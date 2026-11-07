"""Monitor evaluation on degraded clips: AUC, calibration, mode switching, banks for control."""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from relmon.dataset import build_rig, run_clip, run_sequence
from relmon.degrade import CONDITIONS
from relmon.monitor import MonitorConfig, ReliabilityModel, ReliabilityMonitor
from relmon.quality import FEATURE_NAMES

SEVERITIES = (0.0, 0.25, 0.5, 0.75, 1.0)
RELIABLE_REL = 0.6   # reliable = rolling recall >= 0.6 x clear-weather recall of this camera
FEATURE_SETS = {"quality_only": list(range(7)), "health_only": [7, 8, 9],
                "combined": list(range(10))}


def auc(score: np.ndarray, y: np.ndarray) -> float:
    """Rank-based AUC (ties handled by average ranks)."""
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    for v in np.unique(score):
        m = score == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    n1, n0 = int(y.sum()), int((1 - y).sum())
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error with equal-width bins."""
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def bootstrap_ci(values, n_boot: int = 2000, level: float = 0.95, seed: int = 0):
    """Percentile bootstrap interval of the mean of ``values`` (None when fewer than two values)."""
    v = np.asarray([x for x in values if x is not None], dtype=float)
    if len(v) < 2:
        return None
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return [float(lo), float(hi)]


def cluster_bootstrap_metrics(p: np.ndarray, y: np.ndarray, clip_ids: np.ndarray,
                              n_boot: int = 1000, level: float = 0.95, seed: int = 0):
    """AUC and ECE intervals that resample whole clips (frames of a clip are dependent)."""
    ids = np.unique(clip_ids)
    members = {c: np.flatnonzero(clip_ids == c) for c in ids}
    rng = np.random.default_rng(seed)
    aucs, eces = [], []
    for _ in range(n_boot):
        pick = rng.choice(ids, size=len(ids), replace=True)
        rows = np.concatenate([members[c] for c in pick])
        yy = y[rows]
        if yy.min() == yy.max():
            continue
        aucs.append(auc(p[rows], yy))
        eces.append(ece(p[rows], yy))
    q = [(1 - level) / 2, 1 - (1 - level) / 2]
    return {"auc_ci": [float(v) for v in np.quantile(aucs, q)],
            "ece_ci": [float(v) for v in np.quantile(eces, q)], "n_clips": int(len(ids)),
            "n_boot_used": int(len(aucs))}


def cluster_bootstrap_auc_difference(p_a: np.ndarray, p_b: np.ndarray, y: np.ndarray,
                                     clip_ids: np.ndarray, n_boot: int = 1000,
                                     level: float = 0.95, seed: int = 0):
    """AUC(a) - AUC(b) on the same resampled clips (an interval for the paired difference)."""
    ids = np.unique(clip_ids)
    members = {c: np.flatnonzero(clip_ids == c) for c in ids}
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        rows = np.concatenate([members[c] for c in rng.choice(ids, size=len(ids), replace=True)])
        yy = y[rows]
        if yy.min() == yy.max():
            continue
        diffs.append(auc(p_a[rows], yy) - auc(p_b[rows], yy))
    q = [(1 - level) / 2, 1 - (1 - level) / 2]
    return {"diff": float(auc(p_a, y) - auc(p_b, y)),
            "ci": [float(v) for v in np.quantile(diffs, q)], "n_clips": int(len(ids))}


def stream_stats(modes, segment, fps: float = 10.0, warmup_s: float = 8.0) -> dict[str, Any]:
    """Mode statistics of one clear -> bad -> clear stream (segment indices 0, 1, 2).

    * ``fallback_frac_bad`` / ``normal_frac_bad``: fractions of the bad segment spent in FALLBACK
      and in NORMAL.
    * ``fallback_frac_clear``: fraction of the clear frames spent in FALLBACK, taking the first
      clear segment after a ``warmup_s`` warm-up and the whole second one, so that the slow
      recovery after the bad episode counts as false fallback.
    * ``fallback_frac_clear_steady``: the same after ``warmup_s`` in both clear
      segments.
    * ``carryover``: the monitor was already in FALLBACK in the frame before the onset, so a
      FALLBACK in the bad segment is not a reaction to it.
    * ``delay_s``: time from the onset of the bad segment to the first FALLBACK frame, only for
      streams without carry-over (None if the monitor never reacted).
    """
    modes, seg = np.asarray(modes), np.asarray(segment)
    fb, nm = modes == "fallback", modes == "normal"
    bad = seg == 1
    idx = np.arange(len(seg))
    start = {k: int(np.argmax(seg == k)) for k in (0, 2)}
    warm = {k: idx >= start[k] + warmup_s * fps for k in (0, 2)}
    clear = ((seg == 0) & warm[0]) | (seg == 2)
    steady = ((seg == 0) & warm[0]) | ((seg == 2) & warm[2])
    hit = np.flatnonzero(bad & fb)
    onset = int(np.argmax(bad))
    carry = bool(onset > 0 and fb[onset - 1])
    return {"fallback_frac_bad": float(fb[bad].mean()), "normal_frac_bad": float(nm[bad].mean()),
            "fallback_frac_clear": float(fb[clear].mean()),
            "fallback_frac_clear_steady": float(fb[steady].mean()),
            "carryover": carry,
            "delay_s": float((hit[0] - onset) / fps) if len(hit) and not carry else None}


def summarize_streams(per_rep: list[dict[str, Any]], n_boot: int = 2000) -> dict[str, Any]:
    """Mean over replicate streams, with bootstrap intervals when there are at least two."""
    out: dict[str, Any] = {"n_reps": len(per_rep)}
    for k in ("fallback_frac_bad", "normal_frac_bad", "fallback_frac_clear",
              "fallback_frac_clear_steady"):
        vals = [r[k] for r in per_rep]
        out[k] = float(np.mean(vals))
        out[k + "_ci"] = bootstrap_ci(vals, n_boot)
    delays = [r["delay_s"] for r in per_rep if r["delay_s"] is not None]
    out["carryover_frac"] = sum(bool(r.get("carryover")) for r in per_rep) / len(per_rep)
    out["detected_frac"] = len(delays) / len(per_rep)
    out["delay_s"] = float(np.mean(delays)) if delays else None
    out["delay_s_ci"] = bootstrap_ci(delays, n_boot)
    return out


def collect(rig, seeds: list[int], n_frames: int = 160, return_clip_ids: bool = False):
    thr = RELIABLE_REL * rig.clear_recall
    """Run all (condition, severity) clips; returns rows and per-setting statistics."""
    xs, ys, meta, clip_ids = [], [], [], []
    stats: dict[str, dict] = {}
    for cond in CONDITIONS:
        for s in ((0.0,) if cond == "clear" else SEVERITIES[1:]):
            tps = ngs = nds = 0
            ratio_bank, feat_bank = [], []
            for sd in seeds:
                clip_seed = sd * 131 + int(s * 100) + CONDITIONS.index(cond)
                r = run_clip(rig, cond, s, clip_seed, n_frames)
                keep = slice(60, None)       # warm-up for health / obstruction windows
                ok = ~np.isnan(r.recall_win[keep])
                xs.append(r.features[keep][ok])
                ys.append((r.recall_win[keep][ok] >= thr).astype(float))
                meta += [(cond, s)] * int(ok.sum())
                clip_ids += [len(stats) * 1000 + sd] * int(ok.sum())
                tps += int(r.tp[keep].sum())
                ngs += int(r.n_gt[keep].sum())
                nds += int(r.n_det[keep].sum())
                feat_bank.append(r.features[keep])
                pairs = zip(r.n_det[keep], r.n_gt[keep], strict=True)
                ratio_bank += [d / g for d, g in pairs if g > 0]
            stats[f"{cond}@{s}"] = {
                "recall": tps / max(ngs, 1), "precision": tps / max(nds, 1),
                "count_ratio_bank": [float(v) for v in ratio_bank],
                "feature_bank": np.vstack(feat_bank).tolist()}
    if return_clip_ids:
        return np.vstack(xs), np.concatenate(ys), meta, stats, np.array(clip_ids)
    return np.vstack(xs), np.concatenate(ys), meta, stats


_RIG = None


def _stream_rep(job):
    """One replicate of a clear -> bad -> clear stream (runs in a worker process)."""
    global _RIG
    cond, sev, seed, model = job
    if _RIG is None:
        _RIG = build_rig(0)
    r, seg = run_sequence(_RIG, [("clear", 0.0, 200), (cond, sev, 160), ("clear", 0.0, 200)], seed)
    mon = ReliabilityMonitor(model, MonitorConfig(fps=10.0))
    modes = [mon.update(f)[1].value for f in r.features]
    return stream_stats(modes, seg)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (``relmon-eval``)."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dev-seeds", type=int, default=2)
    ap.add_argument("--test-seeds", type=int, default=4)
    ap.add_argument("--out", default="results/eval")
    ap.add_argument("--stream-reps", type=int, default=1,
                    help="replicate streams per setting (the first keeps the original seed); "
                         ">= 2 gives intervals")
    ap.add_argument("--jobs", type=int, default=1,
                    help="worker processes for the extra stream replicates")
    ap.add_argument("--bootstrap", type=int, default=1000,
                    help="bootstrap resamples for the AUC / ECE intervals")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rig = build_rig(0)
    xd, yd, _, _ = collect(rig, [1000 + i for i in range(args.dev_seeds)])
    xt, yt, meta, stats, clip_ids = collect(rig, list(range(args.test_seeds)), return_clip_ids=True)
    models, report = {}, {}
    preds: dict[str, np.ndarray] = {}
    for name, cols in FEATURE_SETS.items():
        m = ReliabilityModel().fit(xd[:, cols], yd)
        p = np.array([m.predict(r[cols]) for r in xt])
        models[name] = m
        preds[name] = p
        report[name] = {"auc": auc(p, yt), "ece": ece(p, yt), "cols": cols,
                        **cluster_bootstrap_metrics(p, yt, clip_ids, n_boot=args.bootstrap)}
    m = models["combined"]
    p = np.array([m.predict(r) for r in xt])
    per_cond: dict[str, dict[str, Any]] = {}
    for cond in CONDITIONS:
        idx = np.array([c == cond for c, _ in meta])
        if idx.any() and 0 < yt[idx].mean() < 1:
            per_cond[cond] = {"auc": auc(p[idx], yt[idx]), "reliable_frac": float(yt[idx].mean())}
        else:
            per_cond[cond] = {"auc": None, "reliable_frac": float(yt[idx].mean())}
    # monitor behaviour on continuous streams: clear (warm-up + steady) -> bad -> clear
    seg_rows = []
    for cond in CONDITIONS[1:]:
        for sev in ((0.5, 0.75, 1.0) if cond == "blackout" else (0.5, 0.75)):
            r, seg = run_sequence(rig, [("clear", 0.0, 200), (cond, sev, 160), ("clear", 0.0, 200)],
                                  777 + CONDITIONS.index(cond) * 10 + int(sev * 10))
            mon = ReliabilityMonitor(m, MonitorConfig(fps=10.0))
            rs, modes = zip(*[(lambda o: (o[0], o[1].value))(mon.update(f)) for f in r.features],
                            strict=True)
            seg_rows.append({"condition": cond, "severity": sev, "modes": list(modes),
                             "r": [float(v) for v in rs], "segment": seg.tolist(),
                             "recall_win": [None if np.isnan(v) else float(v)
                                            for v in r.recall_win]})
    auc_differences = {
        f"{a} - {b}": cluster_bootstrap_auc_difference(preds[a], preds[b], yt, clip_ids,
                                                       n_boot=args.bootstrap)
        for a, b in (("combined", "quality_only"), ("combined", "health_only"))}
    stream_summary, stream_raw = [], []
    def rep_seed(k: int, row: dict[str, Any]) -> int:
        return 9000 + 1000 * k + CONDITIONS.index(row["condition"]) * 10 + int(row["severity"] * 10)

    extra = [(row["condition"], row["severity"], rep_seed(k, row), m)
             for k in range(1, args.stream_reps) for row in seg_rows]
    if extra and args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            extra_stats = list(ex.map(_stream_rep, extra, chunksize=1))
    else:
        global _RIG
        _RIG = rig
        extra_stats = [_stream_rep(j) for j in extra]
    for row in seg_rows:
        reps = [stream_stats(row["modes"], row["segment"])]
        reps += [st for (c, sv, _, _), st in zip(extra, extra_stats, strict=True)
                 if c == row["condition"] and sv == row["severity"]]
        stream_raw.append({"condition": row["condition"], "severity": row["severity"],
                           "reps": reps})
        stream_summary.append({"condition": row["condition"], "severity": row["severity"],
                               **summarize_streams(reps)})
    result = {"report": report, "stream_summary": stream_summary, "stream_reps": args.stream_reps,
              "stream_reps_raw": stream_raw, "auc_differences": auc_differences,
              "per_condition_auc_combined": per_cond,
              "setting_stats": {k: {"recall": v["recall"], "precision": v["precision"]}
                                for k, v in stats.items()},
              "feature_names": list(FEATURE_NAMES) + ["mean_score", "count_ratio", "fragmentation"],
              "reliable_recall_threshold": RELIABLE_REL * rig.clear_recall,
              "clear_recall_calibration": rig.clear_recall, "segments": seg_rows,
              "n_test_frames": int(len(yt)), "reliable_frac": float(yt.mean()),
              "combined_weights": m.weights.tolist(), "combined_mean": m.mean.tolist(),
              "combined_scale": m.scale.tolist()}
    (out / "results.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    (out / "banks.json").write_text(json.dumps(
        {k: {"count_ratio_bank": v["count_ratio_bank"], "feature_bank": v["feature_bank"]}
         for k, v in stats.items()}), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("report", "stream_summary",
                                               "per_condition_auc_combined",
                                               "setting_stats", "n_test_frames",
                                               "reliable_frac")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
