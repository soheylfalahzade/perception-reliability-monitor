"""Monitor evaluation on degraded clips: AUC, calibration, mode switching, banks for control."""
from __future__ import annotations

import argparse
import json
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


def collect(rig, seeds: list[int], n_frames: int = 160):
    thr = RELIABLE_REL * rig.clear_recall
    """Run all (condition, severity) clips; returns rows and per-setting statistics."""
    xs, ys, meta = [], [], []
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
    return np.vstack(xs), np.concatenate(ys), meta, stats


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (``relmon-eval``)."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dev-seeds", type=int, default=2)
    ap.add_argument("--test-seeds", type=int, default=4)
    ap.add_argument("--out", default="results/eval")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rig = build_rig(0)
    xd, yd, _, _ = collect(rig, [1000 + i for i in range(args.dev_seeds)])
    xt, yt, meta, stats = collect(rig, list(range(args.test_seeds)))
    models, report = {}, {}
    for name, cols in FEATURE_SETS.items():
        m = ReliabilityModel().fit(xd[:, cols], yd)
        p = np.array([m.predict(r[cols]) for r in xt])
        models[name] = m
        report[name] = {"auc": auc(p, yt), "ece": ece(p, yt), "cols": cols}
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
    result = {"report": report, "per_condition_auc_combined": per_cond,
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
    print(json.dumps({k: result[k] for k in ("report", "per_condition_auc_combined",
                                               "setting_stats", "n_test_frames",
                                               "reliable_frac")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
