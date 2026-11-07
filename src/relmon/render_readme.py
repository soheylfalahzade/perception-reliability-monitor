"""Render the result tables of README.md from docs/results/eval_results.json.

Tables sit between markers ``<!-- BEGIN:name -->`` ... ``<!-- END:name -->`` and are generated here,
so no number inside them is typed by hand.

    python -m relmon.render_readme            # rewrite the tables in README.md
    python -m relmon.render_readme --check    # exit 1 if README.md differs from the recorded data

Verdict rules for the claims table (thresholds fixed here, not tuned to the result):
  * joint prediction: "supported" when the interval of the paired AUC difference excludes zero for
    both single-source models, otherwise "not supported";
  * reaction: among streams where the detector fails (relative recall below the reliability
    threshold), "supported" when the monitor reacted in at least 90% of the replicate streams for
    every such setting, otherwise "partially supported";
  * false fallbacks: "supported" when no stream exceeds 0.10 of its clear frames in FALLBACK
    counting the slow recovery after the bad episode, otherwise "partially supported".
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"
RESULTS = "docs/results/eval_results.json"
SETS = [
    ("quality_only", "image quality only"),
    ("health_only", "detector health only"),
    ("combined", "**combined**"),
]
REACT_MIN, FALSE_MAX = 0.9, 0.10


def load(path: str = RESULTS) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def md(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    return "\n".join(out + ["| " + " | ".join(r) + " |" for r in rows])


def with_ci(v: float | None, ci: list[float] | None, nd: int = 2) -> str:
    if v is None:
        return "-"
    return f"{v:.{nd}f}" if ci is None else f"{v:.{nd}f} ({ci[0]:.{nd}f} to {ci[1]:.{nd}f})"


def fails(d: dict[str, Any], condition: str, severity: float) -> bool:
    return d["setting_stats"][f"{condition}@{severity}"]["recall"] < d["reliable_recall_threshold"]


def block_frame_level(d: dict[str, Any]) -> str:
    rows = []
    for key, label in SETS:
        r = d["report"][key]
        rows.append([label, with_ci(r["auc"], r["auc_ci"], 3), with_ci(r["ece"], r["ece_ci"], 3)])
    n_clips = d["report"]["combined"]["n_clips"]
    boot = d["report"]["combined"]["n_boot_used"]
    lines = [
        md(["features", "AUC (95% interval)", "ECE (95% interval)"], rows),
        "",
        f"Intervals resample whole clips ({n_clips} test clips, {boot} resamples), because the "
        "frames of one clip are strongly dependent. Paired AUC difference on the same resampled "
        "clips:",
        "",
    ]
    for key, diff in d["auc_differences"].items():
        a, b = key.split(" - ")
        lines.append(
            f"- {a.replace('_', ' ')} minus {b.replace('_', ' ')}: "
            f"{diff['diff']:+.3f} ({diff['ci'][0]:+.3f} to {diff['ci'][1]:+.3f})"
        )
    per = ", ".join(f"{c} {v['auc']:.2f}" for c, v in d["per_condition_auc_combined"].items())
    lines += ["", f"Per-condition AUC of the combined model: {per}."]
    return "\n".join(lines)


def block_detector(d: dict[str, Any]) -> str:
    sev = ["0.25", "0.5", "0.75", "1.0"]
    st = d["setting_stats"]
    conds = []
    for key in st:
        c = key.split("@")[0]
        if c != "clear" and c not in conds:
            conds.append(c)
    rows = [
        [
            "clear",
            f"{st['clear@0.0']['recall']:.2f} / {st['clear@0.0']['precision']:.2f}",
            "",
            "",
            "",
        ]
    ]
    for c in conds:
        cells = []
        for v in sev:
            r = st.get(f"{c}@{float(v)}")
            cells.append("" if r is None else f"{r['recall']:.2f} / {r['precision']:.2f}")
        rows.append([c] + cells)
    return md(["condition"] + sev, rows)


def block_streams(d: dict[str, Any]) -> str:
    rows = []
    for s in d["stream_summary"]:
        cond, sev = s["condition"], s["severity"]
        rows.append(
            [
                f"{cond} {sev:g}",
                "fails" if fails(d, cond, sev) else "copes",
                with_ci(s["fallback_frac_bad"], s["fallback_frac_bad_ci"]),
                f"{s['normal_frac_bad']:.2f}",
                with_ci(s["fallback_frac_clear"], s["fallback_frac_clear_ci"]),
                with_ci(s["fallback_frac_clear_steady"], s["fallback_frac_clear_steady_ci"]),
                f"{s['detected_frac']:.2f}",
                with_ci(s["delay_s"], s["delay_s_ci"], 1),
                f"{s['carryover_frac']:.2f}",
            ]
        )
    n = d["stream_summary"][0]["n_reps"]
    head = [
        "stream",
        "detector",
        "FALLBACK in bad",
        "NORMAL in bad (mean)",
        "FALLBACK in clear",
        "in clear, after recovery",
        "reacted",
        "delay [s]",
        "carry-over",
    ]
    return "\n".join(
        [
            md(head, rows),
            "",
            f"Mean over {n} replicate streams per setting, with 95% bootstrap intervals "
            "over replicates. *FALLBACK in clear* counts the first clear segment after an 8 s "
            "warm-up and the whole second one, so slow recovery counts as false fallback; "
            "*after recovery* applies the warm-up to both. *Reacted* is the share of replicates "
            "in which the monitor entered FALLBACK during the bad segment without having been "
            "in FALLBACK just before it; *carry-over* is the share that entered the bad segment "
            "already in FALLBACK. The delay is measured from the onset over replicates that "
            "reacted. *detector* marks whether the detector's relative recall in that setting "
            "is below the reliability threshold.",
        ]
    )


def verdicts(d: dict[str, Any]) -> tuple[str, str, str, dict[str, Any]]:
    diffs = d["auc_differences"]
    v1 = "supported" if all(x["ci"][0] > 0 for x in diffs.values()) else "not supported"
    failing = [s for s in d["stream_summary"] if fails(d, s["condition"], s["severity"])]
    v2 = (
        "supported"
        if failing and all(s["detected_frac"] >= REACT_MIN for s in failing)
        else "partially supported"
    )
    worst = max(d["stream_summary"], key=lambda s: s["fallback_frac_clear"])
    v3 = "supported" if worst["fallback_frac_clear"] <= FALSE_MAX else "partially supported"
    return v1, v2, v3, {"failing": failing, "worst": worst}


def block_claims(d: dict[str, Any]) -> str:
    v1, v2, v3, x = verdicts(d)
    rep = d["report"]
    diffs = d["auc_differences"]
    low = min(diffs.values(), key=lambda t: t["diff"])
    e1 = (
        f"**{v1}**: AUC {rep['combined']['auc']:.3f} (combined) vs "
        f"{rep['quality_only']['auc']:.3f} (quality) and {rep['health_only']['auc']:.3f} "
        f"(health); the smallest paired difference is {low['diff']:+.3f} "
        f"({low['ci'][0]:+.3f} to {low['ci'][1]:+.3f})"
    )
    fl = x["failing"]
    delays = [s["delay_s"] for s in fl if s["delay_s"] is not None]
    if fl and delays:
        lo_r = 100 * min(s["detected_frac"] for s in fl)
        hi_r = 100 * max(s["detected_frac"] for s in fl)
        reacted = f"{lo_r:.0f}%" if round(lo_r) == round(hi_r) else f"{lo_r:.0f}% to {hi_r:.0f}%"
        e2 = (
            f"**{v2}**: in the {len(fl)} streams where the detector fails the monitor reacted in "
            f"{reacted} of replicates, with a mean delay of {min(delays):.1f} to "
            f"{max(delays):.1f} s; reaction elsewhere is in the stream table"
        )
    else:
        e2 = f"**{v2}**: no stream in which the detector fails reacted; see the stream table"
    w = x["worst"]
    steady = max(d["stream_summary"], key=lambda s: s["fallback_frac_clear_steady"])
    e3 = (
        f"**{v3}**: up to {w['fallback_frac_clear']:.2f} of clear frames in FALLBACK, counting the "
        f"recovery after the bad episode ({w['condition']} {w['severity']}); after the recovery "
        f"up to {steady['fallback_frac_clear_steady']:.2f} ({steady['condition']} "
        f"{steady['severity']})"
    )
    return md(
        ["claim", "evidence", "file"],
        [
            ["Image quality and detector health jointly predict reliability", e1, f"`{RESULTS}`"],
            ["The monitor reacts when the detector fails", e2, f"`{RESULTS}`"],
            ["The monitor limits false fallbacks", e3, f"`{RESULTS}`"],
        ],
    )


BLOCKS = {
    "detector": block_detector,
    "frame_level": block_frame_level,
    "streams": block_streams,
    "claims_monitor": block_claims,
}


def render(text: str, d: dict[str, Any] | None = None) -> str:
    d = d or load()
    for name, fn in BLOCKS.items():
        pat = re.compile(rf"(<!-- BEGIN:{name} -->)\n.*?(<!-- END:{name} -->)", re.S)
        if not pat.search(text):
            sys.exit(f"README.md has no marker block for '{name}'")
        body = fn(d)

        def fill(m: re.Match[str], body: str = body) -> str:
            return m.group(1) + "\n" + body + "\n" + m.group(2)

        text = pat.sub(fill, text)
    return text


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    current = README.read_text(encoding="utf-8")
    rendered = render(current)
    if args.check:
        if rendered != current:
            print(
                "README.md tables differ from the recorded data; "
                "run: python -m relmon.render_readme"
            )
            sys.exit(1)
        print("README.md tables match the recorded data.")
        return
    README.write_text(rendered, encoding="utf-8")
    print("README.md tables rendered from the recorded data")


if __name__ == "__main__":
    main()
