"""Figures from ``relmon-eval`` output."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def plot_streams(results_json: str | Path, out_png: str | Path,
                 picks=(("rain", 0.75), ("glare", 0.75), ("blackout", 1.0))) -> None:
    """Reliability score, rolling recall and mode for clear -> bad -> clear streams."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = json.loads(Path(results_json).read_text())
    rows = [s for s in res["segments"] if (s["condition"], s["severity"]) in set(picks)]
    fig, axes = plt.subplots(len(rows), 1, figsize=(9, 2.6 * len(rows)), sharex=True)
    axes = np.atleast_1d(axes)
    colors = {"normal": "#d9f2d9", "degraded": "#fff1c1", "fallback": "#f8c9c9"}
    for ax, s in zip(axes, rows, strict=True):
        t = np.arange(len(s["r"])) / 10.0
        modes = s["modes"]
        start = 0
        for i in range(1, len(modes) + 1):
            if i == len(modes) or modes[i] != modes[start]:
                ax.axvspan(t[start], t[min(i, len(t) - 1)], color=colors[modes[start]], lw=0)
                start = i
        ax.plot(t, s["r"], color="k", lw=1.4, label="reliability r(t)")
        rec = np.array([np.nan if v is None else v for v in s["recall_win"]])
        ax.plot(t, rec / max(np.nanmax(rec), 1e-6), color="tab:blue", lw=1, alpha=0.8,
                label="rolling recall (normalised, ground truth)")
        seg = np.array(s["segment"])
        ax.axvline(t[np.argmax(seg == 1)], color="gray", ls="--", lw=0.8)
        ax.axvline(t[len(seg) - 1 - np.argmax(seg[::-1] == 1)], color="gray", ls="--", lw=0.8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(f"{s['condition']} {s['severity']}")
    axes[-1].set_xlabel("time [s]  (dashed: condition starts / ends; shading: green normal, "
                        "yellow degraded, red fallback)")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
