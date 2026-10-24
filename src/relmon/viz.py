"""Data-driven animation: degraded frames, detections and the monitor's reliability score."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from relmon.dataset import build_rig
from relmon.degrade import Degrader
from relmon.health import DetectorHealth
from relmon.monitor import MonitorConfig, ReliabilityModel, ReliabilityMonitor
from relmon.scene import spawn_traffic, visible_box


def render_monitor_gif(out: str | Path, results_json: str | Path, condition: str = "rain",
                       severity: float = 0.75, seed: int = 21, every: int = 4,
                       fps: int = 10) -> dict[str, Any]:
    """Stream clear -> ``condition`` -> clear and animate frame, detections and ``r(t)``.

    Left: the frame as the camera sees it; green boxes are ground truth, red boxes detections.
    Right: reliability ``r(t)`` with the mode shading (green normal, yellow degraded, red fallback).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.patches import Rectangle
    from PIL import Image

    res = json.loads(Path(results_json).read_text())
    model = ReliabilityModel(np.array(res["combined_weights"]), np.array(res["combined_mean"]),
                             np.array(res["combined_scale"]))
    rig = build_rig(0)
    rig.quality.reset_obstruction()
    rng = np.random.default_rng(seed)
    deg = Degrader(seed=int(rng.integers(1 << 30)))
    health = DetectorHealth(reference_count=rig.ref_count)
    monitor = ReliabilityMonitor(model, MonitorConfig(fps=10.0))
    plan = [("clear", 0.0)] * 200 + [(condition, severity)] * 160 + [("clear", 0.0)] * 200
    recs: list[dict[str, Any]] = []
    for v, (cnd, sev) in zip(spawn_traffic(rng, len(plan), 0.08), plan, strict=True):
        img, light = rig.scene.render(v)
        img = deg.apply(img, light, cnd, sev)
        boxes = rig.detector(img)
        feats = np.concatenate([rig.quality.extract(img), health.update(boxes)])
        r, mode = monitor.update(feats)
        recs.append({"img": img, "gt": [b for b in (visible_box(x) for x in v) if b],
                     "det": boxes, "r": r, "mode": mode.value, "cond": cnd})
    colors = {"normal": "#d9f2d9", "degraded": "#fff1c1", "fallback": "#f8c9c9"}
    images = []
    for k in range(120, len(recs), every):
        rec = recs[k]
        fig, (a, b) = plt.subplots(1, 2, figsize=(8.6, 3.3), dpi=72,
                                   gridspec_kw={"width_ratios": [1, 1.25]})
        a.imshow(rec["img"][..., ::-1])
        for g in rec["gt"]:
            a.add_patch(Rectangle((g[0], g[1]), g[2] - g[0], g[3] - g[1], fill=False,
                                  ec="#2ca02c", lw=1.4))
        for d in rec["det"]:
            a.add_patch(Rectangle((d.x1, d.y1), d.x2 - d.x1, d.y2 - d.y1, fill=False,
                                  ec="#d62728", lw=1.0, ls="--"))
        a.set_xticks([])
        a.set_yticks([])
        a.set_title(f"camera view: {rec['cond']}", fontsize=9)
        t = np.arange(k + 1) / 10.0
        start = 0
        for i in range(1, k + 2):
            if i == k + 1 or recs[i]["mode"] != recs[start]["mode"]:
                b.axvspan(t[start], t[min(i, k)], color=colors[recs[start]["mode"]], lw=0)
                start = i
        b.plot(t, [x["r"] for x in recs[: k + 1]], color="k", lw=1.6)
        for edge in (20.0, 36.0):
            b.axvline(edge, color="gray", ls="--", lw=0.8)
        b.set_xlim(12, len(recs) / 10.0)
        b.set_ylim(0, 1.05)
        b.set_xlabel("time [s]  (dashed: condition starts / ends)", fontsize=8)
        b.set_ylabel("reliability r(t)", fontsize=8)
        b.set_title(f"mode: {rec['mode']}", fontsize=9)
        b.tick_params(labelsize=7)
        fig.tight_layout()
        canvas = cast(FigureCanvasAgg, fig.canvas)
        canvas.draw()
        images.append(Image.fromarray(np.asarray(canvas.buffer_rgba())).convert("RGB")
                      .quantize(colors=96))
        plt.close(fig)
    images[0].save(out, save_all=True, append_images=images[1:], duration=int(1000 / fps),
                   loop=0, optimize=True)
    return {"frames": len(images)}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (``relmon-gif``)."""
    import argparse

    ap = argparse.ArgumentParser(description=render_monitor_gif.__doc__)
    ap.add_argument("--out", default="docs/assets/monitor.gif")
    ap.add_argument("--results", default="docs/results/eval_results.json")
    ap.add_argument("--condition", default="rain")
    ap.add_argument("--severity", type=float, default=0.75)
    args = ap.parse_args(argv)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    print(render_monitor_gif(args.out, args.results, args.condition, args.severity))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
