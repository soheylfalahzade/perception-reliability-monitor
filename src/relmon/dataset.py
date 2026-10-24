"""Builds labelled monitor datasets by running the reference detector on degraded clips."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from relmon.degrade import Degrader
from relmon.detector import BackgroundBlobDetector, match_recall
from relmon.health import DetectorHealth
from relmon.quality import QualityExtractor
from relmon.scene import Scene, spawn_traffic, visible_box


@dataclass
class Rig:
    """A calibrated camera: scene, detector, quality extractor and clear-weather references."""

    scene: Scene
    detector: BackgroundBlobDetector
    quality: QualityExtractor
    ref_count: float
    clear_f: float
    clear_recall: float


def build_rig(seed: int = 0, n_calib: int = 200) -> Rig:
    """Calibrate background, quality baselines and reference detection count (clear weather)."""
    scene = Scene(seed)
    rng = np.random.default_rng(10_000 + seed)
    clips = list(spawn_traffic(rng, n_calib + 60, 0.08))
    imgs = [scene.render(v)[0] for v in clips[:n_calib]]
    det = BackgroundBlobDetector()
    det.calibrate(imgs[::4])
    q = QualityExtractor()
    q.calibrate(imgs[::2])
    counts, ratios, tps, ngs = [], [], 0, 0
    for v in clips[n_calib:]:
        d = det(scene.render(v)[0])
        gt = [b for b in (visible_box(x) for x in v) if b is not None]
        tp, n_gt, _ = match_recall(gt, d)
        counts.append(len(d))
        tps, ngs = tps + tp, ngs + n_gt
        if n_gt:
            ratios.append(len(d) / n_gt)
    return Rig(scene, det, q, float(np.mean(counts)), float(np.mean(ratios)) if ratios else 1.0,
               tps / max(ngs, 1))


@dataclass
class ClipResult:
    """Per-frame outputs of one clip."""

    features: np.ndarray      # (T, 10): 7 quality + 3 health
    recall_win: np.ndarray    # rolling-window recall (label source)
    n_det: np.ndarray
    n_gt: np.ndarray
    tp: np.ndarray


def run_clip(rig: Rig, condition: str, severity: float, seed: int, n_frames: int = 160,
             density: float = 0.08, window: int = 30) -> ClipResult:
    """Render, degrade, detect and featurise one clip (fresh health and obstruction state)."""
    rng = np.random.default_rng(seed)
    deg = Degrader(seed=int(rng.integers(1 << 30)))
    health = DetectorHealth(reference_count=rig.ref_count)
    rig.quality.reset_obstruction()
    feats, nd, ng, tps = [], [], [], []
    for v in spawn_traffic(rng, n_frames, density):
        img, light = rig.scene.render(v)
        img = deg.apply(img, light, condition, severity)
        boxes = rig.detector(img)
        gt = [b for b in (visible_box(x) for x in v) if b is not None]
        tp, n_g, n_d = match_recall(gt, boxes)
        feats.append(np.concatenate([rig.quality.extract(img), health.update(boxes)]))
        nd.append(n_d)
        ng.append(n_g)
        tps.append(tp)
    tp_a, ng_a = np.array(tps), np.array(ng)
    win_tp: deque[int] = deque(maxlen=window)
    win_gt: deque[int] = deque(maxlen=window)
    rec = []
    for a, b in zip(tp_a, ng_a, strict=True):
        win_tp.append(a)
        win_gt.append(b)
        rec.append(sum(win_tp) / sum(win_gt) if sum(win_gt) > 0 else np.nan)
    return ClipResult(np.array(feats), np.array(rec), np.array(nd), ng_a, tp_a)


def run_sequence(rig: Rig, segments: list[tuple[str, float, int]], seed: int,
                 density: float = 0.08, window: int = 30) -> tuple[ClipResult, np.ndarray]:
    """One continuous stream whose condition changes over time (health state persists).

    ``segments`` lists ``(condition, severity, n_frames)``. Returns the clip result and, per
    frame, the index of the segment it belongs to.
    """
    rng = np.random.default_rng(seed)
    deg = Degrader(seed=int(rng.integers(1 << 30)))
    health = DetectorHealth(reference_count=rig.ref_count)
    rig.quality.reset_obstruction()
    plan = [(c, sv, k) for k, (c, sv, n) in enumerate(segments) for _ in range(n)]
    feats, nd, ng, tps = [], [], [], []
    for v, (cond, sev, _) in zip(spawn_traffic(rng, len(plan), density), plan, strict=True):
        img, light = rig.scene.render(v)
        img_d = deg.apply(img, light, cond, sev)
        boxes = rig.detector(img_d)
        gt = [b for b in (visible_box(x) for x in v) if b is not None]
        tp, n_g, n_d = match_recall(gt, boxes)
        feats.append(np.concatenate([rig.quality.extract(img_d), health.update(boxes)]))
        nd.append(n_d)
        ng.append(n_g)
        tps.append(tp)
    win_tp: deque[int] = deque(maxlen=window)
    win_gt: deque[int] = deque(maxlen=window)
    rec = []
    for a, b in zip(tps, ng, strict=True):
        win_tp.append(a)
        win_gt.append(b)
        rec.append(sum(win_tp) / sum(win_gt) if sum(win_gt) > 0 else np.nan)
    res = ClipResult(np.array(feats), np.array(rec), np.array(nd), np.array(ng), np.array(tps))
    return res, np.array([k for _, _, k in plan])
