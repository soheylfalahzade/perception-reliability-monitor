"""A deliberately simple, classical vehicle detector whose quality really depends on the image.

Background differencing against a clear-weather median background, after per-frame photometric
normalisation. It is a stand-in for any perception module: what matters for the monitor is that its
recall genuinely degrades when the *pixels* degrade (night, fog, rain, glare, dirt), without
any hand-written recall model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from relmon.quality import to_gray


@dataclass(frozen=True)
class Box:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class BackgroundBlobDetector:
    """Detects foreground blobs on a normalised difference image."""

    def __init__(self, threshold: float = 0.9, min_area: int = 40, roi_top: int = 60) -> None:
        self.threshold, self.min_area, self.roi_top = threshold, min_area, roi_top
        self._bg: np.ndarray | None = None

    @staticmethod
    def _norm(g: np.ndarray) -> np.ndarray:
        return (g - g.mean()) / (g.std() + 1e-3)

    def calibrate(self, clear_frames: list[np.ndarray]) -> None:
        """Median background from clear frames (vehicles move, so the median removes them)."""
        med = np.median(np.stack([to_gray(f) for f in clear_frames]), axis=0)
        self._bg = self._norm(med)

    def __call__(self, img: np.ndarray) -> list[Box]:
        assert self._bg is not None, "call calibrate() first"
        diff = np.abs(self._norm(to_gray(img)) - self._bg)
        diff[: self.roi_top] = 0.0
        mask = ndi.binary_opening(diff > self.threshold, structure=np.ones((3, 3)), iterations=1)
        mask = ndi.binary_closing(mask, structure=np.ones((5, 5)), iterations=3)
        mask = ndi.binary_fill_holes(mask)
        lab, n = ndi.label(mask)
        boxes: list[Box] = []
        for i, sl in enumerate(ndi.find_objects(lab), start=1):
            if sl is None:
                continue
            area = int((lab[sl] == i).sum())
            if area < self.min_area:
                continue
            region = diff[sl][lab[sl] == i]
            boxes.append(Box(sl[1].start, sl[0].start, sl[1].stop, sl[0].stop,
                             float(min(1.0, region.mean() / 2.5))))
        return boxes


def match_recall(gt: list[tuple[int, int, int, int]], det: list[Box],
                 thr: float = 0.3) -> tuple[int, int, int]:
    """Greedy IoU matching; returns ``(true_positives, n_gt, n_det)``."""
    used: set[int] = set()
    tp = 0
    for g in gt:
        best, bj = 0.0, -1
        for j, d in enumerate(det):
            if j in used:
                continue
            v = iou(g, (d.x1, d.y1, d.x2, d.y2))
            if v > best:
                best, bj = v, j
        if best >= thr:
            tp += 1
            used.add(bj)
    return tp, len(gt), len(det)
