"""Detector-side health signals that need no ground truth: scores, counts and track stability."""
from __future__ import annotations

from collections import deque

import numpy as np

from relmon.detector import Box, iou

HEALTH_NAMES = ("mean_score", "count_ratio", "fragmentation")


class IouTracker:
    """Minimal greedy IoU tracker; exposes births and track lengths for health statistics."""

    def __init__(self, iou_thr: float = 0.2, max_age: int = 3) -> None:
        self.iou_thr, self.max_age = iou_thr, max_age
        self._tracks: dict[int, tuple[tuple[int, int, int, int], int]] = {}
        self._next = 1
        self.births = 0

    def update(self, boxes: list[Box]) -> list[int]:
        """Assign IDs to ``boxes``; returns IDs in the same order."""
        ids, taken = [], set()
        for b in boxes:
            bb = (b.x1, b.y1, b.x2, b.y2)
            best, bid = 0.0, None
            for tid, (tb, _) in self._tracks.items():
                v = iou(bb, tb) if tid not in taken else 0.0
                if v > best:
                    best, bid = v, tid
            if bid is not None and best >= self.iou_thr:
                taken.add(bid)
                ids.append(bid)
            else:
                bid = self._next
                self._next += 1
                self.births += 1
                taken.add(bid)
                ids.append(bid)
            self._tracks[bid] = (bb, 0)
        for tid in list(self._tracks):
            if tid not in taken:
                tb, age = self._tracks[tid]
                if age + 1 > self.max_age:
                    del self._tracks[tid]
                else:
                    self._tracks[tid] = (tb, age + 1)
        return ids


class DetectorHealth:
    """Rolling detector-health features over a window of frames.

    * ``mean_score``: mean detection confidence;
    * ``count_ratio``: window mean count over a clear-weather reference mean count;
    * ``fragmentation``: new track IDs per detection (high = unstable tracking).
    """

    def __init__(self, window: int = 60, reference_count: float = 1.0) -> None:
        self.window, self.ref = window, max(reference_count, 1e-3)
        self.tracker = IouTracker()
        self._counts: deque[int] = deque(maxlen=window)
        self._scores: deque[float] = deque(maxlen=window * 4)
        self._births: deque[int] = deque(maxlen=window)

    def update(self, boxes: list[Box]) -> np.ndarray:
        """Add a frame of detections; returns the feature vector in :data:`HEALTH_NAMES` order."""
        before = self.tracker.births
        self.tracker.update(boxes)
        self._births.append(self.tracker.births - before)
        self._counts.append(len(boxes))
        self._scores.extend(b.score for b in boxes)
        n_det = max(sum(self._counts), 1)
        return np.array([
            float(np.mean(self._scores)) if self._scores else 0.0,
            float(np.mean(self._counts)) / self.ref,
            float(sum(self._births)) / n_det,
        ])
