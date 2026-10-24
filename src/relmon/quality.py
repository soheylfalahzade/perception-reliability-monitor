"""Reference-free image-quality features and a persistent lens-obstruction map (numpy only)."""
from __future__ import annotations

import numpy as np

FEATURE_NAMES = ("log_brightness", "log_contrast", "log_sharpness", "log_noise",
                 "saturated_frac", "log_dynamic_range", "obstruction_frac")


def to_gray(img: np.ndarray) -> np.ndarray:
    """Float32 luminance in [0, 255] from a BGR or grayscale uint8 image."""
    a = img.astype(np.float32)
    if a.ndim == 3:
        return 0.114 * a[..., 0] + 0.587 * a[..., 1] + 0.299 * a[..., 2]
    return a


def _laplacian(g: np.ndarray) -> np.ndarray:
    return 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]


def _block_gradient(g: np.ndarray, grid: int) -> np.ndarray:
    gy, gx = np.abs(np.diff(g, axis=0))[:, :-1], np.abs(np.diff(g, axis=1))[:-1, :]
    mag = gx + gy
    h, w = mag.shape
    bh, bw = h // grid, w // grid
    return mag[: bh * grid, : bw * grid].reshape(grid, bh, grid, bw).mean(axis=(1, 3))


class QualityExtractor:
    """Computes the seven monitor features; relative to a clear-weather calibration.

    ``calibrate`` stores baseline statistics (brightness, contrast, sharpness, noise, dynamic
    range, per-block gradient map) from clear frames. At run time, features are log-ratios to the
    baseline (so they are zero in clear conditions), plus the saturated-pixel fraction and the
    fraction of image blocks whose slowly-averaged detail fell below 85% of the baseline, which
    flags persistent lens contamination.
    """

    def __init__(self, grid: int = 8, obstruction_ema: float = 0.05) -> None:
        self.grid, self.ema = grid, obstruction_ema
        self.base: dict[str, float] | None = None
        self.base_map: np.ndarray | None = None
        self._map: np.ndarray | None = None

    @staticmethod
    def _raw(g: np.ndarray) -> dict[str, float]:
        lap = _laplacian(g)
        p5, p95 = np.percentile(g, (5, 95))
        return {
            "brightness": float(g.mean()) + 1.0, "contrast": float(g.std()) + 1e-3,
            "sharpness": float(lap.var()) + 1e-3,
            "noise": float(1.4826 * np.median(np.abs(lap - np.median(lap)))) + 1e-3,
            "dynamic_range": float(p95 - p5) + 1.0, "saturated_frac": float((g >= 250).mean()),
        }

    def calibrate(self, clear_frames: list[np.ndarray]) -> None:
        """Store clear-weather baselines from a list of images."""
        raws = [self._raw(to_gray(f)) for f in clear_frames]
        self.base = {k: float(np.mean([r[k] for r in raws])) for k in raws[0]}
        self.base_map = np.mean([_block_gradient(to_gray(f), self.grid) for f in clear_frames],
                                axis=0) + 1e-3
        self._map = self.base_map.copy()

    def extract(self, img: np.ndarray) -> np.ndarray:
        """Feature vector (order: :data:`FEATURE_NAMES`) for one frame."""
        assert self.base is not None and self.base_map is not None, "call calibrate() first"
        g = to_gray(img)
        r = self._raw(g)
        cur = _block_gradient(g, self.grid)
        assert self._map is not None
        self._map = (1 - self.ema) * self._map + self.ema * cur
        obstruction = float(np.mean(self._map / self.base_map < 0.85))
        b = self.base
        return np.array([
            np.log(r["brightness"] / b["brightness"]), np.log(r["contrast"] / b["contrast"]),
            np.log(r["sharpness"] / b["sharpness"]), np.log(r["noise"] / b["noise"]),
            r["saturated_frac"], np.log(r["dynamic_range"] / b["dynamic_range"]), obstruction,
        ])

    def reset_obstruction(self) -> None:
        """Restart the slow obstruction average (e.g. after cleaning)."""
        if self.base_map is not None:
            self._map = self.base_map.copy()
