"""Image degradations emulating adverse conditions (night, rain, fog, glare, lens dirt)."""
from __future__ import annotations

import numpy as np

CONDITIONS = ("clear", "night", "rain", "fog", "glare", "dirt", "blackout")


def _box_blur(img: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return img
    out = img.copy()
    for ax in (0, 1):
        c = np.cumsum(np.pad(out, [(k // 2 + 1, k // 2) if a == ax else (0, 0)
                                    for a in range(out.ndim)], mode="edge"), axis=ax)
        sl_hi = [slice(None)] * out.ndim
        sl_lo = [slice(None)] * out.ndim
        sl_hi[ax], sl_lo[ax] = slice(k, k + out.shape[ax]), slice(0, out.shape[ax])
        out = (c[tuple(sl_hi)] - c[tuple(sl_lo)]) / k
    return out


class Degrader:
    """Applies a condition at severity in [0, 1] to a rendered frame; dirt blobs are persistent."""

    def __init__(self, seed: int = 0, h: int = 180, w: int = 320) -> None:
        self.rng = np.random.default_rng(seed)
        yy, xx = np.mgrid[0:h, 0:w]
        self._dirt = np.zeros((h, w), np.float32)
        for _ in range(6):
            cy, cx, r = self.rng.uniform(20, h - 20), self.rng.uniform(20, w - 20), \
                self.rng.uniform(14, 34)
            self._dirt = np.maximum(self._dirt, np.clip(1 - ((yy - cy) ** 2 + (xx - cx) ** 2)
                                                         / r**2, 0, 1))
        self._glare_c = (self.rng.uniform(40, 100), self.rng.uniform(80, 240))

    def apply(self, img: np.ndarray, light: np.ndarray, condition: str, s: float) -> np.ndarray:
        """Return a degraded uint8 image. ``s`` = 0 leaves the image unchanged."""
        x = img.astype(np.float32)
        h, w = x.shape[:2]
        if condition == "clear" or s <= 0:
            out = x
        elif condition == "night":
            out = x * (1 - 0.85 * s) + light * s * 0.9
            out += self.rng.normal(0, 2 + 7 * s, x.shape)
        elif condition == "rain":
            out = _box_blur(x, 1 + int(3 * s)) * (1 - 0.25 * s) + 40 * s
            n = int(160 * s)
            ys, xs = self.rng.integers(0, h - 14, n), self.rng.integers(0, w - 6, n)
            for y0, x0 in zip(ys, xs, strict=True):
                for k in range(10):
                    out[y0 + k, x0 + k // 3] += 55 * s
        elif condition == "fog":
            a = (s * (0.92 - 0.5 * np.linspace(0, 1, h)))[:, None, None]
            out = x * (1 - a) + 215 * a
        elif condition == "glare":
            yy, xx = np.mgrid[0:h, 0:w]
            cy, cx = self._glare_c
            blob = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * (28 + 40 * s) ** 2))
            out = x + (330 * s * blob)[..., None] + 25 * s
        elif condition == "dirt":
            m = _box_blur(self._dirt, 5)[..., None] * min(1.0, 1.2 * s)
            out = x * (1 - 0.92 * m) + 35 * 0.92 * m
        elif condition == "blackout":      # dead / covered feed: signal fades to sensor noise
            out = x * (1 - s) + self.rng.normal(0, 1.5 * s, x.shape)
        else:
            raise ValueError(f"unknown condition {condition!r}")
        return np.clip(out, 0, 255).astype(np.uint8)
