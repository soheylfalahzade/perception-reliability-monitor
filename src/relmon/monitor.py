"""Reliability score and graceful-degradation state machine."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class Mode(str, Enum):
    NORMAL = "normal"
    DEGRADED = "degraded"
    FALLBACK = "fallback"


@dataclass
class ReliabilityModel:
    """Logistic model mapping monitor features to P(perception is reliable)."""

    weights: np.ndarray = field(default_factory=lambda: np.zeros(1))
    mean: np.ndarray = field(default_factory=lambda: np.zeros(1))
    scale: np.ndarray = field(default_factory=lambda: np.ones(1))

    def _z(self, x: np.ndarray) -> np.ndarray:
        return np.concatenate([(x - self.mean) / self.scale, [1.0]])

    def predict(self, x: np.ndarray) -> float:
        return float(1.0 / (1.0 + np.exp(-float(self._z(x) @ self.weights))))

    def fit(self, x: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 100):
        """Standardise features and fit by L2-regularised Newton iterations (IRLS)."""
        self.mean, self.scale = x.mean(axis=0), x.std(axis=0) + 1e-6
        z = np.hstack([(x - self.mean) / self.scale, np.ones((len(x), 1))])
        w = np.zeros(z.shape[1])
        reg = np.eye(len(w)) * l2
        reg[-1, -1] = 0.0
        for _ in range(iters):
            p = 1.0 / (1.0 + np.exp(-(z @ w)))
            grad = z.T @ (p - y) + reg @ w
            hess = (z * (p * (1 - p))[:, None]).T @ z + reg + 1e-9 * np.eye(len(w))
            step = np.linalg.solve(hess, grad)
            w -= step
            if np.abs(step).max() < 1e-8:
                break
        self.weights = w
        return self


@dataclass
class MonitorConfig:
    """Smoothing and hysteresis (all times in frames or seconds as noted)."""

    ema: float = 0.15
    degraded_below: float = 0.65
    fallback_below: float = 0.35
    recover_above: float = 0.55
    dwell_down_s: float = 3.0
    dwell_up_s: float = 6.0
    fps: float = 10.0


class ReliabilityMonitor:
    """Smooths the model output and switches NORMAL / DEGRADED / FALLBACK with hysteresis.

    Going *down* requires the score to stay below a threshold for ``dwell_down_s``; going back up
    to NORMAL requires staying above ``recover_above`` for ``dwell_up_s``. The asymmetry makes the
    controller fall back quickly and return cautiously.
    """

    def __init__(self, model: ReliabilityModel, cfg: MonitorConfig | None = None) -> None:
        self.model, self.cfg = model, cfg or MonitorConfig()
        self.mode = Mode.NORMAL
        self.r = 1.0
        self._init = False
        self._below_fb = self._below_deg = self._above = 0

    def update(self, features: np.ndarray) -> tuple[float, Mode]:
        """Consume one frame's features; return ``(smoothed_reliability, mode)``."""
        raw = self.model.predict(features)
        self.r = raw if not self._init else self.cfg.ema * raw + (1 - self.cfg.ema) * self.r
        self._init = True
        c, fps = self.cfg, self.cfg.fps
        self._below_fb = self._below_fb + 1 if self.r < c.fallback_below else 0
        self._below_deg = self._below_deg + 1 if self.r < c.degraded_below else 0
        self._above = self._above + 1 if self.r > c.recover_above else 0
        if self._below_fb >= c.dwell_down_s * fps:
            self.mode = Mode.FALLBACK
        elif self.mode == Mode.NORMAL and self._below_deg >= c.dwell_down_s * fps:
            self.mode = Mode.DEGRADED
        elif self.mode != Mode.NORMAL and self._above >= c.dwell_up_s * fps:
            self.mode = Mode.NORMAL
        elif self.mode == Mode.FALLBACK and self.r >= c.fallback_below and self._below_deg == 0:
            self.mode = Mode.DEGRADED
        return self.r, self.mode


def fallback_rate(model: ReliabilityModel, cfg: MonitorConfig,
                  streams: list[np.ndarray]) -> float:
    """Fraction of frames spent in FALLBACK over feature streams (each ``(frames, features)``)."""
    n = hit = 0
    for feats in streams:
        mon = ReliabilityMonitor(model, cfg)
        for x in feats:
            _, mode = mon.update(x)
            hit += int(mode == Mode.FALLBACK)
            n += 1
    return hit / max(n, 1)


THRESHOLD_GRID: tuple[float, ...] = tuple(round(0.05 * k, 2) for k in range(1, 15))


def calibrate_fallback_threshold(model: ReliabilityModel, healthy_streams: list[np.ndarray],
                                 target: float = 0.01, base: MonitorConfig | None = None,
                                 grid: tuple[float, ...] = THRESHOLD_GRID) -> MonitorConfig:
    """Most sensitive ``fallback_below`` whose false-fallback rate on healthy streams is <= target.

    ``healthy_streams`` are feature streams recorded in good conditions. The returned config keeps
    all other settings of ``base``; the degraded/recover thresholds are moved up with the
    fallback threshold so that the ordering fallback < recover < degraded is preserved.
    """
    from dataclasses import replace

    base = base or MonitorConfig()
    best = replace(base, fallback_below=float(grid[0]))
    for thr in grid:
        cand = replace(base, fallback_below=float(thr),
                       recover_above=max(base.recover_above, float(thr) + 0.1),
                       degraded_below=max(base.degraded_below, float(thr) + 0.2))
        if fallback_rate(model, cand, healthy_streams) <= target:
            best = cand
        else:
            break
    return best
