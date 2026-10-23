"""Synthetic fixed-camera road scene with moving vehicles and exact ground-truth boxes."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

H, W = 180, 320
LANES_X = (110.0, 160.0, 210.0)  # lane centre columns at the bottom of the image


@dataclass
class Vehicle:
    lane: int
    y: float          # image row of the vehicle's bottom edge
    speed: float      # rows per frame
    color: tuple[int, int, int]
    kind: str = "car"


def _scale(y: float) -> float:
    return 0.25 + 0.75 * (y / H)


def vehicle_box(v: Vehicle) -> tuple[int, int, int, int]:
    """Ground-truth ``(x1, y1, x2, y2)`` in pixels (perspective: smaller further up)."""
    s = _scale(v.y)
    w = (30 if v.kind == "car" else 38) * s
    h = (26 if v.kind == "car" else 44) * s
    cx = W / 2 + (LANES_X[v.lane] - W / 2) * (0.35 + 0.65 * v.y / H)
    return int(cx - w / 2), int(v.y - h), int(cx + w / 2), int(v.y)


def visible_box(v: Vehicle, min_visible: float = 0.6) -> tuple[int, int, int, int] | None:
    """Ground-truth box clipped to the image and the ROI; ``None`` if mostly outside."""
    x1, y1, x2, y2 = vehicle_box(v)
    full = max((x2 - x1) * (y2 - y1), 1)
    cx1, cy1, cx2, cy2 = max(x1, 0), max(y1, 62), min(x2, W), min(y2, H)
    if cx2 <= cx1 or cy2 <= cy1 or (cx2 - cx1) * (cy2 - cy1) < min_visible * full:
        return None
    return cx1, cy1, cx2, cy2


class Scene:
    """Static textured background + vehicle sprites (+ a light layer for night rendering)."""

    def __init__(self, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        bg = np.zeros((H, W, 3), np.float32)
        bg[:] = (95, 100, 98)
        bg[:60] = (140, 150, 160)                      # sky / far buildings
        for x0 in range(0, W, 40):
            bg[20:60, x0:x0 + 30] = rng.uniform(70, 130, 3)
        bg[60:, :70] = (80, 110, 80)
        bg[60:, 250:] = (80, 110, 80)
        for lane_x in (135.0, 185.0):                  # dashed lane markings
            for y in range(70, H, 14):
                s = y / H
                x = int(W / 2 + (lane_x - W / 2) * (0.35 + 0.65 * s))
                bg[y:y + 6, x - 1:x + 1] = (230, 230, 230)
        bg += rng.normal(0, 4.0, bg.shape)             # fixed asphalt texture
        self.background = np.clip(bg, 0, 255)
        self.rng = rng

    def render(self, vehicles: list[Vehicle]) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(image uint8 BGR, light_layer float32)`` for the given vehicles."""
        img = self.background.copy()
        light = np.zeros_like(img)
        for v in sorted(vehicles, key=lambda q: q.y):
            x1, y1, x2, y2 = vehicle_box(v)
            x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, W), min(y2, H)
            if x2 <= x1 or y2 <= y1:
                continue
            img[y1:y2, x1:x2] = v.color
            wy = y1 + (y2 - y1) // 5
            img[wy:wy + (y2 - y1) // 4, x1 + 1:x2 - 1] = (35, 35, 45)   # windscreen
            img[y2 - 1:min(y2 + 2, H), x1:x2] = (40, 40, 40)            # shadow
            for xl in (x1 + 1, x2 - 3):
                light[max(y2 - 5, 0):y2 - 1, xl:xl + 3] = (255, 255, 235)
        return np.clip(img, 0, 255).astype(np.uint8), light


def spawn_traffic(rng: np.random.Generator, n_steps: int, density: float = 0.06):
    """Yield the vehicle list at each frame (vehicles enter at the horizon, move down)."""
    vehicles: list[Vehicle] = []
    for _ in range(n_steps):
        if rng.random() < density:
            lane = int(rng.integers(3))
            if all(abs(v.y - 62.0) > 14 for v in vehicles if v.lane == lane):
                col = tuple(int(c) for c in rng.integers(30, 235, 3))
                while abs(0.114 * col[0] + 0.587 * col[1] + 0.299 * col[2] - 99) < 50:
                    col = tuple(int(c) for c in rng.integers(30, 235, 3))
                vehicles.append(Vehicle(lane, 62.0, float(rng.uniform(0.6, 1.4)), col,
                                        "truck" if rng.random() < 0.2 else "car"))
        for v in vehicles:
            v.y += v.speed * (0.4 + v.y / H)
        vehicles = [v for v in vehicles if v.y < H + 20]
        yield list(vehicles)
