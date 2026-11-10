"""Small alpha-beta tracker used to bridge brief missed detections per eye."""
from __future__ import annotations

import math


class AlphaBeta:
    def __init__(self, alpha: float = .72, beta: float = .16, max_lost: int = 3):
        self.alpha, self.beta, self.max_lost = alpha, beta, max_lost
        self.x = self.y = self.vx = self.vy = None
        self.lost = 0

    def update(self, measurement, dt: float = 1.0):
        if measurement is None:
            if self.x is None:
                return {"xy": None, "state": "lost", "lost_frames": self.lost + 1}
            self.x += self.vx * dt; self.y += self.vy * dt; self.lost += 1
            if self.lost > self.max_lost:
                self.x = self.y = self.vx = self.vy = None
                return {"xy": None, "state": "lost", "lost_frames": self.lost}
            return {"xy": [self.x, self.y], "state": "predicted", "lost_frames": self.lost}
        mx, my = map(float, measurement)
        if self.x is None:
            self.x, self.y, self.vx, self.vy, self.lost = mx, my, 0., 0., 0
            return {"xy": [self.x, self.y], "state": "detected", "lost_frames": 0}
        px, py = self.x + self.vx * dt, self.y + self.vy * dt
        rx, ry = mx - px, my - py
        self.x, self.y = px + self.alpha * rx, py + self.alpha * ry
        self.vx += self.beta * rx / max(dt, 1e-6); self.vy += self.beta * ry / max(dt, 1e-6)
        self.lost = 0
        return {"xy": [self.x, self.y], "state": "detected", "lost_frames": 0}


def euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])
