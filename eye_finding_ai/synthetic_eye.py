"""Deterministic synthetic eye-crop renderer with pixel labels."""
from __future__ import annotations

import math
import random
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HEIGHT, WIDTH = 128, 192
N_CLASSES = 4  # background, sclera, iris, pupil


def render_eye(seed: int, sequence_offset: float | None = None, blink: bool = False) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float]]:
    rng = random.Random(int(seed))
    np_rng = np.random.default_rng(int(seed) ^ 0x6A09E667)
    w, h = WIDTH, HEIGHT
    skin = tuple(rng.randint(a, b) for a, b in ((92, 174), (54, 112), (42, 91)))
    image = Image.new("RGB", (w, h), skin)
    draw = ImageDraw.Draw(image)

    cx = w * 0.5 + rng.uniform(-5, 5)
    cy = h * 0.51 + rng.uniform(-4, 4)
    opening = rng.uniform(25, 39)
    left, right = rng.uniform(7, 18), rng.uniform(w - 18, w - 7)
    upper, lower = [], []
    for i in range(49):
        t = i / 48
        x = left + t * (right - left)
        shape = max(0.0, math.sin(math.pi * t)) ** rng.uniform(0.62, 0.9)
        upper.append((x, cy - opening * shape + rng.uniform(-1.0, 1.0)))
        lower.append((x, cy + opening * shape + rng.uniform(-1.0, 1.0)))
    aperture = upper + list(reversed(lower))

    # Lid folds and brows provide nuisance texture outside the labeled aperture.
    brow = [(x, cy - opening - 13 - 5 * math.sin(math.pi * i / 48)) for i, x in enumerate(np.linspace(left - 3, right + 3, 49))]
    draw.line(brow, fill=tuple(max(0, c - rng.randint(22, 42)) for c in skin), width=rng.randint(2, 5))
    draw.polygon(aperture, fill=(220, 214, 201))

    labels = Image.new("L", (w, h), 0)
    ld = ImageDraw.Draw(labels)
    sclera_color = (rng.randint(205, 246), rng.randint(200, 239), rng.randint(190, 229))
    draw.polygon(aperture, fill=sclera_color)
    ld.polygon(aperture, fill=1)

    # Iris travels within the visible eye opening. Sequence offsets represent gaze motion.
    if sequence_offset is None:
        gaze_x = rng.uniform(-0.38, 0.38)
    else:
        gaze_x = max(-0.48, min(0.48, sequence_offset + rng.uniform(-0.015, 0.015)))
    iris_x = cx + gaze_x * (right - left) * 0.43
    iris_y = cy + rng.uniform(-3.0, 3.0)
    iris_r = rng.uniform(20, 29)
    iris_box = (iris_x - iris_r, iris_y - iris_r, iris_x + iris_r, iris_y + iris_r)
    iris_color = (rng.randint(35, 103), rng.randint(44, 100), rng.randint(28, 76))
    draw.ellipse(iris_box, fill=iris_color)
    ld.ellipse(iris_box, fill=2)
    for _ in range(rng.randint(18, 34)):
        ang = rng.random() * 2 * math.pi
        rad = iris_r * rng.uniform(0.58, 0.91)
        x, y = iris_x + rad * math.cos(ang), iris_y + rad * math.sin(ang)
        draw.ellipse((x - 0.7, y - 0.7, x + 0.7, y + 0.7), fill=tuple(min(255, c + 35) for c in iris_color))

    pupil_r = rng.uniform(8, 14)
    pupil_x, pupil_y = iris_x + rng.uniform(-1.2, 1.2), iris_y + rng.uniform(-1.2, 1.2)
    draw.ellipse((pupil_x - pupil_r, pupil_y - pupil_r, pupil_x + pupil_r, pupil_y + pupil_r), fill=(7, 8, 10))
    ld.ellipse((pupil_x - pupil_r, pupil_y - pupil_r, pupil_x + pupil_r, pupil_y + pupil_r), fill=3)

    # Corneal reflections and small vessels are image nuisances, not target labels.
    for _ in range(rng.randint(1, 3)):
        gx = pupil_x + rng.uniform(-pupil_r * 0.7, pupil_r * 0.7)
        gy = pupil_y + rng.uniform(-pupil_r * 0.7, pupil_r * 0.7)
        r = rng.uniform(1.0, 2.7)
        draw.ellipse((gx-r, gy-r, gx+r, gy+r), fill=(250, 249, 241))
    for _ in range(rng.randint(2, 6)):
        x0 = rng.uniform(left + 4, right - 4)
        y0 = cy + rng.uniform(-opening * 0.72, opening * 0.72)
        draw.line((x0, y0, x0 + rng.uniform(4, 14), y0 + rng.uniform(-1.5, 1.5)), fill=(177, 112, 106), width=1)

    # Partial lids/occluders hide the true surface and clear the corresponding labels.
    if blink:
        gap = rng.uniform(2, 7)
    else:
        gap = rng.uniform(opening * 0.45, opening * 1.25)
    if gap < opening * 1.8:
        top_y, bot_y = cy - gap, cy + gap
        draw.rectangle((0, 0, w, max(0, int(top_y))), fill=skin)
        draw.rectangle((0, min(h, int(bot_y)), w, h), fill=skin)
        ld.rectangle((0, 0, w, max(0, int(top_y))), fill=0)
        ld.rectangle((0, min(h, int(bot_y)), w, h), fill=0)
    if blink:
        draw.polygon(aperture, fill=skin)
        ld.polygon(aperture, fill=0)

    # Random illumination, blur, and sensor noise.
    arr = np.asarray(image, dtype=np.float32)
    shade = np_rng.uniform(0.78, 1.18, size=(1, 1, 3))
    arr = np.clip(arr * shade + np_rng.normal(0, rng.uniform(1.5, 8.0), size=arr.shape), 0, 255).astype(np.uint8)
    image = Image.fromarray(arr, "RGB")
    if rng.random() < 0.28:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.25, 1.25)))
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    mask = np.asarray(labels, dtype=np.uint8)
    pupil_points = np.argwhere(mask == 3)
    if len(pupil_points):
        py, px = pupil_points.mean(axis=0)
        center = (float(px), float(py))
    else:
        center = (float("nan"), float("nan"))
    return gray, mask, center
