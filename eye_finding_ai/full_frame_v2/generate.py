"""Fresh deterministic full-face synthetic eye-finding dataset generator.

Each independent subject seed produces one complete face frame and a seven-class
mask: background, left sclera/iris/pupil, right sclera/iris/pupil. Splits are
assigned by non-overlapping seed ranges before any model training.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

WIDTH, HEIGHT = 320, 224
CLASSES = ["background", "left_sclera", "left_iris", "left_pupil",
           "right_sclera", "right_iris", "right_pupil"]


def render(seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    r = random.Random(seed)
    nr = np.random.default_rng(seed ^ 0xA17E_8159)
    bg = tuple(r.randint(12, 38) for _ in range(3))
    skin = tuple(r.randint(a, b) for a, b in ((118, 210), (69, 157), (46, 126)))
    image = Image.new("RGB", (WIDTH, HEIGHT), bg)
    draw = ImageDraw.Draw(image)
    mask = Image.new("L", (WIDTH, HEIGHT), 0)
    md = ImageDraw.Draw(mask)

    fx0, fy0 = r.randint(26, 39), r.randint(8, 20)
    fx1, fy1 = WIDTH - r.randint(26, 39), HEIGHT + r.randint(2, 24)
    draw.ellipse((fx0, fy0, fx1, fy1), fill=skin)
    mid = (fx0 + fx1) / 2
    eye_y = r.randint(77, 103)
    nose_color = tuple(max(0, c - r.randint(8, 28)) for c in skin)
    draw.line((mid + r.randint(-5, 5), eye_y + 12, mid + r.randint(-10, 10), eye_y + 55),
              fill=nose_color, width=r.randint(2, 5))
    draw.arc((mid - 16, eye_y + 47, mid + 17, eye_y + 66), 8, 170,
             fill=nose_color, width=2)
    mouth_y = min(HEIGHT - 8, eye_y + r.randint(79, 99))
    draw.arc((mid - r.randint(12, 23), mouth_y - 5, mid + r.randint(12, 23), mouth_y + 9),
             5, 175, fill=tuple(max(0, c - 45) for c in skin), width=2)

    left_center = (r.randint(91, 112), eye_y + r.randint(-5, 5))
    right_center = (r.randint(208, 229), eye_y + r.randint(-5, 5))
    eye_w, eye_h = r.randint(35, 46), r.randint(17, 24)
    conditions = {"blink_left": False, "blink_right": False,
                  "occlusion": False, "blur_radius": 0.0, "exposure": 1.0,
                  "noise_sigma": 0.0, "camera_style": r.choice(["warm", "cool", "neutral"])}

    for side, (cx, cy) in enumerate((left_center, right_center)):
        base = 1 if side == 0 else 4
        x0, x1, y0, y1 = cx-eye_w, cx+eye_w, cy-eye_h, cy+eye_h
        socket = tuple(max(0, c-r.randint(17, 48)) for c in skin)
        draw.ellipse((x0-5, y0-5, x1+5, y1+5), fill=socket)
        blink = r.random() < 0.045
        conditions["blink_left" if side == 0 else "blink_right"] = blink
        # The eyelid outline is part of the image; closed-eye frames have no
        # pupil labels, which makes the state observable to the pipeline.
        draw.ellipse((x0, y0, x1, y1), fill=(236, 230, 218) if not blink else skin,
                     outline=tuple(max(0, c-35) for c in skin), width=2)
        if blink:
            draw.arc((x0, cy-eye_h//2, x1, cy+eye_h//2), 0, 180,
                     fill=tuple(max(0, c-38) for c in skin), width=3)
            continue

        md.ellipse((x0, y0, x1, y1), fill=base)
        gaze_x = int(r.uniform(-eye_w*0.28, eye_w*0.28))
        gaze_y = int(r.uniform(-eye_h*0.2, eye_h*0.2))
        ix, iy = cx+gaze_x, cy+gaze_y
        iris_r = r.randint(8, 12)
        pupil_r = r.randint(3, 6)
        draw.ellipse((ix-iris_r, iy-iris_r, ix+iris_r, iy+iris_r),
                     fill=r.choice([(57, 83, 86), (83, 61, 42), (51, 57, 76), (67, 81, 50)]))
        md.ellipse((ix-iris_r, iy-iris_r, ix+iris_r, iy+iris_r), fill=base+1)
        draw.ellipse((ix-pupil_r, iy-pupil_r, ix+pupil_r, iy+pupil_r), fill=(13, 16, 19))
        md.ellipse((ix-pupil_r, iy-pupil_r, ix+pupil_r, iy+pupil_r), fill=base+2)
        # Small specular highlights are deliberately not mislabeled as pupil.
        if r.random() < 0.7:
            draw.ellipse((ix-pupil_r+1, iy-pupil_r+1, ix-pupil_r+3, iy-pupil_r+3),
                         fill=(245, 244, 237))
        brow = tuple(max(0, c-34) for c in skin)
        draw.line((x0+4, y0-8, cx, y0-10+r.randint(-3, 3), x1-3, y0-7),
                  fill=brow, width=r.randint(2, 5), joint="curve")
        # Lid occlusion clips visible semantic labels in a controlled fraction.
        if r.random() < 0.14:
            conditions["occlusion"] = True
            cut = r.randint(2, max(3, eye_h//2))
            draw.rectangle((x0, y0, x1, y0+cut), fill=skin)
            md.rectangle((x0, y0, x1, y0+cut), fill=0)

    # Acquisition shifts are applied to pixels only; masks stay geometric.
    rgb = np.asarray(image, dtype=np.float32)
    style = conditions["camera_style"]
    if style == "warm": rgb[..., 0] *= 1.08; rgb[..., 2] *= 0.91
    elif style == "cool": rgb[..., 0] *= 0.91; rgb[..., 2] *= 1.08
    exposure = r.uniform(0.68, 1.32)
    sigma = r.uniform(0, 8)
    conditions["exposure"], conditions["noise_sigma"] = round(exposure, 4), round(sigma, 3)
    rgb = np.clip(rgb * exposure + nr.normal(0, sigma, rgb.shape), 0, 255).astype(np.uint8)
    image = Image.fromarray(rgb, "RGB")
    blur = r.choices([0.0, 0.45, 0.9, 1.4], weights=[0.62, 0.20, 0.13, 0.05])[0]
    conditions["blur_radius"] = blur
    if blur:
        image = image.filter(ImageFilter.GaussianBlur(blur))
    # Occasional mild crop/rotation simulates pose and framing variation while
    # keeping the labels exactly aligned via nearest-neighbour transforms.
    angle = r.uniform(-3.0, 3.0)
    image = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=bg)
    mask = mask.rotate(angle, resample=Image.Resampling.NEAREST, fillcolor=0)
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    labels = np.asarray(mask, dtype=np.uint8)
    return gray, labels, {**conditions, "rotation_deg": round(angle, 3)}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="data/full_frame_v2")
    ap.add_argument("--train", type=int, default=1800)
    ap.add_argument("--validation", type=int, default=300)
    ap.add_argument("--test", type=int, default=400)
    ap.add_argument("--seed", type=int, default=815900)
    a = ap.parse_args()
    out = Path(a.output); rows=[]; counts={"train":a.train,"validation":a.validation,"test":a.test}
    start=a.seed
    for split, n in counts.items():
        for j in range(n):
            subject=start+j
            gray, mask, cond=render(subject)
            name=f"{split}_{subject:07d}"
            ip=out/"images"/f"{name}.png"; mp=out/"masks"/f"{name}.png"
            ip.parent.mkdir(parents=True,exist_ok=True); mp.parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray(gray).save(ip,optimize=True)
            Image.fromarray(mask).save(mp,optimize=True)
            rows.append({"image":str(ip),"mask":str(mp),"split":split,"subject_id":f"synthetic-{subject:07d}","seed":subject,
                         "left_eye_visible":int(bool((mask==3).any())),"right_eye_visible":int(bool((mask==6).any())),
                         "label_counts":json.dumps({str(k):int((mask==k).sum()) for k in range(7)},separators=(",",":")),
                         "conditions":json.dumps(cond,separators=(",",":"))})
        start += n
    manifest=out/"manifest.csv"; manifest.parent.mkdir(parents=True,exist_ok=True)
    with manifest.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    cfg={"generator":"golden_retina.full_frame_v2","seed":a.seed,"size_wh":[WIDTH,HEIGHT],"classes":CLASSES,
         "counts":counts,"split_policy":"disjoint deterministic subject-seed ranges; split assigned before generation",
         "image_augmentation":"per-image exposure/color/noise/blur/rotation; paired label transforms",
         "synthetic_only":True}
    (out/"generator_config.json").write_text(json.dumps(cfg,indent=2)+"\n")
    summary={"images":len(rows),"counts":counts,"unique_subjects":len({x['subject_id'] for x in rows}),
             "manifest_sha256":sha256(manifest),"status":"GENERATED_SYNTHETIC_FULL_FRAME_DATASET_ONLY"}
    (out/"generation_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))

if __name__=="__main__": main()
