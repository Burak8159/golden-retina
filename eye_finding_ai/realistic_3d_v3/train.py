"""Train and evaluate the RGB full-frame v3 model on disjoint synthetic subjects."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from model import FaceEyeUNet

CLASSES = ["background", "face", "left_sclera", "left_iris", "left_pupil", "left_glint",
           "right_sclera", "right_iris", "right_pupil", "right_glint", "left_lid", "right_lid"]
SWAP = {2: 6, 3: 7, 4: 8, 5: 9, 6: 2, 7: 3, 8: 4, 9: 5, 10: 11, 11: 10}


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


class FaceEyeDataset(Dataset):
    def __init__(self, manifest: str, split: str, augment: bool = False, size: int = 384):
        self.rows = [r for r in read_rows(Path(manifest)) if r["split"] == split]
        if not self.rows:
            raise ValueError(f"manifest has no {split} rows")
        self.augment, self.size = augment, size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        image = Image.open(r["image"]).convert("RGB")
        mask = Image.open(r["mask"]).convert("L")
        if self.size:
            image = image.resize((self.size, self.size), Image.Resampling.BILINEAR)
            mask = mask.resize((self.size, self.size), Image.Resampling.NEAREST)
        # Deterministic per-sample augmentation: repeated epochs vary only when
        # the worker RNG changes, while the subject/split assignment stays fixed.
        if self.augment:
            if random.random() < .5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                m = np.asarray(mask, dtype=np.uint8).copy()
                for src, dst in SWAP.items():
                    m[np.asarray(mask) == src] = dst
                mask = Image.fromarray(m, mode="L")
            if random.random() < .8:
                image = ImageEnhance.Brightness(image).enhance(random.uniform(.72, 1.28))
                image = ImageEnhance.Contrast(image).enhance(random.uniform(.78, 1.25))
                image = ImageEnhance.Color(image).enhance(random.uniform(.85, 1.15))
            if random.random() < .18:
                image = image.filter(ImageFilter.GaussianBlur(random.uniform(.15, .8)))
            if random.random() < .20:
                encoded = io.BytesIO()
                image.save(encoded, format="JPEG", quality=random.randint(72, 96))
                encoded.seek(0)
                image = Image.open(encoded).convert("RGB")
        x = np.asarray(image, dtype=np.float32).copy() / 255.0
        if self.augment:
            if random.random() < .30:
                # Approximate sensor read/shot noise; labels stay unchanged.
                x = np.clip(x + np.random.normal(0, random.uniform(.003, .025), x.shape).astype(np.float32), 0, 1)
            if random.random() < .20:
                # Low-amplitude radial falloff for lens/illumination variation.
                yy, xx = np.mgrid[-1:1:complex(x.shape[0]), -1:1:complex(x.shape[1])]
                radius2 = xx * xx + yy * yy
                gain = np.clip(1 - random.uniform(.04, .20) * radius2, .72, 1).astype(np.float32)
                x *= gain[..., None]
        y = np.asarray(mask, dtype=np.int64).copy()
        return torch.from_numpy(x).permute(2, 0, 1), torch.from_numpy(y), int(r["left_blink"]), int(r["right_blink"])


def validate_manifest(path: Path) -> dict:
    rows = read_rows(path)
    by_split = {s: [r for r in rows if r["split"] == s] for s in ("train", "validation", "test")}
    ids = {s: {r["subject_id"] for r in rr} for s, rr in by_split.items()}
    overlaps = {f"{a}_{b}": sorted(ids[a] & ids[b]) for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))}
    missing = [r["image"] for r in rows if not Path(r["image"]).is_file()]
    missing += [r["mask"] for r in rows if not Path(r["mask"]).is_file()]
    if missing or any(overlaps.values()):
        raise ValueError(f"invalid manifest; missing={len(missing)}, subject overlaps={ {k:len(v) for k,v in overlaps.items()} }")
    return {"rows": len(rows), "split_frames": {k: len(v) for k, v in by_split.items()},
            "split_subjects": {k: len(v) for k, v in ids.items()}, "subject_overlap_counts": {k: len(v) for k, v in overlaps.items()},
            "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def calculate_metrics(model, loader, device) -> dict:
    intersections = np.zeros(12, dtype=np.float64)
    unions = np.zeros(12, dtype=np.float64)
    denominators = np.zeros(12, dtype=np.float64)
    detected = {"left": 0, "right": 0}; open_count = {"left": 0, "right": 0}
    centers = {"left": [], "right": []}; blink_hallucinations = {"left": 0, "right": 0}; blink_counts = {"left": 0, "right": 0}
    model.eval()
    with torch.inference_mode():
        for x, y, lb, rb in loader:
            x = x.to(device, non_blocking=True)
            pred = model(x).argmax(1).cpu()
            for cls in range(1, 12):
                p, t = pred == cls, y == cls
                intersections[cls] += (p & t).sum().item()
                unions[cls] += (p | t).sum().item()
                denominators[cls] += p.sum().item() + t.sum().item()
            for j in range(len(y)):
                for side, gt_blink, pupil_cls, iris_cls in (("left", lb[j].item(), 4, 3), ("right", rb[j].item(), 8, 7)):
                    pred_points = torch.nonzero(pred[j] == pupil_cls)
                    true_points = torch.nonzero(y[j] == pupil_cls)
                    if gt_blink:
                        blink_counts[side] += 1
                        if pred_points.numel() >= 5:
                            blink_hallucinations[side] += 1
                    else:
                        open_count[side] += 1
                        if pred_points.numel() >= 5:
                            detected[side] += 1
                            centers[side].append(float(torch.linalg.vector_norm(pred_points.float().mean(0) - true_points.float().mean(0))))
    class_dice = {CLASSES[k]: float(2 * intersections[k] / max(denominators[k], 1)) for k in range(1, 12)}
    class_iou = {CLASSES[k]: float(intersections[k] / max(unions[k], 1)) for k in range(1, 12)}
    return {"class_dice": class_dice, "class_iou": class_iou,
            "foreground_macro_dice": float(np.mean(list(class_dice.values()))),
            "eye_pupil_detection_recall_open": {s: detected[s] / max(open_count[s], 1) for s in detected},
            "eye_open_frame_count": open_count,
            "pupil_center_error_px_384_mean": {s: float(np.mean(centers[s])) if centers[s] else None for s in centers},
            "pupil_center_error_px_384_p95": {s: float(np.percentile(centers[s], 95)) if centers[s] else None for s in centers},
            "blink_pupil_false_positive_rate": {s: blink_hallucinations[s] / max(blink_counts[s], 1) for s in blink_counts},
            "blink_frame_count": blink_counts, "n_frames": len(loader.dataset)}


def dice_loss(logits, target):
    prob = logits.softmax(1)
    one = torch.nn.functional.one_hot(target, num_classes=12).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = (prob * one).sum(dims)
    den = prob.sum(dims) + one.sum(dims)
    return 1 - ((2 * inter[1:] + 1) / (den[1:] + 1)).mean()


def atomic_save(obj, path: Path):
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, temp)
    os.replace(temp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--output", required=True)
    ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--seed", type=int, default=815911)
    ap.add_argument("--size", type=int, default=384); ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--resume", action="store_true"); ap.add_argument("--test-after-freeze", action="store_true")
    a = ap.parse_args(); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.benchmark = torch.cuda.is_available()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = Path(a.manifest); split_audit = validate_manifest(manifest)
    train = DataLoader(FaceEyeDataset(a.manifest, "train", True, a.size), batch_size=a.batch_size, shuffle=True, num_workers=a.workers,
                       pin_memory=dev.type == "cuda", persistent_workers=a.workers > 0)
    val = DataLoader(FaceEyeDataset(a.manifest, "validation", False, a.size), batch_size=a.batch_size, shuffle=False, num_workers=a.workers,
                     pin_memory=dev.type == "cuda", persistent_workers=a.workers > 0)
    model = FaceEyeUNet().to(dev); opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=.5, patience=2, min_lr=1e-5)
    weights = torch.tensor([.06, .15, .9, 1.4, 3.0, 1.0, .9, 1.4, 3.0, 1.0, 1.0, 1.0], device=dev)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    best = -1.; best_epoch = 0; stale = 0; history = []; start_epoch = 1; start = time.time()
    last_path = out / "last_checkpoint.pt"
    if a.resume and last_path.exists():
        ck = torch.load(last_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"]); sched.load_state_dict(ck["scheduler"]); scaler.load_state_dict(ck["scaler"])
        start_epoch = ck["epoch"] + 1; best = ck["best_validation_dice"]; best_epoch = ck["best_epoch"]; stale = ck["stale_epochs"]
        hp = out / "training_history.json"
        if hp.exists(): history = json.loads(hp.read_text())
    for epoch in range(start_epoch, a.epochs + 1):
        model.train(); losses = []
        for x, y, _, _ in train:
            x = x.to(dev, non_blocking=True); y = y.to(dev, non_blocking=True); opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda", dtype=torch.float16):
                logits = model(x); loss = nn.functional.cross_entropy(logits, y, weight=weights) + .40 * dice_loss(logits, y)
            scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.); scaler.step(opt); scaler.update(); losses.append(float(loss.detach()))
        vm = calculate_metrics(model, val, dev); sched.step(vm["foreground_macro_dice"])
        rec = {"epoch": epoch, "train_loss": float(np.mean(losses)), "validation": vm, "lr": opt.param_groups[0]["lr"],
               "elapsed_min": round((time.time() - start) / 60, 2)}
        history.append(rec); print(json.dumps(rec), flush=True)
        if vm["foreground_macro_dice"] > best:
            best = vm["foreground_macro_dice"]; best_epoch = epoch; stale = 0
            atomic_save({"model": model.state_dict(), "epoch": epoch, "base": 16, "classes": CLASSES, "size": a.size,
                        "seed": a.seed, "manifest_sha256": split_audit["manifest_sha256"], "validation_dice": best}, out / "best_checkpoint.pt")
        else: stale += 1
        state = {"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
                 "epoch": epoch, "best_validation_dice": best, "best_epoch": best_epoch, "stale_epochs": stale, "seed": a.seed}
        atomic_save(state, last_path); (out / "training_history.json").write_text(json.dumps(history, indent=2) + "\n")
        if stale >= a.patience: break
    report = {"status": "SYNTHETIC_ONLY_MODEL_DEVELOPMENT", "device": str(dev), "split_audit": split_audit,
              "train_frames": len(train.dataset), "validation_frames": len(val.dataset), "best_epoch": best_epoch,
              "best_validation_macro_dice": best, "epochs_run": len(history), "elapsed_min": round((time.time() - start) / 60, 2),
              "test_used_for_selection": False, "checkpoint": str(out / "best_checkpoint.pt")}
    (out / "training_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if a.test_after_freeze:
        ck = torch.load(out / "best_checkpoint.pt", map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        test = DataLoader(FaceEyeDataset(a.manifest, "test", False, a.size), batch_size=a.batch_size, shuffle=False, num_workers=a.workers,
                          pin_memory=dev.type == "cuda", persistent_workers=a.workers > 0)
        result = calculate_metrics(model, test, dev)
        locked = {**report, "status": "LOCKED_SYNTHETIC_HELD_OUT_TEST", "test": result, "selection_split": "validation_only",
                  "checkpoint_sha256": hashlib.sha256((out / "best_checkpoint.pt").read_bytes()).hexdigest()}
        (out / "locked_test_metrics.json").write_text(json.dumps(locked, indent=2) + "\n")
        print(json.dumps({"locked_test": result, "status": locked["status"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
