"""Train a compact five-class eye segmenter on the single-asset CharMorph set.

Metrics are same-renderer, held-out morph-seed diagnostics only. They are not
person-held-out or real-camera performance estimates.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from model import FaceEyeUNet


FIVE_CLASS_NAMES = ["background", "image_left_iris", "image_left_pupil",
                    "image_right_iris", "image_right_pupil"]
SHARED_EYE_NAMES = ["background", "iris", "pupil"]


def rows_from(path: Path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


class EyeDataset(Dataset):
    def __init__(self, manifest: Path, split: str, size: int, augment: bool,
                 class_mode: str = "five_class"):
        self.rows = [r for r in rows_from(manifest) if r["split"] == split]
        if not self.rows:
            raise ValueError(f"manifest has no {split} rows")
        self.size, self.augment, self.class_mode = size, augment, class_mode

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        image = Image.open(row["image"]).convert("RGB")
        mask = Image.open(row["mask"]).convert("L")
        image = image.resize((self.size, self.size), Image.Resampling.BILINEAR)
        mask = mask.resize((self.size, self.size), Image.Resampling.NEAREST)
        if self.augment:
            if random.random() < .5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                m = np.asarray(mask, dtype=np.uint8)[:, ::-1].copy()
                if self.class_mode == "shared_eye":
                    mask = Image.fromarray(m, mode="L")
                else:
                    mapping = np.arange(256, dtype=np.uint8)
                    mapping[1], mapping[3] = 3, 1
                    mapping[2], mapping[4] = 4, 2
                    mask = Image.fromarray(mapping[m], mode="L")
            if random.random() < .8:
                image = ImageEnhance.Brightness(image).enhance(random.uniform(.78, 1.22))
                image = ImageEnhance.Contrast(image).enhance(random.uniform(.82, 1.18))
                image = ImageEnhance.Color(image).enhance(random.uniform(.9, 1.1))
            if random.random() < .14:
                image = image.filter(ImageFilter.GaussianBlur(random.uniform(.1, .55)))
            if random.random() < .15:
                buf = io.BytesIO()
                image.save(buf, format="JPEG", quality=random.randint(75, 97))
                buf.seek(0)
                image = Image.open(buf).convert("RGB")
        x = np.asarray(image, dtype=np.float32).copy() / 255.0
        if self.augment and random.random() < .25:
            x = np.clip(x + np.random.normal(0, .012, x.shape).astype(np.float32), 0, 1)
        y = np.asarray(mask, dtype=np.int64).copy()
        return torch.from_numpy(x).permute(2, 0, 1), torch.from_numpy(y)


def manifest_audit(path: Path):
    rows = rows_from(path)
    splits = {s: [r for r in rows if r["split"] == s] for s in ("train", "validation", "test")}
    missing = [r[k] for r in rows for k in ("image", "mask") if not Path(r[k]).is_file()]
    ids = {s: {r["sample_id"] for r in rr} for s, rr in splits.items()}
    overlaps = {"train_validation": sorted(ids["train"] & ids["validation"]),
                "train_test": sorted(ids["train"] & ids["test"]),
                "validation_test": sorted(ids["validation"] & ids["test"])}
    if missing or any(overlaps.values()):
        raise ValueError(f"invalid manifest missing={len(missing)} sample_overlap={overlaps}")
    return {"n_rows": len(rows), "split_rows": {k: len(v) for k, v in splits.items()},
            "split_sample_overlap": {k: len(v) for k, v in overlaps.items()},
            "shared_base_identity": len({r["subject_id"] for r in rows}) == 1,
            "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def dice_loss(logits, target):
    probs = logits.softmax(1)
    onehot = torch.nn.functional.one_hot(target, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
    probs, onehot = probs[:, 1:], onehot[:, 1:]
    inter = (probs * onehot).sum((0, 2, 3))
    denom = probs.sum((0, 2, 3)) + onehot.sum((0, 2, 3))
    return 1 - ((2 * inter + 1) / (denom + 1)).mean()


@torch.inference_mode()
def evaluate(model, loader, device, class_names, class_mode):
    n_classes = len(class_names)
    inter = np.zeros(n_classes, dtype=np.float64)
    pred_n = np.zeros(n_classes, dtype=np.float64)
    true_n = np.zeros(n_classes, dtype=np.float64)
    detected = {"image_left": 0, "image_right": 0}
    open_counts = {"image_left": 0, "image_right": 0}
    center_errors = {"image_left": [], "image_right": []}
    model.eval()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x).argmax(1)
        for cls in range(1, n_classes):
            p, t = pred == cls, y == cls
            inter[cls] += (p & t).sum().item()
            pred_n[cls] += p.sum().item()
            true_n[cls] += t.sum().item()
        for j in range(len(y)):
            for side, x_slice, cls in (("image_left", slice(0, y.shape[-1] // 2),
                                        2 if class_mode == "shared_eye" else 2),
                                       ("image_right", slice(y.shape[-1] // 2, y.shape[-1]),
                                        2 if class_mode == "shared_eye" else 4)):
                gt_map = y[j, :, x_slice] == cls
                pred_map = pred[j, :, x_slice] == cls
                gt = torch.nonzero(gt_map)
                pp = torch.nonzero(pred_map)
                if gt.numel():
                    open_counts[side] += 1
                    if pp.shape[0] >= 5:
                        detected[side] += 1
                        center_errors[side].append(float(torch.linalg.vector_norm(
                            pp.float().mean(0) - gt.float().mean(0))))
    dice = {class_names[c]: float(2 * inter[c] / max(pred_n[c] + true_n[c], 1))
            for c in range(1, n_classes)}
    iou = {class_names[c]: float(inter[c] / max(pred_n[c] + true_n[c] - inter[c], 1))
           for c in range(1, n_classes)}
    return {"class_dice": dice, "class_iou": iou,
            "foreground_macro_dice": float(np.mean(list(dice.values()))),
            "pupil_detection_recall": {s: detected[s] / max(open_counts[s], 1) for s in detected},
            "pupil_open_frame_counts": open_counts,
            "pupil_center_error_px_at_input_size_mean": {
                s: float(np.mean(center_errors[s])) if center_errors[s] else None for s in center_errors},
            "pupil_center_error_px_at_input_size_p95": {
                s: float(np.percentile(center_errors[s], 95)) if center_errors[s] else None for s in center_errors},
            "n_frames": len(loader.dataset)}


def atomic_save(obj, path: Path):
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, temp)
    temp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--size", type=int, default=384)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=815920)
    p.add_argument("--test-after-freeze", action="store_true")
    p.add_argument("--class-mode", choices=("five_class", "shared_eye"), default="five_class")
    a = p.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audit = manifest_audit(a.manifest)
    train = DataLoader(EyeDataset(a.manifest, "train", a.size, True, a.class_mode), batch_size=a.batch_size,
                       shuffle=True, num_workers=a.workers, pin_memory=device.type == "cuda")
    val = DataLoader(EyeDataset(a.manifest, "validation", a.size, False, a.class_mode), batch_size=a.batch_size,
                     shuffle=False, num_workers=a.workers, pin_memory=device.type == "cuda")
    class_names = SHARED_EYE_NAMES if a.class_mode == "shared_eye" else FIVE_CLASS_NAMES
    dataset_report = json.loads((a.manifest.parent / "dataset_report.json").read_text())
    if dataset_report.get("class_mode", "five_class") != a.class_mode:
        raise ValueError("--class-mode does not match dataset_report.json")
    n_classes = len(class_names)
    model = FaceEyeUNet(base=16, classes=n_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    weights = (torch.tensor([.05, 1.0, 4.0], device=device) if a.class_mode == "shared_eye"
               else torch.tensor([.05, 1.0, 2.5, 1.0, 2.5], device=device))
    a.output.mkdir(parents=True, exist_ok=True)
    history = []; best = -1.0; best_epoch = 0; stale = 0
    start = time.time()
    for epoch in range(1, a.epochs + 1):
        model.train(); losses = []
        for x, y in train:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=torch.float16):
                logits = model(x)
                loss = nn.functional.cross_entropy(logits, y, weight=weights) + .4 * dice_loss(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            losses.append(float(loss.detach()))
        metrics = evaluate(model, val, device, class_names, a.class_mode)
        rec = {"epoch": epoch, "train_loss": float(np.mean(losses)),
               "validation": metrics, "elapsed_min": round((time.time() - start) / 60, 2)}
        history.append(rec); print(json.dumps(rec), flush=True)
        if metrics["foreground_macro_dice"] > best:
            best = metrics["foreground_macro_dice"]; best_epoch = epoch; stale = 0
            atomic_save({"model": model.state_dict(), "base": 16, "classes": class_names,
                         "size": a.size, "seed": a.seed, "epoch": epoch,
                         "manifest_sha256": audit["manifest_sha256"],
                         "validation_metrics": metrics}, a.output / "best_checkpoint.pt")
        else:
            stale += 1
        atomic_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "scaler": scaler.state_dict(), "epoch": epoch, "best_epoch": best_epoch},
                    a.output / "last_checkpoint.pt")
        (a.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        if stale >= 4:
            break
    report = {"status": "SYNTHETIC_SINGLE_ASSET_PIPELINE_SMOKE",
              "device": str(device), "manifest_audit": audit,
              "dataset_identity_count": 1, "split_scope": "held-out morph seeds from one base asset; not person-held-out",
              "best_epoch": best_epoch, "best_validation_macro_dice": best,
              "epochs_run": len(history), "elapsed_min": round((time.time() - start) / 60, 2),
              "test_used_for_selection": False,
              "checkpoint": str(a.output / "best_checkpoint.pt")}
    (a.output / "training_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if a.test_after_freeze:
        ck = torch.load(a.output / "best_checkpoint.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        test = DataLoader(EyeDataset(a.manifest, "test", a.size, False, a.class_mode), batch_size=a.batch_size,
                          shuffle=False, num_workers=a.workers, pin_memory=device.type == "cuda")
        result = evaluate(model, test, device, class_names, a.class_mode)
        locked = {**report, "status": "SYNTHETIC_SINGLE_ASSET_HELD_OUT_MORPH_TEST",
                  "class_mode": a.class_mode,
                  "test": result, "selection_split": "validation_only",
                  "checkpoint_sha256": hashlib.sha256((a.output / "best_checkpoint.pt").read_bytes()).hexdigest()}
        (a.output / "held_out_morph_test.json").write_text(json.dumps(locked, indent=2) + "\n")
        print(json.dumps({"held_out_test": result,
                          "status": locked["status"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
