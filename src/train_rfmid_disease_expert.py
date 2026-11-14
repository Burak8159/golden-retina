"""Train one support-qualified RFMiD fundus finding/disease specialist.

This is a retrospective research tool. RFMiD's image-level split does not
establish patient independence, and a model trained on these images is not a
clinically validated diagnostic device.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

SPLITS = {
    "train": ("Training_Set/Training_Set/Training", "Training_Set/Training_Set/RFMiD_Training_Labels.csv"),
    "valid": ("Evaluation_Set/Evaluation_Set/Validation", "Evaluation_Set/Evaluation_Set/RFMiD_Validation_Labels.csv"),
    "test": ("Test_Set/Test_Set/Test", "Test_Set/Test_Set/RFMiD_Testing_Labels.csv"),
}
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
DINOV2_MODEL_ID = "facebook/dinov2-small"
DINOV2_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"


def read_rows(root: Path, split: str, target: str) -> list[dict]:
    image_rel, labels_rel = SPLITS[split]
    path = root / labels_rel
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        if target not in columns:
            raise ValueError(f"Target {target!r} absent from {path}; available={columns}")
        id_col = columns[0]
        rows = []
        for item in reader:
            image_id = str(int(item[id_col]))
            image_path = root / image_rel / f"{image_id}.png"
            if not image_path.is_file():
                matches = list((root / image_rel).glob(f"{image_id}.*"))
                if len(matches) != 1:
                    raise FileNotFoundError(f"Expected one image for ID {image_id} in {image_rel}")
                image_path = matches[0]
            value = int(item[target])
            if value not in (0, 1):
                raise ValueError(f"Non-binary {target}={value} at image {image_id}")
            rows.append({"id": image_id, "path": image_path, "y": value})
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError(f"Duplicate IDs in {split}")
    return rows


class Fundus(Dataset):
    def __init__(self, rows: list[dict], size: int, train: bool):
        self.rows = rows
        ops = [transforms.Resize((size, size))]
        if train:
            ops += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(8),
                    transforms.ColorJitter(brightness=.08, contrast=.08, saturation=.06)]
        ops += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
        self.transform = transforms.Compose(ops)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        r = self.rows[index]
        with Image.open(r["path"]) as im:
            x = self.transform(im.convert("RGB"))
        return x, torch.tensor(r["y"], dtype=torch.float32), r["id"]


def make_model(architecture: str, pretrained: bool) -> nn.Module:
    if architecture == "convnext_tiny":
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 1)
    elif architecture == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, 1)
    elif architecture == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 1)
    elif architecture == "dinov2_vits14":
        if not pretrained:
            raise ValueError("DINOv2 expert requires the official pretrained checkpoint")
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise RuntimeError("Install transformers to use the DINOv2 expert") from exc

        class DINOv2Binary(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = AutoModel.from_pretrained(DINOV2_MODEL_ID, revision=DINOV2_REVISION)
                self.classifier = nn.Linear(self.encoder.config.hidden_size, 1)

            def forward(self, x):
                return self.classifier(self.encoder(pixel_values=x).last_hidden_state[:, 0])

        m = DINOv2Binary()
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    return m


def auc_ap(y: np.ndarray, s: np.ndarray) -> tuple[float | None, float | None]:
    pos = int(y.sum()); neg = len(y) - pos
    auc = None
    if pos and neg:
        order = np.argsort(s, kind="mergesort")
        ranks = np.empty(len(y), dtype=float)
        sorted_s = s[order]
        i = 0
        while i < len(y):
            j = i + 1
            while j < len(y) and sorted_s[j] == sorted_s[i]:
                j += 1
            ranks[order[i:j]] = (i + 1 + j) / 2
            i = j
        auc = float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))
    ap = None
    if pos:
        order = np.argsort(-s, kind="mergesort")
        ys, ss = y[order], s[order]
        ends = np.r_[np.flatnonzero(ss[:-1] != ss[1:]), len(y) - 1]
        tp = np.cumsum(ys)[ends]
        fp = np.cumsum(1 - ys)[ends]
        prev = np.r_[0, tp[:-1]]
        ap = float(np.sum((tp - prev) / pos * tp / (tp + fp)))
    return auc, ap


def predict(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval(); ys, scores, ids = [], [], []
    with torch.inference_mode():
        for x, y, image_ids in loader:
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                z = model(x).flatten()
            ys.extend(y.numpy().astype(int).tolist())
            scores.extend(torch.sigmoid(z.float()).cpu().numpy().tolist())
            ids.extend(image_ids)
    return np.asarray(ys), np.asarray(scores), ids


def metrics(y: np.ndarray, s: np.ndarray) -> dict:
    p = s >= .5
    tp = int(np.sum((y == 1) & p)); tn = int(np.sum((y == 0) & ~p))
    fp = int(np.sum((y == 0) & p)); fn = int(np.sum((y == 1) & ~p))
    auc, ap = auc_ap(y, s)
    return {"n": len(y), "positive": int(y.sum()), "negative": int(len(y)-y.sum()),
            "accuracy_at_0_5": float(np.mean(p == y)),
            "sensitivity_at_0_5": tp / int(y.sum()) if y.sum() else None,
            "specificity_at_0_5": tn / int(len(y)-y.sum()) if len(y)-y.sum() else None,
            "auroc": auc, "average_precision": ap,
            "confusion_at_0_5": {"tp": tp, "tn": tn, "fp": fp, "fn": fn}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--target", required=True, help="Exact RFMiD label column, e.g. DR or ARMD")
    p.add_argument("--architecture", choices=["convnext_tiny", "resnet50", "efficientnet_b0", "dinov2_vits14"], required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--size", type=int, default=320)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--max-seconds", type=int, default=1800)
    p.add_argument("--min-positive", type=int, default=10)
    p.add_argument("--min-negative", type=int, default=10)
    p.add_argument("--seed", type=int, default=20261114)
    p.add_argument("--evidence-tier", choices=["primary_support", "exploratory_low_support"],
                   default="primary_support", help="Explicit support tier recorded in each result.")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--evaluate-checkpoint", type=Path,
                   help="Evaluate one validation-selected checkpoint on the official test split, without training.")
    a = p.parse_args()
    if a.architecture == "dinov2_vits14" and a.size % 14:
        p.error("DINOv2 ViT-S/14 input size must be divisible by 14 (use --size 224 or 336).")
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this L4 expert run")
    root = a.data_root.resolve()
    splits = {s: read_rows(root, s, a.target) for s in SPLITS}
    support = {s: {"positive": sum(r["y"] for r in rows), "negative": sum(1-r["y"] for r in rows)}
               for s, rows in splits.items()}
    train = support["train"]
    if train["positive"] < a.min_positive or train["negative"] < a.min_negative:
        raise SystemExit(json.dumps({"status": "NOT_TRAINED_INSUFFICIENT_TRAIN_SUPPORT", "target": a.target,
                                     "required": {"positive": a.min_positive, "negative": a.min_negative},
                                     "support": support}, indent=2))
    a.outdir.mkdir(parents=True, exist_ok=True); a.artifact.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    model = make_model(a.architecture, not a.no_pretrained).to(device)
    ratio = train["negative"] / train["positive"]
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([min(10., max(1., ratio ** .5))], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=2e-4)
    loaders = {
      "train": DataLoader(Fundus(splits["train"], a.size, True), batch_size=a.batch_size, shuffle=True,
                          num_workers=a.workers, pin_memory=True, persistent_workers=False),
      "valid": DataLoader(Fundus(splits["valid"], a.size, False), batch_size=a.batch_size, shuffle=False,
                          num_workers=a.workers, pin_memory=True, persistent_workers=False),
      "test": DataLoader(Fundus(splits["test"], a.size, False), batch_size=a.batch_size, shuffle=False,
                         num_workers=a.workers, pin_memory=True, persistent_workers=False),
    }
    if a.evaluate_checkpoint:
        ckpt = torch.load(a.evaluate_checkpoint, map_location=device, weights_only=False)
        if ckpt.get("target") != a.target or ckpt.get("architecture") != a.architecture:
            raise ValueError("Checkpoint target/architecture does not match the requested test run")
        model.load_state_dict(ckpt["state_dict"])
        ty, ts, _ = predict(model, loaders["test"], device)
        report = {
            "status": "RFMiD_OFFICIAL_IMAGE_SPLIT_RESEARCH_ONLY_PATIENT_INDEPENDENCE_UNVERIFIED",
            "target": a.target, "architecture": a.architecture,
            "evidence_tier": a.evidence_tier,
            "support_caution": ("Low-support exploratory result; metrics are unstable and not a performance claim."
                                if a.evidence_tier == "exploratory_low_support" else
                                "Primary-support image-level development result; not clinical validation."),
            "pretrained": not a.no_pretrained,
            "pretraining_source": "self-supervised DINOv2 checkpoint" if a.architecture == "dinov2_vits14" else ("ImageNet" if not a.no_pretrained else "none"),
            "pretrained_model_id": ckpt.get("pretrained_model_id"),
            "pretrained_model_revision": ckpt.get("pretrained_model_revision"),
            "software_versions": {"torch": torch.__version__, "torchvision": __import__("torchvision").__version__},
            "input": "color fundus photograph", "input_size": a.size, "checkpoint_epoch": ckpt.get("epoch"),
            "split_support": support, "test_metrics": metrics(ty, ts),
            "patient_independence_verified": False, "image_ids_available": False,
            "checkpoint_sha256": hashlib.sha256(a.evaluate_checkpoint.read_bytes()).hexdigest(),
            "limitation": "Retrospective public-dataset experiment; not device-matched clinical validation or a clinical diagnosis.",
        }
        a.outdir.mkdir(parents=True, exist_ok=True)
        path = a.outdir / f"{a.target}_{a.architecture}_FINAL_TEST.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"summary": str(path), "test_metrics": report["test_metrics"]}), flush=True)
        return
    scaler = torch.amp.GradScaler("cuda"); best_auc = -1.; best_epoch = 0; stale = 0
    started = time.monotonic(); history = []
    for epoch in range(1, a.epochs + 1):
        model.train(); losses = []
        for x, y, _ in loaders["train"]:
            if time.monotonic() - started >= a.max_seconds: break
            x=x.to(device, non_blocking=True); y=y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss=criterion(model(x).flatten(), y)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); losses.append(float(loss.detach().cpu()))
        if not losses: break
        vy, vs, _ = predict(model, loaders["valid"], device); vm = metrics(vy, vs)
        score = vm["auroc"] if vm["auroc"] is not None else -1.
        record={"epoch":epoch,"train_loss":float(np.mean(losses)),"validation":vm}
        history.append(record); print(json.dumps(record), flush=True)
        if score > best_auc:
            best_auc=score; best_epoch=epoch; stale=0
            torch.save({"state_dict":model.state_dict(),"target":a.target,"architecture":a.architecture,
                        "evidence_tier":a.evidence_tier,"epoch":epoch,"size":a.size,"seed":a.seed,
                        "pretrained_model_id": DINOV2_MODEL_ID if a.architecture == "dinov2_vits14" else f"torchvision.models.{a.architecture}",
                        "pretrained_model_revision": DINOV2_REVISION if a.architecture == "dinov2_vits14" else "torchvision DEFAULT weights; exact package version recorded in run summary"},a.artifact)
        else: stale += 1
        if stale >= 5 or time.monotonic()-started >= a.max_seconds: break
    ckpt=torch.load(a.artifact,map_location=device,weights_only=False)
    report={"status":"RFMiD_VALIDATION_SELECTED_CANDIDATE_TEST_NOT_EVALUATED",
            "target":a.target,"architecture":a.architecture,"evidence_tier":a.evidence_tier,
            "support_caution":("Low-support exploratory candidate; validation and test estimates are unstable."
                               if a.evidence_tier == "exploratory_low_support" else
                               "Primary-support candidate on image-level splits; not clinical validation."),
            "pretrained":not a.no_pretrained,
            "pretraining_source":"self-supervised DINOv2 checkpoint" if a.architecture == "dinov2_vits14" else ("ImageNet" if not a.no_pretrained else "none"),
            "pretrained_model_id": DINOV2_MODEL_ID if a.architecture == "dinov2_vits14" else f"torchvision.models.{a.architecture}",
            "pretrained_model_revision": DINOV2_REVISION if a.architecture == "dinov2_vits14" else "torchvision DEFAULT weights; exact package version recorded below",
            "software_versions": {"torch": torch.__version__, "torchvision": __import__("torchvision").__version__},
            "input":"color fundus photograph","input_size":a.size,"seed":a.seed,"best_epoch_by_validation_auroc":best_epoch,
            "best_validation_auroc":best_auc,
            "split_support":support,"minimum_train_support":{"positive":a.min_positive,"negative":a.min_negative},
            "config":{"optimizer":"AdamW","learning_rate":2e-5,"weight_decay":2e-4,
                      "loss":"BCEWithLogitsLoss with square-root inverse-prevalence weight clipped to [1,10]",
                      "validation_selection":"AUROC; test split is not evaluated during candidate training",
                      "report_threshold":0.5},"history":history,"test_metrics":None,
            "patient_independence_verified":False,"image_ids_available":False,
            "checkpoint_sha256":hashlib.sha256(a.artifact.read_bytes()).hexdigest(),
            "limitation":"Retrospective public-dataset experiment; not device-matched clinical validation or a clinical diagnosis."}
    path=a.outdir/f"{a.target}_{a.architecture}_summary.json";path.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"summary":str(path),"best_epoch":best_epoch,"best_validation_auroc":best_auc,
                      "checkpoint_sha256":report["checkpoint_sha256"]}),flush=True)

if __name__ == "__main__": main()
