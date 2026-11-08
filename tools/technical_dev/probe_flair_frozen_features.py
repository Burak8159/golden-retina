#!/usr/bin/env python3
"""Frozen FLAIR retinal image embeddings with development-only linear probes.

Uses the official FLAIR vision preprocessing (long edge mapped to 512 px,
zero-pad bottom/right, RGB in [0,1]) and the vision encoder/projection only.
No Zhejiang holdout rows are read. Raw paths and per-image outputs stay local.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, roc_auc_score, average_precision_score)
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Resize


class FLAIRImages(Dataset):
    def __init__(self, rows: list[dict[str, str]], image_size: int = 512):
        self.rows = rows
        self.image_size = image_size
        self.resize = Resize((1, 1), antialias=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i: int):
        row = self.rows[i]
        with Image.open(row["image_path"]) as image:
            arr = np.asarray(image.convert("RGB"), dtype=np.float32)
        if arr.max() > 0:
            arr /= 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1)
        h, w = x.shape[-2:]
        scale = max(h, w) / self.image_size
        size = (int(h / scale), int(w / scale))
        x = Resize(size, antialias=True)(x)
        x = torch.nn.functional.pad(x, (0, self.image_size-x.shape[-1], 0,
                                        self.image_size-x.shape[-2]))
        return x, i


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r.get("split") in {"train", "valid"}]
    for r in rows:
        p = Path(r["image_path"])
        if not p.is_file():
            raise FileNotFoundError(p)
    return rows


def binary_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y = y.astype(int)
    pred = (p >= 0.5).astype(int)
    out = {"n": int(len(y)), "positives": int(y.sum()),
           "accuracy_at_0_5": float(accuracy_score(y, pred)),
           "balanced_accuracy_at_0_5": float(balanced_accuracy_score(y, pred)),
           "sensitivity_at_0_5": float((pred[y == 1] == 1).mean()) if (y == 1).any() else None,
           "specificity_at_0_5": float((pred[y == 0] == 0).mean()) if (y == 0).any() else None,
           "f1_macro_at_0_5": float(f1_score(y, pred, average="macro", zero_division=0))}
    out["auroc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None
    out["auprc"] = float(average_precision_score(y, p)) if (y == 1).any() else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--flair-source", type=Path, required=True)
    ap.add_argument("--checkpoint", default="jusiro2/FLAIR",
                    help="Official Hugging Face model ID or local directory with config.json and model.safetensors")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20261107)
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    sys.path.insert(0, str(args.flair_source))
    from flair import FLAIRModel

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.manifest)
    split = np.array([r["split"] for r in rows])
    # The full FLAIR checkpoint contains its trained vision encoder. Disable
    # constructor-time ImageNet initialization so loading the checkpoint does
    # not trigger an unrelated torchvision weight download.
    model = FLAIRModel.from_pretrained(args.checkpoint, vision_pretrained=False)
    model.eval(); model.vision_model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.vision_model.to(device)
    loader = DataLoader(FLAIRImages(rows), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True,
                        persistent_workers=args.workers > 0)
    features = np.zeros((len(rows), model.vision_model.out_dim), dtype=np.float32)
    started = time.time(); done = 0
    with torch.inference_mode():
        for x, idx in loader:
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=device.type == "cuda"):
                z = model.vision_model(x)
            z = z.float().cpu().numpy()
            features[idx.numpy()] = z
            done += len(idx)
            if done % 512 < len(idx) or done == len(rows):
                print(json.dumps({"phase":"embedding","done":done,"total":len(rows),
                                  "elapsed_s":round(time.time()-started,1)}), flush=True)

    train = split == "train"; valid = split == "valid"
    tasks = {
        "dr_grade_0_4": lambda r: int(r["dr_grade"]) if int(r["dr_grade"]) >= 0 else None,
        "referable_dr_ge2": lambda r: int(int(r["dr_grade"]) >= 2) if int(r["dr_grade"]) >= 0 else None,
        "ungradable_ddr": lambda r: int(r["ungradable"]) if int(r["ungradable"]) >= 0 else None,
        "glaucoma_hygd": lambda r: int(r["gon"]) if int(r["gon"]) >= 0 else None,
    }
    checkpoint_path = Path(args.checkpoint)
    checkpoint_file = checkpoint_path / "model.safetensors" if checkpoint_path.is_dir() else None
    checkpoint_sha256 = None
    if checkpoint_file is not None and checkpoint_file.is_file():
        digest = hashlib.sha256()
        with checkpoint_file.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        checkpoint_sha256 = digest.hexdigest()
    results = {"status":"development train/validation only; no external test read",
               "encoder":"FLAIR jusiro2/FLAIR official pretrained model; frozen vision encoder plus projection",
               "checkpoint_reference":args.checkpoint,
               "checkpoint_sha256":checkpoint_sha256,
               "features":"512-dimensional projected normalized image embeddings",
               "preprocessing":"official FLAIR resize-long-edge-to-512, bottom/right zero pad, RGB [0,1]",
               "manifest_sha256":hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
               "n_train_valid_rows":len(rows),"train_rows":int(train.sum()),"validation_rows":int(valid.sum()),
               "feature_extraction_seconds":round(time.time()-started,2),"tasks":{}}
    private_preds=[]
    for name, label_fn in tasks.items():
        labs=[label_fn(r) for r in rows]
        eligible=np.array([v is not None for v in labs])
        tr=train & eligible; va=valid & eligible
        ytr=np.array([labs[i] for i in np.where(tr)[0]],dtype=int)
        yva=np.array([labs[i] for i in np.where(va)[0]],dtype=int)
        if len(np.unique(ytr)) < 2 or len(np.unique(yva)) < 2:
            results["tasks"][name]={"status":"insufficient class support","n_train":int(tr.sum()),"n_valid":int(va.sum())}
            continue
        clf=LogisticRegression(max_iter=1000,class_weight="balanced",C=1.0,
                               random_state=args.seed,solver="lbfgs")
        clf.fit(features[tr],ytr)
        pv=clf.predict_proba(features[va])
        source_res={}
        for src in sorted(set(r["source"] for i,r in enumerate(rows) if va[i])):
            idx=np.array([bool(va[i] and rows[i]["source"]==src) for i in range(len(rows))])
            if idx.sum()<2: continue
            if name=="dr_grade_0_4":
                pred=clf.predict(features[idx]); yy=np.array([labs[i] for i in np.where(idx)[0]])
                source_res[src]={"n":int(idx.sum()),"accuracy":float(accuracy_score(yy,pred)),
                                 "balanced_accuracy":float(balanced_accuracy_score(yy,pred)),
                                 "macro_f1":float(f1_score(yy,pred,average="macro",zero_division=0))}
            else:
                pos=list(clf.classes_).index(1)
                yy=np.array([labs[i] for i in np.where(idx)[0]])
                source_res[src]=binary_metrics(yy,clf.predict_proba(features[idx])[:,pos])
        if name=="dr_grade_0_4":
            pred=clf.predict(features[va])
            summary={"n_train":int(tr.sum()),"n_valid":int(va.sum()),
                     "accuracy":float(accuracy_score(yva,pred)),
                     "balanced_accuracy":float(balanced_accuracy_score(yva,pred)),
                     "macro_f1":float(f1_score(yva,pred,average="macro",zero_division=0)),
                     "class_support_valid":{str(int(c)):int((yva==c).sum()) for c in sorted(set(yva))},
                     "by_source":source_res}
        else:
            pos=list(clf.classes_).index(1)
            summary={"n_train":int(tr.sum()),"n_valid":int(va.sum()),
                     **binary_metrics(yva,pv[:,pos]),"by_source":source_res}
        results["tasks"][name]=summary
        # Private row-level validation predictions retained only on the VM.
        for j,i in enumerate(np.where(va)[0]):
            private_preds.append({"source":rows[i]["source"],"image_id":rows[i]["image_id"],
                                  "patient_id":rows[i]["patient_id"],"task":name,
                                  "truth":int(labs[i]),"score":float(pv[j,pos]) if name!="dr_grade_0_4" else float(clf.predict_proba(features[[i]])[0].max())})
    results["status"]="complete"; results["elapsed_seconds_total"]=round(time.time()-started,2)
    (args.output_dir/"aggregate_summary.json").write_text(json.dumps(results,indent=2)+"\n")
    with (args.output_dir/"private_validation_predictions.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(private_preds[0]) if private_preds else ["source","image_id","patient_id","task","truth","score"])
        w.writeheader();w.writerows(private_preds)
    # Embeddings are derived row-level data: retain only on private VM storage.
    np.save(args.output_dir/"private_embeddings.npy",features.astype(np.float16))
    print(json.dumps(results,indent=2),flush=True)


if __name__ == "__main__":
    main()
