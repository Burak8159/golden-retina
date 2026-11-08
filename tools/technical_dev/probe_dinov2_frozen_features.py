#!/usr/bin/env python3
"""Official DINOv2 ViT-B/14 frozen-feature probes on train/validation only."""
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, f1_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class FundusRows(Dataset):
    def __init__(self, rows):
        self.rows = rows
        self.transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        with Image.open(self.rows[i]["image_path"]) as im:
            x = self.transform(im.convert("RGB"))
        return x, i


def binary_metrics(y, p):
    y = y.astype(int); pred = (p >= 0.5).astype(int)
    return {"n": int(len(y)), "positives": int(y.sum()),
            "accuracy_at_0_5": float(accuracy_score(y, pred)),
            "balanced_accuracy_at_0_5": float(balanced_accuracy_score(y, pred)),
            "sensitivity_at_0_5": float((pred[y == 1] == 1).mean()) if (y == 1).any() else None,
            "specificity_at_0_5": float((pred[y == 0] == 0).mean()) if (y == 0).any() else None,
            "f1_macro_at_0_5": float(f1_score(y, pred, average="macro", zero_division=0)),
            "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
            "auprc": float(average_precision_score(y, p)) if (y == 1).any() else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--dinov2-source", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-iter", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20261107)
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    with args.manifest.open(newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r.get("split") in {"train", "valid"}]
    for r in rows:
        if not Path(r["image_path"]).is_file(): raise FileNotFoundError(r["image_path"])
    model = torch.hub.load(str(args.dinov2_source), "dinov2_vitb14", source="local",
                           pretrained=True, trust_repo=True)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for p in model.parameters(): p.requires_grad_(False)
    # Verify the selected official representation and runtime before the cohort pass.
    with torch.inference_mode():
        smoke = model.forward_features(torch.zeros(1, 3, 224, 224, device=device))
    if not isinstance(smoke, dict) or "x_norm_clstoken" not in smoke:
        raise RuntimeError(f"Unexpected DINOv2 output structure: {type(smoke)}")
    if smoke["x_norm_clstoken"].shape[-1] != 768:
        raise RuntimeError(f"Unexpected DINOv2 feature dimension: {smoke['x_norm_clstoken'].shape}")
    loader = DataLoader(FundusRows(rows), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=device.type == "cuda",
                        persistent_workers=args.workers > 0)
    features = np.zeros((len(rows), 768), dtype=np.float32); started = time.time()
    with torch.inference_mode():
        for x, indices in loader:
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                z = model.forward_features(x)["x_norm_clstoken"]
            features[indices.numpy()] = z.float().cpu().numpy()
            done = int(indices[-1]) + 1
            if done % 512 < len(indices) or done == len(rows):
                print(json.dumps({"phase":"embedding","done":done,"total":len(rows),
                                  "elapsed_s":round(time.time()-started,1)}),flush=True)
    split=np.asarray([r["split"] for r in rows]); train=split=="train"; valid=split=="valid"
    task_fns={"dr_grade_0_4":lambda r:int(r["dr_grade"]) if int(r["dr_grade"])>=0 else None,
              "referable_dr_ge2":lambda r:int(int(r["dr_grade"])>=2) if int(r["dr_grade"])>=0 else None,
              "ungradable_ddr":lambda r:int(r["ungradable"]) if int(r["ungradable"])>=0 else None,
              "glaucoma_hygd":lambda r:int(r["gon"]) if int(r["gon"])>=0 else None}
    # Hash the exact official cached checkpoint; do not export it.
    ckpts=list((Path(torch.hub.get_dir())/"checkpoints").glob("dinov2_vitb14*.pth"))
    if len(ckpts)!=1: raise RuntimeError(f"Expected one official cached DINOv2 ViT-B/14 checkpoint, found {[x.name for x in ckpts]}")
    digest=hashlib.sha256()
    with ckpts[0].open("rb") as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b""): digest.update(chunk)
    result={"status":"completed; development train/validation only; no internal test or Zhejiang rows loaded",
            "encoder":"Official DINOv2 ViT-B/14; frozen normalized CLS token", "checkpoint_file":ckpts[0].name,
            "checkpoint_sha256":digest.hexdigest(),"model_license":"Apache-2.0 per official facebookresearch/dinov2 repository",
            "dinov2_repository_commit":__import__("subprocess").check_output(["git","-C",str(args.dinov2_source),"rev-parse","HEAD"],text=True).strip(),
            "preprocessing":"ImageNet normalization; bicubic resize 256 and center crop 224",
            "probe":f"training-only StandardScaler + balanced LogisticRegression(C=1,max_iter={args.max_iter})",
            "manifest_sha256":hashlib.sha256(args.manifest.read_bytes()).hexdigest(),"rows_train_valid":len(rows),
            "train_rows":int(train.sum()),"validation_rows":int(valid.sum()),"embedding_dim":768,
            "feature_extraction_seconds":round(time.time()-started,2),"tasks":{}}
    private=[]
    for name,fn in task_fns.items():
        labels=[fn(r) for r in rows]; eligible=np.asarray([v is not None for v in labels]); tr=train&eligible; va=valid&eligible
        ytr=np.asarray([labels[i] for i in np.where(tr)[0]],dtype=int); yva=np.asarray([labels[i] for i in np.where(va)[0]],dtype=int)
        if len(np.unique(ytr))<2 or len(np.unique(yva))<2:
            result["tasks"][name]={"status":"insufficient class support","n_train":int(tr.sum()),"n_valid":int(va.sum())}; continue
        scaler=StandardScaler(); xtr=scaler.fit_transform(features[tr]); xva=scaler.transform(features[va])
        clf=LogisticRegression(max_iter=args.max_iter,class_weight="balanced",C=1.0,random_state=args.seed,solver="lbfgs")
        clf.fit(xtr,ytr); probs=clf.predict_proba(xva); by_source={}
        for source in sorted({rows[i]["source"] for i in np.where(va)[0]}):
            mask=np.asarray([bool(va[i] and rows[i]["source"]==source) for i in range(len(rows))]); yy=np.asarray([labels[i] for i in np.where(mask)[0]],dtype=int)
            if name=="dr_grade_0_4":
                pred=clf.predict(scaler.transform(features[mask])); by_source[source]={"n":int(mask.sum()),"accuracy":float(accuracy_score(yy,pred)),"balanced_accuracy":float(balanced_accuracy_score(yy,pred)),"macro_f1":float(f1_score(yy,pred,average="macro",zero_division=0))}
            else:
                pos=list(clf.classes_).index(1); by_source[source]=binary_metrics(yy,clf.predict_proba(scaler.transform(features[mask]))[:,pos])
        if name=="dr_grade_0_4":
            pred=clf.predict(xva); m={"n_train":int(tr.sum()),"n_valid":int(va.sum()),"accuracy":float(accuracy_score(yva,pred)),"balanced_accuracy":float(balanced_accuracy_score(yva,pred)),"macro_f1":float(f1_score(yva,pred,average="macro",zero_division=0)),"class_support_valid":{str(int(c)):int((yva==c).sum()) for c in sorted(set(yva))}}
        else:
            pos=list(clf.classes_).index(1); m={"n_train":int(tr.sum()),"n_valid":int(va.sum()),**binary_metrics(yva,probs[:,pos])}
        m["by_source"]=by_source; result["tasks"][name]=m
        for j,i in enumerate(np.where(va)[0]):
            score=float(probs[j,pos]) if name!="dr_grade_0_4" else float(clf.predict_proba(scaler.transform(features[[i]]))[0].max())
            private.append({"source":rows[i]["source"],"image_id":rows[i]["image_id"],"patient_id":rows[i]["patient_id"],"task":name,"truth":int(labels[i]),"score":score})
    result["elapsed_seconds_total"]=round(time.time()-started,2); args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/"aggregate_summary.json").write_text(json.dumps(result,indent=2)+"\n")
    with (args.output_dir/"private_validation_predictions.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["source","image_id","patient_id","task","truth","score"]);w.writeheader();w.writerows(private)
    np.save(args.output_dir/"private_embeddings.npy",features.astype(np.float16));print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__": main()
