#!/usr/bin/env python3
"""Validation-only DINOv2 frozen-feature comparison for Zhejiang disease tasks.

Never reads or scores test rows. Outputs are validation-only development
summaries; all row-level manifests/scores remain on the VM.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             brier_score_loss, confusion_matrix, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class ImageRows(Dataset):
    def __init__(self, rows):
        self.rows = rows
        self.transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __len__(self): return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["image_path"]) as image:
            x = self.transform(image.convert("RGB"))
        return x, index


def val_eye_rows(rows, scores):
    grouped = {}
    for row, score in zip(rows, scores):
        if row["split"] != "validation":
            continue
        key = (row["patient_id"], row["eye_id"])
        rec = grouped.setdefault(key, {"patient_id":row["patient_id"],"eye_id":row["eye_id"],
                                       "target":int(row["target"]),"score":float(score)})
        if rec["target"] != int(row["target"]):
            raise ValueError("Conflicting label within eye")
        rec["score"] = max(rec["score"], float(score))
    return list(grouped.values())


def threshold_for_specificity(y, score, target=0.90):
    candidates = np.r_[np.unique(score), np.nextafter(np.max(score), np.inf)]
    feasible=[]
    for t in candidates:
        pred=score>=t
        tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
        spec=tn/max(tn+fp,1); sens=tp/max(tp+fn,1)
        if spec+1e-12>=target:
            feasible.append((sens,spec,float(t)))
    return max(feasible)[2]


def summarize_eye_level(rows, image_scores):
    eyes=val_eye_rows(rows,image_scores)
    y=np.asarray([r["target"] for r in eyes],dtype=int)
    score=np.asarray([r["score"] for r in eyes],dtype=float)
    threshold=threshold_for_specificity(y,score,0.90)
    pred=score>=threshold
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    patients={r["patient_id"] for r in eyes}
    return {
      "n_validation_eyes":int(len(y)),"positive_eyes":int(y.sum()),
      "negative_eyes":int(len(y)-y.sum()),"validation_patients":int(len(patients)),
      "accuracy_at_validation_selected_threshold":float(accuracy_score(y,pred)),
      "sensitivity_at_validation_selected_threshold":float(tp/max(tp+fn,1)),
      "specificity_at_validation_selected_threshold":float(tn/max(tn+fp,1)),
      "false_negatives":int(fn),"false_positives":int(fp),
      "auroc":float(roc_auc_score(y,score)),
      "auprc":float(average_precision_score(y,score)),
      "brier_score":float(brier_score_loss(y,score)),
      "threshold_selected_on_this_validation_set":float(threshold),
      "selection_note":"All these metrics are on the model-selection/threshold-selection validation set and are optimistic; use for candidate comparison only, not final performance claims."
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--task-dir",type=Path,required=True)
    ap.add_argument("--dinov2-source",type=Path,required=True)
    ap.add_argument("--output-dir",type=Path,required=True)
    ap.add_argument("--batch-size",type=int,default=16)
    ap.add_argument("--workers",type=int,default=4)
    ap.add_argument("--seed",type=int,default=20261121)
    args=ap.parse_args()
    task_files=sorted(args.task_dir.glob("*_vs_healthy.csv"))
    tasks={}; unique={}
    for path in task_files:
        with path.open(newline="",encoding="utf-8-sig") as f:
            rows=[r for r in csv.DictReader(f) if r.get("split") in {"train","validation"}]
        if not rows: raise ValueError(f"No train/validation rows in {path.name}")
        for r in rows:
            if not Path(r["image_path"]).is_file(): raise FileNotFoundError("Unresolved VM image path")
            unique[r["image_id"]]=r
        tasks[path.stem]=rows
    rows=list(unique.values())
    source=args.dinov2_source.resolve()
    model=torch.hub.load(str(source),"dinov2_vitb14",source="local",pretrained=True,trust_repo=True)
    model.eval();device=torch.device("cuda" if torch.cuda.is_available() else "cpu");model.to(device)
    for param in model.parameters(): param.requires_grad_(False)
    with torch.inference_mode():
        smoke=model.forward_features(torch.zeros(1,3,224,224,device=device))
    if not isinstance(smoke,dict) or smoke.get("x_norm_clstoken",torch.empty(0)).shape[-1]!=768:
        raise RuntimeError("Unexpected DINOv2 ViT-B/14 feature output")
    loader=DataLoader(ImageRows(rows),batch_size=args.batch_size,shuffle=False,
                      num_workers=args.workers,pin_memory=device.type=="cuda")
    features=np.empty((len(rows),768),dtype=np.float32)
    with torch.inference_mode():
        for x,indices in loader:
            x=x.to(device,non_blocking=True)
            with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=device.type=="cuda"):
                out=model.forward_features(x)["x_norm_clstoken"]
            features[indices.numpy()]=out.float().cpu().numpy()
    fidx={r["image_id"]:i for i,r in enumerate(rows)}
    weights=list((Path(torch.hub.get_dir())/"checkpoints").glob("dinov2_vitb14*.pth"))
    if len(weights)!=1: raise RuntimeError("Expected one cached official DINOv2 ViT-B/14 checkpoint")
    digest=hashlib.sha256()
    with weights[0].open("rb") as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b""):digest.update(chunk)
    output={"status":"VALIDATION_ONLY_CANDIDATE_COMPARISON","encoder":"Official DINOv2 ViT-B/14; frozen normalized CLS token",
            "checkpoint_sha256":digest.hexdigest(),"preprocessing":"Bicubic resize 256, center crop 224, ImageNet normalization",
            "probe":"Training-only StandardScaler + L2 LogisticRegression(C=1); class-balanced; fit on train rows only",
            "patient_grouping":"All images/eyes from one patient remain in a single partition; eye scores max-pooled",
            "model_selection":"Validation only; no test rows were loaded or scored.","tasks":{}}
    private=[]
    for task,task_rows in tasks.items():
        train=[r for r in task_rows if r["split"]=="train"]
        valid=[r for r in task_rows if r["split"]=="validation"]
        y=np.asarray([int(r["target"]) for r in train],dtype=int)
        x=np.stack([features[fidx[r["image_id"]]] for r in train])
        patient_count=Counter(r["patient_id"] for r in train)
        n_patients={c:len({r["patient_id"] for r in train if int(r["target"])==c}) for c in (0,1)}
        w=np.asarray([1.0/(n_patients[int(r["target"])]*patient_count[r["patient_id"]]) for r in train])
        scaler=StandardScaler().fit(x);xtr=scaler.transform(x)
        clf=LogisticRegression(C=1.0,solver="lbfgs",max_iter=2000,random_state=args.seed)
        clf.fit(xtr,y,sample_weight=w)
        xv=np.stack([features[fidx[r["image_id"]]] for r in valid])
        pos=list(clf.classes_).index(1)
        sv=clf.predict_proba(scaler.transform(xv))[:,pos]
        metrics=summarize_eye_level(valid,sv)
        output["tasks"][task]={"train_patients":int(len(patient_count)),"validation":metrics}
        args.output_dir.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(args.output_dir/f"{task}_head_private.npz",coef=clf.coef_.astype(np.float32),
                            intercept=clf.intercept_.astype(np.float32),scaler_mean=scaler.mean_.astype(np.float32),
                            scaler_scale=scaler.scale_.astype(np.float32))
        for r,score in zip(valid,sv): private.append({"task":task,"image_id":r["image_id"],"score":float(score)})
    output["boundary"]="Small single-source development validation only; results are used for candidate selection and threshold choice and are optimistic. Not external or clinical validation. Test rows were never loaded."
    out=args.output_dir/"validation_only_summary.json"
    out.write_text(json.dumps(output,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    with (args.output_dir/"validation_scores_private.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=("task","image_id","score"));writer.writeheader();writer.writerows(private)
    print(json.dumps(output,indent=2,ensure_ascii=False),flush=True)

if __name__=="__main__": main()
