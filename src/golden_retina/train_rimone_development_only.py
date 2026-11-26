#!/usr/bin/env python3
"""Train source-only RIM-ONE DL glaucoma candidates using training images only.

The official test directories are never enumerated. Random initialization and
within-dataset training comply with the source README's no-external-data rule.
Patient IDs are not exposed, so duplicate-image groups are the only split
clusters and cannot support patient-level claims.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

SEED = 8159
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def training_rows(root: Path) -> list[dict]:
    candidates = [p for p in root.rglob("training_set") if p.is_dir()]
    candidates = [p for p in candidates if (p / "normal").is_dir() and (p / "glaucoma").is_dir()]
    if not candidates:
        raise FileNotFoundError("Could not identify official RIM-ONE training_set/normal and training_set/glaucoma folders")
    # Prefer the hospital-held-out official variant when both variants are present.
    candidates.sort(key=lambda p: ("hospital" not in str(p.parent).lower(), len(str(p))))
    base = candidates[0]
    rows = []
    for class_name, label in (("normal", 0), ("glaucoma", 1)):
        for p in sorted((base / class_name).rglob("*")):
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
                rows.append({"path": p, "y": label, "sha": sha256(p)})
    if not rows or {r["y"] for r in rows} != {0, 1}:
        raise ValueError("RIM-ONE training partition must contain both classes")
    by_hash: dict[str, set[int]] = {}
    for r in rows:
        by_hash.setdefault(r["sha"], set()).add(r["y"])
    cross_label = sum(len(v) > 1 for v in by_hash.values())
    if cross_label:
        raise RuntimeError(f"Exact duplicate image hashes with conflicting labels: {cross_label}")
    for r in rows:
        r["group"] = r["sha"]
    return rows


class Fundus(Dataset):
    def __init__(self, rows, train=False, size=224):
        self.rows = rows
        ops = [transforms.Resize(256), transforms.CenterCrop(size)]
        if train:
            ops += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(8),
                    transforms.ColorJitter(.08, .08, .06, .01)]
        ops += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
        self.tf = transforms.Compose(ops)

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        with Image.open(r["path"]) as im:
            x = self.tf(ImageOps.exif_transpose(im).convert("RGB"))
        return x, torch.tensor(r["y"], dtype=torch.float32)


def make(name: str):
    if name == "resnet18":
        m = models.resnet18(weights=None); m.fc = nn.Linear(m.fc.in_features, 1)
    elif name == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=None); m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 1)
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None); m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 1)
    else: raise ValueError(name)
    return m


def infer(model, rows, batch, workers):
    dl = DataLoader(Fundus(rows, False), batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True)
    model.eval(); ys=[]; ps=[]
    with torch.inference_mode():
        for x, y in dl:
            with torch.autocast("cuda", dtype=torch.float16): z=model(x.cuda(non_blocking=True)).squeeze(1)
            ys.extend(y.numpy().astype(int).tolist()); ps.extend(torch.sigmoid(z.float()).cpu().numpy().tolist())
    return np.asarray(ys, dtype=int), np.asarray(ps, dtype=float)


def metrics(y, p, threshold):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"n_images": int(len(y)), "positive_images": int(y.sum()), "negative_images": int(len(y)-y.sum()),
            "threshold": float(threshold), "accuracy": float((pred == y).mean()),
            "sensitivity": float(tp/(tp+fn)) if tp+fn else None,
            "specificity": float(tn/(tn+fp)) if tn+fp else None,
            "false_negatives": int(fn), "false_positives": int(fp),
            "auroc": float(roc_auc_score(y,p)), "average_precision": float(average_precision_score(y,p))}


def threshold_at_sensitivity(y, p, target=.95):
    pos = p[y == 1]
    if not len(pos): return float("inf")
    q = max(0., 1-target)
    return float(np.nextafter(np.quantile(pos, q, method="lower"), -np.inf))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-root",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--epochs",type=int,default=12)
    ap.add_argument("--seconds-per-model",type=int,default=900)
    ap.add_argument("--batch-size",type=int,default=2)
    ap.add_argument("--workers",type=int,default=2)
    ap.add_argument("--size",type=int,default=224)
    ap.add_argument("--sensitivity-target",type=float,default=.95)
    ap.add_argument("--audit-only",action="store_true",help="Audit only the extracted official training partition; do not fit models")
    a=ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("RTX CUDA unavailable; refusing CPU fallback")
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    rows=training_rows(a.data_root)
    y=np.asarray([r["y"] for r in rows]); groups=np.asarray([r["group"] for r in rows])
    folds=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED).split(np.zeros(len(rows)),y,groups))
    # Fold 0 calibration, fold 1 validation, folds 2-4 training. The official
    # test partition is not included in rows, file enumeration, or this report.
    cal_i, val_i, train_i=folds[0][1], folds[1][1], np.concatenate([folds[2][1],folds[3][1],folds[4][1]])
    train=[rows[i] for i in train_i]; cal=[rows[i] for i in cal_i]; val=[rows[i] for i in val_i]
    split_groups={k:{r["group"] for r in v} for k,v in (("train",train),("calibration",cal),("validation",val))}
    overlaps={f"{x}_{z}":len(split_groups[x]&split_groups[z]) for x,z in (("train","calibration"),("train","validation"),("calibration","validation"))}
    if any(overlaps.values()): raise RuntimeError(f"duplicate group overlap: {overlaps}")
    if a.audit_only:
        def summary(part):
            yy=[r["y"] for r in part]
            return {"images":len(part),"exact_sha_groups":len({r["group"] for r in part}),"positive_images":int(sum(yy)),"negative_images":int(len(yy)-sum(yy))}
        audit={"status":"TRAINING_PARTITION_AUDIT_ONLY","source":"RIM-ONE DL official hospital-partition training folders","split":"StratifiedGroupKFold on exact-image SHA-256 groups; official test directories not extracted or read","counts":{"all_training_partition_images":len(rows),"all_exact_sha_groups":len(set(groups)),"exact_duplicate_images":int(len(rows)-len(set(groups))),"positive_training_images":int(y.sum()),"negative_training_images":int(len(y)-y.sum()),"fit":summary(train),"calibration":summary(cal),"validation":summary(val)},"group_overlap":overlaps,"patient_ids_available":False,"official_test_accessed":False,"row_level_paths_exported":False,"dataset_archive_sha256":"85aed5f95c794f52d11b6ed953032108a66576ffcff4b587365df467491b603d","limitation":"Exact-image grouping is not patient grouping; development only."}
        a.out.mkdir(parents=True,exist_ok=True);out=a.out/"training_partition_audit_aggregate.json";out.write_text(json.dumps(audit,indent=2)+"\n");print(json.dumps(audit),flush=True);return
    a.out.mkdir(parents=True,exist_ok=True); candidates=[]; started=time.time()
    pos=sum(r["y"] for r in train); neg=len(train)-pos
    pos_weight=torch.tensor([neg/max(pos,1)],device="cuda",dtype=torch.float32)
    train_dl=DataLoader(Fundus(train,True,a.size),batch_size=a.batch_size,shuffle=True,num_workers=a.workers,pin_memory=True)
    for arch in ("resnet18","mobilenet_v3_small","efficientnet_b0"):
        model=make(arch).cuda(); opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-3)
        best=-1.; best_epoch=0; stale=0; t0=time.time(); ck=a.out/f"{arch}_best_private.pt"
        for ep in range(1,a.epochs+1):
            model.train(); loss_sum=0.; seen=0
            for x, yy in train_dl:
                x=x.cuda(non_blocking=True); yy=yy.cuda(non_blocking=True); opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda",dtype=torch.float16): z=model(x).squeeze(1); loss=nn.functional.binary_cross_entropy_with_logits(z,yy,pos_weight=pos_weight)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step(); loss_sum+=float(loss.detach())*len(yy); seen+=len(yy)
            vy,vp=infer(model,val,a.batch_size*2,a.workers); auc=float(roc_auc_score(vy,vp)); apv=float(average_precision_score(vy,vp))
            print(json.dumps({"candidate":arch,"epoch":ep,"train_loss":loss_sum/max(1,seen),"validation_auroc":auc,"validation_ap":apv}),flush=True)
            if apv>best+1e-4:
                best=apv;best_epoch=ep;stale=0;torch.save({"state_dict":model.state_dict(),"architecture":arch,"epoch":ep,"input_size":a.size,"seed":SEED,"pretraining":"none; RIM-ONE-only per source terms"},ck)
            else: stale+=1
            if stale>=3 or time.time()-t0>=a.seconds_per_model: break
        candidates.append({"architecture":arch,"best_epoch":best_epoch,"validation_average_precision":best})
        del model;torch.cuda.empty_cache()
    chosen=max(candidates,key=lambda x:x["validation_average_precision"])
    ck=a.out/f"{chosen['architecture']}_best_private.pt"; state=torch.load(ck,map_location="cpu",weights_only=False)
    model=make(chosen["architecture"]);model.load_state_dict(state["state_dict"]);model.cuda()
    cy,cp=infer(model,cal,a.batch_size*2,a.workers); vy,vp=infer(model,val,a.batch_size*2,a.workers)
    th=threshold_at_sensitivity(cy,cp,a.sensitivity_target)
    report={"status":"DEVELOPMENT_ONLY_RIMONE_TRAINING_PARTITION","source":"official miag-ull/rim-one-dl repository; hospital-partition training set only when present","task":"source-labeled glaucoma versus normal fundus images","counts":{"training_partition_images":len(rows),"training_fit_images":len(train),"calibration_images":len(cal),"validation_images":len(val),"training_unique_exact_sha_groups":len(set(groups)),"positive_training_rows":int(y.sum()),"positive_fit":int(sum(r['y'] for r in train)),"positive_calibration":int(sum(r['y'] for r in cal)),"positive_validation":int(sum(r['y'] for r in val))},"patient_ids_available":False,"split":"StratifiedGroupKFold on exact-image SHA groups only; no official validation/test directory was opened. Group split is not patient split.","exact_sha_group_overlap":overlaps,"candidate_selection":"highest average precision on internal validation; no external pretraining and no outside dataset used, consistent with source README","candidates":candidates,"selected_model":chosen["architecture"],"selected_epoch":chosen["best_epoch"],"calibration_sensitivity_target":a.sensitivity_target,"calibration_threshold":th,"calibration_metrics":metrics(cy,cp,th),"validation_metrics_at_calibration_threshold":metrics(vy,vp,th),"official_test_accessed":False,"checkpoint_sha256_private":sha256(ck),"limitations":["No patient identifiers; duplicate-image grouping does not guarantee patient independence.","The held-out official test has been inspected in prior work and is not reopened.","Internal validation is small and selected; no independent external or clinical validation.","This is a research classifier, not an autonomous clinical diagnosis."],"elapsed_seconds":round(time.time()-started,1)}
    out=a.out/"aggregate_report.json";out.write_text(json.dumps(report,indent=2)+"\n");print(json.dumps({"summary":str(out),"selected_model":chosen["architecture"],"validation_metrics":report["validation_metrics_at_calibration_threshold"]}),flush=True)


if __name__=="__main__":main()
