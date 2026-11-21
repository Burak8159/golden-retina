"""Source-only exploratory RIM-ONE DL glaucoma run (random initialization).

This avoids any non-RIM-ONE image pretraining. The official hospital test was
already exposed by a prior exploratory run, so this script records an
exploratory re-evaluation and makes no confirmatory-test claim.
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
from PIL import Image, ImageOps
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

SEED = 8159
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class Images(Dataset):
    def __init__(self, rows, root, augment=False):
        self.rows, self.root = rows, Path(root)
        ops = [transforms.Resize(256), transforms.CenterCrop(224)]
        if augment:
            ops += [transforms.RandomHorizontalFlip(.5),
                    transforms.RandomRotation(8),
                    transforms.ColorJitter(brightness=.10, contrast=.10, saturation=.08)]
        ops += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
        self.tf = transforms.Compose(ops)

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        path, label = self.rows[i]
        with Image.open(self.root / path) as im:
            x = self.tf(ImageOps.exif_transpose(im).convert("RGB"))
        return x, torch.tensor(label, dtype=torch.float32)


def gather(root, part):
    out = []
    for name, label in (("normal", 0), ("glaucoma", 1)):
        out += [(p.relative_to(root).as_posix(), label)
                for p in sorted((root / part / name).iterdir()) if p.is_file()]
    if not out or {y for _, y in out} != {0, 1}:
        raise RuntimeError(f"Both classes required for {part}")
    return out


def infer(model, loader, device):
    model.eval(); logits=[]; labels=[]
    with torch.inference_mode():
        for x, y in loader:
            x=x.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16): z=model(x).squeeze(1)
            logits.extend(z.float().cpu().numpy().tolist()); labels.extend(y.numpy().tolist())
    return np.asarray(logits, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def calculate(y, p, threshold):
    pred=(p >= threshold).astype(int)
    tn, fp, fn, tp=confusion_matrix(y, pred, labels=[0,1]).ravel()
    return {"n":int(len(y)),"positive":int(y.sum()),"negative":int(len(y)-y.sum()),
            "threshold":float(threshold),"accuracy":float(accuracy_score(y,pred)),
            "sensitivity":float(tp/(tp+fn)),"specificity":float(tn/(tn+fp)),
            "false_negative":int(fn),"false_positive":int(fp),"true_positive":int(tp),
            "true_negative":int(tn),"auroc":float(roc_auc_score(y,p)),
            "average_precision":float(average_precision_score(y,p))}


def threshold90(y,p):
    candidates=[]
    for t in np.unique(np.r_[0.0,p,1.0]):
        m=calculate(y,p,t)
        if m["sensitivity"] >= .90: candidates.append((m["specificity"],m["accuracy"],-t,float(t),m))
    if candidates:
        best=max(candidates)
        return best[3],best[4],True
    options=[(calculate(y,p,t),float(t)) for t in np.unique(np.r_[0.0,p,1.0])]
    m,t=max(options,key=lambda z:(z[0]["sensitivity"],z[0]["specificity"]))
    return t,m,False


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-root",type=Path,required=True)
    ap.add_argument("--output-dir",type=Path,required=True)
    ap.add_argument("--epochs",type=int,default=30)
    ap.add_argument("--batch-size",type=int,default=16)
    ap.add_argument("--workers",type=int,default=4)
    args=ap.parse_args(); random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    if not torch.cuda.is_available(): raise RuntimeError("L4 CUDA required; refusing CPU fallback")
    args.output_dir.mkdir(parents=True,exist_ok=True)
    train_all=gather(args.data_root,"training_set"); test=gather(args.data_root,"test_set")
    tr,va=train_test_split(train_all,test_size=.20,random_state=SEED,stratify=[y for _,y in train_all])
    split_path=args.output_dir/"scratch_split_manifest_private.csv"
    with split_path.open("w",newline="") as f:
        w=csv.writer(f);w.writerow(("relative_path","label","split"));w.writerows((p,y,"train") for p,y in tr);w.writerows((p,y,"validation") for p,y in va);w.writerows((p,y,"hospital_test_exposed") for p,y in test)
    device=torch.device("cuda")
    train_loader=DataLoader(Images(tr,args.data_root,True),batch_size=args.batch_size,shuffle=True,num_workers=args.workers,pin_memory=True)
    val_loader=DataLoader(Images(va,args.data_root),batch_size=args.batch_size,shuffle=False,num_workers=args.workers,pin_memory=True)
    test_loader=DataLoader(Images(test,args.data_root),batch_size=args.batch_size,shuffle=False,num_workers=args.workers,pin_memory=True)
    model=models.resnet18(weights=None)
    model.fc=nn.Linear(model.fc.in_features,1);model.to(device)
    counts=np.bincount([y for _,y in tr],minlength=2)
    lossfn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([counts[0]/counts[1]],device=device))
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.03)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="max",factor=.5,patience=2,min_lr=1e-6)
    scaler=torch.amp.GradScaler("cuda")
    best_auc=-1;best_state=None;best_epoch=0;stale=0;hist=[]
    start=time.time()
    for ep in range(1,args.epochs+1):
        model.train();loss_sum=0;seen=0
        for x,y in train_loader:
            x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True);opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.float16): z=model(x).squeeze(1);loss=lossfn(z,y)
            scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);scaler.step(opt);scaler.update()
            loss_sum+=float(loss.detach())*len(y);seen+=len(y)
        vz,vy=infer(model,val_loader,device);vp=1/(1+np.exp(-np.clip(vz,-40,40)))
        va_auc=float(roc_auc_score(vy,vp));va_ap=float(average_precision_score(vy,vp));sched.step(va_auc)
        row={"epoch":ep,"train_loss":loss_sum/max(1,seen),"validation_auroc":va_auc,"validation_ap":va_ap,"lr":opt.param_groups[0]["lr"]};hist.append(row);print(json.dumps(row),flush=True)
        if va_auc>best_auc+1e-4:
            best_auc=va_auc;best_epoch=ep;stale=0;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            stale+=1
            if stale>=6:break
    if best_state is None:raise RuntimeError("No validation checkpoint")
    model.load_state_dict(best_state);vz,vy=infer(model,val_loader,device);vp=1/(1+np.exp(-np.clip(vz,-40,40)))
    th,vm,target=threshold90(vy,vp)
    # This partition was already opened in a non-compliant exploratory run; do
    # not claim this later score is an untouched or confirmatory test.
    tz,ty=infer(model,test_loader,device);tp=1/(1+np.exp(-np.clip(tz,-40,40)));tm=calculate(ty,tp,th)
    ckpt=args.output_dir/"rimone_resnet18_scratch_private.pt"
    torch.save({"state_dict":best_state,"architecture":"torchvision.resnet18","weights":"random initialization; no external image pretraining","threshold":th,"seed":SEED,"best_epoch":best_epoch,"input_size":224,"mean":MEAN,"std":STD},ckpt)
    predpath=args.output_dir/"scratch_test_predictions_private.csv"
    with predpath.open("w",newline="") as f:
        w=csv.writer(f);w.writerow(("relative_path","label","score","predicted_label","false_negative"))
        for (path,y),p in zip(test,tp):
            pred=int(p>=th);w.writerow((path,y,f"{float(p):.10f}",pred,int(y==1 and pred==0)))
    report={"status":"EXPLORATORY_RIMONE_HOSPITAL_RERUN_TEST_ALREADY_EXPOSED","task":"RIM-ONE DL glaucoma versus normal fundus image classification","training_data":"Only official RIM-ONE DL hospital training partition; random initialization; no non-RIM-ONE image pretraining.","counts":{"official_train":len(train_all),"internal_train":len(tr),"internal_validation":len(va),"hospital_test":len(test),"train_positive":int(sum(y for _,y in train_all)),"test_positive":int(sum(y for _,y in test))},"split":"Stratified image-level 80/20 split in HUC training images, seed 8159; patient identifiers absent; patient independence unverified.","selection":{"criterion":"highest internal-validation AUROC with early stopping; threshold maximizes specificity subject to validation sensitivity >=90%","best_epoch":best_epoch,"validation_auroc":best_auc,"validation_operating_point":vm,"target_sensitivity_feasible":target,"threshold":th},"hospital_partition_metrics_exploratory_only":tm,"checkpoint_sha256_private":sha256(ckpt),"split_manifest_sha256_private":sha256(split_path),"history":hist,"limitations":["The official hospital test labels/scores were already examined in an earlier ImageNet-pretrained exploratory run; this rerun is not an untouched confirmatory test.","No patient IDs are supplied; image-level intervals do not quantify patient-level uncertainty.","Small single-dataset retrospective result; not clinical validation or autonomous diagnostic evidence.","Dataset, row-level files, and checkpoint are private under source terms."]}
    out=args.output_dir/"scratch_summary_aggregate.json";out.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"summary":str(out),"best_epoch":best_epoch,"validation_auroc":best_auc,"validation_ap":hist[best_epoch-1]["validation_ap"],"threshold":th,"test_exploratory":tm,"elapsed_seconds":round(time.time()-start,1)},indent=2),flush=True)


if __name__=="__main__":main()
