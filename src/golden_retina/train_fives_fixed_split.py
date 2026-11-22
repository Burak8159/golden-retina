#!/usr/bin/env python3
"""Train a disease classifier on FIVES' official train partition.

Validation is carved only from official training data and grouped by exact
image hash. The official image-level test is scored once after model selection.
FIVES has no patient IDs in this pipeline, so results are explicitly exploratory
and not patient-independent. Images and per-image predictions stay on the VM.
"""
from __future__ import annotations

import argparse, hashlib, json, random, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

SEED=8159
CODES=("A","D","G","N")
NAMES=("AMD","Diabetic Retinopathy","Glaucoma","Healthy")
CODE_TO_LABEL=dict(zip(CODES,range(len(CODES))))
MEAN=(.485,.456,.406); STD=(.229,.224,.225)
EXTS={".png",".jpg",".jpeg"}

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def read_split(root: Path, split: str):
    folder=root/split/"Original"
    files=sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTS)
    rows=[]; bad=[]
    for p in files:
        code=p.stem.rsplit("_",1)[-1]
        if code not in CODE_TO_LABEL: raise ValueError(f"Unknown FIVES label code in {p.name}")
        try:
            with Image.open(p) as im: im.load()
            digest=sha(p)
        except Exception as exc:
            bad.append({"split":split,"status":"excluded_decode_error","error_type":type(exc).__name__})
            continue
        rows.append({"path":p,"label":CODE_TO_LABEL[code],"code":code,"sha":digest})
    seen={}; unique=[]; dup=0
    for r in rows:
        if r["sha"] in seen: dup+=1; continue
        seen[r["sha"]]=r["path"].name; unique.append(r)
    return unique,{"files_seen":len(files),"decode_failures":len(bad),"within_split_duplicate_rows_excluded":dup,"class_counts":dict(Counter(r["code"] for r in unique))}

class ImageRows(Dataset):
    def __init__(self,rows,size,train=False):
        self.rows=rows
        ops=[transforms.Resize((size,size))]
        if train: ops += [transforms.RandomHorizontalFlip(),transforms.RandomRotation(8),transforms.ColorJitter(.06,.06,.05,.01)]
        ops += [transforms.ToTensor(),transforms.Normalize(MEAN,STD)]
        self.tf=transforms.Compose(ops)
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        row=self.rows[i]
        with Image.open(row["path"]) as im: x=self.tf(ImageOps.exif_transpose(im).convert("RGB"))
        return x,row["label"]

def make(arch,n):
    if arch=="convnext_tiny":
        m=models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT);m.classifier[2]=nn.Linear(m.classifier[2].in_features,n)
    elif arch=="resnet50":
        m=models.resnet50(weights=models.ResNet50_Weights.DEFAULT);m.fc=nn.Linear(m.fc.in_features,n)
    else: raise ValueError(arch)
    return m

def predict(model,loader):
    model.eval(); ps=[];ys=[]
    with torch.inference_mode():
        for x,y in loader:
            with torch.autocast("cuda",dtype=torch.float16): z=model(x.cuda(non_blocking=True))
            ps.extend(torch.softmax(z.float(),1).cpu().numpy());ys.extend(y.numpy())
    return np.asarray(ps),np.asarray(ys)

def metrics(y,p):
    pred=p.argmax(1); cm=confusion_matrix(y,pred,labels=range(4)); out={"n_images":int(len(y)),"accuracy":float(accuracy_score(y,pred)),"confusion_matrix_rows_true_A_D_G_N_cols_pred_A_D_G_N":cm.tolist(),"class_order":list(NAMES),"per_disease":{}}
    for i,name in enumerate(NAMES):
        yy=(y==i).astype(int); pp=p[:,i]; pr=(pred==i); tn,fp,fn,tp=confusion_matrix(yy,pr,labels=[0,1]).ravel()
        out["per_disease"][name]={"support":int(yy.sum()),"sensitivity":float(tp/(tp+fn)) if tp+fn else None,"specificity":float(tn/(tn+fp)) if tn+fp else None,"false_negatives":int(fn),"false_positives":int(fp),"one_vs_rest_auroc":float(roc_auc_score(yy,pp)) if len(np.unique(yy))==2 else None,"one_vs_rest_average_precision":float(average_precision_score(yy,pp)) if yy.sum() else None}
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,required=True);ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--exclude-root",type=Path,action="append",default=[],help="Known source-training image roots; exact matches in FIVES test are excluded.")
    ap.add_argument("--image-size",type=int,default=384);ap.add_argument("--epochs",type=int,default=10);ap.add_argument("--batch-size",type=int,default=8);ap.add_argument("--workers",type=int,default=4);ap.add_argument("--seconds-per-model",type=int,default=2400)
    a=ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA L4 required")
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    tr_all,tr_audit=read_split(a.root,"train");test_all,test_audit=read_split(a.root,"test")
    tr_hashes={r["sha"] for r in tr_all}; external_hashes=set()
    for root in a.exclude_root:
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in EXTS:
                try: external_hashes.add(sha(p))
                except OSError: pass
    test=[]; cross_split_removed=0; external_removed=0
    for r in test_all:
        if r["sha"] in tr_hashes: cross_split_removed+=1;continue
        if r["sha"] in external_hashes: external_removed+=1;continue
        test.append(r)
    if len(test)<40 or any(not any(r["label"]==i for r in test) for i in range(4)):
        raise RuntimeError(f"Insufficient test support after exact duplicate exclusions: n={len(test)}")
    y=np.asarray([r["label"] for r in tr_all]);groups=np.asarray([r["sha"] for r in tr_all])
    folds=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED)
    tr_idx,va_idx=next(folds.split(np.zeros(len(y)),y,groups))
    train=[tr_all[i] for i in tr_idx];valid=[tr_all[i] for i in va_idx]
    if {r["sha"] for r in train}&{r["sha"] for r in valid}: raise RuntimeError("Exact duplicate crossed train/validation")
    loaders={k:DataLoader(ImageRows(v,a.image_size,k=="train"),batch_size=a.batch_size,shuffle=k=="train",num_workers=a.workers,pin_memory=True) for k,v in {"train":train,"validation":valid,"test":test}.items()}
    a.out.mkdir(parents=True,exist_ok=True)
    split_summary={"dataset":"FIVES v1","class_codes":dict(zip(CODES,NAMES)),"patient_ids_available":False,"official_source_partition":{"train":tr_audit,"test":test_audit},"model_train_counts":dict(Counter(r["code"] for r in train)),"validation_counts":dict(Counter(r["code"] for r in valid)),"test_counts_after_exclusions":dict(Counter(r["code"] for r in test)),"fives_train_exact_overlap_test_rows_excluded":cross_split_removed,"known_source_exact_overlap_test_rows_excluded":external_removed,"official_train_rows_used":len(tr_all),"derived_train_rows":len(train),"derived_validation_rows":len(valid),"test_rows_used":len(test),"split_boundary":"Validation is exact-hash-grouped within the official training partition. The official test remains image-level; patient IDs are unavailable. Exact duplicates crossing official train/test or matching any provided known source-training root are removed from test. Near-duplicates and same-patient leakage cannot be ruled out."}
    (a.out/"split_audit_aggregate.json").write_text(json.dumps(split_summary,indent=2)+"\n")
    results=[]
    for arch in ("convnext_tiny","resnet50"):
        model=make(arch,4).cuda();opt=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=2e-4);best=float("inf");best_epoch=0;stale=0;start=time.time();ck=a.out/f"{arch}_validation_best.pt"
        counts=np.asarray([sum(r["label"]==i for r in train) for i in range(4)],dtype=float); weights=(counts.mean()/counts)**.5; cw=torch.tensor(weights,dtype=torch.float32,device="cuda")
        for epoch in range(1,a.epochs+1):
            model.train();loss_sum=0.;seen=0
            for x,yb in loaders["train"]:
                x=x.cuda(non_blocking=True);yb=yb.cuda(non_blocking=True);opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda",dtype=torch.float16): z=model(x);loss=nn.functional.cross_entropy(z,yb,weight=cw)
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step();loss_sum+=loss.item()*len(yb);seen+=len(yb)
            vp,vy=predict(model,loaders["validation"]);vl=float(nn.functional.cross_entropy(torch.tensor(np.log(np.maximum(vp,1e-8)),dtype=torch.float32),torch.tensor(vy)).item())
            print(json.dumps({"architecture":arch,"epoch":epoch,"train_loss":loss_sum/max(1,seen),"validation_loss":vl,"validation":metrics(vy,vp)}),flush=True)
            if vl<best-1e-4:
                best=vl;best_epoch=epoch;stale=0;torch.save({"state_dict":model.state_dict(),"architecture":arch,"classes":list(NAMES),"epoch":epoch,"image_size":a.image_size},ck)
            else: stale+=1
            if stale>=3 or time.time()-start>=a.seconds_per_model: break
        results.append({"architecture":arch,"best_epoch":best_epoch,"best_validation_loss":best,"checkpoint":str(ck)})
        del model;torch.cuda.empty_cache()
    chosen=min(results,key=lambda z:z["best_validation_loss"])
    ck=torch.load(chosen["checkpoint"],map_location="cpu",weights_only=False);model=make(chosen["architecture"],4);model.load_state_dict(ck["state_dict"]);model.cuda()
    tp,ty=predict(model,loaders["test"])
    report={"status":"complete","dataset":"FIVES v1","selected_architecture":chosen["architecture"],"selected_epoch":chosen["best_epoch"],"selection":"Lowest validation cross-entropy; official test not used for model/epoch selection.","candidate_summaries":[{k:v for k,v in z.items() if k!="checkpoint"} for z in results],"test_metrics":metrics(ty,tp),"split_audit":split_summary,"boundary":"Exploratory same-source image-level holdout, not patient-independent. Exact duplicates were excluded, but patient and near-duplicate leakage are unresolved. Not clinical validation."}
    (a.out/"aggregate_report.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps({"status":"complete","selected_architecture":chosen["architecture"],"test_metrics":report["test_metrics"],"boundary":report["boundary"]}),flush=True)
if __name__=="__main__":main()
