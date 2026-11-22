#!/usr/bin/env python3
"""Exploratory six-class UWF tumor model with exact-duplicate-grouped splits.

The source archive has no patient/eye manifest and includes repeated visits.
This program therefore reports image/duplicate-group metrics only; it must not
be described as patient-independent or clinical validation. All image paths,
manifests, checkpoints, and per-image outputs stay on the cloud VM.
"""
from __future__ import annotations

import argparse, hashlib, json, random, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

SEED=8159
MEAN=(.485,.456,.406); STD=(.229,.224,.225)
EXTS={".jpg",".jpeg",".png"}

def digest(p:Path)->str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()

def scan(root:Path):
    class_dirs=sorted(p for p in root.iterdir() if p.is_dir())
    names=[p.name for p in class_dirs]
    if len(names)!=6: raise ValueError(f"Expected six label folders, found {names}")
    rows=[];decode_errors=[]
    for label,folder in enumerate(class_dirs):
        for p in sorted(folder.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in EXTS:continue
            try:
                with Image.open(p) as im: im.load()
                h=digest(p)
            except Exception as exc:
                decode_errors.append({"class":folder.name,"error_type":type(exc).__name__})
                continue
            rows.append({"path":p,"label":label,"class_name":folder.name,"sha":h})
    by_hash=defaultdict(list)
    for i,r in enumerate(rows):by_hash[r["sha"]].append(i)
    conflicts={h for h,ids in by_hash.items() if len({rows[i]["label"] for i in ids})>1}
    excluded_conflict=sum(len(by_hash[h]) for h in conflicts)
    rows=[r for r in rows if r["sha"] not in conflicts]
    # A duplicated byte-identical image is one observation. Keep its first
    # deterministic path while recording duplicate counts in the audit.
    seen=set();unique=[];duplicate_rows=0;duplicate_groups=0
    for r in rows:
        if r["sha"] in seen:
            duplicate_rows+=1;continue
        seen.add(r["sha"]);unique.append(r)
    duplicate_groups=sum(len(v)>1 for h,v in by_hash.items() if h not in conflicts)
    if min(Counter(r["label"] for r in unique).values())<30:raise ValueError("Too little clean support for at least one class")
    summary={"class_names":names,"files_decodable":len(rows)+excluded_conflict,"decode_failures":len(decode_errors),"cross_label_exact_hash_images_excluded":excluded_conflict,"exact_duplicate_groups":duplicate_groups,"exact_duplicate_rows_excluded":duplicate_rows,"eligible_unique_images":len(unique),"eligible_class_counts":dict(Counter(r["class_name"] for r in unique)),"decode_error_classes":dict(Counter(x["class"] for x in decode_errors)),"patient_and_eye_ids_available":False,"source_split_files_available":False,"boundary":"Only exact duplicate groups are controlled. The source paper reports repeated images from the same eyes and a random image-level split; patient-level leakage cannot be ruled out."}
    return unique,names,summary

def split(rows):
    y=np.asarray([r["label"] for r in rows]);g=np.asarray([r["sha"] for r in rows]);
    outer=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED)
    trainval,test=next(outer.split(np.zeros(len(y)),y,g))
    ytv=y[trainval];gtv=g[trainval]
    inner=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED+1)
    tr,va=next(inner.split(np.zeros(len(ytv)),ytv,gtv))
    parts={"train":[rows[trainval[i]] for i in tr],"validation":[rows[trainval[i]] for i in va],"test":[rows[i] for i in test]}
    groups={k:{r["sha"] for r in v} for k,v in parts.items()}
    if any(groups[a]&groups[b] for a,b in (("train","validation"),("train","test"),("validation","test"))):raise RuntimeError("Exact-hash group leakage")
    return parts

class PadSquare:
    def __init__(self,size):self.size=size
    def __call__(self,im):
        im=ImageOps.exif_transpose(im).convert("RGB")
        scale=self.size/max(im.size); nw=max(1,round(im.width*scale));nh=max(1,round(im.height*scale))
        im=im.resize((nw,nh),Image.Resampling.BILINEAR)
        canvas=Image.new("RGB",(self.size,self.size),(0,0,0));canvas.paste(im,((self.size-nw)//2,(self.size-nh)//2));return canvas

class Images(Dataset):
    def __init__(self,rows,size,train=False):
        self.rows=rows;ops=[PadSquare(size)]
        if train:ops += [transforms.RandomHorizontalFlip(),transforms.RandomRotation(5),transforms.ColorJitter(.05,.05,.04,.01)]
        ops += [transforms.ToTensor(),transforms.Normalize(MEAN,STD)];self.tf=transforms.Compose(ops)
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        with Image.open(r["path"]) as im:x=self.tf(im)
        return x,r["label"]

def make(arch,n):
    if arch=="convnext_tiny":
        m=models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT);m.classifier[2]=nn.Linear(m.classifier[2].in_features,n)
    elif arch=="resnet50":
        m=models.resnet50(weights=models.ResNet50_Weights.DEFAULT);m.fc=nn.Linear(m.fc.in_features,n)
    else:raise ValueError(arch)
    return m

def evaluate(model,loader,names):
    model.eval();ys=[];ps=[]
    with torch.inference_mode():
        for x,y in loader:
            with torch.autocast("cuda",dtype=torch.float16):z=model(x.cuda(non_blocking=True))
            ys.extend(y.numpy());ps.extend(torch.softmax(z.float(),1).cpu().numpy())
    y=np.asarray(ys);p=np.asarray(ps);pred=p.argmax(1);cm=confusion_matrix(y,pred,labels=range(len(names)));per={};aucs=[];aps=[]
    for i,name in enumerate(names):
        yy=(y==i).astype(int);score=p[:,i];tn,fp,fn,tp=confusion_matrix(yy,pred==i,labels=[0,1]).ravel()
        auc=float(roc_auc_score(yy,score)) if len(np.unique(yy))==2 else None
        ap=float(average_precision_score(yy,score)) if yy.sum() else None
        if auc is not None:aucs.append(auc)
        if ap is not None:aps.append(ap)
        per[name]={"support":int(yy.sum()),"sensitivity":float(tp/(tp+fn)) if tp+fn else None,"specificity":float(tn/(tn+fp)) if tn+fp else None,"false_negatives":int(fn),"false_positives":int(fp),"one_vs_rest_auroc":auc,"one_vs_rest_average_precision":ap}
    return {"n_images":int(len(y)),"accuracy":float((pred==y).mean()),"macro_auroc":float(np.mean(aucs)) if aucs else None,"macro_average_precision":float(np.mean(aps)) if aps else None,"class_order":names,"confusion_matrix_rows_true_cols_pred":cm.tolist(),"per_disease":per}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--data",type=Path,required=True);ap.add_argument("--out",type=Path,required=True);ap.add_argument("--image-size",type=int,default=384);ap.add_argument("--epochs",type=int,default=10);ap.add_argument("--batch-size",type=int,default=8);ap.add_argument("--workers",type=int,default=4);ap.add_argument("--seconds-per-model",type=int,default=2400)
    a=ap.parse_args()
    if not torch.cuda.is_available():raise RuntimeError("CUDA L4 required")
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    rows,names,audit=scan(a.data);parts=split(rows);a.out.mkdir(parents=True,exist_ok=True)
    audit["split_counts"]={k:{"images":len(v),"class_counts":dict(Counter(r["class_name"] for r in v))} for k,v in parts.items()}
    (a.out/"split_audit_aggregate.json").write_text(json.dumps(audit,indent=2)+"\n")
    loaders={k:DataLoader(Images(v,a.image_size,k=="train"),batch_size=a.batch_size,shuffle=k=="train",num_workers=a.workers,pin_memory=True) for k,v in parts.items()}
    results=[]
    for arch in ("convnext_tiny","resnet50"):
        model=make(arch,len(names)).cuda();opt=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=2e-4);counts=np.asarray([sum(r["label"]==i for r in parts["train"]) for i in range(len(names))],dtype=float);cw=torch.tensor((counts.mean()/counts)**.5,dtype=torch.float32,device="cuda")
        best=-1.;best_epoch=0;stale=0;beg=time.time();ck=a.out/f"{arch}_validation_best.pt"
        for epoch in range(1,a.epochs+1):
            model.train();loss_sum=0.;seen=0
            for x,y in loaders["train"]:
                x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True);opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda",dtype=torch.float16):z=model(x);loss=nn.functional.cross_entropy(z,y,weight=cw)
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step();loss_sum+=loss.item()*len(y);seen+=len(y)
            v=evaluate(model,loaders["validation"],names);val=v["macro_average_precision"] or -1.
            print(json.dumps({"architecture":arch,"epoch":epoch,"train_loss":loss_sum/max(1,seen),"validation":v}),flush=True)
            if val>best+1e-4:
                best=val;best_epoch=epoch;stale=0;torch.save({"state_dict":model.state_dict(),"architecture":arch,"classes":names,"epoch":epoch,"image_size":a.image_size},ck)
            else:stale+=1
            if stale>=3 or time.time()-beg>=a.seconds_per_model:break
        results.append({"architecture":arch,"best_epoch":best_epoch,"validation_macro_average_precision":best,"checkpoint":str(ck)})
        del model;torch.cuda.empty_cache()
    chosen=max(results,key=lambda z:z["validation_macro_average_precision"])
    ck=torch.load(chosen["checkpoint"],map_location="cpu",weights_only=False);model=make(chosen["architecture"],len(names));model.load_state_dict(ck["state_dict"]);model.cuda()
    test=evaluate(model,loaders["test"],names)
    report={"status":"complete","dataset":"UWF Fundus Images of Intraocular Tumors · Figshare v1","classes":names,"eligible_unique_images":audit["eligible_unique_images"],"decode_failures_excluded":audit["decode_failures"],"exact_duplicate_groups":audit["exact_duplicate_groups"],"exact_duplicate_rows_excluded":audit["exact_duplicate_rows_excluded"],"cross_label_exact_hash_images_excluded":audit["cross_label_exact_hash_images_excluded"],"candidate_comparison":[{k:v for k,v in r.items() if k!="checkpoint"} for r in results],"selected_model":chosen["architecture"],"selected_epoch":chosen["best_epoch"],"validation_selection":"Highest validation macro-average-precision; exact-hash groups kept intact; test not used for selection.","test_metrics":test,"split_counts":audit["split_counts"],"patient_ids_available":False,"test_boundary":"Exploratory image/duplicate-group split only. The source contains repeat images from the same eyes/people, and no patient IDs were supplied. Patient-level leakage cannot be ruled out. UWF device/single-centre data; not clinical validation or standard-camera performance."}
    (a.out/"aggregate_report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"status":"complete","selected_model":report["selected_model"],"test_metrics":test,"boundary":report["test_boundary"]}),flush=True)
if __name__=="__main__":main()
