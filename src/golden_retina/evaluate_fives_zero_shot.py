#!/usr/bin/env python3
"""One-shot source-transfer evaluation of a frozen multiclass fundus model on FIVES.

Uses only the official FIVES test/Original directory and its filename class code.
No threshold fitting or model selection occurs on FIVES. Only aggregate metrics
are written; image paths and per-image predictions stay in the VM process.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

MEAN=(0.485,0.456,0.406); STD=(0.229,0.224,0.225)
SOURCE_TO_MODEL={"D":"Diabetic Retinopathy","G":"Glaucoma","N":"Healthy"}

class Images(Dataset):
    def __init__(self, root: Path):
        self.files=sorted(root.glob("*.png"))
        if not self.files: raise FileNotFoundError(f"No PNGs in {root}")
        self.tf=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
        self.codes=[]
        for p in self.files:
            code=p.stem.rsplit("_",1)[-1]
            if code not in {"A","D","G","N"}: raise ValueError(f"Unexpected FIVES class code: {p.name}")
            self.codes.append(code)
    def __len__(self): return len(self.files)
    def __getitem__(self,i):
        with Image.open(self.files[i]) as im: x=ImageOps.exif_transpose(im).convert("RGB")
        return self.tf(x), self.codes[i]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",type=Path,required=True)
    ap.add_argument("--test-original",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--batch-size",type=int,default=32)
    a=ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA L4 required")
    ck=torch.load(a.checkpoint,map_location="cpu",weights_only=False)
    classes=ck["classes"]; arch=ck["architecture"]
    if arch!="convnext_tiny": raise ValueError(f"Unsupported architecture {arch}")
    model=models.convnext_tiny(weights=None); model.classifier[2]=nn.Linear(model.classifier[2].in_features,len(classes)); model.load_state_dict(ck["state_dict"]); model.cuda().eval()
    ds=Images(a.test_original); dl=DataLoader(ds,batch_size=a.batch_size,shuffle=False,num_workers=4,pin_memory=True)
    probs=[]; codes=[]
    with torch.inference_mode():
        for x,code in dl:
            with torch.autocast("cuda",dtype=torch.float16): logits=model(x.cuda(non_blocking=True))
            probs.extend(torch.softmax(logits.float(),1).cpu().numpy()); codes.extend(code)
    p=np.asarray(probs); codes=np.asarray(codes)
    out={"dataset":"FIVES official test / Original only","source":"Figshare 19688169 v1","n_images":int(len(codes)),"class_counts":{c:int((codes==c).sum()) for c in "ADGN"},"architecture":arch,"checkpoint_epoch":int(ck.get("epoch",-1)),"threshold_policy":"fixed probability threshold 0.5; no FIVES threshold tuning; no model selection on FIVES","patient_ids_available_to_evaluator":False,"test_partition_used_for_any_selection":False,"metrics":{},"three_way_DRG_normal_accuracy":None,"boundary":"Exploratory source-transfer measurement. FIVES filenames provide mutually exclusive A/D/G/N categories. FIVES test images are source-level holdout, not patient-independent: the public split is image-based and patient identifiers are unavailable. Not clinical validation."}
    for code,cls in SOURCE_TO_MODEL.items():
        if cls not in classes: continue
        y=(codes==code).astype(int); score=p[:,classes.index(cls)]; pred=score>=.5
        tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
        out["metrics"][cls]={"positive_class_code":code,"n_positive":int(y.sum()),"n_negative":int((1-y).sum()),"threshold":0.5,"sensitivity":float(tp/(tp+fn)) if tp+fn else None,"specificity":float(tn/(tn+fp)) if tn+fp else None,"accuracy":float(accuracy_score(y,pred)),"false_negatives":int(fn),"false_positives":int(fp),"auroc":float(roc_auc_score(y,score)) if len(np.unique(y))==2 else None}
    sel=np.isin(codes,["D","G","N"]); ymap={"D":0,"G":1,"N":2}; cols=[classes.index(SOURCE_TO_MODEL[c]) for c in ("D","G","N")]
    y=np.asarray([ymap[c] for c in codes[sel]]); pred=p[sel][:,cols].argmax(1)
    out["three_way_DRG_normal_accuracy"]={"n_images":int(sel.sum()),"accuracy":float(accuracy_score(y,pred)),"macro_sensitivity":float(np.mean([((pred==k)&(y==k)).sum()/max(1,(y==k).sum()) for k in range(3)])),"confusion_matrix_rows_true_D_G_N_cols_pred_D_G_N":confusion_matrix(y,pred,labels=[0,1,2]).tolist(),"class_order":["Diabetic Retinopathy","Glaucoma","Healthy"],"AMD_excluded":int((codes=="A").sum())}
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(out,indent=2)+"\n"); print(json.dumps(out,indent=2))
if __name__=="__main__": main()
