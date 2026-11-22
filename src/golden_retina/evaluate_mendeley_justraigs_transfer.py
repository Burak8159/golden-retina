#!/usr/bin/env python3
"""One-shot glaucoma transfer check from Mendeley to JustRAIGS; aggregates only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


class Rows(Dataset):
    def __init__(self, items):
        self.items = items
        self.tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
            transforms.ToTensor(), transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        path, label = self.items[i]
        with Image.open(path) as im: x = self.tf(ImageOps.exif_transpose(im).convert('RGB'))
        return x, label


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--images',type=Path,required=True)
    ap.add_argument('--predictions',type=Path,required=True)
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--threshold',type=float,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    by_stem={}
    for p in a.images.rglob('*'):
        if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png','.tif','.tiff'}:
            by_stem.setdefault(p.stem,p)
    rows=[]; unmatched=0
    with a.predictions.open(newline='') as f:
        for r in csv.DictReader(f):
            p=by_stem.get(str(r['eye_id']).strip())
            if p is None: unmatched+=1; continue
            rows.append((p,int(r['label_rg'])))
    if not rows or len({y for _,y in rows})<2: raise RuntimeError('No matched two-class JustRAIGS test cohort')
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    m=models.convnext_tiny(weights=None); m.classifier[2]=nn.Linear(m.classifier[2].in_features,2); m.load_state_dict(ck['state_dict'],strict=True); m.cuda().eval()
    ys=[];ps=[]
    with torch.inference_mode():
        for x,y in DataLoader(Rows(rows),batch_size=64,num_workers=4,pin_memory=True):
            with torch.autocast('cuda',dtype=torch.float16): z=m(x.cuda(non_blocking=True))
            ys.extend(y.numpy().tolist()); ps.extend(torch.softmax(z.float(),1)[:,1].cpu().numpy().tolist())
    y=np.asarray(ys,dtype=np.int8); p=np.asarray(ps,dtype=np.float64); pred=p>=a.threshold
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    report={'status':'complete','source_train':'Mendeley 6ry9n79gb4.1 mixed-source, exact-image-group split',
      'target_evaluation':'JustRAIGS internal held-out image/fellow-eye group split already used in project; cross-source diagnostic, not pristine or official challenge test',
      'endpoint':'referable glaucoma (RG) vs non-referable glaucoma (NRG)',
      'checkpoint_sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),'threshold_from_mendeley_validation':a.threshold,
      'n':int(len(y)),'positive':int(y.sum()),'negative':int(len(y)-y.sum()),'unmatched_rows_private_count':int(unmatched),
      'accuracy':float(accuracy_score(y,pred)),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
      'sensitivity':float(tp/max(1,tp+fn)),'specificity':float(tn/max(1,tn+fp)),
      'false_negatives':int(fn),'false_positives':int(fp),'auroc':float(roc_auc_score(y,p)),
      'limitation':'The JustRAIGS cohort was previously used in project development. This is a new frozen model cross-source diagnostic only, not independent external validation. No row-level predictions or IDs are written.'}
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__': main()
