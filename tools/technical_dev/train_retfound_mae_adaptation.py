#!/usr/bin/env python3
"""Partial supervised adaptation of the access-approved RETFound-MAE encoder.

Development-only, patient-disjoint train/validation run. No test rows are read.
The official pretrained weights remain private and are loaded with restricted
PyTorch unpickling (only argparse.Namespace is allow-listed).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             roc_auc_score, average_precision_score,
                             f1_score)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class Rows(Dataset):
    def __init__(self, rows, training):
        self.rows = rows
        if training:
            self.tf = transforms.Compose([
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomCrop(224), transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), transforms.Normalize((.485,.456,.406),(.229,.224,.225)),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224), transforms.ToTensor(),
                transforms.Normalize((.485,.456,.406),(.229,.224,.225)),
            ])

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        with Image.open(r['image_path']) as im:
            x = self.tf(im.convert('RGB'))
        return (x, int(r['dr_grade']), int(r['gon']), int(r['ungradable']), i)


class Adapted(nn.Module):
    def __init__(self, backbone, width=1024):
        super().__init__()
        self.backbone = backbone
        self.dr = nn.Linear(width, 5)
        self.gon = nn.Linear(width, 1)
        self.quality = nn.Linear(width, 1)
        self.refer = nn.Linear(width, 1)

    def forward(self, x):
        z = self.backbone.forward_features(x)
        return self.dr(z), self.refer(z).squeeze(-1), self.gon(z).squeeze(-1), self.quality(z).squeeze(-1)


def metrics(rows, probs, grade_logits, gon_logits, quality_logits):
    out = {}
    def binary(name, vals, label_key, eligible):
        ix = np.asarray([i for i,r in enumerate(rows) if eligible(r)], dtype=np.int64)
        if not len(ix):
            return {'status':'no validation labels for this task','n':0}
        y = np.array([int(rows[i][label_key]) for i in ix])
        p = vals[ix]
        if len(ix) and len(np.unique(y)) == 2:
            pred = p >= .5
            return {'n': int(len(ix)), 'positive': int(y.sum()), 'accuracy_at_0_5': float(accuracy_score(y,pred)),
                    'sensitivity_at_0_5': float((pred[y==1]).mean()), 'specificity_at_0_5': float((~pred[y==0]).mean()),
                    'auroc': float(roc_auc_score(y,p)), 'auprc': float(average_precision_score(y,p))}
        return {'status':'insufficient class support','n':int(len(ix))}
    ygrade=np.array([int(r['dr_grade']) for r in rows]); graded=ygrade>=0
    if graded.any():
        pred=grade_logits.argmax(1)
        out['dr_grade_0_4']={'n':int(graded.sum()),'accuracy':float(accuracy_score(ygrade[graded],pred[graded])),
                             'balanced_accuracy':float(balanced_accuracy_score(ygrade[graded],pred[graded])),
                             'macro_f1':float(f1_score(ygrade[graded],pred[graded],average='macro',zero_division=0))}
        pos=ygrade[graded]>=2; score=probs[graded]
        yy=pos.astype(int); pp=score[graded] if len(score)==len(ygrade) else probs[graded]
        pr=pp>=.5
        out['referable_dr_ge2']={'n':int(graded.sum()),'positive':int(yy.sum()),'accuracy_at_0_5':float(accuracy_score(yy,pr)),
                                 'sensitivity_at_0_5':float((pr[yy==1]).mean()),'specificity_at_0_5':float((~pr[yy==0]).mean()),
                                 'auroc':float(roc_auc_score(yy,pp)),'auprc':float(average_precision_score(yy,pp))}
    gonp=1/(1+np.exp(-gon_logits)); qualp=1/(1+np.exp(-quality_logits))
    out['glaucoma_hygd']=binary('glaucoma',gonp,'gon',lambda r:int(r['gon'])>=0)
    out['image_quality_ddr']=binary('quality',qualp,'ungradable',lambda r:int(r['ungradable'])>=0)
    out['by_source']={}
    for src in sorted(set(r['source'] for r in rows)):
        idx=[i for i,r in enumerate(rows) if r['source']==src and int(r['dr_grade'])>=0]
        if len(idx) and len(set((int(rows[i]['dr_grade'])>=2) for i in idx))==2:
            y=np.array([int(int(rows[i]['dr_grade'])>=2) for i in idx]); p=probs[idx]; pred=p>=.5
            out['by_source'][src]={'n':len(idx),'referable_dr_accuracy':float(accuracy_score(y,pred)),
                                  'referable_dr_auroc':float(roc_auc_score(y,p)),
                                  'referable_dr_sensitivity':float((pred[y==1]).mean()),
                                  'referable_dr_specificity':float((~pred[y==0]).mean())}
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest',type=Path,required=True); ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--retfound-source',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--epochs',type=int,default=12); ap.add_argument('--batch-size',type=int,default=8)
    ap.add_argument('--workers',type=int,default=6); ap.add_argument('--max-hours',type=float,default=2)
    ap.add_argument('--seed',type=int,default=20261108)
    args=ap.parse_args(); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    rows=list(csv.DictReader(args.manifest.open(newline='',encoding='utf-8-sig')))
    rows=[r for r in rows if r['split'] in {'train','valid'}]
    if any(r['split']=='test' for r in rows): raise ValueError('Test rows must not enter this run')
    tr=[r for r in rows if r['split']=='train']; va=[r for r in rows if r['split']=='valid']
    if not tr or not va: raise ValueError('Nonempty train and validation required')
    tp={r['patient_id'] for r in tr}; vp={r['patient_id'] for r in va}
    if tp & vp: raise ValueError(f'Patient leakage: {len(tp&vp)} IDs')
    for r in rows:
        if not Path(r['image_path']).is_file(): raise FileNotFoundError(r['image_path'])
    sys.path.insert(0,str(args.retfound_source.resolve()))
    import models_vit
    with torch.serialization.safe_globals([argparse.Namespace]):
        ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    backbone=models_vit.RETFound_mae(global_pool=False,num_classes=0,img_size=224)
    missing=backbone.load_state_dict(ckpt['model'],strict=False)
    if missing.missing_keys: raise RuntimeError(f'Missing pretrained weights: {missing.missing_keys[:10]}')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for p in backbone.parameters(): p.requires_grad_(False)
    for p in backbone.blocks[-1].parameters(): p.requires_grad_(True)
    for p in backbone.norm.parameters(): p.requires_grad_(True)
    model=Adapted(backbone).to(device)
    train_loader=DataLoader(Rows(tr,True),batch_size=args.batch_size,shuffle=True,num_workers=args.workers,pin_memory=True,persistent_workers=args.workers>0)
    val_loader=DataLoader(Rows(va,False),batch_size=args.batch_size,shuffle=False,num_workers=args.workers,pin_memory=True,persistent_workers=args.workers>0)
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW([{'params':model.backbone.blocks[-1].parameters(),'lr':1e-5},
       {'params':model.backbone.norm.parameters(),'lr':1e-5},
       {'params':list(model.dr.parameters())+list(model.refer.parameters())+list(model.gon.parameters())+list(model.quality.parameters()),'lr':3e-4}],weight_decay=1e-4)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    start=time.time(); history=[]; best_auc=-1; best_epoch=None
    for epoch in range(1,args.epochs+1):
        model.train(); model.backbone.blocks[:-1].eval(); losses=[]
        for x,g,gon,q,_ in train_loader:
            x=x.to(device,non_blocking=True); g=g.to(device); gon=gon.to(device); q=q.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16,enabled=device.type=='cuda'):
                gl,rl,gonl,ql=model(x); parts=[]
                m=g>=0
                if m.any():
                    parts.append(nn.functional.cross_entropy(gl[m],g[m]))
                    parts.append(nn.functional.binary_cross_entropy_with_logits(rl[m],(g[m]>=2).float()))
                m=gon>=0
                if m.any(): parts.append(nn.functional.binary_cross_entropy_with_logits(gonl[m],gon[m].float()))
                m=q>=0
                if m.any(): parts.append(nn.functional.binary_cross_entropy_with_logits(ql[m],q[m].float()))
                loss=torch.stack(parts).mean()
            scaler.scale(loss).backward(); scaler.unscale_(optimizer); nn.utils.clip_grad_norm_(params,1.0); scaler.step(optimizer); scaler.update(); losses.append(float(loss.detach().cpu()))
        model.eval(); gls=[]; rps=[]; gons=[]; quals=[]
        with torch.inference_mode():
            for x,g,gon,q,_ in val_loader:
                x=x.to(device,non_blocking=True)
                with torch.autocast('cuda',dtype=torch.float16,enabled=device.type=='cuda'):
                    gl,rl,gonl,ql=model(x)
                gls.append(gl.float().cpu().numpy()); rps.append(torch.sigmoid(rl.float()).cpu().numpy()); gons.append(gonl.float().cpu().numpy()); quals.append(ql.float().cpu().numpy())
        gls=np.concatenate(gls); rp=np.concatenate(rps); gonp=np.concatenate(gons); qualp=np.concatenate(quals)
        met=metrics(va,rp,gls,gonp,qualp)
        auc=met.get('referable_dr_ge2',{}).get('auroc',-1)
        rec={'epoch':epoch,'train_loss':float(np.mean(losses)),'validation':met,'elapsed_seconds':round(time.time()-start,1)}; history.append(rec); print(json.dumps(rec),flush=True)
        if auc>best_auc:
            best_auc=auc; best_epoch=epoch; torch.save({'state_dict':model.state_dict(),'epoch':epoch,'validation':met,'manifest_sha256':hashlib.sha256(args.manifest.read_bytes()).hexdigest(),'claim_boundary':'development validation only'},args.output_dir/'best.pt')
        (args.output_dir/'progress.json').write_text(json.dumps({'history':history,'best_epoch':best_epoch,'best_auroc':best_auc},indent=2)+'\n')
        if time.time()-start>=args.max_hours*3600: break
    result={'status':'DEVELOPMENT_TRAIN_VALID_ONLY','backbone':'RETFound-MAE ViT-L/16; last transformer block, final norm, and four task heads trainable; other backbone parameters frozen','checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),'manifest_sha256':hashlib.sha256(args.manifest.read_bytes()).hexdigest(),'train_rows':len(tr),'validation_rows':len(va),'patient_overlap':0,'test_rows_loaded':0,'trainable_parameters':sum(p.numel() for p in params),'loss':'DR 5-grade cross-entropy + referable DR BCE + masked glaucoma BCE + masked image-ungradable BCE; available task losses averaged per batch','best_epoch':best_epoch,'best_validation_auroc':best_auc,'history':history,'elapsed_seconds':round(time.time()-start,1),'warning':'Internal validation used for model selection; not independent clinical validation.'}
    (args.output_dir/'summary.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({'status':result['status'],'best_epoch':best_epoch,'best_validation':next((x['validation'] for x in history if x['epoch']==best_epoch),None)},indent=2),flush=True)

if __name__=='__main__': main()
