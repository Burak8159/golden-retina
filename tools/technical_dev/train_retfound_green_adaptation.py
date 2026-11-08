#!/usr/bin/env python3
"""Patient-grouped RETFound-Green partial adaptation; train/validation only."""
from __future__ import annotations
import argparse, csv, hashlib, json, random, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score, balanced_accuracy_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import v2 as T

class Rows(Dataset):
    def __init__(self, rows, train):
        self.rows=rows
        ops=[T.Resize((392,392),antialias=True)]
        if train: ops += [T.RandomHorizontalFlip(),T.RandomRotation(5),T.ColorJitter(brightness=.08,contrast=.08,saturation=.04)]
        ops += [T.ToImage(),T.ToDtype(torch.float32,scale=True),T.Normalize((.5,.5,.5),(.5,.5,.5))]
        self.tf=T.Compose(ops)
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        with Image.open(r['image_path']) as im: x=self.tf(im.convert('RGB'))
        q=int(r.get('ungradable',-1)) if r['source']=='ddr_hf_specificity300' else -1
        return x,int(r['dr_grade']),int(r['gon']),q,r['source'],r['patient_id']

class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, strength):
        ctx.strength=float(strength)
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad):
        return -ctx.strength*grad, None

class MultiTask(nn.Module):
    def __init__(self, backbone, dim, quality_aux=False, source_count=0, source_reversal=0.0):
        super().__init__(); self.backbone=backbone
        self.grade=nn.Linear(dim,5); self.refer=nn.Linear(dim,1); self.gon=nn.Linear(dim,1)
        self.quality=nn.Linear(dim,1) if quality_aux else None
        self.source_head=nn.Linear(dim,source_count) if source_count>1 else None
        self.source_reversal=float(source_reversal)
    def forward(self,x):
        z=self.backbone(x)
        if z.ndim>2: z=z.mean(dim=1)
        out=(self.grade(z),self.refer(z).squeeze(-1),self.gon(z).squeeze(-1))
        if self.quality is not None: out=(*out,self.quality(z).squeeze(-1))
        if self.source_head is not None: out=(*out,self.source_head(GradientReverse.apply(z,self.source_reversal)))
        return out

def getmetrics(rows, grade, refer, gon):
    out={}
    g=np.array([int(r['dr_grade']) for r in rows]); m=g>=0
    if m.any():
        pred=grade.argmax(1); out['dr_grade_0_4']={'n':int(m.sum()),'accuracy':float(accuracy_score(g[m],pred[m])),'balanced_accuracy':float(balanced_accuracy_score(g[m],pred[m])),'macro_f1':float(f1_score(g[m],pred[m],average='macro',zero_division=0))}
        y=(g[m]>=2).astype(int); p=refer[m]; q=p>=.5
        out['referable_dr_ge2']={'n':int(len(y)),'positive':int(y.sum()),'accuracy_at_0_5':float(accuracy_score(y,q)),'sensitivity_at_0_5':float(q[y==1].mean()),'specificity_at_0_5':float((~q[y==0]).mean()),'auroc':float(roc_auc_score(y,p)),'auprc':float(average_precision_score(y,p))}
    y=np.array([int(r['gon']) for r in rows]); m=y>=0
    if m.any():
        p=gon[m]; q=p>=.5; yy=y[m]
        out['glaucoma_hygd']={'n':int(len(yy)),'positive':int(yy.sum()),'accuracy_at_0_5':float(accuracy_score(yy,q)),'sensitivity_at_0_5':float(q[yy==1].mean()),'specificity_at_0_5':float((~q[yy==0]).mean()),'auroc':float(roc_auc_score(yy,p)),'auprc':float(average_precision_score(yy,p))}
    out['by_source']={}
    for s in sorted(set(r['source'] for r in rows)):
        ix=np.array([i for i,r in enumerate(rows) if r['source']==s and int(r['dr_grade'])>=0])
        if len(ix) and len(set((g[ix]>=2).astype(int)))==2:
            y=(g[ix]>=2).astype(int);p=refer[ix]
            out['by_source'][s]={'n':int(len(ix)),'referable_dr_auroc':float(roc_auc_score(y,p)),'accuracy_at_0_5':float(accuracy_score(y,p>=.5)),'sensitivity_at_0_5':float((p[y==1]>=.5).mean()),'specificity_at_0_5':float((p[y==0]<.5).mean())}
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--epochs',type=int,default=16);ap.add_argument('--batch-size',type=int,default=8);ap.add_argument('--workers',type=int,default=6);ap.add_argument('--max-hours',type=float,default=2);ap.add_argument('--seed',type=int,default=20261108)
    ap.add_argument('--include-train-only-ddr',action='store_true',help='Add labeled DDR train rows only; DDR has no validation/test rows and unverified IDs are excluded from split-overlap checks.')
    ap.add_argument('--quality-aux-weight',type=float,default=0.0,help='Train auxiliary gradability BCE on DDR training labels only; validation has no quality labels, so only downstream disease metrics are evaluated.')
    ap.add_argument('--source-invariance-weight',type=float,default=0.0,help='Adversarially reduce source predictability; fixed coefficient is the gradient-reversal strength and validation also reports source balanced accuracy.')
    a=ap.parse_args(); random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    allrows=list(csv.DictReader(a.manifest.open(newline='',encoding='utf-8-sig')))
    if any(r['split']=='test' for r in allrows): raise RuntimeError('Test rows present: abort before reading them')
    rows=[r for r in allrows if r['split'] in ('train','valid') and r['source']!='ddr_hf_specificity300' and r['patient_group_verified'].lower()=='true']
    if a.include_train_only_ddr:
        rows.extend(r for r in allrows if r['source']=='ddr_hf_specificity300' and r['split']=='train' and (int(r['dr_grade'])>=0 or r.get('ungradable','-1') in ('0','1')))
    tr=[r for r in rows if r['split']=='train']; va=[r for r in rows if r['split']=='valid']
    if not tr or not va: raise RuntimeError('Empty train/validation split')
    tp={r['patient_id'] for r in tr if r['patient_group_verified'].lower()=='true'}
    vp={r['patient_id'] for r in va if r['patient_group_verified'].lower()=='true'}
    if tp&vp: raise RuntimeError(f'Patient overlap: {len(tp&vp)}')
    for r in rows:
        if not Path(r['image_path']).is_file(): raise FileNotFoundError(r['image_path'])
    import timm
    # Match the official RETFound-Green recipe: load the released token-pooling
    # checkpoint first, then set global_pool='avg' (fc_norm remains Identity).
    backbone=timm.create_model('vit_small_patch14_reg4_dinov2',img_size=392,num_classes=0)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    if isinstance(ck,dict) and 'model' in ck: ck=ck['model']
    if isinstance(ck,dict) and 'state_dict' in ck: ck=ck['state_dict']
    if not isinstance(ck,dict): raise TypeError('Checkpoint is not a state dictionary')
    ck={k.removeprefix('module.'):v for k,v in ck.items() if isinstance(v,torch.Tensor)}
    incompat=backbone.load_state_dict(ck,strict=False)
    missing=[k for k in incompat.missing_keys if not k.startswith(('head.','fc_norm.'))]
    unexpected=[k for k in incompat.unexpected_keys if not any(x in k for x in ('head','fc_norm'))]
    if missing or unexpected: raise RuntimeError(f'Encoder checkpoint mismatch; missing={missing[:8]} unexpected={unexpected[:8]}')
    backbone.global_pool='avg'
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for p in backbone.parameters():p.requires_grad_(False)
    for p in backbone.blocks[-1].parameters():p.requires_grad_(True)
    for name in ('norm','fc_norm'):
        mod=getattr(backbone,name,None)
        if mod is not None:
            for p in mod.parameters():p.requires_grad_(True)
    # DDR is train-only in this manifest. Keep it out of the adversarial
    # source classifier because there are no DDR validation rows with which
    # to assess source predictability; DDR still contributes disease labels.
    source_domains=sorted({r['source'] for r in tr if r['source']!='ddr_hf_specificity300'})
    source_to_idx={s:i for i,s in enumerate(source_domains)} if a.source_invariance_weight>0 else {}
    if source_to_idx and not {r['source'] for r in va}.issubset(source_to_idx):
        raise RuntimeError('Every validation source must be represented in the source-adversarial training classes')
    model=MultiTask(backbone,backbone.num_features,quality_aux=a.quality_aux_weight>0,source_count=len(source_to_idx),source_reversal=a.source_invariance_weight).to(device)
    train_ds=Rows(tr,True);valid_ds=Rows(va,False)
    # Balance source and referable status without splitting patients.
    key=[(r['source'],int(r['dr_grade'])>=2 if int(r['dr_grade'])>=0 else -1,int(r['gon']) if int(r['gon'])>=0 else -1) for r in tr]
    counts={k:key.count(k) for k in set(key)}; weights=torch.tensor([1/counts[k] for k in key],dtype=torch.double)
    sampler_gen=torch.Generator().manual_seed(a.seed+100)
    loader_gen=torch.Generator().manual_seed(a.seed+200)
    sampler=WeightedRandomSampler(weights,len(weights),replacement=True,generator=sampler_gen)
    tl=DataLoader(train_ds,batch_size=a.batch_size,sampler=sampler,num_workers=a.workers,pin_memory=True,persistent_workers=a.workers>0,generator=loader_gen)
    vl=DataLoader(valid_ds,batch_size=a.batch_size,shuffle=False,num_workers=a.workers,pin_memory=True,persistent_workers=a.workers>0,generator=torch.Generator().manual_seed(a.seed+300))
    enc_params=[p for p in model.backbone.parameters() if p.requires_grad]; head_params=list(model.grade.parameters())+list(model.refer.parameters())+list(model.gon.parameters())
    if model.quality is not None: head_params += list(model.quality.parameters())
    if model.source_head is not None: head_params += list(model.source_head.parameters())
    opt=torch.optim.AdamW([{'params':enc_params,'lr':2e-5},{'params':head_params,'lr':3e-4}],weight_decay=1e-4)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda');a.output_dir.mkdir(parents=True,exist_ok=True)
    start=time.time();history=[];best=-1;best_epoch=None
    for epoch in range(1,a.epochs+1):
        model.train();model.backbone.blocks[:-1].eval();losses=[]
        for x,g,gon,q,src,pid in tl:
            x=x.to(device,non_blocking=True);g=g.to(device);gon=gon.to(device);q=q.to(device);sid=torch.tensor([source_to_idx.get(s,-1) for s in src],device=device) if source_to_idx else None;opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16,enabled=device.type=='cuda'):
                outputs=model(x);gl,rl,gn=outputs[:3];idx=3;ql=None;slog=None
                if model.quality is not None: ql=outputs[idx];idx+=1
                if model.source_head is not None: slog=outputs[idx]
                parts=[];m=g>=0
                if m.any():parts.extend([nn.functional.cross_entropy(gl[m],g[m]),nn.functional.binary_cross_entropy_with_logits(rl[m],(g[m]>=2).float())])
                m=gon>=0
                if m.any():parts.append(nn.functional.binary_cross_entropy_with_logits(gn[m],gon[m].float()))
                loss=torch.stack(parts).mean() if parts else (ql.sum()*0 if ql is not None else torch.zeros((),device=device,requires_grad=True))
                qm=q>=0
                if a.quality_aux_weight>0 and qm.any() and ql is not None:loss=loss+a.quality_aux_weight*nn.functional.binary_cross_entropy_with_logits(ql[qm],q[qm].float())
                source_active=False
                if model.source_head is not None and slog is not None:
                    source_mask=sid>=0
                    source_active=bool(source_mask.any())
                    if source_active: loss=loss+nn.functional.cross_entropy(slog[source_mask],sid[source_mask])
                quality_active=bool(a.quality_aux_weight>0 and qm.any() and ql is not None)
                if not parts and not quality_active and not source_active:
                    continue
            scaler.scale(loss).backward();scaler.unscale_(opt);nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0);scaler.step(opt);scaler.update();losses.append(float(loss.detach().cpu()))
        model.eval();glog=[];rlog=[];nlog=[];slog_all=[]
        with torch.inference_mode():
            for x,g,gon,q,src,pid in vl:
                x=x.to(device,non_blocking=True)
                with torch.autocast('cuda',dtype=torch.float16,enabled=device.type=='cuda'): outputs=model(x);gl,rl,gn=outputs[:3];idx=3
                if model.quality is not None: idx+=1
                if model.source_head is not None: slog_all.append(outputs[idx].float().cpu().numpy())
                glog.append(gl.float().cpu().numpy());rlog.append(torch.sigmoid(rl.float()).cpu().numpy());nlog.append(torch.sigmoid(gn.float()).cpu().numpy())
        gl=np.concatenate(glog);rp=np.concatenate(rlog);gp=np.concatenate(nlog);met=getmetrics(va,gl,rp,gp)
        if slog_all:
            from sklearn.metrics import balanced_accuracy_score
            sy=np.array([source_to_idx[r['source']] for r in va]);sp=np.concatenate(slog_all).argmax(1);met['source_predictability']={'balanced_accuracy':float(balanced_accuracy_score(sy,sp)),'source_classes':source_to_idx}
        # Lead task is DR discrimination; glaucoma is separately monitored.
        score=met.get('referable_dr_ge2',{}).get('auroc',-1)
        rec={'epoch':epoch,'train_loss':float(np.mean(losses)),'validation':met,'elapsed_seconds':round(time.time()-start,1)};history.append(rec);print(json.dumps(rec),flush=True)
        if score>best:
            best=score;best_epoch=epoch
            torch.save({'state_dict':model.state_dict(),'epoch':epoch,'validation':met,'source_to_idx':source_to_idx,'source_reversal':a.source_invariance_weight,'quality_aux_weight':a.quality_aux_weight,'manifest_sha256':hashlib.sha256(a.manifest.read_bytes()).hexdigest(),'claim_boundary':'internal development only; train/validation only'},a.output_dir/'best.pt')
        (a.output_dir/'progress.json').write_text(json.dumps({'history':history,'best_epoch':best_epoch,'best_auroc':best},indent=2)+'\n')
        if time.time()-start>=a.max_hours*3600:break
    summary={'status':'DEVELOPMENT_TRAIN_VALID_ONLY','backbone':'RETFound-Green ViT-S/14 reg4 DINOv2 architecture at 392px; last transformer block, final norms and DR/glaucoma heads trainable; optional auxiliary quality and source-adversarial heads; all other encoder parameters frozen','train_rows':len(tr),'train_rows_by_source':{s:sum(r['source']==s for r in tr) for s in sorted({r['source'] for r in tr})},'validation_rows':len(va),'validation_rows_by_source':{s:sum(r['source']==s for r in va) for s in sorted({r['source'] for r in va})},'include_train_only_ddr':bool(a.include_train_only_ddr),'quality_aux_weight':a.quality_aux_weight,'quality_supervised_train_rows':sum(r['source']=='ddr_hf_specificity300' and r.get('ungradable','-1') in ('0','1') for r in tr) if a.quality_aux_weight>0 else 0,'quality_validation_rows':sum(r.get('ungradable','-1') in ('0','1') for r in va),'source_invariance_weight':a.source_invariance_weight,'source_to_idx':source_to_idx,'sources':sorted(set(r['source'] for r in rows)),'train_verified_patients':len(tp),'validation_verified_patients':len(vp),'patient_overlap':0,'test_rows_loaded':0,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'loss':'masked DR 5-grade cross entropy + referable DR BCE + masked glaucoma BCE; optional masked DDR quality BCE and gradient-reversed source CE; source/label group-balanced sampling','selection_metric':'validation referable-DR AUROC; glaucoma tracked separately; quality validation unavailable','checkpoint_sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),'manifest_sha256':hashlib.sha256(a.manifest.read_bytes()).hexdigest(),'best_epoch':best_epoch,'best_validation':next((r['validation'] for r in history if r['epoch']==best_epoch),None),'history':history,'elapsed_seconds':round(time.time()-start,1),'warning':'Internal development validation used for checkpoint selection. Not independent clinical validation. RETFound-Green terms restrict this to non-commercial research.'}
    (a.output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({'status':summary['status'],'best_epoch':best_epoch,'best_validation':summary['best_validation']},indent=2),flush=True)
if __name__=='__main__':main()
