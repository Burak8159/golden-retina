#!/usr/bin/env python3
"""Train disease-versus-normal specialists on Mendeley 6ry9n79gb4.1.

Uses exact-duplicate-grouped splits and an optional private cross-source hash
exclusion list. The archive has no patient IDs or source-to-image provenance;
all estimates are exploratory image-grouped development results.
"""
from __future__ import annotations
import argparse, hashlib, json, random, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

SEED=8159; EXT={'.jpg','.jpeg','.png','.tif','.tiff','.bmp'}
MEAN=(.485,.456,.406); STD=(.229,.224,.225)
TARGETS={
 'amd':['dry_amd','wet_amd'],
 'diabetic_retinopathy':['mild_dr','moderate_dr','severe_dr','very_severe_npdr','proliferative_dr','advanced_pdr'],
 'glaucoma':['glaucoma'],
 'cataract':['cataract'],
 'hypertensive_retinopathy':['hypertensive_retinopathy'],
 'pathological_myopia':['pathological_myopia'],
}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def scan(root,external):
 dirs=sorted(p for p in root.iterdir() if p.is_dir() and p.name.lower() not in {'pterygium','pt'})
 names=[p.name.lower() for p in dirs]; rows=[]; excluded=0
 for ci,d in enumerate(dirs):
  for p in d.rglob('*'):
   if p.is_file() and p.suffix.lower() in EXT:
    h=sha(p)
    if h in external: excluded+=1; continue
    rows.append({'path':p,'class':ci,'name':d.name.lower(),'group':h})
 outer=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED)
 y=np.array([r['class'] for r in rows]); g=np.array([r['group'] for r in rows])
 tv,te=next(outer.split(np.zeros(len(y)),y,g))
 inner=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED+1)
 tr,va=next(inner.split(np.zeros(len(tv)),y[tv],g[tv]))
 parts={'train':[rows[i] for i in tv[tr]],'validation':[rows[i] for i in tv[va]],'test':[rows[i] for i in te]}
 gs={k:{r['group'] for r in v} for k,v in parts.items()}
 if any(gs[a]&gs[b] for a,b in [('train','validation'),('train','test'),('validation','test')]): raise RuntimeError('exact duplicate leakage')
 return parts,names,excluded
class DS(Dataset):
 def __init__(self,rows,train):
  self.rows=rows; ops=[transforms.Resize(256),transforms.CenterCrop(224)]
  if train: ops += [transforms.RandomHorizontalFlip(),transforms.RandomRotation(8),transforms.ColorJitter(.08,.08,.06,.02)]
  ops += [transforms.ToTensor(),transforms.Normalize(MEAN,STD)]; self.tf=transforms.Compose(ops)
 def __len__(self): return len(self.rows)
 def __getitem__(self,i):
  r=self.rows[i]
  with Image.open(r['path']) as im: x=self.tf(ImageOps.exif_transpose(im).convert('RGB'))
  return x,r['label']
def model(arch,state):
 if arch=='convnext_tiny': m=models.convnext_tiny(weights=None);m.classifier[2]=nn.Linear(m.classifier[2].in_features,2)
 else: m=models.resnet50(weights=None);m.fc=nn.Linear(m.fc.in_features,2)
 own=m.state_dict();m.load_state_dict({k:v for k,v in state.items() if k in own and own[k].shape==v.shape},strict=False)
 return m.cuda()
def infer(m,dl):
 m.eval(); p=[];y=[]
 with torch.inference_mode():
  for x,z in dl:
   x=x.cuda(non_blocking=True)
   with torch.autocast('cuda',dtype=torch.float16): q=m(x)
   p.extend(torch.softmax(q.float(),1)[:,1].cpu().numpy());y.extend(z.numpy())
 return np.asarray(y,dtype=int),np.asarray(p)
def pack(y,p,t):
 z=p>=t;tn,fp,fn,tp=confusion_matrix(y,z,labels=[0,1]).ravel()
 return {'n':int(len(y)),'positive':int(y.sum()),'negative':int(len(y)-y.sum()),'threshold':float(t),'accuracy':float(accuracy_score(y,z)),'balanced_accuracy':float(balanced_accuracy_score(y,z)),'macro_f1':float(f1_score(y,z,average='macro',zero_division=0)),'sensitivity':float(tp/max(1,tp+fn)),'specificity':float(tn/max(1,tn+fp)),'false_negatives':int(fn),'false_positives':int(fp),'auroc':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None}
def threshold(y,p):
 y=np.asarray(y,dtype=np.int8);p=np.asarray(p,dtype=np.float64)
 candidates=np.r_[np.unique(p),np.nextafter(p.max(),np.inf)]
 pos=np.sort(p[y==1]);neg=np.sort(p[y==0])
 tp=pos.size-np.searchsorted(pos,candidates,side='left')
 fp=neg.size-np.searchsorted(neg,candidates,side='left')
 sens=tp/max(1,pos.size);spec=(neg.size-fp)/max(1,neg.size);bal=(sens+spec)/2
 best=np.lexsort((candidates,sens,bal))[-1]
 return float(candidates[best])
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--base-checkpoint',type=Path,required=True);ap.add_argument('--exclude-hash-file',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--epochs',type=int,default=5);ap.add_argument('--seconds-per-task',type=int,default=360);ap.add_argument('--batch-size',type=int,default=24);ap.add_argument('--workers',type=int,default=4);ap.add_argument('--negative-scope',choices=['healthy','all_other_labeled'],default='healthy');ap.add_argument('--max-per-split',type=int,default=6000);a=ap.parse_args()
 if not torch.cuda.is_available(): raise RuntimeError('CUDA L4 required')
 random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
 external=set(a.exclude_hash_file.read_text().split()); parts,names,excluded=scan(a.data,external); name_to_idx={n:i for i,n in enumerate(names)}
 if 'normal_fundus' not in name_to_idx: raise RuntimeError(f'normal_fundus absent: {names}')
 ck=torch.load(a.base_checkpoint,map_location='cpu',weights_only=False); base=ck['state_dict'];arch=ck['architecture']
 a.out.mkdir(parents=True,exist_ok=True); results={}; normal=name_to_idx['normal_fundus']
 for target,positive_names in TARGETS.items():
  positive_ids={name_to_idx[n] for n in positive_names if n in name_to_idx}
  if not positive_ids: results[target]={'status':'unsupported_label_folder','folders':positive_names}; continue
  subsets={}
  for split,rs in parts.items():
   chosen=[dict(r,label=int(r['class'] in positive_ids)) for r in rs if a.negative_scope=='all_other_labeled' or r['class'] in positive_ids or r['class']==normal]
   if not chosen or len({r['label'] for r in chosen})<2: raise RuntimeError(f'{target}: one class absent from {split}')
   if a.max_per_split>0 and len(chosen)>a.max_per_split:
    rng=np.random.default_rng(SEED+sum(map(ord,target))+{'train':0,'validation':1,'test':2}[split]);yy=np.asarray([r['label'] for r in chosen]);keep=[]
    for lab in (0,1):
     ii=np.flatnonzero(yy==lab);quota=round(a.max_per_split*len(ii)/len(chosen))
     if len(ii) and quota==0:quota=1
     keep.extend(rng.choice(ii,size=min(quota,len(ii)),replace=False).tolist())
    if len(keep)>a.max_per_split:keep=sorted(rng.choice(np.asarray(keep),size=a.max_per_split,replace=False).tolist())
    elif len(keep)<a.max_per_split:
     rest=np.setdiff1d(np.arange(len(chosen)),np.asarray(keep,dtype=int));keep.extend(rng.choice(rest,size=min(a.max_per_split-len(keep),len(rest)),replace=False).tolist())
    chosen=[chosen[i] for i in sorted(keep)]
   subsets[split]=chosen
  loaders={k:DataLoader(DS(v,k=='train'),batch_size=a.batch_size,shuffle=k=='train',num_workers=a.workers,pin_memory=True) for k,v in subsets.items()}
  path=a.out/f'{target}_{arch}_best_private.pt'
  if path.exists():
   best_ep=int(torch.load(path,map_location='cpu',weights_only=False).get('epoch',0))
   print(json.dumps({'task':target,'resume':'saved best checkpoint; training skipped','best_epoch':best_ep}),flush=True)
  else:
   m=model(arch,base);counts=Counter(r['label'] for r in subsets['train']);cw=torch.tensor([len(subsets['train'])/(2*counts[i]) for i in (0,1)],dtype=torch.float32,device='cuda');opt=torch.optim.AdamW(m.parameters(),lr=1.5e-5,weight_decay=3e-4)
   best=float('inf');best_ep=0;stale=0;start=time.time()
   for ep in range(1,a.epochs+1):
    m.train();loss_sum=0;seen=0
    for x,y in loaders['train']:
     x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True);opt.zero_grad(set_to_none=True)
     with torch.autocast('cuda',dtype=torch.float16): z=m(x);loss=nn.functional.cross_entropy(z,y,weight=cw)
     loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),2.);opt.step();loss_sum+=float(loss.detach())*len(y);seen+=len(y)
    vy,vp=infer(m,loaders['validation']);vl=float(nn.functional.cross_entropy(torch.tensor(np.stack([1-vp,vp],1).clip(1e-7,1-1e-7),dtype=torch.float32).log(),torch.tensor(vy)).item());vm=pack(vy,vp,.5)
    print(json.dumps({'task':target,'epoch':ep,'train_loss':loss_sum/max(1,seen),'validation_loss':vl,'validation_at_0_5':vm}),flush=True)
    if vl<best-1e-4:best,best_ep,stale=vl,ep,0;torch.save({'state_dict':m.state_dict(),'architecture':arch,'epoch':ep,'task':target},path)
    else: stale+=1
    if stale>=2 or time.time()-start>=a.seconds_per_task: break
   del m;torch.cuda.empty_cache()
  state=torch.load(path,map_location='cpu',weights_only=False)['state_dict'];m=model(arch,state)
  vy,vp=infer(m,loaders['validation']);th=threshold(vy,vp);ty,tp=infer(m,loaders['test']);results[target]={'status':'complete','positive_folders':positive_names,'normal_folder':'normal_fundus','positive_support_test':int(ty.sum()),'best_epoch':best_ep,'threshold_validation_balanced_accuracy':th,'validation':pack(vy,vp,th),'test':pack(ty,tp,th),'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest()};print(json.dumps({'finished':target,'metrics':results[target]['test']}),flush=True);del m;torch.cuda.empty_cache()
 report={'dataset':'Mendeley 6ry9n79gb4.1','architecture':arch,'classes':names,'negative_scope':a.negative_scope,'max_per_split':a.max_per_split,'cross_source_exact_hash_rows_excluded':excluded,'tasks':results,'data_boundary':'No patient IDs or source-to-image provenance; exact-hash groups only. The archive compiles three underlying data sources and image-grouped estimates are exploratory; no independent clinical validation.'}
 (a.out/'condition_specialists_aggregate.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':'complete','tasks':{k:v.get('test') for k,v in results.items()}},indent=2),flush=True)
if __name__=='__main__': main()
