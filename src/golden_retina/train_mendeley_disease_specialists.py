#!/usr/bin/env python3
"""Fit per-class-versus-all-other-label specialists on frozen image groups.

The source provides no patient IDs. The script reconstructs the same exact-
duplicate-grouped split used by train_mendeley_folder_multiclass.py. A supplied
joint checkpoint is transferred only into each specialist's training fold;
validation selects thresholds, and the held-out image-group test is evaluated
once per prespecified specialist. Results are not patient-independent or
clinical validation; after the joint model opens the same test, these follow-on
specialist results are exploratory and must not be treated as a fresh test.
"""
from __future__ import annotations
import argparse,hashlib,json,random,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageOps
from sklearn.metrics import accuracy_score,balanced_accuracy_score,confusion_matrix,roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader,Dataset
from torchvision import models,transforms

SEED=8159;EXT={'.jpg','.jpeg','.png','.tif','.tiff'};MEAN=(.485,.456,.406);STD=(.229,.224,.225)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def load(root):
 dirs=sorted(p for p in root.iterdir() if p.is_dir() and p.name.lower() not in {'pterygium','pt'})
 rows=[]; hashes=defaultdict(list)
 for ci,d in enumerate(dirs):
  for p in d.rglob('*'):
   if p.is_file() and p.suffix.lower() in EXT:
    h=sha(p);r={'path':p,'cls':ci,'name':d.name,'group':h};hashes[h].append(len(rows));rows.append(r)
 bad=set()
 for ids in hashes.values():
  if len({rows[i]['cls'] for i in ids})>1:bad.update(ids)
 rows=[r for i,r in enumerate(rows) if i not in bad]
 return rows,[d.name for d in dirs]
def split(rows):
 y=np.array([r['cls'] for r in rows]);g=np.array([r['group'] for r in rows]);outer=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED)
 tv,te=next(outer.split(np.zeros(len(y)),y,g));inner=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED+1);tr,va=next(inner.split(np.zeros(len(tv)),y[tv],g[tv]))
 out={'train':[rows[i] for i in tv[tr]],'validation':[rows[i] for i in tv[va]],'test':[rows[i] for i in te]}
 gs={k:{r['group'] for r in v} for k,v in out.items()}
 if any(gs[a]&gs[b] for a,b in [('train','validation'),('train','test'),('validation','test')]):raise RuntimeError('duplicate split leakage')
 return out
class DS(Dataset):
 def __init__(self,rows,train):
  self.rows=rows;ops=[transforms.Resize(256),transforms.CenterCrop(224)]
  if train:ops += [transforms.RandomHorizontalFlip(),transforms.RandomRotation(8),transforms.ColorJitter(.08,.08,.06,.02)]
  ops += [transforms.ToTensor(),transforms.Normalize(MEAN,STD)];self.tf=transforms.Compose(ops)
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  r=self.rows[i]
  with Image.open(r['path']) as im:x=ImageOps.exif_transpose(im).convert('RGB')
  return self.tf(x),r['binary']
def build(arch,base,device):
 if arch=='convnext_tiny':m=models.convnext_tiny(weights=None);m.classifier[2]=nn.Linear(m.classifier[2].in_features,2)
 else:m=models.resnet50(weights=None);m.fc=nn.Linear(m.fc.in_features,2)
 state={k:v for k,v in base.items() if k in m.state_dict() and m.state_dict()[k].shape==v.shape}
 m.load_state_dict(state,strict=False);return m.to(device)
def collect(m,dl):
 m.eval();ps=[];ys=[]
 with torch.inference_mode():
  for x,y in dl:
   x=x.cuda(non_blocking=True)
   with torch.autocast('cuda',dtype=torch.float16):z=m(x)
   ps.extend(torch.softmax(z.float(),1)[:,1].cpu().numpy());ys.extend(y.numpy())
 return np.asarray(ps),np.asarray(ys,dtype=int)
def pack(y,p,t):
 pred=p>=t;tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
 return {'n':len(y),'positive':int(y.sum()),'negative':int(len(y)-y.sum()),'threshold':float(t),'accuracy':float(accuracy_score(y,pred)),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),'sensitivity':float(tp/(tp+fn)) if tp+fn else None,'specificity':float(tn/(tn+fp)) if tn+fp else None,'false_negatives':int(fn),'false_positives':int(fp),'auroc':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None}
def choose(y,p):
 c=np.r_[np.unique(p),np.nextafter(p.max(),np.inf)];return max(((pack(y,p,float(t)),float(t)) for t in c),key=lambda z:(z[0]['balanced_accuracy'],z[0]['sensitivity'] or 0))[1]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--base-checkpoint',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--epochs',type=int,default=5);ap.add_argument('--workers',type=int,default=4);ap.add_argument('--batch-size',type=int,default=16);ap.add_argument('--seconds-per-model',type=int,default=1200);a=ap.parse_args()
 if not torch.cuda.is_available():raise RuntimeError('CUDA required')
 torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED)
 rows,names=load(a.data);parts=split(rows);base_ck=torch.load(a.base_checkpoint,map_location='cpu',weights_only=False);arch=base_ck['architecture'];base=base_ck['state_dict']
 a.out.mkdir(parents=True,exist_ok=True);results={}
 for target,name in enumerate(names):
  sets={}
  for split_name,subset in parts.items():
   rs=[dict(r,binary=int(r['cls']==target)) for r in subset]
   if not rs or len({r['binary'] for r in rs})<2:raise RuntimeError(f'Binary target lacks both classes in {split_name}')
   sets[split_name]=rs
  loaders={k:DataLoader(DS(v,k=='train'),batch_size=a.batch_size,shuffle=k=='train',num_workers=a.workers,pin_memory=True) for k,v in sets.items()}
  m=build(arch,base,'cuda');counts=Counter(r['binary'] for r in sets['train']);cw=torch.tensor([np.sqrt(len(sets['train'])/(2*counts[i])) for i in range(2)],dtype=torch.float32,device='cuda');opt=torch.optim.AdamW(m.parameters(),lr=1.5e-5,weight_decay=3e-4)
  best=float('inf');bestep=0;stale=0;start=time.time();out=a.out/f'{target:02d}_{name.lower().replace(" ","_")}.pt'
  for ep in range(1,a.epochs+1):
   m.train();loss_sum=0.;seen=0
   for x,y in loaders['train']:
    x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True);opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.float16):z=m(x);loss=nn.functional.cross_entropy(z,y,weight=cw)
    loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),2.);opt.step();loss_sum+=loss.item()*len(y);seen+=len(y)
   vp,vy=collect(m,loaders['validation']);vl=float(nn.functional.cross_entropy(torch.tensor(np.stack([1-vp,vp],1).clip(1e-7,1),dtype=torch.float32).log(),torch.tensor(vy)).item());thr=choose(vy,vp)
   vm=pack(vy,vp,thr);print(json.dumps({'target':name,'epoch':ep,'train_loss':loss_sum/max(1,seen),'validation_loss':vl,'validation':vm}),flush=True)
   if vl<best-1e-4:best=vl;bestep=ep;stale=0;torch.save({'architecture':arch,'state_dict':m.state_dict(),'target':name,'epoch':ep},out)
   else:stale+=1
   if stale>=2 or time.time()-start>=a.seconds_per_model:break
  # Reload selected validation checkpoint; choose threshold on validation only.
  ck=torch.load(out,map_location='cuda',weights_only=False);m.load_state_dict(ck['state_dict']);vp,vy=collect(m,loaders['validation']);thr=choose(vy,vp);tp,ty=collect(m,loaders['test']);results[name]={'architecture':arch,'best_epoch':bestep,'train_n':len(sets['train']),'validation':pack(vy,vp,thr),'test':pack(ty,tp,thr),'test_boundary':'Exact-hash-grouped image split only; patient IDs absent. Target condition vs source-defined healthy class, not versus all other disease.'};del m;torch.cuda.empty_cache()
  print(json.dumps({'completed_target':name,'test':results[name]['test']}),flush=True)
 report={'status':'complete','source':'Mendeley Eye Disease Image Dataset v1','selected_joint_initialization':arch,'selection':'Base model selected on same frozen validation; each specialist threshold selected on its validation subset. Test partitions are not used for selection.','patient_ids_available':False,'results':results,'boundary':'Each specialist separates one source label from all other source labels on an exact-duplicate-grouped image split only. Patient IDs are absent. The shared test was already opened for the joint model; follow-on specialist test figures are exploratory, not a fresh confirmatory test. Not externally or clinically validated.'}
 (a.out/'aggregate_specialists_report.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
