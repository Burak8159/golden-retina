#!/usr/bin/env python3
"""Train an exploratory multiclass fundus classifier from folder-labeled data.

Rows, split membership, predictions and checkpoints stay in the VM output dir.
Because this source has no patient identifiers, duplicate-connected image groups
are kept intact and results are explicitly marked image-grouped, not patient
independent or clinically validated.
"""
from __future__ import annotations

import argparse, hashlib, json, random, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

SEED=8159
EXTS={'.jpg','.jpeg','.png','.tif','.tiff','.bmp'}
MEAN=(0.485,0.456,0.406); STD=(0.229,0.224,0.225)

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def scan(root, excluded_hashes=None):
 dirs=sorted(p for p in root.iterdir() if p.is_dir())
 dirs=[p for p in dirs if p.name.lower() not in {'pterygium','pt'}]
 rows=[]
 for ci,d in enumerate(dirs):
  for p in d.rglob('*'):
   if p.is_file() and p.suffix.lower() in EXTS:
    h=sha(p); rows.append({'path':p,'class':ci,'class_name':d.name,'hash':h})
 if len(dirs)<2 or len(rows)<100: raise ValueError('Unexpected folder dataset size/class layout')
 excluded_hashes=excluded_hashes or set()
 excluded_rows=sum(r['hash'] in excluded_hashes for r in rows)
 rows=[r for r in rows if r['hash'] not in excluded_hashes]
 # Union identical bytes; conflicting labels are excluded rather than allowed to leak.
 hashes=defaultdict(list)
 for i,r in enumerate(rows): hashes[r['hash']].append(i)
 bad=set(); parent=list(range(len(rows)))
 def find(x):
  while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
  return x
 def union(a,b):
  a,b=find(a),find(b)
  if a!=b: parent[b]=a
 for ids in hashes.values():
  labs={rows[i]['class'] for i in ids}
  if len(labs)>1: bad.update(ids)
  for i in ids[1:]: union(ids[0],i)
 rows=[r for i,r in enumerate(rows) if i not in bad]
 # Regroup after removal; same-hash rows are one leakage component.
 for r in rows: r['group']=r['hash']
 return rows,[p.name for p in dirs],len(bad),sum(len(v)-1 for v in hashes.values()),excluded_rows

def split(rows):
 y=np.array([r['class'] for r in rows]); g=np.array([r['group'] for r in rows])
 # Five-fold group split; choose folds by seeded order and preserve every class.
 folds=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED)
 trainval,test=next(folds.split(np.zeros(len(y)),y,g))
 inner=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED+1)
 tr,va=next(inner.split(np.zeros(len(trainval)),y[trainval],g[trainval]))
 part={'train':[rows[i] for i in trainval[tr]],'validation':[rows[i] for i in trainval[va]],'test':[rows[i] for i in test]}
 groups={k:{r['group'] for r in v} for k,v in part.items()}
 if any(groups[a]&groups[b] for a,b in [('train','validation'),('train','test'),('validation','test')]): raise RuntimeError('duplicate group leakage')
 classes=set(range(len(np.unique(y))))
 if any({r['class'] for r in v}!=classes for v in part.values()): raise RuntimeError('a split lacks a class')
 return part

class DS(Dataset):
 def __init__(self,rows,train=False):
  self.rows=rows
  ops=[transforms.Resize(256),transforms.CenterCrop(224)]
  if train: ops += [transforms.RandomHorizontalFlip(),transforms.RandomRotation(8),transforms.ColorJitter(.08,.08,.06,.02)]
  ops += [transforms.ToTensor(),transforms.Normalize(MEAN,STD)]; self.tf=transforms.Compose(ops)
 def __len__(self): return len(self.rows)
 def __getitem__(self,i):
  r=self.rows[i]
  with Image.open(r['path']) as im: x=ImageOps.exif_transpose(im).convert('RGB')
  return self.tf(x),r['class']

def make(name,n):
 if name=='convnext_tiny':
  m=models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT); m.classifier[2]=nn.Linear(m.classifier[2].in_features,n)
 else:
  m=models.resnet50(weights=models.ResNet50_Weights.DEFAULT); m.fc=nn.Linear(m.fc.in_features,n)
 return m

def infer(m,loader,device):
 m.eval(); ps=[]; ys=[]
 with torch.inference_mode():
  for x,y in loader:
   x=x.to(device,non_blocking=True)
   with torch.autocast('cuda',dtype=torch.float16): z=m(x)
   ps.extend(torch.softmax(z.float(),1).cpu().numpy()); ys.extend(y.numpy())
 return np.asarray(ps),np.asarray(ys)

def metrics(y,p,names):
 pred=p.argmax(1); cm=confusion_matrix(y,pred,labels=range(len(names))); result={'n_images':int(len(y)),'accuracy':float(accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),'classes':{}}
 for i,name in enumerate(names):
  tp=int(cm[i,i]); fn=int(cm[i,:].sum()-tp); fp=int(cm[:,i].sum()-tp); tn=int(len(y)-tp-fn-fp)
  try: auc=float(roc_auc_score((y==i).astype(int),p[:,i]))
  except ValueError: auc=None
  result['classes'][name]={'support':int((y==i).sum()),'sensitivity':tp/(tp+fn) if tp+fn else None,'specificity':tn/(tn+fp) if tn+fp else None,'accuracy':(tp+tn)/len(y),'false_negatives':fn,'false_positives':fp,'auroc':auc}
 return result

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--data',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--exclude-hash-file',type=Path,default=None,help='Private newline-delimited SHA-256 hashes found in other staged datasets'); ap.add_argument('--epochs',type=int,default=8); ap.add_argument('--batch-size',type=int,default=16); ap.add_argument('--workers',type=int,default=6); ap.add_argument('--seconds',type=int,default=21600); args=ap.parse_args()
 random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
 if not torch.cuda.is_available(): raise RuntimeError('CUDA L4 required')
 args.out.mkdir(parents=True,exist_ok=True)
 excluded_hashes=set(args.exclude_hash_file.read_text().split()) if args.exclude_hash_file and args.exclude_hash_file.exists() else set()
 rows,names,conflicts,dups,cross_excluded=scan(args.data,excluded_hashes); parts=split(rows)
 summary={'dataset':'Mendeley Data 6ry9n79gb4.1: Diabetic Retinopathy and Retinal Diseases Fundus Image Dataset','classes':names,'n_images':len(rows),'cross_label_exact_hash_rows_excluded':conflicts,'cross_source_exact_hash_rows_excluded':cross_excluded,'duplicate_rows':dups,'split_counts':{k:{'n':len(v),'groups':len({r['group'] for r in v}),'class_counts':dict(Counter(r['class_name'] for r in v))} for k,v in parts.items()},'patient_ids_available':False,'source_provenance':'Mendeley record says the compilation merges three publicly available sources; image-level source provenance and patient linkage are not supplied in the staged files.','boundary':'Exact-duplicate-grouped image split only. Exact hashes found in scanned project datasets are excluded; no patient identity or source-level split is available, and 3-source compilation may contain near-duplicate/original-augmentation relatives across folds. Development estimate only, not patient-independent, external or clinical validation.'}
 (args.out/'private_split_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 loaders={k:DataLoader(DS(v,k=='train'),batch_size=args.batch_size,shuffle=k=='train',num_workers=args.workers,pin_memory=True) for k,v in parts.items()}
 counts=np.array([sum(r['class']==i for r in parts['train']) for i in range(len(names))],dtype=float); weights=(counts.mean()/counts)**.5; cw=torch.tensor(weights,dtype=torch.float32,device='cuda')
 results=[]; start=time.time()
 for arch in ('convnext_tiny','resnet50'):
  model=make(arch,len(names)).cuda(); opt=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=2e-4); best=float('inf'); best_epoch=0; best_path=args.out/f'{arch}_best.pt'; stale=0
  for epoch in range(1,args.epochs+1):
   model.train(); loss_sum=0.; seen=0
   for x,y in loaders['train']:
    x=x.cuda(non_blocking=True); y=y.cuda(non_blocking=True); opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.float16): logits=model(x); loss=nn.functional.cross_entropy(logits,y,weight=cw)
    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),2.); opt.step(); loss_sum+=loss.item()*len(y); seen+=len(y)
   vp,vy=infer(model,loaders['validation'],'cuda'); val_loss=float(nn.functional.cross_entropy(torch.tensor(np.log(np.maximum(vp,1e-8)),dtype=torch.float32),torch.tensor(vy)).item())
   print(json.dumps({'model':arch,'epoch':epoch,'train_loss':loss_sum/max(1,seen),'validation_loss':val_loss,'validation':metrics(vy,vp,names)}),flush=True)
   if val_loss<best-1e-4:
    best=val_loss; best_epoch=epoch; stale=0; torch.save({'state_dict':model.state_dict(),'architecture':arch,'classes':names,'epoch':epoch},best_path)
   else: stale+=1
   if stale>=3 or time.time()-start>=args.seconds: break
  results.append({'architecture':arch,'best_epoch':best_epoch,'best_validation_loss':best,'checkpoint':str(best_path)})
  del model; torch.cuda.empty_cache()
 # Select on validation only; final image-grouped test is opened once.
 chosen=min(results,key=lambda x:x['best_validation_loss']); ck=torch.load(chosen['checkpoint'],map_location='cpu',weights_only=False); model=make(chosen['architecture'],len(names)); model.load_state_dict(ck['state_dict']); model.cuda()
 tp,ty=infer(model,loaders['test'],'cuda'); final=metrics(ty,tp,names)
 report={'status':'complete','selected_model':chosen['architecture'],'selection_rule':'lowest validation cross-entropy; test not used for selection','candidates':results,'test':final,'test_boundary':summary['boundary'],'data_summary':summary}
 (args.out/'aggregate_report.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps({'final_aggregate':final,'selected_model':chosen['architecture'],'test_boundary':summary['boundary']}),flush=True)
if __name__=='__main__': main()
