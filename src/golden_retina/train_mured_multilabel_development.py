#!/usr/bin/env python3
"""Fine-tune a ConvNeXt-Tiny multi-label head on MuReD's published train/val.

MuReD combines ARIA, STARE and RFMiD training data. This script reports only
validation-development evidence; it does not create an independent test claim.
All images, row identifiers, predictions and weights must stay on the VM.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,random,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageOps
from sklearn.metrics import average_precision_score,roc_auc_score
from torch import nn
from torch.utils.data import DataLoader,Dataset
from torchvision import models,transforms

SEED=8159; EXT={'.jpg','.jpeg','.png','.tif','.tiff'}; MEAN=(.485,.456,.406); STD=(.229,.224,.225)

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def load_split(images,csv_path):
 with csv_path.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
 if not rows: raise ValueError(f'empty label table: {csv_path.name}')
 fields=list(rows[0]); id_col=fields[0]; names=fields[1:]
 if len(names)!=20: raise ValueError(f'unexpected MuReD label count {len(names)}')
 index={p.stem:p for p in images.rglob('*') if p.is_file() and p.suffix.lower() in EXT}
 out=[]
 for r in rows:
  key=Path(str(r[id_col]).strip()).stem
  if key not in index: raise FileNotFoundError('MuReD row could not be matched to an image')
  y=[]
  for n in names:
   v=str(r[n]).strip()
   if v not in {'0','1','0.0','1.0'}: raise ValueError(f'nonbinary value in label {n}')
   y.append(float(v))
  out.append({'path':index[key],'targets':y,'sha':sha(index[key])})
 return out,names

class Img(Dataset):
 def __init__(self,rows,train):
  self.rows=rows; ops=[transforms.Resize((320,320))]
  if train: ops += [transforms.RandomHorizontalFlip(),transforms.RandomRotation(8),transforms.ColorJitter(.08,.08,.06)]
  ops += [transforms.ToTensor(),transforms.Normalize(MEAN,STD)]; self.tf=transforms.Compose(ops)
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  r=self.rows[i]
  with Image.open(r['path']) as im:x=self.tf(ImageOps.exif_transpose(im).convert('RGB'))
  return x,torch.tensor(r['targets'],dtype=torch.float32)

def collect(m,dl):
 m.eval(); yy=[]; pp=[]
 with torch.inference_mode():
  for x,y in dl:
   x=x.cuda(non_blocking=True)
   with torch.autocast('cuda',dtype=torch.float16): z=m(x)
   yy.append(y.numpy()); pp.append(torch.sigmoid(z.float()).cpu().numpy())
 return np.concatenate(yy),np.concatenate(pp)

def report(y,p,names,train_counts):
 out={}; auc=[]; aps=[]
 for j,n in enumerate(names):
  yy=y[:,j].astype(int); pred=p[:,j]>=.5; pos=int(yy.sum()); neg=len(yy)-pos
  try:a=float(roc_auc_score(yy,p[:,j]))
  except ValueError:a=None
  try:ap=float(average_precision_score(yy,p[:,j])) if pos else None
  except ValueError:ap=None
  if a is not None:auc.append(a)
  if ap is not None:aps.append(ap)
  out[n]={'train_positive':int(train_counts[j]),'validation_positive':pos,'validation_negative':neg,'auroc':a,'average_precision':ap,'sensitivity_at_0_5':float(((yy==1)&pred).sum()/pos) if pos else None,'specificity_at_0_5':float(((yy==0)&~pred).sum()/neg) if neg else None,'accuracy_at_0_5':float((yy==pred).mean()),'false_negatives_at_0_5':int(((yy==1)&~pred).sum())}
 return {'n_validation':len(y),'macro_auroc':float(np.mean(auc)) if auc else None,'macro_average_precision':float(np.mean(aps)) if aps else None,'per_label':out}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data-root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--epochs',type=int,default=10);ap.add_argument('--seconds',type=int,default=7200);ap.add_argument('--batch-size',type=int,default=12);ap.add_argument('--workers',type=int,default=4);a=ap.parse_args()
 if not torch.cuda.is_available():raise RuntimeError('CUDA required')
 random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
 base=a.data_root; image_root=base/'images'; train,names=load_split(image_root,base/'train_data.csv'); val,names2=load_split(image_root,base/'val_data.csv')
 if names!=names2:raise ValueError('train/validation label names differ')
 train_hash={r['sha'] for r in train}; val_hash={r['sha'] for r in val}; overlap=train_hash&val_hash
 if overlap:raise RuntimeError(f'exact train/validation image overlap: {len(overlap)}; remove duplicate groups before fitting')
 ytrain=np.asarray([r['targets'] for r in train],dtype=np.float32); counts=ytrain.sum(0); negatives=len(train)-counts
 active=(counts>=10)&(negatives>=10); pos_weight=np.sqrt(np.maximum(negatives,1)/np.maximum(counts,1)).clip(1,8);pos_weight[~active]=0
 a.out.mkdir(parents=True,exist_ok=True)
 safe={'source':'MuReD v2','n_train':len(train),'n_validation':len(val),'labels':names,'train_positive_counts':counts.astype(int).tolist(),'validation_positive_counts':np.asarray([r['targets'] for r in val]).sum(0).astype(int).tolist(),'exact_train_validation_sha_overlap':0,'patient_ids_available':False,'class_weighted_labels':[n for n,m in zip(names,active) if m],'excluded_from_supported_training':[n for n,m in zip(names,active) if not m],'evidence_boundary':'MuReD is composed from ARIA, STARE and RFMiD training data; no patient linkage and no independent test partition in this run. Development validation only.'}
 (a.out/'private_split_summary.json').write_text(json.dumps(safe,indent=2)+'\n')
 dl={'train':DataLoader(Img(train,True),batch_size=a.batch_size,shuffle=True,num_workers=a.workers,pin_memory=True),'validation':DataLoader(Img(val,False),batch_size=a.batch_size*2,shuffle=False,num_workers=a.workers,pin_memory=True)}
 m=models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT);m.classifier[-1]=nn.Linear(m.classifier[-1].in_features,len(names));m=m.cuda()
 opt=torch.optim.AdamW(m.parameters(),lr=2e-5,weight_decay=2e-4); pw=torch.tensor(pos_weight,dtype=torch.float32,device='cuda'); best=-1;bestep=0;stale=0; start=time.time(); ck=a.out/'convnext_tiny_best.pt'
 for ep in range(1,a.epochs+1):
  m.train();ls=0.;seen=0
  for x,y in dl['train']:
   x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True);opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.float16):z=m(x);loss=nn.functional.binary_cross_entropy_with_logits(z,y,pos_weight=pw,reduction='none')[:,active].mean()
   loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),2.);opt.step();ls+=loss.item()*len(y);seen+=len(y)
  yv,pv=collect(m,dl['validation']);r=report(yv,pv,names,counts); score=r['macro_average_precision'] or 0
  print(json.dumps({'epoch':ep,'train_loss':ls/max(1,seen),'validation':r}),flush=True)
  if score>best+1e-4:best=score;bestep=ep;stale=0;torch.save({'state_dict':m.state_dict(),'labels':names,'epoch':ep},ck)
  else:stale+=1
  if stale>=3 or time.time()-start>=a.seconds:break
 (a.out/'aggregate_report.json').write_text(json.dumps({'status':'complete','best_epoch':bestep,'best_validation_macro_average_precision':best,'validation':r,'data_summary':safe},indent=2)+'\n')
if __name__=='__main__':main()
