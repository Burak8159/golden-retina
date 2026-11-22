#!/usr/bin/env python3
"""Rajbari image-group AMD specialist with exploratory BMRD site readout.

The Rajbari archive has no patient IDs or augmentation lineage. Exact-byte
duplicates are grouped across train/validation/test; this cannot prevent
near-duplicate or source-family leakage. BMRD has been opened by prior runs,
so its per-hospital figures are secondary exploratory cross-source results.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, random, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

SEED=8159
MEAN=(.485,.456,.406); STD=(.229,.224,.225)
EXT={'.jpg','.jpeg','.png','.tif','.tiff','.bmp'}

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def load_rajbari(root):
 dirs={p.name:p for p in root.iterdir() if p.is_dir()}
 wanted={
  'No Age-Related Macular Degeneration (Healthy regarding AMD)':0,
  'Dry Age-Related Macular Degeneration':1,
  'Wet Age-Related Macular Degeneration':1,
 }
 rows=[]
 for name,y in wanted.items():
  if name not in dirs: raise FileNotFoundError(f'missing class folder: {name}')
  for p in dirs[name].rglob('*'):
   if p.is_file() and p.suffix.lower() in EXT: rows.append({'path':p,'y':y,'group':sha(p),'source':'Rajbari','name':name})
 groups=defaultdict(list)
 for i,r in enumerate(rows):groups[r['group']].append(i)
 bad=set()
 for ids in groups.values():
  if len({rows[i]['y'] for i in ids})>1:bad.update(ids)
 return [r for i,r in enumerate(rows) if i not in bad],len(bad),sum(len(v)-1 for v in groups.values())

def split_rows(rows):
 y=np.array([r['y'] for r in rows]);g=np.array([r['group'] for r in rows])
 cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED)
 tv,te=next(cv.split(np.zeros(len(y)),y,g))
 inner=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED+1)
 tr,va=next(inner.split(np.zeros(len(tv)),y[tv],g[tv]))
 out={'train':[rows[i] for i in tv[tr]],'validation':[rows[i] for i in tv[va]],'test':[rows[i] for i in te]}
 sets={k:{r['group'] for r in v} for k,v in out.items()}
 if sets['train']&sets['validation'] or sets['train']&sets['test'] or sets['validation']&sets['test']:raise RuntimeError('exact hash group leakage')
 return out

class Images(Dataset):
 def __init__(self,rows,train=False):
  ops=[transforms.Resize((256,256))]
  if train:ops += [transforms.RandomHorizontalFlip(),transforms.RandomRotation(7),transforms.ColorJitter(.05,.05,.04,.01)]
  ops += [transforms.ToTensor(),transforms.Normalize(MEAN,STD)];self.tf=transforms.Compose(ops);self.rows=rows
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  r=self.rows[i]
  with Image.open(r['path']) as im:x=self.tf(ImageOps.exif_transpose(im).convert('RGB'))
  return x,int(r['y'])

def model_init(arch):
 if arch=='convnext_tiny':
  m=models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT);m.classifier[2]=nn.Linear(m.classifier[2].in_features,1)
 elif arch=='resnet50':
  m=models.resnet50(weights=models.ResNet50_Weights.DEFAULT);m.fc=nn.Linear(m.fc.in_features,1)
 else:raise ValueError(arch)
 return m

def score(m,loader):
 m.eval();ys=[];ps=[]
 with torch.inference_mode():
  for x,y in loader:
   with torch.autocast('cuda',dtype=torch.float16):z=m(x.cuda(non_blocking=True)).flatten()
   ps.extend(torch.sigmoid(z.float()).cpu().numpy());ys.extend(y.numpy())
 return np.asarray(ys,dtype=int),np.asarray(ps)

def pack(y,p,t):
 pred=p>=t;tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
 return {'n':int(len(y)),'positive_support':int(y.sum()),'negative_support':int(len(y)-y.sum()),'threshold':float(t),
  'accuracy':float(accuracy_score(y,pred)),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
  'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),
  'auroc':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
  'sensitivity':float(tp/max(1,tp+fn)),'specificity':float(tn/max(1,tn+fp)),
  'false_negatives':int(fn),'false_positives':int(fp)}

def select_threshold(y,p):
 candidates=np.r_[np.unique(p),np.nextafter(float(p.max()),np.inf)]
 return max(((pack(y,p,float(t)),float(t)) for t in candidates),key=lambda z:(z[0]['balanced_accuracy'],z[0]['sensitivity']))[1]

def load_bmrd(path):
 rows=[]
 for r in csv.DictReader(open(path,encoding='utf-8-sig')):
  if r.get('disease','').lower()!='amd':continue
  lab=r['label'].lower()
  if lab=='amd_none':y=0
  elif lab in {'amd_dry','amd_wet'}:y=1
  else:continue
  rows.append({'path':Path(r['path']),'y':y,'group':r['sha256'],'source':r['hospital'],'name':r['label']})
 return rows

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--rajbari',type=Path,required=True);ap.add_argument('--bmrd-manifest',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--epochs',type=int,default=8);ap.add_argument('--seconds',type=int,default=2400);ap.add_argument('--workers',type=int,default=6);ap.add_argument('--batch-size',type=int,default=32);a=ap.parse_args()
 if not torch.cuda.is_available():raise RuntimeError('CUDA L4 required')
 random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
 rows,bad,dups=load_rajbari(a.rajbari);parts=split_rows(rows);a.out.mkdir(parents=True,exist_ok=True)
 summary={'dataset':'Rajbari Retinal Disease Dataset 2026 (Mendeley z227d626vm.1)','mapping':{'positive':['Dry Age-Related Macular Degeneration','Wet Age-Related Macular Degeneration'],'negative':['No Age-Related Macular Degeneration (Healthy regarding AMD)']},'n':len(rows),'conflicting_exact_hash_rows_excluded':bad,'duplicate_rows':dups,'patient_ids_available':False,'lineage':'Archive has 5,500 images, 500 per 11 classes, while source describes 1,222 original plus 5,500 augmented; no image-parent map in staged archive. Exact-byte grouping cannot guarantee augmentation-family separation.','split_counts':{k:{'n':len(v),'positive':sum(r['y'] for r in v),'negative':sum(1-r['y'] for r in v),'groups':len({r['group'] for r in v})} for k,v in parts.items()},'interpretation':'Exploratory image-grouped development estimate; not patient-level or clinical validation.'}
 (a.out/'private_split_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 loaders={k:DataLoader(Images(v,k=='train'),batch_size=a.batch_size,shuffle=k=='train',num_workers=a.workers,pin_memory=True) for k,v in parts.items()}
 counts=np.bincount([r['y'] for r in parts['train']],minlength=2);cw=torch.tensor([np.sqrt(len(parts['train'])/(2*max(1,c))) for c in counts],device='cuda',dtype=torch.float32)
 results=[];start=time.time()
 for arch in ('convnext_tiny','resnet50'):
  m=model_init(arch).cuda();opt=torch.optim.AdamW(m.parameters(),lr=2e-5,weight_decay=2e-4);best=float('inf');stale=0;path=a.out/f'{arch}_best.pt';bestep=0
  for ep in range(1,a.epochs+1):
   m.train();ss=0.;nn_=0
   for x,y in loaders['train']:
    x=x.cuda(non_blocking=True);y=y.float().cuda(non_blocking=True);opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.float16):z=m(x).flatten();loss=nn.functional.binary_cross_entropy_with_logits(z,y,pos_weight=(cw[1]/cw[0]).clamp(.5,4))
    loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),2.);opt.step();ss+=float(loss.item())*len(y);nn_+=len(y)
   vy,vp=score(m,loaders['validation']);vl=float(nn.functional.binary_cross_entropy(torch.tensor(vp.clip(1e-7,1-1e-7)),torch.tensor(vy,dtype=torch.float32)))
   print(json.dumps({'architecture':arch,'epoch':ep,'train_loss':ss/max(1,nn_),'validation_loss':vl,'validation':pack(vy,vp,.5)}),flush=True)
   if vl<best-1e-4:best=vl;stale=0;bestep=ep;torch.save({'state_dict':m.state_dict(),'architecture':arch,'target':'AMD vs AMD-none'},path)
   else:stale+=1
   if stale>=3 or time.time()-start>=a.seconds:break
  del m;torch.cuda.empty_cache();results.append({'architecture':arch,'checkpoint':str(path),'best_epoch':bestep,'validation_loss':best})
  if time.time()-start>=a.seconds:break
 chosen=min(results,key=lambda r:r['validation_loss']);ck=torch.load(chosen['checkpoint'],map_location='cpu',weights_only=False);m=model_init(chosen['architecture']);m.load_state_dict(ck['state_dict']);m.cuda()
 vy,vp=score(m,loaders['validation']);thr=select_threshold(vy,vp);ty,tp=score(m,loaders['test']);fit_test=pack(ty,tp,thr)
 bmrd=load_bmrd(a.bmrd_manifest);bmrd_out={}
 for src in sorted({r['source'] for r in bmrd}):
  subset=[r for r in bmrd if r['source']==src]
  dl=DataLoader(Images(subset),batch_size=a.batch_size,shuffle=False,num_workers=a.workers,pin_memory=True);y,p=score(m,dl);bmrd_out[src]=pack(y,p,thr)
 report={'status':'complete','target':'AMD (dry or wet) vs source-defined AMD-none','candidate_selection':'lowest Rajbari validation BCE among trained architectures; threshold selected for maximum balanced accuracy on Rajbari validation only','selected':chosen,'validation_threshold':thr,'rajbari_image_grouped_test':fit_test,'bmrd_cross_source_exploratory':bmrd_out,'data_summary':summary,'test_boundary':'Rajbari exact-byte image groups only; no patients/augmentation lineage. BMRD is a distinct source with zero exact-byte cross-source overlap but BMRD outcomes were already viewed in earlier project runs. BMRD readings are exploratory and not confirmatory.','not_clinically_validated':True}
 (a.out/'aggregate_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'final':report}),flush=True)
if __name__=='__main__':main()
