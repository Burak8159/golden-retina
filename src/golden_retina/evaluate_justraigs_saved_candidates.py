#!/usr/bin/env python3
"""Fast final selection and one-shot evaluation of saved JustRAIGS candidates."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from .train_justraigs_patient_grouped import (
    SEED,EXTS,DS,make_model,metrics,predict,threshold_at_specificity,
)

def rows_and_parts(labels:Path,images:Path,size:int,cap:int):
 raw=[]
 with labels.open(newline='',encoding='utf-8-sig') as f:
  for r in csv.DictReader(f,delimiter=';'):
   eye=(r.get('Eye ID') or '').strip();fellow=(r.get('Fellow Eye ID') or '').strip();lab=(r.get('Final Label') or '').strip().upper()
   if eye and lab in {'RG','NRG'}:raw.append({'eye':eye,'fellow':fellow or None,'label':int(lab=='RG')})
 parent={r['eye']:r['eye'] for r in raw}
 def find(x):
  while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
  return x
 def union(a,b):
  if a not in parent:parent[a]=a
  if b not in parent:parent[b]=b
  ra,rb=find(a),find(b)
  if ra!=rb:parent[max(ra,rb)]=min(ra,rb)
 for r in raw:
  if r['fellow']:union(r['eye'],r['fellow'])
 for r in raw:r['group']=find(r['eye'])
 by={}
 for p in images.rglob('*'):
  if p.is_file() and p.suffix.lower() in EXTS:by.setdefault(p.stem,[]).append(p)
 rows=[{**r,'path':by[r['eye']][0]} for r in raw if by.get(r['eye'])]
 groups={}
 for r in rows:groups[r['group']]=max(groups.get(r['group'],0),r['label'])
 g=np.array(list(groups));gl=np.array([groups[x] for x in g])
 gtv,gte=train_test_split(g,test_size=.15,random_state=SEED,stratify=gl)
 gltv=np.array([groups[x] for x in gtv]);gtr,gva=train_test_split(gtv,test_size=.1764705882,random_state=SEED+1,stratify=gltv)
 gs={'train':set(gtr),'validation':set(gva),'test':set(gte)}
 parts={k:[r for r in rows if r['group'] in v] for k,v in gs.items()}
 rng=np.random.default_rng(SEED)
 for k,items in parts.items():
  if cap<=0 or len(items)<=cap:continue
  y=np.asarray([r['label'] for r in items]);chosen=[]
  for lab in (0,1):
   ids=np.flatnonzero(y==lab);q=round(cap*len(ids)/len(items))
   if len(ids) and q==0:q=1
   chosen.extend(rng.choice(ids,size=min(q,len(ids)),replace=False).tolist())
  if len(chosen)>cap:chosen=sorted(rng.choice(np.asarray(chosen),size=cap,replace=False).tolist())
  elif len(chosen)<cap:
   rest=np.setdiff1d(np.arange(len(items)),np.asarray(chosen,dtype=int));chosen.extend(rng.choice(rest,size=min(cap-len(chosen),len(rest)),replace=False).tolist())
  parts[k]=[items[i] for i in sorted(chosen)]
 return parts

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--labels',type=Path,required=True);ap.add_argument('--images',type=Path,required=True);ap.add_argument('--checkpoints',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--image-size',type=int,default=320);ap.add_argument('--cap',type=int,default=6000);ap.add_argument('--batch-size',type=int,default=32);ap.add_argument('--workers',type=int,default=4);a=ap.parse_args()
 parts=rows_and_parts(a.labels,a.images,a.image_size,a.cap)
 loaders={k:DataLoader(DS(v,a.image_size,False),batch_size=a.batch_size,shuffle=False,num_workers=a.workers,pin_memory=True) for k,v in parts.items()}
 candidates=[];preds={}
 for arch in ('convnext_tiny','resnet50'):
  p=a.checkpoints/f'{arch}_validation_best_private.pt'
  ck=torch.load(p,map_location='cpu',weights_only=False);m=make_model(arch);m.load_state_dict(ck['state_dict']);m.cuda()
  y,s,g,ids=predict(m,loaders['validation']);val=metrics(y,s)
  candidates.append({'architecture':arch,'best_epoch':int(ck['epoch']),'checkpoint':str(p),'validation':val})
  preds[arch]=(y,s,g,ids);del m;torch.cuda.empty_cache()
 chosen=max(candidates,key=lambda x:x['validation']['auprc'])
 arch=chosen['architecture'];vy,vs,_,_=preds[arch];th=threshold_at_specificity(vy,vs,.90)
 m=make_model(arch);ck=torch.load(a.checkpoints/f'{arch}_validation_best_private.pt',map_location='cpu',weights_only=False);m.load_state_dict(ck['state_dict']);m.cuda()
 ty,ts,tg,ids=predict(m,loaders['test']);test=metrics(ty,ts,th['threshold'])
 rng=np.random.default_rng(SEED);unique=np.unique(tg);index={g:np.flatnonzero(tg==g) for g in unique};boot={k:[] for k in ('accuracy','sensitivity','specificity','auroc','auprc')}
 for _ in range(1000):
  ix=np.concatenate([index[g] for g in rng.choice(unique,size=len(unique),replace=True)])
  if len(np.unique(ty[ix]))<2:continue
  mm=metrics(ty[ix],ts[ix],th['threshold'])
  for key in boot:boot[key].append(mm[key])
 ci={k:[float(np.quantile(v,.025)),float(np.quantile(v,.975))] if v else None for k,v in boot.items()}
 result={'status':'complete','dataset':'JustRAIGS training release, deterministic capped rerun of frozen partitions','task':'referable glaucoma vs no referable glaucoma','candidate_models':candidates,'selected_model':arch,'validation_threshold_target_specificity_90':th,'one_shot_internal_test':test,'group_bootstrap_95_ci':ci,'validation_n':len(vy),'test_n':len(ty),'cap_per_partition':a.cap,'boundary':'This is an internal split of the official training release. Fellow-eye components were reconstructed from available links; missing links can split true patients. It is not the official challenge test or independent clinical validation.'}
 a.out.mkdir(parents=True,exist_ok=True);(a.out/'aggregate_report.json').write_text(json.dumps(result,indent=2)+'\n')
 with (a.out/'private_test_predictions.csv').open('w') as f:
  w=csv.writer(f);w.writerow(['eye_id','group','label_rg','score_rg','prediction_locked_threshold'])
  for eye,g,y,s in zip(ids,tg,ty,ts):w.writerow([eye,g,int(y),float(s),int(s>=th['threshold'])])
 print(json.dumps({'selected_model':arch,'validation_threshold':th,'one_shot_internal_test':test,'group_bootstrap_95_ci':ci},indent=2),flush=True)
if __name__=='__main__':main()
