#!/usr/bin/env python3
"""Cloud-only JustRAIGS referral-glaucoma experiment with fellow-eye grouping."""
from __future__ import annotations
import argparse, csv, hashlib, json, random, time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, roc_auc_score, brier_score_loss)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

SEED=8159
MEAN=(.485,.456,.406); STD=(.229,.224,.225)
EXTS={'.jpg','.jpeg','.png','.tif','.tiff'}

def sha256(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

class DS(Dataset):
 def __init__(self,rows,size,train=False):
  self.rows=rows;self.size=size
  t=[transforms.Resize((size,size))]
  if train:t += [transforms.RandomHorizontalFlip(),transforms.RandomRotation(7),transforms.ColorJitter(.04,.04,.03,.01)]
  t += [transforms.ToTensor(),transforms.Normalize(MEAN,STD)]
  self.tf=transforms.Compose(t)
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  r=self.rows[i]
  with Image.open(r['path']) as im:
   # JPEG draft asks libjpeg for a reduced decode before RGB conversion; this
   # avoids full multi-megapixel CPU decode when the model input is 320 px.
   if im.format=='JPEG':im.draft('RGB',(self.size,self.size))
   x=self.tf(ImageOps.exif_transpose(im).convert('RGB'))
  return x,r['label'],r['group'],r['eye']

def make_model(name):
 if name=='convnext_tiny':
  m=models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT); m.classifier[2]=nn.Linear(m.classifier[2].in_features,1)
 elif name=='resnet50':
  m=models.resnet50(weights=models.ResNet50_Weights.DEFAULT); m.fc=nn.Linear(m.fc.in_features,1)
 else: raise ValueError(name)
 return m

def metrics(y,s,t=.5):
 pred=s>=t; tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
 return {'n_images':int(len(y)),'positive_support':int(y.sum()),'accuracy':float(accuracy_score(y,pred)),
  'balanced_accuracy':float(balanced_accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),
  'auroc':float(roc_auc_score(y,s)) if len(np.unique(y))==2 else None,
  'auprc':float(average_precision_score(y,s)) if y.sum() else None,'brier_score':float(brier_score_loss(y,s)),
  'threshold':float(t),'sensitivity':float(tp/max(1,tp+fn)),'specificity':float(tn/max(1,tn+fp)),
  'false_negatives':int(fn),'false_positives':int(fp),'confusion_matrix_rows_true_NRG_RG':[[int(tn),int(fp)],[int(fn),int(tp)]]}

def threshold_at_specificity(y,s,target=.90):
 # Vectorized count sweep: the previous implementation recomputed AUROC,
 # AUPRC and confusion matrices for every score threshold, creating an O(n^2)
 # post-training stall on several thousand validation images.
 y=np.asarray(y,dtype=np.int8);s=np.asarray(s,dtype=np.float64)
 candidates=np.r_[np.unique(s),np.nextafter(float(s.max()),np.inf)]
 pos=np.sort(s[y==1]);neg=np.sort(s[y==0])
 tp=pos.size-np.searchsorted(pos,candidates,side='left')
 fp=neg.size-np.searchsorted(neg,candidates,side='left')
 sens=tp/max(1,pos.size);spec=(neg.size-fp)/max(1,neg.size)
 valid=np.flatnonzero(spec>=target)
 if not len(valid):return {'target_specificity':target,'threshold':float('inf'),'validation_sensitivity':0.0,'validation_specificity':1.0}
 best=valid[np.lexsort((candidates[valid],spec[valid],sens[valid]))[-1]]
 return {'target_specificity':target,'threshold':float(candidates[best]),'validation_sensitivity':float(sens[best]),'validation_specificity':float(spec[best])}

def predict(m,loader):
 m.eval(); scores=[]; ys=[]; groups=[]; eyes=[]
 with torch.inference_mode():
  for x,y,g,e in loader:
   x=x.cuda(non_blocking=True)
   with torch.autocast('cuda',dtype=torch.float16):z=m(x).flatten()
   scores.extend(torch.sigmoid(z.float()).cpu().numpy());ys.extend(y.numpy());groups.extend(g);eyes.extend(e)
 return np.asarray(ys,dtype=int),np.asarray(scores),np.asarray(groups),eyes

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--labels',type=Path,required=True);ap.add_argument('--images',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
 ap.add_argument('--exclude-root',type=Path,action='append',default=[]);ap.add_argument('--exclude-hash-file',type=Path,default=None,help='Private newline-delimited hashes of exact cross-source matches from the completed cloud audit');ap.add_argument('--verified-exact-overlap-report',type=Path,default=None,help='Completed private audit report for this exact source root; skips a redundant full-image rehash only after validation');ap.add_argument('--image-size',type=int,default=320)
 ap.add_argument('--epochs',type=int,default=8);ap.add_argument('--seconds-per-model',type=int,default=3600)
 ap.add_argument('--batch-size',type=int,default=32);ap.add_argument('--workers',type=int,default=6)
 ap.add_argument('--max-images-per-partition',type=int,default=12000,help='Deterministic per-label cap applied only after group-disjoint splits are frozen; 0 keeps the full partition')
 ap.add_argument('--architectures',nargs='+',default=['convnext_tiny','resnet50'])
 a=ap.parse_args(); assert torch.cuda.is_available(), 'CUDA required; no CPU fallback'
 random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
 # Explicit CSV parser: published file is semicolon-delimited.
 raw=[]
 with a.labels.open(newline='',encoding='utf-8-sig') as f:
  for r in csv.DictReader(f,delimiter=';'):
   eye=(r.get('Eye ID') or '').strip(); fellow=(r.get('Fellow Eye ID') or '').strip(); lab=(r.get('Final Label') or '').strip().upper()
   if eye and lab in {'RG','NRG'}:raw.append({'eye':eye,'fellow':fellow or None,'label':int(lab=='RG')})
 # Build connected components from Eye ID <-> Fellow Eye ID. Missing links remain singletons.
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
 # Discover files recursively; names are expected to encode Eye ID in their stem.
 by_stem={}
 for p in a.images.rglob('*'):
  if p.is_file() and p.suffix.lower() in EXTS:by_stem.setdefault(p.stem,[]).append(p)
 rows=[];missing=[]
 for r in raw:
  cand=by_stem.get(r['eye'],[])
  if not cand:missing.append(r['eye']);continue
  rows.append({**r,'path':cand[0]})
 if not rows:raise RuntimeError('no label-to-image matches; aborting before training')
 verified_report=None
 if a.verified_exact_overlap_report and a.verified_exact_overlap_report.exists():
  verified_report=json.loads(a.verified_exact_overlap_report.read_text());root_audit=verified_report.get('roots',{}).get('justraigs')
  if not root_audit or root_audit.get('file_paths')!=root_audit.get('unique_hashes') or any('justraigs' in k for k in verified_report.get('pairs_with_exact_overlap',{})):
   raise RuntimeError('completed exact-overlap report does not clear JustRAIGS; refusing hash-scan skip')
 # Exact-hash audit against known project sources, and within JustRAIGS.
 exclude_hash=set();exclude_counts={}
 if a.exclude_hash_file and a.exclude_hash_file.exists():
  exclude_hash=set(a.exclude_hash_file.read_text().split())
  exclude_counts['completed_exact_overlap_hash_file']=len(exclude_hash)
 else:
  for root in a.exclude_root:
   n=0
   for p in root.rglob('*'):
    if p.is_file() and p.suffix.lower() in EXTS:
     try:exclude_hash.add(sha256(p));n+=1
     except OSError:pass
   exclude_counts[str(root)]=n
 seen={};conflict=set();cross=0;dups=0;kept=[]
 if verified_report:
  # Audit covered this same 101,423-file root and all staged candidate roots.
  kept=rows
 else:
  for r in rows:
   h=sha256(r['path']);r['sha256']=h
   if h in exclude_hash:cross+=1;continue
   if h in seen:
    dups+=1
    if seen[h]['label']!=r['label']:conflict.add(h)
   else:seen[h]=r;kept.append(r)
  rows=[r for r in kept if r['sha256'] not in conflict]
 if len(rows)<1000:raise RuntimeError(f'only {len(rows)} rows after mapping; investigate before training')
 # Stratify at reconstructed patient-group level; each patient's linked eyes travel together.
 # Aggregate labels in one pass; rescanning all rows per patient is O(N^2)
 # and stalls on the 100k-image release before the first epoch.
 group_labels={}
 for r in rows:
  g=r['group'];group_labels[g]=max(group_labels.get(g,0),r['label'])
 groups=np.array(list(group_labels)); gl=np.array([group_labels[g] for g in groups])
 g_trainval,g_test=train_test_split(groups,test_size=.15,random_state=SEED,stratify=gl)
 gltv=np.array([group_labels[g] for g in g_trainval])
 g_train,g_val=train_test_split(g_trainval,test_size=.1764705882,random_state=SEED+1,stratify=gltv)
 group_sets={'train':set(g_train),'validation':set(g_val),'test':set(g_test)}
 part={k:[r for r in rows if r['group'] in group_sets[k]] for k in group_sets}
 sets=[set(r['group'] for r in part[k]) for k in part]
 assert not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2]
 # Bound epoch time on the L4. Sampling occurs within each already frozen
 # partition, stratified by eye-level label. No group is moved across splits.
 pre_cap={k:len(v) for k,v in part.items()}
 if a.max_images_per_partition>0:
  rng=np.random.default_rng(SEED)
  for k,items in part.items():
   if len(items)<=a.max_images_per_partition:continue
   labs=np.asarray([r['label'] for r in items]);chosen=[]
   for lab in (0,1):
    ids=np.flatnonzero(labs==lab);quota=round(a.max_images_per_partition*len(ids)/len(items))
    if len(ids) and quota==0:quota=1
    chosen.extend(rng.choice(ids,size=min(quota,len(ids)),replace=False).tolist())
   if len(chosen)>a.max_images_per_partition:
    chosen=sorted(rng.choice(np.asarray(chosen),size=a.max_images_per_partition,replace=False).tolist())
   elif len(chosen)<a.max_images_per_partition:
    rest=np.setdiff1d(np.arange(len(items)),np.asarray(chosen,dtype=int),assume_unique=False)
    chosen.extend(rng.choice(rest,size=min(a.max_images_per_partition-len(chosen),len(rest)),replace=False).tolist())
   part[k]=[items[i] for i in sorted(chosen)]
 a.out.mkdir(parents=True,exist_ok=True)
 split={'dataset':'JustRAIGS Zenodo 10035093','label_csv_rows':len(raw),'rows_with_image':len(rows),'missing_image_ids_count':len(missing),'missing_image_ids_sample':missing[:30],
 'final_label_counts':{'NRG':sum(r['label']==0 for r in rows),'RG':sum(r['label']==1 for r in rows)},'reconstructed_patient_groups':len(group_labels),
 'groups_with_missing_fellow_eye_links':sum(not r['fellow'] for r in raw),'resolved_fellow_links':sum(bool(r['fellow']) for r in raw),
 'published_patient_count':'55,736 as reported in source paper; this implementation reconstructs groups from Eye ID/Fellow Eye ID and does not claim exact equivalence',
 'cross_source_roots_scanned':exclude_counts,'cross_source_exact_match_rows_removed':cross,'within_source_duplicate_rows_removed':dups,'conflicting_duplicate_hashes_excluded':len(conflict),'validated_prior_exact_hash_audit_used':bool(verified_report),
 'split_method':'stratified split of connected components formed from Eye ID and Fellow Eye ID, seed 8159; 70/15/15 approximately; no component crosses split',
 'pre_cap_partition_images':pre_cap,'max_images_per_partition':a.max_images_per_partition,
 'split':{k:{'images':len(v),'groups':len({r['group'] for r in v}),'NRG':sum(r['label']==0 for r in v),'RG':sum(r['label']==1 for r in v)} for k,v in part.items()},
 'boundary':'Official training partition only, subdivided internally. Patient links are reconstructed and incomplete; 19 metadata/image mismatches and blank fellow-eye links are disclosed. No official challenge test or independent clinical validation.'}
 (a.out/'split_audit_aggregate.json').write_text(json.dumps(split,indent=2)+'\n')
 loaders={k:DataLoader(DS(v,a.image_size,k=='train'),batch_size=a.batch_size,shuffle=False if k!='train' else False,
   sampler=None,num_workers=a.workers,pin_memory=True,persistent_workers=a.workers>0) for k,v in part.items()}
 # Weighted sampling raises positive draw share to 20% while keeping dataset epoch length.
 ytr=np.array([r['label'] for r in part['train']]); n0=max(1,int((ytr==0).sum()));n1=max(1,int((ytr==1).sum()))
 weights=torch.tensor([(.8/n0 if r['label']==0 else .2/n1) for r in part['train']],dtype=torch.double)
 sampler=WeightedRandomSampler(weights,num_samples=len(part['train']),replacement=True)
 loaders['train']=DataLoader(DS(part['train'],a.image_size,True),batch_size=a.batch_size,sampler=sampler,num_workers=a.workers,pin_memory=True,persistent_workers=a.workers>0)
 candidate_results=[]
 for arch in a.architectures:
  m=make_model(arch).cuda();opt=torch.optim.AdamW(m.parameters(),lr=2e-5,weight_decay=2e-4)
  best=-1.;best_ep=0;stale=0;start=time.time();ckpt=a.out/f'{arch}_validation_best_private.pt'
  for ep in range(1,a.epochs+1):
   m.train();tot=0.;cnt=0
   for x,y,_,_ in loaders['train']:
    x=x.cuda(non_blocking=True);y=y.float().cuda(non_blocking=True);opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.float16):z=m(x).flatten();loss=nn.functional.binary_cross_entropy_with_logits(z,y,pos_weight=torch.tensor([4.],device='cuda'))
    loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),2.);opt.step();tot+=float(loss.detach())*len(y);cnt+=len(y)
   vy,vs,vg,_=predict(m,loaders['validation']);vm=metrics(vy,vs)
   print(json.dumps({'architecture':arch,'epoch':ep,'train_loss':tot/max(1,cnt),'validation':vm}),flush=True)
   if vm['auprc'] is not None and vm['auprc']>best+1e-5:
    best,best_ep,stale=vm['auprc'],ep,0;torch.save({'state_dict':m.state_dict(),'architecture':arch,'epoch':ep,'image_size':a.image_size},ckpt)
   else:stale+=1
   if stale>=2 or time.time()-start>=a.seconds_per_model:break
  candidate_results.append({'architecture':arch,'best_epoch':best_ep,'best_validation_auprc':best,'checkpoint_sha256':sha256(ckpt) if ckpt.exists() else None})
  del m;torch.cuda.empty_cache()
  # Respect total time budget: allow outer supervisor to stop between models safely.
 chosen=max(candidate_results,key=lambda x:x['best_validation_auprc'])
 cpath=a.out/f"{chosen['architecture']}_validation_best_private.pt"; ck=torch.load(cpath,map_location='cpu',weights_only=False)
 m=make_model(chosen['architecture']);m.load_state_dict(ck['state_dict']);m.cuda()
 vy,vs,vg,_=predict(m,loaders['validation']); threshold=threshold_at_specificity(vy,vs,.90)
 ty,ts,tg,ids=predict(m,loaders['test']); test=metrics(ty,ts,threshold['threshold'])
 # Patient-cluster bootstrap for AUROC, AP and locked-threshold performance.
 unique=np.unique(tg);rng=np.random.default_rng(SEED);boot={k:[] for k in ['accuracy','sensitivity','specificity','auroc','auprc']};ix={g:np.flatnonzero(tg==g) for g in unique}
 for _ in range(1000):
  idx=np.concatenate([ix[g] for g in rng.choice(unique,size=len(unique),replace=True)])
  if len(np.unique(ty[idx]))<2:continue
  mm=metrics(ty[idx],ts[idx],threshold['threshold'])
  for k in boot:boot[k].append(mm[k])
 ci={k:[float(np.quantile(v,.025)),float(np.quantile(v,.975))] if v else None for k,v in boot.items()}
 rep={'status':'complete','dataset':'JustRAIGS Zenodo training release','task':'Referable glaucoma (RG) versus no referable glaucoma (NRG)','selected_model':chosen,'candidate_models':candidate_results,'validation_threshold':threshold,'test_metrics_one_shot':test,'patient_group_bootstrap_95_ci':ci,'split_audit':split,'threshold_rule':'max sensitivity subject to validation specificity >= 90%; applied once on internal test','terms':'CC BY-NC-SA 4.0 source data; images and row-level predictions remain on cloud VM. Do not publish model weights or derived data without a license-specific review.','boundary':split['boundary']}
 (a.out/'aggregate_report.json').write_text(json.dumps(rep,indent=2)+'\n')
 with (a.out/'private_test_predictions.csv').open('w') as f:
  w=csv.writer(f);w.writerow(['eye_id','patient_group','label_rg','score_rg','prediction_at_locked_threshold'])
  for e,g,y,s in zip(ids,tg,ty,ts):w.writerow([e,g,int(y),float(s),int(s>=threshold['threshold'])])
 print(json.dumps({'status':'complete','selected_model':chosen,'validation_threshold':threshold,'test_metrics_one_shot':test,'patient_group_bootstrap_95_ci':ci,'boundary':rep['boundary']}),flush=True)
if __name__=='__main__':main()
