#!/usr/bin/env python3
"""Exploratory site-held-out disease severity models for BMRD.

The source has no documented patient IDs. This script therefore reports image-level
held-out-hospital metrics, does not call the estimates patient independent, and
excludes exact-hash label conflicts in the precomputed private manifest.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,random,time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageOps
from sklearn.metrics import accuracy_score,balanced_accuracy_score,confusion_matrix,f1_score,roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader,Dataset
from torchvision import models,transforms
SEED=8159
MEAN=(.485,.456,.406);STD=(.229,.224,.225)
TASKS={
 'dr':{'train':'Shiromoni BNSB Hospital','test':'Khulna Laser Clinic','prefix':'DR_Level_'},
 'amd':{'train':'Shiromoni BNSB Hospital','test':'Khulna Laser Clinic','prefix':'AMD_'},
 'glaucoma':{'train':'Khulna Laser Clinic','test':'Anwar khan Modern Medical','prefix':'Glaucoma_Level_'}
}
class Fundus(Dataset):
 def __init__(self,rows,size,train=False):
  self.rows=rows;ops=[transforms.Resize((size,size))]
  if train:ops += [transforms.RandomHorizontalFlip(),transforms.RandomRotation(8),transforms.ColorJitter(.06,.06,.04,.01)]
  ops += [transforms.ToTensor(),transforms.Normalize(MEAN,STD)];self.tf=transforms.Compose(ops)
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  r=self.rows[i]
  with Image.open(r['path']) as im:x=self.tf(ImageOps.exif_transpose(im).convert('RGB'))
  return x,r['y']
def build(name,n):
 if name=='convnext_tiny':
  m=models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT);m.classifier[2]=nn.Linear(m.classifier[2].in_features,n);return m
 m=models.resnet50(weights=models.ResNet50_Weights.DEFAULT);m.fc=nn.Linear(m.fc.in_features,n);return m
def infer(m,loader):
 m.eval();pp=[];yy=[]
 with torch.inference_mode():
  for x,y in loader:
   x=x.cuda(non_blocking=True)
   with torch.autocast('cuda',dtype=torch.float16):z=m(x)
   pp.extend(torch.softmax(z.float(),1).cpu().numpy());yy.extend(y.numpy())
 return np.asarray(pp),np.asarray(yy,dtype=int)
def metric(y,p,names):
 pred=p.argmax(1);cm=confusion_matrix(y,pred,labels=range(len(names)));out={'n_images':len(y),'accuracy':float(accuracy_score(y,pred)),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),'confusion_matrix_rows_true_columns_pred':cm.tolist(),'classes':{}}
 auc=[]
 for i,n in enumerate(names):
  tp=int(cm[i,i]);fn=int(cm[i].sum()-tp);fp=int(cm[:,i].sum()-tp);tn=int(len(y)-tp-fn-fp)
  try:a=float(roc_auc_score((y==i).astype(int),p[:,i]));auc.append(a)
  except ValueError:a=None
  out['classes'][n]={'support':int((y==i).sum()),'sensitivity':tp/(tp+fn) if tp+fn else None,'specificity':tn/(tn+fp) if tn+fp else None,'false_negatives':fn,'false_positives':fp,'auroc_ovr':a}
 out['macro_auroc_ovr']=float(np.mean(auc)) if auc else None
 return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--epochs',type=int,default=7);ap.add_argument('--seconds-per-model',type=int,default=900);ap.add_argument('--size',type=int,default=256);ap.add_argument('--batch',type=int,default=16);ap.add_argument('--workers',type=int,default=4);a=ap.parse_args()
 if not torch.cuda.is_available():raise RuntimeError('L4 CUDA required')
 random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);a.out.mkdir(parents=True,exist_ok=True)
 rows=list(csv.DictReader(a.manifest.open()));report={'dataset':'BMRD Mendeley 959whsbc6v.1','selection':'validation macro-F1; one test hospital per disease, opened once','patient_ids':'not supplied/documented; validation and test metrics are image-level, with test source held out by hospital','label_scope':'folder names Level 1-4 and AMD dry/wet/no-AMD are reported as source classes; grade thresholds are not mapped to a clinical severity standard','results':{}}
 for task,cfg in TASKS.items():
  prefix=cfg['prefix']; selected=[r for r in rows if r['label'].startswith(prefix) and r['hospital'] in {cfg['train'],cfg['test']}]
  names=sorted({r['label'] for r in selected});mapping={n:i for i,n in enumerate(names)}
  tr=[{**r,'y':mapping[r['label']]} for r in selected if r['hospital']==cfg['train']];te=[{**r,'y':mapping[r['label']]} for r in selected if r['hospital']==cfg['test']]
  if len(names)<2 or not tr or not te or any(sum(r['y']==i for r in tr)<5 or sum(r['y']==i for r in te)<2 for i in range(len(names))):
   report['results'][task]={'status':'not_fitted_insufficient_class_support','train_site':cfg['train'],'test_site':cfg['test'],'class_names':names,'train_counts':dict(Counter(r['label'] for r in tr)),'test_counts':dict(Counter(r['label'] for r in te))};print(json.dumps({'task':task,**report['results'][task]}),flush=True);continue
  yy=np.array([r['y'] for r in tr]); ix=np.arange(len(tr));ixtr,ixv=train_test_split(ix,test_size=.2,random_state=SEED,stratify=yy)
  train=[tr[i] for i in ixtr];val=[tr[i] for i in ixv]
  loaders={'train':DataLoader(Fundus(train,a.size,True),batch_size=a.batch,shuffle=True,num_workers=a.workers,pin_memory=True,persistent_workers=a.workers>0),'validation':DataLoader(Fundus(val,a.size),batch_size=a.batch,shuffle=False,num_workers=a.workers,pin_memory=True,persistent_workers=a.workers>0),'test':DataLoader(Fundus(te,a.size),batch_size=a.batch,shuffle=False,num_workers=a.workers,pin_memory=True,persistent_workers=a.workers>0)}
  candidates=[];taskstart=time.time()
  for arch in ['convnext_tiny','resnet50']:
   m=build(arch,len(names)).cuda();opt=torch.optim.AdamW(m.parameters(),lr=2e-5,weight_decay=2e-4);cnt=np.bincount(yy,minlength=len(names));cw=torch.tensor((cnt.mean()/np.maximum(cnt,1))**.5,dtype=torch.float32,device='cuda');best=-1;bestep=0;stale=0;start=time.time();ck=a.out/f'{task}_{arch}_validation_best_private.pt'
   for ep in range(1,a.epochs+1):
    m.train();loss_sum=0;seen=0
    for x,y in loaders['train']:
     x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True);opt.zero_grad(set_to_none=True)
     with torch.autocast('cuda',dtype=torch.float16):z=m(x);loss=nn.functional.cross_entropy(z,y,weight=cw)
     loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),2);opt.step();loss_sum+=float(loss.detach())*len(y);seen+=len(y)
    vp,vy=infer(m,loaders['validation']);vm=metric(vy,vp,names);score=vm['macro_f1'];print(json.dumps({'task':task,'model':arch,'epoch':ep,'train_loss':loss_sum/max(1,seen),'validation':vm}),flush=True)
    if score>best+1e-4:best=score;bestep=ep;stale=0;torch.save({'state_dict':m.state_dict(),'classes':names,'epoch':ep,'task':task},ck)
    else:stale+=1
    if stale>=2 or time.time()-start>=a.seconds_per_model:break
   candidates.append({'architecture':arch,'best_epoch':bestep,'best_validation_macro_f1':best,'checkpoint_sha256':hashlib.sha256(ck.read_bytes()).hexdigest() if ck.exists() else None});del m;torch.cuda.empty_cache()
   if time.time()-taskstart>=2*a.seconds_per_model+120:break
  viable=[x for x in candidates if x['checkpoint_sha256']]
  if not viable:report['results'][task]={'status':'no_checkpoint','candidates':candidates};continue
  chosen=max(viable,key=lambda x:x['best_validation_macro_f1']);ck=a.out/f"{task}_{chosen['architecture']}_validation_best_private.pt";state=torch.load(ck,map_location='cpu',weights_only=False);m=build(chosen['architecture'],len(names));m.load_state_dict(state['state_dict']);m.cuda();tp,ty=infer(m,loaders['test']);tm=metric(ty,tp,names)
  report['results'][task]={'status':'complete','train_site':cfg['train'],'validation_source':cfg['train'],'test_site':cfg['test'],'class_names':names,'train_counts':dict(Counter(r['label'] for r in train)),'validation_counts':dict(Counter(r['label'] for r in val)),'test_counts':dict(Counter(r['label'] for r in te)),'candidate_models':candidates,'selected_model':chosen['architecture'],'test_metrics_one_shot':tm,'boundary':'hospital/source-held-out, but no patient IDs; image-level estimates. Not patient-independent or clinical validation.'};print(json.dumps({'task':task,'selected':chosen,'test_metrics':tm}),flush=True);del m;torch.cuda.empty_cache()
 (a.out/'aggregate_report.json').write_text(json.dumps(report,indent=2)+'\n');print('REPORT_WRITTEN',flush=True)
if __name__=='__main__':main()
