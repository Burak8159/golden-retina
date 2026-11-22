#!/usr/bin/env python3
"""Train a 45-label RFMiD-derived fundus model on MedIMeta's expert-adjudicated labels.

The images are an RFMiD-derived cohort already seen in this project. This run
tests label refinement and disease coverage, never independent generalization.
"""
from __future__ import annotations
import argparse,csv,json,random,time
from collections import Counter
from pathlib import Path
import numpy as np
import torch,yaml
from PIL import Image,ImageOps
from sklearn.metrics import accuracy_score,average_precision_score,balanced_accuracy_score,confusion_matrix,roc_auc_score
from torch import nn
from torch.utils.data import DataLoader,Dataset
from torchvision import models,transforms

SEED=8159;MEAN=(.485,.456,.406);STD=(.229,.224,.225)
class Fundus(Dataset):
 def __init__(self,rows,train):
  self.rows=rows;ops=[transforms.Resize(256),transforms.CenterCrop(224)]
  if train:ops += [transforms.RandomHorizontalFlip(),transforms.RandomRotation(7),transforms.ColorJitter(.05,.05,.04,.01)]
  ops += [transforms.ToTensor(),transforms.Normalize(MEAN,STD)];self.tf=transforms.Compose(ops)
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  r=self.rows[i]
  with Image.open(r['path']) as im:x=self.tf(ImageOps.exif_transpose(im).convert('RGB'))
  return x,torch.tensor(r['labels'],dtype=torch.float32)
def make():
 m=models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT);m.classifier[2]=nn.Linear(m.classifier[2].in_features,45);return m.cuda()
def infer(m,dl):
 m.eval();p=[];y=[]
 with torch.inference_mode():
  for x,z in dl:
   x=x.cuda(non_blocking=True)
   with torch.autocast('cuda',dtype=torch.float16):q=m(x)
   p.extend(torch.sigmoid(q.float()).cpu().numpy());y.extend(z.numpy())
 return np.asarray(y,dtype=int),np.asarray(p)
def per_label(y,p,names,thresholds):
 out={}
 for j,name in enumerate(names):
  yy=y[:,j];pp=p[:,j];t=float(thresholds[j]);pred=pp>=t;tn,fp,fn,tp=confusion_matrix(yy,pred,labels=[0,1]).ravel()
  out[name]={'support':int(yy.sum()),'negative_support':int(len(yy)-yy.sum()),'threshold':t,'accuracy':float(accuracy_score(yy,pred)),'sensitivity':float(tp/max(1,tp+fn)),'specificity':float(tn/max(1,tn+fp)),'false_negatives':int(fn),'false_positives':int(fp),'auroc':float(roc_auc_score(yy,pp)) if len(np.unique(yy))==2 else None,'average_precision':float(average_precision_score(yy,pp)) if yy.sum() else None}
 return out
def thresholds_on_validation(y,p,min_pos=10,min_neg=10):
 result=[]
 for j in range(y.shape[1]):
  yy=y[:,j];pp=p[:,j]
  if yy.sum()<min_pos or (len(yy)-yy.sum())<min_neg:result.append(.5);continue
  c=np.r_[0.,np.unique(pp),1.]
  scores=[]
  for t in c:
   pred=pp>=t;tn,fp,fn,tp=confusion_matrix(yy,pred,labels=[0,1]).ravel();sens=tp/max(1,tp+fn);spec=tn/max(1,tn+fp);scores.append((.5*(sens+spec),sens,t))
  result.append(float(max(scores)[2]))
 return result
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--epochs',type=int,default=8);ap.add_argument('--seconds',type=int,default=2400);ap.add_argument('--batch-size',type=int,default=32);ap.add_argument('--workers',type=int,default=4);a=ap.parse_args()
 if not torch.cuda.is_available():raise RuntimeError('CUDA L4 required')
 random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
 meta=yaml.safe_load((a.root/'info.yaml').read_text());names=[v for _,v in sorted(((int(k),v) for k,v in meta['tasks'][1]['labels'].items()))]
 rows=[]
 with (a.root/'annotations.csv').open(newline='') as f:
  for r in csv.DictReader(f):rows.append({'path':a.root/r['filepath'],'split':r['split'],'labels':json.loads(r['tasks/disease labels']),'original_filepath':r['original_filepath']})
 if len(rows)!=3200 or len(names)!=45 or any(not r['path'].is_file() for r in rows):raise RuntimeError('MedIMeta manifest/image audit failed')
 parts={k:[r for r in rows if r['split']==k] for k in ['train','val','test']}
 if [len(parts[k]) for k in parts]!=[1920,640,640]:raise RuntimeError('official split counts mismatch')
 a.out.mkdir(parents=True,exist_ok=True)
 support={k:{n:int(np.asarray([r['labels'] for r in v])[:,i].sum()) for i,n in enumerate(names)} for k,v in parts.items()}
 (a.out/'private_split_support.json').write_text(json.dumps({'dataset':'MedIMeta fundus / RFMiD-derived','license':'CC BY-SA 4.0','counts':{k:len(v) for k,v in parts.items()},'positive_support':support,'patient_ids_available':False,'split':'publisher-provided 1920/640/640 image split; same RFMiD-derived image identities were previously used in this project'},indent=2)+'\n')
 loaders={k:DataLoader(Fundus(v,k=='train'),batch_size=a.batch_size,shuffle=k=='train',num_workers=a.workers,pin_memory=True) for k,v in parts.items()}
 yt=np.asarray([r['labels'] for r in parts['train']],dtype=np.float32);pos=yt.sum(0);neg=len(yt)-pos;pw=torch.tensor(np.clip(neg/np.maximum(pos,1),1,15),dtype=torch.float32,device='cuda')
 m=make();opt=torch.optim.AdamW(m.parameters(),lr=1.5e-5,weight_decay=3e-4);best=float('inf');bestep=0;stale=0;start=time.time();ck=a.out/'convnext_tiny_best_private.pt'
 for ep in range(1,a.epochs+1):
  m.train();ls=0.;seen=0
  for x,y in loaders['train']:
   x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True);opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.float16):z=m(x);loss=nn.functional.binary_cross_entropy_with_logits(z,y,pos_weight=pw)
   loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),2.);opt.step();ls+=float(loss.detach())*len(y);seen+=len(y)
  vy,vp=infer(m,loaders['val']);vl=float(nn.functional.binary_cross_entropy(torch.tensor(vp.clip(1e-7,1-1e-7),dtype=torch.float32),torch.tensor(vy,dtype=torch.float32)).item());print(json.dumps({'epoch':ep,'train_loss':ls/max(1,seen),'val_bce':vl}),flush=True)
  if vl<best-1e-5:best,bestep,stale=vl,ep,0;torch.save({'state_dict':m.state_dict(),'architecture':'convnext_tiny','labels':names,'epoch':ep},ck)
  else:stale+=1
  if stale>=2 or time.time()-start>=a.seconds:break
 del m;torch.cuda.empty_cache();state=torch.load(ck,map_location='cpu',weights_only=False)['state_dict'];m=make();m.load_state_dict(state)
 vy,vp=infer(m,loaders['val']);ths=thresholds_on_validation(vy,vp);ty,tp=infer(m,loaders['test']);lab=per_label(ty,tp,names,ths)
 eligible=[v for v in lab.values() if v['support']>=10 and v['auroc'] is not None]
 report={'status':'complete','dataset':'MedIMeta fundus / RFMiD-derived 45-label task','architecture':'ConvNeXt-Tiny','best_epoch':bestep,'best_validation_bce':best,'n_test':len(ty),'macro_auroc_all_supported':float(np.mean([v['auroc'] for v in eligible])) if eligible else None,'evaluated_label_count_support_ge_10':len(eligible),'per_label':lab,'threshold_policy':'validation-only balanced-accuracy threshold if val positive and negative supports are each >=10; otherwise fixed 0.5','boundary':'Publisher image split only; no patient IDs. Images are the RFMiD-derived cohort, whose test identities were previously opened in this project. This measures development under expert-adjudicated 45-condition labels, not independent external performance or clinical validity. Rare labels remain exploratory.'}
 (a.out/'aggregate_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':'complete','best_epoch':bestep,'macro_auroc':report['macro_auroc_all_supported'],'evaluated_labels':len(eligible),'per_label':lab,'boundary':report['boundary']},indent=2),flush=True)
if __name__=='__main__':main()
