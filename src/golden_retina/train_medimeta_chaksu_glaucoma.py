#!/usr/bin/env python3
"""Train glaucoma-suspect heads on MedIMeta's five-expert Chákṣu labels.

Known patient IDs are grouped; rows without patient IDs are singleton groups
and explicitly prevent a full patient-independent interpretation.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,random,time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageOps
from sklearn.metrics import accuracy_score,average_precision_score,balanced_accuracy_score,confusion_matrix,f1_score,roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader,Dataset,WeightedRandomSampler
from torchvision import models,transforms

SEED=8159;MEAN=(.485,.456,.406);STD=(.229,.224,.225)
class DS(Dataset):
 def __init__(self,rows,train):
  self.rows=rows;ops=[transforms.Resize((224,224))]
  if train:ops += [transforms.RandomHorizontalFlip(),transforms.RandomRotation(7),transforms.ColorJitter(.04,.04,.03,.01)]
  ops += [transforms.ToTensor(),transforms.Normalize(MEAN,STD)];self.tf=transforms.Compose(ops)
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  r=self.rows[i]
  with Image.open(r['path']) as im:x=self.tf(ImageOps.exif_transpose(im).convert('RGB'))
  return x,r['label'],r['group']
def make(arch):
 if arch=='convnext_tiny':m=models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT);m.classifier[2]=nn.Linear(m.classifier[2].in_features,1)
 else:m=models.resnet50(weights=models.ResNet50_Weights.DEFAULT);m.fc=nn.Linear(m.fc.in_features,1)
 return m.cuda()
def predict(m,dl):
 m.eval();ys=[];ps=[];gs=[]
 with torch.inference_mode():
  for x,y,g in dl:
   x=x.cuda(non_blocking=True)
   with torch.autocast('cuda',dtype=torch.float16):z=m(x).flatten()
   ys.extend(y.numpy());ps.extend(torch.sigmoid(z.float()).cpu().numpy());gs.extend(g)
 return np.asarray(ys,dtype=int),np.asarray(ps),np.asarray(gs)
def metric(y,p,t):
 pred=p>=t;tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
 return {'n':int(len(y)),'positive_support':int(y.sum()),'accuracy':float(accuracy_score(y,pred)),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),'auroc':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,'auprc':float(average_precision_score(y,p)) if y.sum() else None,'threshold':float(t),'sensitivity':float(tp/max(1,tp+fn)),'specificity':float(tn/max(1,tn+fp)),'false_negatives':int(fn),'false_positives':int(fp)}
def threshold90(y,p):
 candidates=np.r_[np.unique(p),np.nextafter(p.max(),np.inf)]; valid=[metric(y,p,float(t)) for t in candidates]; valid=[x for x in valid if x['specificity']>=.90]
 return max(valid,key=lambda z:(z['sensitivity'],z['specificity']))['threshold'] if valid else 1.01
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--epochs',type=int,default=7);ap.add_argument('--seconds-per-model',type=int,default=900);ap.add_argument('--batch-size',type=int,default=32);ap.add_argument('--workers',type=int,default=4);a=ap.parse_args()
 if not torch.cuda.is_available():raise RuntimeError('CUDA L4 required')
 random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
 rows=[];missing_patient=0;label_counts=Counter()
 with (a.root/'annotations.csv').open(newline='') as f:
  for r in csv.DictReader(f):
   label=str(r['Glaucoma suspect']).strip().lower(); y=int(label in {'suspect','1','true'})
   path=a.root/r['filepath']
   if not path.is_file():continue
   pid=str(r.get('patient_id','')).strip()
   if not pid:missing_patient+=1
   rows.append({'path':path,'label':y,'patient_id':pid or None,'group':f'patient:{pid}' if pid else f'unlinked:{r["filepath"]}','split_original':r['split'],'eye':r.get('image_name','')})
   label_counts[y]+=1
 if len(rows)!=1345:raise RuntimeError(f'expected 1345 mapped images, got {len(rows)}')
 y=np.asarray([r['label'] for r in rows]);g=np.asarray([r['group'] for r in rows]);outer=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED);tv,te=next(outer.split(np.zeros(len(y)),y,g));inner=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED+1);tr,va=next(inner.split(np.zeros(len(tv)),y[tv],g[tv]));parts={'train':[rows[i] for i in tv[tr]],'validation':[rows[i] for i in tv[va]],'test':[rows[i] for i in te]}
 gs={k:{r['group'] for r in v} for k,v in parts.items()}
 if any(gs[a]&gs[b] for a,b in [('train','validation'),('train','test'),('validation','test')]):raise RuntimeError('group leakage')
 split={'dataset':'MedIMeta glaucoma / Chákṣu; CC BY-SA 4.0','counts':{k:{'n':len(v),'positive':sum(r['label'] for r in v),'groups':len({r['group'] for r in v})} for k,v in parts.items()},'n_images':len(rows),'positive_count':label_counts[1],'negative_count':label_counts[0],'patient_id_present':len(rows)-missing_patient,'patient_id_missing':missing_patient,'patient_id_coverage':(len(rows)-missing_patient)/len(rows),'grouping':'known patient_id values are kept together; rows without IDs are singleton image groups. Partial grouping only, not a patient-independent test.','source_official_split_counts':dict(Counter(r['split_original'] for r in rows))}
 a.out.mkdir(parents=True,exist_ok=True);(a.out/'private_split_audit.json').write_text(json.dumps(split,indent=2)+'\n')
 loaders={k:DataLoader(DS(v,k=='train'),batch_size=a.batch_size,shuffle=k=='train',num_workers=a.workers,pin_memory=True) for k,v in parts.items()}
 train_y=np.asarray([r['label'] for r in parts['train']]);n0=max(1,int((train_y==0).sum()));n1=max(1,int((train_y==1).sum()));weights=torch.tensor([(.8/n0 if y==0 else .2/n1) for y in train_y],dtype=torch.double);loaders['train']=DataLoader(DS(parts['train'],True),batch_size=a.batch_size,sampler=WeightedRandomSampler(weights,len(train_y),replacement=True),num_workers=a.workers,pin_memory=True)
 candidates=[]
 for arch in ['convnext_tiny','resnet50']:
  m=make(arch);opt=torch.optim.AdamW(m.parameters(),lr=2e-5,weight_decay=2e-4);best=-1.;bestep=0;stale=0;start=time.time();path=a.out/f'{arch}_validation_best_private.pt'
  for ep in range(1,a.epochs+1):
   m.train();loss_sum=0;seen=0
   for x,yy,_ in loaders['train']:
    x=x.cuda(non_blocking=True);yy=yy.float().cuda(non_blocking=True);opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.float16):z=m(x).flatten();loss=nn.functional.binary_cross_entropy_with_logits(z,yy,pos_weight=torch.tensor([min(8,n0/n1)],device='cuda'))
    loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),2.);opt.step();loss_sum+=float(loss.detach())*len(yy);seen+=len(yy)
   vy,vp,_=predict(m,loaders['validation']);va=metric(vy,vp,.5);print(json.dumps({'architecture':arch,'epoch':ep,'train_loss':loss_sum/max(1,seen),'validation':va}),flush=True)
   if va['auprc'] is not None and va['auprc']>best+1e-5:best,bestep,stale=va['auprc'],ep,0;torch.save({'state_dict':m.state_dict(),'architecture':arch,'epoch':ep},path)
   else:stale+=1
   if stale>=2 or time.time()-start>=a.seconds_per_model:break
  candidates.append({'architecture':arch,'best_epoch':bestep,'best_validation_auprc':best,'checkpoint':str(path)});del m;torch.cuda.empty_cache()
 chosen=max(candidates,key=lambda z:z['best_validation_auprc']);ck=torch.load(chosen['checkpoint'],map_location='cpu',weights_only=False);m=make(chosen['architecture']);m.load_state_dict(ck['state_dict']);vy,vp,_=predict(m,loaders['validation']);th=threshold90(vy,vp);ty,tp,tg=predict(m,loaders['test']);test=metric(ty,tp,th)
 rep={'status':'complete','task':'glaucoma suspect vs normal, five-expert majority label','dataset':'MedIMeta glaucoma / Chákṣu','selected_model':{k:v for k,v in chosen.items() if k!='checkpoint'},'candidate_models':[{k:v for k,v in c.items() if k!='checkpoint'} for c in candidates],'validation_threshold_target_specificity_90':th,'test_metrics_one_shot':test,'split_audit':split,'license':'CC BY-SA 4.0; weights not publicly redistributed by this run','boundary':'Only 60/1,345 images have a patient_id (30 distinct values); missing IDs are singleton groups. Partial known-ID grouping does not establish patient-level independence. Cohort has been introduced here for this project, but exact image-hash overlap audit is pending; not clinical validation.'}
 (a.out/'aggregate_report.json').write_text(json.dumps(rep,indent=2)+'\n');print(json.dumps({'status':'complete','selected':rep['selected_model'],'test':test,'boundary':rep['boundary']},indent=2),flush=True)
if __name__=='__main__':main()
