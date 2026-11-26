#!/usr/bin/env python3
"""Train/select multi-label fundus models on CLARUS700 patient splits.

The official test partition is not used for architecture, epoch, or threshold
selection. In development-only mode, the test manifest is never read; only
aggregate outputs are written and row identifiers/predictions remain on RTX.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,random,time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageOps
from sklearn.metrics import average_precision_score,roc_auc_score,balanced_accuracy_score,confusion_matrix
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision import models,transforms

SEED=8159; LABELS=("DR","RVO","high_myopia","other_disease"); MEAN=(.485,.456,.406); STD=(.229,.224,.225)
MISSING_IMAGE_ROWS=[]

def digest(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()

def rows_for(root,csv_path):
 with csv_path.open(encoding="utf-8-sig",newline="") as f: rows=list(csv.DictReader(f))
 if not rows: raise ValueError(f"Empty official split: {csv_path.name}")
 files=list(root.rglob("*")); imgmap=defaultdict(list)
 for p in files:
  if p.is_file() and p.suffix.lower() in {".jpg",".jpeg",".png",".tif",".tiff"}: imgmap[p.name].append(p)
 out=[]
 for r in rows:
  if str(r.get("model_eligible","1")) not in {"1","1.0"}: continue
  name=Path(r["image_file"]).name
  cand=imgmap.get(name,[])
  if len(cand)==0:
   split_name={"train_final_public":"train","val_final_public":"validation","test_final_public":"test"}.get(csv_path.stem,csv_path.stem)
   MISSING_IMAGE_ROWS.append({"split":split_name,"patient_id":r.get("patient_id"),"image_file_in_manifest":name,"image_id":r.get("image_id"),"status":"excluded: official manifest filename absent from downloaded image archives"})
   continue
  if len(cand)!=1: raise FileNotFoundError(f"Expected one unique image for a public split row; found {len(cand)}")
  y=[int(float(r[k])) for k in LABELS]
  out.append({"path":cand[0],"patient":r["patient_id"],"y":y,"sha":digest(cand[0]),"quality":r.get("quality","")})
 return out

def audit_parts(parts):
 patients={k:{r["patient"] for r in v} for k,v in parts.items()}
 hashes={k:{r["sha"] for r in v} for k,v in parts.items()}
 pairs=[(a,b) for i,a in enumerate(parts) for b in list(parts)[i+1:]]
 overlaps={f"{a}_{b}":len(patients[a]&patients[b]) for a,b in pairs}
 hash_overlaps={f"{a}_{b}":len(hashes[a]&hashes[b]) for a,b in pairs}
 if any(overlaps.values()): raise RuntimeError(f"Patient overlap in official partitions: {overlaps}")
 if any(hash_overlaps.values()): raise RuntimeError(f"Exact image overlap in official partitions: {hash_overlaps}")
 missing_by_split={k:sum(x["split"]==k for x in MISSING_IMAGE_ROWS) for k in parts}
 return {"split_counts":{k:{"images":len(v),"patients":len(patients[k]),"positives":{lab:int(sum(r["y"][i] for r in v)) for i,lab in enumerate(LABELS)}} for k,v in parts.items()},"patient_overlap":overlaps,"exact_hash_overlap":hash_overlaps,"missing_manifest_images_excluded_count":len(MISSING_IMAGE_ROWS),"missing_manifest_images_excluded_by_split":missing_by_split}

def remove_cross_split_exact_duplicates(parts):
 """Keep first copy by train > validation > test; never expose an exact copy across splits."""
 seen={};removed=[]
 for split in parts:
  kept=[]
  for r in parts[split]:
   prior=seen.get(r["sha"])
   if prior is not None:
    removed.append({"excluded_split":split,"excluded_patient_id":r["patient"],"excluded_filename":r["path"].name,"retained_split":prior["split"],"retained_patient_id":prior["patient"],"retained_filename":prior["filename"],"sha256":r["sha"]})
    continue
   seen[r["sha"]]={"split":split,"patient":r["patient"],"filename":r["path"].name};kept.append(r)
  parts[split]=kept
 return removed

class Img(Dataset):
 def __init__(self,rows,train,size):
  self.rows=rows; ops=[transforms.Resize((size,size))]
  if train: ops += [transforms.RandomHorizontalFlip(),transforms.RandomRotation(8),transforms.ColorJitter(.06,.06,.05,.01)]
  ops += [transforms.ToTensor(),transforms.Normalize(MEAN,STD)]; self.tf=transforms.Compose(ops)
 def __len__(self): return len(self.rows)
 def __getitem__(self,i):
  r=self.rows[i]
  with Image.open(r["path"]) as im: x=self.tf(ImageOps.exif_transpose(im).convert("RGB"))
  return x,torch.tensor(r["y"],dtype=torch.float32),i

def make(arch,n):
 if arch in {"convnext_tiny","convnext_mendeley"}:
  m=models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT if arch=="convnext_tiny" else None); m.classifier[2]=nn.Linear(m.classifier[2].in_features,n)
 elif arch=="resnet50":
  m=models.resnet50(weights=models.ResNet50_Weights.DEFAULT);m.fc=nn.Linear(m.fc.in_features,n)
 elif arch=="efficientnet_b0":
  m=models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT);m.classifier[1]=nn.Linear(m.classifier[1].in_features,n)
 else: raise ValueError(arch)
 return m

def score(y,p):
 auc=[];ap=[];per={}
 for j,name in enumerate(LABELS):
  yy=y[:,j].astype(int)
  if len(np.unique(yy))<2: continue
  auc.append(float(roc_auc_score(yy,p[:,j])));ap.append(float(average_precision_score(yy,p[:,j])))
  per[name]={"n_positive":int(yy.sum()),"n_negative":int(len(yy)-yy.sum()),"auroc":auc[-1],"average_precision":ap[-1]}
 return {"macro_auroc":float(np.mean(auc)) if auc else None,"macro_average_precision":float(np.mean(ap)) if ap else None,"per_label":per}

def select_thresholds(y,p):
 thresholds={}
 for j,name in enumerate(LABELS):
  yy=y[:,j].astype(int)
  if len(np.unique(yy))<2: thresholds[name]=.5; continue
  candidates=np.r_[np.unique(p[:,j]),np.nextafter(p[:,j].max(),np.inf)]
  vals=[]
  for t in candidates:
   pred=p[:,j]>=t; tn,fp,fn,tp=confusion_matrix(yy,pred,labels=[0,1]).ravel()
   sens=tp/(tp+fn) if tp+fn else 0.;spec=tn/(tn+fp) if tn+fp else 0.
   vals.append((.5*(sens+spec),sens,-float(t),float(t)))
  thresholds[name]=max(vals)[-1]
 return thresholds

def select_thresholds_for_sensitivity(y,p,target=0.95):
 """Select the highest threshold meeting target sensitivity on calibration patients."""
 thresholds={}
 for j,name in enumerate(LABELS):
  yy=y[:,j].astype(int); pos=p[yy==1,j]
  if len(pos)==0:
   thresholds[name]=.5; continue
  # Moving just below the lower quantile includes enough calibration positives.
  q=max(0.0,1.0-target)
  thresholds[name]=float(np.nextafter(np.quantile(pos,q,method="lower"),-np.inf))
 return thresholds

def operating_metrics(y,p,thresholds):
 out={}
 for j,name in enumerate(LABELS):
  yy=y[:,j].astype(int);t=thresholds[name]; pred=p[:,j]>=t;tn,fp,fn,tp=confusion_matrix(yy,pred,labels=[0,1]).ravel()
  out[name]={"support":int(yy.sum()),"negative":int(len(yy)-yy.sum()),"threshold":float(t),"sensitivity":float(tp/(tp+fn)) if tp+fn else None,"specificity":float(tn/(tn+fp)) if tn+fp else None,"accuracy":float((pred==yy).mean()),"false_negatives":int(fn),"false_positives":int(fp)}
 return out

def patient_bootstrap(y,p,rows,thresholds,n_boot=1000,seed=8159):
 groups=defaultdict(list)
 for i,r in enumerate(rows): groups[r["patient"]].append(i)
 keys=sorted(groups);rng=np.random.default_rng(seed);out={}
 for j,name in enumerate(LABELS):
  aucs=[];sens=[];spec=[]
  for _ in range(n_boot):
   draw=rng.choice(keys,size=len(keys),replace=True); idx=np.asarray([i for patient in draw for i in groups[patient]],dtype=int)
   yy=y[idx,j].astype(int);pp=p[idx,j]
   if len(np.unique(yy))<2: continue
   aucs.append(float(roc_auc_score(yy,pp)))
   pr=pp>=thresholds[name];tn,fp,fn,tp=confusion_matrix(yy,pr,labels=[0,1]).ravel()
   if tp+fn:sens.append(float(tp/(tp+fn)))
   if tn+fp:spec.append(float(tn/(tn+fp)))
  def ci(v): return [float(np.quantile(v,.025)),float(np.quantile(v,.975))] if v else None
  out[name]={"patient_cluster_bootstrap_replicates_requested":n_boot,"valid_auroc_replicates":len(aucs),"auroc_95ci":ci(aucs),"sensitivity_95ci":ci(sens),"specificity_95ci":ci(spec)}
 return out

def evaluate(m,dl):
 m.eval(); ys=[];ps=[]
 with torch.inference_mode():
  for x,y,_ in dl:
   with torch.autocast("cuda",dtype=torch.float16): z=m(x.cuda(non_blocking=True))
   ys.append(y.numpy());ps.append(torch.sigmoid(z.float()).cpu().numpy())
 return np.concatenate(ys),np.concatenate(ps)

def main():
 global LABELS
 ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,required=True);ap.add_argument("--splits",type=Path,required=True);ap.add_argument("--out",type=Path,required=True);ap.add_argument("--mendeley-checkpoint",type=Path);ap.add_argument("--targets",default=",".join(LABELS),help="Comma-separated CLARUS targets; use one target to train a disease-specific specialist.");ap.add_argument("--epochs",type=int,default=12);ap.add_argument("--seconds-per-model",type=int,default=2400);ap.add_argument("--image-size",type=int,default=384);ap.add_argument("--batch-size",type=int,default=8);ap.add_argument("--workers",type=int,default=4);ap.add_argument("--audit-only",action="store_true");ap.add_argument("--allow-official-test",action="store_true",help="Explicitly open the official test manifest and labels; development-only is the default.");ap.add_argument("--calibration-sensitivity",type=float,default=.95);a=ap.parse_args()
 a.development_only=not a.allow_official_test
 requested=tuple(x.strip() for x in a.targets.split(",") if x.strip())
 if not requested or any(x not in LABELS for x in requested): ap.error(f"--targets must be a comma-separated subset of {LABELS}")
 LABELS=requested
 if not torch.cuda.is_available(): raise RuntimeError("CUDA GPU required")
 random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
 split_files=(("train","train"),("validation","val"))
 if not a.development_only: split_files += (("test","test"),)
 parts={k:rows_for(a.root,a.splits/f"{f}_final_public.csv") for k,f in split_files}
 if a.development_only:
  # Calibration is drawn only from official training patients; the sealed test
  # manifest and its labels are never read in this mode.
  patients=sorted({r["patient"] for r in parts["train"]})
  rng=np.random.default_rng(SEED);rng.shuffle(patients)
  n_cal=max(1,round(.20*len(patients)));cal_patients=set(patients[:n_cal])
  calibration=[r for r in parts["train"] if r["patient"] in cal_patients]
  train=[r for r in parts["train"] if r["patient"] not in cal_patients]
  parts["train"]=train;parts["calibration"]=calibration
 exact_duplicate_exclusions=remove_cross_split_exact_duplicates(parts)
 audit=audit_parts(parts);audit["exact_duplicate_rows_excluded_count"]=len(exact_duplicate_exclusions);audit["exact_duplicate_rows_excluded_by_split"]={k:sum(x["excluded_split"]==k for x in exact_duplicate_exclusions) for k in parts};a.out.mkdir(parents=True,exist_ok=True)
 if a.audit_only:
  (a.out/"official_split_audit_aggregate.json").write_text(json.dumps(audit,indent=2)+"\n")
  print(json.dumps({"status":"audit_complete","split_audit":audit}),flush=True);return
 train_y=np.asarray([r["y"] for r in parts["train"]],dtype=np.float32); counts=train_y.sum(0); neg=len(train_y)-counts
 if np.any(counts<5): raise RuntimeError(f"insufficient train support for configured labels: {counts.tolist()}")
 pw=torch.tensor(np.sqrt(neg/np.maximum(counts,1)).clip(1,8),device="cuda",dtype=torch.float32)
 loaders={k:DataLoader(Img(v,k=="train",a.image_size),batch_size=a.batch_size*(2 if k!="train" else 1),shuffle=k=="train",num_workers=a.workers,pin_memory=True) for k,v in parts.items()}
 candidate_results=[]; started=time.time(); specifications=["convnext_tiny","resnet50","efficientnet_b0"]
 base_state=None
 if a.mendeley_checkpoint:
  base=torch.load(a.mendeley_checkpoint,map_location="cpu",weights_only=False); base_state=base["state_dict"]
  specifications.insert(1,"convnext_mendeley")
 for arch in specifications:
  model=make(arch,len(LABELS))
  if arch=="convnext_mendeley":
   own=model.state_dict(); transfer={k:v for k,v in base_state.items() if not k.startswith("classifier.") and k in own and own[k].shape==v.shape}
   if len(transfer)<100: raise RuntimeError(f"unexpected Mendeley encoder transfer coverage {len(transfer)}")
   model.load_state_dict(transfer,strict=False)
  model=model.cuda();opt=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=2e-4)
  best=-1.;best_auc=-1.;best_ep=0;stale=0;beg=time.time();ck=a.out/f"{arch}_validation_best.pt"
  for ep in range(1,a.epochs+1):
   model.train();loss_sum=0.;seen=0
   for x,y,_ in loaders["train"]:
    x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True);opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda",dtype=torch.float16): z=model(x);loss=nn.functional.binary_cross_entropy_with_logits(z,y,pos_weight=pw)
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step();loss_sum+=loss.item()*len(y);seen+=len(y)
   vy,vp=evaluate(model,loaders["validation"]);s=score(vy,vp)
   print(json.dumps({"candidate":arch,"epoch":ep,"train_loss":loss_sum/max(seen,1),"validation":s}),flush=True)
   key=s["macro_average_precision"] or -1
   if key>best+1e-4:
    best=key;best_auc=s["macro_auroc"] or -1;best_ep=ep;stale=0;torch.save({"state_dict":model.state_dict(),"architecture":arch,"labels":LABELS,"epoch":ep,"image_size":a.image_size},ck)
   else: stale+=1
   if stale>=3 or time.time()-beg>=a.seconds_per_model: break
  candidate_results.append({"architecture":arch,"best_epoch":best_ep,"validation_macro_average_precision":best,"validation_macro_auroc":best_auc,"checkpoint":str(ck)})
  del model;torch.cuda.empty_cache()
 chosen=max(candidate_results,key=lambda x:(x["validation_macro_average_precision"],x["validation_macro_auroc"]))
 ck=torch.load(chosen["checkpoint"],map_location="cpu",weights_only=False); model=make(chosen["architecture"],len(LABELS));model.load_state_dict(ck["state_dict"]);model.cuda()
 vy,vp=evaluate(model,loaders["validation"])
 if a.development_only:
  cy,cp=evaluate(model,loaders["calibration"]);thresholds=select_thresholds_for_sensitivity(cy,cp,a.calibration_sensitivity)
  report={"status":"complete","dataset":"CLARUS700 Figshare 2026 public release","modality":"single-centre ZEISS CLARUS 700 approximately 133-degree widefield fundus","labels":list(LABELS),"eligible_images":{k:len(v) for k,v in parts.items()},"split_audit":audit,"candidate_selection":"highest validation macro-average-precision; ties by validation macro-AUROC; internal validation used for candidate/epoch selection; calibration subset is patient-disjoint within official training patients","selected_model":chosen["architecture"],"selected_epoch":chosen["best_epoch"],"candidates":[{k:v for k,v in c.items() if k!="checkpoint"} for c in candidate_results],"calibration_metrics":score(cy,cp),"calibration_sensitivity_target":a.calibration_sensitivity,"calibration_thresholds":thresholds,"validation_metrics":score(vy,vp),"validation_operating_points":operating_metrics(vy,vp,thresholds),"validation_patient_cluster_bootstrap_95ci":patient_bootstrap(vy,vp,parts["validation"],thresholds),"test_accessed":False,"sealed_test_labels_parsed":False,"boundary":"Development-only: calibration patients were held out from official training patients, while the official validation partition selected candidates and epochs and is therefore selection-exposed. The official test manifest and labels were not read. Single-site/device; no clinical validation."}
 else:
  thresholds=select_thresholds(vy,vp);ty,tp=evaluate(model,loaders["test"])
  report={"status":"complete","dataset":"CLARUS700 Figshare 2026 public release","modality":"single-centre ZEISS CLARUS 700 approximately 133-degree widefield fundus","labels":list(LABELS),"eligible_images":{"train":len(parts["train"]),"validation":len(parts["validation"]),"test":len(parts["test"])},"split_audit":audit,"candidate_selection":"highest validation macro-average-precision; ties by validation macro-AUROC; official test not used for candidate selection, epoch selection or thresholds","candidates":[{k:v for k,v in c.items() if k!="checkpoint"} for c in candidate_results],"selected_model":chosen["architecture"],"selected_epoch":chosen["best_epoch"],"validation_metrics":score(vy,vp),"validation_thresholds":thresholds,"test_metrics":score(ty,tp),"test_operating_points":operating_metrics(ty,tp,thresholds),"test_patient_cluster_bootstrap_95ci":patient_bootstrap(ty,tp,parts["test"],thresholds),"test_patients":len({r["patient"] for r in parts["test"]}),"image_quality":"Test results exclude official model_eligible=0 rows; subgroup metrics are not used for selection.","boundary":"Official patient-disjoint test set; one site/device, small positive support especially RVO. Research development evidence, not clinical validation or universal retinal disease detection."}
 (a.out/"aggregate_report.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps({"final":{k:report.get(k) for k in ("selected_model","selected_epoch","validation_metrics","validation_operating_points","test_accessed")}}),flush=True)
if __name__=="__main__":main()
