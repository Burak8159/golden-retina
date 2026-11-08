#!/usr/bin/env python3
"""Paired patient/source bootstrap for two private validation prediction CSVs."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score,average_precision_score,roc_auc_score

def load(path:Path,label_col:str,score_col:str):
    groups={}
    with path.open(newline='',encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            y=int(r[label_col])
            if y<0: continue
            k=(r['source'],r['patient_id']); z=groups.setdefault(k,[[],[]]);z[0].append(y);z[1].append(float(r[score_col]))
    return {k:(int(max(y)),float(max(s))) for k,(y,s) in groups.items()}

def metrics(y,p):
    pred=p>=.5
    return {'accuracy':float(accuracy_score(y,pred)),'sensitivity':float(pred[y==1].mean()),'specificity':float((~pred[y==0]).mean()),'auroc':float(roc_auc_score(y,p)),'auprc':float(average_precision_score(y,p))}

def main():
    a=argparse.ArgumentParser();a.add_argument('--candidate',type=Path,required=True);a.add_argument('--candidate-label',default='referable_truth');a.add_argument('--candidate-score',default='referable_score');a.add_argument('--reference',type=Path,required=True);a.add_argument('--reference-label',default='referable_truth');a.add_argument('--reference-score',default='referable_score');a.add_argument('--candidate-name',default='candidate');a.add_argument('--reference-name',default='reference');a.add_argument('--output',type=Path,required=True);a.add_argument('--draws',type=int,default=5000);a.add_argument('--seed',type=int,default=20261110);x=a.parse_args()
    c=load(x.candidate,x.candidate_label,x.candidate_score);r=load(x.reference,x.reference_label,x.reference_score);keys=sorted(set(c)&set(r))
    if not keys: raise ValueError('No common source/patient groups')
    if any(c[k][0]!=r[k][0] for k in keys): raise ValueError('Patient-level label mismatch')
    y=np.array([c[k][0] for k in keys]);pc=np.array([c[k][1] for k in keys]);pr=np.array([r[k][1] for k in keys]);mc,mr=metrics(y,pc),metrics(y,pr);names=tuple(mc);rng=np.random.default_rng(x.seed);vals={n:[] for n in names}
    for _ in range(x.draws):
        ix=rng.integers(0,len(y),len(y));m1,m0=metrics(y[ix],pc[ix]),metrics(y[ix],pr[ix])
        for n in names: vals[n].append(m1[n]-m0[n])
    out={'status':'PAIRED_PATIENT_SOURCE_INTERNAL_VALIDATION_ONLY','n_common_patient_source_groups':len(keys),'positive_patients':int(y.sum()),'threshold':0.5,'candidate':{'name':x.candidate_name,'metrics':mc},'reference':{'name':x.reference_name,'metrics':mr},'delta_candidate_minus_reference':{n:mc[n]-mr[n] for n in names},'paired_bootstrap_95ci_delta':{n:[float(np.quantile(v,.025)),float(np.quantile(v,.975))] for n,v in vals.items()},'warning':'Repeated development validation; not independent external validation. Bootstrap unit is source-patient group.'}
    x.output.parent.mkdir(parents=True,exist_ok=True);x.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
