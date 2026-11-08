#!/usr/bin/env python3
"""Paired patient-cluster bootstrap comparing validation-only retinal scores."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

def aggregate(rows):
    d={}
    for source,pid,y,s in rows:
        k=(source,pid);a=d.setdefault(k,{'source':source,'labels':[],'scores':[]});a['labels'].append(int(y));a['scores'].append(float(s))
    return {k:(max(v['labels']),max(v['scores'])) for k,v in d.items()}

def read_rfg(root):
    p=root/'retfound_green_adaptation/validation_predictions_private.csv'; rows=[]
    with p.open(newline='',encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            if int(r['referable_truth'])>=0:rows.append((r['source'],r['patient_id'],int(r['referable_truth']),float(r['referable_score'])))
    return aggregate(rows)

def read_convnext(root):
    p=root/'convnext_baseline/private_validation_predictions.csv';rows=[]
    with p.open(newline='',encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            if r.get('split')!='valid':continue
            y=int(r['dr_grade_true'])
            if y<0:continue
            rows.append((r['source'],r['patient_id'],int(y>=2),float(r['dr_probability_referable_ge_2'])))
    return aggregate(rows)

def read_task(root,model):
    p=root/model/'private_validation_predictions.csv';rows=[]
    with p.open(newline='',encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            if r.get('task')!='referable_dr_ge2':continue
            rows.append((r['source'],r['patient_id'],int(r['truth']),float(r['score'])))
    return aggregate(rows)

def scores(y,p):
    q=p>=.5
    return np.array([np.mean(q==y),np.mean(q[y==1]) if np.any(y==1) else np.nan,np.mean(~q[y==0]) if np.any(y==0) else np.nan,roc_auc_score(y,p) if len(np.unique(y))==2 else np.nan])

def compare(a,b,keys,boots,seed):
    y=np.array([a[k][0] for k in keys],int);pa=np.array([a[k][1] for k in keys],float);pb=np.array([b[k][1] for k in keys],float)
    delta=scores(y,pa)-scores(y,pb);rng=np.random.default_rng(seed);samples=np.empty((boots,4))
    for i in range(boots):
        ix=rng.integers(0,len(y),len(y)); samples[i]=scores(y[ix],pa[ix])-scores(y[ix],pb[ix])
    return {'n_patients':len(keys),'delta_RFG_minus_comparator':{'accuracy':float(delta[0]),'sensitivity':float(delta[1]),'specificity':float(delta[2]),'auroc':float(delta[3])},'paired_patient_bootstrap_95ci':{name:[float(np.nanquantile(samples[:,j],.025)),float(np.nanquantile(samples[:,j],.975))] for j,name in enumerate(('accuracy','sensitivity','specificity','auroc'))}}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results-root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--boots',type=int,default=2000);a=ap.parse_args()
    root=a.results_root;models={'RETFound-Green':read_rfg(root),'ConvNeXt-Tiny':read_convnext(root),'FLAIR':read_task(root,'flair'),'UrFound':read_task(root,'urf')}
    base=models['RETFound-Green'];out={'status':'PAIRED_INTERNAL_VALIDATION_PATIENT_BOOTSTRAP','metric_order':['accuracy','sensitivity','specificity','auroc'],'common_patients':{},'comparisons':{},'interpretation':'All intervals are paired patient-bootstrap intervals on the exposed internal validation cohort; they do not establish independent clinical superiority.'}
    for m,d in models.items():
        common=sorted(set(base)&set(d));bad=[k for k in common if base[k][0]!=d[k][0]]
        if bad:raise ValueError(f'label mismatch for {m}: {len(bad)} common patients')
        out['common_patients'][m]=len(common)
    common=sorted(set.intersection(*(set(v) for v in models.values())))
    if any(len({models[m][k][0] for m in models})!=1 for k in common):raise ValueError('Cross-model label mismatch')
    for comp in ('ConvNeXt-Tiny','FLAIR','UrFound'):
        out['comparisons'][comp]=compare(base,models[comp],common,a.boots,20261108+len(comp))
        out['comparisons'][comp]['by_source']={}
        for source in sorted({k[0] for k in common}):
            ks=[k for k in common if k[0]==source]
            out['comparisons'][comp]['by_source'][source]=compare(base,models[comp],ks,a.boots,20261200+len(comp)+len(source))
    a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
