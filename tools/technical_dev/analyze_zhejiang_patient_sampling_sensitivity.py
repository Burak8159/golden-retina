"""Check how unequal images-per-patient affects external ranking estimates."""
from __future__ import annotations

import argparse, csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def bootstrap(groups, positive, negative, rng, reps):
    p=list(groups[positive]); n=list(groups[negative]); aucs=[]; aps=[]
    for _ in range(reps):
        ps=rng.choice(p,len(p),replace=True); ns=rng.choice(n,len(n),replace=True)
        y=np.r_[np.ones(len(ps)),np.zeros(len(ns))]
        scores=np.r_[[groups[positive][x] for x in ps],[groups[negative][x] for x in ns]]
        aucs.append(roc_auc_score(y,scores)); aps.append(average_precision_score(y,scores))
    return {"auroc_ci95":np.quantile(aucs,[.025,.975]).tolist(),"auprc_ci95":np.quantile(aps,[.025,.975]).tolist()}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--predictions',required=True); ap.add_argument('--output',required=True); ap.add_argument('--replicates',type=int,default=2000); a=ap.parse_args()
    with open(a.predictions,newline='',encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
    patients=defaultdict(list)
    for r in rows: patients[r['patient_id']].append(r)
    groups=defaultdict(lambda: defaultdict(dict))
    for pid,rs in patients.items():
        labels={r['diagnosis'] for r in rs}
        if len(labels)!=1: raise ValueError(f'patient has conflicting source diagnoses: {pid}')
        d=labels.pop(); by_eye=defaultdict(list)
        for r in rs: by_eye[r['eye_id']].append(r)
        eye_scores={eye:float(np.mean([float(x['risk_score']) for x in xrs])) for eye,xrs in by_eye.items()}
        groups['patient_mean'][d][pid]=float(np.mean([float(r['risk_score']) for r in rs]))
        groups['patient_max'][d][pid]=float(max(float(r['risk_score']) for r in rs))
        first_eye=min(by_eye, key=lambda e:(min(x['image_id'] for x in by_eye[e]),e))
        groups['first_eye'][d][pid]=eye_scores[first_eye]
    rng=np.random.default_rng(20261108); cats=sorted(set(r['diagnosis'] for r in rows)); report={
      'status':'EXPLORATORY_PATIENT_SAMPLING_SENSITIVITY_ANALYSIS',
      'important_design_issue':'The cohort has different images-per-patient distributions by diagnosis; a max-over-images patient score can advantage groups with more images. Report patient-mean and deterministic single-eye results alongside max.',
      'single_eye_rule':'Select eye containing the lexicographically first image ID per patient; average repeated images within that eye if present.',
      'patient_mean_rule':'Arithmetic mean frozen score across every available image for a patient.',
      'patient_max_rule':'Maximum frozen score across every available image for a patient.',
      'conditions':{}}
    for d in cats:
        if d=='Healthy': continue
        report['conditions'][d]={'positive_patients':len(groups['patient_mean'][d]),'healthy_patients':len(groups['patient_mean']['Healthy']),'images_per_patient_positive':{'min':min(len(v) for v in patients.values() if v[0]['diagnosis']==d),'max':max(len(v) for v in patients.values() if v[0]['diagnosis']==d)},'methods':{}}
        for method in ('patient_mean','patient_max','first_eye'):
            positive=groups[method][d]; negative=groups[method]['Healthy']
            y=np.r_[np.ones(len(positive)),np.zeros(len(negative))]
            s=np.r_[list(positive.values()),list(negative.values())]
            report['conditions'][d]['methods'][method]={
              'auroc':float(roc_auc_score(y,s)),
              'auprc':float(average_precision_score(y,s)),
              'patient_bootstrap_95_percentile':bootstrap(groups[method],d,'Healthy',rng,a.replicates),
            }
    report['replicates']=a.replicates
    report['limitations']=['Single-center, enriched source-defined diagnosis categories; not prevalence representative.',
      'This risk head is referable-DR-trained and was not trained as a classifier for non-DR categories.',
      'The first-eye rule is deterministic sensitivity analysis, not a prespecified clinical laterality choice.',
      'No weights or thresholds changed after the frozen one-shot prediction export.']
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__': main()
