"""Patient-cluster bootstrap for disease-score ranking within quality strata."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def load_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def grouped_auc(rows, positive, negative, quality_value=None):
    selected = [r for r in rows if r["diagnosis"] in {positive, negative}
                and (quality_value is None or int(r["quality_overall_quality"]) == quality_value)]
    groups = {label: defaultdict(list) for label in (positive, negative)}
    for r in selected:
        groups[r["diagnosis"]][r["patient_id"]].append(float(r["risk_score"]))
    by = {label: groups[label] for label in groups}
    y = np.asarray([1]*sum(map(len, by[positive].values())) + [0]*sum(map(len, by[negative].values())))
    scores = np.asarray([s for vals in by[positive].values() for s in vals] +
                        [s for vals in by[negative].values() for s in vals])
    if len(by[positive]) < 2 or len(by[negative]) < 2:
        return None
    auc = float(roc_auc_score(y, scores))
    return auc, by, len(selected)


def bootstrap_auc(by, positive, negative, rng, n_boot=2000):
    pos_ids, neg_ids = list(by[positive]), list(by[negative])
    vals=[]
    for _ in range(n_boot):
        pos_draw=rng.choice(pos_ids,len(pos_ids),replace=True)
        neg_draw=rng.choice(neg_ids,len(neg_ids),replace=True)
        ps=[v for pid in pos_draw for v in by[positive][pid]]
        ns=[v for pid in neg_draw for v in by[negative][pid]]
        vals.append(roc_auc_score([1]*len(ps)+[0]*len(ns),ps+ns))
    return np.quantile(vals,[.025,.975]).tolist()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--predictions',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--replicates',type=int,default=2000)
    args=ap.parse_args()
    rows=load_rows(args.predictions)
    categories=sorted({r['diagnosis'] for r in rows})
    healthy_patients=len({r['patient_id'] for r in rows if r['diagnosis']=='Healthy'})
    rng=np.random.default_rng(20261108)
    report={'status':'EXPLORATORY_QUALITY_STRATIFIED_SOURCE_TRANSFER_ANALYSIS',
      'positive_label':'dataset diagnosis category; comparator is dataset Healthy category',
      'quality_label':'Ground Truth Overall quality: 0=poor, 1=good',
      'score':'frozen referable-DR risk score',
      'unit':'patient-cluster bootstrap; images retained within sampled patient; AUROC computed image-level within stratum',
      'replicates':args.replicates,'healthy_patients_total':healthy_patients,'conditions':{}}
    for disease in categories:
        if disease=='Healthy': continue
        item={'strata':{}}
        for name,qv in [('all',None),('good_quality',1),('poor_quality',0)]:
            result=grouped_auc(rows,disease,'Healthy',qv)
            if result is None:
                item['strata'][name]={'status':'insufficient class support'}
                continue
            auc,by,nimg=result
            counts={d:{'patients':len(by[d]),'images':sum(map(len,by[d].values()))} for d in (disease,'Healthy')}
            # Do not report unstable poor-quality intervals when either class
            # has fewer than ten patient clusters.
            if min(len(by[disease]),len(by['Healthy']))<10:
                ci=None; status='point estimate only; fewer than 10 patients in a class'
            else:
                ci=bootstrap_auc(by,disease,'Healthy',rng,args.replicates); status='patient-cluster bootstrap'
            item['strata'][name]={'status':status,'image_rows':nimg,'counts':counts,'auroc':auc,'ci95':ci}
        all_auc=item['strata'].get('all',{}).get('auroc')
        good_auc=item['strata'].get('good_quality',{}).get('auroc')
        item['good_quality_minus_all_auroc']=None if all_auc is None or good_auc is None else good_auc-all_auc
        report['conditions'][disease]=item
    report['limitations']=[
      'Retrospective single-center case-control-like UWF source; not prevalence-representative.',
      'The risk score was trained/selected for referable DR, not for the other diagnosis categories.',
      'Quality stratification is an exploratory confounding check, not causal adjustment.',
      'Poor-quality stratum can be sparse, especially among healthy controls; intervals are omitted below ten patient clusters per class.',
      'No thresholds or weights were changed after the one-shot prediction export.'
    ]
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__': main()
