#!/usr/bin/env python3
"""Private patient-cluster validation comparison for RETFound-Green."""
from __future__ import annotations
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np
import torch,timm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score,roc_auc_score,average_precision_score
from train_retfound_green_adaptation import Rows,MultiTask

def patient_rows(rows, scores, label_key, task, threshold=.5):
    groups={}
    for r,s in zip(rows,scores):
        y=int(r[label_key])
        if y<0: continue
        # Dataset-local identifiers are not globally unique. Keep source in
        # the cluster key so e.g. patient "001" in two datasets stays distinct.
        key=(r['source'],r['patient_id'])
        g=groups.setdefault(key,{'source':r['source'],'patient_id':r['patient_id'],'labels':[],'scores':[]})
        g['labels'].append(y);g['scores'].append(float(s))
    return [{'patient_id':v['patient_id'],'source':v['source'],'label':int(max(v['labels'])),'score':float(max(v['scores']))} for v in groups.values()]

def metrics(records, threshold=.5, boots=2000, seed=20261108):
    if not records:return {'n_patients':0,'status':'no eligible patients'}
    y=np.asarray([r['label'] for r in records],dtype=int);p=np.asarray([r['score'] for r in records],dtype=float)
    def point(ix):
        yy=y[ix];pp=p[ix];pred=pp>=threshold
        return {'n_patients':int(len(ix)),'positive_patients':int(yy.sum()),'accuracy_at_0_5':float(accuracy_score(yy,pred)),'sensitivity_at_0_5':float(pred[yy==1].mean()) if (yy==1).any() else None,'specificity_at_0_5':float((~pred[yy==0]).mean()) if (yy==0).any() else None,'auroc':float(roc_auc_score(yy,pp)) if len(np.unique(yy))==2 else None,'auprc':float(average_precision_score(yy,pp)) if (yy==1).any() else None}
    out=point(np.arange(len(y)))
    rng=np.random.default_rng(seed);vals={k:[] for k in ('accuracy_at_0_5','sensitivity_at_0_5','specificity_at_0_5','auroc','auprc')}
    for _ in range(boots):
        ix=rng.integers(0,len(y),len(y))
        m=point(ix)
        for k in vals:
            if m[k] is not None and np.isfinite(m[k]):vals[k].append(m[k])
    out['patient_bootstrap_95ci']={k:[float(np.quantile(v,.025)),float(np.quantile(v,.975))] if v else None for k,v in vals.items()}
    return out

def pred_for_model(path, model, task):
    rows=[];scores=[]
    with open(path,newline='',encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            if model=='convnext':
                if r.get('split')!='valid':continue
                if task=='dr':
                    raw=int(r['dr_grade_true'])
                    if raw<0:continue
                    y=int(raw>=2);s=float(r['dr_probability_referable_ge_2'])
                else:y=int(r['gon_true']);s=float(r['gon_probability'])
            else:
                want='referable_dr_ge2' if task=='dr' else 'glaucoma_hygd'
                if r.get('task')!=want:continue
                y=int(r['truth']);s=float(r['score'])
            if y<0:continue
            rows.append({'patient_id':r['patient_id'],'source':r['source'],('dr_grade' if task=='dr' else 'gon'):str(y)})
            scores.append(s)
    return patient_rows(rows,scores,'dr_grade' if task=='dr' else 'gon',task)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--base-checkpoint',type=Path,required=True);ap.add_argument('--adapted-checkpoint',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--results-root',type=Path,required=True);ap.add_argument('--boots',type=int,default=2000);a=ap.parse_args()
    allrows=list(csv.DictReader(a.manifest.open(newline='',encoding='utf-8-sig')))
    if any(r['split']=='test' for r in allrows):raise RuntimeError('Test rows must not be read')
    va=[r for r in allrows if r['split']=='valid' and r['source']!='ddr_hf_specificity300' and r['patient_group_verified'].lower()=='true']
    private_pred=a.output_dir/'validation_predictions_private.csv'
    if private_pred.exists():
        # Reuse the already-generated private row predictions; avoid a second
        # inference pass while keeping the correction strictly label-side.
        dr_rows=[];dr_scores=[];gn_rows=[];gn_scores=[]
        with private_pred.open(newline='',encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                if int(r['referable_truth'])>=0:
                    dr_rows.append({'patient_id':r['patient_id'],'source':r['source'],'dr_grade':r['referable_truth']});dr_scores.append(float(r['referable_score']))
                if int(r['gon_truth'])>=0:
                    gn_rows.append({'patient_id':r['patient_id'],'source':r['source'],'gon':r['gon_truth']});gn_scores.append(float(r['gon_score']))
        rfg_dr=patient_rows(dr_rows,dr_scores,'dr_grade','dr');rfg_g=patient_rows(gn_rows,gn_scores,'gon','glaucoma')
    else:
        model=timm.create_model('vit_small_patch14_reg4_dinov2',img_size=392,num_classes=0)
        base=torch.load(a.base_checkpoint,map_location='cpu',weights_only=True);load=model.load_state_dict(base,strict=False)
        if load.missing_keys or load.unexpected_keys:raise RuntimeError(f'base-key mismatch: missing={load.missing_keys}, unexpected={load.unexpected_keys}')
        model.global_pool='avg'
        ck=torch.load(a.adapted_checkpoint,map_location='cpu',weights_only=True)
        has_quality=any(k.startswith('quality.') for k in ck['state_dict'])
        source_to_idx=ck.get('source_to_idx',{})
        net=MultiTask(model,model.num_features,quality_aux=has_quality,source_count=len(source_to_idx),source_reversal=float(ck.get('source_reversal',0.0)));net.load_state_dict(ck['state_dict'],strict=True)
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');net.eval().to(device)
        dl=DataLoader(Rows(va,False),batch_size=8,shuffle=False,num_workers=4,pin_memory=device.type=='cuda')
        rg=[];rr=[];gg=[]
        with torch.inference_mode():
            for x,g,gon,q,src,pid in dl:
                x=x.to(device,non_blocking=True)
                with torch.autocast('cuda',dtype=torch.float16,enabled=device.type=='cuda'): outputs=net(x);gl,rl,gn=outputs[:3]
                rg.extend(gl.argmax(1).cpu().numpy().tolist());rr.extend(torch.sigmoid(rl.float()).cpu().numpy().tolist());gg.extend(torch.sigmoid(gn.float()).cpu().numpy().tolist())
        # Persist row-level predictions only on the private VM.
        a.output_dir.mkdir(parents=True,exist_ok=True)
        with private_pred.open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['source','patient_id','dr_grade_true','dr_grade_pred','referable_truth','referable_score','gon_truth','gon_score'])
            for r,grade,ref,gon in zip(va,rg,rr,gg):w.writerow([r['source'],r['patient_id'],r['dr_grade'],grade,int(int(r['dr_grade'])>=2) if int(r['dr_grade'])>=0 else -1,ref,r['gon'],gon])
        binary_dr_rows=[dict(r,dr_grade=str(int(int(r['dr_grade'])>=2) if int(r['dr_grade'])>=0 else -1)) for r in va]
        rfg_dr=patient_rows(binary_dr_rows,rr,'dr_grade','dr');rfg_g=patient_rows(va,gg,'gon','glaucoma')
    out={'status':'PATIENT_LEVEL_INTERNAL_VALIDATION_ONLY','manifest_sha256':hashlib.sha256(a.manifest.read_bytes()).hexdigest(),'base_checkpoint_sha256':hashlib.sha256(a.base_checkpoint.read_bytes()).hexdigest(),'adapted_checkpoint_sha256':hashlib.sha256(a.adapted_checkpoint.read_bytes()).hexdigest(),'validation_images':len(va),'test_rows_read':0,'patient_rule':'patient positive if any labeled eye/image is positive; patient score is maximum per-image score; bootstrap resamples patient clusters.','models':{},'by_source':{}}
    all_models={'RETFound-Green':{'dr':rfg_dr,'glaucoma':rfg_g}}
    root=a.results_root
    for name,rel in [('ConvNeXt-Tiny','convnext_baseline/private_validation_predictions.csv'),('FLAIR','flair/private_validation_predictions.csv'),('UrFound','urf/private_validation_predictions.csv')]:
        path=root/rel
        all_models[name]={'dr':pred_for_model(path,'convnext' if name=='ConvNeXt-Tiny' else 'features','dr'),'glaucoma':pred_for_model(path,'convnext' if name=='ConvNeXt-Tiny' else 'features','glaucoma')}
    for name,bytask in all_models.items():out['models'][name]={task:metrics(records,boots=a.boots,seed=20261108+(i*19)) for i,(task,records) in enumerate(bytask.items())}
    for source in sorted(set(r['source'] for r in va)):
        out['by_source'][source]={}
        for name,bytask in all_models.items():
            out['by_source'][source][name]={task:metrics([r for r in records if r['source']==source],boots=a.boots,seed=20261108+(i*19)) for i,(task,records) in enumerate(bytask.items())}
    (a.output_dir/'patient_level_summary.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'status':out['status'],'models':out['models'],'by_source':out['by_source']},indent=2),flush=True)
if __name__=='__main__':main()
