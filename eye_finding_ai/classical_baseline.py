"""Tune a simple dark-pixel centroid on validation seeds, then evaluate test seeds."""
import argparse, json
from pathlib import Path
import numpy as np
from synthetic_eye import render_eye


def score(seed0, n, threshold):
    errors=[]; misses=0
    for i in range(n):
        gray,mask,truth=render_eye(seed0+i)
        roi=gray[25:106,28:165]
        yy,xx=np.where(roi<threshold)
        if len(xx)<3:
            misses+=1; continue
        pred=(float(xx.mean()+28),float(yy.mean()+25))
        errors.append(float(np.hypot(pred[0]-truth[0],pred[1]-truth[1])))
    return {"threshold":threshold,"detections":len(errors),"misses":misses,"mean_center_error_px":float(np.mean(errors)) if errors else None,
            "p95_center_error_px":float(np.percentile(errors,95)) if errors else None}


def main():
    p=argparse.ArgumentParser();p.add_argument('--valid-count',type=int,default=400);p.add_argument('--test-count',type=int,default=600);p.add_argument('--seed',type=int,default=8159);p.add_argument('--output',default='artifacts/classical_baseline.json');a=p.parse_args()
    val_seed=a.seed+1_000_000;test_seed=a.seed+2_000_000
    candidates=[score(val_seed,a.valid_count,t) for t in (12,18,24,30,36)]
    chosen=min(candidates,key=lambda x:(x['misses'],x['mean_center_error_px'] if x['mean_center_error_px'] is not None else 1e9))['threshold']
    report={"status":"SYNTHETIC_HELD_OUT_ONLY","method":"dark-pixel centroid, threshold selected on validation seeds","selected_threshold":chosen,
            "validation_candidates":candidates,"test":score(test_seed,a.test_count,chosen),"test_seed_start":test_seed,"validation_seed_start":val_seed}
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
