"""Run a frozen shared-eye model and alpha-beta tracker on a synthetic sequence."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from model import FaceEyeUNet
from prepare_charmorph_eye_dataset import components
from sequence_filter import AlphaBeta
from train_charmorph_eye import SHARED_EYE_NAMES


def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument("--sequence",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--size",type=int,default=384);p.add_argument("--minimum-pupil-pixels",type=int,default=5)
    a=p.parse_args();device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck=torch.load(a.checkpoint,map_location="cpu",weights_only=False)
    if ck.get("classes")!=SHARED_EYE_NAMES:raise ValueError("checkpoint must use shared-eye labels")
    model=FaceEyeUNet(base=int(ck.get("base",16)),classes=3).to(device);model.load_state_dict(ck["model"]);model.eval()
    tracker={"left":AlphaBeta(),"right":AlphaBeta()}; errors={s:[] for s in tracker};filtered={s:[] for s in tracker}
    detected={s:0 for s in tracker};trace=[];truth_series={s:[] for s in tracker}
    with torch.inference_mode():
        for i in range(int(json.loads((a.sequence/"sequence_report.json").read_text())["frames"])):
            folder=a.sequence/f"frame_{i:04d}";im=Image.open(folder/"beauty.png").convert("RGB")
            raw=np.load(folder/"labels.npy"); iris,_=components(raw==1);pupil,_=components(raw==3)
            if len(iris)!=2 or len(pupil)!=2:raise RuntimeError(f"invalid frame labels: {folder}")
            gt=np.zeros(raw.shape,dtype=np.uint8)
            for c in iris:gt[c["pixels"][:,0],c["pixels"][:,1]]=1
            for c in pupil:gt[c["pixels"][:,0],c["pixels"][:,1]]=2
            im_size=im.size
            im_small=im.resize((a.size,a.size),Image.Resampling.BILINEAR)
            gt_small=np.asarray(Image.fromarray(gt).resize((a.size,a.size),Image.Resampling.NEAREST))
            x=np.asarray(im_small,dtype=np.float32).copy()/255
            probs=model(torch.from_numpy(x).permute(2,0,1)[None].to(device)).softmax(1)[0].cpu().numpy()
            pred=probs.argmax(0);row={"frame":i,"eyes":{}}
            for side,xs in (("left",slice(0,a.size//2)),("right",slice(a.size//2,a.size))):
                g=np.where(gt_small[:,xs]==2); truth=None if not len(g[0]) else [float(g[1].mean()+(0 if side=="left" else a.size//2)),float(g[0].mean())]
                pts=np.where(pred[:,xs]==2);measurement=None
                if len(pts[0])>=a.minimum_pupil_pixels:
                    cx=float(pts[1].mean()+(0 if side=="left" else a.size//2));cy=float(pts[0].mean())
                    offset=0 if side=="left" else a.size//2
                    conf=float(probs[2][pts[0],pts[1]+offset].mean());measurement=[cx,cy];detected[side]+=1
                    if truth is not None:errors[side].append(math.dist(measurement,truth))
                tracked=tracker[side].update(measurement)
                if tracked["xy"] is not None and truth is not None:filtered[side].append(math.dist(tracked["xy"],truth))
                if truth is not None:truth_series[side].append(truth)
                row["eyes"][side]={"truth_model_px":truth,"measurement_model_px":measurement,
                                    "measurement_confidence":conf if measurement else None,
                                    "track_state":tracked["state"],"track_model_px":tracked["xy"]}
            trace.append(row)
    scale=im_size[0]/a.size
    summary={"status":"SYNTHETIC_TRANSLATION_SEQUENCE_EVALUATION_ONLY","device":str(device),
       "frames":len(trace),"fresh_sequence_seed":json.loads((a.sequence/"sequence_report.json").read_text()).get("seed"),
       "checkpoint_sha256":digest(a.checkpoint),"frames_manifest_sha256":digest(a.sequence/"sequence_report.json"),
       "pupil_detection_coverage":{s:detected[s]/max(1,len(trace)) for s in detected},
       "measurement_center_error_mean_model_px":{s:float(np.mean(errors[s])) if errors[s] else None for s in errors},
       "measurement_center_error_p95_model_px":{s:float(np.percentile(errors[s],95)) if errors[s] else None for s in errors},
       "measurement_center_error_mean_render_px":{s:float(np.mean(errors[s])*scale) if errors[s] else None for s in errors},
       "alpha_beta_track_error_mean_model_px":{s:float(np.mean(filtered[s])) if filtered[s] else None for s in filtered},
       "synthetic_movement":"labelled iris/pupil geometry translation; not physical gaze rotation",
       "limits":["single synthetic identity", "fixed camera and face", "no blink, head movement, hardware, or real humans",
                 "not a camera performance estimate"],"trace":trace}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps({k:v for k,v in summary.items() if k!="trace"},indent=2),flush=True)


if __name__=="__main__":main()
