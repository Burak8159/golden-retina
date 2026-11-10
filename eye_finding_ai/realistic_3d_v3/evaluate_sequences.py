"""Evaluate detection and temporal filtering on rendered identity-consistent sequences."""
from __future__ import annotations

import argparse, csv, json, math
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from model import FaceEyeUNet
from sequence_filter import AlphaBeta


def main():
    p=argparse.ArgumentParser();p.add_argument("--manifest",required=True);p.add_argument("--checkpoint",required=True);p.add_argument("--output",required=True)
    p.add_argument("--size",type=int,default=None);p.add_argument("--minimum-confidence",type=float,default=.35);a=p.parse_args()
    rows=list(csv.DictReader(open(a.manifest,newline="")));device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck=torch.load(a.checkpoint,map_location=device,weights_only=False);size=a.size or int(ck.get("size",384));model=FaceEyeUNet(base=ck.get("base",16)).to(device);model.load_state_dict(ck["model"]);model.eval()
    groups={}
    for row in rows:groups.setdefault(row["sequence_id"],[]).append(row)
    errors={s:[] for s in ("left","right")};visible={s:0 for s in errors};detected={s:0 for s in errors};track_visible={s:0 for s in errors};blink_count={s:0 for s in errors};blink_pupil={s:0 for s in errors};reacquire=[]
    with torch.inference_mode():
        for sid,frames in groups.items():
            filters={s:AlphaBeta() for s in errors};was_blink={s:False for s in errors};wait={s:0 for s in errors}
            for row in sorted(frames,key=lambda r:int(r["frame"])):
                im=Image.open(row["image"]).convert("RGB");w,h=im.size
                x=np.asarray(im.resize((size,size),Image.Resampling.BILINEAR),dtype=np.float32).copy()/255
                probs=model(torch.from_numpy(x).permute(2,0,1)[None].to(device)).softmax(1)[0].cpu().numpy();pred=probs.argmax(0)
                for side,cls in (("left",4),("right",8)):
                    blink=bool(int(row[side+"_blink"]));pts=np.where(pred==cls);xy=None
                    if len(pts[0])>=5:
                        conf=float(probs[cls,pts[0],pts[1]].mean())
                        if conf>=a.minimum_confidence:xy=[float(pts[1].mean()*w/size),float(pts[0].mean()*h/size)]
                    state=filters[side].update(xy);truth=json.loads(row[side+"_pupil_xy"]) if row[side+"_pupil_xy"] else None
                    if blink:
                        blink_count[side]+=1
                        if len(pts[0])>=5:blink_pupil[side]+=1
                    else:
                        visible[side]+=1
                        if xy is not None:
                            detected[side]+=1
                            if truth is not None:errors[side].append(math.dist(xy,truth))
                        if state["xy"] is not None:track_visible[side]+=1
                        if was_blink[side] and state["state"]=="detected":reacquire.append(wait[side]);wait[side]=0
                        elif was_blink[side] and state["state"]!="detected":wait[side]+=1
                    was_blink[side]=blink
    summary={"status":"SYNTHETIC_SEQUENCE_EVALUATION_ONLY","device":str(device),"sequence_count":len(groups),"frame_count":len(rows),
      "visible_frames":visible,"detected_visible_frames":detected,"detection_coverage_visible":{s:detected[s]/max(1,visible[s]) for s in visible},
      "temporally_tracked_visible_frames":track_visible,"temporal_coverage_visible":{s:track_visible[s]/max(1,visible[s]) for s in visible},
      "pupil_center_error_px_mean":{s:float(np.mean(errors[s])) if errors[s] else None for s in errors},
      "pupil_center_error_px_p95":{s:float(np.percentile(errors[s],95)) if errors[s] else None for s in errors},
      "blink_frames":blink_count,"blink_false_pupil_rate":{s:blink_pupil[s]/max(1,blink_count[s]) for s in blink_count},
      "post_blink_reacquire_frames_median":float(np.median(reacquire)) if reacquire else None,"physical_camera_validated":False,
      "checkpoint":str(a.checkpoint)}
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
