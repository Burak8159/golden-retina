"""Synthetic temporal tracking evaluation for the trained pupil segmenter."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import torch

from synthetic_eye import render_eye
from train_segmenter import TinyUNet


def center_from_mask(mask):
    pts=torch.nonzero(mask==3)
    return None if pts.numel()<8 else (float(pts[:,1].float().mean()),float(pts[:,0].float().mean()))


class AlphaBeta:
    def __init__(self, alpha=.78, beta=.22):
        self.alpha,self.beta=alpha,beta; self.x=self.y=self.vx=self.vy=None; self.lost=0
    def step(self, measurement, dt=1.0):
        if self.x is None and measurement is not None:
            self.x,self.y=measurement; self.vx=self.vy=0.; self.lost=0
            return (self.x,self.y,"detected")
        if self.x is None:
            self.lost+=1; return (None,None,"lost")
        px,py=self.x+self.vx*dt,self.y+self.vy*dt
        if measurement is None:
            self.x,self.y=px,py; self.lost+=1
            if self.lost>8: self.x=self.y=self.vx=self.vy=None; return (None,None,"lost")
            return (px,py,"predicted")
        rx,ry=measurement[0]-px,measurement[1]-py
        self.x,self.y=px+self.alpha*rx,py+self.alpha*ry
        self.vx+=self.beta*rx/dt; self.vy+=self.beta*ry/dt; self.lost=0
        return (self.x,self.y,"detected")


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True); ap.add_argument('--output',default='artifacts/track_summary.json'); ap.add_argument('--sequences',type=int,default=50); ap.add_argument('--frames',type=int,default=50); args=ap.parse_args()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    model=TinyUNet().to(device); model.load_state_dict(ck['state_dict']); model.eval()
    errors=[]; valid_frames=0; kept_frames=0; blink_predicted=0; blink_frames=0; recoveries=[]; sequence_reacquire=[]
    with torch.no_grad():
        for s in range(args.sequences):
            tracker=AlphaBeta(); was_blink=False; reacquire_count=0
            for t in range(args.frames):
                blink=(t%19 in (8,9,10))
                gaze=.3*math.sin(2*math.pi*t/args.frames)+.12*math.sin(6*math.pi*t/args.frames)
                image,mask,truth=render_eye(900000+s*args.frames+t,sequence_offset=gaze,blink=blink)
                x=torch.from_numpy(image.copy()).float()[None,None].to(device)/255.
                pred=model(x).argmax(1)[0]
                measurement=None if blink else center_from_mask(pred)
                ox,oy,state=tracker.step(measurement)
                if blink:
                    blink_frames+=1
                    if state=='predicted': blink_predicted+=1
                else:
                    valid_frames+=1
                    if truth[0]==truth[0] and ox is not None:
                        kept_frames+=1; errors.append(math.hypot(ox-truth[0],oy-truth[1]))
                    if was_blink:
                        if state=='detected': recoveries.append(reacquire_count)
                        reacquire_count=0
                if was_blink and not blink and state!='detected': reacquire_count+=1
                was_blink=blink
            sequence_reacquire.append(reacquire_count)
    report={"status":"SYNTHETIC_TEMPORAL_EVALUATION_ONLY","device":str(device),"sequences":args.sequences,"frames_per_sequence":args.frames,
      "visible_frames":valid_frames,"tracked_visible_frames":kept_frames,"visible_track_coverage":kept_frames/max(1,valid_frames),
      "tracked_center_error_px_mean":float(np.mean(errors)) if errors else None,"tracked_center_error_px_p95":float(np.percentile(errors,95)) if errors else None,
      "blink_frames":blink_frames,"blink_frames_with_prediction":blink_predicted,"blink_prediction_fraction":blink_predicted/max(1,blink_frames),
      "post_blink_reacquire_frames_median":float(np.median(recoveries)) if recoveries else None,"checkpoint":args.checkpoint}
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
