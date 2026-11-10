"""Versioned per-frame pupil observation interface for the compact model."""
from __future__ import annotations
import hashlib, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from train_segmenter import TinyUNet


class EyeFinder:
    def __init__(self, checkpoint: str, device: str = 'cpu'):
        self.device=torch.device(device if device=='cpu' or torch.cuda.is_available() else 'cpu')
        self.checkpoint=Path(checkpoint); ck=torch.load(self.checkpoint,map_location='cpu',weights_only=False)
        self.model=TinyUNet(); self.model.load_state_dict(ck['state_dict']); self.model.to(self.device).eval()
        self.model_version=hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()[:16]

    @torch.no_grad()
    def predict(self, image: Image.Image, eye_side='unknown', timestamp_ns=None):
        gray=image.convert('L').resize((192,128),Image.Resampling.BILINEAR)
        a=np.array(gray,dtype=np.uint8,copy=True)
        x=torch.from_numpy(a).float()[None,None].to(self.device)/255.
        prob=self.model(x).softmax(1)[0]
        mask=prob.argmax(0)
        pts=torch.nonzero(mask==3)
        if pts.numel()<8:
            center=None; confidence=None; state='uncertain'
        else:
            center=[float(pts[:,1].float().mean()*image.width/192),float(pts[:,0].float().mean()*image.height/128)]
            confidence=float(prob[3][mask==3].mean().item())
            state='detected' if confidence>=0.65 else 'uncertain'
        return {'schema_version':'0.1.0','timestamp_ns':int(timestamp_ns or time.time_ns()),'eye_side':eye_side,
          'pupil_center_xy':center,'pupil_mask_rle':None,'confidence':confidence,'track_state':state,
          'frame_size_wh':[image.width,image.height],'model_version':self.model_version,
          'coordinate_frame':'input_eye_crop_pixel_xy_origin_top_left'}
