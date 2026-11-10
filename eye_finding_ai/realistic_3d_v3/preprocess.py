"""Aspect-ratio-preserving image preprocessing and coordinate remapping."""
from __future__ import annotations

import numpy as np
from PIL import Image


def letterbox_rgb(image: Image.Image, size: int):
    image=image.convert("RGB");w,h=image.size;scale=min(size/w,size/h)
    rw=max(1,round(w*scale));rh=max(1,round(h*scale))
    resized=image.resize((rw,rh),Image.Resampling.BILINEAR)
    ox=(size-rw)//2;oy=(size-rh)//2
    canvas=Image.new("RGB",(size,size),(0,0,0));canvas.paste(resized,(ox,oy))
    return np.asarray(canvas,dtype=np.float32).copy()/255.,{"scale":scale,"offset_x":ox,"offset_y":oy,"width":rw,"height":rh}


def remap_xy(x: float, y: float, transform: dict):
    scale=transform["scale"]
    return [(x-transform["offset_x"])/scale,(y-transform["offset_y"])/scale]


def unletterbox_mask(mask: np.ndarray, original_size: tuple[int,int], transform: dict):
    ox,oy=transform["offset_x"],transform["offset_y"]
    crop=Image.fromarray(mask.astype(np.uint8)[oy:oy+transform["height"],ox:ox+transform["width"]],mode="L")
    return crop.resize(original_size,Image.Resampling.NEAREST)
