"""Synthetic full-face grayscale frames with image-left/right eye and pupil labels."""
import math,random
import numpy as np
from PIL import Image,ImageDraw,ImageFilter
from synthetic_eye import render_eye

H,W=224,320

def render_face(seed:int):
    r=random.Random(seed); nr=np.random.default_rng(seed^0xBB67AE85)
    bg=tuple(r.randint(18,45) for _ in range(3)); skin=tuple(r.randint(a,b) for a,b in ((136,205),(83,143),(62,116)))
    im=Image.new('RGB',(W,H),bg); d=ImageDraw.Draw(im)
    face=(r.randint(22,38),r.randint(8,20),r.randint(W-38,W-22),r.randint(H-15,H+15))
    d.ellipse(face,fill=skin)
    # Low-detail face landmarks vary independently of the eye labels.
    fx0,fy0,fx1,fy1=face; mid=(fx0+fx1)/2; ey=r.randint(72,91)
    d.line((mid,ey+12,mid-r.randint(-4,4),ey+54),fill=tuple(max(0,c-18) for c in skin),width=r.randint(2,4))
    d.ellipse((mid-9,ey+52,mid+10,ey+59),fill=tuple(max(0,c-22) for c in skin))
    labels=Image.new('L',(W,H),0); ld=ImageDraw.Draw(labels)
    crop_w,crop_h=112,74
    base_y=ey+r.randint(-4,6)
    positions=[(r.randint(38,51),base_y+r.randint(-4,4)),(r.randint(169,182),base_y+r.randint(-4,4))]
    for side,(x,y) in enumerate(positions):
        offset=r.uniform(-0.33,0.33)
        blink=r.random()<0.035
        gray,mask,_=render_eye(seed*13+side,sequence_offset=offset,blink=blink)
        eye=Image.fromarray(gray,'L').resize((crop_w,crop_h),Image.Resampling.BILINEAR)
        m=Image.fromarray(mask,'L').resize((crop_w,crop_h),Image.Resampling.NEAREST)
        angle=r.uniform(-6,6)
        eye=eye.rotate(angle,resample=Image.Resampling.BILINEAR,fillcolor=int(np.mean(skin)))
        m=m.rotate(angle,resample=Image.Resampling.NEAREST,fillcolor=0)
        im.paste(eye,(x,y))
        a=np.asarray(m,dtype=np.uint8)
        mapped=np.zeros_like(a)
        if side==0: mapped[a==1]=1; mapped[a==2]=2; mapped[a==3]=3
        else: mapped[a==1]=4; mapped[a==2]=5; mapped[a==3]=6
        labels.paste(Image.fromarray(mapped),(x,y))
        # Eyebrow and eyelid folds are nuisance signals, not labels.
        d.line((x+5,y-5,x+crop_w-7,y-7+r.uniform(-3,3)),fill=tuple(max(0,c-38) for c in skin),width=r.randint(2,4))
    arr=np.asarray(im,dtype=np.float32)
    arr=np.clip(arr*nr.uniform(.82,1.16)+nr.normal(0,r.uniform(1,7),arr.shape),0,255).astype(np.uint8)
    im=Image.fromarray(arr,'RGB')
    if r.random()<.3: im=im.filter(ImageFilter.GaussianBlur(r.uniform(.2,.9)))
    gray=np.asarray(im.convert('L'),dtype=np.uint8)
    return gray,np.asarray(labels,dtype=np.uint8)

def centers_and_boxes(mask):
    out={}
    for name,classes in [('frame_left',[1,2,3]),('frame_right',[4,5,6])]:
        eye=np.isin(mask,classes); ys,xs=np.where(eye)
        pupil=np.where(mask==classes[-1])
        out[name]={'bbox_xyxy':[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)] if len(xs) else None,
                   'pupil_center_xy':[float(pupil[1].mean()),float(pupil[0].mean())] if len(pupil[0]) else None}
    return out
