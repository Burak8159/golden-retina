"""Single-frame inference contract. Input is one full-face grayscale/RGB image."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from model import FullFrameUNet

def predict(image_path,checkpoint,output_dir,device=None):
    device=torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"));ckpt=torch.load(checkpoint,map_location="cpu",weights_only=False)
    model=FullFrameUNet().to(device);model.load_state_dict(ckpt["model"]);model.eval()
    image=Image.open(image_path).convert("L").resize((320,224),Image.Resampling.BILINEAR);arr=np.asarray(image,dtype=np.float32)/255.
    x=torch.from_numpy(arr.copy()).unsqueeze(0).unsqueeze(0).to(device)
    with torch.inference_mode():prob=model(x).softmax(1)[0];pred=prob.argmax(0).cpu().numpy()
    result={"model":"golden-retina-full-frame-v2","source_image":str(image_path),"status":"synthetic-domain-inference-only","coordinate_frame":"resized_input_pixels_320x224","eyes":{}}
    for side,classes,pupil in (("frame_left",(1,2,3),3),("frame_right",(4,5,6),6)):
        ys,xs=np.where(np.isin(pred,classes));py,px=np.where(pred==pupil);visible=len(px)>0
        conf=float(prob[pupil][pred==pupil].mean().item()) if visible else 0.
        result["eyes"][side]={"eye_found":bool(len(xs)>0),"pupil_visible":visible,"pupil_center_xy":[float(px.mean()),float(py.mean())] if visible else None,"pupil_mean_confidence":conf,"state":"tracking" if visible and conf>=0.35 else "uncertain_or_occluded"}
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True);stem=Path(image_path).stem;Image.fromarray(pred.astype(np.uint8),"L").save(out/f"{stem}_mask.png");(out/f"{stem}_observation.json").write_text(json.dumps(result,indent=2)+"\n");return result

def main():
    p=argparse.ArgumentParser();p.add_argument("--image",required=True);p.add_argument("--checkpoint",required=True);p.add_argument("--output-dir",default="artifacts/full_frame_v2/inference");p.add_argument("--device",default=None);a=p.parse_args();print(json.dumps(predict(a.image,a.checkpoint,a.output_dir,a.device),indent=2))
if __name__=="__main__":main()
