"""One-shot held-out synthetic test evaluation for a frozen checkpoint."""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from model import FullFrameUNet
from train import SegDataset,metrics,NAMES

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument("--manifest",required=True);p.add_argument("--checkpoint",required=True);p.add_argument("--output",default="artifacts/full_frame_v2/test_metrics.json");p.add_argument("--batch-size",type=int,default=16);a=p.parse_args()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu");rows=list(csv.DictReader(open(a.manifest)));test_rows=[r for r in rows if r["split"]=="test"]
    loader=DataLoader(SegDataset(a.manifest,"test"),batch_size=a.batch_size,shuffle=False,num_workers=2,pin_memory=device.type=="cuda")
    ckpt=torch.load(a.checkpoint,map_location="cpu",weights_only=False);model=FullFrameUNet().to(device);model.load_state_dict(ckpt["model"]);model.eval()
    core=metrics(model,loader,device);records=[];err={"left":[],"right":[]};det={"left":{"found":0,"visible":0,"false_when_closed":0,"closed":0},"right":{"found":0,"visible":0,"false_when_closed":0,"closed":0}}
    with torch.inference_mode():
        for row in test_rows:
            x=torch.from_numpy(np.asarray(Image.open(row["image"]).convert("L"),dtype=np.float32).copy()/255.).unsqueeze(0).unsqueeze(0).to(device)
            y=np.asarray(Image.open(row["mask"]).convert("L"),dtype=np.uint8);z=model(x);prob=z.softmax(1)[0];pred=z.argmax(1)[0].cpu().numpy()
            for side,cls in (("left",3),("right",6)):
                gt=np.argwhere(y==cls);pr=np.argwhere(pred==cls);key=side
                if len(gt):
                    det[key]["visible"]+=1
                    if len(pr):
                        det[key]["found"]+=1;err[key].append(float(np.linalg.norm(gt.mean(0)-pr.mean(0))))
                else:
                    det[key]["closed"]+=1
                    if len(pr):det[key]["false_when_closed"]+=1
            cond=json.loads(row["conditions"]);blur="sharp" if cond["blur_radius"]<.2 else "mild" if cond["blur_radius"]<1 else "strong"
            records.append({"subject_id":row["subject_id"],"camera_style":cond["camera_style"],"blur_group":blur,"exposure":cond["exposure"],"left_visible":bool((y==3).any()),"right_visible":bool((y==6).any()),"frame_left_pupil_confidence":float(prob[3][pred==3].mean()) if (pred==3).any() else 0.,"frame_right_pupil_confidence":float(prob[6][pred==6].mean()) if (pred==6).any() else 0.})
    summary={"status":"LOCKED_SYNTHETIC_TEST_ONLY","model":"FullFrameUNet base=16 classes=7","checkpoint_sha256":sha(a.checkpoint),"test_manifest_sha256":sha(a.manifest),"device":str(device),"n_test_frames":len(test_rows),"selection_used_validation_only":True,"pixel_metrics":core,"per_eye_pupil_detection":{k:{"visible_frames":v["visible"],"detection_rate":v["found"]/max(1,v["visible"]),"closed_frames":v["closed"],"false_detection_on_closed_frames":v["false_when_closed"],"center_error_mean_px":float(np.mean(err[k])) if err[k] else None,"center_error_p95_px":float(np.percentile(err[k],95)) if err[k] else None} for k,v in det.items()},"subgroup_counts":{"camera_style":{x:sum(r["camera_style"]==x for r in records) for x in sorted({r["camera_style"] for r in records})},"blur_group":{x:sum(r["blur_group"]==x for r in records) for x in ["sharp","mild","strong"]}},"limitations":["All images are renderer-generated; no physical camera or real-human domain validation.","Pixel and pupil metrics do not establish clinical or device performance."]}
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(summary,indent=2)+"\n");(out.with_name("test_frame_summary.json")).write_text(json.dumps(records,indent=2)+"\n");print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
