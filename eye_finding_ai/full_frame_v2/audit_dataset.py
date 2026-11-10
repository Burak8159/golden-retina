"""Integrity and leakage audit for the generated train/validation/test manifest."""
import argparse,csv,json
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

def main():
    p=argparse.ArgumentParser();p.add_argument("--manifest",required=True);p.add_argument("--output",default=None);a=p.parse_args()
    rows=list(csv.DictReader(open(a.manifest,newline="")));subjects={s:set() for s in ("train","validation","test")};seeds={s:set() for s in subjects};counts=Counter();fail=[];pixel_hist={s:Counter() for s in subjects}
    for i,r in enumerate(rows):
        split=r["split"];counts[split]+=1
        if split not in subjects:fail.append(f"row {i}: invalid split {split}");continue
        if r["subject_id"] in subjects[split]:fail.append(f"row {i}: duplicate subject within split")
        subjects[split].add(r["subject_id"]);seeds[split].add(r["seed"])
        ip,mp=Path(r["image"]),Path(r["mask"])
        if not ip.is_file() or not mp.is_file():fail.append(f"row {i}: missing image or mask");continue
        im=np.asarray(Image.open(ip).convert("L"));mask=np.asarray(Image.open(mp).convert("L"))
        if im.shape!=(224,320) or mask.shape!=im.shape:fail.append(f"row {i}: bad shape {im.shape}/{mask.shape}")
        classes=set(map(int,np.unique(mask)))
        if not classes.issubset(set(range(7))):fail.append(f"row {i}: unknown class values {sorted(classes)}")
        pupil_l=bool((mask==3).any());pupil_r=bool((mask==6).any())
        if pupil_l!=bool(int(r["left_eye_visible"])) or pupil_r!=bool(int(r["right_eye_visible"])):fail.append(f"row {i}: pupil visibility metadata mismatch")
        pixel_hist[split].update(map(int,np.unique(mask)))
    overlap={"train_validation":sorted(subjects["train"]&subjects["validation"]),"train_test":sorted(subjects["train"]&subjects["test"]),"validation_test":sorted(subjects["validation"]&subjects["test"])}
    if any(overlap.values()):fail.append("subject overlap across splits")
    # Save a human-readable overlay sheet as part of label QA.
    colors=np.array([[0,0,0],[35,180,255],[80,230,130],[255,80,80],[180,120,255],[255,205,65],[255,90,180]],dtype=np.uint8)
    preview=Image.new("RGB",(640,480),(8,12,18));pd=ImageDraw.Draw(preview)
    samples=[r for r in rows if r["split"]=="train"][:2]+[r for r in rows if r["split"]=="validation"][:1]+[r for r in rows if r["split"]=="test"][:1]
    for j,r in enumerate(samples):
        im=np.asarray(Image.open(r["image"]).convert("L"));mask=np.asarray(Image.open(r["mask"]).convert("L"));rgb=np.repeat(im[...,None],3,axis=2);active=mask>0
        rgb[active]=(rgb[active]*.38+colors[mask[active]]*.62).astype(np.uint8)
        tile=Image.fromarray(rgb).resize((320,224));x=(j%2)*320;y=(j//2)*240;preview.paste(tile,(x,y+16));pd.text((x+5,y+2),f"{r['split']} {r['subject_id']}",fill=(230,235,245))
    overlay_path=Path(a.manifest).with_name("label_overlay_preview.png");preview.save(overlay_path)
    report={"status":"FAIL" if fail else "PASS","rows":len(rows),"counts_by_split":dict(counts),"unique_subjects_by_split":{k:len(v) for k,v in subjects.items()},"cross_split_subject_overlap":overlap,"class_presence_by_split":{s:{str(k):int(v[k]>0) for k in range(7)} for s,v in pixel_hist.items()},"label_overlay_preview":str(overlay_path),"errors":fail}
    dest=Path(a.output) if a.output else Path(a.manifest).with_name("dataset_audit.json");dest.write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2))
    if fail:raise SystemExit(2)
if __name__=="__main__":main()
