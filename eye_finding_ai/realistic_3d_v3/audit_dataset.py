"""Audit generated paths, split independence, mask semantics and landmark alignment."""
from __future__ import annotations

import argparse, csv, hashlib, json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

CLASSES = {"background":0,"face":1,"left_sclera":2,"left_iris":3,"left_pupil":4,"left_glint":5,
           "right_sclera":6,"right_iris":7,"right_pupil":8,"right_glint":9,"left_lid":10,"right_lid":11}


def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument("--manifest",required=True);p.add_argument("--output",required=True);p.add_argument("--max-landmark-error-px",type=float,default=5.)
    a=p.parse_args();rows=list(csv.DictReader(open(a.manifest,newline="")))
    if not rows:raise SystemExit("empty manifest")
    by={s:[r for r in rows if r["split"]==s] for s in ("train","validation","test")}
    subjects={s:{r["subject_id"] for r in rr} for s,rr in by.items()}
    overlap={f"{x}_{y}":len(subjects[x]&subjects[y]) for x,y in (("train","validation"),("train","test"),("validation","test"))}
    image_hashes=Counter();mask_hashes=Counter();class_pixels=Counter();sizes=Counter();bad_blinks=[];bad_open=[];landmark_errors=[];bad_paths=[];bad_labels=[]
    for row in rows:
        im_path,mask_path=Path(row["image"]),Path(row["mask"])
        if not im_path.is_file() or not mask_path.is_file():bad_paths.append(row["subject_id"]);continue
        image=Image.open(im_path).convert("RGB");mask=Image.open(mask_path).convert("L")
        if image.size!=mask.size or image.size!=(512,512):sizes[str((image.size,mask.size))]+=1
        values,counts=np.unique(np.asarray(mask,dtype=np.uint8),return_counts=True)
        for v,c in zip(values,counts):
            class_pixels[int(v)]+=int(c)
            if int(v) not in CLASSES.values():bad_labels.append((row["subject_id"],int(v)))
        for side,pupil,glint,blink in (("left",4,5,int(row["left_blink"])),("right",8,9,int(row["right_blink"]))):
            pm=np.asarray(mask)==pupil
            if blink:
                if pm.any() or (np.asarray(mask)==glint).any():bad_blinks.append({"subject_id":row["subject_id"],"side":side,"pupil_pixels":int(pm.sum()),"glint_pixels":int((np.asarray(mask)==glint).sum())})
            elif not pm.any():bad_open.append({"subject_id":row["subject_id"],"side":side})
            else:
                actual=np.argwhere(pm)[:,::-1].mean(0)
                meta=json.loads(row["conditions_json"]);uv=meta["landmarks"][side+"_pupil_center_uv"]
                truth=np.array([uv[0]*image.width,uv[1]*image.height]);err=float(np.linalg.norm(actual-truth))
                landmark_errors.append({"subject_id":row["subject_id"],"side":side,"error_px":err})
        image_hashes[digest(im_path)]+=1;mask_hashes[digest(mask_path)]+=1
    exact_image_dupes=sum(n-1 for n in image_hashes.values() if n>1)
    exact_mask_dupes=sum(n-1 for n in mask_hashes.values() if n>1)
    excessive=[x for x in landmark_errors if x["error_px"]>a.max_landmark_error_px]
    expected_classes=set(CLASSES.values());missing_classes=sorted(expected_classes-set(class_pixels))
    result={"status":"PASS" if not any(overlap.values()) and not bad_paths and not bad_labels and not bad_blinks and not bad_open and not excessive and not sizes and not exact_image_dupes and not missing_classes else "FAIL",
      "manifest":str(Path(a.manifest)),"frame_count":len(rows),"split_frames":{s:len(rr) for s,rr in by.items()},"split_subjects":{s:len(subjects[s]) for s in by},
      "subject_overlap_counts":overlap,"missing_paths":len(bad_paths),"invalid_label_values":len(bad_labels),"non_512_or_mismatched_images":dict(sizes),
      "distinct_mask_class_ids":sorted(class_pixels),"missing_mask_class_ids":missing_classes,"mask_pixels_by_class":dict(sorted(class_pixels.items())),
      "exact_duplicate_images":exact_image_dupes,"exact_duplicate_masks":exact_mask_dupes,
      "blink_cases_checked":sum(int(r["left_blink"])+int(r["right_blink"]) for r in rows),"blink_mask_violations":bad_blinks[:30],"open_eye_without_pupil_label":bad_open[:30],
      "open_eye_landmark_checks":len(landmark_errors),"landmark_error_px_mean":float(np.mean([x["error_px"] for x in landmark_errors])) if landmark_errors else None,
      "landmark_error_px_p95":float(np.percentile([x["error_px"] for x in landmark_errors],95)) if landmark_errors else None,
      "landmark_errors_over_limit":excessive[:30],"max_landmark_error_px":a.max_landmark_error_px,
      "note":"Synthetic renderer audit only; does not assess realism or camera-domain validity."}
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2))
    if result["status"]!="PASS":raise SystemExit(2)

if __name__=="__main__":main()
