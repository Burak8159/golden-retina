#!/usr/bin/env python3
"""Cloud-side structural audit for the official FIVES archive; emits aggregates only."""
from __future__ import annotations
import argparse, collections, hashlib, json
from pathlib import Path
from PIL import Image

def sha(p: Path) -> str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("root",type=Path); ap.add_argument("--out",type=Path,required=True); a=ap.parse_args()
    report={"source":"FIVES official archive","split_audit":{},"patient_ids_available":False,"boundary":"Image split supplied by source; not patient-grouped. Structural audit only."}
    digests={}
    for split in ("train","test"):
        base=a.root/split; orig=sorted((base/"Original").glob("*.png")); masks=sorted((base/"Ground truth").glob("*.png"))
        cls=collections.Counter(p.stem.rsplit("_",1)[-1] for p in orig)
        om={p.stem for p in orig}; mm={p.stem for p in masks}
        bad=0; image_hashes=[]
        for p in orig:
            try:
                with Image.open(p) as im: im.load()
            except Exception: bad+=1
            else: image_hashes.append(sha(p))
        counts=collections.Counter(image_hashes); hashes=set(image_hashes); digests[split]=hashes
        report["split_audit"][split]={"original_images":len(orig),"ground_truth_masks":len(masks),"unpaired_originals":len(om-mm),"unpaired_masks":len(mm-om),"class_counts":dict(sorted(cls.items())),"image_decode_failures":bad,"unique_exact_hashes":len(hashes),"within_split_exact_duplicate_groups":sum(n>1 for n in counts.values()),"within_split_duplicate_rows":sum(n-1 for n in counts.values() if n>1)}
    report["train_test_exact_image_hash_overlap"]=len(digests["train"]&digests["test"])
    report["structural_complete"]=(all(report["split_audit"][s]["original_images"]==({"train":600,"test":200}[s]) and report["split_audit"][s]["ground_truth_masks"]==({"train":600,"test":200}[s]) and report["split_audit"][s]["unpaired_originals"]==0 and report["split_audit"][s]["unpaired_masks"]==0 and report["split_audit"][s]["image_decode_failures"]==0 for s in ("train","test")) and report["train_test_exact_image_hash_overlap"]==0)
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,indent=2))
if __name__=="__main__":main()
