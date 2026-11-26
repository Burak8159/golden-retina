"""Extract only the official RIM-ONE hospital-partition training images."""
from __future__ import annotations
import argparse, hashlib, json, zipfile
from pathlib import Path, PurePosixPath

IMAGE = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument("--archive",type=Path,required=True);p.add_argument("--destination",type=Path,required=True);p.add_argument("--report",type=Path,required=True);a=p.parse_args()
    if a.destination.exists() and any(a.destination.iterdir()): raise SystemExit("Destination is nonempty; refusing overwrite")
    a.destination.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(a.archive) as z:
        infos=z.infolist()
        safe=[]
        for info in infos:
            name=info.filename.replace("\\","/"); rel=PurePosixPath(name)
            if rel.is_absolute() or ".." in rel.parts: raise SystemExit("Unsafe ZIP path")
            safe.append((info,name,rel))
        variants={}
        for info,name,rel in safe:
            parts=[x.lower() for x in rel.parts]
            for i,x in enumerate(parts):
                if x=="training_set" and i+1<len(parts) and parts[i+1] in {"normal","glaucoma"} and Path(name).suffix.lower() in IMAGE and info.file_size:
                    variant="/".join(rel.parts[:i])
                    variants.setdefault(variant,[]).append((info,name,rel.parts[i:]))
        picks=[(k,v) for k,v in variants.items() if "hospital" in k.lower()]
        if len(picks)!=1: raise SystemExit(f"Expected one hospital-partition training tree, found {len(picks)}")
        variant,entries=picks[0]
        classes={"normal":0,"glaucoma":0}; total_bytes=0
        for info,name,tail in entries:
            out=a.destination.joinpath(*tail)
            out.parent.mkdir(parents=True,exist_ok=True)
            # Reading each selected training image validates its CRC. We never
            # stream official test image payloads or parse their labels.
            with z.open(info) as src,out.open("wb") as dst:
                while True:
                    chunk=src.read(1<<20)
                    if not chunk: break
                    dst.write(chunk)
            if tail[1].lower() not in classes: raise SystemExit("Unexpected class directory")
            classes[tail[1].lower()]+=1;total_bytes+=info.file_size
    report={"status":"HOSPITAL_TRAINING_ONLY_EXTRACTED","source_variant":variant,"image_counts_by_class":classes,"total_training_images":sum(classes.values()),"uncompressed_bytes":total_bytes,"archive_sha256":sha(a.archive),"official_test_entries_extracted":False,"random_partition_entries_extracted":False,"row_level_paths_exported":False,"terms":"research and educational use only; no redistribution"}
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8");print(json.dumps(report))

if __name__=="__main__":main()
