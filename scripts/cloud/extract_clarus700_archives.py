#!/usr/bin/env python3
"""Validate and extract the five public CLARUS700 image archives on the VM."""
from __future__ import annotations
import argparse, hashlib, json, zipfile
from pathlib import Path, PurePosixPath

EXPECTED={
    "PICTURE-1.zip":(1_992_461_975,"aea57c4423edda2b6d90d9e5af2e9cc9"),
    "PICTURE-2.zip":(2_565_766_184,"048e26fce581e47dbe7b04003fd53124"),
    "PICTURE-3.zip":(2_417_148_517,"5b49aeabaa9b68b38f4beace2e6e4d78"),
    "PICTURE-4.zip":(2_664_149_703,"9b4228aca215aba9e30faf64b07749de"),
    "PICTURE-5.zip":(2_266_149_861,"9846e11720053222019d8b0718ecd497"),
}
def md5(p:Path)->str:
    h=hashlib.md5()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""):h.update(b)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument("archive_root",type=Path);ap.add_argument("destination",type=Path);a=ap.parse_args()
    report=[];a.destination.mkdir(parents=True,exist_ok=True)
    for name,(size,expected) in EXPECTED.items():
        p=a.archive_root/name
        if p.stat().st_size!=size or md5(p)!=expected:raise RuntimeError(f"official archive checksum mismatch: {name}")
        with zipfile.ZipFile(p) as z:
            bad=z.testzip()
            if bad is not None:raise RuntimeError(f"ZIP CRC failure in {name}")
            for member in z.infolist():
                q=PurePosixPath(member.filename)
                if q.is_absolute() or ".." in q.parts:raise RuntimeError(f"unsafe path in archive {name}")
            z.extractall(a.destination)
            report.append({"archive":name,"archive_bytes":size,"md5":expected,"members":len(z.infolist()),"expanded_bytes":sum(x.file_size for x in z.infolist())})
            print(json.dumps(report[-1]),flush=True)
    images=list(a.destination.rglob("*.jpg"))+list(a.destination.rglob("*.jpeg"))+list(a.destination.rglob("*.tif"))+list(a.destination.rglob("*.tiff"))+list(a.destination.rglob("*.png"))
    out={"status":"complete","archives":report,"image_files":len(images),"expanded_bytes":sum(x.stat().st_size for x in images)}
    (a.destination/"cloud_extraction_aggregate.json").write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps(out),flush=True)
if __name__=="__main__":main()
