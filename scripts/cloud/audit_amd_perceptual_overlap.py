#!/usr/bin/env python3
"""Private perceptual near-duplicate screen for saturated AMD results.

Computes 64-bit DCT perceptual hashes for BMRD AMD, Rajbari AMD/AMD-none, and
Mendeley13 AMD/normal images. Path-level nearest-neighbour pairs stay on VM.
This heuristic is a review trigger, not proof of identity or independence.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from scipy.fft import dctn

EXT={'.jpg','.jpeg','.png','.tif','.tiff','.bmp'}
def phash(path: Path) -> int:
    with Image.open(path) as im:
        a=np.asarray(ImageOps.exif_transpose(im).convert('L').resize((32,32)),dtype=np.float32)
    d=dctn(a,norm='ortho')[:8,:8].reshape(-1); med=np.median(d[1:])
    return sum((int(v>med)<<i) for i,v in enumerate(d))
def dhash(path: Path) -> int:
    with Image.open(path) as im:
        a=np.asarray(ImageOps.exif_transpose(im).convert('L').resize((9,8)),dtype=np.uint8)
    bits=(a[:,1:]>a[:,:-1]).reshape(-1)
    return sum((int(v)<<i) for i,v in enumerate(bits))
def images(root, predicate):
    return [p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in EXT and predicate(p)]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bmrd',type=Path,required=True);ap.add_argument('--rajbari',type=Path,required=True);ap.add_argument('--mendeley13',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    roots={
      'bmrd_amd':images(a.bmrd,lambda p:'amd' in str(p.parent).lower()),
      'rajbari_amd_or_none':images(a.rajbari,lambda p:any(k in str(p.parent).lower() for k in ('age-related macular degeneration','no age-related macular degeneration'))),
      'mendeley13_amd_or_normal':images(a.mendeley13,lambda p:p.parent.name.lower() in {'dry_amd','wet_amd','normal_fundus'}),
    }
    hashes={k:[(phash(p),dhash(p)) for p in v] for k,v in roots.items()}
    pairs={}
    names=list(roots)
    for i,left in enumerate(names):
      for right in names[i+1:]:
        near=[]; mins=[]
        for li,(lh,ld) in enumerate(hashes[left]):
          ds=[(lh^rh[0]).bit_count() for rh in hashes[right]]
          if not ds: continue
          j=int(np.argmin(ds)); mins.append(ds[j]); dd=(ld^hashes[right][j][1]).bit_count()
          if ds[j]<=8: near.append({'phash_distance':int(ds[j]),'dhash_distance':int(dd),'left':str(roots[left][li]),'right':str(roots[right][j])})
        pairs[left+'__'+right]={'left_count':len(roots[left]),'right_count':len(roots[right]),'left_images_nearest_phash_hamming_le_8':len(near),'joint_phash_and_dhash_le_8':sum(x['dhash_distance']<=8 for x in near),'pHash_minimum_distance_histogram':{str(d):mins.count(d) for d in sorted(set(mins)) if d<=16},'private_near_pairs':near}
    report={'method':'64-bit DCT pHash, 32x32 grayscale, threshold Hamming distance <=8; not rotation invariant, no claim of identity.','counts':{k:len(v) for k,v in roots.items()},'pairs':pairs}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'counts':report['counts'],'pairs':{k:{kk:vv for kk,vv in v.items() if kk!='private_near_pairs'} for k,v in pairs.items()}},indent=2))
if __name__=='__main__':main()
