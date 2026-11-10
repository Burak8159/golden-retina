"""Check whether the imported face asset exposes usable material-index masks."""
from __future__ import annotations

import bpy
import json
import struct
import sys
import zlib
from collections import Counter
from pathlib import Path
import numpy as np


def write_png_rgb(path: Path, rgb: np.ndarray) -> None:
    h,w,_=rgb.shape
    def chunk(tag:bytes,data:bytes)->bytes:
        return struct.pack(">I",len(data))+tag+data+struct.pack(">I",zlib.crc32(tag+data)&0xffffffff)
    raw=b"".join(b"\x00"+rgb[y].astype(np.uint8,copy=False).tobytes() for y in range(h))
    payload=b"\x89PNG\r\n\x1a\n"+chunk(b"IHDR",struct.pack(">IIBBBBB",w,h,8,2,0,0,0))+chunk(b"IDAT",zlib.compress(raw,6))+chunk(b"IEND",b"")
    path.write_bytes(payload)


def main() -> None:
    args=sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else []
    if len(args)!=1:
        raise SystemExit("usage: probe_charmorph_material_masks.py -- output_dir")
    out=Path(args[0]);out.mkdir(parents=True,exist_ok=True)
    scene=bpy.context.scene
    scene.view_layers[0].use_pass_material_index=True
    scene.cycles.samples=1
    scene.cycles.use_denoising=False
    scene.use_nodes=True
    tree=scene.node_tree;tree.nodes.clear()
    layers=tree.nodes.new("CompositorNodeRLayers")
    composite=tree.nodes.new("CompositorNodeComposite")
    if "IndexMA" not in layers.outputs:
        raise RuntimeError("Blender compositor does not expose IndexMA material pass")
    tree.links.new(layers.outputs["IndexMA"],composite.inputs["Image"])
    scene.render.filepath=str(out/"material_index")
    scene.render.image_settings.file_format="OPEN_EXR"
    scene.render.image_settings.color_mode="BW"
    scene.render.image_settings.color_depth="32"
    bpy.ops.render.render(write_still=True)
    exr=out/"material_index.exr"
    if not exr.exists():
        candidates=sorted(out.glob("material_index*.exr"))
        if not candidates:raise RuntimeError("Material-index EXR was not written")
        exr=candidates[0]
    image=bpy.data.images.load(str(exr),check_existing=False)
    channels=int(image.channels);pixels=np.asarray(image.pixels[:],dtype=np.float32)
    w,h=map(int,image.size)
    if channels==1:idx=pixels.reshape(h,w)
    else:idx=pixels.reshape(h,w,channels)[...,0]
    # Save a compact integer map and a visually inspectable palette preview.
    labels=np.rint(idx).astype(np.uint8)[::-1].copy()
    np.save(out/"material_index.npy",labels)
    palette=np.array([[22,26,34],[230,230,230],[170,90,60],[8,8,8],[240,220,190],
                      [180,130,105],[20,20,20],[60,35,20],[220,170,145],[25,25,25],
                      [0,200,180],[0,120,220],[240,160,40],[130,80,220],[40,180,80]],dtype=np.uint8)
    rgb=palette[np.clip(labels,0,len(palette)-1)]
    write_png_rgb(out/"material_index_preview.png",rgb)
    vals,counts=np.unique(labels,return_counts=True)
    mats=[]
    for mat in bpy.data.materials:
        if hasattr(mat,"pass_index") and mat.pass_index:
            mats.append({"name":mat.name,"pass_index":int(mat.pass_index)})
    report={"status":"PASS" if len(vals)>1 else "FAIL","shape":[h,w],
            "unique_material_indices":{str(int(v)):int(c) for v,c in zip(vals,counts)},
            "materials":mats,"note":"single-frame synthetic material ID pass; not yet audited for anatomical completeness"}
    (out/"material_mask_probe.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":main()
