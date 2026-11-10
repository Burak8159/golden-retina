"""Render a controlled, fixed-face synthetic iris/pupil translation sequence.

The movement is a labelled geometric translation, not physical eyeball
rotation or real camera motion. It is for pipeline/tracker smoke evaluation.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_charmorph_variants import (
    apply_gaze_offset, frame_camera_on_eyes, make_gaze_snapshot,
    render_mask, set_morphs,
)


def main():
    argv=sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else []
    p=argparse.ArgumentParser();p.add_argument("output_dir");p.add_argument("--frames",type=int,default=32)
    p.add_argument("--resolution",type=int,default=512);p.add_argument("--samples",type=int,default=2)
    p.add_argument("--seed",type=int,default=817001);a=p.parse_args(argv)
    root=Path(a.output_dir);root.mkdir(parents=True,exist_ok=True)
    scene=bpy.context.scene;head=bpy.data.objects.get("cm_vitruvian")
    if head is None or not head.data.shape_keys:raise RuntimeError("expected CharMorph base mesh")
    scene.render.engine="CYCLES";scene.render.resolution_x=a.resolution;scene.render.resolution_y=a.resolution
    scene.render.resolution_percentage=100;scene.cycles.samples=a.samples;scene.cycles.use_denoising=True
    # A fixed eye morphology and face preset for a repeatable trajectory.
    case={"size":.60,"distance":0.0,"vertical_shift":-.02,"upper_lid":.86,
          "lower_lid":.60,"monolid":.20,"hooded":.22}
    set_morphs(head,case)
    bpy.context.view_layer.update(); framing=frame_camera_on_eyes(scene,head)
    snapshot=make_gaze_snapshot(head);records=[]
    for i in range(a.frames):
        phase=2*math.pi*i/max(a.frames,1)
        dx=.0032*math.sin(phase);dz=.0017*math.sin(2*phase)
        apply_gaze_offset(head,snapshot,dx,dz)
        out=root/f"frame_{i:04d}";out.mkdir(parents=True,exist_ok=True)
        scene.use_nodes=False;scene.cycles.samples=a.samples;scene.cycles.use_denoising=True
        scene.render.image_settings.file_format="PNG";scene.render.image_settings.color_mode="RGB"
        scene.render.filepath=str(out/"beauty");bpy.ops.render.render(write_still=True)
        labels=render_mask(scene,head,out)
        rec={"frame":i,"seed":a.seed,"gaze_translation_x_m":dx,"gaze_translation_z_m":dz,
             "camera_framing_baseline":framing,"labels":labels}
        (out/"metadata.json").write_text(json.dumps(rec,indent=2)+"\n");records.append(rec)
        print(json.dumps({"frame":i,"pupil_pixels":labels["pupil"]["pixels"]}),flush=True)
    report={"status":"CONTROLLED_SYNTHETIC_TRANSLATION_SEQUENCE_RENDERED","frames":len(records),
            "seed":a.seed,"resolution":[a.resolution,a.resolution],"samples":a.samples,
            "independent_identity_count":1,"sequence":records,
            "limitations":["single synthetic base identity","iris/pupil geometry translated rather than eyeball-rotated",
                           "fixed camera and face; no head movement, blinking, real sensor, or real-person footage",
                           "tracking metrics are synthetic pipeline diagnostics only"]}
    (root/"sequence_report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"status":report["status"],"frames":len(records)}),flush=True)


if __name__=="__main__":main()
