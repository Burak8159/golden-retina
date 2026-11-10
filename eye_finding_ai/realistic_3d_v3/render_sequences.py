"""Render identity-consistent synthetic eye-motion sequences in Blender."""
from __future__ import annotations

import bpy, sys, csv, json, math, time
from pathlib import Path
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_dataset as rd

args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
import argparse
p = argparse.ArgumentParser(); p.add_argument("--output", required=True); p.add_argument("--sequences", type=int, default=8)
p.add_argument("--frames", type=int, default=24); p.add_argument("--samples", type=int, default=8); p.add_argument("--seed", type=int, default=6100000)
a = p.parse_args(args); out = Path(a.output); (out/"images").mkdir(parents=True, exist_ok=True); (out/"masks").mkdir(parents=True, exist_ok=True)
rows=[];start=time.time()
for seq in range(a.sequences):
    subject_seed=a.seed+seq; _,base=rd.sample_params(subject_seed)
    for frame in range(a.frames):
        phase=2*math.pi*frame/max(a.frames,1)
        overrides={"gaze_x":.052*math.sin(phase)+.01*math.sin(3*phase), "gaze_z":.026*math.sin(2*phase),
                   "head_yaw_deg":base["head_yaw_deg"]+3.0*math.sin(phase*.5),
                   "head_pitch_deg":base["head_pitch_deg"]+1.5*math.cos(phase*.6),
                   "head_roll_deg":base["head_roll_deg"]+.8*math.sin(phase*.4),
                   "camera_yaw_deg":base["camera_yaw_deg"]+1.2*math.sin(phase*.3),
                   "exposure":base["exposure"]*(1+.06*math.sin(phase*.25)),
                   "blink_left":frame%17 in (8,9), "blink_right":frame%23==14}
        scene,params,eyes=rd.build_scene(subject_seed,overrides);scene.cycles.samples=a.samples
        name=f"sequence_{seq:03d}_frame_{frame:04d}";image=out/"images"/(name+".png");mask_path=out/"masks"/(name+".png")
        scene.render.filepath=str(image);bpy.ops.render.render(write_still=True)
        mask=rd.get_index_mask(scene,[o for e in eyes for o in e["corneas"]],out/"masks"/(name+"_index"));rd.write_png_gray(mask_path,mask)
        row={"image":str(image),"mask":str(mask_path),"sequence_id":f"sequence-{seq:03d}","frame":frame,"subject_seed":subject_seed,
             "left_blink":int(overrides["blink_left"]),"right_blink":int(overrides["blink_right"])}
        for e in eyes:
            # project_point stores y in top-down image coordinates already.
            uv=rd.project_point(scene,e["pupil_center"]);row[e["side"]+"_pupil_xy"]="" if e["blink"] else json.dumps([round(uv[0]*rd.W,3),round(uv[1]*rd.H,3)])
        rows.append(row)
        if len(rows)%20==0:print(json.dumps({"frames":len(rows),"sequences":a.sequences,"elapsed_min":round((time.time()-start)/60,2)}),flush=True)
with (out/"sequence_manifest.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(out/"sequence_config.json").write_text(json.dumps({"status":"SYNTHETIC_IDENTITY_CONSISTENT_MOTION","sequences":a.sequences,"frames_per_sequence":a.frames,
    "total_frames":len(rows),"seed":a.seed,"same_generated_identity_within_sequence":True,"physical_camera_validated":False},indent=2)+"\n")
print(json.dumps({"status":"SEQUENCE_RENDER_COMPLETE","frames":len(rows),"elapsed_min":round((time.time()-start)/60,2),"manifest":str(out/"sequence_manifest.csv")},indent=2),flush=True)
