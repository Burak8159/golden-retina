import bpy, sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import render_dataset as r
out=Path('/data/golden-retina/results/realistic_3d_v3_debug');out.mkdir(parents=True,exist_ok=True)
scene,p,eyes=r.build_scene(815900)
print('PARAMS', {k:p[k] for k in ['head_yaw_deg','head_pitch_deg','head_roll_deg','camera_yaw_deg','camera_pitch_deg','hair_style','capture_mode','camera_distance','camera_lens_mm']},flush=True)
print('OBJECTS',[(o.name,o.pass_index,tuple(round(v,3) for v in o.matrix_world.translation),tuple(round(v,3) for v in o.dimensions)) for o in scene.objects if o.pass_index in (1,2,3,4,5,6,7,8,9,10,11)],flush=True)
print('GLINT_SCALE',[(o.name,tuple(o.scale),tuple(o.matrix_basis.to_scale()),tuple(o.matrix_parent_inverse.to_scale())) for o in scene.objects if 'glint' in o.name],flush=True)
scene.cycles.samples=8;scene.render.filepath=str(out/'face.png');bpy.ops.render.render(write_still=True)
mask=r.get_index_mask(scene,[o for e in eyes for o in e['corneas']],out/'index');r.write_png_gray(out/'mask.png',mask)
print('MASK_IDS',np.unique(mask,return_counts=True),flush=True)
