"""Summarize eye geometry/material topology in a saved CharMorph scene."""
from __future__ import annotations
import bpy, json, sys
from collections import Counter, defaultdict
from pathlib import Path
from bpy_extras.object_utils import world_to_camera_view

def main():
    args=sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else []
    if len(args)!=1:raise SystemExit("usage: inspect_charmorph_scene.py -- output.json")
    out=Path(args[0]);out.parent.mkdir(parents=True,exist_ok=True)
    meshes=[o for o in bpy.data.objects if o.type=="MESH"]
    head=next((o for o in meshes if o.name=="cm_vitruvian"),meshes[0] if meshes else None)
    if head is None:raise RuntimeError("no mesh in saved scene")
    depsgraph=bpy.context.evaluated_depsgraph_get();head_eval=head.evaluated_get(depsgraph);eval_mesh=head_eval.to_mesh()
    counts=Counter();faces_by={}
    for i,slot in enumerate(head.material_slots):
        name=slot.material.name if slot.material else f"slot_{i}"
        faces={p.index for p in head.data.polygons if p.material_index==i}
        counts[name]=len(faces);faces_by[name]=faces
    # Connected components of polygons in the Sclera_Cornea slot reveal whether
    # the two eye shells are topologically independent.
    sclera_idx=next((i for i,s in enumerate(head.material_slots) if s.material and s.material.name=="Sclera_Cornea"),None)
    component_details={}
    for material_name in ("Sclera_Cornea","Iris","Pupil"):
        material_idx=next((i for i,s in enumerate(head.material_slots) if s.material and s.material.name==material_name),None)
        if material_idx is None:continue
        polygons=[p for p in head.data.polygons if p.material_index==material_idx]
        owner=defaultdict(list);adj=defaultdict(set)
        for p in polygons:
            for v in p.vertices:owner[v].append(p.index)
        for p in polygons:
            for v in p.vertices:
                for other in owner[v]:
                    if other!=p.index:adj[p.index].add(other)
        seen=set();details=[]
        for p in polygons:
            if p.index in seen:continue
            stack=[p.index];seen.add(p.index);n=0
            component_faces=[];verts=set()
            while stack:
                q=stack.pop();n+=1
                component_faces.append(q);verts.update(head.data.polygons[q].vertices)
                for z in adj[q]-seen:seen.add(z);stack.append(z)
            local=sum((eval_mesh.vertices[v].co for v in verts),start=eval_mesh.vertices[0].co*0)/max(len(verts),1)
            world=head_eval.matrix_world @ local
            uv=world_to_camera_view(bpy.context.scene,bpy.context.scene.camera,world)
            details.append({"faces":n,"vertices":len(verts),"center_local_xyz":[round(float(x),6) for x in local],
                            "center_uv":[round(float(uv.x),6),round(float(uv.y),6)],
                            "center_px_top_origin":[round(float(uv.x*bpy.context.scene.render.resolution_x),2),
                                                     round(float((1-uv.y)*bpy.context.scene.render.resolution_y),2)]})
        component_details[material_name]=sorted(details,key=lambda x:x["center_uv"][0])
    keys=[]
    if head.data.shape_keys:
        keys=[k.name for k in head.data.shape_keys.key_blocks if any(t in k.name.lower() for t in ("eye","iris","pupil","gaze","look"))]
        shape_key_controls=[{"name":k.name,"value":round(float(k.value),5),"min":round(float(k.slider_min),4),"max":round(float(k.slider_max),4)} for k in head.data.shape_keys.key_blocks]
    else:shape_key_controls=[]
    report={"head_object":head.name,"mesh_vertices":len(head.data.vertices),"mesh_polygons":len(head.data.polygons),
            "material_face_counts":dict(counts),"eye_geometry_components":component_details,
            "eye_related_shape_keys":keys,"shape_key_controls":shape_key_controls,
            "mesh_objects":[{"name":o.name,"vertices":len(o.data.vertices)} for o in meshes],
            "other_scene_objects":[{"name":o.name,"type":o.type} for o in bpy.data.objects if o.type!="MESH"]}
    head_eval.to_mesh_clear()
    out.write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2),flush=True)
if __name__=="__main__":main()
