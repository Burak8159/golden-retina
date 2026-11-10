"""Render a high-detail, CC0 Vitruvian GLB asset for a visual feasibility check."""
from __future__ import annotations

import bpy
import sys
from mathutils import Vector


def main() -> None:
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if len(args) != 2:
        raise SystemExit("usage: probe_vitruvian_glb.py -- input.glb output.png")
    source, output = args
    bpy.ops.import_scene.gltf(filepath=source)
    for obj in bpy.context.scene.objects:
        if obj.name == "Cube":
            obj.hide_render = True
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH" and o.name != "Cube"]
    if not meshes:
        raise RuntimeError("GLB import produced no mesh objects")
    bpy.context.view_layer.update()
    head = next((o for o in meshes if o.name == "cm_vitruvian"), max(meshes, key=lambda o: len(o.data.vertices)))
    # The GLB contains a helper/background object named Cube; framing all mesh
    # bounds would shrink the actual face to a thumbnail.
    corners = [head.matrix_world @ Vector(c) for c in head.bound_box]
    lo = Vector((min(v[i] for v in corners) for i in range(3)))
    hi = Vector((max(v[i] for v in corners) for i in range(3)))
    center = (lo + hi) * 0.5
    dimensions = hi - lo

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"
    scene.cycles.samples = 32
    scene.cycles.use_denoising = True
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "OPTIX"
    prefs.get_devices()
    for device in prefs.devices:
        device.use = device.type == "OPTIX"
    scene.render.resolution_x = 768
    scene.render.resolution_y = 768
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "AgX"
    scene.render.film_transparent = False
    world = bpy.data.worlds.new("vitruvian_probe_world")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.16, 0.17, 0.19, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.3
    scene.world = world

    # CharMorph/GLTF head convention faces -Y. The camera is placed front-on.
    cam_data = bpy.data.cameras.new("vitruvian_probe_camera")
    cam = bpy.data.objects.new("vitruvian_probe_camera", cam_data)
    scene.collection.objects.link(cam)
    cam.location = center + Vector((0, -max(dimensions.z * 3.2, 0.8), 0.025 * dimensions.z))
    cam.data.lens = 64
    cam.data.sensor_width = 36
    cam.rotation_euler = (center - cam.location).to_track_quat("-Z", "Y").to_euler()
    scene.camera = cam
    for name, offset, energy, size, color in (
        ("vit_key", (-.42, -.72, .66), 45, .46, (1.0, .9, .82)),
        ("vit_fill", (.46, -.64, .20), 24, .54, (.8, .87, 1.0)),
        ("vit_rim", (0, .46, .62), 32, .48, (1, .96, .9)),
    ):
        data = bpy.data.lights.new(name, "AREA")
        light = bpy.data.objects.new(name, data)
        scene.collection.objects.link(light)
        light.location = center + Vector(offset)
        light.data.energy = energy
        light.data.shape = "DISK"
        light.data.size = size
        light.data.color = color
        light.rotation_euler = (center - light.location).to_track_quat("-Z", "Y").to_euler()
    scene.render.filepath = output
    bpy.ops.render.render(write_still=True)
    print({"source": source, "mesh_objects": len(meshes),
           "vertices": sum(len(o.data.vertices) for o in meshes),
           "shape_keys": {o.name: len(o.data.shape_keys.key_blocks) if o.data.shape_keys else 0 for o in meshes},
           "materials": sorted({m.name for o in meshes for m in o.data.materials if m}),
           "dimensions": tuple(round(v, 4) for v in dimensions), "output": output}, flush=True)


if __name__ == "__main__":
    main()
