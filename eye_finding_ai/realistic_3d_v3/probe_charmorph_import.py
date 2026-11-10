"""Feasibility probe: create and render a textured/morphable CharMorph head."""
from __future__ import annotations

import bpy
import os
import random
import sys
from mathutils import Vector


def main() -> None:
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if len(args) != 2:
        raise SystemExit("usage: probe_charmorph_import.py -- char_morph_root output.png")
    addon_root, output = args
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    sys.path.insert(0, os.path.dirname(addon_root))
    import CharMorph
    CharMorph.register()
    ui = bpy.context.window_manager.charmorph_ui
    ui.base_model = "Vitruvian"
    ui.tex_downscale = "2K"
    ui.import_morphs = True
    ui.import_expressions = False
    ui.use_sk = True
    result = bpy.ops.charmorph.import_char()
    if "FINISHED" not in result:
        raise RuntimeError(f"CharMorph character import failed: {result}")
    head = bpy.context.view_layer.objects.active
    if not head or head.type != "MESH":
        raise RuntimeError("CharMorph did not return an active mesh")

    # Seeded moderate variation demonstrates the native morph interface without
    # changing the data used by the separate production renderer.
    random.seed(815900)
    ui.randomize_mode = "RL1"
    ui.randomize_strength = 0.16
    ui.randomize_morphs = True
    ui.randomize_mats = False
    ui.randomize_incl = ".*"
    ui.randomize_excl = "Age|Fantasy|Expression|Viseme|Pose"
    bpy.ops.charmorph.randomize()
    bpy.context.view_layer.update()

    world_corners = [head.matrix_world @ Vector(c) for c in head.bound_box]
    lo = Vector((min(v[i] for v in world_corners) for i in range(3)))
    hi = Vector((max(v[i] for v in world_corners) for i in range(3)))
    center = (lo + hi) * 0.5
    dims = hi - lo
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
    world = bpy.data.worlds.new("charmorph_probe_world")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (.16, .17, .19, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = .3
    scene.world = world

    camera_data = bpy.data.cameras.new("charmorph_probe_camera")
    camera = bpy.data.objects.new("charmorph_probe_camera", camera_data)
    scene.collection.objects.link(camera)
    face_target = Vector((center.x, center.y, lo.z + .86 * dims.z))
    camera.location = face_target + Vector((0, -max(dims.z * .78, .9), .015 * dims.z))
    camera.data.lens = 82
    camera.data.sensor_width = 36
    camera.rotation_euler = (face_target - camera.location).to_track_quat("-Z", "Y").to_euler()
    scene.camera = camera
    for name, offset, energy, size, color in (
        ("probe_key", (-.8, -1.3, .9), 145, 1.0, (1, .9, .82)),
        ("probe_fill", (.75, -1.1, .25), 75, 1.25, (.8, .87, 1)),
        ("probe_rim", (0, .75, .8), 100, 1.1, (1, .96, .9)),
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
    print({"asset": "CharMorph Vitruvian (CC0 character data)", "object": head.name,
           "vertices": len(head.data.vertices),
           "shape_keys": len(head.data.shape_keys.key_blocks) if head.data.shape_keys else 0,
           "materials": [m.name for m in head.data.materials if m],
           "dimensions": tuple(round(v, 4) for v in dims), "output": output}, flush=True)


if __name__ == "__main__":
    main()
