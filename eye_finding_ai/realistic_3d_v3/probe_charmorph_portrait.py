"""One-frame portrait feasibility probe for the CC0 CharMorph Vitruvian base.

This is deliberately separate from the current training renderer. It checks
whether the supplied morphable head, hair asset, eye materials, and portrait
framing can form a better-controlled synthetic capture before any dataset is
generated from it.
"""
from __future__ import annotations

import bpy
import os
import random
import sys
from pathlib import Path
from mathutils import Vector


def main() -> None:
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if len(args) not in (2, 3):
        raise SystemExit("usage: probe_charmorph_portrait.py -- char_morph_root output.png [scene.blend]")
    addon_root, output = args[:2]
    scene_output = args[2] if len(args) == 3 else None
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    sys.path.insert(0, os.path.dirname(addon_root))
    import CharMorph
    from CharMorph.lib.charlib import library

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

    # Seeded face-shape variation; age is allowed to vary, while fantasy and
    # expression/pose morphs are excluded from this static eye-finding probe.
    random.seed(815900)
    ui.randomize_mode = "RL1"
    ui.randomize_strength = 0.24
    ui.randomize_morphs = True
    ui.randomize_incl = ".*"
    ui.randomize_excl = "Fantasy|Expression|Viseme|Pose"
    bpy.ops.charmorph.randomize()

    # Use the CC0 character pack's supplied hair library rather than the
    # procedural scalp cap, then frame a face portrait so torso censor geometry
    # does not dominate the training-style input.
    char_conf = library.obj_char(head)
    styles = list(char_conf.hairstyles)
    hair_style = next((s for s in ("Bob", "Eve", "SlickedBack") if s in styles), None)
    hair_color = next(iter(library.hair_colors), None)
    if hair_style and hair_color:
        ui.hair_style = hair_style
        ui.hair_color = hair_color
        hair_result = bpy.ops.charmorph.hair_create()
        if "FINISHED" not in hair_result:
            raise RuntimeError(f"Hair import failed for {hair_style}: {hair_result}")

    bpy.context.view_layer.update()
    corners = [head.matrix_world @ Vector(c) for c in head.bound_box]
    lo = Vector((min(v[i] for v in corners) for i in range(3)))
    hi = Vector((max(v[i] for v in corners) for i in range(3)))
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
    world = bpy.data.worlds.new("portrait_probe_world")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (.10, .11, .13, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = .22
    scene.world = world

    # Camera values are derived from the imported head bounds, not the body.
    target = Vector((0, 0, lo.z + .91 * dims.z))
    distance = max(.49 * dims.z, .78)
    camera_data = bpy.data.cameras.new("portrait_probe_camera")
    camera = bpy.data.objects.new("portrait_probe_camera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = target + Vector((0, -distance, .035 * dims.z))
    camera.data.lens = 82
    camera.data.sensor_width = 36
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
    scene.camera = camera
    for name, offset, energy, size, color in (
        ("key", (-.65, -1.0, .65), 125, .72, (1, .91, .83)),
        ("fill", (.62, -.9, .10), 58, 1.0, (.78, .85, 1)),
        ("rim", (0, .62, .55), 86, .9, (1, .96, .9)),
    ):
        data = bpy.data.lights.new(name, "AREA")
        light = bpy.data.objects.new(name, data)
        scene.collection.objects.link(light)
        light.location = target + Vector(offset)
        light.data.energy = energy
        light.data.shape = "DISK"
        light.data.size = size
        light.data.color = color
        light.rotation_euler = (target - light.location).to_track_quat("-Z", "Y").to_euler()
    scene.render.filepath = output
    scene.view_layers[0].use_pass_material_index = True
    material_summary = []
    for index, material in enumerate(head.data.materials, start=1):
        if material is None:
            continue
        if hasattr(material, "pass_index"):
            material.pass_index = index
        material_summary.append({"name": material.name,
                                 "has_pass_index": hasattr(material, "pass_index"),
                                 "pass_index": getattr(material, "pass_index", None)})
    if scene_output:
        Path(scene_output).parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=scene_output)
    bpy.ops.render.render(write_still=True)
    print({"asset": "CharMorph Vitruvian CC0", "vertices": len(head.data.vertices),
           "shape_keys": len(head.data.shape_keys.key_blocks) if head.data.shape_keys else 0,
           "hair_style": hair_style, "hair_color": hair_color,
           "material_index_summary": material_summary,
           "scene_output": scene_output,
           "output": output}, flush=True)


if __name__ == "__main__":
    main()
