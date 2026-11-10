"""Render a one-frame check of a CharMorph character asset already opened in Blender.

This is an asset-quality probe, not a training pipeline or proof of identity coverage.
Run Blender with the external character .blend as the startup file, then pass an output PNG.
"""
from __future__ import annotations

import bpy
import math
import sys
from mathutils import Vector


def main() -> None:
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if len(args) != 1:
        raise SystemExit("usage: probe_charmorph_asset.py -- output.png")
    output = args[0]
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    if not meshes:
        raise RuntimeError("No mesh found in the loaded character file")
    head = max(meshes, key=lambda o: len(o.data.vertices))
    bpy.context.view_layer.update()
    corners = [head.matrix_world @ Vector(c) for c in head.bound_box]
    center = sum(corners, Vector()) / len(corners)
    dimensions = head.dimensions

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"
    scene.cycles.samples = 24
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
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "AgX"

    world = bpy.data.worlds.new("probe_world")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.18, 0.19, 0.21, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.35
    scene.world = world

    cam_loc = center + Vector((0, -max(dimensions.z * 2.4, 3.2), dimensions.z * 0.03))
    camera_data = bpy.data.cameras.new("probe_camera")
    camera = bpy.data.objects.new("probe_camera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = cam_loc
    camera.data.lens = 62
    camera.data.sensor_width = 36
    camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
    scene.camera = camera

    for name, offset, energy, size, color in (
        ("key", (-2.2, -3.0, 2.5), 420, 2.3, (1.0, 0.89, 0.79)),
        ("fill", (2.0, -2.7, 0.8), 220, 2.8, (0.78, 0.86, 1.0)),
        ("rim", (0.0, 1.7, 2.1), 300, 2.5, (1.0, 0.95, 0.88)),
    ):
        light_data = bpy.data.lights.new(name, type="AREA")
        light = bpy.data.objects.new(name, light_data)
        scene.collection.objects.link(light)
        light.location = center + Vector(offset)
        light.name = name
        light.data.energy = energy
        light.data.shape = "DISK"
        light.data.size = size
        light.data.color = color
        light.rotation_euler = (center - light.location).to_track_quat("-Z", "Y").to_euler()

    scene.render.filepath = output
    bpy.ops.render.render(write_still=True)
    print({"asset_object": head.name, "vertices": len(head.data.vertices),
           "dimensions": tuple(round(v, 4) for v in dimensions),
           "materials": [m.name for m in head.data.materials if m],
           "shape_keys": len(head.data.shape_keys.key_blocks) if head.data.shape_keys else 0,
           "output": output}, flush=True)


if __name__ == "__main__":
    main()
