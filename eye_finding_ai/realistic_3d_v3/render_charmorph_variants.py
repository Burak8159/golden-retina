"""Render a controlled four-case CharMorph eye-label feasibility set.

This is a label/variation probe, not a full synthetic cohort or human-data
surrogate. The four outputs share one CC0 base identity and differ through
explicitly recorded morph controls.
"""
from __future__ import annotations

import bpy
import argparse
import json
import math
import random
import struct
import sys
import zlib
from pathlib import Path

import numpy as np


CASES = [
    {"name": "open_neutral", "size": 0.52, "distance": -0.25, "upper_lid": 0.88, "lower_lid": 0.52,
     "monolid": 0.12, "hooded": 0.18},
    {"name": "hooded", "size": 0.68, "distance": 0.18, "upper_lid": 0.70, "lower_lid": 0.42,
     "monolid": 0.20, "hooded": 0.72},
    {"name": "partially_closed", "size": 0.46, "distance": -0.08, "upper_lid": 0.42, "lower_lid": 0.62,
     "monolid": 0.38, "hooded": 0.62},
    {"name": "monolid", "size": 0.78, "distance": 0.32, "upper_lid": 0.78, "lower_lid": 0.72,
     "monolid": 0.84, "hooded": 0.22},
]


def write_png(path: Path, rgb: np.ndarray) -> None:
    h, w, _ = rgb.shape

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff))

    raw = b"".join(b"\x00" + rgb[y].astype(np.uint8, copy=False).tobytes()
                    for y in range(h))
    path.write_bytes(b"\x89PNG\r\n\x1a\n"
                     + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def set_morphs(head, case):
    keys = head.data.shape_keys.key_blocks
    values = {
        "L2__Eyes_Size": case["size"],
        "L2__Eyes_Distance": case["distance"],
        "L2__Eyes_VerticalShift": case.get("vertical_shift", -0.04),
        "L2__Eyes_UpperLidOpenness": case["upper_lid"],
        "L2__Eyes_LowerLidOpenness": case["lower_lid"],
        "L2__Eyes_Eyelid_Monolid": case["monolid"],
        "L2__Eyes_Eyelid_hooded": case["hooded"],
    }
    for name, value in values.items():
        if name in keys:
            keys[name].value = float(value)
    bpy.context.view_layer.update()
    return values


def sample_cases(count: int, seed: int):
    """Build deterministic bounded eye-morph and gaze-offset samples.

    Iris/pupil motion is a small local X/Z translation for exact label checks;
    it is not a physically accurate eyeball-rotation model.
    """
    rng = random.Random(seed)
    gaze_anchors = [(0.0, 0.0), (-0.0035, 0.0), (0.0035, 0.0),
                    (0.0, -0.0020), (0.0, 0.0020)]
    face_presets = ["asset_default", "male_morph_delta", "female_morph_delta", "young_morph_delta",
                    "older_morph_delta", "male_older_morph_delta", "female_young_morph_delta",
                    "asset_default"]
    result = []
    for i in range(count):
        gx, gz = gaze_anchors[i] if i < len(gaze_anchors) else (
            rng.uniform(-0.0035, 0.0035), rng.uniform(-0.0020, 0.0020))
        result.append({
            "name": f"seed_{seed}_{i:04d}",
            "size": rng.uniform(0.46, 0.76),
            "distance": rng.uniform(-0.25, 0.28),
            "vertical_shift": rng.uniform(-0.12, 0.08),
            "upper_lid": rng.uniform(0.48, 0.91),
            "lower_lid": rng.uniform(0.38, 0.78),
            "monolid": rng.uniform(0.0, 0.82),
            "hooded": rng.uniform(0.0, 0.78),
            "gaze_x_m": gx,
            "gaze_z_m": gz,
            "face_preset": face_presets[i % len(face_presets)],
        })
    return result


def apply_face_preset(head, preset: str, base_values):
    """Apply bounded deltas around the scene's original face morph mix.

    The source asset already has multiple age/gender controls mixed in. Hard
    zeroing those controls can move the face out of the fixed camera view, so
    this probe retains the base mix and tests only small, recorded deltas.
    """
    keys = head.data.shape_keys.key_blocks
    assignments = {
        "male_morph_delta": {"L2__Gender_Male": 0.10},
        "female_morph_delta": {"L2__Gender_Female": 0.10},
        "young_morph_delta": {"L2__Age_Baby": 0.20},
        "older_morph_delta": {"L2__Age_Old": 0.20},
        "male_older_morph_delta": {"L2__Gender_Male": 0.08, "L2__Age_Old": 0.15},
        "female_young_morph_delta": {"L2__Gender_Female": 0.08, "L2__Age_Baby": 0.15},
    }
    values = dict(base_values)
    for key, delta in assignments.get(preset, {}).items():
        if key in keys:
            value = min(1.0, max(0.0, base_values.get(key, keys[key].value) + delta))
            keys[key].value = value
            values[key] = value
    bpy.context.view_layer.update()
    return values


def make_gaze_snapshot(head):
    material_ids = {i for i, slot in enumerate(head.material_slots)
                    if slot.material and slot.material.name in {"Iris", "Pupil"}}
    vertices = {v for poly in head.data.polygons if poly.material_index in material_ids
                for v in poly.vertices}
    blocks = head.data.shape_keys.key_blocks
    original = {block.name: {idx: block.data[idx].co.copy() for idx in vertices}
                for block in blocks}
    return vertices, original


def apply_gaze_offset(head, snapshot, dx: float, dz: float):
    vertices, original = snapshot
    for block in head.data.shape_keys.key_blocks:
        for idx in vertices:
            co = original[block.name][idx]
            block.data[idx].co = (co.x + dx, co.y, co.z + dz)
    bpy.context.view_layer.update()


def frame_camera_on_eyes(scene, head, target_y: float = 0.60):
    """Re-aim the existing camera so the evaluated eye center stays in frame."""
    from mathutils import Vector
    from bpy_extras.object_utils import world_to_camera_view

    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = head.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    eye_material_ids = {i for i, slot in enumerate(head.material_slots)
                        if slot.material and slot.material.name in {"Iris", "Pupil"}}
    vertex_ids = {v for poly in mesh.polygons if poly.material_index in eye_material_ids
                  for v in poly.vertices}
    if not vertex_ids:
        evaluated.to_mesh_clear()
        raise RuntimeError("no evaluated iris/pupil geometry for camera framing")
    points = [evaluated.matrix_world @ mesh.vertices[i].co for i in vertex_ids]
    eye_center = sum(points, Vector((0.0, 0.0, 0.0))) / len(points)
    evaluated.to_mesh_clear()

    camera = scene.camera
    distance = (eye_center - camera.location).length
    up = camera.matrix_world.to_3x3() @ Vector((0.0, 1.0, 0.0))
    half_angle = float(camera.data.angle_y) / 2.0
    view_angle = math.atan((2.0 * target_y - 1.0) * math.tan(half_angle))
    target = eye_center - up * (distance * math.tan(view_angle))
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
    bpy.context.view_layer.update()
    projected = world_to_camera_view(scene, camera, eye_center)
    return {"eye_center_world": [float(v) for v in eye_center],
            "eye_center_camera_xy": [float(projected.x), float(projected.y)],
            "target_y_fraction_from_bottom": target_y,
            "camera_reframed": True}


def render_mask(scene, head, out: Path):
    idx = next(i for i, slot in enumerate(head.material_slots)
               if slot.material and slot.material.name == "Sclera_Cornea")
    original = head.material_slots[idx].material
    transparent = original.copy()
    transparent.name = "PROBE_TRANSPARENT_Sclera_Cornea"
    transparent.use_nodes = True
    nodes = transparent.node_tree.nodes
    nodes.clear()
    shader = nodes.new("ShaderNodeBsdfTransparent")
    output = nodes.new("ShaderNodeOutputMaterial")
    transparent.node_tree.links.new(shader.outputs[0], output.inputs["Surface"])
    head.material_slots[idx].material = transparent
    scene.view_layers[0].use_pass_material_index = True
    scene.use_nodes = True
    scene.node_tree.nodes.clear()
    layer = scene.node_tree.nodes.new("CompositorNodeRLayers")
    composite = scene.node_tree.nodes.new("CompositorNodeComposite")
    scene.node_tree.links.new(layer.outputs["IndexMA"], composite.inputs["Image"])
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_mode = "BW"
    scene.render.image_settings.color_depth = "32"
    scene.render.filepath = str(out / "labels")
    scene.cycles.samples = 1
    bpy.ops.render.render(write_still=True)
    exr = out / "labels.exr"
    if not exr.exists():
        exr = next(iter(sorted(out.glob("labels*.exr"))), None)
    if exr is None:
        raise RuntimeError("missing material-index EXR")
    image = bpy.data.images.load(str(exr), check_existing=False)
    h, w = map(int, image.size)
    pixels = np.asarray(image.pixels[:], dtype=np.float32)
    c = int(image.channels)
    array = pixels.reshape(h, w, c)[..., 0] if c > 1 else pixels.reshape(h, w)
    labels = np.rint(array).astype(np.uint8)[::-1].copy()
    np.save(out / "labels.npy", labels)
    stats = {}
    for value, name in ((1, "iris"), (3, "pupil"), (4, "sclera_cornea")):
        ys, xs = np.where(labels == value)
        stats[name] = {"pixels": int(len(xs)),
                       "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
                       if len(xs) else None}
    preview = np.zeros((h, w, 3), dtype=np.uint8)
    preview[labels == 1] = (255, 145, 25)
    preview[labels == 3] = (45, 255, 80)
    preview[labels == 4] = (40, 125, 255)
    write_png(out / "labels_preview.png", preview)
    head.material_slots[idx].material = original
    scene.use_nodes = False
    return stats


def main():
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=815900)
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--samples", type=int, default=3)
    opts = parser.parse_args(args)
    root = Path(opts.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    head = bpy.data.objects.get("cm_vitruvian")
    if head is None or not head.data.shape_keys:
        raise RuntimeError("expected shape-keyed cm_vitruvian mesh")
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = opts.resolution
    scene.render.resolution_y = opts.resolution
    scene.render.resolution_percentage = 100
    scene.cycles.samples = opts.samples
    scene.cycles.use_denoising = True
    scene.render.image_settings.file_format = "PNG"
    if opts.count < 1:
        raise ValueError("count must be positive")
    cases = sample_cases(opts.count, opts.seed)
    gaze_snapshot = make_gaze_snapshot(head)
    face_keys = head.data.shape_keys.key_blocks
    base_face_values = {name: face_keys[name].value for name in (
        "L2__Gender_Female", "L2__Gender_Male", "L2__Age_Baby", "L2__Age_Old")
        if name in face_keys}
    records = []
    for i, case in enumerate(cases):
        out = root / case["name"]
        out.mkdir(parents=True, exist_ok=True)
        for name, value in base_face_values.items():
            face_keys[name].value = value
        face_controls = apply_face_preset(head, case["face_preset"], base_face_values)
        controls = set_morphs(head, case)
        apply_gaze_offset(head, gaze_snapshot, case["gaze_x_m"], case["gaze_z_m"])
        framing = frame_camera_on_eyes(scene, head)
        scene.use_nodes = False
        # render_mask() intentionally uses one sample; restore the requested
        # beauty quality on every loop iteration to prevent setting leakage.
        scene.cycles.samples = opts.samples
        scene.cycles.use_denoising = True
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGB"
        scene.render.filepath = str(out / "beauty")
        bpy.ops.render.render(write_still=True)
        stats = render_mask(scene, head, out)
        record = {"seed": opts.seed + i, "case": case, "face_controls": face_controls,
                  "camera_framing": framing,
                  "controls": controls,
                  "labels": stats, "beauty": "beauty.png",
                  "label_preview": "labels_preview.png"}
        (out / "metadata.json").write_text(json.dumps(record, indent=2) + "\n")
        records.append(record)
        print(json.dumps({"case": case["name"], "labels": stats}), flush=True)
    report = {
        "status": "PASS" if all(x["labels"]["iris"]["pixels"] > 100
                                   and x["labels"]["pupil"]["pixels"] > 20
                                   for x in records) else "FAIL",
        "n_synthetic_morph_cases": len(records),
        "sampling_seed": opts.seed,
        "render_resolution": [opts.resolution, opts.resolution],
        "cycles_samples": opts.samples,
        "single_base_mesh": "CharMorph Vitruvian CC0",
        "independent_identity_count": 1,
        "cases": records,
        "limitations": ["synthetic-only", "one base identity", "asset face morphs do not establish demographic representation",
                        "controlled eye morphology and iris/pupil translation only",
                        "gaze translation is not physical eyeball rotation", "no real camera calibration",
                        "no real eye-tracker footage", "not demographic validation"],
    }
    (root / "variant_label_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "n_cases": len(records),
                      "pupil_pixels": [x["labels"]["pupil"]["pixels"] for x in records]}), flush=True)


if __name__ == "__main__":
    main()
