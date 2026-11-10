"""Probe whether the CC0 face scene can yield visible iris/pupil masks.

The scene is synthetic feasibility evidence only. The beauty render is saved
before changing the cornea material; the label pass then temporarily makes the
cornea transparent so the renderer can expose the underlying eye materials.
"""
from __future__ import annotations

import bpy
import json
import struct
import sys
import zlib
from pathlib import Path

import numpy as np


def write_png(path: Path, rgb: np.ndarray) -> None:
    h, w, _ = rgb.shape

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff))

    raw = b"".join(b"\x00" + rgb[y].astype(np.uint8, copy=False).tobytes()
                    for y in range(h))
    payload = (b"\x89PNG\r\n\x1a\n"
               + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
               + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
    path.write_bytes(payload)


def main() -> None:
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if len(args) != 1:
        raise SystemExit("usage: probe_transparent_eye_labels.py -- output_dir")
    out = Path(args[0])
    out.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 12
    scene.cycles.use_denoising = True
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.use_nodes = False
    scene.render.filepath = str(out / "beauty")
    bpy.ops.render.render(write_still=True)
    beauty = out / "beauty.png"

    head = bpy.data.objects.get("cm_vitruvian")
    if head is None:
        raise RuntimeError("saved scene has no cm_vitruvian mesh")
    idx_by_name = {slot.material.name: i for i, slot in enumerate(head.material_slots)
                   if slot.material}
    needed = {"Iris", "Pupil", "Sclera_Cornea"}
    if not needed.issubset(idx_by_name):
        raise RuntimeError(f"missing expected eye materials: {sorted(needed-set(idx_by_name))}")

    # Preserve the original material datablock and restore it before exit.
    sclera_index = idx_by_name["Sclera_Cornea"]
    original = head.material_slots[sclera_index].material
    transparent = original.copy()
    transparent.name = "PROBE_TRANSPARENT_Sclera_Cornea"
    transparent.use_nodes = True
    nodes = transparent.node_tree.nodes
    nodes.clear()
    trans = nodes.new("ShaderNodeBsdfTransparent")
    output = nodes.new("ShaderNodeOutputMaterial")
    transparent.node_tree.links.new(trans.outputs[0], output.inputs["Surface"])
    head.material_slots[sclera_index].material = transparent

    scene.view_layers[0].use_pass_material_index = True
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()
    layer = tree.nodes.new("CompositorNodeRLayers")
    composite = tree.nodes.new("CompositorNodeComposite")
    if "IndexMA" not in layer.outputs:
        raise RuntimeError("Blender compositor has no IndexMA output")
    tree.links.new(layer.outputs["IndexMA"], composite.inputs["Image"])
    scene.render.filepath = str(out / "eye_labels")
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_mode = "BW"
    scene.render.image_settings.color_depth = "32"
    bpy.ops.render.render(write_still=True)
    exr = out / "eye_labels.exr"
    if not exr.exists():
        exr = next(iter(sorted(out.glob("eye_labels*.exr"))), None)
    if exr is None:
        raise RuntimeError("label EXR not created")
    image = bpy.data.images.load(str(exr), check_existing=False)
    h, w = map(int, image.size)
    pixels = np.asarray(image.pixels[:], dtype=np.float32)
    channels = int(image.channels)
    values = pixels.reshape(h, w, channels)[..., 0] if channels > 1 else pixels.reshape(h, w)
    labels = np.rint(values).astype(np.uint8)[::-1].copy()
    np.save(out / "eye_labels.npy", labels)
    unique, counts = np.unique(labels, return_counts=True)
    mapping = {"1": "Iris", "3": "Pupil", "4": "Sclera_Cornea"}
    palette = {1: (255, 150, 30), 3: (50, 255, 90), 4: (45, 140, 255)}
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    for value, color in palette.items():
        overlay[labels == value] = color
    write_png(out / "eye_labels_preview.png", overlay)
    stats = {}
    for value, count in zip(unique, counts):
        v = int(value)
        if str(v) not in mapping:
            continue
        ys, xs = np.where(labels == v)
        stats[mapping[str(v)]] = {
            "pass_index": v, "visible_pixel_count": int(count),
            "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            if len(xs) else None,
        }
    report = {
        "status": "PASS" if stats.get("Iris", {}).get("visible_pixel_count", 0) > 0
                          and stats.get("Pupil", {}).get("visible_pixel_count", 0) > 0
                  else "PARTIAL_OR_FAIL",
        "render_size": [w, h], "visible_eye_materials": stats,
        "source_scene": "CharMorph Vitruvian CC0 synthetic scene",
        "beauty_render": beauty.name,
        "label_render": exr.name,
        "caveat": "Single synthetic frame; visibility labels are not yet validated across morphs, gaze, pose, or camera conditions.",
    }
    (out / "transparent_eye_label_probe.json").write_text(json.dumps(report, indent=2) + "\n")
    head.material_slots[sclera_index].material = original
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
