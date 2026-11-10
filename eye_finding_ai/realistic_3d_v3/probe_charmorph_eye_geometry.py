"""Inspect connected Iris/Pupil/Sclera mesh regions in the CC0 CharMorph scene.

This is a geometry feasibility probe. It does not modify the scene or generate
training data. Run with Blender 4.5 against the saved CharMorph .blend file.
"""
from __future__ import annotations

import bpy
import json
import sys
from collections import defaultdict
from pathlib import Path


def components_for_material(mesh, material_index: int):
    polygons = [p for p in mesh.polygons if p.material_index == material_index]
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for poly in polygons:
        verts = list(poly.vertices)
        for v in verts[1:]:
            union(verts[0], v)
    groups = defaultdict(set)
    for v in parent:
        groups[find(v)].add(v)
    result = []
    for vertices in sorted(groups.values(), key=len, reverse=True):
        points = [mesh.vertices[v].co[:] for v in vertices]
        bounds = [[float(min(p[i] for p in points)), float(max(p[i] for p in points))]
                  for i in range(3)]
        center = [sum(p[i] for p in points) / len(points) for i in range(3)]
        result.append({"vertex_count": len(vertices), "bounds_xyz": bounds,
                       "centroid_xyz": center})
    return {"polygon_count": len(polygons), "components": result}


def main():
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if len(args) != 1:
        raise SystemExit("usage: probe_charmorph_eye_geometry.py -- output.json")
    head = bpy.data.objects.get("cm_vitruvian")
    if head is None or head.type != "MESH":
        raise RuntimeError("expected mesh object cm_vitruvian")
    target = {"Iris", "Pupil", "Sclera_Cornea"}
    slots = {slot.material.name: i for i, slot in enumerate(head.material_slots)
             if slot.material and slot.material.name in target}
    report = {
        "source": "CC0 CharMorph Vitruvian scene; geometry inspection only",
        "object": head.name,
        "regions": {name: components_for_material(head.data, index)
                    for name, index in slots.items()},
        "limitations": ["one base mesh", "not a human dataset", "no gaze labels"]
    }
    path = Path(args[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
