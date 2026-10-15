"""Generate explicitly tagged ASCII MSH2 reference clouds without meshio."""

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

import numpy as np

from examples.fracture2d._legacy import nonlocal_baseline as direct


def parse_msh2(path):
    sections = {}
    with Path(path).open(encoding="utf-8") as stream:
        lines = iter(stream)
        for line in lines:
            name = line.strip()
            if name.startswith("$") and not name.startswith("$End"):
                values = []
                for item in lines:
                    if item.strip() == "$End" + name[1:]:
                        break
                    values.append(item.strip())
                sections[name[1:]] = values
    if sections["MeshFormat"] != ["2.2 0 8"]:
        raise ValueError("reference cloud requires explicit ASCII MSH 2.2 doubles")
    names = {}
    for line in sections["PhysicalNames"][1:]:
        dim, tag, name = shlex.split(line)
        names[int(dim), int(tag)] = name
    required = {(2, 10): "domain", (0, 9): "anchor", (1, 1): "bottom", (1, 3): "top",
                (1, 2): "right", (1, 4): "left_upper", (1, 5): "notch_upper",
                (1, 6): "notch_tip", (1, 7): "notch_lower", (1, 8): "left_lower"}
    if names != required:
        raise ValueError(f"reference mesh physical groups differ: {names}")
    nodes = {}
    for line in sections["Nodes"][1:]:
        tag, x, y, z = line.split()
        if float(z) != 0:
            raise ValueError("reference mesh must be exactly planar")
        if int(tag) in nodes:
            raise ValueError("duplicate mesh node tag")
        nodes[int(tag)] = [float(x), float(y)]
    triangles, boundaries, anchors = [], {key: [] for key in range(1, 9)}, []
    for line in sections["Elements"][1:]:
        entries = [int(value) for value in line.split()]
        _, kind, n_tags = entries[:3]
        if n_tags < 1:
            raise ValueError("reference mesh elements require physical tags")
        physical = entries[3]
        vertices = entries[3 + n_tags:]
        if kind == 2 and physical == 10 and len(vertices) == 3:
            triangles.append(vertices)
        elif kind == 1 and physical in boundaries and len(vertices) == 2:
            boundaries[physical].append(vertices)
        elif kind == 15 and physical == 9 and len(vertices) == 1:
            anchors.extend(vertices)
        else:
            raise ValueError(f"unsupported/unowned reference mesh element: {line}")
    if len(anchors) != 1 or not triangles or not all(boundaries.values()):
        raise ValueError("reference mesh must contain material, all boundaries and one anchor")
    used = sorted(set(np.asarray(triangles).reshape(-1)))
    coords = np.asarray([nodes[tag] for tag in used], np.float64)
    order = np.lexsort((coords[:, 1], coords[:, 0]))
    coords = coords[order]
    tags = np.asarray(used)[order]
    mapping = {int(tag): index for index, tag in enumerate(tags)}
    if mapping[anchors[0]] != 0 or not np.array_equal(coords[0], [0, 0]):
        raise ValueError("tagged material anchor is not the ordered physical origin")
    tri = np.asarray([[mapping[tag] for tag in row] for row in triangles], np.int64)
    masks = []
    for tag in (1, 3):
        mask = np.zeros(len(coords), bool)
        for node in np.asarray(boundaries[tag]).reshape(-1):
            mask[mapping[int(node)]] = True
        masks.append(mask)
    _, volumes, _ = direct._g_triangle_quadrature(coords, tri, "reference cloud triangles")
    if min(volumes) <= 0 or len(np.unique(coords, axis=0)) != len(coords):
        raise ValueError("reference cloud has invalid material volume/coordinates")
    tip = np.flatnonzero(np.all(coords == [0.5, 0.5], axis=1))
    if len(tip) != 1:
        raise ValueError("reference mesh must retain the exact rounded tip point")
    return {"coordinates": coords, "triangles": tri, "volumes": volumes,
            "bottom": masks[0], "top": masks[1], "anchor_particle": np.asarray(0, np.int64)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--h-far", type=float, required=True)
    parser.add_argument("--h-near", type=float, required=True)
    parser.add_argument("--gmsh-python", type=Path, required=True)
    args = parser.parse_args()
    if not np.all(np.isfinite([args.h_far, args.h_near])) or not 0 < args.h_near <= args.h_far:
        raise ValueError("reference cloud mesh sizes must be explicitly positive and ordered")
    if not args.output.parent.is_dir() or not args.log.parent.is_dir() or not args.geo.is_file():
        raise FileNotFoundError("reference cloud input or output/log parent is missing")
    mesh = args.output.with_suffix(".msh")
    if args.output.exists() or mesh.exists() or args.log.exists():
        raise FileExistsError("refusing to overwrite a reference cloud/mesh/log")
    if not args.gmsh_python.is_file():
        raise FileNotFoundError(args.gmsh_python)
    command = [str(args.gmsh_python), str(Path(__file__).with_name("gmsh_reference_mesh.py")),
               "--geo", str(args.geo), "--mesh", str(mesh),
               "--h-far", repr(args.h_far), "--h-near", repr(args.h_near)]
    with args.log.open("x") as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
    cloud = parse_msh2(mesh)
    for name in ("bottom", "top"):
        values = np.sort(cloud["coordinates"][cloud[name], 0])
        if max(np.diff(values)) > args.h_far * 1.01:
            raise ValueError("Gmsh did not apply the declared reference-cloud size override")
    metadata = {"schema": "fracture2d-reference-cloud-v1", "gmsh_version": subprocess.check_output(["gmsh", "-version"], text=True, stderr=subprocess.STDOUT).strip(),
                "geo_sha256": hashlib.sha256(args.geo.read_bytes()).hexdigest(),
                "mesh_sha256": hashlib.sha256(mesh.read_bytes()).hexdigest(),
                "h_far": args.h_far, "h_near": args.h_near, "material_nodes": len(cloud["coordinates"]),
                "mesh_algorithm": 6, "element_order": 1, "mesh_binary": False, "mesh_threads": 1,
                "material_area": float(sum(cloud["volumes"])), "input_geometry": str(args.geo),
                "command": command}
    with args.output.open("xb") as stream:
        np.savez_compressed(stream, **cloud, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
    metadata["cloud_sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(".manifest.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
