"""Check common reference queries on the actual coarse/fine CAD meshes.

CPU geometry construction only: no DOLFIN import, PDE solve or JAX GPU work.
The separate Gmsh interpreter avoids mixing its libraries into the JAX process.
"""
import argparse
import importlib
import json
from pathlib import Path
import subprocess
import time

import matplotlib
matplotlib.use("Agg")
from matplotlib.tri import Triangulation
import numpy as np


def run(module, output):
    config = importlib.import_module(module).get_config()
    from examples.confined_ch2d.operator_problem import case_config, specification

    native = Path(config.operator.dataset_path)
    bank = json.loads((native / "manifest.json").read_text())
    if bank["status"] != "passed" or bank["specification"] != specification(config):
        raise ValueError("Reference mesh check needs the matching qualified native inputs")
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for index in bank["test_indices"]:
        original = bank["conditions"][index]
        with np.load(native / original["file"]) as data:
            points = data["points"]
            holes = data["holes"]
        np.testing.assert_array_equal(holes, original["holes"])
        case = case_config(config, holes, index)
        for resolution, size in enumerate(config.reference.mesh_sizes):
            stem = output / f"case_{index:04d}_resolution_{resolution}"
            request = stem.with_suffix(".json")
            arrays = stem.with_suffix(".npz")
            request.write_text(json.dumps(dict(
                config=case.to_dict(), h=float(size), path=str(stem),
                output_arrays=str(arrays)), indent=2) + "\n")
            started = time.perf_counter()
            subprocess.run([
                config.operator.reference_python, "-m",
                "examples.confined_ch2d.geometry", "--input", str(request)
            ], cwd=Path(__file__).resolve().parents[2], check=True,
                capture_output=True, text=True)
            with np.load(arrays) as mesh:
                vertices, cells = mesh["vertices"], mesh["cells"]
            triangulation = Triangulation(vertices[:, 0], vertices[:, 1], cells)
            located = triangulation.get_trifinder()(points[:, 0], points[:, 1])
            if np.any(located < 0):
                missing = np.flatnonzero(located < 0).tolist()
                raise ValueError(f"Geometry {index}, h={size}: queries outside mesh: {missing}")
            triangles = vertices[cells[located]]
            basis = np.stack((triangles[:, 1] - triangles[:, 0],
                              triangles[:, 2] - triangles[:, 0]), axis=-1)
            coordinates = np.linalg.solve(basis, (points - triangles[:, 0])[..., None])[..., 0]
            barycentric = np.column_stack((1 - coordinates.sum(axis=1), coordinates))
            if not np.isfinite(barycentric).all() or barycentric.min() < -1e-11:
                raise ValueError("Independent barycentric check failed")
            reconstructed = np.einsum("ni,nij->nj", barycentric, triangles)
            np.testing.assert_allclose(reconstructed, points, atol=2e-14, rtol=0)
            rows.append(dict(
                index=index, mesh_size=float(size), vertices=len(vertices),
                triangles=len(cells), queries=len(points), outside=0,
                minimum_barycentric=float(barycentric.min()),
                maximum_reconstruction_error=float(np.max(np.abs(reconstructed - points))),
                seconds=time.perf_counter() - started,
                native_input=str(native / original["file"]), request=str(request),
                mesh_arrays=str(arrays)))
            print(json.dumps(rows[-1]), flush=True)
    report = dict(status="passed", configuration=module, cases=rows,
                  native_manifest=str(native / "manifest.json"),
                  pde_solves=0, gpu_work=False,
                  scope="Physical query containment only; not reference convergence or accuracy")
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.output.resolve())
