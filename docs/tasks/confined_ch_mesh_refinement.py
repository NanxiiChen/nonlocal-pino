"""Independent two-geometry native spatial check, before any GPU work."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.tri as mtri
import numpy as np
from ml_collections import ConfigDict
from scipy.spatial import Delaunay

from examples.confined_ch2d.geometry import quadrature, visible_pairs
from examples.confined_ch2d.native_reference import native_matrix, native
from examples.confined_ch2d.operator_problem import case_config, fixed_initial
from examples.confined_ch2d.problem import relative


def mapping(data, queries):
    tri = Delaunay(data["points"])
    faces = tri.simplices
    edges = np.stack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=1)
    visible = visible_pairs(data["points"], edges.reshape(-1, 2), data["holes"])
    good = np.all(visible.reshape(-1, 3), axis=1)
    owner = tri.find_simplex(data["holes"][:, :2])
    good[owner[owner >= 0]] = False
    owner = tri.find_simplex(queries)
    safe = np.maximum(owner, 0)
    valid = (owner >= 0) & good[safe]
    for cx, cy, radius in data["holes"]:
        valid &= np.sum((queries - [cx, cy])**2, axis=1) > radius**2
    first = np.einsum("nij,nj->ni", tri.transform[safe, :2], queries - tri.transform[safe, 2])
    weights = np.column_stack((first, 1 - first.sum(axis=1)))
    vertices = faces[safe]
    np.testing.assert_allclose(weights[valid].sum(axis=1), 1., atol=1e-13, rtol=0)
    np.testing.assert_allclose(np.einsum("nkd,nk->nd", data["points"][vertices[valid]], weights[valid]),
                               queries[valid], atol=1e-12, rtol=0)
    if np.min(weights[valid]) < -1e-12:
        raise ValueError("Nonpositive interior barycentric reconstruction")
    return valid, vertices, weights


def case(request):
    dataset, output, index, h = request
    dataset, output = Path(dataset), Path(output)
    original = ConfigDict(json.loads((dataset / "config.json").read_text()))
    manifest = json.loads((dataset / "manifest.json").read_text())
    if index not in manifest["test_indices"] or manifest["status"] != "passed":
        raise ValueError("Require an existing validated reserved geometry")
    record = manifest["conditions"][index]
    keys = ("points", "volumes", "phi", "mu", "time", "holes", "mesh_points", "mesh_cells")
    with np.load(dataset / record["file"], allow_pickle=False) as f:
        old = {key: f[key] for key in keys}
    c = ConfigDict(original.to_dict())
    c.geometry.mesh_size = h
    c.operator.dataset_path = str(output)
    cc = case_config(c, old["holes"], index)
    target = output / f"case_{index:03d}"
    target.mkdir(exist_ok=False)
    started = time.perf_counter()
    request = target / "mesh_request.json"
    request.write_text(cc.to_json(indent=2)+"\n")
    env = dict(os.environ)
    env.pop("VIRTUAL_ENV", None)
    # Same explicit meshing interpreter as operator_precompute.build_one.
    subprocess.run([c.operator.reference_python, "-c",
        "import sys,json,numpy as np; from pathlib import Path; from ml_collections import ConfigDict; "
        "from examples.confined_ch2d.geometry import create_mesh; "
        "c=ConfigDict(json.loads(Path(sys.argv[1]).read_text())); "
        "v,t=create_mesh(c,c.geometry.mesh_size,Path(sys.argv[2])); "
        "np.savez(sys.argv[3],vertices=v,cells=t)",
        str(request), str(target / "mesh.xdmf"), str(target / "mesh.npz")],
        check=True, cwd=Path(__file__).resolve().parents[2], env=env)
    with np.load(target / "mesh.npz", allow_pickle=False) as mesh:
        vertices, cells = mesh["vertices"], mesh["cells"]
    points, volumes = quadrature(vertices, cells)
    A = native_matrix(points, volumes, cc)
    np.testing.assert_allclose(A @ np.ones(len(points)), 0., atol=1e-10, rtol=0)
    coarse = native(cc, points, volumes, A, c.reference.dt, 1e-7)
    fine = native(cc, points, volumes, A, c.reference.refined_dt, 1e-9)
    time_errors = {key: max(relative(a, b, volumes) for a, b in zip(coarse[key], fine[key], strict=True))
                   for key in ("phi", "mu")}
    if time_errors["phi"] >= c.reference.phase_tolerance or time_errors["mu"] >= c.reference.chemical_tolerance:
        raise ValueError(f"Time refinement failed: {index}: {time_errors}")
    mass_drift = float(np.max(np.abs(fine["mass"] - fine["mass"][0])))
    energy_increase = float(max(0., np.max(np.diff(fine["energy"]))))
    if mass_drift > 1e-9 or energy_increase > 1e-9:
        raise ValueError("Native mass/energy gates failed")
    new = dict(points=points, volumes=volumes, holes=old["holes"], mesh_points=vertices, mesh_cells=cells,
               time=old["time"], phi=fine["phi"], mu=fine["mu"])
    np.testing.assert_allclose(new["phi"][0], fixed_initial(points, cc, xp=np), rtol=0, atol=1e-14)
    np.savez_compressed(target / "reference.npz", **new, mass=fine["mass"], energy=fine["energy"])
    (target / "config.json").write_text(cc.to_json(indent=2) + "\n")

    x, y = np.meshgrid(np.linspace(.005, .995, 200), np.linspace(.105, .895, 160), indexing="xy")
    queries = np.column_stack((x.ravel(), y.ravel()))
    physical = np.sum(((queries - np.asarray(c.geometry.center)) / np.asarray(c.geometry.axes))**2, axis=1) < 1
    for cx, cy, radius in old["holes"]:
        physical &= np.sum((queries - [cx, cy])**2, axis=1) > radius**2
    mappings = [mapping(d, queries) for d in (old, new)]
    p1_keep = physical & mappings[0][0] & mappings[1][0]
    owners = [mtri.Triangulation(d["mesh_points"][:, 0], d["mesh_points"][:, 1], d["mesh_cells"]).get_trifinder()(*queries.T)
              for d in (old, new)]
    p0_keep = physical & (owners[0] >= 0) & (owners[1] >= 0)
    if min(p1_keep.sum(), p0_keep.sum()) < 100:
        raise ValueError("Insufficient common physical support")
    def statistics(a, b):
        return dict(pooled=relative(a, b, np.ones(a.shape[-1])), terminal=relative(a[-1], b[-1], np.ones(a.shape[-1])),
                    by_time=[relative(u, v, np.ones(len(u))) for u, v in zip(a, b, strict=True)])
    interpolants = [np.einsum("tnk,nk->tn", d["phi"][:, vertices[p1_keep]], weights[p1_keep])
                    for d, (_, vertices, weights) in zip((old, new), mappings, strict=True)]
    p0 = [d["phi"][:, owner[p0_keep]] for d, owner in zip((old, new), owners, strict=True)]
    exact_initial = fixed_initial(queries[p1_keep], cc, xp=np)
    initial_errors = [relative(phi[0], exact_initial, np.ones(len(exact_initial))) for phi in interpolants]
    result = dict(index=index, old_mesh_size=original.geometry.mesh_size, new_mesh_size=h,
                  old_points=len(old["points"]), new_points=len(points), time_refinement=time_errors,
                  native_mass_drift=mass_drift, native_energy_increase=energy_increase,
                  common_p1_queries=int(p1_keep.sum()), common_p0_queries=int(p0_keep.sum()),
                  physical_queries=int(physical.sum()), p1_phase_difference=statistics(*interpolants),
                  p0_phase_difference=statistics(*p0), p1_initial_reconstruction_relative_error=initial_errors,
                  reference=str(target / "reference.npz"), seconds=time.perf_counter() - started,
                  scope="Uniform common-query L2; coarse versus fine, normalized by fine. P0/P1 reconstruction included; no IC error subtraction. Not local FEM/continuum error or a pure operator-error estimate.")
    (target / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result), flush=True)
    return result


def main(dataset, output, indices, h):
    source = ConfigDict(json.loads((dataset / "config.json").read_text()))
    if not 0 < h < source.geometry.mesh_size or len(set(indices)) != len(indices):
        raise ValueError("Require a finer positive h and distinct explicit indices")
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    requests = [(str(dataset), str(output), index, h) for index in indices]
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
        rows = list(pool.map(case, requests))
    result = dict(status="passed", dataset=str(dataset), indices=indices, conditions=rows,
                  wall_seconds=time.perf_counter()-start, fixed_delta=source.nl.delta,
                  fixed_epsilon=source.physical.epsilon,
                  scope="Two existing development geometries selected diagnostically; independent native graph/BDF spatial check, no FEM/JAX or new training labels.")
    (output / "validation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--indices", type=int, nargs="+", required=True)
    parser.add_argument("--mesh-size", type=float, required=True)
    a = parser.parse_args()
    main(a.dataset.resolve(), a.output.resolve(), a.indices, a.mesh_size)
