"""Matrix-free intact/damaged tangent audit, with independently checked Ritz defects."""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

from examples.fracture2d._legacy.fp64 import FP64A2Backend
from examples.fracture2d._legacy.inner_probe import load_snapshot


def run(args):
    _, model, _ = load_snapshot(args.base, args.parent_digest)
    field_path = args.fields if args.fields is not None else args.trial / "fields.npz"
    with np.load(field_path, allow_pickle=False) as archive:
        damage = archive["d"]
    if (damage.shape != (model.n_material,) or damage.dtype != np.float64
            or not np.all(np.isfinite(damage)) or np.min(damage) < 0 or np.max(damage) > 1):
        raise ValueError("spectrum requires a finite bounded FP64 material damage field")
    backend = FP64A2Backend(model.static, model.constraints, jax.devices()[0])
    free = np.asarray(backend.free_dofs)
    if not args.output.parent.is_dir():
        raise FileNotFoundError(args.output.parent)
    args.output.mkdir()
    results = {}
    operators = (("intact", np.zeros_like(damage)), ("damaged", damage))
    if args.operators == "damaged":
        operators = operators[1:]
    for name, d in operators:
        factor = backend._edge_factor(jnp.asarray(d))
        action = jax.jit(lambda direction: backend._tangent_from_factor(factor, direction))
        volumes = np.repeat(np.asarray(model.geometry.union_volumes), 2)[free]
        if np.any(volumes <= 0) or not np.all(np.isfinite(volumes)):
            raise ValueError("mechanics displacement norm requires positive union volumes")
        inverse_root = 1 / np.sqrt(volumes) if args.metric == "volume" else np.ones(len(free))
        def matvec(vector):
            direction = np.zeros(model.n_dofs, np.float64)
            direction[free] = inverse_root * np.asarray(vector).reshape(-1)
            return inverse_root * np.asarray(action(jnp.asarray(direction)))[free]
        A = LinearOperator((len(free), len(free)), matvec=matvec, dtype=np.float64)
        largest, _ = eigsh(A, k=1, which="LA", tol=1e-6, maxiter=args.maxiter, ncv=40)
        values, vectors = eigsh(A, k=3, which="SA", tol=1e-6, maxiter=args.maxiter, ncv=80)
        defects = []
        modes = []
        for value, vector in zip(values, vectors.T):
            applied = matvec(vector)
            relative = float(np.linalg.norm(applied - value * vector) / np.linalg.norm(applied))
            if not np.isfinite(relative) or relative > 1e-4:
                raise ValueError(f"{name} Ritz eigenpair is unconverged: {relative}")
            defects.append(relative)
            full = np.zeros(model.n_dofs)
            full[free] = inverse_root * vector
            nodal = np.linalg.norm(full.reshape(-1, 2), axis=1)
            fractions = (np.asarray(model.geometry.union_volumes) * nodal**2
                         if args.metric == "volume" else nodal**2)
            peak = int(np.argmax(nodal))
            modes.append({"peak_node": peak,
                          "peak_coordinate": np.asarray(model.geometry.coordinates)[peak].tolist(),
                          "nodal_participation_count": float(1 / np.sum(fractions**2)),
                          "peak_nodal_amplitude": float(nodal[peak])})
        results[name] = {"minimum_eigenvalues": values.tolist(), "maximum_eigenvalue": float(largest[0]),
                         "condition_estimate": float(largest[0] / values[0]),
                         "ritz_relative_defects": defects, "modes": modes,
                         "metric": args.metric,
                         "field_path": str(field_path), "direct_model_digest": model.model_digest,
                         "diagnostic_only": True, "training_targets": False}
        (args.output / "spectrum.json").write_text(json.dumps(results, indent=2))
        np.savez_compressed(args.output / f"{name}_modes.npz", values=values,
                            vectors=inverse_root[:, None] * vectors, free_dofs=free)
        print(json.dumps({"operator": name, **results[name]}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent-digest", required=True)
    parser.add_argument("--maxiter", type=int, required=True)
    parser.add_argument("--metric", choices=("euclidean", "volume"), required=True)
    parser.add_argument("--fields", type=Path)
    parser.add_argument("--operators", choices=("both", "damaged"), required=True)
    args = parser.parse_args()
    if args.maxiter < 1:
        raise ValueError("maxiter must be positive")
    run(args)


if __name__ == "__main__":
    main()
