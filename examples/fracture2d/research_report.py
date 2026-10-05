"""Independently recompute saved research fields and render diagnostic figures."""

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from examples.fracture2d.fp64 import FP64A2Backend
from examples.fracture2d.inner_probe import load_snapshot
from examples.fracture2d.losses import FractureResidual


def run(args):
    config, model, state = load_snapshot(args.base, args.parent_digest)
    with np.load(args.fields, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    if args.fields.name == "fields_0036.npz":
        for actual, bound in zip((data["q"], data["d"], data["H"]), state[4:]):
            np.testing.assert_array_equal(actual, np.asarray(bound))
    backend = FP64A2Backend(model.static, model.constraints, jax.devices()[0])
    residual = FractureResidual(model, backend)
    factor = backend._edge_factor(jnp.asarray(data["d"]))
    r = np.asarray(backend._residual_from_factor(factor, jnp.asarray(data["q"])))
    rho = np.asarray(residual.gc_over_l0 * data["d"]
        + residual.gc_l0 * residual.operator.deterministic_matvec(jnp.asarray(data["d"]))
        + 2 * data["H"] * (data["d"] - 1))
    xy = np.asarray(model.geometry.coordinates)
    free = np.asarray(backend.free_dofs)
    hotspots = free[np.argsort(np.abs(r[free]))[-10:][::-1]]
    damage_hotspots = np.argsort(np.abs(rho))[-10:][::-1]
    report = {"fields_sha256": hashlib.sha256(args.fields.read_bytes()).hexdigest(),
              "mechanics_inf": float(np.max(np.abs(r[free]))),
              "at2_inf": float(np.max(np.abs(rho))),
              "reaction": float(r @ np.asarray(backend.outer_top_selector)),
              "damage_hotspots": [{"node": int(i), "coordinate": xy[i].tolist(),
                                    "residual": float(rho[i])} for i in damage_hotspots],
              "mechanics_hotspots": [{"dof": int(i), "coordinate": xy[i // 2].tolist(),
                                       "residual": float(r[i])} for i in hotspots]}
    print(json.dumps(report, indent=2))
    root = args.output
    if not root.is_dir():
        raise FileNotFoundError(root)
    (root / "independent_field_audit.json").write_text(json.dumps(report, indent=2))
    n = model.n_material
    q = data["q"][:2 * n].reshape(n, 2)
    d = data["d"]
    with np.load(config.data.reference, allow_pickle=False) as archive:
        triangles = archive["triangles"]
        load_index = int(np.argmin(np.abs(archive["loads"] - args.load)))
        reference_available = (args.comparison == "archive_warning"
            and abs(float(archive["loads"][load_index]) - args.load) <= 1e-12)
        q_ref = archive["union_q"][load_index][:2*n].reshape(n, 2)
        d_ref = archive["damage"][load_index]
    triangulation = mtri.Triangulation(xy[:n, 0], xy[:n, 1], triangles)
    for suffix, zoom in (("", None), ("_notch", (0.38, 0.72, 0.34, 0.66))):
        fig, axes = plt.subplots(3 if reference_available else 1, 3,
            figsize=(14, 12 if reference_available else 4), constrained_layout=True, squeeze=False)
        predicted = (q[:, 0], q[:, 1], d)
        reference = (q_ref[:, 0], q_ref[:, 1], d_ref)
        for col, label in enumerate(("ux", "uy", "damage")):
            low = min(predicted[col].min(), reference[col].min()) if reference_available else predicted[col].min()
            high = max(predicted[col].max(), reference[col].max()) if reference_available else predicted[col].max()
            panels = (predicted[col], reference[col], predicted[col] - reference[col]) if reference_available else (predicted[col],)
            for row, values in enumerate(panels):
                if row == 2:
                    high_error = float(np.max(np.abs(values)))
                    vmin, vmax, cmap = -high_error, high_error, "RdBu_r"
                else:
                    vmin, vmax, cmap = low, high, "viridis"
                artist = axes[row, col].tripcolor(triangulation, values, shading="gouraud",
                    vmin=vmin, vmax=vmax, cmap=cmap)
                fig.colorbar(artist, ax=axes[row, col])
                axes[row, col].set_aspect("equal")
                axes[row, col].set_title(("Research PINN", "Warning-tail comparator", "Difference")[row] + ": " + label)
                if zoom:
                    axes[row, col].set_xlim(zoom[:2])
                    axes[row, col].set_ylim(zoom[2:])
        fig.suptitle(f"load {args.load:.4f}; diagnostic fields, no strict acceptance claim")
        fig.savefig(root / f"physical_fields{suffix}.png", dpi=140)
        plt.close(fig)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--fields", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent-digest", required=True)
    parser.add_argument("--load", type=float, required=True)
    parser.add_argument("--comparison", choices=("pinn_only", "archive_warning"), required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
