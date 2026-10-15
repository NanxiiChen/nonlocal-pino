"""Plot identity-verified strict PINN displacements against the direct archive.

Use the accepted network's committed material-node displacement, not a newly
evaluated network or a warning-labelled archived state. No fields are trained.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from ml_collections import ConfigDict
import numpy as np

from examples.fracture2d._legacy.continuation import _load_accepted, _source_root_for_state
from examples.fracture2d._legacy.pinn_paths import physical_pinn_path
from examples.fracture2d._legacy.train import _build_grid, _reference


def _state_config(config, root, index):
    if ("fork_after_state_index" in config.output
            and index <= config.output.fork_after_state_index):
        ancestor = _source_root_for_state(config, config.output.fork_from, index)
    else:
        ancestor = root
    state_dir = ancestor / f"state_{index:04d}"
    path = state_dir / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"accepted load has no stored configuration: {path}")
    state_config = ConfigDict(json.loads(path.read_text(encoding="utf-8")))
    if (int(state_config.data.state_index) != index
            or physical_pinn_path(state_config.output.dir).resolve() != state_dir.resolve()):
        raise ValueError(f"stored load {index} does not belong to {state_dir}")
    return state_config


def _verified_fields(root, indices):
    root = physical_pinn_path(root).resolve()
    config_path = root / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"continuation root has no config: {config_path}")
    config = ConfigDict(json.loads(config_path.read_text(encoding="utf-8")))
    if physical_pinn_path(config.output.dir).resolve() != root:
        raise ValueError("continuation root config points to another directory")
    if (not indices or len(indices) != len(set(indices))
            or list(indices) != sorted(indices)
            or any(type(i) is not int or i < 1 or i > 59 for i in indices)
            or max(indices) > int(config.data.end_state_index)):
        raise ValueError("plot indices must be increasing unique strict states in 1..59")
    model = _build_grid(config)
    selected = {}
    parent_sha256 = None
    for index in range(1, max(indices) + 1):
        state_config = _state_config(config, root, index)
        state, parent_sha256 = _load_accepted(state_config, model, parent_sha256)
        if index in indices:
            load, _, _, q_ref, _, _, _, warning = _reference(state_config, model)
            if warning:
                raise ValueError(f"reference state {index} is warning-labelled")
            n = model.n_material
            coordinates = np.asarray(model.geometry.coordinates[:n])
            predicted = np.asarray(state[4][:2*n]).reshape(n, 2)
            reference = np.asarray(q_ref[:2*n]).reshape(n, 2)
            if not all(np.all(np.isfinite(x)) for x in (coordinates, predicted, reference)):
                raise FloatingPointError(f"displacement comparison at state {index} is not finite")
            selected[index] = (float(load), coordinates, predicted, reference, parent_sha256)
    return config, selected


def _metrics(predicted, reference, coordinates):
    error = predicted - reference
    lengths = np.linalg.norm(error, axis=1)
    peak = int(np.argmax(lengths))
    ref_lengths = np.linalg.norm(reference, axis=1)
    return {
        "relative_l2_material": float(np.linalg.norm(error) / np.linalg.norm(reference)),
        "ux_relative_l2": float(np.linalg.norm(error[:, 0]) / np.linalg.norm(reference[:, 0])),
        "uy_relative_l2": float(np.linalg.norm(error[:, 1]) / np.linalg.norm(reference[:, 1])),
        "vector_rmse": float(np.sqrt(np.mean(lengths**2))),
        "max_vector_error": float(lengths[peak]),
        "max_vector_error_over_reference_peak": float(lengths[peak] / np.max(ref_lengths)),
        "max_error_coordinate": coordinates[peak].tolist(),
        "ux_error_min_max": [float(error[:, 0].min()), float(error[:, 0].max())],
        "uy_error_min_max": [float(error[:, 1].min()), float(error[:, 1].max())],
    }


def _plot(triangles, coordinates, predicted, reference, index, load, metrics, path,
          zoom):
    triangulation = mtri.Triangulation(
        coordinates[:, 0], coordinates[:, 1], triangles,
    )
    error = predicted - reference
    rows = (
        (reference[:, 0], reference[:, 1], np.linalg.norm(reference, axis=1)),
        (predicted[:, 0], predicted[:, 1], np.linalg.norm(predicted, axis=1)),
        (error[:, 0], error[:, 1], np.linalg.norm(error, axis=1)),
    )
    labels = ("$u_x$", "$u_y$", "$|u|$ / $|u_{PINN}-u_{ref}|$")
    bounds = (
        (min(reference[:, 0].min(), predicted[:, 0].min()),
         max(reference[:, 0].max(), predicted[:, 0].max())),
        (min(reference[:, 1].min(), predicted[:, 1].min()),
         max(reference[:, 1].max(), predicted[:, 1].max())),
        (0.0, max(np.linalg.norm(reference, axis=1).max(),
                  np.linalg.norm(predicted, axis=1).max())),
    )
    figure, axes = plt.subplots(3, 3, figsize=(14.0, 12.0), constrained_layout=True)
    for row, values in enumerate(rows):
        for column, panel in enumerate(values):
            axis = axes[row, column]
            if row < 2:
                vmin, vmax = bounds[column]
                color = "coolwarm" if column < 2 else "viridis"
            elif column < 2:
                vmax = float(np.max(np.abs(panel)))
                vmin = -vmax
                color = "RdBu_r"
            else:
                vmin, vmax, color = 0.0, float(panel.max()), "magma"
            artist = axis.tripcolor(
                triangulation, panel, shading="gouraud", cmap=color,
                vmin=float(vmin), vmax=float(vmax),
            )
            figure.colorbar(artist, ax=axis, fraction=0.046, pad=0.03)
            axis.set_aspect("equal", adjustable="box")
            if zoom is not None:
                axis.set_xlim(zoom[0], zoom[1])
                axis.set_ylim(zoom[2], zoom[3])
            axis.set_title(labels[column])
            axis.set_xlabel("x")
            axis.set_ylabel(("Direct reference", "Accepted PINN", "PINN - reference")[row]
                            + "  |  y")
    figure.suptitle(
        f"Fracture2D strict state {index}, load {load:.4f}  |  "
        f"material u relative L2 {metrics['relative_l2_material']:.3%}",
    )
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--states", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--zoom", type=float, nargs=4, metavar=("X0", "X1", "Y0", "Y1"))
    args = parser.parse_args()
    zoom = args.zoom
    if zoom is not None and (not all(np.isfinite(zoom))
                             or zoom[0] >= zoom[1] or zoom[2] >= zoom[3]):
        raise ValueError("zoom bounds must be finite with lower < upper")
    if not args.output_dir.parent.is_dir():
        raise FileNotFoundError(f"plot parent does not exist: {args.output_dir.parent}")
    config, fields = _verified_fields(args.root, args.states)
    outputs = [args.output_dir / f"displacement_state_{index:04d}.png"
               for index in args.states]
    summary_path = args.output_dir / "displacement_summary.json"
    if any(path.exists() for path in (*outputs, summary_path)):
        raise FileExistsError("refusing to overwrite a displacement plot or summary")
    with np.load(config.data.reference, allow_pickle=False) as archive:
        triangles = np.asarray(archive["triangles"])
        archived_coordinates = np.asarray(archive["coordinates"])
    args.output_dir.mkdir(exist_ok=True)
    summary = {}
    for index, path in zip(args.states, outputs):
        load, coordinates, predicted, reference, digest = fields[index]
        if not np.array_equal(coordinates, archived_coordinates):
            raise ValueError("plot triangles and verified material coordinates differ")
        metrics = _metrics(predicted, reference, coordinates)
        _plot(triangles, coordinates, predicted, reference, index, load, metrics, path, zoom)
        summary[str(index)] = {
            "load": load, "accepted_sha256": digest, "plot": str(path), **metrics,
        }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "states": summary}, sort_keys=True))


if __name__ == "__main__":
    main()
