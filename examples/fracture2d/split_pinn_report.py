"""Independent comparison-only field and reaction audit of a split PINN prefix."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

from examples.fracture2d.coupling_diagnostics import atomic_json
from examples.fracture2d.split_horizon_model import read_cloud
from examples.fracture2d.split_pinn import comparison_reference, neural_state_root


def damage_metrics(actual, reference, coordinates, volumes, *, tip, tip_radius):
    """Report nodal/volume norms and local errors; no training or gate changes."""
    n = len(actual)
    if (actual.shape != (n,) or reference.shape != (n,)
            or coordinates.shape != (n, 2) or volumes.shape != (n,)
            or n == 0 or tip.shape != (2,) or not np.isfinite(tip_radius) or tip_radius <= 0
            or any(not np.all(np.isfinite(x)) for x in (actual, reference, coordinates, volumes, tip))
            or np.min(volumes) <= 0):
        raise ValueError("damage audit requires finite matching fields, positive volumes and explicit tip region")
    local = np.linalg.norm(coordinates - tip, axis=1) <= tip_radius
    if not np.any(local):
        raise ValueError("declared damage tip region contains no material nodes")
    difference = actual - reference
    norm = np.linalg.norm(reference)
    weighted_norm = np.sqrt(np.dot(volumes, reference**2))
    if norm == 0 or weighted_norm == 0:
        raise ValueError("nonzero-load damage audit requires a nonzero reference field norm")
    actual_peak, reference_peak = int(np.argmax(actual)), int(np.argmax(reference))
    return {
        "damage_rmse": float(np.sqrt(np.mean(difference**2))),
        "damage_relative_l2": float(np.linalg.norm(difference) / norm),
        "damage_volume_rmse": float(np.sqrt(np.dot(volumes, difference**2) / np.sum(volumes))),
        "damage_volume_relative_l2": float(np.sqrt(np.dot(volumes, difference**2)) / weighted_norm),
        "damage_max_error": float(np.max(np.abs(difference))),
        "tip_node_count": int(np.count_nonzero(local)),
        "tip_damage_rmse": float(np.sqrt(np.mean(difference[local]**2))),
        "tip_damage_max_error": float(np.max(np.abs(difference[local]))),
        "damage_peak": float(actual[actual_peak]),
        "reference_damage_peak": float(reference[reference_peak]),
        "damage_peak_coordinates": coordinates[actual_peak].tolist(),
        "reference_peak_coordinates": coordinates[reference_peak].tolist(),
        "peak_location_distance": float(np.linalg.norm(coordinates[actual_peak] - coordinates[reference_peak])),
    }


def plot_tip_comparison(path, cloud, actual, reference, *, tip, radius, load, status):
    """Zoom the same piecewise-linear mesh fields, with shared damage scales."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation
    mesh = Triangulation(cloud["coordinates"][:, 0], cloud["coordinates"][:, 1], cloud["triangles"])
    upper = max(float(np.max(actual)), float(np.max(reference)))
    error_upper = max(float(np.max(np.abs(actual - reference))), np.finfo(np.float64).eps)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, values, label in zip(axes, (actual, reference, actual - reference),
                                ("PINN damage", "Strict reference damage", "PINN minus reference")):
        error = label == "PINN minus reference"
        artist = ax.tripcolor(mesh, values, shading="gouraud", cmap="coolwarm" if error else "viridis",
                             vmin=-error_upper if error else 0., vmax=error_upper if error else upper)
        ax.set(xlim=(tip[0] - radius, tip[0] + radius), ylim=(tip[1] - radius, tip[1] + radius), title=label)
        ax.set_aspect("equal")
        fig.colorbar(artist, ax=ax)
    fig.suptitle(f"Notch-tip comparison, load {load:.6g}; {status} research state")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(root, last_index, output, *, plots):
    config = json.loads((root / "identity.json").read_text())["config"]
    cloud = read_cloud(config["data"]["cloud"], config["data"]["cloud_sha256"])
    curve = json.loads((root / "reaction_curve.json").read_text())
    if last_index < 1 or len(curve) < last_index:
        raise ValueError("requested damage audit exceeds completed neural prefix")
    rows = []
    last_fields = None
    for index, report in enumerate(curve[:last_index], 1):
        if report["index"] != index or not report["research_advance"]:
            raise ValueError("damage audit requires a contiguous research-usable neural prefix")
        state_dir = neural_state_root(root, index)
        snapshot = state_dir / "state.zip"
        with zipfile.ZipFile(snapshot) as archive:
            identity = json.loads(archive.read("metadata.json"))["identity"]
        fields_path = state_dir / f"fields_{report['sweep']:04d}.npz"
        with np.load(fields_path, allow_pickle=False) as archive:
            q, d = (archive[name] for name in ("q", "d"))
            if not np.array_equal(archive["coordinates"], cloud["coordinates"]):
                raise ValueError("damage audit field coordinates differ from bound cloud")
        _, reference, reaction, digest = comparison_reference(config["data"]["reference_root"], index,
            identity["model_digest"], len(q), len(d), report["load"])
        if digest != identity["reference_sha256"] or reaction != report["reference_reaction"]:
            raise ValueError("damage audit comparator differs from neural snapshot")
        metrics = damage_metrics(d, reference, cloud["coordinates"], cloud["volumes"],
                                 tip=np.asarray([0.5, 0.5]), tip_radius=config["physical"]["l0"])
        if metrics["damage_rmse"] != report["damage_rmse"]:
            raise ValueError("damage audit fields differ from recorded sweep RMSE")
        rows.append({"index": index, "load": report["load"], "status": report["status"],
                     "reaction": report["reaction"], "reference_reaction": reaction,
                     "reaction_relative": report["reaction_relative"], "drift_count": report["drift_count"],
                     "failed_checks": report["failed_checks"], "snapshot_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                     **metrics})
        last_fields = (d, reference)
    actual = np.asarray([row["reaction"] for row in rows])
    reference = np.asarray([row["reference_reaction"] for row in rows])
    summary = {"comparison_only": True, "field_equations_validated": False,
               "tip": [0.5, 0.5], "tip_radius": config["physical"]["l0"],
               "last_index": last_index, "reaction_curve_relative_l2": float(np.linalg.norm(actual - reference) / np.linalg.norm(reference)),
               "maximum_reaction_relative_error": max(row["reaction_relative"] for row in rows), "states": rows}
    atomic_json(output, summary)
    if plots:
        if last_fields is None:
            raise RuntimeError("damage audit has no final fields for plotting")
        plot_tip_comparison(output.parent / f"tip_damage_comparison_{last_index:04d}.png", cloud, *last_fields,
                            tip=np.asarray([0.5, 0.5]), radius=3 * config["physical"]["l0"], load=rows[-1]["load"],
                            status=rows[-1]["status"])
    print(json.dumps({key: value for key, value in summary.items() if key != "states"}
                     | {"last": rows[-1]}, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--last-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plots", action="store_true", help="Write a 3*l0 notch-tip damage comparison crop")
    args = parser.parse_args()
    run(args.root, args.last_index, args.output, plots=args.plots)


if __name__ == "__main__":
    main()
