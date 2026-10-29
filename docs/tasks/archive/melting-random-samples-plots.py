"""Reproduce three randomly preselected held-out melting phase galleries."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parent
selection = json.loads((OUT / "melting-random-samples-selection.json").read_text())
run = Path(selection["run"])
manifest_path = Path(selection["manifest"])
manifest = json.loads(manifest_path.read_text())
summary = json.loads((run / "summary.json").read_text())
config = json.loads((run / "config.json").read_text())
indices = selection["selected_indices"]
assert len(indices) == len(set(indices)) == 3
assert set(indices).issubset(manifest["test_indices"])
records = {row["index"]: row for row in summary["test"]["conditions"]}
fields = {}


def relative_l2(prediction, target):
    """Compute physical-field L2 using FP64 accumulation, without a norm floor."""
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    assert np.sum(target**2) > 0
    return float(np.sqrt(np.sum((prediction - target)**2) / np.sum(target**2)))


for index in indices:
    with np.load(run / f"prediction_{index:03d}.npz") as stored:
        data = {name: stored[name] for name in stored.files}
    assert all(np.isfinite(array).all() for array in data.values())
    np.testing.assert_allclose(data["time"], np.linspace(0, 1, 5), rtol=0, atol=0)
    np.testing.assert_array_equal(data["parameters"], manifest["parameters"][index])
    assert data["prediction"].shape == data["target"].shape == (5, 24**3)
    assert data["slice_prediction"].shape == data["slice_target"].shape == (5, 128**2)
    points = data["slice_points"]
    assert points.shape == (128**2, 3)
    assert np.ptp(points[:, 2]) == 0
    # C-order XYZ ij coordinates require a transpose for image rows Y, columns X.
    grid = points.reshape(128, 128, 3)
    assert np.all(np.diff(grid[:, 0, 0]) > 0)
    assert np.all(np.diff(grid[0, :, 1]) > 0)
    with np.load(manifest_path.parent / f"condition_{index:03d}.npz") as reference:
        np.testing.assert_array_equal(data["target"], reference["phi"])
        np.testing.assert_array_equal(data["slice_target"], reference["slice_phi"])
        np.testing.assert_array_equal(points, reference["slice_points"])
    fields[index] = data

error_max = max(float(np.max(np.abs(d["slice_prediction"] - d["slice_target"])))
                for d in fields.values())
metrics = []
for index, data in fields.items():
    pred, target = data["prediction"], data["target"]
    row = {
        "index": index,
        "semiaxes": data["parameters"][3:6].tolist(),
        "pooled_phi": relative_l2(pred, target),
        "terminal_phi": relative_l2(pred[-1], target[-1]),
        "terminal_solid": relative_l2((pred[-1] + 1) / 2, (target[-1] + 1) / 2),
        "phi_by_time": [relative_l2(p, t) for p, t in zip(pred, target)],
        "slice_z": float(data["slice_points"][0, 2]),
    }
    for key in ("pooled_phi", "terminal_phi", "terminal_solid", "phi_by_time"):
        np.testing.assert_allclose(row[key], records[index][key], rtol=2e-5, atol=1e-10)
    metrics.append(row)
    target_slice, pred_slice = data["slice_target"], data["slice_prediction"]
    error = np.abs(pred_slice - target_slice)
    fig, axes = plt.subplots(3, 5, figsize=(13, 7.4), constrained_layout=True)
    for r, (values, label) in enumerate(((target_slice, "Reference"),
                                        (pred_slice, "Prediction"),
                                        (error, "Absolute error"))):
        for c, t in enumerate(data["time"]):
            ax = axes[r, c]
            image = ax.imshow(values[c].reshape(128, 128).T, origin="lower",
                              extent=(0, 1, 0, 1),
                              cmap="magma" if r == 2 else "coolwarm",
                              vmin=0 if r == 2 else -1,
                              vmax=error_max if r == 2 else 1,
                              interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"t* = {t:.2f}  ({config['nondim']['T_c'] * t:g} s)")
            if c == 0:
                ax.set_ylabel(label, fontsize=12)
        bar = fig.colorbar(image, ax=axes[r, :], fraction=.018, pad=.012)
        bar.set_label("Absolute phase error" if r == 2 else "Phase phi")
    semiaxes = ", ".join(f"{a:.3f}" for a in row["semiaxes"])
    fig.suptitle(f"Random held-out ellipsoid ID {index} | initial semiaxes ({semiaxes})\n"
                 f"z = {row['slice_z']:.4f} slice | ice +1, water -1", fontsize=14)
    fig.savefig(OUT / f"melting-random-sample-{index:03d}.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)

report = {
    "selection": selection,
    "checkpoint": str(run / "model_final.eqx"),
    "metric_scope": "Physical phi, 24^3 three-dimensional diagnostic points; pooled across five times or terminal only.",
    "plot_scope": "128^2 central z slice; common phase [-1,1] and absolute-error [0,max] scales across all three inputs.",
    "common_slice_error_max": error_max,
    "conditions": metrics,
    "checks": "Finite/shape/time/coordinate-order/manifest/reference equality and saved 3D metric replay passed.",
}
(OUT / "melting-random-samples-metrics.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
