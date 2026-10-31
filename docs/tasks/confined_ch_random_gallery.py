"""Blind random gallery of already validated confined-CH saved physical fields."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

from examples.confined_ch2d.operator_plotting import SmoothMeshDisplay


def relative_l2(prediction, reference, areas):
    denominator = np.sum(reference**2 * areas)
    if denominator <= 0:
        raise ValueError("Reference phase must have a positive squared norm")
    return float(np.sqrt(np.sum((prediction - reference)**2 * areas) / denominator))


def make_gallery(run, output, count):
    config = json.loads((run / "config.json").read_text())
    dataset = Path(config["operator"]["dataset_path"])
    manifest = json.loads((dataset / "manifest.json").read_text())
    candidates = manifest["test_indices"]
    if manifest["status"] != "passed" or len(set(candidates)) != len(candidates):
        raise ValueError("A validated bank with unique reserved geometry IDs is required")
    if not 1 <= count <= len(candidates):
        raise ValueError("Sample count must be within the reserved geometry count")
    if set(candidates) & set(manifest["train_indices"]):
        raise ValueError("Reserved geometry IDs overlap the training pool")

    output.mkdir(parents=True, exist_ok=False)
    chosen = np.random.default_rng().choice(candidates, size=count, replace=False).tolist()
    selection = dict(
        selected_utc=datetime.now(timezone.utc).isoformat(),
        run=str(run), dataset=str(dataset), candidates=candidates, chosen=chosen,
        sampling="Uniform without replacement, fresh entropy; IDs written before field/error reads",
        cohort="Eight reserved development geometries, reused during model development; absent from training draws",
    )
    (output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    print(json.dumps({"chosen_before_scores": chosen}), flush=True)

    # Establish shared scales across the entire reserved cohort, after committing the draw.
    keys = ("phi", "points", "volumes", "holes", "time", "mesh_points", "mesh_cells")
    records = {row["index"]: row for row in manifest["conditions"]}
    saved_metrics = json.loads((run / "evaluation/metrics.json").read_text())
    saved_by_index = {row["index"]: row for row in saved_metrics["conditions"]}
    data, scores = {}, {}
    bound, error_bound = 0., 0.
    for index in candidates:
        with np.load(run / f"prediction_{index:03d}.npz", allow_pickle=False) as payload:
            prediction = {key: payload[key] for key in keys}
            saved_reference = payload["phi_reference"]
        reference_path = dataset / records[index]["file"]
        with np.load(reference_path, allow_pickle=False) as payload:
            reference = {key: payload[key] for key in keys}
        for key in keys[1:]:
            np.testing.assert_array_equal(prediction[key], reference[key])
        np.testing.assert_array_equal(saved_reference, reference["phi"])
        np.testing.assert_allclose(reference["time"], [0., .75, 1.5, 2.25, 3.], rtol=0, atol=1e-14)
        areas, actual, target = reference["volumes"], prediction["phi"], reference["phi"]
        if actual.shape != (5, len(areas)) or target.shape != actual.shape:
            raise ValueError(f"Unexpected phase shape in geometry {index}")
        if np.any(areas <= 0) or not all(np.isfinite(a).all() for a in (areas, actual, target)):
            raise ValueError(f"Invalid physical phase or quadrature in geometry {index}")
        np.testing.assert_allclose(actual[0], target[0], rtol=0, atol=1e-13)
        pooled = relative_l2(actual, target, areas)
        terminal = relative_l2(actual[-1], target[-1], areas)
        np.testing.assert_allclose(
            [pooled, terminal],
            [saved_by_index[index]["phi_pooled"], saved_by_index[index]["phi_terminal"]],
            rtol=1e-13, atol=1e-15,
        )
        scores[index] = dict(index=index, phi_pooled=pooled, phi_terminal=terminal,
                             phase_l2_by_time=[relative_l2(a, b, areas) for a, b in zip(actual, target, strict=True)],
                             cells=len(areas), reference_path=str(reference_path))
        bound = max(bound, float(np.max(np.abs(actual))), float(np.max(np.abs(target))))
        error_bound = max(error_bound, float(np.max(np.abs(actual - target))))
        if index in chosen:
            data[index] = prediction, reference
    if min(bound, error_bound) <= 0:
        raise ValueError("Shared phase and absolute-error scales must be positive")

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "figure.facecolor": "white"})
    figure_records = []
    for index in chosen:
        prediction, reference = data[index]
        display = SmoothMeshDisplay(
            reference["mesh_points"], reference["mesh_cells"], reference["volumes"], reference["holes"],
            extent=(0., 1., .07, .93), resolution=(600, 480),
        )
        raw_error = np.abs(prediction["phi"] - reference["phi"])
        fields = (display(reference["phi"]), display(prediction["phi"]), display(raw_error))
        for values in fields:
            if not np.isfinite(values.compressed()).all():
                raise ValueError(f"Non-finite interpolated display in geometry {index}")
        if np.ma.min(fields[2]) < 0 or np.ma.max(fields[2]) > error_bound * (1 + 1e-12):
            raise ValueError("Positive display reconstruction must preserve absolute-error bounds")
        fig, axes = plt.subplots(3, 5, figsize=(14.6, 7.7), layout="constrained")
        for row, values in enumerate(fields):
            for col, time in enumerate(reference["time"]):
                ax = axes[row, col]
                im = ax.imshow(
                    values[col], origin="lower", extent=display.extent, interpolation="nearest",
                    cmap="RdBu_r" if row < 2 else "magma", vmin=-bound if row < 2 else 0,
                    vmax=bound if row < 2 else error_bound,
                )
                for cx, cy, radius in reference["holes"]:
                    ax.add_patch(Circle((cx, cy), radius, facecolor="white", edgecolor="black", linewidth=.6))
                ax.set(aspect="equal", xticks=[], yticks=[])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if row == 0:
                    ax.set_title(f"t = {time:.2f}")
                if col == 0:
                    ax.set_ylabel(("Native reference", "Operator prediction", "Absolute error")[row], labelpad=12)
            fig.colorbar(im, ax=list(axes[row]), shrink=.85, pad=.012,
                         label=r"$\phi$" if row < 2 else r"$|\phi_{NO}-\phi_{ref}|$")
        score = scores[index]
        fig.suptitle(
            f"Random reserved geometry {index} | phase relative L2: pooled {score['phi_pooled']:.2%}, "
            f"final {score['phi_terminal']:.2%}\nFixed initial field; variable hole geometry; common color scales across all samples",
            fontsize=12,
        )
        png, pdf = output / f"fields_{index:03d}.png", output / f"fields_{index:03d}.pdf"
        fig.savefig(png, dpi=190)
        fig.savefig(pdf, dpi=190)
        plt.close(fig)
        figure_records.append({**score, "png": str(png), "pdf": str(pdf)})
        print(json.dumps(figure_records[-1]), flush=True)

    receipt = dict(
        **selection, figures=figure_records, phase_bound=bound, absolute_error_bound=error_bound,
        times=[0., .75, 1.5, 2.25, 3.], reference="Independent native graph/BDF; local FEM deferred",
        display="Area-weighted incident-cell projection, linear interpolation within original triangles; analytic hole masks. Raw absolute errors interpolated directly; metrics use unchanged cell fields and areas.",
        verification="All eight field/geometry/time/reference identities and independently recomputed scores match saved evaluation; selected five finite positive display bounds pass.",
    )
    (output / "metrics.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--count", required=True, type=int)
    args = parser.parse_args()
    make_gallery(args.run.resolve(), args.output.resolve(), args.count)
