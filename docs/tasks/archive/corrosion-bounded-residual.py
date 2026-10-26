"""Replay bounded-residual pilots by direct file paths; reference is evaluation-only."""

import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ml_collections import ConfigDict
import numpy as np

from examples.corrosion2d.residual_experiment import (
    boundary_error, boundary_values, build_model, field_metrics, load_reference,
)


ROOT = Path("/root/autodl-tmp/tf-logs/corrosion2d")
RUNS = {
    "historical original": ROOT / "20261024-residual-phase-seed42",
    "original": ROOT / "20261025-residual-phase-control-seed42",
    "full inverse": ROOT / "20261024-residual-latent-seed42",
    "cap2": ROOT / "20261025-residual-bounded-eta1.0-mapkks-ac1e-07-ch1.0-seed42",
    "cap11": ROOT / "20261025-residual-bounded-eta0.1-mapkks-ac1e-07-ch1.0-seed42",
    "cap101": ROOT / "20261025-residual-bounded-eta0.01-mapkks-ac1e-07-ch1.0-seed42",
    "original seed7": ROOT / "20261025-residual-phase-control-seed7",
    "cap11 seed7": ROOT / "20261025-residual-bounded-eta0.1-mapkks-ac1e-07-ch1.0-seed7",
    "original no scale": ROOT / "20261025-residual-phase-noscale-seed42",
    "cap11 no scale": ROOT / "20261025-residual-bounded-eta0.1-mapkks-ac1.0-ch1.0-seed42",
}


def flatten(data, prefix=""):
    result = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten(value, name))
        else:
            result[name] = value
    return result


def main():
    if jax.default_backend() != "cpu":
        raise ValueError("Replay requires JAX_PLATFORMS=cpu")
    configs = {name: json.loads((path / "config.json").read_text()) for name, path in RUNS.items()}
    original_config = flatten(configs["original"])
    initial_bytes = {seed: (RUNS[label] / "model_initial.eqx").read_bytes()
                     for seed, label in ((42, "original"), (7, "original seed7"))}
    ignored = {"log.run_name", "experiment.formulation", "experiment.bounded_eta"}
    result = {"reference_used_for_training": False, "runs": {}}
    predictions, histories = {}, {}
    for name, path in RUNS.items():
        raw_config = configs[name]
        flattened = flatten(raw_config)
        differences = [key for key in original_config.keys() | flattened.keys()
                       if key not in ignored and (key not in original_config or key not in flattened
                                                  or original_config[key] != flattened[key])]
        allowed = {"train.seed"} if raw_config["train"]["seed"] == 7 else set()
        if raw_config["loss"]["ac_scale"] == 1:
            allowed.add("loss.ac_scale")
        if set(differences) != allowed:
            raise ValueError(f"Unmatched controls for {name}: {differences}")
        if (path / "model_initial.eqx").read_bytes() != initial_bytes[raw_config["train"]["seed"]]:
            raise ValueError(f"Unmatched raw initialization for {path}")
        c = ConfigDict(raw_config)
        reference = load_reference(c)
        summary = json.loads((path / "summary.json").read_text())
        if summary["updates"] != 5000 or summary["final_metrics"]["step"] != 5000:
            raise ValueError(f"Wrong completed budget: {path}")
        with np.load(path / "prediction.npz") as saved:
            np.testing.assert_array_equal(saved["X"], reference["X"])
            np.testing.assert_array_equal(saved["Y"], reference["Y"])
            np.testing.assert_array_equal(saved["time"], reference["time"])
            prediction = np.stack([saved["phi"], saved["c"]], axis=-1)
        if not np.all(np.isfinite(prediction)):
            raise ValueError(f"Nonfinite saved field: {path}")
        metrics = field_metrics(prediction, reference)
        for key in ("terminal_rel_l2_phi", "terminal_rel_l2_c", "pooled_rel_l2_phi", "pooled_rel_l2_c"):
            np.testing.assert_allclose(metrics[key], summary["final_metrics"][key], rtol=1e-10)
        points = np.stack([reference["X"].ravel(), reference["Y"].ravel()], axis=-1)
        x = jnp.asarray(points, dtype=jnp.float32)
        net = build_model(c, jax.random.split(jax.random.PRNGKey(c.train.seed))[0])
        net = eqx.tree_deserialise_leaves(path / "model_final.eqx", net)

        @eqx.filter_jit
        def predict_terminal(model):
            t = jnp.array(1., dtype=x.dtype)
            fields = jax.vmap(model, in_axes=(0, None))(x, t)
            raw = jax.vmap(model.net, in_axes=(0, None))(x, t)
            dc_dv = jax.vmap(jax.jacfwd(model.constraint))(raw)[:, 1, 1]
            return fields, raw, dc_dv

        replay, raw, dc_dv = (np.asarray(a) for a in predict_terminal(net))
        replay = replay.reshape(*reference["X"].shape, 2)
        replay_difference = float(np.max(np.abs(replay - prediction[-1])))
        np.testing.assert_allclose(replay, prediction[-1], rtol=0, atol=5e-6)
        zero, one, _ = boundary_values(c, points)
        phase = prediction[-1, ..., 0]
        crossings = np.sum(np.diff(phase > .5, axis=0), axis=0)
        history = [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines()]
        late = [row for row in history if row["step"] >= 4000]
        result["runs"][name] = {
            "path": str(path), "reference": c.experiment.reference_path,
            "config_differences_beyond_formulation_eta_name": differences,
            "initial_parameters_byte_equal": True,
            "cpu_terminal_replay_max_abs_difference": replay_difference,
            "metrics": metrics,
            "boundary_max_abs_error": boundary_error(prediction, zero, one),
            "physical_phi_range": [float(prediction[..., 0].min()), float(prediction[..., 0].max())],
            "physical_c_range": [float(prediction[..., 1].min()), float(prediction[..., 1].max())],
            "columns_with_multiple_terminal_vertical_crossings": int(np.sum(crossings > 1)),
            "terminal_vertical_crossings_max": int(crossings.max()),
            "terminal_fraction_zero_concentration_output_derivative": float(np.mean(dc_dv == 0)),
            "terminal_raw_concentration_range": [float(raw[:, 1].min()), float(raw[:, 1].max())],
            "last_1000_update_terminal_phi_l2_median": float(np.median([r["terminal_rel_l2_phi"] for r in late])),
            "last_1000_update_terminal_c_l2_median": float(np.median([r["terminal_rel_l2_c"] for r in late])),
        }
        predictions[name], histories[name] = prediction, history
        print(name, metrics["terminal_rel_l2_phi"], metrics["terminal_rel_l2_c"], flush=True)
    baseline = result["runs"]["original"]["metrics"]
    scores = {name: max(result["runs"][name]["metrics"][f"terminal_rel_l2_{field}"] /
                        baseline[f"terminal_rel_l2_{field}"] for field in ("phi", "c"))
              for name in ("cap2", "cap11", "cap101")}
    best = min(scores, key=scores.__getitem__)
    result["screening"] = {"worst_field_error_ratio_to_original": scores,
                           "best_bounded": best, "both_terminal_errors_improved": scores[best] < 1,
                           "scope": "Three-eta same-seed screen; not independent hyperparameter validation."}
    result["validation"] = {}
    for label, original, bounded in (("second_seed", "original seed7", "cap11 seed7"),
                                     ("without_pre_scales", "original no scale", "cap11 no scale")):
        result["validation"][label] = {
            field: {"original": result["runs"][original]["metrics"][f"terminal_rel_l2_{field}"],
                    "bounded": result["runs"][bounded]["metrics"][f"terminal_rel_l2_{field}"]}
            for field in ("phi", "c")}
    output = Path(__file__)
    output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    primary = ("historical original", "original", "full inverse", "cap2", "cap11", "cap101")
    for name in primary:
        axes[0].semilogy([r["step"] for r in histories[name]],
                         [r["terminal_rel_l2_phi"] for r in histories[name]], label=name)
        axes[1].semilogy(reference["physical_time"],
                         result["runs"][name]["metrics"]["rel_l2_by_time_phi"], label=name)
    axes[0].set(xlabel="Completed updates", ylabel="50 s phase relative L2")
    axes[1].set(xlabel="Physical time (s)", ylabel="Final checkpoint phase relative L2")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend()
    fig.savefig(output.with_suffix(".png"), dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for row, labels in enumerate((("original seed7", "cap11 seed7"), ("original no scale", "cap11 no scale"))):
        for col, field in enumerate(("phi", "c")):
            for name in labels:
                axes[row, col].semilogy([r["step"] for r in histories[name]],
                                        [r[f"terminal_rel_l2_{field}"] for r in histories[name]], label=name)
            axes[row, col].set(xlabel="Completed updates", ylabel=f"50 s {field} relative L2")
            axes[row, col].grid(alpha=.2)
            axes[row, col].legend()
    fig.savefig(output.with_name(output.stem + "-validation.png"), dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    for ax, label, values in zip(axes, ("Reference", "Original", best),
                                 (reference["phi"][-1], predictions["original"][-1, ..., 0], predictions[best][-1, ..., 0])):
        plot = ax.pcolormesh(reference["X"], reference["Y"], values, vmin=0, vmax=1, shading="nearest")
        ax.contour(reference["X"], reference["Y"], reference["phi"][-1], levels=[.5], colors="white", linewidths=.8)
        ax.set(title=label, xlabel="x*", ylabel="y*", aspect="equal")
    fig.colorbar(plot, ax=axes, label="Phase at 50 s")
    fig.savefig(output.with_name(output.stem + "-fields.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
