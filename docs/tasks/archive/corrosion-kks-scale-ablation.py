"""Replay six corrosion controls from explicit run/config/checkpoint paths."""

import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ml_collections import ConfigDict

from examples.corrosion2d.residual_experiment import build_model, load_reference


def flatten(mapping, prefix=""):
    result = {}
    for key, value in mapping.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten(value, path))
        else:
            result[path] = value
    return result


def main():
    if jax.default_backend() != "cpu":
        raise ValueError("Use JAX_PLATFORMS=cpu")
    root = Path("/root/autodl-tmp/tf-logs/corrosion2d")
    groups = [("kks", 1e-7), ("kks", 1e-9), ("direct", 1e-9)]
    output = Path(__file__).with_suffix("")
    result = {"seed": 42, "updates": 5000, "physical_final_time_seconds": 50,
              "hard_ic": False, "diagnostic_range_tolerance": 1e-6, "runs": {}}
    curves, axes = plt.subplots(1, 2, figsize=(12, 4))
    fields, field_axes = plt.subplots(3, 3, figsize=(12, 8))
    initial_weights = None
    base_config = flatten(json.loads((root / "20261024-residual-phase-seed42/config.json").read_text()))
    for row, (mapping, scale) in enumerate(groups):
        group = f"{mapping}, AC scale {scale:g}"
        for col, formulation in enumerate(("phase", "latent"), start=1):
            if scale == 1e-7:
                run = root / f"20261024-residual-{formulation}-seed42"
            else:
                run = root / f"20261024-residual-{formulation}-map{mapping}-ac1e-09-ch1.0-seed42"
            config = ConfigDict(json.loads((run / "config.json").read_text()))
            flat_config = flatten(config.to_dict())
            differences = [key for key in sorted(base_config.keys() | flat_config.keys())
                           if key not in base_config or key not in flat_config or base_config[key] != flat_config[key]]
            assert set(differences) <= {"experiment.formulation", "log.run_name", "arch.use_hard_constraint", "loss.ac_scale"}, differences
            assert config.arch.use_hard_constraint == (mapping == "kks")
            assert config.loss.ac_scale == scale and config.loss.ch_scale == 1
            assert config.train.iterations == 5000 and config.train.seed == 42
            initial = (run / "model_initial.eqx").read_bytes()
            if initial_weights is None:
                initial_weights = initial
            assert initial == initial_weights, f"Raw initial weights differ: {run}"
            reference = load_reference(config)
            history = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
            assert history[-1]["step"] == 5000
            with np.load(run / "prediction.npz") as data:
                prediction = {key: data[key] for key in data.files}
            summary = json.loads((run / "summary.json").read_text())
            check = {}
            for k, name in enumerate(("phi", "c")):
                error = float(np.linalg.norm(prediction[name][-1].astype(np.float64) - reference[name][-1]) / np.linalg.norm(reference[name][-1]))
                assert abs(error - summary["final_metrics"][f"terminal_rel_l2_{name}"]) < 1e-12
                check[f"terminal_rel_l2_{name}"] = error
                axes[k].plot([h["step"] for h in history], [h[f"terminal_rel_l2_{name}"] for h in history],
                    color=f"C{row}", linestyle="-" if formulation == "phase" else "--", label=f"{group}; {formulation}")
                tail = [h[f"terminal_rel_l2_{name}"] for h in history if h["step"] >= 4000]
                check[f"last1000_terminal_{name}_l2_median"] = float(np.median(tail))
            c = prediction["c"]
            phi_all = prediction["phi"]
            dc = config.physical.c_se - config.physical.c_le
            liquid = c - dc * (3 * phi_all**2 - 2 * phi_all**3)
            liquid_max = 1 - config.physical.c_se + config.physical.c_le
            check.update(concentration_min=float(c.min()), concentration_max=float(c.max()),
                         concentration_fraction_outside_0_1=float(np.mean((c < 0) | (c > 1))),
                         concentration_terminal_min=float(c[-1].min()), concentration_terminal_max=float(c[-1].max()),
                         reconstructed_liquid_min=float(liquid.min()), reconstructed_liquid_max=float(liquid.max()),
                         reconstructed_liquid_fraction_outside_bounds_tol=float(np.mean((liquid < -1e-6) | (liquid > liquid_max + 1e-6))),
                         physical_initial_phi_rel_l2_before_training=history[0]["rel_l2_by_time_phi"][0],
                         physical_initial_c_rel_l2_before_training=history[0]["rel_l2_by_time_c"][0])
            crossings = ((prediction["phi"][-1, :-1] - .5) * (prediction["phi"][-1, 1:] - .5) < 0).sum(axis=0)
            check["terminal_vertical_crossing_histogram"] = {str(int(a)): int(b) for a, b in zip(*np.unique(crossings, return_counts=True))}
            template = build_model(config, jax.random.split(jax.random.PRNGKey(config.train.seed))[0])
            model = eqx.tree_deserialise_leaves(run / "model_final.eqx", template)
            x = jnp.asarray(np.stack([reference["X"].ravel(), reference["Y"].ravel()], axis=-1), dtype=jnp.float32)
            times = jnp.array([0., .5, 1.])

            @eqx.filter_jit
            def probe(net):
                raw = jax.vmap(lambda t: jax.vmap(net.net, in_axes=(0, None))(x, t))(times)
                physical = net.constraint(raw)
                derivative = jax.jvp(net.constraint, (raw,), (jnp.zeros_like(raw).at[..., 1].set(1.),))[1][..., 1]
                return physical, derivative

            predicted, derivative = (np.asarray(a) for a in probe(model))
            saved = np.stack([prediction["phi"][[0, 10, 20]], prediction["c"][[0, 10, 20]]], axis=-1).reshape(predicted.shape)
            discrepancy = float(np.max(np.abs(predicted - saved)))
            assert discrepancy < 5e-4, discrepancy
            check["checkpoint_cpu_gpu_max_abs_difference"] = discrepancy
            check["fraction_zero_dc_draw1"] = float(np.mean(derivative == 0))
            key = f"{mapping}_ac{scale:g}_{formulation}"
            result["runs"][key] = {"path": str(run), "summary": summary, "checks": check,
                                   "config_differences_from_original_baseline": differences}
            if (run / "loss_history.jsonl").exists():
                losses = [json.loads(line) for line in (run / "loss_history.jsonl").read_text().splitlines()]
                result["runs"][key]["last_logged_losses_weights"] = losses[-1]
            print(key, json.dumps(check), flush=True)
            for target_col, phase in ((0, reference["phi"][-1]), (col, prediction["phi"][-1])):
                ax = field_axes[row, target_col]
                im = ax.pcolormesh(reference["X"], reference["Y"], phase, shading="auto", vmin=0, vmax=1, cmap="coolwarm")
                ax.contour(reference["X"], reference["Y"], reference["phi"][-1], levels=[.5], colors="black", linewidths=.8)
                if target_col:
                    ax.contour(reference["X"], reference["Y"], phase, levels=[.5], colors="white", linewidths=.8)
                ax.set(title=f"{group}\n{'Reference' if target_col == 0 else formulation}", aspect="equal")
    result["raw_initial_weights_identical"] = True
    for name, ax in zip(("phi", "c"), axes):
        ax.set(title=f"Corrosion50 s: terminal {name} relative L2", xlabel="Completed updates", ylabel="Relative L2", yscale="log")
        ax.legend(fontsize=7)
        ax.grid(alpha=.25)
    curves.tight_layout()
    curves.savefig(str(output) + ".png", dpi=180)
    fields.subplots_adjust(wspace=.3, hspace=.55, right=.87)
    fields.colorbar(im, cax=fields.add_axes([.91, .2, .015, .6]), label="phi")
    fields.savefig(str(output) + "-fields.png", dpi=180)
    output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
