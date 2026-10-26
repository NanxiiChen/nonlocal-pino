"""Replay paired-run comparisons from direct config/field/metric file paths."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def checkpoint_probe(run, config, reference, prediction):
    """Load the saved model by path; check fields and output-branch saturation."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    from ml_collections import ConfigDict
    from examples.corrosion2d.residual_experiment import build_model
    from nonlocal_pinn.nn.constraints import kks_constraint

    if jax.default_backend() != "cpu":
        raise ValueError("Run the independent report with JAX_PLATFORMS=cpu")
    c = ConfigDict(config)
    template = build_model(c, jax.random.split(jax.random.PRNGKey(c.train.seed))[0])
    model = eqx.tree_deserialise_leaves(run / "model_final.eqx", template)
    x = jnp.asarray(np.stack([reference["X"].ravel(), reference["Y"].ravel()], axis=-1), dtype=jnp.float32)
    indices = np.array([0, len(reference["time"]) // 2, len(reference["time"]) - 1])
    times = jnp.asarray(reference["time"][indices], dtype=jnp.float32)

    @eqx.filter_jit
    def predict_raw(net):
        return jax.vmap(lambda t: jax.vmap(net, in_axes=(0, None))(x, t))(times)

    raw = predict_raw(model.net)
    fields = np.asarray(kks_constraint(raw, c.physical.c_se, c.physical.c_le)).reshape(len(indices), *reference["X"].shape, 2)
    saved = np.stack([prediction["phi"][indices], prediction["c"][indices]], axis=-1)
    max_error = float(np.max(np.abs(fields - saved)))
    if max_error > 5e-4:
        raise ValueError(f"Independent checkpoint predictions differ from saved fields: {run}: {max_error}")
    cl_max = 1.0 - c.physical.c_se + c.physical.c_le
    tanh_v = jnp.tanh(raw[..., 1])
    dcl = np.asarray(0.5 * cl_max * (1.0 - tanh_v**2))
    cl = np.asarray(0.5 * cl_max * (1.0 + tanh_v))
    result = {"checkpoint_prediction_max_abs_difference_cpu_gpu": max_error,
              "raw_phase_min": float(jnp.min(raw[..., 0])), "raw_phase_max": float(jnp.max(raw[..., 0])),
              "raw_concentration_min": float(jnp.min(raw[..., 1])), "raw_concentration_max": float(jnp.max(raw[..., 1])),
              "liquid_concentration_min": float(cl.min()), "liquid_concentration_max": float(cl.max()),
              "fraction_zero_liquid_concentration_derivative": float(np.mean(dcl == 0))}
    for label, mask in (("all_sampled_times", np.ones(raw.shape[:2], dtype=bool)), ("terminal", np.broadcast_to(np.arange(len(indices))[:, None] == len(indices) - 1, raw.shape[:2]))):
        result[label + "_fraction_cL_zero"] = float(np.mean(cl[mask] == 0))
    return result


def read_json(path):
    return json.loads(path.read_text())


def config_differences(a, b, prefix=""):
    differences = []
    for key in sorted(a.keys() | b.keys()):
        path = f"{prefix}.{key}" if prefix else key
        if key not in a or key not in b:
            differences.append(path)
        elif isinstance(a[key], dict) and isinstance(b[key], dict):
            differences.extend(config_differences(a[key], b[key], path))
        elif a[key] != b[key]:
            differences.append(path)
    return differences


def main():
    output = Path(__file__).with_suffix("")
    cases = ("corrosion2d", "electropolish2d")
    result = {"protocol": "matched5000 updates, soft ICs, normalized phase residual only",
              "seed": 42, "cases": {}}
    curves, curve_axes = plt.subplots(2, 2, figsize=(11, 7), squeeze=False)
    fields, field_axes = plt.subplots(2, 3, figsize=(13, 6), squeeze=False)
    colors = {"phase": "#2467a6", "latent": "#cf671e"}
    for row, case in enumerate(cases):
        root = Path("/root/autodl-tmp/tf-logs") / case
        runs = {kind: root / f"20261024-residual-{kind}-seed42" for kind in ("phase", "latent")}
        configs = {kind: read_json(run / "config.json") for kind, run in runs.items()}
        diff = config_differences(configs["phase"], configs["latent"])
        assert diff == ["experiment.formulation", "log.run_name"], diff
        assert (runs["phase"] / "model_initial.eqx").read_bytes() == (runs["latent"] / "model_initial.eqx").read_bytes()
        refpath = Path(configs["phase"]["experiment"]["reference_path"])
        with np.load(refpath) as data:
            ref = {key: data[key] for key in data.files}
        metadata = json.loads(str(ref["metadata_json"]))
        data_result = {"reference_path": str(refpath), "reference_validation": metadata,
                       "initial_weights_identical": True, "config_differences": diff, "runs": {}}
        snapshots = [ref["phi"][-1]]
        reference_crossings = ((ref["phi"][-1, :-1] - 0.5) * (ref["phi"][-1, 1:] - 0.5) < 0).sum(axis=0)
        data_result["reference_vertical_crossing_histogram"] = {str(int(k)): int(v) for k, v in zip(*np.unique(reference_crossings, return_counts=True))}
        for kind, run in runs.items():
            summary = read_json(run / "summary.json")
            history = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
            assert summary["updates"] == 5000 and not summary["hard_initial_condition"]
            assert history[0]["step"] == 0 and history[-1]["step"] == 5000
            with np.load(run / "prediction.npz") as data:
                prediction = {key: data[key] for key in data.files}
            recomputed = {}
            for name in ("phi", "c"):
                error = prediction[name][-1].astype(np.float64) - ref[name][-1]
                value = float(np.linalg.norm(error) / np.linalg.norm(ref[name][-1]))
                assert abs(value - summary["final_metrics"][f"terminal_rel_l2_{name}"]) < 1e-12
                recomputed[f"terminal_rel_l2_{name}"] = value
                col = 0 if name == "phi" else 1
                ax = curve_axes[row, col]
                ax.plot([h["step"] for h in history],
                        [h[f"terminal_rel_l2_{name}"] for h in history],
                        label=kind, color=colors[kind], linewidth=1.6)
                ax.set(title=f"{case}: terminal {name} relative L2", xlabel="Completed updates", ylabel="Relative L2", yscale="log")
                ax.grid(alpha=0.25)
                ax.legend()
            tail = [h["terminal_rel_l2_phi"] for h in history if h["step"] >= 4000]
            recomputed["last1000_terminal_phi_l2_median"] = float(np.median(tail))
            recomputed["last1000_terminal_phi_l2_min"] = float(np.min(tail))
            recomputed["last1000_terminal_phi_l2_max"] = float(np.max(tail))
            recomputed["best_logged_terminal_phi_l2"] = float(min(h["terminal_rel_l2_phi"] for h in history))
            recomputed["initial_phi_rel_l2_at_final_model"] = summary["final_metrics"]["rel_l2_by_time_phi"][0]
            recomputed["terminal_solid_fraction"] = float(np.mean(prediction["phi"][-1] > 0.5))
            phase = prediction["phi"][-1]
            crossings = ((phase[:-1] - 0.5) * (phase[1:] - 0.5) < 0).sum(axis=0)
            recomputed["vertical_crossing_histogram"] = {str(int(k)): int(v) for k, v in zip(*np.unique(crossings, return_counts=True))}
            recomputed["fraction_columns_with_extra_crossings"] = float(np.mean(crossings > reference_crossings))
            target_phase = ref["phi"][-1]
            squared_error = (phase - target_phase)**2
            recomputed["phase_error_partition"] = {}
            for name, mask in (("solid", target_phase > 0.95), ("liquid", target_phase < 0.05), ("interface", (target_phase >= 0.05) & (target_phase <= 0.95))):
                if not np.any(mask):
                    raise ValueError(f"Empty reference error-partition region: {case}/{name}")
                recomputed["phase_error_partition"][name] = {"rmse": float(np.sqrt(np.mean(squared_error[mask]))), "squared_error_fraction": float(squared_error[mask].sum() / squared_error.sum())}
            recomputed["checkpoint_probe"] = checkpoint_probe(run, configs[kind], ref, prediction)
            data_result["runs"][kind] = {"path": str(run), "summary": summary, "independent_field_checks": recomputed}
            snapshots.append(prediction["phi"][-1])
        assert data_result["runs"]["phase"]["summary"]["reference"] == data_result["runs"]["latent"]["summary"]["reference"]
        for col, (phase, title) in enumerate(zip(snapshots, ("Aligned graph reference", "Original phase residual", "Transformed phase residual"))):
            ax = field_axes[row, col]
            im = ax.pcolormesh(ref["X"], ref["Y"], phase, shading="auto", vmin=0, vmax=1, cmap="coolwarm")
            ax.contour(ref["X"], ref["Y"], ref["phi"][-1], levels=[0.5], colors="black", linewidths=0.9)
            if col:
                ax.contour(ref["X"], ref["Y"], phase, levels=[0.5], colors="white", linewidths=0.9)
            ax.set(title=f"{case}, {metadata['spec']['nondim']['T_c']:g} s\n{title}", aspect="equal", xlabel="x*", ylabel="y*")
        data_result["reference_terminal_solid_fraction"] = float(np.mean(ref["phi"][-1] > 0.5))
        data_result["original_over_latent_terminal_phi_l2"] = data_result["runs"]["phase"]["independent_field_checks"]["terminal_rel_l2_phi"] / data_result["runs"]["latent"]["independent_field_checks"]["terminal_rel_l2_phi"]
        result["cases"][case] = data_result
    curves.tight_layout()
    curves.savefig(str(output) + ".png", dpi=180)
    fields.subplots_adjust(wspace=0.3, hspace=0.5, right=0.86)
    colorbar_ax = fields.add_axes([0.90, 0.22, 0.015, 0.56])
    fields.colorbar(im, cax=colorbar_ax, label="phi")
    fields.savefig(str(output) + "-fields.png", dpi=180)
    output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    for case, data in result["cases"].items():
        print(case, "phase/latent terminal phi ratio", data["original_over_latent_terminal_phi_l2"])
        for kind, run in data["runs"].items():
            print(kind, json.dumps(run["independent_field_checks"]))


if __name__ == "__main__":
    main()
