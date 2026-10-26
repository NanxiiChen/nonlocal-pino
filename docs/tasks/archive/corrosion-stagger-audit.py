"""Independent four-run stagger replay, including concentration/transport fidelity.

Derived from the validated concentration audit; no archived outputs are changed.
"""

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

from examples.corrosion2d.residual_experiment import boundary_error, boundary_values, build_model, field_metrics, load_reference, nondim
from examples.corrosion2d.residuals import compute_pde


ROOT = Path("/root/autodl-tmp/tf-logs/corrosion2d")
RUNS = {
    "phase joint": ROOT / "20261025-stagger0-every25-phase-mapkks-ac1e-07-ch1.0-seed42",
    "phase stagger": ROOT / "20261025-stagger1000-every25-phase-mapkks-ac1e-07-ch1.0-seed42",
    "latent joint": ROOT / "20261025-stagger0-every25-latent-mapkks-ac1e-07-ch1.0-seed42",
    "latent stagger": ROOT / "20261025-stagger1000-every25-latent-mapkks-ac1e-07-ch1.0-seed42",
}


def h(phi):
    return 3 * phi**2 - 2 * phi**3


def relative_norm(error, target):
    denominator = np.linalg.norm(target)
    if denominator == 0:
        raise ValueError("Reference norm vanishes; relative error is undefined")
    return float(np.linalg.norm(error) / denominator)


def flatten(data, prefix=""):
    out = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(flatten(value, name))
        else:
            out[name] = value
    return out


def check_control(run, config, baseline, initial_bytes, source_bytes):
    values = flatten(config.to_dict())
    allowed = {"log.run_name", "experiment.formulation", "loss.stagger_steps"}
    differences = [k for k in baseline.keys() | values.keys()
                   if k not in allowed and (k not in baseline or k not in values or baseline[k] != values[k])]
    if differences:
        raise ValueError(f"Unmatched paired config: {run}: {differences}")
    if (run / "model_initial.eqx").read_bytes() != initial_bytes:
        raise ValueError(f"Unmatched initialization: {run}")
    for path, expected in source_bytes.items():
        if (run / "source" / path).read_bytes() != expected:
            raise ValueError(f"Runtime snapshot differs between paired runs: {run}: {path}")
    summary = json.loads((run / "summary.json").read_text())
    if summary["updates"] != 5000 or summary["final_metrics"]["step"] != 5000:
        raise ValueError(f"Wrong training budget: {run}")
    expected_counts = {"ac_only": 500, "ch_only": 500, "joint": 4000} if config.loss.stagger_steps else {"ac_only": 0, "ch_only": 0, "joint": 5000}
    if summary["activation_counts"] != expected_counts:
        raise ValueError(f"Wrong activation counts: {run}")
    history = [json.loads(line) for line in (run / "loss_history.jsonl").read_text().splitlines()]
    switches = set()
    for row in history:
        it = row["step"] - 1
        stage = (it // config.loss.stagger_switch_every) % 2 if it < config.loss.stagger_steps else 2
        mask = ([True,False,True,True,True], [False,True,True,True,True], [True]*5)[stage]
        if row["active"] != mask:
            raise ValueError(f"Wrong activation at {run}:{it}")
        adaptive, effective = np.asarray(row["adaptive_weights"]), np.asarray(row["weights"])
        if not np.all(np.isfinite(adaptive)) or np.any(adaptive <= 0):
            raise ValueError(f"Invalid adaptive state at {run}:{it}")
        np.testing.assert_allclose(effective, adaptive*np.asarray(config.loss.gradnorm_bias)*mask, rtol=0, atol=0)
        if not np.any(effective[:2] > 0) or np.any(effective[2:] <= 0):
            raise ValueError(f"Missing active PDE or IC/BC term at {run}:{it}")
        switches.add(it)
    if config.loss.stagger_steps and not set(range(0,1001,25)).issubset(switches):
        raise ValueError(f"Stage transition not logged: {run}")
    metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    late = [row for row in metrics if row["step"] >= 4000]
    return {"config_initialization_source_match": True, "activation_counts": expected_counts,
            "all_logged_effective_weights_verified": True,
            "late_terminal_phi_l2_median": float(np.median([r["terminal_rel_l2_phi"] for r in late])),
            "late_terminal_c_l2_median": float(np.median([r["terminal_rel_l2_c"] for r in late]))}


def main():
    if jax.default_backend() != "cpu":
        raise ValueError("Use JAX_PLATFORMS=cpu for checkpoint audit")
    result = {"scope": "Fresh corrected stagger x original/full-latent matched controls", "runs": {}}
    baseline_run = RUNS["phase joint"]
    baseline = flatten(json.loads((baseline_run / "config.json").read_text()))
    initial_bytes = (baseline_run / "model_initial.eqx").read_bytes()
    snapshot_files = json.loads((baseline_run / "source/files.json").read_text())
    source_bytes = {path: (baseline_run / "source" / path).read_bytes() for path in snapshot_files}
    predictions, histories = {}, {}
    for name, run in RUNS.items():
        config = ConfigDict(json.loads((run / "config.json").read_text()))
        checks = check_control(run, config, baseline, initial_bytes, source_bytes)
        ref = load_reference(config)
        nd = nondim(config)
        nch, _, _, _, dc, cle = nd
        target_phi = ref["phi"].astype(np.float64)
        target_c = ref["c"].astype(np.float64)
        target_cl = target_c - dc * h(target_phi)
        target_q = target_cl - cle
        with np.load(run / "prediction.npz") as data:
            np.testing.assert_array_equal(data["time"], ref["time"])
            phi, c = data["phi"].astype(np.float64), data["c"].astype(np.float64)
        cl = c - dc * h(phi)
        q = cl - cle
        total_error, phase_error = c - target_c, phi - target_phi
        geometry_error = dc * (h(phi) - h(target_phi))
        cl_error = cl - target_cl
        np.testing.assert_allclose(total_error, geometry_error + cl_error, rtol=0, atol=1e-15)
        np.testing.assert_allclose(q - target_q, cl_error, rtol=0, atol=1e-15)
        points = np.stack([ref["X"].ravel(), ref["Y"].ravel()], axis=-1)
        zero, one, _ = boundary_values(config, points)
        free = ~(zero | one)
        row, col, weight = (ref[key] for key in ("graph_row", "graph_col", "graph_weight"))
        n = len(points)

        def graph(values):
            flat = values.reshape(len(values), n).T
            output = np.zeros_like(flat)
            np.add.at(output, row, weight[:, None] * (flat[row] - flat[col]))
            return output.T

        dq_error, dq_ref = graph(q - target_q), graph(target_q)
        # Reload raw outputs to avoid cancellation when reconstructing c_L
        # from saved physical fields. Reference c_t follows its checked ODE.
        x = jnp.asarray(points, dtype=jnp.float32)
        times = jnp.array([1.], dtype=jnp.float32)
        sub = tuple(jnp.asarray(a) for a in (row, col, -weight, np.bincount(row, weights=weight, minlength=n).astype(np.float32)))
        net = build_model(config, jax.random.split(jax.random.PRNGKey(config.train.seed))[0])
        net = eqx.tree_deserialise_leaves(run / "model_final.eqx", net)

        @eqx.filter_jit
        def checkpoint_values(model):
            def raw_at_times(ts):
                return jax.vmap(lambda t: jax.vmap(model.net, in_axes=(0, None))(x, t))(ts)
            raw, raw_t = jax.jvp(raw_at_times, (times,), (jnp.ones_like(times),))
            fields, ft = jax.jvp(model.constraint, (raw,), (raw_t,))
            phi_j = fields[..., 0]
            native_q = fields[..., 1] - (-2 * phi_j**3 + 3 * phi_j**2) * dc - cle
            _, native_ch = compute_pde(model, x, times, sub, nd, "phase", .5)
            return raw, ft[..., 1], native_q, native_ch, fields

        raw, ct, native_q, native_ch, reloaded_fields = (np.asarray(a) for a in checkpoint_values(net))
        saved_terminal = np.stack([phi[-1],c[-1]],axis=-1)
        reloaded_fields = reloaded_fields.reshape(saved_terminal.shape)
        replay_difference = float(np.max(np.abs(reloaded_fields-saved_terminal)))
        np.testing.assert_allclose(reloaded_fields,saved_terminal,rtol=0,atol=5e-6)
        stable_cl = (1 - config.physical.c_se + cle) * .5 * (1 + np.tanh(raw[0, :, 1].astype(np.float64)))
        stable_q = (stable_cl-cle).reshape(1, *ref["X"].shape)
        stable_dq = graph(stable_q)[0]
        target_dq = graph(target_q[-1:])[0]
        target_ct = -nch * target_dq
        ct = ct[0].astype(np.float64)
        stable_ch = ct + nch * stable_dq
        stable_le = stable_cl - target_cl[-1].ravel()
        native_q_dq = graph(native_q.reshape(1, *ref["X"].shape))[0]
        rms = lambda v: float(np.sqrt(np.mean(v**2)))
        stable_probe = {
            "scope": "CPU raw c_L in float64; c_t from AD of actual output map; reference c_t=-N_CH*Dq on free nodes. Not a separately trained model.",
            "stable_q_relative_l2_free": relative_norm(stable_le[free], target_q[-1].ravel()[free]),
            "stable_cL_relative_l2_free": relative_norm(stable_le[free], target_cl[-1].ravel()[free]),
            "stable_Dq_relative_l2_free": relative_norm((stable_dq-target_dq)[free], target_dq[free]),
            "reference_Dq_rms_free": rms(target_dq[free]),
            "stable_Dq_error_rms_free": rms((stable_dq-target_dq)[free]),
            "native_q_Dq_difference_from_stable_rms_free": rms((native_q_dq-stable_dq)[free]),
            "reference_c_t_rms_free": rms(target_ct[free]),
            "predicted_c_t_error_relative_l2_free": relative_norm((ct-target_ct)[free], target_ct[free]),
            "stable_full_CH_rms_free": rms(stable_ch[free]),
            "native_full_CH_rms_free": rms(native_ch[:, 0][free]),
            "stable_CH_rms_over_reference_c_t_rms": rms(stable_ch[free])/rms(target_ct[free]),
        }
        scopes = {}
        for label, indices in (("all_times", np.arange(len(phi))), ("terminal", np.array([len(phi)-1]))):
            p, t = phi[indices], target_phi[indices]
            ce, ge, le = total_error[indices], geometry_error[indices], cl_error[indices]
            targetc, targetcl, targetq = target_c[indices], target_cl[indices], target_q[indices]
            free_mask = np.broadcast_to(free.reshape(*ref["X"].shape), t.shape)
            bulk = ((t < .01) | (t > .99)) & free_mask
            interface = (t >= .01) & (t <= .99) & free_mask
            entries = {}
            for region, mask in (("all_nodes", np.ones_like(t, dtype=bool)),
                                 ("free_nodes", free_mask), ("bulk", bulk), ("interface", interface)):
                a, b = ge[mask], le[mask]
                norm_product = np.linalg.norm(a) * np.linalg.norm(b)
                cosine = float(np.dot(a, b) / norm_product) if norm_product > 0 else None
                entries[region] = {
                    "count": int(mask.sum()),
                    "phi_relative_l2": relative_norm((p-t)[mask], t[mask]),
                    "c_relative_l2": relative_norm(ce[mask], targetc[mask]),
                    "cL_relative_l2": relative_norm(le[mask], targetcl[mask]),
                    "q_relative_l2": relative_norm(le[mask], targetq[mask]),
                    "c_rmse": float(np.sqrt(np.mean(ce[mask]**2))),
                    "cL_q_rmse": float(np.sqrt(np.mean(le[mask]**2))),
                    "cL_q_rmse_over_c_le": float(np.sqrt(np.mean(le[mask]**2)) / cle),
                    "geometry_part_rmse": float(np.sqrt(np.mean(a**2))),
                    "geometry_concentration_error_cosine": cosine,
                    "cL_reference_rms": float(np.sqrt(np.mean(targetcl[mask]**2))),
                    "q_reference_rms": float(np.sqrt(np.mean(targetq[mask]**2))),
                }
            entries["transport_operator_relative_l2_free_nodes"] = relative_norm(dq_error[indices][:, free], dq_ref[indices][:, free])
            scopes[label] = entries
        saved_fields = np.stack([phi,c],axis=-1)
        physical = field_metrics(saved_fields,ref)
        summary = json.loads((run/"summary.json").read_text())
        for key in ("terminal_rel_l2_phi","terminal_rel_l2_c","pooled_rel_l2_phi","pooled_rel_l2_c"):
            np.testing.assert_allclose(physical[key],summary["final_metrics"][key],rtol=1e-10)
        raw_cl_gradient = (1-config.physical.c_se+cle)*.5*(1-np.asarray(jnp.tanh(jnp.asarray(raw[0,:,1])))**2)
        record = {"path": str(run), "checks": checks,
                  "cpu_terminal_replay_max_abs_difference": replay_difference,
                  "physical_metrics": physical,
                  "boundary_max_abs_error": boundary_error(saved_fields,zero,one),
                  "ic_phi_rmse": float(np.sqrt(np.mean((phi[0]-target_phi[0])**2))),
                  "ic_c_rmse": float(np.sqrt(np.mean((c[0]-target_c[0])**2))),
                  "terminal_zero_concentration_derivative_fraction": float(np.mean(raw_cl_gradient==0)), "stagger_steps": config.loss.stagger_steps,
                  "stagger_switch_every": config.loss.stagger_switch_every,
                  "physical_c_le": cle, "metrics": scopes,
                  "stable_raw_checkpoint_transport_probe": stable_probe,
                  "note": "cL and q reconstructed in float64 from saved FP32 fields; decomposition components are correlated, not additive error fractions. Dq comparison is transport operator only, not full CH residual."}
        result["runs"][name] = record
        predictions[name] = saved_fields
        histories[name] = [json.loads(line) for line in (run/"metrics.jsonl").read_text().splitlines()]
        terminal = scopes["terminal"]["all_nodes"]
        print(name, "c/cL/q relative L2", terminal["c_relative_l2"], terminal["cL_relative_l2"], terminal["q_relative_l2"],
              "cL RMSE/c_le", terminal["cL_q_rmse_over_c_le"], flush=True)
    output = Path(__file__)
    output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    fig, axes = plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for ax, field, label in zip(axes.ravel(),("phi","c","q","Dq_free"),("Phase","Concentration","Driving field q","Graph transport Dq")):
        for name, history in histories.items():
            ax.semilogy([r["step"] for r in history],[r[f"terminal_rel_l2_{field}"] for r in history],label=name)
        ax.set(xlabel="Completed updates",ylabel=f"50 s {label} relative L2")
        ax.axvline(1000,color="gray",linewidth=.8,alpha=.5)
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.savefig(output.with_suffix(".png"),dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    for ax,(name,fields) in zip(axes.ravel(),predictions.items()):
        image=ax.pcolormesh(ref["X"],ref["Y"],fields[-1,...,0],vmin=0,vmax=1,shading="nearest")
        ax.contour(ref["X"],ref["Y"],ref["phi"][-1],levels=[.5],colors="white",linewidths=.8)
        ax.set(title=name,xlabel="x*",ylabel="y*",aspect="equal")
    fig.colorbar(image,ax=axes.ravel().tolist(),label="Phase at 50 s; white=reference interface")
    fig.savefig(output.with_name(output.stem+"-fields.png"),dpi=150);plt.close(fig)


if __name__ == "__main__":
    main()
