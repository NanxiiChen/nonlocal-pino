"""Independent fixed-field direct audit; these fields are never training targets."""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from examples.fracture2d._legacy import nonlocal_baseline as direct
from examples.fracture2d._legacy.fp64 import FP64A2Backend
from examples.fracture2d._legacy.inner_probe import load_snapshot
from examples.fracture2d._legacy.losses import FractureResidual


def plot_sweeps(sweeps, load, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    x = [sample["sweep"] for sample in sweeps]
    axes[0, 0].plot(x, [sample["reaction"] for sample in sweeps], "o-")
    axes[0, 0].set_ylabel("Reaction; oracle diagnostic only")
    axes[0, 1].semilogy(x, [sample["damage_inf"] for sample in sweeps], "o-")
    axes[0, 1].set_ylabel("Coupled AT2 residual infinity norm")
    axes[1, 0].semilogy(x, [sample["history_max"] for sample in sweeps], "o-")
    axes[1, 0].set_ylabel("Maximum current history")
    axes[1, 1].plot(x, [sample["max_displacement_ratio"] for sample in sweeps], "o-")
    axes[1, 1].axhline(direct.DRIFT_ENVELOPE, color="red", linestyle="--", label="All-material drift gate")
    axes[1, 1].set_ylabel("Maximum material displacement / load")
    axes[1, 1].legend()
    for axis in axes.flat:
        axis.set_xlabel("Fixed-load oracle sweep")
        axis.grid(alpha=0.3)
    figure.suptitle(f"Fixed load {load:.4f}: exact inner solves, unconverged coupled diagnostic")
    figure.savefig(output / "oracle_coupling_summary.png", dpi=150)
    plt.close(figure)


def run(args):
    if args.plot_only:
        audit = json.loads((args.output / "audit.json").read_text())
        sweeps = json.loads((args.output / "oracle_sweeps.json").read_text())
        if len(sweeps) != args.oracle_sweeps or not all(
                sample["diagnostic_only"] and not sample["training_targets"] for sample in sweeps):
            raise ValueError("plot-only requires the recorded diagnostic sweep budget")
        plot_sweeps(sweeps, audit["load"], args.output)
        return
    config, model, _ = load_snapshot(args.base, args.parent_digest)
    with np.load(args.trial / "fields.npz", allow_pickle=False) as archive:
        q, d, H = (archive[name] for name in ("q", "d", "H"))
    with np.load(args.parent / "fields.npz", allow_pickle=False) as archive:
        old_d, old_H = archive["d"], archive["H"]
    policy = json.loads((args.trial / "policy.json").read_text())
    load = policy["load"]
    backend = FP64A2Backend(model.static, model.constraints, jax.devices()[0])
    residual = FractureResidual(model, backend)
    mechanical = backend.solve(d, load, 1e-9, 1e-10, 10000, initial_guess=q)
    q_exact = np.asarray(mechanical.displacement)
    updated = residual.evaluate(jnp.asarray(q_exact), jnp.asarray(d), None, jnp.asarray(old_H))
    H_exact_q = np.asarray(updated.trial_history)
    D = residual.operator.to_scipy().tocsr()
    gc0, gc1 = config.physical.Gc / config.physical.l0, config.physical.Gc * config.physical.l0
    def damage_solve(history):
        A = gc1 * D + diags(gc0 + 2 * history)
        solved = spsolve(A, 2 * history)
        rho = A @ solved - 2 * history
        if not np.all(np.isfinite(solved)) or max(abs(rho)) > 1e-7:
            raise ValueError("independent fixed-H solve failed its residual check")
        return solved, float(max(abs(rho)))
    d_exact_H, rho_H = damage_solve(H)
    d_exact_q, rho_q = damage_solve(H_exact_q)
    n = model.n_material
    xy = np.asarray(model.geometry.coordinates)
    ratios = np.linalg.norm(q_exact[:2*n].reshape(n, 2), axis=1) / load
    order = np.argsort(ratios)[-10:][::-1]
    report = {"diagnostic_only": True, "training_targets": False, "load": load,
        "cg_explicit_converged": bool(mechanical.explicit_converged),
        "cg_status_code": int(mechanical.status_code), "cg_iterations": int(mechanical.cg_iterations),
        "cg_relative_residual": float(mechanical.free_residual_relative_norm),
        "fixed_d_direct_reaction": float(mechanical.top_reaction),
        "fixed_d_direct_drift_count": int(np.count_nonzero(ratios > direct.DRIFT_ENVELOPE)),
        "fixed_d_direct_max_displacement_ratio": float(max(ratios)),
        "q_neural_vs_fixed_d_direct_relative_l2": float(np.linalg.norm(q[:2*n] - q_exact[:2*n]) / np.linalg.norm(q_exact[:2*n])),
        "fixed_H_damage_residual_inf": rho_H,
        "fixed_H_damage_neural_max_error": float(max(abs(d - d_exact_H))),
        "fixed_H_exact_damage_healing": float(max(np.maximum(old_d - d_exact_H, 0))),
        "fixed_H_exact_damage_max": float(max(d_exact_H)),
        "updated_direct_q_history_max": float(max(H_exact_q)),
        "updated_direct_q_damage_residual_inf": rho_q,
        "updated_direct_q_damage_max": float(max(d_exact_q)),
        "largest_displacement_nodes": [{"node": int(i), "coordinate": xy[i].tolist(),
                                       "ratio": float(ratios[i])} for i in order]}
    if not args.output.parent.is_dir():
        raise FileNotFoundError(args.output.parent)
    args.output.mkdir()
    (args.output / "audit.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(args.output / "diagnostic_fields.npz", q_fixed_d_direct=q_exact,
                        d_fixed_H_direct=d_exact_H, H_after_direct_q=H_exact_q, d_after_direct_q=d_exact_q)
    print(json.dumps(report, indent=2), flush=True)
    sweeps = []
    d_iter, q_guess = d.copy(), q_exact
    for sweep in range(1, args.oracle_sweeps + 1):
        before = backend.solve_strict(d_iter, load, 1e-9, 1e-10, 10000, initial_guess=q_guess)
        H_used = np.asarray(residual.evaluate(before.displacement, jnp.asarray(d_iter), None,
                                             jnp.asarray(old_H)).trial_history)
        solved_d, _ = damage_solve(H_used)
        trial_d = 0.5 * (d_iter + solved_d)
        after = backend.solve_strict(trial_d, load, 1e-9, 1e-10, 10000,
                                     initial_guess=np.asarray(before.displacement))
        q_guess = np.asarray(after.displacement)
        fields = residual.evaluate(after.displacement, jnp.asarray(trial_d), None, jnp.asarray(old_H))
        trial_H = np.asarray(fields.trial_history)
        ratios = np.linalg.norm(q_guess[:2*n].reshape(n, 2), axis=1) / load
        sample = {"sweep": sweep, "diagnostic_only": True, "training_targets": False,
                  "reaction": float(fields.reaction), "damage_max": float(max(trial_d)),
                  "damage_inf": float(max(abs(np.asarray(fields.damage)))),
                  "damage_change_inf": float(max(abs(trial_d - d_iter))),
                  "history_max": float(max(trial_H)),
                  "history_change_inf": float(max(abs(trial_H - H_used))),
                  "drift_count": int(np.count_nonzero(ratios > direct.DRIFT_ENVELOPE)),
                  "max_displacement_ratio": float(max(ratios))}
        if not all(np.isfinite(value) for value in sample.values()):
            raise FloatingPointError("oracle diagnostic produced nonfinite fields")
        sweeps.append(sample)
        print(json.dumps(sample), flush=True)
        (args.output / "oracle_sweeps.json").write_text(json.dumps(sweeps, indent=2))
        np.savez_compressed(args.output / "oracle_latest.npz", q=q_guess, d=trial_d, H=trial_H)
        d_iter = trial_d
    if sweeps:
        plot_sweeps(sweeps, load, args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent-digest", required=True)
    parser.add_argument("--oracle-sweeps", type=int, required=True)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.oracle_sweeps < 0:
        raise ValueError("oracle-sweeps must be nonnegative")
    run(args)


if __name__ == "__main__":
    main()
