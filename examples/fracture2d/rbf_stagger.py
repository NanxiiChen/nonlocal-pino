"""Target-free RBF-enriched neural stagger pilot on the frozen physical graph.

The inherited network is a frozen base; localized Gaussian hidden units have
trainable readout weights. Mechanics weights minimize the existing fixed-d
A2 energy in this neural subspace, not a reference field or nodal exact solve.
Damage weights minimize the checked fixed-H energy with the existing smooth
nonnegative activation. Raw physical residuals are always reported.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict
from scipy.linalg import lstsq
from scipy.optimize import least_squares, minimize
from scipy.special import expit

from examples.fracture2d.coupling_diagnostics import metrics
from examples.fracture2d import nonlocal_baseline as direct
from examples.fracture2d.diagnostic_continuation import _strict_parent
from examples.fracture2d.fixed_load_sweeps import _experiment_configs
from examples.fracture2d.fp64 import FP64A2Backend
from examples.fracture2d.inner_probe import load_snapshot
from examples.fracture2d.losses import FractureResidual
from examples.fracture2d.pinn_paths import physical_pinn_path


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    os.replace(temporary, path)


def run(args):
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    config, model, initial = load_snapshot(args.initial, args.parent_digest)
    source_path = physical_pinn_path(args.source)
    source = ConfigDict(json.loads((source_path / "config.json").read_text()))
    validation, _ = _experiment_configs(source, args.source, args.output, 36,
        tensorboard_root=Path(args.output).parent / "events")
    parent, parent_digest = _strict_parent(validation, model)
    if parent_digest != args.parent_digest:
        raise ValueError("RBF parent digest differs from verified strict state 59")
    output = Path(args.output)
    if not output.parent.is_dir():
        raise FileNotFoundError(output.parent)
    output.mkdir()
    policy = vars(args).copy()
    policy = {k: str(v) if isinstance(v, Path) else v for k, v in policy.items()}
    policy.update({"schema": "fracture2d-rbf-stagger-pilot-v1",
                   "neural_basis": "frozen base MLP plus trainable Gaussian hidden-unit readouts",
                   "mechanics_objective": args.mechanics_objective,
                   "damage_objective": args.damage_objective,
                   "reference_targets": False, "reduced_svd_rtol": 1e-12,
                   "damage_maxiter": 1000, "damage_residual_max_nfev": 100, "omega": 0.5,
                   "converged_required_consecutive_sweeps": 3,
                   "at2_relative_inf_limit": 1e-3,
                   "damage_change_inf_limit": 1e-4,
                   "H_change_relative_inf_limit": 1e-3,
                   "q_change_relative_inf_limit": 1e-3,
                   "reaction_change_relative_limit": 5e-3,
                   "mechanics_normalized_inf_limit": 0.1,
                   "mechanics_loss_limit": 1e-2,
                   "healing_atol": 1e-3, "range_atol": 1e-10,
                   "status": "research diagnostic; no strict accepted states"})
    atomic_json(output / "policy.json", policy)
    backend = FP64A2Backend(model.static, model.constraints, jax.devices()[0])
    residual = FractureResidual(model, backend)
    xy = np.asarray(model.geometry.coordinates)
    # Multi-scale geometry-owned centers; no reference fields select centers.
    gx, gy = np.meshgrid(np.linspace(0, 1, 8), np.linspace(0, 1, 8))
    lx, ly = np.meshgrid(np.linspace(0.4, 1.0, 25), np.linspace(0.38, 0.62, 9))
    centers = np.concatenate((np.stack((gx.ravel(), gy.ravel()), axis=1),
                              np.stack((lx.ravel(), ly.ravel()), axis=1)))
    radii = np.concatenate((np.full(gx.size, 0.15), np.full(lx.size, 0.045)))
    if args.basis in ("tip_refined", "tip_collar_whitened", "tip_multiscale_whitened"):
        material_xy = xy[:model.n_material]
        tip = material_xy[np.linalg.norm(material_xy - [0.5, 0.5], axis=1) <= 0.06]
        centers = np.concatenate((centers, tip))
        radii = np.concatenate((radii, np.full(len(tip), 0.008)))
    if args.basis in ("tip_collar_whitened", "tip_multiscale_whitened"):
        collar = xy[((xy[:, 1] >= 0.96) | (xy[:, 1] <= 0.04))][::3]
        centers = np.concatenate((centers, collar))
        radii = np.concatenate((radii, np.full(len(collar), 0.016)))
    if args.basis == "tip_multiscale_whitened":
        narrow = xy[:model.n_material][np.linalg.norm(xy[:model.n_material] - [0.5, 0.5], axis=1) <= 0.10]
        centers = np.concatenate((centers, narrow))
        radii = np.concatenate((radii, np.full(len(narrow), 0.003)))
    phi = np.exp(-np.sum((xy[:, None, :] - centers[None, :, :])**2, axis=2)
                 / (2 * radii[None, :]**2))
    m = phi.shape[1]
    print(json.dumps({"event": "rbf_basis", "hidden_units": m,
                      "mechanics_weights": 2 * m, "free_dofs": len(model.constraints.free_dofs)}), flush=True)
    B = np.zeros((model.n_dofs, 2 * m))
    B[0::2, :m] = phi
    B[1::2, m:] = phi
    B[model.constraints.constrained_dofs] = 0.0
    free = np.asarray(backend.free_dofs)
    Bfree = B[free]
    B_device = jnp.asarray(B)
    phi_d = phi[:model.n_material]
    damage_transform = np.eye(m)
    if args.basis in ("tip_collar_whitened", "tip_multiscale_whitened"):
        U, singular, Vt = np.linalg.svd(phi_d, full_matrices=False)
        retained = singular > singular[0] * 1e-10
        damage_transform = Vt[retained].T / singular[retained]
        phi_d = U[:, retained]
        print(json.dumps({"event": "damage_basis_whitened", "rank": int(np.sum(retained)),
                          "cutoff": 1e-10, "original_columns": m}), flush=True)
    volumes = np.asarray(model.geometry.material_reference_volumes)
    D = residual.operator.to_scipy().tocsr()
    points = jnp.asarray(xy)
    base_q_raw = np.asarray(jax.vmap(initial[0])(points))
    base_d_raw = np.asarray(jax.vmap(initial[2])(points[:model.n_material]))[:, 0]
    sigmoid_base = None
    if args.damage_link == "sigmoid_log_base":
        anchor_raw = (config.physical.E / (config.physical.Gc / config.physical.l0)) * 0.006**2 * base_d_raw
        magnitude = np.sqrt(anchor_raw**2 + config.physical.damage_floor**2)
        positive = anchor_raw >= 0
        anchor_positive = np.empty_like(anchor_raw)
        anchor_positive[positive] = 0.5 * (anchor_raw[positive] + magnitude[positive])
        anchor_positive[~positive] = config.physical.damage_floor**2 / (2 * (magnitude[~positive] - anchor_raw[~positive]))
        if np.any(anchor_positive <= 0) or not np.all(np.isfinite(anchor_positive)):
            raise ValueError("log-base transfer requires finite positive inherited network damage")
        sigmoid_base = np.log(anchor_positive)
    first_load = 1e-4
    affine = np.stack((np.zeros(len(xy)), first_load * xy[:, 1]), axis=1)
    affine = residual.lift(jnp.asarray(affine.reshape(-1)), first_load)
    affine_r = backend._residual_from_factor(
        backend._edge_factor(jnp.zeros(model.n_material, jnp.float64)), affine)[free]
    first_scale = float(jnp.sqrt(jnp.mean(affine_r**2)))
    q_weights = np.zeros(2 * m)
    d_weights = np.zeros(phi_d.shape[1])
    committed_H = np.asarray(parent[6])
    committed_d = np.asarray(parent[5])
    d_iter = np.asarray(initial[5])
    if args.damage_link in ("sigmoid", "sigmoid_log_base"):
        if np.any(d_iter <= 0) or np.any(d_iter >= 1):
            raise ValueError("sigmoid transfer requires an interior initial PINN damage field")
        anchor_base = (sigmoid_base if args.damage_link == "sigmoid_log_base" else
            (config.physical.E / (config.physical.Gc / config.physical.l0)) * 0.006**2 * base_d_raw)
        logit = np.log(d_iter) - np.log1p(-d_iter)
        d_weights, _, _, _ = lstsq(phi_d, logit - anchor_base, cond=1e-12, lapack_driver="gelsd")
    if args.fork_rbf is not None:
        fork = Path(args.fork_rbf)
        fork_policy = json.loads((fork / "policy.json").read_text())
        fork_result = json.loads((fork / "result.json").read_text())
        previous_dirs = sorted(fork.glob("load_*"))
        if args.fork_load_index is not None:
            selected_dir = fork / f"load_{args.fork_load_index:04d}"
            selected_reports = sorted(selected_dir.glob("sweep_*.json"))
            if not selected_reports:
                raise ValueError("selected RBF fork load has no reports")
            selected = json.loads(selected_reports[-1].read_text())
        else:
            if not fork_result["complete"]:
                raise ValueError("failed RBF root requires an explicit converged fork-load-index")
            selected_dir = previous_dirs[-1]
            selected = fork_result["last"]
        if selected["consecutive"] < 3 or not all(selected["gates"].values()):
            raise ValueError("RBF fork requires a completed coupled-checked source load")
        for key in ("basis", "mechanics_objective", "damage_objective", "parent_digest", "initial"):
            if fork_policy[key] != policy[key]:
                raise ValueError(f"RBF fork changes model/ansatz identity: {key}")
        if args.damage_link != "positive":
            if "damage_link" not in fork_policy or fork_policy["damage_link"] != args.damage_link:
                raise ValueError("RBF fork cannot change the damage link")
        previous_load = selected["load"]
        if abs(args.first_load - previous_load - args.load_step) > 1e-12:
            raise ValueError("RBF fork must begin at the next declared load")
        with np.load(fork / "basis.npz", allow_pickle=False) as archive:
            for key, value in (("centers", centers), ("radii", radii), ("damage_transform", damage_transform)):
                np.testing.assert_allclose(archive[key], value, rtol=1e-12, atol=1e-12)
        field_path = selected_dir / "latest.npz"
        with np.load(field_path, allow_pickle=False) as archive:
            q_weights, d_weights = archive["q_weights"], archive["d_weights"]
            d_iter = archive["d"]
            committed_d, committed_H = archive["d"].copy(), archive["H"].copy()
            q_saved = archive["q"]
        base = previous_load * (np.stack((np.zeros(len(xy)), xy[:, 1]), axis=1) + base_q_raw)
        q_rebuilt = np.asarray(residual.lift(jnp.asarray(base.reshape(-1)), previous_load)) + previous_load * B @ q_weights
        np.testing.assert_allclose(q_rebuilt, q_saved, rtol=1e-10, atol=1e-12)
        fork_fields = residual.evaluate(jnp.asarray(q_saved), jnp.asarray(d_iter), None, jnp.asarray(committed_H))
        np.testing.assert_allclose(fork_fields.trial_history, committed_H, rtol=1e-12, atol=1e-12)
        atomic_json(output / "fork_identity.json", {
            "source": str(fork), "field_sha256": hashlib.sha256(field_path.read_bytes()).hexdigest(),
            "policy_sha256": hashlib.sha256((fork / "policy.json").read_bytes()).hexdigest(),
            "last_load": previous_load, "strict_acceptance": False})
    np.savez_compressed(output / "basis.npz", centers=centers, radii=radii,
                        damage_transform=damage_transform)
    eqx.tree_serialise_leaves(output / "base_networks.eqx", (initial[0], initial[2]))

    @eqx.filter_jit
    def tangent_columns(factor, columns):
        return jax.vmap(lambda column: backend._tangent_from_factor(factor, column),
                        in_axes=1, out_axes=1)(columns)

    @eqx.filter_jit
    def q_residual(factor, q):
        return backend._residual_from_factor(factor, q)

    def solve_q(d, load):
        nonlocal q_weights
        base = load * (np.stack((np.zeros(len(xy)), xy[:, 1]), axis=1) + base_q_raw)
        q0 = np.asarray(residual.lift(jnp.asarray(base.reshape(-1)), load))
        factor = backend._edge_factor(jnp.asarray(d))
        RB = np.empty_like(Bfree)
        # Fixed four-column batches bound compiled temporary sizes.
        for start in range(0, 2 * m, 4):
            count = min(4, 2 * m - start)
            columns = B_device[:, start:start + count]
            if count < 4:
                columns = jnp.pad(columns, ((0, 0), (0, 4 - count)))
            RB[:, start:start + count] = np.asarray(tangent_columns(factor, columns))[free, :count]
        reduced = Bfree.T @ RB
        rhs = -(Bfree.T @ np.asarray(q_residual(factor, jnp.asarray(q0)))[free]) / load
        symmetry = np.max(np.abs(reduced - reduced.T)) / np.max(np.abs(reduced))
        if symmetry > 1e-10:
            raise ValueError(f"RBF energy tangent is not symmetric: {symmetry}")
        if args.mechanics_objective == "residual":
            q_weights, _, rank, singular = lstsq(RB,
                -np.asarray(q_residual(factor, jnp.asarray(q0)))[free] / load,
                cond=1e-12, lapack_driver="gelsd")
        else:
            q_weights, _, rank, singular = lstsq(reduced, rhs, cond=1e-12, lapack_driver="gelsd")
        q = q0 + load * B @ q_weights
        r = np.asarray(q_residual(factor, jnp.asarray(q)))
        return q, r, int(rank), float(singular[-1] / singular[0])

    def activation(raw):
        if args.damage_link in ("sigmoid", "sigmoid_log_base"):
            d = expit(raw)
            return d, d * (1 - d)
        floor = config.physical.damage_floor
        magnitude = np.sqrt(raw**2 + floor**2)
        positive = raw >= 0
        # Stable algebraic form of the existing one-sided activation.
        d = np.empty_like(raw)
        d[positive] = 0.5 * (raw[positive] + magnitude[positive])
        d[~positive] = floor**2 / (2 * (magnitude[~positive] - raw[~positive]))
        derivative = np.empty_like(raw)
        derivative[positive] = 0.5 * (1 + raw[positive] / magnitude[positive])
        derivative[~positive] = floor**2 / (2 * magnitude[~positive] * (magnitude[~positive] - raw[~positive]))
        return d, derivative

    def solve_d(H, load):
        nonlocal d_weights
        damage_load = 0.006 if args.damage_link in ("sigmoid", "sigmoid_log_base") else load
        base = (sigmoid_base if args.damage_link == "sigmoid_log_base" else
            (config.physical.E / (config.physical.Gc / config.physical.l0)) * damage_load**2 * base_d_raw)
        def objective(weights):
            d, derivative = activation(base + phi_d @ weights)
            Dd = D @ d
            rho = config.physical.Gc / config.physical.l0 * d + config.physical.Gc * config.physical.l0 * Dd + 2 * H * (d - 1)
            energy = np.sum(volumes * (0.5 * config.physical.Gc / config.physical.l0 * d**2
                + 0.5 * config.physical.Gc * config.physical.l0 * d * Dd + H * (d - 1)**2))
            gradient = phi_d.T @ (volumes * rho * derivative)
            return energy, gradient
        if args.damage_objective == "residual":
            from scipy.sparse import diags
            A = (config.physical.Gc * config.physical.l0 * D
                 + diags(config.physical.Gc / config.physical.l0 + 2 * H))
            scale = 2 * np.sqrt(np.mean(H**2))
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError("fixed-H residual scale must be positive")
            def rho_at(weights):
                d, _ = activation(base + phi_d @ weights)
                return (A @ d - 2 * H) / scale
            def jacobian(weights):
                _, derivative = activation(base + phi_d @ weights)
                return A @ (derivative[:, None] * phi_d) / scale
            solved = least_squares(rho_at, d_weights, jac=jacobian,
                method="trf", max_nfev=100, ftol=1e-12, xtol=1e-12, gtol=1e-10,
                x_scale="jac")
            iterations = solved.nfev
        else:
            solved = minimize(objective, d_weights, jac=True, method="L-BFGS-B",
                options={"maxiter": 1000, "maxls": 40, "maxcor": 30, "ftol": 1e-15, "gtol": 1e-10})
            iterations = solved.nit
        d_weights = solved.x
        d, _ = activation(base + phi_d @ d_weights)
        return d, {"iterations": int(iterations), "message": str(solved.message),
                   "success": bool(solved.success)}

    previous = None
    consecutive = 0
    all_reports = []
    loads = np.arange(args.first_load, args.last_load + args.load_step * 0.5, args.load_step)
    for load_index, load in enumerate(loads):
        state_dir = output / f"load_{load_index:04d}"
        state_dir.mkdir()
        for sweep in range(1, args.max_sweeps + 1):
            q_before, _, _, _ = solve_q(d_iter, load)
            H_used = np.asarray(residual.evaluate(jnp.asarray(q_before), jnp.asarray(d_iter), None,
                                                 jnp.asarray(committed_H)).trial_history)
            d_solved, optimizer = solve_d(H_used, load)
            d = d_iter + 0.5 * (d_solved - d_iter)
            q, r, rank, singular_ratio = solve_q(d, load)
            fields = residual.evaluate(jnp.asarray(q), jnp.asarray(d), None, jnp.asarray(committed_H))
            H = np.asarray(fields.trial_history)
            local = config.physical.Gc / config.physical.l0 * d
            nonlocal_term = config.physical.Gc * config.physical.l0 * (D @ d)
            history = 2 * H * (d - 1)
            rho = local + nonlocal_term + history
            np.testing.assert_allclose(rho, np.asarray(fields.damage), rtol=1e-8, atol=1e-10)
            scale_q = first_scale * load / first_load
            healing = float(np.max(np.maximum(committed_d - d, 0)))
            drift = int(np.count_nonzero(np.linalg.norm(q[:2 * model.n_material].reshape(-1, 2), axis=1) / load > direct.DRIFT_ENVELOPE))
            gates = {"finite": all(np.all(np.isfinite(v)) for v in (q, d, H, r, rho)),
                     "mechanics_loss": float(np.mean((r[free] / scale_q)**2)) <= 1e-2,
                     "mechanics_inf": float(np.max(np.abs(r[free] / scale_q))) <= 0.1,
                     "damage_range": np.min(d) >= -1e-10 and np.max(d) <= 1.0,
                     "healing": healing <= 1e-3,
                     "history_monotonic": np.min(H - committed_H) >= 0,
                     "drift": drift == 0}
            report = {"sweep": sweep, "load": float(load), "reaction": float(fields.reaction),
                      "damage_inf": float(np.max(np.abs(rho))),
                      "mechanics_loss": float(np.mean((r[free] / scale_q)**2)),
                      "gates": {k: bool(v) for k, v in gates.items()}}
            data = {"q": q, "d": d, "H": H, "H_used": H_used,
                    "at2_local": local, "at2_nonlocal": nonlocal_term,
                    "at2_history": history,
                    "at2_fixed_relaxed": local + nonlocal_term + 2 * H_used * (d - 1),
                    "at2_fixed_solved": config.physical.Gc / config.physical.l0 * d_solved
                        + config.physical.Gc * config.physical.l0 * (D @ d_solved) + 2 * H_used * (d_solved - 1),
                    "coordinates": xy[:model.n_material]}
            diag = metrics(report, data, previous)
            changes = diag["adjacent_sweep"]
            coupled_ok = changes is not None and diag["at2_relative_inf"] <= 1e-3 and (
                changes["damage_change_inf"] <= 1e-4 and changes["H_change_relative_inf"] <= 1e-3
                and changes["q_change_relative_inf"] <= 1e-3 and changes["reaction_change_relative"] <= 5e-3)
            consecutive = consecutive + 1 if coupled_ok and all(gates.values()) else 0
            record = {**report, **diag, "consecutive": consecutive, "damage_max": float(np.max(d)),
                      "healing": healing, "drift": drift, "reduced_rank": rank,
                      "reduced_singular_ratio": singular_ratio, "damage_optimizer": optimizer}
            print(json.dumps(record), flush=True)
            atomic_json(state_dir / f"sweep_{sweep:04d}.json", record)
            np.savez_compressed(state_dir / "latest.npz", **data, q_weights=q_weights, d_weights=d_weights)
            all_reports.append(record)
            atomic_json(output / "reports.json", all_reports)
            previous = data, report
            d_iter = d
            if consecutive >= 3:
                break
        if consecutive < 3:
            atomic_json(output / "result.json", {"complete": False, "reason": "coupled gates failed within budget", "last": record})
            return
        committed_d, committed_H = d.copy(), H.copy()
        previous, consecutive = None, 0
    atomic_json(output / "result.json", {"complete": True, "last": record})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent-digest", required=True)
    parser.add_argument("--first-load", type=float, required=True)
    parser.add_argument("--last-load", type=float, required=True)
    parser.add_argument("--load-step", type=float, required=True)
    parser.add_argument("--max-sweeps", type=int, required=True)
    parser.add_argument("--basis", choices=("coarse", "tip_refined", "tip_collar_whitened", "tip_multiscale_whitened"), required=True)
    parser.add_argument("--mechanics-objective", choices=("energy", "residual"), required=True)
    parser.add_argument("--damage-objective", choices=("energy", "residual"), required=True)
    parser.add_argument("--fork-rbf", type=Path)
    parser.add_argument("--fork-load-index", type=int)
    parser.add_argument("--damage-link", choices=("positive", "sigmoid", "sigmoid_log_base"), required=True)
    args = parser.parse_args()
    if not (0 < args.first_load <= args.last_load and args.load_step > 0 and args.max_sweeps > 0):
        raise ValueError("invalid explicit RBF load/budget settings")
    if args.first_load != 0.006 and args.fork_rbf is None:
        raise ValueError("RBF pilot must first re-equilibrate state 60 from its original state-59 context")
    run(args)


if __name__ == "__main__":
    main()
