"""Matrix-free joint physical-residual optimization of the enriched PINN.

History is the unchanged accepted maximum rule, evaluated from the current
displacement and a frozen committed PINN parent. No reference fields enter
training. This probes an unstable outer Picard branch without modifying PDEs.
"""

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse.linalg import LinearOperator

from examples.fracture2d import nonlocal_baseline as direct
from examples.fracture2d.fp64 import FP64A2Backend
from examples.fracture2d.inner_probe import load_snapshot
from examples.fracture2d.losses import FractureResidual
from examples.fracture2d.rbf_stagger import atomic_json


def run(args):
    config, model, initial = load_snapshot(args.base, args.parent_digest)
    source_policy = json.loads((args.source / "policy.json").read_text())
    if source_policy["damage_link"] != "sigmoid_log_base":
        raise ValueError("joint probe requires the bounded log-base parent")
    source_dir = args.source / f"load_{args.source_index:04d}"
    source_report = json.loads(sorted(source_dir.glob("sweep_*.json"))[-1].read_text())
    if source_report["consecutive"] < 3 or not all(source_report["gates"].values()):
        raise ValueError("joint source must be a coupled-checked research parent")
    with np.load(args.source / "basis.npz", allow_pickle=False) as archive:
        centers, radii, damage_transform = (archive[name] for name in ("centers", "radii", "damage_transform"))
    field_path = source_dir / "latest.npz"
    with np.load(field_path, allow_pickle=False) as archive:
        old_q, old_d, old_H = (archive[name] for name in ("q", "d", "H"))
        q_weights, d_weights = archive["q_weights"], archive["d_weights"]
    if args.load <= source_report["load"]:
        raise ValueError("joint target load must exceed its committed source load")
    xy = np.asarray(model.geometry.coordinates)
    phi = np.exp(-np.sum((xy[:, None, :] - centers[None, :, :])**2, axis=2)
                 / (2 * radii[None, :]**2))
    m = phi.shape[1]
    backend = FP64A2Backend(model.static, model.constraints, jax.devices()[0])
    residual = FractureResidual(model, backend)
    free = np.asarray(backend.free_dofs)
    q_centers, q_radii = centers.copy(), radii.copy()
    feature_q, feature_d = old_q, old_d
    if args.joint_parent is not None:
        with np.load(args.joint_parent / "fields.npz", allow_pickle=False) as archive:
            feature_q, feature_d = archive["q"], archive["d"]
            if "q_centers" in archive.files:
                q_centers, q_radii = archive["q_centers"], archive["q_radii"]
    if args.q_feature_source == "warm_trial":
        if args.warm_trial is None:
            raise ValueError("warm-trial feature selection requires a warm trial")
        with np.load(args.warm_trial / "fields.npz", allow_pickle=False) as archive:
            feature_q, feature_d = archive["q"], archive["d"]
            if "q_centers" in archive.files:
                q_centers, q_radii = archive["q_centers"], archive["q_radii"]
    if args.extra_q_units:
        r = np.asarray(backend._residual_from_factor(backend._edge_factor(jnp.asarray(feature_d)),
                                                    jnp.asarray(feature_q))).copy()
        r[model.constraints.constrained_dofs] = 0
        order = np.argsort(np.linalg.norm(r.reshape(-1, 2), axis=1))[::-1]
        additions = []
        for node in order:
            duplicate = np.any((np.linalg.norm(q_centers - xy[node], axis=1) < 1e-12)
                               & (abs(q_radii - 0.003) < 1e-12))
            if not duplicate:
                additions.append(xy[node])
            if len(additions) == args.extra_q_units:
                break
        if len(additions) != args.extra_q_units:
            raise ValueError("insufficient distinct residual-hotspot feature centers")
        q_centers = np.concatenate((q_centers, np.asarray(additions)))
        q_radii = np.concatenate((q_radii, np.full(len(additions), 0.003)))
    q_phi = np.exp(-np.sum((xy[:, None, :] - q_centers[None, :, :])**2, axis=2)
                   / (2 * q_radii[None, :]**2))
    qm = q_phi.shape[1]

    def extend_weights(weights, old_count):
        return np.concatenate((weights[:old_count], np.zeros(qm - old_count),
                               weights[old_count:], np.zeros(qm - old_count)))

    q_weights = extend_weights(q_weights, m)
    B = np.zeros((model.n_dofs, 2 * qm))
    B[0::2, :qm], B[1::2, qm:] = q_phi, q_phi
    B[model.constraints.constrained_dofs] = 0
    U, singular, Vt = np.linalg.svd(B[free], full_matrices=False)
    retained = singular > singular[0] * 1e-10
    q_transform = Vt[retained].T / singular[retained]
    QB = jnp.asarray(B @ q_transform)
    QD = jnp.asarray(phi[:model.n_material] @ damage_transform)
    nq = int(np.sum(retained))
    # Transform only the existing PINN's own coefficients, never reference data.
    q_initial = singular[retained] * (Vt[retained] @ q_weights)
    parent_vector = None
    if args.joint_parent is not None:
        parent_policy = json.loads((args.joint_parent / "policy.json").read_text())
        parent_result = json.loads((args.joint_parent / "result.json").read_text())
        if not parent_result["physical_checks_pass"] or not parent_result["optimizer_success"]:
            raise ValueError("joint continuation requires a checked completed neural parent")
        for key, value in (("base", str(args.base)), ("source", str(args.source)),
                           ("source_index", args.source_index), ("parent_digest", args.parent_digest)):
            if parent_policy[key] != value:
                raise ValueError(f"joint parent differs in foundation identity: {key}")
        field_path = args.joint_parent / "fields.npz"
        with np.load(field_path, allow_pickle=False) as archive:
            previous_q_transform = archive["q_transform"]
            np.testing.assert_allclose(archive["damage_transform"], damage_transform, rtol=1e-12, atol=1e-12)
            old_q, old_d, old_H = (archive[name] for name in ("q", "d", "H"))
            parent_vector = archive["weights"]
        if args.load <= parent_result["load"]:
            raise ValueError("joint continuation load must increase")
        source_report = parent_result
        previous_nq = previous_q_transform.shape[1]
        q_weights = extend_weights(previous_q_transform @ parent_vector[:previous_nq],
                                   previous_q_transform.shape[0] // 2)
        q_initial = singular[retained] * (Vt[retained] @ q_weights)
        d_weights = parent_vector[previous_nq:]
    points = jnp.asarray(xy)
    raw_q = jax.vmap(initial[0])(points)
    raw_d = jax.vmap(initial[2])(points[:model.n_material])[:, 0]
    anchor_raw = (config.physical.E / (config.physical.Gc / config.physical.l0)) * 0.006**2 * raw_d
    magnitude = jnp.sqrt(anchor_raw**2 + config.physical.damage_floor**2)
    anchor_positive = jnp.where(anchor_raw >= 0, 0.5 * (anchor_raw + magnitude),
        config.physical.damage_floor**2 / (2 * (magnitude - anchor_raw)))
    log_base = jnp.log(anchor_positive)
    committed = jnp.asarray(old_H)

    def q_base(load):
        base = load * (jnp.stack((jnp.zeros_like(points[:, 1]), points[:, 1]), axis=1) + raw_q)
        return residual.lift(base.reshape(-1), load)

    rebuilt = np.asarray(q_base(source_report["load"]) + source_report["load"] * B @ q_weights)
    np.testing.assert_allclose(rebuilt, old_q, rtol=1e-10, atol=1e-12)
    if parent_vector is not None:
        rebuilt_d = jax.nn.sigmoid(log_base + QD @ jnp.asarray(d_weights))
        np.testing.assert_allclose(np.asarray(rebuilt_d), old_d, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(np.asarray(residual.evaluate(jnp.asarray(old_q), jnp.asarray(old_d), None,
        committed).trial_history), old_H, rtol=1e-12, atol=1e-12)
    affine = residual.lift(jnp.stack((jnp.zeros_like(points[:, 1]), 1e-4 * points[:, 1]), axis=1).reshape(-1), 1e-4)
    zero_factor = backend._edge_factor(jnp.zeros(model.n_material, jnp.float64))
    q_scale = float(jnp.sqrt(jnp.mean(backend._residual_from_factor(zero_factor, affine)[free]**2))) * args.load / 1e-4
    d_scale = float(2 * np.sqrt(np.mean(old_H**2)))
    if not np.all(np.isfinite([q_scale, d_scale])) or min(q_scale, d_scale) <= 0:
        raise ValueError("joint fixed normalization scales must be finite and positive")
    base = q_base(args.load)
    mechanics_row_weights = jnp.ones(len(free), jnp.float64)

    def fields_at(vector):
        q = base + args.load * QB @ vector[:nq]
        d = jax.nn.sigmoid(log_base + QD @ vector[nq:])
        fields = residual.evaluate(q, d, None, committed)
        return q, d, fields

    def physical_residual(vector):
        _, _, fields = fields_at(vector)
        return jnp.concatenate((mechanics_row_weights * fields.mechanics / (q_scale * np.sqrt(len(free))),
                                args.damage_weight * fields.damage / (d_scale * np.sqrt(model.n_material))))

    def branch_residual(vector, driver0, history0, mask):
        _, d, fields = fields_at(vector)
        H = history0 + mask * (fields.driver - driver0)
        rho = (residual.gc_over_l0 * d + residual.gc_l0 * residual.operator.deterministic_matvec(d)
               + 2.0 * H * (d - 1.0))
        return jnp.concatenate((mechanics_row_weights * fields.mechanics / (q_scale * np.sqrt(len(free))),
                                args.damage_weight * rho / (d_scale * np.sqrt(model.n_material))))

    evaluate_driver = jax.jit(lambda vector: fields_at(vector)[2].driver)

    def branch_data(vector):
        driver = np.asarray(evaluate_driver(jnp.asarray(vector)))
        gap = driver - old_H
        tie_tolerance = 128 * np.finfo(np.float64).eps * np.maximum(np.maximum(abs(driver), abs(old_H)), 1.0)
        mask = np.where(abs(gap) <= tie_tolerance, 0.5, np.where(gap > 0, 1.0, 0.0))
        return tuple(jnp.asarray(x) for x in (driver, np.maximum(driver, old_H), mask))

    vector0 = jnp.asarray(np.concatenate((q_initial, d_weights)))
    if args.warm_trial is not None:
        warm_policy = json.loads((args.warm_trial / "policy.json").read_text())
        for key, value in (("base", str(args.base)), ("source", str(args.source)),
                           ("source_index", args.source_index), ("parent_digest", args.parent_digest),
                           ("load", args.load)):
            if warm_policy[key] != value:
                raise ValueError(f"joint warm trial differs in committed context: {key}")
        if warm_policy["source_field_sha256"] != hashlib.sha256(field_path.read_bytes()).hexdigest():
            raise ValueError("joint warm trial has a different committed field digest")
        with np.load(args.warm_trial / "fields.npz", allow_pickle=False) as archive:
            np.testing.assert_allclose(archive["damage_transform"], damage_transform, rtol=1e-12, atol=1e-12)
            previous_transform = archive["q_transform"]
            previous_nq = previous_transform.shape[1]
            previous_count = previous_transform.shape[0] // 2
            previous_weights = archive["weights"]
            if "q_centers" in archive.files:
                np.testing.assert_allclose(q_centers[:previous_count], archive["q_centers"], rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(q_radii[:previous_count], archive["q_radii"], rtol=1e-12, atol=1e-12)
            raw_weights = extend_weights(previous_transform @ previous_weights[:previous_nq], previous_count)
            transformed = singular[retained] * (Vt[retained] @ raw_weights)
            vector0 = jnp.asarray(np.concatenate((transformed, previous_weights[previous_nq:])))
            transferred_q = np.asarray(fields_at(vector0)[0])
            np.testing.assert_allclose(transferred_q, archive["q"], rtol=1e-7, atol=1e-8)
            np.testing.assert_allclose(np.asarray(fields_at(vector0)[1]), archive["d"], rtol=1e-10, atol=1e-12)
        if vector0.shape != (nq + len(d_weights),):
            raise ValueError("joint warm trial weight shape differs")
    if args.mechanics_weighting == "initial_outliers":
        initial_mechanics = np.asarray(fields_at(vector0)[2].mechanics)
        initial_rms = float(np.sqrt(np.mean(initial_mechanics**2)))
        if not np.isfinite(initial_rms) or initial_rms <= 0:
            raise ValueError("initial mechanics outlier scale must be finite and positive")
        mechanics_row_weights = jnp.asarray(np.sqrt(1 + (initial_mechanics / initial_rms)**2))
    evaluate = jax.jit(physical_residual)
    np.testing.assert_allclose(np.asarray(evaluate(vector0)), np.asarray(physical_residual(vector0)),
                               rtol=1e-7, atol=1e-8)

    @jax.jit
    def apply_jacobian(vector, direction, driver0, history0, mask):
        fn = lambda v: branch_residual(v, driver0, history0, mask)
        return jax.jvp(fn, (vector,), (direction,))[1]

    @jax.jit
    def apply_transpose(vector, direction, driver0, history0, mask):
        fn = lambda v: branch_residual(v, driver0, history0, mask)
        _, pullback = jax.vjp(fn, vector)
        return pullback(direction)[0]

    parameter_scale = np.ones(len(vector0))
    if args.parameter_scaling == "column_norm":
        batched_jvp = jax.jit(jax.vmap(apply_jacobian, in_axes=(None, 0, None, None, None)))
        initial_branch = branch_data(vector0)
        column_norms = np.empty(len(vector0))
        for start in range(0, len(vector0), 4):
            count = min(4, len(vector0) - start)
            directions = np.zeros((4, len(vector0)))
            directions[np.arange(count), start + np.arange(count)] = 1.0
            columns = np.asarray(batched_jvp(vector0, jnp.asarray(directions), *initial_branch))
            column_norms[start:start + count] = np.linalg.norm(columns[:count], axis=1)
        if not np.all(np.isfinite(column_norms)) or np.any(column_norms <= 0):
            raise ValueError("joint Jacobian column scales must be finite and nonzero")
        parameter_scale = 1.0 / column_norms

    output = args.output
    if not output.parent.is_dir():
        raise FileNotFoundError(output.parent)
    output.mkdir()
    policy = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    policy.update({"schema": "fracture2d-joint-rbf-research-v1", "q_basis_svd_cutoff": 1e-10,
                   "mechanics_scale": q_scale, "damage_scale": d_scale,
                   "committed_load": source_report["load"],
                   "source_field_sha256": hashlib.sha256(field_path.read_bytes()).hexdigest(),
                   "reference_targets": False, "method": "joint TRF / matrix-free LSMR",
                   "history": "max of unchanged committed history and current Amor driver",
                   "lsmr_maxiter": args.lsmr_maxiter, "lsmr_atol": args.lsmr_tolerance,
                   "lsmr_btol": args.lsmr_tolerance,
                   "ftol": 1e-12, "xtol": 1e-12, "gtol": 1e-10,
                   "x_scale": args.parameter_scaling})
    policy["history_derivative"] = "piecewise max derivative; midpoint at 128-ulp numerical ties"
    policy["mechanics_row_weight_formula"] = (
        "sqrt(1 + (initial raw mechanics residual / initial RMS)^2), fixed during solve"
        if args.mechanics_weighting == "initial_outliers" else "uniform one")
    atomic_json(output / "policy.json", policy)
    np.savez_compressed(output / "parameter_scaling.npz", scale=parameter_scale)
    np.savez_compressed(output / "mechanics_row_weights.npz", weights=np.asarray(mechanics_row_weights))
    reports = []

    def diagnostics(vector):
        q, d, fields = fields_at(jnp.asarray(vector))
        q, d, H = map(np.asarray, (q, d, fields.trial_history))
        local = np.asarray(residual.gc_over_l0 * d)
        nonlocal_term = np.asarray(residual.gc_l0 * residual.operator.deterministic_matvec(jnp.asarray(d)))
        history = 2 * H * (d - 1)
        term_scale = max(abs(local)) + max(abs(nonlocal_term)) + max(abs(history))
        raw_q_residual = np.asarray(fields.mechanics)
        healing = float(np.max(np.maximum(old_d - d, 0)))
        drift = int(np.count_nonzero(np.linalg.norm(q[:2 * model.n_material].reshape(-1, 2), axis=1)
                                    / args.load > direct.DRIFT_ENVELOPE))
        gates = {"finite": all(np.all(np.isfinite(v)) for v in (q, d, H, raw_q_residual, history)),
                 "mechanics_loss": np.mean((raw_q_residual / q_scale)**2) <= 1e-2,
                 "mechanics_inf": max(abs(raw_q_residual)) / q_scale <= 0.1,
                 "damage_residual": max(abs(np.asarray(fields.damage))) / term_scale <= 1e-3,
                 "damage_range": min(d) >= 0 and max(d) <= 1,
                 "healing": healing <= 1e-3, "drift": drift == 0,
                 "history_monotonic": min(H - old_H) >= 0}
        report = {"load": args.load, "reaction": float(fields.reaction), "damage_max": float(max(d)),
                  "damage_inf": float(max(abs(np.asarray(fields.damage)))),
                  "damage_relative_inf": float(max(abs(np.asarray(fields.damage))) / term_scale),
                  "mechanics_loss": float(np.mean((raw_q_residual / q_scale)**2)),
                  "mechanics_normalized_inf": float(max(abs(raw_q_residual)) / q_scale),
                  "healing": healing, "drift": drift,
                  "gates": {k: bool(v) for k, v in gates.items()},
                  "physical_checks_pass": bool(all(gates.values())), "strict_accepted": False}
        return report, q, d, H

    def objective(vector):
        value = np.asarray(evaluate(jnp.asarray(vector)))
        if not np.all(np.isfinite(value)):
            raise FloatingPointError("joint residual is nonfinite")
        return value

    def jacobian(vector):
        values = jnp.asarray(vector)
        branch = branch_data(values)
        report, _, _, _ = diagnostics(vector)
        reports.append(report)
        print(json.dumps({"event": "joint_iteration", "iteration": len(reports)-1, **report}), flush=True)
        atomic_json(output / "iterations.json", reports)
        return LinearOperator((len(free) + model.n_material, len(vector)), dtype=np.float64,
            matvec=lambda v: np.asarray(apply_jacobian(values, jnp.asarray(np.asarray(v).reshape(-1)), *branch)),
            rmatvec=lambda v: np.asarray(apply_transpose(values, jnp.asarray(np.asarray(v).reshape(-1)), *branch)))

    # A directional derivative check exercises the composed link/history rule.
    direction = np.random.default_rng(0).normal(size=len(vector0))
    direction /= np.linalg.norm(direction)
    initial_branch = branch_data(vector0)
    branch_evaluate = jax.jit(branch_residual)
    np.testing.assert_allclose(np.asarray(branch_evaluate(vector0, *initial_branch)), objective(np.asarray(vector0)),
                               rtol=1e-7, atol=1e-8)
    jv = np.asarray(apply_jacobian(vector0, jnp.asarray(direction), *initial_branch))
    derivative_checks = []
    for h in (1e-6, 1e-7, 1e-8, 1e-9):
        plus = np.asarray(vector0) + h * direction
        minus = np.asarray(vector0) - h * direction
        fd = (objective(plus) - objective(minus)) / (2 * h)
        relative = np.linalg.norm(fd - jv) / np.linalg.norm(jv)
        mask_plus = np.asarray(fields_at(jnp.asarray(plus))[2].driver) > old_H
        mask_minus = np.asarray(fields_at(jnp.asarray(minus))[2].driver) > old_H
        check = {"h": h, "relative_defect": relative,
                 "history_branch_switches": int(np.count_nonzero(mask_plus != mask_minus))}
        branch_fd = (np.asarray(branch_evaluate(jnp.asarray(plus), *initial_branch))
                     - np.asarray(branch_evaluate(jnp.asarray(minus), *initial_branch))) / (2 * h)
        check["fixed_branch_relative_defect"] = float(np.linalg.norm(branch_fd - jv) / np.linalg.norm(jv))
        derivative_checks.append(check)
        print(json.dumps({"event": "joint_jvp_check", **check}), flush=True)
        if np.isfinite(check["fixed_branch_relative_defect"]) and check["fixed_branch_relative_defect"] <= 1e-4:
            break
    atomic_json(output / "derivative_checks.json", derivative_checks)
    if not np.isfinite(check["fixed_branch_relative_defect"]) or check["fixed_branch_relative_defect"] > 1e-4:
        raise ValueError(f"joint fixed-branch JVP finite-difference check failed: {check}")
    cotangent = np.random.default_rng(1).normal(size=len(jv))
    jt = np.asarray(apply_transpose(vector0, jnp.asarray(cotangent), *initial_branch))
    np.testing.assert_allclose(jv @ cotangent, direction @ jt, rtol=1e-7, atol=1e-9)
    print(json.dumps({"event": "joint_generalized_jacobian_verified", **check}), flush=True)
    solved = least_squares(objective, np.asarray(vector0), jac=jacobian, method="trf",
        tr_solver="lsmr", tr_options={"maxiter": args.lsmr_maxiter,
                                      "atol": args.lsmr_tolerance, "btol": args.lsmr_tolerance},
        max_nfev=args.max_evaluations, ftol=1e-12, xtol=1e-12, gtol=1e-10, x_scale=parameter_scale)
    report, q, d, H = diagnostics(solved.x)
    report.update({"optimizer_success": bool(solved.success), "optimizer_message": str(solved.message),
                   "evaluations": int(solved.nfev)})
    atomic_json(output / "result.json", report)
    np.savez_compressed(output / "fields.npz", q=q, d=d, H=H, coordinates=xy[:model.n_material],
        weights=solved.x, q_transform=q_transform, damage_transform=damage_transform, centers=centers, radii=radii,
        q_centers=q_centers, q_radii=q_radii)
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent-digest", required=True)
    parser.add_argument("--load", type=float, required=True)
    parser.add_argument("--max-evaluations", type=int, required=True)
    parser.add_argument("--lsmr-maxiter", type=int, required=True)
    parser.add_argument("--lsmr-tolerance", type=float, required=True)
    parser.add_argument("--parameter-scaling", choices=("none", "column_norm"), required=True)
    parser.add_argument("--damage-weight", type=float, required=True)
    parser.add_argument("--warm-trial", type=Path)
    parser.add_argument("--joint-parent", type=Path)
    parser.add_argument("--extra-q-units", type=int, required=True)
    parser.add_argument("--mechanics-weighting", choices=("none", "initial_outliers"), required=True)
    parser.add_argument("--q-feature-source", choices=("parent", "warm_trial"), required=True)
    args = parser.parse_args()
    if args.max_evaluations < 1 or args.source_index < 0 or args.lsmr_maxiter < 1 or not 0 < args.lsmr_tolerance < 1:
        raise ValueError("invalid joint evaluation budget/source index")
    if not np.isfinite(args.damage_weight) or args.damage_weight <= 0:
        raise ValueError("damage-weight must be finite and positive")
    if args.extra_q_units < 0:
        raise ValueError("extra-q-units must be nonnegative")
    run(args)


if __name__ == "__main__":
    main()
