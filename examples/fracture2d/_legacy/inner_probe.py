"""Target-free frozen-subproblem GN probe from a bound diagnostic snapshot."""

import argparse
import hashlib
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from ml_collections import ConfigDict

from examples.fracture2d._legacy.fp64 import FP64A2Backend
from examples.fracture2d._legacy.gauss_newton import make_gauss_newton_step
from examples.fracture2d._legacy.losses import FractureResidual
from examples.fracture2d._legacy.stagger_checkpoint import SCHEMA_DIAGNOSTIC, read_checkpoint
from examples.fracture2d._legacy.train import _build_grid, _build_network, _nonnegative_damage


def load_snapshot(root, parent_digest):
    root = Path(root)
    config = ConfigDict(json.loads((root / "config.json").read_text()))
    if config.train.dtype != "float64" or config.data.state_index != 60:
        raise ValueError("probe requires FP64 diagnostic state 60")
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    model = _build_grid(config)
    values = json.loads(config.to_json_best_effort())
    values["output"]["resume"] = False
    identity = {
        "config_sha256": hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest(),
        "cloud_sha256": config.data.cloud_sha256,
        "reference_sha256": config.data.reference_sha256,
        "direct_model_digest": model.model_digest,
        "precision": "float64", "parent_accepted_sha256": parent_digest,
        "warning_diagnostic": True,
        "parent_schema": "fracture2d-fp64-accepted-load-v2", "state_index": 60,
    }
    q_net = _build_network(config, jax.random.PRNGKey(config.train.seed), out_dim=2)
    d_net = _build_network(config, jax.random.PRNGKey(config.train.seed + 1), out_dim=1)
    q_net, d_net = (jax.tree.map(lambda x: x.astype(jnp.float64), n) for n in (q_net, d_net))
    optim = optax.adam(config.train.learning_rate)
    template = (q_net, optim.init(eqx.filter(q_net, eqx.is_inexact_array)),
                d_net, optim.init(eqx.filter(d_net, eqx.is_inexact_array)),
                jnp.zeros(model.n_dofs, jnp.float64),
                jnp.zeros(model.n_material, jnp.float64),
                jnp.zeros(model.n_material, jnp.float64))
    metadata, state = read_checkpoint(root / "diagnostic.ckpt.zip", identity,
                                      template, schema=SCHEMA_DIAGNOSTIC)
    if metadata["phase"] != "complete":
        raise ValueError("probe requires a complete diagnostic snapshot")
    return config, model, state


def run(root, output, parent_digest, steps, max_cg, method):
    output = Path(output)
    if not output.parent.is_dir():
        raise FileNotFoundError(output.parent)
    output.mkdir()
    config, model, state = load_snapshot(root, parent_digest)
    backend = FP64A2Backend(model.static, model.constraints, jax.devices()[0])
    residual = FractureResidual(model, backend)
    points = jnp.asarray(model.geometry.coordinates)
    with np.load(config.data.reference, allow_pickle=False) as archive:
        load = float(archive["loads"][60])
        first_load = float(archive["loads"][1])
    factor = backend._edge_factor(state[5])
    affine = jnp.stack((jnp.zeros_like(points[:, 1]), first_load * points[:, 1]), axis=1)
    affine = residual.lift(affine.reshape(-1), first_load)
    affine_r = backend._residual_from_factor(
        backend._edge_factor(jnp.zeros(model.n_material, jnp.float64)), affine)[backend.free_dofs]
    mechanics_scale = float(jnp.sqrt(jnp.mean(affine_r**2))) * load / first_load
    if not np.isfinite(mechanics_scale) or mechanics_scale <= 0:
        raise ValueError("mechanics normalization must be finite and positive")

    def q_at(network):
        raw = jax.vmap(network)(points)
        base = jnp.stack((jnp.zeros_like(points[:, 1]), load * points[:, 1]), axis=1)
        return residual.lift((base + load * raw).reshape(-1), load)

    def d_at(network):
        scale = (config.physical.E / (config.physical.Gc / config.physical.l0)) * load**2
        raw = scale * jax.vmap(network)(points[:model.n_material])[:, 0]
        return _nonnegative_damage(raw, config.physical.damage_floor)

    def q_residual(network):
        return backend._residual_from_factor(factor, q_at(network))[backend.free_dofs] / mechanics_scale

    H = state[6]
    H_scale = 2.0 * jnp.sqrt(jnp.mean(H**2))

    def d_residual(network):
        d = d_at(network)
        return (residual.gc_over_l0 * d + residual.gc_l0 * residual.operator.deterministic_matvec(d)
                + 2.0 * H * (d - 1.0)) / H_scale

    policy = {"root": str(root), "parent_digest": parent_digest, "steps": steps,
              "max_cg": max_cg, "method": method,
              "reference_targets": False, "initial_damping": 1e-2, "cg_tolerance": 1e-6}
    (output / "policy.json").write_text(json.dumps(policy, indent=2))
    summaries = {}
    for name, network, fn, field_fn in (("mechanics", state[0], q_residual, q_at),
                                       ("damage", state[2], d_residual, d_at)):
        if method == "energy_lbfgs":
            from jax.flatten_util import ravel_pytree
            from scipy.optimize import minimize
            energy_scale = float(backend._energy_from_factor(factor, affine * load / first_load))
            volumes = jnp.asarray(model.static.material_phase_volumes[:model.n_material])

            def energy(n):
                if name == "mechanics":
                    return backend._energy_from_factor(factor, q_at(n)) / energy_scale
                d = d_at(n)
                Dd = residual.operator.deterministic_matvec(d)
                return jnp.sum(volumes * (0.5 * residual.gc_over_l0 * d**2
                                          + 0.5 * residual.gc_l0 * d * Dd
                                          + H * (d - 1.0)**2))

            # Independently compare dE/dfield with the declared physical residual.
            if name == "mechanics":
                grad_q = jax.grad(lambda q: backend._energy_from_factor(factor, q))(state[4])
                expected = backend._residual_from_factor(factor, state[4])
            else:
                def field_energy(d):
                    return jnp.sum(volumes * (0.5 * residual.gc_over_l0 * d**2
                        + 0.5 * residual.gc_l0 * d * residual.operator.deterministic_matvec(d)
                        + H * (d - 1.0)**2))
                grad_q = jax.grad(field_energy)(state[5])
                # The stored relaxed field differs from the network's solved field.
                expected = volumes * (residual.gc_over_l0 * state[5]
                    + residual.gc_l0 * residual.operator.deterministic_matvec(state[5])
                    + 2.0 * H * (state[5] - 1.0))
            np.testing.assert_allclose(np.asarray(grad_q), np.asarray(expected), rtol=1e-7, atol=1e-9)
            print(json.dumps({"event": "energy_gradient_verified", "stage": name,
                              "max_defect": float(jnp.max(jnp.abs(grad_q - expected)))}), flush=True)
            trainable, static = eqx.partition(network, eqx.is_inexact_array)
            flat, unravel = ravel_pytree(trainable)

            @jax.jit
            def flat_energy(vector):
                return energy(eqx.combine(unravel(vector), static))

            evaluate = jax.jit(jax.value_and_grad(flat_energy))
            evaluate_rho = eqx.filter_jit(fn)
            reports = []

            def record(vector):
                net = eqx.combine(unravel(jnp.asarray(vector)), static)
                rho = np.asarray(evaluate_rho(net))
                report = {"stage": name, "step": len(reports),
                          "loss": float(np.mean(rho**2)),
                          "normalized_inf": float(np.max(np.abs(rho))),
                          "energy": float(flat_energy(jnp.asarray(vector)))}
                if name == "mechanics":
                    report["reaction"] = float(backend._residual_from_factor(factor, q_at(net))
                                               @ backend.outer_top_selector)
                reports.append(report)
                if report["step"] % 100 == 0:
                    print(json.dumps(report), flush=True)
                    (output / f"{name}.json").write_text(json.dumps(reports, indent=2))

            def objective(vector):
                value, gradient = evaluate(jnp.asarray(vector))
                return float(value), np.asarray(gradient, dtype=np.float64)

            record(np.asarray(flat))
            solved = minimize(objective, np.asarray(flat), jac=True, method="L-BFGS-B",
                              callback=record, options={"maxiter": steps, "maxls": 40,
                                  "maxcor": 30, "ftol": 1e-15, "gtol": 1e-10})
            record(solved.x)
            network = eqx.combine(unravel(jnp.asarray(solved.x)), static)
            (output / f"{name}.json").write_text(json.dumps(reports, indent=2))
            (output / f"{name}_optimizer.json").write_text(json.dumps({
                "success": bool(solved.success), "message": str(solved.message),
                "iterations": int(solved.nit), "evaluations": int(solved.nfev)}))
            eqx.tree_serialise_leaves(output / f"{name}.eqx", network)
            np.savez_compressed(output / f"{name}_fields.npz", field=np.asarray(field_fn(network)))
            summaries[name] = reports
            continue
        init, step = make_gauss_newton_step(
            fn, initial_damping=1e-2, max_cg=max_cg, cg_tolerance=1e-6,
            max_attempts=4, max_backtracks=20, armijo_c1=1e-4,
            damping_decrease=0.3, damping_increase=10.0,
            minimum_damping=1e-8, maximum_damping=1e8)
        evaluate = eqx.filter_jit(fn)
        optimizer_state = init(network)
        reports = []
        for index in range(steps + 1):
            if index:
                network, optimizer_state, _ = step(network, optimizer_state)
            rho = np.asarray(evaluate(network))
            report = {"stage": name, "step": index, "loss": float(np.mean(rho**2)),
                      "normalized_inf": float(np.max(np.abs(rho))),
                      "damping": float(optimizer_state.damping),
                      "cg_iterations": int(optimizer_state.cg_iterations),
                      "cg_relative_residual": float(optimizer_state.cg_relative_residual),
                      "step_size": float(optimizer_state.step_size),
                      "accepted": bool(optimizer_state.accepted)}
            if name == "mechanics":
                report["reaction"] = float(backend._residual_from_factor(factor, field_fn(network))
                                           @ backend.outer_top_selector)
            if not all(np.isfinite(v) for v in report.values() if isinstance(v, (int, float))):
                raise FloatingPointError("nonfinite inner probe report")
            reports.append(report)
            if index % 10 == 0 or index == steps:
                print(json.dumps(report), flush=True)
                (output / f"{name}.json").write_text(json.dumps(reports, indent=2))
                eqx.tree_serialise_leaves(output / f"{name}.eqx", network)
                np.savez_compressed(output / f"{name}_fields.npz", field=np.asarray(field_fn(network)))
        summaries[name] = reports
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent-digest", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--max-cg", type=int, required=True)
    parser.add_argument("--method", choices=("residual_gn", "energy_lbfgs"), required=True)
    args = parser.parse_args()
    if min(args.steps, args.max_cg) < 1:
        raise ValueError("steps and max-cg must be positive")
    run(args.root, args.output, args.parent_digest, args.steps, args.max_cg, args.method)


if __name__ == "__main__":
    main()
