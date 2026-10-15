"""Paired fixed-d original A2 objectives and common-energy audit; diagnostic only."""

import argparse
import json
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from ml_collections import ConfigDict

from examples.fracture2d._legacy import nonlocal_baseline as direct
from examples.fracture2d._legacy.continuation import _digest
from examples.fracture2d._legacy.coupling_diagnostics import atomic_json
from examples.fracture2d._legacy.diagnostic_continuation import _strict_parent
from examples.fracture2d._legacy.fixed_load_sweeps import _experiment_configs
from examples.fracture2d._legacy.fp64 import FP64A2Backend
from examples.fracture2d._legacy.legacy_coupled_loading import load_diagnostic
from examples.fracture2d._legacy.losses import FractureResidual
from examples.fracture2d._legacy.stagger import _inner_update_ratio
from examples.fracture2d._legacy.train import _build_grid


def energy_audit(backend, residual, q, d, floor_H, volumes, coordinates, load):
    """Check individual gradients and quadrature-weighted cross-field derivatives."""
    V = jnp.asarray(volumes)

    def elastic(q, d):
        return backend._energy_from_factor(backend._edge_factor(d), q)

    fixed_H = residual.evaluate(q, d, None, floor_H).trial_history

    def damage_energy(d):
        return jnp.sum(V * (0.5 * residual.gc_over_l0 * d**2
            + 0.5 * residual.gc_l0 * d * residual.operator.deterministic_matvec(d)
            + fixed_H * (d - 1)**2))

    grad_q, grad_d = jax.grad(elastic, 0)(q, d), jax.grad(damage_energy)(d)
    force = backend._residual_from_factor(backend._edge_factor(d), q)
    rho = residual.evaluate(q, d, fixed_H, floor_H).damage
    np.testing.assert_allclose(np.asarray(grad_q), np.asarray(force), rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(np.asarray(grad_d), np.asarray(V*rho), rtol=1e-7, atol=1e-8)
    report: dict = dict(mechanics_gradient_inf=float(jnp.max(jnp.abs(grad_q-force))),
                        fixed_H_damage_gradient_inf=float(jnp.max(jnp.abs(grad_d-V*rho))))

    def mechanics(q, d):
        return backend._residual_from_factor(backend._edge_factor(d), q)

    def damage(q, d, *, current):
        H = None if current else fixed_H
        return V * residual.evaluate(q, d, H, floor_H).damage

    @jax.jit
    def cross(v, w):
        mixed_u = jax.jvp(lambda x: mechanics(q, x), (d,), (w,))[1]
        frozen = jax.jvp(lambda x: damage(x, d, current=False), (q,), (v,))[1]
        current = jax.jvp(lambda x: damage(x, d, current=True), (q,), (v,))[1]
        return jnp.vdot(v, mixed_u), jnp.vdot(w, frozen), jnp.vdot(w, current)

    rng = np.random.default_rng(20261005)
    tip_distance = np.linalg.norm(coordinates[:len(d)] - [0.5, 0.5], axis=1)
    samples = []
    for name in ("tip", "global_0", "global_1"):
        v = rng.normal(size=q.shape)
        v[np.asarray(backend.constraints.constrained_dofs)] = 0
        w = rng.normal(size=d.shape)
        if name == "tip":
            mask = tip_distance <= 2*residual.model.l0
            v.reshape(-1, 2)[:len(d)][~mask] = 0
            v.reshape(-1, 2)[len(d):] = 0
            w[~mask] = 0
        v *= load / np.max(np.abs(v))
        w /= np.max(np.abs(w))
        a, b, c = map(float, cross(jnp.asarray(v), jnp.asarray(w)))
        # Independent central differences validate both mixed derivative directions.
        fd_rows = []
        for epsilon in (1e-4, 1e-5):
            av = float(jnp.vdot(jnp.asarray(v),
                (mechanics(q, d+epsilon*w)-mechanics(q, d-epsilon*w))/(2*epsilon)))
            cv = float(jnp.vdot(jnp.asarray(w),
                (damage(q+epsilon*v, d, current=True)
                 - damage(q-epsilon*v, d, current=True))/(2*epsilon)))
            fd_rows.append(dict(epsilon=epsilon, mechanics=av, current_damage=cv))
        np.testing.assert_allclose(fd_rows[-1]["mechanics"], a, rtol=1e-5, atol=1e-8)
        # max-history switching can invalidate a finite-difference linearization.
        fd_agrees = bool(np.isclose(fd_rows[-1]["current_damage"], c, rtol=1e-4, atol=1e-8))
        samples.append(dict(direction=name, mechanics_mixed=a, frozen_H_damage_mixed=b,
            current_H_damage_mixed=c, frozen_H_defect=a-b, current_H_defect=a-c,
            current_H_fd_agrees=fd_agrees, finite_differences=fd_rows))
    deformation = backend._deformation_gradient(backend._edge_factor(d), q)
    strain, _ = backend._strain_stress(deformation)
    report.update(mixed_derivatives=samples,
        negative_trace_material_edges=int(jnp.count_nonzero(
            jnp.trace(strain[residual.material_edges], axis1=1, axis2=2) < 0)),
        parent_active_nodes=int(jnp.count_nonzero(floor_H > residual.evaluate(q,d,None,floor_H).driver)),
        elastic_damage_gradient=np.asarray(jax.grad(elastic, 1)(q,d)).tolist(),
        history_damage_gradient=np.asarray(2*V*fixed_H*(d-1)).tolist(),
        interpretation="Individual subproblem gradients checked; cross derivatives audit the original unscaled equations, not a new potential.")
    return report


def run(args):
    if args.steps < 1 or args.report_every < 1 or args.steps % args.report_every:
        raise ValueError("positive steps must be divisible by report-every")
    if args.output.exists() or not args.output.parent.is_dir():
        raise ValueError("output requires a fresh root beneath an existing parent")
    if jax.devices()[0].platform != "gpu":
        raise RuntimeError("objective pair requires GPU")
    source = ConfigDict(json.loads((args.source_root/"config.json").read_text()))
    pc, _ = _experiment_configs(source, args.source_root, args.output, 4,
                                 tensorboard_root=args.output.parent/"events")
    model = _build_grid(pc)
    strict, sd = _strict_parent(pc, model)
    c, seed, seed_digest = load_diagnostic(args.seed_state, args.seed_sha256, 61,
        model, source, strict, sd, require_ready=False)
    _, parent, pd = load_diagnostic(c.output.coupled_parent_state,
        c.output.coupled_parent_sha256, 60, model, source, strict, sd, require_ready=True)
    if args.steps != c.stagger.mechanics_steps:
        raise ValueError("paired budget must equal original mechanics block")
    backend = FP64A2Backend(model.static, model.constraints, jax.devices()[0])
    residual = FractureResidual(model, backend)
    points = jnp.asarray(model.geometry.coordinates)
    d, floor_H = seed[5], parent[6]
    original_d, original_H = np.asarray(d).copy(), np.asarray(floor_H).copy()
    with np.load(c.data.reference, allow_pickle=False) as data:
        load, first_load = float(data["loads"][61]), float(data["loads"][1])

    @eqx.filter_jit
    def field(network):
        base = jnp.stack((jnp.zeros_like(points[:,1]), load*points[:,1]), axis=1)
        return residual.lift((base+load*jax.vmap(network)(points)).reshape(-1), load)

    np.testing.assert_allclose(np.asarray(field(seed[0])), np.asarray(seed[4]), rtol=1e-11, atol=1e-12)
    factor = backend._edge_factor(d)
    affine = residual.lift(jnp.stack((jnp.zeros_like(points[:,1]), first_load*points[:,1]), axis=1).reshape(-1), first_load)
    affine_force = backend._residual_from_factor(
        backend._edge_factor(jnp.zeros_like(d)), affine)[backend.free_dofs]
    force_scale = float(jnp.sqrt(jnp.mean(affine_force**2)))*load/first_load
    energy_scale = float(backend._energy_from_factor(factor, affine*load/first_load))
    if not np.all(np.isfinite([force_scale,energy_scale])) or min(force_scale,energy_scale) <= 0:
        raise ValueError("objective scales must be positive finite")
    exact = backend.solve_strict(np.asarray(d), load, 1e-9, 1e-10, 10000,
                                 initial_guess=np.asarray(seed[4]))
    exact_q = exact.displacement
    shadow = residual.evaluate(exact_q, d, None, floor_H)
    volumes = np.asarray(model.geometry.material_reference_volumes)
    tip = np.linalg.norm(np.asarray(points[:model.n_material])-[0.5,0.5],axis=1) <= 2*model.l0
    args.output.mkdir()
    policy = dict(schema="fracture2d-fixed-d-objective-pair-v1", seed_sha256=seed_digest,
        parent_sha256=pd, strict_ancestor_sha256=sd, model_digest=model.model_digest,
        static_digest=model.static_digest, load=load, steps=args.steps, report_every=args.report_every,
        learning_rate=c.train.learning_rate, fresh_adam_both=True, original_taper=True,
        ratios=[_inner_update_ratio(c,i,args.steps) for i in (1,args.steps)],
        force_scale=force_scale, energy_scale=energy_scale, tip_radius=2*model.l0,
        original_config=json.loads(c.to_json_best_effort()), reference_targets=False,
        diagnostic_only=True, strict_accepted=False, history_committed=False,
        exact_comparison_only=True, drift_envelope=direct.DRIFT_ENVELOPE)
    atomic_json(args.output/"policy.json", policy)
    audit = energy_audit(backend,residual,seed[4],d,floor_H,volumes,np.asarray(points),load)
    atomic_json(args.output/"energy_audit.json",audit)
    print(json.dumps(dict(event="energy_audit", **{k:v for k,v in audit.items()
        if k not in ("elastic_damage_gradient","history_damage_gradient")})),flush=True)
    np.savez_compressed(args.output/"shadow.npz", q=np.asarray(exact_q),d=original_d,
        driver=np.asarray(shadow.driver),H=np.asarray(shadow.trial_history),floor_H=original_H,
        coordinates=np.asarray(points),tip_mask=tip,force=np.asarray(shadow.mechanics),
        reaction=float(shadow.reaction))
    optimizer = optax.adam(c.train.learning_rate)
    summaries = {}
    for objective in ("residual", "energy"):
        root = args.output/objective
        root.mkdir()
        network = seed[0]
        opt_state = optimizer.init(eqx.filter(network,eqx.is_inexact_array))

        def loss(net):
            q = field(net)
            if objective == "energy":
                return backend._energy_from_factor(factor,q)/energy_scale
            force = backend._residual_from_factor(factor,q)[backend.free_dofs]
            return jnp.mean((force/force_scale)**2)

        @eqx.filter_jit
        def step(net, opt_state, ratio):
            value, gradient = eqx.filter_value_and_grad(loss)(net)
            update, opt_state = optimizer.update(gradient,opt_state,net)
            return eqx.apply_updates(net,jax.tree.map(lambda x: ratio*x,update)),opt_state,value

        rows = []
        start = time.monotonic()
        status = "budget_complete"
        for iteration in range(args.steps+1):
            if iteration:
                network,opt_state,_ = step(network,opt_state,
                    jnp.asarray(_inner_update_ratio(c,iteration,args.steps),jnp.float64))
            if iteration % args.report_every:
                continue
            q = field(network)
            fields = residual.evaluate(q,d,None,floor_H)
            q_np = np.asarray(q)
            de = np.asarray(fields.driver-shadow.driver)
            he = np.asarray(fields.trial_history-shadow.trial_history)
            qdiff = q-exact_q
            energy_error = float(backend._energy_from_factor(factor,qdiff))
            norm_ratio = np.linalg.norm(q_np.reshape(-1,2),axis=1)/load
            row = dict(iteration=iteration, objective=objective,
                mechanics_loss=float(jnp.mean((fields.mechanics/force_scale)**2)),
                mechanics_inf=float(jnp.max(jnp.abs(fields.mechanics))),
                energy=float(backend._energy_from_factor(factor,q)), energy_error=energy_error,
                reaction=float(fields.reaction),exact_reaction=float(shadow.reaction),
                reaction_relative_error=abs(float(fields.reaction-shadow.reaction))/abs(float(shadow.reaction)),
                q_relative_l2=float(jnp.linalg.norm(qdiff)/jnp.linalg.norm(exact_q)),
                driver_error_inf=float(np.max(abs(de))),driver_error_rmse=float(np.sqrt(np.mean(de**2))),
                driver_error_min=float(np.min(de)),driver_error_max=float(np.max(de)),
                tip_driver_error_inf=float(np.max(abs(de[tip]))),
                tip_driver_error_rmse=float(np.sqrt(np.mean(de[tip]**2))),
                H_error_inf=float(np.max(abs(he))),at2_current_inf=float(jnp.max(jnp.abs(fields.damage))),
                drift_count=int(np.count_nonzero(norm_ratio>direct.DRIFT_ENVELOPE)),
                max_norm_over_load=float(np.max(norm_ratio)),elapsed_seconds=time.monotonic()-start)
            if not all(np.isfinite(v) for v in row.values() if isinstance(v,(int,float))):
                raise FloatingPointError(f"nonfinite {objective} report at {iteration}")
            fixed = model.constraints.constrained_dofs
            np.testing.assert_array_equal(q_np[fixed],model.constraints.prescribed_vector(load)[fixed])
            rows.append(row)
            np.savez_compressed(root/f"fields_{iteration:05d}.npz",q=q_np,d=original_d,
                H=np.asarray(fields.trial_history),driver=np.asarray(fields.driver),
                force=np.asarray(fields.mechanics),at2=np.asarray(fields.damage))
            eqx.tree_serialise_leaves(root/"network_adam.eqx",(network,opt_state))
            atomic_json(root/"reports.json",rows)
            print(json.dumps(row),flush=True)
            if row["drift_count"]:
                status = "rejected_drift"
                break
        np.testing.assert_array_equal(np.asarray(d),original_d)
        np.testing.assert_array_equal(np.asarray(floor_H),original_H)
        summaries[objective] = dict(status=status,final=rows[-1],diagnostic_only=True,
                                    accepted=False,history_committed=False)
        atomic_json(args.output/"result.json",summaries)
    if _digest(args.seed_state/"diagnostic.ckpt.zip") != seed_digest:
        raise RuntimeError("source checkpoint changed during paired study")
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root",type=Path,required=True)
    parser.add_argument("--seed-state",type=Path,required=True)
    parser.add_argument("--seed-sha256",required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--steps",type=int,required=True)
    parser.add_argument("--report-every",type=int,required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
