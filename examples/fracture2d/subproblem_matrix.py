"""Common-floor fixed-load direct/neural q/d diagnostic, never a production path."""
import argparse
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg
import numpy as np
import optax
from ml_collections import ConfigDict

from examples.fracture2d import nonlocal_baseline as direct
from examples.fracture2d.coupling_diagnostics import atomic_json, metrics
from examples.fracture2d.diagnostic_continuation import _strict_parent
from examples.fracture2d.fixed_load_sweeps import _experiment_configs
from examples.fracture2d.fp64 import FP64A2Backend
from examples.fracture2d.legacy_coupled_loading import LIMITS, load_diagnostic
from examples.fracture2d.legacy_coupling_refine import readiness
from examples.fracture2d.losses import FractureResidual
from examples.fracture2d.stagger import _inner_update_ratio, _warning_advancement_gates
from examples.fracture2d.train import _build_grid, _nonnegative_damage


def damage_solver(operator, volumes, gc0, gc1, *, tol, atol, maxiter):
    """Original volume-SPD AT2 solve; CG convergence is checked explicitly."""
    if (not np.isfinite(tol) or not 0 < tol < 1 or not np.isfinite(atol) or not 0 < atol
            or type(maxiter) is not int or maxiter < 1 or not np.all(np.isfinite([gc0, gc1]))
            or min(gc0, gc1) <= 0
            or volumes.shape != (operator.shape[0],) or not np.all(np.isfinite(volumes))
            or np.min(volumes) <= 0):
        raise ValueError("invalid fixed-H CG controls/volumes")
    V = jnp.asarray(volumes, jnp.float64)
    D = operator.to_scipy().tocsr()
    from scipy.sparse import diags
    mass_D = diags(volumes) @ D
    skew = mass_D-mass_D.T
    if skew.nnz and np.max(np.abs(skew.data)) > 1e-10*max(1., np.max(np.abs(mass_D.data))):
        raise ValueError("fixed-H CG requires volume-symmetric damage operator")
    diagonal_D = jnp.asarray(D.diagonal(), jnp.float64)

    @jax.jit
    def solve(H, initial):
        diagonal = V * (gc0 + 2*H + gc1*diagonal_D)
        def matvec(d):
            return V*((gc0+2*H)*d + gc1*operator.deterministic_matvec(d))
        d, _ = cg(matvec, 2*V*H, x0=initial, tol=tol, atol=atol,
                  maxiter=maxiter, M=lambda r: r/diagonal)
        return d, matvec(d)-2*V*H

    def checked(H, initial):
        if (H.shape != V.shape or initial.shape != V.shape
                or np.asarray(H).dtype != np.float64 or np.asarray(initial).dtype != np.float64
                or not np.all(np.isfinite(np.asarray(H)))
                or not np.all(np.isfinite(np.asarray(initial))) or float(jnp.min(H)) < 0):
            raise ValueError("fixed-H CG received invalid history/shape")
        d, r = solve(H, initial)
        norm = float(jnp.linalg.norm(r))
        limit = max(atol, tol*float(jnp.linalg.norm(2*V*H)))
        if not np.all(np.isfinite(np.asarray(d))) or norm > limit*(1+1e-3):
            raise RuntimeError(f"GPU AT2 CG failed explicit residual: {norm} > {limit}")
        return d
    return checked


def run(args):
    if args.sweeps < 4 or args.direct_sweeps < args.sweeps or not 0 < args.mixing <= 1:
        raise ValueError("invalid matched/direct sweep budget or mixing")
    if len(set(args.load_indices)) != len(args.load_indices) or any(i not in (61, 62) for i in args.load_indices):
        raise ValueError("matrix loads must be distinct members of 61/62")
    if args.output.exists() or not args.output.parent.is_dir():
        raise ValueError("matrix output must be a new root under an existing parent")
    if not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("matrix solve requires GPU")
    source = ConfigDict(json.loads((args.source_root/'config.json').read_text()))
    pc, _ = _experiment_configs(source, args.source_root, args.output, max(4, args.sweeps),
                                tensorboard_root=args.output.parent/'events')
    model = _build_grid(pc)
    strict, sd = _strict_parent(pc, model)
    c, parent, pd = load_diagnostic(args.parent_state, args.parent_sha256, 60,
        model, source, strict, sd, require_ready=True)
    backend = FP64A2Backend(model.static, model.constraints, jax.devices()[0])
    residual = FractureResidual(model, backend)
    points = jnp.asarray(model.geometry.coordinates)
    floor_d, floor_H = parent[5], parent[6]
    original_H = np.asarray(floor_H).copy()
    volumes = np.asarray(model.geometry.material_reference_volumes)
    gc0, gc1 = float(model.Gc/model.l0), float(model.Gc*model.l0)
    solve_d = damage_solver(residual.operator, volumes, gc0, gc1,
                            tol=1e-11, atol=1e-13, maxiter=10000)
    args.output.mkdir()
    policy = dict(schema="fracture2d-common-floor-matrix-v1", parent_sha256=pd,
        strict_ancestor_sha256=sd, model_digest=model.model_digest, static_digest=model.static_digest,
        load_indices=args.load_indices, independent_same_floor=True, mixing=args.mixing,
        matched_sweeps=args.sweeps, direct_sweeps=args.direct_sweeps,
        original_config=json.loads(c.to_json_best_effort()), limits=LIMITS,
        mechanics_cg=dict(tol=1e-9, atol=1e-10, maxiter=10000),
        damage_cg=dict(tol=1e-11, atol=1e-13, maxiter=10000),
        diagnostic_only=True, strict_accepted=False, reference_targets=False)
    atomic_json(args.output/'policy.json', policy)
    # Independent validation only: the generated trajectory always uses GPU d.
    gpu_d = solve_d(floor_H, floor_d)
    cpu = direct.solve_fixed_history_damage(residual.operator, model.Gc, model.l0, np.asarray(floor_H))
    np.testing.assert_allclose(np.asarray(gpu_d), cpu.damage, rtol=1e-8, atol=1e-10)
    atomic_json(args.output/'damage_backend_check.json', dict(
        max_difference=float(np.max(np.abs(np.asarray(gpu_d)-cpu.damage))),
        cpu_validation_only=True, fallback=False))
    optimizer = optax.adam(c.train.learning_rate)
    with np.load(c.data.reference, allow_pickle=False) as archive:
        loads = archive['loads'].copy()
        ref_damage, ref_reactions = archive['damage'].copy(), archive['top_reaction'].copy()
    first_load = float(loads[1])
    affine = jnp.stack((jnp.zeros_like(points[:, 1]), first_load*points[:, 1]), axis=1)
    affine_q = residual.lift(affine.reshape(-1), first_load)
    first_r = backend._residual_from_factor(backend._edge_factor(jnp.zeros(model.n_material, jnp.float64)), affine_q)[backend.free_dofs]
    first_scale = jnp.sqrt(jnp.mean(first_r**2))

    def q_at(net, load):
        base = jnp.stack((jnp.zeros_like(points[:, 1]), load*points[:, 1]), axis=1)
        return residual.lift((base+load*jax.vmap(net)(points)).reshape(-1), load)
    def d_at(net, load):
        raw = (c.physical.E/gc0)*load**2*jax.vmap(net)(points[:model.n_material])[:, 0]
        return _nonnegative_damage(raw, c.physical.damage_floor)
    @eqx.filter_jit
    def q_step(net, state, d, load, ratio):
        def objective(n):
            r = backend._residual_from_factor(backend._edge_factor(d), q_at(n, load))[backend.free_dofs]
            return jnp.mean((r/(first_scale*load/first_load))**2)
        _, grad = eqx.filter_value_and_grad(objective)(net)
        updates, state = optimizer.update(grad, state, net)
        return eqx.apply_updates(net, jax.tree.map(lambda u: ratio*u, updates)), state
    @eqx.filter_jit
    def d_step(net, state, H, load, ratio):
        def objective(n):
            d = d_at(n, load)
            rho = gc0*d+gc1*residual.operator.deterministic_matvec(d)+2*H*(d-1)
            return jnp.mean((rho/(2*jnp.sqrt(jnp.mean(H**2))))**2)
        _, grad = eqx.filter_value_and_grad(objective)(net)
        updates, state = optimizer.update(grad, state, net)
        return eqx.apply_updates(net, jax.tree.map(lambda u: ratio*u, updates)), state

    def trained(net, state, field, load, count, step):
        for iteration in range(1, count+1):
            ratio = jnp.asarray(_inner_update_ratio(c, iteration, count), jnp.float64)
            net, state = step(net, state, field, load, ratio)
        return net, state
    def exact_q(d, load, guess):
        return backend.solve_strict(np.asarray(d), float(load), 1e-9, 1e-10, 10000,
                                    initial_guess=np.asarray(guess)).displacement
    summary = []
    modes = ((False, False), (False, True), (True, False), (True, True))
    for index in args.load_indices:
        load = jnp.asarray(loads[index], jnp.float64)
        for neural_q, neural_d in modes:
            name = f"{'neural' if neural_q else 'direct'}_q_{'neural' if neural_d else 'direct'}_d"
            root = args.output/f'load_{index:04d}'/name
            root.mkdir(parents=True)
            qnet, qs, dnet, ds = parent[:4]
            d_iter = floor_d
            guess = residual.lift(parent[4]*(load/loads[60]), load)
            previous, consecutive, rows = None, 0, []
            budget = args.sweeps if neural_q or neural_d else args.direct_sweeps
            status = 'budget_exhausted'
            for sweep in range(1, budget+1):
                phase = 'mechanics_before'
                try:
                    qb = exact_q(d_iter, load, guess)
                    if neural_q:
                        qnet, qs = trained(qnet, qs, d_iter, load, c.stagger.mechanics_steps, q_step)
                        q_before = q_at(qnet, load)
                    else:
                        q_before = qb
                    before = residual.evaluate(q_before, d_iter, None, floor_H)
                    bench_before = residual.evaluate(qb, d_iter, None, floor_H)
                    H_used = before.trial_history
                    phase = 'damage'
                    db = solve_d(H_used, d_iter)
                    if neural_d:
                        dnet, ds = trained(dnet, ds, H_used, load, c.stagger.damage_steps, d_step)
                        d_solved = d_at(dnet, load)
                    else:
                        d_solved = db
                    d_trial = d_iter+args.mixing*(d_solved-d_iter)
                    phase = 'mechanics_after'
                    qa = exact_q(d_trial, load, q_before)
                    if neural_q:
                        qnet, qs = trained(qnet, qs, d_trial, load, c.stagger.mechanics_steps, q_step)
                        q = q_at(qnet, load)
                    else:
                        q = qa
                    evaluated = residual.evaluate(q, d_trial, None, floor_H)
                    bench_after = residual.evaluate(qa, d_trial, None, floor_H)
                except RuntimeError as error:
                    status = 'numerical_failure'
                    atomic_json(root/'failure.json', dict(sweep=sweep, phase=phase, reason=str(error)))
                    print(json.dumps(dict(event='matrix_numerical_failure', index=index, mode=name,
                                          sweep=sweep, phase=phase, reason=str(error))), flush=True)
                    break
                q_np, d_np, H_np = map(np.asarray, (q, d_trial, evaluated.trial_history))
                residual.validate_fields(q_np, d_np, H_np, original_H, np.asarray(load))
                ratio = np.linalg.norm(q_np[:2*model.n_material].reshape(-1, 2), axis=1)/float(load)
                row = dict(sweep=sweep, load=float(load), reaction=float(evaluated.reaction),
                    damage_inf=float(jnp.max(jnp.abs(evaluated.damage))),
                    mechanics_loss=float(jnp.mean((evaluated.mechanics/(first_scale*load/first_load))**2)),
                    mechanics_inf=float(jnp.max(jnp.abs(evaluated.mechanics))),
                    damage_min=float(np.min(d_np)), damage_max=float(np.max(d_np)),
                    damage_healing_max=float(np.max(np.maximum(np.asarray(floor_d)-d_np, 0))),
                    history_decrease_max=float(np.max(np.maximum(original_H-H_np, 0))),
                    drift_count=int(np.count_nonzero(ratio>direct.DRIFT_ENVELOPE)),
                    max_displacement_ratio=float(np.max(ratio)), accepted=False, diagnostic_only=True,
                    reaction_relative=abs(float(evaluated.reaction)-float(ref_reactions[index]))/abs(float(ref_reactions[index])),
                    d_rmse=float(np.sqrt(np.mean((d_np-ref_damage[index])**2))),
                    before_q_vs_exact_l2=float(jnp.linalg.norm(q_before-qb)/jnp.linalg.norm(qb)),
                    before_H_vs_exact_inf=float(jnp.max(jnp.abs(H_used-bench_before.trial_history))),
                    after_q_vs_exact_l2=float(jnp.linalg.norm(q-qa)/jnp.linalg.norm(qa)),
                    after_H_vs_exact_inf=float(jnp.max(jnp.abs(evaluated.trial_history-bench_after.trial_history))),
                    after_driver_error_min=float(jnp.min(evaluated.driver-bench_after.driver)),
                    after_driver_error_max=float(jnp.max(evaluated.driver-bench_after.driver)),
                    solved_d_vs_exact_inf=float(jnp.max(jnp.abs(d_solved-db))))
                gates = _warning_advancement_gates(c, row)
                row['gates'] = {key: bool(value) for key, value in gates.items()}
                fields = dict(q=q_np, d=d_np, H=H_np, H_used=np.asarray(H_used),
                    d_solved=np.asarray(d_solved), d_previous=np.asarray(d_iter),
                    at2_local=gc0*d_np,
                    at2_nonlocal=np.asarray(gc1*residual.operator.deterministic_matvec(d_trial)),
                    at2_history=2*H_np*(d_np-1),
                    at2_fixed_relaxed=np.asarray(gc0*d_trial+gc1*residual.operator.deterministic_matvec(d_trial)+2*H_used*(d_trial-1)),
                    at2_fixed_solved=np.asarray(gc0*d_solved+gc1*residual.operator.deterministic_matvec(d_solved)+2*H_used*(d_solved-1)),
                    coordinates=np.asarray(model.geometry.coordinates[:model.n_material]))
                diagnostic = metrics(row, fields, previous)
                checks = readiness(diagnostic, row['gates'], ConfigDict(LIMITS))
                consecutive = consecutive+1 if all(checks.values()) else 0
                row.update(coupling=diagnostic, checks=checks, consecutive=consecutive,
                           coupled_ready=consecutive>=3, strict_accepted=False)
                np.savez_compressed(root/f'fields_{sweep:04d}.npz', **fields,
                    q_exact_same_d=np.asarray(qa), H_exact_same_d=np.asarray(bench_after.trial_history),
                    d_exact_same_H=np.asarray(db), driver=np.asarray(evaluated.driver),
                    exact_driver=np.asarray(bench_after.driver))
                eqx.tree_serialise_leaves(root/f'neural_{sweep:04d}.eqx', (qnet, qs, dnet, ds))
                rows.append(row)
                atomic_json(root/'sweeps.json', rows)
                print(json.dumps(dict(event='matrix_sweep', index=index, mode=name,
                    sweep=sweep, reaction=row['reaction'], damage_max=row['damage_max'],
                    at2_relative=diagnostic['at2_relative_inf'], drift=row['drift_count'],
                    H_error=row['after_H_vs_exact_inf'], d_error=row['solved_d_vs_exact_inf'])), flush=True)
                previous = fields, row
                d_iter, guess = d_trial, q
                if not all(gates.values()):
                    status = 'physical_hard_stop'
                    break
                if consecutive >= 3 and sweep >= args.sweeps:
                    status = 'research_coupled_ready'
                    break
            if not np.array_equal(original_H, np.asarray(floor_H)):
                raise RuntimeError('common history floor was mutated')
            final = dict(index=index, mode=name, status=status, completed_sweeps=len(rows),
                matched_budget=args.sweeps, parent_sha256=pd, diagnostic_only=True,
                strict_accepted=False, final=None if not rows else rows[-1])
            atomic_json(root/'result.json', final)
            summary.append(final)
            atomic_json(args.output/'result.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root', 'parent-state', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--parent-sha256', required=True)
    parser.add_argument('--load-indices', type=int, nargs='+', required=True)
    parser.add_argument('--sweeps', type=int, required=True)
    parser.add_argument('--direct-sweeps', type=int, required=True)
    parser.add_argument('--mixing', type=float, required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
