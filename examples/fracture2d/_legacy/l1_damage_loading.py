"""Fresh L1/random Adam staggered loading; unready ancestors stay provisional."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import diags, load_npz

from examples.fracture2d._legacy import nonlocal_baseline as direct
from examples.fracture2d._legacy.conditioned_damage import make_features
from examples.fracture2d._legacy.core_halo_damage import (
    crack_sampling_probabilities, pack_rows, validate_probabilities)
from examples.fracture2d._legacy.coupling_diagnostics import atomic_json
from examples.fracture2d._legacy.local_damage_trial import load_geometry, sha
from examples.fracture2d._legacy.low_order_damage import (
    make_loss_fn, make_optimizer, make_step, optimizer_count)
from examples.fracture2d._legacy.outer_stopping import evaluate, validate_controls
from examples.fracture2d._legacy.output_damage_continue import atomic_fields
from examples.fracture2d._legacy.plain_damage_head import fresh_network
from examples.fracture2d._legacy.plain_sigmoid_damage import latent_basis_scaling
from examples.fracture2d._legacy.prebuilt_damage_pool import build_sampling_pool, make_pool_objective
from examples.fracture2d._legacy.precomputed_damage_sampling import make_sampling_rows


def loading_grid(*, transition_load, coarse_step, fine_step, end_load):
    """Construct exact requested endpoints using integer increment counts."""
    if (not np.isfinite([transition_load, coarse_step, fine_step, end_load]).all()
            or min(transition_load, coarse_step, fine_step) <= 0
            or end_load <= transition_load or fine_step >= coarse_step):
        raise ValueError('invalid explicit two-stage load grid')
    coarse_count = round(transition_load/coarse_step)
    fine_count = round((end_load-transition_load)/fine_step)
    if (coarse_count < 1 or fine_count < 1
            or not np.isclose(coarse_count*coarse_step, transition_load, rtol=0, atol=1e-14)
            or not np.isclose(transition_load+fine_count*fine_step, end_load, rtol=0, atol=1e-14)):
        raise ValueError('transition/end loads must lie on their explicit increment grids')
    return np.concatenate((np.arange(1, coarse_count+1)*coarse_step,
        transition_load+np.arange(1, fine_count+1)*fine_step)).astype(np.float64)


def committed_history(parent, current_driver):
    """Freeze the preceding load's history floor, not the preceding outer H."""
    if parent.shape != current_driver.shape:
        raise ValueError('committed history and current driver shapes differ')
    return jax.lax.stop_gradient(jnp.maximum(parent, current_driver))


def residual_metrics(d, H, D, *, gc0, gc1):
    parts = [gc0*d, gc1*(D@d), 2*H*(d-1)]
    r = sum(parts)
    denominator = sum(float(abs(part).max()) for part in parts)
    if not np.isfinite(denominator) or denominator <= 0:
        raise FloatingPointError('AT2 relative metric requires a positive physical scale')
    return r, float(abs(r).max()/denominator)


def validate(c):
    a = c.l1_loading
    grid = loading_grid(transition_load=a.transition_load, coarse_step=a.coarse_step,
        fine_step=a.fine_step, end_load=a.end_load)
    validate_controls(c.outer_stop)
    if (a.initial_physical_damage != 0 or a.initial_physical_history != 0
            or a.output_head != 'sigmoid' or a.history_feature_scaling != 'physical'
            or a.parameterization != 'latent_basis_rms' or a.trunk_gain != 1
            or a.loss_norm != 'l1' or a.explicit_residual_weighting is not False
            or a.optimizer != 'adam' or a.resample_every != 1
            or a.decay_rate != 1 or a.minimum_rate != a.learning_rate
            or a.user_provisional is not True or not c.outer_stop.enabled
            or c.stagger.omega != .5 or not 0 < c.head.initial_damage < 1
            or any(type(v) is not int or v < 1 for v in
                (a.inner_updates, a.sample_count, a.halo_capacity, a.pool_size, a.cpu_threads))
            or a.inner_updates % a.pool_size
            or any(type(v) is not int or not 1 <= v <= c.outer_stop.max_sweeps
                   for v in a.checkpoint_sweeps)):
        raise ValueError('L1 cold loading requires explicit declared physical/optimizer/sampling settings')
    return grid


def run(c, config_path):
    grid = validate(c)
    a = c.l1_loading
    if jax.devices()[0].platform != 'gpu' or not jax.config.jax_enable_x64:
        raise ValueError('L1 loading requires serialized FP64 GPU execution')
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:a.cpu_threads])
    root = Path(a.output_dir)
    # A supervisor may create only its own receipts/log before the worker starts.
    if (root/'identity.json').exists():
        raise ValueError('refusing to overwrite an existing L1 loading experiment')
    root.mkdir(parents=True, exist_ok=True)
    m, op, geometry = load_geometry(c)
    fem = json.loads((Path(c.trial.geometry_dir)/'fenics_parity.json').read_text())
    if not fem['pass'] or fem['mesh_sha256'] != geometry['mesh_sha256']:
        raise ValueError('cold loading requires independently validated local FEM geometry')
    if sha(a.basis_path) != a.basis_sha256:
        raise ValueError('geometry-only compact basis hash differs')
    B, D = load_npz(a.basis_path), op.to_scipy()
    if B.shape[0] != m.n_material or not a.sample_count < a.halo_capacity <= m.n_material:
        raise ValueError('basis/material ordering or sampled capacities invalid')
    gc0, gc1 = c.physical.Gc/c.physical.l0, c.physical.Gc*c.physical.l0
    xy = m.coordinates[:m.n_material]
    features, info = make_features(op, xy, gc0=gc0, length=c.physical.l0,
        tip=list(c.conditioned.tip), powers=list(c.conditioned.powers),
        roundoff_factor=c.validation.roundoff_factor)
    features = eqx.filter_jit(features)
    net = fresh_network(c, info['input_dimension'], B, head=a.output_head)
    net, scales = latent_basis_scaling(net, B, np.ones(B.shape[1]))
    bc, bv = pack_rows(B@diags(scales), width=int(np.max(np.diff(B.indptr))))
    dc, dv = pack_rows(D, width=int(np.max(np.diff(D.indptr))))
    bc, bv, dc, dv = map(jnp.asarray, (bc, bv, dc, dv))
    image, objective = make_loss_fn(gc0=gc0, gc1=gc1, gain=a.trunk_gain,
        population_size=m.n_material, norm=a.loss_norm)
    image = eqx.filter_jit(image)
    def full_context(H, X):
        return dict(H=H, core_H=H, features=X, basis_columns=bc, basis_values=bv,
            core_local=jnp.arange(m.n_material, dtype=jnp.int32),
            operator_columns=dc, operator_values=dv,
            p_core=jnp.full(m.n_material, 1/m.n_material, jnp.float64))
    optimizer, schedule = make_optimizer(kind=a.optimizer, learning_rate=a.learning_rate,
        decay_steps=a.decay_steps, decay_rate=a.decay_rate, minimum_rate=a.minimum_rate,
        grad_clip=a.grad_clip, b1=a.b1, b2=a.b2, eps=a.eps, eps_root=a.eps_root,
        shampoo_beta=a.shampoo_beta, correct_bias=a.correct_bias,
        precondition_frequency=a.precondition_frequency, max_precond_dim=a.max_precond_dim,
        precondition_1d=a.precondition_1d, weight_decay=a.weight_decay)
    state = optimizer.init(eqx.filter(net, eqx.is_inexact_array))
    step = make_step(make_pool_objective(objective, basis_columns=bc, basis_values=bv), optimizer)
    p = np.asarray(crack_sampling_probabilities(jnp.asarray(xy), start=list(a.segment_start),
        end=list(a.segment_end), width=a.sampling_width, boost=a.sampling_boost,
        power=a.sampling_power))
    validate_probabilities(p, population_size=m.n_material)
    sampling_rows = make_sampling_rows(D, row_width=dc.shape[1])
    np.savez_compressed(root/'parameterization.npz', column_scale=scales, trunk_gain=a.trunk_gain)
    np.save(root/'sampling_probabilities.npy', p)
    protected = {str(Path(a.basis_path)): sha(a.basis_path)}
    for name in ('identity.json', 'mesh.npz', 'damage_operator.npz', 'fenics_parity.json'):
        path = Path(c.trial.geometry_dir)/name
        protected[str(path)] = sha(path)
    repo = Path(__file__).resolve().parents[3]
    code = {}
    for module in list(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename is not None:
            path = Path(filename).resolve()
            if path.suffix == '.py' and (path.is_relative_to(repo/'examples') or path.is_relative_to(repo/'src')):
                code[str(path.relative_to(repo))] = sha(path)
    code[str(Path(config_path).resolve().relative_to(repo))] = sha(config_path)
    for name in code:
        target = root/'code_snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo/name, target)
    atomic_json(root/'identity.json', dict(schema='l1-adam-own-zero-loading-v1', config=c.to_dict(),
        loads=grid.tolist(), geometry=geometry, features=info, source=protected, code=code,
        own_zero=True, fresh_network=True, old_weights=False, direct_targets=False,
        damage_correction=False, explicit_residual_weighting=False,
        importance_correction='1/(N*p): expected uniform nodal MAE',
        optimizer_persistent_across_batches_outers_loads=True))
    q = jnp.zeros(len(m.load_direction), jnp.float64)
    d = jnp.zeros(m.n_material, jnp.float64)
    H = jnp.zeros_like(d)
    zero = root/'state_0000'
    zero.mkdir()
    atomic_fields(zero/'fields.npz', q=np.asarray(q), d=np.asarray(d), H=np.asarray(H))
    eqx.tree_serialise_leaves(root/'initial_network.eqx', net)
    eqx.tree_serialise_leaves(root/'initial_optimizer.eqx', state)
    atomic_json(zero/'result.json', dict(index=0, load=0., reaction=0., local_ready=True,
        certified_path=True, status='exact_physical_zero', optimizer_count=0,
        neural_image_is_physical_zero=False, fields_sha256=sha(zero/'fields.npz')))
    solve = m.gpu_solver(tol=c.validation.cg_rtol, atol=c.validation.cg_atol,
        maxiter=c.validation.cg_maxiter)
    cache = dict(slot=a.pool_size, pool=None, block=0)
    rng = np.random.default_rng(a.seed)
    journal = root/'sampling_blocks.jsonl'
    def sample(H_used, X, *, generator, warmup):
        if cache['slot'] == a.pool_size:
            before = generator.bit_generator.state
            cores = np.stack([generator.choice(m.n_material, size=a.sample_count,
                replace=True, p=p).astype(np.int32) for _ in range(a.pool_size)])
            pool, halos = build_sampling_pool(sampling_rows, cores, p, node_capacity=a.halo_capacity)
            cache.update(slot=0, pool=jax.device_put(pool), block=cache['block']+1)
            if not warmup:
                with journal.open('a') as stream:
                    stream.write(json.dumps(dict(block=cache['block'], rng_before=before,
                        rng_after=generator.bit_generator.state, core_shape=list(cores.shape),
                        cores_sha256=hashlib.sha256(cores.tobytes()).hexdigest(),
                        halo_min=int(halos.min()), halo_max=int(halos.max())))+'\n')
        slot = cache['slot']
        cache['slot'] += 1
        return dict(H=H_used, features=X, pool=cache['pool'], slot=jnp.asarray(slot, jnp.int32))
    started = time.perf_counter()
    curve, ancestry_ready, parent_sha, expected_count = [], True, sha(zero/'fields.npz'), 0
    for index, target in enumerate(grid, 1):
        load_value = float(target)
        load = jnp.asarray(load_value, jnp.float64)
        parent_d, parent_H = d, H
        q, reaction, mechanical = solve(d, load, q)
        H = committed_history(parent_H, jnp.asarray(m.driver(np.asarray(q))))
        folder = root/f'state_{index:04d}'
        folder.mkdir()
        consecutive, rows, load_started = 0, [], time.perf_counter()
        atomic_fields(folder/'initial_fields.npz', q=np.asarray(q), d=np.asarray(d), H=np.asarray(H),
            d_parent=np.asarray(parent_d), H_parent=np.asarray(parent_H))
        if index == 1:
            X = jax.lax.stop_gradient(features(H))
            np.testing.assert_allclose(image(net, full_context(H, X)), c.head.initial_damage,
                rtol=1e-12, atol=1e-14)
            context = sample(H, X, generator=np.random.default_rng(a.warmup_seed), warmup=True)
            jax.block_until_ready(step(net, state, context, load))  # compile; discard returned state/weights
            cache.update(slot=a.pool_size, pool=None, block=0)
            atomic_json(root/'preflight.json', dict(pass_initial_image=True, physical_zero=True,
                positive_history=float(H.max()), fresh_optimizer_count=optimizer_count(state, kind='adam'),
                initial_elastic_mechanics=mechanical, gpu_step_compiled=True))
        for sweep in range(1, c.outer_stop.max_sweeps+1):
            previous_q, previous_d, previous_H, previous_R = q, d, H, reaction
            H_used = jax.lax.stop_gradient(H)
            if not np.isfinite(np.asarray(H_used)).all() or float(H_used.min()) < 0 or float(H_used.max()) <= 0:
                raise FloatingPointError('positive-load L1 training needs finite nonnegative nonzero own history')
            X = jax.lax.stop_gradient(features(H_used))
            for update in range(a.inner_updates):
                context = sample(H_used, X, generator=rng, warmup=False)
                net, state, value, gn = step(net, state, context, load)
            expected_count += a.inner_updates
            if optimizer_count(state, kind='adam') != expected_count:
                raise RuntimeError('Adam count reset across batches, outers or loads')
            if not np.isfinite([float(value), float(gn)]).all():
                raise FloatingPointError('nonfinite L1 training objective/gradient')
            damage_image = image(net, full_context(H_used, X))
            r, inner = residual_metrics(np.asarray(damage_image), np.asarray(H_used), D, gc0=gc0, gc1=gc1)
            rho = r/(2*np.sqrt(np.mean(np.asarray(H_used)**2)))
            d = d+c.stagger.omega*(damage_image-d)
            q, reaction, mechanical = solve(d, load, q)
            H = committed_history(parent_H, jnp.asarray(m.driver(np.asarray(q))))
            raw_current = image(net, full_context(H, jax.lax.stop_gradient(features(H))))
            current = residual_metrics(np.asarray(d), np.asarray(H), D, gc0=gc0, gc1=gc1)[1]
            raw = residual_metrics(np.asarray(raw_current), np.asarray(H), D, gc0=gc0, gc1=gc1)[1]
            scale = max(float(abs(H).max()), float(abs(previous_H).max()), c.outer_stop.scale_floor)
            ratio = np.linalg.norm(np.asarray(q).reshape(-1, 2)[:m.n_material], axis=1)/load_value
            row = dict(index=index, load=float(load_value), sweep=sweep, reaction=float(reaction),
                damage_min=float(d.min()), damage_max=float(d.max()),
                fixed_H_image_damage_min=float(damage_image.min()), fixed_H_image_damage_max=float(damage_image.max()),
                fixed_H_image_AT2_relative=inner, coupled_AT2_relative=current,
                raw_current_H_AT2_relative=raw, full_MAE=float(abs(rho).mean()), full_MSE=float((rho**2).mean()),
                damage_change=float(abs(d-previous_d).max()), q_change_over_load=float(abs(q-previous_q).max())/load_value,
                H_change_relative=float(abs(H-previous_H).max())/scale,
                reaction_change_relative=abs(reaction-previous_R)/max(abs(reaction), abs(previous_R), c.outer_stop.scale_floor),
                healing=float(jnp.maximum(parent_d-d, 0).max()), history_decrease=float(jnp.maximum(parent_H-H, 0).max()),
                drift_count=int(np.count_nonzero(ratio > direct.DRIFT_ENVELOPE)), mechanics=mechanical,
                optimizer_count=expected_count, learning_rate=float(schedule(expected_count-1)),
                seconds=time.perf_counter()-load_started, parent_sha256=parent_sha,
                parent_certified_path=ancestry_ready, last_training_loss=float(value))
            outer = evaluate(row, c.outer_stop, c.reporting, sweep=sweep, previous_consecutive=consecutive)
            raw_pass = raw <= c.reporting.coupled_AT2_relative_limit
            outer['checks']['raw_current_H_AT2'] = dict(value=raw,
                limit=c.reporting.coupled_AT2_relative_limit, **{'pass':raw_pass})
            if not raw_pass:
                outer.update(pass_this_sweep=False, consecutive=0, converged=False, reason='raw_current_H_rejected')
            consecutive = outer['consecutive']
            row['outer_convergence'] = outer
            rows.append(row)
            atomic_json(folder/f'sweep_{sweep:04d}.json', row)
            # Every sweep has a complete restart record; named snapshots remain immutable.
            checkpoint = folder/f'outer_{sweep:04d}' if sweep in a.checkpoint_sweeps or outer['converged'] else folder/'pending'
            checkpoint.mkdir(exist_ok=True)
            atomic_fields(checkpoint/'fields.npz', q=np.asarray(q), d=np.asarray(d), H=np.asarray(H),
                H_used=np.asarray(H_used), damage_image=np.asarray(damage_image), raw_current_image=np.asarray(raw_current),
                q_previous=np.asarray(previous_q), d_previous=np.asarray(previous_d), H_previous=np.asarray(previous_H),
                d_parent=np.asarray(parent_d), H_parent=np.asarray(parent_H))
            eqx.tree_serialise_leaves(checkpoint/'network.eqx', net)
            eqx.tree_serialise_leaves(checkpoint/'optimizer.eqx', state)
            atomic_json(checkpoint/'report.json', dict(**row, fields_sha256=sha(checkpoint/'fields.npz'),
                network_sha256=sha(checkpoint/'network.eqx'), optimizer_sha256=sha(checkpoint/'optimizer.eqx'),
                rng_state=rng.bit_generator.state, pool_slot=cache['slot'], pool_block=cache['block']))
            atomic_json(root/'result.json', dict(status='running', complete=False, curve=curve, current=row,
                certified_path=ancestry_ready, requested_loads=grid.tolist()))
            print(json.dumps(dict(event='l1_loading_outer', index=index, load=float(load_value), sweep=sweep,
                reaction=float(reaction), inner_AT2=inner, physical_AT2=current, damage_max=float(d.max()),
                optimizer_count=expected_count, local_ready=outer['converged'])), flush=True)
            if outer['converged']:
                break
        for name in ('fields.npz', 'network.eqx', 'optimizer.eqx'):
            shutil.copyfile(checkpoint/name, folder/name)
        force = m.matrix(np.asarray(d))@np.asarray(q)
        force_norm = float(np.linalg.norm(force[m.free]))
        if force_norm > mechanical['convergence_limit']*(1+1e-3):
            raise RuntimeError('independent host FEM force residual rejected final load')
        np.testing.assert_allclose(force@m.load_direction, reaction, rtol=1e-9, atol=1e-6)
        ancestry_ready = ancestry_ready and outer['converged']
        report = dict(**row, status='ready' if outer['converged'] else 'provisional_budget_exhausted',
            local_ready=outer['converged'], certified_path=ancestry_ready,
            fields_sha256=sha(folder/'fields.npz'), network_sha256=sha(folder/'network.eqx'),
            optimizer_sha256=sha(folder/'optimizer.eqx'), host_force_norm=force_norm,
            host_reaction=float(force@m.load_direction), completed_sweeps=sweep)
        atomic_json(folder/'result.json', report)
        curve.append(report)
        parent_sha = report['fields_sha256']
        atomic_json(root/'result.json', dict(status='running', complete=False, curve=curve, current=report,
            certified_path=ancestry_ready, requested_loads=grid.tolist()))
        print(json.dumps(dict(event='l1_load_complete', index=index, load=float(load_value),
            reaction=float(reaction), status=report['status'], certified_path=ancestry_ready)), flush=True)
    for path, digest in protected.items():
        if sha(path) != digest:
            raise RuntimeError(f'protected physical input changed: {path}')
    for name, digest in code.items():
        if sha(repo/name) != digest:
            raise RuntimeError(f'protected numerical source changed: {name}')
    atomic_json(root/'result.json', dict(status='complete', complete=True, curve=curve, current=curve[-1],
        certified_path=ancestry_ready, requested_loads=grid.tolist(), seconds=time.perf_counter()-started,
        optimizer_count=expected_count, fresh_draws=expected_count, sampling_blocks=cache['block'],
        own_zero=True, fresh_network=True, loss_norm='l1', direct_targets=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    config_path = parser.parse_args().config
    c = runpy.run_path(config_path)['get_config']()
    jax.config.update('jax_enable_x64', True)
    jax.config.update('jax_default_matmul_precision', 'highest')
    run(c, config_path)
