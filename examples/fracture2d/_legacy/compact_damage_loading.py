"""Own-zero staggered FEM mechanics and continuous neural nonlocal damage."""
import argparse
import json
from pathlib import Path
import runpy
import time
from typing import Any
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import load_npz

from examples.fracture2d._legacy import nonlocal_baseline as direct
from examples.fracture2d._legacy.conditioned_damage import make_features, _build_network
from examples.fracture2d._legacy.compact_damage_features import Readout, pack_basis, residual_power_loss
from examples.fracture2d._legacy.coupling_diagnostics import atomic_json
from examples.fracture2d._legacy.damage_optimization import optimize_lbfgs
from examples.fracture2d._legacy.local_damage_trial import load_geometry, sha, at2_metrics
from examples.fracture2d._legacy.outer_stopping import evaluate, validate_controls
from examples.fracture2d._legacy.split_pinn_gradients import make_padded_matvec
from examples.fracture2d._legacy.train import _nonnegative_damage


class JointNetwork(eqx.Module):
    trunk: Any
    readout: Readout


def history_feature_input(H, load, reference_load, mode):
    """Known coefficient feature normalization; residual always uses physical H."""
    H = jax.lax.stop_gradient(H)
    if mode == 'physical':
        return H
    if mode == 'load_squared':
        return H*(reference_load/load)**2
    raise ValueError('loading.history_feature_scaling must be explicit and recognized')


def make_problem(raw_trunk, phi, matvec, c, reference_load):
    """Only known current H and load enter; no frozen physical H closure."""
    if not np.isfinite(reference_load) or reference_load <= 0:
        raise ValueError('neural load reference must be finite positive')
    gc0, gc1 = c.physical.Gc/c.physical.l0, c.physical.Gc*c.physical.l0
    def image(net, H, load):
        raw = raw_trunk(net, H, load)+(load/reference_load)**2*phi(net)
        return _nonnegative_damage(raw, c.physical.damage_floor)
    def metrics(net, H, load):
        d = image(net, H, load)
        parts = (gc0*d, gc1*matvec(d), 2*H*(d-1))
        return dict(AT2_relative=jnp.max(abs(sum(parts)))/sum(jnp.max(abs(p)) for p in parts),
                    damage_min=jnp.min(d), damage_max=jnp.max(d))
    def mse(net, H, load):
        d = image(net, H, load)
        rho = (gc0+2*H)*d+gc1*matvec(d)-2*H
        return jnp.mean((rho/(2*jnp.sqrt(jnp.mean(H**2))))**2)
    def peak(power):
        def objective(net, H, load):
            d = image(net, H, load)
            return residual_power_loss((gc0+2*H)*d+gc1*matvec(d)-2*H, H,
                                       power=power, floor=c.conditioned.loss_floor)
        return objective
    return image, metrics, dict(residual_mse=mse, residual_l16=peak(16), residual_l32=peak(32))


def run(c):
    validate_controls(c.outer_stop)
    targets = np.asarray(c.data.reference_indices, np.int64)*c.data.load_step
    if (jax.devices()[0].platform != 'gpu' or not jax.config.jax_enable_x64
            or not c.outer_stop.enabled or c.stagger.omega != .5
            or np.any(np.diff(targets) <= 0) or targets[0] <= 0
            or not 1 <= c.data.end_state_index <= len(targets)):
        raise ValueError('invalid explicit own-zero neural loading controls/device')
    source = Path(c.loading.source)
    source_result = json.loads((source/'result.json').read_text())
    source_identity = json.loads((source.parent/'identity.json').read_text())
    if (source_result['optimization']['status'] != 'residual_reached'
            or not json.loads((source.parent/'result.json').read_text())['ready']
            or source_identity['damage_correction'] or source_identity['direct_targets']
            or source_identity['config']['compact']['baseline_mode'] != 'raw_conditioned'):
        raise ValueError('own-zero loading requires a ready uncorrected raw-baseline neural seed')
    if (source_result['readout_sha256'] != sha(source/'readout.eqx')
            or source_result['fields_sha256'] != sha(source/'fields.npz')):
        raise ValueError('neural loading seed differs')
    for key in ('physical', 'geometry', 'nl', 'arch'):
        if source_identity['config'][key] != c.to_dict()[key]:
            raise ValueError(f'neural loading seed {key} differs')
    mechanics, operator, geometry = load_geometry(c)
    if geometry != source_identity['geometry']:
        raise ValueError('neural loading geometry differs')
    parity = json.loads((Path(c.trial.geometry_dir)/'fenics_parity.json').read_text())
    if not parity['pass'] or parity['mesh_sha256'] != geometry['mesh_sha256']:
        raise ValueError('verified FEM parity is required')
    matrix = load_npz(source.parent/'basis.npz')
    with np.load(source.parent/'basis_geometry.npz', allow_pickle=False) as f:
        scaling = f['column_scale'].copy()
    phi, _ = pack_basis(matrix, scaling)
    net = eqx.tree_deserialise_leaves(source/'readout.eqx', Readout(jnp.zeros(matrix.shape[1], jnp.float64)))
    trunk_path = Path(c.compact.source)/'network.eqx'
    if sha(trunk_path) != source_identity['source']['network.eqx']:
        raise ValueError('frozen neural trunk differs')
    gc0, gc1 = c.physical.Gc/c.physical.l0, c.physical.Gc*c.physical.l0
    features, info = make_features(operator, mechanics.coordinates[:mechanics.n_material],
        gc0=gc0, length=c.physical.l0, tip=list(c.conditioned.tip),
        powers=list(c.conditioned.powers), roundoff_factor=c.validation.roundoff_factor)
    trunk = eqx.tree_deserialise_leaves(trunk_path, _build_network(c, info['input_dimension']))
    reference_load = float(json.loads((Path(c.optimization.source)/'result.json').read_text())['load'])
    if (type(c.loading.train_trunk) is not bool
            or c.loading.history_feature_scaling not in ('physical', 'load_squared')):
        raise ValueError('loading.train_trunk must be explicit boolean')
    if c.loading.train_trunk:
        net = JointNetwork(trunk, net)
        def raw_trunk(n, H, load):
            known = history_feature_input(H, load, reference_load, c.loading.history_feature_scaling)
            return c.physical.E/gc0*load**2*jax.vmap(n.trunk)(features(known))[:, 0]
        def readout_field(n):
            return phi(n.readout)
    else:
        def raw_trunk(n, H, load):
            known = history_feature_input(H, load, reference_load, c.loading.history_feature_scaling)
            return c.physical.E/gc0*load**2*jax.vmap(trunk)(features(known))[:, 0]
        readout_field = phi
    matvec, _ = make_padded_matvec(operator)
    matvec = jax.jit(matvec)
    image, metrics, objectives = make_problem(raw_trunk, readout_field, matvec, c, reference_load)
    if (not c.loading.methods or set(c.loading.methods)-objectives.keys()
            or len(set(c.loading.methods)) != len(c.loading.methods)):
        raise ValueError('own-zero loading objective stages must be explicit and unique')
    predict, measure = eqx.filter_jit(image), eqx.filter_jit(metrics)
    # This is an ansatz reconstruction audit only. Physical q/d/H below start
    # at zero and never use these diagnostic arrays as a parent or history floor.
    with np.load(source/'fields.npz', allow_pickle=False) as f:
        np.testing.assert_allclose(predict(net, jnp.asarray(f['H_used']), jnp.asarray(reference_load)),
                                   f['damage'], rtol=1e-11, atol=1e-13)
    source_files = [source/'readout.eqx', source/'fields.npz', source/'result.json',
                    source.parent/'identity.json', source.parent/'basis.npz',
                    source.parent/'basis_geometry.npz', trunk_path]
    source_shas = {str(path): sha(path) for path in source_files}
    configuration = c.to_dict()
    configuration['data']['end_state_index'] = 0
    configuration['output']['resume'] = False
    identity = dict(schema='own-zero-compact-neural-loading-v1', config=configuration,
        geometry=geometry, seed=source_shas, reference_load=reference_load,
        runner_sha256=sha(__file__), feature_code_sha256=sha(Path(__file__).with_name('compact_damage_features.py')),
        conditioned_code_sha256=sha(Path(__file__).with_name('conditioned_damage.py')),
        optimizer_sha256=sha(Path(__file__).with_name('damage_optimization.py')),
        own_zero_initial_state=True, direct_targets=False, damage_correction=False,
        frozen_conditioned_trunk=not c.loading.train_trunk, trainable_readouts=matrix.shape[1],
        trainable_parameters=sum(x.size for x in jax.tree.leaves(eqx.filter(net, eqx.is_inexact_array))))
    root = Path(c.loading.output_dir)
    if c.output.resume:
        if json.loads((root/'identity.json').read_text()) != identity:
            raise ValueError('own-zero neural loading resume identity differs')
    else:
        root.mkdir()
        atomic_json(root/'identity.json', identity)
    solve_q = mechanics.gpu_solver(tol=c.validation.cg_rtol, atol=c.validation.cg_atol,
                                   maxiter=c.validation.cg_maxiter)
    q_parent = jnp.zeros(len(mechanics.load_direction), jnp.float64)
    d_parent = jnp.zeros(mechanics.n_material, jnp.float64)
    H_parent = jnp.zeros_like(d_parent)
    curve, parent_sha = [], None
    begin = time.perf_counter()
    for index, load_value in enumerate(targets[:c.data.end_state_index], 1):
        load_value, load = float(load_value), jnp.asarray(load_value)
        folder = root/f'state_{index:04d}'
        if c.output.resume and (folder/'result.json').exists():
            r = json.loads((folder/'result.json').read_text())
            if r['status'] != 'ready' or r['parent_sha256'] != parent_sha or r['state_sha256'] != sha(folder/'fields.npz') or r['network_sha256'] != sha(folder/'network.eqx'):
                raise ValueError('cannot advance a rejected or changed neural loading state')
            with np.load(folder/'fields.npz', allow_pickle=False) as f:
                q_parent, d_parent, H_parent = [jnp.asarray(f[k]) for k in ('q', 'd', 'H')]
            net = eqx.tree_deserialise_leaves(folder/'network.eqx', net)
            parent_sha = r['state_sha256']
            curve.append(r)
            continue
        folder.mkdir()
        started = time.perf_counter()
        d = d_parent
        q, reaction, mechanical = solve_q(d, load, q_parent)
        H = jnp.maximum(H_parent, jnp.asarray(mechanics.driver(np.asarray(q))))
        count, rows, status = 0, [], 'stagger_budget_exhausted'
        for sweep in range(1, c.outer_stop.max_sweeps+1):
            q_previous, d_previous, H_previous, reaction_previous = q, d, H, reaction
            H_used = H
            optimizations = []
            for method in c.loading.methods:
                def progress(row):
                    print(json.dumps(dict(event='inner', index=index, sweep=sweep, method=method, **row)), flush=True)
                net, optimization = optimize_lbfgs(net, objectives[method], metrics, H_used, load,
                    c.loading.lbfgs.to_dict(), progress=progress)
                optimizations.append(dict(method=method, **optimization))
                if optimization['status'] == 'residual_reached':
                    break
            damage_image = predict(net, H_used, load)
            inner = {k: float(v) for k, v in measure(net, H_used, load).items()}
            atomic_json(folder/f'inner_{sweep:04d}.json', dict(stages=optimizations, metrics=inner))
            if optimization['status'] != 'residual_reached':
                status = 'inner_rejected'
                break
            d = d+c.stagger.omega*(damage_image-d)
            q, reaction, mechanical = solve_q(d, load, q)
            H = jnp.maximum(H_parent, jnp.asarray(mechanics.driver(np.asarray(q))))
            field_scale = max(float(jnp.max(abs(H))), float(jnp.max(abs(H_previous))), c.outer_stop.scale_floor)
            ratio = np.linalg.norm(np.asarray(q).reshape(-1, 2)[:mechanics.n_material], axis=1)/load_value
            r = dict(index=index, load=load_value, sweep=sweep, reaction=reaction,
                damage_min=float(jnp.min(d)), damage_max=float(jnp.max(d)),
                fixed_H_image_damage_min=inner['damage_min'], fixed_H_image_damage_max=inner['damage_max'],
                fixed_H_image_AT2_relative=inner['AT2_relative'],
                coupled_AT2_relative=at2_metrics(d, H, matvec, gc0, gc1),
                raw_current_H_AT2_relative=float(measure(net, H, load)['AT2_relative']),
                damage_change=float(jnp.max(abs(d-d_previous))),
                q_change_over_load=float(jnp.max(abs(q-q_previous)))/load_value,
                H_change_relative=float(jnp.max(abs(H-H_previous)))/field_scale,
                reaction_change_relative=abs(reaction-reaction_previous)/max(abs(reaction), abs(reaction_previous), c.outer_stop.scale_floor),
                healing=float(jnp.max(jnp.maximum(d_parent-d, 0.))),
                history_decrease=float(jnp.max(jnp.maximum(H_parent-H, 0.))),
                drift_count=int(np.count_nonzero(ratio > direct.DRIFT_ENVELOPE)),
                mechanics=mechanical, parent_sha256=parent_sha)
            outer = evaluate(r, c.outer_stop, c.reporting, sweep=sweep, previous_consecutive=count)
            raw_pass = r['raw_current_H_AT2_relative'] <= c.reporting.coupled_AT2_relative_limit
            outer['checks']['raw_current_H_AT2'] = dict(value=r['raw_current_H_AT2_relative'],
                limit=c.reporting.coupled_AT2_relative_limit, **{'pass': raw_pass})
            if not raw_pass:
                outer.update(pass_this_sweep=False, consecutive=0, converged=False, reason='raw_neural_current_H_rejected')
            count = outer['consecutive']
            r['outer_convergence'] = outer
            rows.append(r)
            atomic_json(folder/f'sweep_{sweep:04d}.json', r)
            print(json.dumps(dict(event='outer', index=index, sweep=sweep,
                current_AT2=r['coupled_AT2_relative'], raw_current_AT2=r['raw_current_H_AT2_relative'],
                damage_change=r['damage_change'], ready=outer['converged'])), flush=True)
            if outer['converged']:
                status = 'ready'
                break
        # Rejected candidates are retained for diagnostics, never committed parents.
        np.savez_compressed(folder/'fields.npz', q=np.asarray(q), d=np.asarray(d), H=np.asarray(H),
            H_used=np.asarray(H_used), damage_image=np.asarray(damage_image),
            d_parent=np.asarray(d_parent), H_parent=np.asarray(H_parent),
            raw_current_image=np.asarray(predict(net, H, load)))
        eqx.tree_serialise_leaves(folder/'network.eqx', net)
        force = mechanics.matrix(np.asarray(d))@np.asarray(q)
        host_force = float(np.linalg.norm(force[mechanics.free]))
        if host_force > mechanical['convergence_limit']*(1+1e-3):
            raise RuntimeError('independent host neural FEM force replay failed')
        report = dict(index=index, load=load_value, status=status, parent_sha256=parent_sha,
            state_sha256=sha(folder/'fields.npz'), network_sha256=sha(folder/'network.eqx'),
            inner=inner, outer_convergence=rows[-1]['outer_convergence'] if rows else None,
            inner_calls=sweep, accepted_inner_sweeps=len(rows), reaction=reaction,
            seconds=time.perf_counter()-started, host_force_norm=host_force,
            host_reaction=float(force@mechanics.load_direction), physical_state_promoted=status=='ready')
        atomic_json(folder/'result.json', report)
        curve.append(report)
        atomic_json(root/'result.json', dict(complete=True, status=status, curve=curve,
            ready_loads=sum(r['status']=='ready' for r in curve),
            requested_loads=c.data.end_state_index, seconds=time.perf_counter()-begin))
        if status != 'ready':
            break
        q_parent, d_parent, H_parent, parent_sha = q, d, H, report['state_sha256']
    if source_shas != {str(path): sha(path) for path in source_files}:
        raise RuntimeError('neural loading seed changed')
    return curve


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    c = runpy.run_path(args.config)['get_config']()
    c.output.resume = args.resume
    jax.config.update('jax_enable_x64', True)
    jax.config.update('jax_default_matmul_precision', 'highest')
    run(c)
