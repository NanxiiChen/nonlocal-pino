"""Local mechanics and neural-predicted, nodally corrected nonlocal damage.

Only predictor weights are imported from the own-H optimizer experiment.
Physical d/H start at zero; no direct arrays enter training or histories.
Each load trains the predictor once, then the same AT2 equation is solved by
explicit checked Jacobi corrections, warm-started between outer sweeps.
This is a hybrid nodal damage method, not a pure PINN solution.
"""
import argparse
import json
from pathlib import Path
import runpy
import shutil
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from examples.fracture2d._legacy import nonlocal_baseline as baseline
from examples.fracture2d._legacy.coupling_diagnostics import atomic_json
from examples.fracture2d._legacy.damage_correction import make_corrector
from examples.fracture2d._legacy.damage_optimization import make_objectives, optimize_lbfgs
from examples.fracture2d._legacy.local_damage_trial import at2_metrics, load_geometry, sha
from examples.fracture2d._legacy.outer_stopping import evaluate, validate_controls
from examples.fracture2d._legacy.split_pinn_gradients import make_padded_matvec
from examples.fracture2d._legacy.train import _build_network, _nonnegative_damage


def run(c):
    validate_controls(c.outer_stop)
    if (jax.devices()[0].platform != 'gpu' or not jax.config.jax_enable_x64
            or c.trial.damage_solver != 'pinn_with_nodal_correction' or c.stagger.omega != .5
            or c.predictor.initial_projection not in ('none', 'unit_interval')):
        raise ValueError('corrected trial requires explicit FP64 GPU/hybrid/relaxation controls')
    targets = np.asarray(c.data.reference_indices, np.int64)*c.data.load_step
    if np.any(np.diff(targets) <= 0) or not 1 <= c.data.end_state_index <= len(targets):
        raise ValueError('invalid corrected-trial loading schedule')
    source = Path(c.predictor.source)
    if sha(source) != c.predictor.source_sha256:
        raise ValueError('neural predictor source SHA differs')
    start = time.perf_counter()
    mechanics, operator, geometry = load_geometry(c)
    validation = json.loads((Path(c.trial.geometry_dir)/'fenics_parity.json').read_text())
    if not validation['pass'] or validation['mesh_sha256'] != geometry['mesh_sha256']:
        raise ValueError('independent UFL validation is missing or mismatched')
    root = Path(c.trial.output_base)
    config = c.to_dict()
    config['data']['end_state_index'], config['output']['resume'] = 0, False
    identity = dict(schema='neural-predictor-nodal-corrector-v2', config=config, geometry=geometry,
        runner_sha256=sha(__file__), correction_sha256=sha(Path(__file__).with_name('damage_correction.py')),
        optimizer_sha256=sha(Path(__file__).with_name('damage_optimization.py')))
    if c.output.resume:
        if json.loads((root/'identity.json').read_text()) != identity:
            raise ValueError('corrected-trial resume identity differs')
    else:
        root.mkdir()
        atomic_json(root/'identity.json', identity)
        if c.trial.fork_source:
            predecessor = Path(c.trial.fork_source)
            predecessor_identity = json.loads((predecessor/'identity.json').read_text())
            if (predecessor_identity['schema'] != 'neural-predictor-nodal-corrector-v1'
                    or predecessor_identity['geometry'] != geometry
                    or type(c.trial.fork_ready_loads) is not int
                    or not 1 <= c.trial.fork_ready_loads <= c.data.end_state_index):
                raise ValueError('invalid ready-prefix hybrid fork')
            previous_config = predecessor_identity['config']
            expected = json.loads(json.dumps(config))
            del previous_config['trial']['output_base']
            for key in ('output_base', 'fork_source', 'fork_ready_loads'):
                del expected['trial'][key]
            del expected['predictor']['initial_projection']
            if expected != previous_config:
                raise ValueError('hybrid fork changed physical/history/solver controls')
            rows = json.loads((predecessor/'result.json').read_text())['curve']
            if len(rows) != c.trial.fork_ready_loads:
                raise ValueError('hybrid fork count does not match completed ready prefix')
            parent = None
            for index, row in enumerate(rows, 1):
                folder = predecessor/f'state_{index:04d}'
                if (row['status'] != 'ready' or not row['outer_convergence']['converged']
                        or not all(v['pass'] for v in row['outer_convergence']['checks'].values())
                        or row['parent_sha256'] != parent
                        or row['state_sha256'] != sha(folder/'fields.npz')
                        or row['network_sha256'] != sha(folder/'network.eqx')
                        or not row['host_force_pass']):
                    raise ValueError('hybrid fork predecessor is not a verified ready state')
                shutil.copytree(folder, root/f'state_{index:04d}')
                parent = row['state_sha256']
            shutil.copy2(predecessor/'prefix2_hashes.json', root/'prefix2_hashes.json')
            atomic_json(root/'fork_origin.json', dict(source=str(predecessor),
                source_identity_sha256=sha(predecessor/'identity.json'),
                source_report_sha256=sha(predecessor/'result.json'), ready_loads=len(rows),
                changed_algorithm='project initial predictor only; unchanged AT2 fixed point'))
            c.output.resume = True
        elif c.trial.fork_ready_loads != 0:
            raise ValueError('fork count requires an explicit source')
    solve_q = mechanics.gpu_solver(tol=c.validation.cg_rtol, atol=c.validation.cg_atol,
                                   maxiter=c.validation.cg_maxiter)
    gc0, gc1 = c.physical.Gc/c.physical.l0, c.physical.Gc*c.physical.l0
    correct = make_corrector(operator, mechanics.volumes, gc0, gc1, c.correction.to_dict(),
                             roundoff_factor=c.validation.roundoff_factor)
    matvec, _ = make_padded_matvec(operator)
    matvec = jax.jit(matvec)
    points, volumes = jnp.asarray(mechanics.coordinates[:mechanics.n_material]), jnp.asarray(mechanics.volumes)
    template = jax.tree.map(lambda x: x.astype(jnp.float64),
        _build_network(c, jax.random.PRNGKey(c.train.seed+1), out_dim=1))
    net = eqx.tree_deserialise_leaves(source, template)

    def predict(n, load):
        return _nonnegative_damage(c.physical.E/gc0*load**2*jax.vmap(n)(points)[:, 0], c.physical.damage_floor)
    _, objective = make_objectives(predict, matvec, volumes, gc0, gc1)
    def metrics(n, H, load):
        d = predict(n, load)
        parts = (gc0*d, gc1*matvec(d), 2*H*(d-1))
        return dict(AT2_relative=jnp.max(abs(sum(parts)))/sum(jnp.max(abs(x)) for x in parts),
                    damage_min=jnp.min(d), damage_max=jnp.max(d))
    predict = eqx.filter_jit(predict)
    q_parent = jnp.zeros(len(mechanics.load_direction), jnp.float64)
    d_parent = jnp.zeros(mechanics.n_material, jnp.float64)
    H_parent = jnp.zeros_like(d_parent)
    parent_sha, curve = None, []
    for index, load_value in enumerate(targets[:c.data.end_state_index], 1):
        load_value, folder = float(load_value), root/f'state_{index:04d}'
        load = jnp.asarray(load_value)
        if c.output.resume and (folder/'result.json').exists():
            report = json.loads((folder/'result.json').read_text())
            if not report['outer_convergence']['converged']:
                raise ValueError('cannot resume beyond rejected corrected load')
            if (report['parent_sha256'] != parent_sha or report['state_sha256'] != sha(folder/'fields.npz')
                    or report['network_sha256'] != sha(folder/'network.eqx')):
                raise ValueError('corrected-trial checkpoint ancestry differs')
            with np.load(folder/'fields.npz', allow_pickle=False) as f:
                q_parent, d_parent, H_parent = [jnp.asarray(f[k]) for k in ('q', 'd', 'H')]
            net = eqx.tree_deserialise_leaves(folder/'network.eqx', net)
            curve.append(report)
            parent_sha = report['state_sha256']
            continue
        folder.mkdir()
        load_start = time.perf_counter()
        d = d_parent
        q, reaction, mechanical = solve_q(d, load, q_parent)
        H = jnp.maximum(H_parent, jnp.asarray(mechanics.driver(np.asarray(q))))
        def progress(row):
            print(json.dumps(dict(event='predictor', index=index, **row)), flush=True)
        net, predictor = optimize_lbfgs(net, objective, metrics, H, load,
                                      c.predictor.lbfgs.to_dict(), progress=progress)
        atomic_json(folder/'predictor.json', predictor)
        neural_image = predict(net, load)
        damage_image = (jnp.clip(neural_image, 0., 1.)
                        if c.predictor.initial_projection == 'unit_interval' else neural_image)
        projection_inf = float(jnp.max(abs(damage_image-neural_image)))
        count, rows = 0, []
        for sweep in range(1, c.outer_stop.max_sweeps+1):
            q_previous, d_previous, H_previous, reaction_previous = q, d, H, reaction
            H_used = H
            damage_image, correction = correct(H_used, damage_image)
            d = d+c.stagger.omega*(damage_image-d)
            q, reaction, mechanical = solve_q(d, load, q)
            H = jnp.maximum(H_parent, jnp.asarray(mechanics.driver(np.asarray(q))))
            ratio = np.linalg.norm(np.asarray(q).reshape(-1, 2)[:mechanics.n_material], axis=1)/load_value
            field_scale = max(float(jnp.max(abs(H))), float(jnp.max(abs(H_previous))), c.outer_stop.scale_floor)
            report = dict(index=index, load=load_value, sweep=sweep, reaction=reaction,
                damage_min=float(jnp.min(d)), damage_max=float(jnp.max(d)),
                fixed_H_image_damage_min=float(jnp.min(damage_image)),
                fixed_H_image_damage_max=float(jnp.max(damage_image)),
                fixed_H_image_AT2_relative=at2_metrics(damage_image, H_used, matvec, gc0, gc1),
                coupled_AT2_relative=at2_metrics(d, H, matvec, gc0, gc1),
                raw_neural_AT2_relative=at2_metrics(neural_image, H, matvec, gc0, gc1),
                damage_change=float(jnp.max(abs(d-d_previous))),
                q_change_over_load=float(jnp.max(abs(q-q_previous)))/load_value,
                H_change_relative=float(jnp.max(abs(H-H_previous)))/field_scale,
                reaction_change_relative=abs(reaction-reaction_previous)/max(abs(reaction), abs(reaction_previous), c.outer_stop.scale_floor),
                healing=float(jnp.max(jnp.maximum(d_parent-d, 0.))),
                history_decrease=float(jnp.max(jnp.maximum(H_parent-H, 0.))),
                drift_count=int(np.count_nonzero(ratio > baseline.DRIFT_ENVELOPE)),
                max_displacement_ratio=float(np.max(ratio)), mechanics=mechanical, correction=correction,
                diagnostic_only=True, parent_sha256=parent_sha,
                damage_method='neural_predictor_plus_nodal_jacobi')
            outer = evaluate(report, c.outer_stop, c.reporting, sweep=sweep, previous_consecutive=count)
            count, report['outer_convergence'] = outer['consecutive'], outer
            rows.append(report)
            atomic_json(folder/f'sweep_{sweep:04d}.json', report)
            print(json.dumps({k: report[k] for k in ('index','load','sweep','reaction','coupled_AT2_relative','raw_neural_AT2_relative','damage_change')}), flush=True)
            if outer['converged']:
                break
        report.update(status='ready' if outer['converged'] else 'stagger_budget_exhausted',
            seconds=time.perf_counter()-load_start, sweeps=len(rows), predictor=predictor,
            initial_projection_inf=projection_inf,
            correction_iterations=sum(r['correction']['iterations'] for r in rows),
            correction_seconds=sum(r['correction']['seconds'] for r in rows))
        np.savez_compressed(folder/'fields.npz', q=np.asarray(q), d=np.asarray(d), H=np.asarray(H),
            H_used=np.asarray(H_used), damage_image=np.asarray(damage_image),
            d_parent=np.asarray(d_parent), H_parent=np.asarray(H_parent), neural_image=np.asarray(neural_image))
        eqx.tree_serialise_leaves(folder/'network.eqx', net)
        report.update(state_sha256=sha(folder/'fields.npz'), network_sha256=sha(folder/'network.eqx'))
        force = mechanics.matrix(np.asarray(d))@np.asarray(q)
        report['host_force_norm'] = float(np.linalg.norm(force[mechanics.free]))
        report['host_reaction'] = float(force@mechanics.load_direction)
        report['host_force_pass'] = report['host_force_norm'] <= mechanical['convergence_limit']*(1+1e-3)
        if not report['host_force_pass']:
            raise RuntimeError('corrected-trial independent host force replay fails')
        atomic_json(folder/'result.json', report)
        curve.append(report)
        atomic_json(root/'result.json', dict(completed_load_snapshot=True,
            ready_loads=sum(r['status']=='ready' for r in curve), curve=curve,
            seconds=time.perf_counter()-start, damage_method='neural_predictor_plus_nodal_jacobi'))
        if not outer['converged']:
            break
        q_parent, d_parent, H_parent, parent_sha = q, d, H, report['state_sha256']
    if sha(source) != c.predictor.source_sha256:
        raise RuntimeError('neural predictor source changed during corrected trial')
    return curve


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--last-index', type=int)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = runpy.run_path(args.config)['get_config']()
    if args.last_index is not None:
        config.data.end_state_index = args.last_index
    config.output.resume = args.resume
    jax.config.update('jax_enable_x64', True)
    jax.config.update('jax_default_matmul_precision', 'highest')
    run(config)
