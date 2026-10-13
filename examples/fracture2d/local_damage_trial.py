"""Independent local-mechanics/direct-or-PINN nonlocal damage experiment.

Prepare geometry on CPU, verify it in fenicsenv, then run serialized JAX GPU
work in .venv. The preceding accepted own H remains the floor; no reference
history or direct-damage fields enter neural training. No warning advancement.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from examples.fracture2d import nonlocal_baseline as direct
from examples.fracture2d.at2_cg import damage_solver
from examples.fracture2d.coupling_diagnostics import atomic_json
from examples.fracture2d.local_mechanics import LocalMechanics
from examples.fracture2d.outer_stopping import evaluate, validate_controls
from examples.fracture2d.split_horizon_model import read_cloud, visibility_graph
from examples.fracture2d.split_pinn_gradients import make_padded_matvec
from examples.fracture2d.train import _build_network, _nonnegative_damage
from pinn.operators import ExplicitVolumeSparseD


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def geometry_identity(c):
    return dict(cloud_sha256=c.data.cloud_sha256, delta_d=c.nl.delta_d,
                visibility_tolerance=c.nl.visibility_tolerance,
                grip_extent=c.geometry.grip_extent, h_grip=c.geometry.h_grip,
                E=c.physical.E, nu=c.physical.nu, k_res=c.physical.k_res,
                schema='p1-local-volume-grips-amor-lumped-v1')


def prepare(c):
    root = Path(c.trial.geometry_dir)
    root.mkdir()
    cloud = read_cloud(c.data.cloud, c.data.cloud_sha256)
    points, tri, V = cloud['coordinates'], cloud['triangles'], cloud['volumes']
    print(json.dumps(dict(event='prepare', material_nodes=len(points), cells=len(tri))), flush=True)
    offsets = np.arange(c.geometry.h_grip, c.geometry.grip_extent, c.geometry.h_grip)
    offsets = np.concatenate((offsets[offsets < c.geometry.grip_extent], [c.geometry.grip_extent]))
    coordinates = points.tolist()
    bottom = direct._g_validate_interface_coordinates(points, cloud['bottom'], 0., 'local bottom')
    top = direct._g_validate_interface_coordinates(points, cloud['top'], 1., 'local top')
    btri, _ = direct._g_structured_grip_triangles(points, bottom, offsets, -1., coordinates, 'bottom')
    ttri, _ = direct._g_structured_grip_triangles(points, top, offsets, 1., coordinates, 'top')
    coordinates = np.asarray(coordinates, np.float64)
    triangles = np.concatenate((tri, btri, ttri)).astype(np.int64)
    collar = np.arange(len(coordinates)) >= len(points)
    bnodes = np.flatnonzero(collar & (coordinates[:, 1] < 0))
    tnodes = np.flatnonzero(collar & (coordinates[:, 1] > 1))
    constrained = np.sort(np.concatenate((2*bnodes+1, 2*tnodes+1, [0]))).astype(np.int64)
    direction = np.zeros(2*len(coordinates))
    direction[2*tnodes+1] = 1.
    _, _, edges, _ = visibility_graph(points, tri, c.nl.delta_d, c.nl.visibility_tolerance)
    operator = ExplicitVolumeSparseD(points, V, edges, c.nl.delta_d, 'cubic')
    from scipy.sparse import save_npz
    save_npz(root/'damage_operator.npz', operator.to_scipy())
    np.savez_compressed(root/'mesh.npz', coordinates=coordinates, triangles=triangles,
        n_material=np.int64(len(points)), n_material_cells=np.int64(len(tri)),
        volumes=V, constrained=constrained, load_direction=direction, damage_edges=edges)
    atomic_json(root/'identity.json', dict(config=geometry_identity(c),
        mesh_sha256=sha(root/'mesh.npz'), operator_sha256=sha(root/'damage_operator.npz')))
    print(json.dumps(dict(event='prepared', root=str(root), union_nodes=len(coordinates), damage_edges=len(edges))), flush=True)


def load_geometry(c):
    root = Path(c.trial.geometry_dir)
    identity = json.loads((root/'identity.json').read_text())
    if (identity['config'] != geometry_identity(c) or identity['mesh_sha256'] != sha(root/'mesh.npz')
            or identity['operator_sha256'] != sha(root/'damage_operator.npz')):
        raise ValueError('prepared local trial geometry identity differs')
    with np.load(root/'mesh.npz', allow_pickle=False) as f:
        values = {name: f[name].copy() for name in f.files}
    mechanics = LocalMechanics(values['coordinates'], values['triangles'], int(values['n_material']),
        int(values['n_material_cells']), values['volumes'], E=c.physical.E, nu=c.physical.nu,
        k_res=c.physical.k_res, constrained=values['constrained'], load_direction=values['load_direction'])
    operator = ExplicitVolumeSparseD(mechanics.coordinates[:mechanics.n_material], mechanics.volumes,
                                    values['damage_edges'], c.nl.delta_d, 'cubic')
    from scipy.sparse import load_npz
    rebuilt, prepared = operator.to_scipy(), load_npz(root/'damage_operator.npz')
    difference = rebuilt-prepared
    coefficient_limit = c.validation.roundoff_factor*np.finfo(float).eps*np.max(abs(prepared.data))
    if (not np.array_equal(rebuilt.indices, prepared.indices)
            or not np.array_equal(rebuilt.indptr, prepared.indptr)
            or difference.nnz and np.max(abs(difference.data)) > coefficient_limit):
        raise ValueError('rebuilt damage operator differs from prepared coefficients')
    return mechanics, operator, identity


def at2_metrics(d, H, matvec, gc0, gc1):
    parts = (gc0*d, gc1*matvec(d), 2*H*(d-1))
    scale = sum(float(jnp.max(jnp.abs(p))) for p in parts)
    if not np.isfinite(scale) or scale <= 0:
        raise FloatingPointError('invalid AT2 reporting scale')
    return float(jnp.max(jnp.abs(sum(parts))))/scale


def run(c):
    validate_controls(c.outer_stop)
    targets = np.asarray(c.data.reference_indices, np.int64)*c.data.load_step
    if (c.trial.damage_solver not in ('direct', 'pinn') or np.any(np.diff(targets) <= 0)
            or not 1 <= c.data.end_state_index <= len(targets) or c.stagger.omega != .5):
        raise ValueError('invalid explicit local trial solver/loading/relaxation')
    if jax.devices()[0].platform != 'gpu':
        raise RuntimeError('local physical trial requires serialized GPU execution')
    start = time.perf_counter()
    mechanics, operator, geo_identity = load_geometry(c)
    fem_validation = json.loads((Path(c.trial.geometry_dir)/'fenics_parity.json').read_text())
    if not fem_validation['pass'] or fem_validation['mesh_sha256'] != geo_identity['mesh_sha256']:
        raise ValueError('independent UFL validation missing or bound to another mesh')
    root = Path(c.trial.output_base+'_'+c.trial.damage_solver)
    config = c.to_dict()
    config['data']['end_state_index'] = 0
    config['output']['resume'] = False
    identity = dict(schema='local-mechanics-nonlocal-damage-trial-v1', config=config,
                    geometry=geo_identity, boundary='same_engineering_volume_grips',
                    driver='plane_strain_Amor_cell_to_node_lumped_L2',
                    runner_sha256=sha(__file__),
                    mechanics_code_sha256=sha(Path(__file__).with_name('local_mechanics.py')))
    if c.output.resume:
        if json.loads((root/'identity.json').read_text()) != identity:
            raise ValueError('local trial resume identity differs')
    else:
        root.mkdir()
        atomic_json(root/'identity.json', identity)
    solve_q = mechanics.gpu_solver(tol=c.validation.cg_rtol, atol=c.validation.cg_atol,
                                   maxiter=c.validation.cg_maxiter)
    gc0, gc1 = c.physical.Gc/c.physical.l0, c.physical.Gc*c.physical.l0
    solve_d = damage_solver(operator, mechanics.volumes, gc0, gc1,
                           tol=c.validation.cg_rtol, atol=c.validation.cg_atol, maxiter=c.validation.cg_maxiter)
    matvec, _ = make_padded_matvec(operator)
    matvec = jax.jit(matvec)
    q_parent = jnp.zeros(len(mechanics.load_direction), jnp.float64)
    d_parent = jnp.zeros(mechanics.n_material, jnp.float64)
    H_parent = jnp.zeros_like(d_parent)
    net, state, optimizer = None, None, None
    if c.trial.damage_solver == 'pinn':
        net = jax.tree.map(lambda x: x.astype(jnp.float64),
            _build_network(c, jax.random.PRNGKey(c.train.seed+1), out_dim=1))
        optimizer = optax.adam(c.train.learning_rate)
        state = optimizer.init(eqx.filter(net, eqx.is_inexact_array))
    points = jnp.asarray(mechanics.coordinates[:mechanics.n_material])
    def image(n, load):
        raw = c.physical.E/gc0*load**2*jax.vmap(n)(points)[:, 0]
        return _nonnegative_damage(raw, c.physical.damage_floor)
    @eqx.filter_jit
    def step(n, s, H, load, scale):
        def objective(nn):
            d = image(nn, load)
            rho = gc0*d+gc1*matvec(d)+2*jax.lax.stop_gradient(H)*(d-1)
            return jnp.mean((rho/scale)**2)
        loss, grad = eqx.filter_value_and_grad(objective)(n)
        updates, s = optimizer.update(grad, s, eqx.filter(n, eqx.is_inexact_array))
        return eqx.apply_updates(n, updates), s, loss
    curve, parent_sha = [], None
    for index, load_value in enumerate(targets[:c.data.end_state_index], 1):
        load_value = float(load_value)
        load = jnp.asarray(load_value)
        folder = root/f'state_{index:04d}'
        if c.output.resume and (folder/'result.json').exists():
            report = json.loads((folder/'result.json').read_text())
            if not report['outer_convergence']['converged']:
                raise ValueError('cannot advance/resume a rejected local trial')
            if report['parent_sha256'] != parent_sha or report['state_sha256'] != sha(folder/'fields.npz'):
                raise ValueError('local trial checkpoint ancestry differs')
            with np.load(folder/'fields.npz', allow_pickle=False) as f:
                q_parent, d_parent, H_parent = [jnp.asarray(f[k]) for k in ('q', 'd', 'H')]
            if c.trial.damage_solver == 'pinn':
                net, state = eqx.tree_deserialise_leaves(folder/'network.eqx', (net, state))
                if report['network_sha256'] != sha(folder/'network.eqx'):
                    raise ValueError('local neural checkpoint SHA differs')
            parent_sha = report['state_sha256']
            curve.append(report)
            continue
        folder.mkdir()
        load_start = time.perf_counter()
        d = d_parent
        q, reaction, mechanical = solve_q(d, load, q_parent)
        H = jnp.maximum(H_parent, jnp.asarray(mechanics.driver(np.asarray(q))))
        count, rows = 0, []
        for sweep in range(1, c.outer_stop.max_sweeps+1):
            q_previous, d_previous, H_previous, reaction_previous = q, d, H, reaction
            H_used = H
            damage_start = time.perf_counter()
            if c.trial.damage_solver == 'direct':
                damage_image = solve_d(H_used, d)
            else:
                scale = 2*jnp.sqrt(jnp.mean(H_used**2))
                for update in range(1, c.stagger.damage_steps+1):
                    net, state, loss = step(net, state, H_used, load, scale)
                    if update % c.output.log_every == 0:
                        print(json.dumps(dict(event='train', index=index, sweep=sweep,
                                              update=update, loss=float(loss))), flush=True)
                damage_image = image(net, load)
            damage_image.block_until_ready()
            damage_seconds = time.perf_counter()-damage_start
            d = d+c.stagger.omega*(damage_image-d)
            q, reaction, mechanical = solve_q(d, load, q)
            H = jnp.maximum(H_parent, jnp.asarray(mechanics.driver(np.asarray(q))))
            field_scale = max(float(jnp.max(abs(H))), float(jnp.max(abs(H_previous))), c.outer_stop.scale_floor)
            ratio = np.linalg.norm(np.asarray(q).reshape(-1, 2)[:mechanics.n_material], axis=1)/load_value
            report = dict(index=index, load=load_value, sweep=sweep, reaction=reaction,
                damage_min=float(jnp.min(d)), damage_max=float(jnp.max(d)),
                fixed_H_image_damage_min=float(jnp.min(damage_image)),
                fixed_H_image_damage_max=float(jnp.max(damage_image)),
                fixed_H_image_AT2_relative=at2_metrics(damage_image, H_used, matvec, gc0, gc1),
                coupled_AT2_relative=at2_metrics(d, H, matvec, gc0, gc1),
                damage_change=float(jnp.max(abs(d-d_previous))),
                q_change_over_load=float(jnp.max(abs(q-q_previous)))/load_value,
                H_change_relative=float(jnp.max(abs(H-H_previous)))/field_scale,
                reaction_change_relative=abs(reaction-reaction_previous)/max(abs(reaction), abs(reaction_previous), c.outer_stop.scale_floor),
                healing=float(jnp.max(jnp.maximum(d_parent-d, 0.))),
                history_decrease=float(jnp.max(jnp.maximum(H_parent-H, 0.))),
                drift_count=int(np.count_nonzero(ratio > direct.DRIFT_ENVELOPE)),
                max_displacement_ratio=float(np.max(ratio)), mechanics=mechanical,
                damage_seconds=damage_seconds, diagnostic_only=True, parent_sha256=parent_sha)
            outer = evaluate(report, c.outer_stop, c.reporting, sweep=sweep, previous_consecutive=count)
            count = outer['consecutive']
            report['outer_convergence'] = outer
            rows.append(report)
            atomic_json(folder/f'sweep_{sweep:04d}.json', report)
            print(json.dumps({k: report[k] for k in ('index', 'load', 'sweep', 'reaction', 'damage_max', 'coupled_AT2_relative', 'damage_change', 'drift_count')}), flush=True)
            if outer['converged']:
                break
        report['status'] = 'ready' if outer['converged'] else 'stagger_budget_exhausted'
        report['seconds'] = time.perf_counter()-load_start
        report['sweeps'] = len(rows)
        report['damage_total_seconds'] = sum(r['damage_seconds'] for r in rows)
        np.savez_compressed(folder/'fields.npz', q=np.asarray(q), d=np.asarray(d), H=np.asarray(H),
            H_used=np.asarray(H_used), damage_image=np.asarray(damage_image), d_parent=np.asarray(d_parent), H_parent=np.asarray(H_parent))
        report['state_sha256'] = sha(folder/'fields.npz')
        if c.trial.damage_solver == 'pinn':
            eqx.tree_serialise_leaves(folder/'network.eqx', (net, state))
            report['network_sha256'] = sha(folder/'network.eqx')
        # Independent host replay, in addition to the GPU solve's explicit check.
        force = mechanics.matrix(np.asarray(d)) @ np.asarray(q)
        report['host_force_norm'] = float(np.linalg.norm(force[mechanics.free]))
        report['host_reaction'] = float(force @ mechanics.load_direction)
        report['host_force_pass'] = report['host_force_norm'] <= mechanical['convergence_limit']*(1+1e-3)
        if not report['host_force_pass']:
            raise RuntimeError('independent host FEM force replay failed')
        atomic_json(folder/'result.json', report)
        curve.append(report)
        atomic_json(root/'result.json', dict(complete=True, status=report['status'],
            damage_solver=c.trial.damage_solver, ready_loads=sum(r['status']=='ready' for r in curve),
            seconds=time.perf_counter()-start, curve=curve))
        if not outer['converged']:
            break
        q_parent, d_parent, H_parent, parent_sha = q, d, H, report['state_sha256']
    return curve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--damage-solver', choices=['direct', 'pinn'])
    parser.add_argument('--last-index', type=int)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    c = runpy.run_path(args.config)['get_config']()
    if args.damage_solver is not None:
        c.trial.damage_solver = args.damage_solver
    if args.last_index is not None:
        c.data.end_state_index = args.last_index
    c.output.resume = args.resume
    jax.config.update('jax_enable_x64', True)
    jax.config.update('jax_default_matmul_precision', 'highest')
    if args.prepare:
        prepare(c)
    else:
        run(c)


if __name__ == '__main__':
    main()
