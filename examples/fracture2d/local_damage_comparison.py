"""Matched lumped-P1 damage control for local-mechanics/nonlocal-AT2 trials.

Only the damage spatial operator changes: M D_local = S, where S is the
standard P1 gradient weak form on material cells. Natural zero-flux boundaries,
nodal-lumped reaction/history, mechanics, loading and acceptance stay explicit.
This is a discrete FEM control, not a continuum-converged reference.
"""
import argparse
import json
from pathlib import Path
import runpy
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import coo_matrix, diags, load_npz

from examples.fracture2d.at2_cg import damage_solver
from examples.fracture2d.coupling_diagnostics import atomic_json
from examples.fracture2d.local_damage_trial import at2_metrics, load_geometry, sha
from examples.fracture2d.outer_stopping import evaluate, validate_controls
from examples.fracture2d import nonlocal_baseline as direct


class LocalDamageOperator:
    """Integral grad(N_i).grad(N_j), divided by lumped nodal volumes."""

    def __init__(self, mechanics):
        n, nc = mechanics.n_material, mechanics.n_material_cells
        tri = mechanics.triangles[:nc]
        # The engineering-strain B stores dN/dx and dN/dy in these entries.
        gradients = np.stack((mechanics.B[:nc, 0, 0::2],
                              mechanics.B[:nc, 1, 1::2]), axis=2)
        element = mechanics.areas[:nc, None, None] * np.einsum('eia,eja->eij', gradients, gradients)
        rows = np.broadcast_to(tri[:, :, None], element.shape)
        columns = np.broadcast_to(tri[:, None, :], element.shape)
        self.stiffness = coo_matrix((element.ravel(), (rows.ravel(), columns.ravel())),
                                   shape=(n, n)).tocsr()
        self.matrix = (diags(1 / mechanics.volumes) @ self.stiffness).tocsr()
        self.matrix.sort_indices()
        self.shape = self.matrix.shape
        skew = self.stiffness - self.stiffness.T
        limit = 256 * np.finfo(float).eps * np.max(abs(self.stiffness.data))
        if ((skew.nnz and np.max(abs(skew.data)) > limit)
                or np.max(abs(self.stiffness @ np.ones(n))) > limit
                or np.any(self.stiffness.diagonal() <= 0)):
            raise ValueError('invalid symmetric P1 gradient stiffness/constant mode')

    def to_scipy(self):
        return self.matrix.copy()

    def gpu_matvec(self):
        """Pack the fixed CSR rows once; use the same coefficients as checked CG."""
        matrix = self.matrix
        width = int(np.max(np.diff(matrix.indptr)))
        columns = np.zeros((matrix.shape[0], width), np.int32)
        weights = np.zeros_like(columns, dtype=np.float64)
        for row in range(matrix.shape[0]):
            start, end = matrix.indptr[row:row + 2]
            columns[row, :end-start] = matrix.indices[start:end]
            weights[row, :end-start] = matrix.data[start:end]
        columns, weights = jnp.asarray(columns), jnp.asarray(weights)
        return jax.jit(lambda x: jnp.sum(weights * x[columns], axis=1))


def checked_local_operator(mechanics, root, geometry, factor):
    operator = LocalDamageOperator(mechanics)
    parity = json.loads((root / 'local_scalar_parity.json').read_text())
    if not parity['pass'] or parity['mesh_sha256'] != geometry['mesh_sha256']:
        raise ValueError('local scalar UFL validation is missing or uses another mesh')
    cached = load_npz(root / 'local_scalar_stiffness.npz')
    difference = operator.stiffness - cached
    limit = factor * np.finfo(float).eps * np.max(abs(cached.data))
    if (cached.shape != operator.shape
            or difference.nnz and np.max(abs(difference.data)) > limit):
        raise ValueError('rebuilt local scalar stiffness differs from UFL-checked cache')
    return operator


def run(c):
    validate_controls(c.outer_stop)
    targets = np.asarray(c.comparison.loads, np.float64)
    if (jax.devices()[0].platform != 'gpu' or targets.ndim != 1 or len(targets) == 0
            or not np.isfinite(targets).all() or np.min(targets) <= 0
            or np.any(np.diff(targets) <= 0) or c.stagger.omega != .5
            or not 1 <= c.data.end_state_index <= len(targets)):
        raise ValueError('matched control requires GPU, ordered positive loads and explicit policy')
    begin = time.perf_counter()
    mechanics, _, geometry = load_geometry(c)
    cache = Path(c.trial.geometry_dir)
    parity = json.loads((cache / 'fenics_parity.json').read_text())
    if not parity['pass'] or parity['mesh_sha256'] != geometry['mesh_sha256']:
        raise ValueError('independent UFL mechanics validation missing or mesh differs')
    operator = checked_local_operator(mechanics, cache, geometry, c.validation.roundoff_factor)
    root = Path(c.comparison.output_dir)
    config = c.to_dict()
    config['data']['end_state_index'] = 0
    config['output']['resume'] = False
    identity = dict(schema='local-gradient-damage-control-v1', config=config,
                    geometry=geometry, boundary='same_engineering_volume_grips',
                    driver='plane_strain_Amor_cell_to_node_lumped_L2',
                    damage_operator='material_P1_gradient_stiffness_over_lumped_volume',
                    scalar_cache_sha256=sha(cache / 'local_scalar_stiffness.npz'),
                    runner_sha256=sha(__file__),
                    mechanics_code_sha256=sha(Path(__file__).with_name('local_mechanics.py')),
                    damage_cg_sha256=sha(Path(__file__).with_name('at2_cg.py')))
    if c.output.resume:
        if json.loads((root / 'identity.json').read_text()) != identity:
            raise ValueError('matched local control resume identity differs')
    else:
        root.mkdir()
        atomic_json(root / 'identity.json', identity)
    solve_q = mechanics.gpu_solver(tol=c.validation.cg_rtol, atol=c.validation.cg_atol,
                                    maxiter=c.validation.cg_maxiter)
    gc0, gc1 = c.physical.Gc / c.physical.l0, c.physical.Gc * c.physical.l0
    solve_d = damage_solver(operator, mechanics.volumes, gc0, gc1,
                           tol=c.validation.cg_rtol, atol=c.validation.cg_atol,
                           maxiter=c.validation.cg_maxiter)
    matvec = operator.gpu_matvec()
    q_parent = jnp.zeros(len(mechanics.load_direction), jnp.float64)
    d_parent = jnp.zeros(mechanics.n_material, jnp.float64)
    H_parent = jnp.zeros_like(d_parent)
    parent_sha, curve = None, []
    for index, target in enumerate(targets[:c.data.end_state_index], 1):
        load = float(target)
        folder = root / f'state_{index:04d}'
        if c.output.resume and (folder / 'result.json').exists():
            report = json.loads((folder / 'result.json').read_text())
            if (report['status'] != 'ready' or not report['outer_convergence']['converged']
                    or report['parent_sha256'] != parent_sha or report['load'] != load
                    or report['state_sha256'] != sha(folder / 'fields.npz')):
                raise ValueError('cannot resume rejected/mismatched local control state')
            with np.load(folder / 'fields.npz', allow_pickle=False) as fields:
                q_parent, d_parent, H_parent = [jnp.asarray(fields[k]) for k in ('q', 'd', 'H')]
            parent_sha = report['state_sha256']
            curve.append(report)
            continue
        folder.mkdir()
        start = time.perf_counter()
        d = d_parent
        q, reaction, mechanical = solve_q(d, load, q_parent)
        H = jnp.maximum(H_parent, jnp.asarray(mechanics.driver(np.asarray(q))))
        consecutive, rows = 0, []
        for sweep in range(1, c.outer_stop.max_sweeps + 1):
            q_old, d_old, H_old, reaction_old = q, d, H, reaction
            H_used = H
            image = solve_d(H_used, d)
            d = d + c.stagger.omega * (image - d)
            q, reaction, mechanical = solve_q(d, load, q)
            H = jnp.maximum(H_parent, jnp.asarray(mechanics.driver(np.asarray(q))))
            H_scale = max(float(jnp.max(H)), float(jnp.max(H_old)), c.outer_stop.scale_floor)
            ratio = np.linalg.norm(np.asarray(q).reshape(-1, 2)[:mechanics.n_material], axis=1) / load
            report = dict(index=index, load=load, sweep=sweep, reaction=reaction,
                damage_min=float(jnp.min(d)), damage_max=float(jnp.max(d)),
                fixed_H_image_damage_min=float(jnp.min(image)),
                fixed_H_image_damage_max=float(jnp.max(image)),
                fixed_H_image_AT2_relative=at2_metrics(image, H_used, matvec, gc0, gc1),
                coupled_AT2_relative=at2_metrics(d, H, matvec, gc0, gc1),
                damage_change=float(jnp.max(abs(d - d_old))),
                q_change_over_load=float(jnp.max(abs(q - q_old))) / load,
                H_change_relative=float(jnp.max(abs(H - H_old))) / H_scale,
                reaction_change_relative=abs(reaction - reaction_old) / max(
                    abs(reaction), abs(reaction_old), c.outer_stop.scale_floor),
                healing=float(jnp.max(jnp.maximum(d_parent - d, 0))),
                history_decrease=float(jnp.max(jnp.maximum(H_parent - H, 0))),
                drift_count=int(np.count_nonzero(ratio > direct.DRIFT_ENVELOPE)),
                max_displacement_ratio=float(np.max(ratio)), mechanics=mechanical,
                diagnostic_only=True, parent_sha256=parent_sha)
            outer = evaluate(report, c.outer_stop, c.reporting, sweep=sweep,
                             previous_consecutive=consecutive)
            consecutive = outer['consecutive']
            report['outer_convergence'] = outer
            rows.append(report)
            atomic_json(folder / f'sweep_{sweep:04d}.json', report)
            if sweep % c.comparison.log_every == 0 or outer['converged']:
                print(json.dumps({k: report[k] for k in ('index', 'load', 'sweep', 'reaction',
                    'damage_max', 'coupled_AT2_relative', 'damage_change', 'drift_count')}), flush=True)
            if outer['converged']:
                break
        report.update(status='ready' if outer['converged'] else 'stagger_budget_exhausted',
                      seconds=time.perf_counter() - start, sweeps=len(rows))
        np.savez_compressed(folder / 'fields.npz', q=np.asarray(q), d=np.asarray(d), H=np.asarray(H),
            H_used=np.asarray(H_used), damage_image=np.asarray(image),
            d_parent=np.asarray(d_parent), H_parent=np.asarray(H_parent))
        report['state_sha256'] = sha(folder / 'fields.npz')
        force = mechanics.matrix(np.asarray(d)) @ np.asarray(q)
        report['host_force_norm'] = float(np.linalg.norm(force[mechanics.free]))
        report['host_reaction'] = float(force @ mechanics.load_direction)
        report['host_force_pass'] = report['host_force_norm'] <= mechanical['convergence_limit'] * (1 + 1e-3)
        if not report['host_force_pass']:
            raise RuntimeError('independent host FEM force replay failed')
        atomic_json(folder / 'result.json', report)
        curve.append(report)
        atomic_json(root / 'result.json', dict(complete=outer['converged'] and index == c.data.end_state_index,
            status=report['status'], ready_loads=sum(r['status'] == 'ready' for r in curve),
            seconds=time.perf_counter() - begin, curve=curve))
        if not outer['converged']:
            break
        q_parent, d_parent, H_parent, parent_sha = q, d, H, report['state_sha256']
    return curve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--last-index', type=int)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    c = runpy.run_path(args.config)['get_config']()
    if args.last_index is not None:
        c.data.end_state_index = args.last_index
    c.output.resume = args.resume
    jax.config.update('jax_enable_x64', True)
    jax.config.update('jax_default_matmul_precision', 'highest')
    run(c)


if __name__ == '__main__':
    main()
