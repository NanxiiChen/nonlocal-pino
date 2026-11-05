"""Prepare held-out references and discretization checks, never training labels.

Run with JAX_PLATFORMS=cpu. The local FV matrix is assembled independently of
the neural spatial operator. The native reference integrates the actual FP32
pair weights with FP64 arithmetic to isolate neural approximation error.
"""
import argparse
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import time

import numpy as np
from scipy import sparse

from .problem import (grid, initial_fields, local_matrix, material_field,
                      material_image, quadrature, sample_parameters)
from .solver import solve


def field_comparison(prediction, reference, c):
    """Common physical-node errors; do not hide minority-phase differences."""
    result = {}
    for i, name in enumerate(('phi', 'c')):
        for stage, a, b in (('pooled', prediction[..., i], reference[..., i]),
                            ('terminal', prediction[-1, :, i], reference[-1, :, i])):
            result[f'{stage}_{name}_relative_l2'] = float(np.linalg.norm(a-b)/np.linalg.norm(b))
    result['terminal_minority_relative_l2'] = float(
        np.linalg.norm(prediction[-1, :, 0]-reference[-1, :, 0])
        / np.linalg.norm(1-reference[-1, :, 0]))
    def q(a):
        phi = a[..., 0]
        return a[..., 1]-(c.physical.c_se-c.physical.c_le)*(3*phi**2-2*phi**3)-c.physical.c_le
    result['q_over_cle_rmse'] = float(np.sqrt(np.mean((q(prediction)-q(reference))**2))/c.physical.c_le)
    return result


def shape_statistics(fields, points, weights):
    initial_area = float(weights @ (1-fields[0, :, 0]))
    terminal_area = float(weights @ (1-fields[-1, :, 0]))
    dissolved = fields[-1, :, 0] < .5
    if not np.any(dissolved):
        raise ValueError('Reference lost its initial pit contour')
    return dict(initial_diffuse_area=initial_area, terminal_diffuse_area=terminal_area,
                diffuse_area_advance=terminal_area-initial_area,
                terminal_depth=float(np.max(points[dissolved, 1])))


def render_reference(path, c, points, fields, local_fields, parameters, times):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    xx, yy = points[:, 0].reshape(c.sample.Ny, c.sample.Nx), points[:, 1].reshape(c.sample.Ny, c.sample.Nx)
    indices = np.linspace(0, len(times)-1, 5).astype(int)
    fig, axes = plt.subplots(len(fields), 6, figsize=(15, 2.2*len(fields)), squeeze=False,
                             constrained_layout=True)
    for b in range(len(fields)):
        kappa = material_field(c, parameters[b], points, np).reshape(xx.shape)
        material = axes[b, 0].pcolormesh(xx, yy, kappa, shading='gouraud',
                                       cmap='viridis', vmin=c.material.minimum, vmax=c.material.maximum)
        axes[b, 0].set_title(f'Material {b+1}: kinetics')
        for j, index in enumerate(indices, 1):
            ax = axes[b, j]
            phase = ax.pcolormesh(xx, yy, fields[b, index, :, 0].reshape(xx.shape),
                                 shading='gouraud', cmap='coolwarm', vmin=0, vmax=1)
            ax.contour(xx, yy, local_fields[b, index, :, 0].reshape(xx.shape),
                       levels=[.5], colors='black', linewidths=.7, linestyles='--')
            ax.set_title(f't* = {times[index]:.2f}')
        for ax in axes[b]:
            ax.set_aspect('equal')
            ax.set_xlim(0, c.geometry.length)
            ax.set_ylim(c.geometry.height, 0)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.colorbar(material, ax=axes[:, 0], shrink=.6, label='kappa')
    fig.colorbar(phase, ax=axes[:, 1:].ravel().tolist(), shrink=.6, label='phi: electrolyte 0, metal 1')
    fig.suptitle('Direct nonlocal reference; dashed contour: independent local FV (not PINO)')
    fig.savefig(path, dpi=160)
    plt.close(fig)


def prepare(output, seed, config_module):
    import jax
    from nonlocal_pinn.operators import build_D
    if jax.default_backend() != 'cpu':
        raise ValueError('Reference preparation requires JAX_PLATFORMS=cpu')
    jax.config.update('jax_enable_x64', False)
    c = importlib.import_module(config_module).get_config()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    xx, yy, points = grid(c)
    # Host IC/material inputs use the exact same FP64 coordinates as training;
    # only operator construction is intentionally the production FP32 path.
    op = build_D(points.astype(np.float32), c.nl.delta, method=c.nl.method, kernel=c.nl.kernel)
    row, col = np.asarray(op._row_idx), np.asarray(op._col_idx)
    weight = -np.asarray(op._vals)
    diagonal = np.bincount(row, weights=weight.astype(np.float64), minlength=len(points))
    native = (sparse.diags(diagonal)
              + sparse.coo_matrix((-weight.astype(np.float64), (row, col)),
                                  shape=(len(points), len(points)))).tocsr()
    local = local_matrix(c)
    parameters = sample_parameters(c, np.random.default_rng(seed), c.operator.reference_conditions)
    times = np.linspace(0, c.sample.t_max, c.operator.reference_snapshots)
    np.savez_compressed(output/'heldout_inputs.npz', parameters=parameters, points=points,
                        time=times, config_json=c.to_json(), seed=np.asarray(seed, dtype=np.uint64))
    cfg = c.reference
    native_fields, local_fields, records = [], [], []
    for i, p in enumerate(parameters):
        kappa = material_field(c, p, points, np)
        print(f'Reference condition {i+1}/{len(parameters)}', flush=True)
        nf, ni = solve(c, native, points, kappa, times, cfg.rtol, cfg.atol, cfg.max_step)
        lf, li = solve(c, local, points, kappa, times, cfg.rtol, cfg.atol, cfg.max_step)
        native_fields.append(nf)
        local_fields.append(lf)
        record = dict(condition=i, native=ni, local=li,
                      nonlocal_vs_local=field_comparison(nf, lf, c),
                      local_shape=shape_statistics(lf, points, quadrature(c)))
        records.append(record)
        print(json.dumps(record), flush=True)
    native_fields, local_fields = np.asarray(native_fields), np.asarray(local_fields)
    common = dict(points=points, time=times, parameters=parameters,
                  images=material_image(c, parameters, np),
                  kappa=material_field(c, parameters, points, np), config_json=c.to_json(),
                  graph_row=row, graph_col=col, graph_weight=weight)
    np.savez_compressed(output/'native_reference.npz', **common, fields=native_fields,
                        backend=np.asarray('native_nonlocal_fp32_weights_fp64_integration'))
    np.savez_compressed(output/'local_reference.npz', **common, fields=local_fields,
                        backend=np.asarray('independent_local_fv'))
    fine = c.copy_and_resolve_references()
    fine.sample.Nx, fine.sample.Ny = 2*c.sample.Nx-1, 2*c.sample.Ny-1
    _, _, fine_points = grid(fine)
    print('Refining held-out condition 0 on independent local FV', flush=True)
    ff, fi = solve(fine, local_matrix(fine), fine_points,
                   material_field(fine, parameters[0], fine_points, np), times,
                   cfg.refined_rtol, cfg.refined_atol, cfg.refined_max_step)
    restricted = ff.reshape(len(times), fine.sample.Ny, fine.sample.Nx, 2)[:, ::2, ::2].reshape(len(times), len(points), 2)
    tf, ti = solve(c, local, points, material_field(c, parameters[0], points, np), times,
                   cfg.refined_rtol, cfg.refined_atol, cfg.refined_max_step)
    refinement = dict(fine_solve=fi, fine_grid=[fine.sample.Nx, fine.sample.Ny],
                      local_coarse_vs_fine=field_comparison(local_fields[0], restricted, c),
                      tightened_time_solve=ti,
                      local_time_tolerance=field_comparison(local_fields[0], tf, c))
    np.savez_compressed(output/'local_refinement.npz', points=fine_points, time=times,
                        parameters=parameters[0], fields=ff, restricted_fields=restricted,
                        tightened_time_fields=tf)
    render_reference(output/'reference_evolution.png', c, points, native_fields,
                     local_fields, parameters, times)
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), seed=seed,
                  reference_role='held-out evaluation only; never used by the training objective',
                  configuration=json.loads(c.to_json()), conditions=records,
                  refinement=refinement, seconds=time.perf_counter()-start)
    (output/'qualification.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print('COMPLETE '+str(output), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, required=True, help='Held-out input provenance, not a training setting')
    parser.add_argument('--config', default='examples.heterogeneous_corrosion2d.configs.train_base',
                        help='Module exposing the explicit get_config() factory')
    args = parser.parse_args()
    prepare(args.output, args.seed, args.config)
