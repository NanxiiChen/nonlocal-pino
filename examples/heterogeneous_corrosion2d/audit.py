"""Audit a saved endpoint on every graph node without reference solution fields."""
import argparse
import json
from pathlib import Path
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict

from .losses import residuals
from .network import build_network, physical_fields, prepare
from .problem import boundary_mask, initial_fields, material_field, material_image, parameter_count


def audit(run, inputs, output):
    run, inputs, output = Path(run), Path(inputs), Path(output)
    if output.exists():
        raise FileExistsError(f'Audit output already exists: {output}')
    c = ConfigDict(json.loads((run/'config.json').read_text()))
    runtime = json.loads((run/'runtime.json').read_text())
    jax.config.update('jax_enable_x64', False)
    jax.config.update('jax_default_matmul_precision', c.train.matmul_precision)
    with np.load(run/'graph.npz', allow_pickle=False) as saved:
        points = saved['points']
        row, col, weights = saved['graph_row'], saved['graph_col'], saved['graph_weight']
        visited = np.unique(saved['core_ids'][saved['core_mask']])
    # Access only known inputs. This file need not contain any evolved solution.
    with np.load(inputs, allow_pickle=False) as source:
        parameters, times = source['parameters'], source['time']
        if not np.array_equal(points, source['points']):
            raise ValueError('Audit input coordinates differ from saved training graph')
        source_config = json.loads(str(source['config_json'].item()))
    for section in ('geometry', 'material', 'physical'):
        if source_config[section] != json.loads(c.to_json())[section]:
            raise ValueError(f'Audit input {section} differs from saved training configuration')
    if (parameters.ndim != 2 or parameters.shape[1] != parameter_count(c)
            or not np.isfinite(parameters).all() or times.ndim != 1 or len(times) < 2
            or not np.isfinite(times).all() or times[0] != 0
            or times[-1] != c.sample.t_max or np.any(np.diff(times) <= 0)):
        raise ValueError('Audit needs finite material coefficients and increasing times from0to t_max')
    net = build_network(c, jax.random.split(jax.random.PRNGKey(runtime['seed']), 2)[0])
    net = eqx.tree_deserialise_leaves(run/'model_final.eqx', net)
    n_array_scalars = sum(a.size for a in jax.tree.leaves(net) if eqx.is_array(a))
    n_frozen = int(net.spatial_frequencies.size)
    queries, ts = jnp.asarray(points, dtype=jnp.float32), jnp.asarray(times, dtype=jnp.float32)
    sub = (jnp.asarray(row), jnp.asarray(col), -jnp.asarray(weights), None)

    @eqx.filter_jit
    def predict_and_residual(model, image, kappa):
        cached = prepare(model, image, queries)
        raw, raw_t = jax.jvp(lambda t: model.decode(cached, t/c.sample.t_max),
                            (ts,), (jnp.ones_like(ts),))
        fields, fields_t = jax.jvp(lambda z: physical_fields(z, c), (raw,), (raw_t,))
        ac, ch = residuals(raw, fields_t, kappa, sub, c)
        return fields, jnp.stack((ac.T, ch.T), axis=-1)

    images = material_image(c, parameters, np).astype(np.float32)
    kappas = material_field(c, parameters, points, np).astype(np.float32)
    predictions, residual_fields = [], []
    started = time.perf_counter()
    for image, kappa in zip(images, kappas):
        fields, rs = predict_and_residual(net, jnp.asarray(image), jnp.asarray(kappa))
        predictions.append(np.asarray(fields))
        residual_fields.append(np.asarray(rs))
    fields, rs = np.stack(predictions), np.stack(residual_fields)
    if not np.isfinite(fields).all() or not np.isfinite(rs).all():
        raise FloatingPointError('Nonfinite full-grid audit output')
    mouth = boundary_mask(c, points)
    boundary = (np.isclose(points[:, 0], 0) | np.isclose(points[:, 0], c.geometry.length)
                | np.isclose(points[:, 1], 0) | np.isclose(points[:, 1], c.geometry.height))
    seen = np.zeros(len(points), dtype=bool)
    seen[visited] = True
    masks = dict(all_free=~mouth, seen_core=seen & ~mouth, unseen_core=~seen & ~mouth,
                 natural_boundary=boundary & ~mouth, interior=~boundary,
                 unseen_natural_boundary=~seen & boundary & ~mouth)

    def statistics(values):
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            return None
        return dict(mean_absolute=float(np.mean(np.abs(values))),
                    rms=float(np.sqrt(np.mean(values**2))), maximum_absolute=float(np.max(np.abs(values))))

    report = dict(run=str(run.resolve()), inputs=str(inputs.resolve()),
                  checkpoint=str((run/'model_final.eqx').resolve()),
                  parameter_count=int(n_array_scalars-n_frozen),
                  network_array_scalar_count=int(n_array_scalars),
                  frozen_coordinate_scalar_count=n_frozen,
                  conditions=len(parameters), times=times.tolist(), reference_fields_used=False,
                  aggregation='All materials and selected times/nodes; unweighted original physical residuals',
                  pde_positive_times={}, pde_initial_time={}, node_counts={})
    for name, mask in masks.items():
        report['node_counts'][name] = int(mask.sum())
        report['pde_positive_times'][name] = {
            eq: statistics(rs[:, 1:, mask, i]) for i, eq in enumerate(('ac', 'ch'))}
        report['pde_initial_time'][name] = {
            eq: statistics(rs[:, :1, mask, i]) for i, eq in enumerate(('ac', 'ch'))}
    initial = initial_fields(c, points)
    report['initial_phi_error'] = statistics(fields[:, 0, :, 0]-initial[:, 0])
    report['initial_c_over_cle_error'] = statistics(
        (fields[:, 0, :, 1]-initial[:, 1])/c.physical.c_le)
    report['mouth_phi_error'] = statistics(fields[:, :, mouth, 0])
    report['mouth_c_over_cle_error'] = statistics(fields[:, :, mouth, 1]/c.physical.c_le)
    report['seconds_including_compile'] = time.perf_counter()-started
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/'full_grid_residuals.npz', residual=rs,
                        time=times, points=points, parameters=parameters, **masks)
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--inputs', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    audit(args.run, args.inputs, args.output)
