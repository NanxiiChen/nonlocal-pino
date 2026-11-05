"""Held-out material evaluation, isolated from every physics training update."""
import argparse
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict
from scipy.spatial import cKDTree

from examples.heterogeneous_corrosion2d.network import build_network, predict_fields
from examples.heterogeneous_corrosion2d.problem import grid, initial_fields, material_image, parameter_count


def load_reference(path, c, points, graph=None):
    """Strict explicit-path validation; evolved arrays are evaluation-only."""
    path = Path(path).resolve(strict=True)
    with np.load(path, allow_pickle=False) as source:
        data = {name: source[name] for name in source.files}
    required = {'fields', 'points', 'time', 'parameters', 'config_json', 'backend'}
    if not required.issubset(data):
        raise ValueError(f'Reference missing fields: {sorted(required-set(data))}')
    native_backend = 'native_nonlocal_fp32_weights_fp64_integration'
    allowed_backends = (native_backend, 'independent_local_fv')
    backend = data['backend']
    if (backend.shape != () or not isinstance(backend.item(), str)
            or backend.item() not in allowed_backends):
        raise ValueError(f'Reference backend must be a scalar string in {allowed_backends}')
    if graph is not None and backend.item() != native_backend:
        raise ValueError(f'Native graph matching requires backend {native_backend}; got {backend.item()}')
    time, fields, parameters = data['time'], data['fields'], data['parameters']
    if (time.ndim != 1 or len(time) < 2 or time[0] != 0 or time[-1] != c.sample.t_max
            or np.any(np.diff(time) <= 0) or not np.isfinite(time).all()
            or parameters.ndim != 2 or parameters.shape[1] != parameter_count(c)
            or not np.isfinite(parameters).all()
            or fields.shape != (len(parameters), len(time), len(points), 2)
            or not np.isfinite(fields).all()
            or data['points'].shape != points.shape
            or not np.allclose(data['points'], points, atol=1e-7, rtol=0)):
        raise ValueError('Reference time, material parameters, coordinates or field shape mismatch')
    ref_config = json.loads(str(data['config_json'].item()))
    actual = json.loads(c.to_json())
    for section in ('physical', 'geometry', 'material'):
        if ref_config[section] != actual[section]:
            raise ValueError(f'Reference {section} differs from training problem')
    for name in ('Nx', 'Ny', 't_max'):
        if ref_config['sample'][name] != actual['sample'][name]:
            raise ValueError(f'Reference sample.{name} differs from training problem')
    if not np.allclose(fields[:, 0], initial_fields(c, points)[None], atol=1e-8, rtol=0):
        raise ValueError('Reference initial state differs from the prescribed initial condition')
    if graph is not None:
        for name, expected in zip(('graph_row', 'graph_col', 'graph_weight'), graph):
            if name not in data or not np.array_equal(data[name], expected):
                raise ValueError(f'Native reference does not share the exact training {name}')
    data['path'] = str(path)
    return data


def _relative(pred, target):
    denominator = float(np.linalg.norm(target))
    if denominator == 0:
        return None
    return float(np.linalg.norm(pred-target)/denominator)


def _interface(c, values):
    import contourpy
    xx, yy, _ = grid(c)
    contours = contourpy.contour_generator(x=xx[0], y=yy[:, 0],
        z=values.reshape(c.sample.Ny, c.sample.Nx)).lines(.5)
    return np.concatenate(contours) if contours else np.empty((0, 2))


def field_metrics(pred, target, initial, c):
    result = {}
    for i, name in enumerate(('phi', 'c')):
        result['pooled_rel_l2_'+name] = _relative(pred[..., i], target[..., i])
        result['terminal_rel_l2_'+name] = _relative(pred[-1, ..., i], target[-1, ..., i])
        result['rel_l2_by_time_'+name] = [_relative(a, b) for a, b in zip(pred[..., i], target[..., i])]
    result['corroded_rel_l2'] = _relative(1-pred[..., 0], 1-target[..., 0])
    result['terminal_corroded_rel_l2'] = _relative(1-pred[-1, ..., 0], 1-target[-1, ..., 0])
    result['dissolution_increment_rel_l2'] = _relative(initial[None, :, 0]-pred[1:, ..., 0],
                                                      initial[None, :, 0]-target[1:, ..., 0])
    dc, cle = c.physical.c_se-c.physical.c_le, c.physical.c_le
    def q(fields):
        phi = fields[..., 0]
        return (fields[..., 1]-dc*phi**2*(3-2*phi)-cle)/cle
    result['q_over_cle_rmse'] = float(np.sqrt(np.mean((q(pred)-q(target))**2)))
    result['terminal_q_over_cle_rmse'] = float(np.sqrt(np.mean((q(pred[-1])-q(target[-1]))**2)))
    result['predicted_q_over_cle_range'] = [float(q(pred).min()), float(q(pred).max())]
    a, b = _interface(c, pred[-1, :, 0]), _interface(c, target[-1, :, 0])
    result['terminal_interface_present'] = bool(len(a) and len(b))
    if len(a) and len(b):
        distances = np.r_[cKDTree(a).query(b)[0], cKDTree(b).query(a)[0]]
        result['terminal_interface_mean_distance'] = float(distances.mean())
        result['terminal_interface_p95_distance'] = float(np.quantile(distances, .95))
        result['terminal_pit_depth_prediction'] = float(a[:, 1].max())
        result['terminal_pit_depth_reference'] = float(b[:, 1].max())
    return result


def make_evaluator(c, reference):
    points = jnp.asarray(reference['points'], dtype=jnp.float32)
    times = jnp.asarray(reference['time'], dtype=jnp.float32)
    images = np.asarray(material_image(c, reference['parameters'], np), dtype=np.float32)
    initial = initial_fields(c, reference['points'])
    predict = eqx.filter_jit(lambda net, image: predict_fields(net, image, points, times, c))

    def evaluate(net):
        predictions, records = [], []
        for i, image in enumerate(images):
            pred = np.asarray(predict(net, jnp.asarray(image)))
            if not np.isfinite(pred).all():
                raise FloatingPointError('Nonfinite held-out predictions')
            predictions.append(pred)
            records.append(dict(condition=i, **field_metrics(pred, reference['fields'][i], initial, c)))
        summary = {'conditions': records}
        for name in ('pooled_rel_l2_phi', 'terminal_rel_l2_phi', 'pooled_rel_l2_c',
                     'terminal_rel_l2_c', 'corroded_rel_l2', 'terminal_corroded_rel_l2',
                     'dissolution_increment_rel_l2', 'q_over_cle_rmse'):
            values = [record[name] for record in records]
            summary[name] = None if any(value is None for value in values) else float(np.mean(values))
        return np.stack(predictions), summary

    return evaluate


def render_reports(run, c, reference, prediction, history):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    xx, yy, _ = grid(c)
    count = min(len(prediction), 4)
    time_ids = np.unique(np.linspace(0, len(reference['time'])-1, 4).astype(int))
    from examples.heterogeneous_corrosion2d.problem import material_field
    materials = material_field(c, reference['parameters'], reference['points'], np)
    fig, axes = plt.subplots(count, len(time_ids)+1, figsize=(3*(len(time_ids)+1), 2.4*count),
                             squeeze=False, constrained_layout=True)
    for i in range(count):
        axes[i, 0].pcolormesh(xx, yy, materials[i].reshape(xx.shape), vmin=c.material.minimum,
                             vmax=c.material.maximum, cmap='viridis', shading='gouraud')
        axes[i, 0].set_title(f'Material {i+1}: kinetics')
        for j, ti in enumerate(time_ids, start=1):
            phi = prediction[i, ti, :, 0].reshape(xx.shape)
            target = reference['fields'][i, ti, :, 0].reshape(xx.shape)
            shown = axes[i, j].pcolormesh(xx, yy, phi, vmin=0, vmax=1, cmap='coolwarm', shading='gouraud')
            axes[i, j].contour(xx, yy, target, levels=[.5], colors='black', linewidths=1)
            axes[i, j].contour(xx, yy, phi, levels=[.5], colors='#ffd65a', linewidths=1)
            axes[i, j].set_title(f't* = {reference["time"][ti]:.2f}')
        for ax in axes[i]:
            ax.set_aspect('equal')
            ax.invert_yaxis()
    fig.colorbar(shown, ax=axes[:, 1:].ravel().tolist(), shrink=.7, label='Metal phase fraction')
    fig.suptitle('Held-out materials: predicted phase; black = reference interface, yellow = PINO')
    fig.savefig(Path(run)/'fields.png', dpi=170)
    plt.close(fig)
    if history:
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        for name, label in (('pooled_rel_l2_phi', 'Phase'), ('pooled_rel_l2_c', 'Concentration'),
                            ('corroded_rel_l2', 'Corroded fraction')):
            ax.semilogy([row['step'] for row in history], [100*row[name] for row in history], label=label)
        ax.set(xlabel='Training updates', ylabel='Held-out relative L2 (%)')
        ax.legend()
        ax.grid(alpha=.25)
        fig.savefig(Path(run)/'convergence.png', dpi=170)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    c = ConfigDict(json.loads((args.run/'config.json').read_text()))
    runtime = json.loads((args.run/'runtime.json').read_text())
    _, _, points = grid(c)
    reference = load_reference(args.reference, c, points)
    model = build_network(c, jax.random.split(jax.random.PRNGKey(runtime['seed']), 2)[0])
    model = eqx.tree_deserialise_leaves(args.run/'model_final.eqx', model)
    prediction, metrics = make_evaluator(c, reference)(model)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False)+'\n')
    np.savez_compressed(args.output/'prediction.npz', prediction=prediction, target=reference['fields'],
                        parameters=reference['parameters'], time=reference['time'], points=points)
    render_reports(args.output, c, reference, prediction, [])
    print(json.dumps(metrics, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
