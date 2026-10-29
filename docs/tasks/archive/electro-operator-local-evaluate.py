"""Frozen-checkpoint64test, shared native/local targets and cold-start parity."""
import argparse
import json
from pathlib import Path
import time

import equinox as eqx
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
import numpy as np

from nonlocal_pinn.training.runtime import configure_gpu_policy
from examples.corrosion2d.residual_experiment import grid, initial_fields, boundary_values
from examples.electropolish2d.operator_data import condition_config, load_dataset
from examples.electropolish2d.operator_training import _build_network, predict_fields
from examples.electropolish2d.operator_local import normal_derivatives


BASE = Path('/root/autodl-tmp/data/electropolish2d/operator_test64_20261028')
LOCAL = Path('/root/autodl-tmp/data/electropolish2d/operator_local_test64_20261029')
RUNS = Path('/root/autodl-tmp/tf-logs/electropolish2d')
NL = RUNS/'20261029-electro-operator-match-nonlocal-01'
AD = RUNS/'20261029-electro-operator-match-local-01'
OUT = Path(__file__).resolve().parent


def relative(a, b, mask):
    if mask is not None:
        a, b = a[..., mask], b[..., mask]
    denominator = np.linalg.norm(b)
    if denominator <= 0:
        raise ValueError('Relative error target must have nonzero norm')
    return float(np.linalg.norm(a-b)/denominator)


def metrics(predicted, target, regular, early, side, interior):
    result = {}
    for ch, label in enumerate(('phi', 'c')):
        p, r = predicted[..., ch], target[..., ch]
        result['pooled_'+label] = relative(p[regular], r[regular], None)
        result['terminal_'+label] = relative(p[-1], r[-1], None)
        result['initial_'+label] = relative(p[0], r[0], None)
        result['early_side_'+label] = relative(p[early], r[early], side)
        result['early_interior_'+label] = relative(p[early], r[early], interior)
        result['terminal_side_'+label] = relative(p[-1], r[-1], side)
        result[label+'_by_time'] = [relative(a, b, None) for a, b in zip(p, r)]
        result['side_'+label+'_by_time'] = [relative(a, b, side) for a, b in zip(p, r)]
    band = ((target[early, ..., 0] > .1) & (target[early, ..., 0] < .9)
            & side[None])
    result['early_side_interface_phi'] = relative(predicted[early, ..., 0][band], target[early, ..., 0][band], None)
    return result


def aggregate(records, key):
    chosen = [r[key] for r in records if key in r]
    output = dict(count=len(chosen), statistics={}, mean_curves={})
    for name in chosen[0]:
        array = np.asarray([r[name] for r in chosen])
        if array.ndim == 1:
            output['statistics'][name] = dict(mean=float(array.mean()), median=float(np.median(array)),
                p90=float(np.quantile(array, .9)), maximum=float(array.max()), std=float(array.std(ddof=1)))
        else:
            output['mean_curves'][name] = array.mean(0).tolist()
    return output


def evaluate(run):
    c = ConfigDict(json.loads((run/'config.json').read_text()))
    configure_gpu_policy(c.train.gpu_policy)
    jax.config.update('jax_enable_x64', False)
    jax.config.update('jax_default_matmul_precision', c.train.matmul_precision)
    assert jax.default_backend() == 'gpu' and c.arch.local_input == 'zero'
    manifest = json.loads((BASE/'manifest.json').read_text())
    local_manifest = json.loads((LOCAL/'manifest.json').read_text())
    times = np.asarray(local_manifest['times'])
    regular = np.searchsorted(times, np.linspace(0., 1., 21))
    np.testing.assert_array_equal(times[regular], np.linspace(0., 1., 21))
    early = (times > 0.) & (times <= .05)
    xx, yy, points = grid(c)
    side = ((xx <= c.nl.delta+1e-12) | (xx >= 1-c.nl.delta-1e-12)) & (yy > 0.) & (yy < .5)
    interior = (xx > .1) & (xx < .9) & (yy > 0.) & (yy < .5)
    x = jnp.asarray(points, dtype=jnp.float32)
    ts = jnp.asarray(times, dtype=jnp.float32)
    template = _build_network(c.arch, c.operator.image_shape, jax.random.key(0))
    net = eqx.tree_deserialise_leaves(run/'model_final.eqx', template)
    predict = eqx.filter_jit(lambda model, image, known: predict_fields(model, image, x, ts, known, c))
    _, height, width = c.operator.image_shape
    sx, sy = np.meshgrid(np.linspace(0., 1., width), np.linspace(0., .5, height), indexing='xy')
    sensors = np.column_stack([sx.ravel(), sy.ravel()])
    z, o, values = boundary_values(c, points)
    wall_ids = np.flatnonzero(((np.isclose(points[:, 0], 0.)) | np.isclose(points[:, 0], 1.)) & ~(z | o))
    walls = x[wall_ids]
    normals = jnp.zeros_like(walls).at[:, 0].set(jnp.where(walls[:, 0] == 0., -1., 1.))
    normal = eqx.filter_jit(lambda model, image: normal_derivatives(model, model.encode(image), walls, normals, ts, c))
    start = time.perf_counter()
    records, predictions, images, initials, normal_fields = [], [], [], [], []
    spatial_errors = []
    for i, identity in enumerate(manifest['surface_ids']):
        cc = condition_config(c, identity, '')
        initial = initial_fields(cc, points); initial[z | o] = values[z | o]
        image_fields = initial_fields(cc, sensors)
        sz, so, sv = boundary_values(c, sensors); image_fields[sz | so] = sv[sz | so]
        image = image_fields[:, 0].reshape(1, height, width).astype(np.float32)
        p = np.asarray(predict(net, jnp.asarray(image), jnp.asarray(initial, dtype=jnp.float32)))
        p = p.reshape(len(times), c.sample.Ny, c.sample.Nx, 2)
        n = np.asarray(normal(net, jnp.asarray(image)))
        assert np.isfinite(p).all() and np.isfinite(n).all()
        with np.load(BASE/f'condition_{i:03d}/reference.npz') as f:
            graph = np.stack([f['phi'], f['c']], -1)
        np.testing.assert_allclose(initial, graph[0].reshape(-1, 2), rtol=0, atol=3e-7)
        with np.load(LOCAL/f'condition_{i:03d}/local101.npz') as f:
            local = f['fields']
        np.testing.assert_allclose(local[0], graph[0], rtol=0, atol=3e-7)
        record = dict(index=i, surface_id=identity,
            graph64=metrics(p[regular], graph, np.arange(21), np.arange(21)==1, side, interior),
            local101_64=metrics(p, local, regular, early, side, interior),
            early_normal_phi_rms=float(np.sqrt(np.mean(n[early, :, 0]**2))),
            early_normal_q_over_cle_rms=float(np.sqrt(np.mean(n[early, :, 1]**2))),
            positive_normal_phi_rms=float(np.sqrt(np.mean(n[times > 0, :, 0]**2))),
            positive_normal_q_over_cle_rms=float(np.sqrt(np.mean(n[times > 0, :, 1]**2))))
        records.append(record); predictions.append(p); images.append(image); initials.append(initial); normal_fields.append(n)
        print('EVAL', run.name, i, record['graph64']['pooled_phi'], record['local101_64']['pooled_phi'], flush=True)
    # Replay stored final monitors through the same25time predictor, then select
    # original21times. This also detects saved-config/checkpoint mismatches.
    data, meta, _ = load_dataset(c)
    replay_gaps = []
    with np.load(run/'prediction.npz') as saved:
        for k, index in enumerate(meta['eval_indices']):
            replay = np.asarray(predict(net, jnp.asarray(data['images'][index]), jnp.asarray(data['initial'][index]))).reshape(len(times), *xx.shape, 2)[regular]
            np.testing.assert_allclose(replay, saved['prediction'][k], rtol=0, atol=2e-5)
            replay_gaps.append(float(np.max(abs(replay-saved['prediction'][k]))))
    summary = dict(run=str(run), checkpoint=str(run/'model_final.eqx'),
        times=times.tolist(), regular_time_indices=regular.tolist(), early_times=times[early].tolist(),
        side_strip_width=c.nl.delta, side_strip_excludes_Dirichlet_corners=True,
        groups={key: aggregate(records, key) for key in ('graph64', 'local101_64')},
        records=records, reference_spatial_errors=spatial_errors, monitor_replay_max_gap=max(replay_gaps),
        status='base64_complete_refinement_pending',
        evaluation_seconds=time.perf_counter()-start,
        scope='Final fixed-budget model; no checkpoint selection;64common graph/localFD101targets. '
              'Local spatial refinement201/401on four predeclared interfaces; do not call64continuum-converged. '
              'Early strip metrics exclude t0; graph finite-horizon closure differs from classical pointwise Neumann.')
    path = run/'evaluation64.json'
    assert not path.exists(), path
    path.write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    np.savez_compressed(run/'evaluation64.npz', prediction=np.stack(predictions), images=np.stack(images),
                        initial=np.stack(initials), normal=np.stack(normal_fields), time=times, X=xx, Y=yy)
    (OUT/f'electro-operator-{c.nl.method}-evaluation64.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print('COMPLETE', run, summary['evaluation_seconds'], flush=True)


def finalize():
    """Add predeclared refined targets to saved fields; no second GPU inference."""
    manifest = json.loads((LOCAL/'manifest.json').read_text())
    for run in (NL, AD):
        c = ConfigDict(json.loads((run/'config.json').read_text()))
        result = json.loads((run/'evaluation64.json').read_text())
        assert result['status'] == 'base64_complete_refinement_pending'
        with np.load(run/'evaluation64.npz') as stored:
            predictions, xx, yy = stored['prediction'], stored['X'], stored['Y']
        times = np.asarray(result['times']); regular = np.asarray(result['regular_time_indices'])
        early = (times > 0.) & (times <= .05)
        side = ((xx <= c.nl.delta+1e-12) | (xx >= 1-c.nl.delta-1e-12)) & (yy > 0.) & (yy < .5)
        interior = (xx > .1) & (xx < .9) & (yy > 0.) & (yy < .5)
        errors = []
        for index in manifest['refinement_indices']:
            p = predictions[index]
            levels = {}
            for nx in (101, 201, 401):
                with np.load(LOCAL/f'condition_{index:03d}/local{nx}.npz') as stored:
                    np.testing.assert_array_equal(stored['time'], times)
                    levels[nx] = stored['fields'][:, ::(nx-1)//100, ::(nx-1)//100]
                if nx > 101:
                    result['records'][index][f'local{nx}_4'] = metrics(p, levels[nx], regular, early, side, interior)
            with np.load(LOCAL/f'condition_{index:03d}/native101.npz') as stored:
                native = stored['fields']
            with np.load(BASE/f'condition_{index:03d}/reference.npz') as stored:
                graph = np.stack([stored['phi'], stored['c']], -1)
            np.testing.assert_allclose(native[regular], graph, rtol=0, atol=2e-7)
            result['records'][index]['native_early4'] = metrics(p, native, regular, early, side, interior)
            errors.append(dict(index=index,
                local101_to_201=metrics(levels[101], levels[201], regular, early, side, interior),
                local201_to_401=metrics(levels[201], levels[401], regular, early, side, interior),
                native_to_local401=metrics(native, levels[401], regular, early, side, interior)))
        for key in ('local201_4', 'local401_4', 'native_early4'):
            result['groups'][key] = aggregate(result['records'], key)
        result['reference_spatial_errors'] = errors
        result['status'] = 'complete'
        text = json.dumps(result, indent=2, allow_nan=False)+'\n'
        (run/'evaluation64.json').write_text(text)
        (OUT/f'electro-operator-{c.nl.method}-evaluation64.json').write_text(text)
        print('FINALIZED', run, flush=True)


def audit():
    configs = [ConfigDict(json.loads((run/'config.json').read_text())) for run in (NL, AD)]
    jax.config.update('jax_enable_x64', False)
    initial = [eqx.tree_deserialise_leaves(run/'model_initial.eqx', _build_network(c.arch, c.operator.image_shape, jax.random.key(0))) for run, c in zip((NL, AD), configs)]
    count = 0
    for a, b in zip(jax.tree.leaves(initial[0]), jax.tree.leaves(initial[1])):
        if eqx.is_array(a):
            np.testing.assert_array_equal(a, b); count += 1
    for name in ('online_coefficients.npy', 'online_accepted.npy'):
        np.testing.assert_array_equal(np.load(NL/name), np.load(AD/name))
    with np.load(NL/'sampling.npz') as a, np.load(AD/'sampling.npz') as b:
        for name in a.files:
            np.testing.assert_array_equal(a[name], b[name])
    assert np.load(NL/'online_coefficients.npy').shape == (5000, 8, 6, 3)
    def diff(a, b, key=''):
        if isinstance(a, dict) and isinstance(b, dict):
            return sum((diff(a[k], b[k], key+'.'+k) if k in a and k in b else [key+'.'+k]
                        for k in sorted(a.keys() | b.keys())), [])
        return [] if a == b else [key.lstrip('.')]
    differences = [name.lstrip('.') for name in diff(configs[0].to_dict(), configs[1].to_dict())]
    assert set(differences) == {'nl.method', 'loss.gradnorm_bias', 'loss.neumann_scale', 'log.run_name'}, differences
    assert configs[0].arch == configs[1].arch
    for path in (NL/'source/files.json', AD/'source/files.json'):
        for name in json.loads(path.read_text()):
            assert (NL/'source'/name).read_bytes() == (AD/'source'/name).read_bytes(), name
    result = dict(initial_arrays_equal=count, actual_profiles_equal=40000,
                  times_core_IC_BC_equal=True, config_differences=differences,
                  common_source_snapshots_equal=True,
                  scope='One matched initialization/online-stream pair; six local losses versus five native losses, no seed-stability claim')
    (OUT/'electro-operator-local-parity.json').write_text(json.dumps(result, indent=2)+'\n')
    print('PASS PARITY', json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('nonlocal', 'local', 'audit', 'finalize'))
    action = parser.parse_args().action
    if action == 'audit':
        audit()
    elif action == 'finalize':
        finalize()
    else:
        evaluate(NL if action == 'nonlocal' else AD)
