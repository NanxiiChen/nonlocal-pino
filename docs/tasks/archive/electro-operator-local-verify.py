"""Independent NumPy field/target/aggregate and direct-path provenance checks."""
import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path('/root/autodl-tmp/tf-logs/electropolish2d')
RUNS = [ROOT/f'20261029-electro-operator-match-{kind}-01' for kind in ('nonlocal', 'local')]
GRAPH = Path('/root/autodl-tmp/data/electropolish2d/operator_test64_20261028')
LOCAL = Path('/root/autodl-tmp/data/electropolish2d/operator_local_test64_20261029')
OUT = Path(__file__).resolve().parent


def reject(value):
    raise ValueError(f'Nonfinite JSON value: {value}')


def read(path):
    return json.loads(path.read_text(), parse_constant=reject)


def rel(a, b):
    return float(np.sqrt(np.sum((a-b)**2)/np.sum(b**2)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('base', 'all'))
    stage = parser.parse_args().stage
    manifest = read(LOCAL/'manifest.json')
    records = []
    for run in RUNS:
        config = read(run/'config.json'); result = read(run/'evaluation64.json')
        assert read(run/'summary.json')['iterations'] == 5000
        assert not read(run/'summary.json')['training_uses_positive_time_labels']
        assert config['arch']['local_input'] == 'zero'
        assert config['train']['compute_dtype'] == 'float32' and config['train']['gpu_policy'] == 'ordinary'
        if stage == 'all':
            assert result['status'] == 'complete'
        times = np.asarray(result['times']); regular = np.asarray(result['regular_time_indices'])
        early = (times > 0) & (times <= .05)
        with np.load(run/'evaluation64.npz') as stored:
            prediction, normal = stored['prediction'], stored['normal']
            xx, yy = stored['X'], stored['Y']
            assert prediction.shape == (64, 25, 51, 101, 2)
            assert np.isfinite(prediction).all() and np.isfinite(normal).all()
            assert prediction[..., 0].min() >= 0. and prediction[..., 0].max() <= 1.
        side = ((xx <= .02+1e-12) | (xx >= .98-1e-12)) & (yy > 0.) & (yy < .5)
        for i in range(64):
            with np.load(GRAPH/f'condition_{i:03d}/reference.npz') as stored:
                graph = np.stack([stored['phi'], stored['c']], -1)
            with np.load(LOCAL/f'condition_{i:03d}/local101.npz') as stored:
                local = stored['fields']
            p = prediction[i]
            for group, a, target in (('graph64', p[regular], graph), ('local101_64', p[regular], local[regular])):
                for ch, label in enumerate(('phi', 'c')):
                    np.testing.assert_allclose(rel(a[..., ch], target[..., ch]), result['records'][i][group]['pooled_'+label], rtol=2e-6)
                    np.testing.assert_allclose(rel(a[-1, ..., ch], target[-1, ..., ch]), result['records'][i][group]['terminal_'+label], rtol=2e-6)
            np.testing.assert_allclose(rel(p[early, ..., 0][:, side], local[early, ..., 0][:, side]),
                result['records'][i]['local101_64']['early_side_phi'], rtol=2e-6)
            for ch, label in enumerate(('phi', 'q_over_cle')):
                actual = float(np.sqrt(np.mean(normal[i, early, :, ch]**2)))
                np.testing.assert_allclose(actual, result['records'][i]['early_normal_'+label+'_rms'], rtol=2e-6)
        if stage == 'all':
            for i in manifest['refinement_indices']:
                targets = {}
                for nx in (101, 201, 401):
                    with np.load(LOCAL/f'condition_{i:03d}/local{nx}.npz') as stored:
                        stride = (nx-1)//100
                        targets[nx] = stored['fields'][:, ::stride, ::stride, :]
                        np.testing.assert_array_equal(stored['time'], times)
                with np.load(LOCAL/f'condition_{i:03d}/native101.npz') as stored:
                    native = stored['fields']
                spatial = next(r for r in result['reference_spatial_errors'] if r['index'] == i)
                comparisons = [
                    (prediction[i], targets[201], result['records'][i]['local201_4']),
                    (prediction[i], targets[401], result['records'][i]['local401_4']),
                    (prediction[i], native, result['records'][i]['native_early4']),
                    (targets[101], targets[201], spatial['local101_to_201']),
                    (targets[201], targets[401], spatial['local201_to_401']),
                    (native, targets[401], spatial['native_to_local401'])]
                for a, b, measured in comparisons:
                    assert a.shape == b.shape == (25, 51, 101, 2)
                    for ch, label in enumerate(('phi', 'c')):
                        np.testing.assert_allclose(rel(a[regular, ..., ch], b[regular, ..., ch]),
                            measured['pooled_'+label], rtol=2e-6)
                        np.testing.assert_allclose(rel(a[-1, ..., ch], b[-1, ..., ch]),
                            measured['terminal_'+label], rtol=2e-6)
                        np.testing.assert_allclose(rel(a[early, ..., ch][:, side], b[early, ..., ch][:, side]),
                            measured['early_side_'+label], rtol=2e-6)
        for key, group in result['groups'].items():
            selected = [r[key] for r in result['records'] if key in r]
            assert group['count'] == len(selected)
            for metric, summary in group['statistics'].items():
                values = np.asarray([r[metric] for r in selected])
                np.testing.assert_allclose([values.mean(), np.median(values), np.quantile(values, .9), values.max()],
                    [summary[n] for n in ('mean', 'median', 'p90', 'maximum')], rtol=1e-12)
        for name in read(run/'source/files.json'):
            assert (run/'source'/name).read_bytes() == Path(name).read_bytes(), name
        records.append(dict(run=str(run), phase_min=float(prediction[..., 0].min()),
            phase_max=float(prediction[..., 0].max()), concentration_min=float(prediction[..., 1].min()),
            concentration_max=float(prediction[..., 1].max()), all64field_metrics_independently_replayed=True,
            monitor_replay_max_gap=result['monitor_replay_max_gap']))
    with np.load(RUNS[0]/'evaluation64.npz') as a, np.load(RUNS[1]/'evaluation64.npz') as b:
        for key in ('images', 'initial', 'time', 'X', 'Y'):
            np.testing.assert_array_equal(a[key], b[key])
    reference_records = []
    for path in sorted(LOCAL.glob('condition_*/*.npz')):
        with np.load(path) as stored:
            f, coarse = stored['fields'], stored['coarse']
            metadata = json.loads(str(stored['metadata_json']), parse_constant=reject)
            a = f.reshape(len(times), -1, 2); b = coarse.reshape(a.shape)
            error = np.linalg.norm(a-b, axis=1)/np.linalg.norm(a, axis=1)
            np.testing.assert_allclose(error.max(), metadata['maximum_time_refinement_error'], rtol=1e-10, atol=1e-15)
            assert error.max() < 1e-4 and np.isfinite(f).all()
            assert f[..., 0].min() >= -1e-5 and f[..., 0].max() <= 1.00001
            np.testing.assert_allclose(f[:, 0, :, :], 1., rtol=0, atol=1e-12)
            np.testing.assert_allclose(f[:, -1, :, :], 0., rtol=0, atol=1e-12)
            reference_records.append(dict(path=str(path), maximum_time_error=float(error.max()),
                concentration_min=float(f[..., 1].min()), concentration_max=float(f[..., 1].max())))
    if stage == 'all':
        assert len(reference_records) == 76
        assert not read(LOCAL/'summary.json')['failures']
    result = dict(stage=stage, runs=records, actual_eval_input_images_initial_arrays_equal=True,
                  refined_field_metrics_independently_replayed=(stage == 'all'),
                  references=reference_records, all_checks_pass=True)
    (OUT/f'electro-operator-local-verification-{stage}.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print('PASS', stage, 'all64fields/aggregates/normals/saved inputs/current source and', len(reference_records), 'reference time-refinement/BC/range checks', flush=True)
