"""Audit final Spinodal fields against both references with identical norms.

CPU-only postprocessing of saved arrays. This does not load a neural model,
solve a reference, or modify the live training/figure queues.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def errors(prediction, reference, volumes):
    """Full space-time channel norms; equal-node primary, volume secondary."""
    p = np.asarray(prediction, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    v = np.asarray(volumes, dtype=np.float64)
    if p.shape != r.shape or p.ndim != 3 or p.shape[-1] != 2:
        raise ValueError('Expected matching [time, point, two channels] fields')
    if p.shape[0] < 2 or v.shape != (p.shape[1],):
        raise ValueError('Missing positive times or mismatched spatial volumes')
    if not (np.isfinite(p).all() and np.isfinite(r).all()
            and np.isfinite(v).all() and np.all(v > 0)):
        raise ValueError('Nonfinite field or nonpositive physical volume')
    results = {}
    for name, weight in (('equal_node', np.ones_like(v)), ('physical_volume', v)):
        denominator = np.sum(r * r * weight[None, :, None], axis=(0, 1))
        later_denominator = np.sum(r[1:, :, 1] ** 2 * weight[None, :])
        if np.any(denominator <= 0) or later_denominator <= 0:
            raise ValueError('Zero reference norm')
        numerator = np.sum((p - r) ** 2 * weight[None, :, None], axis=(0, 1))
        relative = np.sqrt(numerator / denominator)
        later = np.sqrt(np.sum((p[1:, :, 1] - r[1:, :, 1]) ** 2
                               * weight[None, :]) / later_denominator)
        results[name] = dict(phi=float(relative[0]), chemical=float(relative[1]),
                             channel_mean=float(relative.mean()),
                             chemical_positive_times=float(later))
    return results


def evaluate_run(run, native_bank, local_bank, expected_steps):
    run, native_bank, local_bank = map(Path, (run, native_bank, local_bank))
    config = read(run / 'config.json')
    native = read(native_bank / 'manifest.json')
    local = read(local_bank / 'manifest.json')
    if config['train']['iterations'] != expected_steps:
        raise ValueError('Run budget differs from the requested final budget')
    if Path(config['operator']['dataset_path']) != native_bank:
        raise ValueError('Run uses a different native bank')
    if (local['status'] != 'passed' or local['indices'] != native['test_indices']
            or local['specification'] != native['specification']
            or [c['index'] for c in local['cases']] != local['indices']):
        raise ValueError('References differ in specification, cohort or ordering')
    monitor = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
    if not monitor or monitor[-1]['step'] != expected_steps:
        raise ValueError('Final training monitor is not complete')
    checkpoint = run / 'model_final.eqx'
    if not checkpoint.is_file():
        raise ValueError('Final checkpoint is absent')
    rows = []
    for case in local['cases']:
        if case['status'] != 'passed':
            raise ValueError('Unqualified local case')
        index = case['index']
        prediction_path = run / f'prediction_{index:03d}.npz'
        native_path = native_bank / native['conditions'][index]['file']
        local_path = local_bank / case['file']
        with np.load(prediction_path) as pred, np.load(native_path) as nr, np.load(local_path) as lr:
            for key in ('points', 'time', 'volumes', 'holes'):
                np.testing.assert_array_equal(pred[key], nr[key])
                np.testing.assert_array_equal(pred[key], lr[key])
            for key in ('mesh_points', 'mesh_cells'):
                np.testing.assert_array_equal(pred[key], nr[key])
            for field in ('phi', 'mu'):
                np.testing.assert_array_equal(pred[field + '_reference'], nr[field])
            times = pred['time']
            if (times.ndim != 1 or times[0] != 0 or not np.isfinite(times).all()
                    or not np.all(np.diff(times) > 0)):
                raise ValueError('Expected ordered zero and positive evaluation times')
            p = np.stack((pred['phi'], pred['mu']), axis=-1)
            n = np.stack((nr['phi'], nr['mu']), axis=-1)
            l = np.stack((lr['phi'], lr['mu']), axis=-1)
            if p.shape[:2] != (len(times), len(pred['points'])):
                raise ValueError('Field and evaluation-coordinate shapes differ')
            rows.append(dict(index=index, prediction=str(prediction_path),
                             native_reference=str(native_path), local_reference=str(local_path),
                             native=errors(p, n, pred['volumes']),
                             local=errors(p, l, pred['volumes']),
                             native_reference_vs_local=errors(n, l, pred['volumes'])))
    groups = ('native', 'local', 'native_reference_vs_local')
    measures = ('equal_node', 'physical_volume')
    fields = ('phi', 'chemical', 'channel_mean', 'chemical_positive_times')
    means = {g: {m: {f: float(np.mean([r[g][m][f] for r in rows]))
                      for f in fields} for m in measures} for g in groups}
    return dict(run=str(run), checkpoint=str(checkpoint), steps=expected_steps,
                indices=local['indices'], cases=rows, means=means)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--final-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    final = read(args.final_report)
    methods = ['nonlocal', 'fem']
    if (final['status'] != 'passed' or final['methods'] != methods
            or set(final['results']) != set(methods)):
        raise ValueError('Both completed and audited Spinodal methods are required')
    results = {}
    for method in methods:
        source = final['results'][method]
        run = Path(source['run'])
        config = read(run / 'config.json')
        result = evaluate_run(run, config['operator']['dataset_path'],
                              config['operator']['comparison_reference']['path'], 50000)
        if Path(result['checkpoint']) != Path(source['checkpoint']):
            raise ValueError('Final audit names a different checkpoint')
        if result['indices'] != [row['index'] for row in source['cases']]:
            raise ValueError('Final audit has a different ordered cohort')
        for measured, audited in zip(result['cases'], source['cases'], strict=True):
            for field, value in measured['local']['equal_node'].items():
                np.testing.assert_allclose(value, audited[field], rtol=2e-13, atol=2e-14)
        for field, value in result['means']['local']['equal_node'].items():
            np.testing.assert_allclose(value, source['means'][field], rtol=2e-13, atol=2e-14)
        results[method] = result
    if results['nonlocal']['indices'] != results['fem']['indices']:
        raise ValueError('Method cohorts differ')
    report = dict(status='passed', final_report=str(args.final_report), results=results,
                  primary_metric='Equal evaluation-node full space-time norm per channel/sample, then arithmetic means',
                  secondary_metric='Physical cell-area weighted spatial norm with equal saved-time weights',
                  chemical_scope='Full-time primary; positive-time chemical diagnostic separately excludes differing initial conventions',
                  limitations='Eight reused geometries and one initialization; reference errors cannot be subtracted to infer causes')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps({m: r['means'] for m, r in results.items()}, indent=2))


if __name__ == '__main__':
    main()
