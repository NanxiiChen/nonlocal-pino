"""Independent draw and endpoint checks for the original-scheme row control."""
import argparse
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def row_replay(run, config, conditions, geometry_ids):
    """Reconstruct every CDF draw from recorded physical inputs, not the pool."""
    settings = config['sample']['residual_rows']
    known = {}
    for file in sorted((run/'residual_row_geometry').glob('*.npz')):
        with np.load(file, allow_pickle=False) as data:
            for i, (slot, version) in enumerate(zip(data['slots'], data['versions'], strict=True)):
                key = int(slot), int(version)
                if key in known:
                    raise ValueError(f'Duplicate physical input version {key}')
                x, v = data['points'][i], data['volumes'][i]
                ids = np.flatnonzero(v > 0)
                ids = ids[np.lexsort((x[ids, 1], x[ids, 0]))]
                mass = v[ids].astype(np.float64)
                cdf = np.cumsum(mass/mass.sum())
                cdf[-1] = 1.
                known[key] = ids, cdf
    rows = [json.loads(line) for line in (run/'residual_row_inputs.jsonl').read_text().splitlines()]
    with np.load(run/'residual_row_draws.npz', allow_pickle=False) as saved:
        cores, weights = saved['core_ids'], saved['core_weights']
    count, steps, capacity = settings['count'], len(conditions), config['operator']['train_conditions']
    if cores.shape != (*conditions.shape, count) or len(rows) != steps:
        raise ValueError('Recorded row batches do not cover the actual training draws')
    np.testing.assert_array_equal(weights, np.full(count, 1/count, dtype=weights.dtype))
    selected = 0
    for step, record in enumerate(rows, 1):
        if record['step'] != step or len(record['samples']) != conditions.shape[1]:
            raise ValueError('Residual row step/order mismatch')
        epoch = (step-1)//settings['refresh_every']
        pools = np.random.default_rng(np.random.SeedSequence(
            [config['train']['seed'], 58114, step])).integers(settings['pools'], size=conditions.shape[1])
        for j, item in enumerate(record['samples']):
            slot, draw = int(conditions[step-1, j]), int(geometry_ids[step-1, j])
            version = 0 if draw < 0 else draw//capacity+1
            if (item['slot'], item['version'], item['epoch'], item['pool']) != (slot, version, epoch, pools[j]):
                raise ValueError(f'Residual pool identity mismatch at step {step}, batch {j}')
            ids, cdf = known[(slot, version)]
            rng = np.random.default_rng(np.random.SeedSequence(
                [config['train']['seed'], 58113, slot, version, epoch, int(pools[j])]))
            u = (np.arange(count)+rng.random(count))/count
            expected = ids[np.searchsorted(cdf, u, side='right')]
            np.testing.assert_array_equal(cores[step-1, j], expected)
            if item['distinct_rows'] != len(np.unique(expected)):
                raise ValueError('Distinct-row record does not match actual indices')
            selected += 1
    return dict(status='passed', selected_sets=selected, scalar_row_indices=int(cores.size),
                physical_input_versions=len(known), decoder_support='Full physical graph')


def paired(full, rows, output, steps):
    a, b = read(full/'config.json'), read(rows/'config.json')
    original_b = read(rows/'config.json')
    for c in (a, b):
        c['log'].pop('run_name')
        if c['operator']['metric'] != 'linear_stable_resolvent' or 'formulation' in c['operator']:
            raise ValueError('This audit requires the original direct-phase/stable-metric recipe')
    b['sample'].pop('residual_rows')
    if a != b:
        raise ValueError('Configurations differ beyond residual-row sampling and run name')
    for name in ('model_initial.eqx', 'optimizer_initial.eqx'):
        if (full/name).read_bytes() != (rows/name).read_bytes():
            raise ValueError(f'Cold state differs: {name}')
    sources = sorted((full/'source').rglob('*.py'))
    if not sources or len(sources) != len(list((rows/'source').rglob('*.py'))):
        raise ValueError('Frozen source file counts differ')
    for file in sources:
        if file.read_bytes() != (rows/'source'/file.relative_to(full/'source')).read_bytes():
            raise ValueError(f'Training sources differ: {file}')
    with np.load(full/'training_draws.npz') as left, np.load(rows/'training_draws.npz') as right:
        if set(left.files) != {'condition_ids', 'geometry_draw_ids', 'times'} or set(left.files) != set(right.files):
            raise ValueError('Expected complete cold online training draws')
        for key in left.files:
            np.testing.assert_array_equal(left[key], right[key])
        replay = row_replay(rows, original_b, right['condition_ids'], right['geometry_draw_ids'])
    for filename, keys in (
            ('online_inputs.jsonl', ('draw_id', 'holes', 'nodes', 'degree', 'area')),
            ('online_refresh.jsonl', ('step', 'batch', 'slots', 'draw_ids'))):
        left = [json.loads(line) for line in (full/filename).read_text().splitlines()]
        right = [json.loads(line) for line in (rows/filename).read_text().splitlines()]
        if len(left) != len(right):
            raise ValueError(f'Geometry stream lengths differ: {filename}')
        for x, y in zip(left, right, strict=True):
            if any(x[key] != y[key] for key in keys):
                raise ValueError(f'Actual geometry identities/metadata differ: {filename}')
    reports = []
    for run in (full, rows):
        summary = read(run/'summary.json')
        if summary['new_updates'] != steps or summary['start_step'] != 0:
            raise ValueError('Cold fixed-budget requirement failed')
        scores = []
        for index in summary['final_metrics']['test_indices']:
            name = f'prediction_{index:03d}.npz'
            with np.load(full/name) as ref, np.load(run/name) as pred:
                for key in ('points', 'volumes', 'time', 'holes', 'phi_reference', 'mu_reference'):
                    np.testing.assert_array_equal(ref[key], pred[key])
                v = pred['volumes'].astype(np.float64)
                diff = pred['phi'].astype(np.float64)-pred['phi_reference']
                score = np.sqrt(np.sum(diff*diff*v)/np.sum(pred['phi_reference']**2*v))
                scores.append(float(score))
        np.testing.assert_allclose(np.mean(scores), summary['final_metrics']['test_phi_pooled'], rtol=2e-12, atol=1e-12)
        with np.load(run/'update_timing.npz') as timing:
            total = timing['update_seconds']+timing['batch_preparation_seconds']
        reports.append(dict(run=str(run), phase_mean=float(np.mean(scores)), phase_by_sample=scores,
            warm_median_seconds=float(np.median(total[100:])), wall_seconds=summary['wall_seconds'],
            peak_bytes=summary['device_memory_stats']['peak_bytes_in_use']))
    output.write_text(json.dumps(dict(status='passed', arms=reports, row_replay=replay,
        identical_cold_states=True, all_geometry_time_draws_equal=True, source_files=len(sources),
        reference='Same eight native reserved geometries, volume-weighted phase space-time norm',
        limitation='Both arms query the full physical graph; not compact core/halo accuracy'), indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full', type=Path, required=True)
    parser.add_argument('--rows', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, required=True)
    args = parser.parse_args()
    paired(args.full, args.rows, args.output, args.steps)
