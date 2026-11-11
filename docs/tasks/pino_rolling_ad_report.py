"""Recompute current paired-control metrics and plot explicitly labeled histories."""
import argparse
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def weighted_relative(prediction, reference, weights):
    weights = np.broadcast_to(weights, reference.shape)
    denominator = np.sum(weights*reference**2, dtype=np.float64)
    if denominator <= 0:
        raise ValueError('Reported region must have positive reference norm')
    return float(np.sqrt(np.sum(weights*(prediction-reference)**2, dtype=np.float64)/denominator))


def regional(prediction, reference, times, spatial_weights, boundary_mask):
    """Positive-time diagnostic masks from reference/geometry, never training."""
    p, r = np.asarray(prediction, np.float64), np.asarray(reference, np.float64)
    if p.shape != r.shape or p.shape[:2] != (len(times), len(spatial_weights)):
        raise ValueError('Regional fields and quadrature do not align')
    if not np.isfinite(p).all() or not np.isfinite(r).all():
        raise FloatingPointError('Nonfinite diagnostic fields')
    positive = np.asarray(times) > 0
    p, r = p[positive], r[positive]
    weights = np.broadcast_to(np.asarray(spatial_weights)[None, :], p.shape[:2])
    masks = {'whole': np.ones(p.shape[:2], bool),
             'boundary_strip': np.broadcast_to(boundary_mask, p.shape[:2]),
             'reference_interface': (r[..., 0] > .1) & (r[..., 0] < .9)}
    result = {}
    for name, mask in masks.items():
        count = int(mask.sum())
        if count == 0:
            result[name] = dict(count=0, phi=None, c=None)
            continue
        w = weights*mask
        result[name] = dict(count=count, phi=weighted_relative(p[..., 0], r[..., 0], w),
                            c=weighted_relative(p[..., 1], r[..., 1], w))
    result['measure'] = 'Equal saved positive-time weights; physical spatial weights; phi in(.1,.9) reference band'
    return result


def particle_metrics(run):
    from ml_collections import ConfigDict
    from scipy.spatial import cKDTree
    from examples.particle_corrosion2d.compare import saved_predictions
    from examples.particle_corrosion2d.evaluate import load_references
    c = ConfigDict(read(run/'config.json'))
    references, manifest = load_references(read(run/'runtime.json')['reference'], c)
    predictions, metrics = saved_predictions(run, references)
    saved = read(run/'summary.json')['final']
    for key, value in metrics.items():
        if key != 'conditions':
            np.testing.assert_allclose(value, saved[key], rtol=1e-10, atol=1e-12)
    extras = []
    for p, r in zip(predictions, references, strict=True):
        boundary = r['boundary_tokens']
        walls = boundary[boundary[:, -1] != 2, :2]
        distance = cKDTree(walls).query(r['points'])[0]
        extras.append(regional(p, r['fields'], r['times'], r['volumes'], distance <= c.nl.delta))
    return dict(reference=read(run/'runtime.json')['reference'], cohort=len(references),
                metrics=metrics, regional=extras, boundary_strip_width=c.nl.delta,
                boundary_strip_definition='Distance to sampled wall tokens, excluding reservoir opening')


def electro_metrics(run):
    from docs.tasks.electro_spatial_evaluate import LOCAL, metrics as compute
    evaluated = read(run/'spatial_evaluation64.json')
    if evaluated['status'] != 'passed' or evaluated['count'] != 64:
        raise ValueError('Missing full64common-reference evaluation')
    with np.load(run/'spatial_evaluation64.npz', allow_pickle=False) as data:
        predictions, times, xx = data['prediction'], data['time'], data['X']
    regular = np.asarray(evaluated['regular_time_indices'])
    c = read(run/'config.json')
    width = c['nl']['delta']
    wall = (xx <= width) | (xx >= float(xx.max())-width)
    rows = []
    for i, p in enumerate(predictions):
        with np.load(LOCAL/f'condition_{i:03d}/local101.npz', allow_pickle=False) as data:
            reference = data['fields']
            np.testing.assert_array_equal(times, data['time'])
        actual = compute(p[regular], reference[regular])
        for name in ('pooled_phi', 'terminal_phi', 'pooled_c', 'terminal_c'):
            np.testing.assert_allclose(actual[name], evaluated['records'][i]['local101_64'][name],
                                       rtol=1e-12, atol=1e-14)
        rows.append(regional(p.reshape(len(times), -1, 2), reference.reshape(len(times), -1, 2),
                             times, np.ones(xx.size), wall.ravel()))
    return dict(reference=str(LOCAL), cohort=64, metrics=evaluated['groups'], regional=rows,
                boundary_strip_width=width, boundary_strip_definition='Left/right strips on common FD101 query grid')


def histories(run, case):
    rows = [json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines() if line.strip()]
    if case == 'particle_corrosion2d':
        return [(r['step'], r['seconds'], r['phi_pooled']) for r in rows]
    values = []
    for r in rows:
        # Explicit held-out subset of the small native monitoring bank, NOT64local test cases.
        chosen = [v for v in r['conditions'] if v['split'] == 'test']
        if not chosen:
            raise ValueError('Electropolish monitor has no declared test subset')
        values.append((r['step'], r['seconds'], float(np.mean([v['pooled_rel_l2_phi'] for v in chosen]))))
    return values


def report(root):
    journal = read(root/'journal.json')
    expected = {(case, seed, method, steps) for case, steps in
                (('particle_corrosion2d', 20000), ('electropolish2d', 10000))
                for seed in (42, 43) for method in ('nonlocal', 'local_ad')}
    actual = {(r['case'], r['seed'], r['method'], r['steps']) for r in journal['runs']}
    if len(journal['runs']) != 8 or actual != expected:
        raise ValueError('Final report requires all eight declared cold runs')
    for case, seed, _, _ in expected:
        stages = [s for s in journal['stages'] if s['label'] == f'{case}-s{seed}-pair']
        if len(stages) != 1 or stages[0]['status'] != 'passed':
            raise ValueError('A declared pair has not passed its actual-input/state audit')
    rows = []
    for record in journal['runs']:
        run, case = Path(record['run']), record['case']
        result = particle_metrics(run) if case == 'particle_corrosion2d' else electro_metrics(run)
        if case == 'electropolish2d':
            midpoint = read(run/'spatial_evaluation64_step_005000.json')
            if midpoint['status'] != 'passed' or midpoint['iterations'] != 5000 or midpoint['count'] != 64:
                raise ValueError('Missing common-reference midpoint evaluation')
            result['midpoint_5000'] = midpoint['groups']
        rows.append(dict(**record, evaluation=result))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'nonlocal': '#246d97', 'local_ad': '#d16b3c'}
    cases = ('particle_corrosion2d', 'electropolish2d')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for row_index, case in enumerate(cases):
        for row in rows:
            if row['case'] != case:
                continue
            h = np.asarray(histories(Path(row['run']), case))
            for column, x in enumerate((h[:, 0], h[:, 1]/60)):
                axes[row_index, column].plot(x, 100*h[:, 2],
                    color=colors[row['method']], linestyle='-' if row['seed'] == 42 else '--',
                    label=f'{row["method"]}, seed{row["seed"]}')
        for column in range(2):
            ax = axes[row_index, column]
            ax.set_yscale('log'); ax.grid(alpha=.2); ax.legend(frameon=False)
            ax.set_xlabel('Optimizer updates' if column == 0 else 'Trainer elapsed minutes (incl. evaluation)')
            ax.set_ylabel('Phase relative L2 (%)')
            ax.set_title('Particle: same4local-FEM monitors' if row_index == 0
                         else 'Electropolish: same4native monitors (not64local evaluation)')
    fig.savefig(root/'convergence.png', dpi=160)
    plt.close(fig)
    result = dict(status='passed', device=journal['device'], results=rows,
                  qualification='Two paired seeds; reused development cohorts; graph-building harness overhead remains in AD process time')
    (root/'results.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print('REPORT_COMPLETE', root, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    report(parser.parse_args().root.resolve(strict=True))
