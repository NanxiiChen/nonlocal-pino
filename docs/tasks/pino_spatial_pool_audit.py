"""Check matched draws and report completed fixed/rolling pairs on the same GPU."""
import argparse
import json
from pathlib import Path
import time

import numpy as np


def read(path):
    return json.loads(path.read_text())


def audit(case, stages):
    fixed, rolling = (Path(s['path']) for s in stages)
    a, b = (read(p/'config.json') for p in (fixed, rolling))
    for config, mode in ((a, 'fixed_pool'), (b, 'rolling_pool')):
        if config['sample']['spatial'].pop('mode') != mode:
            raise ValueError(f'Spatial mode does not match stage {mode}')
        config['log'].pop('run_name')
    if a != b:
        raise ValueError('Matched controls differ in a non-spatial configuration field')
    for name in ('model_initial.eqx', 'optimizer_initial.eqx'):
        if (fixed/name).read_bytes() != (rolling/name).read_bytes():
            raise ValueError(f'Matched cold state differs for {name}')
    n, pool = a['train']['iterations'], a['sample']['K_pool']
    electro = case == 'electropolish2d'
    with np.load(fixed/'sampling.npz') as fa, np.load(rolling/'sampling.npz') as fb:
        for key in (('times', 'ic_ids', 'dirichlet_ids', 'side_ids') if electro else ('times', 'parameters')):
            np.testing.assert_array_equal(fa[key], fb[key])
        np.testing.assert_array_equal(fa['draw_ids'], np.arange(n) % pool)
        np.testing.assert_array_equal(fb['draw_ids'], np.arange(n))
        for key in (('core_ids', 'core_mask', 'nonlocal_halo_counts') if electro else ('core_ids', 'actual_nodes')):
            np.testing.assert_array_equal(fa[key], fb[key][:pool])
    if electro:
        for name in ('online_coefficients.npy', 'online_accepted.npy'):
            np.testing.assert_array_equal(np.load(fixed/name), np.load(rolling/name))

    results = {}
    for path, stage, mode in zip((fixed, rolling), stages, ('fixed_pool', 'rolling_pool'), strict=True):
        summary = read(path/'summary.json')
        timing = read(path/'timing.json')
        update, sample, spatial = [np.asarray(timing[key]) for key in
                                   ('update_seconds', 'sample_seconds', 'spatial_seconds')]
        if not all(x.shape == (n,) and np.isfinite(x).all() for x in (update, sample, spatial)):
            raise ValueError('Expected one finite timing record per update')
        memory = read(path/'memory.json')
        coverage = read(path/'spatial_sampling.json')
        if electro:
            evaluated = read(path/'spatial_evaluation64.json')
            if evaluated['status'] != 'passed':
                raise ValueError('Saved Electropolish fields did not pass evaluator checks')
            errors = {name: group['statistics'] for name, group in evaluated['groups'].items()}
            peak = max(record['allocator_stats']['peak_bytes_in_use'] for record in memory)/1024**3
        else:
            errors = summary['test']
            peak = summary['peak_active_GiB']
        results[mode] = dict(relative_errors=errors, active_peak_GiB=peak,
                             first_update_seconds=float(update[0]), cold_wall_seconds=stage['seconds'],
                             warm_update_median_ms=float(np.median(update[100:])*1000),
                             warm_effective_median_ms=float(np.median((update+sample+spatial)[100:])*1000),
                             warm_effective_mean_ms=float(np.mean((update+sample+spatial)[100:])*1000),
                             spatial_sampling=coverage)
    return dict(case=case, seed=a['train']['seed'], steps=n, matched_config_cold_state_and_draws=True, results=results)


def summarize(root):
    journal = read(root/'journal.json')
    passed = {s['label']: s for s in journal['stages'] if s['status'] == 'passed'}
    rows = []
    for label, stage in passed.items():
        if not label.endswith('-fixed_pool'):
            continue
        prefix = label.removesuffix('-fixed_pool')
        case = prefix.rsplit('-s', 1)[0]
        rolling = prefix+'-rolling_pool'
        if rolling not in passed:
            continue
        if case == 'electropolish2d' and any(x+'-evaluate64' not in passed for x in (label, rolling)):
            continue
        rows.append(audit(case, (stage, passed[rolling])))
    destination = root/'paired-results.json'
    temp = destination.with_suffix('.partial')
    temp.write_text(json.dumps(dict(queue_status=journal['status'], pairs=rows), indent=2, allow_nan=False)+'\n')
    temp.replace(destination)
    return len(rows), journal['status']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_root', type=Path)
    parser.add_argument('--follow', action='store_true', help='Update the report while the queue is active')
    args = parser.parse_args()
    while True:
        count, status = summarize(args.run_root)
        print(f'Queue {status}. Completed and validated {count} pairs.', flush=True)
        if not args.follow or status != 'running':
            break
        time.sleep(60)
