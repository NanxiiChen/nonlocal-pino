"""Pair and audit the explicit local AD channel control, without reference solves."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from docs.tasks.channel_fem_support import HISTORICAL, COMMON, LOGS, paired, json_write


def screen(bundle, sampled=False):
    from examples.channel_nsch2d.configs.train_default import get_config as baseline
    from examples.channel_nsch2d.configs.train_operator_local_ad import get_config
    if sampled:
        from examples.channel_nsch2d.configs.train_operator_local_ad_sampled import get_config
    from examples.channel_nsch2d.problem import validate_config, identity
    from examples.channel_nsch2d.data import geometry_data
    from examples.channel_nsch2d.geometry import initial_parameters
    from examples.channel_nsch2d.online import build_input
    from nonlocal_pinn.training.runtime import with_run_seed
    c = with_run_seed(get_config(), 42); validate_config(c)
    baseline_config = json.loads(with_run_seed(baseline(), 42).to_json())
    old = json.loads((HISTORICAL/'config.json').read_text())
    baseline_config['log']['run_name'] = old['log']['run_name']
    if baseline_config != old:
        raise ValueError('Current baseline changed relative to accepted historical configuration')
    new = json.loads(c.to_json())
    differences = {key: dict(baseline=baseline_config[key], local_ad=new[key] if key in new else None)
                   for key in baseline_config if key not in new or baseline_config[key] != new[key]}
    repaired = json.loads(c.to_json())
    for key in ('nl', 'derivatives', 'phase_transport', 'log'):
        repaired[key] = baseline_config[key]
    repaired['train']['microbatch'] = baseline_config['train']['microbatch']
    if sampled:
        del repaired['sample']['ad_sampling']
    if repaired != baseline_config:
        raise ValueError('Unexpected AD changes beyond derivative method, time block and run name')
    for relative in ('examples/channel_nsch2d/network.py', 'examples/channel_nsch2d/geometry.py',
                     'src/nonlocal_pinn/nn/point_conditional.py', 'src/nonlocal_pinn/training/optimizers.py'):
        if Path(relative).read_bytes() != (HISTORICAL/'source'/relative).read_bytes():
            raise ValueError('Network, geometry or SOAP changed: '+relative)
    started = time.perf_counter()
    with np.load(HISTORICAL/'known_inputs.npz') as original:
        for i, a in enumerate(initial_parameters(c)):
            raw, _ = geometry_data(a, c)
            for key in (*COMMON, 'transport_measure'):
                np.testing.assert_array_equal(raw[key], original[key][i], err_msg=key)
    count = ((c.train.iterations-1)//c.operator.online.refresh_every+c.operator.online.prefetch_batches)*c.operator.online.refresh_size
    areas = []
    for draw in range(count):
        data, report = build_input(c.to_dict(), draw)
        areas.append(report['operator']['area'])
        if report['operator']['field_derivative_stencils'] != 0:
            raise ValueError('Local control unexpectedly contains field stencils')
    result = dict(status='passed', historical=str(HISTORICAL), config=c.to_dict(),
        differences=differences, physical_identity=identity(c), initial_cases=c.operator.train_conditions,
        online_cases=count, smallest_area=min(areas), largest_area=max(areas),
        seconds=time.perf_counter()-started, pde_solves=0, field_derivative_stencils=0)
    json_write(bundle/'screen.json', result)
    print(json.dumps({k: result[k] for k in ('status', 'online_cases', 'seconds')}), flush=True)


def qualify(bundle, prefix, sampled=False):
    methods = ('nonlocal', 'ad1024', 'ad2048') if sampled else ('nonlocal', 'ad')
    runs = [LOGS/(prefix+'-probe-'+method) for method in methods]
    pair = paired(runs, 60)
    rows = []
    for method, run in zip(methods, runs, strict=True):
        summary = json.loads((run/'summary.json').read_text())
        if summary['status'] != 'completed' or summary['iterations'] != 60:
            raise ValueError('A complete 60-update production probe is required')
        memory = summary['device_memory_bytes']
        if memory['peak_bytes_in_use'] > .97*memory['bytes_limit']:
            raise ValueError('Insufficient active allocator headroom')
        with np.load(run/'spatial_timing.npz') as timing:
            duration = timing['update'][10:]+timing['batch_preparation'][10:]
            if duration.shape != (50,) or not np.isfinite(duration).all():
                raise ValueError('Incomplete or invalid warm update timing')
        rows.append(dict(method=method, run=str(run),
            inclusive_median_seconds=float(np.median(duration)),
            inclusive_mean_seconds=float(np.mean(duration)),
            peak_active_gib=memory['peak_bytes_in_use']/1024**3,
            projected_20k_update_hours=float(np.median(duration)*20000/3600)))
    result = dict(status='passed', pairing=pair, resources=rows)
    if sampled:
        # Predeclared accuracy-oriented choice; neither the monitor nor test
        # errors select a method after observing the reserved references.
        result['selected_probe'] = str(runs[-1])
        result['selection'] = '2048 interior queries, timeblock9; larger sampled PDE set, subject to resource gate'
    json_write(bundle/'resources.json', result)
    print(json.dumps(result), flush=True)


def finish(bundle, prefix):
    run = LOGS/(prefix+'-ad-cold20k')
    pair = paired([run], 20000)
    record = json.loads((run/'unseen3/evaluation.json').read_text())
    cases = []
    for case in record['per_geometry']:
        with np.load(run/'unseen3'/f'case_{case["index"]:03d}'/'prediction.npz') as prediction, np.load(case['reference']) as ref:
            for key in ('points', 'time', 'amplitudes'):
                np.testing.assert_array_equal(prediction[key], ref[key])
            np.testing.assert_array_equal(prediction['reference'], ref['fields'])
            row = {}
            for name, channels, start in (('phi', slice(0, 1), 0), ('velocity', slice(2, 4), 1),
                                           ('pressure', slice(4, 5), 1), ('chemical', slice(1, 2), 1)):
                actual = np.asarray(prediction['fields'][start:, ..., channels], dtype=np.float64)
                target = np.asarray(ref['fields'][start:, ..., channels], dtype=np.float64)
                row[name] = float(np.linalg.norm(actual-target)/np.linalg.norm(target))
                np.testing.assert_allclose(row[name], case['metrics'][name], atol=1e-12, rtol=0)
            cases.append(row)
    means = {name: float(np.mean([c[name] for c in cases])) for name in cases[0]}
    for name, value in means.items():
        np.testing.assert_allclose(value, record['aggregate'][name]['mean'], atol=1e-12, rtol=0)
    files = list((run/'source').rglob('*.py'))
    for source in files:
        if source.read_bytes() != source.relative_to(run/'source').read_bytes():
            raise ValueError('Saved/current source mismatch: '+str(source))
    result = dict(status='passed', local_ad_run=str(run), pairing=pair,
        resources=json.loads((bundle/'resources.json').read_text()),
        summary=json.loads((run/'summary.json').read_text()), cohort=record,
        independent_means=means, source_files_checked=len(files),
        limitations=['One initialization, three reused reserved geometries.',
                     'Effective condition/time counts match; sampled AD uses an unbiased spatial estimator and its own execution block.',
                     'Historical nonlocal accuracy and present resource probes use different physical GPU UUIDs.'])
    json_write(bundle/'comparison.json', result)
    print(json.dumps(dict(status='passed', means=means)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=('screen', 'qualify', 'finish'), required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--sampled', action='store_true')
    args = parser.parse_args()
    if args.action == 'screen':
        screen(args.bundle, args.sampled)
    elif args.action == 'qualify':
        qualify(args.bundle, args.prefix, args.sampled)
    else:
        finish(args.bundle, args.prefix)
