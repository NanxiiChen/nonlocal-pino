"""Recover cold elapsed axes from original files; freeze evidence for reproduction.

The launch log is created immediately after the parent's timer starts and before
Popen. Its birth time is a launch proxy, not a recorded monotonic timestamp.
The metric file's final write calibrates the trainer clock. Its initial creation
independently checks that calibration. Report this reconstruction explicitly.
"""
import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
import subprocess

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def birth(path):
    value = subprocess.check_output(['stat', '--format=%w', str(path)], text=True).strip()
    if value == '-':
        raise ValueError('Original file birth timestamp is required: ' + str(path))
    return dict(text=value, epoch_seconds=datetime.fromisoformat(value).timestamp())


def recover(seconds, evaluation_seconds, *, launch_epoch, metric_created_epoch,
            metric_written_epoch, last_counter, cold_total, tolerance_seconds):
    seconds = np.asarray(seconds, float)
    evaluation_seconds = np.asarray(evaluation_seconds, float)
    scalars = [launch_epoch, metric_created_epoch, metric_written_epoch,
               last_counter, cold_total, tolerance_seconds]
    if (seconds.ndim != 1 or len(seconds) < 2
            or evaluation_seconds.shape != seconds.shape
            or not np.isfinite(seconds).all() or not np.isfinite(evaluation_seconds).all()
            or not np.isfinite(scalars).all() or np.any(np.diff(seconds) <= 0)
            or np.any(seconds < 0) or np.any(evaluation_seconds < 0)
            or tolerance_seconds <= 0 or cold_total <= 0 or last_counter < seconds[-1]):
        raise ValueError('Require ordered finite observed counters and explicit clock evidence')
    offset = metric_written_epoch - launch_epoch - last_counter
    drift = metric_created_epoch - launch_epoch - (seconds[0] + offset)
    observed = seconds + offset + evaluation_seconds
    if (offset < 0 or abs(drift) > tolerance_seconds
            or metric_written_epoch < metric_created_epoch
            or np.any(np.diff(observed) <= 0) or observed[-1] > cold_total):
        raise ValueError('File timestamps disagree with the cold-process/metric clock')
    plotted = observed.copy()
    # At process exit the final fixed-budget model is available. This does not
    # insert a new evaluation, extrapolate errors, or add extra observations.
    plotted[-1] = cold_total
    return dict(elapsed_seconds=plotted.tolist(),
                observed_evaluation_seconds=observed.tolist(),
                trainer_origin_offset_seconds=float(offset),
                first_write_calibration_difference_seconds=float(drift),
                final_availability_shift_seconds=float(cold_total-observed[-1]))


def capture(comparison, costs):
    comparison = Path(comparison).resolve(strict=True)
    report = read(comparison)
    data = report['data']
    if report['status'] != 'passed' or data['comparison_kind'] != 'time_fusion':
        raise ValueError('Require a completed time-fusion comparison')
    result = dict(example=data['example'], comparison=str(comparison), steps=data['steps'], arms=[])
    for arm in data['arms']:
        cost = costs['temporal_controls'][data['example']][arm['method']]
        stage = [r for r in read(cost['journal'])['stages'] if r['name'] == cost['stage']]
        if len(stage) != 1 or stage[0]['exit_code'] != 0 or stage[0]['seconds'] != cost['seconds']:
            raise ValueError('Parent stage does not match the selected full cold timing')
        if stage[0]['seconds'] != arm['timing']['cold_process_seconds']:
            raise ValueError('Selected run and comparison timing disagree')
        if '--config.log.run_name=' + Path(arm['run']).name not in stage[0]['command']:
            raise ValueError('Clock stage belongs to another run')
        log = Path(cost['journal']).parent / (cost['stage'] + '.log')
        metric = Path(arm['run']) / 'metrics.jsonl'
        all_rows = rows(metric)
        if data['example'] == 'electropolish2d':
            selected = all_rows
            local = rows(arm['monitor'])
            if [r['step'] for r in local] != data['steps']:
                raise ValueError('Local metrics and timed metrics do not align')
            durations = [r['evaluation_seconds'] for r in local]
        elif data['example'] == 'icemelting3d':
            selected = [r for r in all_rows if r['split'] == 'monitor']
            durations = [0.] * len(selected)  # Counters already include evaluation.
        else:
            raise ValueError('No timing convention declared for this example')
        if [r['step'] for r in selected] != data['steps']:
            raise ValueError('Require every actual timed evaluation')
        created, metric_created = birth(log), birth(metric)
        evidence = dict(launch_epoch=created['epoch_seconds'],
                        metric_created_epoch=metric_created['epoch_seconds'],
                        metric_written_epoch=metric.stat().st_mtime,
                        last_counter=all_rows[-1]['seconds'], cold_total=cost['seconds'],
                        tolerance_seconds=.1)
        counters = [r['seconds'] for r in selected]
        recovered = recover(counters, durations, **evidence)
        result['arms'].append(dict(method=arm['method'], run=arm['run'],
            parent_journal=cost['journal'], parent_stage=cost['stage'],
            launch_log=str(log), metric_file=str(metric), monitor=arm['monitor'],
            launch_log_birth=created, metric_birth=metric_created,
            clock_evidence=evidence, trainer_counters=counters,
            following_local_evaluation_seconds=durations, **recovered))
    return result


def attach(data, evidence_path):
    evidence_path = Path(evidence_path).resolve(strict=True)
    evidence = read(evidence_path)
    if evidence['status'] != 'passed' or evidence['schema'] != 'pino_time_fusion_clock_v1':
        raise ValueError('Require qualified saved cold-clock evidence')
    matches = [case for case in evidence['cases'] if case['example'] == data['example']]
    if len(matches) != 1 or matches[0]['steps'] != data['steps']:
        raise ValueError('Clock evidence does not match the comparison')
    case = matches[0]
    if [a['method'] for a in case['arms']] != [a['method'] for a in data['arms']]:
        raise ValueError('Clock methods differ')
    output = copy.deepcopy(data)
    for arm, clock in zip(output['arms'], case['arms'], strict=True):
        if arm['run'] != clock['run'] or arm['monitor'] != clock['monitor']:
            raise ValueError('Clock evidence belongs to another training run')
        actual = rows(clock['metric_file'])
        if data['example'] == 'icemelting3d':
            actual = [r for r in actual if r['split'] == 'monitor']
        if ([r['step'] for r in actual] != data['steps']
                or [r['seconds'] for r in actual] != clock['trainer_counters']):
            raise ValueError('Saved trainer observations changed')
        recalculated = recover(clock['trainer_counters'], clock['following_local_evaluation_seconds'],
                               **clock['clock_evidence'])
        np.testing.assert_array_equal(recalculated['elapsed_seconds'], clock['elapsed_seconds'])
        arm['elapsed_seconds'] = recalculated['elapsed_seconds']
    output['convergence_axis'] = dict(kind='cold_wall_time', unit='seconds',
        label='Elapsed wall time (s)', evidence=str(evidence_path),
        origin='Original launch-log creation before Popen; filesystem-calibrated trainer counters',
        endpoint='Final existing evaluation placed at measured complete process exit',
        limitation='Reconstructed observation times, not direct per-evaluation parent timestamps; '
                   'subsecond precision is not claimed. No step-rate extrapolation.')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, action='append', required=True)
    parser.add_argument('--costs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    costs = read(args.costs)
    if costs['status'] != 'passed':
        raise ValueError('Require qualified parent-process timings')
    result = dict(schema='pino_time_fusion_clock_v1', status='passed',
        costs=str(args.costs.resolve()), cases=[capture(p, costs) for p in args.comparison])
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
