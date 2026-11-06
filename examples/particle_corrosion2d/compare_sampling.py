"""Compare completed fixed/rolling runs from saved fields only, without JAX.

The optional evaluation pair refers to the same eight reused geometries. This
command never evaluates a network, starts training, or rebuilds input operators.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from ml_collections import ConfigDict

from .compare import (_history, _plot_convergence, run_costs, saved_predictions)
from .evaluate import load_references
from .online_validation import validate_control_result


TABLE_METRICS = ('phi_pooled', 'phi_terminal', 'c_pooled', 'c_terminal')


def _read(path):
    return json.loads(Path(path).read_text())


def _verify_metrics(actual, recorded, label):
    """Check both cohort means and individual physical-field errors."""
    for name, value in actual.items():
        if name == 'conditions':
            if len(value) != len(recorded[name]):
                raise ValueError(f'{label}: saved metric cohort size differs')
            for index, (condition, previous) in enumerate(zip(value, recorded[name], strict=True)):
                _verify_metrics(condition, previous, f'{label}/condition{index}')
        elif (not np.isfinite(value) or not np.isfinite(recorded[name])
              or not np.isclose(value, recorded[name], rtol=1e-10, atol=1e-12)):
            raise ValueError(f'{label}: recomputed metric differs: {name}')


def _online_receipt(run, c):
    receipt = _read(run/'online_validation.json')
    if (receipt['status'] != 'passed' or Path(receipt['run']).resolve() != run
            or receipt['updates'] != c.train.iterations
            or receipt['exact_final_slots_reconstructed']+receipt['unchanged_initial_slots']
            != c.operator.geometry_count
            or receipt['evaluation_overlap'] != 0 or receipt['future_training_fields'] != 0):
        raise ValueError('Completed, matching, label-free online input audit is required')
    return receipt


def _costs(run, c, rolling):
    result = run_costs(run, c)
    if not rolling:
        result.update(online_update_total_seconds=0., producer_wait_seconds=0.,
                      online_summary=None)
        return result
    producer = _read(run/'online_summary.json')
    overhead = np.asarray(_read(run/'timing.json')['online_update_seconds'], dtype=float)
    if (overhead.shape != (c.train.iterations,) or not np.isfinite(overhead).all()
            or np.any(overhead < 0) or not np.isfinite(producer['producer_wait_seconds'])
            or producer['producer_wait_seconds'] < 0
            or producer['capacity'] != c.operator.geometry_count
            or producer['pde_solves'] != 0 or producer['fem_solves'] != 0
            or producer['evolved_training_fields'] != 0):
        raise ValueError('Invalid completed rolling-production timings/provenance')
    result.update(online_update_total_seconds=float(overhead.sum()),
        producer_wait_seconds=producer['producer_wait_seconds'], online_summary=producer,
        online_update_scope=('Per-step controller plus FIFO wait/validation and synchronized '
            'device replacement, including its first compilation; separate from neural updates. '
            'Producer wait is a subset and must not be added again.'))
    return result


def _saved_evaluation(directory, run, c):
    directory = Path(directory).resolve(strict=True)
    receipt = _read(directory/'metrics.json')
    if (Path(receipt['run']).resolve() != run or receipt['checkpoint'] != 'model_final.eqx'
            or receipt['training_performed'] is not False):
        raise ValueError('Evaluation must use this run\'s final checkpoint without training')
    reference = Path(receipt['reference']).resolve(strict=True)
    refs, manifest = load_references(reference, c)
    if manifest['backend'] != 'local_p1_lumped_fem' or len(refs) != 8:
        raise ValueError('The optional comparison requires the same eight local FEM references')
    _, measured = saved_predictions(directory, refs)
    _verify_metrics(measured, receipt['metrics'], str(directory))
    return dict(evaluation=str(directory), run=str(run), reference=str(reference),
                conditions=len(refs), metrics=measured)


def _cohort_table(fixed, rolling):
    if fixed['reference'] != rolling['reference']:
        raise ValueError('Fixed and rolling evaluations must use the identical reference path')
    if fixed['conditions'] != rolling['conditions']:
        raise ValueError('Fixed and rolling evaluation cohort sizes differ')
    rows = []
    for index, (left, right) in enumerate(zip(fixed['metrics']['conditions'],
                                             rolling['metrics']['conditions'], strict=True)):
        rows.append(dict(condition=index, fixed={name: left[name] for name in TABLE_METRICS},
                         rolling={name: right[name] for name in TABLE_METRICS}))
    return dict(reference=fixed['reference'], condition_count=fixed['conditions'],
        fixed=fixed, rolling=rolling, table=rows,
        interpretation='Previously inspected reused comparison cohort; not a new untouched final test.')


def compare_sampling(run, control, output, rolling_evaluation, fixed_evaluation):
    if (rolling_evaluation is None) != (fixed_evaluation is None):
        raise ValueError('Supply both rolling-evaluation and fixed-evaluation, or neither')
    run, control = Path(run).resolve(strict=True), Path(control).resolve(strict=True)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Comparison output must be a new directory: {output}')
    configurations = [ConfigDict(_read(path/'config.json')) for path in (control, run)]
    audit = _online_receipt(run, configurations[1])
    paired = validate_control_result(run, control)
    records = []
    for label, path, c, rolling in zip(('fixed pool', 'rolling pools'), (control, run),
                                      configurations, (False, True), strict=True):
        runtime, summary = _read(path/'runtime.json'), _read(path/'summary.json')
        if (summary['steps'] != c.train.iterations or not (path/'model_final.eqx').is_file()
                or not (path/'optimizer_final.eqx').is_file()):
            raise ValueError(f'Run has not completed its declared fixed budget: {path}')
        reference = Path(runtime['reference']).resolve(strict=True)
        refs, manifest = load_references(reference, c)
        if manifest['backend'] != 'local_p1_lumped_fem':
            raise ValueError('Development comparison requires the common local FEM reference')
        history = _history(path, c.train.iterations)
        _, measured = saved_predictions(path, refs)
        _verify_metrics(measured, history[-1], f'{path}/last_history')
        _verify_metrics(measured, summary['final'], f'{path}/summary')
        for row in history:
            if any(not np.isfinite(row[name]) for name in measured if name != 'conditions'):
                raise ValueError(f'Nonfinite historical convergence metric: {path}')
        bank = _read(Path(runtime['inputs'])/'manifest.json')
        shape = dict(halo=c.operator.online.halo if rolling else bank['halo_size'],
                     neighbors=c.operator.online.neighbors if rolling else bank['neighbor_size'],
                     cores=c.sample.K_core, pools=c.sample.K_pool,
                     geometry_capacity=c.operator.geometry_count)
        records.append(dict(run=str(path), method=label, spatial_derivative=c.nl.method,
            reference=str(reference), condition_count=len(refs), metrics=measured,
            costs=_costs(path, c, rolling), execution_shape=shape, history=history))
    if records[0]['reference'] != records[1]['reference']:
        raise ValueError('Development reference paths differ')
    reused = None
    if rolling_evaluation is not None:
        fixed = _saved_evaluation(fixed_evaluation, control, configurations[0])
        rolling = _saved_evaluation(rolling_evaluation, run, configurations[1])
        reused = _cohort_table(fixed, rolling)
    result = dict(validation=paired, online_validation=audit,
        reference=records[0]['reference'], runs=records, reused8=reused,
        metric_definition=('Arithmetic mean of per-geometry relative L2, using physical P1 '
            'volumes and trapezoidal snapshot time weights; terminal errors use physical volumes.'),
        limitations=[
            'Common cold model/optimizer and slot/pool/time draws are identical; rolling physical inputs change.',
            'Initial-bank and rolling padding sizes are recorded per run. This historical comparison '
            'does not isolate sampling from execution shape, compilation or reduction-order effects.',
            'The rolling sampler repairs candidate admission and logs rejection/retry statistics; '
            'the original fixed inputs and physical geometry thresholds are unchanged. '
            'Documented sampler source differences are listed in the validation receipt.',
            'One paired initialization and finite reused evaluation cohorts do not establish cross-seed robustness.'],
        model_inference_performed=False, training_performed=False)
    # Publish only after every completion, provenance, metric and timing gate passes.
    output.mkdir(parents=True, exist_ok=False)
    _plot_convergence(output, records)
    (output/'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(output), runs=[dict(method=row['method'], metrics=row['metrics'],
        costs=row['costs'], execution_shape=row['execution_shape']) for row in records],
        reused8=reused), allow_nan=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path, help='Completed rolling run')
    parser.add_argument('--control', required=True, type=Path, help='Completed fixed-bank control')
    parser.add_argument('--output', required=True, type=Path, help='New comparison directory')
    parser.add_argument('--rolling-evaluation', type=Path,
                        help='Saved final rolling-model evaluation on the reused eight geometries')
    parser.add_argument('--fixed-evaluation', type=Path,
                        help='Saved final fixed-model evaluation on the same eight geometries')
    args = parser.parse_args()
    compare_sampling(args.run, args.control, args.output, args.rolling_evaluation, args.fixed_evaluation)


if __name__ == '__main__':
    main()
