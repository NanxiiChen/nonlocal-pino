"""Audit joint-query training against the explicitly reused historical FiLM run."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

from scripts.time_fusion_wall_clock import birth, recover


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def matched_configs(film, joint):
    left, right = copy.deepcopy(film), copy.deepcopy(joint)
    if 'temporal_mode' in left['arch'] or right['arch'].pop('temporal_mode') != 'joint_query':
        raise ValueError('Require historical default FiLM and explicit joint queries')
    expected = dict(expected_conditions=64, chunk_size=512,
                    metric='equal_point_space_time', render_training_figures=False)
    if right.pop('monitor') != expected or 'monitor' in left:
        raise ValueError('Unexpected historical/new monitoring configuration')
    if left['log']['eval_every'] != 1000 or right['log']['eval_every'] != 500:
        raise ValueError('Unexpected evaluation intervals')
    right['log']['eval_every'] = 1000
    if left != right:
        raise ValueError('Physics, optimization or sampling changed along with time injection')


def baseline_clock(run, parent_path):
    parent = read(parent_path)
    stages = [s for s in parent['stages'] if s['name'] == 'train']
    if (parent['status'] != 'complete' or Path(parent['run']) != run
            or len(stages) != 1 or stages[0]['status'] != 'passed'):
        raise ValueError('Missing completed historical training stage')
    total = stages[0]['seconds']
    metric, log = run/'metrics.jsonl', Path(parent_path).parent/'train.log'
    observed = rows(metric)
    evidence = dict(launch_epoch=birth(log)['epoch_seconds'],
        metric_created_epoch=birth(metric)['epoch_seconds'],
        metric_written_epoch=metric.stat().st_mtime,
        last_counter=observed[-1]['seconds'], cold_total=total, tolerance_seconds=.1)
    result = recover([r['seconds'] for r in observed], np.zeros(len(observed)), **evidence)
    return dict(parent=str(parent_path), launch_log=str(log), metrics=str(metric),
                evidence=evidence, **result)


def paired_arrays(left, right, name):
    with np.load(left/name, allow_pickle=False) as a, np.load(right/name, allow_pickle=False) as b:
        if set(a.files) != set(b.files):
            raise ValueError('Paired input schema differs: '+name)
        for key in a.files:
            if a[key].dtype != b[key].dtype or not np.array_equal(a[key], b[key]):
                raise ValueError('Paired arrays differ: '+name+'/'+key)


def field_errors(predictions, reference):
    manifest = read(reference/'manifest.json')
    if len(manifest['cases']) != 64:
        raise ValueError('Final comparison requires the complete64case bank')
    result = []
    # One case at a time: no GPU inference or unnecessary full-bank allocation.
    for index, case in enumerate(manifest['cases']):
        with np.load(reference/case['file'], allow_pickle=False) as r, np.load(
                predictions/f'prediction_{index:03d}.npz', allow_pickle=False) as p:
            np.testing.assert_array_equal(p['points'], r['points'])
            np.testing.assert_array_equal(p['times'], r['times'])
            target, prediction = r['fields'].astype(float), p['prediction'].astype(float)
        if prediction.shape != target.shape or target.shape[-1] != 2:
            raise ValueError('Final prediction grid/channels differ from the reference')
        result.append([float(np.linalg.norm((prediction[..., k]-target[..., k]).ravel()) /
                             np.linalg.norm(target[..., k].ravel())) for k in range(2)])
    value = np.array(result)
    if not np.isfinite(value).all() or np.any(value <= 0):
        raise ValueError('Final relative errors must be finite and positive')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    args = parser.parse_args(); bundle = args.bundle.resolve(strict=True)
    import equinox as eqx
    import jax
    from ml_collections import ConfigDict
    from examples.particle_corrosion2d.network import build_network
    from examples.particle_corrosion2d.online_validation import audit_online_run
    journal = read(bundle/'journal.json')
    if journal['baseline_reused'] is not True:
        raise ValueError('This audit is only for the user-approved reused FiLM comparison')
    runs = {arm: Path(path) for arm, path in journal['runs'].items()}
    configs = {arm: read(run/'config.json') for arm, run in runs.items()}
    matched_configs(configs['film'], configs['joint'])
    runtime = {arm: read(run/'runtime.json') for arm, run in runs.items()}
    for key in ('seed', 'inputs'):
        if runtime['film'][key] != runtime['joint'][key]:
            raise ValueError('Unmatched '+key)
    exclusions = [{r['reference'], *r['excluded_reference_paths']} for r in runtime.values()]
    if exclusions[0] != exclusions[1]:
        raise ValueError('Evaluation/exclusion bank union changed')
    for r in runtime.values():
        if not r['cold_initialization'] or r['training_uses_reference_targets']:
            raise ValueError('Both models must be cold physics-only training')
    for name in ('sampling.npz', 'online_sampling.npz',
                 'online_initial_inputs.npz', 'online_final_inputs.npz'):
        paired_arrays(runs['film'], runs['joint'], name)
    models = {arm: eqx.tree_deserialise_leaves(run/'model_initial.eqx',
        build_network(ConfigDict(configs[arm]), jax.random.PRNGKey(runtime[arm]['seed'])))
        for arm, run in runs.items()}
    common = eqx.tree_at(lambda m: m.time, models['film'], None)
    for a, b in zip(jax.tree.leaves(common), jax.tree.leaves(models['joint'].backbone), strict=True):
        if eqx.is_array(a) and not np.array_equal(a, b):
            raise ValueError('Common initial parameter leaves differ')
    historical = read(journal['baseline_evaluation'])
    if (historical['status'] != 'passed' or historical['count'] != 64
            or historical['steps'] != 20000 or historical['checkpoint'] != 'model_final.eqx'
            or historical['run'] != str(runs['film']) or historical['reference'] != journal['reference']):
        raise ValueError('Historical final64 evaluation belongs to another study')
    old_online = read(runs['film']/'online_validation.json')
    if old_online['status'] != 'passed' or old_online['updates'] != 20000:
        raise ValueError('Missing historical completed online-input audit')
    clock = baseline_clock(runs['film'], journal['baseline_journal'])
    (bundle/'baseline_clock.json').write_text(json.dumps(clock, indent=2)+'\n')
    result = dict(status='passed', baseline_reused=True,
        common_parameters_identical=True, actual_input_streams_identical=True,
        final_metric='Equal-point per-channel L2 over all saved space-time points; then channel/sample mean',
        convergence_metric='Physical spatial/time quadrature weighted per-channel space-time L2',
        convergence_scope='FiLM:4development cases/21records; joint:64test cases/41records. Different cohorts.',
        timing_scope='Entire cold train process including initialization, compilation, evaluation, writes and exit',
        hardware_scope='Same GPU model, different physical UUIDs; different monitoring overhead',
        baseline_hardware=read(journal['baseline_hardware']), joint_hardware=journal['gpu'],
        methods={})
    for arm, run in runs.items():
        summary = read(run/'summary.json')
        count, interval = (4, 1000) if arm == 'film' else (64, 500)
        history = rows(run/'metrics.jsonl')
        if (summary['steps'] != 20000 or summary['heldout_count'] != count
                or [r['step'] for r in history] != list(range(0, 20001, interval))
                or not (run/'model_final.eqx').is_file()):
            raise ValueError('Incomplete20k budget or monitoring history: '+arm)
        weighted = np.array([[[c['phi_pooled'], c['c_pooled']] for c in r['conditions']] for r in history])
        if weighted.shape != (20000//interval+1, count, 2) or not np.isfinite(weighted).all():
            raise ValueError('Incomplete cohort histories: '+arm)
        np.testing.assert_allclose(weighted.mean(axis=1),
            [[r['phi_pooled'], r['c_pooled']] for r in history], rtol=1e-12, atol=0)
        final = field_errors(Path(journal['prediction_roots'][arm]), Path(journal['reference']))
        if arm == 'film':
            if [r['index'] for r in historical['records']] != list(range(64)):
                raise ValueError('Historical64sample ordering changed')
            np.testing.assert_allclose(final, [[r['phi'], r['concentration']] for r in historical['records']],
                                       rtol=2e-12, atol=2e-14)
            elapsed = np.array(clock['elapsed_seconds']); total = clock['evidence']['cold_total']
            online = old_online
        else:
            equal = np.array([[[c['equal_phi'], c['equal_c']] for c in r['conditions']] for r in history])
            if equal.shape != (41, 64, 2) or not np.isfinite(equal).all():
                raise ValueError('Incomplete64case equal-point monitoring')
            np.testing.assert_allclose(final, equal[-1], rtol=2e-12, atol=2e-14)
            np.testing.assert_allclose(equal.mean(axis=(1, 2)), [r['channel_mean'] for r in history],
                                       rtol=1e-12, atol=0)
            elapsed = np.array([r['cold_wall_seconds'] for r in history])
            stage = [s for s in journal['stages'] if s['name'] == 'train-joint']
            if len(stage) != 1 or stage[0]['status'] != 'passed':
                raise ValueError('Missing new complete cold parent stage')
            total = stage[0]['seconds']
            if total < elapsed[-1]:
                raise ValueError('Process exit precedes evaluation')
            elapsed[-1] = total
            online = audit_online_run(run, workers=4)
        if np.any(np.diff(elapsed) <= 0) or elapsed[0] <= 0:
            raise ValueError('Cold elapsed observations must remain ordered')
        np.savez_compressed(bundle/(arm+'-convergence.npz'),
            steps=np.array([r['step'] for r in history]), cold_wall_seconds=elapsed,
            weighted_errors=weighted, final_errors=final, parent_seconds=total)
        result['methods'][arm] = dict(run=str(run), prediction_root=journal['prediction_roots'][arm],
            parameters=runtime[arm]['parameters'], full_cold_seconds=total,
            final_phi=float(final[:, 0].mean()), final_c=float(final[:, 1].mean()),
            final_channel_mean=float(final.mean()), final_field_norms=final.tolist(),
            records=len(history), monitor_count=count, final_count=64, online_audit=online)
    (bundle/'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print('REUSED FILM / JOINT AUDIT PASSED', flush=True)


if __name__ == '__main__':
    main()
