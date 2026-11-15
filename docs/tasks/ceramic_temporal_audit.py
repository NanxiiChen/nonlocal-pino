"""Independent final-field and paired-input audit for the Ceramic temporal study."""
import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--bundle', type=Path, required=True)
    a = p.parse_args(); bundle = a.bundle.resolve(strict=True)
    import equinox as eqx
    import jax
    import numpy as np
    from ml_collections import ConfigDict
    from examples.particle_corrosion2d.network import build_network
    from examples.particle_corrosion2d.evaluate import load_references
    from examples.particle_corrosion2d.online_validation import audit_online_run
    journal = read(bundle/'journal.json')
    runs = {arm: Path(journal['runs'][arm]) for arm in ('film', 'joint')}
    configs = {arm: read(run/'config.json') for arm, run in runs.items()}
    if configs['film']['arch'].pop('temporal_mode') != 'decoder_film': raise ValueError('Wrong FiLM arm')
    if configs['joint']['arch'].pop('temporal_mode') != 'joint_query': raise ValueError('Wrong joint arm')
    if configs['film'] != configs['joint']: raise ValueError('Unmatched physical/training configurations')
    runtime = {arm: read(run/'runtime.json') for arm, run in runs.items()}
    for key in ('seed', 'inputs', 'reference', 'excluded_reference_paths'):
        if runtime['film'][key] != runtime['joint'][key]: raise ValueError('Unmatched '+key)
    for name in ('sampling.npz', 'online_sampling.npz', 'online_initial_inputs.npz', 'online_final_inputs.npz'):
        with np.load(runs['film']/name, allow_pickle=False) as left, np.load(runs['joint']/name, allow_pickle=False) as right:
            if set(left.files) != set(right.files): raise ValueError('Unmatched schema '+name)
            for key in left.files:
                if left[key].dtype != right[key].dtype or not np.array_equal(left[key], right[key]):
                    raise ValueError('Unmatched paired input '+name+'/'+key)
    models = {}
    for arm, run in runs.items():
        c = ConfigDict(read(run/'config.json'))
        models[arm] = eqx.tree_deserialise_leaves(run/'model_initial.eqx',
            build_network(c, jax.random.PRNGKey(runtime[arm]['seed'])))
    common = eqx.tree_at(lambda m: m.time, models['film'], None)
    for left, right in zip(jax.tree.leaves(common), jax.tree.leaves(models['joint'].backbone), strict=True):
        if eqx.is_array(left) and not np.array_equal(left, right): raise ValueError('Unpaired common initialization')
    result = dict(status='passed', metric='equal-point full-space-time per-channel L2, then channel/sample mean',
        common_parameters_identical=True, actual_input_streams_identical=True, methods={})
    for arm, run in runs.items():
        c = ConfigDict(read(run/'config.json')); refs, _ = load_references(runtime[arm]['reference'], c)
        summary = read(run/'summary.json')
        if summary['steps'] != 20000 or len(refs) != 64 or summary['heldout_count'] != 64:
            raise ValueError('Incomplete training/evaluation budget')
        records = [json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
        if [r['step'] for r in records] != list(range(0, 20001, 500)):
            raise ValueError('Missing convergence records')
        values = np.array([[[r['equal_phi'], r['equal_c']] for r in row['conditions']] for row in records])
        if values.shape != (41, 64, 2) or not np.isfinite(values).all(): raise ValueError('Incomplete full cohort')
        if any(row['channel_mean'] != float(np.mean([r['channel_mean'] for r in row['conditions']])) for row in records):
            raise ValueError('Incorrect cohort aggregation')
        times = np.array([row['cold_wall_seconds'] for row in records])
        if not np.all(np.diff(times) > 0) or times[0] <= 0: raise ValueError('Invalid direct cold clock')
        final = []
        for i, ref in enumerate(refs):
            with np.load(run/f'prediction_{i:03d}.npz', allow_pickle=False) as stored:
                np.testing.assert_array_equal(stored['points'], ref['points'])
                np.testing.assert_array_equal(stored['times'], ref['times'])
                prediction = stored['prediction'].astype(np.float64)
            target = ref['fields'].astype(np.float64)
            if prediction.shape != target.shape: raise ValueError('Wrong final evaluation grid')
            # Independent flatten-and-norm oracle, not the training metric helper.
            final.append([float(np.linalg.norm((prediction[..., k]-target[..., k]).ravel())/
                                np.linalg.norm(target[..., k].ravel())) for k in range(2)])
        np.testing.assert_allclose(values[-1], final, rtol=2e-12, atol=2e-14)
        online = audit_online_run(run, workers=4)
        stages = [s for s in journal['stages'] if s['name'] == 'train-'+arm]
        if len(stages) != 1 or stages[0]['status'] != 'passed': raise ValueError('Missing parent completion')
        elapsed = stages[0]['seconds']
        if elapsed < times[-1]: raise ValueError('Parent total precedes final evaluation')
        np.savez_compressed(bundle/(arm+'-convergence.npz'), steps=np.arange(0, 20001, 500),
            cold_wall_seconds=times, errors=values, final_errors=np.array(final), parent_seconds=elapsed)
        result['methods'][arm] = dict(run=str(run), parameters=runtime[arm]['parameters'],
            final_phi=float(np.mean(np.array(final)[:, 0])), final_c=float(np.mean(np.array(final)[:, 1])),
            final_channel_mean=float(np.mean(final)), full_cold_seconds=elapsed, records=41, test_count=64,
            final_field_norms=final, online_audit=online)
    (bundle/'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print('PAIRED AUDIT PASSED', flush=True)


if __name__ == '__main__': main()
