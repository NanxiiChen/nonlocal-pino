"""Compare matched cold optimizers by saved paths, without target selection."""
import argparse
import json
from pathlib import Path
import numpy as np
import equinox as eqx
import jax
from ml_collections import ConfigDict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from examples.confined_ch2d.operator_training import build_network


def compare(baseline, candidate, output):
    output.mkdir(parents=True, exist_ok=False)
    runs = (baseline, candidate)
    configs = [json.loads((r/'config.json').read_text()) for r in runs]
    labels = [c['train']['optimizer'].upper() for c in configs]
    a, b = [json.loads(json.dumps(c)) for c in configs]
    for c in (a, b):
        c['log']['run_name'] = ''
        c['train'].pop('optimizer')
    assert a == b, 'Optimizer must be the only strategy change'
    for relative in ('examples/confined_ch2d/operator_training.py',
                     'src/nonlocal_pinn/nn/point_conditional.py',
                     'src/nonlocal_pinn/training/optimizers.py',
                     'src/nonlocal_pinn/training/runtime.py'):
        assert (baseline/'source'/relative).read_bytes()==(candidate/'source'/relative).read_bytes(), relative
    jax.config.update('jax_enable_x64', True)
    template = build_network(ConfigDict(configs[0]), jax.random.key(0))
    models = [eqx.tree_deserialise_leaves(r/'model_initial.eqx', template) for r in runs]
    for x, y in zip(jax.tree.leaves(models[0]), jax.tree.leaves(models[1]), strict=True):
        if eqx.is_array(x):np.testing.assert_array_equal(x, y)
    draws = []
    for r in runs:
        with np.load(r/'training_draws.npz', allow_pickle=False) as f:draws.append(f['condition_ids'])
    np.testing.assert_array_equal(*draws)
    histories = [[json.loads(s) for s in (r/'metrics.jsonl').read_text().splitlines()] for r in runs]
    metrics = [json.loads((r/'evaluation/metrics.json').read_text()) for r in runs]
    summaries = [json.loads((r/'summary.json').read_text()) for r in runs]
    effects = [json.loads((r/'geometry_effect/metrics.json').read_text()) for r in runs]
    assert effects[0]['reference']==effects[1]['reference']
    assert effects[0]['common_points']==effects[1]['common_points']
    assert all(h[-1]['step'] == 20000 for h in histories)
    for axis_kind in ('updates', 'wall_time'):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
        for h, label in zip(histories, labels, strict=True):
            x = [v['step'] if axis_kind == 'updates' else v['elapsed_seconds']/60 for v in h]
            for ax, field in zip(axes, ('phi_pooled', 'phi_terminal'), strict=True):
                ax.plot(x, [100*v['test_'+field] for v in h], marker='o', markersize=2.5, label=label)
                ax.set(xlabel='Optimizer calls' if axis_kind == 'updates' else 'Cold elapsed time (min)',
                       ylabel='Validation phase relative L2 (%)',
                       title='All five saved times' if field == 'phi_pooled' else 'Terminal time t = 3')
                ax.grid(alpha=.2);ax.legend()
        fig.suptitle('Same architecture, initialization, physics and geometry draws; native reference')
        fig.savefig(output/f'convergence_{axis_kind}.png', dpi=180)
        fig.savefig(output/f'convergence_{axis_kind}.pdf')
        plt.close(fig)
    target=metrics[0]['test']
    crossings=[r for r in histories[1] if r['test_phi_pooled']<=target['phi_pooled']
               and r['test_phi_terminal']<=target['phi_terminal']]
    first=({k:crossings[0][k] for k in ('step','elapsed_seconds','test_phi_pooled','test_phi_terminal')}
           if crossings else None)
    report = dict(baseline=str(baseline), candidate=str(candidate), optimizers=labels,
        only_optimizer_changed=True, numerical_training_sources_identical=True, initial_parameters_identical=True,
        actual_condition_draws_identical=True, optimizer_calls=20000,
        native_validation=[m['test'] for m in metrics],
        candidate_relative_phase_error_reduction=1-metrics[1]['test']['phi_pooled']/metrics[0]['test']['phi_pooled'],
        first_candidate_matching_both_baseline_final_phase_metrics=first,
        geometry_response_relative_error=[e['geometry_response_relative_error'] for e in effects],
        geometry_response_amplitude_ratio=[e['geometry_response_amplitude_ratio'] for e in effects],
        geometry_response_correlation=[e['geometry_response_correlation'] for e in effects],
        cold_seconds=[s['wall_seconds'] for s in summaries],
        warm_update_seconds=[s['mean_update_seconds_after_100'] for s in summaries],
        milestones=[dict(optimizer=label, values=[{k:r[k] for k in
            ('step','test_phi_pooled','test_phi_terminal','test_mu_pooled')}
            for r in h if r['step'] in (5000,10000,15000,20000)]) for label,h in zip(labels,histories,strict=True)],
        selection='Fixed 20000-call final in both runs; no held-out checkpoint selection',
        scope='One cold run per optimizer on eight reused development geometries; no stable optimizer ordering or local-PDE claim')
    (output/'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args();compare(a.baseline, a.candidate, a.output)
