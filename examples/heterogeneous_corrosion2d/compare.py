"""Re-evaluate completed runs against one independent local-FV reference.

Only saved checkpoints are compared; native-reference training logs are never
mixed into these convergence curves. This reporting command does not train or
change any run. Invoke after the compared training runs have completed.
"""
import argparse
import json
import math
from pathlib import Path
import re

import numpy as np


def _read_json(path):
    return json.loads(Path(path).read_text())


def _write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def _checkpoint_paths(run, config, completion):
    """Select each step once, preferring the completed final model at its step."""
    final_step = completion['steps']
    if (isinstance(final_step, bool) or not isinstance(final_step, int)
            or final_step < 1 or final_step != config['train']['iterations']):
        raise ValueError(f'Completed step count does not match the saved configuration: {run}')
    initial, final = run/'model_initial.eqx', run/'model_final.eqx'
    if not initial.is_file() or not final.is_file():
        raise FileNotFoundError(f'Comparison requires initial and completed final models: {run}')
    selected = {0: initial}
    for path in sorted(run.glob('model_*.eqx')):
        match = re.fullmatch(r'model_(\d+)\.eqx', path.name)
        if match is None:
            continue
        step = int(match.group(1))
        if step < 1 or step > final_step or step in selected:
            raise ValueError(f'Duplicate or invalid numeric checkpoint step: {path}')
        marker = run/f'checkpoint_{step:06d}.json'
        if not marker.is_file() or _read_json(marker)['step'] != step:
            raise ValueError(f'Numeric checkpoint lacks its matching completion record: {path}')
        selected[step] = path
    selected[final_step] = final
    return sorted(selected.items())


def _resources(run):
    timing = _read_json(run/'timing.json')
    names = ('first_update_compile_seconds', 'warm_update_mean_seconds', 'total_seconds')
    result = {}
    for name in names:
        value = timing[name]
        if value is None and name == 'warm_update_mean_seconds':
            result[name] = None
        elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f'Invalid {name} in {run}/timing.json')
        else:
            result[name] = float(value)
    memory = _read_json(run/'memory.json')
    if not isinstance(memory, list):
        raise ValueError(f'Expected a memory-record list: {run}/memory.json')
    active_peaks, pool_peaks = [], []
    for record in memory:
        allocator = record['allocator']
        if allocator is None:
            continue
        for name, values in (('peak_bytes_in_use', active_peaks), ('peak_pool_bytes', pool_peaks)):
            if name in allocator:
                value = allocator[name]
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(f'Invalid allocator counter {name}: {run}/memory.json')
                values.append(value)
    result['peak_active_bytes'] = max(active_peaks) if active_peaks else None
    result['peak_active_gib'] = max(active_peaks)/2**30 if active_peaks else None
    result['peak_pool_bytes'] = max(pool_peaks) if pool_peaks else None
    result['peak_memory_status'] = 'recorded_allocator_high_water_mark' if active_peaks else 'not_recorded'
    result['timing_note'] = 'First update includes compilation; total includes setup, sampling, evaluation and checkpoint I/O.'
    result['memory_note'] = 'Active peak is peak_bytes_in_use; allocator pool reservation is reported separately.'
    return result


def _label(config):
    method, optimizer = config['nl']['method'], config['train']['optimizer']
    if method not in ('sparse', 'local_ad'):
        raise ValueError(f'Unsupported comparison spatial method: {method}')
    if optimizer not in ('adam', 'soap'):
        raise ValueError(f'Unsupported comparison optimizer: {optimizer}')
    spatial = 'Nonlocal' if method == 'sparse' else 'Local AD'
    optimizer_label = 'Adam' if optimizer == 'adam' else 'SOAP'
    return f'{spatial} / {optimizer_label} (LR {config["train"]["lr_init"]:g})'


def _plots(output, reports, predictions, reference, c):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from examples.heterogeneous_corrosion2d.problem import grid

    colors = plt.get_cmap('tab10')(np.arange(len(reports)) % 10)
    panels = (
        ('pooled_rel_l2_phi', 'Phase: all saved times'),
        ('terminal_rel_l2_phi', 'Phase: final time'),
        ('pooled_rel_l2_c', 'Concentration: all saved times'),
        ('terminal_rel_l2_c', 'Concentration: final time'),
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
    for ax, (metric, title) in zip(axes.ravel(), panels, strict=True):
        for color, report in zip(colors, reports, strict=True):
            rows = [row for row in report['records'] if row['step'] > 0]
            values = [row['metrics'][metric] for row in rows]
            if any(value is None or not np.isfinite(value) or value <= 0 for value in values):
                raise ValueError(f'Convergence plot requires positive finite {metric}: {report["run"]}')
            ax.semilogy([row['step'] for row in rows], 100*np.asarray(values),
                        color=color, marker='o', markersize=3.5, linewidth=1.6,
                        label=report['label'])
        ax.set(title=title, xlabel='Training updates', ylabel='Mean relative L2 (%)')
        ax.grid(alpha=.25, which='both')
    axes[0, 0].legend(fontsize=9)
    fig.suptitle('Saved checkpoints evaluated against the same independent local-FV reference\n'
                 'Equal-weight mean over held-out materials; step 0 is recorded in JSON')
    fig.savefig(output/'convergence.png', dpi=180)
    plt.close(fig)

    xx, yy, _ = grid(c)
    count = 4
    if len(reference['parameters']) < count:
        raise ValueError('The final interface comparison requires at least four held-out materials')
    fig, axes = plt.subplots(1, count, figsize=(15, 3.4), layout='constrained')
    for condition, ax in enumerate(axes):
        target = reference['fields'][condition, -1, :, 0].reshape(xx.shape)
        ax.contour(xx, yy, target, levels=[.5], colors='black', linestyles='--', linewidths=1.8)
        missing = []
        for color, predicted, report in zip(colors, predictions, reports, strict=True):
            phase = predicted[condition, -1, :, 0].reshape(xx.shape)
            if phase.min() < .5 < phase.max():
                ax.contour(xx, yy, phase, levels=[.5], colors=[color], linestyles='-', linewidths=1.6)
            else:
                missing.append(report['label'])
        ax.set(title=f'Held-out material {condition+1}', xlabel='x',
               xlim=(0, c.geometry.length), ylim=(c.geometry.height, 0), aspect='equal')
        if condition == 0:
            ax.set_ylabel('Depth y')
        ax.grid(alpha=.15)
        if missing:
            ax.text(.02, .02, 'No interface:\n'+'\n'.join(missing), transform=ax.transAxes,
                    fontsize=6, va='bottom', bbox=dict(facecolor='white', alpha=.8, edgecolor='none'))
    handles = [Line2D([0], [0], color='black', linestyle='--', linewidth=1.8,
                      label='Independent local FV')]
    handles.extend(Line2D([0], [0], color=color, linewidth=1.6, label=report['label'])
                   for color, report in zip(colors, reports, strict=True))
    fig.legend(handles=handles, loc='outside lower center', ncol=min(len(handles), 4), fontsize=9)
    final_time = float(reference['time'][-1])
    fig.suptitle(f'Final corrosion interfaces, phase = 0.5, t* = {final_time:g}')
    fig.savefig(output/'final_interfaces.png', dpi=180)
    plt.close(fig)


def compare(runs, reference_path, output):
    """Evaluate completed checkpoints without loading optimizer or training labels."""
    import equinox as eqx
    import jax
    from ml_collections import ConfigDict

    from examples.heterogeneous_corrosion2d.evaluate import load_reference, make_evaluator
    from examples.heterogeneous_corrosion2d.network import build_network
    from examples.heterogeneous_corrosion2d.problem import grid
    from nonlocal_pinn.training.runtime import configure_gpu_policy

    run_paths = [Path(run).resolve(strict=True) for run in runs]
    if not run_paths or len(set(run_paths)) != len(run_paths):
        raise ValueError('Comparison needs at least one run and no duplicate run paths')
    reference_path = Path(reference_path).resolve(strict=True)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Comparison output must be a new directory: {output}')
    specs = []
    for run in run_paths:
        config, runtime = _read_json(run/'config.json'), _read_json(run/'runtime.json')
        completion = _read_json(run/'summary.json')
        if config['train']['compute_dtype'] != 'float32':
            raise ValueError(f'The shared evaluator currently requires FP32: {run}')
        specs.append(dict(run=run, config=config, runtime=runtime,
            checkpoints=_checkpoint_paths(run, config, completion),
            resources=_resources(run), label=_label(config)))
    policy = specs[0]['config']['train']['gpu_policy']
    precision = specs[0]['config']['train']['matmul_precision']
    original_labels = [spec['label'] for spec in specs]
    for spec in specs:
        if (spec['config']['train']['gpu_policy'] != policy
                or spec['config']['train']['matmul_precision'] != precision):
            raise ValueError('Compared runs must share the declared compilation and matmul policy')
        if original_labels.count(spec['label']) > 1:
            spec['label'] += ' ['+spec['run'].name+']'
        c = ConfigDict(spec['config'])
        _, _, points = grid(c)
        reference = load_reference(reference_path, c, points)
        if reference['backend'].item() != 'independent_local_fv':
            raise ValueError('Comparison requires the independent_local_fv backend, not the native graph reference')
        if len(reference['parameters']) < 4:
            raise ValueError('Comparison needs at least four held-out material samples')
        spec['model_config'], spec['reference'] = c, reference
    configure_gpu_policy(policy)
    jax.config.update('jax_enable_x64', False)
    jax.config.update('jax_default_matmul_precision', precision)
    output.mkdir(parents=True, exist_ok=False)
    reports, summaries, predictions = [], [], []
    common_reference, common_config = None, None
    # All configurations were checked against this one physical problem and
    # grid. Reuse the evaluator so optimizer-only differences do not recompile
    # an identical prediction path; model architecture remains a JIT input.
    evaluate = make_evaluator(specs[0]['model_config'], specs[0]['reference'])
    for index, spec in enumerate(specs, start=1):
        run, runtime = spec['run'], spec['runtime']
        c, reference = spec['model_config'], spec['reference']
        common_reference, common_config = reference, c
        template = build_network(c, jax.random.split(jax.random.PRNGKey(runtime['seed']), 2)[0])
        records, final_prediction = [], None
        run_output = output/f'run_{index:02d}'
        run_output.mkdir()
        for step, checkpoint in spec['checkpoints']:
            model = eqx.tree_deserialise_leaves(checkpoint, template)
            predicted, metrics = evaluate(model)
            record = dict(step=step, checkpoint=str(checkpoint), metrics=metrics)
            records.append(record)
            _write_json(run_output/f'metrics_{step:06d}.json', record)
            final_prediction = predicted
            print(json.dumps(dict(run=str(run), label=spec['label'], step=step,
                pooled_rel_l2_phi=metrics['pooled_rel_l2_phi'],
                terminal_rel_l2_phi=metrics['terminal_rel_l2_phi']), allow_nan=False), flush=True)
        report = dict(run=str(run), label=spec['label'], reference=str(reference_path), records=records)
        _write_json(run_output/'metrics.json', report)
        reports.append(report)
        predictions.append(final_prediction)
        summaries.append(dict(run=str(run), label=spec['label'],
            spatial_derivative=c.nl.method, optimizer=c.train.optimizer,
            seed=runtime['seed'], devices=runtime['devices'],
            checkpoint_steps=[row['step'] for row in records],
            final_step=records[-1]['step'], final_metrics=records[-1]['metrics'],
            resources=spec['resources']))
    _write_json(output/'metrics.json', dict(reference=str(reference_path), runs=reports))
    summary = dict(reference=str(reference_path), reference_backend='independent_local_fv',
        material_count=len(common_reference['parameters']),
        final_interface_condition_indices=[0, 1, 2, 3],
        aggregation='Arithmetic mean of per-material relative L2; pooled metrics use all reference times and spatial nodes.',
        checkpoint_selection='All published numeric checkpoints, initial model and completed final; final step evaluated once.',
        convergence_step_zero='Included in JSON, excluded from plotted curves.', runs=summaries)
    _write_json(output/'summary.json', summary)
    _plots(output, reports, predictions, common_reference, common_config)
    adam_ids = [i for i, spec in enumerate(specs) if spec['config']['train']['optimizer'] == 'adam']
    if 1 < len(adam_ids) < len(specs):
        selected_output = output/'adam_comparison'
        selected_output.mkdir()
        _plots(selected_output, [reports[i] for i in adam_ids],
               [predictions[i] for i in adam_ids], common_reference, common_config)
    print('COMPARISON '+str(output), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--reference', type=Path, required=True, help='Shared independent local-FV NPZ')
    parser.add_argument('--output', type=Path, required=True, help='New report directory')
    args = parser.parse_args()
    compare(args.runs, args.reference, args.output)


if __name__ == '__main__':
    main()
