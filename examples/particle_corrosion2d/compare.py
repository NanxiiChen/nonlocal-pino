"""Strict paired comparison, defaulting to saved fields and CPU-only analysis.

``--reevaluate`` explicitly reloads the final networks and may use the GPU.
Without that flag, importing/running this module never imports JAX or Equinox.
All artifact identities are explicit paths and direct byte/array comparisons.
"""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
from ml_collections import ConfigDict

from .evaluate import load_references, metrics, render_gallery


def _json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def _same_file(a, b):
    """Direct file comparison; no artifact hashes or hash-based loading."""
    a, b = Path(a), Path(b)
    if a.stat().st_size != b.stat().st_size:
        return False
    with a.open('rb') as left, b.open('rb') as right:
        while True:
            block = left.read(1024*1024)
            other = right.read(1024*1024)
            if block != other:
                return False
            if not block:
                return True


def _canonical_config(config):
    result = copy.deepcopy(config)
    method = result['nl'].pop('method')
    biases = result['loss'].pop('gradnorm_bias')
    expected = {'nonlocal': 5, 'local_ad': 7}
    if method not in expected or len(biases) != expected[method]:
        raise ValueError('Nonlocal/local AD require exactly five/seven explicit loss biases')
    if not np.isfinite(biases).all() or min(biases) <= 0:
        raise ValueError('Loss biases must be finite and positive')
    if method == 'local_ad' and biases[5:] != [1., 1.]:
        raise ValueError('Matched local control must append the two declared unit Neumann biases')
    result['loss']['shared_gradnorm_bias'] = biases[:5]
    return result, method


def validate_pair(runs):
    """Verify common setup, actual cold bytes, all input draws and common sources."""
    runs = [Path(run).resolve(strict=True) for run in runs]
    if len(runs) < 2 or len(set(runs)) != len(runs):
        raise ValueError('Comparison needs at least two distinct run directories')
    configs = [_json(run/'config.json') for run in runs]
    runtimes = [_json(run/'runtime.json') for run in runs]
    canonical = [_canonical_config(config) for config in configs]
    if {method for _, method in canonical} != {'nonlocal', 'local_ad'}:
        raise ValueError('The paired comparison must include both nonlocal and local_ad')
    for i, ((config, _), runtime) in enumerate(zip(canonical, runtimes, strict=True)):
        if config != canonical[0][0]:
            raise ValueError(f'Run {i} changes configuration beyond spatial method/Neumann biases')
        for key in ('inputs', 'reference', 'seed'):
            if runtime[key] != runtimes[0][key]:
                raise ValueError(f'Run {i} has different {key} provenance')
        if runtime['training_uses_reference_targets'] or not runtime['cold_initialization']:
            raise ValueError('Comparison requires cold physics-only runs')
    source_names = (
        'examples/particle_corrosion2d/network.py',
        'examples/particle_corrosion2d/losses.py',
        'examples/particle_corrosion2d/local_ad.py',
        'examples/particle_corrosion2d/problem.py',
        'examples/particle_corrosion2d/inputs.py',
        'src/nonlocal_pinn/nn/point_conditional.py',
        'src/nonlocal_pinn/nn/conditional.py',
        'src/nonlocal_pinn/training/optimizers.py',
    )
    for other in runs[1:]:
        for filename in ('model_initial.eqx', 'optimizer_initial.eqx'):
            if not _same_file(runs[0]/filename, other/filename):
                raise ValueError(f'Cold initial state differs: {other/filename}')
        for source in source_names:
            if not _same_file(runs[0]/'source'/source, other/'source'/source):
                raise ValueError(f'Paired source differs: {source}')
    with np.load(runs[0]/'sampling.npz', allow_pickle=False) as source:
        first = {name: source[name] for name in source.files}
    if set(first) != {'condition_ids', 'pool_ids', 'times'}:
        raise ValueError('Unexpected sampling schema')
    steps, batch = configs[0]['train']['iterations'], configs[0]['operator']['condition_batch']
    expected_shapes = {'condition_ids': (steps, batch), 'pool_ids': (steps, batch),
                       'times': (steps, configs[0]['sample']['Nt'])}
    for name, array in first.items():
        if array.shape != expected_shapes[name] or not np.isfinite(array).all():
            raise ValueError(f'Invalid sampled {name} shape/values')
    for other in runs[1:]:
        with np.load(other/'sampling.npz', allow_pickle=False) as source:
            if set(source.files) != set(first):
                raise ValueError(f'Different sampling schema: {other}')
            for name, actual in first.items():
                if source[name].dtype != actual.dtype or not np.array_equal(source[name], actual):
                    raise ValueError(f'Different actual sampling array: {other}/{name}')
    return dict(runs=[str(run) for run in runs], methods=[method for _, method in canonical],
        seed=runtimes[0]['seed'], inputs=runtimes[0]['inputs'], reference=runtimes[0]['reference'],
        identical_initial_model=True, identical_initial_optimizer=True,
        identical_sampling_arrays=True, identical_common_sources=list(source_names),
        permitted_config_differences=['nl.method', 'loss.gradnorm_bias[5:]'],
        limitation='Local AD adds two Neumann terms and balances seven losses; nonlocal balances five.')


def run_costs(run, c):
    """Separate first compilation, refresh, held updates, sampling and evaluation."""
    timing = _json(Path(run)/'timing.json')
    updates = np.asarray(timing['update_seconds'], dtype=float)
    samples = np.asarray(timing['sample_seconds'], dtype=float)
    if (updates.shape != (c.train.iterations,) or samples.shape != updates.shape
            or not np.isfinite(updates).all() or np.any(updates <= 0)
            or not np.isfinite(samples).all() or np.any(samples < 0)):
        raise ValueError('Completed update/sample timing arrays must cover the full declared budget')
    refresh = np.arange(len(updates)) % c.train.gradnorm_update_every == 0
    warm = np.arange(len(updates)) > 0
    def group(mask):
        values = updates[mask]
        return dict(count=int(len(values)), mean_seconds=None if not len(values) else float(values.mean()),
                    median_seconds=None if not len(values) else float(np.median(values)))
    memories = _json(Path(run)/'memory.json')
    allocators = [record['allocator'] for record in memories if record['allocator'] is not None]
    peak = None if not allocators else max(item['peak_bytes_in_use'] for item in allocators)
    pools = [item['peak_pool_bytes'] for item in allocators if 'peak_pool_bytes' in item]
    return dict(first_update_compile_and_refresh_seconds=float(updates[0]),
        warm_all=group(warm), warm_held=group(warm & ~refresh), warm_refresh=group(warm & refresh),
        update_total_seconds=float(updates.sum()), sample_total_seconds=float(samples.sum()),
        evaluation_total_seconds=float(np.sum(timing['evaluation_seconds'])),
        setup_seconds=timing['setup_seconds'], total_seconds=timing['total_seconds'],
        peak_bytes_in_use=peak, peak_gib=None if peak is None else peak/2**30,
        peak_pool_bytes=None if not pools else max(pools),
        timing_scope='All updates synchronized; first includes compilation. Total includes setup, evaluation and I/O.',
        memory_scope='Cumulative JAX active peak of the whole process, including evaluation; allocator pool reported separately.')


def _history(run, steps):
    with (Path(run)/'metrics.jsonl').open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    actual = np.asarray([row['step'] for row in rows])
    if len(rows) < 2 or actual[0] != 0 or actual[-1] != steps or np.any(np.diff(actual) <= 0):
        raise ValueError('Metrics history must increase from initialization to the final fixed budget')
    return rows


def _summary(predictions, references):
    records = [metrics(prediction, ref['fields'], ref['volumes'], ref['times'])
               for prediction, ref in zip(predictions, references, strict=True)]
    if not records:
        raise ValueError('Evaluation cohort must be nonempty')
    result = {name: float(np.mean([row[name] for row in records])) for name in records[0]}
    result['conditions'] = records
    return result


def saved_predictions(run, references):
    predictions = []
    for i, ref in enumerate(references):
        with np.load(Path(run)/f'prediction_{i:03d}.npz', allow_pickle=False) as source:
            prediction = source['prediction']
            if (prediction.shape != ref['fields'].shape or not np.isfinite(prediction).all()
                    or not np.array_equal(source['points'], ref['points'])
                    or not np.array_equal(source['times'], ref['times'])):
                raise ValueError(f'Saved prediction {i} does not match this reference coordinate/time set')
            predictions.append(prediction)
    return predictions, _summary(predictions, references)


def evaluate_run(run, reference, output):
    """Explicit opt-in model inference on a common or separately reserved cohort."""
    import equinox as eqx
    import jax
    from nonlocal_pinn.training.runtime import configure_gpu_policy
    from .network import build_network
    from .evaluate import make_evaluator
    run, output = Path(run).resolve(strict=True), Path(output).resolve()
    c = ConfigDict(_json(run/'config.json'))
    runtime = _json(run/'runtime.json')
    if not (run/'model_final.eqx').is_file():
        raise ValueError('Only a completed fixed-budget final checkpoint may be evaluated')
    repo = Path(__file__).resolve().parents[2]
    for source in ('examples/particle_corrosion2d/network.py',
                   'src/nonlocal_pinn/nn/point_conditional.py', 'src/nonlocal_pinn/nn/conditional.py'):
        if not _same_file(run/'source'/source, repo/source):
            raise ValueError(f'Current inference source differs from the saved run: {source}')
    refs, manifest = load_references(reference, c)
    if manifest['backend'] != 'local_p1_lumped_fem':
        raise ValueError('This comparison requires the declared common local FEM reference')
    configure_gpu_policy(c.train.gpu_policy)
    jax.config.update('jax_enable_x64', c.train.compute_dtype == 'float64')
    jax.config.update('jax_default_matmul_precision', c.train.matmul_precision)
    net = build_network(c, jax.random.PRNGKey(runtime['seed']))
    net = eqx.tree_deserialise_leaves(run/'model_final.eqx', net)
    started = time.perf_counter()
    predictions, summary = make_evaluator(c, refs)(net)
    output.mkdir(parents=True, exist_ok=False)
    for i, (pred, ref) in enumerate(zip(predictions, refs, strict=True)):
        np.savez_compressed(output/f'prediction_{i:03d}.npz', prediction=pred,
                            points=ref['points'], times=ref['times'])
    _write(output/'metrics.json', dict(run=str(run), reference=str(Path(reference).resolve()),
        seconds=time.perf_counter()-started, metrics=summary,
        checkpoint='model_final.eqx', training_performed=False))
    render_gallery(output/'fields.png', refs, predictions, f'{c.nl.method}: final model evaluation')
    return predictions, summary


def _plot_convergence(output, records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for ax, (metric, title) in zip(axes.ravel(), (
            ('phi_pooled', 'Space-time phase'), ('phi_terminal', 'Terminal phase'),
            ('c_pooled', 'Space-time concentration'), ('corroded_terminal', 'Terminal corroded phase')), strict=True):
        for record in records:
            rows = record['history']
            ax.semilogy([row['step'] for row in rows], [100*row[metric] for row in rows],
                        label=record['method'])
        ax.set(title=title, xlabel='Training updates', ylabel='Relative L2 (%)')
        ax.grid(alpha=.25)
        ax.legend()
    fig.suptitle('Recorded training-monitor convergence')
    fig.savefig(Path(output)/'convergence.png', dpi=160)
    plt.close(fig)


def _differences(output, references, first, second, labels):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    differences = [b-a for a, b in zip(first, second, strict=True)]
    limit = max(float(np.max(np.abs(diff[..., 0]))) for diff in differences)
    # A plotting scale only; exactly identical fields remain recorded as zero.
    plot_limit = 1. if limit == 0 else limit
    columns = 5
    fig, axes = plt.subplots(len(references), columns, figsize=(15, 2.6*len(references)),
                             squeeze=False, constrained_layout=True)
    records = []
    for i, (ref, diff) in enumerate(zip(references, differences, strict=True)):
        np.savez_compressed(output/f'difference_{i:03d}.npz', difference=diff,
                            points=ref['points'], times=ref['times'])
        triangle = Triangulation(ref['points'][:, 0], ref['points'][:, 1], ref['triangles'])
        ids = np.rint(np.linspace(0, len(ref['times'])-1, columns)).astype(int)
        for ax, index in zip(axes[i], ids, strict=True):
            image = ax.tripcolor(triangle, diff[index, :, 0], shading='gouraud', cmap='RdBu_r',
                                vmin=-plot_limit, vmax=plot_limit, rasterized=True)
            if ref['fields'][index, :, 0].min() < .5 < ref['fields'][index, :, 0].max():
                ax.tricontour(triangle, ref['fields'][index, :, 0], levels=[.5], colors='black', linewidths=.7)
            ax.set(aspect='equal', title=f't*={ref["times"][index]:.2f}')
            ax.invert_yaxis()
            ax.set_facecolor('#667680')
        records.append(dict(condition=i, max_abs_phi=float(np.abs(diff[..., 0]).max()),
                            max_abs_c=float(np.abs(diff[..., 1]).max())))
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=.7, label='Phase difference')
    fig.suptitle(f'{labels[1]} minus {labels[0]} | black: reference interface')
    fig.savefig(output/'phase_difference.png', dpi=150)
    plt.close(fig)
    return records


def compare(runs, reference, output, reevaluate):
    check = validate_pair(runs)
    runs = [Path(run) for run in check['runs']]
    reference = Path(check['reference'] if reference is None else reference).resolve(strict=True)
    if not reevaluate and str(reference) != str(Path(check['reference']).resolve()):
        raise ValueError('A different reference cohort requires explicit --reevaluate')
    output = Path(output).resolve()
    # Validate completion and reference compatibility before publishing an output.
    configs, histories, costs = [], [], []
    references = None
    for run in runs:
        c = ConfigDict(_json(run/'config.json'))
        summary = _json(run/'summary.json')
        if summary['steps'] != c.train.iterations or not (run/'model_final.eqx').is_file():
            raise ValueError(f'Run has not completed its fixed budget: {run}')
        refs, manifest = load_references(reference, c)
        if manifest['backend'] != 'local_p1_lumped_fem':
            raise ValueError('Use the common independently solved local FEM reference cohort')
        references = refs
        configs.append(c)
        histories.append(_history(run, c.train.iterations))
        costs.append(run_costs(run, c))
    output.mkdir(parents=True, exist_ok=False)
    records, predictions = [], []
    for i, (run, c, history, cost) in enumerate(zip(runs, configs, histories, costs, strict=True)):
        if reevaluate:
            predicted, result = evaluate_run(run, reference, output/f'evaluation_{i:02d}')
        else:
            predicted, result = saved_predictions(run, references)
            for key, value in result.items():
                if key != 'conditions' and not np.isclose(value, history[-1][key], rtol=1e-10, atol=1e-12):
                    raise ValueError(f'Recomputed saved-field metric differs from the run record: {run}/{key}')
        predictions.append(predicted)
        records.append(dict(run=str(run), method=c.nl.method, metrics=result, costs=cost, history=history))
    _plot_convergence(output, records)
    differences = []
    for i in range(1, len(records)):
        differences.append(dict(first=records[0]['run'], second=records[i]['run'],
            conditions=_differences(output/f'field_difference_00_{i:02d}', references,
                                     predictions[0], predictions[i], (records[0]['method'], records[i]['method']))))
    result = dict(validation=check, reference=str(reference), convergence_reference=check['reference'], reevaluated=reevaluate,
                  runs=records, field_differences=differences)
    _write(output/'comparison.json', result)
    print(json.dumps(dict(output=str(output), runs=[dict(method=row['method'], metrics=row['metrics'],
        costs=row['costs']) for row in records]), allow_nan=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reevaluate', action='store_true', help='Explicitly reload networks; may use the GPU')
    args = parser.parse_args()
    compare(args.runs, args.reference, args.output, args.reevaluate)


if __name__ == '__main__':
    main()
