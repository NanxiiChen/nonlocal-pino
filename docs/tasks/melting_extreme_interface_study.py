"""Finite owner for extreme-interface and independently separated horizon controls."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
WIDTHS = (.0075, .00375, .001875, .0009375)
SEEDS = (42, 43)


def tag(value):
    return str(value).replace('.', 'p')


def matrix():
    cases = []
    for seed in SEEDS:
        for epsilon in WIDTHS:
            for method in ('sparse', 'local'):
                cases.append(dict(group='thickness', epsilon=epsilon, seed=seed,
                    method=method, n=128, delta=.01, normalization='raw'))
        for delta in (.01, .015, .02):
            for normalization in ('raw', 'moment'):
                if delta == .01 and normalization == 'raw':
                    continue  # The identical thickness arm is the shared control.
                cases.append(dict(group='horizon', epsilon=.001875, seed=seed,
                    method='sparse', n=128, delta=delta, normalization=normalization))
        for method, normalization in (('sparse', 'raw'), ('sparse', 'moment'), ('local', 'raw')):
            cases.append(dict(group='refinement', epsilon=.001875, seed=seed,
                method=method, n=512, delta=1.27/511, normalization=normalization))
    return cases


def configuration(case):
    from examples.icemelting3d.configs.train_extreme_interface import get_config
    c = get_config()
    c.physical.eps_star = case['epsilon']
    c.train.seed = case['seed']
    c.sample.Nx = c.sample.Ny = c.sample.Nz = case['n']
    c.nl.delta = case['delta']
    c.nl.method = case['method']
    c.nl.normalization = case['normalization']
    return c


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def analyze(output):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from examples.icemelting3d.extreme_reference import weighted_metrics
    journal = read(output/'journal.json')
    results = []
    for stage in journal['stages']:
        if stage['kind'] != 'train':
            continue
        case, folder = stage['case'], Path(stage['run'])
        summary, saved = read(folder/'summary.json'), read(folder/'config.json')
        c = configuration(case)
        c.experiment.reference_path = str(Path(journal['reference_root'])/tag(case['epsilon'])/'reference.npz')
        expected = json.loads(c.to_json_best_effort())
        for key in ('physical', 'nondim', 'sample', 'nl', 'arch', 'phase', 'train', 'loss', 'experiment', 'reference'):
            if saved[key] != expected[key]:
                raise ValueError(f'Final configuration mismatch: {folder}: {key}')
        with np.load(folder/'prediction.npz') as p, np.load(c.experiment.reference_path) as r:
            check = weighted_metrics(p['phi'], r['local'], r['weights'])
        for key, value in check.items():
            np.testing.assert_allclose(value, summary['metrics'][key], rtol=1e-12, atol=1e-14)
        if summary['requested_steps'] != 10000 or (summary['status'] == 'complete' and summary['completed_steps'] != 10000):
            raise ValueError('Unqualified final training budget')
        if summary['status'] not in ('complete', 'numerical_failure'):
            raise ValueError('Unexpected training termination status')
        expected_attempts = 10000 if summary['status'] == 'complete' else summary['failure']['attempted_step']
        with np.load(folder/'sampling.npz') as data:
            if set(data.files) != {'ids', 'times', 'boundary', 'normal'}:
                raise ValueError('Sampling trace schema mismatch')
            if any(len(data[key]) != expected_attempts for key in data.files):
                raise ValueError('Sampling trace does not cover every attempted update')
        results.append(dict(case=case, run=str(folder), **summary))
    if len(results) != len(matrix()):
        raise ValueError('Every declared arm must have a completed or explicitly failed result')
    identity = lambda case: tuple(case[key] for key in ('group', 'epsilon', 'seed', 'method', 'n', 'delta', 'normalization'))
    if {identity(row['case']) for row in results} != {identity(case) for case in matrix()}:
        raise ValueError('Actual completed identities differ from the predeclared experiment matrix')
    pairing = []
    for epsilon in WIDTHS:
        for seed in SEEDS:
            for n in (128, 512):
                group = [r for r in results if r['case']['epsilon'] == epsilon
                         and r['case']['seed'] == seed and r['case']['n'] == n]
                if not group:
                    continue
                first = Path(group[0]['run'])
                with np.load(first/'sampling.npz') as data:
                    original = {k: data[k] for k in data.files}
                for row in group[1:]:
                    folder = Path(row['run'])
                    for name in ('model_initial.eqx', 'optimizer_initial.eqx'):
                        if (first/name).read_bytes() != (folder/name).read_bytes():
                            raise ValueError(f'Cold-state pairing failed: {folder}/{name}')
                    with np.load(folder/'sampling.npz') as data:
                        count = min(len(original['ids']), len(data['ids']))
                        for key in original:
                            np.testing.assert_array_equal(original[key][:count], data[key][:count])
                    pairing.append(dict(a=str(first), b=str(folder), common_attempts=count,
                        full_budget_pair=count == 10000))
    report = dict(results=results, pairing=pairing, cold_model_optimizer_pairing=True,
        actual_common_prefix_pairing=True, reference='Qualified radial local far-field surrogate with common cube cubature',
        qualification='No reference labels in training; fixed-grid extremes can be spatially underresolved; no new native reference')
    write(output/'results.json', report)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, key, title in zip(axes, ('pooled', 'interface', 'radius_relative_rmse'),
                            ('Whole-field L2', 'Interface-band L2', 'Interface radius RMSE'), strict=True):
        for method, label, color in [('sparse', 'Nonlocal, original', '#246b98'), ('local', 'Local AD', '#cb673e')]:
            values = np.array([[r['metrics'][key] if r['status'] == 'complete' else np.nan
                for r in results if r['case']['group'] == 'thickness'
                and r['epsilon'] == epsilon and r['method'] == method] for epsilon in WIDTHS], dtype=float)*100
            ax.plot(WIDTHS, np.mean(values, axis=1), 'o-', color=color, label=label)
            ax.fill_between(WIDTHS, values.min(axis=1), values.max(axis=1), color=color, alpha=.15)
        ax.set(xscale='log', yscale='log', title=title, xlabel='epsilon (thinner →)', ylabel='Error (%)')
        ax.invert_xaxis(); ax.grid(alpha=.2); ax.legend(frameon=False, fontsize=8)
    fig.suptitle('Fixed 128-cube grid · delta = 0.01 · 10,000 updates · two paired seeds')
    fig.savefig(output/'extreme-error-vs-thickness.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, epsilon in zip(axes.flat, WIDTHS, strict=True):
        for r in results:
            if r['case']['group'] != 'thickness' or r['epsilon'] != epsilon:
                continue
            path = Path(r['run'])/'metrics.jsonl'
            trace = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
            ax.plot([v['step'] for v in trace], [100*v['pooled'] for v in trace], marker='.',
                color='#246b98' if r['method'] == 'sparse' else '#cb673e',
                linestyle='-' if r['seed'] == SEEDS[0] else '--', label=f'{r["method"]}, seed {r["seed"]}')
        ax.set(yscale='log', title=f'epsilon = {epsilon}', xlabel='Training updates', ylabel='Whole-field L2 (%)')
        ax.grid(alpha=.2); ax.legend(frameon=False, fontsize=8)
    fig.savefig(output/'extreme-convergence.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, metric in zip(axes, ('pooled', 'interface'), strict=True):
        for norm, label, color in [('raw', 'Original weights', '#246b98'), ('moment', 'Unit second moment', '#52946c')]:
            values = np.array([[r['metrics'][metric] if r['status'] == 'complete' else np.nan for r in results
                if r['epsilon'] == .001875 and r['method'] == 'sparse'
                and r['case']['n'] == 128 and r['case']['delta'] == delta
                and r['case']['normalization'] == norm] for delta in (.01, .015, .02)])*100
            ax.plot((.01, .015, .02), values.mean(axis=1), 'o-', color=color, label=label)
            ax.fill_between((.01, .015, .02), values.min(axis=1), values.max(axis=1), color=color, alpha=.15)
        local = [r['metrics'][metric]*100 for r in results if r['epsilon'] == .001875
                 and r['method'] == 'local' and r['case']['n'] == 128 and r['status'] == 'complete']
        if local:
            ax.axhline(np.mean(local), color='#cb673e', linestyle='--', label='Local AD')
        ax.set(xlabel='Horizon delta', ylabel='Relative L2 (%)', yscale='log', title=metric)
        ax.grid(alpha=.2); ax.legend(frameon=False, fontsize=8)
    fig.suptitle('Fixed grid: horizon and second-moment controls · epsilon = 0.001875')
    fig.savefig(output/'horizon-comparison.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    controls = [(n, method, norm) for n in (128, 512)
                for method, norm in (('sparse', 'raw'), ('sparse', 'moment'), ('local', 'raw'))]
    labels = [f'{n}³\n{method}\n{norm if method == "sparse" else "AD"}' for n, method, norm in controls]
    for ax, metric in zip(axes, ('pooled', 'interface', 'step_ms'), strict=True):
        values = []
        for n, method, norm in controls:
            selected = [r for r in results if r['epsilon'] == .001875 and r['method'] == method
                and r['case']['n'] == n and r['case']['normalization'] == norm
                and (n == 512 or r['case']['delta'] == .01)]
            if len(selected) != 2:
                raise ValueError('Refinement comparison requires both paired seeds')
            values.append([np.nan if r['status'] != 'complete' else
                1000*r['median_step_seconds'] if metric == 'step_ms' else 100*r['metrics'][metric]
                for r in selected])
        values = np.asarray(values)
        mid = values.mean(axis=1)
        ax.bar(range(len(controls)), mid, color=['#246b98', '#52946c', '#cb673e']*2,
               yerr=np.stack((mid-values.min(axis=1), values.max(axis=1)-mid)), capsize=3)
        ax.set(yscale='log', title=metric, ylabel='Milliseconds' if metric == 'step_ms' else 'Relative L2 (%)')
        ax.set_xticks(range(len(controls)), labels, fontsize=8); ax.grid(axis='y', alpha=.2)
    fig.suptitle('Joint point-spacing/horizon refinement · epsilon = 0.001875')
    fig.savefig(output/'refinement-comparison.png', dpi=180); plt.close(fig)
    print('ANALYZED', output, 'runs', len(results), flush=True)


def dense(output):
    import numpy as np
    import equinox as eqx
    import jax
    from ml_collections import ConfigDict
    from examples.icemelting3d.extreme_reference import RadialSolution, evaluation_points, sample_solution, weighted_metrics
    from examples.icemelting3d.extreme_interface import predictor
    from examples.icemelting3d.thin_interface import build_network
    from nonlocal_pinn.training.runtime import configure_gpu_policy
    results = read(output/'results.json')['results']
    selected = [r for r in results if r['case']['group'] == 'thickness' and r['epsilon'] == min(WIDTHS)]
    first = ConfigDict(read(Path(selected[0]['run'])/'config.json'))
    configure_gpu_policy(first.train.gpu_policy)
    jax.config.update('jax_default_matmul_precision', first.train.matmul_precision)
    finer = first.copy_and_resolve_references()
    finer.experiment.quadrature_directions = 512
    finer.experiment.quadrature_radial = 8192
    points, weights, _, _ = evaluation_points(finer)
    profile = Path(first.experiment.reference_path).parent/'radial-profile.npz'
    with np.load(profile) as data:
        solution = RadialSolution(data['radius'], data['time'], data['latent'], {'source': str(profile)})
    truth = sample_solution(solution, points)
    receipt = dict(status='running', quadrature_directions=512, quadrature_radial=8192,
                   volume=float(weights.sum()), reference_profile=str(profile), results=[])
    write(output/'dense-evaluation.json', receipt)
    for row in selected:
        folder = Path(row['run'])
        c = ConfigDict(read(folder/'config.json'))
        net = eqx.tree_deserialise_leaves(folder/'model_final.eqx', build_network(c))
        evaluate = predictor(c, solution.time)
        with np.load(c.experiment.reference_path) as r, np.load(folder/'prediction.npz') as p:
            replay = evaluate(net, r['points'])
            max_abs = float(np.max(np.abs(replay-p['phi'])))
            initial_max_abs = float(np.max(np.abs(replay[0]-p['phi'][0])))
            positive_max_abs = float(np.max(np.abs(replay[1:]-p['phi'][1:])))
            replay_relative_l2 = float(np.sqrt(np.sum(r['weights']*(replay[1:]-p['phi'][1:])**2)
                / np.sum(r['weights']*p['phi'][1:]**2)))
            atol = 1e-5*(.0075/c.physical.eps_star)
            np.testing.assert_allclose(replay, p['phi'], rtol=0., atol=atol)
        del replay
        metrics = weighted_metrics(evaluate(net, points), truth, weights)
        receipt['results'].append(dict(run=str(folder), trained_steps=row['completed_steps'],
            status=row['status'], seed=row['seed'], method=row['method'], metrics=metrics,
            replay_max_abs=max_abs, replay_absolute_tolerance=atol,
            replay_initial_max_abs=initial_max_abs, replay_positive_time_max_abs=positive_max_abs,
            replay_relative_l2=replay_relative_l2,
            change_percentage_points={key: 100*(metrics[key]-row['metrics'][key]) for key in ('pooled', 'terminal', 'interface')},
            max_absolute_change=metrics['max_absolute']-row['metrics']['max_absolute']))
        write(output/'dense-evaluation.json', receipt)
    receipt['status'] = 'complete'
    write(output/'dense-evaluation.json', receipt)


def queue(output, reference_root, cube_check):
    bank = read(reference_root/'bank-validation.json')
    cube = read(cube_check)
    if (bank['status'] != 'passed' or not bank['all_finite'] or not bank['source_bytes_match']
            or sorted(row['epsilon'] for row in bank['rows']) != sorted(WIDTHS)):
        raise ValueError('Qualified complete reference bank is required before training')
    for source in (reference_root/'source').rglob('*'):
        if source.is_file() and source.read_bytes() != (REPO/source.relative_to(reference_root/'source')).read_bytes():
            raise ValueError(f'Reference source changed after validation: {source}')
    if (cube['status'] != 'passed' or cube['epsilon'] != .00375
            or cube['grids'] != [512, 768] or cube['stage'] != 'complete'):
        raise ValueError('Qualified cube/radial crosscheck is required before training')
    checked_config = read(cube_check.parent/'config.json')
    from examples.icemelting3d.configs.train_extreme_interface import get_config
    expected_config = json.loads(get_config().to_json_best_effort())
    for key in ('physical', 'nondim', 'reference'):
        if checked_config[key] != expected_config[key]:
            raise ValueError(f'Cube crosscheck {key} does not match the declared study')
    output.mkdir(parents=True, exist_ok=False)
    files = ['examples/icemelting3d/'+name for name in ('train.py', 'phase.py', 'losses.py',
        'thin_interface.py', 'thin_interface_reference.py', 'thin_interface_operators.py',
        'extreme_interface.py', 'extreme_reference.py', 'configs/train_base.py',
        'configs/train_thin_interface.py', 'configs/train_extreme_interface.py')]
    files += ['docs/tasks/melting_extreme_interface_study.py', 'docs/tasks/melting_extreme_cube_check.py']
    for name in files:
        target = output/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO/name, target)
    state = dict(status='running', pid=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat(),
        reference_root=str(reference_root), cube_check=str(cube_check), reference_bank_validation=bank,
        steps=10000, seeds=list(SEEDS), widths=list(WIDTHS), plan=matrix(), stages=[])
    write(output/'journal.json', state)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
        JAX_PLATFORMS='cuda', JAX_ENABLE_X64='False', XLA_PYTHON_CLIENT_MEM_FRACTION='.55')
    def run(command, item):
        state['stages'].append(item)
        item['command'] = command
        start = time.perf_counter()
        with Path(item['log']).open('x') as log:
            child = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
            item.update(status='running', pid=child.pid); write(output/'journal.json', state)
            code = child.wait()
        item.update(status='passed' if code == 0 else 'failed', exit_code=code, seconds=time.perf_counter()-start)
        write(output/'journal.json', state)
        if code:
            raise RuntimeError(f'Child {item["kind"]} failed; inspect {item["log"]}')
    try:
        for i, case in enumerate(matrix()):
            name = f'{output.name}-{i:02d}-e{tag(case["epsilon"])}-{case["method"]}-{case["normalization"]}-n{case["n"]}-s{case["seed"]}'
            folder = Path('/root/autodl-tmp/tf-logs/icemelting3d')/name
            command = [sys.executable, '-m', 'examples.icemelting3d.train',
                '--config=examples/icemelting3d/configs/train_extreme_interface.py',
                f'--seed={case["seed"]}', f'--config.physical.eps_star={case["epsilon"]}',
                f'--config.nl.method={case["method"]}', f'--config.nl.delta={case["delta"]}',
                f'--config.nl.normalization={case["normalization"]}',
                *[f'--config.sample.N{axis}={case["n"]}' for axis in ('x', 'y', 'z')],
                f'--config.log.run_name={name}',
                f'--config.experiment.reference_path={reference_root/tag(case["epsilon"])/"reference.npz"}']
            item = dict(kind='train', case=case, run=str(folder), log=str(output/f'{name}.log'))
            run(command, item)
            summary = read(folder/'summary.json')
            item.update(training_status=summary['status'], completed_steps=summary['completed_steps'])
            write(output/'journal.json', state)
        analyze(output)
        run([sys.executable, '-m', 'docs.tasks.melting_extreme_interface_study', 'dense', '--output', str(output)],
            dict(kind='dense', log=str(output/'dense.log')))
        state['status'] = 'complete'
    except Exception as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        state['finished_utc'] = datetime.now(timezone.utc).isoformat()
        write(output/'journal.json', state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('queue', 'analyze', 'dense'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--reference-root', type=Path)
    parser.add_argument('--cube-check', type=Path)
    args = parser.parse_args()
    if args.action == 'queue':
        if args.reference_root is None or args.cube_check is None:
            parser.error('queue requires --reference-root and --cube-check')
        queue(args.output, args.reference_root, args.cube_check)
    elif args.action == 'dense':
        dense(args.output)
    else:
        analyze(args.output)
