"""Finite serial owner for reference validation, matched training and curves."""
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
WIDTHS = (.0225, .015, .010, .0075)


def config(epsilon):
    from examples.icemelting3d.configs.train_thin_interface import get_config
    c = get_config()
    c.physical.eps_star = epsilon
    return c


def tag(epsilon):
    return str(epsilon).replace('.', 'p')


def reference(epsilon, output):
    import numpy as np
    from examples.icemelting3d.thin_interface_reference import prepare, sample_reference
    c = config(epsilon)
    prepare(c, output)
    if epsilon == min(WIDTHS):
        half, seconds = sample_reference(c, c.experiment.reference_grids[0],
                                        c.experiment.reference_dt/2, local=True)
        with np.load(output/'reference.npz') as f:
            coarse = f['local_coarse']
            error = np.linalg.norm(half[:, :coarse.shape[1]]-coarse)/np.linalg.norm(coarse)
        if not np.isfinite(error) or error > 1e-5:
            raise ValueError(f'Thin local reference time-refinement failure: {error}')
        p = output/'validation.json'
        report = json.loads(p.read_text())
        report.update(thinnest_time_relative_l2=float(error), extra_time_check_seconds=seconds)
        p.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print('TIME_CHECK', error, flush=True)


def analyze(output):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from examples.icemelting3d.thin_interface import field_metrics
    journal = json.loads((output/'journal.json').read_text())
    runs = [s for s in journal['stages'] if s['kind'] == 'train']
    results = []
    for run in runs:
        folder = Path(run['run'])
        summary = json.loads((folder/'summary.json').read_text())
        refpath = Path(json.loads((folder/'config.json').read_text())['experiment']['reference_path'])
        with np.load(folder/'prediction.npz') as f, np.load(refpath) as r:
            check = field_metrics(f['phi'], r['local'])
            for key, value in check.items():
                np.testing.assert_allclose(value, summary['metrics'][key], rtol=1e-12, atol=1e-14)
            summary['native_comparison'] = field_metrics(f['phi'], r['native'])
            summary['native_baseline'] = field_metrics(r['native'], r['local'])
            if 'native_floor' in summary:
                del summary['native_floor']
            summary['local_space_positive_time'] = field_metrics(r['local_coarse'], r['local'])
        results.append(dict(run=str(folder), **summary))
    for eps in WIDTHS:
        for seed in journal['seeds']:
            pair = [r for r in results if r['epsilon'] == eps and r['seed'] == seed]
            if len(pair) != 2:
                raise ValueError('Every epsilon/seed requires both completed derivative methods')
            a, b = [Path(r['run']) for r in pair]
            for name in ('model_initial.eqx', 'optimizer_initial.eqx'):
                if (a/name).read_bytes() != (b/name).read_bytes():
                    raise ValueError(f'Cold-state pairing failed: {name}')
            with np.load(a/'sampling.npz') as x, np.load(b/'sampling.npz') as y:
                for name in x.files:
                    np.testing.assert_array_equal(x[name], y[name])
    report = dict(results=results, actual_input_pairing=True, cold_model_optimizer_pairing=True,
        reference='Independent local FD512 cube reference; positive-time diagnostics',
        qualifications=f'Finite {journal["steps"]}-update budget; existing IC hard constraint and latent preconditioning; no seed selection')
    (output/'results.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})

    def curve(axes, specs):
        for ax, (metric, title, ylabel) in zip(axes, specs, strict=True):
            for method, label, color in [('sparse', 'Nonlocal', '#246b98'), ('local', 'Local AD', '#cb673e')]:
                values = np.array([[r['metrics'][metric] if metric != 'time' else r['median_step_seconds']
                    for r in results if r['method'] == method and r['epsilon'] == eps] for eps in WIDTHS], dtype=float)
                scale = 1000 if metric == 'time' else 100
                mid, lo, hi = np.median(values, axis=1)*scale, values.min(axis=1)*scale, values.max(axis=1)*scale
                ax.plot(WIDTHS, mid, 'o-', color=color, label=label, lw=2, markersize=5)
                ax.fill_between(WIDTHS, lo, hi, color=color, alpha=.12)
            ax.set(xlabel=r'Interface parameter $\epsilon^*$ (thinner →)', ylabel=ylabel, title=title)
            ax.invert_xaxis(); ax.grid(alpha=.18); ax.set_yscale('log'); ax.legend(frameon=False)
            ax.set_xticks(WIDTHS, [str(eps) for eps in WIDTHS])
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8), constrained_layout=True)
    curve(axes, [('pooled', 'Whole-field relative L2', 'Relative L2 (%)'),
        ('terminal', 'Final-time relative L2', 'Relative L2 (%)'),
        ('interface', r'Interface band: $|\phi_{ref}|<0.9$', 'Relative L2 (%)')])
    fig.suptitle(f'3D melting · SOAP {journal["steps"]:,} cold updates · {len(journal["seeds"])} paired seed(s)', fontsize=12)
    fig.savefig(output/'error-vs-thickness.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
    curve(axes, [('radius_relative_rmse', 'Zero-interface radius RMSE', 'Relative RMSE (%)'),
                 ('time', 'Warm update + sampling', 'Milliseconds per update')])
    fig.savefig(output/'position-and-time.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, epsilon in zip(axes.flat, WIDTHS, strict=True):
        for method, label, color in [('sparse', 'Nonlocal', '#246b98'), ('local', 'Local AD', '#cb673e')]:
            for seed in journal['seeds']:
                run = next(r for r in results if r['epsilon'] == epsilon
                           and r['method'] == method and r['seed'] == seed)
                trace = [json.loads(line) for line in (Path(run['run'])/'metrics.jsonl').read_text().splitlines()]
                if trace[-1]['step'] != journal['steps']:
                    raise ValueError('Convergence trace must include the final budget')
                ax.plot([r['step'] for r in trace], [100*r['pooled'] for r in trace],
                        marker='.', color=color, alpha=.8,
                        linestyle='-' if seed == journal['seeds'][0] else '--',
                        label=f'{label}, seed {seed}')
        ax.set(title=f'epsilon = {epsilon}', xlabel='Training updates',
               ylabel='Whole-field relative L2 (%)', yscale='log')
        ax.grid(alpha=.18); ax.legend(frameon=False, fontsize=8)
    fig.suptitle('Matched cold starts · common local FD512 reference', fontsize=12)
    fig.savefig(output/'convergence.png', dpi=180); plt.close(fig)
    print('ANALYSIS', json.dumps([dict(epsilon=r['epsilon'], seed=r['seed'], method=r['method'], metrics=r['metrics'], median_step_seconds=r['median_step_seconds']) for r in results]), flush=True)


def queue(output, seeds, steps, reference_root, lr_decay_every):
    if steps <= 0 or lr_decay_every <= 0 or len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds):
        raise ValueError('Require positive budget/decay period and distinct nonnegative runtime seeds')
    output.mkdir(parents=True, exist_ok=False)
    source = output/'source'
    for name in ('train.py', 'losses.py', 'phase.py', 'thin_interface.py', 'thin_interface_reference.py',
                 'configs/train_base.py', 'configs/train_thin_interface.py'):
        p = source/name
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO/'examples/icemelting3d'/name, p)
    for name in ('melting_thin_interface_study.py', 'melting_thin_interface_finish.py'):
        shutil.copy2(REPO/'docs/tasks'/name, source/name)
    state = dict(status='running', pid=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat(),
                 seeds=seeds, steps=steps, lr_decay_every=lr_decay_every, widths=WIDTHS, stages=[])
    refs = output/'references' if reference_root is None else reference_root
    state['reference_root'] = str(refs)
    def save():
        p = output/'journal.partial'
        p.write_text(json.dumps(state, indent=2, allow_nan=False)+'\n'); p.replace(output/'journal.json')
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', JAX_PLATFORMS='cuda',
               XLA_PYTHON_CLIENT_MEM_FRACTION='.55')
    def run(command, item, *, x64):
        item['command'] = command
        state['stages'].append(item); save()
        start = time.perf_counter()
        with Path(item['log']).open('x') as stream:
            child = subprocess.Popen(command, cwd=REPO, env=dict(env, JAX_ENABLE_X64=str(x64)), stdout=stream, stderr=subprocess.STDOUT)
            item.update(pid=child.pid, status='running'); save()
            code = child.wait()
        item.update(exit_code=code, status='passed' if code == 0 else 'failed', seconds=time.perf_counter()-start)
        save()
        if code:
            raise RuntimeError(f'{item["kind"]} exited {code}: {item["log"]}')
    save()
    try:
        for epsilon in WIDTHS:
            path = refs/tag(epsilon)
            if reference_root is not None:
                saved = json.loads((path/'config.json').read_text())
                expected = json.loads(config(epsilon).to_json_best_effort())
                for key in ('physical', 'nondim', 'sample', 'nl', 'experiment'):
                    if saved[key] != expected[key]:
                        raise ValueError(f'Reused reference {epsilon} {key} mismatch')
                report = json.loads((path/'validation.json').read_text())
                if (report['representation'] != 'latent' or report['local_space_relative_l2'] > .003
                        or report['local_space_interface_relative_l2'] > .03
                        or (epsilon == min(WIDTHS) and report['thinnest_time_relative_l2'] > 1e-5)
                        or not (path/'reference.npz').is_file()):
                    raise ValueError('Reused reference must pass the original validation gates')
                state['stages'].append(dict(kind='reference', epsilon=epsilon, path=str(path),
                                            status='reused', validation=report))
                save()
                continue
            run([sys.executable, '-m', 'docs.tasks.melting_thin_interface_study', 'reference',
                 '--epsilon', str(epsilon), '--output', str(path)],
                dict(kind='reference', epsilon=epsilon, path=str(path), log=str(output/f'reference-{tag(epsilon)}.log')), x64=True)
        for seed in seeds:
            for epsilon in WIDTHS:
                for method in ('sparse', 'local'):
                    name = f'{output.name}-e{tag(epsilon)}-s{seed}-{method}'
                    path = Path('/root/autodl-tmp/tf-logs/icemelting3d')/name
                    run([sys.executable, '-m', 'examples.icemelting3d.train',
                         '--config=examples/icemelting3d/configs/train_thin_interface.py',
                         f'--seed={seed}', f'--config.physical.eps_star={epsilon}', f'--config.nl.method={method}',
                         f'--config.train.iterations={steps}', f'--config.log.run_name={name}',
                         f'--config.train.lr_decay_every={lr_decay_every}',
                         f'--config.experiment.reference_path={refs/tag(epsilon)/"reference.npz"}'],
                        dict(kind='train', seed=seed, epsilon=epsilon, method=method, run=str(path),
                             log=str(output/f'{name}.log')), x64=False)
        analyze(output)
        state['status'] = 'complete'
    except Exception as error:
        state.update(status='failed', failure=str(error)); raise
    finally:
        state['finished_utc'] = datetime.now(timezone.utc).isoformat(); save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('queue', 'reference', 'analyze'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--steps', type=int)
    parser.add_argument('--lr-decay-every', type=int)
    parser.add_argument('--epsilon', type=float)
    parser.add_argument('--reference-root', type=Path)
    args = parser.parse_args()
    if args.action == 'queue':
        if not args.seeds or args.steps is None or args.lr_decay_every is None:
            parser.error('queue requires --seeds, --steps and --lr-decay-every')
        queue(args.output, args.seeds, args.steps, args.reference_root, args.lr_decay_every)
    elif args.action == 'reference':
        if args.epsilon is None:
            parser.error('reference requires --epsilon')
        reference(args.epsilon, args.output)
    else:
        analyze(args.output)
