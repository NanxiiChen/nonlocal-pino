"""Fixed201x101horizon-ratio extension; reuse one final model and cold train five.

Use prepare on CPU, probe on GPU, then run the finite background queue.
All numerical work reuses the established train.py and common64evaluator.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from docs.tasks.electro_horizon_study import REPO, PYTHON, command, read, train_command, write

PARENT = Path('/root/autodl-tmp/tf-logs/electropolish2d/horizon-density-cold10k-20261112-02')
BASELINE = PARENT/'runs/n201-d010'
RATIOS = [('ratio150', 1.5), ('ratio175', 1.75), ('ratio225', 2.25),
          ('ratio250', 2.5), ('ratio300', 3.)]


def fixed_particle_config(config):
    """Only horizon, complete-halo padding and output directories may differ."""
    c = copy.deepcopy(config)
    del c['nl']['delta']
    for key in ('max_nodes', 'max_edges'):
        del c['sample']['spatial'][key]
    for key in ('run_name', 'log_dir'):
        del c['log'][key]
    return c


def verify_reuse():
    parent = read(PARENT/'report.json')
    source = read(BASELINE/'config.json')
    evaluated = read(BASELINE/'evaluation64.json')
    if (parent['status'] != 'passed' or evaluated['count'] != 64 or evaluated['steps'] != 10000
            or source['sample']['Nx'] != 201 or source['sample']['Ny'] != 101
            or source['nl']['delta'] != .01 or source['train']['seed'] != 42):
        raise ValueError('Unexpected or unqualified reused fixed201x101baseline')
    for name in read(BASELINE/'source/files.json'):
        if (BASELINE/'source'/name).read_bytes() != (REPO/name).read_bytes():
            raise ValueError('Numerical source changed since reused run: '+name)
    return source


def prepare(root):
    import jax
    import jax.numpy as jnp
    from ml_collections import ConfigDict
    from examples.corrosion2d.residual_experiment import grid
    from examples.electropolish2d.horizon_study import load_monitor, operator_audit
    from nonlocal_pinn.operators import build_D
    if jax.default_backend() != 'cpu' or jax.config.x64_enabled:
        raise ValueError('Prepare with JAX_PLATFORMS=cpu JAX_ENABLE_X64=false')
    source = verify_reuse()
    root.mkdir(parents=True, exist_ok=False)
    (root/'configs').mkdir(); (root/'logs').mkdir()
    c = ConfigDict(source)
    load_monitor(c)
    baseline_audit = read(BASELINE/'operator_audit.json')
    records = [dict(name='ratio200', ratio=2., reused=True, config=str(BASELINE/'config.json'),
                    run=str(BASELINE), audit=baseline_audit)]
    _, _, points = grid(c)
    h = 1/(c.sample.Nx-1)
    for name, ratio in RATIOS:
        c = ConfigDict(copy.deepcopy(source))
        c.nl.delta = ratio*h
        c.log.log_dir, c.log.run_name = str(root/'runs'), name
        op = build_D(jnp.asarray(points, dtype=jnp.float32), c.nl.delta, method='sparse', kernel=c.nl.kernel)
        audit = operator_audit(op, points.astype(np.float32), c.nl.delta, c.sample.K_core)
        c.sample.spatial.max_nodes, c.sample.spatial.max_edges = audit['max_nodes'], audit['max_edges']
        config = json.loads(c.to_json())
        if fixed_particle_config(config) != fixed_particle_config(source):
            raise ValueError('Unexpected non-horizon change before training')
        path = root/'configs'/f'{name}.json'
        write(path, config)
        records.append(dict(name=name, ratio=ratio, reused=False, config=str(path),
                            run=str(root/'runs'/name), audit=audit))
        print(name, json.dumps(audit), flush=True)
    probe = read(root/'configs/ratio300.json')
    probe['train']['iterations'] = 50
    probe['log'].update(run_name='probe-ratio300', eval_every=50, log_every=10)
    write(root/'configs/probe.json', probe)
    parent = read(PARENT/'manifest.json')
    manifest = {k:copy.deepcopy(parent[k]) for k in ('steps', 'seed', 'allocator_fraction', 'preallocate',
        'pdf_audit_packages', 'phase_thickness', 'core_count', 'reference', 'inputs', 'count',
        'evaluation_times', 'metric', 'checkpoint')}
    manifest.update(created=datetime.now(timezone.utc).isoformat(), arms=records,
        parent=str(PARENT), reused_run=str(BASELINE), particle_count=20301, grid=[201,101], spacing=h,
        probe_config=str(root/'configs/probe.json'),
        device=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv'],text=True))
    write(root/'manifest.json', manifest)
    for name in ('docs/tasks/electro_horizon_ratio_study.py', 'docs/tasks/electro_horizon_study.py',
                 'scripts/render_electro_horizon_ratio.py', 'scripts/render_electro_horizon_study.py',
                 'scripts/render_pino_derivative_comparisons.py'):
        target = root/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO/name, target)


def queue(root, probe):
    manifest = read(root/'manifest.json')
    verify_reuse()
    if probe:
        command(root, 'probe', train_command(manifest['probe_config']))
        return
    if not (root/'runs/probe-ratio300/summary.json').is_file():
        raise ValueError('Inspect successful largest-halo probe before full runs')
    for arm in manifest['arms']:
        if arm['reused']:
            continue
        command(root, arm['name']+'-train', train_command(arm['config']))
        command(root, arm['name']+'-eval', [str(PYTHON), 'docs/tasks/electro_horizon_study.py',
                                          'evaluate', '--root', str(root), '--run', arm['run']])
    command(root, 'audit-and-plot', [str(PYTHON), 'scripts/render_electro_horizon_ratio.py', '--root', str(root)])
    write(root/'complete.json', dict(status='complete', new_runs=5, reused_runs=1,
                                    at=datetime.now(timezone.utc).isoformat()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'probe', 'run'))
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.root)
    else:
        queue(args.root, args.action == 'probe')
