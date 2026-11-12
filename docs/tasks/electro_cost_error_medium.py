"""Complete two predeclared middle-grid controls for a three-curve plot."""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from docs.tasks.electro_horizon_study import REPO, PYTHON, command, read, train_command, write
from docs.tasks.electro_horizon_ratio_study import fixed_particle_config

PARENT = Path('/root/autodl-tmp/tf-logs/electropolish2d/horizon-density-cold10k-20261112-02')
BASELINE = PARENT / 'runs/n151-ratio2'
ARMS = [('n151-ratio150', 1.5), ('n151-ratio250', 2.5)]


def verify_source():
    config = read(BASELINE / 'config.json')
    if (config['sample']['Nx'], config['sample']['Ny'], config['train']['seed'],
            config['train']['iterations']) != (151, 76, 42, 10000):
        raise ValueError('Unexpected frozen middle-grid baseline')
    for name in read(BASELINE / 'source/files.json'):
        if (BASELINE / 'source' / name).read_bytes() != (REPO / name).read_bytes():
            raise ValueError('Source changed since baseline: ' + name)
    return config


def prepare(root):
    import jax
    import jax.numpy as jnp
    from ml_collections import ConfigDict
    from examples.corrosion2d.residual_experiment import grid
    from examples.electropolish2d.horizon_study import load_monitor, operator_audit
    from nonlocal_pinn.operators import build_D
    if jax.default_backend() != 'cpu' or jax.config.x64_enabled:
        raise ValueError('Prepare on CPU in FP32')
    baseline = verify_source()
    root.mkdir(parents=True, exist_ok=False)
    (root / 'configs').mkdir(); (root / 'logs').mkdir()
    records = []
    larger = read(PARENT / 'runs/n151-d020/operator_audit.json')
    for name, ratio in ARMS:
        c = ConfigDict(copy.deepcopy(baseline))
        c.nl.delta = ratio / 150
        c.log.log_dir, c.log.run_name = str(root / 'runs'), name
        load_monitor(c)
        _, _, points = grid(c)
        op = build_D(jnp.asarray(points, dtype=jnp.float32), c.nl.delta,
                     method='sparse', kernel=c.nl.kernel)
        audit = operator_audit(op, points.astype(np.float32), c.nl.delta, c.sample.K_core)
        c.sample.spatial.max_nodes, c.sample.spatial.max_edges = audit['max_nodes'], audit['max_edges']
        config = json.loads(c.to_json())
        if fixed_particle_config(config) != fixed_particle_config(baseline):
            raise ValueError('Undeclared scheme change')
        if not all(audit[k] <= larger[k] for k in ('max_nodes','max_edges')):
            raise ValueError('Halo exceeds the completed same-grid ratio3 control')
        path = root / 'configs' / (name + '.json')
        write(path, config)
        records.append(dict(name=name, config=str(path), run=str(root / 'runs' / name), audit=audit))
        print(name, audit, flush=True)
    parent = read(PARENT / 'manifest.json')
    manifest = {k:copy.deepcopy(parent[k]) for k in ('steps','seed','allocator_fraction','preallocate',
        'pdf_audit_packages','phase_thickness','core_count','reference','inputs','count',
        'evaluation_times','metric','checkpoint')}
    device = subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version',
                                      '--format=csv'], text=True)
    if device != parent['device']:
        raise ValueError('GPU model/memory/driver differs from the comparison timing records')
    manifest.update(created=datetime.now(timezone.utc).isoformat(), baseline=str(BASELINE),
        arms=records, device=device, selection='Predeclared151x76 ratios1.5and2.5; reuse2and3',
        memory_check='Both halos smaller than completed same-grid ratio3; serial runs')
    write(root / 'manifest.json', manifest)
    target = root / 'source/docs/tasks/electro_cost_error_medium.py'
    target.parent.mkdir(parents=True); shutil.copyfile(Path(__file__), target)


def audit(root):
    from scripts.render_electro_horizon_study import collect
    manifest, records = collect(root, fixed_particle_config)
    baseline_config = fixed_particle_config(verify_source())
    for arm in manifest['arms']:
        run = Path(arm['run'])
        if fixed_particle_config(read(run / 'config.json')) != baseline_config:
            raise ValueError('Final config differs from baseline: ' + arm['name'])
        for name in ('model_initial.eqx','optimizer_initial.eqx'):
            if (run / name).read_bytes() != (BASELINE / name).read_bytes():
                raise ValueError('Cold state mismatch: ' + name)
        for name in ('online_coefficients.npy','online_accepted.npy'):
            np.testing.assert_array_equal(np.load(run / name), np.load(BASELINE / name))
        fields = ('times','draw_ids','entry_draw_ids','core_ids','core_mask','ic_ids','dirichlet_ids','side_ids')
        with np.load(run / 'sampling.npz') as a, np.load(BASELINE / 'sampling.npz') as b:
            for key in fields:
                np.testing.assert_array_equal(a[key], b[key], err_msg=arm['name']+':'+key)
    write(root / 'report.json', dict(status='passed', manifest=str(root / 'manifest.json'),
        records=records, paired_baseline=str(BASELINE), cold_state='exact for both arms',
        sampling='All10000time/core/IC/BC/draw identities and condition acceptances match baseline',
        norm_replay='All128sample channel norms independently replayed from saved fields',
        caveat='Halo shapes/FP32 operations differ; one paired seed'))


def run(root):
    verify_source()
    for arm in read(root / 'manifest.json')['arms']:
        command(root, arm['name'] + '-train', train_command(arm['config']))
        command(root, arm['name'] + '-eval', [str(PYTHON), 'docs/tasks/electro_horizon_study.py',
                                              'evaluate', '--root', str(root), '--run', arm['run']])
    command(root, 'audit', [str(PYTHON), str(Path(__file__).resolve()), 'audit', '--root', str(root)])
    write(root / 'complete.json', dict(status='complete', at=datetime.now(timezone.utc).isoformat()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare','run','audit'))
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    {'prepare':prepare,'run':run,'audit':audit}[args.action](args.root)
