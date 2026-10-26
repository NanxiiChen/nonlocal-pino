"""After six controls: contemporary random-data baseline42/7 and candidate7.

Candidate: lowest terminal phi+c L2 among final5000 models with exactly one
vertical phi0.5 crossing in every column at all21 saved times. This rule is
declared before seed7 is run; no earlier checkpoint is selected.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT = Path('/root/autodl-tmp/tf-logs/electropolish2d')
BASE = Path(__file__).with_name('electropolish-optimization-controls.json')


def execute(form, mode, ramp, seed, initial, boundary):
    name = f'20261025-data-{mode}-ic{initial}-bc{boundary}-{form}-stagger0-seed{seed}'
    if ramp:
        name += '-timeramp3000-early0.05'
    run = ROOT/name
    log = Path('/tmp')/('electropolish-'+name+'.log')
    if run.exists():
        raise FileExistsError(run)
    command = [sys.executable, '-u', '-m', 'examples.corrosion2d.residual_experiment', 'train',
               '--case', 'electropolish2d', '--formulation', form, '--seed', str(seed),
               '--data-loss-mode', mode, '--initial-sampling', initial, '--boundary-sampling', boundary]
    if ramp:
        command += ['--time-ramp-steps', '3000', '--time-ramp-early', '0.05']
    env = dict(os.environ)
    env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2', XLA_PYTHON_CLIENT_PREALLOCATE='false',
               XLA_FLAGS='--xla_gpu_deterministic_ops=true',
               JAX_COMPILATION_CACHE_DIR='/tmp/nonlocal-pinn-electropolish-optimization-cache')
    print('Starting validation', name, flush=True)
    with log.open('x') as stream:
        result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if run.is_dir():
        (run/'train.log').write_bytes(log.read_bytes())
    record = {'path': str(run), 'formulation': form, 'data_loss_mode': mode, 'time_ramp': ramp,
              'seed': seed, 'initial_sampling': initial, 'boundary_sampling': boundary,
              'returncode': result.returncode, 'command': command, 'maximum_workers': 3}
    if result.returncode:
        print('FAILED', name, log.read_text(), flush=True)
    else:
        final = json.loads((run/'summary.json').read_text())['final_metrics']
        print('Finished validation', name, {k:final[k] for k in ('terminal_rel_l2_phi','terminal_rel_l2_c','terminal_rel_l2_q','terminal_rel_l2_Dq_free')}, flush=True)
    return record


def main():
    while True:
        if BASE.exists():
            records = json.loads(BASE.read_text())
            if len(records) == 6:
                break
        time.sleep(5)
    assert all(v['returncode'] == 0 for v in records)
    candidates = []
    for record in records:
        run = Path(record['path'])
        with np.load(run/'prediction.npz') as saved:
            p = saved['phi']
        assert not np.any(p == .5)
        counts = ((p[:, :-1]-.5)*(p[:, 1:]-.5) < 0).sum(axis=1)
        if np.all(counts == 1):
            metrics = json.loads((run/'summary.json').read_text())['final_metrics']
            candidates.append((metrics['terminal_rel_l2_phi']+metrics['terminal_rel_l2_c'], record))
    if not candidates:
        raise RuntimeError('No topology-valid candidate; preserve all outcomes and diagnose before seed validation')
    score, winner = min(candidates, key=lambda v:v[0])
    choice = {'selection_rule': 'lowest sum of terminal phi/c relative L2 among final5000 checkpoints with one vertical crossing in all columns/all21 times',
              'selected_record': winner, 'terminal_field_score': score, 'eligible_count': len(candidates),
              'note': 'Seed7 validation is independent; this is a declared field criterion, not proof of transport superiority'}
    Path(__file__).with_name('electropolish-optimization-selection.json').write_text(json.dumps(choice,indent=2)+'\n')
    print('Candidate selected', json.dumps(choice), flush=True)
    plan = [('phase','physical',False,42,'random_interior','random'),
            ('phase','physical',False,7,'random_interior','random'),
            (winner['formulation'],winner['data_loss_mode'],winner['time_ramp'],7,'matched_graph','matched_graph')]
    completed = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = [pool.submit(execute,*item) for item in plan]
        for job in as_completed(jobs):
            completed.append(job.result())
            Path(__file__).with_suffix('.json').write_text(json.dumps(completed,indent=2)+'\n')
    if any(v['returncode'] for v in completed):
        raise SystemExit('Validation failure; inspect records')


if __name__ == '__main__':
    main()
