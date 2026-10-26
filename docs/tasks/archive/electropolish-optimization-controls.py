"""Six predefined data-objective/time-window cold starts, max3 GPU workers."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/root/autodl-tmp/tf-logs/electropolish2d')
PLAN = [(form, mode, ramp) for ramp in (False, True)
        for form, mode in (('phase', 'physical'), ('phase', 's_bc'), ('latent', 's_bc'))]


def execute(form, mode, ramp):
    name = f'20261025-data-{mode}-icmatched_graph-bcmatched_graph-{form}-stagger0-seed42'
    if ramp:
        name += '-timeramp3000-early0.05'
    run = ROOT / name
    log = Path('/tmp') / ('electropolish-' + name + '.log')
    if run.exists():
        raise FileExistsError(run)
    command = [sys.executable, '-u', '-m', 'examples.corrosion2d.residual_experiment', 'train',
               '--case', 'electropolish2d', '--formulation', form, '--seed', '42',
               '--data-loss-mode', mode, '--initial-sampling', 'matched_graph',
               '--boundary-sampling', 'matched_graph']
    if ramp:
        command += ['--time-ramp-steps', '3000', '--time-ramp-early', '0.05']
    env = dict(os.environ)
    env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2', XLA_PYTHON_CLIENT_PREALLOCATE='false',
               XLA_FLAGS='--xla_gpu_deterministic_ops=true',
               JAX_COMPILATION_CACHE_DIR='/tmp/nonlocal-pinn-electropolish-optimization-cache')
    print('Starting', name, flush=True)
    with log.open('x') as stream:
        result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if run.is_dir():
        (run / 'train.log').write_bytes(log.read_bytes())
    record = {'path': str(run), 'formulation': form, 'data_loss_mode': mode, 'time_ramp': ramp,
              'seed': 42, 'command': command, 'xla_flags': env['XLA_FLAGS'],
              'returncode': result.returncode, 'maximum_workers': 3}
    if result.returncode:
        print('FAILED', name, log.read_text(), flush=True)
    else:
        final = json.loads((run / 'summary.json').read_text())['final_metrics']
        print('Finished', name, {k: final[k] for k in ('terminal_rel_l2_phi', 'terminal_rel_l2_c', 'terminal_rel_l2_q', 'terminal_rel_l2_Dq_free')}, flush=True)
    return record


def main():
    records = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = [pool.submit(execute, *item) for item in PLAN]
        for job in as_completed(jobs):
            records.append(job.result())
            Path(__file__).with_suffix('.json').write_text(json.dumps(records, indent=2) + '\n')
    if any(r['returncode'] for r in records):
        raise SystemExit('Failed control; inspect records/logs')


if __name__ == '__main__':
    main()
