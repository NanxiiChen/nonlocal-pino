"""Finite serial continuation: Spinodal row costs and ceramic spatial accuracy."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from docs.tasks.pino_unified_sampling_queue import gpu

ROOT = Path(__file__).resolve().parents[2]
CERAMIC = Path('/root/autodl-tmp/data/particle_corrosion2d/20261107-spaced-six-01')
TEST64 = Path('/root/autodl-tmp/data/particle_corrosion2d/local_test64_20261111-01')
PRESETS = ('train_physical_spaced_w256', 'train_physical_spaced_w256_cdf',
           'train_physical_spaced_w256_stratified')


def read(path):
    return json.loads(Path(path).read_text())


def run(output, after):
    output.mkdir(parents=True, exist_ok=False)
    journal = output/'journal.json'
    predecessor = read(after)
    state = dict(status='waiting_for_probes', pid=os.getpid(), gpu=gpu(), after=str(after), stages=[],
                 started_utc=datetime.now(timezone.utc).isoformat())
    def save():
        temp = journal.with_suffix('.partial')
        temp.write_text(json.dumps(state, indent=2, allow_nan=False)+'\n')
        temp.replace(journal)
    save()
    environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       MPLBACKEND='Agg', PYTHONUNBUFFERED='1')
    def stage(label, command, *, cpu):
        env = dict(environment, JAX_PLATFORMS='cpu' if cpu else 'cuda')
        if cpu:
            env['CUDA_VISIBLE_DEVICES'] = ''
        else:
            card = gpu()
            if card['uuid'] != state['gpu']['uuid'] or card['used_MiB'] > 1024:
                raise RuntimeError(f'Same idle GPU required: {card}')
            env.update(XLA_PYTHON_CLIENT_PREALLOCATE='true', XLA_PYTHON_CLIENT_MEM_FRACTION='0.90')
        item = dict(label=label, command=command, platform='cpu' if cpu else 'cuda', status='running')
        state['stages'].append(item)
        tick = time.perf_counter()
        with (output/f'{label}.log').open('x') as log:
            process = subprocess.Popen([sys.executable, *command], cwd=ROOT, env=env,
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            item['pid'] = process.pid
            save()
            code = process.wait()
        item.update(status='passed' if code == 0 else 'failed', exit_code=code, seconds=time.perf_counter()-tick)
        save()
        if code:
            raise RuntimeError(f'{label} failed; retained log and partial artifacts')
    try:
        while predecessor['status'] == 'running':
            pid = predecessor['pid']
            command = Path(f'/proc/{pid}/cmdline')
            if not command.exists() or b'pino_unified_sampling_queue' not in command.read_bytes():
                predecessor = read(after)
                if predecessor['status'] == 'running':
                    raise RuntimeError('Preflight owner is absent without a terminal journal; do not restart implicitly')
                break
            time.sleep(30)
            predecessor = read(after)
        if predecessor['status'] != 'complete':
            raise RuntimeError('Predecessor resource probes did not all complete')
        state['status'] = 'running'
        save()
        inputs = output/'spinodal-inputs'
        stage('spinodal-prepare', ['-m', 'docs.tasks.pino_spinodal_row_probe', '--prepare', '--inputs', str(inputs)], cpu=True)
        for rows in (0, 256, 512):
            stage(f'spinodal-rows{rows}', ['-m', 'docs.tasks.pino_spinodal_row_probe', '--inputs', str(inputs),
                  '--rows', str(rows), '--output', str(output/f'spinodal-rows{rows}.json')], cpu=False)
        banks = [CERAMIC/'inputs64', output/'ceramic-cdf-inputs', output/'ceramic-wall-inputs']
        for label, preset, bank in zip(('cdf', 'wall'), PRESETS[1:], banks[1:], strict=True):
            stage('ceramic-prepare-'+label, ['-m', 'examples.particle_corrosion2d.prepare',
                '--config', 'examples.particle_corrosion2d.configs.'+preset, '--output', str(bank),
                '--workers', '4', '--seed', '1307500965'], cpu=True)
        stage('ceramic-input-audit', ['-m', 'docs.tasks.pino_unified_ceramic_inputs',
            '--banks', *map(str, banks), '--presets', *PRESETS, '--output', str(output/'ceramic-input-audit.json')], cpu=True)
        runs = []
        for label, preset, bank in zip(('area', 'cdf', 'wall'), PRESETS, banks, strict=True):
            destination = Path('/root/autodl-tmp/tf-logs/particle_corrosion2d')/(output.name+'-'+label)
            stage('ceramic-train-'+label, ['-m', 'examples.particle_corrosion2d.train',
                '--config', f'examples/particle_corrosion2d/configs/{preset}.py', '--inputs', str(bank),
                '--reference', str(CERAMIC/'local_dev4'), '--exclude-reference', str(TEST64),
                '--output', str(destination), '--seed', '42'], cpu=False)
            stage('ceramic-online-audit-'+label, ['-m', 'examples.particle_corrosion2d.online_validation',
                '--run', str(destination), '--workers', '4'], cpu=True)
            stage('ceramic-native-local-'+label, ['-m', 'docs.tasks.pino_geometry_sampling_results',
                'ceramic-eval', str(destination), str(CERAMIC/'native_dev4')], cpu=False)
            stage('ceramic-eval64-'+label, ['-m', 'docs.tasks.pino_unified_ceramic_evaluate',
                '--run', str(destination), '--reference', str(TEST64),
                '--output', str(destination/'evaluation64')], cpu=False)
            runs.append(destination)
        from docs.tasks.pino_geometry_sampling_results import paired
        paired(runs, 'ceramic', output/'ceramic-paired.json')
        state['status'] = 'complete'
    except Exception as error:
        state.update(status='failed', failure=str(error))
        raise
    finally:
        state['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    args = parser.parse_args()
    run(args.output, args.after)
