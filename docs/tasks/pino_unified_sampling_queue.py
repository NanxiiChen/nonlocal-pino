"""Finite serial GPU preflights for the unified rolling-sampling study."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
LOGS = Path('/root/autodl-tmp/tf-logs/channel_nsch2d')


def gpu():
    text = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,memory.total,memory.used',
                                   '--format=csv,noheader,nounits'], text=True).strip()
    uuid, name, total, used = (s.strip() for s in text.split(','))
    return dict(uuid=uuid, name=name, total_MiB=int(total), used_MiB=int(used))


def run_probes(output):
    output.mkdir(parents=True, exist_ok=True)
    journal = output/'probe-journal.json'
    if journal.exists():
        raise FileExistsError(journal)
    support = json.loads((output/'support-audit.json').read_text())
    if support['status'] != 'passed':
        raise ValueError('Support audit must pass before training probes')
    state = dict(status='running', pid=os.getpid(), gpu=gpu(), stages=[],
                 started_utc=datetime.now(timezone.utc).isoformat())
    def save():
        temporary = journal.with_suffix('.partial')
        temporary.write_text(json.dumps(state, indent=2, allow_nan=False)+'\n')
        temporary.replace(journal)
    save()
    arms = [('full', 'train_spatial_control', []),
            ('tiles512', 'train_spatial_control', ['--config.sample.spatial.mode=tiles',
                                                '--config.sample.spatial.node_capacity=8192']),
            ('strat4_bulk512', 'train_stratified_spatial', ['--config.sample.spatial.wall_groups=0',
                                                         '--config.sample.spatial.node_capacity=10240']),
            ('strat4_wall512', 'train_stratified_spatial', ['--config.sample.spatial.node_capacity=10240']),
            ('strat4_wall1024', 'train_stratified_spatial', ['--config.sample.spatial.core_count=1024',
                '--config.sample.spatial.wall_groups=16', '--config.sample.spatial.node_capacity=12288'])]
    environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
        JAX_PLATFORMS='cuda', XLA_PYTHON_CLIENT_PREALLOCATE='true', XLA_PYTHON_CLIENT_MEM_FRACTION='0.90')
    try:
        for label, preset, options in arms:
            card = gpu()
            if card['uuid'] != state['gpu']['uuid'] or card['used_MiB'] > 1024:
                raise RuntimeError(f'Original idle GPU required for serial timing: {card}')
            name = output.name+'-probe-'+label
            destination = LOGS/name
            if destination.exists():
                raise FileExistsError(destination)
            command = [sys.executable, '-m', 'examples.channel_nsch2d.train',
                f'--config=examples/channel_nsch2d/configs/{preset}.py', '--seed=42',
                '--config.train.iterations=60', f'--config.log.run_name={name}', *options]
            record = dict(label=label, command=command, run=str(destination), status='running',
                          started_utc=datetime.now(timezone.utc).isoformat())
            state['stages'].append(record)
            tick = time.perf_counter()
            with (output/f'probe-{label}.log').open('x') as log:
                child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
                record['pid'] = child.pid
                save()
                code = child.wait()
            record.update(exit_code=code, seconds=time.perf_counter()-tick, status='passed' if code == 0 else 'failed')
            save()
            if code:
                raise RuntimeError(f'Probe {label} failed; see its retained log')
            result = json.loads((destination/'summary.json').read_text())
            record['spatial'] = result['spatial']
            record['peak_active_GiB'] = result['device_memory_bytes']['peak_bytes_in_use']/1024**3
            save()
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
    run_probes(parser.parse_args().output)
