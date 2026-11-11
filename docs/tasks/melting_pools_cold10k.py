"""Serial paired cold Melting runs; snapshots preserve fixed-budget diagnostics."""
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from docs.tasks.pino_spatial_pool_audit import audit

REPO = Path(__file__).resolve().parents[2]
ROOT = Path('/root/autodl-tmp/tf-logs/icemelting3d')


def main(output):
    output.mkdir(parents=True, exist_ok=False)
    journal = output/'journal.json'
    state = dict(status='running', pid=os.getpid(), cold_start=True, seed=43,
                 steps=10000, started_utc=datetime.now(timezone.utc).isoformat(), stages=[])

    def save():
        temp = journal.with_suffix('.partial')
        temp.write_text(json.dumps(state, indent=2, allow_nan=False)+'\n')
        temp.replace(journal)

    env = dict(os.environ, JAX_PLATFORMS='cuda', OPENBLAS_NUM_THREADS='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', XLA_PYTHON_CLIENT_MEM_FRACTION='0.40')
    save()
    stages = []
    try:
        for mode in ('fixed_pool', 'rolling_pool'):
            name = output.name+'-'+mode
            path = ROOT/name
            command = [sys.executable, '-m', 'examples.icemelting3d.train',
                       '--config=examples/icemelting3d/configs/train_operator_pool_10k.py',
                       '--seed=43', '--config.sample.spatial.mode='+mode,
                       '--config.log.run_name='+name]
            log = output/(mode+'.log')
            item = dict(mode=mode, path=str(path), command=command, log=str(log), status='running')
            state['stages'].append(item)
            start = time.perf_counter()
            with log.open('x') as stream:
                child = subprocess.Popen(command, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT)
                item['pid'] = child.pid
                save()
                code = child.wait()
            item.update(status='passed' if code == 0 else 'failed', pid=None,
                        seconds=time.perf_counter()-start, exit_code=code)
            save()
            if code:
                raise RuntimeError(f'{mode} exited {code}; see {log}')
            summary = json.loads((path/'summary.json').read_text())
            if summary['iterations'] != 10000 or not summary['cold_start']:
                raise ValueError('A complete cold10000run is required')
            stages.append(item)
        result = audit('icemelting3d', stages)
        (output/'paired-results.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    except Exception as error:
        state.update(status='failed', failure=str(error))
        raise
    else:
        state['status'] = 'complete'
    finally:
        state['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
