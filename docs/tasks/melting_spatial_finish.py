"""Wait for serial pools, then fresh GPU replay and independent scientific report."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    repo = Path('/root/nonlocal-pinn')
    root = Path('/root/autodl-tmp/tf-logs/icemelting3d')
    journal = root/'20261031-melting-spatial-01-pools-process.json'
    receipt = repo/'docs/tasks/melting-spatial-finish.json'
    if receipt.exists():
        raise FileExistsError(receipt)
    record = dict(status='waiting_for_training', pid=os.getpid(), stages=[])
    def write():
        receipt.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    write()
    while True:
        state = json.loads(journal.read_text())
        if state['status'] == 'training_complete':
            break
        if state['status'] != 'running':
            raise RuntimeError(f'Training journal status{state["status"]}')
        time.sleep(2)
    env = dict(os.environ, JAX_PLATFORMS='cuda', OPENBLAS_NUM_THREADS='1',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', XLA_PYTHON_CLIENT_MEM_FRACTION='0.12')
    for name, script in [('GPU_checkpoint_replay', 'melting_spatial_replay.py'),
                         ('independent_inputs_fields_report', 'melting_spatial_report.py')]:
        log = root/('20261031-melting-spatial-01-'+name+'.log')
        record['status'] = name; write()
        with log.open('x') as stream:
            result = subprocess.run([sys.executable, str(repo/'docs/tasks'/script)],
                cwd=repo, env=env, stdout=stream, stderr=subprocess.STDOUT)
        record['stages'].append(dict(name=name, exit_code=result.returncode, log=str(log)))
        if result.returncode != 0:
            record['status'] = 'failed'; write(); raise RuntimeError(f'{name} failed: {log}')
        write()
    record.update(status='passed', finished_utc=datetime.now(timezone.utc).isoformat()); write()


if __name__ == '__main__':
    main()
