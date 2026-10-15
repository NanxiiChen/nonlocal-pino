"""Detached ownership and receipts for cold core+halo loading through .0060."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import traceback

BASE = Path('/root/autodl-tmp/tf-logs/fracture2d')
NAME = 'pinn_core+halo_tip_tol2e5'
ROOT = BASE/NAME
EVENTS = BASE.parent/NAME
RECEIPT = BASE/(NAME+'_process.json')
LOG = BASE/(NAME+'.log')
ARGV = ['/root/nonlocal-KKS/.venv/bin/python', '-u', '-m',
    'examples.fracture2d.train',
    '--config=examples/fracture2d/configs/train_core_halo_tip.py',
    '--config.loading.damage_tolerance=2e-5',
    '--config.output.root='+str(ROOT),
    '--config.output.event_dir='+str(EVENTS)]
record = dict(status='preparing', supervisor_pid=os.getpid(), worker_pid=None,
    argv=ARGV, started=datetime.now(timezone.utc).isoformat(), log=str(LOG),
    root=str(ROOT), event_dir=str(EVENTS), cold_start=True,
    terminal_load=.0060, expected_positive_endpoints=60)
worker = None
cancelled = False

def save():
    temporary = RECEIPT.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(record, indent=2)+'\n')
    temporary.replace(RECEIPT)

def stop(signum, frame):
    global cancelled
    cancelled = True
    if worker is not None and worker.poll() is None:
        worker.terminate()

for sig in (signal.SIGTERM, signal.SIGINT):
    signal.signal(sig, stop)
try:
    if ROOT.exists() or EVENTS.exists() or RECEIPT.exists() or LOG.exists():
        raise RuntimeError('refusing to reuse run/events/receipt/log paths')
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
            '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('GPU occupied; no training started')
    env = dict(os.environ, PYTHONPATH='.:src', OPENBLAS_NUM_THREADS='1',
        OMP_NUM_THREADS='1', XLA_PYTHON_CLIENT_PREALLOCATE='false',
        MPLBACKEND='Agg', JAX_PLATFORMS='cuda')
    with LOG.open('xb', buffering=0) as stream:
        worker = subprocess.Popen(ARGV, cwd='/root/nonlocal-KKS', env=env,
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
        record.update(status='running', worker_pid=worker.pid)
        save()
        code = worker.wait()
    record['worker_exit_code'] = code
    if cancelled:
        record['status'] = 'cancelled'
    elif code:
        record['status'] = 'failed'
    else:
        result = json.loads((ROOT/'result.json').read_text())
        if (not result['complete'] or len(result['curve']) != 60
                or abs(result['current']['load']-.0060) > 1e-14
                or not result['current']['q_d_synchronized']):
            raise RuntimeError('worker exited without complete synchronized .0060 path')
        record.update(status='complete', completed_endpoints=60,
            final_load=result['current']['load'],
            final_reaction=result['current']['reaction'])
except BaseException:
    record.update(status='failed', error=traceback.format_exc())
    if worker is not None and worker.poll() is None:
        worker.terminate()
        worker.wait()
finally:
    record['ended'] = datetime.now(timezone.utc).isoformat()
    save()
