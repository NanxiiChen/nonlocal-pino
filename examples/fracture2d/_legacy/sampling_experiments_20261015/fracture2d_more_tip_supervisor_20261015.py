"""Detached ownership and termination receipts for a finite sampling probe."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import traceback

BASE = Path('/root/autodl-tmp/tf-logs/fracture2d')
RECEIPT = BASE/'core_halo_more_tip_20261015_process.json'
LOG = BASE/'core_halo_more_tip_20261015.log'
ARGV = ['/root/nonlocal-KKS/.venv/bin/python', '-u', '/tmp/fracture2d_more_tip_probe_20261015.py']
record = dict(status='preparing', supervisor_pid=os.getpid(), worker_pid=None,
    argv=ARGV, started=datetime.now(timezone.utc).isoformat(), log=str(LOG),
    root=str(BASE/'core_halo_more_tip_20261015'))
worker = None
cancelled = False
def save():
    temp = RECEIPT.with_suffix('.tmp.json')
    temp.write_text(json.dumps(record, indent=2)+'\n')
    temp.replace(RECEIPT)
def stop(signum, frame):
    global cancelled
    cancelled = True
    if worker is not None and worker.poll() is None:
        worker.terminate()
for signum in (signal.SIGTERM, signal.SIGINT):
    signal.signal(signum, stop)
try:
    if Path(record['root']).exists():
        raise RuntimeError('probe root already exists')
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('GPU is occupied; no probe started')
    env = dict(os.environ, PYTHONPATH='.:src', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               XLA_PYTHON_CLIENT_PREALLOCATE='false', MPLBACKEND='Agg')
    with LOG.open('ab', buffering=0) as stream:
        worker = subprocess.Popen(ARGV, cwd='/root/nonlocal-KKS', env=env, stdin=subprocess.DEVNULL,
                                  stdout=stream, stderr=subprocess.STDOUT)
        record.update(status='running', worker_pid=worker.pid)
        save()
        code = worker.wait()
    record['worker_exit_code'] = code
    if cancelled:
        record['status'] = 'cancelled'
    elif code:
        record['status'] = 'failed'
    else:
        result = json.loads((Path(record['root'])/'result.json').read_text())
        assert result['complete'] and len(result['summaries']) == 3
        record['status'] = 'complete'
except BaseException:
    record.update(status='failed', error=traceback.format_exc())
    if worker is not None and worker.poll() is None:
        worker.terminate()
        worker.wait()
finally:
    record['ended'] = datetime.now(timezone.utc).isoformat()
    save()
