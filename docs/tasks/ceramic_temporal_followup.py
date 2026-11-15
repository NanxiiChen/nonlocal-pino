"""Process-identity wait, then CPU figures only after the paired audit succeeds."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def process_identity(pid):
    try:
        value = (Path('/proc')/str(pid)/'stat').read_text()
    except FileNotFoundError:
        return None
    fields = value[value.rindex(')')+2:].split()
    return None if fields[0] in ('Z', 'X') else int(fields[19])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--owner-pid', type=int, required=True)
    a = p.parse_args(); root = Path(__file__).resolve().parents[2]
    record = a.bundle/'figure_followup.json'
    if record.exists(): raise FileExistsError('Preserve existing observer record: '+str(record))
    journal = json.loads((a.bundle/'journal.json').read_text())
    if journal['pid'] != a.owner_pid: raise ValueError('Owner PID differs from the selected journal')
    reused = 'baseline_reused' in journal and journal['baseline_reused'] is True
    if reused:
        raise ValueError('User requires equal-point histories; use the matched64case continuation bundle')
    elif 'predecessor_start_ticks' in journal:
        owner_module = 'docs.tasks.ceramic_temporal_equal_queue'
    else:
        owner_module = 'docs.tasks.ceramic_temporal_queue'
    figure_module = 'docs.tasks.ceramic_temporal_figures'
    identity = process_identity(a.owner_pid)
    if identity is not None:
        command = (Path('/proc')/str(a.owner_pid)/'cmdline').read_bytes().split(b'\0')
        if owner_module.encode() not in command: raise ValueError('Unexpected owner command')
    state = dict(status='waiting_for_owner', pid=os.getpid(), owner_pid=a.owner_pid,
                 owner_start_ticks=identity, process_poll_seconds=60,
                 output=str(a.bundle/'figures-v1'), metrics_polled=False)
    def save():
        temporary = record.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2)+'\n'); temporary.replace(record)
    save()
    try:
        # This background observer checks no training logs, metrics or GPU state.
        deadline = time.monotonic()+12*3600
        while identity is not None and process_identity(a.owner_pid) == identity:
            if time.monotonic() >= deadline: raise TimeoutError('Twelve-hour observer deadline; trainer not modified')
            time.sleep(60)
        journal = json.loads((a.bundle/'journal.json').read_text())
        if journal['status'] != 'complete': raise RuntimeError('Owner/audit incomplete; no figures generated')
        state['status'] = 'rendering'; save()
        command = [sys.executable, '-m', figure_module,
                   '--bundle', str(a.bundle), '--output', state['output']]
        state['command'] = command; save()
        result = subprocess.run(command, cwd=root, check=False, env=dict(os.environ,
            JAX_PLATFORMS='cpu', PYTHONPATH='/tmp/pino-pdf-audit', OPENBLAS_NUM_THREADS='1',
            OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'))
        if result.returncode: raise RuntimeError('Figure rendering failed; completed training results retained')
        state.update(status='complete', needs_visual_review=True); save()
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}'); save(); raise


if __name__ == '__main__': main()
