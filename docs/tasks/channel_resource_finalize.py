"""Finish the already-running resource probe without resuming its old owner."""
import json
import os
from pathlib import Path
import signal
import time
from datetime import datetime, timezone

ROOT = Path('/root/autodl-tmp/tf-logs/channel-uniform-pro6000-20261113-02')
OWNER, CHILD = 5245, 57170


def process(pid):
    p = Path(f'/proc/{pid}')
    raw = (p/'stat').read_text().rsplit(')', 1)[1].split()
    return raw[0], int(raw[19]), int(raw[49])


def main():
    owner_state, owner_start, _ = process(OWNER)
    if owner_state != 'T':
        raise RuntimeError('Old owner must be stopped before adopting the final probe')
    journal = ROOT/'journal.json'
    original = json.loads(journal.read_text())
    last = original['stages'][-1]
    if (original['pid'] != OWNER or last['pid'] != CHILD
            or last['label'] != 'probe-core2048' or last['status'] != 'running'):
        raise RuntimeError('Unexpected resource owner/child identity')
    child_start = process(CHILD)[1]
    while True:
        state, start, wait_status = process(CHILD)
        if start != child_start or process(OWNER)[:2] != ('T', owner_start):
            raise RuntimeError('Owned process identity/state changed')
        if state == 'Z':
            break
        time.sleep(10)
    code = os.waitstatus_to_exitcode(wait_status)
    (ROOT/'journal-before-resource-finalization.json').write_text(json.dumps(original, indent=2)+'\n')
    # Kill only the stopped scheduler, after its final resource child has exited.
    # Resuming that scheduler could start the explicitly cancelled long training.
    os.kill(OWNER, signal.SIGKILL)
    last.update(exit_code=code, status='passed' if code == 0 else 'failed',
                observed_finished_utc=datetime.now(timezone.utc).isoformat())
    records = []
    if code == 0:
        for stage in original['stages'][1:]:
            summary = json.loads((Path(stage['run'])/'summary.json').read_text())
            if summary['status'] != 'completed' or summary['iterations'] != 60:
                raise RuntimeError('Every resource probe must finish its declared60updates')
            s = summary['spatial']
            records.append(dict(label=stage['label'].removeprefix('probe-'),
                core_count=0 if s['mode'] == 'full' else s['core_count'],
                node_capacity=s['node_capacity'],
                seconds=s['warm_step_with_batch_median_seconds'],
                peak_bytes=summary['device_memory_bytes']['peak_bytes_in_use'],
                run=stage['run']))
        (ROOT/'resources.json').write_text(json.dumps(records, indent=2)+'\n')
    original.update(status='completed_resources_only' if code == 0 else 'failed_resource_probe',
        long_training_cancelled_by_user=True, finalizer_pid=os.getpid(),
        finished_utc=datetime.now(timezone.utc).isoformat(),
        reason='User requests resource records only, then ceramic FEM comparison; no channel long run')
    original.pop('progress', None)
    journal.write_text(json.dumps(original, indent=2)+'\n')
    print(json.dumps(dict(status=original['status'], resources=records), indent=2), flush=True)
    if code:
        raise RuntimeError(f'Final resource child exited with code{code}')


if __name__ == '__main__':
    main()
