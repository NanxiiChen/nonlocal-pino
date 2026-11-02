"""Wait for one complete checkpoint, then run existing CPU-only field tools."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]


def main(run, owner, step, output, timeout_seconds):
    run, owner, output = map(Path, (run, owner, output))
    if not all(p.is_absolute() for p in (run, owner, output)):
        raise ValueError('Use absolute run, owner-journal and output paths')
    c = json.loads((run/'config.json').read_text())
    if (step <= 0 or step > c['train']['iterations']
            or step % c['log']['checkpoint_every'] or timeout_seconds <= 0):
        raise ValueError('Require a scheduled positive checkpoint and finite wait budget')
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = run/f'ckpt_{step:06d}.eqx'
    started = time.monotonic()
    record = dict(status='waiting_for_complete_checkpoint', pid=os.getpid(),
        run=str(run), owner=str(owner), checkpoint=str(checkpoint), step=step,
        output=str(output), timeout_seconds=timeout_seconds,
        started_utc=datetime.now(timezone.utc).isoformat(),
        scope='One CPU-only partial-checkpoint gallery; no training, FEM or optimizer recovery')

    def write():
        temporary = output/'status.tmp'
        temporary.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
        temporary.replace(output/'status.json')

    write()
    try:
        while True:
            state = json.loads(owner.read_text())
            if Path(state['run']) != run:
                raise ValueError('Owner journal belongs to a different run')
            # The log is flushed before serialisation. A later full update or
            # final summary establishes that serialisation has actually ended.
            complete = ('latest_loss' in state and state['latest_loss']['step'] > step)
            complete = complete or (run/'summary.json').is_file()
            if checkpoint.is_file() and complete:
                break
            if state['status'] != 'training':
                raise RuntimeError(f"Training ended before the requested checkpoint: {state['status']}")
            if time.monotonic()-started >= timeout_seconds:
                raise TimeoutError('Requested checkpoint did not arrive within the finite wait budget')
            time.sleep(30)
        for saved in (run/'source').rglob('*.py'):
            if saved.read_bytes() != (ROOT/saved.relative_to(run/'source')).read_bytes():
                raise ValueError(f'Frozen source changed: {saved}')
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-4:])
        env = dict(os.environ, JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='',
            OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MPLBACKEND='Agg')
        record.update(status='cpu_postprocessing')
        write()
        payload = run/f'partial_{step:06d}.npz'
        commands = [
            ['docs/tasks/channel_partial_evaluate.py', '--run', str(run), '--step', str(step)],
            ['-m', 'examples.channel_nsch2d.evaluate', '--fields', str(payload),
             '--output', str(output), '--field', 'phi', '--times', '0', '1.5', '3', '4.5', '6'],
            ['-m', 'examples.channel_nsch2d.evaluate', '--fields', str(payload),
             '--output', str(output), '--field', 'speed', '--field', 'pressure',
             '--times', '.5', '1.5', '3', '4.5', '6'],
            ['docs/tasks/channel_droplet_metrics.py', '--fields', str(payload),
             '--output', str(output/'droplet-shape')],
        ]
        for command in commands:
            subprocess.run([sys.executable, *command], cwd=ROOT, env=env, check=True)
        record.update(status='partial_checkpoint_gallery_completed',
            completed_utc=datetime.now(timezone.utc).isoformat())
        write()
    except Exception as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}')
        write()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--owner', required=True)
    parser.add_argument('--step', required=True, type=int)
    parser.add_argument('--output', required=True)
    parser.add_argument('--timeout-seconds', required=True, type=int)
    args = parser.parse_args()
    main(args.run, args.owner, args.step, args.output, args.timeout_seconds)
