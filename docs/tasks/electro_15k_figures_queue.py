"""Wait for the existing finite suite, then create CPU-only draft figures.

This queue never starts training, retries a failed experiment, or publishes.
Visual inspection and README/website selection remain explicit later actions.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    args = parser.parse_args()
    record = args.bundle/'figure_queue.json'
    if record.exists():
        raise FileExistsError('An existing figure owner must be inspected, not duplicated')
    state = dict(status='waiting', pid=os.getpid(), bundle=str(args.bundle), stages=[],
                 started_utc=datetime.now(timezone.utc).isoformat(), publication=False)

    def save():
        temporary = record.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(state, indent=2)+'\n')
        temporary.replace(record)

    save()
    try:
        while True:
            current = json.loads((args.bundle/'journal.json').read_text())
            if current['status'] == 'complete':
                break
            if current['status'] not in ('running', 'waiting_dependency'):
                raise RuntimeError('Numerical suite did not complete: ' + current['status'])
            if not Path(f'/proc/{current["pid"]}').exists():
                raise RuntimeError('Numerical owner is absent; inspect its existing logs')
            time.sleep(60)
        state['status'] = 'rendering'
        save()
        commands = [
            ('library', ['-m','scripts.electro_15k_results','--bundle',str(args.bundle),'--action','library',
                         '--output',str(args.bundle/'library-figures')]),
            ('fusion-data', ['-m','scripts.electro_15k_results','--bundle',str(args.bundle),'--action','fusion']),
            ('fusion-figures', ['-m','scripts.render_pino_time_fusion',
                               '--comparison',str(args.bundle/'time_fusion_comparison15k.json'),
                               '--output',str(args.bundle/'temporal-figures'),'--x-axis','cold-wall-time',
                               '--clock-evidence',str(args.bundle/'time_fusion_clock15k.json')])]
        env = dict(os.environ, JAX_PLATFORMS='cpu', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                   PYTHONPATH='/tmp/pino-pdf-audit:' + str(ROOT) + ':' + str(ROOT/'src'))
        for name, command in commands:
            row = dict(name=name, args=command, status='running')
            state['stages'].append(row)
            save()
            start = time.monotonic()
            with (args.bundle/('figure-'+name+'.log')).open('x') as log:
                result = subprocess.run([sys.executable,*command], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            row.update(exit_code=result.returncode, seconds=time.monotonic()-start,
                       status='passed' if result.returncode == 0 else 'failed')
            save()
            if result.returncode != 0:
                raise RuntimeError('Figure preparation failed; inspect ' + name)
        state['status'] = 'ready_for_visual_review'
        state['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save()
    except Exception as error:
        state.update(status='failed', error=str(error))
        save()
        raise


if __name__ == '__main__':
    main()
