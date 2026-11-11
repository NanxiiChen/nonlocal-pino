"""Queue matched cold fixed/rolling comparisons for the current regular-domain PINOs."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[2]
LOGS = Path('/root/autodl-tmp/tf-logs')
MODES = ('fixed_pool', 'rolling_pool')
CASES = {'electropolish2d': (10_000, (42, 43)), 'icemelting3d': (5_000, (43,))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    journal = output / 'journal.json'
    state = dict(status='running', pid=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat(), stages=[])

    def save():
        partial = journal.with_suffix('.partial')
        partial.write_text(json.dumps(state, indent=2, allow_nan=False) + '\n')
        partial.replace(journal)

    env = dict(os.environ, JAX_PLATFORMS='cuda', OPENBLAS_NUM_THREADS='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', XLA_PYTHON_CLIENT_MEM_FRACTION='0.60')

    def run(label, command, path):
        log = output / f'{label}.log'
        stage = dict(label=label, command=command, path=str(path), log=str(log), status='running')
        state['stages'].append(stage)
        start = time.perf_counter()
        with log.open('x') as stream:
            child = subprocess.Popen([sys.executable, *command], cwd=REPO, env=env,
                                     stdout=stream, stderr=subprocess.STDOUT)
            stage['pid'] = child.pid
            save()
            code = child.wait()
        stage.update(pid=None, exit_code=code, seconds=time.perf_counter()-start,
                     status='passed' if code == 0 else 'failed')
        save()
        if code != 0:
            raise RuntimeError(f'{label} exited {code}, see {log}')

    save()
    try:
        for case, (steps, seeds) in CASES.items():
            for seed in seeds:
                paths = {}
                for mode in MODES:
                    label = f'{case}-s{seed}-{mode}'
                    name = f'{output.name}-{label}'
                    path = LOGS / case / name
                    command = ['-m', f'examples.{case}.train',
                               f'--config=examples/{case}/configs/train_operator_{mode}.py',
                               f'--seed={seed}', f'--config.log.run_name={name}']
                    if case == 'electropolish2d':
                        command += ['--config.arch.width=96', '--config.sample.Nt=16',
                                    f'--config.train.iterations={steps}']
                    run(label, command, path)
                    summary = json.loads((path/'summary.json').read_text())
                    if summary['iterations'] != steps:
                        raise ValueError(f'{label} missed its fixed training budget {steps}')
                    paths[mode] = path
                if case == 'electropolish2d':
                    for mode, path in paths.items():
                        run(f'{case}-s{seed}-{mode}-evaluate64',
                            ['-m', 'docs.tasks.electro_spatial_evaluate', '--run', str(path)], path)
    except Exception as error:
        state.update(status='failed', failure=str(error))
        raise
    else:
        state['status'] = 'complete'
    finally:
        state['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == '__main__':
    main()
