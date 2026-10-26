"""Three predeclared seed2026 cold starts through the canonical train entries."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'reports/phasefield-seed-stability-20261025'
CASES = ('corrosion2d', 'electropolish2d', 'icemelting3d')


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    specs = []
    for case in CASES:
        name = '20261025-independent-seed2026-default-strategy'
        run = Path('/root/autodl-tmp/tf-logs')/case/name
        if run.exists():
            raise FileExistsError(run)
        command = [str(ROOT/'.venv/bin/python'), '-u', '-m', f'examples.{case}.train',
                   '--seed=2026', '--config.log.run_name='+name]
        if case == 'icemelting3d':
            command.append('--config.eval.graph_reference_path='+str(
                ROOT/'reports/icemelting-weak-prior-20261024/reference.npz'))
        specs.append({'case': case, 'seed': 2026, 'updates': 5000,
                      'run': str(run), 'command': command, 'stdout': str(OUT/(case+'.log'))})
    manifest = {'status': 'running', 'seed_predeclared': 2026,
                'maximum_workers': 3, 'runs': specs,
                'selection': 'Every final5000 checkpoint, including failures; no intermediate selection'}
    (OUT/'launch.json').write_text(json.dumps(manifest, indent=2)+'\n')
    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE='false',
               XLA_FLAGS='--xla_gpu_deterministic_ops=true')
    env.pop('JAX_PLATFORMS', None)

    def execute(spec):
        start = time.monotonic()
        with Path(spec['stdout']).open('x') as stream:
            process = subprocess.Popen(spec['command'], cwd=ROOT, env=env,
                                       stdout=stream, stderr=subprocess.STDOUT)
            print(f"Started {spec['case']}, pid={process.pid}, seed=2026", flush=True)
            returncode = process.wait()
        return {**spec, 'returncode': returncode, 'wall_seconds': time.monotonic()-start}

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for future in concurrent.futures.as_completed([pool.submit(execute, s) for s in specs]):
            result = future.result()
            results.append(result)
            (OUT/'completed.json').write_text(json.dumps(results, indent=2)+'\n')
            print(f"Finished {result['case']}, code={result['returncode']}, seconds={result['wall_seconds']:.1f}", flush=True)
    manifest['status'] = 'completed' if all(r['returncode'] == 0 for r in results) else 'failed'
    manifest['runs'] = sorted(results, key=lambda r: CASES.index(r['case']))
    (OUT/'launch.json').write_text(json.dumps(manifest, indent=2)+'\n')
    if manifest['status'] != 'completed':
        raise RuntimeError('A predeclared control failed; inspect logs without replacing its seed')


if __name__ == '__main__':
    main()
