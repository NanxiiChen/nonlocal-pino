"""Queue the previously successful seed42, preserving maximum three workers."""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'reports/corrosion-robust-objective-20261026'


def main():
    name = '20261026-objective-bounded_frozen-seed42'
    run = Path('/root/autodl-tmp/tf-logs/corrosion2d')/name
    if run.exists(): raise FileExistsError(run)
    command = [str(ROOT/'.venv/bin/python'), '-u', '-m', 'examples.corrosion2d.train',
               '--seed=42', '--config.experiment.formulation=bounded_frozen', '--config.log.run_name='+name]
    spec = dict(status='queued', label='bounded_frozen-seed42', seed=42,
                formulation='bounded_frozen', run=str(run), command=command,
                stdout=str(OUT/'bounded_frozen-seed42.log'), updates=5000,
                rationale='Check the previously successful seed without seed selection',
                maximum_total_workers=3)
    manifest = OUT/'validation.json'
    if manifest.exists(): raise FileExistsError(manifest)
    manifest.write_text(json.dumps(spec, indent=2)+'\n')
    while not (OUT/'completed.json').exists():
        time.sleep(30)
    completed = json.loads((OUT/'completed.json').read_text())
    if not any(v['returncode']==0 for v in completed):
        raise RuntimeError('No successful original worker; inspect retained errors')
    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE='false',
               XLA_FLAGS='--xla_gpu_deterministic_ops=true', OPENBLAS_NUM_THREADS='1',
               OMP_NUM_THREADS='2', JAX_COMPILATION_CACHE_DIR='/tmp/corrosion-objective-cache')
    env.pop('JAX_PLATFORMS', None)
    start = time.monotonic()
    with Path(spec['stdout']).open('x') as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        spec.update(status='running', pid=process.pid)
        manifest.write_text(json.dumps(spec, indent=2)+'\n')
        print('Started frozen42 validation, pid='+str(process.pid), flush=True)
        rc = process.wait()
    if run.is_dir(): (run/'train.log').write_bytes(Path(spec['stdout']).read_bytes())
    spec.update(status='completed' if rc==0 else 'failed', returncode=rc,
                wall_seconds=time.monotonic()-start)
    manifest.write_text(json.dumps(spec, indent=2)+'\n')
    print(json.dumps(spec), flush=True)
    if rc: raise RuntimeError('Validation failed; inspect retained log')


if __name__ == '__main__': main()
