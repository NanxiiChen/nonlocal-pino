"""Run the planned CPU channel audits after the existing GPU queues finish."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
LOGS = Path('/root/autodl-tmp/tf-logs/channel_nsch2d')
MANIFEST = Path('/root/autodl-tmp/data/channel_nsch2d/random-walls-20261102-02/unseen3/manifest.json')
HELPERS = ('pino_channel_final_audit.py', 'pino_channel_spatial_replay.py',
           'pino_channel_regional_errors.py')


def read(path):
    return json.loads(path.read_text())


def queue_ready(journal, module):
    """Require terminal success or an actual matching live owner, never a lock."""
    state = read(journal)
    if state['status'] == 'complete':
        return True
    if state['status'] in ('failed', 'superseded'):
        raise RuntimeError(f'Predecessor {journal} is {state["status"]}')
    pid = state['pid']
    if type(pid) is not int or pid <= 0:
        raise ValueError(f'Invalid predecessor PID: {journal}')
    try:
        args = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    except FileNotFoundError:
        # The owner can publish success and exit between the two reads.
        if read(journal)['status'] == 'complete':
            return True
        raise RuntimeError(f'Predecessor handle {pid} missing: {journal}') from None
    if module.encode() not in args or str(journal.parent).encode() not in args:
        if read(journal)['status'] == 'complete':
            return True
        raise RuntimeError(f'Predecessor handle {pid} has a different command: {journal}')
    return False


def run(output, after, idle_after, cpus, poll_seconds):
    if (after.resolve() == idle_after.resolve() or len(cpus) != 2
            or len(set(cpus)) != 2 or not set(cpus) <= os.sched_getaffinity(0)
            or not 1 <= poll_seconds <= 60):
        raise ValueError('Two distinct queues, two available CPUs and 1..60s polling required')
    os.sched_setaffinity(0, cpus)
    predecessors = ((after, 'docs.tasks.pino_unified_accuracy_queue'),
                    (idle_after, 'docs.tasks.pino_spinodal_band_queue'))
    for path, module in predecessors:
        queue_ready(path, module)
    runs = [LOGS / (after.parent.name + '-channel-cold20k-' + label)
            for label in ('bulk512', 'wall512', 'full')]
    frozen = runs[0] / 'source'
    source = {p.relative_to(frozen): p.read_bytes() for p in frozen.rglob('*.py')}
    if not source:
        raise ValueError('The current channel run must contain its frozen production sources')
    production_source_files = len(source)
    for path, content in source.items():
        if (ROOT / path).read_bytes() != content:
            raise ValueError(f'Current implementation differs from training: {path}')
    # The trainer freezes selected shared modules, not an independently
    # importable package. Keep their bytes and add the package dependencies.
    dependencies = [ROOT / 'examples/__init__.py',
                    *sorted((ROOT / 'src/nonlocal_pinn').rglob('*.py'))]
    supplemental = []
    for path in dependencies:
        relative = path.relative_to(ROOT)
        if relative not in source:
            supplemental.append(str(relative))
            source[relative] = path.read_bytes()
    for name in HELPERS:
        path = Path('docs/tasks') / name
        source[path] = (ROOT / path).read_bytes()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, output / 'controller.py')
    runtime = output / 'source'
    for path, content in source.items():
        destination = runtime / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    (output / 'source_paths.json').write_text(json.dumps(
        [str(path) for path in sorted(source)], indent=2) + '\n')
    state = dict(status='preflight', pid=os.getpid(), after=str(after),
                 idle_after=str(idle_after), runtime=str(runtime),
                 production_source_files=production_source_files,
                 supplemental_dependency_files=supplemental, runtime_source_files=len(source),
                 cpus=cpus, poll_seconds=poll_seconds, stages=[],
                 started_utc=datetime.now(timezone.utc).isoformat())
    journal = output / 'journal.json'

    def save():
        temporary = journal.with_suffix('.partial')
        temporary.write_text(json.dumps(state, indent=2, allow_nan=False) + '\n')
        temporary.replace(journal)

    environment = dict(os.environ, JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='',
                       OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       PYTHONPATH=os.pathsep.join((str(runtime), str(runtime / 'src'))),
                       PYTHONUNBUFFERED='1', MPLBACKEND='Agg')

    def stage(label, command):
        current = {p.relative_to(runtime): p.read_bytes() for p in runtime.rglob('*.py')}
        if current != source:
            raise RuntimeError('Frozen audit source changed')
        record = dict(label=label, command=[sys.executable, *command], status='running',
                      started_utc=datetime.now(timezone.utc).isoformat())
        state['stages'].append(record)
        started = time.perf_counter()
        with (output / (label + '.log')).open('x') as stream:
            child = subprocess.Popen(record['command'], cwd=runtime, env=environment,
                                     stdin=subprocess.DEVNULL, stdout=stream,
                                     stderr=subprocess.STDOUT)
            record['pid'] = child.pid
            save()
            code = child.wait()
        record.update(exit_code=code, seconds=time.perf_counter() - started,
                      status='passed' if code == 0 else 'failed')
        save()
        if code:
            raise RuntimeError(f'{label} failed; retained output, no automatic retry')

    save()
    try:
        # Check dependency origins without a device allocation or model call.
        stage('runtime-imports', ['-c',
            "import importlib,json,pathlib,sys; root=pathlib.Path.cwd(); "
            "names=['docs.tasks.pino_channel_final_audit',"
            "'docs.tasks.pino_channel_spatial_replay','docs.tasks.pino_channel_regional_errors',"
            "'examples.channel_nsch2d.online']; "
            "[importlib.import_module(n) for n in names]; "
            "paths={n:str(pathlib.Path(m.__file__).resolve()) for n,m in sys.modules.items() "
            "if n.startswith(('docs.tasks.','examples.','nonlocal_pinn')) and "
            "getattr(m,'__file__',None)}; "
            "assert paths and all(pathlib.Path(p).is_relative_to(root) for p in paths.values()), paths; "
            "print(json.dumps(paths,indent=2))"])
        state['status'] = 'waiting_for_existing_queues'
        save()
        while True:
            ready = [queue_ready(path, module) for path, module in predecessors]
            if all(ready):
                break
            time.sleep(poll_seconds)
        if read(after.parent / 'channel-paired.json')['status'] != 'passed':
            raise ValueError('Original channel pairing must pass before independent checks')
        state['status'] = 'running'
        save()
        arguments = ['--runs', *map(str, runs), '--evaluations',
                     *(str(p / 'reserved') for p in runs), '--manifest', str(MANIFEST)]
        stage('field-and-spatial-audit', ['-m', 'docs.tasks.pino_channel_final_audit',
              *arguments, '--output', str(output / 'field-and-spatial')])
        stage('regional-field-audit', ['-m', 'docs.tasks.pino_channel_regional_errors',
              *arguments, '--band', '0.075', '--output', str(output / 'regional')])
        for name, expected in (('field-and-spatial', 'passed'),
                               ('regional', 'diagnostic_completed')):
            if read(output / name / 'summary.json')['status'] != expected:
                raise RuntimeError(f'{name} did not report a successful audit')
        state['status'] = 'complete'
    except Exception as error:
        state.update(status='failed', failure=str(error))
        raise
    finally:
        state['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--idle-after', type=Path, required=True)
    parser.add_argument('--cpus', type=int, nargs=2, required=True)
    parser.add_argument('--poll-seconds', type=int, required=True)
    args = parser.parse_args()
    run(args.output, args.after, args.idle_after, args.cpus, args.poll_seconds)
