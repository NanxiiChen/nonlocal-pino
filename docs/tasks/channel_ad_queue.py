"""Bounded local-AD probes or cold20k/audits, with an explicit launch per phase."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
LOGS = Path('/root/autodl-tmp/tf-logs/channel_nsch2d')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--phase', choices=('probe', 'train', 'sampled'), required=True)
    parser.add_argument('--wait-for-probes', action='store_true',
                        help='Wait for this bundle\'s finite probes before the cold20k stage')
    parser.add_argument('--ad-memory-fraction', type=float,
                        help='Required for sampled phase: AD allocator fraction; leave driver headroom for CUDA graphs')
    args = parser.parse_args()
    if args.phase == 'sampled':
        if args.ad_memory_fraction is None or not 0 < args.ad_memory_fraction <= .97:
            parser.error('sampled phase requires --ad-memory-fraction in (0, 0.97]')
    elif args.ad_memory_fraction is not None:
        parser.error('--ad-memory-fraction is only supported by the sampled phase')
    args.bundle.mkdir(parents=True, exist_ok=args.phase == 'train')
    env = dict(os.environ, JAX_PLATFORMS='cuda', XLA_FLAGS='', JAX_ENABLE_COMPILATION_CACHE='false',
        XLA_PYTHON_CLIENT_PREALLOCATE='true', XLA_PYTHON_CLIENT_MEM_FRACTION='.97',
        OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,memory.total,driver_version',
                                   '--format=csv,noheader'], text=True).strip()
    state = dict(status='running', phase=args.phase, pid=os.getpid(),
        started_utc=datetime.now(timezone.utc).isoformat(), gpu=gpu, prefix=args.prefix, stages=[])
    if args.phase == 'sampled':
        state['allocator'] = dict(ad_fraction=args.ad_memory_fraction, nonlocal_fraction=.97,
            preallocate=True, compilation_cache=False,
            scope='Execution resource setting only; AD probes and production use the same fraction')
    journal = args.bundle/(args.phase+'-journal.json')
    if journal.exists():
        raise FileExistsError(journal)

    def save():
        tmp = journal.with_suffix('.partial')
        tmp.write_text(json.dumps(state, indent=2)+'\n'); tmp.replace(journal)

    def stage(name, command, *, cpu=False):
        entry = dict(name=name, command=command, status='running')
        memory_fraction = .97
        if args.phase == 'sampled' and (name.startswith('probe-ad') or name == 'train-ad'):
            memory_fraction = args.ad_memory_fraction
        child_env = dict(env, JAX_PLATFORMS='cpu' if cpu else 'cuda',
                         XLA_PYTHON_CLIENT_MEM_FRACTION=str(memory_fraction))
        entry['execution_environment'] = {key: child_env[key] for key in
            ('JAX_PLATFORMS', 'XLA_FLAGS', 'JAX_ENABLE_COMPILATION_CACHE',
             'XLA_PYTHON_CLIENT_PREALLOCATE', 'XLA_PYTHON_CLIENT_MEM_FRACTION')}
        state['stages'].append(entry); save()
        start = time.perf_counter()
        with (args.bundle/(name+'.log')).open('x') as output:
            child = subprocess.Popen([sys.executable, *command], cwd=ROOT,
                env=child_env, stdout=output, stderr=subprocess.STDOUT)
            entry['pid'] = child.pid; save()
            while True:
                try:
                    code = child.wait(timeout=30); break
                except subprocess.TimeoutExpired:
                    state['heartbeat_utc'] = datetime.now(timezone.utc).isoformat(); save()
        entry.update(status='passed' if code == 0 else 'failed', exit_code=code, seconds=time.perf_counter()-start)
        save()
        if code:
            raise RuntimeError(name+' failed; remaining stages are not launched')

    def support(action):
        result = ['-m', 'docs.tasks.channel_ad_support', '--action', action,
                  '--bundle', str(args.bundle), '--prefix', args.prefix]
        if args.phase == 'sampled':
            result.append('--sampled')
        return result

    def train(config, name, probe, *, core=None, block=None):
        command = ['-m', 'examples.channel_nsch2d.train',
            '--config=examples/channel_nsch2d/configs/'+config+'.py', '--seed=42', '--config.log.run_name='+name]
        if probe:
            command.append('--config.train.iterations=60')
        if core is not None:
            command.append('--config.sample.ad_sampling.core_count='+str(core))
        if block is not None:
            command.append('--config.train.microbatch.size='+str(block))
        return command

    save()
    try:
        if args.phase == 'sampled':
            stage('screen', support('screen'), cpu=True)
            for core, block in ((1024, 18), (2048, 9)):
                stage('probe-ad'+str(core), train('train_operator_local_ad_sampled',
                    args.prefix+'-probe-ad'+str(core), True, core=core, block=block))
            stage('probe-nonlocal', train('train_full_fem_comparison_probe', args.prefix+'-probe-nonlocal', True))
            stage('qualify', support('qualify'), cpu=True)
            resources = json.loads((args.bundle/'resources.json').read_text())
            if resources['status'] != 'passed':
                raise ValueError('Successful sampled and nonlocal resource probes required')
            probe = Path(resources['selected_probe'])
            for source in (probe/'source').rglob('*.py'):
                if source.read_bytes() != (ROOT/source.relative_to(probe/'source')).read_bytes():
                    raise ValueError('Source changed after sampled probe: '+str(source))
            selected = json.loads((probe/'config.json').read_text())
            if (selected['sample']['ad_sampling']['core_count'] != 2048
                    or selected['train']['microbatch']['size'] != 9):
                raise ValueError('Selected sampled probe differs from predeclared production settings')
            run = LOGS/(args.prefix+'-ad-cold20k')
            stage('train-ad', train('train_operator_local_ad_sampled', run.name, False, core=2048, block=9))
            stage('validate', ['-m', 'examples.channel_nsch2d.validate_run', '--run', str(run)])
            stage('cohort', ['-m', 'examples.channel_nsch2d.cohort_evaluate', '--run', str(run),
                '--manifest', selected['operator']['validation_manifest'], '--output', str(run/'unseen3')])
            stage('finish', support('finish'), cpu=True)
            state.update(status='complete', finished_utc=datetime.now(timezone.utc).isoformat()); save()
            return
        if args.phase == 'probe':
            stage('screen', support('screen'), cpu=True)
            stage('probe-ad', train('train_operator_local_ad', args.prefix+'-probe-ad', True))
            stage('probe-nonlocal', train('train_full_fem_comparison_probe', args.prefix+'-probe-nonlocal', True))
            stage('qualify', support('qualify'), cpu=True)
        else:
            if args.wait_for_probes:
                while True:
                    probe_state = json.loads((args.bundle/'probe-journal.json').read_text())
                    if probe_state['status'] == 'failed':
                        raise RuntimeError('Resource probes failed; cold training is not launched')
                    if probe_state['status'] == 'complete':
                        break
                    if probe_state['status'] != 'running':
                        raise ValueError('Unexpected resource-probe status')
                    os.kill(probe_state['pid'], 0)
                    state['waiting_for'] = 'qualified resource probes'
                    state['heartbeat_utc'] = datetime.now(timezone.utc).isoformat(); save()
                    time.sleep(30)
                if probe_state['gpu'] != gpu:
                    raise ValueError('GPU differs from the resource-qualified device')
                del state['waiting_for']
                save()
            resources = json.loads((args.bundle/'resources.json').read_text())
            if resources['status'] != 'passed':
                raise ValueError('Qualified full-update resources are required before long training')
            probe = LOGS/(args.prefix+'-probe-ad')
            from examples.channel_nsch2d.configs.train_operator_local_ad import get_config
            from nonlocal_pinn.training.runtime import with_run_seed
            current = json.loads(with_run_seed(get_config(), 42).to_json())
            previous = json.loads((probe/'config.json').read_text())
            previous['train']['iterations'] = current['train']['iterations']
            previous['log']['run_name'] = current['log']['run_name']
            if previous != current:
                raise ValueError('Production configuration differs from measured probe')
            for source in (probe/'source').rglob('*.py'):
                if source.read_bytes() != (ROOT/source.relative_to(probe/'source')).read_bytes():
                    raise ValueError('Production source changed since measured probe: '+str(source))
            run = LOGS/(args.prefix+'-ad-cold20k')
            stage('train-ad', train('train_operator_local_ad', run.name, False))
            stage('validate', ['-m', 'examples.channel_nsch2d.validate_run', '--run', str(run)])
            stage('cohort', ['-m', 'examples.channel_nsch2d.cohort_evaluate', '--run', str(run),
                '--manifest', current['operator']['validation_manifest'], '--output', str(run/'unseen3')])
            stage('finish', support('finish'), cpu=True)
        state.update(status='complete', finished_utc=datetime.now(timezone.utc).isoformat()); save()
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}'); save(); raise


if __name__ == '__main__':
    main()
