"""One detached cold20k/LR channel experiment and finite audits; no FEM/retries."""
import argparse
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main(a):
    from examples.channel_nsch2d.budget_validation import comparison_config, resource_evidence, audit_cold, audit_completed
    from examples.channel_nsch2d.online import validate_construction_screen
    from examples.channel_nsch2d.problem import validate_config, identity

    output = Path(a.output)
    if not output.is_absolute() or Path(a.run_name).name != a.run_name:
        raise ValueError('Absolute output and a simple fresh run name required')
    c = importlib.import_module(a.config).get_config()
    c.train.seed = a.seed
    c.log.run_name = a.run_name
    validate_config(c)
    comparison_config(c, a.control, a.execution_time_block)
    run = Path(c.log.log_dir)/a.run_name
    if run.exists() or output.exists():
        raise FileExistsError('Fresh run and owner output paths are required')
    query = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,memory.total',
                                     '--format=csv,noheader,nounits'], text=True).strip().split(', ')
    gpu = dict(name=query[0], uuid=query[1], total_memory_bytes=int(query[2])*2**20)
    resource = resource_evidence(c, a.control, a.resource_probe, gpu, a.memory_fraction,
                                 a.execution_time_block)
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('GPU occupied; this owner starts no concurrent experiment')
    if shutil.disk_usage(run.parent).free < 4*2**30:
        raise RuntimeError('Require4GiB free; no automatic artifact cleanup')
    gate = json.loads((Path(c.reference.root)/'reference-validation.json').read_text())
    if gate['status'] != 'passed' or gate['identity'] != json.loads(json.dumps(identity(c))):
        raise ValueError('Matching existing development FEM gate required')
    output.mkdir(parents=True)
    journal = output/'process.json'
    (output/'owner-source.py').write_bytes(Path(__file__).read_bytes())
    (output/'planned-config.json').write_text(c.to_json(indent=2)+'\n')
    record = dict(status='preflight_passed', runner_pid=os.getpid(),
                  started_utc=datetime.now(timezone.utc).isoformat(), run=str(run),
                  config=a.config, control=a.control, seed=a.seed, requested_steps=20000,
                  initialization='Cold model and complete fresh SOAP; no checkpoint or optimizer import',
                  execution_time_block=a.execution_time_block,
                  continuous_time_probe_samples=252 if a.execution_time_block == 18 else 240,
                  resource=resource, prepared_config=json.loads(c.to_json()),
                  active_child_pid=None, stages=[], new_FEM_solves=0, automatic_retries=0,
                  scope='One initialization; same reused development/3reserved FEM cases, not a population guarantee')
    env = dict(os.environ, JAX_PLATFORMS='cuda', XLA_PYTHON_CLIENT_MEM_FRACTION=str(a.memory_fraction),
               OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               NUMEXPR_NUM_THREADS='4', PYTHONUNBUFFERED='1')
    cpu = dict(env, JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')

    def save():
        temp = journal.with_suffix('.partial')
        temp.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
        temp.replace(journal)

    def terminate(signum, frame):
        raise SystemExit(128+signum)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)

    def execute(stage, arguments, environment):
        log = output/(stage+'.log')
        with log.open('x') as stream:
            child = subprocess.Popen([str(ROOT/'.venv/bin/python'), *arguments], cwd=ROOT,
                                     env=environment, stdout=stream, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            entry = dict(stage=stage, pid=child.pid, arguments=arguments, log=str(log),
                         started_utc=datetime.now(timezone.utc).isoformat())
            record['stages'].append(entry)
            record.update(status=stage, active_child_pid=child.pid)
            save()
            try:
                while child.poll() is None:
                    record['heartbeat_utc'] = datetime.now(timezone.utc).isoformat()
                    if stage == 'training':
                        for name, key in (('metrics.jsonl', 'latest_monitor'),
                                          ('loss.jsonl', 'latest_loss'),
                                          ('online_refresh.jsonl', 'latest_refresh')):
                            path = run/name
                            if path.is_file():
                                lines = [s for s in path.read_bytes().splitlines(keepends=True) if s.endswith(b'\n')]
                                if lines:
                                    record[key] = json.loads(lines[-1])
                        if 'latest_monitor' in record and 'actual_cold_state' not in record:
                            record['actual_cold_state'] = audit_cold(run, a.control, a.execution_time_block)
                        if 'latest_loss' in record:
                            step = record['latest_loss']['step']
                            record['scheduled_lr_at_last_logged_update'] = max(c.train.lr_min,
                                c.train.lr_init*c.train.lr_decay_rate**((step-1)/c.train.lr_decay_every))
                    save()
                    time.sleep(10)
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    child.wait(timeout=30)
            entry.update(exit_code=child.returncode, completed_utc=datetime.now(timezone.utc).isoformat())
            record['active_child_pid'] = None
            save()
            if child.returncode:
                raise RuntimeError(f'{stage} failed with exit{child.returncode}; see {log}; no retry')

    save()
    try:
        draws = ((c.train.iterations-1)//c.operator.online.refresh_every
                 + c.operator.online.prefetch_batches)*c.operator.online.refresh_size
        screen = output/'input-screen.json'
        execute('input_construction_screen', ['docs/tasks/channel_operator_prior_probe.py',
            '--config', a.config, '--draws', str(draws), '--workers', str(a.screen_workers),
            '--output', str(screen)], cpu)
        record['construction_screen'] = validate_construction_screen(c, screen)
        save()
        if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
            raise RuntimeError('GPU became occupied during input preflight; no automatic wait or overlap')
        execute('training', ['-m', 'examples.channel_nsch2d.train',
            '--config='+a.config.replace('.', '/')+'.py', '--config.log.run_name='+a.run_name,
            '--seed='+str(a.seed)], env)
        summary = json.loads((run/'summary.json').read_text())
        if summary['status'] != 'completed' or summary['iterations'] != 20000:
            raise ValueError('Training did not publish the prescribed20k endpoint')
        if 'actual_cold_state' not in record:
            record['actual_cold_state'] = audit_cold(run, a.control, a.execution_time_block)
        for name in ('validate_run', 'diagnose_phase'):
            execute(name, ['-m', 'examples.channel_nsch2d.'+name, '--run='+str(run)], env)
        comparison = audit_completed(run, a.control, a.execution_time_block)
        (run/'budget-control-validation.json').write_text(json.dumps(comparison, indent=2)+'\n')
        record['budget_control'] = comparison
        save()
        execute('continuous_time_residual', ['-m', 'examples.channel_nsch2d.diagnose_time',
            '--run='+str(run), '--output='+str(run/'continuous-time-residual'),
            '--time-samples='+str(record['continuous_time_probe_samples']), '--probe-seed=20261103'], cpu)
        execute('droplet_shape', ['docs/tasks/channel_droplet_metrics.py',
            '--fields='+str(run/'prediction.npz'), '--output='+str(run/'droplet-shape')], cpu)
        execute('phi_gallery', ['-m', 'examples.channel_nsch2d.evaluate',
            '--fields='+str(run/'prediction.npz'), '--output='+str(run/'phi-fields'),
            '--field=phi', '--times', '0', '1.5', '3', '4.5', '6'], cpu)
        execute('unseen_geometry_evaluation', ['-m', 'examples.channel_nsch2d.cohort_evaluate',
            '--run='+str(run), '--manifest='+c.operator.validation_manifest,
            '--output='+str(run/'unseen3')], env)
        for index in range(3):
            case = run/'unseen3'/f'case_{index:03d}'
            execute(f'unseen_phi_gallery_{index}', ['-m', 'examples.channel_nsch2d.evaluate',
                '--fields='+str(case/'prediction.npz'), '--output='+str(case/'phi-fields'),
                '--field=phi', '--times', '0', '1.5', '3', '4.5', '6'], cpu)
        execute('convergence', ['docs/tasks/channel_random_wall_convergence.py',
            '--run='+str(run), '--output='+str(run/'convergence')], cpu)
        sources = []
        for path in (run/'source').rglob('*.py'):
            relative = path.relative_to(run/'source')
            if path.read_bytes() != (ROOT/relative).read_bytes():
                raise ValueError(f'Trainer source changed while running: {relative}')
            sources.append(str(relative))
        record.update(status='completed_with_saved_run_audits', active_child_pid=None,
                      completed_utc=datetime.now(timezone.utc).isoformat(),
                      final_metrics=summary['metrics'], source_files=sources,
                      finite_cohort=str(run/'unseen3/evaluation.json'))
        save()
    except BaseException as error:
        record.update(status='failed_or_interrupted', error=repr(error), active_child_pid=None)
        save()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('config', 'run-name', 'control', 'resource-probe', 'output'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--memory-fraction', type=float, required=True)
    parser.add_argument('--screen-workers', type=int, required=True)
    parser.add_argument('--execution-time-block', type=int, choices=(8, 18),
                        help='Explicit40/8 or36/18 time sampling/block extension')
    main(parser.parse_args())
