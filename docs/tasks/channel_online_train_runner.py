"""One detached online channel stage and finite saved-field audits; no FEM."""
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


def main(config, run_name, seed, reference_root, construction_screen, comparison_control=None, geometry_control=None,balance_control=None,operator_control=None,random_wall_control=None,capacity_control=None,capacity_probe=None,hardware_trial=False,memory_fraction=None):
    from examples.channel_nsch2d.problem import identity, validate_config
    from examples.channel_nsch2d.cohort import reserved_amplitudes
    from examples.channel_nsch2d.online import validate_construction_screen

    c = importlib.import_module(config).get_config()
    c.train.seed = seed
    c.reference.root = reference_root
    c.log.run_name = run_name
    validate_config(c)
    from examples.channel_nsch2d.initialization import model_initialization_spec
    initialization=model_initialization_spec(c)
    if hardware_trial and capacity_control is None:
        raise ValueError('Hardware execution trial requires the explicit capacity control/probe')
    if memory_fraction is not None and (not hardware_trial or not 0<memory_fraction<=.97):
        raise ValueError('Explicit allocator fraction in(0,.97] requires the hardware execution trial')
    if capacity_control is not None:
        if any(v is not None for v in (comparison_control,geometry_control,balance_control,operator_control,random_wall_control,initialization)):
            raise ValueError('Capacity trial requires an independent cold width/batch/budget comparison')
        if capacity_probe is None or not Path(capacity_probe).is_absolute():
            raise ValueError('Capacity trial requires its absolute qualified resource-probe path')
        if hardware_trial:
            from examples.channel_nsch2d.capacity_validation import execution_config as comparison_config
        else:
            from examples.channel_nsch2d.capacity_validation import comparison_config
        comparison_config(c,capacity_control)
        resource=json.loads((Path(capacity_probe)/'result.json').read_text())
        if hardware_trial:
            gpu_uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
            if resource['hardware_trial']['gpu_uuid']!=gpu_uuid:
                raise ValueError('Hardware resource probe belongs to a different physical GPU')
            if 'allocator_fraction' in resource['hardware_trial']:
                if memory_fraction!=resource['hardware_trial']['allocator_fraction']:
                    raise ValueError('Actual allocator fraction must explicitly match the aggressive probe')
                memory_limit=min(.95*resource['hardware_trial']['total_memory_bytes'],
                    .97*resource['memory']['bytes_limit'])
            else:
                if memory_fraction is not None:raise ValueError('Legacy hardware probe did not qualify an allocator override')
                memory_limit=min(.9*resource['hardware_trial']['total_memory_bytes'],
                    .95*resource['memory']['bytes_limit'])
        else:memory_limit=28*2**30
        actual=json.loads(c.to_json());qualified=resource['config']
        actual['log']['run_name']=qualified['log']['run_name']
        if (resource['status']!='finite_production_probe_passed' or resource['updates']<7
                or resource['memory']['peak_bytes_in_use']>memory_limit
                or actual!=qualified or resource['fitted_probe_weights_exported']
                or not resource['capacity_trial']['full_geometry_pool_on_device']
                or not resource['capacity_trial']['finite_nominal_evaluation']):
            raise ValueError('Capacity resource probe did not qualify this complete cold configuration')
    elif capacity_probe is not None:
        raise ValueError('Capacity probe belongs only to the declared capacity comparison')
    elif random_wall_control is not None:
        if any(v is not None for v in (comparison_control,geometry_control,balance_control,operator_control,initialization)):
            raise ValueError('Random-wall control requires a cold geometry-family extension only')
        from examples.channel_nsch2d.random_wall_validation import comparison_config
        comparison_config(c,random_wall_control)
    elif operator_control is not None:
        if any(v is not None for v in (comparison_control,geometry_control,balance_control,initialization)):
            raise ValueError('Operator control requires an independent cold derivative-only comparison')
        from examples.channel_nsch2d.integral_validation import comparison_config
        comparison_config(c,operator_control)
    elif balance_control is not None:
        if comparison_control is not None or geometry_control is not None or initialization is not None:
            raise ValueError('Balance control requires cold training and no other comparison')
        from examples.channel_nsch2d.balance_validation import comparison_config
        comparison_config(c,balance_control)
    elif geometry_control is not None:
        if comparison_control is not None or initialization is not None:
            raise ValueError('Geometry control requires cold training and no same-geometry comparison')
        from ml_collections import ConfigDict
        from docs.tasks.channel_geometry_transfer import check_transfer
        parent=Path(geometry_control)
        if not parent.is_absolute():raise ValueError('Geometry control requires an absolute source run')
        check_transfer(ConfigDict(json.loads((parent/'config.json').read_text())),c)
        if json.loads((parent/'validation.json').read_text())['status']!='replay_and_hard_constraints_passed':
            raise ValueError('A completed audited geometry source is required')
    elif 'causal' in c or 'time_derivative'in c:
        if comparison_control is None:raise ValueError('Causal/timeJVP run requires an explicit audited comparison control')
        from examples.channel_nsch2d.causal import comparison_config
        comparison_config(c,comparison_control)
    elif comparison_control is not None:
        raise ValueError('Comparison-control argument belongs to the explicit causal/timeJVP pilot')
    if ('online' not in c.operator or 'continuation' in c.train or 'refinement' in c.train
            or 'droplet_metric' in c or 'phase_metric' in c or 'residual_transform' in c):
        raise ValueError('This finite cold online pilot requires original residuals and fresh optimizer state')
    if capacity_control is None and (c.operator.train_conditions != 16 or c.operator.condition_batch != 2
            or (c.train.iterations,c.sample.Nt) not in ((20000,4),(5000,40))):
        raise ValueError('Finite pilot requires16pool/B2 and either20k/Nt4 or5k/Nt40')
    if capacity_control is None and c.sample.Nt==40 and ('microbatch' not in c.train
            or c.train.microbatch.to_dict()!=dict(kind='time',size=8)):
        raise ValueError('The40-time pilot requires the explicitly validated8-time accumulation')
    if not Path(reference_root).is_absolute() or not Path(construction_screen).is_absolute():
        raise ValueError('Reference and input-screen paths must be absolute')
    gate = json.loads((Path(reference_root)/'reference-validation.json').read_text())
    if gate['status'] != 'passed' or gate['identity'] != json.loads(json.dumps(identity(c))):
        raise ValueError('Matching existing development reference gate is required')
    screen = validate_construction_screen(c, construction_screen)
    reserved = reserved_amplitudes(c)
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
            '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('GPU is occupied; do not overlap another experiment')
    run = Path(c.log.log_dir)/run_name
    data = Path('/root/autodl-tmp/data/channel_nsch2d')
    journal = data/(run_name+'-process.json')
    if run.exists() or journal.exists():
        raise FileExistsError('A fresh run name and process journal are required')
    if shutil.disk_usage(run.parent).free < 2**30:
        raise RuntimeError('Require1GiBfree for input/state snapshots and finite diagnostics')
    env = dict(os.environ, JAX_PLATFORMS='cuda', OPENBLAS_NUM_THREADS='4',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4', PYTHONUNBUFFERED='1')
    if memory_fraction is not None:env['XLA_PYTHON_CLIENT_MEM_FRACTION']=str(memory_fraction)
    record = dict(status='preflight_passed', runner_pid=os.getpid(), run=str(run),
        started_utc=datetime.now(timezone.utc).isoformat(), config=config, seed=seed,
        initialization=(initialization if initialization is not None else 'cold; no checkpoint or optimizer import'), requested_steps=c.train.iterations,
        time_samples_per_update=c.sample.Nt,
        comparison_control=comparison_control,
        geometry_control=geometry_control,
        balance_control=balance_control,
        operator_control=operator_control,
        random_wall_control=random_wall_control,
        capacity_control=capacity_control,capacity_probe=capacity_probe,
        hardware_trial=hardware_trial,memory_fraction=memory_fraction,
        construction_screen=construction_screen, screen_summary=screen,
        reserved_geometry_count=len(reserved), prepared_config=json.loads(c.to_json()),
        stages=[], automatic_goal=None, new_FEM_solves=0,
        scope='One user-authorized finite online run; no old queue, retries, FEM or further experiments; nominal geometry metrics are not population accuracy')
    owner_source=data/(run_name+'-owner.py')
    with owner_source.open('xb') as stream:stream.write(Path(__file__).read_bytes())
    record['owner_source']=str(owner_source)
    if geometry_control is not None:
        helper=data/(run_name+'-geometry-control.py')
        with helper.open('xb')as stream:stream.write((ROOT/'docs/tasks/channel_geometry_transfer.py').read_bytes())
        record['geometry_control_source']=str(helper)

    def write():
        temporary = journal.with_suffix('.json.partial')
        temporary.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
        os.replace(temporary, journal)

    def terminate(signum, frame):
        raise SystemExit(128+signum)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)

    def execute(stage, arguments, environment):
        log = data/(run_name+'-'+stage+'.log')
        with log.open('x') as stream:
            child = subprocess.Popen([str(ROOT/'.venv/bin/python'), *arguments],
                cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            entry = dict(stage=stage, pid=child.pid, arguments=arguments, log=str(log),
                         started_utc=datetime.now(timezone.utc).isoformat())
            record['stages'].append(entry)
            record.update(status=stage, active_child_pid=child.pid); write()
            try:
                while child.poll() is None:
                    record['heartbeat_utc'] = datetime.now(timezone.utc).isoformat()
                    if stage == 'training':
                        for name, key in (('metrics.jsonl', 'latest_monitor'), ('loss.jsonl', 'latest_loss'),
                                          ('online_refresh.jsonl', 'latest_refresh')):
                            path = run/name
                            if path.is_file():
                                complete = [s for s in path.read_bytes().splitlines(keepends=True) if s.endswith(b'\n')]
                                if complete:
                                    record[key] = json.loads(complete[-1])
                        parity_key='model_stage_state_parity'if initialization is not None else 'cold_state_parity'
                        if any(v is not None for v in (comparison_control,geometry_control,balance_control,operator_control,random_wall_control)) and 'latest_monitor' in record and parity_key not in record:
                            import numpy as np
                            original=Path(random_wall_control if random_wall_control is not None else
                                operator_control if operator_control is not None else
                                balance_control if balance_control is not None else
                                comparison_control if comparison_control is not None else geometry_control)
                            if initialization is not None:
                                from examples.channel_nsch2d.initialization import audit_model_initialization
                                record['model_initialization_parity']=audit_model_initialization(run,c)
                            else:
                                for name in ('model_initial.eqx','optimizer_initial.eqx'):
                                    if (run/name).read_bytes()!=(original/name).read_bytes():
                                        raise ValueError(f'Actual causal cold state differs: {name}')
                            if operator_control is not None:
                                from examples.channel_nsch2d.integral_validation import audit_cold_inputs
                                record[parity_key]=audit_cold_inputs(run,operator_control)
                            elif geometry_control is None and random_wall_control is None:
                                with np.load(run/'known_inputs.npz',allow_pickle=False) as x,np.load(original/'known_inputs.npz',allow_pickle=False) as y:
                                    if set(x.files)!=set(y.files):raise ValueError('Actual causal initial input schema differs')
                                    for name in x.files:np.testing.assert_array_equal(x[name],y[name])
                            if operator_control is None:
                                record[parity_key]=dict(status='passed',cold=initialization is None,
                                    fresh_geometry_bank_exact=geometry_control is None and random_wall_control is None,first_monitor_step=record['latest_monitor']['step'])
                            if operator_control is not None:
                                record[parity_key]['cold']=True
                            elif geometry_control is not None or random_wall_control is not None:
                                record[parity_key].update(initial_model_complete_optimizer_exact=True,
                                    geometry_inputs_intentionally_changed=True)
                            elif initialization is None:
                                record[parity_key]['initial_model_complete_optimizer_and_geometry_bank_exact']=True
                            else:
                                record[parity_key].update(initial_model_exactly_parent_final=True,
                                    complete_optimizer_exactly_fresh=True,optimizer_and_pool_counters_resumed=False)
                        if capacity_control is not None and 'latest_monitor' in record and 'capacity_cold_state' not in record:
                            from examples.channel_nsch2d.capacity_validation import audit_probe_state
                            record['capacity_cold_state']=audit_probe_state(run,capacity_probe)
                    write(); time.sleep(15)
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    child.wait(timeout=30)
            entry.update(exit_code=child.returncode, completed_utc=datetime.now(timezone.utc).isoformat())
            record['active_child_pid'] = None; write()
            if child.returncode:
                raise RuntimeError(f'{stage} failed: exit{child.returncode}; see {log}')

    write()
    try:
        execute('training', ['-m', 'examples.channel_nsch2d.train',
            '--config='+config.replace('.', '/')+'.py', '--config.reference.root='+reference_root,
            '--config.log.run_name='+run_name, '--seed='+str(seed)], env)
        summary = json.loads((run/'summary.json').read_text())
        if summary['status'] != 'completed' or summary['iterations'] != c.train.iterations:
            raise ValueError('Training did not publish the prescribed complete endpoint')
        for module in ('validate_run', 'diagnose_phase'):
            execute(module, ['-m', 'examples.channel_nsch2d.'+module, '--run='+str(run)], env)
        cpu = dict(env, JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
        execute('droplet_shape', ['docs/tasks/channel_droplet_metrics.py', '--fields='+str(run/'prediction.npz'),
            '--output='+str(run/'droplet-shape')], cpu)
        execute('phi_gallery', ['-m', 'examples.channel_nsch2d.evaluate', '--fields='+str(run/'prediction.npz'),
            '--output='+str(run/'phi-fields'), '--field=phi', '--times', '0', '1.5', '3', '4.5', '6'], cpu)
        if comparison_control is not None:
            from examples.channel_nsch2d.causal import audit_comparison
            matched=audit_comparison(run,comparison_control)
            (run/'matched-control-validation.json').write_text(json.dumps(matched,indent=2,allow_nan=False)+'\n')
            record['matched_control']=matched;write()
        if geometry_control is not None:
            from docs.tasks.channel_geometry_transfer import audit_cold_geometry
            matched=audit_cold_geometry(run,geometry_control)
            (run/'geometry-control-validation.json').write_text(json.dumps(matched,indent=2,allow_nan=False)+'\n')
            record['geometry_control_validation']=matched;write()
        if balance_control is not None:
            from examples.channel_nsch2d.balance_validation import audit_comparison
            matched=audit_comparison(run,balance_control)
            (run/'balance-control-validation.json').write_text(json.dumps(matched,indent=2,allow_nan=False)+'\n')
            record['balance_control_validation']=matched;write()
        if operator_control is not None:
            from examples.channel_nsch2d.integral_validation import audit_comparison
            matched=audit_comparison(run,operator_control)
            (run/'operator-control-validation.json').write_text(json.dumps(matched,indent=2,allow_nan=False)+'\n')
            record['operator_control_validation']=matched;write()
        if random_wall_control is not None:
            from examples.channel_nsch2d.random_wall_validation import audit_comparison
            matched=audit_comparison(run,random_wall_control)
            (run/'random-wall-control-validation.json').write_text(json.dumps(matched,indent=2,allow_nan=False)+'\n')
            record['random_wall_control_validation']=matched;write()
        if capacity_control is not None:
            from examples.channel_nsch2d.capacity_validation import audit_comparison
            matched=audit_comparison(run,capacity_control,capacity_probe,hardware_trial=hardware_trial)
            (run/'capacity-control-validation.json').write_text(json.dumps(matched,indent=2,allow_nan=False)+'\n')
            record['capacity_control_validation']=matched;write()
        if 'time_derivative' in c:
            execute('continuous_time_residual',['-m','examples.channel_nsch2d.diagnose_time',
                '--run='+str(run),'--output='+str(run/'continuous-time-residual'),
                '--time-samples=240','--probe-seed=20261102'],cpu)
        elif c.sample.Nt==40:
            for name in ('channel_pde_bc_diagnosis.py','channel_frozen_ch.py'):
                target=run/'source/docs/tasks'/name;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(ROOT/'docs/tasks'/name,target)
            execute('full_time_residual', ['docs/tasks/channel_pde_bc_diagnosis.py',
                '--run='+str(run),'--output='+str(run/'full-time-residual'),
                '--residual-only'],cpu)
        if random_wall_control is not None or capacity_control is not None:
            execute('unseen_geometry_evaluation',['-m','examples.channel_nsch2d.cohort_evaluate',
                '--run='+str(run),'--manifest='+c.operator.validation_manifest,'--output='+str(run/'unseen3')],env)
            for index in range(3):
                case=run/'unseen3'/f'case_{index:03d}'
                execute(f'unseen_phi_gallery_{index}',['-m','examples.channel_nsch2d.evaluate',
                    '--fields='+str(case/'prediction.npz'),'--output='+str(case/'phi-fields'),
                    '--field=phi','--times','0','1.5','3','4.5','6'],cpu)
            record['finite_cohort']=dict(geometries=3,evaluation=str(run/'unseen3'/'evaluation.json'),
                scope='Predeclared held-out conditional-prior draws; no continuous-prior guarantee')
        if capacity_control is not None:
            sources=[]
            for saved_source in (run/'source').rglob('*.py'):
                relative=saved_source.relative_to(run/'source')
                if saved_source.read_bytes()!=(ROOT/relative).read_bytes():
                    raise ValueError(f'Capacity training source changed: {relative}')
                sources.append(str(relative))
            record.update(source_byte_checks=len(sources),source_files=sources)
        record.update(status='completed_with_saved_run_audits', completed_utc=datetime.now(timezone.utc).isoformat(),
            active_child_pid=None, final_metrics=summary['metrics'], online=summary['online'],
            shape_report=str(run/'droplet-shape/metrics.json'), population_accuracy_validated=False)
        write()
    except BaseException as error:
        record.update(status='failed_or_interrupted', error=repr(error), active_child_pid=None)
        write(); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True); parser.add_argument('--run-name', required=True)
    parser.add_argument('--seed', required=True, type=int); parser.add_argument('--reference-root', required=True)
    parser.add_argument('--construction-screen', required=True)
    parser.add_argument('--comparison-control',help='Required for a matched causal/timeJVP pilot')
    parser.add_argument('--geometry-control',help='Completed source for an explicit geometry-only cold study')
    parser.add_argument('--balance-control',help='Completed fixed-scale source for the nine-term adaptive control')
    parser.add_argument('--operator-control',help='Completed GMLS source for the integral derivative control')
    parser.add_argument('--random-wall-control',help='Completed integral single-bump control for the explicit random-wall extension')
    parser.add_argument('--capacity-control',help='Audited random-wall128/B2/5k control for the192/B8/10k trial')
    parser.add_argument('--capacity-probe',help='Qualified production-grid capacity resource probe')
    parser.add_argument('--hardware-trial',action='store_true',help='Use a qualified40-time divisor and this physical GPU memory limit')
    parser.add_argument('--memory-fraction',type=float,help='Explicit allocator fraction, required for an aggressive resource probe')
    args = parser.parse_args()
    main(args.config, args.run_name, args.seed, args.reference_root, args.construction_screen,args.comparison_control,args.geometry_control,args.balance_control,args.operator_control,args.random_wall_control,args.capacity_control,args.capacity_probe,args.hardware_trial,args.memory_fraction)
