"""Finite serial goal driver: dense assessment, optional strict continuation, online training."""
from pathlib import Path
import argparse
import json
import importlib
import os
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def main(after_pid,refined,dense_root,tag,continuation_config,online_config,operator_validation,operator_validation_pid):
    root=Path(__file__).resolve().parents[2];python=root/'.venv/bin/python'
    refined=Path(refined);dense_root=Path(dense_root)
    # This task recipe deliberately names one complete parent; never import a
    # different checkpoint merely because the waiting driver points elsewhere.
    get_config=importlib.import_module(continuation_config.replace('/','.').removesuffix('.py')).get_config
    if Path(get_config().train.continuation.parent_run)!=refined:
        raise ValueError('Queued continuation recipe names a different refined parent')
    kind=get_config().train.continuation.kind
    if kind=='operator' and (operator_validation is None or operator_validation_pid is None):
        raise ValueError('Operator refinement requires an explicit input-prior validation file and process')
    data=Path('/root/autodl-tmp/data/channel_nsch2d');runs=Path('/root/autodl-tmp/tf-logs/channel_nsch2d')
    progress=data/f'{tag}-progress.json'
    if progress.exists():raise FileExistsError(progress)
    def record(stage,**fields):
        progress.write_text(json.dumps(dict(stage=stage,**fields),indent=2,allow_nan=False)+'\n')
        print(stage,flush=True)
    def launch(command,label):
        env=dict(os.environ);env['JAX_PLATFORMS']='cuda'
        with (data/f'{tag}-{label}.log').open('x') as log:
            subprocess.run([str(python),*command],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    record('waiting_for_dense_reference_driver',pid=after_pid)
    while Path(f'/proc/{after_pid}').exists():time.sleep(2)
    gate=dense_root/'reference-validation.json'
    if json.loads(gate.read_text())['status']!='passed':raise ValueError('Dense independent FEM gate failed')
    if not (refined/'dense-fem-figures').is_dir():raise ValueError('Dense driver did not finish frozen-model evaluation/plots')
    current=refined;assessment=json.loads((current/'dense-fem-evaluation.json').read_text())
    if not assessment['single_geometry_target_met']:
        suffix={'operator':'operator','time_refine':'time','strict':'long','refine':'refined'}[kind]
        current=runs/f'{tag}-single-{suffix}-01'
        record('single_geometry_continuation',kind=kind,run=str(current),previous_assessment=assessment)
        launch(['-m','examples.channel_nsch2d.train',f'--config={continuation_config}',
            f'--config.log.run_name={current.name}'],'single-long-train')
        # The subprocess driver can wait for PID0 (not an actual /proc entry)
        # because synchronous training has already exited successfully.
        launch(['docs/tasks/channel_single_validate.py','--driver-pid','0','--run',str(current),
            '--baseline',str(refined)],'single-long-validation')
        launch(['docs/tasks/channel_dense_evaluate.py','--run',str(current),'--validation',str(gate)],'single-long-dense')
        assessment=json.loads((current/'dense-fem-evaluation.json').read_text())
    if not assessment['single_geometry_target_met']:
        record('single_geometry_target_still_unmet',run=str(current),assessment=assessment,
            online_training_started=False,goal_complete=False)
        return
    if operator_validation is not None:
        if operator_validation_pid is None:raise ValueError('Operator validation needs an explicit process handle')
        record('waiting_for_input_prior_validation',pid=operator_validation_pid)
        while Path(f'/proc/{operator_validation_pid}').exists():time.sleep(2)
        from nonlocal_pinn.training.runtime import with_run_seed
        from examples.channel_nsch2d.online import validate_construction_screen
        selected=with_run_seed(importlib.import_module(online_config.replace('/','.').removesuffix('.py')).get_config(),None)
        validate_construction_screen(selected,operator_validation)
    online=runs/f'{tag}-online-01'
    record('cold_online_training',single_geometry_run=str(current),assessment=assessment,run=str(online))
    launch(['-m','examples.channel_nsch2d.train',f'--config={online_config}',
        f'--config.reference.root={dense_root}',f'--config.log.run_name={online.name}'],'online-train')
    for module in ('validate_run','diagnose_phase'):
        launch(['-m',f'examples.channel_nsch2d.{module}','--run',str(online)],f'online-{module}')
    record('online_training_and_input_replay_completed_unseen_validation_required',run=str(online),
        summary=json.loads((online/'summary.json').read_text()),goal_complete=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--after-pid',required=True,type=int)
    p.add_argument('--refined',required=True);p.add_argument('--dense-root',required=True);p.add_argument('--tag',required=True)
    p.add_argument('--continuation-config',default='examples/channel_nsch2d/configs/train_single_conservative_long.py')
    p.add_argument('--online-config',default='examples/channel_nsch2d/configs/train_operator_conservative_online.py')
    p.add_argument('--operator-validation');p.add_argument('--operator-validation-pid',type=int)
    a=p.parse_args();main(a.after_pid,a.refined,a.dense_root,a.tag,a.continuation_config,a.online_config,a.operator_validation,a.operator_validation_pid)
