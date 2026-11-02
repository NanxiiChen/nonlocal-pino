"""One conditional cold time-feature control after the prescribed strict continuation."""
from pathlib import Path
import argparse
import json
import importlib
import os
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def main(after_pid,previous_progress,dense_root,tag,single_config,online_config,operator_validation):
    root=Path(__file__).resolve().parents[2];python=root/'.venv/bin/python'
    data=Path('/root/autodl-tmp/data/channel_nsch2d');runs=Path('/root/autodl-tmp/tf-logs/channel_nsch2d')
    progress=data/f'{tag}-progress.json'
    if progress.exists():raise FileExistsError(progress)
    def record(stage,**kwargs):
        progress.write_text(json.dumps(dict(stage=stage,**kwargs),indent=2,allow_nan=False)+'\n')
        print(stage,flush=True)
    def launch(command,label):
        env=dict(os.environ);env['JAX_PLATFORMS']='cuda'
        with (data/f'{tag}-{label}.log').open('x') as log:
            subprocess.run([str(python),*command],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    record('waiting_for_prescribed_continuation_and_audits',pid=after_pid)
    while Path(f'/proc/{after_pid}').exists():time.sleep(2)
    prior=json.loads(Path(previous_progress).read_text())
    if prior['stage']=='online_training_and_input_replay_completed_unseen_validation_required':
        record('previous_recipe_passed_no_new_control_needed',goal_complete=False);return
    if prior['stage']!='single_geometry_target_still_unmet':
        raise ValueError('Previous driver did not complete a valid single-geometry accuracy assessment')
    single=runs/f'{tag}-single-01'
    record('cold_dual_time_single_geometry_training',run=str(single))
    launch(['-m','examples.channel_nsch2d.train',f'--config={single_config}',
        f'--config.log.run_name={single.name}'],'single-train')
    launch(['docs/tasks/channel_single_validate.py','--driver-pid','0','--run',str(single),
        '--baseline',prior['run']],'single-validation')
    gate=Path(dense_root)/'reference-validation.json'
    launch(['docs/tasks/channel_dense_evaluate.py','--run',str(single),'--validation',str(gate)],'single-dense')
    assessment=json.loads((single/'dense-fem-evaluation.json').read_text())
    if not assessment['single_geometry_target_met']:
        record('single_geometry_dual_time_target_still_unmet',run=str(single),assessment=assessment,goal_complete=False);return
    from nonlocal_pinn.training.runtime import with_run_seed
    from examples.channel_nsch2d.online import validate_construction_screen
    selected=with_run_seed(importlib.import_module(online_config.replace('/','.').removesuffix('.py')).get_config(),None)
    if selected.nl.degree>=4 and operator_validation is None:raise ValueError('Higher-order online recipe requires its input-prior screen')
    if operator_validation is not None:validate_construction_screen(selected,operator_validation)
    online=runs/f'{tag}-online-01'
    record('cold_dual_time_online_training',run=str(online),single_geometry_assessment=assessment)
    launch(['-m','examples.channel_nsch2d.train',f'--config={online_config}',
        f'--config.reference.root={dense_root}',f'--config.log.run_name={online.name}'],'online-train')
    for module in ('validate_run','diagnose_phase'):
        launch(['-m',f'examples.channel_nsch2d.{module}','--run',str(online)],f'online-{module}')
    record('online_training_and_input_replay_completed_unseen_validation_required',run=str(online),goal_complete=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--after-pid',required=True,type=int)
    p.add_argument('--previous-progress',required=True);p.add_argument('--dense-root',required=True);p.add_argument('--tag',required=True)
    p.add_argument('--single-config',default='examples/channel_nsch2d/configs/train_single_dual_time.py')
    p.add_argument('--online-config',default='examples/channel_nsch2d/configs/train_operator_dual_time_online.py')
    p.add_argument('--operator-validation')
    a=p.parse_args();main(a.after_pid,a.previous_progress,a.dense_root,a.tag,a.single_config,a.online_config,a.operator_validation)
