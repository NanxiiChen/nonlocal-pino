"""Finite canonical single-geometry control, then conditional cold online study."""
from pathlib import Path
import argparse
import json
import importlib
import os
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def main(tag,single_config,online_config,after_pid,previous_progress,baseline_run):
    root=Path(__file__).resolve().parents[2];python=str(root/'.venv/bin/python')
    data=Path('/root/autodl-tmp/data/channel_nsch2d');runs=Path('/root/autodl-tmp/tf-logs/channel_nsch2d')
    progress=data/f'{tag}-progress.json'
    if progress.exists():raise FileExistsError(progress)
    def record(stage,**extra):
        progress.write_text(json.dumps(dict(stage=stage,**extra),indent=2,allow_nan=False)+'\n')
        print(stage,flush=True)
    def launch(command,label):
        env=dict(os.environ);env['JAX_PLATFORMS']='cuda'
        with (data/f'{tag}-{label}.log').open('x') as log:
            subprocess.run([python,*command],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    if after_pid and previous_progress is None:raise ValueError('A waiting follow-up requires the previous finite driver progress')
    record('waiting_for_previous_finite_driver',pid=after_pid)
    while Path(f'/proc/{after_pid}').exists():time.sleep(2)
    if previous_progress is not None:
        prior=json.loads(Path(previous_progress).read_text())
        if prior['stage']=='online_training_and_input_replay_completed_unseen_validation_required':
            record('previous_recipe_proceeded_no_new_control',goal_complete=False);return
        if prior['stage']!='single_geometry_target_still_unmet':raise ValueError('Previous accuracy gate is not a completed failure')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise RuntimeError('Finish existing GPU work before the coupled refinement control')
    parent=Path(baseline_run);single=runs/f'{tag}-single-01'
    if not parent.is_absolute():raise ValueError('Reporting baseline must be an explicit absolute run path')
    for name in ('config.json','prediction.npz','metrics.jsonl'):
        if not (parent/name).is_file():raise FileNotFoundError(parent/name)
    selected=importlib.import_module(single_config.replace('/','.').removesuffix('.py')).get_config()
    record('coupled_full_time_lbfgs' if 'refinement' in selected.train else 'single_geometry_cold_training',run=str(single))
    launch(['-m','examples.channel_nsch2d.train',f'--config={single_config}',
        f'--config.log.run_name={single.name}'],'single-train')
    launch(['docs/tasks/channel_single_validate.py','--driver-pid','0','--run',str(single),'--baseline',str(parent)],'single-validation')
    gate=data/'dense-reference-20261101/reference-validation.json'
    launch(['docs/tasks/channel_dense_evaluate.py','--run',str(single),'--validation',str(gate)],'single-dense')
    assessment=json.loads((single/'dense-fem-evaluation.json').read_text())
    if not assessment['single_geometry_target_met']:
        record('single_geometry_target_still_unmet',run=str(single),assessment=assessment,goal_complete=False);return
    online=runs/f'{tag}-online-01'
    record('cold_online_training',run=str(online),single_geometry_assessment=assessment)
    launch(['-m','examples.channel_nsch2d.train',f'--config={online_config}',
        f'--config.reference.root={data / "dense-reference-20261101"}',f'--config.log.run_name={online.name}'],'online-train')
    for name in ('validate_run','diagnose_phase'):
        launch(['-m',f'examples.channel_nsch2d.{name}','--run',str(online)],f'online-{name}')
    record('online_training_and_input_replay_completed_unseen_validation_required',run=str(online),goal_complete=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True)
    p.add_argument('--single-config',default='examples/channel_nsch2d/configs/train_single_lbfgs.py')
    p.add_argument('--online-config',default='examples/channel_nsch2d/configs/train_operator_time_refined_online.py')
    p.add_argument('--after-pid',type=int,default=0);p.add_argument('--previous-progress')
    p.add_argument('--baseline-run',default='/root/autodl-tmp/tf-logs/channel_nsch2d/20261101-5090-time-refine-single-time-01')
    a=p.parse_args();main(a.tag,a.single_config,a.online_config,a.after_pid,a.previous_progress,a.baseline_run)
