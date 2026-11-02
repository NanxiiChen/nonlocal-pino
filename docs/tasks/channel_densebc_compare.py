"""Two finite cold20k BC/coupling controls, followed by paired-input checks.

No parameter search, reference solve, continuation, retry or default promotion.
The existing per-run owner performs canonical saved-run/input/shape audits.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def run(a):
    import numpy as np
    checked=json.loads(Path(a.preflight).read_text())
    if (checked['status']!='passed' or checked['control']!=a.control
            or checked['seed']!=a.seed or checked['physical_fields_maximum_gap']!=0
            or not checked['encoder_and_unchanged_domain_arrays_exact']
            or checked['existing_laplacian_prior_screen']!=a.construction_screen):
        raise ValueError('Matching production boundary/input/field preflight must pass')
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False)
    record=dict(status='preparing_finite_pair',runner_pid=os.getpid(),
        started_utc=datetime.now(timezone.utc).isoformat(),seed=a.seed,
        steps_per_arm=20000,arms=[],automatic_goal=None,new_reference_solves=0,
        existing_control=a.control,preflight=a.preflight,
        scope='Two cold runs on reused development reference; not population or multi-seed accuracy')
    def save():
        p=output/'status.partial';p.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
        p.replace(output/'status.json')
    save()
    def terminate(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
    (output/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    configs=(('uniform','train_operator_width128_gmls_densebc_online',a.uniform_name),
             ('balanced','train_operator_width128_gmls_densebc_balanced_online',a.balanced_name))
    try:
        for label,config,name in configs:
            command=[sys.executable,'-u','docs/tasks/channel_online_train_runner.py',
                '--config=examples.channel_nsch2d.configs.'+config,'--run-name='+name,
                '--seed='+str(a.seed),'--reference-root='+a.reference_root,
                '--construction-screen='+a.construction_screen]
            entry=dict(arm=label,command=command,started_utc=datetime.now(timezone.utc).isoformat())
            record['arms'].append(entry);record['status']='running_'+label;save()
            with (output/(label+'-runner.log')).open('x') as stream:
                child=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                    start_new_session=True)
                entry['runner_pid']=child.pid;save()
                try:code=child.wait()
                finally:
                    if child.poll() is None:
                        os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=45)
                if code:raise subprocess.CalledProcessError(code,command)
            journal=Path('/root/autodl-tmp/data/channel_nsch2d')/(name+'-process.json')
            j=json.loads(journal.read_text())
            if j['status']!='completed_with_saved_run_audits':raise ValueError('Missing complete per-run canonical audits')
            entry.update(status='completed_with_saved_run_audits',run=j['run'],journal=str(journal),
                final_metrics=j['final_metrics'],completed_utc=datetime.now(timezone.utc).isoformat())
            save()
        uniform,balanced=[Path(e['run']) for e in record['arms']]
        control=Path(a.control)
        for name in ('model_initial.eqx','optimizer_initial.eqx'):
            if not (uniform/name).read_bytes()==(balanced/name).read_bytes()==(control/name).read_bytes():
                raise ValueError('Cold initial model/SOAP state must match all three controls exactly')
        for other in (balanced,control):
            with np.load(uniform/'sampling.npz') as x,np.load(other/'sampling.npz') as y:
                if set(x.files)!=set(y.files):raise ValueError('Actual sampling fields differ')
                for k in x.files:np.testing.assert_array_equal(x[k],y[k])
        with np.load(uniform/'known_inputs.npz') as x,np.load(balanced/'known_inputs.npz') as y:
            if set(x.files)!=set(y.files):raise ValueError('Known-input schemas differ')
            for k in x.files:np.testing.assert_array_equal(x[k],y[k])
        comparisons=[]
        for label,p in [('original_midpoint_uniform',control),('densebc_uniform',uniform),('densebc_balanced',balanced)]:
            s=json.loads((p/'summary.json').read_text());shape=json.loads((p/'droplet-shape/metrics.json').read_text())
            comparisons.append(dict(label=label,run=str(p),summary=s,shape=shape))
        record.update(status='finite_pair_completed_with_exact_initial_and_actual_input_parity',
            comparisons=comparisons,exact_initial_model_and_SOAP=True,
            all_20000_update_condition_and_time_arrays_identical=True,
            condition_draws_per_arm=40000,time_draws_per_arm=80000,
            completed_utc=datetime.now(timezone.utc).isoformat())
        save()
    except BaseException as error:
        record.update(status='failed_or_interrupted_no_retry',error=repr(error));save();raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('output','control','uniform-name','balanced-name','reference-root','construction-screen','preflight'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--seed',required=True,type=int)
    run(p.parse_args())
