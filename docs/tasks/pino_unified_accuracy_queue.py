"""Serial canonical Spinodal pilots and full-budget channel sampling controls."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

from docs.tasks.pino_unified_sampling_queue import gpu
from docs.tasks.pino_geometry_sampling_results import paired

ROOT=Path(__file__).resolve().parents[2]
LOGS=Path('/root/autodl-tmp/tf-logs')
COHORT=Path('/root/autodl-tmp/data/channel_nsch2d/random-walls-20261102-02/unseen3/manifest.json')


def potential_audit(paths,output):
    base=paths[0]
    result=[]
    for run in paths:
        for name in ('model_initial.eqx','optimizer_initial.eqx'):
            if (base/name).read_bytes()!=(run/name).read_bytes():
                raise ValueError('Potential cold model/optimizer mismatch: '+str(run))
        with np.load(base/'training_draws.npz') as a,np.load(run/'training_draws.npz') as b:
            if set(a.files)!=set(b.files):raise ValueError('Potential draw schemas differ')
            for key in a.files:np.testing.assert_array_equal(a[key],b[key])
        summary=json.loads((run/'summary.json').read_text())
        if summary['new_updates']!=5000 or summary['start_step']!=0:
            raise ValueError('Potential pilot budget or cold start missing')
        for file in sorted(base.glob('prediction_*.npz')):
            with np.load(file) as a,np.load(run/file.name) as b:
                for key in ('points','volumes','time','holes','phi_reference','mu_reference'):
                    np.testing.assert_array_equal(a[key],b[key])
        with np.load(run/'update_timing.npz') as timing:
            step=timing['update_seconds']+timing['batch_preparation_seconds']
        result.append(dict(run=str(run),final=summary['final_metrics'],
            warm_median_seconds=float(np.median(step[100:])),effective_mean_seconds=float(np.mean(step[100:])),
            wall_seconds=summary['wall_seconds'],peak_bytes=summary['device_memory_stats']['peak_bytes_in_use']))
    output.write_text(json.dumps(dict(status='passed',cold_states_geometry_time_and_reference_equal=True,
        arms=result),indent=2,allow_nan=False)+'\n')


def run(output,after):
    output.mkdir(parents=True,exist_ok=False)
    journal=output/'journal.json'
    state=dict(status='waiting_for_followup',pid=os.getpid(),gpu=gpu(),after=str(after),stages=[],
        started_utc=datetime.now(timezone.utc).isoformat())
    def save():
        temp=journal.with_suffix('.partial')
        temp.write_text(json.dumps(state,indent=2,allow_nan=False)+'\n');temp.replace(journal)
    save()
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
             MPLBACKEND='Agg',PYTHONUNBUFFERED='1')
    def stage(label,command,*,cpu):
        environment=dict(env,JAX_PLATFORMS='cpu' if cpu else 'cuda')
        if cpu:environment['CUDA_VISIBLE_DEVICES']=''
        else:
            card=gpu()
            if card['uuid']!=state['gpu']['uuid'] or card['used_MiB']>1024:
                raise RuntimeError(f'Same idle GPU required: {card}')
            environment.update(XLA_PYTHON_CLIENT_PREALLOCATE='true',XLA_PYTHON_CLIENT_MEM_FRACTION='0.90')
        item=dict(label=label,command=command,platform='cpu' if cpu else 'cuda',status='running',
                  started_utc=datetime.now(timezone.utc).isoformat())
        state['stages'].append(item)
        tick=time.perf_counter()
        with (output/(label+'.log')).open('x') as stream:
            child=subprocess.Popen([sys.executable,*command],cwd=ROOT,env=environment,
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
            item['pid']=child.pid;save();code=child.wait()
        item.update(exit_code=code,status='passed' if code==0 else 'failed',seconds=time.perf_counter()-tick)
        save()
        if code:raise RuntimeError(f'{label} failed; retained log/run, no implicit retries')
    try:
        while True:
            predecessor=json.loads(after.read_text())
            if predecessor['status']=='complete':break
            if predecessor['status']=='failed':raise RuntimeError('Predecessor failed; inspect its retained stage')
            process=Path(f'/proc/{predecessor["pid"]}/cmdline')
            if not process.exists() or b'pino_unified_sampling_followup' not in process.read_bytes():
                predecessor=json.loads(after.read_text())
                if predecessor['status']=='complete':break
                raise RuntimeError('Predecessor handle absent without a successful terminal journal')
            time.sleep(30)
        state['status']='running';save()
        for policy in ('ordinary','deterministic'):
            stage('ceramic-update-replay-'+policy,[
                '-m','docs.tasks.pino_ceramic_update_replay',
                '--source-run',str(LOGS/'particle_corrosion2d/pino-unified-followup-20261112-01-area'),
                '--input-audit',str(LOGS/'pino-unified-repeatability-20261112-01'),
                '--output',str(output/('ceramic-update-replay-'+policy)),
                '--gpu-policy',policy,'--repeats','4'],cpu=False)
        regular=[]
        for case in ('electropolish2d','icemelting3d'):
            name=output.name+'-regular-probe-'+case
            options=['--config.train.iterations=60'] if case=='electropolish2d' else [
                '--config.operator.action=memory','--config.operator.probe_steps=60']
            stage('regular-probe-'+case,['-m','examples.'+case+'.train',
                '--config=examples/'+case+'/configs/train_operator.py','--seed=42',
                '--config.log.run_name='+name,*options],cpu=False)
            path=LOGS/case/name
            timing=json.loads((path/'timing.json').read_text())
            seconds=sum(np.asarray(timing[k]) for k in ('update_seconds','sample_seconds','spatial_seconds'))
            memory=json.loads((path/'memory.json').read_text())
            allocator_key='allocator_stats' if case=='electropolish2d' else 'allocator'
            regular.append(dict(case=case,run=str(path),updates=60,
                warm_step_median_seconds=float(np.median(seconds[10:])),
                peak_active_GiB=max(row[allocator_key]['peak_bytes_in_use'] for row in memory)/1024**3,
                scope='Current default rolling recipe; resource probe, not final accuracy'))
        (output/'regular-resources.json').write_text(json.dumps(dict(status='passed',gpu=state['gpu'],arms=regular),indent=2)+'\n')
        # First prove memory/complete support at canonical B16 and online pool.
        for label,preset,options in (
            ('full','train_operator_potential',[]),
            ('bulk','train_operator_potential_spatial',['--config.sample.spatial.wall_groups=0']),
            ('wall','train_operator_potential_spatial',[])):
            name=output.name+'-potential-probe-'+label
            stage('potential-probe-'+label,['-m','examples.confined_ch2d.train',
                '--config=examples/confined_ch2d/configs/'+preset+'.py','--seed=42',
                '--config.train.iterations=60','--config.log.run_name='+name,*options],cpu=False)
        # Equal initial state/geometry/time draws, native references only at eval.
        potential=[]
        for label,preset in (('full','train_operator_potential'),('wall','train_operator_potential_spatial')):
            name=output.name+'-potential-cold5k-'+label
            from examples.confined_ch2d.configs.train_operator_potential import get_config
            potential.append(Path(get_config().log.log_dir)/name)
            stage('potential-cold5k-'+label,['-m','examples.confined_ch2d.train',
                '--config=examples/confined_ch2d/configs/'+preset+'.py','--seed=42',
                '--config.log.run_name='+name],cpu=False)
        potential_audit(potential,output/'potential-paired.json')
        channel=[]
        for label,preset,options in (
            ('bulk512','train_stratified_spatial',['--config.sample.spatial.wall_groups=0',
                '--config.sample.spatial.node_capacity=10240']),
            ('wall512','train_stratified_spatial',['--config.sample.spatial.node_capacity=10240']),
            ('full','train_spatial_control',[])):
            name=output.name+'-channel-cold20k-'+label
            path=LOGS/'channel_nsch2d'/name
            stage('channel-cold20k-'+label,['-m','examples.channel_nsch2d.train',
                '--config=examples/channel_nsch2d/configs/'+preset+'.py','--seed=42',
                '--config.train.iterations=20000','--config.log.checkpoint_every=5000',
                '--config.log.run_name='+name,*options],cpu=False)
            stage('channel-online-audit-'+label,['-m','docs.tasks.pino_geometry_sampling_results',
                'channel-audit',str(path)],cpu=True)
            stage('channel-reserved-'+label,['-m','examples.channel_nsch2d.cohort_evaluate',
                '--run',str(path),'--manifest',str(COHORT),'--output',str(path/'reserved')],cpu=False)
            channel.append(path)
        paired(channel,'channel',output/'channel-paired.json')
        state['status']='complete'
    except Exception as error:
        state.update(status='failed',failure=str(error));raise
    finally:
        state['finished_utc']=datetime.now(timezone.utc).isoformat();save()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--after',type=Path,required=True)
    args=parser.parse_args();run(args.output,args.after)
