"""Finite Spinodal reference/probe/cold50k queue after a verified serial owner."""
import argparse
from datetime import datetime,timezone
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from docs.tasks.electro_threeway_queue import process_ticks,read,write
from docs.tasks.spinodal_threeway_support import METHODS

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True);parser.add_argument('--prefix',required=True)
    parser.add_argument('--wait-journal',type=Path,required=True)
    parser.add_argument('--wait-pid',type=int,required=True);parser.add_argument('--wait-start-ticks',type=int,required=True)
    parser.add_argument('--max-projected-update-hours',type=float,required=True)
    parser.add_argument('--reference-reuse',type=Path)
    parser.add_argument('--reference-finer-mesh',type=float,nargs='+')
    args=parser.parse_args()
    if (args.reference_reuse is None)!=(args.reference_finer_mesh is None):
        raise ValueError('Reference recovery requires both prior bank and explicit finer mesh sizes')
    if not 0<args.max_projected_update_hours<=24:raise ValueError('Explicit update-time ceiling in(0,24]hours required')
    args.bundle.mkdir(parents=True,exist_ok=False)
    state=dict(status='waiting',pid=os.getpid(),prefix=args.prefix,wait_journal=str(args.wait_journal),
        wait_pid=args.wait_pid,wait_start_ticks=args.wait_start_ticks,stages=[],
        started_utc=datetime.now(timezone.utc).isoformat(),max_projected_update_hours=args.max_projected_update_hours)
    if args.reference_reuse is not None:
        state['reference_recovery']=dict(source=str(args.reference_reuse),finer_mesh_sizes=args.reference_finer_mesh)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        JAX_PLATFORMS='cuda',XLA_FLAGS='',JAX_ENABLE_COMPILATION_CACHE='false',
        XLA_PYTHON_CLIENT_PREALLOCATE='true',XLA_PYTHON_CLIENT_MEM_FRACTION='.8',PYTHONUNBUFFERED='1')

    def save():
        state['heartbeat_utc']=datetime.now(timezone.utc).isoformat();write(args.bundle/'journal.json',state)

    def gpu():
        return subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.total,driver_version',
            '--format=csv,noheader'],text=True).strip()

    def stage(name,command,interpreter,platform):
        occupied=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        if occupied or gpu()!=state['gpu']:raise RuntimeError('GPU occupied or changed before serial stage: '+occupied)
        if shutil.disk_usage('/root/autodl-tmp').free<8*1024**3:raise RuntimeError('Less than8GiBdisk headroom; no deletion or launch')
        row=dict(name=name,status='running',command=[interpreter,*command]);state['stages'].append(row);save()
        before=time.perf_counter();child_env=dict(env,JAX_PLATFORMS=platform)
        if interpreter!=sys.executable:
            child_env.pop('VIRTUAL_ENV',None);child_env.pop('PYTHONPATH',None)
        with (args.bundle/(name+'.log')).open('x') as stream:
            child=subprocess.Popen([interpreter,*command],cwd=ROOT,env=child_env,stdout=stream,stderr=subprocess.STDOUT)
            row['pid']=child.pid;save()
            while True:
                try:code=child.wait(timeout=30);break
                except subprocess.TimeoutExpired:save()
        row.update(status='passed' if code==0 else 'failed',exit_code=code,seconds=time.perf_counter()-before);save()
        if code:raise RuntimeError(name+' failed; no later stage or numerical fallback launched')

    def train(method,probe):
        command=['-m','examples.confined_ch2d.train','--seed=42',
            '--config=examples/confined_ch2d/configs/train_operator_compare_'+method+'.py',
            '--config.log.run_name='+args.prefix+('-probe-' if probe else '-')+method]
        if probe:command+=['--config.train.iterations=30']
        return command

    def audit(name):
        return ['-m','docs.tasks.spinodal_threeway_support','--bundle',str(args.bundle),'--prefix',args.prefix,'--stage',name]

    save()
    try:
        while True:
            dependency=read(args.wait_journal)
            if dependency['status']=='complete':break
            if dependency['status'] not in ('waiting','running'):raise RuntimeError('Dependency failed or stopped: '+dependency['status'])
            if process_ticks(args.wait_pid)!=args.wait_start_ticks:raise RuntimeError('Dependency identity changed')
            save();time.sleep(30)
        state.update(status='running',gpu=gpu());save()
        from examples.confined_ch2d.configs.train_operator_compare_nonlocal import get_config
        c=get_config()
        reference=['-m','docs.tasks.spinodal_local_reference',
            '--config','examples.confined_ch2d.configs.train_operator_compare_nonlocal']
        if args.reference_reuse is not None:
            reference[1]='docs.tasks.spinodal_reference_recovery'
            reference+=['--source',str(args.reference_reuse),'--finer-mesh',*map(str,args.reference_finer_mesh)]
        stage('local-reference',reference,c.operator.reference_python,'cpu')
        for method in METHODS:stage('probe-'+method,train(method,True),sys.executable,'cuda')
        stage('resources',audit('resources'),sys.executable,'cpu')
        resources=read(args.bundle/'resources.json')
        if any(r['projected_50k_update_hours']>args.max_projected_update_hours for r in resources['resources']):
            raise RuntimeError('Measured50kupdate estimate exceeds explicit execution ceiling; inspect before long training')
        for method in METHODS:stage('train-'+method,train(method,False),sys.executable,'cuda')
        stage('final-audit',audit('final'),sys.executable,'cpu')
        state.update(status='complete',finished_utc=datetime.now(timezone.utc).isoformat());save()
    except BaseException as error:
        state.update(status='failed',error=f'{type(error).__name__}: {error}');save();raise


if __name__=='__main__':main()
