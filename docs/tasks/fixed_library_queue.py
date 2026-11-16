"""Finite isolated probes, eight cold runs, independent audits and final figure."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from docs.tasks.fixed_library_prepare import SIZES,CERAMIC_REFERENCE,CERAMIC_DEV

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True);parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--prefix',required=True);a=parser.parse_args()
    inputs=a.inputs.resolve(strict=True);prepared=json.loads((inputs/'preparation.json').read_text())
    if prepared['status']!='passed' or prepared['sizes']!=list(SIZES):raise ValueError('Require the complete nested512input preparation')
    bundle=a.bundle.resolve();bundle.mkdir(parents=True,exist_ok=False);source=bundle/'source'
    ignored=shutil.ignore_patterns('__pycache__','*.pyc')
    for name in ('src','examples/particle_corrosion2d','examples/electropolish2d','examples/corrosion2d'):
        shutil.copytree(ROOT/name,source/name,ignore=ignored)
    for name in ('docs/tasks/ceramic_temporal_probe.py','docs/tasks/fixed_library_prepare.py',
                 'docs/tasks/fixed_library_audit.py','docs/tasks/fixed_library_figures.py',
                 'docs/tasks/fixed_library_queue.py','docs/tasks/electro_spatial_evaluate.py',
                 'docs/tasks/ceramic_fem_comparison_figures.py'):
        target=source/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    configs=source/'study_configs';configs.mkdir();(configs/'__init__.py').write_text('')
    runs={case:[] for case in ('electropolish','ceramic')}
    for count in SIZES:
        name=f'{a.prefix}-electro-n{count}'
        runs['electropolish'].append(str(Path('/root/autodl-tmp/tf-logs/electropolish2d')/name))
        text=("from ml_collections import ConfigDict\n"
            "from examples.electropolish2d.configs.train_operator_threeway import get_config as baseline\n"
            "def get_config():\n    c=baseline()\n"
            f"    c.operator.condition_library=ConfigDict(dict(path={str(inputs/'electropolish')!r},count={count},sampling='uniform_with_replacement'))\n"
            f"    c.log.run_name={name!r}\n    return c\n")
        (configs/f'electro_{count}.py').write_text(text)
        name=f'{a.prefix}-ceramic-n{count}'
        runs['ceramic'].append(str(Path('/root/autodl-tmp/tf-logs/particle_corrosion2d')/name))
        (configs/f'ceramic_{count}.py').write_text(
            'from examples.particle_corrosion2d.configs.train_fixed_geometry import get_config as baseline\n'
            f'def get_config():\n    c=baseline()\n    c.operator.geometry_count={count}\n    return c\n')
    (configs/'electro_probe.py').write_text(
        'from study_configs.electro_512 import get_config as baseline\n'
        'def get_config():\n    c=baseline()\n    c.train.iterations=51\n'
        f'    c.log.run_name={a.prefix+"-electro-probe"!r}\n    c.log.checkpoint_step=0\n    return c\n')
    def gpu():return subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.total,driver_version',
                                             '--format=csv,noheader'],text=True).strip()
    state=dict(status='running',pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat(),
        source=str(source),inputs=str(inputs),gpu=gpu(),sizes=list(SIZES),runs=runs,stages=[],
        budget=dict(electropolish=10000,ceramic=20000),seed=42,
        metric='Equal-point full-space-time per-channel norms, then channel/sample mean',
        uncertainty='Across64samples,ddof1standard deviation',automatic_retries=False)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        JAX_ENABLE_X64='false',JAX_ENABLE_COMPILATION_CACHE='false',XLA_FLAGS='',
        XLA_PYTHON_CLIENT_PREALLOCATE='true',XLA_PYTHON_CLIENT_MEM_FRACTION='.90',
        PYTHONPATH=str(source)+':'+str(source/'src')+':/tmp/pino-pdf-audit',PYTHONUNBUFFERED='1')
    def save():
        p=bundle/'journal.tmp';p.write_text(json.dumps(state,indent=2,allow_nan=False)+'\n');p.replace(bundle/'journal.json')
    def stage(name,args,cpu=False):
        if not cpu:
            busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            if gpu()!=state['gpu'] or busy:raise RuntimeError('GPU changed or occupied before isolated stage: '+busy)
        row=dict(name=name,args=args,status='running',cpu=cpu);state['stages'].append(row);save()
        with (bundle/(name+'.log')).open('x') as log:
            start=time.monotonic_ns()
            child=subprocess.Popen([sys.executable,*args],cwd=source,stdout=log,stderr=subprocess.STDOUT,
                env=dict(env,JAX_PLATFORMS='cpu' if cpu else 'cuda',CERAMIC_COLD_START_MONOTONIC_NS=str(start)))
            row.update(pid=child.pid,started_utc=datetime.now(timezone.utc).isoformat());save()
            code=child.wait();row['seconds']=(time.monotonic_ns()-start)/1e9
        row.update(exit_code=code,status='passed' if code==0 else 'failed');save()
        if code:raise RuntimeError(name+' failed; no subsequent stage or automatic retry')
    save()
    try:
        stage('probe-ceramic-512',['-m','docs.tasks.ceramic_temporal_probe','--config','study_configs.ceramic_512',
            '--inputs',str(inputs/'ceramic-512'),'--reference',str(CERAMIC_REFERENCE),
            '--max-active-gib','85','--output',str(bundle/'ceramic-probe.json')])
        stage('probe-electropolish-512',['-m','examples.electropolish2d.train',
            '--config='+str(configs/'electro_probe.py'),'--seed=42'])
        for case in ('electropolish','ceramic'):
            for count,run in zip(SIZES,runs[case],strict=True):
                if case=='electropolish':
                    args=['-m','examples.electropolish2d.train','--config='+str(configs/f'electro_{count}.py'),'--seed=42']
                else:
                    args=['-m','examples.particle_corrosion2d.train','--config',str(configs/f'ceramic_{count}.py'),
                        '--inputs',str(inputs/f'ceramic-{count}'),'--reference',str(CERAMIC_REFERENCE),
                        '--exclude-reference',str(CERAMIC_DEV),'--output',run,'--seed','42']
                stage(f'train-{case}-{count}',args)
                if case=='electropolish':
                    stage(f'evaluate-{case}-{count}',['-m','docs.tasks.electro_spatial_evaluate','--run',run])
                stage(f'audit-{case}-{count}',['-m','docs.tasks.fixed_library_audit','--case',case,
                    '--run',run,'--first',runs[case][0]],cpu=True)
        state['status']='audited';save()
        stage('figures',['-m','docs.tasks.fixed_library_figures','--bundle',str(bundle),
            '--output',str(bundle/'figures')],cpu=True)
        state.update(status='complete',finished_utc=datetime.now(timezone.utc).isoformat());save()
    except BaseException as error:
        state.update(status='failed',error=f'{type(error).__name__}: {error}');save();raise


if __name__=='__main__':main()
