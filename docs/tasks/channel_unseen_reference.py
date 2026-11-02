"""Wait for online training, validate every reserved FEM geometry, evaluate frozen NO."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse
import json
import multiprocessing
import os
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def reference_case(config_path,case,query_grid):
    from ml_collections import ConfigDict
    from examples.channel_nsch2d.validate_reference import assess
    root=Path(__file__).resolve().parents[2];folder=Path(case['root']);folder.mkdir(exist_ok=False)
    config=ConfigDict(json.loads(Path(config_path).read_text()))
    config.geometry.upper,config.geometry.lower=case['amplitudes'];config.reference.root=str(folder)
    env=dict(os.environ);env.pop('VIRTUAL_ENV',None);env['JAX_PLATFORMS']='cpu'
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[name]='1'
    fenics='/root/miniconda3/envs/fenicsenv/bin/python'
    for tag,grid,dt in (('coarse',config.reference.coarse_grid,config.reference.refined_dt),
                       ('time',config.reference.fine_grid,config.reference.dt),
                       ('fine',config.reference.fine_grid,config.reference.refined_dt)):
        command=[fenics,'-m','examples.channel_nsch2d.solver','--saved-config',str(config_path),
            '--grid',*[str(v) for v in grid],'--dt',str(dt),'--query-grid',*[str(v) for v in query_grid],
            '--common-query-grid',*[str(v) for v in query_grid],
            '--upper',str(case['amplitudes'][0]),'--lower',str(case['amplitudes'][1]),
            '--output',str(folder/f'bdf-{tag}.npz')]
        with (folder/f'{tag}.log').open('x') as log:
            subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    report=assess(config,folder/'bdf-coarse.npz',folder/'bdf-time.npz',folder/'bdf-fine.npz')
    if report['common_queries']['status']!='passed':raise ValueError('Common physical reference refinement failed')
    return dict(index=case['index'],status=report['status'],common_query_status=report['common_queries']['status'],
        reference_validation=str(folder/'reference-validation.json'))


def main(after_pid,run,manifest,workers,tag,response_protocol):
    from ml_collections import ConfigDict
    from examples.channel_nsch2d.cohort import load
    from examples.channel_nsch2d.response_queries import protocol
    if isinstance(workers,bool) or workers<1:raise ValueError('Positive FEM worker count required')
    run=Path(run);manifest=Path(manifest);folder=manifest.parent;root=Path(__file__).resolve().parents[2]
    response_protocol=Path(response_protocol)
    if not response_protocol.is_absolute() or not response_protocol.is_file():
        raise ValueError('Explicit existing absolute common-physical response protocol required')
    declared=json.loads(response_protocol.read_text())
    progress=folder/f'{tag}-pipeline-progress.json'
    if progress.exists():raise FileExistsError(progress)
    def record(stage,**kwargs):
        progress.write_text(json.dumps(dict(stage=stage,**kwargs),indent=2,allow_nan=False)+'\n')
        print(stage,flush=True)
    record('waiting_for_online_training_and_audits',pid=after_pid,run=str(run))
    while Path(f'/proc/{after_pid}').exists():time.sleep(2)
    if not (run/'summary.json').is_file():
        record('online_training_not_completed',goal_complete=False);return
    summary=json.loads((run/'summary.json').read_text())
    if summary['status']!='completed' or 'online' not in summary:raise ValueError('A completed actual online run is required')
    if json.loads((run/'validation.json').read_text())['status']!='replay_and_hard_constraints_passed':
        raise ValueError('Online physical and input replay must pass before cohort assessment')
    c=ConfigDict(json.loads((run/'config.json').read_text()));cohort=load(manifest,c)
    if declared!=protocol(c,cohort['query_grid']):raise ValueError('Frozen common-physical response protocol differs from the actual model family')
    if cohort['count']!=16:raise ValueError('This pilot driver requires the predeclared16geometry cohort')
    while subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],
                         capture_output=True,text=True,check=True).stdout.strip():time.sleep(5)
    smoke=folder/f'{tag}-common-query-export-smoke'
    record('isolated_FEM_export_compatibility_smoke',output=str(smoke))
    env=dict(os.environ);env.pop('VIRTUAL_ENV',None);env['JAX_PLATFORMS']='cpu'
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[name]='1'
    with (folder/f'{tag}-common-query-export-smoke.log').open('x') as log:
        subprocess.run(['/root/miniconda3/envs/fenicsenv/bin/python','docs/tasks/channel_common_fem_smoke.py',
            '--saved-config',str(run/'config.json'),'--output',str(smoke)],cwd=root,env=env,
            stdout=log,stderr=subprocess.STDOUT,check=True)
    smoke_report=json.loads((smoke/'validation.json').read_text())
    if smoke_report['status']!='common_query_export_legacy_parity_passed':raise ValueError('FEM diagnostic export changed the primary solution')
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
    record('isolated_parallel_FEM_reference_gates',workers=workers,completed=0,cases=cohort['count'])
    records=[];development=None
    development_case=dict(index=cohort['count'],amplitudes=[c.geometry.upper,c.geometry.lower],root=str(folder/'development'))
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as executor:
        futures=[executor.submit(reference_case,str(run/'config.json'),case,cohort['query_grid']) for case in [*cohort['cases'],development_case]]
        for future in as_completed(futures):
            result=future.result()
            if result['index']==cohort['count']:development=result
            else:records.append(result)
            record('isolated_parallel_FEM_reference_gates',workers=workers,
                completed=len(records),cases=cohort['count'],nominal_development_completed=development is not None,
                references=sorted(records,key=lambda r:r['index']))
    if development is None:raise ValueError('Nominal off-grid development reference is missing')
    gate=dict(status='passed',cases=cohort['count'],manifest=str(manifest),references=sorted(records,key=lambda r:r['index']),
        nominal_development_reference=development,query_grid=cohort['query_grid'],
        common_query_protocol=declared,response_protocol=str(response_protocol),
        FEM_export_compatibility_smoke=str(smoke/'validation.json'),
        scope='Every frozen geometry passed the unchanged independent space/time/flow gates; no training labels')
    (folder/'cohort-reference-validation.json').write_text(json.dumps(gate,indent=2)+'\n')
    record('frozen_operator_GPU_evaluation',cases=cohort['count'])
    env=dict(os.environ);env['JAX_PLATFORMS']='cuda';python=str(root/'.venv/bin/python')
    with (folder/'nominal-model-evaluation.log').open('x') as log:
        subprocess.run([python,'docs/tasks/channel_dense_evaluate.py','--run',str(run),
            '--validation',development['reference_validation']],cwd=root,env=env,
            stdout=log,stderr=subprocess.STDOUT,check=True)
    with (folder/'model-evaluation.log').open('x') as log:
        subprocess.run([python,'-m','examples.channel_nsch2d.cohort_evaluate','--run',str(run),
            '--manifest',str(manifest),'--output',str(run/'unseen16')],cwd=root,env=env,
            stdout=log,stderr=subprocess.STDOUT,check=True)
    env['JAX_PLATFORMS']='cpu'
    subprocess.run([python,'-m','examples.channel_nsch2d.evaluate','--fields',str(run/'dense-fem-prediction.npz'),
        '--output',str(run/'dense-fem-figures')],cwd=root,env=env,check=True)
    for index in (0,5,10):
        case=run/'unseen16'/f'case_{index:03d}'
        subprocess.run([python,'-m','examples.channel_nsch2d.evaluate','--fields',str(case/'prediction.npz'),
            '--output',str(case/'figures')],cwd=root,env=env,check=True)
    record('unseen_cohort_evaluation_completed',evaluation=str(run/'unseen16'/'evaluation.json'),
        nominal_development_evaluation=str(run/'dense-fem-evaluation.json'),goal_complete=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--after-pid',required=True,type=int)
    p.add_argument('--run',required=True);p.add_argument('--manifest',required=True);p.add_argument('--workers',required=True,type=int)
    p.add_argument('--tag',required=True)
    p.add_argument('--response-protocol',required=True)
    a=p.parse_args();main(a.after_pid,a.run,a.manifest,a.workers,a.tag,a.response_protocol)
