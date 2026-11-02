"""Finite four-geometry independent FEM qualification, before any GPU training."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse
import importlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from docs.tasks.channel_unseen_reference import reference_case


def main(output,config,workers,reuse_root):
    from examples.channel_nsch2d.cohort import prepare
    from examples.channel_nsch2d.geometry import nominal_parameters
    from examples.channel_nsch2d.response_queries import protocol
    from nonlocal_pinn.training.runtime import with_run_seed
    root=Path(output)
    if not root.is_absolute():raise ValueError('Absolute external output required')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise RuntimeError('FEM qualification cannot overlap JAX GPU work')
    root.mkdir(exist_ok=False)
    c=with_run_seed(importlib.import_module(config).get_config(),None)
    c.reference.root=str(root/'development')
    c.operator.validation_manifest=str(root/'unseen3'/'manifest.json')
    cohort=prepare(c,c.operator.validation_manifest,3,(96,40))
    (root/'config.json').write_text(c.to_json(indent=2)+'\n')
    journal=dict(stage='FEM_reference_trios',started=time.time(),workers=workers,cases=4,
        config=str(root/'config.json'),manifest=c.operator.validation_manifest,reuse_root=reuse_root,completed=[],
        training_labels=0,automatic_training=False)
    def save():
        temp=root/'progress.tmp';temp.write_text(json.dumps(journal,indent=2)+'\n');temp.replace(root/'progress.json')
    save()
    cases=[dict(index=3,amplitudes=nominal_parameters(c).tolist(),root=str(root/'development')),*cohort['cases']]
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
    try:
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            pending=[]
            for case in cases:
                source=None
                if reuse_root is not None:
                    relative=Path(case['root']).relative_to(root)
                    candidate=Path(reuse_root)/relative/'bdf-fine.npz'
                    if candidate.exists():source=str(candidate)
                pending.append(pool.submit(reference_case,str(root/'config.json'),case,cohort['query_grid'],source))
            for job in as_completed(pending):
                result=job.result();journal['completed'].append(result);save();print(json.dumps(result),flush=True)
        development=next(r for r in journal['completed'] if r['index']==3)
        gate=dict(status='passed',cases=3,manifest=c.operator.validation_manifest,
            references=sorted([r for r in journal['completed'] if r['index']!=3],key=lambda r:r['index']),
            nominal_development_reference=development,query_grid=cohort['query_grid'],
            common_query_protocol=protocol(c,cohort['query_grid']),
            scope='Three frozen prior draws and one nominal; independent space/time/flux gates, no evolved training labels')
        (root/'unseen3'/'cohort-reference-validation.json').write_text(json.dumps(gate,indent=2)+'\n')
        journal.update(stage='completed',seconds=time.time()-journal['started']);save()
    except BaseException as exc:
        journal.update(stage='failed',error=repr(exc));save();raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--config',required=True);p.add_argument('--workers',required=True,type=int)
    p.add_argument('--reuse-root',help='Explicit previous attempt; only complete matching fine exports are eligible as new coarse')
    a=p.parse_args();main(a.output,a.config,a.workers,a.reuse_root)
