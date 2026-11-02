"""Input-only conditioning checks over the complete planned online prefix and initial bank."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse
import importlib
import json
import multiprocessing
import os
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def one(config,kind,index,amplitudes):
    import numpy as np
    from ml_collections import ConfigDict
    from examples.channel_nsch2d.online import build_input,validate_payload
    from examples.channel_nsch2d.data import geometry_data
    c=ConfigDict(config)
    if kind=='online':
        _,record=build_input(config,index)
        return dict(kind=kind,index=index,amplitudes=record['amplitudes'],operator=record['operator'])
    raw,report=geometry_data(np.asarray(amplitudes),c)
    data={k:v.astype('int32' if k in ('neighbors','boundary_neighbors') else c.train.compute_dtype) for k,v in raw.items()}
    validate_payload(data,c)
    return dict(kind=kind,index=index,amplitudes=amplitudes,operator=report)


def main(config_module,draws,workers,output):
    import numpy as np
    from nonlocal_pinn.training.runtime import with_run_seed
    from examples.channel_nsch2d.problem import validate_config,identity
    c=with_run_seed(importlib.import_module(config_module).get_config(),None);validate_config(c)
    if draws<1 or workers<1:raise ValueError('Positive online draw and CPU worker counts required')
    path=Path(output)
    if path.exists():raise FileExistsError(path)
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
    os.environ['JAX_PLATFORMS']='cpu';begin=time.perf_counter()
    rng=np.random.default_rng(c.operator.dataset_seed)
    initial=rng.uniform(c.geometry.amplitude_min,c.geometry.amplitude_max,(c.operator.train_conditions,2))
    rows=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        tasks=[pool.submit(one,c.to_dict(),'initial',i,a.tolist()) for i,a in enumerate(initial)]
        tasks += [pool.submit(one,c.to_dict(),'online',i,None) for i in range(draws)]
        for task in as_completed(tasks):
            rows.append(task.result())
            if len(rows)%64==0:print(f'Validated {len(rows)}/{len(tasks)} input-only operators',flush=True)
    report=dict(status='passed',operator=c.nl.to_dict(),grid=[c.sample.Nx,c.sample.Ny],
        run_seed=c.train.seed,initial_bank_seed=c.operator.dataset_seed,physical_family=identity(c),
        online_draws=draws,initial_geometries=len(initial),workers=workers,seconds=time.perf_counter()-begin,
        maximum_condition=max(row['operator']['maximum_condition'] for row in rows),
        maximum_polynomial_error=max(row['operator']['polynomial_error'] for row in rows),
        pde_solves=0,fem_solves=0,stored_operator_arrays=0,
        cases=sorted(rows,key=lambda row:(row['kind'],row['index'])),
        scope='Canonical input construction/conditioning over an explicit finite prior prefix, not a continuum or solution-accuracy proof')
    with path.open('x') as stream:stream.write(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--draws',required=True,type=int);p.add_argument('--workers',required=True,type=int);p.add_argument('--output',required=True)
    a=p.parse_args();main(a.config,a.draws,a.workers,a.output)
