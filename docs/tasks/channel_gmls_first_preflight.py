"""CPU-only finite-prior boundary-gradient checks for the matched PINO control."""
from pathlib import Path
import argparse
import importlib
import json
import os
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from examples.channel_nsch2d.problem import quadrature,boundary,validate_config
from examples.channel_nsch2d.operators import first_derivative_stencils
from examples.channel_nsch2d.online import validate_construction_screen,validate_payload
from examples.channel_nsch2d.data import geometry_data


def check(args):
    output=Path(args.output)
    if output.exists():raise FileExistsError(output)
    c=importlib.import_module(args.config).get_config();c.train.seed=args.seed
    old=json.loads((Path(args.baseline_run)/'config.json').read_text())
    c.reference.root=old['reference']['root'];c.log.run_name=old['log']['run_name']
    validate_config(c);before=json.loads(c.to_json());del before['derivatives']
    before['phase_transport']=old['phase_transport']
    if before!=old:raise ValueError('Unexpected settings change outside declared derivative/flux control')
    existing=validate_construction_screen(c,args.prior_screen)
    cases=json.loads(Path(args.prior_screen).read_text())['cases']
    start=time.perf_counter();records=[]
    for row in cases:
        a=np.asarray(row['amplitudes']);points,_=quadrature(a,c,c.sample.Nx,c.sample.Ny)
        b,_=boundary(a,c)
        _,_,_,report=first_derivative_stencils(points,b[:,:2],c)
        records.append(dict(kind=row['kind'],index=row['index'],amplitudes=row['amplitudes'],boundary_gradient=report))
    with np.load(Path(args.baseline_run)/'known_inputs.npz') as bank:
        for slot in (0,15):
            raw,_=geometry_data(bank['amplitudes'][slot],c)
            for name in bank.files:np.testing.assert_array_equal(raw[name],bank[name][slot])
            converted={name:v.astype('int32' if name in ('neighbors','boundary_neighbors') else c.train.compute_dtype)
                for name,v in raw.items()}
            validate_payload(converted,c)
    nominal,report=geometry_data((c.geometry.upper,c.geometry.lower),c)
    result=dict(status='finite_prior_GMLS_boundary_gradients_and_shared_inputs_passed',
        config=args.config,baseline_run=args.baseline_run,run_seed=args.seed,
        existing_laplacian_prior_screen=args.prior_screen,existing_screen=existing,
        new_boundary_gradient_cases=len(records),
        maximum_boundary_condition=max(r['boundary_gradient']['maximum_condition'] for r in records),
        maximum_boundary_polynomial_error=max(r['boundary_gradient']['polynomial_error'] for r in records),
        inherited_array_slots_checked=[0,15],inherited_arrays_exact=True,
        nominal_operator=report,known_input_arrays=len(nominal),seconds=time.perf_counter()-start,
        cases=records,scope='Finite prescribed training-prior prefix, no PDE/FEM/NN solve; domain rank uses identical basis/neighbors as old Laplacian, all actual domain gradients also checked during construction')
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    output.with_suffix('.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases','existing_screen')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','baseline-run','prior-screen','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--seed',type=int,required=True)
    check(p.parse_args())
