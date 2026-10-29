"""Validation-only native IVP check of global chemical integration error.

No generated values are consumed by training. Existing reference is unchanged.
This script imports no JAX or FEM implementation.
"""
from pathlib import Path
import sys,json,time
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.integrate import solve_ivp
from examples.confined_ch2d.configs.train_coarsening_integral import get_config
from examples.confined_ch2d.time_integration import integration_nodes,integration_matrix
from examples.confined_ch2d.problem import load_reference,initial_field,relative
from examples.confined_ch2d.native_reference import native_matrix


def main():
    c=get_config();ref=load_reference(c);p=c.physical;v=ref['volumes']
    A=native_matrix(ref['points'],v,c);A2=A@A;K=p.mobility*p.epsilon**2*A2
    grids={n:integration_nodes(count=n,duration=p.terminal_time,time_scale=c.loss.phase_metric.time_scale) for n in (32,64,128)}
    times=np.unique(np.concatenate([ref['time'],*grids.values()]))
    phi0=initial_field(ref['points'],c,xp=np)
    def rhs(t,phi):return -p.mobility*(A@(p.beta*(phi**3-phi))+p.epsilon**2*(A2@phi))
    def jac(t,phi):return -p.mobility*(A@sparse.diags(p.beta*(3*phi**2-1))+p.epsilon**2*A2)
    tick=time.perf_counter()
    sol=solve_ivp(rhs,(0,p.terminal_time),phi0,method='BDF',jac=jac,rtol=1e-9,atol=1e-11,max_step=c.reference.refined_dt,t_eval=times)
    if not sol.success:raise RuntimeError(sol.message)
    original=sol.y[:,np.searchsorted(times,ref['time'])].T
    comparison=relative(original,ref['phi_nonlocal'],v)
    if comparison>1e-5:raise ValueError('Validation native IVP differs from existing reference: '+str(comparison))
    report=dict(scope='Validation only; no data enters any PINN loss',existing_native_phi_relative=comparison,grids={})
    for n,nodes in grids.items():
        phi=sol.y[:,np.searchsorted(times,nodes)].T;mu=p.beta*(phi**3-phi)+p.epsilon**2*(A@phi.T).T
        W=integration_matrix(nodes,time_scale=c.loss.phase_metric.time_scale)
        residual=phi-phi0+p.mobility*(A@(W@mu).T).T
        errors=np.stack([spsolve(sparse.eye(len(v),format='csc')+t*K,r) for t,r in zip(nodes,residual,strict=True)])
        rms=lambda a:np.sqrt(np.sum(v*a*a,axis=1)/v.sum())
        record=dict(integrated_defect_max_rms=float(np.max(rms(residual))),
            preconditioned_defect_max_rms=float(np.max(rms(errors))),
            preconditioned_defect_terminal_rms=float(rms(errors)[-1]),
            quadrature_objective=float(np.dot(W[-1],rms(errors)**2)/p.terminal_time))
        report['grids'][str(n)]=record;print(n,record,flush=True)
    report['wall_seconds']=time.perf_counter()-tick
    Path(__file__).with_suffix('.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
