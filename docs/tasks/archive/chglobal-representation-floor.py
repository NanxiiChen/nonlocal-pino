"""Validation-only temporal approximation audit, never PINN training targets.

Free coefficients at each grid node test the global time trial space alone;
they are not a neural-network solution or a learned PINN checkpoint.
"""
from pathlib import Path
import sys,json
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy import sparse
from scipy.integrate import solve_ivp
from numpy.polynomial.chebyshev import chebvander,chebder
from examples.confined_ch2d.configs.train_coarsening_global_basis import get_config
from examples.confined_ch2d.time_integration import integration_nodes,integration_matrix
from examples.confined_ch2d.problem import load_reference,initial_field,relative
from examples.confined_ch2d.native_reference import native_matrix


def basis(times,rank,scale,duration):
    L=np.log1p(duration/scale);s=np.log1p(times/scale)/L;u=2*s-1
    C=chebvander(u,rank-1)
    Cp=chebvander(u,rank-2)@chebder(np.eye(rank),axis=0)
    return s[:,None]*C,(C+2*s[:,None]*Cp)/((scale+times)*L)[:,None]


def main():
    c=get_config();ref=load_reference(c);p=c.physical;v=ref['volumes'];scale=c.phase.initial_time_scale
    A=native_matrix(ref['points'],v,c);A2=A@A;phi0=initial_field(ref['points'],c,xp=np)
    fit_times=integration_nodes(count=128,duration=p.terminal_time,time_scale=scale)
    validation_times=np.unique(np.r_[np.linspace(0,p.terminal_time,201),ref['time']])
    times=np.unique(np.r_[fit_times,validation_times])
    def rhs(t,phi):return -p.mobility*(A@(p.beta*(phi**3-phi))+p.epsilon**2*(A2@phi))
    def jac(t,phi):return -p.mobility*(A@sparse.diags(p.beta*(3*phi**2-1))+p.epsilon**2*A2)
    sol=solve_ivp(rhs,(0,p.terminal_time),phi0,method='BDF',jac=jac,rtol=1e-9,atol=1e-11,max_step=c.reference.refined_dt,t_eval=times)
    if not sol.success:raise RuntimeError(sol.message)
    original=sol.y[:,np.searchsorted(times,ref['time'])].T
    gap=relative(original,ref['phi_nonlocal'],v)
    if gap>1e-5:raise ValueError('Validation IVP disagrees with existing native reference')
    y=sol.y[:,np.searchsorted(times,fit_times)].T-phi0
    truth=sol.y[:,np.searchsorted(times,validation_times)].T
    true_mu=p.beta*(truth**3-truth)+p.epsilon**2*(A@truth.T).T
    weights=integration_matrix(fit_times,time_scale=scale)[-1]/p.terminal_time
    report=dict(scope='Validation only. Free nodal time coefficients are not a PINN or a training target. They isolate temporal approximation, not spatial neural capacity.',native_reference_replay=gap,ranks={})
    rms=lambda a:np.sqrt(np.sum(v*a*a,axis=-1)/v.sum())
    for rank in (8,16,24,48,64):
        B,_=basis(fit_times,rank,scale,p.terminal_time)
        coeff,_,_,_=np.linalg.lstsq(np.sqrt(weights[:,None])*B,np.sqrt(weights[:,None])*y,rcond=None)
        coeff-=((coeff@v)/v.sum())[:,None]
        E,Edot=basis(validation_times,rank,scale,p.terminal_time)
        phi=phi0+E@coeff;dt=Edot@coeff
        mu=p.beta*(phi**3-phi)+p.epsilon**2*(A@phi.T).T
        residual=dt+p.mobility*(A@mu.T).T
        row=dict(phase_pooled=relative(phi,truth,v),phase_terminal=relative(phi[-1],truth[-1],v),
            mu_pooled=relative(mu,true_mu,v),mu_terminal=relative(mu[-1],true_mu[-1],v),
            initial_derivative_residual_rms=float(rms(residual)[0]),differential_residual_max_rms=float(np.max(rms(residual))))
        report['ranks'][str(rank)]=row;print(rank,row,flush=True)
    Path(__file__).with_suffix('.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
