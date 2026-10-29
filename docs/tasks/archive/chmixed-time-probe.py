"""Validation-only free temporal coefficients: NOT a neural/PINN training run."""
from pathlib import Path
import sys,json
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file());sys.path.insert(0,str(ROOT))
import numpy as np
from scipy import sparse,linalg
from scipy.integrate import solve_ivp
from numpy.polynomial.chebyshev import chebvander
from examples.confined_ch2d.configs.train_coarsening_global_mixed import get_config
from examples.confined_ch2d.problem import load_reference,initial_field,relative
from examples.confined_ch2d.time_integration import integration_nodes,integration_matrix
from examples.confined_ch2d.native_reference import native_matrix
import importlib.util
spec=importlib.util.spec_from_file_location('temporal_audit',Path(__file__).with_name('chglobal-representation-floor.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);basis=module.basis

c=get_config();ref=load_reference(c);p=c.physical;v=ref['volumes'];scale=c.phase.initial_time_scale
A=native_matrix(ref['points'],v,c);A2=A@A;phi0=initial_field(ref['points'],c,xp=np)
mu0=p.beta*(phi0**3-phi0)+p.epsilon**2*(A@phi0);rate0=-p.mobility*(A@mu0)
fit=integration_nodes(count=128,duration=p.terminal_time,time_scale=scale)
strong=np.r_[0.,c.sample.temporal.nodes];check=np.unique(np.r_[strong,ref['time'],np.linspace(0,1.5,201)])
times=np.unique(np.r_[fit,check])
rhs=lambda t,phi:-p.mobility*(A@(p.beta*(phi**3-phi))+p.epsilon**2*(A2@phi))
jac=lambda t,phi:-p.mobility*(A@sparse.diags(p.beta*(3*phi**2-1))+p.epsilon**2*A2)
sol=solve_ivp(rhs,(0,p.terminal_time),phi0,method='BDF',jac=jac,rtol=1e-9,atol=1e-11,max_step=c.reference.refined_dt,t_eval=times)
if not sol.success:raise RuntimeError(sol.message)
truth=sol.y.T;true_mu=p.beta*(truth**3-truth)+p.epsilon**2*(A@truth.T).T
replay=relative(truth[np.searchsorted(times,ref['time'])],ref['phi_nonlocal'],v)
if replay>1e-5:raise ValueError('Native reference disagreement')
w=integration_matrix(fit,time_scale=scale)[-1]/p.terminal_time;rootw=np.sqrt(w)[:,None]
B,Bd=basis(fit,c.arch.rank,scale,p.terminal_time)
C=chebvander(2*np.log1p(fit/scale)/np.log1p(p.terminal_time/scale)-1,c.arch.rank-1)
Y=truth[np.searchsorted(times,fit)]-phi0;U=true_mu[np.searchsorted(times,fit)]
E,Ed=basis(check,c.arch.rank,scale,p.terminal_time)
F=chebvander(2*np.log1p(check/scale)/np.log1p(p.terminal_time/scale)-1,c.arch.rank-1)
root=np.sqrt(v);S=root[:,None]*A.toarray()/root[None,:];q=root/np.linalg.norm(root)
chol=linalg.cho_factor(.5*(S+S.T)+np.outer(q,q),lower=True)
rms=lambda a:np.sqrt(np.sum(v*a*a,axis=-1)/v.sum())
def constrained(matrix,target,row,boundary):
    Z=linalg.null_space(row[None,:]);part=row[:,None]*boundary/(row@row)
    coef=np.linalg.lstsq(rootw*(matrix@Z),rootw*(target-matrix@part),rcond=None)[0]
    return part+Z@coef
report=dict(scope='Validation-only free spatial coefficients, not neural capacity or a PINN result. No fitted arrays enter training.',native_replay=replay,cases={})
for name in ('free','known_initial_rate_and_mu'):
    if name=='free':
        phase=np.linalg.lstsq(rootw*B,rootw*Y,rcond=None)[0];chemical=np.linalg.lstsq(rootw*C,rootw*U,rcond=None)[0]
    else:
        phase=constrained(B,Y,Bd[0],rate0);chemical=constrained(C,U,C[0],mu0)
    phase-=((phase@v)/v.sum())[:,None]
    phi=phi0+E@phase;dt=Ed@phase;mu=F@chemical
    evolution=dt+p.mobility*(A@mu.T).T
    inverse=linalg.cho_solve(chol,(evolution*root).T).T/root/p.mobility
    chem=mu-p.beta*(phi**3-phi)-p.epsilon**2*(A@phi.T).T
    i=np.searchsorted(check,strong);tr=truth[np.searchsorted(times,check)];tm=true_mu[np.searchsorted(times,check)]
    row=dict(phase=relative(phi,tr,v),phase_terminal=relative(phi[-1],tr[-1],v),mu=relative(mu,tm,v),mu_terminal=relative(mu[-1],tm[-1],v),
        strong_phase_loss=float(np.mean(rms(inverse[i])**2)/c.loss.chemical_scale**2),
        chemical_loss=float(np.mean(rms(chem[i])**2)/c.loss.chemical_scale**2),
        initial_rate_defect_rms=float(rms(dt[0]-rate0)),initial_mu_defect_rms=float(rms(mu[0]-mu0)))
    report['cases'][name]=row;print(name,row,flush=True)
Path(__file__).with_suffix('.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
