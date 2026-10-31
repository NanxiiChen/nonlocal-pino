"""Frozen linear causal defect norm: independent scalar checks, no training.

For z_t+q*z=r(t), z(0)=0, interpolate r linearly on each interval h:
z_i=exp(-q*h)*z_(i-1)+a*r_(i-1)+b*r_i.
a=h*[1-(1+d)exp(-d)]/d^2, b=h*[d-1+exp(-d)]/d^2, d=q*h.
The additional first output (h_1/2)*r_0 makes the nodal map injective;
it is not propagated as the physical initial response, which remains zero.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy import sparse,linalg


def coefficients(q,h):
    d=np.asarray(q)*h
    small=abs(d)<1e-3;safe=np.where(small,1.,d);em1=np.expm1(-safe)
    a=h*(-safe-(1+safe)*em1)/safe**2
    b=h*(safe+em1)/safe**2
    a=np.where(small,h*(.5-d/3+d*d/8-d**3/30+d**4/144),a)
    b=np.where(small,h*(.5-d/6+d*d/24-d**3/120+d**4/720),b)
    return np.exp(-d),a,b


def causal_defect(residual,times,rates):
    r=np.asarray(residual);t=np.asarray(times);q=np.asarray(rates)
    if r.shape!=(len(t),len(q)) or t[0]!=0 or np.any(np.diff(t)<=0) or not all(np.isfinite(a).all() for a in (r,t,q)):
        raise ValueError('Finite aligned nodal residual/rates and increasing times starting at0 required')
    z=np.zeros_like(r);z[0]=.5*(t[1]-t[0])*r[0];state=np.zeros(len(q))
    for i,h in enumerate(np.diff(t),1):
        E,a,b=coefficients(q,h);state=E*state+a*r[i-1]+b*r[i];z[i]=state
    return z


def scalar_checks():
    maximum=0.
    for q in (-1.4,-1e-8,0.,1e-8,.3,162.):
        for h in (.006,.05,.37):
            _,a,b=coefficients(q,h)
            exact_a=quad(lambda s:np.exp(-q*(h-s))*(1-s/h),0,h,epsabs=1e-13)[0]
            exact_b=quad(lambda s:np.exp(-q*(h-s))*(s/h),0,h,epsabs=1e-13)[0]
            maximum=max(maximum,float(abs(a-exact_a)),float(abs(b-exact_b)))
            np.testing.assert_allclose([a,b],[exact_a,exact_b],rtol=1e-10,atol=1e-13)
    times=np.array([0.,.01,.07,.2,.7,1.5,3.]);rates=np.array([-1.4,0.,162.])
    for linear in (False,True):
        r=np.broadcast_to(times[:,None] if linear else np.ones((len(times),1)),(len(times),3))
        actual=causal_defect(r,times,rates)
        for i,t in enumerate(times[1:],1):
            expected=[quad(lambda s:np.exp(-q*(t-s))*(s if linear else 1),0,t,epsabs=1e-11)[0] for q in rates]
            np.testing.assert_allclose(actual[i],expected,rtol=1e-10,atol=1e-11)
    singular=[]
    for q in rates:
        H=np.column_stack([causal_defect(np.eye(len(times))[:,i,None],times,np.array([q]))[:,0] for i in range(len(times))])
        singular.append(float(np.linalg.svd(H,compute_uv=False).min()))
        if singular[-1]<=0:raise ValueError('Causal nodal preconditioner must be injective')
    return dict(quadrature_max_abs_difference=maximum,minimum_singular_values=singular,
        scalar_constant_and_linear_forcing=True,scope='Exact piecewise-linear residual response; no nonlinear-error equality claim')


def diagnose(run,step,index,dense,output):
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    from ml_collections import ConfigDict
    from examples.confined_ch2d.operator_training import build_network,prepare,phase,apply_operator
    from examples.confined_ch2d.operator_problem import fixed_initial
    checks=scalar_checks();jax.config.update('jax_enable_x64',True)
    c=ConfigDict(json.loads((run/'config.json').read_text()));root=Path(c.operator.dataset_path)
    manifest=json.loads((root/'manifest.json').read_text())
    with np.load(root/manifest['conditions'][index]['file'],allow_pickle=False) as f:d={k:f[k] for k in f.files}
    with np.load(dense/f'native_{index:03d}.npz',allow_pickle=False) as f:truth={k:f[k] for k in f.files}
    np.testing.assert_array_equal(d['points'],truth['points']);v=d['volumes'];times=truth['time']
    n,k=d['neighbors'].shape
    A=sparse.diags(d['coefficients'].sum(axis=1))+sparse.coo_matrix((-d['coefficients'].ravel(),
        (np.repeat(np.arange(n),k),d['neighbors'].ravel())),shape=(n,n)).tocsr()
    rootv=np.sqrt(v);S=(sparse.diags(rootv)@A@sparse.diags(1/rootv)).toarray()
    np.testing.assert_allclose(S,S.T,atol=1e-12)
    lam,U=linalg.eigh(S);mean=float(v@d['phi'][0]/v.sum())
    q=c.physical.mobility*lam*(c.physical.epsilon**2*lam+c.physical.beta*(3*mean**2-1))
    data={k:jnp.asarray(d[k]) for k in ('points','volumes','neighbors','coefficients','condition','condition_weights')}
    data['initial']=jnp.asarray(fixed_initial(d['points'],c,xp=np))
    net=eqx.tree_deserialise_leaves(run/f'ckpt_{step:06d}.eqx',build_network(c,jax.random.key(0)))
    @eqx.filter_jit
    def evaluate(model,data):
        prepared=prepare(model,data)
        def at(t):
            phi,dt=jax.jvp(lambda x:phase(model,prepared,x,data,c),(t,),(jnp.ones_like(t),))
            mu=c.physical.beta*(phi**3-phi)+c.physical.epsilon**2*apply_operator(phi,data)
            return phi,dt+c.physical.mobility*apply_operator(mu,data)
        return jax.vmap(at)(jnp.asarray(times))
    phi,r=map(np.asarray,evaluate(net,data));modal=r@(rootv[:,None]*U)
    z=causal_defect(modal,times,q)
    weight=np.empty(len(times));weight[0]=(times[1]-times[0])/2;weight[-1]=(times[-1]-times[-2])/2
    weight[1:-1]=(times[2:]-times[:-2])/2
    rms=lambda a:float(np.sqrt(np.sum(weight[:,None]*a*a)/(times[-1]*v.sum())))
    if c.operator.metric=='biharmonic_resolvent':w=1/(1+times[-1]*c.physical.mobility*c.physical.epsilon**2*lam**2)
    elif c.operator.metric=='linear_stable_resolvent':w=1/(1+times[-1]*np.maximum(q,0))
    else:raise ValueError('Unsupported source metric')
    source_w=w
    w=1/(1+times[-1]*np.maximum(q,0))
    actual=(phi-truth['phi'])*rootv
    report=dict(run=str(run),step=step,index=index,scalar_checks=checks,
        physical_error_rms=rms(actual),spatial_residual_rms=rms(times[-1]*modal*w),
        source_training_metric=c.operator.metric,source_spatial_residual_rms=rms(times[-1]*modal*source_w),
        diagnostic_spatial_metric='linear_stable_resolvent',
        causal_defect_rms=rms(z),physical_terminal_error_rms=float(np.linalg.norm(actual[-1])/np.sqrt(v.sum())),
        causal_terminal_defect_rms=float(np.linalg.norm(z[-1])/np.sqrt(v.sum())),
        maximum_linear_growth_amplification=float(np.exp(-q.min()*times[-1])),
        scope='Frozen checkpoint diagnosis only; piecewise-linear residual quadrature, no training or future-label loss')
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','dense','output'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('step','index'):p.add_argument('--'+name,type=int,required=True)
    a=p.parse_args();diagnose(a.run,a.step,a.index,a.dense,a.output)
