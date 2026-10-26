"""CPU float64 manufactured local-PDE and paired AD/data checks."""
import importlib
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from examples.corrosion2d.configs.train_base import get_config
from examples.corrosion2d.residuals import make_normalized_data_loss_vector
from nonlocal_pinn.nn.constraints import ConstrainedForward,kks_constraint

benchmark=importlib.import_module(__package__+'.local-ad-performance')


class Raw(eqx.Module):
    a:jax.Array
    b:jax.Array

    def __call__(self,x,t):
        return jnp.stack([self.a*x[0]+self.b*x[1]**2+.2*t,
                          .1*x[0]*x[1]+.3*t-.2])


def main():
    if jax.default_backend()!='cpu':raise RuntimeError('CPU-only preflight')
    jax.config.update('jax_enable_x64',True)
    cle=.04;dc=.96
    raw=Raw(jnp.array(.7),jnp.array(.4))
    net=ConstrainedForward(raw,lambda r:kks_constraint(r,1.,cle))
    x=jnp.array([[.2,.3],[.5,.1],[.8,.4]])
    ts=jnp.array([.15,.75])
    nd=(.7,1.2,.8,.4,dc,cle)
    point_errors=[];chain_errors=[]
    for xx in x:
        for t in ts:
            p,c,pt,ct,lap_p,lap_q=benchmark.local_point(net,xx,t,cle)
            z,v=raw(xx,t);s=jnp.tanh(z);sv=jnp.tanh(v)
            gp2=raw.a**2+(2*raw.b*xx[1])**2
            expected_lap_p=.5*(1-s*s)*(2*raw.b-2*s*gp2)
            expected_lap_q=-cle*sv*(1-sv*sv)*.01*jnp.sum(xx**2)
            expected_pt=.1*(1-s*s)
            expected_ct=.15*cle*(1-sv*sv)+dc*6*p*(1-p)*expected_pt
            expected=jnp.array([expected_pt,expected_ct,expected_lap_p,expected_lap_q])
            actual=jnp.array([pt,ct,lap_p,lap_q])
            np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-14)
            point_errors.append(float(jnp.max(jnp.abs(actual-expected))))
            physical_mu=lambda xx:net(xx,t)[1]-dc*(3*net(xx,t)[0]**2-2*net(xx,t)[0]**3)-cle
            chain=jnp.trace(jax.hessian(physical_mu)(xx))
            np.testing.assert_allclose(lap_q,chain,rtol=1e-11,atol=1e-14)
            chain_errors.append(float(jnp.abs(lap_q-chain)))
    pairs=jnp.concatenate([jnp.tile(x,(len(ts),1)),jnp.repeat(ts,len(x))[:,None]],axis=1)
    tensor=benchmark.local_residual(net,ts,x,nd,'local_tensor')
    joint=benchmark.local_residual(net,ts,pairs,nd,'local_joint')
    for a,b in zip(tensor,joint,strict=True):
        np.testing.assert_allclose(a.T.ravel(),b,rtol=1e-12,atol=1e-14)
    def values(time):return jax.vmap(lambda xx,t:net(xx,t))(pairs[:,:2],time)
    jvp=jax.jvp(values,(pairs[:,2],),(jnp.ones(len(pairs)),))[1]
    grad=jax.vmap(lambda z:jnp.stack([jax.grad(lambda t:net(z[:2],t)[i])(z[2]) for i in range(2)]))(pairs)
    np.testing.assert_allclose(jvp,grad,rtol=1e-12,atol=1e-14)
    c=get_config();c.physical.c_le=cle
    xic=x[:2];ic=jnp.array([[.2,.1],[.7,.6]])
    zero=x[:1];one=x[-1:]
    local,local_pde=benchmark.local_loss_factory(c,nd,xic,ic,zero,one,'local_tensor')
    physical=make_normalized_data_loss_vector(c,nd,xic,ic,zero,one,normalize_initial=False)
    empty=jnp.zeros(0,dtype=jnp.int32)
    sub=(empty,empty,jnp.zeros(0),jnp.zeros(len(x)))
    mask=jnp.array([True,True,False])
    l=local(net,ts,x,None,mask);p=physical(net,ts,x,sub,mask)
    np.testing.assert_allclose(l[2:],p[2:],rtol=0,atol=0)
    # Mixed parameter/spatial derivatives required by actual PINN training.
    def objective(a):
        n=eqx.tree_at(lambda m:m.net.a,net,a)
        return jnp.sum(local(n,ts,x,None,mask))
    ad=float(jax.grad(objective)(raw.a));eps=1e-5
    fd=float((objective(raw.a+eps)-objective(raw.a-eps))/(2*eps))
    np.testing.assert_allclose(ad,fd,rtol=2e-7,atol=1e-10)
    result={'status':'passed','manufactured_lap_time_max_error':max(point_errors),
            'stable_q_vs_mixture_laplacian_max_error':max(chain_errors),
            'tensor_joint_same_points_equal':True,'normalized_BC_IC_losses_exactly_equal':True,
            'joint_time_grad_jvp_max_error':float(jnp.max(jnp.abs(jvp-grad))),
            'mixed_parameter_gradient':ad,'finite_difference':fd,
            'local_sign_reference':'examples/electropolish2d/losses.py:152; D_delta -> -Laplacian'}
    Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
