"""Independent polynomial strong/weak momentum and pressure-gradient identities."""
import unittest
import numpy as np
import jax
import jax.numpy as jnp
from examples.channel_nsch2d.configs.train_base import get_config
from examples.channel_nsch2d.problem import walls,wall_slopes
from docs.tasks.channel_momentum_closure_probe import global_momentum


def manufactured(points,t):
    x,y=points.T;t=t[:,None];shape=(len(t),len(x),5)
    value=np.zeros(shape);rate=np.zeros(shape);dx=np.zeros(shape);dy=np.zeros(shape)
    value[...,0]=x-y;value[...,1]=t*(x*x+y);value[...,2]=2*t*x*x*y
    value[...,3]=-2*t*x*y*y;value[...,4]=t*(x**3+2*y*y)
    rate[...,2]=2*x*x*y;rate[...,3]=-2*x*y*y
    dx[...,0]=1;dy[...,0]=-1;dx[...,1]=2*t*x;dy[...,1]=t
    dx[...,2]=4*t*x*y;dy[...,2]=2*t*x*x
    dx[...,3]=-2*t*y*y;dy[...,3]=-4*t*x*y
    dx[...,4]=3*t*x*x;dy[...,4]=4*t*y
    return value,rate,dx,dy


def gauss_domain(c,amplitudes):
    r,w=np.polynomial.legendre.leggauss(32);e=(r+1)/2;we=w/2
    cuts=(0,c.geometry.center-c.geometry.half_length,c.geometry.center+c.geometry.half_length,c.geometry.length)
    xx=[];wx=[]
    for lo,hi in zip(cuts[:-1],cuts[1:],strict=True):xx.append(lo+(r+1)*(hi-lo)/2);wx.append(w*(hi-lo)/2)
    x=np.concatenate(xx);wx=np.concatenate(wx);b,t=walls(x,amplitudes,c,xp=np)
    X,E=np.meshgrid(x,e,indexing='ij');Y=b[:,None]+E*(t-b)[:,None]
    points=np.stack((X,Y),axis=-1).reshape(-1,2)
    measure=(wx[:,None]*we[None,:]*(t-b)[:,None]).ravel()
    db,dt=wall_slopes(x,amplitudes,c,xp=np)
    lower=np.stack((db,-np.ones_like(x)),axis=-1);upper=np.stack((-dt,np.ones_like(x)),axis=-1)
    wl=np.linalg.norm(lower,axis=-1);wu=np.linalg.norm(upper,axis=-1)
    y=e*c.geometry.height
    boundary=np.concatenate((np.column_stack((x,b,lower/wl[:,None],np.zeros_like(x))),
        np.column_stack((x,t,upper/wu[:,None],np.zeros_like(x))),
        np.column_stack((np.zeros_like(y),y,-np.ones_like(y),np.zeros_like(y),np.ones_like(y))),
        np.column_stack((np.full_like(y,c.geometry.length),y,np.ones_like(y),np.zeros_like(y),np.full_like(y,2)))))
    bw=np.concatenate((wx*wl,wx*wu,we*c.geometry.height,we*c.geometry.height))
    return points,measure,boundary,bw


class MomentumClosure(unittest.TestCase):
    def test_exact_flat_and_curved_divergence_theorem_and_pressure_gradient(self):
        c=get_config();times=np.array([.2,.7]);rho=1.3;nu=.2
        for amplitudes in ((0.,0.),(.12,.08)):
            points,w,b,bw=gauss_domain(c,amplitudes)
            value,rate,dx,dy=manufactured(points,times);bv,_,bx,by=manufactured(b[:,:2],times)
            actual=global_momentum(value,rate,dx,dy,bv,bx,by,w,b,bw,rho,nu,xp=np)
            x,y=points.T;t=times[:,None]
            conv=np.stack((4*t*t*x**3*y*y,4*t*t*x*x*y**3),axis=-1)
            lap=np.stack((4*t*y,-4*t*x),axis=-1)
            strong=rho*(rate[...,2:4]+conv)+np.stack((3*t*x*x,4*t*y),axis=-1)-nu*lap
            strong+=(x-y)[None,:,None]*np.stack((2*t*x,np.broadcast_to(t,(len(t),len(x)))),axis=-1)
            expected=np.sum(strong*w[None,:,None],axis=1)/w.sum()
            np.testing.assert_allclose(actual,expected,atol=2e-11,rtol=2e-13)
            if amplitudes==(0.,0.):
                L=c.geometry.length;H=c.geometry.height;t=times
                hand=np.stack((rho*(L*L*H/3+t*t*L**3*H*H/3)+t*L*L-2*nu*t*H+t*(2*L*L/3-L*H/2),
                    rho*(-L*H*H/3+t*t*L*L*H**3/3)+2*t*H+2*nu*t*L+t*(L-H)/2),axis=-1)
                np.testing.assert_allclose(actual,hand,atol=2e-11,rtol=2e-13)
                with jax.enable_x64(True):
                    def f(alpha):
                        boundary_value=jnp.asarray(bv).at[...,4].multiply(alpha)
                        return jnp.sum(global_momentum(*[jnp.asarray(a) for a in (value,rate,dx,dy)],
                            boundary_value,jnp.asarray(bx),jnp.asarray(by),jnp.asarray(w),jnp.asarray(b),jnp.asarray(bw),rho,nu,xp=jnp))
                    derivative=jax.jit(jax.grad(f))(1.)
                    np.testing.assert_allclose(derivative,np.sum(times)*(L*L+2*H),atol=2e-11,rtol=2e-13)


if __name__=='__main__':unittest.main()
