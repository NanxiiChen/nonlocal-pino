"""Mapped conservative phase advection with published diagonal-norm SBP4-2.

Fisher--Carpenter NASA/TM2013-217971, Appendix A.1, equations A1--A4:
https://ntrs.nasa.gov/api/citations/20130011028/downloads/20130011028.pdf
This is a bounded-domain finite difference operator, not a regional pair kernel.
"""
import numpy as np
from examples.channel_nsch2d.problem import walls


def transport_shapes(c):
    nx,ny=c.sample.Nx+1,c.sample.Ny+1
    return dict(transport_dx=(nx,nx),transport_deta=(ny,ny),transport_weight_x=(nx,),
        transport_weight_eta=(ny,),transport_jacobian=(nx,),transport_measure=(nx*ny,))


def sbp_first(nodes,length):
    """Return D,H satisfying H*D+D.T*H=diag(-1,0,...,1)."""
    if isinstance(nodes,bool) or not isinstance(nodes,int) or nodes<9:
        raise ValueError('SBP4-2 requires at least nine uniformly spaced nodes')
    if not np.isfinite(length) or length<=0:raise ValueError('Positive finite SBP interval length required')
    h=length/(nodes-1);weights=np.ones(nodes)
    weights[:4]=np.array([17,59,43,49])/48;weights[-4:]=weights[:4][::-1]
    Q=np.zeros((nodes,nodes))
    left=np.array([[-.5,59/96,-1/12,-1/32,0,0],
        [-59/96,0,59/96,0,0,0],
        [1/12,-59/96,0,59/96,-1/12,0],
        [1/32,0,-59/96,0,2/3,-1/12]])
    Q[:4,:6]=left;Q[-4:,-6:]=-left[::-1,::-1]
    for i in range(4,nodes-4):Q[i,i-2:i+3]=np.array([1/12,-2/3,0,2/3,-1/12])
    norm=weights*h;D=Q/norm[:,None]
    boundary=np.zeros(nodes);boundary[0]=-1;boundary[-1]=1
    np.testing.assert_allclose(norm[:,None]*D+D.T*norm[None,:],np.diag(boundary),atol=2e-15,rtol=0)
    np.testing.assert_allclose(D@np.ones(nodes),0.,atol=2e-13,rtol=0)
    return D,norm


def build_transport(points,amplitudes,c):
    nx,ny=c.sample.Nx+1,c.sample.Ny+1
    Dx,Hx=sbp_first(nx,c.geometry.length);De,He=sbp_first(ny,1.)
    x=np.linspace(0,c.geometry.length,nx);eta=np.linspace(0,1,ny)
    bottom,top=walls(x,amplitudes,c,xp=np);height=top-bottom
    xx=np.broadcast_to(x[:,None],(nx,ny));yy=bottom[:,None]+eta[None,:]*height[:,None]
    expected=np.stack((xx,yy),axis=-1).reshape(-1,2)
    np.testing.assert_allclose(points,expected,atol=1e-12,rtol=0,
        err_msg='Mapped SBP transport requires the declared tensor-node ordering')
    return dict(transport_dx=Dx,transport_deta=De,transport_weight_x=Hx,
        transport_weight_eta=He,transport_jacobian=height,
        transport_measure=(Hx[:,None]*He[None,:]*height[:,None]).ravel())


def conservative_advection(phi,stream,data,*,xp):
    """Discrete mapped div(phi*v), with v represented by the same stream values."""
    nx=len(data['transport_jacobian']);ny=len(data['transport_weight_eta'])
    if phi.shape!=stream.shape or phi.ndim!=2 or phi.shape[1]!=nx*ny:
        raise ValueError('Transport requires aligned(time,ordered tensor nodes) phase and stream arrays')
    p=phi.reshape(-1,nx,ny);s=stream.reshape(p.shape)
    dx=lambda z:xp.einsum('ij,tjk->tik',data['transport_dx'],z)
    de=lambda z:xp.einsum('ij,tkj->tki',data['transport_deta'],z)
    qx=de(s);qe=-dx(s)
    rate=(dx(p*qx)+de(p*qe))/data['transport_jacobian'][None,:,None]
    return rate.reshape(phi.shape),qx,qe
