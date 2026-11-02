"""Prototype native chemical recovery and explicit query interpolation.

Phi, velocity and pressure retain their continuous coordinate decoder. Interior
mu is a nodal constitutive recovery; boundary mu remains a decoder trace. The
query interpolation is a value/time operation, not an AD spatial-mu extension.
Enabled only by the explicit interior chemical-constraint research preset.
"""
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
import numpy as np
from scipy.spatial import cKDTree

from examples.channel_nsch2d.network import fields
from examples.channel_nsch2d.problem import walls
from examples.channel_nsch2d.chemical_projection import project_state


def _decoder_config(c):
    if 'chemical_constraint'not in c:raise ValueError('Explicit interior chemical constraint required')
    q=c.chemical_constraint
    if (set(q)!={'kind','boundary','query'} or q.kind!='interior_gmls_projection'
            or q.boundary!='decoder_trace' or q.query!='mapped_bilinear'):
        raise ValueError('Chemical constraint requires interior_gmls_projection/decoder_trace/mapped_bilinear')
    raw=ConfigDict(c.to_dict());del raw.chemical_constraint
    return raw


def make_decoder_fn(c):
    """Return point-decoded variables; mu is used only as a boundary trace."""
    raw=_decoder_config(c)
    return lambda net,e,x,t,a:fields(net,e,x,t,a,raw)


def make_domain_fn(c):
    decoded=make_decoder_fn(c)
    def value(net,e,data,times):
        raw=decoded(net,e,data['points'],times,data['amplitudes'])
        return project_state(raw,data,c,xp=jnp)
    return value


def make_domain_time_fn(c):
    value=make_domain_fn(c)
    return lambda net,e,data,t:jax.jvp(lambda ts:value(net,e,data,ts),(t,),(jnp.ones_like(t),))


def query_rule(data,points,c):
    """Precompute mapped bilinear values, exact nodal lookup and boundary traces."""
    _decoder_config(c)
    source=np.asarray(data['points']);x=np.asarray(points,dtype=np.float64)
    nx,ny=c.sample.Nx,c.sample.Ny
    if (source.shape!=((nx+1)*(ny+1),2) or x.ndim!=2 or x.shape[1]!=2
            or not np.isfinite(source).all()or not np.isfinite(x).all()):
        raise ValueError('Chemical query rule requires finite mapped source and query coordinates')
    if source.dtype.kind!='f':raise ValueError('Floating source coordinates required')
    eps=max(np.finfo(source.dtype).eps,np.finfo(c.train.compute_dtype).eps)
    xx,eta=np.meshgrid(np.linspace(0,c.geometry.length,nx+1),np.linspace(0,1,ny+1),indexing='ij')
    b,t=walls(xx,np.asarray(data['amplitudes']),c,xp=np)
    expected=np.stack((xx,b+eta*(t-b)),axis=-1).reshape(-1,2)
    tolerance=8*eps*max(c.geometry.length,c.geometry.height)
    if not np.allclose(source,expected,rtol=0,atol=tolerance):raise ValueError('Source ordering differs from the declared mapped tensor grid')
    bottom,top=walls(x[:,0],np.asarray(data['amplitudes']),c,xp=np)
    etaq=(x[:,1]-bottom)/(top-bottom);xi=x[:,0]/c.geometry.length
    if np.any((xi < -8*eps)|(xi>1+8*eps)|(etaq < -8*eps)|(etaq>1+8*eps)):
        raise ValueError('Chemical query coordinates outside the actual channel')
    # Only roundoff-size endpoint deviations are clipped after the range check.
    xi=np.clip(xi,0,1);etaq=np.clip(etaq,0,1)
    qx=nx*xi;qy=ny*etaq;ix=np.minimum(np.floor(qx).astype(np.int32),nx-1);iy=np.minimum(np.floor(qy).astype(np.int32),ny-1)
    a=qx-ix;b=qy-iy;i=ix*(ny+1)+iy
    index=np.stack((i,i+ny+1,i+1,i+ny+2),axis=-1)
    weight=np.stack(((1-a)*(1-b),a*(1-b),(1-a)*b,a*b),axis=-1)
    boundary=(xi<=8*eps)|(xi>=1-8*eps)|(etaq<=8*eps)|(etaq>=1-8*eps)
    distance,nearest=cKDTree(source.astype(np.float64)).query(x)
    exact=distance==0
    # Interpolation at a stored node is exactly its nodal unknown, including
    # boundary nodes. Other boundary queries use the retained decoder trace.
    index[exact]=nearest[exact,None];weight[exact]=np.array([1.,0.,0.,0.])
    if np.any(index<0)or np.any(index>=len(source))or np.any(weight<0):raise ValueError('Invalid chemical interpolation stencil')
    np.testing.assert_allclose(weight.sum(-1),1.,rtol=0,atol=2e-14)
    return dict(indices=index.astype(np.int32),weights=weight.astype(c.train.compute_dtype),boundary_trace=boundary&~exact)


def make_query_fn(c):
    domain=make_domain_fn(c);decoded=make_decoder_fn(c)
    def query(net,e,data,points,times,rule):
        count=len(points)
        if (set(rule)!={'indices','weights','boundary_trace'}or rule['indices'].shape!=(count,4)
                or rule['weights'].shape!=(count,4)or rule['boundary_trace'].shape!=(count,)):
            raise ValueError('Aligned explicit chemical query rule required')
        native=domain(net,e,data,times);raw=decoded(net,e,points,times,data['amplitudes'])
        mu=jnp.sum(native[:,rule['indices'],1]*rule['weights'][None],axis=-1)
        mu=jnp.where(rule['boundary_trace'][None],raw[...,1],mu)
        return jnp.stack((raw[...,0],mu,raw[...,2],raw[...,3],raw[...,4]),axis=-1)
    return query
