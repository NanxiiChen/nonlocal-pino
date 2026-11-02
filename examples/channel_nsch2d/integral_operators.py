"""Original radial integral kernels with polynomial-moment corrections.

The uncorrected 2D quadratic kernel matches ExplicitVolumeSparseD. Correct its
gradient/Laplacian coefficients by minimum weighted-norm changes, preserving
the original kernel's moment-nullspace component. This is NOT the original
symmetric regional matrix or an automatic Neumann boundary closure.

Constrained reproducing quadrature: https://arxiv.org/abs/1801.04488
Higher-order nonlocal moments: https://doi.org/10.1007/s00366-021-01537-x
See README for the explicitly checked derivation and boundary semantics.
"""
import math

import numpy as np
from scipy.spatial import cKDTree


def validate_integral(c):
    s=c.nl
    if (set(s)!={'method','delta','neighbors','degree','kernel','condition_limit'}
            or s.method!='integral_moment' or s.kernel!='quadratic'):
        raise ValueError('Integral operator requires explicit integral_moment/quadratic settings')
    if (isinstance(s.degree,bool) or not isinstance(s.degree,int) or s.degree!=3
            or isinstance(s.neighbors,bool) or not isinstance(s.neighbors,int)
            or s.neighbors<10):
        raise ValueError('Integral pilot requires degree3 and integer padded capacity>=10')
    for name in ('delta','condition_limit'):
        value=s[name]
        if (isinstance(value,(bool,np.bool_)) or not np.isscalar(value)
                or np.asarray(value).dtype.kind not in 'fiu'
                or not np.isfinite(value) or value<=0):
            raise ValueError(f'Integral {name} must be a positive finite real scalar')
    if s.condition_limit<=1:
        raise ValueError('Integral condition_limit must exceed1')


def integral_stencils(points,volumes,targets,c):
    """Return padded indices, x/y gradients and Laplacian for physical targets.

For centered Taylor monomials B and W=diag(kernel*volume), minimize
sum((a-a0)^2/W) subject to B.T*a=t. Weighted QR evaluates the exact correction
W*B*(B.T*W*B)^-1*(t-B.T*a0), avoiding a normal-equation solve. Constants cancel
through neighbor-minus-target field values, including off-node targets.
"""
    validate_integral(c)
    x=np.asarray(points,dtype=np.float64);v=np.asarray(volumes,dtype=np.float64)
    y=np.asarray(targets,dtype=np.float64)
    if (x.ndim!=2 or x.shape[1]!=2 or y.ndim!=2 or y.shape[1]!=2
            or not len(x) or not len(y) or v.shape!=(len(x),)
            or not all(np.isfinite(a).all() for a in (x,v,y)) or np.any(v<=0)):
        raise ValueError('Integral operators require finite(N,2)/(M,2) points and positive(N,) volumes')
    tree=cKDTree(x)
    if np.any(tree.query(x,k=2)[0][:,1]<=0):
        raise ValueError('Duplicate integral source nodes')
    k=c.nl.neighbors;delta=c.nl.delta
    distance,index=tree.query(y,k=k+1,distance_upper_bound=delta)
    if np.any(np.isfinite(distance[:,-1])):
        raise ValueError(f'Integral horizon exceeds padded neighbor capacity{k}; increase explicit capacity')
    distance=distance[:,:k];index=index[:,:k]
    valid=np.isfinite(distance);count=valid.sum(axis=1)
    basis=[(a,total-a) for total in range(1,c.nl.degree+1) for a in range(total+1)]
    if np.any(count<len(basis)):
        raise ValueError(f'Integral horizon unresolved: minimum neighbors={count.min()}')
    index=np.where(valid,index,0).astype(np.int32)
    output=np.zeros((len(y),k,3));maximum_error=0.;maximum_condition=0.
    # Assembly blocks only bound CPU temporary memory; they do not alter support.
    for start in range(0,len(y),256):
        stop=min(start+256,len(y));ix=index[start:stop];mask=valid[start:stop]
        xi=np.where(mask[...,None],x[ix]-y[start:stop,None,:],0.)
        z=xi/delta
        B=np.stack([z[...,0]**a*z[...,1]**b/math.factorial(a)/math.factorial(b)
                    for a,b in basis],axis=-1)
        d=np.where(mask,distance[start:stop],0.)
        w=np.where(mask,120/(np.pi*delta**4)*(1-d/delta)**2*v[ix],0.)
        root=np.sqrt(w)
        Q,R=np.linalg.qr(root[...,None]*B,mode='reduced')
        condition=np.linalg.cond(R)
        if not np.isfinite(condition).all() or condition.max()>c.nl.condition_limit:
            raise ValueError(f'Integral moment matrix unresolved: max QR condition={condition.max()}')
        target=np.zeros((stop-start,len(basis),3))
        target[:,basis.index((1,0)),0]=1/delta
        target[:,basis.index((0,1)),1]=1/delta
        target[:,basis.index((2,0)),2]=1/delta**2
        target[:,basis.index((0,2)),2]=1/delta**2
        baseline=w[...,None]*np.concatenate((xi/2,np.ones((*xi.shape[:2],1))),axis=-1)
        mismatch=target-np.einsum('nkb,nka->nba',B,baseline)
        coefficient=np.linalg.solve(np.swapaxes(R,-1,-2),mismatch)
        stencil=baseline+root[...,None]*np.einsum('nkb,nba->nka',Q,coefficient)
        error=float(np.max(np.abs(np.einsum('nkb,nka->nba',B,stencil)-target)))
        if error>5e-9 or not np.isfinite(stencil).all():
            raise ValueError(f'Integral polynomial reproduction failed: {error}')
        output[start:stop]=stencil
        maximum_error=max(maximum_error,error)
        maximum_condition=max(maximum_condition,float(condition.max()))
    report=dict(method=c.nl.method,kernel=c.nl.kernel,horizon=delta,
        basis_degree=c.nl.degree,padded_capacity=k,minimum_neighbors=int(count.min()),
        maximum_neighbors=int(count.max()),mean_neighbors=float(count.mean()),
        polynomial_error=maximum_error,maximum_condition=maximum_condition,
        construction='Minimum weighted correction of original radial-integral coefficients',
        boundary='Truncated physical support with explicit physical BCs; no natural-flux substitution',
        symmetric_pair_matrix=False)
    return index,output[...,0],output[...,1],output[...,2],report
