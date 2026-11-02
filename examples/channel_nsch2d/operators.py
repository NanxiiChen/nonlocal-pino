"""Finite-neighborhood polynomial-reproducing Laplacian on physical nodes.

GMLS functional reconstruction (Compadre mathematical overview):
https://github.com/sandialabs/compadre#generalized-moving-least-squares-gmls
This is a meshfree differentiation stencil, not the regional energy pair kernel.
Boundary values must be imposed explicitly; no automatic no-slip claim is made.
"""
import math
import numpy as np
from scipy.spatial import cKDTree


def gradient_shapes(c):
    n=(c.sample.Nx+1)*(c.sample.Ny+1);k=c.nl.neighbors
    b=4*(c.boundary_sampling.points_per_side if 'boundary_sampling' in c else c.sample.boundary_points)
    return dict(gradient_x=(n,k),gradient_y=(n,k),boundary_neighbors=(b,k),
        boundary_gradient_x=(b,k),boundary_gradient_y=(b,k))


def first_derivative_stencils(points,targets,c):
    """Physical x/y GMLS functionals, including off-node boundary targets."""
    x=np.asarray(points,dtype=np.float64);y=np.asarray(targets,dtype=np.float64)
    if x.ndim!=2 or x.shape[1]!=2 or y.ndim!=2 or y.shape[1]!=2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('GMLS first derivatives require finite source/target(N,2) arrays')
    degree=c.nl.degree;k=c.nl.neighbors
    basis=[(a,total-a) for total in range(degree+1) for a in range(total+1)]
    if c.nl.method!='gmls' or degree<2 or not len(basis)<k<=len(x) or c.nl.weight_exponent<=0:
        raise ValueError('Require explicit degree>=2 GMLS and more neighbors than polynomial terms')
    if np.any(cKDTree(x).query(x,k=2)[0][:,1]<=0):raise ValueError('Duplicate GMLS source nodes')
    distance,indices=cKDTree(x).query(y,k=k);radius=1.25*distance[:,-1]
    delta=(x[indices]-y[:,None,:])/radius[:,None,None]
    P=np.stack([delta[...,0]**a*delta[...,1]**b/math.factorial(a)/math.factorial(b)
        for a,b in basis],axis=-1)
    weight=(1-distance/radius[:,None])**c.nl.weight_exponent
    Q,R=np.linalg.qr(np.sqrt(weight)[...,None]*P,mode='reduced')
    condition=np.linalg.cond(R)
    if not np.isfinite(condition).all() or condition.max()>c.nl.condition_limit:
        raise ValueError(f'Unresolved GMLS gradient: max condition={condition.max()}')
    target=np.zeros((len(y),len(basis),2))
    target[:,basis.index((1,0)),0]=1/radius
    target[:,basis.index((0,1)),1]=1/radius
    coefficient=np.linalg.solve(np.swapaxes(R,-1,-2),target)
    stencil=np.sqrt(weight)[...,None]*np.einsum('nkj,nja->nka',Q,coefficient)
    error=float(np.max(np.abs(np.einsum('nkj,nka->nja',P,stencil)-target)))
    if error>5e-9:raise ValueError(f'GMLS gradient polynomial reproduction failed: {error}')
    return indices.astype(np.int32),stencil[...,0],stencil[...,1],dict(
        polynomial_error=error,maximum_condition=float(condition.max()),
        maximum_radius=float(radius.max()),basis_degree=degree,neighbors=k)


def laplacian_stencils(points,c):
    x=np.asarray(points,dtype=np.float64);n=len(x);k=c.nl.neighbors;degree=c.nl.degree
    if c.nl.method!='gmls' or x.shape!=(n,2) or not np.isfinite(x).all():
        raise ValueError('Explicit gmls method and finite(N,2) physical nodes required')
    basis=[(a,b) for total in range(degree+1) for a in range(total+1) for b in [total-a]]
    if degree<2 or not len(basis)<k<=n or c.nl.weight_exponent<=0:
        raise ValueError('Need degree>=2, positive weights and more neighbors than basis functions')
    distance,indices=cKDTree(x).query(x,k=k)
    if np.any(distance[:,1]<=0):raise ValueError('Duplicate physical nodes')
    radius=1.25*distance[:,-1]
    delta=(x[indices]-x[:,None,:])/radius[:,None,None]
    P=np.stack([delta[...,0]**a*delta[...,1]**b/math.factorial(a)/math.factorial(b)
                for a,b in basis],axis=-1)
    weight=(1-distance/radius[:,None])**c.nl.weight_exponent
    Q,R=np.linalg.qr(np.sqrt(weight)[...,None]*P,mode='reduced')
    condition=np.linalg.cond(R)
    if not np.isfinite(condition).all() or np.max(condition)>c.nl.condition_limit:
        raise ValueError(f'Unresolved GMLS stencil: max condition={np.max(condition)}')
    target=np.zeros((n,len(basis)))
    target[:,basis.index((2,0))]=1/radius**2
    target[:,basis.index((0,2))]=1/radius**2
    coefficients=np.linalg.solve(np.swapaxes(R,-1,-2),target[...,None])[...,0]
    stencil=np.sqrt(weight)*np.einsum('nkj,nj->nk',Q,coefficients)
    error=float(np.max(np.abs(np.einsum('nk,nkj->nj',stencil,P)-target)))
    if error>5e-9:raise ValueError(f'GMLS polynomial reproduction failed: {error}')
    return indices.astype(np.int32),stencil,dict(polynomial_error=error,
        maximum_condition=float(np.max(condition)),maximum_radius=float(radius.max()),
        basis_degree=degree,neighbors=k)
