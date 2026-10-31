"""Input-only frozen CH residual metrics; NumPy/SciPy, no JAX or FEM."""
import numpy as np
from scipy import linalg,sparse


def linear_stable_resolvent(volumes,neighbors,coefficients,initial,beta,epsilon,mobility,time_scale):
    """Keep growing homogeneous-state modes; damp only the stable stiff part.

    A=-Delta_delta, S=sqrt(V) A /sqrt(V), m0=volume mean(initial),
    q(lambda)=M*lambda*(epsilon^2*lambda+beta*(3*m0^2-1)).
    Return V^-1/2 (I+time_scale*q(S)_+)^-1 V^1/2. The spectral
    positive part belongs to the definition, not an eigenvalue repair/fallback.
    """
    v=np.asarray(volumes);nb=np.asarray(neighbors);co=np.asarray(coefficients);initial=np.asarray(initial)
    if v.ndim!=1 or not len(v):raise ValueError('Nonempty one-dimensional physical volumes required')
    n=len(v)
    if (v.shape!=(n,) or initial.shape!=(n,) or nb.ndim!=2 or nb.shape[0]!=n
            or co.shape!=nb.shape or not np.issubdtype(nb.dtype,np.integer)
            or np.any(nb<0) or np.any(nb>=n) or np.any(v<=0) or np.any(co<0)):
        raise ValueError('Positive physical volumes, aligned initial data and nonnegative graph required')
    if not all(np.isfinite(a).all() for a in (v,co,initial)) or not np.isfinite([beta,epsilon,mobility,time_scale]).all() or min(beta,epsilon,mobility,time_scale)<=0:
        raise ValueError('Finite graph/initial arrays and positive physical/time scales required')
    A=sparse.diags(co.sum(axis=1))+sparse.coo_matrix((-co.ravel(),
        (np.repeat(np.arange(n),nb.shape[1]),nb.ravel())),shape=(n,n)).tocsr()
    root=np.sqrt(v);S=(sparse.diags(root)@A@sparse.diags(1/root)).toarray()
    if not np.isfinite(S).all():raise FloatingPointError('Nonfinite volume-weighted graph')
    tolerance=100*np.finfo(S.dtype).eps*n*max(1.,np.linalg.norm(S,ord=np.inf))
    if np.max(abs(S-S.T))>tolerance:raise ValueError('CH metric requires a volume-self-adjoint graph')
    values,U=linalg.eigh(S,check_finite=False,driver='evd')
    if values[0]<-tolerance:raise ValueError('CH metric requires positive semidefinite -Delta')
    mean=float(v@initial/v.sum());curvature=beta*(3*mean**2-1)
    rates=mobility*values*(epsilon**2*values+curvature)
    weights=1/(1+time_scale*np.maximum(rates,0))
    P=((U*weights)@U.T)*root[None,:]/root[:,None]
    constant_error=float(abs(P@np.ones(n)-1).max())
    if not np.isfinite(P).all() or np.any(weights<=0) or np.any(weights>1) or constant_error>1e-9:
        raise FloatingPointError('Nonfinite/nonpositive metric or failed constant-mode preservation')
    return P,dict(nodes=n,initial_mean=mean,homogeneous_curvature=curvature,
        laplacian_min=float(values[0]),laplacian_max=float(values[-1]),
        minimum_weight=float(weights.min()),maximum_weight=float(weights.max()),
        growing_modes=int(np.sum(rates < -tolerance)),constant_mode_error=constant_error)


def metric_job(arguments):
    """Top-level spawn-safe worker; arguments contain physical inputs only."""
    return linear_stable_resolvent(*arguments)
