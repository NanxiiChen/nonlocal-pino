"""Frozen biharmonic resolvent metric in the physical-volume inner product."""
import equinox as eqx
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
import numpy as np
from scipy import sparse,linalg


class BiharmonicMetric(eqx.Module):
    factor:object
    sqrt_volumes:object

    def __init__(self,D,volumes,*,time_scale,mobility,epsilon):
        if not np.isfinite([time_scale,mobility,epsilon]).all() or min(time_scale,mobility,epsilon)<=0:
            raise ValueError('Resolvent scales must be finite positive')
        v=np.asarray(volumes);neighbors=np.asarray(D.neighbors);weights=np.asarray(D.coefficients)
        n=len(v)
        off=sparse.coo_matrix((weights.ravel(),(np.repeat(np.arange(n),neighbors.shape[1]),neighbors.ravel())),shape=(n,n)).tocsr()
        A=sparse.diags(np.asarray(off.sum(axis=1)).ravel())-off
        root=np.sqrt(v);S=sparse.diags(root)@A@sparse.diags(1/root)
        # S is symmetric in exact arithmetic; S^T S keeps roundoff SPD without
        # jitter or a silent symmetrization. This is A^dagger A in L2(volumes).
        scale=time_scale*mobility*epsilon**2
        P=(sparse.eye(n,format='csr')+scale*(S.T@S)).toarray()
        factor=linalg.cholesky(P,lower=True,check_finite=True)
        self.factor=jnp.asarray(factor);self.sqrt_volumes=jnp.asarray(root)

    def __call__(self,residual):
        z=solve_triangular(self.factor,self.sqrt_volumes*residual,lower=True)
        z=solve_triangular(self.factor.T,z,lower=False)
        return z/self.sqrt_volumes


class SpaceTimeMetric(eqx.Module):
    """Invert a stable linear evolution on residuals, not on predicted fields.

    With K=M*epsilon^2*A^2, solve (I+h_i*K)e_i=e_{i-1}+h_i*r_i.
    The lower triangular map is invertible for positive, ordered query times.
    Parameter gradients pass through every residual and every recurrence step.
    """
    eigenvectors: object
    rates: object
    sqrt_volumes: object

    def __init__(self,D,volumes,*,mobility,epsilon):
        if not np.isfinite([mobility,epsilon]).all() or min(mobility,epsilon)<=0:
            raise ValueError('Space-time metric needs positive finite mobility/epsilon')
        v=np.asarray(volumes)
        if v.ndim!=1 or not np.isfinite(v).all() or np.any(v<=0):
            raise ValueError('Space-time metric requires positive physical volumes')
        n=len(v);neighbors=np.asarray(D.neighbors);weights=np.asarray(D.coefficients)
        off=sparse.coo_matrix((weights.ravel(),(np.repeat(np.arange(n),neighbors.shape[1]),neighbors.ravel())),shape=(n,n)).tocsr()
        A=sparse.diags(np.asarray(off.sum(axis=1)).ravel())-off
        root=np.sqrt(v);S=(sparse.diags(root)@A@sparse.diags(1/root)).toarray()
        defect=np.max(np.abs(S-S.T));scale=max(1.,np.max(np.abs(S)))
        if defect>1e-12*scale:raise ValueError('Space-time metric requires weighted-self-adjoint A')
        # Explicitly remove only checked assembly roundoff in the metric.
        eigenvalues,eigenvectors=linalg.eigh(.5*(S+S.T),check_finite=True)
        if eigenvalues[0]<-1e-10*scale:raise ValueError('Space-time metric requires positive-semidefinite A')
        self.eigenvectors=jnp.asarray(eigenvectors)
        self.rates=jnp.asarray(mobility*epsilon**2*eigenvalues**2)
        self.sqrt_volumes=jnp.asarray(root)

    def __call__(self,residuals,times):
        import jax
        h=jnp.diff(jnp.concatenate((jnp.zeros(1,dtype=times.dtype),times)))
        transformed=(residuals*self.sqrt_volumes)@self.eigenvectors
        def advance(previous,data):
            dt,r=data
            now=(previous+dt*r)/(1+dt*self.rates)
            return now,now
        _,error=jax.lax.scan(advance,jnp.zeros_like(self.rates),(h,transformed))
        return (error@self.eigenvectors.T)/self.sqrt_volumes

    def resolvent(self,residuals,times):
        """Apply (I+t*K)^-1 separately to integrated residual rows."""
        transformed=(residuals*self.sqrt_volumes)@self.eigenvectors
        errors=transformed/(1+times[:,None]*self.rates)
        return (errors@self.eigenvectors.T)/self.sqrt_volumes


class NeumannInverseMetric(eqx.Module):
    """A^-1 on zero-mean residuals, with only the constant null mode lifted."""
    factor: object
    sqrt_volumes: object

    def __init__(self,D,volumes):
        v=np.asarray(volumes)
        if v.ndim!=1 or not np.isfinite(v).all() or np.any(v<=0):raise ValueError('Positive physical volumes required')
        n=len(v);neighbors=np.asarray(D.neighbors);weights=np.asarray(D.coefficients)
        off=sparse.coo_matrix((weights.ravel(),(np.repeat(np.arange(n),neighbors.shape[1]),neighbors.ravel())),shape=(n,n)).tocsr()
        from scipy.sparse.csgraph import connected_components
        off.eliminate_zeros()
        if connected_components(off,directed=False,return_labels=False)!=1:
            raise ValueError('Neumann inverse needs a connected graph with only one constant null mode')
        A=sparse.diags(np.asarray(off.sum(axis=1)).ravel())-off
        root=np.sqrt(v);S=(sparse.diags(root)@A@sparse.diags(1/root)).toarray()
        if np.max(abs(S-S.T))>1e-12*max(1.,np.max(abs(S))):raise ValueError('Weighted symmetry required')
        q=root/np.linalg.norm(root)
        # Explicit constant-mode lift; no jitter. Disconnected graphs fail.
        factor=linalg.cholesky(.5*(S+S.T)+np.outer(q,q),lower=True,check_finite=True)
        self.factor=jnp.asarray(factor);self.sqrt_volumes=jnp.asarray(root)

    def __call__(self,residual):
        z=solve_triangular(self.factor,self.sqrt_volumes*residual,lower=True)
        z=solve_triangular(self.factor.T,z,lower=False)
        return z/self.sqrt_volumes
