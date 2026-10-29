"""Frozen inverse of the global linear mixed CH Volterra residual block."""
import equinox as eqx
import jax.numpy as jnp
import numpy as np
from scipy import sparse,linalg


class MixedBlockMetric(eqx.Module):
    eigenvectors: object
    eigenvalues: object
    sqrt_volumes: object
    time_inverse: object
    integration: object
    mobility: float=eqx.field(static=True)
    epsilon_squared: float=eqx.field(static=True)

    def __init__(self,D,volumes,W,*,mobility,epsilon):
        v=np.asarray(volumes);W=np.asarray(W)
        if v.ndim!=1 or not np.isfinite(v).all() or np.any(v<=0):raise ValueError('Positive physical volumes required')
        if W.ndim!=2 or W.shape[0]!=W.shape[1] or not np.isfinite(W).all() or np.any(W[0]!=0):raise ValueError('Finite square global integration matrix with zero initial row required')
        if not np.isfinite([mobility,epsilon]).all() or min(mobility,epsilon)<=0:raise ValueError('Positive finite mobility/epsilon required')
        n=len(v);neighbors=np.asarray(D.neighbors);weights=np.asarray(D.coefficients)
        off=sparse.coo_matrix((weights.ravel(),(np.repeat(np.arange(n),neighbors.shape[1]),neighbors.ravel())),shape=(n,n)).tocsr()
        A=sparse.diags(np.asarray(off.sum(axis=1)).ravel())-off
        root=np.sqrt(v);S=(sparse.diags(root)@A@sparse.diags(1/root)).toarray()
        scale=max(1.,float(np.max(abs(S))))
        if np.max(abs(S-S.T))>1e-12*scale:raise ValueError('Weighted self-adjoint operator required')
        eigenvalues,eigenvectors=linalg.eigh(.5*(S+S.T),check_finite=True)
        if eigenvalues[0]<-1e-10*scale:raise ValueError('Positive semidefinite operator required')
        # Constant mode is mathematically zero. Remove only checked eigensolver roundoff.
        eigenvalues=np.maximum(eigenvalues,0.)
        rates=mobility*epsilon**2*eigenvalues**2;identity=np.eye(len(W))
        blocks=identity[None,:,:]+rates[:,None,None]*W[None,:,:]
        inverses=np.linalg.solve(blocks,np.broadcast_to(identity,blocks.shape))
        defect=float(np.max(abs(blocks@inverses-identity)))
        if not np.isfinite(inverses).all() or defect>1e-8:raise ValueError(f'Global mixed inverse fails backward check: {defect}')
        self.eigenvectors=jnp.asarray(eigenvectors);self.eigenvalues=jnp.asarray(eigenvalues)
        self.sqrt_volumes=jnp.asarray(root);self.time_inverse=jnp.asarray(inverses)
        self.integration=jnp.asarray(W);self.mobility=mobility;self.epsilon_squared=epsilon**2

    def __call__(self,evolution,chemical):
        expected=(self.integration.shape[0],self.sqrt_volumes.shape[0])
        if evolution.shape!=expected or chemical.shape!=expected:raise ValueError('Mixed residual shape must match fixed(time,space) block')
        first=(evolution*self.sqrt_volumes)@self.eigenvectors
        second=(chemical*self.sqrt_volumes)@self.eigenvectors
        right=first-self.mobility*(self.integration@second)*self.eigenvalues
        phase=jnp.einsum('nij,jn->in',self.time_inverse,right)
        mu=second+self.epsilon_squared*phase*self.eigenvalues
        return (phase@self.eigenvectors.T)/self.sqrt_volumes,(mu@self.eigenvectors.T)/self.sqrt_volumes
