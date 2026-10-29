"""Positive regional pair Laplacian on physical triangle-centroid quadrature."""
import equinox as eqx
import jax.numpy as jnp
import numpy as np
from scipy.spatial import cKDTree
from scipy import sparse
from examples.confined_ch2d.geometry import visible_pairs


class RegionalLaplacian(eqx.Module):
    neighbors: object
    coefficients: object

    def __init__(self,points,volumes,c):
        x=np.asarray(points);v=np.asarray(volumes);delta=c.nl.delta
        if x.ndim!=2 or x.shape[1]!=2 or v.shape!=(len(x),) or not np.isfinite(x).all() or not np.isfinite(v).all() or np.any(v<=0):
            raise ValueError('Require finite(N,2) points and positive physical volumes')
        pairs=cKDTree(x).query_pairs(delta,output_type='ndarray')
        pairs=pairs[visible_pairs(x,pairs,c.geometry.holes)]
        if len(pairs)==0:raise ValueError('No visible bonds; refine geometry or horizon')
        i,j=pairs.T;r=np.linalg.norm(x[i]-x[j],axis=1)
        if np.any(r==0):raise ValueError('Duplicate quadrature points')
        active=r<delta;i,j,r=i[active],j[active],r[active]
        kernel=120/(np.pi*delta**4)*(1-r/delta)**2
        off=sparse.coo_matrix((np.r_[kernel*v[j],kernel*v[i]],(np.r_[i,j],np.r_[j,i])),shape=(len(x),len(x))).tocsr()
        counts=np.diff(off.indptr)
        if np.any(counts==0):raise ValueError('Isolated quadrature points')
        neighbors=np.repeat(np.arange(len(x))[:,None],counts.max(),axis=1)
        weights=np.zeros(neighbors.shape)
        for row in range(len(x)):
            sl=slice(off.indptr[row],off.indptr[row+1]);k=counts[row]
            neighbors[row,:k]=off.indices[sl];weights[row,:k]=off.data[sl]
        self.neighbors=jnp.asarray(neighbors,dtype=jnp.int32)
        self.coefficients=jnp.asarray(weights,dtype=jnp.float64)

    def __matmul__(self,values):
        if values.ndim!=1 or values.shape[0]!=self.neighbors.shape[0]:raise ValueError('Expected scalar full-grid field(N,)')
        return jnp.sum(self.coefficients*(values[:,None]-values[self.neighbors]),axis=1)
