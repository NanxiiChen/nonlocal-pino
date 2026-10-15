"""Original dynamic-H objective and explicit user-provisional loading policy."""
import jax
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from examples.fracture2d._legacy.core_halo_damage import predict_rows
from examples.fracture2d._legacy.compact_damage_features import residual_power_loss
from examples.fracture2d._legacy.range_damage_optimization import augmented_term


def make_dynamic_problem(basis_columns,basis_values,matvec,*,gc0,gc1,power,floor,trunk_gain):
    def predict(net,context,load):
        return predict_rows(net,jax.lax.stop_gradient(context['features']),basis_columns,basis_values,trunk_gain=trunk_gain)
    def objective(net,context,load):
        c=jax.tree.map(jax.lax.stop_gradient,context)
        damage=predict(net,c,load)
        residual=(gc0+2*c['H'])*damage+gc1*matvec(damage)-2*c['H']
        return (residual_power_loss(residual,c['H'],power=power,floor=floor)
            +augmented_term(damage,c['H'],c['lower'],c['upper'],c['penalty'],gc0=gc0))
    return predict,objective


def advancement(*,user_authorized,strict_ready,numerically_valid,provisional_ancestry):
    if not all(type(value) is bool for value in [user_authorized,strict_ready,numerically_valid,provisional_ancestry]):
        raise ValueError('loading policy requires explicit boolean authorization/validity/ancestry')
    advance=numerically_valid and (strict_ready or user_authorized)
    return dict(load_advanced=advance,local_strict_ready=strict_ready,
        status='numerical_rejected' if not numerically_valid else 'converged' if strict_ready
            else 'user_provisional' if user_authorized else 'convergence_rejected',
        user_forced_advance=advance and not strict_ready,
        certified_path=advance and strict_ready and not provisional_ancestry)


def mesh_adjacency(triangles,*,population_size):
    t=np.asarray(triangles)
    if (t.ndim!=2 or t.shape[1]!=3 or not np.issubdtype(t.dtype,np.integer)
            or type(population_size) is not int or population_size<1
            or t.size==0 or t.min()<0 or t.max()>=population_size):
        raise ValueError('invalid material triangles for crack-connectivity diagnostic')
    row=np.concatenate([t[:,0],t[:,1],t[:,2],t[:,1],t[:,2],t[:,0]])
    column=np.concatenate([t[:,1],t[:,2],t[:,0],t[:,0],t[:,1],t[:,2]])
    matrix=coo_matrix((np.ones(len(row),np.float64),(row,column)),shape=(population_size,population_size)).tocsr()
    matrix.data[:]=1.
    return matrix


def crack_extent(damage,points,adjacency,*,threshold,tip,tip_radius,right_edge,tolerance):
    d,x=np.asarray(damage),np.asarray(points);tip=np.asarray(tip,np.float64)
    if (d.ndim!=1 or x.shape!=(len(d),2) or adjacency.shape!=(len(d),len(d))
            or not np.isfinite(d).all() or not np.isfinite(x).all() or tip.shape!=(2,)
            or not np.isfinite(tip).all() or not 0<threshold<=1 or not np.isfinite(tip_radius)
            or tip_radius<=0 or not np.isfinite(right_edge) or not np.isfinite(tolerance) or tolerance<=0):
        raise ValueError('invalid explicit crack-connectivity inputs')
    active=np.flatnonzero(d>=threshold)
    if not len(active):return dict(spans_ligament=False,tip_component_xmax=None,active_nodes=0)
    _,labels=connected_components(adjacency[active][:,active],directed=False)
    near=np.sum((x[active]-tip)**2,axis=1)<=tip_radius**2
    selected=np.isin(labels,np.unique(labels[near]))
    component=active[selected]
    return dict(spans_ligament=bool(len(component) and np.any(abs(x[component,0]-right_edge)<=tolerance)),
        tip_component_xmax=float(np.max(x[component,0])) if len(component) else None,active_nodes=int(len(active)))
