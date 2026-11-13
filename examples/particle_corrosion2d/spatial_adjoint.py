"""Fixed linear adjoints avoid atomic gather-transpose sums in parameter backprop.

Forward edge differences are unchanged. Only field derivatives are exposed:
geometry, indices and operator coefficients are prescribed, not learned inputs.
"""
import jax
import jax.numpy as jnp
import numpy as np


def validate_support(data):
    indices=np.where(np.asarray(data['coefficients'])!=0,np.asarray(data['neighbors']),-1)
    ordered=np.sort(indices,axis=-1)
    if np.any((ordered[...,1:]==ordered[...,:-1]) & (ordered[...,1:]>=0)):
        raise ValueError('Fixed adjoint requires unique nonzero neighbors within each row')


def prepare_adjoint(data):
    """One selected geometry only; do not densify the entire rolling pool.

    Each physical neighbor appears once per row. Zero padding is sent out of
    bounds, so it cannot race with a physical neighbor at index zero. Repeated
    sampled core IDs occupy distinct rows and remain valid.
    """
    n_halo=len(data['graph_points'])
    core,neighbors,coefficients=(data[k] for k in ('core_indices','neighbors','coefficients'))
    selection=jax.nn.one_hot(core,n_halo,dtype=coefficients.dtype)
    rows=jnp.broadcast_to(jnp.arange(len(core))[:,None],neighbors.shape)
    columns=jnp.where(coefficients!=0,neighbors,n_halo)
    edges=jnp.zeros_like(selection).at[rows,columns].add(coefficients,mode='drop')
    matrix=selection*jnp.sum(coefficients,axis=1)[:,None]-edges
    return dict(data,core_selection=jax.lax.stop_gradient(selection),
                operator_matrix=jax.lax.stop_gradient(matrix))


def _edges(values,core,neighbors,coefficients):
    return jnp.sum(coefficients[:,:,None]*(values[core,None,:]-values[neighbors]),axis=1)


@jax.custom_vjp
def fixed_graph(values,core,neighbors,coefficients,matrix):
    return _edges(values,core,neighbors,coefficients)


def _graph_forward(values,core,neighbors,coefficients,matrix):
    return _edges(values,core,neighbors,coefficients),matrix


def _graph_backward(matrix,cotangent):
    return matrix.T@cotangent,None,None,None,None


fixed_graph.defvjp(_graph_forward,_graph_backward)


@jax.custom_vjp
def fixed_gather(values,core,selection):
    return values[core]


def _gather_forward(values,core,selection):
    return values[core],selection


def _gather_backward(selection,cotangent):
    return selection.T@cotangent,None,None


fixed_gather.defvjp(_gather_forward,_gather_backward)
