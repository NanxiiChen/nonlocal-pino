"""PDE-consistent axis reflections with a REEVALUATED fixed ambient IC.

The ellipse and declared prior must be invariant under these isometries.
Pair distances/visibility/volumes are unchanged, so A and its resolvent retain
their entries. Boundary normals transform as vectors. Future fields are never
transformed or used; reflecting phi0 values would change the prescribed IC.
"""
import jax.numpy as jnp
from examples.confined_ch2d.operator_problem import fixed_initial


def axis_reflections(batch,bits,c):
    if bits.shape!=(batch['points'].shape[0],2):
        raise ValueError('Axis reflection bits must have shape(batch,2)')
    signs=jnp.where(bits,-1.,1.)[:,None,:]
    center=jnp.asarray(c.geometry.center,dtype=batch['points'].dtype)
    points=center+(batch['points']-center)*signs
    boundary=batch['condition']
    boundary=jnp.concatenate((center+(boundary[...,:2]-center)*signs,
                              boundary[...,2:4]*signs,boundary[...,4:]),axis=-1)
    result=dict(batch)
    result['points']=points
    result['initial']=jnp.where(batch['volumes']>0,fixed_initial(points,c,xp=jnp),0.)
    result['condition']=boundary
    return result
