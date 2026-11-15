"""P1 Kuhn-tetrahedral weak diffusion with lumped mass on the unit cube.

Six tetrahedra partition each Cartesian cell. Element gradients are exact;
reaction and time terms use nodal/lumped quadrature. All faces retain their
natural zero-flux rows. The stored row weights are -K_ij / M_ii.
"""
from dataclasses import dataclass
from itertools import permutations
import os
import shlex

import numpy as np


METHODS = ('nonlocal', 'fem_p1_lumped', 'local_ad')


def validate_comparison(c):
    if (c.operator.spatial_discretization not in METHODS
            or 'spatial' not in c.sample
            or c.sample.spatial.mode != 'rolling_pool'
            or c.operator.action not in ('train', 'memory')
            or 'autotune_level' not in c.train
            or type(c.train.autotune_level) is not int or c.train.autotune_level != 0):
        raise ValueError('Melting spatial comparison requires an explicit method, rolling pool, '
                         'train/memory action and autotune_level=0')
    if c.operator.spatial_discretization == 'local_ad':
        from examples.icemelting3d.operator_local_ad import validate_local_ad
        validate_local_ad(c)
    elif 'local_ad' in c.operator:
        raise ValueError('Local-AD boundary settings cannot silently affect another method')


def configure_comparison(c):
    validate_comparison(c)
    flags = os.environ['XLA_FLAGS'] if 'XLA_FLAGS' in os.environ else ''
    found = [s for s in shlex.split(flags) if s.startswith('--xla_gpu_autotune_level=')]
    if any(s != '--xla_gpu_autotune_level=0' for s in found):
        raise ValueError('Inherited autotuning conflicts with the paired execution setting')
    if not found:
        os.environ['XLA_FLAGS'] = (flags+' --xla_gpu_autotune_level=0').strip()


def element_templates():
    """Integer vertices, unit-cell volumes and exact P1 stiffness matrices."""
    for order in permutations(range(3)):
        vertices = np.vstack((np.zeros(3, dtype=int),
                              np.cumsum(np.eye(3, dtype=int)[list(order)], axis=0)))
        affine = np.column_stack((np.ones(4), vertices))
        gradients = np.linalg.inv(affine)[1:]
        volume = abs(np.linalg.det(affine))/6
        yield vertices, volume, volume*(gradients.T@gradients)


@dataclass(frozen=True)
class WeakGraph:
    _N: int
    _nei_idx_np: np.ndarray
    _nei_w_np: np.ndarray
    _nei_mask_np: np.ndarray
    mass: np.ndarray


def build_cube(n):
    """Assemble translated element contributions without materializing all tets.

    Flattening is Y,X,Z, matching production meshgrid(indexing='xy'). Only
    axis couplings survive for this particular uniform tetrahedralization.
    No signed element terms are dropped or repaired.
    """
    if type(n) is not int or n < 3:
        raise ValueError('P1 cube requires an integer node count >=3')
    h = 1/(n-1)
    offsets = np.concatenate((np.eye(3, dtype=int), -np.eye(3, dtype=int)))
    direction = {tuple(v): i for i, v in enumerate(offsets)}
    mass = np.zeros((n,n,n), dtype=np.float64)
    conductance = np.zeros((n,n,n,6), dtype=np.float64)
    def region(vertex):
        return tuple(slice(int(v), int(v)+n-1) for v in vertex[[1,0,2]])
    for vertices, volume, stiffness in element_templates():
        for a in range(4):
            mass[region(vertices[a])] += volume*h**3/4
            for b in range(4):
                if a == b or stiffness[a,b] == 0:
                    continue
                displacement = tuple(vertices[b]-vertices[a])
                if stiffness[a,b] > 0 or displacement not in direction:
                    raise ValueError('Unexpected non-axial/signed Kuhn element coupling')
                conductance[region(vertices[a])+(direction[displacement],)] += -h*stiffness[a,b]
    if not np.isfinite(mass).all() or np.any(mass <= 0) or not np.isclose(mass.sum(), 1., rtol=1e-12):
        raise ValueError('Assembled positive lumped masses must integrate unit volume')
    linear = np.arange(n**3, dtype=np.int32).reshape(n,n,n)
    indices = np.empty((n,n,n,6), dtype=np.int32)
    valid = conductance > 0
    for k, offset in enumerate(offsets):
        shift = int(offset[1]*n*n+offset[0]*n+offset[2])
        indices[...,k] = np.where(valid[...,k], linear+shift, linear)
    weights = (conductance/mass[...,None]).reshape(-1,6)
    graph = WeakGraph(n**3, indices.reshape(-1,6), weights,
                      valid.reshape(-1,6), mass.reshape(-1))
    metadata = dict(spatial_discretization='fem_p1_lumped', grid_nodes=n**3,
        tetrahedra=6*(n-1)**3, directed_edges=int(valid.sum()), degree_max=6,
        mass='row-sum P1 mass', reaction='nodal quadrature',
        residual_norm='uniform sampled mass-normalized tanh-latent L1',
        boundary='natural homogeneous Neumann, all six faces',
        mass_sum=float(mass.sum()), mass_min=float(mass.min()), mass_max=float(mass.max()),
        actual_weight_min=float(weights[valid.reshape(-1,6)].min()),
        actual_weight_max=float(weights.max()), interior_quadratic_moment=1.)
    return graph, metadata
