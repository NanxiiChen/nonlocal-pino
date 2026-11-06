"""Known geometry/function measures and fixed-shape sampled PDE payloads.

No evolved solution is computed here. Core and IC nodes are sampled with
replacement in proportion to P1 lumped mass; their unweighted empirical means
estimate the respective physical-domain means. Every core retains all its
visible horizon neighbors. Boundary quadrature is independent of mesh nodes.
"""
import numpy as np

from .geometry import boundary, contains
from .problem import initial_fields


def conditioning(c, record):
    ny,nx = c.operator.domain_shape
    x=(np.arange(nx)+.5)*c.geometry.length/nx
    y=(np.arange(ny)+.5)*c.geometry.height/ny
    xx,yy=np.meshgrid(x,y,indexing='xy')
    points=np.stack((xx.ravel(),yy.ravel()),axis=-1)
    weights=contains(points,record,c).astype(float)*c.geometry.length*c.geometry.height/(nx*ny)
    tokens=np.column_stack((points,initial_fields(c,points)[:,0]))
    walls,normals,line_weights,kind=wall_quadrature(c,record)
    mouth,mouth_weights=mouth_quadrature(c)
    boundary_tokens=np.concatenate((np.column_stack((walls,normals,kind)),
        np.column_stack((mouth,np.tile([0.,-1.],(len(mouth),1)),np.full(len(mouth),2.)))))
    return dict(domain_tokens=tokens,domain_weights=weights,
        boundary_tokens=boundary_tokens,boundary_weights=np.r_[line_weights,mouth_weights],
        wall_points=walls,wall_normals=normals,wall_weights=line_weights,
        mouth_points=mouth,mouth_weights=mouth_weights)


def mouth_quadrature(c):
    n=c.operator.mouth_points
    a=c.geometry.pit_x-c.geometry.mouth_half_width
    length=2*c.geometry.mouth_half_width
    points=np.column_stack((a+(np.arange(n)+.5)*length/n,np.zeros(n)))
    return points,np.full(n,length/n)


def wall_quadrature(c, record):
    """PDE-domain outward normals; ceramic outward normals have opposite sign."""
    length,height=c.geometry.length,c.geometry.height
    a=c.geometry.pit_x-c.geometry.mouth_half_width
    b=c.geometry.pit_x+c.geometry.mouth_half_width
    segments=[((0,0),(a,0),(0,-1)),((b,0),(length,0),(0,-1)),
              ((0,height),(length,height),(0,1)),((0,0),(0,height),(-1,0)),
              ((length,0),(length,height),(1,0))]
    points,normals,weights,kinds=[],[],[],[]
    n=c.operator.wall_points_per_side
    s=(np.arange(n)+.5)/n
    for begin,end,normal in segments:
        begin,end=np.asarray(begin),np.asarray(end)
        points.append(begin+s[:,None]*(end-begin))
        normals.append(np.tile(normal,(n,1)))
        weights.append(np.full(n,np.linalg.norm(end-begin)/n))
        kinds.append(np.zeros(n))
    n=c.operator.wall_points_per_particle
    theta=2*np.pi*(np.arange(n)+.5)/n
    # Arc speed from the analytic contour derivative, evaluated by normals'
    # perpendicular dot exact radial/ellipse derivative (same parameterization).
    for particle in record['particles']:
        xy,normal,_=boundary(particle,theta)
        modes=np.asarray(particle['modes'])
        acoef,bcoef=np.asarray(particle['cosine']),np.asarray(particle['sine'])
        phase=theta[:,None]*modes
        radial=particle['radial_scale']*np.exp(np.cos(phase)@acoef+np.sin(phase)@bcoef)
        dlog=(-np.sin(phase)*modes)@acoef+(np.cos(phase)*modes)@bcoef
        derivative=np.column_stack((radial*(dlog*np.cos(theta)-np.sin(theta)),
                                    radial*(dlog*np.sin(theta)+np.cos(theta))))
        derivative*=np.array([np.sqrt(particle['aspect']),1/np.sqrt(particle['aspect'])])
        speed=np.linalg.norm(derivative,axis=-1)
        points.append(xy); normals.append(-normal); weights.append(speed*2*np.pi/n)
        kinds.append(np.ones(n))
    return np.concatenate(points),np.concatenate(normals),np.concatenate(weights),np.concatenate(kinds)


def graph_payload(c, mesh, matrix, rng):
    """A complete core+halo subgraph, not an induced/truncated core graph."""
    points,volumes=mesh['points'],mesh['volumes']
    free=np.flatnonzero(~mesh['mouth'])
    core=rng.choice(free,size=c.sample.K_core,replace=True,p=volumes[free]/volumes[free].sum())
    rows=[]; coefficients=[]
    for idx in core:
        row=matrix.getrow(int(idx))
        select=(row.indices!=idx)&(row.data<0)
        rows.append(row.indices[select]); coefficients.append(-row.data[select])
    halo=np.unique(np.concatenate([core,*rows]))
    maximum=max(len(row) for row in rows)
    if len(halo)>c.nl.max_halo or maximum>c.nl.max_neighbors:
        raise ValueError(f'Core+halo overflow: halo={len(halo)}/{c.nl.max_halo}, neighbors={maximum}/{c.nl.max_neighbors}')
    neighbors=np.zeros((len(core),maximum),dtype=np.int32)
    coef=np.zeros(neighbors.shape)
    for i,(row,values) in enumerate(zip(rows,coefficients,strict=True)):
        neighbors[i,:len(row)]=np.searchsorted(halo,row)
        coef[i,:len(row)]=values
    return dict(graph_points=points[halo],core_indices=np.searchsorted(halo,core).astype(np.int32),
        neighbors=neighbors,coefficients=coef,pde_weights=np.ones(len(core)))


def initial_payload(c, mesh, rng):
    ids=rng.choice(len(mesh['points']),size=c.operator.ic_points,replace=True,
                   p=mesh['volumes']/mesh['volumes'].sum())
    points=mesh['points'][ids]
    return dict(ic_points=points,initial_ic=initial_fields(c,points),ic_weights=np.ones(len(ids)))


def pad_graph(payload, halo_size, neighbor_size):
    result=dict(payload)
    n,k=payload['neighbors'].shape
    if len(payload['graph_points'])>halo_size or k>neighbor_size:
        raise ValueError('Explicit fixed payload shape is too small')
    # Finite duplicate padding never contributes to physical quadrature.
    points=np.tile(payload['graph_points'][0],(halo_size,1))
    points[:len(payload['graph_points'])]=payload['graph_points']
    neighbors=np.zeros((n,neighbor_size),dtype=np.int32)
    coefficients=np.zeros((n,neighbor_size))
    neighbors[:,:k]=payload['neighbors'];coefficients[:,:k]=payload['coefficients']
    result.update(graph_points=points,neighbors=neighbors,coefficients=coefficients)
    return result
