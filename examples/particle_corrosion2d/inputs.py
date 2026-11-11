"""Known geometry/function measures and fixed-shape sampled PDE payloads.

No evolved solution is computed here. The baseline draws core and IC nodes
with replacement in proportion to P1 lumped mass. The optional wall sampler
uses weighted stratum means to estimate the same physical-domain mean. Every core retains all its
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


def mouth_sampling_policy(c):
    """Absent selector preserves the historical independent midpoint losses."""
    if 'mouth_sampling' not in c.operator:
        return 'midpoint'
    policy = c.operator.mouth_sampling
    if policy not in ('midpoint', 'mesh_dirichlet'):
        raise ValueError('Mouth sampling must be midpoint or mesh_dirichlet')
    return policy


def mesh_mouth_quadrature(c, mesh):
    """Every discrete reservoir node, with lumped P1 boundary-edge measure."""
    points = np.asarray(mesh['points'])
    mouth = np.asarray(mesh['mouth'])
    edges = np.asarray(mesh['boundary_edges'])[np.asarray(mesh['boundary_groups']) == 1]
    if (points.ndim != 2 or points.shape[1] != 2 or mouth.shape != (len(points),)
            or mouth.dtype != np.bool_ or edges.ndim != 2 or edges.shape[1] != 2
            or len(edges) == 0 or not np.issubdtype(edges.dtype, np.integer)
            or np.any(edges < 0) or np.any(edges >= len(points)) or not np.isfinite(points).all()):
        raise ValueError('Invalid mouth-node/edge geometry')
    ids = np.flatnonzero(mouth)
    if not np.array_equal(ids, np.unique(edges)):
        raise ValueError('Mouth edges must cover exactly the discrete Dirichlet nodes')
    capacity = c.operator.mouth_points
    if isinstance(capacity, bool) or not isinstance(capacity, int) or len(ids) > capacity:
        raise ValueError(f'Mouth-node padding overflow: {len(ids)} nodes / {capacity} slots')
    actual = points[ids]
    g = c.geometry
    if (np.any(np.abs(actual[:, 1]) > 1e-12)
            or np.any(np.abs(actual[:, 0]-g.pit_x) > g.mouth_half_width+1e-12)):
        raise ValueError('Discrete mouth nodes lie outside the declared reservoir')
    lengths = np.linalg.norm(points[edges[:, 1]]-points[edges[:, 0]], axis=1)
    if np.any(lengths <= 0) or not np.isclose(lengths.sum(), 2*g.mouth_half_width, rtol=1e-12, atol=1e-12):
        raise ValueError('Mouth edge measure differs from the declared opening length')
    nodal = np.zeros(len(points))
    np.add.at(nodal, edges[:, 0], lengths/2)
    np.add.at(nodal, edges[:, 1], lengths/2)
    if np.any(nodal[ids] <= 0):
        raise ValueError('Every operator mouth node needs positive boundary weight')
    padded_points = np.tile(actual[0], (capacity, 1))
    padded_weights = np.zeros(capacity)
    padded_points[:len(ids)] = actual
    padded_weights[:len(ids)] = nodal[ids]
    return padded_points, padded_weights


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


def particle_wall_band(points, polygons, distance):
    """Distance band of the actual polygonal hole boundaries."""
    near = np.zeros(len(points), dtype=bool)
    for polygon in polygons:
        candidates = np.flatnonzero(np.all(
            (points >= polygon.min(axis=0)-distance) & (points <= polygon.max(axis=0)+distance), axis=1))
        edge = np.roll(polygon, -1, axis=0)-polygon
        squared_length = np.sum(edge**2, axis=1)
        if np.any(squared_length <= 0):
            raise ValueError('A ceramic wall edge has zero length')
        offset = points[candidates, None, :]-polygon[None, :, :]
        parameter = np.clip(np.sum(offset*edge, axis=-1)/squared_length, 0., 1.)
        near[candidates] |= np.any(np.sum((offset-parameter[..., None]*edge)**2, axis=-1) <= distance**2, axis=1)
    return near


def make_core_sampler(c, mesh):
    """Draw complete PDE targets with area sampling or fixed wall strata."""
    volumes = mesh['volumes']
    free = np.flatnonzero(~mesh['mouth'])
    if 'core_wall' not in c.sample:
        probabilities = volumes[free]/volumes[free].sum()

        def area(rng):
            return (rng.choice(free, size=c.sample.K_core, replace=True, p=probabilities),
                    np.ones(c.sample.K_core))

        return area

    near = particle_wall_band(mesh['points'], mesh['hole_polygons'], c.sample.core_wall.distance)
    strata = (free[near[free]], free[~near[free]])
    counts = (c.sample.core_wall.count, c.sample.K_core-c.sample.core_wall.count)
    masses = np.array([volumes[ids].sum() for ids in strata])
    if np.any(masses <= 0):
        raise ValueError('Both ceramic wall and bulk strata must have positive free-node area')
    weights = np.concatenate([np.full(n, mass/masses.sum()*c.sample.K_core/n)
                              for mass, n in zip(masses, counts, strict=True)])

    def stratified(rng):
        core = np.concatenate([rng.choice(ids, n, replace=True, p=volumes[ids]/mass)
                               for ids, n, mass in zip(strata, counts, masses, strict=True)])
        return core, weights

    return stratified


def graph_payload(c, mesh, matrix, rng, *, sample_cores=None):
    """A complete core+halo subgraph, with the prescribed core loss measure."""
    points = mesh['points']
    if sample_cores is None:
        sample_cores = make_core_sampler(c, mesh)
    core, weights = sample_cores(rng)
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
        neighbors=neighbors,coefficients=coef,pde_weights=weights)


def initial_payload(c, mesh, rng):
    ids=rng.choice(len(mesh['points']),size=c.operator.ic_points,replace=True,
                   p=mesh['volumes']/mesh['volumes'].sum())
    points=mesh['points'][ids]
    result = dict(ic_points=points,initial_ic=initial_fields(c,points),ic_weights=np.ones(len(ids)))
    if mouth_sampling_policy(c) == 'mesh_dirichlet':
        result['mouth_points'], result['mouth_weights'] = mesh_mouth_quadrature(c, mesh)
    return result


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
