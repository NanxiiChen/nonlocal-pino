"""Explicit channel geometry families; pure NumPy/JAX graph-wall evaluation.

Random walls use positive quartic B-splines (C3 at simple knots), multiplied
by a compact C3 septic taper. Host-side polynomial extrema qualify each draw;
no geometry-specific network features or evolved fields enter conditioning.
"""
from functools import lru_cache
import numpy as np


def random_walls(c):
    return 'kind' in c.geometry and c.geometry.kind=='random_spline'


def parameter_count(c):
    return 2*c.geometry.controls if random_walls(c) else 2


def nominal_parameters(c):
    return np.asarray(c.geometry.coefficients if random_walls(c)
                      else (c.geometry.upper,c.geometry.lower),dtype=np.float64)


def coefficient_bounds(c):
    g=c.geometry
    return (g.coefficient_min,g.coefficient_max) if random_walls(c) else (g.amplitude_min,g.amplitude_max)


def knots(controls):
    return (0.,)*5+tuple(np.arange(1,controls-4)/(controls-4))+(1.,)*5


def basis(z,controls,degree,*,xp):
    """Cox--de Boor recurrence; static zero denominators are omitted explicitly."""
    t=knots(controls)
    values=xp.stack([((z>=a)&(z<b)).astype(z.dtype) for a,b in zip(t[:-1],t[1:])],axis=-1)
    for p in range(1,degree+1):
        rows=[]
        for i in range(len(t)-p-1):
            left=(z-t[i])/(t[i+p]-t[i])*values[...,i] if t[i+p]>t[i] else xp.zeros_like(z)
            right=(t[i+p+1]-z)/(t[i+p+1]-t[i+1])*values[...,i+1] if t[i+p+1]>t[i+1] else xp.zeros_like(z)
            rows.append(left+right)
        values=xp.stack(rows,axis=-1)
    return values


def taper(z,*,xp):
    z=xp.clip(z,0.,1.)
    return z**4*(35.+z*(-84.+z*(70.-20.*z)))


def taper_derivative(z,*,xp):
    z=xp.clip(z,0.,1.)
    return 140.*z**3*(1.-z)**3


def profiles(x,parameters,c,*,xp,derivative=False):
    g=c.geometry;span=g.end-g.start
    x=xp.asarray(x);z=(x-g.start)/span
    a=xp.reshape(xp.asarray(parameters),(2,g.controls))
    B=basis(z,g.controls,4,xp=xp)
    l=(x-g.start)/g.taper_width;r=(g.end-x)/g.taper_width
    gate=taper(l,xp=xp)*taper(r,xp=xp)
    value=xp.einsum('...k,ak->a...',B,a)
    if derivative:
        lower=basis(z,g.controls,3,xp=xp);t=knots(g.controls)
        dB=xp.stack([(4.*lower[...,i]/(t[i+4]-t[i]) if t[i+4]>t[i] else xp.zeros_like(z))
                    -(4.*lower[...,i+1]/(t[i+5]-t[i+1]) if t[i+5]>t[i+1] else xp.zeros_like(z))
                    for i in range(g.controls)],axis=-1)/span
        dgate=(taper_derivative(l,xp=xp)*taper(r,xp=xp)
               -taper(l,xp=xp)*taper_derivative(r,xp=xp))/g.taper_width
        value=xp.einsum('...k,ak->a...',dB,a)*gate+value*dgate
    else:value=value*gate
    return (value[1],-value[0]) if derivative else (value[1],g.height-value[0])


@lru_cache(maxsize=16)
def _piece_basis(controls,start,end,width):
    """Independent SciPy piece-polynomial basis, with local normalized coordinates."""
    from scipy.interpolate import BSpline,PPoly
    from numpy.polynomial import Polynomial as P
    t=np.asarray(knots(controls));breaks=np.unique(np.r_[t,width/(end-start),1-width/(end-start)])
    result=[]
    for lo,hi in zip(breaks[:-1],breaks[1:]):
        if hi<=lo:continue
        mid=(lo+hi)/2;z=P([lo,hi-lo]);pieces=[]
        for i in range(controls):
            pp=PPoly.from_spline(BSpline(t,np.eye(controls)[i],4))
            j=np.searchsorted(pp.x,mid,side='right')-1
            q=P(pp.c[:,j][::-1])(z-pp.x[j])
            def gate_poly(v):return 35*v**4-84*v**5+70*v**6-20*v**7
            gl=gate_poly(z*(end-start)/width) if mid*(end-start)<width else P([1.])
            gr=gate_poly((1-z)*(end-start)/width) if (1-mid)*(end-start)<width else P([1.])
            pieces.append((q*gl*gr).coef)
        degree=max(map(len,pieces));matrix=np.zeros((controls,degree))
        for i,p in enumerate(pieces):matrix[i,:len(p)]=p
        result.append(((hi-lo)*(end-start),matrix))
    return tuple(result)


def geometry_statistics(parameters,c):
    """Bound gap, slope and curvature at all real polynomial critical points."""
    from numpy.polynomial import Polynomial as P
    g=c.geometry;a=np.asarray(parameters).reshape(2,g.controls)
    minimum=g.height;maximum=0.;curvature=0.
    def extrema(p):
        roots=p.deriv().roots()
        real=roots.real[(np.abs(roots.imag)<1e-8)&(roots.real>0)&(roots.real<1)]
        return p(np.r_[0.,real,1.])
    for dx,B in _piece_basis(g.controls,g.start,g.end,g.taper_width):
        upper,lower=(P(row@B) for row in a)
        minimum=min(minimum,float(extrema(P([g.height])-upper-lower).min()))
        for p in (upper,lower):
            maximum=max(maximum,float(np.max(np.abs(extrema(p.deriv()/dx)))))
            curvature=max(curvature,float(np.max(np.abs(extrema(p.deriv(2)/dx**2)))))
    return dict(minimum_gap=minimum,maximum_slope=maximum,maximum_curvature=curvature)


def valid_statistics(s,c):
    g=c.geometry
    return (g.gap_min<=s['minimum_gap']<=g.gap_max and s['maximum_slope']<=g.slope_max
            and s['maximum_curvature']<=g.curvature_max)


def validate_parameters(parameters,c):
    a=np.asarray(parameters,dtype=np.float64)
    lo,hi=coefficient_bounds(c)
    # Legacy reference amplitudes may lie below the training lower bound.
    lower=lo if random_walls(c) else 0.
    if a.shape!=(parameter_count(c),) or not np.isfinite(a).all() or np.any(a<lower) or np.any(a>hi):
        raise ValueError('Invalid geometry parameter shape, finite range or coefficient bounds')
    if random_walls(c):
        s=geometry_statistics(a,c)
        if not valid_statistics(s,c):raise ValueError(f'Random wall violates declared aperture/slope/curvature bounds: {s}')
    return a


def sample_parameters(c,rng):
    if not random_walls(c):return rng.uniform(c.geometry.amplitude_min,c.geometry.amplitude_max,2)
    g=c.geometry
    # A cheap sample screen only rejects; exact polynomial extrema accept.
    x=np.linspace(g.start,g.end,257)
    for _ in range(g.max_attempts):
        a=rng.uniform(g.coefficient_min,g.coefficient_max,2*g.controls)
        b,t=profiles(x,a,c,xp=np);db,dt=profiles(x,a,c,xp=np,derivative=True)
        gap=float(np.min(t-b))
        if not (g.gap_min<=gap<=g.gap_max and max(np.max(np.abs(db)),np.max(np.abs(dt)))<=g.slope_max):continue
        if valid_statistics(geometry_statistics(a,c),c):return a
    raise RuntimeError(f'Random-wall rejection budget exhausted after {g.max_attempts} attempts')


def initial_parameters(c):
    if c.operator.train_conditions==1:return nominal_parameters(c)[None,:]
    rng=np.random.default_rng(c.operator.dataset_seed)
    # Preserve original vectorized RNG arithmetic for historical presets.
    if not random_walls(c):
        return rng.uniform(c.geometry.amplitude_min,c.geometry.amplitude_max,(c.operator.train_conditions,2))
    return np.stack([sample_parameters(c,rng) for _ in range(c.operator.train_conditions)])


def validate_geometry(c):
    g=c.geometry
    if 'kind' in g and not random_walls(c):raise ValueError('Unknown explicit channel geometry kind')
    if random_walls(c):
        fields={'kind','length','height','controls','start','end','taper_width','coefficient_min','coefficient_max',
                'gap_min','gap_max','slope_max','curvature_max','max_attempts','coefficients'}
        if set(g)!=fields:raise ValueError(f'Random-wall geometry requires exactly {sorted(fields)}')
        if isinstance(g.controls,bool) or not isinstance(g.controls,int) or g.controls<5:raise ValueError('At least five quartic spline controls required')
        if isinstance(g.max_attempts,bool) or not isinstance(g.max_attempts,int) or g.max_attempts<1:raise ValueError('Positive rejection budget required')
        values=[g[k] for k in fields-{'kind','controls','max_attempts','coefficients'}]
        if not np.isfinite(values).all():raise ValueError('Geometry settings must be finite')
        if not (0<g.start<g.end<g.length and 0<2*g.taper_width<g.end-g.start
                and 0<=g.coefficient_min<g.coefficient_max<g.height/2
                and 0<g.gap_min<g.gap_max<g.height and min(g.slope_max,g.curvature_max)>0):
            raise ValueError('Invalid random-wall support, taper, coefficients or geometric bounds')
        validate_parameters(nominal_parameters(c),c)
        return g.start
    if not np.isfinite(g.half_length) or g.half_length<=0 or not np.isfinite(g.center):raise ValueError('Legacy bump length and center must be finite, length positive')
    if not (0<=g.amplitude_min<g.amplitude_max<g.height/2):raise ValueError('Independent inward amplitudes require 0<=min<max<H/2')
    if not (g.half_length<g.center<g.length-g.half_length):raise ValueError('Compact wall indentation must preserve straight inlet/outlet')
    validate_parameters(nominal_parameters(c),c)
    return g.center-g.half_length


def with_nominal(c,parameters):
    """Copy configuration with one explicit reference geometry; leave prior unchanged."""
    from ml_collections import ConfigDict
    cc=ConfigDict(c.to_dict());a=validate_parameters(parameters,c)
    if random_walls(c):cc.geometry.coefficients=tuple(a.tolist())
    else:cc.geometry.upper=float(a[0]);cc.geometry.lower=float(a[1])
    return cc


def envelope_parameters(c):
    """Containing inward-wall envelope for shared physical queries, not a prior draw."""
    return np.full(parameter_count(c),coefficient_bounds(c)[1])


def validate_mesh_queries(query,parameters,c,nx):
    """Queries must be inside the actual piecewise-linear mapped FEM domain."""
    from examples.channel_nsch2d.problem import walls
    q=np.asarray(query);x=np.linspace(0,c.geometry.length,nx+1)
    bottom,top=walls(x,parameters,c,xp=np)
    lower=np.interp(q[:,0],x,bottom);upper=np.interp(q[:,0],x,top)
    valid=((q[:,0]>=0)&(q[:,0]<=c.geometry.length)&(q[:,1]>=lower-1e-12)&(q[:,1]<=upper+1e-12))
    if not np.all(valid):
        raise ValueError(f'{np.count_nonzero(~valid)} FEM queries outside the actual {nx}-interval chord-wall domain; '
                         f'first={q[np.flatnonzero(~valid)[0]].tolist()}. Refine mesh; extrapolation is forbidden.')
