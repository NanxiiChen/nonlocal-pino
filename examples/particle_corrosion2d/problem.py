"""Known initial and boundary data for the open, homogeneous matrix problem."""
import numpy as np


def validate(c):
    from examples.particle_corrosion2d.geometry import _validate_config
    _validate_config(c)
    p = c.physical
    values = [p.kappa,p.n_ch,p.n_ac1,p.n_ac2,p.n_ac3,p.c_se,p.c_le,
              c.sample.t_max,c.mesh.spacing,c.nl.delta]
    if not np.isfinite(values).all() or min(values)<=0 or p.c_se<=p.c_le:
        raise ValueError('Physical scales must be positive finite and c_se>c_le')
    if not np.isclose(p.n_ac2/(2*p.n_ac3),c.geometry.interface_k**2):
        raise ValueError('Interface k must equal sqrt(n_ac2/(2*n_ac3))')
    if c.mesh.boundary_points != c.nl.visibility_points:
        raise ValueError('Mesh and nonlocal visibility must use the same contour polygon')
    if c.nl.method not in ('nonlocal','local_ad'):
        raise ValueError('Explicit method must be nonlocal or local_ad')


def initial_fields(c, points):
    points = np.asarray(points)
    radius = np.hypot(points[...,0]-c.geometry.pit_x,points[...,1])
    phi = .5*(1+np.tanh(c.geometry.interface_k*(radius-c.geometry.pit_radius)))
    mouth = ((np.abs(points[...,1])<1e-12)
             & (np.abs(points[...,0]-c.geometry.pit_x)<=c.geometry.mouth_half_width+1e-12))
    phi = np.where(mouth,0.,phi)
    concentration = c.physical.c_se*phi**2*(3-2*phi)
    return np.stack((phi,concentration),axis=-1)
