"""Known initial and boundary data for the open, homogeneous matrix problem."""
import numpy as np


def validate(c):
    from examples.particle_corrosion2d.dimensional import validate_dimensional
    from examples.particle_corrosion2d.geometry import _validate_config
    from examples.particle_corrosion2d.inputs import mouth_sampling_policy
    _validate_config(c)
    mouth_sampling_policy(c)
    validate_dimensional(c)
    if 'zero_output' in c.arch and not isinstance(c.arch.zero_output, bool):
        raise ValueError('arch.zero_output must be an explicit boolean')
    if 'boundary' in c:
        if (set(c.boundary) != {'dirichlet'} or c.boundary.dirichlet != 'exact_mouth'
                or c.nl.method != 'nonlocal'
                or 'phase' not in c or 'initial_lifting' not in c.phase):
            raise ValueError('Exact mouth projection requires nonlocal initial lifting and exact_mouth')
    if 'pde_reduction' in c.loss:
        if c.loss.pde_reduction not in ('mean_absolute', 'mean_square') or c.nl.method != 'nonlocal':
            raise ValueError('PDE reduction must be mean_absolute/mean_square on the nonlocal path')
    if 'time_coordinate' in c.arch:
        coordinate = c.arch.time_coordinate
        if (set(coordinate) != {'kind', 'time_scale'} or coordinate.kind != 'logarithmic'
                or not np.isfinite(coordinate.time_scale) or coordinate.time_scale <= 0):
            raise ValueError('arch.time_coordinate requires logarithmic and positive time_scale')
        if c.nl.method != 'nonlocal':
            raise ValueError('Logarithmic time coordinate is qualified only for the nonlocal path')
    if 'concentration_scale' in c.loss:
        if not np.isfinite(c.loss.concentration_scale) or c.loss.concentration_scale <= 0:
            raise ValueError('loss.concentration_scale must be positive finite')
        if c.nl.method != 'nonlocal':
            raise ValueError('Explicit concentration data metric is qualified only for the nonlocal path')
    if 'phase' in c and 'initial_lifting' in c.phase:
        lifting = c.phase.initial_lifting
        if (set(lifting) != {'kind', 'time_scale'}
                or lifting.kind not in ('bounded_exponential', 'latent_additive')):
            raise ValueError('phase.initial_lifting requires a supported kind and time_scale')
        if not np.isfinite(lifting.time_scale) or lifting.time_scale <= 0:
            raise ValueError('Initial lifting time_scale must be positive finite')
        if c.nl.method != 'nonlocal':
            raise ValueError('Initial lifting is qualified only for the explicit nonlocal path')
    if 'core_wall' in c.sample:
        spec = c.sample.core_wall
        if (set(spec) != {'distance', 'count'} or c.nl.method != 'nonlocal'
                or 'full_grid' in c.sample or 'online' not in c.operator
                or isinstance(spec.distance, bool) or not np.isfinite(spec.distance) or spec.distance <= 0
                or isinstance(spec.count, bool) or not isinstance(spec.count, int)
                or not 0 < spec.count < c.sample.K_core):
            raise ValueError('core_wall requires a positive distance and both wall and bulk draws in rolling nonlocal core+halo')
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
