"""Shared physical inputs only; independent solver imports no PINN operators."""
import json
from pathlib import Path
import numpy as np


def grid(n, length):
    y, x = np.meshgrid(np.arange(n)*length/n, np.arange(n)*length/n, indexing='ij')
    return np.stack((x.ravel(), y.ravel()), axis=-1)


def initial_field(points, c, *, xp):
    if 'kind' in c.initial and c.initial.kind == 'multiple':
        return multiple_initial_field(points, c.initial.centers, c.initial.levels,
                                      c.initial.width, c.initial.background,
                                      c.geometry.length, xp=xp)
    s = xp.sum(xp.sin(xp.pi*(points-xp.asarray(c.initial.center))/c.geometry.length)**2, axis=-1)
    return c.initial.background+(1-c.initial.background)*.5*(1-xp.tanh((s-c.initial.level)/c.initial.width))


def multiple_initial_field(points, centers, levels, width, background, length, *, xp):
    """Smooth union of the prescribed periodic particles; no future shape data."""
    relative = points[..., None, :]-xp.asarray(centers)
    s = xp.sum(xp.sin(xp.pi*relative/length)**2, axis=-1)
    particles = .5*(1-xp.tanh((s-xp.asarray(levels))/width))
    union = 1-xp.prod(1-particles, axis=-1)
    return background+(1-background)*union


def validate_config(c):
    for name,value in [('sample.Nt',c.sample.Nt),('train.iterations',c.train.iterations),
        ('log.eval_every',c.log.eval_every),('log.checkpoint_every',c.log.checkpoint_every),
        ('arch.hidden_dim',c.arch.hidden_dim),('arch.num_layers',c.arch.num_layers),
        ('arch.spatial_modes',c.arch.spatial_modes)]:
        if isinstance(value,bool) or not isinstance(value,int) or value<1:
            raise ValueError(f'{name} must be a positive integer')
    if c.reference.snapshots<2 or c.reference.error_tolerance<=0 or not np.isfinite(c.reference.error_tolerance):
        raise ValueError('Reference requires >=2 snapshots and a finite positive tolerance')
    if c.reference.refined_local_grid<=c.reference.local_grid or c.reference.refined_dt>=c.reference.dt:
        raise ValueError('Reference refinement requires a larger grid and smaller dt')
    if c.sample.Nx != c.sample.Ny or c.sample.Nx < 9 or c.sample.Nx % 2 != 1:
        raise ValueError('Pilot requires an odd square grid >=9 to avoid real-derivative Nyquist null modes')
    if c.nl.method not in ('nonlocal', 'local_spectral', 'local_ad'):
        raise ValueError('nl.method must be nonlocal, local_spectral or local_ad')
    if not c.geometry.length/c.sample.Nx < c.nl.delta < c.geometry.length/2:
        raise ValueError('Require h<delta<L/2')
    values = [c.physical.beta, c.physical.epsilon, c.physical.mobility,
              c.physical.shear_modulus, c.physical.terminal_time, c.geometry.length,
              c.initial.width, c.train.lr_init, c.loss.phase_weight, c.loss.chemical_weight]
    if not np.isfinite(values).all() or min(values) <= 0:
        raise ValueError('Physical, geometry, width, learning rate and loss weights must be finite positive')
    if not np.isfinite(c.physical.lame_lambda) or c.physical.lame_lambda+c.physical.shear_modulus <= 0:
        raise ValueError('2D isotropic elastic energy requires mu>0 and lambda+mu>0')
    e = np.asarray(c.physical.eigenstrain)
    if e.shape != (3,) or not np.isfinite(e).all():
        raise ValueError('eigenstrain must be finite (E_xx,E_yy,E_xy)')
    if not -1 <= c.initial.background < 0:
        raise ValueError('Require -1<=initial.background<0')
    if 'kind' in c.initial:
        if c.initial.kind != 'multiple':
            raise ValueError('Explicit initial.kind must be multiple; legacy single-particle schema omits kind')
        centers, levels = np.asarray(c.initial.centers), np.asarray(c.initial.levels)
        if centers.ndim != 2 or centers.shape[1] != 2 or len(centers) < 2:
            raise ValueError('Multiple-particle centers must have shape(P,2), P>=2')
        if not np.isfinite(centers).all() or np.any(centers < 0) or np.any(centers >= c.geometry.length):
            raise ValueError('Multiple-particle centers must be finite and inside the periodic cell')
        if levels.shape != (len(centers),) or not np.isfinite(levels).all() or np.any(levels <= 0) or np.any(levels >= 1):
            raise ValueError('Multiple-particle levels must have shape(P,) and lie strictly between0 and1')
        if len(np.unique(centers, axis=0)) != len(centers):
            raise ValueError('Multiple-particle centers must be distinct')
    else:
        if len(c.initial.center) != 2 or not np.isfinite(c.initial.center).all():
            raise ValueError('Initial center must have two finite coordinates')
        if not 0 < c.initial.level < 1:
            raise ValueError('Require 0<initial.level<1')
    if c.train.compute_dtype != 'float64':
        raise ValueError('This pilot explicitly uses float64')
    for n in (c.reference.local_grid, c.reference.refined_local_grid):
        if n % 2 != 1 or n < c.sample.Nx:
            raise ValueError('Reference grids must be odd and at least as large as the training grid')
    for dt in (c.reference.dt, c.reference.refined_dt):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('Reference dt must be finite and positive')
        steps = round(c.physical.terminal_time/dt)
        if not np.isclose(steps*dt, c.physical.terminal_time, rtol=0, atol=1e-12) or steps % (c.reference.snapshots-1):
            raise ValueError('Reference dt must divide duration and align with snapshots')


def identity(c):
    return dict(model='periodic_homogeneous_coherent_ch_v1', geometry=c.geometry.to_dict(),
        physical=c.physical.to_dict(), initial=c.initial.to_dict(),
        grid=(c.sample.Nx, c.sample.Ny), delta=c.nl.delta,
        mechanical_boundary='fixed_zero_macroscopic_strain_periodic_zero_mean_displacement')


def load_reference(c):
    path = Path(c.reference.path)
    with np.load(path, allow_pickle=False) as f:
        data = {k:f[k].copy() for k in f.files}
    if json.loads(str(data['identity'])) != json.loads(json.dumps(identity(c))):
        raise ValueError(f'Reference physical/grid identity mismatch: {path}')
    n, nt = c.sample.Nx, c.reference.snapshots
    np.testing.assert_array_equal(data['points'], grid(n, c.geometry.length))
    np.testing.assert_allclose(data['time'], np.linspace(0, c.physical.terminal_time, nt), atol=1e-14, rtol=0)
    for kind in ('local', 'nonlocal'):
        for field, shape in [('phi',(nt,n,n)), ('q',(nt,n,n)), ('u',(nt,n,n,2)), ('stress',(nt,n,n,3))]:
            a = data[field+'_'+kind]
            if a.shape != shape or not np.isfinite(a).all():
                raise ValueError(f'Invalid reference {field}_{kind}: {a.shape}, expected {shape}')
    return data
