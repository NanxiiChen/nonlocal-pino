"""Explicit periodic pilot data, independent of numerical operators and JAX."""
import json
import numpy as np


def is_constant_flow(config):
    """Velocity schema: explicit two-vector or explicit reversed-vortex record."""
    return isinstance(config.physical.velocity, (tuple, list))


def velocity_field(x, t, config, *, xp):
    if is_constant_flow(config):
        return xp.broadcast_to(xp.asarray(config.physical.velocity), x.shape)
    flow = config.physical.velocity
    if flow.kind != 'reversed_vortex':
        raise ValueError(f'Unknown velocity kind: {flow.kind}')
    angle = xp.pi*x/config.geometry.length
    sx, sy = xp.sin(angle[..., 0]), xp.sin(angle[..., 1])
    cx, cy = xp.cos(angle[..., 0]), xp.cos(angle[..., 1])
    scale = flow.amplitude*xp.cos(xp.pi*t/flow.period)
    return xp.stack((-scale*sx*sx*sy*cy, scale*sx*cx*sy*sy), axis=-1)


def initial_field(x, config, *, xp):
    """Smooth periodic droplet-like IC; no prescribed subsequent motion."""
    phase = xp.pi * (x - xp.asarray(config.initial.center)) / config.geometry.length
    level = xp.sum(xp.sin(phase)**2, axis=-1)
    return xp.tanh((level - config.initial.level) / config.initial.width)


def identity(config):
    specification = dict(
        case='twophase2d-periodic-prescribed-flow', schema=1,
        geometry=config.geometry.to_dict(), physical=config.physical.to_dict(),
        initial=config.initial.to_dict(), grid=[config.sample.Nx, config.sample.Ny],
        delta=config.nl.delta, kernel=config.nl.kernel,
        normalization=config.nl.normalization)
    # Preserve historical Fourier-reference identities. The new native model
    # changes first derivatives and must not silently load the old reference.
    if 'advection' in config and config.advection == 'nonlocal_flux':
        specification['advection'] = 'nonlocal_flux'
    return json.loads(json.dumps(specification))


def validate_config(c):
    if c.sample.Nx != c.sample.Ny or c.sample.Nx < 8 or c.sample.Nx % 2:
        raise ValueError('Pilot requires equal even Nx=Ny>=8, endpoint excluded')
    if c.sample.Nt < 1 or c.train.iterations < 1:
        raise ValueError('Nt and iterations must be positive')
    if 'time_rule' in c.sample and c.sample.time_rule != 'open_stratified':
        raise ValueError('Explicit sample.time_rule must be open_stratified')
    if 'time_rule' in c.sample and 4*np.finfo(np.float32).eps*c.sample.Nt >= .5:
        raise ValueError('Too many time bins to keep FP32 sample intervals distinct')
    if 'include_initial_time' in c.sample:
        if not isinstance(c.sample.include_initial_time, bool):
            raise ValueError('sample.include_initial_time must be an explicit boolean')
        if c.sample.include_initial_time:
            if 'preconditioner' in c and c.preconditioner.kind == 'ch_causal':
                raise ValueError('Causal residual metric requires positive times; t=0 is not an invertible interval')
    if 'phase_mass' in c and c.phase_mass != 'exact_grid_mean':
        raise ValueError('phase_mass must be exact_grid_mean')
    if 'advection' in c and c.advection not in ('spectral_flux', 'nonlocal_flux'):
        raise ValueError('Explicit advection must be spectral_flux or nonlocal_flux')
    if 'advection' in c and c.advection == 'nonlocal_flux' and c.nl.method != 'nonlocal':
        raise ValueError('nonlocal_flux reference requires nonlocal diffusion; no mixed local-diffusion reference is provided')
    for v in (c.geometry.length, c.physical.epsilon, c.physical.mobility,
              c.physical.terminal_time, c.initial.width, c.train.lr_init):
        if not np.isfinite(v) or v <= 0:
            raise ValueError('Lengths, mobility, horizon, width and learning rate must be positive finite')
    if is_constant_flow(c):
        if len(c.physical.velocity) != 2 or not np.isfinite(c.physical.velocity).all():
            raise ValueError('velocity must have two finite components')
    else:
        flow = c.physical.velocity
        if flow.kind != 'reversed_vortex':
            raise ValueError(f'Unknown velocity kind: {flow.kind}')
        if not np.isfinite(flow.amplitude) or flow.amplitude <= 0:
            raise ValueError('Vortex amplitude must be positive finite')
        if not np.isfinite(flow.period) or flow.period != c.physical.terminal_time:
            raise ValueError('Vortex period must equal terminal_time for one reversal')
    if len(c.initial.center) != 2 or not all(0 <= x < c.geometry.length for x in c.initial.center):
        raise ValueError('Initial center must lie in the periodic cell')
    if not 0 < c.initial.level < 1:
        raise ValueError('Initial level must be between0 and1')
    if not c.geometry.length/c.sample.Nx < c.nl.delta < c.geometry.length/2:
        raise ValueError('Require h<delta<L/2 for unambiguous periodic interactions')
    if c.nl.kernel != 'cubic' or c.nl.normalization != 'discrete_second_moment':
        raise ValueError('Pilot implements cubic/discrete_second_moment only')
    if c.nl.method not in ('nonlocal', 'local_ad'):
        raise ValueError('nl.method must be nonlocal or local_ad')
    if c.train.compute_dtype not in ('float32', 'float64'):
        raise ValueError('compute_dtype must be float32 or float64')
    if 'refinement' in c.train:
        r = c.train.refinement
        if r.optimizer != 'lbfgs' or c.train.compute_dtype != 'float64':
            raise ValueError('L-BFGS refinement requires optimizer=lbfgs and compute_dtype=float64')
        for name in ('iterations', 'max_evaluations', 'max_line_search', 'history_size', 'time_points'):
            if not isinstance(r[name], int) or r[name] < 1:
                raise ValueError(f'train.refinement.{name} must be a positive integer')
        for name in ('gradient_tolerance', 'function_tolerance'):
            if not np.isfinite(r[name]) or r[name] <= 0:
                raise ValueError(f'train.refinement.{name} must be positive finite')
    # The original schema has no residual record. The extended schema requires
    # both fields explicitly; historical serialized configs remain unchanged.
    if 'residual' in c:
        if c.residual.mode not in ('original', 'transport', 'both', 'map_transport', 'map_both'):
            raise ValueError('Unknown residual.mode; use original/transport/both/map_transport/map_both')
        if not np.isfinite(c.residual.eta) or c.residual.eta <= 0:
            raise ValueError('residual.eta must be positive finite')
        if c.residual.mode.startswith('map_') and 'phase' not in c:
            raise ValueError('Inverse-map residual requires an explicit phase record')
    if 'phase' in c:
        if c.phase.map == 'leaky_tanh':
            if not np.isfinite(c.phase.leak) or not 0.1 <= c.phase.leak <= 1.:
                raise ValueError('Validated leaky-tanh inverse requires0.1<=phase.leak<=1')
            if not isinstance(c.phase.inverse_steps, int) or c.phase.inverse_steps < 8:
                raise ValueError('Leaky-tanh IC inverse requires at least8 fixed Newton steps')
        elif c.phase.map not in ('additive', 'inverse_cubic'):
            raise ValueError('Explicit phase.map must be additive, leaky_tanh or inverse_cubic')
    if 'residual' in c and c.residual.mode.startswith('map_'):
        if c.phase.map != 'leaky_tanh':
            raise ValueError('map_ residual modes require the leaky_tanh phase map')
        if 'phase_mass' in c:
            raise ValueError('Pointwise inverse-map residuals are incompatible with the spatially coupled mass projection')
    if 'preconditioner' in c:
        if c.preconditioner.kind not in ('ch_schur', 'ch_causal'):
            raise ValueError('preconditioner.kind must be ch_schur or ch_causal')
        if c.preconditioner.kind == 'ch_causal':
            if 'time_rule' not in c.sample or c.sample.time_rule != 'open_stratified':
                raise ValueError('Causal preconditioning requires explicit open_stratified times')
        for name in ('time_scale', 'curvature'):
            if not np.isfinite(c.preconditioner[name]) or c.preconditioner[name] <= 0:
                raise ValueError(f'preconditioner.{name} must be positive finite')
        if 'residual' in c and c.residual.mode != 'original':
            raise ValueError('CH Schur preconditioner requires original physical residuals')
    if c.arch.activation not in ('gelu', 'tanh'):
        raise ValueError('A smooth gelu/tanh activation is required')
    if c.arch.spatial_modes < 1 or c.arch.hidden_dim < 1 or c.arch.num_layers < 1:
        raise ValueError('Network sizes must be positive')
    loss_keys = ('phase_weight', 'chemical_weight') if 'phase_mass' in c else ('phase_weight', 'chemical_weight', 'mass_weight')
    for key in loss_keys:
        if not np.isfinite(c.loss[key]) or c.loss[key] <= 0:
            raise ValueError(f'loss.{key} must be positive finite')
    for key in ('eval_every', 'checkpoint_every'):
        if c.log[key] < 1:raise ValueError(f'log.{key} must be positive')
    if c.reference.local_grid % c.sample.Nx or c.reference.refined_local_grid % c.reference.local_grid:
        raise ValueError('Reference grids must be integer refinements of the evaluation grid')
    if c.reference.refined_local_grid <= c.reference.local_grid:
        raise ValueError('Reference spatial refinement must be larger')
    if not 0 < c.reference.refined_dt < c.reference.dt or c.reference.snapshots < 2:
        raise ValueError('Need positive decreasing reference dt and at least2 snapshots')
    if not np.isfinite(c.reference.error_tolerance) or c.reference.error_tolerance <= 0:
        raise ValueError('reference.error_tolerance must be positive')


def load_reference(config):
    with np.load(config.reference.path, allow_pickle=False) as data:
        reference = {key: data[key] for key in data.files}
    if json.loads(str(reference['identity'])) != identity(config):
        raise ValueError('Reference physical/grid/operator/IC specification differs from config')
    n = config.sample.Nx
    t = reference['time']
    if t.shape != (config.reference.snapshots,) or not np.all(np.diff(t) > 0):
        raise ValueError('Reference time ordering/shape mismatch')
    if t[0] != 0 or t[-1] != config.physical.terminal_time:
        raise ValueError('Reference time range mismatch')
    for key in ('phi_local', 'phi_nonlocal', 'q_local', 'q_nonlocal'):
        if reference[key].shape != (len(t), n, n) or not np.isfinite(reference[key]).all():
            raise ValueError(f'Reference {key} shape/range mismatch')
    if reference['points'].shape != (n*n, 2):
        raise ValueError('Reference points shape mismatch')
    axis = np.arange(n)*config.geometry.length/n
    X, Y = np.meshgrid(axis, axis, indexing='xy')
    expected_points = np.stack((X.ravel(), Y.ravel()), axis=-1)
    if not np.array_equal(reference['points'], expected_points):
        raise ValueError('Reference points must use the endpoint-excluded periodic xy grid ordering')
    return reference
