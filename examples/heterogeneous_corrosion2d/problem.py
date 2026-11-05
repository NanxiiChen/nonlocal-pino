"""Known geometry/material inputs; no neural outputs or evolved reference data.

The variable mobility multiplies the complete Allen--Cahn chemical-potential
bracket. It never enters the concentration diffusion operator. All coordinates
are dimensionless, and the solid occupies y>0 with an opening on y=0.
"""
import numpy as np
from scipy import sparse


def validate(c):
    p, g, s, m = c.physical, c.geometry, c.sample, c.material
    positive = (g.length, g.height, g.pit_radius, g.mouth_half_width,
        g.interface_k, p.n_ch, p.n_ac1, p.n_ac2, p.n_ac3,
        m.minimum, m.maximum, s.t_max)
    if not np.isfinite(positive).all() or min(positive) <= 0:
        raise ValueError('Physical, geometry and material scales must be positive finite values')
    if not (0 < p.c_le < p.c_se == 1.0):
        raise ValueError('Pilot requires 0<c_le<c_se=1')
    if not (g.mouth_half_width < g.pit_radius < min(g.pit_x, g.length-g.pit_x, g.height)):
        raise ValueError('Mouth must lie inside the initial pit; pit cannot touch other walls')
    if m.minimum >= m.maximum:
        raise ValueError('Material mobility minimum must be below maximum')
    if min(s.Nx, s.Ny) < 5 or any(int(v) != v for v in (s.Nx, s.Ny)):
        raise ValueError('Grid dimensions must be integers >=5')
    modes = np.asarray(m.modes)
    if (modes.ndim != 2 or modes.shape[1] != 2 or len(modes) == 0
            or np.any(modes < 0) or np.any(modes != modes.astype(int))
            or np.any(np.sum(modes, axis=1) == 0)):
        raise ValueError('Material modes must be nonzero pairs of nonnegative integers')
    scales = np.asarray(m.coefficient_scale, dtype=float)
    if scales.ndim == 0:
        valid_scales = bool(np.isfinite(scales) and scales > 0)
    elif scales.shape == (1+2*len(modes),):
        valid_scales = bool(np.isfinite(scales).all() and np.all(scales >= 0)
                            and np.any(scales[1:] > 0))
    else:
        valid_scales = False
    if not valid_scales:
        raise ValueError('Material coefficient_scale must be positive scalar or nonnegative '
                         '(1+2*modes,) standard deviations with nonzero spatial variance')
    if not np.isclose(p.n_ac2/(2*p.n_ac3), g.interface_k**2, rtol=1e-12):
        raise ValueError('Initial interface width must match the AC free-energy coefficients')


def nondim(c):
    p = c.physical
    return p.n_ch, p.n_ac1, p.n_ac2, p.n_ac3, p.c_se-p.c_le, p.c_le


def grid(c):
    validate(c)
    xx, yy = np.meshgrid(np.linspace(0, c.geometry.length, c.sample.Nx),
                         np.linspace(0, c.geometry.height, c.sample.Ny), indexing='xy')
    return xx, yy, np.stack([xx.ravel(), yy.ravel()], axis=-1)


def quadrature(c):
    wx, wy = np.ones(c.sample.Nx), np.ones(c.sample.Ny)
    wx[[0, -1]], wy[[0, -1]] = 0.5, 0.5
    return (np.outer(wy, wx)*c.geometry.length*c.geometry.height
            / ((c.sample.Nx-1)*(c.sample.Ny-1))).ravel()


def boundary_mask(c, points):
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError('Points must be a finite (N,2) array')
    return (np.isclose(points[:, 1], 0.0, atol=1e-12, rtol=0)
            & (np.abs(points[:, 0]-c.geometry.pit_x) <= c.geometry.mouth_half_width+1e-12))


def initial_fields(c, points):
    points = np.asarray(points)
    radius = np.hypot(points[:, 0]-c.geometry.pit_x, points[:, 1])
    phi = 0.5*(1+np.tanh(c.geometry.interface_k*(radius-c.geometry.pit_radius)))
    # Same open-pit initial chemistry as corrosion2d: initially c=h(phi)c_se.
    fields = np.stack([phi, (3*phi**2-2*phi**3)*c.physical.c_se], axis=-1)
    fields[boundary_mask(c, points)] = 0.0
    return fields


def parameter_count(c):
    return 1+2*len(c.material.modes)


def sample_parameters(c, rng, count):
    """Independent Gaussian coefficients with explicit scalar or per-mode scales.

    A zero first entry fixes only the latent constant mode, not the physical
    mean after the bounded nonlinear material mapping.
    """
    if not isinstance(count, int) or count <= 0:
        raise ValueError('Material sample count must be a positive integer')
    return rng.normal(0.0, c.material.coefficient_scale, size=(count, parameter_count(c)))


def material_field(c, parameters, points, xp):
    """Smooth logistic Fourier map; accepts NumPy or jax.numpy explicitly.

    Parameters (...,1+2M), points (N,2), output (...,N). Coefficients are
    continuously sampled without a finite material library. No trajectory or
    decoder-local condition enters this known input function.
    """
    coefficients, points = xp.asarray(parameters), xp.asarray(points)
    if coefficients.shape[-1] != parameter_count(c) or points.ndim != 2 or points.shape[-1] != 2:
        raise ValueError('Material parameter or coordinate shape mismatch')
    modes = xp.asarray(c.material.modes, dtype=points.dtype)
    phase = 2*xp.pi*((points/xp.asarray([c.geometry.length, c.geometry.height])) @ modes.T)
    basis = xp.concatenate([xp.cos(phase), xp.sin(phase)], axis=-1)
    latent = coefficients[..., :1] + xp.einsum('...m,nm->...n', coefficients[..., 1:], basis)/xp.sqrt(float(len(c.material.modes)))
    return c.material.minimum+(c.material.maximum-c.material.minimum)*0.5*(1+xp.tanh(latent/2))


def image_points(c):
    channels, height, width = c.operator.image_shape
    if channels != 1 or min(height, width) < 2:
        raise ValueError('Material image requires (1,H,W) with H,W>=2')
    x = (np.arange(width)+0.5)*c.geometry.length/width
    y = (np.arange(height)+0.5)*c.geometry.height/height
    xx, yy = np.meshgrid(x, y, indexing='xy')
    return np.stack([xx.ravel(), yy.ravel()], axis=-1)


def material_image(c, parameters, xp):
    values = material_field(c, parameters, xp.asarray(image_points(c)), xp)
    # Dimensionless condition channel: fixed linear map of physical kappa.
    values = 2*(values-c.material.minimum)/(c.material.maximum-c.material.minimum)-1
    return values.reshape(parameters.shape[:-1]+tuple(c.operator.image_shape))


def local_matrix(c):
    """Independent node-centred FV negative Laplacian, reflecting outer faces.

    Half-width edge cells give doubled one-sided boundary rows. Consequently
    diag(quadrature)@D is symmetric and annihilates constants; D itself need
    not be Euclidean symmetric. The mouth becomes Dirichlet in the solver.
    """
    validate(c)
    def line(n, length):
        h = length/(n-1)
        result = sparse.diags([-np.ones(n-1), 2*np.ones(n), -np.ones(n-1)], [-1, 0, 1], format='lil')
        result[0, 1], result[-1, -2] = -2, -2
        return result.tocsr()/h**2
    return (sparse.kron(sparse.eye(c.sample.Ny), line(c.sample.Nx, c.geometry.length))
            + sparse.kron(line(c.sample.Ny, c.geometry.height), sparse.eye(c.sample.Nx))).tocsr()
