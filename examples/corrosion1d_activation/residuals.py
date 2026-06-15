"""PDE residuals for the 1D KKS phase-field corrosion model (non-local form).

Problem-specific — each example provides its own version of this file.
"""

import jax.numpy as jnp
import jax
import numpy as np


def _h(phi):
    return -2.0 * phi**3 + 3.0 * phi**2


def _hp(phi):
    return -6.0 * phi**2 + 6.0 * phi


def _gp(phi):
    return 4.0 * phi**3 - 6.0 * phi**2 + 2.0 * phi


def pde_residual(net, phi, c, x_free, t_batch, D, nondim_params):
    """Non-dimensional, non-local KKS PDE residuals.

    Args:
        net: (x, t) → (φ, c), single-sample.
        phi, c: (N_t, N_x) network output at ALL spatial points.
        x_free: (N_free, dim) free-point coordinates.
        t_batch: (N_t,) time points.
        D: L_delta operator (N_x, N_x) or condensed (N_free, N_free).
        nondim_params: (N_CH, N_AC1, N_AC2, N_AC3, dc0, c_Le).

    Returns:
        r_phi, r_c: (N_t, N_free) residuals.
    """
    N_CH, N_AC1, N_AC2, N_AC3, dc0, c_Le = nondim_params
    N_free = len(x_free)
    N_t = len(t_batch)

    nl_phi = (D @ phi.T).T
    mu = c - _h(phi) * dc0 - c_Le
    nl_mu = (D @ mu.T).T

    x_flat = jnp.repeat(x_free, N_t, axis=0)
    t_flat = jnp.tile(t_batch, N_free)

    dphi_dt = jax.vmap(jax.grad(lambda x, t: net(x, t)[0], argnums=1))(
        x_flat, t_flat)
    dc_dt = jax.vmap(jax.grad(lambda x, t: net(x, t)[1], argnums=1))(
        x_flat, t_flat)
    dphi_dt = dphi_dt.reshape(N_t, N_free)
    dc_dt = dc_dt.reshape(N_t, N_free)

    phi_free = phi[:, :N_free]
    c_free = c[:, :N_free]
    nl_phi_free = nl_phi[:, :N_free]
    nl_mu_free = nl_mu[:, :N_free]

    r_phi = (dphi_dt
             + N_AC3 * nl_phi_free
             + N_AC2 * _gp(phi_free)
             - N_AC1 * (c_free - _h(phi_free) * dc0 - c_Le) * dc0 * _hp(phi_free))

    r_c = dc_dt + N_CH * nl_mu_free

    return r_phi, r_c


def ic_values(x, phys, L_c):
    """Analytical tanh IC for the 1D corrosion problem.

    φ₀(x*) = 0.5·(1 - tanh(k·(x* - 0.5)))
    c₀(x*) = h(φ₀)·c_Se + (1 - h(φ₀))·c_Le

    Args:
        x: (N, dim) spatial coordinates.
        phys: config.physical (omega_phi, alpha_phi, c_se, c_le).
        L_c: characteristic length.

    Returns:
        phi0, c0: (N,) arrays.
    """
    k = np.sqrt(phys.omega_phi / (2.0 * phys.alpha_phi)) * L_c
    dist = np.array(x)[:, 0] - 0.5
    phi0 = 0.5 * (1.0 - np.tanh(k * dist))
    c0 = _h(phi0) * phys.c_se + (1.0 - _h(phi0)) * phys.c_le
    return jnp.array(phi0), jnp.array(c0)
