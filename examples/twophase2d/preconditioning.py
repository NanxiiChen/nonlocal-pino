"""Frozen bulk mixed-CH block inverse, applied only to the residual metric."""
import jax
import jax.numpy as jnp
import numpy as np


def _symbol(D, config, *, dtype):
    n = config.sample.Nx
    if D is None:
        k = 2*np.pi*np.fft.fftfreq(n, d=config.geometry.length/n)
        lam = k[:, None]**2+k[None, :n//2+1]**2
    else:
        # Compute this metric's symbol from the actual fixed training weights.
        # Integer grid ordering is validated by load_reference at startup.
        nodes = np.asarray(D.indices)[:, 0]
        dy, dx = nodes//n, nodes % n
        ky = np.fft.fftfreq(n)*n
        kx = np.fft.rfftfreq(n)*n
        angles = 2*np.pi*(dy[:, None, None]*ky[None, :, None]
                          + dx[:, None, None]*kx[None, None, :])/n
        lam = np.sum(np.asarray(D.weights)[:, None, None]*(1-np.cos(angles)), axis=0)
    return jnp.asarray(lam, dtype=dtype)


def make_schur_transform(D, config, *, dtype):
    """Apply diag(1/tau,1) A_tau^-1, ignoring transport in A_tau.

    A_tau = [[1/tau, M L], [-(c + eps² L), I]]. Its Schur complement
    is positive for tau,c>0. The actual nonlinear PDE residual is unchanged.
    The constant Fourier mode has no division by a Laplacian eigenvalue.
    """
    n = config.sample.Nx
    tau = config.preconditioner.time_scale
    curvature = config.preconditioner.curvature
    lam = _symbol(D, config, dtype=dtype)
    ml = config.physical.mobility*lam
    b = curvature+config.physical.epsilon**2*lam
    denominator = 1+tau*ml*b

    def transform(residual):
        if residual.shape != (n*n, 2):
            raise ValueError('Schur residual requires the full periodic (Nx*Ny,2) grid')
        r = jnp.fft.rfft2(residual.reshape(n, n, 2), axes=(0, 1))
        phase = (r[..., 0]-ml*r[..., 1])/denominator
        chemical = r[..., 1]+tau*b*phase
        out = jnp.stack((phase, chemical), axis=-1)
        return jnp.fft.irfft2(out, s=(n, n), axes=(0, 1)).reshape(n*n, 2)
    return transform


def make_causal_transform(D, config, *, dtype):
    """Inverse frozen mixed-CH backward-Euler error operator in sorted time.

    This transforms continuous AD residuals, not the physical PDE or its time
    discretization. Time intervals define only an invertible residual metric.
    Advection and variable phase curvature are omitted from the preconditioner.
    """
    n = config.sample.Nx
    lam = _symbol(D, config, dtype=dtype)
    ml = config.physical.mobility*lam
    b = config.preconditioner.curvature+config.physical.epsilon**2*lam
    time_scale = config.preconditioner.time_scale

    def transform(residual, times):
        if residual.shape != (times.shape[0], n*n, 2):
            raise ValueError('Causal preconditioning requires all full-grid time residuals')
        order = jnp.argsort(times)
        times = times[order]
        dt = jnp.diff(jnp.concatenate((jnp.zeros(1, dtype=times.dtype), times)))
        hat = jnp.fft.rfft2(residual[order].reshape(-1, n, n, 2), axes=(1, 2))

        def advance(previous, data):
            r, interval = data
            phase = (previous+interval*(r[..., 0]-ml*r[..., 1]))/(1+interval*ml*b)
            chemical = r[..., 1]+b*phase
            return phase, jnp.stack((phase/time_scale, chemical), axis=-1)

        _, out = jax.lax.scan(advance, jnp.zeros_like(hat[0, ..., 0]), (hat, dt))
        return jnp.fft.irfft2(out, s=(n, n), axes=(1, 2)).reshape(residual.shape)
    return transform
