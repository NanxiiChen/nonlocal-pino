"""Conservative Fourier or finite-horizon divergence of periodic nodal flux."""
import jax.numpy as jnp
import numpy as np
from examples.twophase2d.operators import PeriodicGradient


def make_flux_divergence(config, *, dtype):
    n = config.sample.Nx
    if config.advection == 'nonlocal_flux':
        derivative = PeriodicGradient(n, config.geometry.length, config.nl.delta, dtype=dtype)

        def nonlocal_divergence(phi, velocity):
            if phi.shape != (n*n,) or velocity.shape != (n*n, 2):
                raise ValueError('Nonlocal flux requires full periodic grid phase and velocity')
            return derivative.divergence(phi[:, None]*velocity)
        return nonlocal_divergence
    if config.advection != 'spectral_flux':
        raise ValueError('Flux advection must be spectral_flux or nonlocal_flux')
    frequency = 2*np.pi*np.fft.fftfreq(n, d=config.geometry.length/n)
    frequency[n//2] = 0.  # Real even-grid first derivative at Nyquist.
    ky = jnp.asarray(frequency, dtype=dtype)
    kx = jnp.asarray(frequency[:n//2+1], dtype=dtype)

    def divergence(phi, velocity):
        if phi.shape != (n*n,) or velocity.shape != (n*n, 2):
            raise ValueError('Spectral flux requires full periodic grid phase and velocity')
        flux = (phi[:, None]*velocity).reshape(n, n, 2)
        hat = jnp.fft.rfft2(flux, axes=(0, 1))
        div = 1j*(kx[None, :]*hat[..., 0]+ky[:, None]*hat[..., 1])
        return jnp.fft.irfft2(div, s=(n, n)).ravel()
    return divergence
