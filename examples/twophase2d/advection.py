"""Conservative Fourier divergence of periodic nodal velocity-phase flux."""
import jax.numpy as jnp
import numpy as np


def make_flux_divergence(config, *, dtype):
    n = config.sample.Nx
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
