"""Periodic finite-horizon negative Laplacian with explicit moment calibration."""
import equinox as eqx
import jax.numpy as jnp
import numpy as np


class PeriodicLaplacian(eqx.Module):
    indices: object
    weights: object
    moment_correction: float = eqx.field(static=True)

    def __init__(self, n, length, delta, *, dtype):
        h = length/n
        if n < 8 or not h < delta < length/2:
            raise ValueError('Periodic operator requires n>=8 and h<delta<L/2')
        offsets = []
        weights = []
        for dy in range(-int(np.ceil(delta/h)), int(np.ceil(delta/h))+1):
            for dx in range(-int(np.ceil(delta/h)), int(np.ceil(delta/h))+1):
                distance = h*np.hypot(dx, dy)
                if 0 < distance < delta:
                    offsets.append((dy, dx))
                    weights.append(120/(np.pi*delta**4)*(1-distance/delta)**2*h*h)
        offsets = np.asarray(offsets, dtype=np.int32)
        weights = np.asarray(weights, dtype=np.float64)
        moment = np.einsum('i,ij,ik->jk', weights, h*offsets, h*offsets)
        if not np.allclose(moment, np.eye(2)*moment[0, 0], atol=1e-12, rtol=1e-12):
            raise ValueError('Periodic kernel moment must be isotropic')
        correction = 2/moment[0, 0]
        yy, xx = np.indices((n, n))
        indices = np.stack([(((yy+dy) % n)*n+(xx+dx) % n).ravel() for dy, dx in offsets])
        self.indices = jnp.asarray(indices)
        self.weights = jnp.asarray(weights*correction, dtype=dtype)
        self.moment_correction = float(correction)

    def __matmul__(self, values):
        if values.ndim not in (1, 2) or values.shape[0] != self.indices.shape[1]:
            raise ValueError('Operator input must be(N,) or(N,Nt) on the fixed full grid')
        differences = values[None] - values[self.indices]
        w = self.weights.reshape((-1,) + (1,)*values.ndim)
        return jnp.sum(w*differences, axis=0)


class PeriodicGradient(eqx.Module):
    """Moment-corrected nodal gradient on the full uniform periodic grid.

    For symmetric radial weights, each component is skew-adjoint. The
    divergence below is its negative adjoint in the uniform-grid inner product.
    This centered nodal pair has checkerboard null modes on even grids; it is
    not by itself a pressure-stable NS discretization. Keep the independent
    pair Laplacian for diffusion rather than composing divergence and gradient.
    """
    indices: object
    coefficients: object

    def __init__(self, n, length, delta, *, dtype):
        h = length/n
        if n < 8 or not h < delta < length/2:
            raise ValueError('Periodic gradient requires n>=8 and h<delta<L/2')
        offsets, weights = [], []
        limit = int(np.ceil(delta/h))
        for dy in range(-limit, limit+1):
            for dx in range(-limit, limit+1):
                distance = h*np.hypot(dx, dy)
                if 0 < distance < delta:
                    offsets.append((dy, dx))
                    weights.append((1-distance/delta)**2*h*h)
        offsets = np.asarray(offsets, dtype=np.int32)
        weights = np.asarray(weights, dtype=np.float64)
        xi = h*offsets[:, ::-1]  # Cartesian x,y, independent of grid row order.
        moment = np.einsum('e,ei,ej->ij', weights, xi, xi)
        if not np.allclose(moment, np.eye(2)*moment[0, 0], atol=1e-12, rtol=1e-12):
            raise ValueError('Periodic gradient kernel moment must be isotropic')
        if not np.isfinite(moment[0, 0]) or moment[0, 0] <= 0:
            raise ValueError('Periodic gradient requires positive finite second moment')
        yy, xx = np.indices((n, n))
        indices = np.stack([(((yy+dy) % n)*n+(xx+dx) % n).ravel() for dy, dx in offsets])
        self.indices = jnp.asarray(indices)
        self.coefficients = jnp.asarray(weights[:, None]*xi/moment[0, 0], dtype=dtype)

    def __matmul__(self, values):
        if values.ndim != 1 or values.shape[0] != self.indices.shape[1]:
            raise ValueError('Periodic gradient input must be(N,) on the full grid')
        difference = values[self.indices]-values[None]
        return jnp.einsum('en,ed->nd', difference, self.coefficients)

    def divergence(self, vectors):
        if vectors.shape != (self.indices.shape[1], 2):
            raise ValueError('Periodic divergence input must be(N,2) on the full grid')
        difference = vectors[self.indices]-vectors[None]
        return jnp.einsum('end,ed->n', difference, self.coefficients)
