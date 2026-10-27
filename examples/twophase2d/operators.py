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
