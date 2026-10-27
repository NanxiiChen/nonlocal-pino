"""Differentiable exact spectral equilibration for homogeneous isotropic C.

Derived from the declared periodic elastic energy; zero macrostrain is fixed.
Odd grids have only the translational zero mode, whose displacement is fixed.
"""
import equinox as eqx
import jax.numpy as jnp
import numpy as np


class ElasticEquilibrium(eqx.Module):
    kx: object
    ky: object
    u_factor: object
    chemical_factor: object
    eigenstrain: object
    lame: float = eqx.field(static=True)
    shear: float = eqx.field(static=True)
    constant: float = eqx.field(static=True)
    n: int = eqx.field(static=True)

    def __init__(self, c):
        self.n = n = c.sample.Nx
        if n % 2 != 1:
            raise ValueError('Spectral equilibrium requires odd grid')
        k = 2*np.pi*np.fft.fftfreq(n, d=c.geometry.length/n)
        kx, ky = np.meshgrid(k, k, indexing='xy')
        kvec = np.stack((kx, ky), axis=-1)
        k2 = kx*kx+ky*ky
        nonzero = k2 > 0
        safe_k2 = np.where(nonzero, k2, 1.)  # Explicit translation-mode gauge only.
        self.lame, self.shear = c.physical.lame_lambda, c.physical.shear_modulus
        exx, eyy, exy = c.physical.eigenstrain
        E = np.array([[exx, exy], [exy, eyy]])
        S = self.lame*np.trace(E)*np.eye(2)+2*self.shear*E
        b = np.einsum('ij,...j->...i', S, kvec)
        inv_b = (b-(self.lame+self.shear)/(self.lame+2*self.shear)
                 *kvec*np.sum(kvec*b,axis=-1)[...,None]/safe_k2[...,None])/(self.shear*safe_k2[...,None])
        d = float(np.sum(S*E))
        B = d-np.sum(b*inv_b, axis=-1)
        if B.min() < -1e-12:
            raise ValueError('Reduced elastic energy must be nonnegative')
        self.kx, self.ky = jnp.asarray(kx), jnp.asarray(ky)
        self.u_factor = jnp.asarray(-.5j*inv_b*nonzero[...,None])
        self.chemical_factor = jnp.asarray(B/4)
        self.eigenstrain = jnp.asarray([exx, eyy, exy])
        self.constant = d/4

    def __call__(self, phi):
        shape = phi.shape
        p = phi.reshape(self.n, self.n)
        hat = jnp.fft.fft2(p)
        uh = self.u_factor*hat[...,None]
        u = jnp.fft.ifft2(uh, axes=(0,1)).real
        strain = jnp.stack((jnp.fft.ifft2(1j*self.kx*uh[...,0]).real,
            jnp.fft.ifft2(1j*self.ky*uh[...,1]).real,
            .5*jnp.fft.ifft2(1j*(self.ky*uh[...,0]+self.kx*uh[...,1])).real), axis=-1)
        elastic = strain-(.5*(1+p))[...,None]*self.eigenstrain
        tr = elastic[...,0]+elastic[...,1]
        stress = 2*self.shear*elastic+self.lame*tr[...,None]*jnp.asarray([1.,1.,0.])
        chemical = jnp.fft.ifft2(self.chemical_factor*hat).real+self.constant
        return chemical.reshape(shape), u.reshape(shape+(2,)), stress.reshape(shape+(3,))

    def energy(self, phi):
        chemical, _, _ = self(phi)
        return jnp.mean(.5*(1+phi)*chemical)
