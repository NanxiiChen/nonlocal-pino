"""Nodal constitutive condensation, retained traces and independent derivatives."""
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from examples.channel_nsch2d.chemical_projection import project_state,project_rate,scalar_laplacian
from examples.channel_nsch2d.data import geometry_data
from examples.channel_nsch2d.problem import material
from examples.channel_nsch2d.losses import gmls_first_jets
from tests.test_channel_time_jvp import config


def fixture():
    c=config();c.sample.Nx=24;c.sample.Ny=10;c.nl.degree=3;c.nl.neighbors=36
    c.sample.boundary_points=6
    data,_=geometry_data((.12,.08),c)
    return c,data


class ChannelChemicalProjectionTests(unittest.TestCase):
    def test_dense_reference_idempotence_trace_and_zero_set(self):
        c,d=fixture();n=len(d['points']);rng=np.random.default_rng(12)
        value=rng.normal(size=(3,n,5));value[...,0]*=.3
        matrix=np.zeros((n,n));rows=np.broadcast_to(np.arange(n)[:,None],d['neighbors'].shape)
        np.add.at(matrix,(rows,d['neighbors']),d['laplacian'])
        matrix[np.diag_indices(n)]-=d['laplacian'].sum(-1)
        independent=value[...,0]@matrix.T
        np.testing.assert_allclose(scalar_laplacian(value[...,0],d,xp=np),independent,rtol=2e-12,atol=2e-11)
        A,kappa=material(c);interior=d['interior']>0
        expected=A*(value[...,0]**3-value[...,0])-kappa*independent
        projected=project_state(value,d,c,xp=np)
        np.testing.assert_allclose(projected[:,interior,1],expected[:,interior],rtol=2e-12,atol=2e-12)
        np.testing.assert_array_equal(projected[:,~interior,1],value[:,~interior,1])
        np.testing.assert_array_equal(projected[...,[0,2,3,4]],value[...,[0,2,3,4]])
        np.testing.assert_array_equal(project_state(projected,d,c,xp=np),projected)
        residual=projected[...,1]-A*(projected[...,0]**3-projected[...,0])+kappa*scalar_laplacian(projected[...,0],d,xp=np)
        np.testing.assert_allclose(residual[:,interior],0.,rtol=0,atol=2e-12)
        original_zero=value.copy();original_zero[:,interior,1]=expected[:,interior]
        np.testing.assert_allclose(project_state(original_zero,d,c,xp=np),original_zero,rtol=2e-12,atol=2e-12)
        self.assertGreater(float(np.max(np.abs(projected[:,~interior,1]-expected[:,~interior]))),.1)

    def test_manufactured_nonzero_boundary_gradients(self):
        with jax.enable_x64(True):
            c,raw=fixture();x,y=raw['points'].T;phi=.1+.08*x-.05*y
            A,_=material(c);mu=A*(phi**3-phi);inside=raw['interior']>0
            value=np.zeros((1,len(x),5));value[0,:,0]=phi;value[0,:,1]=mu
            value[0,inside,1]=7.
            projected=project_state(value,raw,c,xp=np)
            np.testing.assert_allclose(projected[0,:,1],mu,rtol=0,atol=3e-12)
            b=raw['boundary'];p=.1+.08*b[:,0]-.05*b[:,1]
            trace=np.zeros((1,len(b),5));trace[0,:,0]=p;trace[0,:,1]=A*(p**3-p)
            d={k:jnp.asarray(v)for k,v in raw.items()}
            gx,gy=gmls_first_jets(jnp.asarray(projected),jnp.asarray(trace),d,boundary=True)
            np.testing.assert_allclose(gx[0,:,1],A*(3*p*p-1)*.08,rtol=0,atol=2e-10)
            np.testing.assert_allclose(gy[0,:,1],A*(3*p*p-1)*(-.05),rtol=0,atol=2e-10)

    def test_projection_tangent_matches_jvp_and_finite_difference(self):
        with jax.enable_x64(True):
            c,raw=fixture();d={k:jnp.asarray(v)for k,v in raw.items()};rng=np.random.default_rng(13)
            value=jnp.asarray(rng.normal(size=(3,len(raw['points']),5))*.2)
            rate=jnp.asarray(rng.normal(size=value.shape)*.01)
            _,actual=jax.jvp(lambda v:project_state(v,d,c,xp=jnp),(value,),(rate,))
            expected=project_rate(value,rate,d,c,xp=jnp)
            np.testing.assert_allclose(actual,expected,rtol=2e-12,atol=2e-12)
            h=1e-5;numeric=(project_state(value+h*rate,d,c,xp=jnp)-project_state(value-h*rate,d,c,xp=jnp))/(2*h)
            np.testing.assert_allclose(expected,numeric,rtol=2e-6,atol=2e-8)
            np.testing.assert_array_equal(np.asarray(expected)[:,raw['interior']==0,1],np.asarray(rate)[:,raw['interior']==0,1])
            with self.assertRaisesRegex(ValueError,'aligned final nodal'):scalar_laplacian(value[...,0][:,:-1],d,xp=jnp)
            with self.assertRaisesRegex(ValueError,'align'):project_rate(value,rate[:,:,:4],d,c,xp=jnp)


if __name__=='__main__':unittest.main()
