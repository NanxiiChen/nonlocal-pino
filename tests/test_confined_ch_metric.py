"""Independent weighted resolvent solve and its residual gradient."""
import unittest
import jax,jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict
from examples.confined_ch2d.configs.train_coarsening_metric import get_config
from examples.confined_ch2d.operators import RegionalLaplacian
from examples.confined_ch2d.native_reference import native_matrix
from examples.confined_ch2d.preconditioning import BiharmonicMetric
from tests.test_confined_ch import fixture


class MetricTests(unittest.TestCase):
    def test_neumann_inverse_rejects_extra_null_modes(self):
        from types import SimpleNamespace
        from examples.confined_ch2d.preconditioning import NeumannInverseMetric
        two_components=SimpleNamespace(neighbors=np.array([[1],[0],[3],[2]]),coefficients=np.ones((4,1)))
        with self.assertRaisesRegex(ValueError,'connected graph'):NeumannInverseMetric(two_components,np.ones(4))

    def test_neumann_inverse_and_mass_subspace(self):
        from examples.confined_ch2d.preconditioning import NeumannInverseMetric
        c,x,v=fixture();A=native_matrix(x,v,c).toarray();P=np.ones((len(x),1))*v[None,:]/v.sum()
        r=np.random.default_rng(17).normal(size=len(x));r-=r@v/v.sum()
        expected=np.linalg.solve(A+P,r)
        with jax.enable_x64(True):
            metric=NeumannInverseMetric(RegionalLaplacian(x,v,c),v)
            actual=np.asarray(metric(jnp.asarray(r)))
            np.testing.assert_allclose(actual,expected,atol=2e-13,rtol=2e-11)
            np.testing.assert_allclose(A@actual,r,atol=1e-12,rtol=1e-12)
            np.testing.assert_allclose(metric(jnp.ones(len(x))),1,atol=1e-11)
            f=lambda scale:jnp.sum(jnp.asarray(v)*metric(jnp.asarray(r)*scale)**2)/v.sum()
            np.testing.assert_allclose(jax.grad(f)(.7),1.4*np.sum(v*expected**2)/v.sum(),rtol=2e-11)

    def test_weighted_solve_root_and_gradient(self):
        c=get_config();_,x,v=fixture();c.nl.delta=.2
        A=native_matrix(x,v,c).toarray();root=np.sqrt(v)
        S=root[:,None]*A/root[None,:]
        P=np.eye(len(x))+c.loss.phase_metric.time_scale*c.physical.mobility*c.physical.epsilon**2*(S.T@S)
        r=np.random.default_rng(7).normal(size=len(x))
        expected=np.linalg.solve(P,root*r)/root
        with jax.enable_x64(True):
            metric=BiharmonicMetric(RegionalLaplacian(x,v,c),v,time_scale=c.loss.phase_metric.time_scale,mobility=c.physical.mobility,epsilon=c.physical.epsilon)
            actual=np.asarray(metric(jnp.asarray(r)))
            np.testing.assert_allclose(actual,expected,atol=2e-13,rtol=2e-13)
            np.testing.assert_allclose(metric(jnp.zeros(len(x))),0.,atol=1e-15)
            np.testing.assert_allclose(metric(jnp.ones(len(x))),1.,atol=1e-13)
            fun=lambda a:jnp.sum(jnp.asarray(v)*metric(jnp.asarray(r)*a)**2)/v.sum()
            np.testing.assert_allclose(jax.grad(fun)(.7),1.4*np.sum(v*expected**2)/v.sum(),rtol=1e-12)


if __name__=='__main__':unittest.main()
