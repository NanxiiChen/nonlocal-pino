"""Independent NumPy SDF identity and fresh batched continuous-input checks."""
import unittest
import jax
import jax.numpy as jnp
import numpy as np

from examples.electropolish2d.configs.train_operator_online import get_config
from examples.electropolish2d.online_conditions import initial_fields, make_sampler
from examples.electropolish2d.operator_data import validate_config
from examples.electropolish2d.losses import _rough_surface, _sdf_to_curve


class OnlineConditionTests(unittest.TestCase):
    def test_initial_fields_match_existing_numpy_sdf(self):
        with jax.enable_x64(True):
            c=get_config();validate_config(c)
            points=np.array([[.01,.2],[.19,.3],[.31,.34],[.5,.4],[.78,.32],[.99,.2],[.5,0.],[.5,.5]])
            curve=np.linspace(0,1,2000)
            k=np.sqrt(c.physical.omega_phi/(2*c.physical.alpha_phi))*c.nondim.L_c
            for identity in (1662057944,1497586430):
                rng=np.random.RandomState(identity)
                coefficients=np.array([(rng.uniform(.3,1)*c.surface.amplitude,
                    rng.uniform(c.surface.freq_lo,c.surface.freq_hi),rng.uniform(0,2*np.pi))
                    for _ in range(c.surface.n_modes)])
                yc=_rough_surface(curve,c.surface.y0,coefficients)
                ys=_rough_surface(points[:,0],c.surface.y0,coefficients)
                sdf=_sdf_to_curve(points[:,0],points[:,1],curve,yc,ys,window=16)
                phi=.5*(1+np.tanh(k*sdf));phi[points[:,1]==0]=1;phi[points[:,1]==.5]=0
                expected=np.stack([phi,c.physical.c_se*(3*phi**2-2*phi**3)],-1)
                actual=jax.jit(lambda a:initial_fields(a,jnp.asarray(points),jnp.asarray(curve),
                    jnp.asarray(np.searchsorted(curve,points[:,0])),y0=c.surface.y0,
                    k=k,c_se=c.physical.c_se,window=16))(jnp.asarray(coefficients))
                np.testing.assert_allclose(actual,expected,rtol=1e-11,atol=1e-12)

    def test_online_samples_are_fresh_batched_and_replayable(self):
        c=get_config();sample,geometry=make_sampler(c)
        points=jnp.array([[.1,.2],[.3,.33],[.8,.4],[.5,0.],[.5,.5]],dtype=jnp.float32)
        nearest=geometry(points);key=jax.random.key(3)
        first=sample(key,points,nearest);same=sample(key,points,nearest)
        other=sample(jax.random.key(4),points,nearest)
        for a,b in zip(first,same):np.testing.assert_array_equal(a,b)
        images,local,coefficients,count=first
        self.assertEqual(images.shape,(8,1,32,64));self.assertEqual(local.shape,(8,5,2))
        self.assertGreaterEqual(int(count),8);self.assertGreaterEqual(int(other[3]),8)
        self.assertFalse(np.array_equal(coefficients,other[2]))
        self.assertEqual(len(np.unique(np.r_[coefficients,other[2]].reshape(16,-1),axis=0)),16)
        np.testing.assert_array_equal(local[:,3],1);np.testing.assert_array_equal(local[:,4],0)
        self.assertTrue(np.isfinite(images).all() and np.isfinite(local).all())
        self.assertTrue(np.all(np.asarray(coefficients)[...,0]>=.3*c.surface.amplitude))
        bad=get_config();bad.operator.training_indices=(0,)
        with self.assertRaisesRegex(ValueError,'finite training subset'):validate_config(bad)
        bad=get_config();bad.operator.online.candidate_factor=0
        with self.assertRaisesRegex(ValueError,'candidate_factor'):validate_config(bad)


if __name__=='__main__':unittest.main()
