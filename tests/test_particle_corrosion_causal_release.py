"""A released causal step equals the ordinary full physical objective/update."""
import unittest

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from ml_collections import ConfigDict

from examples.particle_corrosion2d import losses,strategies
from examples.particle_corrosion2d.configs.train_physical_single_causal_release import get_config
from tests.test_particle_corrosion_losses import fixture


class CausalReleaseTests(unittest.TestCase):
    def test_release_matches_complete_physical_adam_update_and_freezes_controller(self):
        c,net,data,times=fixture()
        c.train.compute_dtype='float32'
        c.loss.gradnorm_norm='l2'
        c.loss.pde_reduction='mean_square'
        c.loss.initial_metric='phase_liquid'
        c.loss.use_causal=True
        c.causal=ConfigDict(dict(step_size=1.1,max_eps=100.,max_last_weight=.1,min_mean_weight=.05))
        payload=jax.tree.map(lambda x:x[None],data)
        optimizer=optax.adam(1e-3);state=optimizer.init(eqx.filter(net,eqx.is_array))
        weights=jnp.array([.7,.9,1.2,1.4,.8]);eps=jnp.array(2.)
        components=losses.make_loss_components(c)
        actual=strategies.make_causal_step(components,optimizer,c,losses.LOSS_NAMES)(
            net,state,weights,eps,times,payload,jnp.array(True),None,jnp.array(False))
        c.loss.use_causal=False
        ordinary=losses.make_step(losses.make_loss_vector(c),optimizer,c,losses.LOSS_NAMES)(
            net,state,weights,times,payload,jnp.array(True))
        self.assertTrue(bool(actual[-1]));self.assertTrue(bool(ordinary[-1]))
        self.assertEqual(float(actual[3]),float(eps))
        np.testing.assert_array_equal(actual[5][0],np.ones(len(times)))
        for a,b in zip(jax.tree.leaves((actual[:3],actual[4])),
                       jax.tree.leaves((ordinary[:3],ordinary[3])),strict=True):
            if eqx.is_array(a):np.testing.assert_allclose(a,b,rtol=2e-5,atol=2e-7)

    def test_schedule_requires_causal_and_an_explicit_integer_within_budget(self):
        c=get_config();strategies.validate_strategies(c)
        for invalid in (True,0,-1,2.5,c.train.iterations):
            c=get_config();del c.loss.causal_steps
            c.loss.causal_steps=invalid
            with self.assertRaisesRegex(ValueError,'Causal release'):
                strategies.validate_strategies(c)
        c.loss.causal_steps=5000;c.loss.use_causal=False
        with self.assertRaisesRegex(ValueError,'Causal release'):
            strategies.validate_strategies(c)


if __name__=='__main__':unittest.main()
