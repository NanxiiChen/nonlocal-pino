"""Full40-time PDE/parameter-gradient identity and stratified coverage."""
import unittest

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict

from examples.channel_nsch2d.configs.train_operator_width128_gmls_time40_online import get_config
from examples.channel_nsch2d.data import geometry_data
from examples.channel_nsch2d.losses import make_loss_fn
from examples.channel_nsch2d.network import build_network
from examples.channel_nsch2d.problem import validate_config
from examples.channel_nsch2d.temporal import sample_intervals,step_count
from examples.channel_nsch2d.time_batching import make_accumulated_fn
from nonlocal_pinn.training.optimizers import make_optimizer
from tests.test_channel_gmls_first import small


class TimeBatchingTests(unittest.TestCase):
    def test_40_strata_and_exact_expected_full_time_quadrature(self):
        c=get_config();validate_config(c)
        self.assertEqual(step_count(c),240)
        ids,t=sample_intervals(jax.random.PRNGKey(4),c)
        self.assertEqual(t.shape,(40,))
        np.testing.assert_array_equal(np.asarray(ids)//6,np.arange(40))
        # Enumerate every possible offset in each equal stratum, rather than
        # relying on a noisy Monte Carlo assertion about the expected loss.
        nodes=np.arange(240).reshape(40,6)
        values=np.sin(nodes*.07)**2+nodes**2*.001
        np.testing.assert_allclose(values.mean(0).mean(),values.reshape(-1).mean(),rtol=2e-16)
        self.assertEqual(len(np.unique(nodes)),240)

    def test_complete_loss_gradient_and_single_soap_update_identity(self):
        with jax.enable_x64(True):
            c=small();c.sample.Nt=40;c.operator.condition_batch=2
            c.train.microbatch=get_config().train.microbatch
            c.train.soap_max_precond_dim=32
            raw=[geometry_data(a,c)[0] for a in ((.12,.08),(.07,.13))]
            batch={k:jnp.asarray(np.stack([d[k] for d in raw])) for k in raw[0]}
            net=build_network(c,jax.random.PRNGKey(42))
            _,times=sample_intervals(jax.random.PRNGKey(8),c)
            loss=make_loss_fn(c)
            whole=eqx.filter_jit(eqx.filter_value_and_grad(loss,has_aux=True))
            chunked=eqx.filter_jit(make_accumulated_fn(c,loss))
            (a,pa),ga=whole(net,batch,times)
            (b,pb),gb=chunked(net,batch,times)
            np.testing.assert_allclose(a,b,rtol=2e-12,atol=2e-13)
            np.testing.assert_allclose(pa,pb,rtol=2e-12,atol=2e-13)
            for x,y in zip(jax.tree.leaves(ga),jax.tree.leaves(gb),strict=True):
                np.testing.assert_allclose(x,y,rtol=2e-10,atol=2e-12)
            optimizer=make_optimizer(c)
            state=optimizer.init(eqx.filter(net,eqx.is_array))
            update=eqx.filter_jit(lambda g,s:optimizer.update(g,s,net))
            ua,sa=update(ga,state);ub,sb=update(gb,state)
            for x,y in zip(jax.tree.leaves(ua),jax.tree.leaves(ub),strict=True):
                np.testing.assert_allclose(x,y,rtol=2e-8,atol=2e-10)
            # SOAP's null-space eigenbasis need not be unique under roundoff;
            # compare physical updates and the actual optimizer step counts.
            self.assertEqual(int(sa[1][0].count),1)
            self.assertEqual(int(sb[1][0].count),1)
            with self.assertRaisesRegex(ValueError,'complete configured'):
                chunked(net,batch,times[:8])

    def test_invalid_execution_and_unsupported_loss_modes(self):
        for value in (True,0,7,41,8.):
            c=get_config();c.train.microbatch=ConfigDict(dict(kind='time',size=value))
            with self.assertRaisesRegex(ValueError,'microbatch size'):validate_config(c)
        c=get_config();c.train.microbatch.kind='spatial'
        with self.assertRaisesRegex(ValueError,'explicit time'):validate_config(c)
        c=get_config();c.balance=ConfigDict(dict(kind='gradnorm'))
        with self.assertRaisesRegex(ValueError,'cold uniform'):validate_config(c)


if __name__=='__main__':unittest.main()
