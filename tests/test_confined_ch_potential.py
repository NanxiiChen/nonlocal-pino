"""Independent conservation and two-hop tests for the experimental CH coordinate."""
import unittest
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from tests.test_confined_ch_operator import small_fixture
from examples.confined_ch2d.operator_training import build_network, encode
from examples.confined_ch2d.operator_potential import (
    compact_payload, compact_state, full_state, fields, length_squared, make_loss)


class ConservativePotentialTests(unittest.TestCase):
    def test_two_encoder_streams_retain_full_compact_gradients(self):
        from examples.confined_ch2d.configs.train_operator_heterogeneous import get_config
        from examples.confined_ch2d import operator_training as canonical
        with jax.enable_x64(True):
            _,raw=small_fixture();c=get_config()
            c.arch.width=8;c.arch.heads=2;c.arch.condition_depth=1
            c.arch.decoder_depth=1;c.arch.spatial_features=2;c.arch.time_features=2
            c.train.compute_dtype='float64'
            c.operator.formulation='conservative_potential';c.operator.metric='identity'
            raw={key:np.asarray(value) for key,value in raw.items() if key!='metric'}
            core=np.array([2,0,1],dtype=np.int32)
            small,_=compact_payload(raw,core,raw['volumes'][core]/raw['volumes'].sum(),inner_capacity=5,query_capacity=6)
            net=build_network(c,jax.random.key(31));times=jnp.array([0.,.02])
            batch=lambda d:jax.tree.map(lambda a:jnp.asarray(a)[None],d)
            fn=eqx.filter_jit(eqx.filter_value_and_grad(canonical.make_loss(c)))
            full,gfull=fn(net,batch(raw),times)
            compact,gcompact=fn(net,batch(small),times)
            np.testing.assert_allclose(full,compact,rtol=2e-12,atol=2e-12)
            for a,b in zip(jax.tree.leaves(gfull),jax.tree.leaves(gcompact),strict=True):
                np.testing.assert_allclose(a,b,rtol=2e-10,atol=2e-10)
            for lift in gcompact.lifts:
                self.assertGreater(float(jnp.linalg.norm(lift.layers[0].weight)),0.)

    def test_canonical_dispatch_uses_identity_without_a_dense_metric(self):
        from examples.confined_ch2d import operator_training as canonical
        from examples.confined_ch2d.operator_problem import validate
        with jax.enable_x64(True):
            c,raw=small_fixture()
            c.operator.formulation='conservative_potential';c.operator.metric='identity'
            validate(c)
            data={key:value for key,value in raw.items() if key!='metric'}
            net=build_network(c,jax.random.key(11));times=jnp.array([0.,.02])
            np.testing.assert_allclose(canonical.fields(net,data,times,c),fields(net,data,times,c),atol=1e-12)
            batch=jax.tree.map(lambda a:a[None],data)
            np.testing.assert_allclose(canonical.make_loss(c)(net,batch,times),make_loss(c)(net,batch,times),atol=1e-12)
            with self.assertRaisesRegex(ValueError,'complete physical graph'):
                canonical.query_phase(net,data,data['points'],times,c)

    def test_independent_matrix_identity_and_gauge(self):
        rng = np.random.default_rng(812)
        n = 11
        volumes = rng.uniform(.2, 1., n)
        pair = rng.uniform(0., 1., (n, n))
        pair = (pair+pair.T)/2
        np.fill_diagonal(pair, 0.)
        D = (np.diag(pair.sum(axis=1))-pair)/volumes[:, None]
        np.testing.assert_allclose(D@np.ones(n), 0., atol=2e-14)
        np.testing.assert_allclose(volumes@D, 0., atol=2e-14)
        phi0, chi, chi_t = rng.normal(size=(3, n))
        epsilon, beta, mobility = .06, 1., .02
        alpha = epsilon**2/beta
        phi = phi0+alpha*D@chi
        mu = beta*(phi**3-phi)+epsilon**2*D@phi
        original = alpha*D@chi_t+mobility*D@mu
        transformed = chi_t+mobility/alpha*mu
        np.testing.assert_allclose(original, alpha*D@transformed, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(volumes@phi, volumes@phi0, atol=1e-13)
        np.testing.assert_allclose(phi0+alpha*D@(chi+4.), phi, atol=1e-13)

    def test_complete_two_hops_and_network_parameter_gradients(self):
        with jax.enable_x64(True):
            c, raw = small_fixture()
            c.train.compute_dtype = 'float64'
            raw = jax.tree.map(np.asarray, raw)
            core = np.array([2, 0, 1], dtype=np.int32)
            weights = raw['volumes'][core]/raw['volumes'].sum()
            compact, report = compact_payload(raw, core, weights, inner_capacity=4, query_capacity=4)
            self.assertEqual(report['query'], 3)
            for key in ('points', 'initial', 'volumes', 'condition', 'condition_weights'):
                self.assertIs(raw[key], compact[key])
            net = build_network(c, jax.random.key(5))
            data, part = jax.tree.map(jnp.asarray, raw), jax.tree.map(jnp.asarray, compact)
            a = net.prepare_encoded(encode(net, data), data['points'])
            b = net.prepare_encoded(encode(net, part), part['query_points'])
            for t in (0., .02):
                full = full_state(net, a, jnp.asarray(t), data, c)
                small = compact_state(net, b, jnp.asarray(t), part, c)
                for x, y in zip(full, small, strict=True):
                    np.testing.assert_allclose(x[core], y, rtol=1e-11, atol=1e-11)
            times = jnp.array([0., .02])
            prediction = fields(net, data, times, c)
            np.testing.assert_array_equal(prediction[0, :, 0], raw['initial'])
            np.testing.assert_allclose(prediction[:, :, 0]@raw['volumes'], raw['initial']@raw['volumes'], atol=1e-12)
            fn = eqx.filter_jit(eqx.filter_value_and_grad(make_loss(c)))
            batch = lambda d: jax.tree.map(lambda a: jnp.asarray(a)[None], d)
            value, grad = fn(net, batch(data), times)
            sampled, sample_grad = fn(net, batch(part), times)
            np.testing.assert_allclose(value, sampled, rtol=1e-11, atol=1e-11)
            for x, y in zip(jax.tree.leaves(grad), jax.tree.leaves(sample_grad), strict=True):
                np.testing.assert_allclose(x, y, rtol=1e-10, atol=1e-10)
            with self.assertRaisesRegex(ValueError, 'overflow'):
                compact_payload(raw, core, weights, inner_capacity=2, query_capacity=4)


if __name__ == '__main__':
    unittest.main()
