"""Independent analytic jets, exact loss accumulation and matched input checks."""
import unittest
from unittest.mock import patch

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from examples.channel_nsch2d.configs.train_operator_local_ad import get_config
from examples.channel_nsch2d.local_ad import spatial_jets, conservative_advection, validate_payload
from examples.channel_nsch2d.data import geometry_data
from examples.channel_nsch2d.geometry import initial_parameters
from examples.channel_nsch2d.losses import make_loss_fn, part_names
from examples.channel_nsch2d.network import build_network, encode, fields
from examples.channel_nsch2d.problem import validate_config
from examples.channel_nsch2d.time_batching import make_accumulated_fn


def small():
    c = get_config(); c.train.seed = 42; c.train.compute_dtype = 'float64'
    c.sample.Nx = 8; c.sample.Ny = 8; c.sample.Nt = 4; c.sample.boundary_points = 4
    c.arch.width = 4; c.arch.heads = 1; c.arch.condition_depth = 1; c.arch.decoder_depth = 1
    c.arch.spatial_features = 2; c.arch.time_features = 4
    c.operator.condition_batch = 1; c.operator.train_conditions = 4
    c.train.soap_max_precond_dim = 32
    return c


class LocalADTests(unittest.TestCase):
    def tearDown(self):
        jax.clear_caches()

    def test_analytic_jets_third_potential_derivative_and_conservative_flux(self):
        with jax.enable_x64(True):
            points = jnp.array([[.37, .29], [1.2, .6], [2.1, .41]])
            times = jnp.array([.3, 1.7])

            def potentials(net, encoded, x, t, amplitudes, c):
                xx, yy = x.T; tt = t[:, None]
                # psi = t*x^3*y^2 -> u = 2t*x^3*y, v = -3t*x^2*y^2.
                return jnp.stack((xx[None]**2+yy[None]**3+tt,
                    net*jnp.sin(xx[None])+yy[None]**2+0*tt,
                    tt*xx[None]**3*yy[None]**2, tt*xx[None]*yy[None]), axis=-1)

            c = small()
            with patch('examples.channel_nsch2d.network.potentials', potentials):
                jets = lambda a: spatial_jets(lambda x: fields(a, None, x, times, None, c), points)
                value, dx, dy, lap = jets(1.3)
                x, y = np.asarray(points).T; t = np.asarray(times)[:, None]
                expected_lap = np.stack((np.broadcast_to(2+6*y, t.shape[:1]+x.shape),
                    np.broadcast_to(2-1.3*np.sin(x), t.shape[:1]+x.shape),
                    12*t*x*y, -6*t*(x*x+y*y), np.zeros_like(t*x)), axis=-1)
                np.testing.assert_allclose(lap, expected_lap, atol=2e-13, rtol=2e-13)
                np.testing.assert_allclose(dx[..., 2]+dy[..., 3], 0., atol=2e-13)
                np.testing.assert_allclose(jax.jacfwd(lambda a: jets(a)[3])(1.3)[..., 1],
                    np.broadcast_to(-np.sin(x), (len(t), len(x))), atol=2e-13)
            # Non-divergence-free manufactured field tests the full product rule.
            fn = lambda z: jnp.stack((z[:, 0]**2+z[:, 1], z[:, 0]*0,
                                      z[:, 0]**2, 3*z[:, 1], z[:, 1]*0), axis=-1)[None]
            v, vx, vy, _ = spatial_jets(fn, points)
            expected = 2*x**3+3*y+(x*x+y)*(2*x+3)
            np.testing.assert_allclose(conservative_advection(v, vx, vy)[0], expected, atol=2e-13)

    def test_input_pairing_with_actual_historical_bank(self):
        from pathlib import Path
        from examples.channel_nsch2d.configs.train_default import get_config as baseline
        from examples.channel_nsch2d.transport import build_transport
        c = get_config(); c.train.seed = 42; validate_config(c)
        old = Path('/root/autodl-tmp/tf-logs/channel_nsch2d/20261103-pro6000-integral-random-walls-online-width192-b8-time18-nt36-cold20k-lr1e5-04/known_inputs.npz')
        if not old.exists():
            self.skipTest('Historical external bank not installed')
        with np.load(old) as saved:
            a = initial_parameters(c)[0]
            raw, _ = geometry_data(a, c)
            data = {k: v.astype(c.train.compute_dtype) for k, v in raw.items()}
            validate_payload(data, c)
            for key in raw:
                np.testing.assert_array_equal(raw[key], saved[key][0], err_msg=key)
            self.assertFalse(any(k in data for k in ('neighbors', 'laplacian', 'transport_dx', 'gradient_x')))
            b = baseline()
            measure = build_transport(raw['points'], a, b)['transport_measure']
            np.testing.assert_array_equal(raw['transport_measure'], measure)

    def test_real_network_derivatives_gradient_and_chunk_equivalence(self):
        with jax.enable_x64(True):
            c = small(); validate_config(c)
            raw, _ = geometry_data(initial_parameters(c)[0], c)
            d = {k: jnp.asarray(v) for k, v in raw.items()}
            batch = {k: v[None] for k, v in d.items()}
            net = build_network(c, jax.random.PRNGKey(6))
            times = jnp.array([.33, 1.4, 3.1, 5.7])
            encoded = encode(net, d)
            # Compare batched directional JVP against dense single-point Hessians.
            point = jnp.array([[.72, .43], [1.43, .61]])
            fn = lambda x: fields(net, encoded, x, times[:1], d['amplitudes'], c)
            actual = eqx.filter_jit(spatial_jets)(fn, point)
            single = lambda x: fn(x[None])[0, 0]
            dx = jax.vmap(jax.jacfwd(single))(point)
            h = jax.vmap(jax.jacfwd(jax.jacfwd(single)))(point)
            np.testing.assert_allclose(actual[1][0], dx[..., 0], rtol=2e-11, atol=2e-10)
            np.testing.assert_allclose(actual[2][0], dx[..., 1], rtol=2e-11, atol=2e-10)
            np.testing.assert_allclose(actual[3][0], jnp.trace(h, axis1=-2, axis2=-1), rtol=2e-11, atol=2e-10)
            loss = make_loss_fn(c)
            full = eqx.filter_jit(eqx.filter_value_and_grad(loss, has_aux=True))
            chunks = eqx.filter_jit(make_accumulated_fn(c, loss))
            (a, pa), ga = full(net, batch, times)
            (b, pb), gb = chunks(net, batch, times)
            self.assertEqual(len(pa), 9); self.assertEqual(len(part_names(c)), 9)
            np.testing.assert_allclose(a, b, rtol=2e-12, atol=1e-12)
            np.testing.assert_allclose(pa, pb, rtol=2e-12, atol=1e-12)
            for x, y in zip(jax.tree.leaves(ga), jax.tree.leaves(gb), strict=True):
                np.testing.assert_allclose(x, y, rtol=2e-9, atol=3e-10)
            from nonlocal_pinn.training.optimizers import make_optimizer
            optimizer = make_optimizer(c)
            state = optimizer.init(eqx.filter(net, eqx.is_array))
            update = eqx.filter_jit(lambda g, s: optimizer.update(g, s, net))
            ua, sa = update(ga, state); ub, sb = update(gb, state)
            for x, y in zip(jax.tree.leaves(ua), jax.tree.leaves(ub), strict=True):
                np.testing.assert_allclose(x, y, rtol=2e-8, atol=2e-10)
            self.assertEqual(int(sa[1][0].count), 1)
            self.assertEqual(int(sb[1][0].count), 1)
            epsilon = 2e-5; index = (0, 0)
            plus = eqx.tree_at(lambda n: n.output.weight, net, net.output.weight.at[index].add(epsilon))
            minus = eqx.tree_at(lambda n: n.output.weight, net, net.output.weight.at[index].add(-epsilon))
            numeric = (full(plus, batch, times)[0][0]-full(minus, batch, times)[0][0])/(2*epsilon)
            self.assertGreater(abs(float(numeric)), 1e-8)
            np.testing.assert_allclose(ga.output.weight[index], numeric, rtol=3e-5, atol=2e-7)

    def test_ambiguous_controls_and_invalid_payload_rejected(self):
        c = small(); c.nl.loss_quadrature = 'uniform'
        with self.assertRaisesRegex(ValueError, 'explicit nl'):
            validate_config(c)
        c = small(); c.derivatives = dict(kind='gmls')
        with self.assertRaisesRegex(ValueError, 'original joint'):
            validate_config(c)
        c = small(); raw, _ = geometry_data(initial_parameters(c)[0], c)
        raw['neighbors'] = np.zeros((81, 2))
        with self.assertRaisesRegex(ValueError, 'only known-input'):
            validate_payload(raw, c)

    def test_sampled_quadrature_expectation_without_self_normalization(self):
        from itertools import combinations
        from ml_collections import ConfigDict
        from examples.channel_nsch2d.local_ad import sample_batch
        with jax.enable_x64(True):
            c=small(); c.sample.Nx=3; c.sample.Ny=3
            c.sample.ad_sampling=ConfigDict(dict(kind='uniform_interior',core_count=2,replacement=False))
            nodes=jnp.array([5,6,9,10]); measure=jnp.zeros(16).at[nodes].set(jnp.array([1.,2.,4.,9.]))
            batch=dict(transport_measure=measure[None],interior=(measure>0)[None])
            residual=jnp.arange(16,dtype=jnp.float64)**2
            loss=[]
            for subset in combinations(np.asarray(nodes).tolist(),2):
                ids=jnp.asarray([subset]); d=sample_batch(batch,ids,c)
                loss.append(jnp.mean(d['ad_core_weights']*residual[ids]**2))
            np.testing.assert_allclose(jnp.mean(jnp.stack(loss)),jnp.sum(measure*residual**2)/measure.sum(),rtol=2e-15)
            self.assertGreater(float(jnp.std(jnp.array(loss))),1.)

    def test_sampled_pde_and_separate_boundary_losses(self):
        from ml_collections import ConfigDict
        from examples.channel_nsch2d.local_ad import query_indices,sample_batch,audit_queries
        with jax.enable_x64(True):
            full_c=small(); c=small(); c.train.iterations=60
            c.sample.ad_sampling=ConfigDict(dict(kind='uniform_interior',core_count=49,replacement=False))
            validate_config(c)
            raw,_=geometry_data(initial_parameters(c)[0],c)
            batch={k:jnp.asarray(v)[None] for k,v in raw.items()}
            key=jax.random.PRNGKey(8); ids=query_indices(key,jnp.uint32(0),c)
            sampled=sample_batch(batch,ids,c)
            for name in ('domain','domain_weights','boundary','boundary_weights'):
                self.assertIs(sampled[name],batch[name])
            net=build_network(c,jax.random.PRNGKey(2)); times=jnp.array([.4,2.1])
            full=eqx.filter_jit(eqx.filter_value_and_grad(make_loss_fn(full_c),has_aux=True))
            selected=eqx.filter_jit(eqx.filter_value_and_grad(make_loss_fn(c),has_aux=True))
            (fa,pa),ga=full(net,batch,times)
            (fb,pb),gb=selected(net,sampled,times)
            np.testing.assert_allclose(pa,pb,rtol=3e-12,atol=2e-12)
            for a,b in zip(jax.tree.leaves(ga),jax.tree.leaves(gb),strict=True):
                np.testing.assert_allclose(a,b,rtol=3e-9,atol=5e-10)
            c.sample.ad_sampling.core_count=7
            short=sample_batch(batch,query_indices(key,jnp.uint32(1),c),c)
            loss=eqx.filter_jit(make_loss_fn(c))
            _,parts=loss(net,short,times)
            # All five explicit BC terms are independent of the PDE subset.
            np.testing.assert_allclose(parts[4:],pa[4:],rtol=3e-12,atol=2e-12)
            draws=np.asarray(jax.jit(jax.vmap(lambda i:query_indices(key,i,c)))(jnp.arange(60,dtype=jnp.uint32)))
            report=audit_queries(draws,key,c)
            self.assertEqual(report['actual_subsets'],60)
            self.assertFalse(np.array_equal(draws[0],draws[1]))


if __name__ == '__main__':
    unittest.main()
