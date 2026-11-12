"""Small CPU checks for fair gradients, reference scope and failed-update recovery."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from examples.icemelting3d.configs.train_extreme_interface import get_config
from examples.icemelting3d.extreme_interface import load_reference, predictor, train, validate
from examples.icemelting3d.thin_interface import build_network, make_loss_fn, sample_batch, stencil
from examples.icemelting3d.thin_interface_operators import compact_stencil
from nonlocal_pinn.training.runtime import with_run_seed


def small_config():
    c = with_run_seed(get_config(), 42)
    c.arch.hidden_dim = 8
    c.arch.num_layers = 1
    c.arch.spatial_emb_dim = 8
    c.arch.temporal_emb_dim = 8
    c.sample.K_core = 3
    c.sample.Nt = 2
    c.experiment.boundary_points = 6
    c.experiment.evaluation_chunk = 8
    c.experiment.radial_directions = 1
    c.experiment.radial_points = 3
    return c


def reference_fields(c):
    radius = np.array([.1, .34, .45])
    rays = np.stack((.5+radius, np.full(3, .5), np.full(3, .5)), -1)[None]
    query_radius = np.array([.1, .349, .45])
    points = np.stack((.5+query_radius, np.full(3, .5), np.full(3, .5)), -1)
    times = np.linspace(0., 1., c.experiment.snapshots)
    phi = np.tanh((.35-query_radius)/(np.sqrt(2)*c.physical.eps_star))
    ray_phi = np.tanh((.35-radius)/(np.sqrt(2)*c.physical.eps_star))
    return dict(points=points, weights=np.array([.2, .3, .5]), time=times,
        local=np.tile(phi, (len(times), 1)), local_coarse=np.tile(phi, (len(times), 1)),
        rays=rays, radius=radius, local_rays=np.tile(ray_phi[None, None, :], (len(times), 1, 1)))


class ExtremeTrainingTest(unittest.TestCase):
    def test_raw_six_neighbor_loss_and_complete_gradient_preserve_pilot(self):
        c = small_config()
        operator = validate(c)
        net = build_network(c)
        net = eqx.tree_at(lambda n: (n.raw.output.weight, n.raw.output.bias), net,
            (.01*jnp.ones_like(net.raw.output.weight), -.2*jnp.ones_like(net.raw.output.bias)))
        ids, times, bc, normal = sample_batch(c, 19)
        legacy_points, legacy_sub = stencil(ids, c)
        new_points, new_sub = compact_stencil(ids, c, operator)
        np.testing.assert_array_equal(new_points, legacy_points)
        for old, new in zip(legacy_sub, new_sub, strict=True):
            np.testing.assert_array_equal(new, old)
        objective = make_loss_fn(c)
        value_grad = eqx.filter_value_and_grad(objective, has_aux=True)
        old = value_grad(net, jnp.asarray(legacy_points), legacy_sub,
                         jnp.asarray(times), jnp.asarray(bc), jnp.asarray(normal))
        new = value_grad(net, jnp.asarray(new_points), new_sub,
                         jnp.asarray(times), jnp.asarray(bc), jnp.asarray(normal))
        for x, y in zip(jax.tree.leaves(old), jax.tree.leaves(new), strict=True):
            np.testing.assert_array_equal(x, y)

    def test_reference_is_shared_across_horizons_but_not_physics(self):
        c = small_config()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            c.experiment.reference_path = str(path/'reference.npz')
            (path/'config.json').write_text(c.to_json_best_effort())
            (path/'validation.json').write_text(json.dumps(dict(status='passed',
                reference_kind='radial_latent_far_field_surrogate')))
            np.savez(path/'reference.npz', **reference_fields(c))
            other = c.copy_and_resolve_references()
            other.nl.normalization = 'moment'
            other.nl.delta = .005
            other.sample.Nx = other.sample.Ny = other.sample.Nz = 512
            loaded = load_reference(other)
            np.testing.assert_array_equal(loaded['weights'], [.2, .3, .5])
            invalid = reference_fields(c)
            invalid['points'][0, 0] = np.nan
            np.savez(path/'reference.npz', **invalid)
            with self.assertRaisesRegex(ValueError, 'Invalid weighted reference'):
                load_reference(other)
            other.physical.eps_star *= .5
            with self.assertRaisesRegex(ValueError, 'physical mismatch'):
                load_reference(other)

    def test_odd_grid_cannot_create_an_artificial_center_hessian_failure(self):
        c = small_config()
        c.sample.Nx = c.sample.Ny = c.sample.Nz = 129
        with self.assertRaises(ValueError):
            validate(c)

    def test_predictor_rejects_nonfinite_latent_even_when_tanh_is_finite(self):
        class InfiniteLatent(eqx.Module):
            def latent(self, x, t):
                return jnp.array([jnp.inf])
        c = small_config()
        evaluate = predictor(c, np.array([0., 1.]))
        with self.assertRaisesRegex(FloatingPointError, 'Nonfinite latent'):
            evaluate(InfiniteLatent(), np.array([[.2, .3, .4]]))

    def _check_rejected_update(self, overflow_norm):
        c = small_config()
        c.train.optimizer = 'adam'
        c.train.iterations = 2
        c.experiment.snapshots = 2
        c.log.eval_every = 1000
        ref = reference_fields(c)
        def bad_loss_factory(_config):
            def objective(net, *_args):
                # The finite nonzero gradient would update the model, but the
                # nonfinite primal loss requires rejection of that whole update.
                value = (1e30*jnp.sum(net.raw.output.weight) if overflow_norm else
                    jnp.sum(net.raw.output.weight)+jax.lax.stop_gradient(jnp.asarray(jnp.nan)))
                return value, jnp.stack((value, jnp.asarray(0.)))
            return objective
        with tempfile.TemporaryDirectory() as directory:
            c.log.log_dir, c.log.run_name = directory, 'failed-update'
            # 1e30 has a finite FP32 gradient but overflows its squared norm.
            # Keep this fixture independent of earlier FP64 geometry tests.
            with jax.enable_x64(False), \
                 patch('examples.icemelting3d.extreme_interface.load_reference', return_value=ref), \
                 patch('examples.icemelting3d.extreme_interface.configure_gpu_policy'), \
                 patch('examples.icemelting3d.extreme_interface.make_loss_fn', side_effect=bad_loss_factory):
                train(c)
            folder = Path(directory)/c.log.run_name
            for initial, final in [('model_initial.eqx', 'model_final.eqx'),
                                   ('optimizer_initial.eqx', 'optimizer_final.eqx')]:
                self.assertEqual((folder/initial).read_bytes(), (folder/final).read_bytes())
            summary = json.loads((folder/'summary.json').read_text())
            self.assertEqual(summary['status'], 'numerical_failure')
            self.assertEqual(summary['completed_steps'], 0)
            self.assertEqual(summary['failure']['attempted_step'], 1)
            flags = dict(zip(summary['failure']['flag_order'], summary['failure']['finite_flags'], strict=True))
            self.assertTrue(flags['gradient'])
            self.assertEqual(flags['loss'], overflow_norm)
            self.assertEqual(flags['gradient_norm'], not overflow_norm)
            with np.load(folder/'prediction.npz') as saved:
                self.assertTrue(np.isfinite(saved['phi']).all())
            with np.load(folder/'sampling.npz') as saved:
                self.assertEqual(len(saved['ids']), 1)

    def test_failed_update_preserves_initial_model_and_optimizer(self):
        self._check_rejected_update(overflow_norm=False)

    def test_finite_gradients_with_overflowing_norm_are_not_accepted_as_stable(self):
        self._check_rejected_update(overflow_norm=True)
