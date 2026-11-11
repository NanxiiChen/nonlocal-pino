"""Rolling core identity, physical input parity and explicit method receipts."""
import json
from pathlib import Path
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict

from examples.electropolish2d.configs.train_operator import get_config as production_config
from examples.electropolish2d.configs.train_operator_local import get_config as historical_local_config
from examples.electropolish2d.configs.train_operator_rolling_local import get_config as local_config
from examples.electropolish2d.configs.train_operator_rolling_nonlocal import get_config as nonlocal_config
from examples.electropolish2d.online_conditions import make_sampler
from examples.electropolish2d.operator_data import validate_config
from examples.electropolish2d.operator_local import encoded_fields, phase_driving_fields, spatial_laplacians
from examples.electropolish2d.operator_pairing import (
    control_record, input_witness, validate_config_pair, validate_pair, witness_steps)
from examples.electropolish2d.operator_training import _build_network
from examples.electropolish2d.spatial_sampling import SpatialSampler
from nonlocal_pinn.operators import build_D


class PairedOperatorTests(unittest.TestCase):
    @staticmethod
    def small_config(factory):
        c = factory()
        c.sample.Nx, c.sample.Ny = 9, 5
        c.sample.K_core, c.sample.K_pool = 4, 3
        c.operator.ic_points, c.operator.condition_batch = 3, 2
        c.sample.spatial = ConfigDict(dict(mode='rolling_pool', max_nodes=36,
            max_edges=32, workers=2, prefetch=3))
        return c

    def test_presets_preserve_current_and_historical_defaults(self):
        a, b = nonlocal_config(), local_config()
        for c in (a, b):
            validate_config(c)
            self.assertEqual((c.arch.width, c.sample.Nt, c.operator.condition_batch, c.train.iterations),
                             (96, 16, 8, 10000))
        validate_config_pair(a.to_dict(), b.to_dict())
        current = production_config().to_dict()
        actual = a.to_dict()
        del actual['operator']['method_control']
        self.assertEqual(actual['log']['checkpoint_step'], 5000)
        actual['log']['checkpoint_step'] = current['log']['checkpoint_step']
        self.assertEqual(current, actual)
        old = historical_local_config()
        self.assertNotIn('spatial', old.sample)
        self.assertEqual((old.arch.width, old.sample.Nt, old.train.iterations), (64, 4, 5000))
        for section, key in (('physical', 'L'), ('loss', 'ch_scale'), ('train', 'lr_init')):
            broken = b.copy_and_resolve_references()
            broken[section][key] *= 2
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'configurations differ'):
                validate_config_pair(a.to_dict(), broken.to_dict())

    def test_rolling_local_uses_same_cores_and_actual_prescribed_inputs_without_halos(self):
        c, d = [self.small_config(factory) for factory in (nonlocal_config, local_config)]
        xx, yy = np.meshgrid(np.linspace(0, 1, 9), np.linspace(0, .5, 5), indexing='xy')
        points = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)
        free = (points[:, 1] > 0) & (points[:, 1] < .5)
        ic, bc = np.array([5, 15, 25]), np.flatnonzero(~free)
        op = build_D(jnp.asarray(points), .18, method='sparse', kernel='cubic')
        sample, _ = make_sampler(c)
        with SpatialSampler(op, points, free, ic, bc, c) as a, SpatialSampler(op, points, free, ic, bc, d) as b:
            for step in range(8):
                left, right = a.take(step), b.take(step)
                ids, x, sub, mask, nearest = right
                self.assertIsNone(sub)
                self.assertEqual(len(mask), c.sample.K_core)
                self.assertEqual(len(ids), c.sample.K_core+len(ic)+len(bc)+6)
                actual_core = np.asarray(left[0])[:len(left[3])][np.asarray(left[3])]
                np.testing.assert_array_equal(np.asarray(ids)[:len(mask)][np.asarray(mask)], actual_core)
                self.assertTrue(np.all(free[np.asarray(ids)[-6:]]))
                self.assertTrue(np.all(np.isin(points[np.asarray(ids)[-6:], 0], [0., 1.])))
                if step in (0, 3, 7):
                    key = jax.random.fold_in(jax.random.PRNGKey(42), step)
                    l_image, l_known, l_coeff, l_accept = sample(key, left[1], left[4])
                    r_image, r_known, r_coeff, r_accept = sample(key, x, nearest)
                    np.testing.assert_array_equal(l_coeff, r_coeff)
                    np.testing.assert_array_equal(l_accept, r_accept)
                    lw = input_witness(l_image, l_known, left[1], len(left[3]), len(ic), len(bc)//2)
                    rw = input_witness(r_image, r_known, x, len(mask), len(ic), len(bc)//2)
                    for name in lw:
                        np.testing.assert_allclose(lw[name], rw[name], rtol=0, atol=3e-7)
            for name in a.trace():
                np.testing.assert_array_equal(a.trace()[name], b.trace()[name])
            self.assertEqual(b.summary()['padded_edges'], 0)
            self.assertFalse(b.summary()['evaluated_graph_halos'])
            self.assertIn('not an optimized local-AD', b.summary()['construction_scope'])

    def test_coordinate_jvp_laplacian_matches_actual_network_pointwise_hessian(self):
        c = local_config()
        c.arch.width, c.arch.heads = 8, 2
        c.arch.encoder_depth, c.arch.decoder_depth = 1, 1
        c.arch.spatial_features, c.arch.time_features = 4, 2
        shape = (1, 8, 16)
        net = _build_network(c.arch, shape, jax.random.PRNGKey(9))
        image = jnp.linspace(0, 1, np.prod(shape), dtype=jnp.float32).reshape(shape)
        encoded = net.encode(image)
        x = jnp.array([[.23, .14], [.68, .37]], dtype=jnp.float32)
        times = jnp.array([.13, .67], dtype=jnp.float32)
        actual = spatial_laplacians(net, encoded, x, times, c)

        def single(point):
            return phase_driving_fields(encoded_fields(net, encoded, point[None], times, c), c)[:, 0]

        full = jax.vmap(jax.jacfwd(jax.jacfwd(single)))(x)
        expected = jnp.trace(full, axis1=-2, axis2=-1).transpose(1, 0, 2)
        np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=3e-5)
        self.assertTrue(np.isfinite(np.asarray(actual)).all())

    def _write_pair(self, root):
        """A small completed receipt fixture, with no trained/reference labels."""
        runs = (root/'nonlocal', root/'local')
        configs = (nonlocal_config(), local_config())
        for run, c in zip(runs, configs, strict=True):
            c.train.iterations, c.sample.K_pool = 2, 1
            c.train.seed = 42  # Runtime draw identity; not a production strategy field.
            c.log.run_name = run.name
            run.mkdir()
            def write(name, value):
                (run/name).write_text(json.dumps(value))
            write('config.json', c.to_dict())
            write('method_control.json', control_record(c, run))
            write('summary.json', dict(iterations=2, final=dict(step=2),
                training_uses_positive_time_labels=False, spatial_operator=c.nl.method))
            (run/'source').mkdir()
            write('source/files.json', ['code.py'])
            (run/'source/code.py').write_text('identical source')
            (run/'model_initial.eqx').write_bytes(b'same model arrays')
            (run/'optimizer_initial.eqx').write_bytes(b'same Adam arrays')
            np.savez(run/'sampling.npz', times=np.full((2, c.sample.Nt), .2, dtype=np.float32),
                draw_ids=np.arange(2), entry_draw_ids=np.arange(2),
                core_ids=np.tile(np.arange(c.sample.K_core), (2, 1)),
                core_mask=np.ones((2, c.sample.K_core), dtype=bool),
                nonlocal_halo_counts=np.array([1000, 1001]), ic_ids=np.arange(c.operator.ic_points),
                dirichlet_ids=np.arange(2*c.sample.Nx), side_ids=np.arange(2*(c.sample.Ny-2)))
            np.save(run/'online_coefficients.npy', np.ones((2, c.operator.condition_batch, c.surface.n_modes, 3), dtype=np.float32))
            np.save(run/'online_accepted.npy', np.full(2, c.operator.condition_batch, dtype=np.int32))
            np.savez(run/'paired_inputs.npz', steps=np.array([0, 1]),
                **{key:np.ones(shape, dtype=np.float32) for key, shape in dict(
                    images=(2, c.operator.condition_batch, *c.operator.image_shape),
                    initial=(2, c.operator.condition_batch, c.operator.ic_points, 2),
                    dirichlet=(2, c.operator.condition_batch, 2*c.sample.Nx, 2),
                    ic_points=(2, c.operator.ic_points, 2), dirichlet_points=(2, 2*c.sample.Nx, 2)).items()})
        return runs

    def test_completed_pair_audit_rejects_changed_actual_draw_or_cold_state(self):
        with tempfile.TemporaryDirectory() as directory:
            runs = self._write_pair(Path(directory))
            result = validate_pair(*runs)
            self.assertEqual(result['status'], 'passed')
            path = runs[1]/'online_coefficients.npy'
            original = np.load(path)
            np.save(path, original*1.01)
            with self.assertRaisesRegex(ValueError, 'coefficients.npy values differ'):
                validate_pair(*runs)
            np.save(path, original)
            (runs[1]/'model_initial.eqx').write_bytes(b'different model')
            with self.assertRaisesRegex(ValueError, 'cold-state mismatch'):
                validate_pair(*runs)
        self.assertEqual(witness_steps(65, 32), [0, 32, 64])
        self.assertEqual(witness_steps(1, 32), [0])


if __name__ == '__main__':
    unittest.main()
