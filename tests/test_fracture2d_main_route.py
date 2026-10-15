"""Numerical regressions for the consolidated local-mechanics damage route."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import equinox as eqx
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
from scipy.sparse import diags, eye, save_npz
from pinn.operators import ExplicitVolumeSparseD
from examples.fracture2d.configs.train_base import get_config
from examples.fracture2d.loading import run, validate
from examples.fracture2d.losses import residual_mean
from examples.fracture2d.mesh import geometry_identity, load_geometry
from examples.fracture2d.storage import atomic_json
from examples.fracture2d.train import DamagePINN
from tests.test_fracture2d_local_mechanics import model


def fixture(root):
    """Manufactured affine patch; no production FEM-validation claim."""
    c = get_config()
    c.runtime.platform = 'cpu'
    c.output.root = str(root/'run')
    c.output.tensorboard = False
    c.output.checkpoint_sweeps = [1, 3]
    c.output.cpu_threads = 2
    c.mesh.prepared_dir = str(root/'mesh')
    c.nl.delta_d = 1.5
    c.arch.hidden_dim, c.arch.num_layers = 8, 2
    c.train.inner_steps = 2
    c.loading.u_final = .0002
    c.loading.prefix_outer_steps, c.loading.outer_steps = 3, 3
    c.loading.field_stop = False
    c.sample.core_count, c.sample.pool_size = 2, 4
    m = model()
    c.data.cloud = str(root/'cloud.npz')
    np.savez(c.data.cloud, coordinates=m.coordinates, triangles=m.triangles,
        volumes=m.volumes, bottom=m.coordinates[:, 1] == 0,
        top=m.coordinates[:, 1] == 1, anchor_particle=np.int64(0))
    edges = np.array([(i, j) for i in range(4) for j in range(4) if i != j], np.int64)
    op = ExplicitVolumeSparseD(m.coordinates, m.volumes, edges, c.nl.delta_d, 'cubic')
    mesh = Path(c.mesh.prepared_dir)
    mesh.mkdir()
    np.savez(mesh/'mesh.npz', coordinates=m.coordinates, triangles=m.triangles,
        n_material=np.int64(4), n_material_cells=np.int64(2), volumes=m.volumes,
        constrained=m.constrained, load_direction=m.load_direction, damage_edges=edges)
    save_npz(mesh/'damage_operator.npz', op.to_scipy())
    identity = dict(config=geometry_identity(c), mesh_path=str(mesh/'mesh.npz'),
                    operator_path=str(mesh/'damage_operator.npz'))
    atomic_json(mesh/'identity.json', identity)
    atomic_json(mesh/'fenics_parity.json', {'pass': True})
    c.data.basis_path = str(root/'basis.npz')
    save_npz(c.data.basis_path, eye(4, format='csr', dtype=np.float64))
    return c


class MainRouteTests(unittest.TestCase):
    def test_tip_sampling_probabilities_are_detached_and_importance_corrected(self):
        from examples.fracture2d.sampling import crack_sampling_probabilities
        points = jnp.array([[.5, .5], [.52, .5], [.75, .5], [.75, .7], [1., 1.]])
        settings = dict(start=[.5, .5], end=[1., .5], width=.024, boost=9., power=2.,
                        tip_width=.024, tip_boost=0., tip_power=2.)
        baseline = crack_sampling_probabilities(points, **settings)
        distance2 = jnp.array([0., 0., 0., .2**2, .5**2])
        raw = 1+9/(1+distance2/.024**2)
        np.testing.assert_allclose(baseline, raw/raw.sum(), rtol=1e-14)
        settings.update(boost=18., tip_boost=36.)
        probability = crack_sampling_probabilities(points, **settings)
        self.assertTrue(np.all(np.asarray(probability) > 0))
        self.assertAlmostEqual(float(probability.sum()), 1.)
        self.assertGreater(float(probability[0]), float(baseline[0]))
        np.testing.assert_array_equal(jax.jacrev(
            lambda x: crack_sampling_probabilities(x, **settings))(points), 0.)
        residual = jnp.array([.2, -.1, .6, -.9, .5])
        H = jnp.ones(5)
        sampled = [float(residual_mean(residual[i:i+1], H, probability[i:i+1], jnp.ones(1),
            population_size=5, norm='l1', floor=1e-15)) for i in range(5)]
        self.assertAlmostEqual(float(probability@jnp.asarray(sampled)), float(abs(residual).mean()/2), places=14)
        with self.assertRaises(ValueError):
            crack_sampling_probabilities(points, **dict(settings, tip_width=0.))

    def test_finite_pool_uses_exact_maximum_and_complete_operator_rows(self):
        from examples.fracture2d.sampling import build_finite_sampling_pool, make_sampling_rows
        D = diags([np.ones(4), -2*np.ones(5), np.ones(4)], [-1, 0, 1], format='csr')
        rows = make_sampling_rows(D, row_width=3)
        cores = np.array([[0, 0], [1, 4], [1, 2]], np.int32)
        pool, counts = build_finite_sampling_pool(rows, cores, np.full(5, .2))
        np.testing.assert_array_equal(counts, [2, 5, 4])
        self.assertEqual(pool['global_ids'].shape, (3, 5))
        values = np.arange(5, dtype=np.float64)**2
        for slot, core in enumerate(cores):
            image = values[pool['global_ids'][slot]]
            actual = np.sum(pool['operator_values'][slot]*image[pool['operator_columns'][slot]], axis=1)
            np.testing.assert_array_equal(actual, (D@values)[core])
            np.testing.assert_array_equal(pool['global_ids'][slot][pool['core_local'][slot]], core)

    def test_fixed_pool_cycles_without_rebuilding_or_resetting_optimizer(self):
        with tempfile.TemporaryDirectory() as temp:
            c = fixture(Path(temp))
            c.sample.mode = 'random'
            m, op, _ = load_geometry(c)
            pinn = DamagePINN(c, m, op)
            pool, state = pinn.pool, pinn.state
            context = pinn.context(jnp.ones(4), jnp.ones(4))
            with patch('examples.fracture2d.train.build_finite_sampling_pool',
                       side_effect=AssertionError('pool rebuilt')):
                for interval in (1, 10):
                    c.sample.resample_every = interval
                    for count in range(3*c.sample.pool_size*interval+129):
                        pinn.count = count
                        batch = pinn.sampled_context(context)
                        self.assertEqual(int(batch['slot']), (count//interval) % c.sample.pool_size)
                        self.assertIs(batch['pool'], pool)
                        self.assertIs(pinn.state, state)

    def test_loss_importance_and_detached_history_weights(self):
        H, r = jnp.array([1., 2., 3.]), jnp.array([.2, -.4, .8])
        p, w = jnp.full(3, 1/3), jnp.ones(3)
        rho = np.asarray(r)/(2*np.sqrt(np.mean(np.asarray(H)**2)))
        for norm, expected in [('l1', np.mean(abs(rho))), ('l2', np.mean(rho**2)),
                               ('l16', np.mean(rho**16)**(2/16))]:
            fun = lambda h, weights: residual_mean(r, h, p, weights,
                population_size=3, norm=norm, floor=1e-15)
            self.assertAlmostEqual(float(fun(H, w)), expected, places=14)
            for gradient in jax.grad(fun, argnums=(0, 1))(H, w):
                np.testing.assert_array_equal(gradient, 0.)
        # Exact expectation of a nonuniform with-replacement L1 moment.
        probabilities = np.array([.1, .3, .6])
        values = [float(residual_mean(r[i:i+1], H, jnp.array([probabilities[i]]),
            jnp.ones(1), population_size=3, norm='l1', floor=1e-15)) for i in range(3)]
        self.assertAlmostEqual(float(probabilities @ values), np.mean(abs(rho)), places=14)

    def test_full_and_exact_halo_gradients_match(self):
        with tempfile.TemporaryDirectory() as temp:
            c = fixture(Path(temp))
            m, op, _ = load_geometry(c)
            full = DamagePINN(c, m, op)
            H = jnp.array([1., 2., 3., 4.])
            context = full.context(H, jnp.ones(4))
            loss, grad = eqx.filter_value_and_grad(full.value)(full.net, context, jnp.array(.0001))
            from examples.fracture2d.sampling import make_sampling_rows, sample_core_halo
            rows = make_sampling_rows(op.to_scipy(), row_width=full.dc.shape[1])
            ids = np.array([3, 0, 2, 1])
            packed = sample_core_halo(rows, ids, node_capacity=4)
            halo = dict(context, core_H=H[ids], core_local=jnp.asarray(packed['core_local']),
                operator_columns=jnp.asarray(packed['columns']), operator_values=jnp.asarray(packed['values']))
            other, other_grad = eqx.filter_value_and_grad(full.value)(full.net, halo, jnp.array(.0001))
            self.assertAlmostEqual(float(loss), float(other), places=13)
            for a, b in zip(jax.tree.leaves(grad), jax.tree.leaves(other_grad)):
                np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-13)
            from examples.fracture2d.sampling import build_finite_sampling_pool, make_pool_objective
            pool, _ = build_finite_sampling_pool(rows, ids[None, :], np.full(4, .25))
            pooled = make_pool_objective(full.value, basis_columns=full.bc, basis_values=full.bv)
            batch = dict(H=H, features=context['features'], weights=jnp.ones(4),
                         pool=jax.device_put(pool), slot=jnp.array(0, jnp.int32))
            pooled_loss, pooled_grad = eqx.filter_value_and_grad(pooled)(full.net, batch, jnp.array(.0001))
            self.assertAlmostEqual(float(loss), float(pooled_loss), places=13)
            for a, b in zip(jax.tree.leaves(grad), jax.tree.leaves(pooled_grad)):
                np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-13)

    def test_direct_stagger_and_pinn_resumed_sampling_are_consistent(self):
        for solver, sampling, interval in [('direct', 'full', 1), ('pinn', 'full', 1),
                                           ('pinn', 'random', 1), ('pinn', 'random', 10)]:
            with self.subTest(solver=solver, sampling=sampling), tempfile.TemporaryDirectory() as temp:
                c = fixture(Path(temp))
                c.sample.mode = sampling
                c.sample.resample_every = interval
                if interval == 10:
                    c.sample.boost, c.sample.tip_boost = 18., 36.
                with contextlib.redirect_stdout(io.StringIO()):
                    result = run(c, damage_solver=solver)
                report = json.loads(result.read_text())
                self.assertTrue(report['complete'])
                self.assertEqual(len(report['curve']), 2)
                for row in report['curve']:
                    self.assertEqual(row['mechanics_calls'], row['sweep']+1)
                    self.assertTrue(row['q_d_synchronized'])
                with np.load(Path(c.output.root)/'state_0000/fields.npz') as fields:
                    for name in ('q', 'd', 'H'):
                        np.testing.assert_array_equal(fields[name], 0.)
                with np.load(Path(c.output.root)/'state_0002/fields.npz') as fields:
                    final = {key: fields[key].copy() for key in ('q', 'd', 'H')}
                # Independent host equilibrium of the saved paired q/d endpoint.
                m, _, _ = load_geometry(c)
                force = m.matrix(final['d']) @ final['q']
                np.testing.assert_allclose(force[m.free], 0., atol=1e-8)
                self.assertAlmostEqual(float(force @ m.load_direction), report['curve'][-1]['reaction'], places=9)
                c.output.root = str(Path(temp)/'resumed')
                def interrupt_after_endpoint(path, data):
                    atomic_json(path, data)
                    if path.name == 'result.json' and 'curve' in data and len(data['curve']) == 1:
                        raise InterruptedError('test stop after first saved endpoint')
                with contextlib.redirect_stdout(io.StringIO()), patch(
                        'examples.fracture2d.loading.atomic_json', side_effect=interrupt_after_endpoint):
                    with self.assertRaises(InterruptedError):
                        run(c, damage_solver=solver)
                c.output.resume = True
                with contextlib.redirect_stdout(io.StringIO()):
                    run(c, damage_solver=solver)
                with np.load(Path(c.output.root)/'state_0002/fields.npz') as fields:
                    for name in final:
                        np.testing.assert_array_equal(fields[name], final[name])
                if solver == 'pinn':
                    resumed = json.loads((Path(c.output.root)/'result.json').read_text())
                    self.assertEqual(resumed['curve'][-1]['optimizer_count'], 12)

    def test_tensorboard_endpoint_fields_and_matched_reference(self):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            c = fixture(root)
            with contextlib.redirect_stdout(io.StringIO()):
                run(c, damage_solver='direct')
            c.output.reference_root = c.output.root
            c.output.root = str(root/'pinn_monitor')
            c.output.tensorboard = True
            c.output.event_dir = str(root/'pinn_baseline')
            c.output.figure_dpi = 40
            with contextlib.redirect_stdout(io.StringIO()):
                run(c, damage_solver='pinn')
            accumulator = EventAccumulator(c.output.event_dir)
            accumulator.Reload()
            self.assertEqual(set(accumulator.Tags()['images']), {
                'train/damage_field', 'train/ux_field', 'train/uy_field',
                'train/history_field', 'train/load_reaction_curve'})
            self.assertEqual(len(accumulator.Scalars('eval/direct_nonlocal/rel_L2_d')), 2)
            self.assertEqual(len(list((Path(c.output.root)/'figures').glob('*.png'))), 10)

    def test_optional_norm_head_optimizer_and_invalid_protocol(self):
        with tempfile.TemporaryDirectory() as temp:
            c = fixture(Path(temp))
            m, op, _ = load_geometry(c)
            for norm, head, optimizer in [('l2', 'exp_square', 'adam'),
                    ('l16', 'sigmoid', 'lbfgs'), ('l1', 'identity', 'soap')]:
                c.loss.norm, c.arch.output_head, c.train.optimizer = norm, head, optimizer
                c.train.lbfgs.maxiter = 2
                p = DamagePINN(c, m, op)
                initial = p.image(p.net, p.context(jnp.ones(4), jnp.ones(4)))
                np.testing.assert_allclose(initial, c.arch.initial_damage, atol=1e-15)
                field, value = p.solve(jnp.array([1., 2., 3., 4.]), jnp.zeros(4), .0001)
                self.assertTrue(np.isfinite(np.asarray(field)).all())
                self.assertTrue(np.isfinite(value))
            c.train.optimizer, c.sample.mode = 'lbfgs', 'random'
            with self.assertRaisesRegex(ValueError, 'fixed full objective'):
                validate(c)
            c.train.optimizer, c.sample.mode = 'adam', 'full'
            c.loading.u_final = .00025
            with self.assertRaisesRegex(ValueError, 'integer multiple'):
                validate(c)


if __name__ == '__main__':
    unittest.main()
