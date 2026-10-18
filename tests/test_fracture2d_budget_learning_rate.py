"""Rate lowering, unchanged Adam moments, budget boundaries and exact resume."""
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
from ml_collections import ConfigDict
from examples.fracture2d.learning_rate import effective_rate, make_rate_step
from examples.fracture2d.loading import run, validate
from examples.fracture2d.optimization import make_optimizer, optimizer_count
from examples.fracture2d.storage import atomic_json
from tests.test_fracture2d_fem_damage import configuration


def controlled(root):
    c = configuration(root)
    c.loading.prefix_outer_steps = c.loading.outer_steps = 2
    c.loading.minimum_sweeps = c.loading.consecutive_passes = 1
    c.loading.field_stop = True
    c.loading.advance_on_budget = False
    c.train.rate_control = ConfigDict(dict(mode='budget_finishing',
        rates=[c.train.learning_rate, c.train.learning_rate/3, c.train.learning_rate/10],
        outers_per_rate=2))
    return c


def stop_after_drop(changes, **kwargs):
    passed = kwargs['sweep'] > 2
    return dict(pass_this_sweep=passed, consecutive=int(passed), converged=passed,
                residual_is_stopping_gate=False)


class Toy(eqx.Module):
    w: jax.Array


class Replay(eqx.Module):
    def batch(self, current, key, enabled, *, old_count, fraction):
        return current


class BudgetLearningRateTests(unittest.TestCase):
    def test_dynamic_rate_matches_warm_adam_without_retracing(self):
        from examples.fracture2d.configs.train_history_fem_cold import get_config
        c = get_config()
        keys = ('decay_steps', 'decay_rate', 'grad_clip', 'b1', 'b2', 'eps',
                'eps_root', 'shampoo_beta', 'correct_bias', 'precondition_frequency',
                'max_precond_dim', 'precondition_1d', 'weight_decay')
        def optimizer(rate):
            return make_optimizer(kind='adam', learning_rate=rate, minimum_rate=rate,
                                  **{k:c.train[k] for k in keys})
        net = Toy(jnp.array([.1, -.7, 2.]))
        original, _ = optimizer(c.train.learning_rate)
        unit, _ = optimizer(1.)
        state = original.init(net)
        for k in range(13):
            _, state = original.update(Toy(jnp.array([.2+k*.01, -.3, .8])), state, net)
        traces = []
        def objective(model, batch):
            traces.append(1)
            return jnp.mean((model.w-batch['H'])**2)
        step = make_rate_step(objective, unit, old_count=1, fraction=.25)
        current = dict(H=jnp.array([.5, .4, .3]))
        gradient = Toy(2*(net.w-current['H'])/3)
        for rate in c.train.rate_control.rates:
            target, schedule = optimizer(rate)
            updates, expected_state = target.update(gradient, state, net)
            expected = eqx.apply_updates(net, updates)
            actual, actual_state, _ = step(net, state, current, Replay(),
                jax.random.PRNGKey(0), jnp.asarray(False), effective_rate(rate))
            np.testing.assert_array_equal(effective_rate(rate),
                jnp.asarray(schedule(jnp.asarray(203904, jnp.int32)), jnp.float64))
            np.testing.assert_allclose(actual.w, expected.w, rtol=1e-14, atol=1e-16)
            for a, b in zip(jax.tree.leaves(actual_state), jax.tree.leaves(expected_state), strict=True):
                np.testing.assert_allclose(a, b, rtol=1e-14, atol=1e-16)
            self.assertEqual(optimizer_count(actual_state, kind='adam'), 14)
        self.assertEqual(len(traces), 1)

    def test_lowering_persists_across_loads_and_exact_endpoint_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = controlled(root)
            validate(c)
            with contextlib.redirect_stdout(io.StringIO()), patch(
                    'examples.fracture2d.loading.field_stop', side_effect=stop_after_drop):
                run(c, damage_solver='pinn')
            base = Path(c.output.root)
            rows = json.loads((base/'result.json').read_text())['curve']
            self.assertEqual([r['sweep'] for r in rows], [3, 3])
            self.assertEqual([r['learning_rate_index'] for r in rows], [1, 2])
            self.assertEqual([r['optimizer_count'] for r in rows], [9, 15])
            for k in ('q', 'd', 'H'):
                np.testing.assert_array_equal(np.load(base/'state_0000/fields.npz')[k], 0.)
            c.output.root = str(root/'interrupted')
            def interrupt(path, data):
                atomic_json(path, data)
                if path.name == 'result.json' and 'curve' in data and len(data['curve']) == 1:
                    raise InterruptedError('endpoint')
            with contextlib.redirect_stdout(io.StringIO()), patch(
                    'examples.fracture2d.loading.field_stop', side_effect=stop_after_drop), patch(
                    'examples.fracture2d.loading.atomic_json', side_effect=interrupt):
                with self.assertRaises(InterruptedError):
                    run(c, damage_solver='pinn')
            c.output.resume = True
            with contextlib.redirect_stdout(io.StringIO()), patch(
                    'examples.fracture2d.loading.field_stop', side_effect=stop_after_drop):
                run(c, damage_solver='pinn')
            actual = Path(c.output.root)/'state_0002'
            for name in ('network.eqx', 'optimizer.eqx', 'rate_control.json'):
                self.assertEqual((base/'state_0002'/name).read_bytes(), (actual/name).read_bytes())
            with np.load(base/'state_0002/fields.npz') as a, np.load(actual/'fields.npz') as b:
                for k in a.files:
                    np.testing.assert_array_equal(a[k], b[k])

    def test_final_block_stops_and_cannot_be_skipped_on_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = controlled(Path(tmp))
            failed = dict(pass_this_sweep=False, consecutive=0, converged=False,
                          residual_is_stopping_gate=False)
            with contextlib.redirect_stdout(io.StringIO()), patch(
                    'examples.fracture2d.loading.field_stop', return_value=failed):
                with self.assertRaisesRegex(RuntimeError, 'exhausted outer budget'):
                    run(c, damage_solver='pinn')
            root = Path(c.output.root)
            row = json.loads((root/'state_0001/result.json').read_text())
            self.assertEqual(row['sweep'], 6)
            self.assertEqual(row['learning_rate_index'], 2)
            self.assertEqual(row['optimizer_count'], 15)
            self.assertFalse((root/'state_0002').exists())
            c.output.resume = True
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, 'cannot resume past'):
                    run(c, damage_solver='pinn')
            c.train.rate_control.rates = [1e-3, 1e-3]
            with self.assertRaisesRegex(ValueError, 'descending positive rates'):
                validate(c)


if __name__ == '__main__':
    unittest.main()
