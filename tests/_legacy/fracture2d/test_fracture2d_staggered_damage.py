"""Pipeline parity, exact P1 norms and inexact-inner stopping semantics."""
import unittest
import jax.numpy as jnp
import numpy as np

from examples.fracture2d._legacy.staggered_damage import (
    case_configs, mechanics_first_sweep, p1_rms, field_changes, field_stop)


class StaggeredDamageTest(unittest.TestCase):
    def test_case_dictionary_list_survives_config_json_roundtrip(self):
        import json
        from examples.fracture2d._legacy.configs.staggered_damage_trial import get_config
        cases = case_configs(json.loads(json.dumps(get_config().to_dict()))['cases'])
        self.assertEqual([case.name for case in cases],
                         ['direct_half', 'direct_full', 'pinn_full', 'pinn_half'])
        with self.assertRaises(ValueError):
            case_configs([dict(name='bad', solver='direct', omega=0., max_sweeps=1)])

    def test_pipeline_matches_old_map_and_calls_mechanics_once(self):
        calls = []
        def mechanics(d, load, q):
            calls.append(1)
            return 2+d, 3., {'converged': True}
        def driver(q): return .1*q
        def damage(h, d, count): return h/(1+h), count+1, {}
        for omega in [.5, 1.]:
            old_d = jnp.array([.1, .2]); parent = jnp.array([.3, .1])
            old_q = 2+old_d
            d, q, carry = old_d, jnp.zeros(2), 0
            calls.clear()
            for iteration in range(5):
                fields, carry = mechanics_first_sweep(q, d, parent, 1., carry,
                    mechanics=mechanics, driver=driver, damage=damage, omega=omega)
                h = jnp.maximum(parent, driver(np.asarray(old_q)))
                image = h/(1+h)
                old_d = old_d+omega*(image-old_d)
                old_q = 2+old_d
                np.testing.assert_allclose(fields['d'], old_d, rtol=1e-6)
                d, q = fields['d'], fields['q']
            self.assertEqual(len(calls), 5)
            self.assertEqual(carry, 5)
            np.testing.assert_allclose(q, 2+fields['d_previous'])

    def test_exact_consistent_mass_norm_for_linear_field(self):
        tri = np.array([[0, 1, 2]]); areas = np.array([.5])
        # x on the unit right triangle: integral(x^2)=1/12, RMS^2=1/6.
        self.assertAlmostEqual(p1_rms(np.array([0., 1., 0.]), tri, areas)**2, 1/6)
        self.assertAlmostEqual(p1_rms(np.full(3, 2.), tri, areas), 2.)
        self.assertAlmostEqual(p1_rms(np.tile([3., 4.], (3, 1)), tri, areas), 5.)
        q = np.tile([0., .005], 3)
        changes = field_changes(q, np.zeros(6), np.zeros(3), np.zeros(3), tri, areas, load=.005)
        self.assertEqual(changes['displacement_rms_over_load'], 1.)

    def test_field_stability_does_not_certify_equation_balance(self):
        changes = dict(displacement_rms_over_load=0., damage_rms=0.)
        count = 0
        for sweep in range(1, 6):
            stop = field_stop(changes, displacement_limit=1e-4, damage_limit=1e-5,
                sweep=sweep, minimum_sweeps=3, consecutive_passes=3, previous_consecutive=count)
            count = stop['consecutive']
        self.assertTrue(stop['converged'])
        self.assertFalse(stop['residual_is_stopping_gate'])
        # An unchanged network d=0 at H=1 still has AT2 residual -2.
        self.assertEqual(2*(0.-1.), -2.)
        with self.assertRaises(ValueError):
            field_stop(dict(changes, damage_rms=np.nan), displacement_limit=1e-4,
                damage_limit=1e-5, sweep=5, minimum_sweeps=3,
                consecutive_passes=3, previous_consecutive=2)


if __name__ == '__main__':
    unittest.main()
