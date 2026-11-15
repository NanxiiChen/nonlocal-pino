"""The recovery must preserve refinement thresholds and the full cohort."""
import copy
from types import SimpleNamespace
import unittest

import numpy as np

from docs.tasks.spinodal_reference_recovery import checks, passes, plan, spatial_failure_only


class ReferenceRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.c = SimpleNamespace(reference=SimpleNamespace(mesh_sizes=(.035, .025),
            phase_tolerance=.01, chemical_tolerance=.05))
        self.prior = dict(status='failed', specification={'physical': 'same'},
            indices=[10, 11, 12], cases=[dict(index=10), dict(index=11)],
            refinement_thresholds=dict(phase=.01, chemical_positive=.05,
                mass=1e-8, energy_increase=1e-8))

    def test_refinement_plan_preserves_cohort_and_thresholds(self):
        plan(self.prior, {'physical': 'same'}, [10, 11, 12], [.018], self.c)
        for levels in ([], [.025], [.018, .02], [np.nan], [-.01]):
            with self.assertRaises(ValueError):
                plan(self.prior, {'physical': 'same'}, [10, 11, 12], levels, self.c)
        changed = copy.deepcopy(self.prior)
        changed['refinement_thresholds']['phase'] = .02
        with self.assertRaises(ValueError):
            plan(changed, {'physical': 'same'}, [10, 11, 12], [.018], self.c)
        changed = copy.deepcopy(self.prior)
        changed['cases'][1]['index'] = 12
        with self.assertRaises(ValueError):
            plan(changed, {'physical': 'same'}, [10, 11, 12], [.018], self.c)

    def test_only_spatial_failures_can_trigger_mesh_refinement(self):
        data = dict(points=np.array([[0., 0.], [1., 1.]]), volumes=np.array([1., 2.]),
            phi=np.ones((3, 2)), mu=np.full((3, 2), 2.))
        fine = dict(points=data['points'], phi=data['phi'], mu=data['mu'],
            mass=np.ones(3), energy=np.array([3., 2., 1.]))
        coarse = copy.deepcopy(fine)
        coarse['phi'][-1] *= 1.013
        values = checks(data, [coarse, fine, fine])
        self.assertFalse(passes(values, self.c))
        self.assertTrue(spatial_failure_only(values, self.c))
        self.assertAlmostEqual(values['space_phase'], .013)
        for key, value in [('time_phase', .011), ('time_chemical_positive', .06),
                           ('mass_drift', 1e-7), ('energy_increase', 1e-7)]:
            wrong = dict(values)
            wrong[key] = value
            self.assertFalse(spatial_failure_only(wrong, self.c))
        bad = copy.deepcopy(fine)
        bad['phi'][0, 0] = np.nan
        with self.assertRaises(ValueError):
            checks(data, [coarse, fine, bad])


if __name__ == '__main__':
    unittest.main()
