"""Common-target identity must ignore only the declared derivative selector."""
from copy import deepcopy
import unittest

from docs.tasks.electro_spatial_evaluate import validate_reference_identity


class ReferenceIdentityTest(unittest.TestCase):
    def setUp(self):
        self.source = dict(physical={'D': 1.}, nondim={'T_c': 5.},
                           surface={'kind': 'rough'},
                           nl={'method': 'sparse', 'delta': .02, 'kernel': 'cubic'})

    def test_both_declared_derivatives_share_target(self):
        for method in ('sparse', 'local_ad'):
            config = deepcopy(self.source)
            config['nl']['method'] = method
            validate_reference_identity(config, self.source)

    def test_physics_or_horizon_changes_are_rejected(self):
        for key, name, value in (('physical', 'D', 2.), ('nondim', 'T_c', 4.),
                                 ('surface', 'kind', 'smooth'), ('nl', 'delta', .03)):
            config = deepcopy(self.source)
            config[key][name] = value
            with self.assertRaises(ValueError):
                validate_reference_identity(config, self.source)

    def test_unknown_method_is_rejected(self):
        config = deepcopy(self.source)
        config['nl']['method'] = 'spectral'
        with self.assertRaises(ValueError):
            validate_reference_identity(config, self.source)


if __name__ == '__main__':
    unittest.main()
