"""Regression for exact uniform-cloud FEM controls and mesh-size ordering."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("dolfin"), "requires fenicsenv")
class LocalMeshConfigTest(unittest.TestCase):
    def setUp(self):
        from examples.fracture2d import solver
        from examples.fracture2d.configs.solver_base import get_config

        self.solver = solver
        self.config = get_config()

    def test_uniform_b_cloud_target_sizes_are_valid(self):
        self.config.mesh.lc_near = 0.004
        self.config.mesh.lc_far = 0.004
        self.solver._validate_config(self.config)

    def test_inverted_near_far_sizes_are_rejected(self):
        self.config.mesh.lc_near = 0.008
        self.config.mesh.lc_far = 0.004
        with self.assertRaisesRegex(ValueError, "lc_near must not exceed"):
            self.solver._validate_config(self.config)


if __name__ == "__main__":
    unittest.main()
