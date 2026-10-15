"""Check nonuniform-volume and localized damage reporting independently."""

import unittest

import numpy as np

from examples.fracture2d._legacy.split_pinn_report import damage_metrics


class SplitDamageReportTest(unittest.TestCase):
    def test_weighted_and_local_errors_do_not_hide_peak_error(self):
        actual = np.asarray([1., 2.2, 2.7])
        reference = np.asarray([1., 2., 3.])
        coordinates = np.asarray([[0.5, 0.5], [0.51, 0.5], [0.9, 0.5]])
        volumes = np.asarray([1., 2., 3.])
        report = damage_metrics(actual, reference, coordinates, volumes,
                                tip=np.asarray([0.5, 0.5]), tip_radius=0.024)
        self.assertAlmostEqual(report["damage_rmse"], np.sqrt(0.13 / 3))
        self.assertAlmostEqual(report["damage_relative_l2"], np.sqrt(0.13 / 14))
        self.assertAlmostEqual(report["damage_volume_rmse"], np.sqrt(0.35 / 6))
        self.assertAlmostEqual(report["damage_volume_relative_l2"], np.sqrt(0.35 / 36))
        self.assertAlmostEqual(report["tip_damage_rmse"], np.sqrt(0.04 / 2))
        self.assertAlmostEqual(report["tip_damage_max_error"], 0.2)
        self.assertAlmostEqual(report["damage_max_error"], 0.3)
        self.assertEqual(report["tip_node_count"], 2)
        self.assertEqual(report["peak_location_distance"], 0.)
        with self.assertRaisesRegex(ValueError, "nonzero reference"):
            damage_metrics(actual, np.zeros(3), coordinates, volumes,
                           tip=np.asarray([0.5, 0.5]), tip_radius=0.024)
        with self.assertRaisesRegex(ValueError, "no material nodes"):
            damage_metrics(actual, reference, coordinates, volumes,
                           tip=np.asarray([0., 0.]), tip_radius=0.024)


if __name__ == "__main__":
    unittest.main()
