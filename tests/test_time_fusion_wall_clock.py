"""Independent cold-clock fixtures and preservation of nonuniform observations."""
import unittest

import numpy as np

from scripts.time_fusion_wall_clock import recover
from scripts.trajectory_trend import trajectory_interval


class ColdClockTests(unittest.TestCase):
    def test_launch_offset_local_evaluation_and_final_teardown(self):
        # Parent starts at 1000, trainer clock at 1007. Native records at
        # 1017/1057/1087 precede the local evaluation by 3/2/2 seconds.
        result = recover([10., 50., 80.], [3., 2., 2.], launch_epoch=1000.,
            metric_created_epoch=1017.002, metric_written_epoch=1087.002,
            last_counter=80., cold_total=95., tolerance_seconds=.1)
        np.testing.assert_allclose(result['observed_evaluation_seconds'], [20.002, 59.002, 89.002])
        np.testing.assert_allclose(result['elapsed_seconds'], [20.002, 59.002, 95.])
        self.assertAlmostEqual(result['first_write_calibration_difference_seconds'], 0.)

    def test_reject_changed_timestamps_and_wrong_process_interval(self):
        evidence = dict(launch_epoch=1000., metric_created_epoch=1017.,
            metric_written_epoch=1087., last_counter=80., cold_total=95., tolerance_seconds=.1)
        for changed in ({'metric_written_epoch': 1088.}, {'cold_total': 82.},
                        {'launch_epoch': 1100.}, {'last_counter': float('nan')}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                recover([10., 50., 80.], [3., 2., 2.], **(evidence | changed))

    def test_melting_last_test_counter_is_not_an_extra_monitor_point(self):
        result = recover([10., 20., 40.], [0., 0., 0.], launch_epoch=1000.,
            metric_created_epoch=1012., metric_written_epoch=1047.,
            last_counter=45., cold_total=50., tolerance_seconds=.1)
        self.assertEqual(result['elapsed_seconds'], [12., 22., 50.])
        self.assertEqual(len(result['elapsed_seconds']), 3)

    def test_smoothing_respects_real_time_and_preserves_actual_errors(self):
        # Exponential in real time has an exact straight log trend, even with
        # an initial compile gap and subsequent nonuniform update durations.
        x = np.array([12., 130., 140., 170., 185., 190., 230., 245., 270., 310.])
        y = np.exp(-.003*x[:, None]) * np.array([1., 2., 3., 4., 5.])
        result = trajectory_interval(x, y, fraction=.6, replicates=100,
                                     seed=42, grid_points=51)
        np.testing.assert_allclose(result['smooth'], 3*np.exp(-.003*result['x']), rtol=1e-12)
        np.testing.assert_array_equal(result['raw_mean'], y.mean(axis=1))
        self.assertEqual(result['x'][0], x[0])
        self.assertEqual(result['x'][-1], x[-1])


if __name__ == '__main__':
    unittest.main()
