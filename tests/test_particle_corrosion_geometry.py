"""Focused geometry checks without GPU, PDE solves or neural training."""
import copy
import json
import pickle
import unittest
from unittest.mock import patch

import numpy as np

from examples.particle_corrosion2d.configs.geometry_base import get_config
from examples.particle_corrosion2d import geometry as geometry_module
from examples.particle_corrosion2d.geometry import boundary, contains, sample_geometry, validate_geometry


class ParticleGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = get_config()
        cls.record = sample_geometry(cls.config, np.random.default_rng(61027))

    def test_sampled_clearances_normalization_and_analytic_membership(self):
        c, record = self.config, self.record
        metrics = validate_geometry(json.loads(json.dumps(record)), c)
        self.assertGreaterEqual(metrics['pair_clearance_lower_bound'], c.particles.particle_clearance)
        self.assertGreaterEqual(metrics['wall_clearance_lower_bound'], c.particles.wall_clearance)
        self.assertGreaterEqual(metrics['pit_clearance_lower_bound'], c.particles.pit_clearance)
        self.assertGreaterEqual(metrics['minimum_sampled_curvature_radius'], c.particles.minimum_curvature_radius)
        self.assertGreaterEqual(metrics['near_pit_count'], c.particles.minimum_near_pit_count)
        centers = np.array([p['center'] for p in record['particles']])
        self.assertFalse(contains(centers, record, c).any())
        self.assertTrue(contains(np.array([[.5, .01], [0., 0.], [1., .6]]), record, c).all())
        self.assertFalse(contains(np.array([[-.1, .2], [1.1, .2]]), record, c).any())
        for particle in record['particles']:
            points, normals, _ = boundary(particle, np.linspace(0, 2*np.pi, 31, endpoint=False))
            self.assertTrue(contains(points+1e-5*normals, record, c).all())
            self.assertFalse(contains(points-1e-5*normals, record, c).any())

    def test_boundary_normals_and_curvature_match_finite_differences(self):
        particle = self.record['particles'][0]
        theta = np.array([.13, 1.18, 2.31, 4.71, 5.87])
        points, normal, curvature = boundary(particle, theta)
        step = 3e-4
        plus = boundary(particle, theta+step)[0]
        minus = boundary(particle, theta-step)[0]
        plus2 = boundary(particle, theta+2*step)[0]
        minus2 = boundary(particle, theta-2*step)[0]
        tangent = (-plus2+8*plus-8*minus+minus2)/(12*step)
        second = (-plus2+16*plus-30*points+16*minus-minus2)/(12*step**2)
        np.testing.assert_allclose(np.sum(normal*tangent, axis=-1), 0, atol=2e-9)
        np.testing.assert_allclose(np.linalg.norm(normal, axis=-1), 1, atol=1e-14)
        numerical_curvature = (tangent[:, 0]*second[:, 1]-tangent[:, 1]*second[:, 0]) / np.linalg.norm(tangent, axis=1)**3
        np.testing.assert_allclose(curvature, numerical_curvature, atol=3e-5, rtol=3e-6)
        self.assertTrue((np.sum(normal*(points-particle['center']), axis=-1) > 0).all())

    def test_invalid_config_overlap_and_area_fail_loudly(self):
        config = get_config()
        config.particles.aspect_range = (0., 1.)
        with self.assertRaisesRegex(ValueError, 'aspect_range'):
            sample_geometry(config, np.random.default_rng(3))
        overlap = copy.deepcopy(self.record)
        overlap['particles'][1] = copy.deepcopy(overlap['particles'][0])
        with self.assertRaisesRegex(ValueError, 'clearance'):
            validate_geometry(overlap, self.config)
        malformed = copy.deepcopy(self.record)
        malformed['particles'][0]['radial_scale'] *= 1.00001
        with self.assertRaisesRegex(ValueError, 'area normalization'):
            validate_geometry(malformed, self.config)
        malformed = copy.deepcopy(self.record)
        malformed['particles'][0]['cosine'][0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'radial particle'):
            validate_geometry(malformed, self.config)

    def test_online_draw251_rejects_coarse_only_curvature_candidate(self):
        # Production regression: the old 512-node gate accepted radius
        # .015003903, but 1024/8192 nodes give .014999393/.014999265 < .015.
        words = [1607243318, 271828, 251]
        first = np.random.default_rng(np.random.SeedSequence(words))
        diagnostic = {}
        accepted = sample_geometry(self.config, first, diagnostics=diagnostic)
        self.assertEqual(diagnostic['outcome'], 'accepted')
        self.assertGreaterEqual(diagnostic['rejections']['refined_clearance_or_curvature'], 1)
        metrics = validate_geometry(accepted, self.config)
        self.assertGreaterEqual(metrics['minimum_sampled_curvature_radius'], .015)
        for particle in accepted['particles']:
            theta = np.arange(8192)*2*np.pi/8192
            self.assertGreaterEqual(1/np.max(np.abs(boundary(particle, theta)[2])), .015)
        second = np.random.default_rng(np.random.SeedSequence(words))
        replay = sample_geometry(self.config, second)
        self.assertEqual(accepted, replay)
        np.testing.assert_array_equal(first.random(16), second.random(16))

    def test_budget_exhaustion_has_auditable_typed_diagnostics(self):
        config = copy.deepcopy(self.config)
        config.particles.max_geometry_attempts = 2
        config.particles.max_particle_attempts = 3
        diagnostic = {}
        with patch.object(geometry_module, '_draw_particle', return_value=None):
            with self.assertRaises(geometry_module.GeometrySamplingExhausted) as failure:
                sample_geometry(config, np.random.default_rng(9), diagnostics=diagnostic)
        self.assertIsInstance(failure.exception, RuntimeError)
        self.assertEqual(diagnostic['outcome'], 'budget_exhausted')
        self.assertEqual(diagnostic['layout_attempts'], 2)
        self.assertEqual(diagnostic['particle_proposals'], 6)
        self.assertEqual(diagnostic['rejections']['coefficient_bounds'], 6)
        self.assertEqual(diagnostic['rejections']['incomplete_layout'], 2)
        self.assertEqual(pickle.loads(pickle.dumps(failure.exception)).diagnostics, diagnostic)
        diagnostic['rejections']['coefficient_bounds'] = -1
        self.assertEqual(failure.exception.diagnostics['rejections']['coefficient_bounds'], 6)

    def test_completed_layout_refinement_is_explicit_rejection(self):
        # A valid particle list is offered twice. Only the first complete
        # layout's diagnostic is perturbed, never any partial-placement test.
        for kind in ('gap', 'curvature'):
            with self.subTest(kind=kind):
                config = copy.deepcopy(self.config)
                config.particles.placement = 'global_rejection'
                config.particles.max_geometry_attempts = 2
                config.particles.max_particle_attempts = 1
                real = geometry_module._separations
                first = True

                def diagnose(particles, c, count):
                    nonlocal first
                    result = real(particles, c, count)
                    if first and len(particles) == c.particles.count and count == c.validation.refined_points:
                        first = False
                        if kind == 'gap':
                            result['wall_clearance_lower_bound'] += 2*c.validation.gap_refinement_tolerance
                        else:
                            result['minimum_sampled_curvature_radius'] *= 1-2*c.validation.curvature_relative_tolerance
                    return result

                proposals = copy.deepcopy(self.record['particles']*2)
                with patch.object(geometry_module, '_draw_particle', side_effect=proposals) as draws, \
                        patch.object(geometry_module, '_admissible', return_value=True), \
                        patch.object(geometry_module, '_separations', side_effect=diagnose):
                    accepted = sample_geometry(config, np.random.default_rng(1))
                self.assertEqual(draws.call_count, 2*config.particles.count)
                self.assertEqual(accepted, self.record)

    def test_unexpected_final_validation_errors_are_not_resampled(self):
        with patch.object(geometry_module, 'validate_geometry', side_effect=ValueError('unexpected area error')) as validator:
            with self.assertRaisesRegex(ValueError, 'unexpected area error'):
                sample_geometry(self.config, np.random.default_rng(61027))
        self.assertEqual(validator.call_count, 1)


if __name__ == '__main__':
    unittest.main()
