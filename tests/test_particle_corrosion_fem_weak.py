"""Signed weak rows, natural-boundary loads and unchanged physical sampling."""
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict

from examples.particle_corrosion2d.fem_weak import diffusion_matrix, signed_payload, _initial_slot
from examples.particle_corrosion2d.mesh import assemble_p1
from examples.particle_corrosion2d.losses import graph_action


class WeakRowsTests(unittest.TestCase):
    def setUp(self):
        points = np.array([[0., 0.], [1., 0.], [.25, .1]])
        cells, mass, stiffness = assemble_p1(points, np.array([[0, 1, 2]]))
        self.mesh = dict(points=points, triangles=cells, volumes=mass,
                         stiffness=stiffness, mouth=np.zeros(3, dtype=bool))
        self.matrix = diffusion_matrix(self.mesh)
        self.c = ConfigDict(dict(sample=dict(K_core=3), operator=dict(online=dict(halo=3, neighbors=2))))

    def test_obtuse_signed_rows_match_matrix_and_preserve_constants(self):
        payload = signed_payload(self.c,self.mesh,self.matrix,np.arange(3),np.ones(3))
        self.assertTrue(np.any(payload['coefficients'] < 0))
        values = np.array([[.2, 1.], [.7, 1.], [-.1, 1.]])
        actual = graph_action(jnp.asarray(values), {k:jnp.asarray(v) for k,v in payload.items()})
        np.testing.assert_allclose(actual, self.matrix@values, rtol=2e-6, atol=2e-6)
        np.testing.assert_array_equal(np.asarray(actual)[:,1], 0.)
        np.testing.assert_allclose(self.mesh['volumes']@np.asarray(actual), 0., atol=2e-7)

    def test_linear_manufactured_boundary_load_and_orientation(self):
        points = self.mesh['points']
        gradient = np.array([2., -3.])
        values = points@gradient
        load = np.zeros(3)
        for a,b in ((0,1),(1,2),(2,0)):
            edge = points[b]-points[a]
            outward_measure = np.array([edge[1],-edge[0]])
            load[[a,b]] += gradient@outward_measure/2
        np.testing.assert_allclose(self.mesh['stiffness']@values,load,atol=1e-14)
        _,mass,stiffness=assemble_p1(points,np.array([[0,2,1]]))
        np.testing.assert_allclose(stiffness.toarray(),self.mesh['stiffness'].toarray(),atol=1e-14)
        np.testing.assert_array_equal(mass,self.mesh['volumes'])

    def test_mouth_columns_and_duplicate_core_gradients_are_retained(self):
        self.mesh['mouth'][0]=True
        core=np.array([1,2,1])
        payload=signed_payload(self.c,self.mesh,self.matrix,core,np.ones(3))
        data={k:jnp.asarray(v) for k,v in payload.items()}
        features=jnp.asarray([[1.,.3],[.2,1.],[.5,-.7]])
        dense=jnp.asarray(self.matrix.toarray()[core])
        def compact(parameters):
            return jnp.mean(graph_action(jnp.tanh(features@parameters)[:,None],data)**2)
        def full(parameters):
            return jnp.mean((dense@jnp.tanh(features@parameters))**2)
        parameters=jnp.array([.3,-.2])
        np.testing.assert_allclose(jax.grad(compact)(parameters),jax.grad(full)(parameters),rtol=2e-6,atol=2e-5)
        self.assertIn(0,np.unique(payload['neighbors']))
        with self.assertRaisesRegex(ValueError,'free physical'):
            signed_payload(self.c,self.mesh,self.matrix,np.arange(3),np.ones(3))

    def test_support_overflow_is_not_truncated(self):
        self.c.operator.online.neighbors=1
        with self.assertRaisesRegex(ValueError,'overflow'):
            signed_payload(self.c,self.mesh,self.matrix,np.arange(3),np.ones(3))


class GeometryAndConfigTests(unittest.TestCase):
    def test_journal_pairing_allows_operator_support_but_not_geometry_changes(self):
        import json
        import tempfile
        from pathlib import Path
        from examples.particle_corrosion2d.method_validation import _journal_physics
        with tempfile.TemporaryDirectory() as directory:
            first,second=Path(directory)/'first',Path(directory)/'second'
            first.mkdir();second.mkdir()
            row=dict(wait_seconds=.1,records=[dict(worker_seconds=.2,cache_hit=False,
                actual_halo=100,actual_neighbors=9,geometry=dict(center=[.2,.3]))])
            (first/'online_refresh.jsonl').write_text(json.dumps(row)+'\n')
            row['records'][0].update(actual_halo=200,actual_neighbors=20)
            (second/'online_refresh.jsonl').write_text(json.dumps(row)+'\n')
            self.assertEqual(_journal_physics(first,different_operators=True),
                             _journal_physics(second,different_operators=True))
            self.assertNotEqual(_journal_physics(first),_journal_physics(second))
            row['records'][0]['geometry']['center'][0]=.4
            (second/'online_refresh.jsonl').write_text(json.dumps(row)+'\n')
            self.assertNotEqual(_journal_physics(first,different_operators=True),
                                _journal_physics(second,different_operators=True))

    def test_real_geometry_same_fixed_inputs_core_draws_and_signed_validation(self):
        from tests.test_particle_corrosion_online import small_config
        from examples.particle_corrosion2d.online import build_input, FIXED_KEYS, validate_payload
        from examples.particle_corrosion2d.method_validation import compare_banks
        c=small_config()
        c.operator.method_comparison='rolling_fem_v1'
        nonlocal_data,meta=build_input(c.to_dict(),314159,0,0,None)
        c.nl.method='fem_weak'
        c.fem=ConfigDict(dict(kind='p1_lumped_nodal',residual_norm='mass_normalized_l1'))
        fem_data,other=build_input(c.to_dict(),314159,0,0,None)
        self.assertEqual(meta['geometry'],other['geometry'])
        for key in FIXED_KEYS:
            np.testing.assert_array_equal(nonlocal_data[key],fem_data[key],err_msg=key)
        compare_banks({k:v[None] for k,v in nonlocal_data.items()},
                      {k:v[None] for k,v in fem_data.items()},'real inputs',True)
        validate_payload(fem_data,c)
        cores=np.take_along_axis(nonlocal_data['graph_points'],nonlocal_data['core_indices'][...,None],axis=1)
        rebuilt=_initial_slot((c.to_dict(),meta['geometry'],cores,nonlocal_data['pde_weights']))
        for key in rebuilt:
            np.testing.assert_array_equal(rebuilt[key],fem_data[key])
        bad={k:v[None].copy() for k,v in fem_data.items()}
        bad['pde_weights'][0,0,0]*=2
        with self.assertRaisesRegex(ValueError,'weights'):
            compare_banks({k:v[None] for k,v in nonlocal_data.items()},bad,'changed',True)

    def test_explicit_presets_and_same_physical_canonical_config(self):
        from examples.particle_corrosion2d.configs.train_fem_comparison_nonlocal import get_config as n
        from examples.particle_corrosion2d.configs.train_fem_comparison_local_ad import get_config as a
        from examples.particle_corrosion2d.configs.train_fem_comparison_weak import get_config as f
        from examples.particle_corrosion2d.problem import validate
        from examples.particle_corrosion2d.online import validate_online
        from examples.particle_corrosion2d.method_validation import validate_config
        from examples.particle_corrosion2d.compare import _canonical_config
        import json
        configs=[n(),a(),f()]
        for c in configs:
            for check in (validate,validate_online,validate_config):check(c)
        canonical=[_canonical_config(json.loads(c.to_json()))[0] for c in configs]
        self.assertEqual(canonical[0],canonical[1])
        self.assertEqual(canonical[0],canonical[2])
        configs[2].fem.kind='unrecognized'
        with self.assertRaisesRegex(ValueError,'FEM requires'):validate(configs[2])


if __name__=='__main__':unittest.main()
