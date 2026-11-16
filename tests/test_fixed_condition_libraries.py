"""Finite-library identity, spatial refresh and prescribed-field invariants."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
import numpy as np

from examples.particle_corrosion2d import online
from tests.test_particle_corrosion_online import small_config, ReverseExecutor


class FixedGeometryTests(unittest.TestCase):
    def test_spatial_refresh_never_replaces_geometry_or_disables_it(self):
        c=small_config(); old=online.plan_events(c,4)
        c.operator.geometry_sampling='fixed_library'
        events=online.plan_events(c,4)
        self.assertEqual([e['completed_updates'] for e in events],[2,4,6,8])
        self.assertTrue(all(t['kind']=='spatial' and t['draw_id']==-1-t['slot']
                            for e in events for t in e['tasks']))
        self.assertTrue(any(t['kind']=='geometry' for e in old for t in e['tasks']))
        records=[online.sample_geometry(c,np.random.default_rng(i+70)) for i in range(4)]
        with tempfile.TemporaryDirectory() as folder, patch.object(online,'ProcessPoolExecutor',ReverseExecutor):
            with patch.object(online,'sample_geometry',side_effect=AssertionError('Fixed geometry changed')):
                with online.RollingPool(c,folder,dict(geometries=records),314159,[]) as pool:
                    values=[]
                    for i in range(c.train.iterations):
                        refreshed=pool.before_step(i)
                        if refreshed is not None: values.append(refreshed)
                    np.testing.assert_array_equal(pool.draw_ids,[-1,-2,-3,-4])
                    np.testing.assert_array_equal(pool.spatial_versions,[2,2,2,2])
                    self.assertEqual(pool.records,records)
                    self.assertEqual(pool.summary()['installed_geometries'],0)
                    self.assertEqual(pool.summary()['installed_spatial_refreshes'],8)
                    first,later=values[0][1],values[2][1]
                    for name in ('domain_tokens','domain_weights','boundary_tokens','boundary_weights'):
                        np.testing.assert_array_equal(first[name],later[name])
                    self.assertFalse(np.array_equal(first['graph_points'],later['graph_points']))
        c.operator.geometry_sampling='unknown'
        with self.assertRaisesRegex(ValueError,'fixed_library'):online.validate_online(c)

    def test_library_size_preserves_pool_time_draws_and_initial_config(self):
        from examples.particle_corrosion2d.configs.train_fixed_geometry import get_config
        from examples.particle_corrosion2d.configs.train_time_film import get_config as base
        from examples.particle_corrosion2d.sampling import sample_times
        c=get_config(); d=c.copy_and_resolve_references();del d.operator['geometry_sampling']
        self.assertEqual(d.to_dict(),base().to_dict())
        expected=None
        for count in (64,128,256,512):
            rng=np.random.default_rng(42)
            ids=rng.integers(count,size=(c.train.iterations,c.operator.condition_batch),dtype=np.int32)
            pools=rng.integers(c.sample.K_pool,size=ids.shape,dtype=np.int32)
            times=sample_times(rng,c)
            if expected is not None:
                np.testing.assert_array_equal(pools,expected[0]);np.testing.assert_array_equal(times,expected[1])
            expected=pools,times


class FixedProfileTests(unittest.TestCase):
    def test_selected_prefix_replay_and_known_fields(self):
        from examples.electropolish2d.configs.train_operator_threeway import get_config
        from examples.electropolish2d.online_conditions import make_sampler
        from examples.electropolish2d.comparison_sampling import prescribed_shared_fields
        from examples.electropolish2d.condition_library import load_library
        from docs.tasks.fixed_library_prepare import electro_coefficients
        c=get_config();c.operator.condition_batch=2;c.operator.image_shape=(1,8,16)
        profiles,_=electro_coefficients(c,4,721)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            np.savez_compressed(path/'coefficients.npz',coefficients=profiles)
            (path/'manifest.json').write_text(json.dumps(dict(schema='electropolish_initial_library_v1',
                count=4,config=c.to_dict(),evolved_fields_generated=False)))
            c.operator.condition_library=ConfigDict(dict(path=str(path),count=2,sampling='uniform_with_replacement'))
            sample,geometry=make_sampler(c)
            points=jnp.array([[.1,.2],[.4,.34],[.7,.4],[.5,0],[.5,.5]],dtype=jnp.float32)
            first=sample(jax.random.key(5),points,geometry(points))
            again=sample(jax.random.key(5),points,geometry(points))
            for a,b in zip(first,again,strict=True):np.testing.assert_array_equal(a,b)
            self.assertEqual(int(first[3]),2)
            for p in np.asarray(first[2]):self.assertTrue(any(np.array_equal(p,q) for q in profiles[:2]))
            expected=prescribed_shared_fields(np.asarray(first[2]),np.asarray(points),c)
            np.testing.assert_allclose(first[1],expected,rtol=2e-5,atol=2e-6)
            np.testing.assert_array_equal(first[1][:,-2],1)
            np.testing.assert_array_equal(first[1][:,-1],0)
            c.operator.condition_library.count=5
            with self.assertRaisesRegex(ValueError,'shape/dtype/count'):load_library(c)


if __name__=='__main__':unittest.main()
