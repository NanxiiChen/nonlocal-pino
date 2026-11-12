"""Nontrivial graph closure and unbiased grouped physical sampling."""
import unittest
import tempfile
import numpy as np
from ml_collections import ConfigDict

from examples.confined_ch2d.operator_core_halo import VolumeGroups,RollingSpatialPool
from examples.confined_ch2d.operator_potential import compact_payload


class CoreHaloTests(unittest.TestCase):
    def test_pool_roll_and_geometry_replacement(self):
        from tests.test_confined_ch_operator import small_fixture
        c,raw=small_fixture()
        c.train.seed=42
        c.operator.train_conditions=2
        c.sample.spatial=ConfigDict(dict(mode='rolling_groups',core_count=4,group_size=2,
            cell_width=.2,wall_groups=0,wall_band=.1,inner_capacity=4,query_capacity=4,
            pools=2,refresh_every=2,workers=2))
        bank={key:np.stack([np.asarray(value)]*2) for key,value in raw.items() if key!='metric'}
        with tempfile.TemporaryDirectory() as directory:
            pool=RollingSpatialPool(c,bank,directory)
            try:
                batch,ids=pool.batch(np.array([0,1]),1)
                for key in ('points','volumes','initial','condition','condition_weights'):
                    np.testing.assert_array_equal(batch[key],bank[key])
                old_initial=np.asarray(batch['initial']).copy()
                updates={key:value[:1].copy() for key,value in bank.items()}
                updates['initial']+=.125
                pool.install(np.array([0]),updates)
                current,_=pool.batch(np.array([0,1]),2)
                np.testing.assert_array_equal(current['initial'][0],updates['initial'][0])
                np.testing.assert_array_equal(current['initial'][1],old_initial[1])
                self.assertEqual(pool.cache[0][0],(1,0))
                pool.batch(np.array([0,1]),3)
                self.assertEqual(pool.cache[0][0],(1,1))
                self.assertEqual(pool.cache[1][0],(0,1))
                self.assertEqual(pool.summary()['geometry_installations'],1)
                self.assertEqual(len(pool.cache),2)
            finally:
                pool.close()

    def test_two_hops_are_distinct_from_full_graph(self):
        n = 9
        nb = np.stack((np.maximum(0,np.arange(n)-1),np.minimum(n-1,np.arange(n)+1)),axis=1)
        data = dict(points=np.column_stack((np.arange(n),np.zeros(n))),volumes=np.ones(n),
            initial=np.linspace(-.2,.3,n),neighbors=nb,coefficients=np.ones((n,2)),
            condition=np.zeros((3,5)),condition_weights=np.ones(3))
        part, report = compact_payload(data,np.array([0]),np.array([1.]),inner_capacity=2,query_capacity=3)
        self.assertEqual(report,dict(core=1,inner=2,query=3,physical_nodes=9))
        np.testing.assert_array_equal(part['query_points'],data['points'][:3])
        for field in (np.arange(n)**3,np.cos(np.arange(n))):
            D = lambda f:np.sum(f[:,None]-f[nb],axis=1)
            first = np.sum(part['inner_coefficients']*(field[part['inner_indices'],None]-field[part['inner_neighbors']]),axis=1)
            second = np.sum(part['core_coefficients']*(first[part['core_inner_indices'],None]-first[part['core_neighbors']]),axis=1)
            np.testing.assert_allclose(second,D(D(field))[:1],atol=1e-12)

    def test_grouped_volume_mean_with_wall_split(self):
        axis = np.linspace(.025,.975,20)
        x,y = np.meshgrid(axis,axis)
        points = np.column_stack((x.ravel(),y.ravel()))
        volume = 1.+points[:,0]+points[:,1]**2
        # Three disjoint ordered polygon loops exercise the schema.
        outer = np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])
        holes = [np.array([[a,b],[a+.04,b],[a+.04,b+.04],[a,b+.04]]) for a,b in ((.3,.3),(.7,.7))]
        data = dict(points=points,volumes=volume,condition=np.pad(np.concatenate([outer,*holes]),((0,0),(0,3))))
        c = ConfigDict(dict(geometry=dict(boundary_outer=4,boundary_per_hole=4,center=(.5,.5),axes=(.5,.5))))
        sampler = VolumeGroups(data,c,cell_width=.2,wall_band=.1)
        f = np.exp(-np.sum((points-np.array([.3,.6]))**2,axis=1)/.01)
        exact = volume@f/volume.sum()
        rng = np.random.default_rng(18)
        for wall_groups in (0,1):
            values = []
            for _ in range(1500):
                ids,w = sampler.draw(core_count=128,group_size=32,wall_groups=wall_groups,rng=rng)
                self.assertEqual(ids.shape,(128,))
                self.assertAlmostEqual(w.sum(),1.)
                values.append(w@f[ids])
            # Independent Monte Carlo uncertainty; no reference-label tuning.
            values = np.asarray(values)
            self.assertLess(abs(values.mean()-exact),5*values.std(ddof=1)/np.sqrt(len(values)))


if __name__ == '__main__':
    unittest.main()
