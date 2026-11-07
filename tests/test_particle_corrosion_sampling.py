"""Independent inverse-CDF and unchanged uniform-stream checks."""
import unittest
import numpy as np
from ml_collections import ConfigDict
from examples.particle_corrosion2d.sampling import sample_times,initial_layer_times,horizon_fractions


class TemporalSamplingTests(unittest.TestCase):
    def config(self):
        return ConfigDict(dict(train=dict(iterations=13), sample=dict(Nt=32, t_max=2.3)))

    def test_historical_uniform_stream_is_exact(self):
        c=self.config();left=np.random.default_rng(712);right=np.random.default_rng(712)
        expected=right.uniform(0,2.3,size=(13,32)).astype(np.float32)
        np.testing.assert_array_equal(sample_times(left,c),expected)
        self.assertEqual(left.random(),right.random())

    def test_global_log_strata_and_initial_node(self):
        c=self.config();c.sample.temporal=ConfigDict(dict(kind='logarithmic_stratified',time_scale=.01))
        first=sample_times(np.random.default_rng(12),c)
        second=sample_times(np.random.default_rng(13),c)
        np.testing.assert_array_equal(first[:,0],0.)
        self.assertTrue(np.all(np.diff(first,axis=1)>0))
        self.assertTrue(np.all(first[:,-1]<2.3))
        coordinate=np.log1p(first[:,1:]/.01)/np.log1p(2.3/.01)
        lower=np.arange(31)/31
        self.assertTrue(np.all(coordinate>=lower-1e-7))
        self.assertTrue(np.all(coordinate<(lower+1/31)+1e-7))
        self.assertFalse(np.array_equal(first,second))
        self.assertTrue(np.all(np.sum(first<.023,axis=1)>=6))

    def test_initial_layer_preserves_global_draws_and_resamples(self):
        c=self.config();c.sample.temporal=ConfigDict(dict(kind='initial_layer',early_count=8,early_end=.012))
        raw=np.random.default_rng(91).uniform(0,2.3,size=(13,32)).astype(np.float32)
        actual=initial_layer_times(raw,c)
        np.testing.assert_array_equal(actual[:,8:],raw[:,8:])
        np.testing.assert_array_equal(actual[:,0],0.)
        self.assertTrue(np.all(np.diff(actual[:,:8],axis=1)>0))
        self.assertTrue(np.all(actual[:,7]<.012))
        bins=actual[:,1:8]/.012*7
        self.assertTrue(np.all(bins>=np.arange(7)-1e-6))
        self.assertTrue(np.all(bins<np.arange(7)+1+1e-6))
        self.assertFalse(np.array_equal(actual[0,1:8],actual[1,1:8]))
        np.testing.assert_array_equal(actual,sample_times(np.random.default_rng(91),c))

    def test_integral_endpoints_include_full_horizon_and_global_strata(self):
        c=self.config();c.sample.temporal=ConfigDict(dict(kind='integral_initial_layer',early_count=8,early_end=.012))
        t=sample_times(np.random.default_rng(91),c)
        np.testing.assert_array_equal(t[:,0],0.)
        np.testing.assert_array_equal(t[:,-1],np.float32(c.sample.t_max))
        global_scaled=t[:,8:-1]/2.3*23
        self.assertTrue(np.all(global_scaled>=np.arange(23)-1e-6))
        self.assertTrue(np.all(global_scaled<np.arange(23)+1+1e-6))
        self.assertTrue(np.all(t[:,1:8]<.012))

    def test_horizon_curriculum_preserves_early_draws_warmup_tail_and_rng(self):
        c=self.config();c.sample.temporal=ConfigDict(dict(kind='initial_layer',early_count=8,early_end=.012))
        source_rng=np.random.default_rng(41);source=sample_times(source_rng,c)
        c.loss=ConfigDict(dict(data_warmup_steps=2))
        c.sample.horizon_curriculum=ConfigDict(dict(kind='exponential',initial_fraction=.02,ramp_steps=5))
        rng=np.random.default_rng(41);actual=sample_times(rng,c)
        fractions=horizon_fractions(c)
        expected=np.ones(13);expected[2:7]=.02**(1-np.arange(5)/4)
        np.testing.assert_allclose(fractions,expected,rtol=1e-7)
        np.testing.assert_array_equal(actual[:,:8],source[:,:8])
        np.testing.assert_array_equal(actual[:2],source[:2])
        np.testing.assert_array_equal(actual[6:],source[6:])
        np.testing.assert_allclose(actual[2:6,8:],source[2:6,8:]*expected[2:6,None],rtol=2e-7)
        self.assertTrue(np.all(actual<=2.3*fractions[:,None]))
        self.assertEqual(rng.random(),source_rng.random())

    def test_horizon_curriculum_rejects_truncated_early_band_and_no_full_tail(self):
        c=self.config();c.sample.temporal=ConfigDict(dict(kind='initial_layer',early_count=8,early_end=.012))
        c.loss=ConfigDict(dict(data_warmup_steps=2))
        c.sample.horizon_curriculum=ConfigDict(dict(kind='exponential',initial_fraction=.001,ramp_steps=5))
        with self.assertRaisesRegex(ValueError,'first horizon'):
            horizon_fractions(c)
        c.sample.horizon_curriculum.initial_fraction=.02
        c.sample.horizon_curriculum.ramp_steps=11
        with self.assertRaisesRegex(ValueError,'full-time tail'):
            horizon_fractions(c)


if __name__=='__main__':unittest.main()
