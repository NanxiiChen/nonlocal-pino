"""Random-wall geometry, host/JAX parity and fixed-shape operator contracts."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline

from examples.channel_nsch2d.configs.train_operator_random_walls import get_config
from examples.channel_nsch2d.geometry import (basis,knots,taper,nominal_parameters,initial_parameters,
    geometry_statistics,sample_parameters,validate_parameters,with_nominal)
from examples.channel_nsch2d.problem import walls,wall_slopes,validate_config


class RandomWallsTest(unittest.TestCase):
    def test_independent_scipy_spline_and_extrema(self):
        c=get_config();g=c.geometry;rng=np.random.default_rng(75)
        x=np.linspace(0,4,20001);z=(x-g.start)/(g.end-g.start)
        for _ in range(3):
            a=sample_parameters(c,rng);validate_parameters(a,c)
            b,t=walls(x,a,c,xp=np)
            gate=taper((x-g.start)/g.taper_width,xp=np)*taper((g.end-x)/g.taper_width,xp=np)
            for coeff,value in ((a[:8],1-t),(a[8:],b)):
                independent=np.nan_to_num(BSpline(knots(8),coeff,4,extrapolate=False)(z))*gate
                np.testing.assert_allclose(value,independent,atol=3e-15,rtol=0)
            db,dt=wall_slopes(x,a,c,xp=np)
            s=geometry_statistics(a,c)
            self.assertLessEqual(s['minimum_gap'],float(np.min(t-b))+2e-12)
            self.assertLess(float(np.min(t-b))-s['minimum_gap'],1e-6)
            self.assertLess(abs(s['maximum_slope']-max(np.max(abs(db)),np.max(abs(dt)))),1e-5)
            np.testing.assert_array_equal(b[x<=g.start],0.);np.testing.assert_array_equal(t[x>=g.end],1.)
        with self.assertRaises(ValueError):validate_parameters(np.full(16,.45),c)
        with self.assertRaises(ValueError):validate_parameters(np.ones(2),c)
        bad=get_config();bad.geometry.kind='typo'
        with self.assertRaises(ValueError):validate_config(bad)

    def test_jax_values_slopes_and_c3_knots(self):
        import jax
        import jax.numpy as jnp
        c=get_config();a=nominal_parameters(c);g=c.geometry
        with jax.enable_x64(True):
            x=np.unique(np.r_[np.linspace(0,4,31),g.start,g.end,g.start+g.taper_width,
                              g.end-g.taper_width,g.start+np.asarray(knots(8))*(g.end-g.start)])
            actual=jax.jit(lambda z:jnp.stack(walls(z,jnp.asarray(a),c,xp=jnp)))(jnp.asarray(x))
            np.testing.assert_allclose(actual,np.stack(walls(x,a,c,xp=np)),atol=2e-15,rtol=0)
            slope=jax.jacfwd(lambda z:jnp.stack(walls(z,jnp.asarray(a),c,xp=jnp)))
            np.testing.assert_allclose(jax.vmap(slope)(jnp.asarray(x)),np.stack(wall_slopes(x,a,c,xp=np),axis=-1),atol=2e-13,rtol=0)
            third=jax.vmap(jax.jacfwd(jax.jacfwd(slope)))
            around=np.r_[g.start,g.end,g.start+g.taper_width,g.end-g.taper_width,
                         g.start+np.asarray(knots(8))[5:-5]*(g.end-g.start)]
            np.testing.assert_allclose(third(jnp.asarray(around-1e-8)),third(jnp.asarray(around+1e-8)),atol=2e-3,rtol=1e-5)

    def test_legacy_rng_and_wall_arithmetic_are_unchanged(self):
        from examples.channel_nsch2d.configs.train_operator_width128_gmls_time40_jvp_stronger_online import get_config as old
        from examples.channel_nsch2d.online import sample_geometry,DRAW_NAMESPACE
        c=old();c.train.seed=42;x=np.linspace(0,4,81);a=np.array([.18,.12])
        shape=np.maximum(1-((x-c.geometry.center)/c.geometry.half_length)**2,0)**4
        np.testing.assert_array_equal(walls(x,a,c,xp=np),(a[1]*shape,1-a[0]*shape))
        np.testing.assert_array_equal(initial_parameters(c),np.random.default_rng(c.operator.dataset_seed).uniform(.06,.20,(16,2)))
        np.testing.assert_array_equal(sample_geometry(c,7),np.random.default_rng(np.random.SeedSequence([42,DRAW_NAMESPACE,7])).uniform(.06,.20,2))

    def test_rolling_payload_and_reserved_cohort_replay(self):
        from examples.channel_nsch2d.online import sample_geometry,build_input,validate_payload,RollingGeometryPool
        from examples.channel_nsch2d.cohort import prepare,load
        c=get_config();c.train.seed=42;c.sample.Nx=48;c.sample.Ny=20;c.nl.delta=.3
        a=sample_geometry(c,3);np.testing.assert_array_equal(a,sample_geometry(c,3))
        d,r=build_input(c.to_dict(),3);validate_payload(d,c)
        self.assertEqual(d['amplitudes'].shape,(16,));self.assertEqual(d['boundary'].shape[1],5)
        self.assertLess(r['operator']['polynomial_error'],5e-9)
        with tempfile.TemporaryDirectory() as directory:
            manifest=Path(directory)/'cohort'/'manifest.json'
            value=prepare(c,manifest,3,(96,40));self.assertEqual(load(manifest,c),value)
            self.assertEqual(np.asarray([v['amplitudes'] for v in value['cases']]).shape,(3,16))
            pool=RollingGeometryPool(c,directory,np.asarray([value['cases'][0]['amplitudes']]))
            self.assertEqual(pool.forbidden.shape,(1,16))
            with self.assertRaises(ValueError):RollingGeometryPool(c,directory,np.zeros((1,2)))
            bad=json.loads(manifest.read_text());bad['cases'][0]['amplitudes'][0]+=.001
            manifest.write_text(json.dumps(bad))
            with self.assertRaises(AssertionError):load(manifest,c)
        self.assertEqual(tuple(with_nominal(c,a).geometry.coefficients),tuple(a))

    def test_fem_chord_domain_rejects_underresolved_curved_wall(self):
        from examples.channel_nsch2d.geometry import validate_mesh_queries
        c=get_config();c.train.seed=42
        from examples.channel_nsch2d.cohort import DRAW_SEED,DRAW_NAMESPACE
        a=sample_parameters(c,np.random.default_rng(np.random.SeedSequence([DRAW_SEED,DRAW_NAMESPACE,1])))
        xx,ee=np.meshgrid((np.arange(96)+.5)*4/96,(np.arange(40)+.5)/40,indexing='ij')
        b,t=walls(xx,a,c,xp=np);q=np.stack((xx,b+ee*(t-b)),axis=-1).reshape(-1,2)
        with self.assertRaisesRegex(ValueError,'chord-wall domain'):validate_mesh_queries(q,a,c,40)
        validate_mesh_queries(q,a,c,64);validate_mesh_queries(q,a,c,96)

    def test_canonical_training_uses_and_audits_sixteen_parameters(self):
        import contextlib,io
        from concurrent.futures import ThreadPoolExecutor
        from unittest.mock import patch
        from examples.channel_nsch2d import train as entry
        from examples.channel_nsch2d.problem import identity
        from examples.channel_nsch2d.online import audit_training_inputs
        c=get_config();c.train.seed=42;c.train.iterations=6
        c.sample.Nx=24;c.sample.Ny=12;c.sample.Nt=2;c.sample.boundary_points=6
        c.nl.delta=.6;c.nl.neighbors=192;c.train.microbatch.size=1
        c.arch.width=4;c.arch.heads=1;c.arch.condition_depth=1;c.arch.decoder_depth=1
        c.arch.spatial_features=2;c.arch.time_features=4;c.train.soap_max_precond_dim=32
        c.operator.train_conditions=4;c.operator.condition_batch=1
        c.operator.online.refresh_every=2;c.log.eval_every=3;c.log.checkpoint_every=3
        del c.operator.validation_manifest
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);c.reference.root=str(root);c.log.log_dir=str(root/'runs');c.log.run_name='fixture'
            # Synthetic labels are exclusively a reporting fixture. Geometry,
            # NS--CH/integral/JVP gradients, SOAP and rolling updates are actual.
            path=root/'reference.npz'
            target=np.ones((13,3,5));target[...,0]=.5
            np.savez(path,points=np.array([[.2,.3],[.8,.5],[2.,.5]]),
                time=np.linspace(0,6,13),amplitudes=nominal_parameters(c),fields=target)
            (root/'reference-validation.json').write_text(json.dumps(dict(status='passed',reference=str(path),identity=identity(c))))
            with patch('examples.channel_nsch2d.online.ProcessPoolExecutor',lambda **kw:ThreadPoolExecutor(max_workers=kw['max_workers'])),contextlib.redirect_stdout(io.StringIO()):
                entry.train(c)
            run=root/'runs'/'fixture';result=audit_training_inputs(run,c)
            self.assertEqual(result['installed_inputs'],4)
            with np.load(run/'sampling.npz') as f:self.assertEqual(f['amplitudes'].shape,(6,1,16))
            self.assertNotEqual((run/'model_initial.eqx').read_bytes(),(run/'model_final.eqx').read_bytes())
            self.assertEqual(json.loads((run/'summary.json').read_text())['supervised_evolved_field_samples'],0)


if __name__=='__main__':unittest.main()
