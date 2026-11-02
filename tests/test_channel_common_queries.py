"""Physical-frame geometry response and independent common-query reference gates."""
from pathlib import Path
import contextlib
import io
import json
import tempfile
import unittest
import numpy as np
from ml_collections import ConfigDict
from examples.channel_nsch2d.configs.train_operator_width128_online import get_config
from examples.channel_nsch2d.problem import walls,identity
from examples.channel_nsch2d.response_queries import points,protocol,freeze
from examples.channel_nsch2d.cohort_evaluate import response_metric,geometry_responses
from examples.channel_nsch2d.validate_reference import assess


class CommonQueries(unittest.TestCase):
    def test_response_protocol_is_frozen_without_changing_geometry_draws(self):
        from examples.channel_nsch2d.cohort import prepare
        c=get_config()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);manifest=root/'reserved'/'manifest.json'
            prepare(c,manifest,2,(7,5));before=manifest.read_bytes();output=root/'protocol.json'
            self.assertEqual(freeze(c,manifest,output),protocol(c,(7,5)))
            self.assertEqual(manifest.read_bytes(),before)
            with self.assertRaises(FileExistsError):freeze(c,manifest,output)

    def test_common_coordinates_inside_every_prior_corner_and_faceted_mesh(self):
        c=get_config();grid=(101,43);common=points(c,grid)
        for upper,lower in ((.04,.04),(.04,.14),(.14,.04),(.14,.14),(.12,.08)):
            q=ConfigDict(c.to_dict());q.geometry.upper=upper;q.geometry.lower=lower
            np.testing.assert_array_equal(points(q,grid),common)
            self.assertEqual(protocol(q,grid),protocol(c,grid))
            b,t=walls(common[:,0],(upper,lower),c,xp=np)
            self.assertTrue(np.all(common[:,1]>b));self.assertTrue(np.all(common[:,1]<t))
            # The coarsest FEM has piecewise-linear wall chords, not exact walls.
            x=np.linspace(0,c.geometry.length,c.reference.coarse_grid[0]+1)
            b,t=walls(x,(upper,lower),c,xp=np)
            self.assertTrue(np.all(common[:,1]>np.interp(common[:,0],x,b)))
            self.assertTrue(np.all(common[:,1]<np.interp(common[:,0],x,t)))

    def test_query_map_false_positive_and_unprescribed_initial_pressure(self):
        c=get_config();x,e=np.meshgrid(np.linspace(.2,3.8,7),np.linspace(.1,.9,5),indexing='ij')
        ambient=[]
        for amplitudes in ((.04,.04),(.14,.14)):
            b,t=walls(x,amplitudes,c,xp=np);ambient.append((b+e*(t-b)).ravel())
        mapped=np.asarray(ambient)[:,None,:,None]
        self.assertEqual(response_metric(mapped,mapped)['relative_error'],0.)
        common=points(c,(7,5));truth=np.broadcast_to(common[None,None,:,1,None],(2,3,len(common),1))
        with self.assertRaisesRegex(ValueError,'no geometry response'):response_metric(truth,truth)
        # A real geometry response, in the SAME physical frame, defeats a blind mean.
        target=np.broadcast_to(np.arange(2)[:,None,None,None]+np.ones((2,3,len(common),5)),(2,3,len(common),5)).copy()
        predicted=target.copy();predicted[0,0,:,4]=1000.;predicted[1,0,:,4]=-500.
        physical,_=geometry_responses(predicted,target,predicted,target)
        self.assertEqual(physical['pressure']['relative_error'],0.)
        blind=np.broadcast_to(target.mean(axis=0,keepdims=True),target.shape)
        physical,_=geometry_responses(blind,target,blind,target)
        for name in ('phi','velocity','pressure'):
            self.assertAlmostEqual(physical[name]['relative_error'],1.)
            self.assertEqual(physical[name]['amplitude_ratio'],0.)

    def test_common_refinement_failure_cannot_hide_behind_passed_main_queries(self):
        c=get_config();c.reference.snapshots=3;grid=(3,2);q=points(c,grid)
        values=np.ones((3,len(q),5));values[0,:,2:]=0.
        payload=dict(fields=values,points=q,time=np.linspace(0,6,3),amplitudes=np.array([.12,.08]),identity_json=json.dumps(identity(c)))
        common=dict(common_points=q,common_fields=values.copy(),common_query_grid=np.array(grid),common_query_identity=json.dumps(protocol(c,grid)))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);c.reference.root=str(root)
            paths=[root/f'{name}.npz' for name in ('coarse','time','fine')]
            diagnostics=[dict(inlet_flux=0.,outlet_flux=0.,wall_speed_l2=0.) for _ in range(3)]
            paths[-1].with_suffix('.json').write_text(json.dumps(dict(diagnostics=diagnostics)))
            for p in paths:np.savez(p,**payload)
            with contextlib.redirect_stdout(io.StringIO()):legacy=assess(c,*paths)
            self.assertEqual(legacy['status'],'passed');self.assertNotIn('common_queries',legacy)
            for p in paths:np.savez(p,**payload,**common)
            with contextlib.redirect_stdout(io.StringIO()):good=assess(c,*paths)
            self.assertEqual(good['common_queries']['status'],'passed')
            bad=common.copy();bad['common_fields']=values.copy();bad['common_fields'][...,0]*=1.5
            np.savez(paths[0],**payload,**bad)
            with contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(RuntimeError,'refinement failed'):assess(c,*paths)
            result=json.loads((root/'reference-validation.json').read_text())
            self.assertEqual(result['checks']['space']['phi']['maximum_time'],0.)
            self.assertEqual(result['common_queries']['status'],'failed');self.assertEqual(result['status'],'failed')
            missing=common.copy();del missing['common_fields'];np.savez(paths[1],**payload,**missing)
            with self.assertRaisesRegex(ValueError,'Incomplete common'):assess(c,*paths)

    def test_full_evaluator_uses_both_frames_and_correct_flux_boundary_slices(self):
        # Manufactured reporting fields exercise orchestration/indexing only;
        # these fixtures are NOT FEM trajectories or evidence of PINO accuracy.
        import equinox as eqx
        import jax
        import jax.numpy as jnp
        from unittest.mock import patch
        from examples.channel_nsch2d import network
        from examples.channel_nsch2d.cohort import prepare
        from examples.channel_nsch2d.cohort_evaluate import evaluate
        from examples.channel_nsch2d.problem import initial_field,flow_rate,wall_slopes
        self.assertEqual(jax.default_backend(),'cpu')
        c=get_config();c.sample.Nx=8;c.sample.Ny=8;c.sample.boundary_points=4
        c.reference.snapshots=3;c.train.seed=42
        def manufactured(model,encoded,x,t,a,config):
            b,top=walls(x[:,0],a,config,xp=jnp);bx,tx=wall_slopes(x[:,0],a,config,xp=jnp)
            h=top-b;eta=(x[:,1]-b)/h;shape=6*eta*(1-eta)
            Q=flow_rate(t,config,xp=jnp)[:,None]
            ux=Q*shape[None,:]/h[None,:]
            uy=Q*shape[None,:]*(bx+eta*(tx-bx))[None,:]/h[None,:]
            gate=-jnp.expm1(-t[:,None])*(x[:,0]/config.geometry.length)[None,:]
            phi=initial_field(x,config,xp=jnp)[None,:]+.02*(a[0]+a[1])*gate
            mu=jnp.broadcast_to(x[None,:,0]/config.geometry.length,phi.shape)
            pressure=Q*(1+a[0]+a[1])*(1-x[None,:,0]/config.geometry.length)
            return jnp.stack((phi,mu,ux,uy,pressure),axis=-1)
        def jets(model,encoded,x,t,a,config):
            f=lambda q:manufactured(model,encoded,q,t,a,config)
            value,dx=jax.jvp(f,(x,),(jnp.broadcast_to(jnp.array([1.,0.]),x.shape),))
            _,dy=jax.jvp(f,(x,),(jnp.broadcast_to(jnp.array([0.,1.]),x.shape),))
            _,dt=jax.jvp(lambda s:manufactured(model,encoded,x,s,a,config),(t,),(jnp.ones_like(t),))
            return value,dt,dx,dy
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run=root/'run';run.mkdir();manifest=root/'reserved'/'manifest.json'
            cohort=prepare(c,manifest,2,(7,5));common=points(c,(7,5));times=np.linspace(0,6,3)
            (run/'config.json').write_text(c.to_json());(run/'summary.json').write_text(json.dumps(dict(status='completed')))
            (run/'online-validation.json').write_text(json.dumps(dict(status='actual_rolling_inputs_reconstructed')))
            (manifest.parent/'cohort-reference-validation.json').write_text(json.dumps(dict(status='passed',cases=2,common_query_protocol=protocol(c,(7,5)))))
            model=eqx.nn.Linear(1,1,key=jax.random.PRNGKey(1));eqx.tree_serialise_leaves(run/'model_final.eqx',model)
            for case in cohort['cases']:
                cc=ConfigDict(c.to_dict());cc.geometry.upper,cc.geometry.lower=case['amplitudes']
                folder=Path(case['root']);folder.mkdir();a=np.asarray(case['amplitudes'])
                xx,ee=np.meshgrid((np.arange(7)+.5)*4/7,(np.arange(5)+.5)/5,indexing='ij')
                b,top=walls(xx,a,c,xp=np);own=np.stack((xx,b+ee*(top-b)),axis=-1).reshape(-1,2)
                target=lambda x:np.asarray(manufactured(model,None,jnp.asarray(x,dtype=jnp.float32),jnp.asarray(times,dtype=jnp.float32),jnp.asarray(a,dtype=jnp.float32),c))
                reference=folder/'fine.npz'
                np.savez(reference,fields=target(own),points=own,amplitudes=a,time=times,identity_json=json.dumps(identity(cc)),
                    common_points=common,common_fields=target(common),common_query_grid=np.array([7,5]),common_query_identity=json.dumps(protocol(c,(7,5))))
                (folder/'reference-validation.json').write_text(json.dumps(dict(status='passed',reference=str(reference),identity=identity(cc),common_queries=dict(status='passed'))))
            with patch.object(network,'build_network',lambda config,key:eqx.nn.Linear(1,1,key=key)),\
                 patch.object(network,'encode',lambda n,data:data['amplitudes']),\
                 patch.object(network,'fields',manufactured),patch.object(network,'field_jets',jets),\
                 contextlib.redirect_stdout(io.StringIO()):
                report=evaluate(run,manifest,root/'evaluation')
            self.assertTrue(report['mean_principal_target_met']);self.assertTrue(report['geometry_response_beats_mean_predictor'])
            for case in report['per_geometry']:
                self.assertLess(case['maximum_flux_error'],2e-5)
                self.assertLess(case['maximum_divergence'],2e-5)
                for name in ('phi','terminal_phi','velocity','pressure'):self.assertLess(case['metrics'][name],2e-6)
            for case in cohort['cases']:
                with np.load(root/'evaluation'/f'case_{case["index"]:03d}'/'prediction.npz',allow_pickle=False) as f:
                    np.testing.assert_array_equal(f['common_points'],common)
                    np.testing.assert_allclose(f['common_fields'],f['common_reference'],atol=1e-6,rtol=1e-6)


if __name__=='__main__':unittest.main()
