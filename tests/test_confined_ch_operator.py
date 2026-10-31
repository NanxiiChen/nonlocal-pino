"""Geometry conditioning, fixed IC, weighted padding and differentiated PDE."""
import unittest
from unittest.mock import patch
import numpy as np
import equinox as eqx
import jax
import jax.numpy as jnp
from examples.confined_ch2d.configs.train_operator_mild import get_config
from examples.confined_ch2d.operator_problem import (
    validate, sample_geometries, valid_holes, case_config, condition, fixed_initial)
from examples.confined_ch2d.problem import initial_field,initial_laplacian
from examples.confined_ch2d.operator_training import build_network,fields,phase,prepare,make_loss,query_phase
from nonlocal_pinn.nn.point_conditional import measure_attention,normalized_integral_attention


def small_fixture():
    c=get_config();c.arch.width=8;c.arch.heads=2;c.arch.latent_tokens=3
    c.arch.encoder_depth=1;c.arch.decoder_depth=1;c.arch.spatial_features=2;c.arch.time_features=2
    c.geometry.boundary_outer=8;c.geometry.boundary_per_hole=4
    holes=sample_geometries(c)[0]
    f,w=condition(holes,c)
    x=jnp.array([[.2,.3],[.5,.5],[.8,.6],[0.,0.]])
    v=jnp.array([.1,.2,.15,0.])
    nb=jnp.array([[1,2],[0,2],[0,1],[3,3]])
    # Positive symmetric pair conductance divided by each receiver volume.
    co=jnp.array([[2.,3.],[1.,2.],[2.,8/3],[0.,0.]])
    data=dict(points=x,volumes=v,initial=jnp.where(v>0,fixed_initial(x,c,xp=jnp),0.),
        neighbors=nb,coefficients=co,metric=jnp.diag(jnp.array([1.,1.,1.,0.])),
        condition=jnp.asarray(f),condition_weights=jnp.asarray(w))
    return c,data


class GeometryOperatorTests(unittest.TestCase):
    def test_explicit_native_only_preparation_without_fem(self):
        import json
        import os
        import tempfile
        from pathlib import Path
        from ml_collections import ConfigDict
        from examples.confined_ch2d.operator_problem import specification
        from examples.confined_ch2d.operator_precompute import prepare as precompute
        from examples.confined_ch2d.operator_reference import local_checks
        c=get_config();c.operator.train_conditions=2;c.operator.test_conditions=1
        c.operator.local_validation_indices=(2,)
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'source';source.mkdir()
            c.operator.dataset_path=str(source)
            (source/'config.json').write_text(c.to_json())
            records=[]
            for i,holes in enumerate(sample_geometries(c)):
                file=f'case_{i:03d}/reference.npz'
                path=source/file;path.parent.mkdir();path.write_bytes(b'reference fixture')
                records.append(dict(index=i,file=file,holes=holes.tolist()))
            manifest=dict(status='passed',specification=specification(c),conditions=records,
                          train_indices=[0,1],test_indices=[2])
            (source/'manifest.json').write_text(json.dumps(manifest))
            c.operator.dataset_path=str(Path(directory)/'native-only')
            c.operator.precompute=ConfigDict(dict(workers=1,source_path=str(source),
                reserved_test_draws=(2,),monitor_train_indices=(0,1)))
            c.operator.local_validation_indices=()
            validate(c)
            with patch.dict(os.environ,{'JAX_PLATFORMS':'cpu'}):precompute(c)
            root=Path(c.operator.dataset_path)
            made=json.loads((root/'manifest.json').read_text())
            self.assertEqual(made['status'],'passed')
            self.assertEqual(made['local_reference_status'],'not_requested')
            self.assertEqual(json.loads((root/'local_checks.json').read_text()),[])
            self.assertFalse(list(root.glob('local_*.npz')))
            self.assertEqual(made['pde_trajectory_solves'],0)
            with self.assertRaisesRegex(ValueError,'No local validation indices'):local_checks(c)
            # A requested local reference must still be present and passed.
            c.operator.dataset_path=str(Path(directory)/'required-local')
            c.operator.local_validation_indices=(2,)
            with patch.dict(os.environ,{'JAX_PLATFORMS':'cpu'}),self.assertRaises(FileNotFoundError):precompute(c)

    def test_fixed_smallscale10_matches_selected_component(self):
        from examples.confined_ch2d.configs.train_operator_smallscale10 import get_config as config
        from examples.confined_ch2d.configs.train_operator_smallscale10_512 import get_config as full_config
        from examples.confined_ch2d.configs.train_operator_multiscale import get_config as candidate
        from examples.confined_ch2d.operator_problem import fixed_initial_laplacian
        c=config();full=full_config();original=candidate();validate(c);validate(full)
        w=np.asarray(c.initial.waves);all_waves=np.asarray(original.initial.waves)
        chosen=all_waves[np.max(abs(all_waves[:,:2]),axis=1)==3]
        np.testing.assert_array_equal(w,chosen[:-2])
        self.assertEqual(len(w),10);self.assertEqual(c.initial.amplitude,original.initial.amplitude)
        self.assertTrue(np.all(np.max(abs(w[:,:2]),axis=1)==3))
        self.assertAlmostEqual(c.initial.amplitude*abs(w[:,2]).sum(),.36)
        self.assertEqual(c.physical.to_dict(),original.physical.to_dict())
        np.testing.assert_array_equal(sample_geometries(c),sample_geometries(original))
        np.testing.assert_array_equal(sample_geometries(full)[512:],sample_geometries(c)[16:])
        x=np.array([[.17,.3],[.41,.56],[.7,.63]])
        with jax.enable_x64(True):
            actual=jax.vmap(lambda p:jnp.trace(jax.hessian(lambda q:fixed_initial(q,c,xp=jnp))(p)))(jnp.asarray(x))
        np.testing.assert_allclose(actual,fixed_initial_laplacian(x,c),rtol=2e-12,atol=2e-12)

    def test_multiscale_fixed_initial_and_shared_geometry_prior(self):
        from examples.confined_ch2d.configs.train_operator_multiscale import get_config as config
        from examples.confined_ch2d.configs.train_operator_multiscale512 import get_config as full_config
        from examples.confined_ch2d.configs.train_operator_richer import get_config as old_config
        from examples.confined_ch2d.operator_problem import fixed_initial_laplacian
        c=config();full=full_config();old=old_config();validate(c);validate(full)
        w=np.asarray(c.initial.waves);axes=np.asarray(c.geometry.axes)
        self.assertEqual(len(w),24);self.assertEqual(c.physical.terminal_time,3.)
        self.assertAlmostEqual(c.initial.amplitude*abs(w[:,2]).sum(),.896)
        k2=np.sum((np.pi*w[:,:2]/axes)**2,axis=1)
        self.assertGreater(2*np.pi/np.sqrt(k2.max()),5*c.geometry.mesh_size)
        growth=c.physical.mobility*k2*(c.physical.beta-c.physical.epsilon**2*k2)
        self.assertTrue(np.any(growth>0) and np.any(growth<0))
        self.assertGreater(np.sum(w[8:,2]**2)/np.sum(w[:,2]**2),.5)
        np.testing.assert_array_equal(sample_geometries(c),sample_geometries(old))
        np.testing.assert_array_equal(sample_geometries(full)[512:],sample_geometries(old)[16:])
        x=np.array([[.17,.3],[.41,.56],[.7,.63]])
        with jax.enable_x64(True):
            actual=jax.vmap(lambda p:jnp.trace(jax.hessian(lambda q:fixed_initial(q,c,xp=jnp))(p)))(jnp.asarray(x))
        np.testing.assert_allclose(actual,fixed_initial_laplacian(x,c),rtol=2e-12,atol=2e-12)
        c.operator.reference_workers=0
        with self.assertRaisesRegex(ValueError,'worker count'):validate(c)

    def test_expanded_bank_reserves_original_validation_draws(self):
        from examples.confined_ch2d.configs.train_operator_richer import get_config as old_config
        from examples.confined_ch2d.configs.train_operator_richer512 import get_config as new_config
        from examples.confined_ch2d.operator_problem import evaluation_indices
        old=old_config();new=new_config();validate(new)
        old_draws=sample_geometries(old);draws=sample_geometries(new)
        self.assertEqual(draws.shape,(520,2,3))
        np.testing.assert_array_equal(draws[:16],old_draws[:16])
        np.testing.assert_array_equal(draws[512:],old_draws[16:])
        self.assertFalse(any(np.array_equal(a,b) for a in draws[:512] for b in draws[512:]))
        self.assertEqual(len(np.unique(draws.reshape(520,-1),axis=0)),520)
        m=dict(format='operator_only_v1',train_indices=[0,1],test_indices=[2],
            monitor_train_indices=[0],conditions=[dict(index=i,has_reference=i!=1) for i in range(3)])
        self.assertEqual(evaluation_indices(m),([0],[0,2]))
        m['conditions'][1]['has_reference']=True
        with self.assertRaises(ValueError):evaluation_indices(m)
        self.assertEqual(evaluation_indices(dict(train_indices=[0,1],test_indices=[2])),([0,1],[0,1,2]))

    def test_operator_only_loader_and_loss_exclude_future_labels(self):
        import json
        from pathlib import Path
        import tempfile
        from examples.confined_ch2d.operator_problem import specification
        from examples.confined_ch2d.operator_training import load_bank
        from examples.confined_ch2d.operators import RegionalLaplacian
        c,data=small_fixture();c.operator.train_conditions=2;c.operator.test_conditions=1
        c.operator.local_validation_indices=(2,)
        x=np.array([[.46,.65],[.5,.65],[.54,.65]]);v=np.asarray(data['volumes'])[:3]
        holes=sample_geometries(c)[0];f,w=condition(holes,c)
        with tempfile.TemporaryDirectory() as directory,jax.enable_x64(True):
            c.operator.dataset_path=directory;root=Path(directory)
            op=RegionalLaplacian(x,v,case_config(c,holes,0))
            initial=fixed_initial(x,c,xp=np)
            common=dict(points=x,volumes=v,holes=holes,condition=f,condition_weights=w,
                neighbors=np.asarray(op.neighbors),coefficients=np.asarray(op.coefficients),metric=np.eye(3),initial=initial)
            for i in range(3):
                fields_ref={} if i==1 else dict(time=np.linspace(0,c.physical.terminal_time,c.reference.snapshots),
                    phi=np.broadcast_to(initial,(c.reference.snapshots,3)),mu=np.zeros((c.reference.snapshots,3)))
                np.savez(root/f'{i}.npz',**common,**fields_ref)
            manifest=dict(format='operator_only_v1',status='passed',specification=specification(c),
                train_indices=[0,1],test_indices=[2],monitor_train_indices=[0],
                conditions=[dict(index=i,file=f'{i}.npz',holes=holes.tolist(),has_reference=i!=1) for i in range(3)])
            (root/'manifest.json').write_text(json.dumps(manifest))
            _,refs,bank=load_bank(c)
            self.assertNotIn('phi',refs[1]);self.assertNotIn('phi',bank);self.assertNotIn('mu',bank)
            np.testing.assert_array_equal(bank['initial'][1],initial)
            net=build_network(c,jax.random.key(0))
            batch=jax.tree.map(lambda a:a[jnp.array([1])],bank)
            value,gradient=eqx.filter_value_and_grad(make_loss(c))(net,batch,jnp.array([0.,.01]))
            self.assertTrue(np.isfinite(float(value)))
            self.assertTrue(all(np.isfinite(a).all() for a in jax.tree.leaves(gradient) if eqx.is_array(a)))
            np.savez(root/'1.npz',**common,phi=np.zeros((2,3)))
            with self.assertRaisesRegex(ValueError,'must not contain evolved'):load_bank(c)

    def test_axis_reflection_covariance_and_fixed_initial_reevaluation(self):
        from examples.confined_ch2d.configs.train_operator_richer_augmented import get_config as config
        from examples.confined_ch2d.operator_augmentation import axis_reflections
        from examples.confined_ch2d.native_reference import native_matrix
        c=config();validate(c)
        holes=np.array([[.37,.58,.06],[.68,.42,.07]])
        x=np.stack(np.meshgrid(np.linspace(.15,.85,11),np.linspace(.2,.8,11)),axis=-1).reshape(-1,2)
        keep=np.sum(((x-np.asarray(c.geometry.center))/np.asarray(c.geometry.axes))**2,axis=-1)<.85**2
        for cx,cy,r in holes:keep &= np.sum((x-[cx,cy])**2,axis=1)>r*r
        x=x[keep];v=np.full(len(x),.006);center=np.asarray(c.geometry.center)
        original=native_matrix(x,v,case_config(c,holes,0)).toarray()
        self.assertGreater(np.count_nonzero(original),len(x))
        f,w=condition(holes,c)
        with jax.enable_x64(True):
            padded=np.pad(x,((0,1),(0,0)));volumes=np.r_[v,0.]
            item=dict(points=jnp.asarray(padded),volumes=jnp.asarray(volumes),
                initial=jnp.where(volumes>0,fixed_initial(padded,c,xp=jnp),0.),
                condition=jnp.asarray(f),condition_weights=jnp.asarray(w),metric=jnp.eye(len(padded)))
            batch=jax.tree.map(lambda a:jnp.stack([a]*4),item)
            bits=jnp.array([[False,False],[True,False],[False,True],[True,True]])
            actual=eqx.filter_jit(lambda a,b:axis_reflections(a,b,c))(batch,bits)
            for i,b in enumerate(np.asarray(bits)):
                signs=np.where(b,-1.,1.)
                reflected=center+(x-center)*signs
                h=holes.copy();h[:,:2]=center+(h[:,:2]-center)*signs
                matrix=native_matrix(reflected,v,case_config(c,h,i)).toarray()
                np.testing.assert_allclose(matrix,original,atol=2e-11,rtol=2e-12)
                np.testing.assert_allclose(actual['initial'][i,:-1],fixed_initial(reflected,c,xp=np),atol=1e-15)
                np.testing.assert_allclose(actual['condition'][i,:,2:4],f[:,2:4]*signs,atol=1e-15)
            self.assertGreater(float(jnp.linalg.norm(actual['initial'][1]-batch['initial'][1])),.1)
            np.testing.assert_array_equal(actual['initial'][:,-1],0.)
            np.testing.assert_array_equal(actual['metric'],batch['metric'])
            np.testing.assert_array_equal(actual['volumes'],batch['volumes'])
        asymmetric=get_config();asymmetric.operator.augmentation='axis_reflections'
        with self.assertRaises(ValueError):validate(asymmetric)

    def test_richer_fixed_waves_geometry_prior_and_analytic_laplacian(self):
        from examples.confined_ch2d.configs.train_operator_richer import get_config as config
        c=config();validate(c);holes=sample_geometries(c)
        self.assertEqual(len(c.initial.waves),8)
        self.assertTrue(all(valid_holes(h,c) for h in holes))
        self.assertGreater(np.ptp(holes[:,:,:2]),.35)
        self.assertGreater(np.ptp(holes[:,:,2]),.05)
        waves=np.asarray(c.initial.waves)
        k2=np.sum((np.pi*waves[:,:2]/np.asarray(c.geometry.axes))**2,axis=1)
        growth=c.physical.mobility*k2*(c.physical.beta-c.physical.epsilon**2*k2)
        self.assertTrue(np.all(growth>0))
        x=np.random.default_rng(7).uniform((.1,.2),(.9,.8),(7,2))
        for i in (0,23):
            np.testing.assert_array_equal(initial_field(x,case_config(c,holes[i],i),xp=np),fixed_initial(x,c,xp=np))
        with jax.enable_x64(True):
            h=jax.vmap(jax.hessian(lambda point:fixed_initial(point,c,xp=jnp)))(jnp.asarray(x))
        np.testing.assert_allclose(initial_laplacian(x,c),np.trace(h,axis1=-2,axis2=-1),atol=5e-13)

    def test_smooth_display_preserves_bounds_constants_and_original_hole_topology(self):
        from examples.confined_ch2d.operator_plotting import SmoothMeshDisplay
        from examples.confined_ch2d.geometry import quadrature
        vertices=np.array([[-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.],
                           [-.3,-.3],[.3,-.3],[.3,.3],[-.3,.3]])
        cells=np.array([triangle for i in range(4) for triangle in
                        ((i,(i+1)%4,4+(i+1)%4),(i,4+(i+1)%4,4+i))])
        _,areas=quadrature(vertices,cells)
        display=SmoothMeshDisplay(vertices,cells,areas,np.array([[0.,0.,.3]]),
                                  extent=(-1.,1.,-1.,1.),resolution=(81,81))
        constant=display(np.ones((2,len(cells))))
        np.testing.assert_allclose(constant.compressed(),1.,atol=1e-15)
        self.assertTrue(constant.mask[:,40,40].all())
        # This point lies outside the circle but INSIDE the original excluded
        # square: its mask tests topology, not just the analytic-circle mask.
        self.assertTrue(constant.mask[:,50,50].all())
        raw=np.linspace(-.9,.7,len(cells))[None,:];copy=raw.copy()
        plotted=display(raw)
        self.assertGreaterEqual(float(plotted.min()),float(raw.min())-1e-14)
        self.assertLessEqual(float(plotted.max()),float(raw.max())+1e-14)
        np.testing.assert_allclose((display(raw+.12)-plotted).compressed(),.12,atol=1e-14)
        np.testing.assert_array_equal(raw,copy)

    def test_normalized_integral_attention_matches_explicit_dense_formula(self):
        with jax.enable_x64(True):
            q,k,v=(jax.random.normal(key,shape,dtype=jnp.float64) for key,shape in zip(
                jax.random.split(jax.random.key(19),3),((6,2,4),(9,2,4),(9,2,3)),strict=True))
            weights=jnp.array([.2,.3,0.,.1,.7,.9,.2,.4,0.])
            def dense(q,k,v):
                qs,ks=jax.nn.softmax(q,axis=-1),jax.nn.softmax(k,axis=-1)
                score=jnp.einsum('mhd,nhd->mhn',qs,ks)*weights
                score=score/jnp.sum(score,axis=-1,keepdims=True)
                return jnp.einsum('mhn,nhe->mhe',score,v)
            fast=lambda q,k,v:normalized_integral_attention(q,k,v,weights)
            np.testing.assert_allclose(fast(q,k,v),dense(q,k,v),atol=3e-16)
            for a,b in zip(jax.grad(lambda *x:jnp.sum(fast(*x)**2),argnums=(0,1,2))(q,k,v),
                           jax.grad(lambda *x:jnp.sum(dense(*x)**2),argnums=(0,1,2))(q,k,v),strict=True):
                np.testing.assert_allclose(a,b,atol=1e-15,rtol=2e-13)

    def test_heterogeneous_queries_measures_hard_constraints_and_pde_gradients(self):
        from examples.confined_ch2d.configs.train_operator_heterogeneous import get_config as config
        with jax.enable_x64(True):
            _,d=small_fixture();c=config()
            c.arch.width=8;c.arch.heads=2;c.arch.condition_depth=2
            c.arch.decoder_depth=1;c.arch.spatial_features=2;c.arch.time_features=2
            validate(c);net=build_network(c,jax.random.key(20))
            inputs=(d['condition'],jnp.column_stack((d['points'],d['initial'])))
            weights=(d['condition_weights'],d['volumes'])
            z=net.encode(inputs,weights);points=d['points'][:3]
            full=net.prepare_encoded(z,points)
            chunked=jnp.concatenate([net.prepare_encoded(z,points[:1]),net.prepare_encoded(z,points[1:])])
            np.testing.assert_allclose(full,chunked,atol=1e-13)
            reverse=net.encode(tuple(a[::-1] for a in inputs),tuple(w[::-1] for w in weights))
            np.testing.assert_allclose(net.prepare_encoded(reverse,points),full,atol=1e-13)
            split=net.encode(tuple(jnp.repeat(a,2,axis=0) for a in inputs),tuple(jnp.repeat(w/2,2) for w in weights))
            np.testing.assert_allclose(net.prepare_encoded(split,points),full,atol=1e-13)
            t=jnp.array([0.,.15,.75]);a=fields(net,d,t,c)
            np.testing.assert_allclose(a[0,:,0],d['initial'],atol=0,rtol=0)
            np.testing.assert_array_equal(a[:,-1],0.)
            np.testing.assert_allclose(a[:,:,0]@d['volumes'],d['initial']@d['volumes'],atol=1e-14)
            np.testing.assert_allclose(query_phase(net,d,points,t,c),a[:,:3,0],atol=1e-13)
            altered=dict(d);altered['points']=d['points'].at[-1].set(jnp.array([8.,-3.]))
            np.testing.assert_allclose(fields(net,altered,t,c)[:,:3],a[:,:3],atol=1e-13)
            prepared=prepare(net,d)
            _,dt=jax.jvp(lambda tt:phase(net,prepared,tt,d,c),(jnp.asarray(.15),),(jnp.asarray(1.),))
            fd=(phase(net,prepared,.150001,d,c)-phase(net,prepared,.149999,d,c))/2e-6
            np.testing.assert_allclose(dt,fd,atol=1e-9,rtol=1e-8)
            batch=jax.tree.map(lambda a:a[None],d)
            value,gradient=eqx.filter_value_and_grad(make_loss(c))(net,batch,t)
            self.assertTrue(np.isfinite(float(value)))
            self.assertTrue(all(np.isfinite(a).all() for a in jax.tree.leaves(eqx.filter(gradient,eqx.is_array))))
            for lift in gradient.lifts:
                self.assertGreater(float(jnp.linalg.norm(lift.layers[0].weight)),0.)
            with self.assertRaises(eqx.EquinoxRuntimeError):
                net.encode(inputs,(weights[0],jnp.zeros_like(weights[1])))

    def test_invalid_physical_prior_and_counts_fail_early(self):
        for section,name,value in (('physical','epsilon',float('nan')),
                                   ('operator','condition_batch',1.5),
                                   ('reference','snapshots',1),
                                   ('initial','waves',((0.,0.,1.,0.),)),
                                   ('arch','input_dim',6)):
            c=get_config()
            with c.ignore_type():c[section][name]=value
            with self.subTest(section=section,name=name),self.assertRaises(ValueError):validate(c)

    def test_direct_measure_tokens_and_unweighted_attention_parity(self):
        with jax.enable_x64(True):
            c,d=small_fixture();c.arch.latent_tokens=0
            net=build_network(c,jax.random.key(8))
            block=net.encoder[0]
            q=jax.random.normal(jax.random.key(1),(5,c.arch.width),dtype=jnp.float64)
            context=jax.random.normal(jax.random.key(2),(7,c.arch.width),dtype=jnp.float64)
            np.testing.assert_allclose(measure_attention(block,q,context,jnp.ones(7)),block(q,context),atol=1e-13)
            t=jnp.asarray([.04,.2]);expected=fields(net,d,t,c)
            perm=np.arange(len(d['condition']))[::-1]
            other=dict(d);other['condition']=d['condition'][perm];other['condition_weights']=d['condition_weights'][perm]
            np.testing.assert_allclose(fields(net,other,t,c),expected,atol=1e-13)
            other['condition']=jnp.repeat(d['condition'],2,axis=0)
            other['condition_weights']=jnp.repeat(d['condition_weights']/2,2)
            np.testing.assert_allclose(fields(net,other,t,c),expected,atol=1e-13)

    def test_prior_and_geometry_independent_initial_derivatives(self):
        c=get_config();validate(c);geometries=sample_geometries(c)
        self.assertEqual(geometries.shape,(24,2,3))
        self.assertTrue(all(valid_holes(h,c) for h in geometries))
        np.testing.assert_array_equal(sample_geometries(c),geometries)
        self.assertFalse(valid_holes([[.5,.5,.1],[.5,.5,.1]],c))
        self.assertFalse(valid_holes([[.99,.5,.1],[.5,.5,.1]],c))
        x=np.array([[.25,.4],[.5,.5],[.72,.65]])
        for index in (0,23):
            cc=case_config(c,geometries[index],index)
            np.testing.assert_array_equal(initial_field(x,cc,xp=np),initial_field(x,c,xp=np))
        with jax.enable_x64(True):
            h=jax.vmap(jax.hessian(lambda point:initial_field(point,c,xp=jnp)))(jnp.asarray(x))
        np.testing.assert_allclose(initial_laplacian(x,c),np.trace(h,axis1=-2,axis2=-1),atol=1e-12)

    def test_measure_encoder_permutation_and_quadrature_splitting(self):
        with jax.enable_x64(True):
            c,d=small_fixture();net=build_network(c,jax.random.key(3))
            z=net.encode(d['condition'],d['condition_weights'])
            permutation=np.random.default_rng(8).permutation(len(d['condition']))
            np.testing.assert_allclose(z,net.encode(d['condition'][permutation],d['condition_weights'][permutation]),atol=1e-13)
            np.testing.assert_allclose(z,net.encode(jnp.repeat(d['condition'],2,axis=0),jnp.repeat(d['condition_weights']/2,2)),atol=1e-13)

    def test_exact_initial_mass_padding_and_time_chain_rule(self):
        with jax.enable_x64(True):
            c,d=small_fixture();net=build_network(c,jax.random.key(4))
            t=jnp.asarray([0.,.07,.2]);a=fields(net,d,t,c)
            np.testing.assert_allclose(a[0,:,0],d['initial'],atol=0,rtol=0)
            np.testing.assert_array_equal(a[:,-1],0.)
            np.testing.assert_allclose(a[:,:,0]@d['volumes'],jnp.dot(d['volumes'],d['initial']),atol=1e-14)
            np.testing.assert_allclose(query_phase(net,d,d['points'][:3],t,c),a[:,:3,0],atol=1e-13)
            prepared=prepare(net,d)
            _,dt=jax.jvp(lambda tt:phase(net,prepared,tt,d,c),(jnp.asarray(.07),),(jnp.asarray(1.),))
            fd=(phase(net,prepared,.070001,d,c)-phase(net,prepared,.069999,d,c))/2e-6
            np.testing.assert_allclose(dt,fd,atol=1e-9,rtol=1e-8)
            self.assertLess(abs(float(dt@d['volumes'])),1e-14)
            # Moving dummy query coordinates cannot alter the physical outputs.
            altered=dict(d);altered['points']=d['points'].at[-1].set(jnp.array([8.,-3.]))
            np.testing.assert_allclose(fields(net,altered,t,c)[:,:3],a[:,:3],atol=1e-13)

    def test_pde_backprop_reaches_geometry_encoder_and_cli_dispatch(self):
        with jax.enable_x64(True):
            c,d=small_fixture();net=build_network(c,jax.random.key(5))
            batch=jax.tree.map(lambda a:a[None],d)
            value,gradient=eqx.filter_value_and_grad(make_loss(c))(net,batch,jnp.array([0.,.08,.2]))
            leaves=jax.tree.leaves(eqx.filter(gradient,eqx.is_array))
            self.assertTrue(np.isfinite(float(value)))
            self.assertTrue(all(np.isfinite(a).all() for a in leaves))
            self.assertGreater(float(jnp.linalg.norm(gradient.latent)),0.)
            self.assertGreater(float(jnp.linalg.norm(gradient.lift.layers[0].weight)),0.)
        from examples.confined_ch2d.train import train
        with patch('examples.confined_ch2d.operator_training.train_operator',return_value='geometry') as run:
            self.assertEqual(train(c),'geometry');run.assert_called_once_with(c)


if __name__=='__main__':
    unittest.main()
