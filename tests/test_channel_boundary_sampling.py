"""Physical boundary integration, unchanged conditioning and adaptive PDE gradients."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
import unittest

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict
from examples.channel_nsch2d.balance import make_balanced_fn,group_losses
from examples.channel_nsch2d.configs.train_stream_gradnorm import get_config as balanced
from examples.channel_nsch2d.data import geometry_data
from examples.channel_nsch2d.losses import make_loss_fn
from examples.channel_nsch2d.network import build_network,encode,fields
from examples.channel_nsch2d.online import build_input,validate_payload
from examples.channel_nsch2d.problem import boundary_loss_quadrature,validate_config,identity
from tests.test_channel_gmls_first import small


def settings():
    c=small()
    c.boundary_sampling=ConfigDict(dict(kind='gauss_legendre',points_per_side=12,
        encoder='unchanged_midpoints'))
    c.balance=balanced().balance
    return c


class BoundarySamplingTests(unittest.TestCase):
    def test_physical_quadrature_and_near_corner_coverage(self):
        c=settings();c.boundary_sampling.points_per_side=192
        a=np.array([.12,.08]);b,w=boundary_loss_quadrature(a,c);n=192
        self.assertTrue(np.all(w>0))
        self.assertLess(b[0,0],.0005)
        self.assertLess(c.geometry.length-b[n-1,0],.0005)
        self.assertTrue(np.all((b[:2*n,0]>0)&(b[:2*n,0]<c.geometry.length)))
        np.testing.assert_allclose(np.linalg.norm(b[:,2:4],axis=1),1.,atol=3e-16)
        # Straight-side exact polynomial moments check normal orientation and ds.
        for section,normal in [(slice(2*n,3*n),[-1.,0.]),(slice(3*n,4*n),[1.,0.])]:
            np.testing.assert_allclose(b[section,2:4],np.broadcast_to(normal,(n,2)))
            for degree in (0,1,2,5):
                self.assertAlmostEqual(float(w[section]@b[section,1]**degree),
                    c.geometry.height**(degree+1)/(degree+1),places=13)
        refined=ConfigDict(c.to_dict());refined.boundary_sampling.points_per_side=512
        rb,rw=boundary_loss_quadrature(a,refined)
        for side in (0,1):
            np.testing.assert_allclose(w[side*n:(side+1)*n].sum(),
                rw[side*512:(side+1)*512].sum(),rtol=0,atol=2e-8)

    def test_encoder_and_physical_fields_exactly_preserved(self):
        c=settings();old=small();old.balance=c.balance
        validate_config(c)
        with jax.enable_x64(True):
            new,_=geometry_data((.12,.08),c);previous,_=geometry_data((.12,.08),old)
            np.testing.assert_array_equal(new['encoder_boundary'],previous['boundary'])
            np.testing.assert_array_equal(new['encoder_boundary_weights'],previous['boundary_weights'])
            for key in ('points','volumes','laplacian','neighbors','gradient_x','gradient_y','domain'):
                np.testing.assert_array_equal(new[key],previous[key])
            key=jax.random.PRNGKey(14);net=build_network(c,key);prior=build_network(old,key)
            for x,y in zip(jax.tree.leaves(net),jax.tree.leaves(prior),strict=True):
                if eqx.is_array(x):np.testing.assert_array_equal(x,y)
            d=jax.tree.map(jnp.asarray,new);o=jax.tree.map(jnp.asarray,previous)
            for x,y in zip(jax.tree.leaves(encode(net,d)),jax.tree.leaves(encode(prior,o)),strict=True):
                np.testing.assert_array_equal(x,y)
            predict=eqx.filter_jit(lambda m,d:fields(m,encode(m,d),d['points'][::9],
                jnp.array([0.,.5,6.]),d['amplitudes'],c))
            np.testing.assert_array_equal(predict(net,d),predict(prior,o))
            payload,_=build_input(c.to_dict(),3);validate_payload(payload,c)
            bad=dict(payload);bad['encoder_boundary']=payload['encoder_boundary'].copy()
            bad['encoder_boundary'][0,0]+=.01
            with self.assertRaisesRegex(ValueError,'encoder_boundary'):validate_payload(bad,c)

    def test_adaptive_gradient_matches_stopped_weighted_physical_loss(self):
        c=settings()
        with jax.enable_x64(True):
            raw,_=geometry_data((.12,.08),c)
            batch={k:jnp.asarray(v)[None] for k,v in raw.items()}
            net=build_network(c,jax.random.PRNGKey(5));t=jnp.array([.5,3.])
            apply=eqx.filter_jit(make_balanced_fn(c))
            total,parts,w,norms,gradient=apply(net,batch,t,jnp.ones(4),jnp.array(True))
            self.assertTrue(np.isfinite(np.asarray(norms)).all());self.assertTrue(np.all(norms>0))
            loss=make_loss_fn(c)
            def weighted(m):return jnp.dot(jax.lax.stop_gradient(w),group_losses(loss(m,batch,t)[1],c))
            expected,expected_g=eqx.filter_jit(eqx.filter_value_and_grad(weighted))(net)
            np.testing.assert_allclose(total,expected,rtol=1e-12)
            for x,y in zip(jax.tree.leaves(gradient),jax.tree.leaves(expected_g),strict=True):
                if eqx.is_array(x):np.testing.assert_allclose(x,y,atol=2e-11,rtol=2e-10)
            _,again,_,_,_=apply(net,batch,t,w,jnp.array(False))
            np.testing.assert_array_equal(parts,again)
            for name,value in [('points_per_side',True),('encoder','changed'),('kind','uniform')]:
                invalid=ConfigDict(c.to_dict());invalid.boundary_sampling[name]=value
                with self.assertRaisesRegex(ValueError,'Boundary'):validate_config(invalid)

    def test_canonical_adaptive_online_training_and_replay(self):
        from examples.channel_nsch2d import train as entry
        from examples.channel_nsch2d.cohort import prepare
        from examples.channel_nsch2d.online import audit_training_inputs
        c=settings();c.train.iterations=3;c.sample.Nt=2;c.train.soap_max_precond_dim=32
        c.balance.update_every=1;c.operator.online.refresh_every=1;c.operator.online.refresh_size=2
        c.log.eval_every=3;c.log.checkpoint_every=3
        with tempfile.TemporaryDirectory() as directory,jax.enable_x64(True):
            root=Path(directory);c.reference.root=str(root);c.log.log_dir=str(root/'runs');c.log.run_name='adaptive'
            c.operator.validation_manifest=str(root/'reserved/manifest.json')
            prepare(c,c.operator.validation_manifest,4,(96,40))
            reference=root/'reference.npz';target=np.ones((13,3,5));target[...,0]=.5
            np.savez(reference,points=np.array([[.2,.3],[.8,.5],[2.,.5]]),
                time=np.linspace(0,6,13),amplitudes=np.array([.12,.08]),fields=target)
            (root/'reference-validation.json').write_text(json.dumps(dict(status='passed',reference=str(reference),identity=identity(c))))
            def executor(**kwargs):return ThreadPoolExecutor(max_workers=kwargs['max_workers'])
            with patch('examples.channel_nsch2d.online.ProcessPoolExecutor',executor),contextlib.redirect_stdout(io.StringIO()):
                entry.train(c)
            run=Path(c.log.log_dir)/'adaptive';report=audit_training_inputs(run,c)
            self.assertEqual(report['installed_inputs'],4)
            weights=np.load(run/'balance_final.npz')['weights']
            self.assertTrue(np.isfinite(weights).all());self.assertFalse(np.array_equal(weights,np.ones(4)))
            self.assertEqual(len((run/'balance.jsonl').read_text().splitlines()),3)
            self.assertNotEqual((run/'model_initial.eqx').read_bytes(),(run/'model_final.eqx').read_bytes())
            summary=json.loads((run/'summary.json').read_text())
            self.assertEqual(summary['boundary_sampling'],c.boundary_sampling.to_dict())


if __name__=='__main__':unittest.main()
