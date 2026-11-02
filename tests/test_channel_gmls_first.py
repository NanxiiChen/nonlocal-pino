"""GMLS first functionals, flux signs, legacy parity and actual loss derivatives."""
import unittest
from pathlib import Path
import contextlib
import io
import json
import tempfile
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from examples.channel_nsch2d.configs.train_operator_width128_gmls_first_online import get_config
from examples.channel_nsch2d.data import geometry_data
from examples.channel_nsch2d.losses import make_loss_fn,gmls_first_jets,gmls_phase_advection
from examples.channel_nsch2d.network import build_network,encode,fields
from examples.channel_nsch2d.online import build_input,validate_payload
from examples.channel_nsch2d.problem import validate_config


def small():
    c=get_config();c.train.seed=42;c.train.compute_dtype='float64'
    c.sample.Nx=16;c.sample.Ny=8;c.sample.boundary_points=6;c.nl.degree=2;c.nl.neighbors=20
    c.arch.width=4;c.arch.heads=1;c.arch.condition_depth=1;c.arch.decoder_depth=1
    c.arch.spatial_features=2;c.arch.time_features=4
    c.operator.train_conditions=4;c.operator.condition_batch=1
    return c


class GMLSFirstTests(unittest.TestCase):
    def test_polynomial_gradients_boundary_and_conservative_flux(self):
        c=get_config();c.train.seed=42;c.sample.Nx=24;c.sample.Ny=10
        raw,_=geometry_data((.12,.08),c);x,y=raw['points'].T;bx,by=raw['boundary'][:,:2].T
        with jax.enable_x64(True):
            d={k:jnp.asarray(v) for k,v in raw.items()}
            value=jnp.asarray(np.stack((x**3+2*x*y+y*y,np.ones_like(x)),axis=-1)[None])
            target=jnp.asarray(np.stack((bx**3+2*bx*by+by*by,np.ones_like(bx)),axis=-1)[None])
            gx,gy=gmls_first_jets(value,target,d,boundary=True)
            np.testing.assert_allclose(gx[0,:,0],3*bx*bx+2*by,atol=2e-10,rtol=0)
            np.testing.assert_allclose(gy[0,:,0],2*bx+2*by,atol=2e-10,rtol=0)
            np.testing.assert_array_equal(gx[0,:,1],0.)
            with self.assertRaisesRegex(ValueError,'aligned'):
                gmls_first_jets(value,target[:,:-1],d,boundary=True)
            phi=x*x+.3*y;u=y;v=-x
            physical=jnp.asarray(np.column_stack((phi,0*x,u,v,0*x))[None])
            np.testing.assert_allclose(gmls_phase_advection(physical,d)[0],2*x*y-.3*x,atol=2e-10,rtol=0)

    def test_input_schema_laplacian_and_physical_output_preserved(self):
        c=small();validate_config(c)
        with jax.enable_x64(True):
            data,_=build_input(c.to_dict(),3);validate_payload(data,c)
            self.assertEqual(data['boundary_neighbors'].dtype,np.dtype('int32'))
            with self.assertRaisesRegex(ValueError,'boundary neighbor'):
                bad=dict(data);bad['boundary_neighbors']=data['boundary_neighbors'].copy()
                bad['boundary_neighbors'][0,0]=-1;validate_payload(bad,c)
            old=small();del old.derivatives
            old.phase_transport.kind='sbp4_mapped_curl';del old.phase_transport.quadrature
            original,_=geometry_data(data['amplitudes'],old)
            for name in original:np.testing.assert_array_equal(data[name],original[name])
            net=build_network(c,jax.random.PRNGKey(4));legacy=build_network(old,jax.random.PRNGKey(4))
            for a,b in zip(jax.tree.leaves(net),jax.tree.leaves(legacy),strict=True):
                if eqx.is_array(a):np.testing.assert_array_equal(a,b)
            d={k:jnp.asarray(v) for k,v in data.items()}
            old_data={k:jnp.asarray(v) for k,v in original.items()}
            point=d['points'][::8];t=jnp.asarray([0.,1.,6.])
            actual=fields(net,encode(net,d),point,t,d['amplitudes'],c)
            expected=fields(legacy,encode(legacy,old_data),point,t,d['amplitudes'],old)
            np.testing.assert_array_equal(actual,expected)

    def test_full_new_loss_parameter_gradient_against_finite_difference(self):
        c=small()
        with jax.enable_x64(True):
            raw,_=geometry_data((.12,.08),c);d={k:jnp.asarray(v)[None] for k,v in raw.items()}
            net=build_network(c,jax.random.PRNGKey(8));times=jnp.array([.5,3.])
            loss=make_loss_fn(c)
            evaluate=eqx.filter_jit(eqx.filter_value_and_grad(loss,has_aux=True))
            (total,parts),gradient=evaluate(net,d,times)
            self.assertEqual(parts.shape,(9,));self.assertTrue(np.isfinite(total))
            epsilon=2e-5;index=(0,0)
            f=eqx.filter_jit(lambda network:loss(network,d,times)[0])
            plus=eqx.tree_at(lambda n:n.output.weight,net,net.output.weight.at[index].add(epsilon))
            minus=eqx.tree_at(lambda n:n.output.weight,net,net.output.weight.at[index].add(-epsilon))
            numeric=float((f(plus)-f(minus))/(2*epsilon))
            self.assertGreater(abs(numeric),1e-8)
            np.testing.assert_allclose(gradient.output.weight[index],numeric,rtol=2e-5,atol=2e-7)

    def test_unsupported_and_ambiguous_configurations_rejected(self):
        for key,value in [('kind','ad'),('boundary','ad'),('velocity','gmls_curl')]:
            c=small();setattr(c.derivatives,key,value)
            with self.assertRaisesRegex(ValueError,'GMLS-first'):validate_config(c)
        c=small();del c.derivatives
        with self.assertRaisesRegex(ValueError,'GMLS conservative'):validate_config(c)

    def test_canonical_online_training_and_final_input_audit(self):
        from examples.channel_nsch2d import train as entry
        from examples.channel_nsch2d.cohort import prepare
        from examples.channel_nsch2d.problem import identity
        from examples.channel_nsch2d.online import audit_training_inputs
        c=small();c.train.iterations=3;c.sample.Nt=2
        c.train.soap_max_precond_dim=32
        c.operator.online.refresh_every=1;c.operator.online.refresh_size=2
        c.log.eval_every=3;c.log.checkpoint_every=3
        with tempfile.TemporaryDirectory() as directory,jax.enable_x64(True):
            root=Path(directory);c.reference.root=str(root);c.log.log_dir=str(root/'runs');c.log.run_name='gmls'
            c.operator.validation_manifest=str(root/'reserved/manifest.json')
            prepare(c,c.operator.validation_manifest,4,(96,40))
            target=np.ones((13,3,5));target[...,0]=.5
            reference=root/'reference.npz'
            np.savez(reference,points=np.array([[.2,.3],[.8,.5],[2.,.5]]),
                time=np.linspace(0,6,13),amplitudes=np.array([.12,.08]),fields=target)
            (root/'reference-validation.json').write_text(json.dumps(dict(status='passed',reference=str(reference),identity=identity(c))))
            def executor(**kwargs):return ThreadPoolExecutor(max_workers=kwargs['max_workers'])
            with patch('examples.channel_nsch2d.online.ProcessPoolExecutor',executor),contextlib.redirect_stdout(io.StringIO()):
                entry.train(c)
            run=Path(c.log.log_dir)/'gmls';report=audit_training_inputs(run,c)
            self.assertEqual(report['refresh_count'],2);self.assertEqual(report['installed_inputs'],4)
            summary=json.loads((run/'summary.json').read_text())
            self.assertIn('GMLS PDE/BC first',summary['method'])
            rows=[json.loads(line) for line in (run/'loss.jsonl').read_text().splitlines()]
            self.assertTrue(all(np.isfinite(row['total']) for row in rows))
            self.assertEqual(len(rows[-1]['parts']),9)
            self.assertNotEqual((run/'model_initial.eqx').read_bytes(),(run/'model_final.eqx').read_bytes())


if __name__=='__main__':unittest.main()
