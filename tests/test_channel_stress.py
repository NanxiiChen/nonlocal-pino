"""Mixed NS equivalence, exact physical-state transfer and full-time gradients."""
from pathlib import Path
import tempfile
import unittest
import equinox as eqx
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
import numpy as np
from examples.channel_nsch2d.configs.train_single_conservative import get_config
from examples.channel_nsch2d.data import geometry_data
from examples.channel_nsch2d.network import build_network,encode,field_jets,extend_stress_outputs,stresses
from examples.channel_nsch2d.losses import make_loss_fn,part_names,residual_components
from examples.channel_nsch2d.stress import constitutive,divergence
from examples.channel_nsch2d.problem import validate_config
from examples.channel_nsch2d.refinement import parent_spec,make_accumulated_objective
from tests.test_channel_continuation import parent_fixture
from tests.test_channel_momentum_closure import manufactured


def configs():
    c=get_config();c.sample.Nx=8;c.sample.Ny=8;c.sample.boundary_points=4
    c.arch.width=4;c.arch.heads=1;c.arch.condition_depth=1;c.arch.decoder_depth=1
    c.arch.spatial_features=2;c.arch.time_features=2
    new=ConfigDict(c.to_dict());new.stress_formulation=ConfigDict(dict(kind='cauchy_mixed',
        initialization='preserve_original',constitutive_scale=1.))
    return c,new


class ChannelStress(unittest.TestCase):
    def test_manufactured_constitutive_and_momentum_equivalence(self):
        with jax.enable_x64(True):
            c=get_config();points=np.array([[.2,.3],[.8,.7],[2.1,.6]]);time=np.array([.2,.7])
            value,dt,dx,dy=manufactured(points,time);x,y=points.T;t=time[:,None];nu=c.physical.viscosity
            p=value[...,4];s=np.stack((-p+8*nu*t*x*y,2*nu*t*(x*x-y*y),-p-8*nu*t*x*y),axis=-1)
            sx=np.stack((-3*t*x*x+8*nu*t*y,4*nu*t*x,-3*t*x*x-8*nu*t*y),axis=-1)
            sy=np.stack((-4*t*y+8*nu*t*x,-4*nu*t*y,-4*t*y-8*nu*t*x),axis=-1)
            np.testing.assert_allclose(-.5*(s[...,0]+s[...,2]),p,atol=2e-15)
            np.testing.assert_allclose(constitutive(s,dx,dy,nu,xp=np),0.,atol=2e-15)
            lap=np.zeros_like(value);lap[...,2]=4*t*y;lap[...,3]=-4*t*x
            old=residual_components(value,dt,dx,dy,lap,c)
            new=residual_components(value,dt,dx,dy,lap,c,stress_divergence=divergence(sx,sy,xp=np))
            np.testing.assert_allclose(new,old,atol=3e-15,rtol=3e-15)

    def test_parent_arrays_physical_jets_and_cold_initialization_are_preserved(self):
        c,new=configs();c.train.compute_dtype='float64';new.train.compute_dtype='float64'
        with jax.enable_x64(True):
            validate_config(new);old=build_network(c,jax.random.PRNGKey(13))
            template=build_network(new,jax.random.PRNGKey(99));model=extend_stress_outputs(old,template)
            before=jax.tree_util.tree_flatten_with_path(old)[0];after=jax.tree_util.tree_flatten_with_path(model)[0]
            for (pa,a),(pb,b) in zip(before,after,strict=True):
                self.assertEqual(jax.tree_util.keystr(pa),jax.tree_util.keystr(pb))
                if eqx.is_array(a) and '.output.' not in jax.tree_util.keystr(pa):np.testing.assert_array_equal(a,b)
            np.testing.assert_array_equal(model.output.weight[4],0.)
            raw,_=geometry_data((.12,.08),c);data={k:jnp.asarray(v) for k,v in raw.items()}
            points=jnp.array([[.8,.4],[2.,.45],[4.,.6]]);times=jnp.array([0.,.025,.5,6.])
            a=eqx.filter_jit(lambda n:field_jets(n,encode(n,data),points,times,data['amplitudes'],c))(old)
            b=eqx.filter_jit(lambda n:field_jets(n,encode(n,data),points,times,data['amplitudes'],new))(model)
            for x,y in zip(a,b,strict=True):np.testing.assert_allclose(x,y,atol=2e-13,rtol=2e-13)
            sigma=stresses(model,encode(model,data),points,times,new)
            np.testing.assert_allclose(-.5*(sigma[...,0]+sigma[...,2]),a[0][...,4],atol=2e-14)
            cold=build_network(new,jax.random.PRNGKey(13))
            for x,y in zip(jax.tree.leaves(model),jax.tree.leaves(cold),strict=True):
                if eqx.is_array(x):np.testing.assert_array_equal(x,y)

    def test_actual_complete_pde_gradient_and_time_accumulation(self):
        _,c=configs();raw,_=geometry_data((.12,.08),c)
        batch={k:jnp.asarray(v[None],dtype=v.dtype if np.issubdtype(v.dtype,np.integer) else jnp.float32) for k,v in raw.items()}
        model=build_network(c,jax.random.PRNGKey(42));loss=make_loss_fn(c)
        nodes=np.array([.05,.5,1.,6.],dtype=np.float32)
        vector,restore,evaluate=make_accumulated_objective(model,loss,batch,nodes,2)
        direct=jax.jit(jax.value_and_grad(lambda x:loss(restore(x),batch,jnp.asarray(nodes)),has_aux=True))
        (v,p),g=direct(jnp.asarray(vector,dtype=jnp.float32));a,b,d=evaluate(vector)
        self.assertEqual(len(part_names(c)),13);self.assertEqual(len(d),13)
        np.testing.assert_allclose(a,v,atol=5e-6,rtol=2e-5)
        np.testing.assert_allclose(d,p,atol=5e-6,rtol=2e-5)
        np.testing.assert_allclose(b,g,atol=8e-5,rtol=3e-4)
        self.assertTrue(np.isfinite(b).all())
        (_,parts),gradient=eqx.filter_jit(eqx.filter_value_and_grad(loss,has_aux=True))(model,batch,jnp.asarray(nodes[:2]))
        self.assertGreater(np.linalg.norm(np.asarray(gradient.output.weight[4])),0.)
        self.assertGreater(float(jnp.sum(parts[-4:])),0.)

    def test_parent_guard_rejects_other_numerical_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);c,_,_,_,_=parent_fixture(root)
            np.savez(root/'known_inputs.npz',a=np.zeros((1,2)));np.savez(root/'sampling.npz',a=np.zeros((1,2)))
            (root/'operators.json').write_text('[]')
            c.train.refinement=ConfigDict(dict(parent_run=directory))
            c.stress_formulation=ConfigDict(dict(kind='cauchy_mixed',initialization='preserve_original',constitutive_scale=1.))
            self.assertEqual(parent_spec(c)[0],root)
            c.nl.degree+=1
            with self.assertRaisesRegex(ValueError,'preserve'):parent_spec(c)
            c.nl.degree-=1;c.loss.chemical_scale*=2
            with self.assertRaisesRegex(ValueError,'preserve'):parent_spec(c)

    def test_soap_takes_actual_mixed_stress_pde_updates(self):
        from nonlocal_pinn.training.optimizers import make_optimizer
        _,c=configs();c.train.optimizer='soap'
        raw,_=geometry_data((.12,.08),c)
        batch={k:jnp.asarray(v[None],dtype=v.dtype if np.issubdtype(v.dtype,np.integer) else jnp.float32) for k,v in raw.items()}
        model=build_network(c,jax.random.PRNGKey(42));optimizer=make_optimizer(c)
        state=optimizer.init(eqx.filter(model,eqx.is_array))
        fn=eqx.filter_jit(eqx.filter_value_and_grad(make_loss_fn(c),has_aux=True))
        for _ in range(3):
            (value,parts),gradient=fn(model,batch,jnp.array([.05,3.],dtype=jnp.float32))
            updates,state=optimizer.update(gradient,state,model);model=eqx.apply_updates(model,updates)
            self.assertTrue(np.isfinite(float(value)));self.assertEqual(parts.shape,(13,))
            self.assertTrue(all(np.isfinite(np.asarray(a)).all() for a in jax.tree.leaves((model,state)) if eqx.is_array(a)))


if __name__=='__main__':unittest.main()
