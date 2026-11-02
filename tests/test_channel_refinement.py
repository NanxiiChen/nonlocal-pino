"""Global gradient accumulation and explicit unchanged-parent provenance."""
from pathlib import Path
import tempfile
import unittest
import equinox as eqx
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from ml_collections import ConfigDict
import numpy as np
from examples.channel_nsch2d.refinement import make_accumulated_objective,parent_spec
from examples.channel_nsch2d.configs.train_single_conservative import get_config
from examples.channel_nsch2d.data import geometry_data
from examples.channel_nsch2d.network import build_network
from examples.channel_nsch2d.losses import make_loss_fn
from tests.test_channel_continuation import parent_fixture


class ChannelRefinement(unittest.TestCase):
    def test_time_accumulation_matches_monolithic_value_parts_and_gradient(self):
        model=eqx.nn.Linear(2,2,key=jax.random.PRNGKey(18))
        nodes=np.arange(1,9,dtype=np.float32)/8
        def loss(net,data,times):
            predicted=jax.vmap(net)(jnp.stack((jnp.sin(times),times**2),axis=-1))
            target=jnp.stack((jnp.cos(times),times**3),axis=-1)
            parts=jnp.mean((predicted-target)**2,axis=0)
            return jnp.sum(parts),parts
        p,static=eqx.partition(model,eqx.is_inexact_array);vector,unravel=ravel_pytree(p)
        direct=jax.jit(jax.value_and_grad(lambda v:loss(eqx.combine(unravel(v),static),None,jnp.asarray(nodes)),has_aux=True))
        (value,parts),gradient=direct(vector)
        for size in (2,4,8):
            initial,restore,evaluate=make_accumulated_objective(model,loss,None,nodes,size)
            v,g,p=evaluate(initial)
            np.testing.assert_allclose(v,value,atol=2e-7,rtol=2e-6)
            np.testing.assert_allclose(p,parts,atol=2e-7,rtol=2e-6)
            np.testing.assert_allclose(g,gradient,atol=2e-7,rtol=2e-6)
            for a,b in zip(jax.tree.leaves(model),jax.tree.leaves(restore(initial)),strict=True):
                if eqx.is_array(a):np.testing.assert_array_equal(a,b)
        with self.assertRaisesRegex(ValueError,'batches'):make_accumulated_objective(model,loss,None,nodes,3)

    def test_actual_mixed_nsch_gradient_includes_all_current_history_states(self):
        c=get_config();c.sample.Nx=8;c.sample.Ny=8;c.sample.boundary_points=4
        c.arch.width=4;c.arch.heads=1;c.arch.condition_depth=1;c.arch.decoder_depth=1
        c.arch.spatial_features=2;c.arch.time_features=2
        raw,_=geometry_data((.12,.08),c)
        data={k:jnp.asarray(v[None],dtype=v.dtype if np.issubdtype(v.dtype,np.integer) else jnp.float32) for k,v in raw.items()}
        model=build_network(c,jax.random.PRNGKey(42));loss=make_loss_fn(c)
        nodes=np.array([.05,.5,1.,6.],dtype=np.float32)
        initial,restore,accumulate=make_accumulated_objective(model,loss,data,nodes,2)
        direct=jax.jit(jax.value_and_grad(lambda v:loss(restore(v),data,jnp.asarray(nodes)),has_aux=True))
        (v,p),g=direct(jnp.asarray(initial,dtype=jnp.float32));a,b,d=accumulate(initial)
        np.testing.assert_allclose(a,v,atol=5e-6,rtol=1e-5)
        np.testing.assert_allclose(d,p,atol=5e-6,rtol=1e-5)
        np.testing.assert_allclose(b,g,atol=5e-5,rtol=2e-4)
        self.assertGreater(np.linalg.norm(b),0.)

    def test_parent_contract_rejects_physics_or_optimizer_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);c,_,_,_,_=parent_fixture(root)
            np.savez(root/'known_inputs.npz',a=np.zeros((1,2)))
            np.savez(root/'sampling.npz',a=np.zeros((1,2)))
            (root/'operators.json').write_text('[]')
            c.train.refinement=ConfigDict(dict(parent_run=directory))
            p,s=parent_spec(c);self.assertEqual(p,root);self.assertEqual(s['iterations'],3)
            c.physical.mobility*=2
            with self.assertRaisesRegex(ValueError,'preserve'):parent_spec(c)
            c.physical.mobility/=2;c.train.lr_init*=.5
            with self.assertRaisesRegex(ValueError,'preserve'):parent_spec(c)


if __name__=='__main__':unittest.main()
