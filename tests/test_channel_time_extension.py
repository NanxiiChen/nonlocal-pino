"""Function-preserving extra time coordinates, without losing old Fourier modes."""
from pathlib import Path
import tempfile
import unittest
import equinox as eqx
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
import numpy as np
from examples.channel_nsch2d.configs.train_single_conservative import get_config
from examples.channel_nsch2d.network import build_network,encode,field_jets,extend_time_basis,temporal_features
from examples.channel_nsch2d.problem import validate_config
from examples.channel_nsch2d.data import geometry_data
from examples.channel_nsch2d.losses import make_loss_fn
from examples.channel_nsch2d.refinement import parent_spec
from tests.test_channel_continuation import parent_fixture


def configs():
    c=get_config();c.sample.Nx=8;c.sample.Ny=8;c.sample.boundary_points=4
    c.arch.width=4;c.arch.heads=1;c.arch.condition_depth=1;c.arch.decoder_depth=1
    c.arch.spatial_features=2;c.arch.time_features=2
    new=ConfigDict(c.to_dict());new.arch.time_features=4
    new.time_embedding=ConfigDict(dict(kind='linear_log',scale=.25,initialization='preserve_linear'))
    return c,new


class ChannelTimeExtension(unittest.TestCase):
    def test_all_shared_arrays_come_from_parent_even_with_different_template_seed(self):
        c,new=configs();parent=build_network(c,jax.random.PRNGKey(13))
        extended=extend_time_basis(parent,build_network(new,jax.random.PRNGKey(99)))
        a=[(jax.tree_util.keystr(p),v) for p,v in jax.tree_util.tree_flatten_with_path(parent)[0] if eqx.is_array(v)]
        b=[(jax.tree_util.keystr(p),v) for p,v in jax.tree_util.tree_flatten_with_path(extended)[0] if eqx.is_array(v)]
        self.assertEqual([p for p,_ in a],[p for p,_ in b])
        for (path,x),(_,y) in zip(a,b,strict=True):
            if path!='.time.layers[0].weight':np.testing.assert_array_equal(x,y)

    def test_cold_initialization_and_physical_time_spatial_jets_preserve_all_old_modes(self):
        c,new=configs();validate_config(new)
        key=jax.random.PRNGKey(42);old=build_network(c,key);extended=build_network(new,key)
        times=jnp.array([0.,.025,.1,.5,3.,6.],dtype=jnp.float32)
        a=jax.vmap(old.time)(temporal_features(old,times,c))
        b=jax.vmap(extended.time)(temporal_features(extended,times,new))
        np.testing.assert_allclose(a,b,atol=2e-7,rtol=2e-6)
        np.testing.assert_array_equal(extended.time.layers[0].weight[:,3:5],0.)
        np.testing.assert_array_equal(extended.time.layers[0].weight[:,7:9],0.)
        raw,_=geometry_data((.12,.08),c)
        data={k:jnp.asarray(v,dtype=v.dtype if np.issubdtype(v.dtype,np.integer) else jnp.float32) for k,v in raw.items()}
        points=jnp.array([[.8,.4],[2.,.45],[4.,.6]],dtype=jnp.float32)
        before=eqx.filter_jit(lambda m:field_jets(m,encode(m,data),points,times,data['amplitudes'],c))(old)
        after=eqx.filter_jit(lambda m:field_jets(m,encode(m,data),points,times,data['amplitudes'],new))(extended)
        for x,y in zip(before,after,strict=True):np.testing.assert_allclose(x,y,atol=3e-6,rtol=3e-5)

    def test_transferred_parent_and_new_time_columns_have_finite_nonzero_pde_gradient(self):
        c,new=configs();parent=build_network(c,jax.random.PRNGKey(13))
        model=extend_time_basis(parent,build_network(new,jax.random.PRNGKey(99)))
        np.testing.assert_array_equal(model.time.layers[0].weight[:,:3],parent.time.layers[0].weight[:,:3])
        np.testing.assert_array_equal(model.time.layers[0].weight[:,5:7],parent.time.layers[0].weight[:,3:5])
        raw,_=geometry_data((.12,.08),c)
        batch={k:jnp.asarray(v[None],dtype=v.dtype if np.issubdtype(v.dtype,np.integer) else jnp.float32) for k,v in raw.items()}
        (value,_),gradient=eqx.filter_jit(eqx.filter_value_and_grad(make_loss_fn(new),has_aux=True))(
            model,batch,jnp.array([.05,3.],dtype=jnp.float32))
        self.assertTrue(np.isfinite(float(value)))
        arrays=[a for a in jax.tree.leaves(gradient) if eqx.is_array(a)]
        self.assertTrue(all(np.isfinite(np.asarray(a)).all() for a in arrays))
        g=np.asarray(gradient.time.layers[0].weight)
        self.assertGreater(np.linalg.norm(g[:,3:5])+np.linalg.norm(g[:,7:9]),0.)

    def test_expansion_parent_guard_still_rejects_other_numerical_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);c,_,_,_,_=parent_fixture(root)
            np.savez(root/'known_inputs.npz',a=np.zeros((1,2)));np.savez(root/'sampling.npz',a=np.zeros((1,2)))
            (root/'operators.json').write_text('[]')
            c.train.refinement=ConfigDict(dict(parent_run=directory))
            c.arch.time_features*=2
            c.time_embedding=ConfigDict(dict(kind='linear_log',scale=.25,initialization='preserve_linear'))
            self.assertEqual(parent_spec(c)[0],root)
            c.arch.time_features-=1
            with self.assertRaisesRegex(ValueError,'function-preserving'):parent_spec(c)
            c.arch.time_features+=1;c.physical.mobility*=2
            with self.assertRaisesRegex(ValueError,'preserve'):parent_spec(c)


if __name__=='__main__':unittest.main()
