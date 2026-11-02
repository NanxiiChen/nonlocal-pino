"""Quartic polynomial reproduction and finite physical mixed-PDE gradients."""
import unittest
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from examples.channel_nsch2d.configs.train_single_gmls4 import get_config
from examples.channel_nsch2d.data import geometry_data
from examples.channel_nsch2d.losses import make_loss_fn
from examples.channel_nsch2d.network import build_network


class ChannelOperatorRefinement(unittest.TestCase):
    def test_quartic_reproduction_and_float32_full_parameter_gradient(self):
        c=get_config();del c.train.continuation
        c.sample.Nx=16;c.sample.Ny=8;c.sample.Nt=2;c.sample.boundary_points=6
        c.arch.width=4;c.arch.heads=1;c.arch.condition_depth=1;c.arch.decoder_depth=1
        c.arch.spatial_features=2;c.arch.time_features=2
        raw,report=geometry_data((.12,.08),c);x,y=raw['points'].T
        for value,target in ((x**4,12*x*x),(x*x*y*y,2*(x*x+y*y)),(x**3*y,6*x*y)):
            lap=np.sum(raw['laplacian']*(value[raw['neighbors']]-value[:,None]),axis=1)
            np.testing.assert_allclose(lap,target,atol=2e-7,rtol=0)
        self.assertLess(report['maximum_condition'],c.nl.condition_limit)
        data={k:jnp.asarray(v,dtype=v.dtype if np.issubdtype(v.dtype,np.integer) else jnp.float32) for k,v in raw.items()}
        net=build_network(c,jax.random.PRNGKey(42));batch=jax.tree.map(lambda a:a[None],data)
        (loss,_),gradient=eqx.filter_jit(eqx.filter_value_and_grad(make_loss_fn(c),has_aux=True))(net,batch,jnp.array([.05,3.],dtype=jnp.float32))
        self.assertTrue(np.isfinite(float(loss)))
        arrays=[v for v in jax.tree.leaves(gradient) if eqx.is_array(v)]
        self.assertTrue(all(np.asarray(v).dtype==np.float32 and np.isfinite(np.asarray(v)).all() for v in arrays))
        self.assertGreater(sum(float(jnp.sum(v*v)) for v in arrays),0.)


if __name__=='__main__':unittest.main()
