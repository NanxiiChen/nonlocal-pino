"""Local boundary closure, independent FD reference, and six-loss training AD."""
import unittest

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy import sparse

from examples.corrosion2d import residual_experiment as r
from examples.corrosion2d.configs.train_local_tensor import get_config
from examples.corrosion2d.local_ad import local_loss_factory
from examples.corrosion2d.local_reference import negative_laplacian,local_spec
from examples.corrosion2d.residuals import normalized_data_fields
from nonlocal_pinn.nn.constraints import ConstrainedForward,kks_constraint


class Raw(eqx.Module):
    a:jax.Array

    def __call__(self,x,t):
        return jnp.stack([self.a*x[0]+.3*x[1]**2+.2*t,.1*x[0]**2+.2*x[1]**2+.1*t-.2])


class CorrosionLocalADTests(unittest.TestCase):
    def test_independent_laplacian_constant_polynomial_and_weighted_symmetry(self):
        nx,ny=11,6
        D=negative_laplacian(nx,ny)
        x,y=np.meshgrid(np.linspace(0,1,nx),np.linspace(0,.5,ny),indexing='xy')
        np.testing.assert_allclose(D@np.ones(x.size),0,atol=1e-12)
        values=(x*x+y*y).ravel();actual=(D@values).reshape(y.shape)
        np.testing.assert_allclose(actual[:-1,1:-1],-4,rtol=0,atol=1e-12)
        wx=np.ones(nx);wy=np.ones(ny);wx[[0,-1]]=.5;wy[[0,-1]]=.5
        WD=sparse.diags(np.outer(wy,wx).ravel())@D
        np.testing.assert_allclose((WD-WD.T).toarray(),0,atol=1e-12)
        import json
        c=get_config();self.assertEqual(local_spec(c),json.loads(json.dumps(local_spec(c))))

    def test_no_flux_and_actual_six_loss_soap_update(self):
        c=get_config()
        with jax.enable_x64(True):
            net=ConstrainedForward(Raw(jnp.array(.7)),lambda raw:kks_constraint(raw,1.,c.physical.c_le))
            points=jnp.array([[.2,.1],[.5,.2],[.8,.1]])
            ts=jnp.array([.2,.7]);neu=jnp.array([[.1,0.],[.9,0.]])
            ic=jax.vmap(net,in_axes=(0,None))(points,jnp.array(0.))
            direction=jnp.zeros_like(neu).at[:,1].set(1.)
            normal=jax.jvp(lambda xx:normalized_data_fields(net,xx,ts),(neu,),(direction,))[1]
            np.testing.assert_array_equal(normal,0.)
            loss,_=local_loss_factory(c,r.nondim(c),points,ic,points[:1],points[-1:],neu,'local_tensor')
            mask=jnp.array([True,True,False]);values=loss(net,ts,points,None,mask)
            self.assertEqual(values.shape,(6,));self.assertEqual(float(values[-1]),0.)
            optimizer=optax.sgd(1e-5)
            state=optimizer.init(eqx.filter(net,eqx.is_array))
            step=r.make_step(loss,optimizer,c)
            after,state,weights,values,finite=step(net,state,jnp.ones(6),ts,points,None,mask,jnp.array(True),jnp.ones(6,dtype=bool))
            self.assertTrue(bool(finite));self.assertEqual(weights.shape,(6,))
            self.assertTrue(np.isfinite(np.asarray(after.net.a)))
            with self.assertRaisesRegex(ValueError,'length6'):
                step(net,state,jnp.ones(6),ts,points,None,mask,jnp.array(False),jnp.ones(5,dtype=bool))


if __name__=='__main__':unittest.main()
