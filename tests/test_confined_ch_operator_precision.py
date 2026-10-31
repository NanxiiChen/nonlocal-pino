"""Full precision paths, shared draws and independent physical-field checks."""
import unittest
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict
from examples.confined_ch2d.configs.train_operator_smallscale10_512_width128_beta09 import get_config
from examples.confined_ch2d.operator_precision import compute_arrays
from examples.confined_ch2d.operator_training import build_network,fields,make_loss,operator_temporal_samples,prepare,phase
from examples.confined_ch2d.operator_problem import validate
from examples.confined_ch2d.operator_evaluate import check_chemical_relation
from examples.confined_ch2d.operator_precision import make_compute_optimizer
from test_confined_ch_operator import small_fixture


class OperatorPrecisionTests(unittest.TestCase):
    def test_rounding_time_jvp_gradients_and_soap_refresh_are_fp32(self):
        with jax.enable_x64(True):
            c=get_config();c.train.seed=42
            c.arch.width=4;c.arch.heads=1;c.arch.condition_depth=1;c.arch.decoder_depth=1
            c.arch.spatial_features=2;c.arch.time_features=2
            c64=ConfigDict(c.to_dict());c.train.compute_dtype='float32';validate(c)
            _,data=small_fixture();data=compute_arrays(data,c)
            batch=jax.tree.map(lambda a:a[None],data)
            key=jax.random.key(3);a=build_network(c,key);b=build_network(c64,key)
            for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b),strict=True):
                if eqx.is_array(x):np.testing.assert_array_equal(x,np.asarray(y).astype(np.float32))
            times,_=operator_temporal_samples(c,2,key);old,_=operator_temporal_samples(c64,2,key)
            np.testing.assert_array_equal(times,np.asarray(old).astype(np.float32))
            times=jnp.array([0.,.03],dtype=jnp.float32)
            h=prepare(a,data)
            derivative=jax.jvp(lambda t:phase(a,h,t,data,c),(times[1],),(jnp.ones_like(times[1]),))[1]
            direct=jax.jacfwd(lambda t:phase(a,h,t,data,c))(times[1])
            np.testing.assert_allclose(derivative,direct,rtol=1e-6,atol=1e-7)
            prediction=fields(a,data,times.astype(jnp.float64),c)
            self.assertEqual(prediction.dtype,jnp.float32)
            np.testing.assert_array_equal(prediction[0,:,0],data['initial'])
            opt=make_compute_optimizer(c);state=opt.init(eqx.filter(a,eqx.is_array));loss=make_loss(c)
            @eqx.filter_jit
            def step(model,state):
                value,gradient=eqx.filter_value_and_grad(loss)(model,batch,times)
                updates,state=opt.update(gradient,state,model)
                return eqx.apply_updates(model,updates),state,value,gradient
            ir=step.lower(a,state).as_text()
            self.assertTrue('xf64>' not in ir,'Unexpected non-scalar FP64 tensor in training IR')
            for _ in range(7):
                a,state,value,gradient=step(a,state)
                for tree in (a,state,gradient,data,value):
                    for leaf in jax.tree.leaves(tree):
                        if eqx.is_inexact_array(leaf):
                            self.assertEqual(leaf.dtype,jnp.float32)
                            self.assertTrue(np.isfinite(leaf).all())

    def test_fp32_saved_chemical_bound_rejects_physical_inconsistency(self):
        c=get_config();c.train.compute_dtype='float32'
        reference=dict(neighbors=np.array([[1],[0]]),coefficients=np.array([[3.],[3.]]))
        phi=np.array([[.12345,-.23456],[.11111,-.22222]],dtype=np.float32)
        action=np.sum(reference['coefficients'].astype(np.float32)[None]*(phi[:,:,None]-phi[:,reference['neighbors']]),axis=-1)
        mu=c.physical.beta*(phi**3-phi)+c.physical.epsilon**2*action
        checked,_=check_chemical_relation(phi,mu,reference,c.to_dict())
        np.testing.assert_array_equal(checked,mu)
        with self.assertRaisesRegex(AssertionError,'roundoff bound'):
            check_chemical_relation(phi,mu+.001,reference,c.to_dict())
        c.train.compute_dtype='float64';phi=phi.astype(np.float64)
        action=np.sum(reference['coefficients'][None]*(phi[:,:,None]-phi[:,reference['neighbors']]),axis=-1)
        exact=c.physical.beta*(phi**3-phi)+c.physical.epsilon**2*action
        check_chemical_relation(phi,exact,reference,c.to_dict())
        with self.assertRaises(AssertionError):check_chemical_relation(phi,exact+1e-8,reference,c.to_dict())
        c.train.compute_dtype='float16'
        with self.assertRaisesRegex(ValueError,'compute_dtype'):validate(c)


if __name__=='__main__':unittest.main()
