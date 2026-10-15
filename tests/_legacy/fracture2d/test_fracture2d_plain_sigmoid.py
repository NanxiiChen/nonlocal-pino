"""No-reference output algebra, detached H and trainable warm-start conversion."""
import unittest
from unittest.mock import patch
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import csr_matrix

from examples.fracture2d._legacy.compact_damage_features import Readout
from examples.fracture2d._legacy.compact_damage_loading import JointNetwork
from examples.fracture2d._legacy.configs.plain_sigmoid_trial import get_config
from examples.fracture2d._legacy.plain_sigmoid_damage import (
    make_problem, make_gradual_problem, make_transfer_objective, recenter_bias, latent_basis_scaling,
)


class TinyTrunk(eqx.Module):
    output: eqx.nn.Linear

    def __call__(self, x):
        return self.output(x)


class PlainSigmoidTest(unittest.TestCase):
    def problem(self, *, gradual=False):
        jax.config.update('jax_enable_x64', True)
        c = get_config()
        D = jnp.asarray([[2., -1., -1.], [-1., 2., -1.], [-1., -1., 2.]])
        B = np.asarray([[1., .2], [.3, 1.], [.5, .4]])
        s = dict(matrix=csr_matrix(B), operator=None, reference_load=.0048,
                 features=lambda H, load: jnp.column_stack((H, H**2)))
        # No frozen_image or diagonal field is present: output does not use them.
        net = JointNetwork(TinyTrunk(eqx.nn.Linear(2, 1, key=jax.random.PRNGKey(0),
            dtype=jnp.float64)), Readout(jnp.asarray([.1, -.2])))
        with patch('examples.fracture2d._legacy.plain_sigmoid_damage.make_padded_matvec',
                   return_value=(lambda d: D@d, {})):
            factory = make_gradual_problem if gradual else make_problem
            functions = factory(c, s, np.ones(2), 2.)
        return c, D, B, net, functions

    def test_gradual_logits_preserve_warm_start_across_load_and_detach_H(self):
        c,D,B,net,(image,metrics,objectives,latent)=self.problem(gradual=True)
        H=jnp.asarray([1.,2.,4.])
        expected=2*jax.vmap(net.trunk)(jnp.column_stack((H,H**2)))[:,0]+B@net.readout.coefficients
        for load in (jnp.asarray(.0001),jnp.asarray(.001),jnp.asarray(.005)):
            np.testing.assert_allclose(latent(net,H,load),expected,rtol=1e-13)
            d=image(net,H,load)
            gc0,gc1=c.physical.Gc/c.physical.l0,c.physical.Gc*c.physical.l0
            parts=(gc0*d,gc1*(D@d),2*H*(d-1))
            np.testing.assert_allclose(metrics(net,H,load)['AT2_relative'],
                jnp.max(abs(sum(parts)))/sum(jnp.max(abs(p)) for p in parts),rtol=1e-13)
            np.testing.assert_array_equal(jax.grad(lambda h:objectives['residual_l16'](net,h,load))(H),jnp.zeros(3))
            grad=eqx.filter_grad(objectives['residual_l16'])(net,H,load)
            eps=1e-6
            plus=eqx.tree_at(lambda n:n.readout.coefficients,net,net.readout.coefficients+jnp.asarray([eps,0.]))
            minus=eqx.tree_at(lambda n:n.readout.coefficients,net,net.readout.coefficients-jnp.asarray([eps,0.]))
            fd=(objectives['residual_l16'](plus,H,load)-objectives['residual_l16'](minus,H,load))/(2*eps)
            np.testing.assert_allclose(grad.readout.coefficients[0],fd,rtol=1e-6,atol=1e-9)
        self.assertGreater(float(jnp.max(abs(image(net,H*2,load)-image(net,H,load)))),0.)

    def test_plain_output_residual_gradients_and_detached_history(self):
        c, D, B, net, (image, metrics, objectives, latent) = self.problem()
        H, load = jnp.asarray([1., 2., 4.]), jnp.asarray(.0053)
        gc0, gc1 = c.physical.Gc/c.physical.l0, c.physical.Gc*c.physical.l0
        expected = 2*c.physical.E/gc0*load**2*jax.vmap(net.trunk)(jnp.column_stack((H,H**2)))[:,0]
        expected += (load/.0048)**2*(B@net.readout.coefficients)
        np.testing.assert_allclose(latent(net,H,load), expected, rtol=1e-13)
        np.testing.assert_allclose(image(net,H,load), jax.nn.sigmoid(expected), rtol=1e-13)
        for history, applied in [(H,load),(H*2,load*1.1)]:
            d=image(net,history,applied)
            parts=(gc0*d,gc1*(D@d),2*history*(d-1))
            np.testing.assert_allclose(metrics(net,history,applied)['AT2_relative'],
                jnp.max(abs(sum(parts)))/sum(jnp.max(abs(p)) for p in parts), rtol=1e-13)
            direction=jax.tree.map(lambda a:jnp.ones_like(a)*.1,eqx.filter(net,eqx.is_inexact_array))
            for loss in objectives.values():
                gradient=eqx.filter_grad(loss)(net,history,applied)
                analytic=sum(jnp.sum(a*b) for a,b in zip(jax.tree.leaves(gradient),jax.tree.leaves(direction)))
                eps=1e-5
                plus=eqx.apply_updates(net,jax.tree.map(lambda a:eps*a,direction))
                minus=eqx.apply_updates(net,jax.tree.map(lambda a:-eps*a,direction))
                fd=(loss(plus,history,applied)-loss(minus,history,applied))/(2*eps)
                np.testing.assert_allclose(fd,analytic,rtol=1e-6,atol=1e-9)
        np.testing.assert_array_equal(jax.grad(lambda h:jnp.sum(image(net,h,load)))(H),jnp.zeros_like(H))
        np.testing.assert_array_equal(jax.grad(lambda h:objectives['residual_l16'](net,h,load))(H),jnp.zeros_like(H))
        self.assertGreater(float(jnp.max(abs(image(net,H*2,load)-image(net,H,load)))),0.)

    def test_bias_conversion_retains_other_weights_and_remains_trainable(self):
        c, _, _, net, (_,_,_,latent)=self.problem()
        H,load=jnp.asarray([1.,2.,4.]),jnp.asarray(.0053)
        teacher=jnp.asarray([.001,.2,.8])
        converted,target,shift=recenter_bias(net,latent,H,load,teacher,
            trunk_factor=2*c.physical.E/(c.physical.Gc/c.physical.l0)*float(load)**2)
        np.testing.assert_array_equal(converted.trunk.output.weight,net.trunk.output.weight)
        np.testing.assert_array_equal(converted.readout.coefficients,net.readout.coefficients)
        np.testing.assert_allclose(jnp.mean(latent(converted,H,load)),jnp.mean(target),atol=1e-14)
        loss=make_transfer_objective(latent,target)
        grad=eqx.filter_grad(loss)(converted,H,load)
        self.assertGreater(float(jnp.linalg.norm(grad.readout.coefficients)),0.)
        self.assertTrue(np.isfinite(shift))
        with self.assertRaises(ValueError):
            recenter_bias(net,latent,H,load,jnp.asarray([0.,.2,.8]),trunk_factor=1.)
        B=csr_matrix(np.asarray([[1.,.2],[.3,1.],[.5,.4]]))
        old_scale=np.asarray([3.,1e8])
        scaled,new_scale=latent_basis_scaling(net,B,old_scale)
        np.testing.assert_allclose(B@(old_scale*np.asarray(net.readout.coefficients)),
                                   B@(new_scale*np.asarray(scaled.readout.coefficients)),rtol=1e-14)
        np.testing.assert_allclose(np.asarray(B.multiply(new_scale).power(2).mean(axis=0)).ravel(),np.ones(2))


if __name__ == '__main__':
    unittest.main()
