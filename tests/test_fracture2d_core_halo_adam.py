"""Changing random batches must preserve Adam state and functional parameters."""
import unittest
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from examples.fracture2d.core_halo_adam import make_adam

jax.config.update('jax_enable_x64',True)


class Parameters(eqx.Module):
    value:jax.Array


class PersistentAdamTest(unittest.TestCase):
    def test_state_survives_batch_change_and_matches_sequential_optax(self):
        objective=lambda net,context,load:jnp.sum((net.value-context['target'])**2)
        optimizer,step,_=make_adam(objective,learning_rate=.01,decay_steps=10,decay_rate=.5,
            minimum_rate=.005,b1=.9,b2=.999,eps=1e-8,eps_root=0.,grad_clip=1000.)
        initial=Parameters(jnp.array([.2,-.1],jnp.float64))
        state=optimizer.init(eqx.filter(initial,eqx.is_inexact_array));reference=initial;reference_state=state
        current=initial
        for target in [jnp.array([1.,2.]),jnp.array([-2.,.5])]:
            context={'target':target}
            current,state,value,norm=step(current,state,context,jnp.asarray(.0053))
            grad=eqx.filter_grad(objective)(reference,context,jnp.asarray(.0053))
            updates,reference_state=optimizer.update(grad,reference_state,eqx.filter(reference,eqx.is_inexact_array))
            reference=eqx.apply_updates(reference,updates)
            np.testing.assert_allclose(current.value,reference.value,rtol=1e-14,atol=1e-14)
            self.assertTrue(np.isfinite(float(value)) and np.isfinite(float(norm)))
        self.assertEqual(int(state[1][0].count),2)
        self.assertEqual(int(state[1][1].count),2)
        first,first_state,_,_=step(initial,optimizer.init(eqx.filter(initial,eqx.is_inexact_array)),
            {'target':jnp.array([1.,2.])},jnp.asarray(.0053))
        reset,_,_,_=step(first,optimizer.init(eqx.filter(first,eqx.is_inexact_array)),
            {'target':jnp.array([-2.,.5])},jnp.asarray(.0053))
        self.assertGreater(float(jnp.linalg.norm(reset.value-current.value)),1e-4)
        np.testing.assert_array_equal(initial.value,jnp.array([.2,-.1]))
        for a,b in zip(jax.tree.leaves(state),jax.tree.leaves(reference_state)):
            np.testing.assert_allclose(a,b,rtol=1e-14,atol=1e-14)

    def test_nonfinite_gradient_is_exposed_not_sanitized(self):
        objective=lambda net,context,load:jnp.sum(net.value)*context['scale']
        optimizer,step,_=make_adam(objective,learning_rate=.01,decay_steps=10,decay_rate=.5,
            minimum_rate=.005,b1=.9,b2=.999,eps=1e-8,eps_root=0.,grad_clip=1000.)
        net=Parameters(jnp.array([.2,-.1],jnp.float64))
        _,_,value,norm=step(net,optimizer.init(eqx.filter(net,eqx.is_inexact_array)),
            {'scale':jnp.asarray(jnp.nan)},jnp.asarray(.0053))
        self.assertFalse(np.isfinite(float(value)))
        self.assertFalse(np.isfinite(float(norm)))


if __name__=='__main__':unittest.main()
