"""Independent two-loss gradients, weight recurrence and stagger masks."""
import unittest
import numpy as np
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from examples.confined_ch2d.loss_balancing import make_balanced_step,frozen_joint_loss,active_losses
from examples.confined_ch2d.configs.train_coarsening_mixed_gradnorm import get_config
from examples.confined_ch2d.configs.train_coarsening_mixed_stagger import get_config as stagger_config
from examples.confined_ch2d.problem import validate_config,initial_field
from examples.confined_ch2d.train import _build_network
from examples.confined_ch2d.losses import make_loss_fn
from examples.confined_ch2d.network import predict_fields
from examples.confined_ch2d.operators import RegionalLaplacian
from tests.test_confined_ch import fixture
from examples.confined_ch2d.native_reference import native_matrix
from jax.flatten_util import ravel_pytree


class Toy(eqx.Module):
    theta:jax.Array


def toy_parts(theta):
    a,b,d=theta
    return jnp.stack(((a+2*b-1)**2+.2*d*d,(3*a-b+.5*d+2)**2+.4*b*b))


class BalancingTests(unittest.TestCase):
    def test_raw_residual_known_initial_scaling_and_parameter_derivative(self):
        from examples.confined_ch2d.configs.train_coarsening_mixed_gradnorm_raw import get_config as raw_config
        with jax.enable_x64(True):
            c=raw_config();c.arch.rank=4;c.arch.hidden_dim=8;c.arch.num_layers=2;c.arch.spatial_modes=2;c.nl.delta=.2
            validate_config(c);_,points,volumes=fixture();x=jnp.asarray(points);v=jnp.asarray(volumes)
            A=native_matrix(points,volumes,c).toarray();D=RegionalLaplacian(points,volumes,c)
            mean=jnp.dot(v,initial_field(x,c,xp=jnp))/v.sum();net=_build_network(c,jax.random.PRNGKey(3))
            loss=make_loss_fn(x,v,D,None,None,None,c);times=jnp.asarray([.001,.2,1.5]);expected=[]
            for time in np.r_[0.,times]:
                fields,dt=jax.jvp(lambda t:predict_fields(net,x,t,mean,v,D),(jnp.asarray(time),),(jnp.asarray(1.),))
                phi,mu=np.asarray(fields).T
                evolution=(np.asarray(dt[:,0])+c.physical.mobility*(A@mu))/100.
                chemical=(mu-c.physical.beta*(phi**3-phi)-c.physical.epsilon**2*(A@phi))/.5
                expected.append([np.dot(volumes,evolution**2)/volumes.sum(),np.dot(volumes,chemical**2)/volumes.sum()])
            value,parts=loss(net,times,mean);np.testing.assert_allclose(parts,np.mean(expected,axis=0),rtol=1e-12)
            theta,unravel=ravel_pytree(eqx.filter(net,eqx.is_inexact_array));_,static=eqx.partition(net,eqx.is_inexact_array)
            objective=eqx.filter_jit(lambda t:loss(eqx.combine(unravel(t),static),times,mean)[0])
            direction=np.random.default_rng(4).normal(size=theta.shape);direction/=np.linalg.norm(direction)
            h=1e-5;fd=(objective(theta+h*direction)-objective(theta-h*direction))/(2*h)
            grad=jax.grad(objective)(theta)
            np.testing.assert_allclose(jnp.dot(grad,direction),fd,rtol=2e-5,atol=1e-7)

    def test_gradient_norm_recurrence_stopped_weights_and_frozen_refinement(self):
        with jax.enable_x64(True):
            c=get_config();c.train.loss_balance.update_every=3
            c.loss.phase_weight=2.;c.loss.chemical_weight=3.
            net=Toy(jnp.asarray([.4,-.2,.3]));optimizer=optax.sgd(.001)
            state=optimizer.init(eqx.filter(net,eqx.is_array))
            weights=jnp.asarray([1.,1.]);norms=jnp.zeros(2)
            loss=lambda net,t,m:(jnp.dot(toy_parts(net.theta),jnp.asarray([2.,3.])),toy_parts(net.theta))
            step=make_balanced_step(loss,optimizer,c)
            for iteration in range(5):
                a,b,d=np.asarray(net.theta)
                z=a+2*b-1;w=3*a-b+.5*d+2
                jac=np.asarray([[2*z,4*z,.4*d],[6*w,-2*w+.8*b,w]])
                expected_norms=np.clip(np.sum(abs(jac),axis=1),1e-6,1e6) if iteration%3==0 else np.asarray(norms)
                expected_weights=np.clip(expected_norms.mean()/expected_norms,1e-6,10000.) if iteration%3==0 else np.asarray(weights)
                expected_gradient=(expected_weights*np.asarray([2.,3.]))@jac
                previous=np.asarray(net.theta)
                net,state,weights,norms,value,parts,effective,active=step(net,state,weights,norms,jnp.ones(2),jnp.asarray(0.),jnp.asarray(iteration))
                np.testing.assert_allclose(weights,expected_weights,rtol=1e-13)
                np.testing.assert_allclose(norms,expected_norms,rtol=1e-13)
                np.testing.assert_allclose(net.theta,previous-.001*expected_gradient,rtol=1e-13,atol=1e-14)
                np.testing.assert_allclose(value,np.dot(expected_weights*[2.,3.],parts),rtol=1e-13)
                self.assertTrue(np.asarray(active).all())
            objective=frozen_joint_loss(loss,weights,c)
            actual_grad=eqx.filter_grad(lambda n:objective(n,None,None)[0])(net).theta
            direction=jnp.asarray([.4,-.1,.2]);h=1e-5
            fd=(objective(Toy(net.theta+h*direction),None,None)[0]-objective(Toy(net.theta-h*direction),None,None)[0])/(2*h)
            np.testing.assert_allclose(jnp.dot(actual_grad,direction),fd,rtol=2e-8)

    def test_stagger_real_mixed_pde_gradient_and_all_stages(self):
        with jax.enable_x64(True):
            c=stagger_config();c.arch.rank=4;c.arch.hidden_dim=8;c.arch.num_layers=2;c.arch.spatial_modes=2;c.nl.delta=.2
            validate_config(c);_,points,volumes=fixture()
            x=jnp.asarray(points);v=jnp.asarray(volumes);D=RegionalLaplacian(points,volumes,c)
            mean=jnp.dot(v,initial_field(x,c,xp=jnp))/v.sum()
            loss=make_loss_fn(x,v,D,None,None,None,c);net=_build_network(c,jax.random.PRNGKey(8))
            optimizer=optax.sgd(1e-6);state=optimizer.init(eqx.filter(net,eqx.is_array))
            step=make_balanced_step(loss,optimizer,c);weights=jnp.ones(2);norms=jnp.zeros(2);times=jnp.asarray([.002,.1,1.5])
            for iteration,expected_active in [(0,[True,False]),(24,[True,False]),(25,[False,True]),(49,[False,True]),(4000,[True,True])]:
                mask=jnp.asarray(expected_active)
                expected_grad=eqx.filter_grad(lambda n:jnp.dot(loss(n,times,mean)[1],mask))(net)
                expected=eqx.apply_updates(net,jax.tree.map(lambda a:None if a is None else -1e-6*a,expected_grad))
                candidate,_,new_weights,_,_,parts,_,active=step(net,state,weights,norms,times,mean,jnp.asarray(iteration))
                np.testing.assert_array_equal(active,expected_active)
                np.testing.assert_array_equal(new_weights,weights)
                for a,b in zip(jax.tree.leaves(eqx.filter(candidate,eqx.is_array)),jax.tree.leaves(eqx.filter(expected,eqx.is_array)),strict=True):
                    np.testing.assert_allclose(a,b,atol=1e-13,rtol=1e-12)
                self.assertEqual(parts.shape,(2,))
                phi=predict_fields(candidate,x,0.,mean,v,D)[:,0]
                np.testing.assert_allclose(phi,initial_field(x,c,xp=jnp),atol=1e-13)
            masks=np.asarray([active_losses(jnp.asarray(i),c.train.loss_balance) for i in range(5000)])
            self.assertEqual(int(np.sum(masks[:,0]&~masks[:,1])),2000)
            self.assertEqual(int(np.sum(masks[:,1]&~masks[:,0])),2000)
            self.assertEqual(int(np.sum(masks.all(axis=1))),1000)
            c.train.loss_balance.stagger_steps=4999
            with self.assertRaises(ValueError):validate_config(c)


if __name__=='__main__':unittest.main()
