"""Exact second derivatives and third derivatives of a streamfunction velocity."""
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from docs.tasks.channel_trained_operator_probe import local_laplacian


class TrainedOperatorProbe(unittest.TestCase):
    def test_vector_laplacian_and_streamfunction_velocity_manufactured(self):
        with jax.enable_x64(True):
            x=jnp.array([[.2,.3],[.7,.5],[1.1,.9]],dtype=jnp.float64)
            t=jnp.array([.1,.5],dtype=jnp.float64)
            def function(q):
                a=jnp.exp(t)[:,None]*(q[:,0]**2+q[:,1]**3)
                b=t[:,None]*jnp.sin(q[:,0])*jnp.cos(q[:,1])
                return jnp.stack((a,b),axis=-1)
            value,lap=jax.jit(lambda q:local_laplacian(function,q))(x)
            expected=jnp.stack((jnp.exp(t)[:,None]*(2+6*x[:,1]),-2*value[...,1]),axis=-1)
            np.testing.assert_allclose(lap,expected,atol=3e-14,rtol=3e-14)
            def velocity(q):
                stream=lambda p:t[:,None]*jnp.sin(p[:,0])*jnp.sin(2*p[:,1])
                ex=jnp.broadcast_to(jnp.array([1.,0.]),q.shape);ey=jnp.broadcast_to(jnp.array([0.,1.]),q.shape)
                _,dx=jax.jvp(stream,(q,),(ex,));_,dy=jax.jvp(stream,(q,),(ey,))
                return jnp.stack((dy,-dx),axis=-1)
            value,lap=jax.jit(lambda q:local_laplacian(velocity,q))(x)
            np.testing.assert_allclose(lap,-5*value,atol=3e-14,rtol=3e-14)


if __name__=='__main__':unittest.main()
