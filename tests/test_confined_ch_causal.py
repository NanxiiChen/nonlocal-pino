"""Independent adaptive-weight direction; never differentiate through weights."""
import unittest
import numpy as np
import jax,jax.numpy as jnp
from examples.confined_ch2d.losses import causal_weights


class CausalTests(unittest.TestCase):
    def test_stopped_weighted_direction_and_floor(self):
        with jax.enable_x64(True):
            b=jnp.asarray([1.,2.,3.]);a=.3;strength=.7;floor=.05
            per=(a-np.asarray(b))**2
            w=np.maximum(np.exp(-strength*np.r_[0.,np.cumsum(per[:-1])]),floor)
            expected=np.dot(w,2*(a-np.asarray(b)))/w.sum()
            def objective(scale):
                losses=(scale-b)**2
                weights=causal_weights(losses,strength=strength,minimum=floor)
                return jnp.dot(weights,losses)/weights.sum()
            np.testing.assert_allclose(jax.jit(jax.grad(objective))(a),expected,rtol=1e-13)
            np.testing.assert_allclose(causal_weights(jnp.asarray(per),strength=strength,minimum=floor),w,rtol=1e-13)
            np.testing.assert_allclose(jax.jacfwd(lambda v:causal_weights(v,strength=strength,minimum=floor))(jnp.asarray(per)),0.,atol=0.)
            np.testing.assert_allclose(causal_weights(jnp.zeros(3),strength=strength,minimum=floor),1.,atol=0.)


if __name__=='__main__':unittest.main()
