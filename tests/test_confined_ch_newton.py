"""Exact manufactured time-PDE solve through the actual Newton-CG helper."""
import unittest,tempfile
from pathlib import Path
import numpy as np
import jax,jax.numpy as jnp,equinox as eqx
from ml_collections import ConfigDict
from examples.confined_ch2d.newton_refinement import refine


class Polynomial(eqx.Module):
    coefficients:object


class NewtonTests(unittest.TestCase):
    def test_manufactured_time_pde(self):
        # y=a*t+b*t^2, hard y(0)=0, PDE y_t=2*t has unique a=0,b=1.
        c=ConfigDict(dict(train=dict(refinement=dict(optimizer='newton_cg',iterations=20,
            time_nodes=(.02,.1,.3,.7,1.),time_points=5,step_tolerance=1e-10,c1=1e-4,c2=.9)),log=dict(eval_every=2)))
        def loss(model,times,mean):
            a,b=model.coefficients;r=a+2*b*times-2*times
            value=jnp.mean(r*r)
            return value,jnp.stack((value,))
        with tempfile.TemporaryDirectory() as d,jax.enable_x64(True):
            model=Polynomial(jnp.asarray([1.3,-.6]));visited=[]
            actual,record=refine(model,loss,jnp.asarray(0.),c,Path(d),lambda net,i:visited.append(i))
            np.testing.assert_allclose(actual.coefficients,[0.,1.],atol=1e-8)
            self.assertLess(record['final_loss'],1e-15)
            self.assertGreater(record['hessian_vector_products'],0)


if __name__=='__main__':unittest.main()
