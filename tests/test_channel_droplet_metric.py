"""Bounded fixed residual norms, genuine objective gradients and shape diagnostics."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from ml_collections import ConfigDict

from examples.channel_nsch2d.configs.train_single_lbfgs_droplet import get_config
from examples.channel_nsch2d.droplet_metric import weights_from_phase,parent_phase
from examples.channel_nsch2d.losses import make_loss_fn
from examples.channel_nsch2d.problem import validate_config
from examples.channel_nsch2d.refinement import parent_spec,make_accumulated_objective
from examples.channel_nsch2d.network import build_network
from examples.channel_nsch2d.data import geometry_data
from tests.test_channel_continuation import parent_fixture


class DropletMetric(unittest.TestCase):
    def test_normalization_positive_bounds_and_zero_contrast(self):
        c=get_config();s=c.droplet_metric
        phase=np.array([[-2.,-1.,0.,1.,2.],[1.,1.,1.,1.,1.]])
        volume=np.array([0.,1.,2.,3.,4.])
        weight=weights_from_phase(phase,volume,s)
        occupancy=1/(1+np.exp(phase/s.temperature))
        raw=1+s.contrast*occupancy
        expected=raw/np.sum(raw*volume[None],axis=1,keepdims=True)*volume.sum()
        np.testing.assert_allclose(weight,expected,atol=2e-15)
        np.testing.assert_allclose(np.sum(weight*volume[None],axis=1)/volume.sum(),1.,atol=3e-16)
        self.assertTrue(np.all(weight>=1/(1+s.contrast)))
        self.assertTrue(np.all(weight<=1+s.contrast))
        s.contrast=0.;np.testing.assert_array_equal(weights_from_phase(phase,volume,s),1.)
        with self.assertRaisesRegex(ValueError,'nonzero'):weights_from_phase(phase,np.zeros(5),s)
        phase[0,0]=np.nan
        with self.assertRaisesRegex(ValueError,'finite'):weights_from_phase(phase,volume,s)

    def test_configuration_and_parent_whitelist_remain_narrow(self):
        c=get_config();validate_config(c)
        c.droplet_metric.source='reference'
        with self.assertRaisesRegex(ValueError,'parent_model'):validate_config(c)
        c=get_config();del c.train.refinement
        with self.assertRaisesRegex(ValueError,'full-time'):validate_config(c)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);c,*_=parent_fixture(root)
            np.savez(root/'known_inputs.npz',a=np.zeros((1,2)))
            np.savez(root/'sampling.npz',a=np.zeros((1,2)))
            (root/'operators.json').write_text('[]')
            c.train.refinement=ConfigDict(dict(parent_run=directory))
            c.droplet_metric=get_config().droplet_metric
            self.assertEqual(parent_spec(c)[0],root)
            c.physical.mobility*=2
            with self.assertRaisesRegex(ValueError,'preserve'):parent_spec(c)

    def test_actual_BDF_SBP_loss_zero_contrast_and_fixed_metric_directional_derivative(self):
        with jax.enable_x64(True):
            c=get_config();c.sample.Nx=c.sample.Ny=8;c.sample.boundary_points=4
            c.arch.width=4;c.arch.heads=1;c.arch.condition_depth=c.arch.decoder_depth=1
            c.arch.spatial_features=c.arch.time_features=2
            c.nl.degree=2;c.nl.neighbors=20;c.train.compute_dtype='float64'
            c.train.refinement.time_batch=2
            raw,_=geometry_data((.12,.08),c)
            data={k:jnp.asarray(v) for k,v in raw.items()}
            model=build_network(c,jax.random.PRNGKey(19))
            nodes=np.arange(1,5,dtype=np.float64)*c.time_discretization.dt
            phase=parent_phase(model,data,nodes,c)
            measure=np.asarray(data['transport_measure']*data['interior'])
            weights=weights_from_phase(phase,measure,c.droplet_metric)
            plain=ConfigDict(c.to_dict());del plain.droplet_metric
            batch=jax.tree.map(lambda x:x[None],data)
            baseline=make_loss_fn(plain)
            weighted=make_loss_fn(c)
            identity=dict(batch,droplet_weights=jnp.ones((1,len(nodes),len(measure))))
            zero=eqx.filter_jit(eqx.filter_value_and_grad(lambda n:weighted(n,identity,jnp.asarray(nodes)),has_aux=True))
            original=eqx.filter_jit(eqx.filter_value_and_grad(lambda n:baseline(n,batch,jnp.asarray(nodes)),has_aux=True))
            (z,zp),zg=zero(model);(o,op),og=original(model)
            np.testing.assert_allclose(z,o,atol=2e-12)
            np.testing.assert_allclose(zp,op,atol=2e-12)
            for a,b in zip(jax.tree.leaves(zg),jax.tree.leaves(og),strict=True):
                if eqx.is_array(a):np.testing.assert_allclose(a,b,atol=2e-10,rtol=2e-11)
            batch=dict(batch,droplet_weights=jnp.asarray(weights[None]))
            initial,restore,objective=make_accumulated_objective(model,weighted,batch,nodes,2)
            value,gradient,_=objective(initial)
            self.assertTrue(np.isfinite(value));self.assertGreater(np.linalg.norm(gradient),0.)
            direction=gradient/np.linalg.norm(gradient)
            epsilon=1e-5
            plus=objective(initial+epsilon*direction)[0];minus=objective(initial-epsilon*direction)[0]
            np.testing.assert_allclose((plus-minus)/(2*epsilon),np.dot(gradient,direction),rtol=2e-5,atol=2e-6)

    def test_contour_self_and_concentric_circle_distance(self):
        path=Path(__file__).resolve().parents[1]/'docs/tasks/channel_droplet_metrics.py'
        spec=importlib.util.spec_from_file_location('channel_shape_metrics',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        x,y=np.meshgrid(np.linspace(0,1,161),np.linspace(0,1,161),indexing='ij')
        points=np.stack((x,y),axis=-1).reshape(-1,2)
        radius=np.sqrt((x-.5)**2+(y-.5)**2).ravel()
        same=module.contour_distance(points,radius-.22,radius-.22,x.shape,1024)
        self.assertEqual(same['mean_distance'],0.)
        larger=module.contour_distance(points,radius-.23,radius-.22,x.shape,1024)
        self.assertAlmostEqual(larger['mean_distance'],.01,delta=1e-4)
        absent=module.contour_distance(points,np.ones_like(radius),radius-.22,x.shape,1024)
        self.assertTrue(absent['contour_missing']);self.assertIsNone(absent['mean_distance'])


if __name__=='__main__':unittest.main()
