"""Window time chain rule, hard state carry and direct trajectory loading."""
from pathlib import Path
import json,tempfile,unittest
import numpy as np
import jax,jax.numpy as jnp,equinox as eqx
from examples.confined_ch2d.configs.train_coarsening_windows import get_config
from examples.confined_ch2d.problem import validate_config,initial_field,identity
from examples.confined_ch2d.window_network import build_window_network,window_fields,make_window_loss
from examples.confined_ch2d.window_training import window_bounds
from examples.confined_ch2d.window_evaluate import replay_trajectory
from examples.confined_ch2d.operators import RegionalLaplacian
from examples.confined_ch2d.native_reference import native_matrix
from nonlocal_pinn.training.runtime import with_run_seed
from tests.test_confined_ch import fixture


class IdentityMetric(eqx.Module):
    def __call__(self,r):return r


class WindowTests(unittest.TestCase):
    def test_chain_rule_independent_residual_and_gradients(self):
        c=with_run_seed(get_config(),42);validate_config(c);c.arch.hidden_dim=8;c.arch.num_layers=2;c.nl.delta=.2
        _,x,v=fixture();A=native_matrix(x,v,c)
        with jax.enable_x64(True):
            p=jnp.asarray(x);vol=jnp.asarray(v);initial=initial_field(p,c,xp=jnp);mean=jnp.dot(vol,initial)/vol.sum()
            net=build_window_network(c,jax.random.PRNGKey(1));D=RegionalLaplacian(x,v,c)
            width=jnp.asarray(.03);t=jnp.asarray(.012)
            fun=lambda tt:window_fields(net,p,tt,width,initial,mean,vol,D,c)
            np.testing.assert_allclose(fun(0.)[:,0],initial,atol=1e-14)
            value,dt=jax.jvp(fun,(t,),(jnp.asarray(1.),))
            np.testing.assert_allclose(dt,(fun(t+1e-7)-fun(t-1e-7))/2e-7,atol=1e-6,rtol=1e-6)
            expected=c.physical.beta*(np.asarray(value[:,0])**3-np.asarray(value[:,0]))+c.physical.epsilon**2*(A@np.asarray(value[:,0]))
            np.testing.assert_allclose(value[:,1],expected,atol=1e-12)
            self.assertLess(abs(float(jnp.dot(vol,dt[:,0]))),1e-12)
            c.sample.include_initial_time=False;loss=make_window_loss(p,vol,D,mean,c)
            actual,g=eqx.filter_jit(eqx.filter_value_and_grad(loss))(net,initial,width,jnp.asarray([t]),IdentityMetric())
            r=float(width)*(np.asarray(dt[:,0])+c.physical.mobility*(A@expected))
            np.testing.assert_allclose(actual,np.sum(v*r*r)/v.sum(),atol=1e-11,rtol=1e-11)
            self.assertTrue(all(np.isfinite(a).all() for a in jax.tree.leaves(eqx.filter(g,eqx.is_array))))

    def test_two_saved_windows_replay_and_state_continuity(self):
        c=with_run_seed(get_config(),42);c.arch.hidden_dim=8;c.arch.num_layers=2;c.nl.delta=.2
        c.train.windows.count=2;c.train.iterations=800;c.train.refinement.iterations=52;c.train.refinement.max_evaluations=104
        validate_config(c);_,x,v=fixture();bounds=window_bounds(c)
        with jax.enable_x64(True),tempfile.TemporaryDirectory() as directory:
            run=Path(directory);p=jnp.asarray(x);vol=jnp.asarray(v);initial=initial_field(p,c,xp=jnp);mean=jnp.dot(vol,initial)/vol.sum()
            D=RegionalLaplacian(x,v,c);manifest=dict(model='sequential_nonlocal_ch_pinn_v1',status='complete',identity=identity(c),bounds=bounds.tolist(),windows=[])
            for i in range(2):
                net=build_window_network(c,jax.random.PRNGKey(i));width=jnp.asarray(bounds[i+1]-bounds[i])
                eqx.tree_serialise_leaves(run/f'model{i}.eqx',net);np.savez(run/f'initial{i}.npz',phi=np.asarray(initial),time=bounds[i])
                manifest['windows'].append(dict(model=f'model{i}.eqx',initial=f'initial{i}.npz',start=bounds[i],end=bounds[i+1]))
                initial=window_fields(net,p,width,width,initial,mean,vol,D,c)[:,0]
            path=run/'trajectory.json';path.write_text(json.dumps(manifest))
            ref=dict(points=x,volumes=v,time=np.linspace(0,1.5,3))
            for key in ('phi_local','phi_nonlocal','mu_local','mu_nonlocal'):ref[key]=np.ones((3,len(x)))
            fields,report=replay_trajectory(run,c,ref,path)
            self.assertEqual(fields.shape,(3,len(x),2));self.assertLess(max(report['continuity_max_absolute']),1e-12)
            self.assertLess(report['mass_max_drift'],1e-12);self.assertLess(report['chemical_constraint_max_absolute'],1e-12)
            manifest['status']='incomplete';path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):replay_trajectory(run,c,ref,path)


if __name__=='__main__':unittest.main()
