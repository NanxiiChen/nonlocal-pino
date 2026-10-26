"""Numerical equivalence, saturated-phase gradients, IC and monitoring checks."""

import json
from pathlib import Path
import tempfile
import unittest

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from examples.icemelting3d.configs.train_base import get_config
from examples.icemelting3d.evaluation import MeltingEvaluator
from examples.icemelting3d.losses import (
    _compute_pde, _compute_pde_local, _compute_pde_latent, ic_values, make_loss_fn,
)
from examples.icemelting3d.phase import InitialPhase
from examples.icemelting3d.train import _build_network, _build_phase_network, _make_step, _nondim_params
from nonlocal_pinn.baselines.fd import FiniteDiff
from nonlocal_pinn.operators import SparseD
from nonlocal_pinn.training.optimizers import make_optimizer


class Correction(eqx.Module):
    scale: jax.Array

    def __call__(self, x, t):
        return jnp.atleast_1d(self.scale * (jnp.sum(x*x) + t))


class Writer:
    def add_scalar(self, *args):
        pass

    def flush(self):
        pass


def tiny_config():
    c = get_config()
    c.arch.hidden_dim = 4
    c.arch.num_layers = 1
    c.arch.spatial_emb_dim = 2
    c.arch.temporal_emb_dim = 2
    c.sample.Nx = c.sample.Ny = c.sample.Nz = 4
    c.nl.delta = .4
    c.eval.chunk_size = 5
    return c


class MeltingLatentTest(unittest.TestCase):
    def test_initial_condition_and_no_prescribed_motion(self):
        c = tiny_config()
        key = jax.random.PRNGKey(2)
        self.assertEqual(c.phase.formulation, "ic_latent")
        net = _build_phase_network(c, key)
        x = jnp.asarray([[.2, .3, .4], [.5, .5, .5], [.8, .7, .3]])
        initial = jax.vmap(net, in_axes=(0, None))(x, 0.)[:, 0]
        np.testing.assert_allclose(initial, ic_values(x, .35, c.nondim.xc, np.sqrt(2)*c.physical.eps_star), atol=1e-7)
        np.testing.assert_array_equal(jax.vmap(net, in_axes=(0, None))(x, 1.)[:, 0], initial)
        net = eqx.tree_at(lambda n: n.raw.output.bias, net, jnp.asarray([.7]))
        np.testing.assert_array_equal(jax.vmap(net, in_axes=(0, None))(x, 0.)[:, 0], initial)
        c.physical.lam = 8.
        changed_speed = _build_phase_network(c, key)
        np.testing.assert_array_equal(jax.vmap(changed_speed, in_axes=(0,None))(x, 1.)[:,0], initial)
        c.phase.formulation = "raw_phase"
        self.assertTrue(eqx.tree_equal(_build_phase_network(c,key), _build_network(c,key)))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"net.eqx"
            eqx.tree_serialise_leaves(path, net)
            restored = eqx.tree_deserialise_leaves(path, _build_phase_network(tiny_config(),key))
            self.assertTrue(eqx.tree_equal(restored, net))

    def test_graph_and_local_residual_chain_rule(self):
        x = jnp.asarray([[.2,.3,.4], [.3,.4,.6], [.7,.4,.2]])
        times = jnp.asarray([.2,.7])
        net = InitialPhase(Correction(jnp.asarray(.3)), .35, (.5,.5,.5), .4)
        row = jnp.asarray([0,1,1,2,0], dtype=jnp.int32)
        col = jnp.asarray([1,0,2,1,0], dtype=jnp.int32)
        sub = (row,col,jnp.asarray([-2.,-2.,-3.,-3.,0.]),jnp.asarray([2.,5.,3.]))
        for operator in [sub, None]:
            if operator is None:
                original, _ = _compute_pde_local(net,x,times,.01,.2,.5)
            else:
                original, _ = _compute_pde(net,x,times,operator,.01,.2,.5)
            transformed, phi = _compute_pde_latent(net,x,times,operator,.01,.2,.5)
            np.testing.assert_allclose(original, (1-phi*phi)*transformed, rtol=3e-6, atol=2e-7)

    def test_saturated_phase_retains_finite_nonzero_latent_gradient(self):
        x = jnp.asarray([[.6,.5,.5], [.7,.5,.5]], dtype=jnp.float32)
        net = InitialPhase(Correction(jnp.float32(.1)), 25., (.5,.5,.5), 1.)
        sub = (jnp.asarray([0,1]),jnp.asarray([1,0]),jnp.asarray([-2.,-2.]),jnp.asarray([2.,2.]))
        r, phi = _compute_pde_latent(net,x,jnp.asarray([.2]),sub,.01,.2,.5)
        np.testing.assert_array_equal(phi, np.ones((2,1)))
        self.assertTrue(np.all(np.isfinite(r)))
        def objective(model):
            return jnp.mean(_compute_pde_latent(model,x,jnp.asarray([.2]),sub,.01,.2,.5)[0]**2)
        grad = eqx.filter_grad(objective)(net)
        self.assertTrue(np.isfinite(float(grad.raw.scale)))
        self.assertGreater(abs(float(grad.raw.scale)), 1e-4)

    def test_padded_sparse_and_fd_training_updates_all_modes(self):
        c = tiny_config()
        v = jnp.linspace(0.,1.,4)
        x = jnp.stack(jnp.meshgrid(v,v,v,indexing="xy"),axis=-1).reshape(-1,3)
        for operator in [SparseD(x,.4), FiniteDiff(x,Nx=4,Ny=4,Nz=4)]:
            entry = operator.build_subgraph_pool(5,2)[0]
            points, sub, mask = x[entry[0]], entry[1], entry[2]
            for mode in ["raw_phase", "ic_phase", "ic_latent"]:
                c.phase.formulation = mode
                net = _build_phase_network(c,jax.random.PRNGKey(2))
                params = _nondim_params(c)
                fn = make_loss_fn(x,operator,*params[:3],x_ic=x[:5],formulation=mode)
                opt = make_optimizer(c)
                state = opt.init(eqx.filter(net,eqx.is_array))
                step = _make_step(fn,opt)
                target = ic_values(x[:5],params[-1],c.nondim.xc,params[4])
                if mode != "raw_phase":
                    target = jnp.full_like(target,jnp.nan)  # Must never read IC targets in hard modes.
                for _ in range(2):
                    net,state,loss,aux,w = step(net,state,jnp.asarray([.2,.7]),target,
                                              jnp.ones(2),jnp.ones(2),True,jnp.asarray(0.),None,points,sub,mask)
                    self.assertTrue(bool(aux['finite_update']))
                    self.assertTrue(np.isfinite(float(loss)))
                if mode != "raw_phase":
                    self.assertEqual(float(aux['loss/ic']),0.)
                    np.testing.assert_array_equal(w,[1.,0.])
                    actual = jax.vmap(net,in_axes=(0,None))(x[:5],0.)[:,0]
                    np.testing.assert_allclose(actual,ic_values(x[:5],params[-1],c.nondim.xc,params[4]),atol=1e-7)
                jax.clear_caches()

    def test_monitor_matches_physical_times_and_full_grid_with_padding(self):
        c = tiny_config()
        v = jnp.linspace(0.,1.,4)
        x = jnp.stack(jnp.meshgrid(v,v,v,indexing="xy"),axis=-1).reshape(-1,3)
        with tempfile.TemporaryDirectory() as folder:
            ref = Path(folder)/"reference.npz"
            identity = {"physical":c.physical.to_dict(),"nl":c.nl.to_dict(),"grid":[4]*3,
                        "nondim":{**c.nondim.to_dict(),"T_c":5.}}
            np.savez(ref,axis=(np.arange(2)+.5)/3,time=np.linspace(0.,1.,21),phi=np.full((21,2,2,2),.2),identity=json.dumps(identity))
            c.eval.graph_reference_path = str(ref)
            evaluator = MeltingEvaluator(c,x,x[:7],jnp.asarray([0.,.25,.5,.75,1.]),folder,Writer())
            result = evaluator.evaluate(lambda xx,t:jnp.asarray([.2]),0,True)
            self.assertEqual(evaluator.reference_indices,[0,5,10,15,20])
            self.assertEqual(result['physical_terminal_s'],5.)
            self.assertLess(result['graph_terminal_full_rel_l2'],1e-7)
            self.assertLess(result['graph_pooled_rel_l2'],1e-7)
            c.nondim.T_c = 4.  # Explicit historical cross-scale control, not the default.
            historical = MeltingEvaluator(c,x,x[:7],jnp.asarray([0.,.25,.5,.75,1.]),folder,Writer())
            self.assertEqual(historical.reference_indices,[0,4,8,12,16])
            result4 = historical.evaluate(lambda xx,t:jnp.asarray([.2]),0,True)
            self.assertEqual(result4['physical_terminal_s'],4.)
            c.physical.M *= 2
            with self.assertRaisesRegex(ValueError,"physical mismatch"):
                MeltingEvaluator(c,x,x[:7],jnp.asarray([0.,.25,.5,.75,1.]),folder,Writer())


if __name__ == "__main__":
    unittest.main()
