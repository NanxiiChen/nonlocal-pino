"""Whole-time term gradients, stopped weighting, cold physics and actual updates."""
import contextlib
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import equinox as eqx
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
import numpy as np

from examples.channel_nsch2d.balance import (
    balanced_value_and_grad,group_names,make_balanced_accumulated_fn)
from examples.channel_nsch2d.balance_validation import comparison_config,audit_comparison
from examples.channel_nsch2d.configs.train_operator_width128_gmls_time40_jvp_allterms_online import get_config
from examples.channel_nsch2d.configs.train_operator_width128_gmls_time40_jvp_stronger_online import get_config as control_config
from examples.channel_nsch2d.data import geometry_data
from examples.channel_nsch2d.losses import make_loss_fn
from examples.channel_nsch2d.network import build_network,encode,fields
from examples.channel_nsch2d.problem import identity,validate_config
from examples.channel_nsch2d.temporal import sample_continuous
from nonlocal_pinn.training.optimizers import make_optimizer
from tests.test_channel_uniform_time_jvp import config as small


def candidate(c):
    c=ConfigDict(c.to_dict());c.loss.chemical_residual_scale=1.
    c.loss.phase_scale=1.;c.loss.boundary_mu_scale=1.;c.balance=get_config().balance
    return c


class ChannelAllTermsBalanceTests(unittest.TestCase):
    def test_explicit_config_raw_losses_and_strict_control(self):
        c=get_config();c.train.seed=42;validate_config(c)
        self.assertEqual(len(group_names(c)),9);self.assertEqual(c.loss.chemical_scale,4.)
        self.assertEqual(c.loss.phase_scale,1.);self.assertEqual(c.loss.boundary_mu_scale,1.)
        bad=get_config();bad.balance.grouping='unknown'
        with self.assertRaisesRegex(ValueError,'cold uniform'):validate_config(bad)
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);source=control_config();source.train.seed=42
            for name in ('sampling.npz','prediction.npz','known_inputs.npz','known_inputs_final.npz',
                         'model_initial.eqx','optimizer_initial.eqx'):(root/name).touch()
            (root/'config.json').write_text(source.to_json())
            (root/'summary.json').write_text(json.dumps(dict(status='completed',iterations=5000)))
            (root/'validation.json').write_text(json.dumps(dict(status='replay_and_hard_constraints_passed')))
            self.assertEqual(comparison_config(c,root),json.loads(source.to_json()))
            for group,name,value in (('loss','chemical_scale',1.),('train','lr_init',.002),
                                     ('sample','Nt',8),('geometry','upper',.19)):
                bad=ConfigDict(c.to_dict());setattr(getattr(bad,group),name,value)
                with self.subTest(group=group,name=name),self.assertRaisesRegex(ValueError,'may change only'):
                    comparison_config(bad,root)

    def test_norm_after_time_mean_not_mean_of_block_norms(self):
        with jax.enable_x64(True):
            c=get_config();model=jnp.array([3.,.7]);times=jnp.linspace(0,6,40);previous=jnp.ones(9)
            def synthetic(p,batch,ts):
                parts=jnp.stack([jnp.mean((p[i%2]-(i+1)*ts)**2)for i in range(9)])
                return parts.sum(),parts
            with patch('examples.channel_nsch2d.balance.make_loss_fn',return_value=synthetic):
                fn=eqx.filter_jit(make_balanced_accumulated_fn(c))
            total,parts,weights,norms,gradient=fn(model,None,times,previous,jnp.array(True))
            # Analytic gradients of each entire-time MSE, with cancellation in term0.
            jac=np.zeros((9,2))
            for i in range(9):jac[i,i%2]=2*(float(model[i%2])-(i+1)*float(times.mean()))
            np.testing.assert_allclose(norms,np.abs(jac).sum(axis=1),rtol=2e-12,atol=2e-12)
            self.assertLess(float(norms[0]),1e-12)
            local=np.mean([abs(2*(3-float(ts.mean())))for ts in np.asarray(times).reshape(5,8)])
            self.assertGreater(local,1.)
            np.testing.assert_allclose(gradient,np.asarray(weights)@jac,rtol=2e-12,atol=2e-12)
            np.testing.assert_allclose(total,np.asarray(weights)@np.asarray(parts),rtol=2e-12)
            _,_,held,zero,g=fn(model,None,times,weights,jnp.array(False))
            np.testing.assert_array_equal(held,weights);np.testing.assert_array_equal(zero,0.)
            np.testing.assert_allclose(g,gradient,rtol=2e-12,atol=2e-12)
            stopped=jax.jacfwd(lambda p:fn(p,None,times,previous,jnp.array(True))[2])(model)
            np.testing.assert_array_equal(stopped,0.)

    def test_full_physics_gradients_frozen_fd_soap_and_physical_primal(self):
        with jax.enable_x64(True):
            source=small();source.loss.chemical_residual_scale=1.;c=candidate(source)
            raw=[geometry_data(a,c)[0]for a in ((.12,.08),(.06,.13))]
            batch={k:jnp.asarray(np.stack([d[k]for d in raw]))for k in raw[0]}
            model=build_network(c,jax.random.PRNGKey(51));legacy=build_network(source,jax.random.PRNGKey(51))
            for a,b in zip(jax.tree.leaves(model),jax.tree.leaves(legacy),strict=True):
                if eqx.is_array(a):np.testing.assert_array_equal(a,b)
            times=jax.jit(lambda k:sample_continuous(k,c))(jax.random.PRNGKey(11)).astype(jnp.float64)
            d={k:a[0]for k,a in batch.items()}
            before=eqx.filter_jit(lambda n:fields(n,encode(n,d),d['points'],times,d['amplitudes'],source))(legacy)
            after=eqx.filter_jit(lambda n:fields(n,encode(n,d),d['points'],times,d['amplitudes'],c))(model)
            np.testing.assert_array_equal(before,after)
            loss=make_loss_fn(c);old=make_loss_fn(source)
            _,oldparts=eqx.filter_jit(old)(model,batch,times)
            _,newparts=eqx.filter_jit(loss)(model,batch,times)
            factors=np.ones(9);factors[0]=.04;factors[[5,8]]=16
            np.testing.assert_allclose(newparts,np.asarray(oldparts)*factors,rtol=2e-12,atol=2e-12)
            def whole(net,previous,refresh):
                def terms(n):
                    _,parts=loss(n,batch,times)
                    return parts,parts
                return balanced_value_and_grad(terms,net,previous,refresh,c.balance)
            direct=eqx.filter_jit(whole);blocked=eqx.filter_jit(make_balanced_accumulated_fn(c))
            previous=jnp.linspace(.5,1.5,9)
            expected=direct(model,previous,jnp.array(True))
            actual=blocked(model,batch,times,previous,jnp.array(True))
            for a,b in zip(jax.tree.leaves(actual),jax.tree.leaves(expected),strict=True):
                np.testing.assert_allclose(a,b,rtol=3e-9,atol=3e-10)
            _,_,weights,_,gradient=actual
            objective=eqx.filter_jit(lambda n:jnp.dot(weights,loss(n,batch,times)[1]))
            h=1e-5;index=(1,0)
            plus=eqx.tree_at(lambda n:n.output.weight,model,model.output.weight.at[index].add(h))
            minus=eqx.tree_at(lambda n:n.output.weight,model,model.output.weight.at[index].add(-h))
            finite_difference=(float(objective(plus))-float(objective(minus)))/(2*h)
            np.testing.assert_allclose(gradient.output.weight[index],finite_difference,rtol=2e-5,atol=2e-6)
            held=blocked(model,batch,times,weights,jnp.array(False))
            np.testing.assert_array_equal(held[2],weights)
            for a,b in zip(jax.tree.leaves(held[-1]),jax.tree.leaves(gradient),strict=True):
                np.testing.assert_allclose(a,b,rtol=3e-9,atol=3e-10)
            optimizer=make_optimizer(c);state=optimizer.init(eqx.filter(model,eqx.is_array))
            step=eqx.filter_jit(lambda g:optimizer.update(g,state,model))
            update_a,state_a=step(gradient);update_b,state_b=step(expected[-1])
            for a,b in zip(jax.tree.leaves(update_a),jax.tree.leaves(update_b),strict=True):
                np.testing.assert_allclose(a,b,rtol=3e-7,atol=3e-9)
            self.assertEqual(int(state_a[1][0].count),1);self.assertEqual(int(state_b[1][0].count),1)

    def test_actual_three_step_training_replay_inputs_and_weight_refresh(self):
        from examples.channel_nsch2d import train as entry
        from examples.channel_nsch2d.cohort import prepare
        from examples.channel_nsch2d.validate_run import validate
        c=small();c.loss.chemical_residual_scale=1.;c.train.iterations=3
        c.operator.online.refresh_every=1;c.operator.online.refresh_size=2
        c.log.eval_every=3;c.log.checkpoint_every=3;c.arch.spatial_scale=.25
        with tempfile.TemporaryDirectory()as directory,jax.enable_x64(True):
            root=Path(directory);c.reference.root=str(root);c.log.log_dir=str(root/'runs');c.log.run_name='fixed'
            c.operator.validation_manifest=str(root/'reserved/manifest.json');prepare(c,c.operator.validation_manifest,4,(96,40))
            reference=root/'reference.npz';target=np.ones((13,3,5));target[...,0]=.5
            np.savez(reference,points=np.array([[.2,.3],[.8,.5],[2.,.5]]),time=np.linspace(0,6,13),
                amplitudes=np.array([.12,.08]),fields=target)
            (root/'reference-validation.json').write_text(json.dumps(dict(status='passed',reference=str(reference),identity=identity(c))))
            def executor(**kwargs):return ThreadPoolExecutor(max_workers=kwargs['max_workers'])
            with patch('examples.channel_nsch2d.online.ProcessPoolExecutor',executor),contextlib.redirect_stdout(io.StringIO()):
                entry.train(c);control=Path(c.log.log_dir)/c.log.run_name;validate(control)
                new=candidate(c);new.log.run_name='adaptive';comparison_config(new,control)
                entry.train(new);run=Path(new.log.log_dir)/new.log.run_name;validate(run)
                audit=audit_comparison(run,control)
            self.assertEqual(audit['status'],'matched_cold_state_all_inputs_time_and_nine_weight_history_passed')
            self.assertEqual(audit['refresh_records'],1)
            self.assertTrue(audit['all_initial_final_input_and_actual_sampling_arrays_exact'])
            self.assertNotEqual((run/'model_initial.eqx').read_bytes(),(run/'model_final.eqx').read_bytes())
            summary=json.loads((run/'summary.json').read_text())
            self.assertEqual(len(summary['balance']['groups']),9)
            self.assertEqual(summary['chemical_scaling']['output_amplitude'],4.)
            self.assertEqual(summary['chemical_scaling']['boundary_normalizer'],1.)


if __name__=='__main__':unittest.main()
