"""Independent penalty derivatives, load-bound ownership and exact restart."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict
from examples.fracture2d.fem_damage import P1DamageOperator
from examples.fracture2d.irreversibility import CurrentEnergyDamagePINN, make_penalty_terms
from examples.fracture2d.loading import run, validate
from examples.fracture2d.mesh import load_geometry
from examples.fracture2d.stagger import damage_driving_field
from examples.fracture2d.storage import atomic_json
from tests.test_fracture2d_budget_learning_rate import controlled, stop_after_drop
from tests.test_fracture2d_local_mechanics import model


def configuration(root):
    c = controlled(root)
    c.irreversibility = ConfigDict(dict(mode='current_psi_l1', beta=1., healing_tolerance=1e-3))
    return c


class IrreversibilityTests(unittest.TestCase):
    def test_weak_gradient_common_normalization_and_detached_inputs(self):
        m = model(); op = P1DamageOperator(m)
        g0, g1, beta = 3., .7, 2.
        terms = make_penalty_terms(op, m.volumes, gc0=g0, gc1=g1, floor=1e-15, beta=beta)
        psi = jnp.array([1., 2., 3., 4.])
        d = jnp.array([.1, .4, .2, .8]); parent = jnp.array([.2, .3, .5, .7])
        normalization = np.sum(m.volumes*2*np.asarray(psi)**2/(g0+2*np.asarray(psi)))
        expected = (m.volumes*((g0+2*np.asarray(psi))*d-2*psi)
                    +g1*op.stiffness@np.asarray(d)
                    -beta*g0*m.volumes*(np.asarray(parent)>np.asarray(d)))/normalization
        loss = lambda d,p,h:terms(d,h,p)['penalized_loss']
        np.testing.assert_allclose(jax.grad(loss)(d,parent,psi),expected,rtol=1e-13,atol=1e-14)
        np.testing.assert_array_equal(jax.grad(loss,1)(d,parent,psi),0.)
        np.testing.assert_array_equal(jax.grad(loss,2)(d,parent,psi),0.)
        metrics = terms(d,psi,parent)
        self.assertAlmostEqual(float(metrics['maximum_healing']),.3)
        self.assertAlmostEqual(float(metrics['irreversibility_loss']),
            beta*g0*np.sum(m.volumes*np.maximum(parent-d,0))/normalization)
        # Under uniform complete unloading, unconstrained d=0 minimizes J.
        # With beta=2 and d_parent=.6 the constrained corner has opposite
        # one-sided derivatives: the L1 penalty prevents that spurious healing.
        unloaded = lambda x:terms(jnp.full(4,x),jnp.zeros(4),jnp.full(4,.6))['penalized_loss']
        self.assertLess(float(jax.grad(unloaded)(.59)),0.)
        self.assertGreater(float(jax.grad(unloaded)(.61)),0.)

    def test_current_driver_can_decrease_and_parent_is_fixed_across_fits(self):
        np.testing.assert_array_equal(damage_driving_field(jnp.ones(4)*10,jnp.ones(4),
            mode='current_psi_l1'),1.)
        np.testing.assert_array_equal(damage_driving_field(jnp.ones(4)*10,jnp.ones(4),
            mode='maximum_history'),10.)
        with tempfile.TemporaryDirectory() as tmp:
            c = configuration(Path(tmp)); m,op,_ = load_geometry(c)
            p = CurrentEnergyDamagePINN(c,m,op)
            parent = jnp.full(4,1e-5)
            p.begin_load(1,1e-4,parent_damage=parent)
            with contextlib.redirect_stdout(io.StringIO()):
                p.solve(jnp.arange(1.,5.),jnp.full(4,.7),1e-4)
                p.solve(jnp.arange(1.,5.)/2,jnp.full(4,.8),1e-4)
            np.testing.assert_array_equal(p.parent_damage,parent)
            context = p.context(jnp.arange(1.,5.))
            for key in ('H','d_parent'):
                np.testing.assert_array_equal(jax.grad(
                    lambda x:p.value(p.net,dict(context,**{key:x})))(context[key]),0.)
            c.history.replay_enabled = True
            with self.assertRaisesRegex(ValueError,'explicit replay policy'):validate(c)

    def test_real_loading_and_exact_endpoint_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); c = configuration(root)
            validate(c)
            with contextlib.redirect_stdout(io.StringIO()),patch(
                    'examples.fracture2d.loading.field_stop',side_effect=stop_after_drop):
                run(c,damage_solver='pinn')
            base = Path(c.output.root)
            rows = json.loads((base/'result.json').read_text())['curve']
            self.assertTrue(all(r['field_stable'] and r['field_stopping']['irreversibility_pass'] for r in rows))
            self.assertTrue(all(r['maximum_healing'] <= 1e-3 for r in rows))
            for row in rows:
                with np.load(row['fields_path']) as f:
                    q = f['q'].copy(); d = f['d'].copy(); psi = f['H'].copy()
                m,_,_ = load_geometry(c)
                np.testing.assert_allclose(psi,m.driver(q),rtol=0,atol=0)
                self.assertTrue(np.isfinite(d).all())
            c.output.root = str(root/'interrupted')
            def interrupt(path,data):
                atomic_json(path,data)
                if path.name == 'result.json' and 'curve' in data and len(data['curve']) == 1:
                    raise InterruptedError('endpoint')
            with contextlib.redirect_stdout(io.StringIO()),patch(
                    'examples.fracture2d.loading.field_stop',side_effect=stop_after_drop),patch(
                    'examples.fracture2d.loading.atomic_json',side_effect=interrupt):
                with self.assertRaises(InterruptedError):run(c,damage_solver='pinn')
            c.output.resume = True
            with contextlib.redirect_stdout(io.StringIO()),patch(
                    'examples.fracture2d.loading.field_stop',side_effect=stop_after_drop):
                run(c,damage_solver='pinn')
            final = Path(c.output.root)/'state_0002'
            for name in ('network.eqx','optimizer.eqx','rate_control.json'):
                self.assertEqual((base/'state_0002'/name).read_bytes(),(final/name).read_bytes())
            with np.load(base/'state_0002/fields.npz') as a,np.load(final/'fields.npz') as b:
                for key in a.files:np.testing.assert_array_equal(a[key],b[key])
            with self.assertRaisesRegex(ValueError,'direct reference is unchanged'):
                run(c,damage_solver='direct')

    def test_healing_gate_rejects_stationary_but_inadmissible_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = configuration(Path(tmp))
            stationary = dict(pass_this_sweep=True, consecutive=3, converged=True,
                              residual_is_stopping_gate=False)
            with contextlib.redirect_stdout(io.StringIO()),patch(
                    'examples.fracture2d.loading.field_stop',return_value=stationary),patch.object(
                    CurrentEnergyDamagePINN,'field_diagnostics',return_value=dict(maximum_healing=.1)):
                with self.assertRaisesRegex(RuntimeError,'exhausted outer budget'):
                    run(c,damage_solver='pinn')
            result = json.loads((Path(c.output.root)/'result.json').read_text())
            self.assertFalse(result['curve'][0]['field_stable'])
            self.assertFalse(result['curve'][0]['field_stopping']['irreversibility_pass'])
            self.assertFalse((Path(c.output.root)/'state_0002').exists())


if __name__ == '__main__':
    unittest.main()
