"""Independent weak-gradient/direct checks and neural FE loading lifecycle."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import equinox as eqx
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve
from examples.fracture2d.at2_cg import damage_solver
from examples.fracture2d.fem_damage import P1DamageOperator, make_energy
from examples.fracture2d.history_training import HistoryDamagePINN
from examples.fracture2d.loading import run, validate
from examples.fracture2d.mesh import load_geometry
from examples.fracture2d.storage import atomic_json
from tests.test_fracture2d_local_mechanics import model
from tests.test_fracture2d_history_conditioned_loading import configuration as residual_configuration


def configuration(root):
    c = residual_configuration(root)
    c.damage.discretization = 'fem'
    c.loss.norm = 'energy'
    c.history.bootstrap_energy_steps = 5
    del c.history.bootstrap_l1_steps
    del c.history.bootstrap_l2_steps
    return c


class FEMDamageTests(unittest.TestCase):
    def test_affine_gradient_energy_derivative_and_direct_minimizer(self):
        m = model()
        op = P1DamageOperator(m)
        affine = 2*m.coordinates[:, 0]-3*m.coordinates[:, 1]+.4
        gradient = np.einsum('ei,eia->ea', affine[op.triangles], op.gradients)
        np.testing.assert_allclose(gradient, [[2., -3.], [2., -3.]], atol=1e-14)
        # Independently known stiffness of the unit-square two-triangle patch.
        expected = np.array([[1.,-.5,-.5,0.],[-.5,1.,0.,-.5],
                             [-.5,0.,1.,-.5],[0.,-.5,-.5,1.]])
        np.testing.assert_allclose(op.stiffness.toarray(), expected, atol=1e-14)
        H, d = jnp.array([1., 2., 3., 4.]), jnp.array([.1, .4, .2, .8])
        g0, g1 = 3., .7
        energy = make_energy(op, m.volumes, gc0=g0, gc1=g1, floor=1e-15)
        A = diags(m.volumes*(g0+2*np.asarray(H)))+g1*csr_matrix(expected)
        b = 2*m.volumes*np.asarray(H)
        release = np.sum(m.volumes*2*np.asarray(H)**2/(g0+2*np.asarray(H)))
        np.testing.assert_allclose(jax.grad(energy)(d,H), (A@d-b)/release, rtol=1e-13)
        np.testing.assert_array_equal(jax.grad(lambda x:energy(d,x))(H), 0.)
        target = spsolve(A.tocsr(), b)
        solve = damage_solver(op,m.volumes,g0,g1,tol=1e-12,atol=1e-14,maxiter=100)
        actual = solve(H,jnp.zeros(4))
        np.testing.assert_allclose(actual,target,rtol=1e-12,atol=1e-14)
        self.assertLess(float(energy(actual,H)), 0.)
        self.assertGreater(float(energy(d,H)),float(energy(actual,H)))
        self.assertEqual(float(energy(jnp.zeros(4),jnp.zeros(4))),0.)
        m.volumes = m.volumes*2
        with self.assertRaisesRegex(ValueError,'triangle-lumped'):
            P1DamageOperator(m)

    def test_energy_network_trains_and_history_inputs_are_detached(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = configuration(Path(tmp))
            c.history.bootstrap_energy_steps = 64
            m,op,_ = load_geometry(c)
            p = HistoryDamagePINN(c,m,op)
            H = jnp.array([1.,2.,3.,4.])
            ctx = p.context(H)
            before = p._current_loss(ctx)
            p.begin_load(1,.0001)
            d,after = p.solve(H,jnp.zeros(4),.0001)
            self.assertLess(after,before)
            self.assertTrue(np.isfinite(d).all())
            batch = dict(H=H[None],fine_ids=ctx['fine_ids'][None],weights=jnp.ones(1))
            np.testing.assert_array_equal(jax.grad(lambda x:p.value(p.net,dict(batch,H=x)))(batch['H']),0.)
            leaves = jax.tree.leaves(eqx.filter_value_and_grad(p.value)(p.net,batch)[1])
            self.assertTrue(all(np.isfinite(x).all() for x in leaves))

    def test_fem_route_zero_resume_and_direct_share_discretization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = configuration(root)
            # FE loading must not read/build the nonlocal operator file.
            (Path(c.mesh.prepared_dir)/'damage_operator.npz').unlink()
            with contextlib.redirect_stdout(io.StringIO()):
                run(c,damage_solver='pinn')
            base = Path(c.output.root)
            report = json.loads((base/'result.json').read_text())
            self.assertEqual([r['optimizer_count'] for r in report['curve']],[9,15])
            identity = json.loads((base/'identity.json').read_text())
            self.assertEqual(identity['schema'],'local-p1-lumped-at2-v1')
            with np.load(base/'state_0000/fields.npz') as f:
                for name in ('q','d','H'): np.testing.assert_array_equal(f[name],0.)
            c.output.root = str(root/'interrupted')
            def interrupt(path,data):
                atomic_json(path,data)
                if path.name=='result.json' and 'curve' in data and len(data['curve'])==1:
                    raise InterruptedError('endpoint')
            with contextlib.redirect_stdout(io.StringIO()), patch(
                    'examples.fracture2d.loading.atomic_json',side_effect=interrupt):
                with self.assertRaises(InterruptedError): run(c,damage_solver='pinn')
            c.output.resume = True
            with contextlib.redirect_stdout(io.StringIO()): run(c,damage_solver='pinn')
            for name in ('network.eqx','optimizer.eqx','fields.npz'):
                # NPZ zip headers need not be byte-identical; compare fields.
                if name.endswith('eqx'):
                    self.assertEqual((base/'state_0002'/name).read_bytes(),
                                     (Path(c.output.root)/'state_0002'/name).read_bytes())
                else:
                    with np.load(base/'state_0002'/name) as a, np.load(Path(c.output.root)/'state_0002'/name) as b:
                        for key in ('q','d','H'): np.testing.assert_array_equal(a[key],b[key])
            c.output.resume = False
            c.output.root = str(root/'direct')
            with contextlib.redirect_stdout(io.StringIO()): run(c,damage_solver='direct')
            direct = json.loads((Path(c.output.root)/'result.json').read_text())
            self.assertTrue(direct['complete'])
            self.assertLess(direct['curve'][-1]['physical_AT2'],1e-7)
            c.sample.mode = 'random'
            with self.assertRaisesRegex(ValueError,'FEM damage'): validate(c)


if __name__ == '__main__':
    unittest.main()
