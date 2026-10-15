"""Conditioned loss, one-time bootstrap and physical-zero endpoint restart."""
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
from examples.fracture2d.history_conditioning import make_conditioned_loss
from examples.fracture2d.history_training import HistoryDamagePINN
from examples.fracture2d.loading import run
from examples.fracture2d.mesh import load_geometry
from examples.fracture2d.sampling import pack_rows
from examples.fracture2d.storage import atomic_json
from tests.test_fracture2d_history_loading import history_config


def configuration(root):
    c = history_config(root)
    c.arch.type = 'conditioned_history_operator'
    c.history.normalized = True
    c.history.local_history = True
    c.history.bootstrap_l1_steps = 2
    c.history.bootstrap_l2_steps = 3
    c.history.replay_enabled = False
    c.loss.norm = 'l2'
    return c


class ConditionedLoadingTests(unittest.TestCase):
    def test_dynamic_losses_use_raw_H_and_keep_one_trace_across_bootstrap(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = configuration(Path(tmp))
            m, op, _ = load_geometry(c)
            p = HistoryDamagePINN(c, m, op)
            H = jnp.array([1., 2., 3., 4.])
            ctx = p.context(H)
            columns, values = pack_rows(op.to_scipy(), width=4)
            points = jnp.asarray(2*m.coordinates[:4]-1)
            image, objective = make_conditioned_loss(coarse=p.coarse, fine=p.fine,
                points=points, columns=jnp.asarray(columns), values=jnp.asarray(values),
                gc0=p.gc0, gc1=c.physical.Gc*c.physical.l0, floor=c.loss.floor)
            d = np.asarray(image(p.net, ctx))
            r = (p.gc0+2*np.asarray(H))*d+c.physical.Gc*c.physical.l0*(op.to_scipy()@d)-2*np.asarray(H)
            rho = r/(2*np.sqrt(np.mean(np.asarray(H)**2)))
            traces = []
            def counted(net, batch):
                traces.append(1)
                return objective(net, batch)
            value = eqx.filter_jit(counted)
            for l2, expected in ((False,np.mean(abs(rho))),(True,np.mean(rho**2))):
                batch = dict(H=H[None], fine_ids=ctx['fine_ids'][None],
                    weights=jnp.ones(1), l2=jnp.asarray(l2))
                self.assertAlmostEqual(float(value(p.net,batch)),float(expected),places=12)
                np.testing.assert_array_equal(jax.grad(lambda x:objective(p.net,dict(batch,H=x)))(batch['H']),0.)
            self.assertEqual(len(traces),1)
            flags = []
            original = p.step
            def record(*args):
                flags.append(bool(args[-1]))
                self.assertFalse(bool(args[-2]))
                return original(*args)
            p.step = record
            p.begin_load(1,.0001)
            p.solve(H,jnp.zeros(4),.0001)
            p.solve(2*H,jnp.zeros(4),.0001)
            self.assertEqual(flags,[False,False,True,True,True,True,True])
            self.assertEqual(p.count,7)
            self.assertEqual(int(p.replay.count),1)
            c.history.bootstrap_l1_steps = 0
            with self.assertRaisesRegex(ValueError,'bootstrap'):
                HistoryDamagePINN(c,m,op)

    def test_cold_stagger_and_endpoint_resume_do_not_repeat_bootstrap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = configuration(root)
            with contextlib.redirect_stdout(io.StringIO()):
                run(c,damage_solver='pinn')
            base = Path(c.output.root)
            report = json.loads((base/'result.json').read_text())
            self.assertTrue(report['complete'])
            self.assertEqual([r['optimizer_count'] for r in report['curve']],[9,15])
            self.assertEqual([r['replay_snapshots'] for r in report['curve']],[2,4])
            with np.load(base/'state_0000/fields.npz') as f:
                for name in ('q','d','H'):
                    np.testing.assert_array_equal(f[name],0.)
            initial = json.loads((base/'state_0000/replay_state.json').read_text())
            self.assertEqual(initial['optimizer_count'],0)
            self.assertEqual(initial['snapshots'],0)
            with np.load(base/'state_0002/fields.npz') as f:
                expected = {name:f[name].copy() for name in ('q','d','H')}
            c.output.root = str(root/'interrupted')
            def interrupt(path, data):
                atomic_json(path,data)
                if path.name=='result.json' and 'curve' in data and len(data['curve'])==1:
                    raise InterruptedError('first endpoint saved')
            with contextlib.redirect_stdout(io.StringIO()), patch(
                    'examples.fracture2d.loading.atomic_json',side_effect=interrupt):
                with self.assertRaises(InterruptedError):
                    run(c,damage_solver='pinn')
            c.output.resume = True
            with contextlib.redirect_stdout(io.StringIO()):
                run(c,damage_solver='pinn')
            resumed = Path(c.output.root)
            with np.load(resumed/'state_0002/fields.npz') as f:
                for name,array in expected.items():np.testing.assert_array_equal(f[name],array)
            self.assertEqual(json.loads((resumed/'result.json').read_text())['current']['optimizer_count'],15)
            for name in ('network.eqx','optimizer.eqx'):
                self.assertEqual((base/'state_0002'/name).read_bytes(),(resumed/'state_0002'/name).read_bytes())


if __name__ == '__main__':
    unittest.main()
