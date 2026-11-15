"""Independent element/quadrature, boundary, sampled-row and residual checks."""
import unittest
import tempfile
from pathlib import Path
import json

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from examples.electropolish2d.configs.train_operator_rolling_fem import get_config
from examples.electropolish2d.fem_weak import assemble_rectangle, RectangleWeakRows
from examples.electropolish2d.operator_data import validate_config
from examples.electropolish2d.operator_training import make_loss_vector, physical_fields
from examples.electropolish2d.spatial_sampling import SpatialSampler
from examples.corrosion2d.residual_experiment import nondim
from nonlocal_pinn.operators import build_D


def grid(nx, ny):
    x, y = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, .5, ny))
    return np.column_stack((x.ravel(), y.ravel()))


def independent_assembly(points, nx, ny):
    """Dense reference using barycentric Vandermonde inverses, not row formulas."""
    k, m = np.zeros((len(points), len(points))), np.zeros(len(points))
    for y in range(ny-1):
        for x in range(nx-1):
            sw = y*nx+x
            for ids in ([sw, sw+1, sw+nx+1], [sw, sw+nx+1, sw+nx]):
                affine = np.column_stack((np.ones(3), points[ids]))
                area = abs(np.linalg.det(affine))/2
                gradients = np.linalg.inv(affine)[1:]
                k[np.ix_(ids, ids)] += area*gradients.T@gradients
                m[ids] += area/3
    return k, m


class ToyField(eqx.Module):
    coefficient: jax.Array

    def prepare(self, image, points, known):
        return points

    def decode(self, points, times):
        x, y = points.T
        a, b = self.coefficient
        return jnp.stack((a*(x[None]**2-y[None])+times[:, None]*.2,
                          b*jnp.cos(2*x[None])+y[None]*times[:, None]), axis=-1)

    def __call__(self, image, points, times, known):
        return self.decode(points, times)


class FemWeakTests(unittest.TestCase):
    def test_common_local_monitor_against_independent_numpy_field_norm(self):
        from examples.electropolish2d.comparison_monitor import make_monitor
        from examples.electropolish2d.configs.train_operator_threeway_fem import get_config as config
        c = config(); validate_config(c)
        with tempfile.TemporaryDirectory() as tmp:
            # Synthetic references keep this unit test portable; the real64case
            # bank is checked separately in the experiment preflight.
            base, refs = Path(tmp)/'inputs', Path(tmp)/'references'
            base.mkdir(); refs.mkdir()
            c.operator.comparison_monitor.inputs_root = str(base)
            c.operator.comparison_monitor.reference_root = str(refs)
            source = json.loads(c.to_json()); source['nl']['method'] = 'sparse'
            source_path = base/'source.json'
            source_path.write_text(json.dumps(source))
            (base/'manifest.json').write_text(json.dumps(dict(count=64, source_config=str(source_path))))
            xx, yy = np.meshgrid(np.linspace(0, 1, 3), np.linspace(0, .5, 2))
            times = np.linspace(0, 1, 21)
            np.savez(base/'prediction.npz', X=xx, Y=yy, time=times.astype(np.float32),
                     images=np.zeros((64, 1, 2, 3), np.float32), initial=np.zeros((64, 6, 2), np.float32))
            for i in range(64):
                folder = refs/f'condition_{i:03d}'; folder.mkdir()
                fields = np.empty((21, 2, 3, 2))
                fields[..., 0] = .4+.02*i+.1*xx[None]+.05*times[:, None, None]
                fields[..., 1] = .3+.01*i+.1*yy[None]+.03*times[:, None, None]
                np.savez(folder/'local101.npz', X=xx, Y=yy, time=times, fields=fields)
            monitor = make_monitor(c, tmp)
            result = monitor(ToyField(jnp.array([.3, .02])), 7)
            record = json.loads((Path(tmp)/'local_metrics.jsonl').read_text())
            self.assertEqual(record, result)
            self.assertEqual(result['step'], 7)
            self.assertEqual([row['index'] for row in result['conditions']], list(range(64)))
            provenance = json.loads((Path(tmp)/'local_monitor.json').read_text())
            self.assertEqual(provenance['count'], 64)
            self.assertEqual(provenance['split'], 'test')
            expected_errors = []
            for row in result['conditions']:
                path = Path(c.operator.comparison_monitor.reference_root)/f"condition_{row['index']:03d}/local101.npz"
                with np.load(path) as data:
                    t = np.linspace(0, 1, 21)
                    ref = data['fields'][np.searchsorted(data['time'], t)]
                    x, y = data['X'][None], data['Y'][None]
                    phi = .5*(1+np.tanh(.3*(x*x-y)+t[:, None, None]*.2))
                    conc = c.physical.c_le*(.02*np.cos(2*x)+y*t[:, None, None])
                    conc += (c.physical.c_se-c.physical.c_le)*(3*phi**2-2*phi**3)
                scores = []
                for k, (name, value) in enumerate((('phi', phi), ('c', conc))):
                    expected = np.linalg.norm(value-ref[..., k])/np.linalg.norm(ref[..., k])
                    self.assertAlmostEqual(row[name], expected, places=6)
                    scores.append(expected)
                expected_errors.append(scores)
            np.testing.assert_allclose([result['phi'], result['c']], np.mean(expected_errors, axis=0), rtol=2e-6)
            for invalid in (tuple(range(4)), tuple(reversed(range(64))), tuple([0]*64)):
                c.operator.comparison_monitor.indices = invalid
                with self.assertRaisesRegex(ValueError, 'complete ordered64'):
                    make_monitor(c, tmp)

    def test_independent_elements_mass_energy_and_validation(self):
        points = grid(7, 5)
        k, mass, cells = assemble_rectangle(points, 7, 5)
        exact_k, exact_m = independent_assembly(points, 7, 5)
        np.testing.assert_allclose(k.toarray(), exact_k, atol=2e-14, rtol=1e-13)
        np.testing.assert_allclose(mass, exact_m, atol=2e-16, rtol=1e-13)
        np.testing.assert_allclose(k@np.ones(len(points)), 0, atol=2e-15)
        self.assertAlmostEqual(mass.sum(), .5)
        linear = 2*points[:, 0]-3*points[:, 1]+.7
        self.assertAlmostEqual(float(linear@k@linear), .5*(4+9), places=12)
        self.assertEqual(len(cells), 48)
        with self.assertRaisesRegex(ValueError, 'Cartesian'):
            assemble_rectangle(points[::-1], 7, 5)
        c = get_config(); validate_config(c)
        c.operator.fem.reaction = 'exact'
        with self.assertRaisesRegex(ValueError, 'FEM control'):
            validate_config(c)

    def test_natural_wall_refinement_and_dirichlet_columns(self):
        errors = []
        for nx in (9, 17, 33):
            ny = (nx+1)//2
            points = grid(nx, ny)
            op = RectangleWeakRows(points, nx, ny)
            u = np.cos(2*np.pi*points[:, 0])*np.cos(4*np.pi*points[:, 1])
            free = (points[:, 1] > 0) & (points[:, 1] < .5)
            target = 20*np.pi**2*u
            errors.append(np.linalg.norm((op.matrix@u-target)[free])/np.linalg.norm(target[free]))
            # y is harmonic and has exactly zero normal derivative at side walls.
            np.testing.assert_allclose((op.matrix@points[:, 1])[free], 0, atol=2e-11)
            self.assertGreater(op.matrix[free][:, ~free].nnz, 0)
        self.assertLess(errors[1], .3*errors[0])
        self.assertLess(errors[2], .3*errors[1])

    def test_complete_rolling_rows_share_actual_core_draws(self):
        points = grid(9, 5).astype(np.float32)
        op = RectangleWeakRows(points, 9, 5)
        nl = build_D(jnp.asarray(points), .18, method='sparse', kernel='cubic')
        c = get_config(); c.sample.K_core = 6; c.sample.K_pool = 3
        free = (points[:, 1] > 0) & (points[:, 1] < .5)
        values = np.random.default_rng(81).normal(size=len(points))
        ic, bc = np.array([12, 16]), np.flatnonzero(~free)
        with SpatialSampler(op, points, free, ic, bc, c) as sample:
            for i in range(12):
                ids, query, sub, mask, _ = sample.take(i)
                ids, mask = np.asarray(ids), np.asarray(mask)
                row, col, weight, diagonal = map(np.asarray, sub)
                z = values[ids[:len(mask)]]
                actual = diagonal*z
                np.add.at(actual, row, weight*z[col])
                np.testing.assert_allclose(actual[mask], (op.matrix@values)[ids[:len(mask)][mask]],
                                           rtol=3e-6, atol=3e-5)
                old_ids, _, old_mask, _ = nl.sub_sample(i, 6)
                np.testing.assert_array_equal(ids[:len(mask)][mask], old_ids[old_mask & free[old_ids]])
                np.testing.assert_array_equal(query[-len(bc):], points[bc])
        self.assertEqual(sample.summary()['entries_used'], 12)

    def test_physical_residual_and_parameter_gradient_against_dense_weak_form(self):
        c = get_config(); nx, ny = 5, 3
        points = grid(nx, ny).astype(np.float32)
        n = len(points)
        op = RectangleWeakRows(points, nx, ny)
        k, m = independent_assembly(points, nx, ny)
        dense = jnp.asarray(k/m[:, None], dtype=jnp.float32)
        free = (points[:, 1] > 0) & (points[:, 1] < .5)
        zero, one = np.flatnonzero(points[:, 1] == .5), np.flatnonzero(points[:, 1] == 0)
        ic = np.array([6, 8])
        queries = jnp.asarray(np.concatenate((points, points[ic], points[zero], points[one])))
        known = jnp.zeros((1, len(queries), 2))
        payload = jnp.zeros((1, 1, 2, 2)), known, queries
        sub = tuple(jnp.asarray(v) for v in (op._row_idx_np, op._col_idx_np, op._vals_np, op._diag))
        times = jnp.array([.1, .6])
        nd = nondim(c)
        vector = make_loss_vector(c, nd, len(ic), nx)
        net = ToyField(jnp.array([.3, .02]))

        def oracle(model):
            f, ft = jax.jvp(lambda t: physical_fields(model.decode(queries[:n], t), c),
                           (times,), (jnp.ones_like(times),))
            phi, conc = f[..., 0], f[..., 1]
            nch, n1, n2, n3, dc, cle = nd
            q = conc-(3*phi**2-2*phi**3)*dc-cle
            ac = ft[..., 0]+n3*(dense@phi.T).T+n2*(4*phi**3-6*phi**2+2*phi)-n1*q*dc*(6*phi-6*phi**2)
            ch = ft[..., 1]+nch*(dense@q.T).T
            top = physical_fields(model.decode(queries[n+len(ic):n+len(ic)+nx], times), c)
            bottom = physical_fields(model.decode(queries[-nx:], times), c)
            initial = physical_fields(model.decode(queries[n:n+len(ic)], jnp.zeros(1)), c)[0]
            return jnp.array([c.loss.ac_scale*jnp.mean(jnp.abs(ac[:, free])),
                              c.loss.ch_scale*jnp.mean(jnp.abs(ch[:, free])),
                              jnp.mean(jnp.sum(top**2, axis=-1)),
                              jnp.mean(jnp.sum((bottom-1)**2, axis=-1)),
                              jnp.mean(jnp.sum(initial**2, axis=-1))])

        def objective(model):
            return vector(model, times, payload, sub, jnp.asarray(free))
        np.testing.assert_allclose(objective(net), oracle(net), rtol=5e-5, atol=2e-5)
        a = eqx.filter_grad(lambda model: objective(model).sum())(net).coefficient
        b = eqx.filter_grad(lambda model: oracle(model).sum())(net).coefficient
        np.testing.assert_allclose(a, b, rtol=2e-4, atol=1e-4)


if __name__ == '__main__':
    unittest.main()
