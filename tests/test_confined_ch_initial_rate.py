"""Canonical initial-rate dispatch, physical derivatives and input lifecycle."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import equinox as eqx
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
import numpy as np

from tests.test_confined_ch_operator import small_fixture
from examples.confined_ch2d.operator_initial_rate import initial_rate_data
from examples.confined_ch2d.operator_online import pad_input
from examples.confined_ch2d.operator_potential import compact_payload, full_state
from examples.confined_ch2d import operator_training as canonical
from examples.confined_ch2d.operator_problem import (
    validate, specification, sample_geometries, condition, case_config, fixed_initial)


def fixture():
    c, raw = small_fixture()
    c.train.seed = 42
    c.operator.formulation = 'conservative_potential'
    c.operator.metric = 'identity'
    c.operator.initial_rate_lift = ConfigDict(dict(kind='exact_pde_rate', timescale='stable_graph_bound'))
    return c, {key: np.asarray(value) for key, value in raw.items() if key != 'metric'}


class CanonicalInitialRateTests(unittest.TestCase):
    def test_explicit_presets_and_default_compatibility(self):
        from examples.confined_ch2d.configs.train_operator import get_config as default
        from examples.confined_ch2d.configs.train_operator_potential import get_config as old
        from examples.confined_ch2d.configs.train_operator_initial_rate import get_config as full
        from examples.confined_ch2d.configs.train_operator_initial_rate_spatial import get_config as spatial
        self.assertNotIn('initial_rate_lift', default().operator)
        a, b = full(), spatial()
        validate(a); validate(b)
        del b.sample.spatial
        self.assertEqual(a.to_dict(), b.to_dict())
        del a.operator.initial_rate_lift
        self.assertEqual(a.to_dict(), old().to_dict())
        for kind in ('bad_kind', 'bad_timescale', 'direct_phase', 'missing_field'):
            c = full()
            if kind == 'bad_kind': c.operator.initial_rate_lift.kind = 'unknown'
            elif kind == 'bad_timescale': c.operator.initial_rate_lift.timescale = 'unknown'
            elif kind == 'direct_phase': del c.operator.formulation
            else: del c.operator.initial_rate_lift.timescale
            with self.assertRaises(ValueError): validate(c)

    def test_initial_jvp_and_compact_parameter_gradients(self):
        with jax.enable_x64(True):
            c, raw = fixture()
            c.loss.initial_residual_fraction = .2
            data, _ = initial_rate_data(raw, c)
            core = np.array([2, 0, 1, 1], dtype=np.int32)
            w = raw['volumes']/raw['volumes'].sum()
            compact, _ = compact_payload(data, core, np.array([w[2], w[0], w[1]/2, w[1]/2]),
                                         inner_capacity=4, query_capacity=5)
            net = canonical.build_network(c, jax.random.key(93))
            arrays = jax.tree.map(jnp.asarray, data)
            prepared = canonical.prepare(net, arrays)
            phi, dt = jax.jvp(lambda t: canonical.phase(net, prepared, t, arrays, c),
                              (jnp.asarray(0.),), (jnp.asarray(1.),))
            mu0 = c.physical.beta*(phi**3-phi)+c.physical.epsilon**2*canonical.apply_operator(phi, arrays)
            np.testing.assert_array_equal(phi, raw['initial'])
            np.testing.assert_allclose(dt, -c.physical.mobility*canonical.apply_operator(mu0, arrays), atol=1e-12)
            times = jnp.array([0., .01, .1, c.physical.terminal_time])
            output = canonical.fields(net, arrays, times, c)
            np.testing.assert_allclose(output[:, :, 0]@raw['volumes'], raw['initial']@raw['volumes'], atol=1e-13)
            for i, t in enumerate(times):
                phase, mu, _ = full_state(net, prepared, t, arrays, c)
                np.testing.assert_allclose(output[i], jnp.stack((phase, mu), axis=-1), atol=1e-12)
            batch = lambda d: jax.tree.map(lambda a: jnp.asarray(a)[None], d)
            fn = eqx.filter_jit(eqx.filter_value_and_grad(canonical.make_loss(c)))
            value, gradient = fn(net, batch(data), times)
            other, derivative = fn(net, batch(compact), times)
            np.testing.assert_allclose(value, other, rtol=1e-10, atol=1e-12)
            for a, b in zip(jax.tree.leaves(gradient), jax.tree.leaves(derivative), strict=True):
                np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-11)

    def test_rounded_initial_rate_padding_and_invalid_input(self):
        with jax.enable_x64(True):
            c, raw = fixture()
            c.train.compute_dtype = 'float32'
            c.operator.online = ConfigDict(dict(workers=1, refresh_size=1, refresh_every=2,
                prefetch_batches=1, max_points=7, max_neighbors=4, origin_step=0))
            data, report = initial_rate_data(raw, c)
            padded = pad_input(data, c)
            reconstructed, _ = initial_rate_data(padded, c)
            np.testing.assert_array_equal(reconstructed['initial_potential_rate'], padded['initial_potential_rate'])
            np.testing.assert_array_equal(padded['initial_potential_rate'][4:], 0.)
            self.assertEqual(padded['initial_rate_timescale'].shape, ())
            self.assertGreater(report['timescale'], 0.)
            self.assertEqual(padded['initial_potential_rate'].dtype, np.dtype('float32'))
            for bad in (dict(data, phi=np.zeros(4)), dict(data, initial_rate_timescale=np.array(-1.))):
                with self.assertRaises(ValueError): pad_input(bad, c)
            with self.assertRaises(ValueError):
                initial_rate_data(dict(raw, coefficients=-raw['coefficients']), c)

    def test_real_bank_loader_and_short_canonical_training(self):
        from examples.confined_ch2d.operators import RegionalLaplacian
        with jax.enable_x64(True), tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c, _ = fixture()
            c.operator.train_conditions=2; c.operator.test_conditions=1; c.operator.condition_batch=2
            c.operator.local_validation_indices=()
            c.operator.dataset_path=str(root/'data'); (root/'data').mkdir()
            c.train.iterations=3; c.sample.Nt=2; c.reference.snapshots=3
            c.log.log_dir=str(root/'runs'); c.log.eval_every=3; c.log.checkpoint_every=3
            x=np.array([[.46,.65],[.5,.65],[.54,.65]]); v=np.array([.1,.2,.15])
            holes=sample_geometries(c)[0]; features, measure=condition(holes,c)
            op=RegionalLaplacian(x,v,case_config(c,holes,0)); initial=fixed_initial(x,c,xp=np)
            common=dict(points=x,volumes=v,holes=holes,condition=features,condition_weights=measure,
                initial=initial,neighbors=np.asarray(op.neighbors),coefficients=np.asarray(op.coefficients),
                metric=np.eye(3),mesh_points=x,mesh_cells=np.array([[0,1,2]]))
            for i in range(3):
                labels={} if i==1 else dict(time=np.linspace(0,c.physical.terminal_time,3),
                    phi=np.broadcast_to(initial,(3,3)),mu=np.full((3,3),123.))
                np.savez(root/'data'/f'{i}.npz',**common,**labels)
            manifest=dict(format='operator_only_v1',status='passed',specification=specification(c),
                train_indices=[0,1],test_indices=[2],monitor_train_indices=[0],
                conditions=[dict(index=i,file=f'{i}.npz',holes=holes.tolist(),has_reference=i!=1) for i in range(3)])
            (root/'data/manifest.json').write_text(json.dumps(manifest))
            _, refs, bank=canonical.load_bank(c)
            self.assertNotIn('phi',refs[1]); self.assertNotIn('mu',refs[1])
            self.assertNotIn('phi',bank); self.assertNotIn('mu',bank)
            expected,_=initial_rate_data(common,c)
            np.testing.assert_array_equal(bank['initial_potential_rate'][1], expected['initial_potential_rate'])
            for name in ('full','compact'):
                c.log.run_name=name
                if name=='compact':
                    c.sample.spatial=ConfigDict(dict(mode='rolling_groups',core_count=8,group_size=4,
                        cell_width=.12,wall_groups=0,wall_band=c.nl.delta,inner_capacity=4,
                        query_capacity=5,pools=2,refresh_every=1,workers=1))
                with redirect_stdout(io.StringIO()): canonical.train_operator(c)
                run=root/'runs'/name
                summary=json.loads((run/'summary.json').read_text())
                self.assertEqual(summary['new_updates'],3)
                self.assertEqual(len(json.loads((run/'initial_rate_lift.json').read_text())['conditions']),3)
                with np.load(run/'prediction_002.npz') as saved:
                    np.testing.assert_array_equal(saved['phi'][0],initial)
            self.assertEqual((root/'runs/full/model_initial.eqx').read_bytes(),(root/'runs/compact/model_initial.eqx').read_bytes())
            np.savez(root/'data/1.npz',**common,initial_potential_rate=np.ones(3))
            with self.assertRaisesRegex(ValueError,'must be derived'):canonical.load_bank(c)


if __name__ == '__main__':
    unittest.main()
