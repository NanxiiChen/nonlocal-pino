"""Known-input parity, deterministic FIFO refresh and safe fixed-shape updates."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import tempfile
import time
import unittest
from unittest.mock import patch
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict
from examples.channel_nsch2d.configs.train_operator import get_config
from examples.channel_nsch2d.data import geometry_data
from examples.channel_nsch2d.online import (
    sample_geometry,build_input,validate_online,validate_payload,replace_slots,RollingGeometryPool,audit_training_inputs,validate_construction_screen)


def fixture():
    c=get_config();c.train.seed=42;c.operator.train_conditions=4
    c.sample.Nx=8;c.sample.Ny=4;c.sample.boundary_points=6;c.nl.neighbors=20;c.nl.degree=2
    c.operator.online=ConfigDict(dict(workers=2,refresh_size=2,refresh_every=2,prefetch_batches=2))
    return c


class ChannelOnline(unittest.TestCase):
    def test_construction_screen_covers_actual_budget_and_prior(self):
        from examples.channel_nsch2d.problem import identity
        c=fixture();c.train.iterations=6
        value=dict(status='passed',operator=c.nl.to_dict(),grid=[c.sample.Nx,c.sample.Ny],
            physical_family=identity(c),online_draws=8,run_seed=c.train.seed,initial_bank_seed=c.operator.dataset_seed,
            initial_geometries=4,maximum_condition=1e6,maximum_polynomial_error=1e-12,
            pde_solves=0,fem_solves=0,stored_operator_arrays=0,
            cases=[dict(kind='initial',index=i) for i in range(4)]+[dict(kind='online',index=i) for i in range(8)])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'screen.json';path.write_text(json.dumps(value))
            self.assertEqual(validate_construction_screen(c,path)['online_draws'],8)
            value['online_draws']=7;path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,'does not cover'):validate_construction_screen(c,path)
            value['online_draws']=8;value['cases'][-1]['index']=0;path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,'missing or duplicate'):validate_construction_screen(c,path)
            value['cases'][-1]['index']=7;path.write_text(json.dumps(value))
            c.nl.weight_exponent*=2
            with self.assertRaisesRegex(ValueError,'does not cover'):validate_construction_screen(c,path)

    def test_canonical_entry_trains_on_replaced_inputs_and_reconstructs_them(self):
        import contextlib
        import io
        from examples.channel_nsch2d import train as entry
        from examples.channel_nsch2d.configs.train_single_conservative import get_config as conservative
        from examples.channel_nsch2d.problem import identity
        c=conservative();c.train.seed=42;c.train.iterations=6
        c.sample.Nx=8;c.sample.Ny=8;c.sample.Nt=2;c.sample.boundary_points=6
        c.nl.neighbors=20;c.nl.degree=2
        c.arch.width=4;c.arch.heads=1;c.arch.condition_depth=1;c.arch.decoder_depth=1
        c.arch.spatial_features=2;c.arch.time_features=2
        c.operator.train_conditions=4;c.operator.condition_batch=1
        c.operator.online=fixture().operator.online;c.log.eval_every=3;c.log.checkpoint_every=3
        c.train.soap_max_precond_dim=32
        def executor(**kw):return ThreadPoolExecutor(max_workers=kw['max_workers'])
        # Synthetic labels supply only the reporting fixture. Canonical network,
        # full NS--CH/SBP/BDF2 loss, SOAP, sampler and rolling updates are real.
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);c.reference.root=str(root);c.log.log_dir=str(root/'runs');c.log.run_name='online'
            from examples.channel_nsch2d.cohort import prepare
            c.operator.validation_manifest=str(root/'reserved'/'manifest.json')
            prepare(c,c.operator.validation_manifest,4,(96,40))
            target=np.ones((13,3,5));target[...,0]=.5
            reference=root/'reference.npz'
            np.savez(reference,points=np.array([[.2,.3],[.8,.5],[2.,.5]]),
                time=np.linspace(0,6,13),amplitudes=np.array([.12,.08]),fields=target)
            (root/'reference-validation.json').write_text(json.dumps(dict(status='passed',reference=str(reference),identity=identity(c))))
            with patch('examples.channel_nsch2d.online.ProcessPoolExecutor',executor),contextlib.redirect_stdout(io.StringIO()):
                entry.train(c)
            run=Path(c.log.log_dir)/'online';report=audit_training_inputs(run,c)
            self.assertEqual(report['refresh_count'],2);self.assertEqual(report['installed_inputs'],4)
            self.assertGreater(report['distinct_sampled_online_draws'],0)
            self.assertNotEqual((run/'model_initial.eqx').read_bytes(),(run/'model_final.eqx').read_bytes())
            summary=json.loads((run/'summary.json').read_text())
            self.assertEqual(summary['online']['installed_inputs'],4)
            self.assertEqual(summary['supervised_evolved_field_samples'],0)
            rows=[json.loads(row) for row in (run/'loss.jsonl').read_text().splitlines()]
            self.assertTrue(all(np.isfinite(row['total']) for row in rows))
            # Deliberate corruptions must not pass provenance validation.
            path=run/'online_refresh.jsonl';original=path.read_text()
            bad=[json.loads(row) for row in original.splitlines()];bad[0]['completed_updates']+=1
            path.write_text(''.join(json.dumps(row)+'\n' for row in bad))
            with self.assertRaisesRegex(ValueError,'Refresh update'):audit_training_inputs(run,c)
            path.write_text(original)

    def test_stateless_prior_and_exact_canonical_input_parity(self):
        c=fixture();validate_online(c)
        for i in range(5):
            a=sample_geometry(c,i);np.testing.assert_array_equal(a,sample_geometry(c,i))
            self.assertTrue(np.all((a>=c.geometry.amplitude_min)&(a<c.geometry.amplitude_max)))
        self.assertFalse(np.array_equal(sample_geometry(c,0),sample_geometry(c,1)))
        payload,record=build_input(c.to_dict(),3);raw,_=geometry_data(sample_geometry(c,3),c)
        for k,value in raw.items():
            np.testing.assert_array_equal(payload[k],value.astype('int32' if k=='neighbors' else c.train.compute_dtype))
        self.assertEqual(record['pde_solves'],0);self.assertEqual(record['fem_solves'],0)
        with self.assertRaisesRegex(ValueError,'ten known-input'):validate_payload(dict(payload,reference=np.zeros(4)),c)
        with self.assertRaisesRegex(ValueError,'Invalid online points'):
            validate_payload(dict(payload,points=payload['points'][:-1]),c)
        c.sample.Ny=8;c.phase_transport=ConfigDict(dict(kind='sbp4_mapped_curl'))
        augmented,_=build_input(c.to_dict(),3);self.assertEqual(len(augmented),16)
        validate_payload(augmented,c)

    def test_fifo_refresh_and_fixed_shape_update_leave_eval_inputs_frozen(self):
        c=fixture();initial,_=build_input(c.to_dict(),99)
        frozen={k:v.copy() for k,v in initial.items()}
        bank={k:jnp.asarray(np.stack([v]*4)) for k,v in initial.items()}
        expected=dict(initial)
        def executor(**kw):return ThreadPoolExecutor(max_workers=kw['max_workers'])
        def delayed(config,draw):
            if draw%2==0:time.sleep(.01)
            return build_input(config,draw)
        with tempfile.TemporaryDirectory() as directory,\
             patch('examples.channel_nsch2d.online.ProcessPoolExecutor',executor),\
             patch('examples.channel_nsch2d.online.build_input',delayed):
            pool=RollingGeometryPool(c,directory,np.empty((0,2))).start()
            replace=jax.jit(replace_slots,donate_argnums=(0,))
            try:
                for step in range(6):
                    result=pool.before_step(step)
                    if result is not None:
                        slots,updates=result;bank=replace(bank,slots,updates)
                np.testing.assert_array_equal(pool.draw_ids,[0,1,2,3])
                state=json.loads(pool.save_state(6).read_text())
                self.assertEqual(state['refresh_count'],2)
                slots,updates=pool.before_step(6);np.testing.assert_array_equal(slots,[0,1])
                bank=replace(bank,slots,updates)
                np.testing.assert_array_equal(pool.draw_ids,[4,5,2,3])
                for slot,draw in enumerate((4,5,2,3)):
                    value,_=build_input(c.to_dict(),draw)
                    for k in value:np.testing.assert_array_equal(np.asarray(bank[k])[slot],value[k])
                with self.assertRaisesRegex(ValueError,'Missed scheduled'):pool.before_step(9)
                rows=[json.loads(s) for s in (Path(directory)/'online_refresh.jsonl').read_text().splitlines()]
                self.assertEqual([r['draw_ids'] for r in rows],[[0,1],[2,3],[4,5]])
                self.assertEqual(pool.summary()['bounded_prefetch_inputs'],4)
            finally:pool.close()
        for k in initial:np.testing.assert_array_equal(frozen[k],expected[k])

    def test_reserved_geometry_and_invalid_schedule_fail(self):
        c=fixture()
        def executor(**kw):return ThreadPoolExecutor(max_workers=kw['max_workers'])
        with tempfile.TemporaryDirectory() as directory,patch('examples.channel_nsch2d.online.ProcessPoolExecutor',executor):
            pool=RollingGeometryPool(c,directory,sample_geometry(c,0)[None]).start()
            try:
                with self.assertRaisesRegex(ValueError,'nonnegative'):pool.before_step(-1)
                with self.assertRaisesRegex(ValueError,'duplicates a reserved'):pool.before_step(2)
            finally:pool.close()
        c.operator.online.refresh_size=3
        with self.assertRaisesRegex(ValueError,'divisible'):validate_online(c)
        c.operator.online.refresh_size=2;c.train.iterations=2
        with self.assertRaisesRegex(ValueError,'actual pool replacement'):validate_online(c)
        c.train.iterations=6;c.train.continuation=ConfigDict(dict(parent_run='/tmp/unused',kind='strict'))
        with self.assertRaisesRegex(ValueError,'recovery'):validate_online(c)


if __name__=='__main__':unittest.main()
