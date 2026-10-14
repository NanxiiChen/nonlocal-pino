"""Fresh-run provenance and safe prefix-to-tail monitoring transition."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from ml_collections import ConfigDict

from examples.fracture2d.configs.cold_full_damage_run import get_config as controls
from examples.fracture2d.configs.cold_full_damage_prefix import get_config as prefix_config
from examples.fracture2d.full_damage_loading import validate
from examples.fracture2d.best_full_damage_loading import targets
from examples.fracture2d.cold_full_damage_run import tail_controls, sha
from examples.fracture2d.cold_full_damage_monitor import selected_root, check_zero


class FreshFullRunTest(unittest.TestCase):
    def test_fresh_prefix_and_same_run_tail_cover_exact_requested_loads(self):
        c=prefix_config();grid=validate(c)
        self.assertEqual(len(grid),52)
        self.assertAlmostEqual(grid[0],1e-4)
        self.assertAlmostEqual(grid[-1],.0052)
        self.assertEqual(c.full_loading.prefix_outer_cap,128)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'independent_review').mkdir();(p/'state_0052').mkdir()
            records={'identity.json':dict(own_zero=True),
                'result.json':dict(complete=True,curve=[dict(index=52)]),
                'independent_review/audit.json':dict(complete=False,pass_replay=True),
                'state_0052/result.json':dict(load=.0052,index=52,
                    q_d_synchronized=True,optimizer_count=384000)}
            for name,value in records.items():(p/name).write_text(json.dumps(value))
            for name in ('parameterization.npz','state_0052/fields.npz',
                    'state_0052/network.eqx','state_0052/optimizer.eqx'):
                (p/name).write_bytes(b'fresh same-run provenance fixture')
            with self.assertRaises(ValueError):tail_controls(controls(),p)
            (p/'independent_review/audit.json').write_text(json.dumps(dict(complete=True,pass_replay=True)))
            frozen=tail_controls(controls(),p)
            self.assertEqual(frozen['source_root'],str(p))
            self.assertEqual(frozen['source_optimizer_count'],384000)
            self.assertEqual(frozen['source_pins']['state_0052/network.eqx'],sha(p/'state_0052/network.eqx'))
            tail=targets(ConfigDict(frozen,allow_dotted_keys=True))
            self.assertEqual(len(tail),8)
            np.testing.assert_allclose(np.concatenate((grid,tail)),np.arange(1,61)*1e-4,rtol=0,atol=1e-14)
            self.assertEqual(frozen['max_outer_sweeps'],1024)

    def test_monitor_requires_actual_zero_and_published_tail_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            prefix=Path(tmp)/'prefix';tail=Path(tmp)/'tail';prefix.mkdir();tail.mkdir()
            zero=prefix/'state_0000';zero.mkdir()
            (prefix/'identity.json').write_text(json.dumps(dict(own_zero=True,optimizer_persistent=True)))
            fields=zero/'fields.npz'
            np.savez(fields,q=np.zeros(2),d=np.zeros(1),H=np.zeros(1))
            row=dict(load=0.,reaction=0.,optimizer_count=0,fields_sha256=sha(fields))
            (zero/'result.json').write_text(json.dumps(row))
            check_zero(prefix)
            self.assertEqual(selected_root(prefix,tail),prefix)
            (tail/'preflight.json').write_text('{}')
            self.assertEqual(selected_root(prefix,tail),prefix)
            (tail/'result.json').write_text('{}')
            self.assertEqual(selected_root(prefix,tail),tail)
            np.savez(fields,q=np.zeros(2),d=np.ones(1)*1e-8,H=np.zeros(1))
            row['fields_sha256']=sha(fields);(zero/'result.json').write_text(json.dumps(row))
            with self.assertRaises(ValueError):check_zero(prefix)


if __name__=='__main__':unittest.main()
