"""Accepted-draw remapping and immutable incremental bank reuse, without solves."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
from ml_collections import ConfigDict
from examples.confined_ch2d.configs.train_operator_mild import get_config
from examples.confined_ch2d.operator_problem import sample_geometries, specification, validate
from examples.confined_ch2d.operator_precompute import prepare


class IncrementalBankTests(unittest.TestCase):
    def fixture(self, directory):
        c=get_config();c.operator.train_conditions=4;c.operator.test_conditions=2
        c.operator.local_validation_indices=()
        source=Path(directory)/'source';source.mkdir()
        c.operator.dataset_path=str(source)
        c.operator.precompute=ConfigDict(dict(workers=1,source_path=str(source),
            reserved_test_draws=(1,3),monitor_train_indices=(0,1),reuse_mode='hardlink'))
        (source/'config.json').write_text(c.to_json())
        records=[];order=[0,2,4,5,1,3]
        for i,(holes,draw) in enumerate(zip(sample_geometries(c),order,strict=True)):
            labeled=i in (0,1,4,5)
            file=f'case_{i:03d}/'+('reference.npz' if labeled else 'operator.npz')
            p=source/file;p.parent.mkdir();p.write_bytes(f'unchanged accepted draw {draw}'.encode())
            records.append(dict(index=i,draw_index=draw,file=file,holes=holes.tolist(),has_reference=labeled))
        manifest=dict(format='operator_only_v1',status='passed',specification=specification(c),
            train_indices=list(range(4)),test_indices=[4,5],monitor_train_indices=[0,1],conditions=records)
        (source/'manifest.json').write_text(json.dumps(manifest))
        c.operator.train_conditions=6;c.operator.dataset_path=str(Path(directory)/'target')
        return c,source,records

    def test_mixed_source_hardlinks_and_new_draws(self):
        with tempfile.TemporaryDirectory() as directory:
            c,source,records=self.fixture(directory)
            calls=[]
            def build(config,index,holes,draw):
                calls.append(draw)
                p=Path(config['operator']['dataset_path'])/f'case_{index:03d}/operator.npz'
                p.parent.mkdir();p.write_bytes(b'new known input only')
                return dict(index=index,draw_index=draw,holes=holes.tolist(),
                    file=f'case_{index:03d}/operator.npz',has_reference=False)
            def executor(max_workers,mp_context):
                return ThreadPoolExecutor(max_workers=max_workers)
            with patch.dict(os.environ,{'JAX_PLATFORMS':'cpu'}),\
                 patch('examples.confined_ch2d.operator_precompute.ProcessPoolExecutor',executor),\
                 patch('examples.confined_ch2d.operator_precompute.build_one',build):
                root=prepare(c)
            made=json.loads((root/'manifest.json').read_text())
            self.assertEqual(sorted(calls),[6,7])
            self.assertEqual([r['draw_index'] for r in made['conditions']],[0,2,4,5,6,7,1,3])
            self.assertEqual(made['new_operators'],2)
            self.assertEqual(made['reused_operators'],2)
            self.assertEqual(made['reused_references'],4)
            self.assertEqual(made['pde_trajectory_solves'],0)
            self.assertEqual(made['fem_solves'],0)
            for r in made['conditions']:
                self.assertEqual(r['has_reference'],r['index'] in (0,1,6,7))
                if r['draw_index'] in (6,7):continue
                old=next(a for a in records if a['draw_index']==r['draw_index'])
                a=source/old['file'];b=root/r['file']
                self.assertEqual(a.read_bytes(),b.read_bytes())
                self.assertEqual(a.stat().st_ino,b.stat().st_ino)
                self.assertEqual(a.read_bytes(),f"unchanged accepted draw {r['draw_index']}".encode())

    def test_invalid_reuse_and_changed_heldout_draws_fail_early(self):
        with tempfile.TemporaryDirectory() as directory:
            c,source,_=self.fixture(directory)
            c.operator.precompute.reuse_mode='fallback'
            with self.assertRaisesRegex(ValueError,'reuse_mode'):validate(c)
            c.operator.precompute.reuse_mode='hardlink'
            c.operator.precompute.reserved_test_draws=(0,3)
            with patch.dict(os.environ,{'JAX_PLATFORMS':'cpu'}),\
                 self.assertRaisesRegex(ValueError,'Geometry prior mismatch|exact original held-out'):prepare(c)
            self.assertFalse(Path(c.operator.dataset_path).exists())
            c.operator.precompute.reserved_test_draws=(1,3)
            c.operator.precompute.monitor_train_indices=(0,4)
            with patch.dict(os.environ,{'JAX_PLATFORMS':'cpu'}),\
                 self.assertRaisesRegex(ValueError,'No existing refined reference'):prepare(c)
            self.assertFalse(Path(c.operator.dataset_path).exists())


if __name__=='__main__':unittest.main()
