"""Check inherited-prefix provenance, then reuse independent native loading replay."""
import argparse
import json
from pathlib import Path
import time

import jax
from ml_collections import ConfigDict

from examples.fracture2d._legacy.best_full_damage_loading import targets
from examples.fracture2d._legacy.coupling_diagnostics import atomic_json
from examples.fracture2d._legacy.full_damage_loading_audit import main as native_replay
from examples.fracture2d._legacy.local_damage_trial import sha


def main(root):
    if jax.devices()[0].platform!='cpu':raise ValueError('best-route independent replay requires CPU')
    while not (root/'preflight.json').exists():
        if (root.parent/'exit.json').exists():raise RuntimeError('best route fails before restore preflight')
        time.sleep(5)
    identity=json.loads((root/'identity.json').read_text())
    a=ConfigDict(identity['driver_config'],allow_dotted_keys=True)
    grid=targets(a);source=Path(a.source_root)
    for name,digest in a.source_pins.items():assert sha(source/name)==digest,name
    receipt=json.loads((root/'prefix_receipt.json').read_text())
    assert receipt['source_root']==str(source)
    assert len(receipt['rows'])==a.source_index+1
    for index,row in enumerate(receipt['rows']):
        assert row['index']==index and row['source']==str(source/f'state_{index:04d}')
        for name,digest in row['files'].items():
            assert sha(Path(row['source'])/name)==digest
            assert sha(root/f'state_{index:04d}'/name)==digest
    actual=json.loads((root/f'state_{a.source_index:04d}/result.json').read_text())
    assert actual['optimizer_count']==a.source_optimizer_count and actual['load']==a.source_load
    assert identity['source_index']==a.source_index and identity['source_optimizer_count']==a.source_optimizer_count
    assert abs(identity['loads'][-1]-a.end_load)<1e-14
    assert len(identity['loads'])==a.source_index+len(grid)
    c=identity['config']['full_loading']
    assert c['inner_updates']==a.inner_updates and c['prefix_outer_cap']==a.max_outer_sweeps
    assert c['critical_outer_cap']==a.max_outer_sweeps and c['symmetry_constraint'] is False
    atomic_json(root/'inheritance_audit.json',dict(pass_prefix_copy=True,
        source_optimizer_count=a.source_optimizer_count,inherited_positive_loads=a.source_index,
        newly_requested_loads=len(grid),auditor_sha256=sha(__file__)))
    native_replay(root,watch=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);args=parser.parse_args()
    jax.config.update('jax_enable_x64',True);main(Path(args.root))
