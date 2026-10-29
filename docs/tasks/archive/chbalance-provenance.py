"""Direct-file/array parity for the matched loss-balancing controls; no hashes."""
from pathlib import Path
import json,sys
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
sys.path.insert(0,str(ROOT))
from examples.confined_ch2d.problem import load_reference
from ml_collections import ConfigDict
OUT=Path(__file__).parent;BASE=Path('/root/autodl-tmp/tf-logs/confined_ch2d')
names=['20261028-chbalance-'+label+'-01' for label in ('joint','gradnorm','stagger')]
runs=[BASE/name for name in names]
first=runs[0];initial=(first/'model_initial.eqx').read_bytes()
sources={str(p.relative_to(first/'source')):p.read_bytes() for p in (first/'source').rglob('*') if p.is_file()}
base_config=json.loads((first/'config.json').read_text());base_config['train'].pop('loss_balance');base_config['log']['run_name']=''
with np.load(first/'training_times.npz') as a:times=a['times'].copy();horizons=a['horizons'].copy()
report=dict(status='passed',runs=[],source_file_count=len(sources),initial_parameters='Direct serialized bytes identical',
    scope='Same architecture/initialization/physical reference/full-horizon times; only loss_balance config and run name vary; no hash loading')
for run in runs:
    c=json.loads((run/'config.json').read_text());strategy=c['train'].pop('loss_balance');c['log']['run_name']=''
    assert c==base_config,run
    assert (run/'model_initial.eqx').read_bytes()==initial,run
    actual={str(p.relative_to(run/'source')):p.read_bytes() for p in (run/'source').rglob('*') if p.is_file()}
    assert actual==sources,run
    with np.load(run/'training_times.npz') as a:
        np.testing.assert_array_equal(a['times'],times);np.testing.assert_array_equal(a['horizons'],horizons)
    ref=load_reference(ConfigDict(json.loads((run/'config.json').read_text())))
    np.testing.assert_array_equal(horizons,np.full(5000,1.5))
    assert ref['phi_local'].shape==(21,3682)
    report['runs'].append(dict(run=str(run),strategy=strategy,full_horizon=True))
(OUT/'chbalance-provenance.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
