"""Separate later raw-control provenance, without a four-way source-parity claim."""
from pathlib import Path
import json
import numpy as np
OUT=Path(__file__).parent;BASE=Path('/root/autodl-tmp/tf-logs/confined_ch2d')
old=BASE/'20261028-chbalance-gradnorm-01';raw=BASE/'20261028-chbalance-gradnorm-raw-01'
assert (old/'model_initial.eqx').read_bytes()==(raw/'model_initial.eqx').read_bytes()
a=json.loads((old/'config.json').read_text());b=json.loads((raw/'config.json').read_text())
a['log']['run_name']='';b['log']['run_name']=''
del a['loss']['phase_metric'];a['loss']['evolution_scale']=100.
assert a==b
with np.load(old/'training_times.npz') as p,np.load(raw/'training_times.npz') as q:
    for k in ('times','horizons'):np.testing.assert_array_equal(p[k],q[k])
old_sources={str(p.relative_to(old/'source')):p.read_bytes() for p in (old/'source').rglob('*') if p.is_file()}
new_sources={str(p.relative_to(raw/'source')):p.read_bytes() for p in (raw/'source').rglob('*') if p.is_file()}
changed=sorted(k for k in old_sources if old_sources[k]!=new_sources[k])
added=sorted(set(new_sources)-set(old_sources))
expected_changed=['examples/confined_ch2d/README.md','examples/confined_ch2d/losses.py','examples/confined_ch2d/problem.py']
assert changed==expected_changed,changed
assert added==['examples/confined_ch2d/configs/train_coarsening_mixed_gradnorm_raw.py'],added
report=dict(status='passed',old_run=str(old),raw_run=str(raw),initial_parameters='Direct bytes identical',
    time_samples='Exact arrays identical',config_difference='Only phase metric removed / explicit evolution_scale100 and run label',
    changed_sources=changed,added_sources=added,
    preserved_sources='All other files including network, train, balancing and operators identical',
    scope='Later raw control; first-three identical57-source claim remains scoped to that cohort; no hashes')
(OUT/'chbalance-raw-provenance.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
