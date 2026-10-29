"""Direct source/initial-file/config parity without hashes."""
from pathlib import Path
import json
base=Path('/root/autodl-tmp/tf-logs/confined_ch2d')
names=['20261028-chmixed-shared-01','20261028-chmixed-separate-01','20261028-chmixed-block-01']
a=base/names[0]/'source/examples/confined_ch2d';b=base/names[1]/'source/examples/confined_ch2d'
files=lambda root:{p.relative_to(root):p for p in root.rglob('*') if p.is_file()}
fa,fb=files(a),files(b);mismatch=[str(p) for p in fa if p not in fb or fa[p].read_bytes()!=fb[p].read_bytes()]
extra=[str(p) for p in fb if p not in fa]
if mismatch or extra:raise ValueError(f'Frozen shared/separate source mismatch: {mismatch}/{extra}')
initial=(base/names[1]/'model_initial.eqx').read_bytes()
if initial!=(base/names[2]/'model_initial.eqx').read_bytes():raise ValueError('Separate/block initial parameters differ')
if initial!=(base/'20261028-chmixed-representation-01/model_initial.eqx').read_bytes():raise ValueError('Supervised diagnostic initial parameters differ')
configs=[json.loads((base/n/'config.json').read_text()) for n in names]
for key in ('geometry','physical','initial','nl','reference','sample','train'):
 if not all(c[key]==configs[0][key] for c in configs[1:]):raise ValueError('Non-strategy config mismatch: '+key)
report=dict(source_files_compared=len(fa),source_difference=mismatch,extra_files=extra,separate_block_initial_files_identical=True,separate_supervised_initial_files_identical=True,
    physical_sampling_optimizer_configs_identical=True,seed=configs[0]['train']['seed'],scope='Shared/separate layouts change parameter count and output initialization shape. Block changes integrated residual/norm; new runs are one-seed exploratory, not stable causal accuracy rankings. Byte comparisons only, no hashes.')
Path(__file__).with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
