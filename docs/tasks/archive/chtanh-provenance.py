"""Direct active-source/config/initial-parameter parity; no hashes."""
from pathlib import Path
import ast,json
base=Path('/root/autodl-tmp/tf-logs/confined_ch2d');out=Path(__file__).with_suffix('.json')
names=['20261028-chtanh-control-m8-01','20261028-chtanh-map-m8-01','20261028-chtanh-jac-m8-01','20261028-chtanh-jac-m5-01']
sources=[base/n/'source' for n in names]
files=lambda root:{p.relative_to(root):p for p in root.rglob('*') if p.is_file()}
reference=files(sources[1]);mismatches={}
for name,source in zip(names,sources,strict=True):
    actual=files(source);mismatches[name]=[str(p) for p in reference if p not in actual or actual[p].read_bytes()!=reference[p].read_bytes()]
    if set(actual)!=set(reference):raise ValueError('Source file set differs: '+name)
for name in names[1:]:
    if mismatches[name]:raise ValueError('Mapped controls must have identical frozen source: '+name)
allowed={'examples/confined_ch2d/network.py','examples/confined_ch2d/evaluate.py','examples/confined_ch2d/tanh_mapping.py'}
if not set(mismatches[names[0]])<=allowed:raise ValueError('Unexpected control source change')
def definition(path,name):
    tree=ast.parse(path.read_text())
    return ast.dump(next(n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==name),include_attributes=False)
for name in ('PhaseNetwork','GlobalBasisNetwork','predict_fields'):
    if definition(sources[0]/'examples/confined_ch2d/network.py',name)!=definition(sources[1]/'examples/confined_ch2d/network.py',name):raise ValueError('Active baseline function changed: '+name)
first=(base/names[0]/'model_initial.eqx').read_bytes()
for name in names[1:3]:
    if (base/name/'model_initial.eqx').read_bytes()!=first:raise ValueError('Modes8 initial parameter files differ')
configs=[json.loads((base/n/'config.json').read_text()) for n in names]
for key in ('geometry','physical','initial','nl','reference','sample','train'):
    if not all(c[key]==configs[0][key] for c in configs[1:]):raise ValueError('Non-strategy inputs differ: '+key)
report=dict(source_files_per_run=len(reference),mapped_sources_identical=True,modes8_initial_parameter_files_identical=True,physical_time_sampling_optimizer_configs_identical=True,baseline_active_definitions_identical=True,source_differences=mismatches,
    explanation='Stable tanh mathematical JVP was added before any mapped run while the unmapped control was running. Changes are restricted to inactive map/diagnostic paths in that control; its active network definitions and all training/operator/optimizer sources agree exactly.',
    frequency_limit='Modes5 changes input dimension and initialization, not a parameter-matched frequency-scale ablation.',seed=42,scope='Direct byte/AST comparisons; known SOAP process variability prevents statistical ranking from this one-seed study.')
out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
