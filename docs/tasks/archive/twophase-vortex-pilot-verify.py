"""Audit paired inputs, source snapshots, complete budgets and measured outcomes."""
import json
from pathlib import Path
import shutil
import sys
import numpy as np
from ml_collections import ConfigDict

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
sys.path.insert(0,str(ROOT))
from examples.twophase2d.problem import load_reference
from examples.twophase2d.configs.train_base import get_config

OUT=ROOT/'reports/twophase-vortex-pilot'


def flatten(d, prefix=''):
    out={}
    for key,value in d.items():
        path=f'{prefix}.{key}' if prefix else key
        if isinstance(value,dict):out.update(flatten(value,path))
        else:out[path]=value
    return out


def main():
    manifest=json.loads((OUT/'manifest.json').read_text())
    audit=json.loads((OUT/'audit.json').read_text())
    if manifest['status']!='completed' or audit['status']!='passed' or len(manifest['runs'])!=2:
        raise ValueError('Incomplete cohort or CPU replay')
    runs=[Path(item['run']) for item in manifest['runs']]
    if any(item['returncode']!=0 for item in manifest['runs']):raise ValueError('Worker failed')
    configs=[flatten(json.loads((run/'config.json').read_text())) for run in runs]
    differences={k:[d[k] for d in configs] for k in configs[0] if configs[0][k]!=configs[1][k]}
    if set(differences)!={'nl.method','log.run_name'}:raise ValueError(differences)
    if (runs[0]/'model_initial.eqx').read_bytes()!=(runs[1]/'model_initial.eqx').read_bytes():
        raise ValueError('Paired initial models differ')
    for run in runs:
        c=ConfigDict(json.loads((run/'config.json').read_text()))
        load_reference(c)
        summary=json.loads((run/'summary.json').read_text())
        if summary['updates']!=5000 or not summary['cold_start']:raise ValueError('Budget/start mismatch')
        if (run/'ckpt_005000.eqx').read_bytes()!=(run/'model_final.eqx').read_bytes():
            raise ValueError('Final checkpoint differs')
        for filename in ['train.py','losses.py','problem.py','network.py','operators.py',
                         'configs/train_vortex.py','configs/train_base.py']:
            relative='examples/twophase2d/'+filename
            if (run/'source'/relative).read_bytes()!=(ROOT/relative).read_bytes():
                raise ValueError('Cohort source changed: '+relative)
        losses=[json.loads(s) for s in (run/'loss.jsonl').read_text().splitlines()]
        if losses[-1]['step']!=5000 or not all(np.isfinite(r['total']) for r in losses):
            raise ValueError('Loss record incomplete/nonfinite')
    checks=json.loads(Path('/root/autodl-tmp/data/twophase2d/results_periodic_vortex.json').read_text())
    if checks['status']!='passed':raise ValueError('Reference did not pass')
    # Legacy constant-flow configs and references remain accepted without migration.
    load_reference(get_config())
    legacy=Path('/root/autodl-tmp/tf-logs/twophase2d/20261026-periodic-transport-nonlocal/config.json')
    load_reference(ConfigDict(json.loads(legacy.read_text())))
    source=OUT/'final-source'
    shutil.copytree(ROOT/'examples/twophase2d',source/'examples/twophase2d',ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
    result=dict(status='passed',paired_initial_models_byte_equal=True,
        config_differences=differences,full_updates_each=5000,
        legacy_config_and_reference_loads=True,source_unchanged_during_training=True,
        reference_checks=checks['checks'],source=str(source),representative_errors=[])
    for r in audit['runs']:
        result['representative_errors'].append(dict(method=r['method'],
            midpoint_own=r['per_time_phi_relative_L2'][10],terminal_own=r['terminal_phi_relative_L2'],
            midpoint_common_local=r['per_time_phi_relative_L2_to_local'][10],
            terminal_common_local=r['terminal_phi_relative_L2_to_local'],
            max_gas_area_relative_drift=r['max_gas_area_relative_drift'],
            wall_seconds=r['trainer_wall_seconds']))
    (OUT/'verification.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
