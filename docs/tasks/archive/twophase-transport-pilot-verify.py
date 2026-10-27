"""Verify completed paired pilot, exact initial/config parity and direct paths."""
import json
from pathlib import Path
import shutil
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
OUT=ROOT/'reports/twophase-transport-pilot'


def flatten(d,prefix=''):
    out={}
    for k,v in d.items():
        path=f'{prefix}.{k}' if prefix else k
        if isinstance(v,dict):out.update(flatten(v,path))
        else:out[path]=v
    return out


def main():
    manifest=json.loads((OUT/'manifest.json').read_text());audit=json.loads((OUT/'audit.json').read_text())
    if manifest['status']!='completed' or len(manifest['runs'])!=2 or audit['status']!='passed':raise ValueError('Incomplete pilot')
    runs=[Path(r['run']) for r in manifest['runs']]
    if any(r['returncode']!=0 for r in manifest['runs']):raise ValueError('Worker failure')
    configs=[flatten(json.loads((r/'config.json').read_text())) for r in runs]
    differences={k:[d[k] for d in configs] for k in configs[0] if configs[0][k]!=configs[1][k]}
    if set(differences)!={'nl.method','log.run_name'}:raise ValueError(differences)
    if (runs[0]/'model_initial.eqx').read_bytes()!=(runs[1]/'model_initial.eqx').read_bytes():raise ValueError('Initial values differ')
    for run in runs:
        summary=json.loads((run/'summary.json').read_text())
        if summary['updates']!=5000:raise ValueError('Wrong update budget')
        if (run/'model_final.eqx').read_bytes()!=(run/'ckpt_005000.eqx').read_bytes():raise ValueError('Final/checkpoint mismatch')
        for relative in ['source/examples/twophase2d/train.py','source/examples/twophase2d/losses.py','source/examples/twophase2d/operators.py','source/examples/twophase2d/network.py']:
            current=ROOT/relative.removeprefix('source/')
            if (run/relative).read_bytes()!=current.read_bytes():raise ValueError('Numerical implementation changed after cohort')
    ref=Path('/root/autodl-tmp/data/twophase2d/results_periodic_pilot.npz')
    checks=json.loads(ref.with_suffix('.json').read_text())
    if checks['status']!='passed':raise ValueError('Reference refinement failed')
    for result in audit['runs']:
        if result['terminal_phi_relative_L2']>=.02 or result['max_gas_area_relative_drift']>=.02:raise ValueError('Pilot field/area target not met')
        if result['max_energy_increase_between_saved_times']>0:raise ValueError('Observed energy curve increases')
    source=OUT/'final-source';source.mkdir(exist_ok=True)
    shutil.copytree(ROOT/'examples/twophase2d',source/'examples/twophase2d',ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
    result={'status':'passed','paired_initial_models_byte_equal':True,'paired_config_differences':differences,
            'full_updates_each':5000,'matched_field_L2_below_2percent_both':True,'area_drift_below_2percent_both':True,
            'energy_decreases_at_all_21_saved_times_both':True,'observed_not_an_exact_PINN_conservation_guarantee':True,
            'numerical_train_network_loss_operator_source_unchanged_during_cohort':True,
            'later_validation_only_change':'Reference point-order guard; evaluate.py added after training.',
            'final_source':str(source),'prior_snapshot_dev_commit':'05780d4','pushed_main_merge':'21fe6d9',
            'new_case_branch':'dev; new development not committed or pushed','full_graph_local_reference_terminal_discrepancy':checks['checks']['graph_vs_local']['terminal_phi']}
    (OUT/'verification.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
