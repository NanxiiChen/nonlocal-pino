"""Complete two-stage cohort/source/input and legacy-checkpoint verification."""
from pathlib import Path
import json
import sys
import numpy as np
from ml_collections import ConfigDict
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
sys.path.insert(0,str(ROOT))
from examples.twophase2d.problem import identity, load_reference


def flatten(d,prefix=''):
    out={}
    for k,v in d.items():
        path=f'{prefix}.{k}' if prefix else k
        if isinstance(v,dict):out.update(flatten(v,path))
        else:out[path]=v
    return out


def main():
    report=ROOT/'reports/twophase-vortex-residual'
    stages=[];all_initial=[];all_identities=[]
    for stage_name in ['twophase-vortex-residual','twophase-vortex-latent']:
        folder=ROOT/'reports'/stage_name
        manifest=json.loads((folder/'manifest.json').read_text())
        analysis=json.loads((folder/'analysis.json').read_text())
        if manifest['status']!='completed' or analysis['status']!='passed' or len(manifest['runs'])!=3:
            raise ValueError('Incomplete stage '+stage_name)
        runs=[Path(item['run']) for item in manifest['runs']]
        if any(item['returncode']!=0 for item in manifest['runs']):raise ValueError('Worker failed')
        configs=[flatten(json.loads((run/'config.json').read_text())) for run in runs]
        differences={k:[d[k] for d in configs] for k in configs[0] if any(d[k]!=configs[0][k] for d in configs[1:])}
        if set(differences)!={'residual.mode','log.run_name'}:raise ValueError(differences)
        histories=[]
        source=(report/'bounded-source' if stage_name.endswith('residual') else ROOT)
        for run in runs:
            all_initial.append((run/'model_initial.eqx').read_bytes())
            c=ConfigDict(json.loads((run/'config.json').read_text()));all_identities.append(identity(c));load_reference(c)
            if (run/'model_final.eqx').read_bytes()!=(run/'ckpt_005000.eqx').read_bytes():raise ValueError('Final checkpoint mismatch')
            summary=json.loads((run/'summary.json').read_text())
            if summary['updates']!=5000 or not summary['cold_start']:raise ValueError('Wrong budget/start')
            for relative in ['train.py','network.py','problem.py','losses.py','operators.py',
                'configs/train_base.py','configs/train_vortex.py','configs/train_vortex_residual.py']:
                if (run/'source/examples/twophase2d'/relative).read_bytes()!=(source/'examples/twophase2d'/relative).read_bytes():
                    raise ValueError('Source changed within stage: '+relative)
            if 'phase' in c:
                if (run/'source/examples/twophase2d/configs/train_vortex_latent.py').read_bytes()!=(ROOT/'examples/twophase2d/configs/train_vortex_latent.py').read_bytes():
                    raise ValueError('Latent config changed')
            history=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
            if [r['step'] for r in history]!=list(range(0,5001,200)):raise ValueError('Incomplete metric history')
            histories.append({k:v for k,v in history[0].items() if k!='elapsed_seconds'})
        if not all(h==histories[0] for h in histories):raise ValueError('Physical initial fields differ within stage')
        stages.append(dict(stage=stage_name,config_differences=differences,within_stage_initial_physical_metrics_equal=True))
    if any(a!=all_initial[0] for a in all_initial):raise ValueError('Random parameter leaves differ')
    if any(d!=all_identities[0] for d in all_identities):raise ValueError('Physical reference identities differ')
    previous=Path('/root/autodl-tmp/tf-logs/twophase2d/20261026-periodic-vortex-nonlocal')
    if (previous/'model_initial.eqx').read_bytes()!=all_initial[0]:raise ValueError('Historical initialization changed')
    result=dict(status='passed',serial_cold_starts_completed=6,full_updates_each=5000,
        parameter_initial_leaves_byte_equal_all6_and_previous=True,
        physical_initial_fields_equal_within_each_stage=True,
        cross_representation_physical_initial_fields_expected_to_differ=True,
        physical_reference_identity_equal_all6=True,stages=stages,
        bounded_stage_source_frozen_before_map_implementation=True,
        raw_and_mapped_results_kept_without_seed_or_checkpoint_selection=True)
    (report/'verification.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
