"""Verify selected canonical defaults, run artifacts and direct provenance."""
import importlib
import json
from pathlib import Path
import shutil

from nonlocal_pinn.training.runtime import with_run_seed

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
OUT=ROOT/'reports/ordinary-policy-defaults-20261026'


def flatten(d,prefix=''):
    out={}
    for k,v in d.items():
        path=f'{prefix}.{k}' if prefix else k
        if isinstance(v,dict):out.update(flatten(v,path))
        else:out[path]=v
    return out


def main():
    manifest=json.loads((OUT/'manifest.json').read_text())
    if manifest['status']!='completed' or len(manifest['runs'])!=6:raise ValueError('Finish six runs')
    kks=json.loads((OUT/'kks-audit.json').read_text());melting=json.loads((OUT/'melting-audit.json').read_text());smokes=json.loads((OUT/'canonical-smokes.json').read_text())
    if any(d['status']!='passed' for d in (kks,melting,smokes)):raise ValueError('Independent audit/smokes failed')
    configs={case:importlib.import_module(f'examples.{case}.configs.train_base').get_config() for case in ['corrosion2d','electropolish2d','icemelting3d']}
    index=json.loads((ROOT/'artifacts/index.yaml').read_text());rows=[];source_count=0
    for case,config in configs.items():
        if config.train.gpu_policy!='ordinary' or 'seed' in config.train or config.train.iterations!=5000 or config.train.lr_init!=1e-3 or config.train.optimizer!='soap':raise ValueError('Wrong canonical strategy')
        spec=next(c for c in index['cases'] if c['case']==case)['default_training']
        run=Path(spec['selected_checkpoint']).parent
        current=flatten(json.loads(with_run_seed(config,42).to_json()));saved=flatten(json.loads((run/'config.json').read_text()))
        if set(current)!=set(saved):raise ValueError(f'Current/saved schema differs: {case}: {set(current)^set(saved)}')
        differences={k:{'current':current[k],'saved':saved[k]} for k in current if current[k]!=saved[k]}
        allowed={'log.run_name','experiment.checkpoint_every','eval.graph_reference_path'}
        if set(differences)-allowed:raise ValueError(f'Unexpected selected configuration: {case}: {differences}')
        if (run/'model_final.eqx').read_bytes()!=(run/'ckpt_005000.eqx').read_bytes():raise ValueError('Checkpoint mismatch')
        rows.append({'case':case,'selected_run':str(run),'current_vs_saved_metadata_only_differences':differences})
    for item in manifest['runs']:
        if item['returncode']!=0 or item['status']!='completed':raise ValueError('Worker failure')
        run=Path(item['run'])
        for name in ['config.json','summary.json','model_initial.eqx','model_final.eqx','ckpt_005000.eqx','metrics.jsonl']:
            if not (run/name).is_file():raise FileNotFoundError(run/name)
        if item['case']!='icemelting3d':
            files=json.loads((run/'source/files.json').read_text())
            for relative in files:
                if not (run/'source'/relative).is_file():raise FileNotFoundError(relative)
                source_count+=1
        else:
            source_count+=len(list((run/'source').rglob('*.py')))
    research={}
    for name,policy in [('train_graph_ordinary','ordinary'),('train_local_tensor','deterministic'),('train_local_joint','deterministic')]:
        c=importlib.import_module('examples.corrosion2d.configs.'+name).get_config()
        if c.train.compute_dtype!='float32' or c.train.gpu_policy!=policy:raise ValueError('Historical research preset changed')
        research[name]={'dtype':c.train.compute_dtype,'policy':c.train.gpu_policy}
    source=OUT/'selected-default-source';source.mkdir(exist_ok=True)
    files=['src/nonlocal_pinn/training/runtime.py','examples/corrosion2d/training_entry.py','examples/corrosion2d/configs/train_base.py','examples/electropolish2d/configs/train_base.py','examples/icemelting3d/configs/train_base.py','examples/icemelting3d/train.py','examples/corrosion2d/configs/train_graph_ordinary.py','examples/corrosion2d/configs/train_local_tensor.py','examples/corrosion2d/configs/train_local_joint.py']
    for relative in files:
        target=source/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/relative,target)
    result={'status':'passed','new_completed_full_runs':6,'independently_replayed_KKS_models':10,'independently_replayed_melting_models':1,'run_snapshot_python_files_present':source_count,'selected_configurations':rows,'historical_research_presets':research,'current_source_direct_snapshot':str(source),'source_files':files,'file_hashes':'none; direct paths and parameter/grid/config checks only','git_history_operations':'none'}
    (OUT/'verification.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
