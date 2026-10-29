"""Explicit finer-reference stages/reconstruction; no silent reuse or relaxed gates.

Run fine_dt/fine in fenicsenv. After the original full solver exits (failed
local mesh gate), assemble with .venv. Existing stage fields are read only.
"""
from pathlib import Path
import argparse,json,shutil,sys
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
sys.path.insert(0,str(ROOT))
import numpy as np
from ml_collections import ConfigDict
from examples.confined_ch2d.configs.train_coarsening import get_config
from examples.confined_ch2d.problem import identity,relative,validate_config


def read(path):
    with np.load(path,allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}


def errors(a,b,v):
    return {k:dict(pooled=relative(a[k],b[k],v),maximum_time=max(relative(x,y,v)
        for x,y in zip(a[k],b[k],strict=True))) for k in ('phi','mu')}


def run_stage(c,stage):
    from examples.confined_ch2d.solver import fem
    root=Path(c.reference.root);tag='fem_refined'+('_dt' if stage=='fine_dt' else '')
    metadata=root/(tag+'-input.json')
    if metadata.exists() or (root/(tag+'-fields.npz')).exists():
        raise FileExistsError('Stage already exists; inspect it explicitly: '+tag)
    geo=read(root/'geometry.npz');h=c.reference.mesh_sizes[1]
    dt=c.reference.dt if stage=='fine_dt' else c.reference.refined_dt
    metadata.write_text(json.dumps(dict(config=c.to_dict(),h=h,dt=dt,tag=tag),indent=2)+'\n')
    fem(c,h,dt,geo['points'],tag)


def assemble(c):
    root=Path(c.reference.root);path=Path(c.reference.path);geo=read(root/'geometry.npz')
    original=read(path);oldchecks=json.loads(path.with_suffix('.json').read_text())
    if oldchecks['status']!='failed':raise ValueError('Expected preserved first failed reference')
    if json.loads(str(original['identity']))!=json.loads(json.dumps(identity(c))):
        raise ValueError('First native/local physical identity mismatch')
    coarse=read(root/'fem_fine-fields.npz')
    request=json.loads((root/'fem_fine.geometry.json').read_text())
    source=ConfigDict(request['config'])
    if json.loads(json.dumps(identity(source)))!=json.loads(json.dumps(identity(c))) or request['h']!=c.reference.mesh_sizes[0] or source.reference.refined_dt!=c.reference.refined_dt:
        raise ValueError('Reused .014 BDF2 stage provenance mismatch')
    stages={}
    for name,tag,dt in [('fine_dt','fem_refined_dt',c.reference.dt),('fine','fem_refined',c.reference.refined_dt)]:
        meta=json.loads((root/(tag+'-input.json')).read_text())
        req=json.loads((root/(tag+'.geometry.json')).read_text())
        if meta['config']!=json.loads(c.to_json()) or req['config']!=meta['config'] or meta['h']!=c.reference.mesh_sizes[1] or req['h']!=meta['h'] or meta['dt']!=dt or meta['tag']!=tag:
            raise ValueError('New stage provenance mismatch: '+tag)
        stages[name]=read(root/(tag+'-fields.npz'))
    fine_dt,fine=stages['fine_dt'],stages['fine'];v=geo['volumes']
    for stage in (coarse,fine_dt,fine):
        np.testing.assert_array_equal(stage['points'],geo['points'])
        for k in ('phi','mu'):
            if stage[k].shape!=(c.reference.snapshots,len(v)) or not np.isfinite(stage[k]).all():
                raise ValueError('Invalid staged fields')
    for k in ('points','volumes','mesh_points','mesh_cells'):
        np.testing.assert_array_equal(original[k],geo[k])
    graph={k:original[k+'_nonlocal'] for k in ('phi','mu','mass','energy')}
    checks=dict(local_space=errors(coarse,fine,v),local_time=errors(fine_dt,fine,v),
        native_time=oldchecks['native_time'],
        reference_gap={k:relative(graph[k],fine[k],v) for k in ('phi','mu')})
    for label,stage in [('local',fine),('native',graph)]:
        checks[label+'_mass_drift']=float(np.max(abs(stage['mass']-stage['mass'][0])))
        checks[label+'_energy_max_increase']=max(0.,float(np.max(np.diff(stage['energy']))))
    checks['resolution']={k:{key:float(data[key]) if key=='seconds' else int(data[key])
        for key in ('vertices','dofs','seconds')} for k,data in [('coarse',coarse),('fine',fine)]}
    checks['provenance']=dict(mesh_sizes=list(c.reference.mesh_sizes),dt=c.reference.dt,refined_dt=c.reference.refined_dt,
        reused_local_stage=str(root/'fem_fine-fields.npz'),new_local_stages=['fem_refined_dt-fields.npz','fem_refined-fields.npz'],
        native_source='reference_mesh02_failed.npz',scope='Same physical identity/quadrature/native solve; only local FEM mesh refined')
    passed=all(checks[label][key]['maximum_time']<tol for label in ('local_space','local_time','native_time')
        for key,tol in [('phi',c.reference.phase_tolerance),('mu',c.reference.chemical_tolerance)])
    passed &= all(checks[label+'_mass_drift']<1e-9 and checks[label+'_energy_max_increase']<1e-8 for label in ('local','native'))
    checks['status']='passed' if passed else 'failed'
    failed=root/'reference_mesh02_failed.npz'
    if failed.exists() or failed.with_suffix('.json').exists():raise FileExistsError('Failed reference archive already exists')
    shutil.copyfile(path,failed);shutil.copyfile(path.with_suffix('.json'),failed.with_suffix('.json'))
    for k in ('phi','mu','mass','energy'):original[k+'_local']=fine[k]
    np.savez_compressed(path,**original)
    path.with_suffix('.json').write_text(json.dumps(checks,indent=2,allow_nan=False)+'\n')
    print(json.dumps(checks,indent=2),flush=True)
    if not passed:raise RuntimeError('Finer reference gates failed; do not train')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=('fine_dt','fine','assemble'))
    args=parser.parse_args();c=get_config();validate_config(c)
    if args.stage=='assemble':assemble(c)
    else:run_stage(c,args.stage)
