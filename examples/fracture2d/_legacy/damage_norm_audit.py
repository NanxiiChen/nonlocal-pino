"""Native CPU replay and common physical-error metrics for L1/L16 controls."""
import argparse
import json
import os
from pathlib import Path
import time
import equinox as eqx
import jax
from ml_collections import ConfigDict
import numpy as np
from scipy.sparse import diags,load_npz
from examples.fracture2d._legacy.coupling_diagnostics import atomic_json
from examples.fracture2d._legacy.damage_norm import eligible_l16
from examples.fracture2d._legacy.full_damage_optimization import adam_from_config
from examples.fracture2d._legacy.local_damage_trial import load_geometry,sha
from examples.fracture2d._legacy.low_order_damage import optimizer_count
from examples.fracture2d._legacy.plain_damage_head import fresh_network
from examples.fracture2d._legacy.staggered_damage_audit import native_image,read_fields,residual


def main(root,*,watch):
    if jax.devices()[0].platform!='cpu':raise ValueError('norm audit requires CPU')
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4])
    while not (root/'preflight.json').exists():
        if (root/'exit.json').exists():raise RuntimeError('norm screen exited before preflight')
        if not watch:raise RuntimeError('norm preflight unavailable')
        time.sleep(5)
    identity=json.loads((root/'identity.json').read_text())
    a,c=ConfigDict(identity['config'],allow_dotted_keys=True),ConfigDict(identity['source_config'],allow_dotted_keys=True)
    for source,pins in [(Path(a.source_root),a.pins),(Path(a.norm_trial.screen_source),a.norm_trial.screen_pins)]:
        for name,digest in pins.items():assert sha(source/name)==digest,name
    for name,digest in identity['code'].items():assert sha(Path('/root/nonlocal-KKS')/name)==digest,name
    m,op,geometry=load_geometry(c);assert geometry==identity['geometry']
    D,B=op.to_scipy(),load_npz(c.l1_loading.basis_path);source=Path(a.source_root)
    scales=read_fields(source/'parameterization.npz');parent=source/f'state_{a.parent_index:04d}'
    pf=read_fields(parent/'fields.npz');initial=read_fields(root/'initial.npz');H=initial['H'];oracle=initial['oracle']
    assert sha(root/'initial.npz')==json.loads((root/'preflight.json').read_text())['initial_sha256']
    np.testing.assert_array_equal(initial['H_parent'],pf['H']);np.testing.assert_array_equal(initial['d_parent'],pf['d'])
    np.testing.assert_array_equal(H,np.maximum(pf['H'],m.driver(initial['q_mechanics'])))
    g0,g1=c.physical.Gc/c.physical.l0,c.physical.Gc*c.physical.l0
    xy=m.coordinates[:m.n_material]
    def image(path):return native_image(path,H,c,D,B,xy,scales['column_scale'],float(scales['trunk_gain']),g0)
    np.testing.assert_allclose(image(parent/'network.eqx'),initial['damage_image'],rtol=1e-9,atol=1e-11)
    A=diags(m.volumes)@(diags(g0+2*H)+g1*D);b=2*m.volumes*H
    oracle_norm=float(np.linalg.norm(A@oracle-b))
    assert oracle_norm<=max(c.validation.cg_atol,c.validation.cg_rtol*np.linalg.norm(b))*(1+1e-3)
    old_report=json.loads((Path(a.norm_trial.screen_source)/'screen/adam/report.json').read_text())
    force=m.matrix(pf['d'])@initial['q_mechanics']
    assert np.linalg.norm(force[m.free])<=old_report['mechanics']['convergence_limit']*(1+1e-3)
    np.testing.assert_allclose(force@m.load_direction,old_report['reaction_at_parent_damage'],rtol=1e-9,atol=1e-6)
    features=json.loads((source/'identity.json').read_text())['features']
    template=fresh_network(c,features['input_dimension'],B,head='sigmoid')
    optimizer,_=adam_from_config(c.l1_loading);memory=optimizer.init(eqx.filter(template,eqx.is_inexact_array))
    review=root/'independent_review';review.mkdir(exist_ok=True);done=set();rows=[]
    while True:
        for case in a.norm_trial.cases:
            case=ConfigDict(case);folder=root/'screen'/case.name
            paths=sorted(folder.glob('update_[0-9][0-9][0-9][0-9][0-9]'))+[folder/'final']
            for path in paths:
                rp=path/'report.json'
                if not rp.exists() or str(rp) in done:continue
                r,f=json.loads(rp.read_text()),read_fields(path/'fields.npz')
                for name,values in f.items():assert values.dtype==np.float64 and np.isfinite(values).all(),name
                np.testing.assert_array_equal(f['H'],H)
                d=image(path/'network.eqx');np.testing.assert_allclose(d,f['damage_image'],rtol=1e-9,atol=1e-11)
                assert sha(path/'fields.npz')==r['fields_sha256'] and sha(path/'network.eqx')==r['network_sha256']
                assert not r['physical_state_promoted'] and r['load']==a.load and r['norm']==case.norm and r['optimizer']==case.optimizer
                raw=(g0+2*H)*d+g1*(D@d)-2*H;rho=raw/(2*np.sqrt(np.mean(H**2)))
                scale=max(float(abs(rho).max()),a.norm_trial.floor)
                objective=float(abs(rho).mean()) if case.norm=='l1' else float(scale**2*(np.mean((rho/scale)**16)+(a.norm_trial.floor/scale)**16)**(1/8))
                delta=d-oracle;v=m.volumes
                expected=dict(objective=objective,uniform_L1=float(abs(rho).mean()),absolute_residual_max=float(abs(raw).max()),
                    AT2_relative=residual(d,H,D,g0,g1),same_H_damage_L2=float(np.sqrt(v@delta**2/(v@oracle**2))),
                    damage_RMSE=float(np.sqrt(v@delta**2/v.sum())),damage_min=float(d.min()),damage_max=float(d.max()))
                for key,value in expected.items():np.testing.assert_allclose(value,r[key],rtol=1e-8,atol=1e-11)
                if case.optimizer=='adam':
                    assert sha(path/'optimizer.eqx')==r['optimizer_sha256']
                    state=eqx.tree_deserialise_leaves(path/'optimizer.eqx',memory)
                    assert optimizer_count(state,kind='adam')==r['optimizer_count']==r['iterations']
                    assert r['learning_rate']==c.l1_loading.learning_rate
                else:
                    assert r['optimizer_count'] is None and 0<=r['iterations']<=a.norm_trial.lbfgs.maxiter
                    assert r['evaluations']>=r['iterations'] and r['learning_rate'] is None
                rows.append(dict(case=case.name,iterations=r['iterations'],seconds=r['seconds'],pass_replay=True,**expected,
                    high_damage_nodal_area=float(v[d>=.9].sum()),fields_sha256=r['fields_sha256'],report_sha256=sha(rp)))
                done.add(str(rp));print(json.dumps(dict(event='norm_CPU_replay',**rows[-1])),flush=True)
        finished=(root/'result.json').exists() and json.loads((root/'result.json').read_text())['complete']
        selection_checked=False
        if finished:
            result=json.loads((root/'result.json').read_text());selection=json.loads((root/'selection.json').read_text())
            finals=[]
            for row in result['cases']:
                matches=[r for r in rows if r['case']==row['case'] and r['iterations']==row['iterations']]
                if matches:finals.append(matches[-1])
            if len(finals)==3:
                assert eligible_l16(finals)==selection['selected']==result['selected'];selection_checked=True
        complete=finished and selection_checked
        atomic_json(review/'audit.json',dict(complete=complete,pass_replay=True,rows=rows,
            selection_checked=selection_checked,oracle_host_residual=oracle_norm,history_fixed=True,
            source_first_mechanics_replayed=True,physical_state_promoted=False,auditor_sha256=sha(__file__)))
        if complete or not watch:return
        if (root/'exit.json').exists() and json.loads((root/'exit.json').read_text())['worker_exit_code']!=0:raise RuntimeError('norm worker failed')
        time.sleep(5)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    jax.config.update('jax_enable_x64',True);main(Path(args.root),watch=args.watch)
