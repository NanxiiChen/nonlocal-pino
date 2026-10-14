"""Independent native replay of frozen-H weighted losses and oracle field errors."""
import argparse
import json
import os
from pathlib import Path
import time

import equinox as eqx
import jax
from ml_collections import ConfigDict
import numpy as np
from scipy.sparse import diags, load_npz

from examples.fracture2d.coupling_diagnostics import atomic_json
from examples.fracture2d.full_damage_optimization import adam_from_config
from examples.fracture2d.local_damage_trial import load_geometry,sha
from examples.fracture2d.low_order_damage import optimizer_count
from examples.fracture2d.plain_damage_head import fresh_network
from examples.fracture2d.staggered_damage_audit import native_image,read_fields,residual


def main(root,*,watch):
    if jax.devices()[0].platform!='cpu':raise ValueError('weighted native replay requires CPU')
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4])
    while not (root/'evaluation_oracle.npz').exists():
        if (root/'exit.json').exists():raise RuntimeError('weighted worker exits before evaluation oracle')
        if not watch:raise RuntimeError('weighted oracle not yet available')
        time.sleep(5)
    identity=json.loads((root/'identity.json').read_text())
    a,c=ConfigDict(identity['config'],allow_dotted_keys=True),ConfigDict(identity['source_config'],allow_dotted_keys=True)
    t=a.weighting;source=Path(t.source_root)
    for parent,pins in [(Path(a.source_root),a.pins),(source,t.pins)]:
        for name,digest in pins.items():assert sha(parent/name)==digest,name
    for name,digest in identity['code'].items():assert sha(Path('/root/nonlocal-KKS')/name)==digest,name
    m,op,geometry=load_geometry(c);assert geometry==identity['geometry']
    D,B=op.to_scipy(),load_npz(c.l1_loading.basis_path)
    f=read_fields(source/'pinn_full/final/fields.npz');H=f['H']
    scales=read_fields(Path(a.source_root)/'parameterization.npz')
    oracle=read_fields(root/'evaluation_oracle.npz');np.testing.assert_array_equal(oracle['H'],H)
    g0,g1=c.physical.Gc/c.physical.l0,c.physical.Gc*c.physical.l0
    A=diags(m.volumes)@(diags(g0+2*H)+g1*D);b=2*m.volumes*H
    norm=float(np.linalg.norm(A@oracle['d']-b))
    assert norm<=max(c.validation.cg_atol,c.validation.cg_rtol*np.linalg.norm(b))*(1+1e-3)
    features=json.loads((Path(a.source_root)/'identity.json').read_text())['features']
    template=fresh_network(c,features['input_dimension'],B,head='sigmoid')
    optimizer,_=adam_from_config(c.l1_loading);memory=optimizer.init(eqx.filter(template,eqx.is_inexact_array))
    region=f['d']>=t.region_threshold
    review=root/'independent_review';review.mkdir(exist_ok=True);done=set();rows=[]
    while True:
        for case in t.cases:
            case=ConfigDict(case);folder=root/case.name
            if not (folder/'weights.npy').exists():continue
            w=np.load(folder/'weights.npy')
            raw=np.ones_like(f['d']) if case.kind=='uniform' else 1/(1-f['d']+case.epsilon)
            if case.cap is not None:raw=np.minimum(raw,case.cap)
            np.testing.assert_array_equal(w,raw/raw.mean())
            for path in sorted(folder.glob('update_[0-9][0-9][0-9][0-9][0-9]')):
                rp=path/'report.json'
                if not rp.exists() or str(rp) in done:continue
                r,z=json.loads(rp.read_text()),read_fields(path/'fields.npz')
                for key,v in z.items():assert v.dtype==np.float64 and np.isfinite(v).all(),key
                np.testing.assert_array_equal(z['H'],H)
                d=native_image(path/'network.eqx',H,c,D,B,m.coordinates[:m.n_material],
                    scales['column_scale'],float(scales['trunk_gain']),g0)
                np.testing.assert_allclose(d,z['damage_image'],rtol=1e-9,atol=1e-11)
                for name in ['fields.npz','network.eqx','optimizer.eqx']:
                    key='fields_sha256' if name=='fields.npz' else name.split('.')[0]+'_sha256'
                    assert sha(path/name)==r[key],name
                assert sha(folder/'weights.npy')==r['weights_sha256']
                state=eqx.tree_deserialise_leaves(path/'optimizer.eqx',memory)
                assert optimizer_count(state,kind='adam')==r['optimizer_count']==t.source_count+r['update']
                assert r['learning_rate']==c.l1_loading.learning_rate and not r['physical_state_promoted']
                rho=((g0+2*H)*d+g1*(D@d)-2*H)/(2*np.sqrt(np.mean(H**2)));delta=d-oracle['d']
                expected=dict(uniform_L1=float(abs(rho).mean()),weighted_L1=float(w@abs(rho)/w.sum()),
                    AT2_relative=residual(d,H,D,g0,g1),same_H_damage_L2=float(np.sqrt(m.volumes@delta**2/(m.volumes@oracle['d']**2))),
                    region_RMSE=float(np.sqrt(m.volumes[region]@delta[region]**2/m.volumes[region].sum())),
                    global_RMSE=float(np.sqrt(m.volumes@delta**2/m.volumes.sum())))
                for key,value in expected.items():np.testing.assert_allclose(value,r[key],rtol=1e-8,atol=1e-11)
                row=dict(case=case.name,update=r['update'],pass_replay=True,**expected,report_sha256=sha(rp),fields_sha256=r['fields_sha256'])
                rows.append(row);done.add(str(rp));print(json.dumps(dict(event='weighted_CPU_replay',**row)),flush=True)
        finished=(root/'result.json').exists() and json.loads((root/'result.json').read_text())['complete']
        complete=finished and len([r for r in rows if r['update']==t.updates])==len(t.cases)
        atomic_json(review/'audit.json',dict(complete=complete,pass_replay=True,rows=rows,
            oracle_host_residual=norm,oracle_sha256=sha(root/'evaluation_oracle.npz'),
            history_and_weights_frozen=True,physical_state_promoted=False,auditor_sha256=sha(__file__)))
        if complete or not watch:return
        if (root/'exit.json').exists() and json.loads((root/'exit.json').read_text())['worker_exit_code']!=0:
            raise RuntimeError('weighted worker failed')
        time.sleep(5)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    jax.config.update('jax_enable_x64',True);main(Path(args.root),watch=args.watch)
