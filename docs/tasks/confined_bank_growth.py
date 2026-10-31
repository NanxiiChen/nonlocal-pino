"""Verify an incremental input-only bank against its immutable native source."""
import argparse
import json
from pathlib import Path
import numpy as np
from ml_collections import ConfigDict
from examples.confined_ch2d.operator_problem import specification,sample_geometries,evaluation_indices


def verify_growth(root,source):
    root,source=Path(root),Path(source)
    c=ConfigDict(json.loads((root/'config.json').read_text()))
    old=ConfigDict(json.loads((source/'config.json').read_text()))
    m=json.loads((root/'manifest.json').read_text());before=json.loads((source/'manifest.json').read_text())
    for cfg,manifest in ((c,m),(old,before)):
        assert manifest['status']=='passed' and manifest['specification']==specification(cfg)
        np.testing.assert_array_equal([r['holes'] for r in manifest['conditions']],sample_geometries(cfg))
        assert manifest['format']=='operator_only_v1'
    assert c.operator.train_conditions>old.operator.train_conditions
    now_spec=specification(c);old_spec=specification(old)
    for k in now_spec:
        if k!='prior':assert now_spec[k]==old_spec[k],k
    for k in now_spec['prior']:
        if k!='train_conditions':assert now_spec['prior'][k]==old_spec['prior'][k],k
    monitors,labeled=evaluation_indices(m);old_monitors,old_labeled=evaluation_indices(before)
    assert monitors==old_monitors
    new_by_draw={r['draw_index']:r for r in m['conditions']}
    assert len(new_by_draw)==len(m['conditions'])
    test_draws=[m['conditions'][i]['draw_index'] for i in m['test_indices']]
    assert test_draws==[before['conditions'][i]['draw_index'] for i in before['test_indices']]
    maps=[];equal_arrays=0;hardlinks=0
    for record in before['conditions']:
        target=new_by_draw[record['draw_index']]
        assert target['has_reference']==record['has_reference']
        np.testing.assert_array_equal(target['holes'],record['holes'])
        if record['index'] in before['train_indices']:assert target['index']==record['index']
        a=source/record['file'];b=root/target['file']
        with np.load(a,allow_pickle=False) as left,np.load(b,allow_pickle=False) as right:
            assert set(left.files)==set(right.files)
            for k in left.files:
                np.testing.assert_array_equal(left[k],right[k]);equal_arrays+=1
        hardlinks+=int(a.stat().st_ino==b.stat().st_ino and a.stat().st_dev==b.stat().st_dev)
        maps.append(dict(draw=record['draw_index'],before=record['index'],after=target['index']))
    old_draws={r['draw_index'] for r in before['conditions']}
    new_records=[r for r in m['conditions'] if r['draw_index'] not in old_draws]
    for r in new_records:
        assert not r['has_reference'] and r['index'] in m['train_indices']
        with np.load(root/r['file'],allow_pickle=False) as f:
            assert not set(f.files)&{'phi','mu','time','mass','energy'}
    assert m['pde_trajectory_solves']==m['fem_solves']==0
    nmax=max(r['nodes'] for r in m['conditions']);kmax=max(r['degree'] for r in m['conditions'])
    total=len(m['conditions']);boundary=c.geometry.boundary_outer+2*c.geometry.boundary_per_hole
    # Eight explicit FP64/int32 payloads, without allocator overhead or neural workspace.
    padded_bytes=total*(8*(nmax*nmax+5*nmax+nmax*kmax+boundary*6)+4*nmax*kmax)
    return dict(status='passed',bank=str(root),source=str(source),training_conditions=c.operator.train_conditions,
        source_training_conditions=old.operator.train_conditions,test_draws=test_draws,
        test_indices=m['test_indices'],source_test_indices=before['test_indices'],
        new_operator_only_inputs=len(new_records),reused_records=len(maps),hardlinked_records=hardlinks,
        exactly_equal_arrays=equal_arrays,label_roles_preserved=True,source_train_prefix_preserved=True,
        pde_trajectory_solves=0,fem_solves=0,max_points=nmax,max_neighbors=kmax,
        estimated_padded_payload_bytes=padded_bytes,new_payload_bytes=sum((root/r['file']).stat().st_size for r in new_records),
        preparation_seconds=m['seconds'],mapping=maps,
        scope='Exact old input/reference arrays and heldout draws; no model accuracy or continuum claim')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('bank','source','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();r=verify_growth(a.bank,a.source)
    a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in r.items() if k!='mapping'},indent=2),flush=True)
