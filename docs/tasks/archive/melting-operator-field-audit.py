"""Independent NumPy diagnostics including small-solid and high-resolution slice errors."""
import argparse
import json
from pathlib import Path
import numpy as np

parser=argparse.ArgumentParser();parser.add_argument('run');args=parser.parse_args()
run=Path(args.run);config=json.loads((run/'config.json').read_text())
manifest=json.loads((Path(config['operator']['dataset_path'])/'manifest.json').read_text())
summary=json.loads((run/'summary.json').read_text())
records=[]
def relative(error,target):
    denominator=np.sqrt(np.sum(target**2))
    return None if denominator==0 else float(np.sqrt(np.sum(error**2))/denominator)
for index in manifest['test_indices']:
    with np.load(run/f'prediction_{index:03d}.npz') as stored:
        p,t=stored['prediction'],stored['target']
        sp,st=stored['slice_prediction'],stored['slice_target']
        assert p.shape==t.shape==(5,24**3) and sp.shape==st.shape==(5,128**2)
        assert np.isfinite(sp).all() and sp.min()>=-1 and sp.max()<=1
        solid=(t+1)/2;delta=(p-t)/2
        item=next(x for x in summary['test']['conditions'] if x['index']==index)
        pooled=relative(delta,solid);terminal=relative(delta[-1],solid[-1])
        np.testing.assert_allclose([pooled,terminal],[item['pooled_solid'],item['terminal_solid']],rtol=1e-12)
        band=abs(t)<.9
        rmse=float(np.sqrt(np.sum((p[band]-t[band])**2)/band.sum()))
        np.testing.assert_allclose(rmse,item['interface_phi_absolute_rmse'],rtol=1e-12)
        frozen=np.broadcast_to(t[0],t.shape)
        records.append(dict(index=index,pooled_solid=pooled,terminal_solid=terminal,
            interface_phi_absolute_rmse=rmse,
            pooled_phi=relative(p-t,t),terminal_phi=relative(p[-1]-t[-1],t[-1]),
            pooled_slice_phi=relative(sp-st,st),terminal_slice_phi=relative(sp[-1]-st[-1],st[-1]),
            terminal_solid_fraction_absolute_error=float(abs(delta[-1].mean())),
            terminal_solid_fraction_relative_error=relative(np.asarray(delta[-1].mean()),np.asarray(solid[-1].mean())),
            frozen_IC_pooled_phi=relative(frozen-t,t),
            hard_IC_maximum_gap=float(abs(p[0]-t[0]).max())))
metrics={key:dict(mean=float(np.mean([r[key] for r in records])),maximum=float(max(r[key] for r in records)))
         for key in records[0] if key!='index' and all(r[key] is not None for r in records)}
result=dict(test_count=len(records),records=records,statistics=metrics,
    sampled_spatial_nodes=24**3,slice_nodes=128**2,full_reference_grid_nodes=128**3,
    scope='Independent physical metrics; diagnostic volumes and one high-resolution z slice, no full-grid PINN L2 claim')
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(test_count=len(records),statistics=metrics),indent=2))
