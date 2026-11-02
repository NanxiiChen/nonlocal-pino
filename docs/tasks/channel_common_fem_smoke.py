"""Isolated short FEM export parity fixture; never a reference-accuracy claim."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def check(saved_config,output):
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise RuntimeError('FEM export smoke must wait until every JAX GPU process exits')
    import numpy as np
    from ml_collections import ConfigDict
    from examples.channel_nsch2d.solver import fem
    from examples.channel_nsch2d.problem import walls
    from examples.channel_nsch2d.response_queries import validate_arrays
    c=ConfigDict(json.loads(Path(saved_config).read_text()));original_horizon=c.physical.terminal_time
    c.physical.terminal_time=.1;c.reference.snapshots=3
    output=Path(output);output.mkdir(parents=True,exist_ok=False);c.reference.root=str(output)
    a=(c.geometry.upper,c.geometry.lower);x,e=np.meshgrid((np.arange(7)+.5)*c.geometry.length/7,(np.arange(5)+.5)/5,indexing='ij')
    b,t=walls(x,a,c,xp=np);query=np.stack((x,b+e*(t-b)),axis=-1).reshape(-1,2)
    legacy,old=fem(c,a,(24,12),.05,query,'legacy_export',output/'legacy.npz')
    extended,new=fem(c,a,(24,12),.05,query,'common_export',output/'common.npz',common_query_grid=(7,5))
    for key in ('fields','points','time','amplitudes','mesh_points','mesh_cells'):
        np.testing.assert_allclose(legacy[key],extended[key],atol=2e-10,rtol=2e-10)
    spec=validate_arrays(extended,c)
    for a,b in zip(old['diagnostics'],new['diagnostics'],strict=True):
        for key in a:np.testing.assert_allclose(a[key],b[key],atol=2e-10,rtol=2e-10)
    report=dict(status='common_query_export_legacy_parity_passed',saved_config=str(saved_config),
        original_horizon=original_horizon,fixture_horizon=.1,grid=[24,12],dt=.05,snapshots=3,
        maximum_primary_field_gap=float(np.max(np.abs(legacy['fields']-extended['fields']))),
        primary_shape=list(legacy['fields'].shape),common_shape=list(extended['common_fields'].shape),
        common_query_protocol=spec,refinement_validated=False,
        scope='Two matched short API fixtures only; no validated reference trajectory, training labels or accuracy claim')
    (output/'validation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--saved-config',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();check(a.saved_config,a.output)
