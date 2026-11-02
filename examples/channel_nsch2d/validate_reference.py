"""Evaluate predeclared FEM space/time gates without importing JAX or DOLFIN."""
from pathlib import Path
import argparse
import importlib
import json
import numpy as np
from examples.channel_nsch2d.problem import walls,identity
from examples.channel_nsch2d.geometry import envelope_parameters,nominal_parameters
from examples.channel_nsch2d.response_queries import KEYS,validate_arrays


def refinement_checks(coarse,time,fine,weight):
    def relative(a,b):
        w=weight.reshape((1,len(weight))+(1,)*(a.ndim-2))
        denominator=np.sum(b*b*w);numerator=np.sum((a-b)**2*w)
        if denominator==0:return 0. if numerator==0 else float('inf')
        return float(np.sqrt(numerator/denominator))
    checks={}
    for name,values in [('space',coarse),('time',time)]:
        result={}
        for field,selection in [('phi',slice(0,1)),('velocity',slice(2,4)),('pressure',slice(4,5)),('chemical',slice(1,2))]:
            a,b=values[...,selection],fine[...,selection];begin=0 if field=='phi' else 1
            result[field]=dict(pooled=relative(a[begin:],b[begin:]),
                maximum_time=max(relative(x[None],y[None]) for x,y in zip(a[begin:],b[begin:],strict=True)))
        checks[name]=result
    return checks


def assess(c,coarse_path,time_path,fine_path):
    arrays=[]
    for path in (coarse_path,time_path,fine_path):
        with np.load(path,allow_pickle=False) as f:arrays.append({k:f[k] for k in f.files})
    coarse,time,fine=arrays
    expected=identity(c)
    for path,d in zip((coarse_path,time_path,fine_path),arrays,strict=True):
        # Explicit identity is mandatory for future files. Early same-session
        # source snapshots without it are rejected rather than silently trusted.
        if 'identity_json' not in d:raise ValueError(f'Missing physical identity: {path}')
        if json.loads(str(d['identity_json']))!=json.loads(json.dumps(expected)):
            raise ValueError(f'FEM physical identity mismatch: {path}')
        if not np.isfinite(d['fields']).all():raise ValueError('Nonfinite reference field')
        if d['fields'].shape!=(c.reference.snapshots,len(d['points']),5):
            raise ValueError('Reference fields must match all snapshots, query points and five physical channels')
        np.testing.assert_array_equal(d['amplitudes'],nominal_parameters(c))
        np.testing.assert_array_equal(d['points'],fine['points'])
        np.testing.assert_array_equal(d['time'],fine['time'])
    bottom,top=walls(fine['points'][:,0],fine['amplitudes'],c,xp=np)
    checks=refinement_checks(coarse['fields'],time['fields'],fine['fields'],top-bottom)
    common=None
    if any(KEYS & set(d) for d in arrays):
        specs=[validate_arrays(d,c) for d in arrays]
        if any(spec!=specs[0] for spec in specs):raise ValueError('Common physical query protocols differ across FEM refinements')
        bottom,top=walls(fine['common_points'][:,0],envelope_parameters(c),c,xp=np)
        common=dict(protocol=specs[0],query_points=len(fine['common_points']),
            checks=refinement_checks(coarse['common_fields'],time['common_fields'],fine['common_fields'],top-bottom))
    diagnostics=json.loads(Path(fine_path).with_suffix('.json').read_text())['diagnostics']
    flux=max(abs(d['outlet_flux']-d['inlet_flux']) for d in diagnostics)
    wall=max(d['wall_speed_l2'] for d in diagnostics)
    limits=dict(phi=c.reference.phase_tolerance,velocity=c.reference.velocity_tolerance,
                pressure=c.reference.pressure_tolerance)
    passed=all(checks[k][f]['maximum_time']<tol for k in ('space','time') for f,tol in limits.items())
    if common is not None:
        common_passed=all(common['checks'][k][f]['maximum_time']<tol for k in ('space','time') for f,tol in limits.items())
        common['status']='passed' if common_passed else 'failed';passed &= common_passed
    passed &= flux<1e-8 and wall<1e-10
    report=dict(status='passed' if passed else 'failed',reference=str(Path(fine_path).resolve()),
        paths=[str(p) for p in (coarse_path,time_path,fine_path)],checks=checks,limits=limits,
        maximum_flux_imbalance=flux,maximum_wall_speed_l2=wall,
        fine_diagnostics=diagnostics,identity=expected,
        chemical_scope='Reported separately; not covered by the phi/velocity/pressure acceptance gate')
    if common is not None:report['common_queries']=common
    output=Path(c.reference.root)/'reference-validation.json'
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    if not passed:raise RuntimeError('Reference refinement failed; training is blocked')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='examples.channel_nsch2d.configs.train_base')
    p.add_argument('--coarse',required=True);p.add_argument('--time',required=True);p.add_argument('--fine',required=True)
    a=p.parse_args();assess(importlib.import_module(a.config).get_config(),a.coarse,a.time,a.fine)
