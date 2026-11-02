"""Frozen geometry-only validation cohorts, independent of the rolling prior stream."""
from pathlib import Path
import json
import numpy as np
from examples.channel_nsch2d.problem import identity
from examples.channel_nsch2d.geometry import random_walls,sample_parameters


DRAW_SEED=20261101
DRAW_NAMESPACE=271828
SCHEMA='channel_unseen_geometry_v1'
RANDOM_SCHEMA='channel_unseen_random_spline_v1'


def prepare(c,path,count,query_grid):
    path=Path(path)
    if not path.is_absolute() or isinstance(count,bool) or not isinstance(count,int) or count<2:
        raise ValueError('Cohort requires an absolute manifest path and count>=2')
    if not isinstance(query_grid,(tuple,list)) or len(query_grid)!=2 or any(isinstance(n,bool) or not isinstance(n,int) or n<1 for n in query_grid):
        raise ValueError('Cohort requires two explicit positive integer query counts')
    path.parent.mkdir(parents=True,exist_ok=False)
    cases=[]
    for i in range(count):
        rng=np.random.default_rng(np.random.SeedSequence([DRAW_SEED,DRAW_NAMESPACE,i]))
        a=sample_parameters(c,rng)
        cases.append(dict(index=i,draw_id=i,amplitudes=a.tolist(),root=str(path.parent/f'case_{i:03d}')))
    value=dict(schema=RANDOM_SCHEMA if random_walls(c) else SCHEMA,physical_family=json.loads(json.dumps(identity(c))),count=count,draw_seed=DRAW_SEED,
        draw_namespace=DRAW_NAMESPACE,cases=cases,query_grid=list(query_grid),
        selection=('Frozen independent conditional spline-coefficient prior draws; no model-score selection' if random_walls(c) else 'Frozen independent uniform-prior draws; no model-score selection'),
        acceptance='Mean of per-geometry phi, terminal_phi, velocity and pressure relative L2 each below0.1; report worst separately')
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    load(path,c)
    return value


def load(path,c):
    path=Path(path)
    if not path.is_absolute():raise ValueError('Cohort manifest path must be absolute')
    value=json.loads(path.read_text())
    required={'schema','physical_family','count','draw_seed','draw_namespace','cases','query_grid','selection','acceptance'}
    if set(value)!=required or value['schema']!=(RANDOM_SCHEMA if random_walls(c) else SCHEMA):
        raise ValueError('Unexpected frozen-geometry cohort schema')
    if value['physical_family']!=json.loads(json.dumps(identity(c))):
        raise ValueError('Frozen cohort physical family differs from training')
    count=value['count']
    if isinstance(count,bool) or not isinstance(count,int) or count<2 or len(value['cases'])!=count:
        raise ValueError('Frozen cohort count must match all predeclared cases')
    if value['draw_seed']!=DRAW_SEED or value['draw_namespace']!=DRAW_NAMESPACE:
        raise ValueError('Frozen cohort draw protocol changed')
    grid=value['query_grid']
    if not isinstance(grid,list) or len(grid)!=2 or any(isinstance(n,bool) or not isinstance(n,int) or n<1 for n in grid):
        raise ValueError('Frozen cohort requires two explicit positive integer query counts')
    amplitudes=[]
    for i,case in enumerate(value['cases']):
        if set(case)!={'index','draw_id','amplitudes','root'} or case['index']!=i or case['draw_id']!=i:
            raise ValueError('Frozen cohort needs consecutive immutable draw identities')
        if Path(case['root'])!=path.parent/f'case_{i:03d}':raise ValueError('Frozen cohort case path changed')
        a=np.asarray(case['amplitudes'],dtype=np.float64)
        rng=np.random.default_rng(np.random.SeedSequence([DRAW_SEED,DRAW_NAMESPACE,i]))
        expected=sample_parameters(c,rng)
        np.testing.assert_array_equal(a,expected)
        amplitudes.append(a)
    if len(np.unique(np.asarray(amplitudes),axis=0))!=count:raise ValueError('Duplicate reserved geometries')
    return value


def reserved_amplitudes(c):
    value=load(c.operator.validation_manifest,c)
    return np.asarray([case['amplitudes'] for case in value['cases']],dtype=np.float64)


if __name__=='__main__':
    import argparse
    import importlib
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--count',required=True,type=int);parser.add_argument('--query-grid',required=True,nargs=2,type=int)
    a=parser.parse_args();prepare(importlib.import_module(a.config).get_config(),a.output,a.count,a.query_grid)
