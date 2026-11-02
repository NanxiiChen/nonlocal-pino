"""Shared physical diagnostic coordinates, independent of the actual wall draw."""
import json
import numpy as np
from examples.channel_nsch2d.problem import walls
from examples.channel_nsch2d.geometry import random_walls,envelope_parameters


KIND='maximum_constriction_common_physical_v1'
KEYS={'common_points','common_fields','common_query_grid','common_query_identity'}


def protocol(c,grid):
    if (not isinstance(grid,(tuple,list)) or len(grid)!=2
            or any(isinstance(n,bool) or not isinstance(n,int) or n<1 for n in grid)):
        raise ValueError('Common physical query grid requires two explicit positive integers')
    if random_walls(c):
        return dict(kind='random_spline_envelope_common_physical_v1',query_grid=list(grid),
            geometry={k:v for k,v in c.geometry.to_dict().items() if k!='coefficients'},
            frame='Identical physical(x,y)points inside the coefficient-maximum inward-wall envelope')
    names=('length','height','center','half_length','amplitude_min','amplitude_max')
    geometry={k:c.geometry[k] for k in names}
    if (not np.isfinite(list(geometry.values())).all()
            or any(geometry[k]<=0 for k in ('length','height','half_length'))
            or not geometry['half_length']<geometry['center']<geometry['length']-geometry['half_length']):
        raise ValueError('Common query geometry requires finite valid length, height and bump support')
    if not (0<=geometry['amplitude_min']<geometry['amplitude_max']<geometry['height']/2):
        raise ValueError('Common subdomain needs the declared bounded inward-wall family')
    return dict(kind=KIND,query_grid=list(grid),geometry=geometry,
        frame='Identical physical(x,y)points inside the maximum-constriction subdomain')


def points(c,grid):
    spec=protocol(c,grid);nx,ny=spec['query_grid']
    x,e=np.meshgrid((np.arange(nx)+.5)*c.geometry.length/nx,(np.arange(ny)+.5)/ny,indexing='ij')
    maximum=envelope_parameters(c)
    bottom,top=walls(x,maximum,c,xp=np)
    return np.stack((x,bottom+e*(top-bottom)),axis=-1).reshape(-1,2)


def validate_arrays(payload,c):
    if not KEYS<=set(payload):raise ValueError('Incomplete common physical FEM query payload')
    grid=np.asarray(payload['common_query_grid'])
    if grid.shape!=(2,) or not np.issubdtype(grid.dtype,np.integer):raise ValueError('Invalid common query-grid metadata')
    grid=grid.tolist();spec=protocol(c,grid)
    if json.loads(str(payload['common_query_identity']))!=spec:raise ValueError('Common physical query protocol changed')
    np.testing.assert_array_equal(payload['common_points'],points(c,grid))
    values=np.asarray(payload['common_fields'])
    if values.shape!=(c.reference.snapshots,int(np.prod(grid)),5) or not np.isfinite(values).all():
        raise ValueError('Common FEM fields must be finite and match every snapshot/query/physical channel')
    return spec


def freeze(c,manifest,output):
    from pathlib import Path
    from examples.channel_nsch2d.cohort import load
    cohort=load(manifest,c);output=Path(output)
    if not output.is_absolute():raise ValueError('Frozen response protocol requires an explicit absolute output path')
    value=protocol(c,cohort['query_grid']);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
    return value


if __name__=='__main__':
    import argparse
    import importlib
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();freeze(importlib.import_module(a.config).get_config(),a.manifest,a.output)
