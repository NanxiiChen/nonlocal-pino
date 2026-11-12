"""Bound every two-group halo on fixed and fresh unlabeled geometries.

A group's complete graph closure contains every possible random core draw
from that group. Maximizing unions of two closures bounds all two-group draws
on a particular geometry, not just the finite sampling audit. This remains a
finite-geometry bound, not a theorem over the continuous geometry prior.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import time
import numpy as np

from examples.confined_ch2d.operator_core_halo import VolumeGroups

POPCOUNT=np.array([int(i).bit_count() for i in range(256)],dtype=np.uint8)


def union_bound(left,right):
    left,right=np.packbits(left,axis=1),np.packbits(right,axis=1)
    return max(int(POPCOUNT[np.bitwise_or(row[None,:],right)].sum(axis=1).max()) for row in left)


def bounds(data,c):
    groups=VolumeGroups(data,c,cell_width=.12,wall_band=c.nl.delta)
    n=len(data['points']);nb,co=data['neighbors'],data['coefficients']
    closure=[]
    for bins,_ in groups.partitions:
        first=[];second=[]
        for ids in bins:
            inner=np.union1d(ids,nb[ids][co[ids]!=0])
            query=np.union1d(inner,nb[inner][co[inner]!=0])
            a,b=np.zeros(n,dtype=bool),np.zeros(n,dtype=bool)
            a[inner]=True;b[query]=True
            first.append(a);second.append(b)
        closure.append((np.stack(first),np.stack(second)))
    result={}
    for name,left,right in (('bulk',0,0),('wall',1,2)):
        result[name]=dict(inner=union_bound(closure[left][0],closure[right][0]),
                         query=union_bound(closure[left][1],closure[right][1]))
    return dict(nodes=int(np.count_nonzero(data['volumes'])),bounds=result)


def fresh_job(config,draw):
    from ml_collections import ConfigDict
    from examples.confined_ch2d.operator_online import build_input
    c=ConfigDict(config)
    data,record=build_input(config,draw)
    return dict(draw_id=draw,geometry=record['holes'],**bounds(data,c))


def main(output,fresh_count):
    from examples.confined_ch2d.configs.train_operator_potential_spatial import get_config
    from examples.confined_ch2d.operator_online import _cpu_worker
    c=get_config();c.train.seed=42
    root=Path(c.operator.dataset_path)
    manifest=json.loads((root/'manifest.json').read_text())
    records=[];start=time.perf_counter()
    for record in manifest['conditions']:
        with np.load(root/record['file'],allow_pickle=False) as arrays:
            data={key:arrays[key] for key in ('points','volumes','condition','neighbors','coefficients')}
        records.append(dict(index=record['index'],**bounds(data,c)))
        if len(records)%128==0:print('finite geometries',len(records),flush=True)
    fresh=[]
    if fresh_count:
        with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn'),initializer=_cpu_worker) as pool:
            for result in pool.map(fresh_job,[c.to_dict()]*fresh_count,range(fresh_count)):
                fresh.append(result)
                if len(fresh)%16==0:print('fresh geometries',len(fresh),flush=True)
    all_rows=records+fresh
    maxima={mode:{key:max(r['bounds'][mode][key] for r in all_rows) for key in ('inner','query')}
            for mode in ('bulk','wall')}
    result=dict(status='passed',scope='All two-cell unions on the listed finite geometry cohort',
        dataset=str(root),seed=42,cell_width=.12,wall_band=c.nl.delta,
        finite_geometries=len(records),fresh_geometries=len(fresh),maxima=maxima,
        capacities=dict(inner=c.sample.spatial.inner_capacity,query=c.sample.spatial.query_capacity),
        records=records,fresh=fresh,seconds=time.perf_counter()-start,pde_solves=0)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','fresh')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fresh',type=int,required=True)
    args=parser.parse_args()
    if args.fresh<0:parser.error('Nonnegative --fresh required')
    main(args.output,args.fresh)
