"""Prepare known geometry operators/payloads only; no PDE trajectory or target."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib
import json
import multiprocessing
from pathlib import Path
import secrets
import time

import numpy as np
from ml_collections import ConfigDict

from .geometry import sample_geometry
from .inputs import conditioning,graph_payload,initial_payload,make_core_sampler,pad_graph
from .problem import validate


def one_geometry(arguments):
    config,seed=arguments
    c=ConfigDict(config)
    from .mesh import build_mesh
    from .operators import build_operator
    rng=np.random.default_rng(seed)
    start=time.perf_counter()
    record=sample_geometry(c,rng)
    mesh=build_mesh(c,record,c.mesh.spacing)
    matrix=build_operator(c,record,mesh['points'],mesh['volumes'])
    fixed={**conditioning(c,record),**initial_payload(c,mesh,rng)}
    sample_cores=make_core_sampler(c,mesh)
    graphs=[graph_payload(c,mesh,matrix,rng,sample_cores=sample_cores) for _ in range(c.sample.K_pool)]
    metadata=dict(nodes=len(mesh['points']),triangles=len(mesh['triangles']),
                  seconds=time.perf_counter()-start,seed=int(seed))
    return record,fixed,graphs,metadata


def prepare(c,output,workers,runtime_seed):
    validate(c)
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    seed=secrets.randbits(32) if runtime_seed is None else runtime_seed
    children=np.random.SeedSequence(seed).generate_state(c.operator.geometry_count)
    args=[(c.to_dict(),int(child)) for child in children]
    start=time.perf_counter();results=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for result in pool.map(one_geometry,args):
            results.append(result)
            print(json.dumps(dict(completed=len(results),total=len(args),**result[3])),flush=True)
    halos=max(len(g['graph_points']) for _,_,graphs,_ in results for g in graphs)
    neighbors=max(g['neighbors'].shape[1] for _,_,graphs,_ in results for g in graphs)
    halos=min(c.nl.max_halo,64*((halos+63)//64))
    neighbor_size=min(c.nl.max_neighbors,8*((neighbors+7)//8))
    fixed={key:np.stack([value[1][key] for value in results]).astype(np.float32)
           for key in results[0][1]}
    padded=[[pad_graph(g,halos,neighbor_size) for g in graphs] for _,_,graphs,_ in results]
    graphs={key:np.stack([[g[key] for g in row] for row in padded])
            .astype(np.int32 if key in ('neighbors','core_indices') else np.float32)
            for key in padded[0][0]}
    np.savez_compressed(output/'inputs.npz',**fixed,**graphs)
    manifest=dict(config=c.to_dict(),runtime_seed=seed,geometries=[r[0] for r in results],
        records=[r[3] for r in results],halo_size=halos,neighbor_size=neighbor_size,
        seconds=time.perf_counter()-start,training_fields_generated=False,
        sampling=('ceramic-wall and bulk stratified core draws; mass-proportional IC draws'
                  if 'core_wall' in c.sample else 'mass-proportional core and IC sampling with replacement'),
        fixed_keys=list(fixed),graph_keys=list(graphs))
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(complete=str(output),seconds=manifest['seconds'],
                          halo_size=halos,neighbor_size=neighbor_size)),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='examples.particle_corrosion2d.configs.train_default')
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--seed',type=int,help='Optional input replay seed; not a strategy/configuration parameter')
    args=parser.parse_args()
    if args.workers<1: raise ValueError('Workers must be positive')
    prepare(importlib.import_module(args.config).get_config(),args.output,args.workers,args.seed)


if __name__=='__main__':main()
