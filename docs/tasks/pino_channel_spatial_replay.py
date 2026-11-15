"""Replay recorded channel spatial choices, cores and complete halo counts.

CPU-only validation of already completed runs. Geometry/time-stream and final
physical-field audits remain separate. No inference, training or reference solve.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
from ml_collections import ConfigDict

from examples.channel_nsch2d.spatial_sampling import NAMESPACE,ROW_WEIGHTS,_build,make_tiles,validate_spatial


def support_nodes(data,cores):
    """Independent union of targets, active domain stencils and all BC stencils."""
    n=len(data['points']);needed=np.zeros(n,dtype=bool);needed[cores]=True
    for key in ROW_WEIGHTS:
        index=data['neighbors'][cores][data[key][cores]!=0]
        needed[index]=True
    for key in ('boundary_gradient_x','boundary_gradient_y'):
        needed[data['boundary_neighbors'][data[key]!=0]]=True
    return np.flatnonzero(needed)


def replay(c,initial,sampled,refresh,records,replacement_builder):
    """Every recorded selection, with bounded per-slot candidate-set caching."""
    steps=c.train.iterations;batch=c.operator.condition_batch;size=c.operator.train_conditions
    s=c.sample.spatial
    shapes={'indices':(steps,batch),'draw_ids':(steps,batch),'spatial_pool_ids':(steps,batch),
            'core_ids':(steps,batch,s.core_count),'halo_counts':(steps,batch)}
    for key,shape in shapes.items():
        if sampled[key].shape!=shape or sampled[key].dtype.kind not in 'iu':
            raise ValueError(f'Complete integer {key} with shape{shape} required')
    if (np.any(sampled['indices']<0) or np.any(sampled['indices']>=size)
            or np.any(sampled['spatial_pool_ids']<0) or np.any(sampled['spatial_pool_ids']>=s.pools)):
        raise ValueError('Saved slot or pool index out of range')
    every=c.operator.online.refresh_every;replace=c.operator.online.refresh_size
    expected_events=(steps-1)//every
    if len(refresh)!=expected_events or len(records)!=expected_events*replace:
        raise ValueError('Complete geometry replacement records required')
    if any(len(a)!=size for a in initial.values()):raise ValueError('Initial bank capacity differs')
    states=[({k:a[i] for k,a in initial.items()},-1-i) for i in range(size)]
    cache=[None]*size;tiles=make_tiles(c.sample.Nx,c.sample.Ny) if s.mode=='tiles' else ()
    event=0;pools=0;sets=0;dtypes=set();maximum_halo=0;weight_error=0.
    for step in range(steps):
        if step and step%every==0:
            row=refresh[event];slots=(event*replace+np.arange(replace))%size
            draws=np.arange(event*replace,(event+1)*replace)
            if row['completed_updates']!=step or row['batch']!=event:
                raise ValueError('Geometry replacement step/order differs')
            np.testing.assert_array_equal(row['slots'],slots)
            np.testing.assert_array_equal(row['draw_ids'],draws)
            for slot,draw in zip(slots,draws,strict=True):
                record=records[int(draw)]
                if record['draw_id']!=draw or record['pde_solves']!=0 or record['fem_solves']!=0:
                    raise ValueError('Invalid online input identity or evolved label')
                data=replacement_builder(int(draw))
                np.testing.assert_array_equal(data['amplitudes'],np.asarray(record['training_amplitudes']))
                if data['points'].dtype!=np.dtype(c.train.compute_dtype):
                    raise ValueError('Canonical replacement coordinates must have training dtype')
                states[int(slot)]=(data,int(draw));cache[int(slot)]=None
            event+=1
        revision=step//s.refresh_every
        choices=np.random.default_rng(np.random.SeedSequence([c.train.seed,NAMESPACE,step])).integers(
            s.pools,size=batch,dtype=np.int32)
        np.testing.assert_array_equal(sampled['spatial_pool_ids'][step],choices)
        for position,(slot,choice) in enumerate(zip(sampled['indices'][step],choices,strict=True)):
            slot=int(slot);data,identity=states[slot]
            if sampled['draw_ids'][step,position]!=identity:raise ValueError('Spatial selection uses stale geometry')
            previous=cache[slot]
            if previous is None or previous[0]!=revision:
                candidates=_build(data,c,identity=identity,revision=revision,tiles=tiles)
                small=[];dtypes.add(str(data['points'].dtype))
                for payload,core,count in candidates:
                    nodes=support_nodes(data,core)
                    if len(nodes)!=count or count>s.node_capacity:
                        raise ValueError('Halo count does not retain all physical/BC support')
                    # Check index remapping against an independently constructed
                    # union rather than accepting the reported count alone.
                    np.testing.assert_array_equal(payload['points'][:count],data['points'][nodes])
                    np.testing.assert_array_equal(nodes[payload['core_indices']],core)
                    for key in ROW_WEIGHTS:
                        np.testing.assert_array_equal(payload[key],data[key][core])
                        mask=data[key][core]!=0
                        np.testing.assert_array_equal(nodes[payload['neighbors'][mask]],data['neighbors'][core][mask])
                    for key in ('boundary_gradient_x','boundary_gradient_y'):
                        mask=data[key]!=0
                        np.testing.assert_array_equal(nodes[payload['boundary_neighbors'][mask]],data['boundary_neighbors'][mask])
                    error=abs(float(np.sum(payload['core_weights'],dtype=float))-1.)
                    # Legacy partial tiles use inverse-inclusion weights; their
                    # sampled constant integral fluctuates around one. Do not
                    # silently renormalize or reject that unbiased estimator.
                    if s.mode!='tiles' and error>2e-6:
                        raise ValueError('Normalized physical quadrature mass differs')
                    weight_error=max(weight_error,error);maximum_halo=max(maximum_halo,count)
                    small.append((core.copy(),count));sets+=1
                cache[slot]=(revision,small);pools+=1
            core,count=cache[slot][1][int(choice)]
            np.testing.assert_array_equal(sampled['core_ids'][step,position],core)
            if sampled['halo_counts'][step,position]!=count:raise ValueError('Saved halo count differs')
    return dict(status='passed',steps=steps,selected_sets=steps*batch,
        compared_core_ids=steps*batch*s.core_count,reconstructed_candidate_pools=pools,
        checked_candidate_sets=sets,installed_online_geometries=event*replace,
        coordinate_dtypes=sorted(dtypes),maximum_halo=maximum_halo,
        maximum_quadrature_mass_error=weight_error)


def audit_run(run):
    from examples.channel_nsch2d.online import build_input
    c=ConfigDict(json.loads((run/'config.json').read_text()));validate_spatial(c)
    if c.sample.spatial.mode=='full':raise ValueError('Spatial replay requires a compact run')
    summary=json.loads((run/'summary.json').read_text())
    if summary['status']!='completed' or summary['iterations']!=c.train.iterations:
        raise ValueError('A completed run with every sampled step is required')
    root=Path(__file__).resolve().parents[2];source=list((run/'source').rglob('*.py'))
    if not source:raise ValueError('Frozen production source required')
    for p in source:
        if p.read_bytes()!=(root/p.relative_to(run/'source')).read_bytes():
            raise ValueError(f'Replay implementation differs from frozen training source: {p}')
    with np.load(run/'known_inputs.npz',allow_pickle=False) as f:initial={k:f[k] for k in f.files}
    with np.load(run/'sampling.npz',allow_pickle=False) as f:
        sampled={k:f[k] for k in ('indices','draw_ids','spatial_pool_ids','core_ids','halo_counts')}
    refresh=[json.loads(s) for s in (run/'online_refresh.jsonl').read_text().splitlines()]
    records=[json.loads(s) for s in (run/'online_inputs.jsonl').read_text().splitlines()]
    result=replay(c,initial,sampled,refresh,records,lambda draw:build_input(c.to_dict(),draw)[0])
    return dict(run=str(run),source_files=len(source),spatial=c.sample.spatial.to_dict(),**result)


def audit(runs,output):
    if not runs or len(set(map(str,runs)))!=len(runs):raise ValueError('Distinct completed runs required')
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-2:])
    output.mkdir(parents=True,exist_ok=False);shutil.copyfile(__file__,output/'source.py')
    started=time.perf_counter();results=[]
    for run in runs:
        result=audit_run(run);results.append(result)
        with (output/'runs.jsonl').open('a') as stream:stream.write(json.dumps(result,allow_nan=False)+'\n')
        print(json.dumps(result),flush=True)
    report=dict(status='passed',results=results,seconds=time.perf_counter()-started,
        training=False,inference=False,pde_solves=0,
        scope='All saved spatial choices/cores/halo counts and complete domain/BC support; geometry/time and physical-field audits are separate')
    (output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();audit(args.runs,args.output)
