"""Common full-graph residual spot check of final ceramic sampling controls."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np


def statistics(values, weights):
    values=np.asarray(values,dtype=np.float64)
    weights=np.asarray(weights,dtype=np.float64)
    if (weights.ndim!=1 or values.shape[-1]!=len(weights)
            or not np.isfinite(values).all() or not np.isfinite(weights).all()
            or np.any(weights<0) or weights.sum()<=0):
        raise ValueError('Finite fields aligned with nonnegative positive-total measure required')
    normalized=weights/weights.sum()
    return dict(mean_absolute=float(np.mean(np.sum(np.abs(values)*normalized,axis=-1))),
        rms=float(np.sqrt(np.mean(np.sum(values**2*normalized,axis=-1)))),
        maximum_absolute=float(np.max(np.abs(values[...,weights>0]))))


def run(paths, reference, output, seed):
    affinity=sorted(os.sched_getaffinity(0))[-2:]
    os.sched_setaffinity(0,affinity)
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    from ml_collections import ConfigDict
    from examples.particle_corrosion2d.dimensional import require_matching_units
    from examples.particle_corrosion2d.geometry import boundary
    from examples.particle_corrosion2d.inputs import mesh_mouth_quadrature, particle_wall_band
    from examples.particle_corrosion2d.losses import residuals
    from examples.particle_corrosion2d.network import build_network, encode, time_jets
    from examples.particle_corrosion2d.operators import build_operator, operator_diagnostics
    from examples.particle_corrosion2d.problem import initial_fields
    if any(d.platform!='cpu' for d in jax.devices()):
        raise ValueError('Full-residual diagnostic requires CPU only')
    started=time.perf_counter();start_utc=datetime.now(timezone.utc).isoformat()
    configs=[json.loads((p/'config.json').read_text()) for p in paths]
    c=ConfigDict(configs[0]);manifest=json.loads((reference/'manifest.json').read_text())
    require_matching_units(manifest['config'],c)
    for section in ('geometry','particles','physical','mesh'):
        if manifest['config'][section]!=configs[0][section]:
            raise ValueError('Native geometry/input mismatch: '+section)
    if c.train.compute_dtype!='float32' or c.operator.mouth_sampling!='mesh_dirichlet':
        raise ValueError('This diagnostic qualifies the current FP32 mesh-mouth recipe')
    for path,config in zip(paths,configs,strict=True):
        for section in ('geometry','particles','physical','mesh','nl','arch','loss'):
            if configs[0][section]!=config[section]:
                raise ValueError('Physical/network/operator/loss mismatch: '+section)
        if any(k in config for k in ('boundary','phase','concentration_scale','transport_residual')):
            raise ValueError('Only original output/residual controls are qualified')
        if config['train']['compute_dtype']!=c.train.compute_dtype or config['train']['matmul_precision']!=c.train.matmul_precision:
            raise ValueError('Numerical precision differs across controls')
        if json.loads((path/'summary.json').read_text())['steps']!=20000:
            raise ValueError('A completed final20k model is required')
        for source in ('examples/particle_corrosion2d/network.py','examples/particle_corrosion2d/losses.py',
                'src/nonlocal_pinn/nn/point_conditional.py','src/nonlocal_pinn/nn/conditional.py'):
            if (path/'source'/source).read_bytes()!=Path(source).read_bytes():
                raise ValueError('Current inference/residual source differs: '+source)
    entry=manifest['cases'][0]
    keys=('points','volumes','triangles','domain_tokens','domain_weights','boundary_tokens','boundary_weights')
    with np.load(reference/entry['file']) as archive:
        known={key:archive[key] for key in keys}
    points,volumes,triangles=(known[k] for k in ('points','volumes','triangles'))
    # Recover actual mesh boundary edges, then identify the declared reservoir.
    candidates=np.sort(np.concatenate([triangles[:,ij] for ij in ([0,1],[1,2],[2,0])]),axis=1)
    edges,counts=np.unique(candidates,axis=0,return_counts=True)
    edges=edges[counts==1]
    g=c.geometry;xy=points[edges]
    mouth_edges=(np.all(np.abs(xy[:,:,1])<1e-12,axis=1)
        & np.all(np.abs(xy[:,:,0]-g.pit_x)<=g.mouth_half_width+1e-12,axis=1))
    mouth=np.zeros(len(points),dtype=bool);mouth[np.unique(edges[mouth_edges])]=True
    mp,mw=mesh_mouth_quadrature(c,dict(points=points,mouth=mouth,
        boundary_edges=edges,boundary_groups=mouth_edges.astype(int)))
    free=np.flatnonzero(~mouth)
    matrix=build_operator(c,entry['geometry'],points,volumes)
    graph_stats=operator_diagnostics(matrix,volumes)
    prescribed=initial_fields(c,points)
    prescribed[mouth]=0.
    initial_phi=prescribed[:,0]
    h=initial_phi**2*(3-2*initial_phi)
    physical=c.physical;dc=physical.c_se-physical.c_le
    liquid_initial=(prescribed[:,1]-dc*h)/physical.c_le
    q=physical.c_le*(liquid_initial-1)
    initial_ga=physical.kappa*(physical.n_ac3*(matrix@initial_phi)
        +physical.n_ac2*2*initial_phi*(1-initial_phi)*(1-2*initial_phi)
        -physical.n_ac1*q*dc*6*initial_phi*(1-initial_phi))
    prescribed_growth=float(volumes[free]@initial_ga[free])
    degrees=[np.count_nonzero(matrix.indices[matrix.indptr[i]:matrix.indptr[i+1]]!=i) for i in free]
    neighbors=np.zeros((len(free),max(degrees)),dtype=np.int32)
    coefficients=np.zeros(neighbors.shape,dtype=np.float32)
    for row,i in enumerate(free):
        a,b=matrix.indptr[i:i+2];selected=matrix.indices[a:b]!=i
        values=matrix.data[a:b][selected]
        if np.any(values>=0): raise ValueError('Visible diffusion edges must be negative')
        neighbors[row,:len(values)]=matrix.indices[a:b][selected]
        coefficients[row,:len(values)]=-values
    payload={k:jnp.asarray(v) for k,v in dict(core_indices=free.astype(np.int32),
        neighbors=neighbors,coefficients=coefficients).items()}
    theta=np.arange(c.mesh.boundary_points)*(2*np.pi/c.mesh.boundary_points)
    polygons=np.stack([boundary(p,theta)[0] for p in entry['geometry']['particles']])
    near=particle_wall_band(points,polygons,c.nl.delta)[free]
    masks=dict(all=np.ones(len(free),dtype=bool),near_particle_wall=near,bulk=~near)
    if not near.any() or near.all(): raise ValueError('Both diagnostic regions must be nonempty')
    rng=np.random.default_rng(seed)
    times=np.r_[0.,(np.arange(16)+rng.random(16))*c.sample.t_max/16].astype(np.float32)
    jax.config.update('jax_enable_x64',False)
    jax.config.update('jax_default_matmul_precision',c.train.matmul_precision)
    context={k:jnp.asarray(known[k],dtype=jnp.float32) for k in keys[3:]}
    queries=np.concatenate((points,mp))
    @eqx.filter_jit
    def infer(model,condition,xy,ts):
        return time_jets(model,model.prepare_encoded(condition,xy),ts,c)
    apply_residual=jax.jit(lambda raw,dt:residuals(raw,dt,payload,c))
    output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(__file__,output/'source.py')
    np.savez_compressed(output/'diagnostic_inputs.npz',points=points,volumes=volumes,
        times=times,free=free,near_particle_wall=near,mouth_points=mp,mouth_weights=mw,
        neighbors=neighbors,coefficients=coefficients)
    records=[]
    for index,path in enumerate(paths):
        model=eqx.tree_deserialise_leaves(path/'model_final.eqx',build_network(c,jax.random.PRNGKey(0)))
        condition=encode(model,context)
        collections=[[],[],[]]
        for offset in range(0,len(queries),512):
            query=queries[offset:offset+512]
            padded=np.pad(query,((0,512-len(query)),(0,0)),mode='edge')
            values=jax.block_until_ready(infer(model,condition,jnp.asarray(padded,dtype=jnp.float32),jnp.asarray(times)))
            for parts,value in zip(collections,values,strict=True):parts.append(np.asarray(value)[:,:len(query)])
        raw,fields,fields_t=[np.concatenate(parts,axis=1) for parts in collections]
        if not all(np.isfinite(v).all() for v in (raw,fields,fields_t)):
            raise FloatingPointError('Nonfinite fields/JVP')
        n=len(points)
        ac,ch=[np.asarray(v).T for v in jax.block_until_ready(apply_residual(jnp.asarray(raw[:,:n]),jnp.asarray(fields_t[:,:n])))]
        full={name:{key:statistics(values[1:,mask],volumes[free[mask]])
            for key,values in (('ac',ac),('ch',ch))} for name,mask in masks.items()}
        ic=fields[0,:n]-initial_fields(c,points)
        bc=fields[1:,n:]
        boundary_metrics=dict(initial_phi=statistics(ic[:,0],volumes),
            initial_c_over_cle=statistics(ic[:,1]/c.physical.c_le,volumes),
            mouth_phi=statistics(bc[:,:,0],mw),mouth_c_over_cle=statistics(bc[:,:,1]/c.physical.c_le,mw))
        # Same independent full-area sample sets for every model, after inference.
        sample_rng=np.random.default_rng(seed+1)
        slots=sample_rng.choice(len(free),size=(64,256),replace=True,p=volumes[free]/volumes[free].sum())
        sampled={key:[statistics(values[1:,ids],np.ones(len(ids)))['mean_absolute'] for ids in slots]
            for key,values in (('ac',ac),('ch',ch))}
        record=dict(run=str(path),checkpoint='model_final.eqx',full=full,boundary=boundary_metrics,
            sampled_area_mean={k:float(np.mean(v)) for k,v in sampled.items()},
            sample_area_std={k:float(np.std(v,ddof=1)) for k,v in sampled.items()},
            phi_t=statistics(fields_t[1:,free,0],volumes[free]),c_t=statistics(fields_t[1:,free,1],volumes[free]),
            diffuse_liquid_area=((1-fields[:,:n,0].astype(float))@volumes).tolist(),
            diffuse_liquid_area_rate=(-fields_t[:,:n,0].astype(float)@volumes).tolist(),
            initial_pde_ac=statistics(ac[0],volumes[free]),
            initial_pde_ch=statistics(ch[0],volumes[free]),
            by_time=[dict(time=float(t),ac=statistics(ac[k],volumes[free]),ch=statistics(ch[k],volumes[free]))
                for k,t in enumerate(times) if k>0])
        np.savez_compressed(output/f'fields_{index:02d}.npz',raw=raw,fields=fields,fields_t=fields_t,ac=ac,ch=ch)
        records.append(record)
        print(json.dumps(dict(run=str(path),full=full,boundary=boundary_metrics)),flush=True)
    report=dict(status='passed',reference_geometry_source=str(reference/entry['file']),
        geometry=entry['geometry'],case_index=0,times=times.tolist(),seed=seed,
        nodes=len(points),pde_nodes=len(free),mouth_nodes=int(mouth.sum()),graph=graph_stats,
        prescribed_initial_diffuse_liquid_area=float(volumes@(1-initial_phi)),
        prescribed_initial_liquid_area_rate=prescribed_growth,
        initial_rate_scope='Direct original nonlocal RHS at the prescribed IC and mouth values; no time integration',
        region_rule='Distance <= one training horizon from actual polygonal ceramic walls',
        norm='Original AC/CH residuals; physical spatial mean, equal positive-time mean; no adaptive loss coefficients',
        reference_fields_loaded=False,training_performed=False,pde_solves=0,fem_solves=0,
        devices=[str(d) for d in jax.devices()],cpu_affinity=affinity,records=records,
        started_utc=start_utc,finished_utc=datetime.now(timezone.utc).isoformat(),seconds=time.perf_counter()-started,
        limitation='One held-out geometry/16times spot check, not a full-cohort convergence certificate; CPU overlaps live GPU training')
    (output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(output),seconds=report['seconds'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,nargs='+',required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,required=True)
    args=parser.parse_args();run(args.runs,args.reference,args.output,args.seed)
