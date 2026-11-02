"""Reference-informed frozen-feature projection; never a trained PINO result."""
import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import equinox as eqx
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
import numpy as np

from examples.channel_nsch2d.network import build_network,encode,temporal_features,lifting_gates,fields
from examples.channel_nsch2d.problem import initial_field
from docs.tasks.channel_droplet_metrics import contour_distance


def probe(run,output,rcond):
    run,output=Path(run),Path(output)
    if not run.is_absolute()or not output.is_absolute():raise ValueError('Absolute source/output paths required')
    if output.exists():raise FileExistsError(output)
    if not np.isfinite(rcond)or not 0<rcond<1:raise ValueError('Explicit relative SVD cutoff in(0,1) required')
    c=ConfigDict(json.loads((run/'config.json').read_text()))
    if 'phase_coordinate'in c or c.lifting.kind!='normalized'or c.train.compute_dtype!='float32':
        raise ValueError('Only the explicit additive-phase FP32 checkpoint is supported')
    if json.loads((run/'summary.json').read_text())['status']!='completed':raise ValueError('Completed model required')
    jax.config.update('jax_enable_x64',True);jax.config.update('jax_default_matmul_precision',c.train.matmul_precision)
    key,_=jax.random.split(jax.random.PRNGKey(c.train.seed))
    net=eqx.tree_deserialise_leaves(run/'model_final.eqx',build_network(c,key))
    net=jax.tree.map(lambda a:a.astype(jnp.float64)if eqx.is_inexact_array(a)else a,net)
    with np.load(run/'evaluation_inputs.npz',allow_pickle=False)as f:
        data={k:jnp.asarray(f[k],dtype=f[k].dtype if np.issubdtype(f[k].dtype,np.integer)else jnp.float64)for k in f.files}
    with np.load(run/'prediction.npz',allow_pickle=False)as f:
        points=f['points'];times=f['time'];reference=f['reference'][...,0];saved=f['fields'][...,0]
    c.train.compute_dtype='float64'

    @eqx.filter_jit
    def design(model,d,x,t):
        e=encode(model,d);hidden=model.prepare_encoded(e,x)
        temporal=temporal_features(model,t,c)
        modulation=jax.vmap(model.time)(temporal).reshape(len(t),len(model.layers),2,model.width)
        hidden=jnp.broadcast_to(hidden,(len(t),*hidden.shape))
        for i,layer in enumerate(model.layers):
            hidden=jax.nn.gelu(jax.vmap(jax.vmap(layer))(hidden)*(1+modulation[:,i,0,None,:])+modulation[:,i,1,None,:])
        xgate,time=lifting_gates(x,t,c);gate=time[:,None]*xgate[None]
        features=model.output_scale*gate[...,None]*jnp.concatenate((hidden,jnp.ones((*hidden.shape[:-1],1),dtype=x.dtype)),axis=-1)
        base=initial_field(x,c,xp=jnp)[None]+jnp.zeros_like(gate)
        actual=fields(model,e,x,t,d['amplitudes'],c)[...,0]
        return features,base,actual

    output.mkdir(parents=True);shutil.copyfile(Path(__file__),output/Path(__file__).name)
    print('Extracting frozen phase-feature subspace; reference used only for this diagnostic',flush=True)
    features,base,actual=[np.asarray(v)for v in design(net,data,jnp.asarray(points),jnp.asarray(times))]
    current=np.r_[np.asarray(net.output.weight[0]),float(net.output.bias[0])]
    reconstruction=base+features@current
    gap=float(np.max(np.abs(reconstruction-actual)));saved_gap=float(np.max(np.abs(actual-saved)))
    if gap>2e-11 or saved_gap>2e-5:raise ValueError(f'Physical phase reconstruction/replay gate failed: {gap}/{saved_gap}')
    a=features.reshape(-1,features.shape[-1]);b=(reference-base).ravel()
    scale=np.linalg.norm(a,axis=0)
    if np.any(scale<=0)or not np.isfinite(a).all()or not np.isfinite(b).all():raise ValueError('Degenerate/nonfinite phase subspace')
    coefficient,_,rank,singular=np.linalg.lstsq(a/scale,b,rcond=rcond)
    coefficient=coefficient/scale
    projected=base+(a@coefficient).reshape(base.shape)
    relative=lambda p,t:float(np.linalg.norm(p-t)/np.linalg.norm(t))
    nx=len(np.unique(points[:,0]));ny=len(points)//nx
    if nx*ny!=len(points):raise ValueError('Mapped query grid required for contour diagnostic')
    contour=contour_distance(points,projected[-1],reference[-1],(nx,ny),1024)
    report=dict(status='frozen_phase_subspace_projection_diagnostic_completed',source_run=str(run),
        diagnostic_reference_source=str(run/'prediction.npz'),reference_rows=int(reference.size),
        evolved_labels_used_for_production_training=0,optimizer_updates=0,fitted_checkpoints_emitted=0,
        precision='Exact saved FP32parameter values promotedCPUFP64',maximum_phase_reconstruction_gap=gap,maximum_saved_GPU_phase_replay_gap=saved_gap,
        columns=a.shape[1],rank=int(rank),relative_SVD_cutoff=rcond,column_scaled_singular_values=singular.tolist(),
        coefficient_norm_ratio=float(np.linalg.norm(coefficient)/np.linalg.norm(current)),
        original=dict(pooled_phi=relative(actual,reference),terminal_phi=relative(actual[-1],reference[-1]),terminal_minority=relative((1-actual[-1])/2,(1-reference[-1])/2)),
        projection=dict(pooled_phi=relative(projected,reference),terminal_phi=relative(projected[-1],reference[-1]),terminal_minority=relative((1-projected[-1])/2,(1-reference[-1])/2),range=[float(projected.min()),float(projected.max())],
            terminal_contour=contour,terminal_contour_distance_over_initial_radius=None if contour['mean_distance']is None else contour['mean_distance']/c.initial.radius),
        scope='Reference-informed unconstrained linearprojection in the CURRENT frozen learned phase features only; no coupled-PDE feasibility, architecture-wide lower bound, generalization or pure-physics accuracy claim. Projected coefficients are not emitted or imported anywhere.')
    (output/'validation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--output',required=True);p.add_argument('--rcond',required=True,type=float)
    a=p.parse_args();probe(a.run,a.output,a.rcond)
