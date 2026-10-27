"""Direct-path matched-case replay, independent metrics and timing plots."""
from pathlib import Path
import json
import sys
ROOT = next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
sys.path.insert(0,str(ROOT))
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from examples.twophase2d.evaluate import audit_run
from examples.twophase2d.network import PhaseNetwork
from examples.twophase2d.fields import predict_fields
from ml_collections import ConfigDict

OUT = ROOT/'reports/twophase-matched-local'


def relative(a,b):
    return float(np.linalg.norm(a-b)/np.linalg.norm(b))


def main():
    if jax.default_backend()!='cpu':raise ValueError('Independent replay requires CPU')
    manifest=json.loads((OUT/'manifest.json').read_text())
    if manifest['status']!='completed':raise ValueError('Both runs must complete')
    results=[];configs=[];histories=[];runs=[]
    for item in manifest['runs']:
        run=Path(item['run']);runs.append(run)
        c=json.loads((run/'config.json').read_text());configs.append(c)
        audit,history=audit_run(run,OUT)
        histories.append(history)
        s=json.loads((run/'summary.json').read_text())
        with np.load(run/'prediction.npz') as f:phi=f['phi'].copy();q=f['q'].copy()
        with np.load(c['reference']['path']) as f:
            target=f['phi_local'].copy();times=f['time'].copy();matched=f['phi_'+('local' if c['nl']['method']=='local_ad' else 'nonlocal')].copy()
        mask=np.abs(target)<.9
        row=dict(method=item['method'],run=str(run),cpu_replay=audit,
                 pooled_phi_relative_L2_local=relative(phi,target),
                 terminal_phi_relative_L2_local=relative(phi[-1],target[-1]),
                 pooled_phi_relative_L2_native=relative(phi,matched),
                 terminal_phi_relative_L2_native=relative(phi[-1],matched[-1]),
                 pooled_interface_phi_relative_L2_local=relative(phi[mask],target[mask]),
                 per_time_phi_relative_L2_local=[relative(a,b) for a,b in zip(phi,target,strict=True)],
                 trainer_wall_seconds=s['wall_seconds'],external_wall_seconds=item['external_wall_seconds'],
                 soap_stage_metrics=next(r for r in history if r['step']==5000),
                 first_soap_update_seconds=s['first_update_seconds'],
                 mean_soap_update_seconds_after_100=s['mean_update_seconds_after_100'],
                 refinement=s['refinement'])
        np.testing.assert_allclose(row['pooled_phi_relative_L2_local'],s['final_metrics']['pooled_rel_l2_phi_local'],atol=1e-12,rtol=0)
        np.testing.assert_allclose(row['terminal_phi_relative_L2_local'],s['final_metrics']['terminal_rel_l2_phi_local'],atol=1e-12,rtol=0)
        if s['soap_updates']!=5000 or s['refinement']['iterations']!=1000:raise ValueError('Unequal completed budget')
        cc=ConfigDict(c)
        with jax.enable_x64(False):template=PhaseNetwork(cc,jax.random.split(jax.random.PRNGKey(42))[0])
        template=jax.tree.map(lambda a:a.astype(np.float64) if eqx.is_inexact_array(a) else a,template)
        # Exact initial reconstruction is checked on the training GPU backend.
        # CPU FP32 random initialization is not bitwise backend invariant.
        model=eqx.tree_deserialise_leaves(run/'model_final.eqx',template)
        np.testing.assert_array_equal(model.modes,np.arange(1,cc.arch.spatial_modes+1))
        max_edge=0.
        for axis in (0,1):
            point=jnp.array([.37,.41]).at[axis].set(0.)
            other=point.at[axis].set(1.)
            for time in (.2,1.,2.):max_edge=max(max_edge,float(jnp.max(jnp.abs(model(point,time)-model(other,time)))))
        if max_edge>1e-12:raise ValueError('Periodic edge violation')
        row['max_periodic_edge_gap']=max_edge
        results.append(row)
    clean=[]
    for c in configs:
        c=json.loads(json.dumps(c));del c['nl']['method'];del c['log']['run_name'];clean.append(c)
    if clean[0]!=clean[1]:raise ValueError('Uncontrolled config difference')
    if (runs[0]/'model_initial.eqx').read_bytes()!=(runs[1]/'model_initial.eqx').read_bytes():raise ValueError('Initial bytes differ')
    snapshots=[]
    for run in runs:
        source=run/'source'
        snapshots.append({str(p.relative_to(source)):p.read_bytes() for p in source.rglob('*') if p.is_file()})
    if snapshots[0]!=snapshots[1]:raise ValueError('Paired sources differ')
    if len(results)!=2:raise ValueError('Expected one matched pair')
    result=dict(status='passed',seed=42,common_reference=configs[0]['reference']['path'],
        identical_config_except=['nl.method','log.run_name'],identical_initial_checkpoint_bytes=True,
        identical_frozen_sources=True,source_files=len(snapshots[0]),
        nonlocal_over_local_trainer_time=results[0]['trainer_wall_seconds']/results[1]['trainer_wall_seconds'],
        local_over_nonlocal_soap_step_time=results[1]['mean_soap_update_seconds_after_100']/results[0]['mean_soap_update_seconds_after_100'],
        limitations=['one seed, one prescribed-flow CH case; no coupled Navier-Stokes',
          'local AD replaces diffusion Hessians only; both arms retain spectral-flux advection',
          'different native equations at finite horizon; common local reference and native errors both reported',
          'external time includes process/backend/setup; trainer time starts after network/operator setup and includes JIT/evaluation/saves/refinement',
          'first update includes its JIT; compiled SOAP time excludes time-sample generation/evaluation/checkpoint work; L-BFGS time includes its JIT/callbacks'],runs=results)
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for row,h in zip(results,histories,strict=True):
        label=row['method']
        axes[0].semilogy([r['step'] for r in h],[r['pooled_rel_l2_phi_local'] for r in h],label=label)
        axes[1].semilogy([r['elapsed_seconds'] for r in h],[r['pooled_rel_l2_phi_local'] for r in h],label=label)
        axes[2].plot(times,row['per_time_phi_relative_L2_local'],label=label)
    axes[0].set(xlabel='SOAP then L-BFGS iterations',ylabel='Pooled phi relative L2 to common local reference')
    axes[0].axvline(5000,color='gray',ls='--',lw=.8)
    axes[1].set(xlabel='Trainer seconds (JIT/evaluation included)',ylabel='Pooled phi relative L2 to common local reference')
    axes[2].set(xlabel='Physical time',ylabel='Final-model phi relative L2 to common local reference')
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Matched seed42: inverse cubic, exact grid mass, spectral flux, FP64, SOAP5000 + L-BFGS1000')
    fig.savefig(OUT/'common-reference-convergence.png',dpi=180)
    plt.close(fig)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
