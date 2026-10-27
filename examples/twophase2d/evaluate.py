"""Direct saved-checkpoint CPU audit, convergence and representative phase fields."""
import argparse
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ml_collections import ConfigDict
import numpy as np

from examples.twophase2d.network import PhaseNetwork
from examples.twophase2d.problem import validate_config, load_reference
from examples.twophase2d.solver import symbols
from examples.twophase2d.fields import predict_fields


def relative(a, b):
    norm = np.linalg.norm(b)
    if norm <= 0:raise ValueError('Relative L2 requires positive reference norm')
    return float(np.linalg.norm(a-b)/norm)


def audit_run(run, output, *, fp32_replay_tolerance=5e-6):
    config = ConfigDict(json.loads((run/'config.json').read_text()))
    validate_config(config)
    summary = json.loads((run/'summary.json').read_text())
    expected_updates = config.train.iterations
    if 'refinement' in config.train:
        expected_updates += summary['refinement']['iterations']
    if summary['updates'] != expected_updates:
        raise ValueError('Saved training budget is incomplete')
    reference = load_reference(config)
    dtype = np.dtype(config.train.compute_dtype)
    jax.config.update('jax_enable_x64', dtype == np.dtype('float64'))
    jax.config.update('jax_default_matmul_precision', config.train.matmul_precision)
    with jax.enable_x64(False):
        key = jax.random.split(jax.random.PRNGKey(config.train.seed))[0]
        template = PhaseNetwork(config, key)
    template = jax.tree.map(lambda a:a.astype(dtype) if eqx.is_inexact_array(a) else a, template)
    model = eqx.tree_deserialise_leaves(run/'model_final.eqx', template)
    if any(np.dtype(a.dtype) != dtype for a in jax.tree.leaves(eqx.filter(model,eqx.is_inexact_array))):
        raise TypeError('Saved model parameter dtype differs from config')
    points = jnp.asarray(reference['points'], dtype=dtype)
    times = jnp.asarray(reference['time'], dtype=dtype)
    predict = eqx.filter_jit(lambda net:jax.vmap(lambda t:predict_fields(net,points,t,config))(times))
    fields = np.asarray(predict(model), dtype=np.float64).reshape(len(times),config.sample.Ny,config.sample.Nx,2)
    phi,q = fields[...,0],fields[...,1]
    # Frozen GPU replay is exact; CPU/GPU FP32 rounding on the 4x64 network
    # reaches 3.31e-6. Keep an explicit absolute tolerance and report differences.
    if not np.isfinite(fp32_replay_tolerance) or fp32_replay_tolerance <= 0:
        raise ValueError('Explicit FP32 replay tolerance must be positive finite')
    tolerance = fp32_replay_tolerance if dtype == np.dtype('float32') else 1e-10
    replay = {}
    with np.load(run/'prediction.npz') as saved:
        for key,values in [('phi',phi),('q',q)]:
            replay[key] = dict(max_absolute=float(np.max(np.abs(values-saved[key]))),
                               relative_L2=relative(values, saved[key]))
            np.testing.assert_allclose(values,saved[key],atol=tolerance,rtol=0)
    name = config.nl.method
    target_key = 'local' if name == 'local_ad' else 'nonlocal'
    target = reference['phi_'+target_key]
    gas = np.mean((1-phi)/2,axis=(1,2))
    gas0 = np.mean((1-target[0])/2)
    lam,_ = symbols(config.sample.Nx,config,config.nl.method)
    action = np.fft.ifftn(lam[None]*np.fft.fftn(phi,axes=(1,2)),axes=(1,2)).real
    energies = np.mean(.25*(phi**2-1)**2+.5*config.physical.epsilon**2*phi*action,axis=(1,2))
    result = dict(run=str(run),method=name,compute_dtype=str(dtype),
        replay_absolute_tolerance=tolerance, replay_differences=replay,
        terminal_phi_relative_L2=relative(phi[-1],target[-1]),
        pooled_phi_relative_L2=relative(phi,target),
        terminal_phi_relative_L2_to_local=relative(phi[-1],reference['phi_local'][-1]),
        terminal_q_rms_error=float(np.sqrt(np.mean((q[-1]-reference['q_'+target_key][-1])**2))),
        max_gas_area_relative_drift=float(np.max(np.abs(gas-gas0))/gas0),
        phase_min=float(phi.min()),phase_max=float(phi.max()),
        reference_phase_min=float(target.min()),reference_phase_max=float(target.max()),
        max_energy_increase_between_saved_times=float(np.max(np.diff(energies))),
        per_time_phi_relative_L2=[relative(a,b) for a,b in zip(phi,target,strict=True)],
        per_time_phi_relative_L2_to_local=[relative(a,b) for a,b in zip(phi,reference['phi_local'],strict=True)],
        per_time_interface_phi_relative_L2=[relative(a[np.abs(b)<.9],b[np.abs(b)<.9]) for a,b in zip(phi,target,strict=True)],
        trainer_wall_seconds=summary['wall_seconds'],
        mean_compiled_step_seconds=summary['mean_update_seconds_after_100'],
        first_update_seconds=summary['first_update_seconds'])
    np.testing.assert_allclose(result['terminal_phi_relative_L2'],summary['final_metrics']['terminal_rel_l2_phi'],atol=tolerance,rtol=0)
    history = [json.loads(s) for s in (run/'metrics.jsonl').read_text().splitlines()]
    expected_steps = list(range(0,config.train.iterations+1,config.log.eval_every))
    if expected_steps[-1] != config.train.iterations:expected_steps.append(config.train.iterations)
    if expected_updates > config.train.iterations:
        expected_steps.extend(range(config.train.iterations+config.log.eval_every, expected_updates+1, config.log.eval_every))
        if expected_steps[-1] != expected_updates:expected_steps.append(expected_updates)
    if [r['step'] for r in history] != expected_steps:
        raise ValueError('Expected complete equally spaced evaluation history')
    late = [r for r in history if r['step'] >= .8*expected_updates]
    result['late_terminal_phi_relative_L2'] = dict(median=float(np.median([r['terminal_rel_l2_phi'] for r in late])),
        maximum=float(np.max([r['terminal_rel_l2_phi'] for r in late])))
    ids = np.linspace(0,len(times)-1,5,dtype=int)
    axis = np.arange(config.sample.Nx)*config.geometry.length/config.sample.Nx
    X,Y = np.meshgrid(axis,axis,indexing='xy')
    error = np.abs(phi-target)
    fig,axes = plt.subplots(3,5,figsize=(14,8),layout='constrained')
    for row,values in enumerate((target,phi,error)):
        for col,i in enumerate(ids):
            ax=axes[row,col]
            im=ax.pcolormesh(X,Y,values[i],shading='nearest',vmin=min(-1.,float(phi.min()),float(target.min())) if row<2 else 0,
                vmax=max(1.,float(phi.max()),float(target.max())) if row<2 else error[ids].max(),cmap='viridis' if row<2 else 'magma')
            ax.set(aspect='equal')
            if row<2:ax.contour(X,Y,values[i],levels=[0],colors='white',linewidths=.6)
            if row==0:ax.set_title(f't={float(times[i]):.2f}')
            if col==0:ax.set_ylabel(('Reference','PINN final','Absolute error')[row]+'\ny')
            if row==2:ax.set_xlabel('x')
            if row==1:
                phi_image=im
                ax.text(.02,.02,f'L2={result["per_time_phi_relative_L2"][i]:.2e}',transform=ax.transAxes,fontsize=8,bbox={'facecolor':'white','alpha':.85,'edgecolor':'none'})
    fig.colorbar(phi_image,ax=axes[:2,:].ravel().tolist(),label='phi (full predicted/reference range)',shrink=.75)
    fig.colorbar(im,ax=axes[2,:].ravel().tolist(),label='Absolute error',shrink=.75)
    fig.suptitle(name+f': prescribed-flow phase-field pilot; matched reference, final{expected_updates} updates')
    fig.savefig(output/(name+'-phi.png'),dpi=170);plt.close(fig)
    return result,history


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,action='append',required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    if jax.default_backend()!='cpu':raise ValueError('Use JAX_PLATFORMS=cpu for independent replay')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    results=[];histories={}
    for run in args.run:
        result,h=audit_run(run,args.output_dir);results.append(result);histories[result['method']]=h
        print(json.dumps(result,indent=2),flush=True)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for name,h in histories.items():
        for col,x in enumerate(('step','elapsed_seconds')):
            axes[col].semilogy([r[x] for r in h],[max(r['terminal_rel_l2_phi'],1e-12) for r in h],label=name)
    for col in range(2):
        axes[col].set(xlabel='Updates' if col==0 else 'Actual elapsed seconds (JIT/eval included)',ylabel='Terminal phi relative L2 to matched reference')
        axes[col].legend();axes[col].grid(alpha=.2)
    fig.suptitle('Same seed/model/SOAP/points; local and finite-horizon targets differ')
    fig.savefig(args.output_dir/'convergence.png',dpi=170);plt.close(fig)
    (args.output_dir/'audit.json').write_text(json.dumps(dict(status='passed',runs=results),indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
