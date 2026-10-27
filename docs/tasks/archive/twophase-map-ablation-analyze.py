"""Direct-path paired map ablation; common and native targets, no file hashes."""
from pathlib import Path
import json
import shutil
import sys
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
sys.path.insert(0,str(ROOT))
import jax
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from examples.twophase2d.evaluate import audit_run,relative

ARCHIVE=Path(__file__).parent
OUT=ROOT/'reports/twophase-map-ablation'
BASE=Path('/root/autodl-tmp/tf-logs/twophase2d')
RUNS=dict(additive=BASE/'20261027-vortex-map-ablation-additive',
          cubic=BASE/'20261027-vortex-map-ablation-cubic')


def main():
    if jax.default_backend()!='cpu':raise ValueError('Use CPU for independent checkpoint replay')
    OUT.mkdir(parents=True,exist_ok=True)
    configs={name:json.loads((run/'config.json').read_text()) for name,run in RUNS.items()}
    clean={}
    for name,c in configs.items():
        clean[name]=json.loads(json.dumps(c))
        del clean[name]['log']['run_name']
        if name=='cubic':
            assert clean[name]['phase']=={'map':'inverse_cubic'}
            del clean[name]['phase']
        assert 'mass_weight' not in c['loss']
    assert clean['additive']==clean['cubic'],'Uncontrolled config difference'
    assert (RUNS['additive']/'model_initial.eqx').read_bytes()==(RUNS['cubic']/'model_initial.eqx').read_bytes(),'Initial parameter bytes differ'
    snapshots={}
    for name,run in RUNS.items():
        snapshots[name]={str(p.relative_to(run/'source')):p.read_bytes() for p in (run/'source').rglob('*') if p.is_file()}
    assert snapshots['additive']==snapshots['cubic'],'Paired source snapshots differ'
    result=dict(status='passed',identical_raw_initial_parameter_bytes=True,
        identical_frozen_sources=True,source_files=len(snapshots['additive']),
        only_config_differences=['phase presence/map','log.run_name'],
        sampling_scope='same seed and identical sampling code/counts; full actual GPU sample trace not stored',runs={})
    fields={};histories={}
    for name,run in RUNS.items():
        folder=OUT/name;folder.mkdir(exist_ok=True)
        audit,history=audit_run(run,folder)
        histories[name]=history
        summary=json.loads((run/'summary.json').read_text())
        assert summary['soap_updates']==5000
        assert 0<=summary['refinement']['iterations']<=configs[name]['train']['refinement']['iterations']==1000
        assert summary['updates']==5000+summary['refinement']['iterations']
        with np.load(run/'prediction.npz',allow_pickle=False) as f:fields[name]={k:f[k].copy() for k in f.files}
        with np.load(configs[name]['reference']['path'],allow_pickle=False) as f:ref={k:f[k].copy() for k in f.files}
        phi,q=fields[name]['phi'],fields[name]['q']
        target=ref['phi_local'];native=ref['phi_nonlocal']
        band=np.abs(target)<.9
        row=dict(run=str(run),cpu_replay=audit,
            pooled_phi_local=relative(phi,target),terminal_phi_local=relative(phi[-1],target[-1]),
            pooled_phi_native=relative(phi,native),terminal_phi_native=relative(phi[-1],native[-1]),
            pooled_interface_phi_local=relative(phi[band],target[band]),
            terminal_interface_phi_local=relative(phi[-1][band[-1]],target[-1][band[-1]]),
            pooled_q_local=relative(q,ref['q_local']),pooled_q_native=relative(q,ref['q_nonlocal']),
            terminal_q_native=relative(q[-1],ref['q_nonlocal'][-1]),
            trainer_seconds=summary['wall_seconds'],first_update_seconds=summary['first_update_seconds'],
            mean_compiled_soap_seconds=summary['mean_update_seconds_after_100'],refinement=summary['refinement'],
            soap_5000=next(r for r in history if r['step']==5000),
            per_time_phi_local=[relative(a,b) for a,b in zip(phi,target,strict=True)])
        np.testing.assert_allclose(row['pooled_phi_local'],summary['final_metrics']['pooled_rel_l2_phi_local'],atol=1e-10,rtol=0)
        np.testing.assert_allclose(row['terminal_phi_local'],summary['final_metrics']['terminal_rel_l2_phi_local'],atol=1e-10,rtol=0)
        assert audit['max_gas_area_relative_drift']<1e-12
        losses=[json.loads(s) for s in (run/'loss.jsonl').read_text().splitlines()]
        assert all(len(r['parts'])==2 for r in losses),'Redundant mass loss was not removed'
        result['runs'][name]=row
        for key in ['config.json','summary.json','metrics.jsonl','loss.jsonl','refinement_summary.json']:
            shutil.copyfile(run/key,ARCHIVE/f'twophase-map-ablation-{name}-{key}')
        shutil.copyfile(folder/'nonlocal-phi.png',ARCHIVE/f'twophase-map-ablation-{name}-phi.png')
    a,b=result['runs']['additive'],result['runs']['cubic']
    result['additive_over_cubic_pooled_phi_ratio']=a['pooled_phi_local']/b['pooled_phi_local']
    result['additive_minus_cubic_pooled_phi_percentage_points']=100*(a['pooled_phi_local']-b['pooled_phi_local'])
    result['scope']='Fresh single-seed pair, same raw parameters but map-dependent initial future fields; not statistical robustness, no coupled NS or SOAP diagnosis.'
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for name,h in histories.items():
        label='Direct phase' if name=='additive' else 'Inverse cubic'
        axes[0].semilogy([r['step'] for r in h],[r['pooled_rel_l2_phi_local'] for r in h],label=label)
        axes[1].semilogy([r['elapsed_seconds'] for r in h],[r['pooled_rel_l2_phi_local'] for r in h],label=label)
        axes[2].plot(ref['time'],result['runs'][name]['per_time_phi_local'],label=label)
    axes[0].set(xlabel='Optimizer iterations',ylabel='Pooled phase relative L2, common local target')
    axes[0].axvline(5000,color='gray',linestyle=':',linewidth=.8)
    axes[1].set(xlabel='Trainer seconds, including JIT/evaluation',ylabel='Pooled phase relative L2')
    axes[2].set(xlabel='Time',ylabel='Final checkpoint phase relative L2')
    for ax in axes:ax.grid(alpha=.25);ax.legend()
    fig.suptitle('Only output map changes: hard IC/mass, original PDE loss, SOAP5000 + up to L-BFGS1000')
    fig.savefig(ARCHIVE/'twophase-map-ablation-convergence.png',dpi=180);plt.close(fig)
    (ARCHIVE/'twophase-map-ablation-analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
