"""Matched saved-field comparison on one common refined local reference."""
import ast
import json
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from ml_collections import ConfigDict
from examples.precipitate2d.problem import load_reference
from examples.precipitate2d.solver import coefficients,elastic_fields,relative

ROOT=Path('/root/autodl-tmp/tf-logs/precipitate2d')
RUNS=dict(nonlocal_phase=ROOT/'20261027-periodic-elastic-pilot-01',
          local_spectral=ROOT/'20261027-periodic-elastic-local-spectral-01',
          local_ad=ROOT/'20261027-periodic-elastic-local-ad-01')
OUT=Path(__file__).parent


def flatten(d,prefix=''):
    result={}
    for key,value in d.items():
        path=prefix+'.'+key if prefix else key
        if isinstance(value,dict):result.update(flatten(value,path))
        else:result[path]=value
    return result


def sampling_code(path):
    module=ast.parse(path.read_text())
    fn=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='train')
    assignments=[]
    for node in ast.walk(fn):
        if isinstance(node,ast.Assign) and ast.unparse(node.targets[0]) in ('(kt, key)','draws','times'):
            assignments.append(ast.dump(node,include_attributes=False))
    if len(assignments)!=3:raise AssertionError('Sampling protocol was not found unambiguously')
    return assignments


def main():
    configs={key:json.loads((run/'config.json').read_text()) for key,run in RUNS.items()}
    baseline=flatten(configs['nonlocal_phase'])
    report=dict(runs={},parity={})
    for key,run in RUNS.items():
        differences={k:[baseline[k],v] for k,v in flatten(configs[key]).items() if v!=baseline[k]}
        if set(differences)-{'nl.method','log.run_name'}:raise AssertionError(differences)
        if (run/'model_initial.eqx').read_bytes()!=(RUNS['nonlocal_phase']/'model_initial.eqx').read_bytes():
            raise AssertionError('Initial parameter bytes differ')
        source='source/examples/precipitate2d'
        for rel in ['mechanics.py','network.py']:
            if (run/source/rel).read_bytes()!=(RUNS['nonlocal_phase']/source/rel).read_bytes():
                raise AssertionError('Core source mismatch: '+rel)
        for rel in ['src/nonlocal_pinn/nn/arch.py','src/nonlocal_pinn/nn/embedding.py',
                    'src/nonlocal_pinn/training/optimizers.py','src/nonlocal_pinn/training/runtime.py',
                    'examples/twophase2d/refinement.py']:
            if (run/'source'/rel).read_bytes()!=(RUNS['nonlocal_phase']/'source'/rel).read_bytes():
                raise AssertionError('Shared source mismatch: '+rel)
        if sampling_code(run/source/'train.py')!=sampling_code(RUNS['nonlocal_phase']/source/'train.py'):
            raise AssertionError('Sampling protocol changed')
        report['parity'][key]=dict(config_differences=differences,initial_parameter_bytes_equal=True,
            network_mechanics_source_equal=True,random_sampling_assignment_AST_equal=True,
            shared_arch_optimizer_runtime_refinement_source_equal=True,
            sampling_scope='same seed/counts/code; full actual GPU time arrays were not logged')
    c=ConfigDict(configs['nonlocal_phase'])
    ref=load_reference(c)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    figfields,fieldaxes=plt.subplots(3,5,figsize=(13,7),layout='constrained')
    selected=np.linspace(0,len(ref['time'])-1,5).round().astype(int)
    n=c.sample.Nx
    for label,run in RUNS.items():
        display_name={'nonlocal_phase':'Nonlocal','local_spectral':'Local spectral','local_ad':'Local AD'}[label]
        cfg=ConfigDict(configs[label])
        target='nonlocal' if cfg.nl.method=='nonlocal' else 'local'
        with np.load(run/'replayed_prediction.npz') as f:pred={k:f[k].copy() for k in f.files}
        best=json.loads((run/'best_metrics.json').read_text())
        summary=json.loads((run/'summary.json').read_text())
        result=dict(best_step=best['step'],native_pooled_phi=best['pooled_rel_l2_phi'],
            native_terminal_phi=best['terminal_rel_l2_phi'],
            wall_seconds=summary['wall_seconds'],first_update_seconds=summary['first_update_seconds'],
            mean_update_seconds_after_100=summary['mean_update_seconds_after_100'],
            refinement_seconds=summary['refinement']['wall_seconds'],
            refinement_iterations=summary['refinement']['iterations'],
            final_native_phi=summary['final_metrics']['pooled_rel_l2_phi'],
            common_local={},native={},per_time_common_local_phi=[])
        for field in ('phi','q','u','stress'):
            result['common_local'][field]=dict(pooled=relative(pred[field],ref[field+'_local']),
                terminal=relative(pred[field][-1],ref[field+'_local'][-1]))
            result['native'][field]=dict(pooled=relative(pred[field],ref[field+'_'+target]),
                terminal=relative(pred[field][-1],ref[field+'_'+target][-1]))
        band=np.abs(ref['phi_local'])<.9
        result['common_local_interface_band_phi']=relative(pred['phi'][band],ref['phi_local'][band])
        result['per_time_common_local_phi']=[relative(a,b) for a,b in zip(pred['phi'],ref['phi_local'],strict=True)]
        result['mass_drift']=float(np.max(np.abs(pred['phi'].mean(axis=(1,2))-pred['phi'][0].mean())))
        coef=coefficients(cfg,n,'nonlocal' if target=='nonlocal' else 'local_spectral')
        L=coef[0]
        energy=[]
        for i,p in enumerate(pred['phi']):
            _,u,s,ee=elastic_fields(p,cfg,coef)
            np.testing.assert_allclose(u,pred['u'][i],rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(s,pred['stress'][i],rtol=1e-12,atol=1e-10)
            lap=np.fft.ifft2(L*np.fft.fft2(p)).real
            energy.append(float(np.mean(.25*cfg.physical.beta*(p*p-1)**2+.5*cfg.physical.epsilon**2*p*lap)+ee))
        result['sampled_max_energy_increase']=max(0.,float(np.max(np.diff(energy))))
        result['initial_q_relative_error']=relative(pred['q'][0],ref['q_'+target][0])
        if result['mass_drift']>1e-12 or result['sampled_max_energy_increase']>1e-8:raise AssertionError(result)
        history=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
        axes[0].semilogy([r['step'] for r in history],[r['pooled_rel_l2_phi_local'] for r in history],label=display_name)
        # Exact IC gives zero at t=0; exclude only this point on the log plot.
        # Reported pooled metrics still include every saved time, including0.
        axes[1].semilogy(ref['time'][1:],result['per_time_common_local_phi'][1:],label=display_name)
        if label=='local_ad':
            error=np.abs(pred['phi']-ref['phi_local'])
            for column,i in enumerate(selected):
                for row,a in enumerate((ref['phi_local'][i],pred['phi'][i],error[i])):
                    im=fieldaxes[row,column].imshow(np.pad(a,((0,1),(0,1)),mode='wrap'),origin='lower',
                        extent=(-.5/n,1+.5/n,-.5/n,1+.5/n),cmap='magma' if row==2 else 'coolwarm',
                        vmin=0 if row==2 else -1,vmax=error.max() if row==2 else 1)
                    fieldaxes[row,column].set_xlim(0,1);fieldaxes[row,column].set_ylim(0,1)
                    fieldaxes[row,column].set_xticks([]);fieldaxes[row,column].set_yticks([])
                fieldaxes[0,column].set_title(f't={ref["time"][i]:.2f}')
            for row,name in enumerate(('Local reference','Local AD PINN','Absolute error')):fieldaxes[row,0].set_ylabel(name)
            figfields.colorbar(fieldaxes[0,0].images[0],ax=fieldaxes[:2,:].ravel().tolist(),shrink=.75,label='phi')
            figfields.colorbar(im,ax=fieldaxes[2,:].tolist(),shrink=.75,label='|error|')
        report['runs'][label]=result
        for name in ('summary.json','best_metrics.json','replay.json','metrics.jsonl','refinement_summary.json','config.json'):
            shutil.copyfile(run/name,OUT/f'precipitate-local-{label}-{name}')
    axes[0].set_xlabel('Optimizer updates');axes[0].set_ylabel('Pooled phase relative L2, common local target')
    axes[1].set_xlabel('Time');axes[1].set_ylabel('Selected checkpoint phase relative L2')
    axes[1].set_title('Common local target; exact phase IC at t=0')
    for ax in axes:ax.legend();ax.grid(alpha=.3)
    fig.savefig(OUT/'precipitate-local-comparison.png',dpi=180);plt.close(fig)
    figfields.savefig(OUT/'precipitate-local-ad-fields.png',dpi=180);plt.close(figfields)
    report['scope']='Single seed, historical nonlocal vs two new controls. Mechanics spectral and solved in all. Timings include JIT/eval/refinement; no matched local-AD accuracy guarantee or statistical ranking.'
    (OUT/'precipitate-local-analysis.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
