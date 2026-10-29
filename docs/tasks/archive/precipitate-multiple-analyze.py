"""Replay final checkpoints and audit the harder coherent-precipitate variant."""
from pathlib import Path
import sys
ROOT = next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
sys.path.insert(0,str(ROOT))
import json
import runpy
import shutil
import jax
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from ml_collections import ConfigDict
from examples.precipitate2d.evaluate import evaluate
from examples.precipitate2d.problem import initial_field, load_reference
from examples.precipitate2d.solver import coefficients, elastic_fields, relative

OUT = Path(__file__).parent
BASE = Path('/root/autodl-tmp/tf-logs/precipitate2d')
RUNS = {'nonlocal':BASE/'20261027-multiple-elastic-nonlocal-01',
        'local_ad':BASE/'20261027-multiple-elastic-local-ad-01'}
component_areas = runpy.run_path(str(OUT/'precipitate-multiple-preflight.py'))['components']


def data(path):
    with np.load(path,allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}


def main():
    if jax.default_backend() != 'cpu':raise ValueError('Use JAX_PLATFORMS=cpu')
    configs = {name:json.loads((run/'config.json').read_text()) for name,run in RUNS.items()}
    cleaned = json.loads(json.dumps(configs))
    for c in cleaned.values():del c['nl']['method'],c['log']['run_name']
    assert cleaned['nonlocal'] == cleaned['local_ad']
    assert (RUNS['nonlocal']/'model_initial.eqx').read_bytes() == (RUNS['local_ad']/'model_initial.eqx').read_bytes()
    sources = [{str(p.relative_to(run/'source')):p.read_bytes() for p in (run/'source').rglob('*') if p.is_file()} for run in RUNS.values()]
    assert sources[0] == sources[1]
    c = ConfigDict(configs['nonlocal']); ref = load_reference(c)
    checks = json.loads(Path(c.reference.path).with_suffix('.json').read_text())
    assert checks['status'] == 'passed'
    result = dict(status='passed',identical_raw_initial_parameters=True,
        identical_source_files=len(sources[0]),only_config_differences=['nl.method','log.run_name'],
        sampling_scope='same seed, shape and sampling code; actual full GPU time-sample trace not stored',
        reference_checks=checks,runs={},reference_morphology={})
    for kind in ('local','nonlocal','uncoupled'):
        p = ref['phi_'+kind]
        areas = [component_areas(v) for v in p]
        result['reference_morphology'][kind] = dict(component_areas=areas,
            component_counts=[len(a) for a in areas],
            initial_terminal_change=relative(p[-1],p[0]))
    histories = {}; predictions = {}
    for name,run in RUNS.items():
        # Each NEW run owns the evaluation outputs. No historical run is rewritten.
        replay = evaluate(run,run/'model_final.eqx')
        pred = data(run/'replayed_prediction.npz'); predictions[name] = pred
        saved = data(run/'prediction.npz')
        replay_errors = {}
        for field in ('phi','q','u','stress'):
            np.testing.assert_allclose(pred[field],saved[field],atol=1e-8,rtol=1e-8)
            replay_errors[field] = float(np.max(np.abs(pred[field]-saved[field])))
        summary = json.loads((run/'summary.json').read_text())
        assert summary['final_metrics']['step'] == 5500
        assert summary['refinement']['iterations'] == 500
        kind = 'nonlocal' if name == 'nonlocal' else 'local'
        row = dict(run=str(run),checkpoint='model_final.eqx',cpu_replay=replay,
            replay_max_absolute=replay_errors,trainer_seconds=summary['wall_seconds'],
            mean_compiled_soap_seconds=summary['mean_update_seconds_after_100'],
            refinement=summary['refinement'],common_local={},native={},
            best_metrics=json.loads((run/'best_metrics.json').read_text()))
        for field in ('phi','q','u','stress'):
            row['common_local'][field] = dict(
                pooled=relative(pred[field],ref[field+'_local']),
                terminal=relative(pred[field][-1],ref[field+'_local'][-1]))
            row['native'][field] = dict(pooled=relative(pred[field],ref[field+'_'+kind]),
                terminal=relative(pred[field][-1],ref[field+'_'+kind][-1]))
            np.testing.assert_allclose(row['native'][field]['pooled'],
                summary['final_metrics']['pooled_rel_l2_'+field],atol=1e-8,rtol=1e-8)
        band = np.abs(ref['phi_'+kind]) < .9
        row['native_interface_phi'] = relative(pred['phi'][band],ref['phi_'+kind][band])
        row['initial_q_relative_error'] = relative(pred['q'][0],ref['q_'+kind][0])
        row['per_time_phi_local'] = [relative(a,b) for a,b in zip(pred['phi'],ref['phi_local'],strict=True)]
        row['component_areas'] = [component_areas(p) for p in pred['phi']]
        row['component_counts'] = [len(a) for a in row['component_areas']]
        training_mean = float(initial_field(ref['points'],c,xp=np).mean())
        row['mass_drift'] = float(np.max(np.abs(pred['phi'].mean(axis=(1,2))-training_mean)))
        row['reference_initial_mean_offset'] = float(ref['phi_'+kind][0].mean()-training_mean)
        row['reference_initial_phi_difference'] = relative(initial_field(ref['points'],c,xp=np).reshape(c.sample.Nx,c.sample.Nx),ref['phi_'+kind][0])
        assert row['mass_drift'] < 1e-12
        coef = coefficients(c,c.sample.Nx,'nonlocal' if kind=='nonlocal' else 'local_spectral')
        energies = []
        maximum_force = 0.
        for i,p in enumerate(pred['phi']):
            _,u,stress,ee = elastic_fields(p,c,coef)
            np.testing.assert_allclose(u,pred['u'][i],atol=1e-12,rtol=1e-12)
            np.testing.assert_allclose(stress,pred['stress'][i],atol=1e-10,rtol=1e-12)
            lap = np.fft.ifft2(coef[0]*np.fft.fft2(p)).real
            energies.append(float(np.mean(.25*c.physical.beta*(p*p-1)**2+.5*c.physical.epsilon**2*p*lap)+ee))
            sx,sy,sxy = np.moveaxis(stress,-1,0); kx,ky = coef[3:5]
            force_x = np.fft.ifft2(1j*(kx*np.fft.fft2(sx)+ky*np.fft.fft2(sxy))).real
            force_y = np.fft.ifft2(1j*(kx*np.fft.fft2(sxy)+ky*np.fft.fft2(sy))).real
            maximum_force = max(maximum_force,float(np.max(np.abs(force_x))),float(np.max(np.abs(force_y))))
        row['energy'] = energies
        row['sampled_max_energy_increase'] = max(0.,float(np.max(np.diff(energies))))
        row['maximum_mechanical_force_residual'] = maximum_force
        assert maximum_force < 1e-8
        assert all(len(json.loads(s)['parts'])==2 for s in (run/'loss.jsonl').read_text().splitlines())
        histories[name] = [json.loads(s) for s in (run/'metrics.jsonl').read_text().splitlines()]
        result['runs'][name] = row
        for f in ('config.json','summary.json','best_metrics.json','metrics.jsonl','loss.jsonl','refinement_summary.json','replay.json'):
            shutil.copyfile(run/f,OUT/f'precipitate-multiple-{name}-{f}')
        for f in ('phi_fields.png','mechanical_fields.png','convergence_and_shape.png'):
            shutil.copyfile(run/f,OUT/f'precipitate-multiple-{name}-{f}')
    fig,axes = plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for name,h in histories.items():
        axes[0].semilogy([v['step'] for v in h],[v['pooled_rel_l2_phi_local'] for v in h],label=name)
        axes[1].plot(ref['time'],result['runs'][name]['per_time_phi_local'],label=name)
        axes[2].plot(ref['time'],result['runs'][name]['energy'],label=name+' prediction')
    for kind in ('nonlocal','local'):axes[2].plot(ref['time'],ref['energy_'+kind],':',label=kind+' reference')
    axes[0].set(xlabel='Optimizer iterations',ylabel='Pooled phase relative L2, common local target')
    axes[1].set(xlabel='Time',ylabel='Final phase relative L2, common local target')
    axes[2].set(xlabel='Time',ylabel='Total discrete free energy')
    for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.25)
    fig.savefig(OUT/'precipitate-multiple-comparison.png',dpi=180);plt.close(fig)
    # Same target and color scale for the requested nonlocal/local-AD comparison.
    indices = (0,5,10,15,20)
    fig,axes = plt.subplots(3,5,figsize=(13,7),layout='constrained')
    rows = ((ref['phi_local'],'Common local reference'),
            (predictions['nonlocal']['phi'],'Nonlocal PINN'),
            (predictions['local_ad']['phi'],'Local AD PINN'))
    for row,(fields,label) in enumerate(rows):
        for col,i in enumerate(indices):
            im = axes[row,col].imshow(fields[i],origin='lower',extent=(0,1,0,1),cmap='coolwarm',vmin=-1.05,vmax=1.05)
            axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
            if row==0:axes[row,col].set_title(f't={ref["time"][i]:.1f}')
            if col==0:axes[row,col].set_ylabel(label)
    fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.65,label='phi')
    fig.savefig(OUT/'precipitate-multiple-common-fields.png',dpi=180);plt.close(fig)
    # Reference dynamics make the elastic effect visible independently of PINN.
    indices = (0,5,10,15,20)
    fig,axes = plt.subplots(2,5,figsize=(13,5),layout='constrained')
    for row,kind in enumerate(('nonlocal','uncoupled')):
        for col,i in enumerate(indices):
            axes[row,col].imshow(ref['phi_'+kind][i],origin='lower',extent=(0,1,0,1),cmap='coolwarm',vmin=-1.05,vmax=1.05)
            axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
            if row==0:axes[row,col].set_title(f't={ref["time"][i]:.1f}')
            if col==0:axes[row,col].set_ylabel('Elastic reference' if row==0 else 'No elastic feedback')
    fig.savefig(OUT/'precipitate-multiple-reference-dynamics.png',dpi=180);plt.close(fig)
    result['scope'] = 'Fresh single-seed matched pair; more complex morphology, still homogeneous solved spectral mechanics. No learned-displacement, calibrated-material or cross-seed robustness claim. Final-budget fields primary; best metrics separate.'
    (OUT/'precipitate-multiple-analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
