"""Independent checkpoint replay and common/native target comparison."""
from pathlib import Path
import json
import shutil
import sys
ROOT = next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
sys.path.insert(0, str(ROOT))
import jax
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from examples.twophase2d.evaluate import audit_run, relative
from examples.twophase2d.problem import velocity_field
from examples.twophase2d.solver import first_derivative_symbols
from ml_collections import ConfigDict

ARCHIVE = Path(__file__).parent
OUT = ROOT/'reports/twophase-nonlocal-flux'
BASE = Path('/root/autodl-tmp/tf-logs/twophase2d')
RUNS = {'spectral':BASE/'20261027-vortex-flux-spectral',
        'nonlocal':BASE/'20261027-vortex-flux-nonlocal'}


def read_npz(path):
    with np.load(path, allow_pickle=False) as f:
        return {k:f[k].copy() for k in f.files}


def main():
    if jax.default_backend() != 'cpu':
        raise ValueError('Use JAX_PLATFORMS=cpu for checkpoint audit')
    configs = {k:json.loads((v/'config.json').read_text()) for k,v in RUNS.items()}
    cleaned = json.loads(json.dumps(configs))
    for c in cleaned.values():
        del c['advection'], c['reference']['path'], c['log']['run_name']
        assert c['phase']['map'] == 'additive'
        assert 'mass_weight' not in c['loss']
    assert cleaned['spectral'] == cleaned['nonlocal']
    assert (RUNS['spectral']/'model_initial.eqx').read_bytes() == (RUNS['nonlocal']/'model_initial.eqx').read_bytes()
    sources = [{str(p.relative_to(run/'source')):p.read_bytes()
                for p in (run/'source').rglob('*') if p.is_file()} for run in RUNS.values()]
    assert sources[0] == sources[1]
    refs = {k:read_npz(c['reference']['path']) for k,c in configs.items()}
    for key in ('points', 'time', 'phi_local', 'q_local', 'energy_local'):
        np.testing.assert_array_equal(refs['spectral'][key], refs['nonlocal'][key])
    result = dict(status='passed', identical_raw_initial_parameters=True,
        identical_source_snapshots=True, source_file_count=len(sources[0]),
        only_config_differences=['advection', 'reference.path', 'log.run_name'],
        shared_local_reference_fields_exact=True,
        sampling='same seed and identical sampling code/counts; full GPU sample trace not stored',
        runs={})
    histories = {}; fields = {}
    for name,run in RUNS.items():
        output = OUT/name; output.mkdir(parents=True, exist_ok=True)
        audit, history = audit_run(run, output)
        histories[name] = history
        summary = json.loads((run/'summary.json').read_text())
        assert summary['soap_updates'] == 5000
        assert summary['refinement']['iterations'] == 1000
        assert summary['updates'] == 6000
        assert all(len(json.loads(s)['parts']) == 2 for s in (run/'loss.jsonl').read_text().splitlines())
        data = read_npz(run/'prediction.npz'); fields[name] = data
        ref = refs[name]; phi = data['phi']; target = ref['phi_local']; native = ref['phi_nonlocal']
        band = np.abs(target) < .9
        row = dict(run=str(run), cpu_replay=audit,
            pooled_phi_local=relative(phi, target), terminal_phi_local=relative(phi[-1], target[-1]),
            pooled_phi_native=relative(phi, native), terminal_phi_native=relative(phi[-1], native[-1]),
            pooled_band_phi_local=relative(phi[band], target[band]),
            terminal_band_phi_local=relative(phi[-1][band[-1]], target[-1][band[-1]]),
            pooled_q_local=relative(data['q'], ref['q_local']),
            pooled_q_native=relative(data['q'], ref['q_nonlocal']),
            reference_native_local_pooled_phi=relative(native, target),
            reference_native_local_terminal_phi=relative(native[-1], target[-1]),
            trainer_seconds=summary['wall_seconds'],
            mean_compiled_soap_seconds=summary['mean_update_seconds_after_100'],
            first_update_seconds=summary['first_update_seconds'],
            refinement=summary['refinement'],
            soap_5000=next(r for r in history if r['step'] == 5000),
            per_time_phi_local=[relative(a,b) for a,b in zip(phi,target,strict=True)])
        np.testing.assert_allclose(row['pooled_phi_local'],
            summary['final_metrics']['pooled_rel_l2_phi_local'], atol=1e-10, rtol=0)
        assert audit['max_gas_area_relative_drift'] < 1e-12
        result['runs'][name] = row
        for f in ('config.json', 'summary.json', 'metrics.jsonl', 'loss.jsonl', 'refinement_summary.json'):
            shutil.copyfile(run/f, ARCHIVE/f'twophase-nonlocal-flux-{name}-{f}')
        shutil.copyfile(output/'nonlocal-phi.png', ARCHIVE/f'twophase-nonlocal-flux-{name}-native-phi.png')

    # Diagnose discrete advection bias on the SAME independently refined field.
    c = ConfigDict(configs['nonlocal']); ref = refs['nonlocal']
    n = c.sample.Nx
    gx,gy = first_derivative_symbols(n,c,'nonlocal')
    sx,sy = first_derivative_symbols(n,c,'local_ad')
    probes = []
    for idx in (2,5,15,18):
        t = ref['time'][idx]
        velocity = velocity_field(ref['points'], t, c, xp=np).reshape(n,n,2)
        flux = ref['phi_local'][idx,...,None]*velocity
        hat = np.fft.fftn(flux,axes=(0,1))
        first = np.fft.ifftn(gx*hat[...,0]+gy*hat[...,1]).real
        spectral = np.fft.ifftn(sx*hat[...,0]+sy*hat[...,1]).real
        vhat = np.fft.fftn(velocity,axes=(0,1))
        divv = np.fft.ifftn(gx*vhat[...,0]+gy*vhat[...,1]).real
        probes.append(dict(time=float(t), flux_derivative_relative_difference=relative(first,spectral),
            nonlocal_velocity_divergence_max=float(np.max(np.abs(divv))),
            nonlocal_flux_sum=float(np.sum(first))))
    result['common_reference_flux_probes'] = probes
    result['scope'] = 'One fresh seed42 pair; prescribed-flow CH only. Native targets differ; no full NS, stable ranking or causal optimizer diagnosis.'

    fig,axes = plt.subplots(1,3,figsize=(14,4),layout='constrained')
    labels = {'spectral':'FFT flux + nonlocal diffusion', 'nonlocal':'Nonlocal flux + nonlocal diffusion'}
    for name,h in histories.items():
        axes[0].semilogy([r['step'] for r in h],[r['pooled_rel_l2_phi_local'] for r in h],label=labels[name])
        axes[1].semilogy([r['elapsed_seconds'] for r in h],[r['pooled_rel_l2_phi_local'] for r in h],label=labels[name])
        axes[2].plot(ref['time'],result['runs'][name]['per_time_phi_local'],label=labels[name])
    axes[0].set(xlabel='Optimizer iterations',ylabel='Pooled phase relative L2, common local target')
    axes[0].axvline(5000,color='gray',linestyle=':')
    axes[1].set(xlabel='Trainer seconds including JIT/evaluation',ylabel='Pooled phase relative L2')
    axes[2].set(xlabel='Time',ylabel='Final checkpoint phase relative L2')
    for ax in axes:
        ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.savefig(ARCHIVE/'twophase-nonlocal-flux-convergence.png',dpi=180); plt.close(fig)
    indices = (0,5,10,15,20)
    fig,axes = plt.subplots(4,5,figsize=(14,10),layout='constrained')
    rows = [ref['phi_local'],fields['spectral']['phi'],fields['nonlocal']['phi'],
            np.abs(fields['nonlocal']['phi']-ref['phi_local'])]
    error_limit = float(np.max(rows[-1][list(indices)]))
    phase_limit = max(1.1, *(float(np.max(np.abs(row))) for row in rows[:3]))
    for i,(row,title) in enumerate(zip(rows,('Common local reference','FFT flux prediction','Nonlocal flux prediction','Nonlocal absolute error'),strict=True)):
        for j,idx in enumerate(indices):
            rendered = axes[i,j].imshow(row[idx],origin='lower',extent=(0,1,0,1),
                cmap='viridis' if i<3 else 'magma',vmin=-phase_limit if i<3 else 0,
                vmax=phase_limit if i<3 else error_limit)
            axes[i,j].set_xticks([]); axes[i,j].set_yticks([])
            if i==0: axes[i,j].set_title(f't={ref["time"][idx]:.2f}')
            if j==0: axes[i,j].set_ylabel(title)
            if i==0 and j==0: phase_image = rendered
    fig.colorbar(phase_image, ax=axes[:3,:].ravel().tolist(), shrink=.7, label='Phase')
    fig.colorbar(rendered, ax=axes[3,:].ravel().tolist(), shrink=.8, label='Absolute error')
    fig.savefig(ARCHIVE/'twophase-nonlocal-flux-fields.png',dpi=180); plt.close(fig)
    (ARCHIVE/'twophase-nonlocal-flux-analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
