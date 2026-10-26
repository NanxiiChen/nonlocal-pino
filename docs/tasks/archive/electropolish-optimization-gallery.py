"""Final physical-field gallery from explicit saved paths and audited metrics."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def main():
    out = Path(__file__).with_suffix('')
    selected = json.loads(out.with_name('electropolish-optimization-selection.json').read_text())
    validation = json.loads(out.with_name('electropolish-optimization-validation.json').read_text())
    assert len(validation) == 3 and all(v['returncode'] == 0 for v in validation)
    seed7 = [v for v in validation if v['seed'] == 7 and v['initial_sampling'] == 'matched_graph']
    assert len(seed7) == 1
    random_runs = {v['seed']:Path(v['path']) for v in validation if v['initial_sampling'] == 'random_interior'}
    assert set(random_runs) == {42,7}
    runs = {'Random seed42': random_runs[42],
            'Candidate seed42': Path(selected['selected_record']['path']),
            'Random seed7': random_runs[7],
            'Candidate seed7': Path(seed7[0]['path'])}
    cfg = json.loads((runs['Candidate seed42']/'config.json').read_text())
    with np.load(cfg['experiment']['reference_path']) as a:
        ref = {k:a[k] for k in a.files}
    times = [0., 1., 2.5, 5.]
    ids = []
    for t in times:
        found = np.flatnonzero(np.isclose(ref['physical_time'],t,rtol=0,atol=1e-12))
        assert len(found) == 1
        ids.append(int(found[0]))
    fields = {'Reference': {k:ref[k][ids] for k in ('phi','c')}}
    metrics = {}
    for label, path in runs.items():
        with np.load(path/'prediction.npz') as a:
            for k in ('X','Y','time'):
                np.testing.assert_array_equal(a[k],ref[k])
            fields[label] = {k:a[k][ids].astype(float) for k in ('phi','c')}
        metrics[label] = {}
        for name in ('phi','c'):
            metrics[label][name] = [float(np.linalg.norm(fields[label][name][j]-fields['Reference'][name][j])/np.linalg.norm(fields['Reference'][name][j])) for j in range(4)]
    plt.rcParams.update({'font.size':10,'axes.titlesize':10})
    legend = [Line2D([0],[0],color='black',ls='--',label='Reference phi=0.5'),
              Line2D([0],[0],color='#ee503f',label='PINN phi=0.5')]
    for name in ('phi','c'):
        fig, axes = plt.subplots(len(fields),4,figsize=(14,2.4*len(fields)),constrained_layout=True)
        for row,label in enumerate(fields):
            for j,t in enumerate(times):
                ax = axes[row,j]
                im = ax.pcolormesh(ref['X'],ref['Y'],fields[label][name][j],vmin=0,vmax=1,cmap='viridis',shading='nearest')
                if label != 'Reference':
                    ax.contour(ref['X'],ref['Y'],fields[label]['phi'][j],levels=[.5],colors='#ee503f',linewidths=.9)
                ax.contour(ref['X'],ref['Y'],fields['Reference']['phi'][j],levels=[.5],colors='black',linestyles='--',linewidths=.9)
                second = 'Graph reference' if label == 'Reference' else f'Relative L2={metrics[label][name][j]:.3e}'
                ax.set(title=f'{label} | {t:g} s\n{second}',xlabel='x/Lc',aspect='equal',xlim=(0,1),ylim=(0,.5),xticks=[0,.5,1],yticks=[0,.25,.5])
                if j == 0:
                    ax.set_ylabel('y/Lc')
        fig.suptitle(f'Electropolish {name}: matched current random-data baselines vs candidate, two seeds\nAll models5000 updates; fixed physical graph/IC/BC; no trajectory labels',fontsize=14)
        fig.colorbar(im,ax=axes,shrink=.85,label=name)
        fig.legend(handles=legend,loc='outside lower center',ncol=2)
        fig.savefig(str(out)+f'-{name}.png',dpi=150)
        plt.close(fig)
    Path(str(out)+'.json').write_text(json.dumps({'reference':cfg['experiment']['reference_path'],
        'runs':{label:str(path) for label,path in runs.items()},'times_seconds':times,'per_time_relative_l2':metrics,
        'note':'Both current-code random-data baselines are shown; older pilots/SOTA are separate historical context in the audit/report'},indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
