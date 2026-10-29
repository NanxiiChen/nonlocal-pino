"""Archived postprocessing only; no training or validation-based model selection."""
from pathlib import Path
import sys, json, shutil
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from examples.confined_ch2d.evaluate import field_plot

OUT=Path(__file__).parent
BASE=Path('/root/autodl-tmp/tf-logs/confined_ch2d')


def main():
    a=json.loads((OUT/'chopt-analysis.json').read_text())
    with np.load('/root/autodl-tmp/data/confined_ch2d/coarsening_multimode10/reference.npz') as data:
        ref={k:data[k] for k in data.files}
    times=ref['time'];indices=np.array([0,2,6,12,20]);view=dict(ref);view['time']=times[indices]
    rows={}
    for name in ('original','windows30-42','windows30-2026','windows15-42'):
        r=a['runs'][name];run=Path(r['run'])
        with np.load(run/'prediction.npz') as data:rows[name]={k:data[k] for k in ('phi','mu')}
        tag='chopt-'+name
        for filename in ('config.json','summary.json','replay.json','metrics.jsonl','loss.jsonl','trajectory.json'):
            if (run/filename).is_file():shutil.copyfile(run/filename,OUT/(tag+'-'+filename))
    best=rows['windows30-42']['phi']
    field_plot(view,[ref['phi_local'][indices],best[indices],abs(best-ref['phi_local'])[indices]],
        OUT/'chopt-optimized-fields.png',['Refined FEM reference','Window PINN (seed42)','Absolute error'])
    field_plot(view,[ref['phi_local'][indices],rows['original']['phi'][indices],best[indices]],
        OUT/'chopt-before-after.png',['Refined FEM reference','Previous global PINN','Window PINN (seed42)'])
    fig,axes=plt.subplots(1,3,figsize=(14,3.8),layout='constrained')
    for name,label in [('original','Previous global'),('windows30-42','30 windows, seed42'),
                       ('windows30-2026','30 windows, seed2026'),('windows15-42','15 windows, seed42')]:
        m=a['runs'][name]['final_metrics']
        for ax,key in zip(axes,('phi_local','phi_nonlocal','mu_nonlocal'),strict=True):
            ax.plot(times,100*np.asarray(m['per_time_rel_l2_'+key]),label=label)
    for ax,title in zip(axes,('Phase vs common FEM','Phase vs native nonlocal','Chemical potential vs native'),strict=True):
        ax.set(xlabel='Dimensionless time',ylabel='Relative L2 (%)',title=title);ax.grid(alpha=.25)
    axes[0].legend(fontsize=8);axes[0].set_ylim(0,105);axes[1].set_ylim(0,105)
    fig.savefig(OUT/'chopt-time-errors.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
    for name,label in [('windows30-42','seed42'),('windows30-2026','seed2026')]:
        m=a['runs'][name]['final_metrics'];r=a['runs'][name]['replay']
        axes[0].plot(times,100*np.asarray(m['per_time_rel_l2_phi_local']),label=label)
        axes[1].plot(times,r['energy'],label=label)
    axes[1].plot(times,ref['energy_nonlocal'],'k--',label='Native reference')
    axes[0].set(xlabel='Dimensionless time',ylabel='Phase relative L2 (%)',title='Common refined FEM reference')
    axes[1].set(xlabel='Dimensionless time',ylabel='Native discrete energy',title='Saved-time energy audit')
    for ax in axes:ax.grid(alpha=.25);ax.legend()
    fig.savefig(OUT/'chopt-accuracy-energy.png',dpi=180);plt.close(fig)
    for name in ('blend015','causal-additive','logtime-causal','metric','eliminated','elim-blend8'):
        run=Path(a['runs'][name]['run'])
        for filename in ('config.json','summary.json','best_metrics.json','loss.jsonl','metrics.jsonl'):
            shutil.copyfile(run/filename,OUT/('chopt-'+name+'-'+filename))
    print('Plots and direct-path raw result records archived')


if __name__=='__main__':main()
