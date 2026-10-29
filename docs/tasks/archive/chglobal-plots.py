"""Scientific figures for the global-time study; never selects training labels."""
from pathlib import Path
import sys,json,argparse
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from examples.confined_ch2d.evaluate import field_plot

OUT=Path(__file__).parent
LABELS={'resolvent':'Differential/spatial inverse','spacetime':'Differential/space-time inverse',
    'integral32':'Integral, joint MLP','inverse':'Neumann inverse, joint MLP',
    'basis24':'Integral, global basis','energy':'Global basis + energy',
    'average':'Global basis, R/t','newton':'Global basis, R/t + Newton-CG',
    'strong':'Global basis, differential inverse','causal':'Global basis, causal differential inverse',
    'wide':'Global basis, differential inverse, width128'}


def main(selected):
    report=json.loads((OUT/'chglobal-analysis.json').read_text())
    with np.load('/root/autodl-tmp/data/confined_ch2d/coarsening_multimode10/reference.npz') as f:ref={k:f[k] for k in f.files}
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for name,r in report['runs'].items():
        label=LABELS[next(key for key in LABELS if '-'+key+'-' in name)]
        run=Path(r['run']);history=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
        for ax,metric in zip(axes,('pooled_rel_l2_phi_local','terminal_rel_l2_phi_local'),strict=True):
            ax.plot([row['step'] for row in history],[100*row[metric] for row in history],label=label,linewidth=1.5)
    for ax,title in zip(axes,('Pooled phase L2, all 21 saved times','Terminal phase L2, t*=1.5'),strict=True):
        ax.set(xlabel='Optimizer iterations (SOAP then L-BFGS)',ylabel='Relative L2 (%)',title=title,ylim=(0,115))
        ax.axvline(5000,color='gray',ls=':',lw=1);ax.grid(alpha=.25)
    axes[0].legend(fontsize=7,ncol=2)
    fig.savefig(OUT/'chglobal-convergence.png',dpi=180);plt.close(fig)
    chosen=report['runs'][selected]
    with np.load(Path(chosen['run'])/'cpu_audit_fields.npz') as f:phi=f['phi'];mu=f['mu']
    indices=np.array([0,2,6,12,20]);view=dict(ref);view['time']=ref['time'][indices]
    field_plot(view,[ref['phi_local'][indices],phi[indices],abs(phi-ref['phi_local'])[indices]],
        OUT/'chglobal-fields.png',['Refined FEM reference',f'Global PINN ({phi.min():.2f}..{phi.max():.2f})','Absolute error'])
    m=chosen['summary']['final_metrics'];fig,axes=plt.subplots(1,3,figsize=(14,3.6),layout='constrained')
    axes[0].plot(ref['time'],100*np.asarray(m['per_time_rel_l2_phi_local']),label='vs refined FEM')
    axes[0].plot(ref['time'],100*np.asarray(m['per_time_rel_l2_phi_nonlocal']),label='vs native')
    axes[0].set(ylabel='Phase relative L2 (%)',title='Error across physical time')
    axes[1].plot(ref['time'],100*np.asarray(m['per_time_rel_l2_mu_nonlocal']))
    axes[1].set(ylabel='Chemical potential relative L2 (%)',title='Chemical error is not hidden')
    axes[2].plot(ref['time'],chosen['energy'],label='Global PINN')
    axes[2].plot(ref['time'],ref['energy_nonlocal'],'k--',label='Native reference')
    axes[2].set(ylabel='Native discrete energy',title='Saved-time energy audit')
    for ax in axes:ax.set_xlabel('Dimensionless time');ax.grid(alpha=.25)
    axes[0].legend();axes[2].legend()
    fig.suptitle('Final global checkpoint; phase range '+str([round(a,3) for a in chosen['phase_range']]),fontsize=11)
    fig.savefig(OUT/'chglobal-errors-energy.png',dpi=180);plt.close(fig)
    text=['| Global control | Common phase pooled / terminal | Native mu pooled / positive-time | Phi range | Trainer seconds |',
          '|---|---:|---:|---:|---:|']
    for name,r in report['runs'].items():
        s=r['summary'];m=s['final_metrics'];label=LABELS[next(key for key in LABELS if '-'+key+'-' in name)]
        text.append(f"| {label} | {100*m['pooled_rel_l2_phi_local']:.2f}% / {100*m['terminal_rel_l2_phi_local']:.2f}% | {100*m['pooled_rel_l2_mu_nonlocal']:.1f}% / {100*r['positive_mu_native']:.1f}% | {r['phase_range'][0]:.2f}..{r['phase_range'][1]:.2f} | {s['wall_seconds']:.1f} |")
    (OUT/'chglobal-results.md').write_text('\n'.join(text)+'\n\nTrainer times exclude initial preprocessing. Newton iterations require Hessian products and are not equivalent to L-BFGS iterations. All outcomes retained; none use reference training labels.\n')
    print('Selected final global run for fields:',selected)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--selected',required=True)
    main(parser.parse_args().selected)
