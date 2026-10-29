"""Matched full-horizon two-field convergence and physical field figures."""
from pathlib import Path
import json,sys
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ml_collections import ConfigDict
from examples.confined_ch2d.problem import load_reference
from examples.confined_ch2d.evaluate import field_plot
OUT=Path(__file__).parent;report=json.loads((OUT/'chbalance-analysis.json').read_text())
names=['20261028-chbalance-'+label+'-01' for label in ('joint','gradnorm','stagger','gradnorm-raw')]
labels=['Fixed joint','Gradient-norm balance','Alternating losses then joint','Raw normalized + gradient balance']
keys=['pooled_rel_l2_phi_local','terminal_rel_l2_phi_local','pooled_rel_l2_mu_nonlocal','terminal_rel_l2_mu_nonlocal']
fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
energy_fig,energy_ax=plt.subplots(figsize=(6.5,4),layout='constrained')
for name,label in zip(names,labels,strict=True):
    row=report['runs'][name];run=Path(row['run']);c=ConfigDict(json.loads((run/'config.json').read_text()));ref=load_reference(c)
    history=[json.loads(s) for s in (run/'metrics.jsonl').read_text().splitlines()]
    for ax,key in zip(axes.flat,keys,strict=True):
        ax.semilogy([r['step'] for r in history],[100*r[key] for r in history],label=label)
    energy_ax.plot(row['energy_time'],row['energy'],label=label)
    with np.load(run/'cpu_audit_fields.npz') as f:
        for stage,key in [('final','phi'),('best','best_phi')]:
            field_plot(ref,[ref['phi_local'],f[key],abs(f[key]-ref['phi_local'])],
                OUT/('chbalance-'+name.removeprefix('20261028-chbalance-').removesuffix('-01')+'-'+stage+'-fields.png'),['Common FEM reference',label,'Absolute error'])
for ax,title in zip(axes.flat,['Phase pooled / FEM','Phase terminal / FEM','Mu pooled / native','Mu terminal / native'],strict=True):
    ax.set(title=title,xlabel='Optimizer updates',ylabel='Relative L2 (%)');ax.grid(alpha=.25)
    ax.axvline(4000,color='grey',ls='--',lw=.7);ax.axvline(5000,color='grey',ls=':',lw=.7)
axes.flat[0].legend(fontsize=8);fig.suptitle('Original ten-wave T1.5; matched shared heads, 5000 SOAP + 500 L-BFGS')
fig.savefig(OUT/'chbalance-convergence.png',dpi=180);plt.close(fig)
energy_ax.plot(ref['time'],ref['energy_nonlocal'],color='black',label='Native reference')
energy_ax.set(xlabel='Time',ylabel='Native free energy');energy_ax.legend(fontsize=8);energy_ax.grid(alpha=.25)
energy_fig.savefig(OUT/'chbalance-energy.png',dpi=180);plt.close(energy_fig)
