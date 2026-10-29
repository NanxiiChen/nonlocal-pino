"""Fixed full-domain convergence and fields, keeping supervised audit separate."""
from pathlib import Path
import sys,json
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file());sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ml_collections import ConfigDict
from examples.confined_ch2d.problem import load_reference
from examples.confined_ch2d.evaluate import field_plot
OUT=Path(__file__).parent;report=json.loads((OUT/'chmixed-analysis.json').read_text())
names=['20261028-chglobal-average-01','20261028-chglobal-strong-01','20261028-chmixed-shared-01','20261028-chmixed-separate-01','20261028-chmixed-block-01']
labels=['Eliminated R/t','Eliminated strong','Mixed shared','Mixed separate','Mixed block inverse']
fig,axes=plt.subplots(1,3,figsize=(15,4.2),layout='constrained')
keys=['pooled_rel_l2_phi_local','terminal_rel_l2_phi_local','pooled_rel_l2_mu_nonlocal']
for name,label in zip(names,labels,strict=True):
    row=report['runs'][name];run=Path(row['run']);history=[json.loads(s) for s in (run/'metrics.jsonl').read_text().splitlines()]
    for ax,key in zip(axes,keys,strict=True):ax.semilogy([x['step'] for x in history],[100*x[key] for x in history],label=label)
for ax,title in zip(axes,['Phase pooled / common FEM','Phase terminal / common FEM','Chemical pooled / native'],strict=True):
    ax.set(title=title,xlabel='Completed optimizer updates',ylabel='Relative L2 (%)');ax.grid(alpha=.25);ax.axvline(5000,color='grey',ls=':',lw=.8)
axes[0].legend(fontsize=8);fig.suptitle('Global-time physics-only cold starts: 5000 SOAP + 500 L-BFGS, seed42')
fig.savefig(OUT/'chmixed-convergence.png',dpi=180);plt.close(fig)
run=Path(report['runs'][names[-1]]['run']);c=ConfigDict(json.loads((run/'config.json').read_text()));ref=load_reference(c)
with np.load(run/'cpu_audit_fields.npz') as f:phi=f['phi']
field_plot(ref,[ref['phi_local'],phi,np.abs(phi-ref['phi_local'])],OUT/'chmixed-block-fields.png',['Common FEM reference','Global mixed block PINN','Absolute error'])
lines=['# Global mixed-field controls','', 'All main rows are final physics-only checkpoints, seed42, same frozen T1.5 physical/reference case, SOAP5000+LBFGS500. No reference loss. Shared/separate have identical40 source files; separate/block initial checkpoint files are identical. Field layout changes parameter count. The block control also changes differential to integrated residual and norm, so benefits cannot be attributed to output scaling alone.','', '| Model | Parameters | Phase pooled FEM % | Phase terminal FEM % | Mu pooled native % | Mu terminal native % | Trainer s | Sampled energy max rise |','|---|---:|---:|---:|---:|---:|---:|---:|']
for name,label in zip(names,labels,strict=True):
    r=report['runs'][name];s=r['summary'];m=s['final_metrics']
    lines.append(f"| {label} | {r['parameter_count']} | {100*m[keys[0]]:.4f} | {100*m[keys[1]]:.4f} | {100*m[keys[2]]:.4f} | {100*m['terminal_rel_l2_mu_nonlocal']:.4f} | {s['wall_seconds']:.2f} | {r['energy_max_increase']:.6g} |")
lines+=['','Block control improves terminal phase/mu and removes sampled energy rises, but pooled phase is worse than prior21.54% model. None is a satisfactory high-accuracy global solution; no default/SOTA promotion or seed stability claim.','', '## Explicitly supervised-only capacity diagnostic','', 'Same separate global network and identical initial weights,5000 SOAP with native field targets: phase pooled0.43846%, terminal0.42996%; mu pooled2.7160%, terminal5.16961%. These are not PINN accuracy results, and no weights enter any physics run. This establishes field expressivity, not accurate derivatives or a joint small-PDE-residual approximation. See chmixed-representation-audit.json for actual physical residuals.','', '## Validation-only temporal coefficients','', 'Free rank24 nodal coefficients: phase0.07697%, mu0.94542%, strong evolution loss1.00657. Known initial derivative/mu constraints retain phase0.07372%/mu0.73586% and reduce that loss to0.054208. This is a constrained field projection, not a residual-minimizing neural/PINN fit or a proof of basis inadequacy. Native reference replay1.90e-14.','', 'CPU final/best replays and independent chemical/mass/energy audits are in chmixed-analysis.json. Both new numerical tests pass. Paths load directly; no hashes, new reference solver or Git history changes.']
(OUT/'chmixed-results.md').write_text('\n'.join(lines)+'\n')
