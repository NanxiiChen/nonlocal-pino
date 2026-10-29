"""Full-horizon curves and field comparison for completed phase-only controls."""
from pathlib import Path
import sys,json,shutil
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file());sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ml_collections import ConfigDict
from examples.confined_ch2d.problem import load_reference
from examples.confined_ch2d.evaluate import field_plot
OUT=Path(__file__).parent;report=json.loads((OUT/'chtanh-analysis.json').read_text())
names=['20261028-chtanh-control-m8-01','20261028-chtanh-map-m8-01','20261028-chtanh-jac-m8-01','20261028-chtanh-jac-m5-01','20261028-chglobal-average-01','20261028-chtanh-average-m5-01']
labels=['Strong / modes8','Tanh / original r / modes8','Tanh / r/J / modes8','Tanh / r/J / modes5','Integral R/t / modes8 (historical)','Integral R/t / modes5']
fig,axes=plt.subplots(1,3,figsize=(15,4.2),layout='constrained')
keys=['pooled_rel_l2_phi_local','terminal_rel_l2_phi_local','pooled_rel_l2_mu_nonlocal']
for name,label in zip(names,labels,strict=True):
    row=report['runs'][name];run=Path(row['run']);history=[json.loads(s) for s in (run/'metrics.jsonl').read_text().splitlines()]
    for ax,key in zip(axes,keys,strict=True):ax.semilogy([x['step'] for x in history],[100*x[key] for x in history],label=label)
for ax,title in zip(axes,['Phase pooled / common FEM','Phase terminal / common FEM','Chemical pooled / native'],strict=True):
    ax.set(title=title,xlabel='Completed updates',ylabel='Relative L2 (%)');ax.grid(alpha=.25);ax.axvline(5000,color='grey',ls=':',lw=.8)
axes[0].legend(fontsize=7);fig.suptitle('Phase-only / eliminated chemical potential: 5000 SOAP + 500 L-BFGS, seed42')
fig.savefig(OUT/'chtanh-convergence.png',dpi=180);plt.close(fig)
run=Path(report['runs'][names[-1]]['run']);c=ConfigDict(json.loads((run/'config.json').read_text()));ref=load_reference(c)
with np.load(run/'cpu_audit_fields.npz') as f:phi=f['phi']
field_plot(ref,[ref['phi_local'],phi,np.abs(phi-ref['phi_local'])],OUT/'chtanh-integral-m5-fields.png',['Common FEM reference','Integral R/t, modes5','Absolute error'])
fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
for name,label in zip(names,labels,strict=True):ax.plot(ref['time'],report['runs'][name]['energy'],label=label)
phi=ref['phi_nonlocal']
from examples.confined_ch2d.native_reference import native_matrix
A=native_matrix(ref['points'],ref['volumes'],c);v=ref['volumes']
energy=np.sum(v*(.25*c.physical.beta*(phi**2-1)**2+.5*c.physical.epsilon**2*phi*(A@phi.T).T),axis=1)
ax.plot(ref['time'],energy,'k--',label='Native reference');ax.set(xlabel='Time',ylabel='Native discrete energy');ax.legend(fontsize=7);ax.grid(alpha=.25)
fig.savefig(OUT/'chtanh-energy.png',dpi=180);plt.close(fig)
lines=['# Phase-only tanh/Jacobian and frequency controls','', 'Five new complete cold starts, each5000SOAP+500L-BFGS, seed42, same T1.5 physics/reference/FP64/ordinary. The retained integral modes8 row is historical, not a fresh paired rerun. Every table entry is final, not best-checkpoint selection. Strong and integral losses are different recipes; do not attribute their cross-row gap to tanh alone.','', '| Model | Parameters | Phase FEM pooled % | Phase FEM terminal % | Mu native pooled % | Mu native terminal % | Trainer s | Energy max rise |','|---|---:|---:|---:|---:|---:|---:|---:|']
for name,label in zip(names,labels,strict=True):
    r=report['runs'][name];s=r['summary'];m=s['final_metrics']
    lines.append(f"| {label} | {r['parameter_count']} | {100*m[keys[0]]:.4f} | {100*m[keys[1]]:.4f} | {100*m[keys[2]]:.4f} | {100*m['terminal_rel_l2_mu_nonlocal']:.4f} | {s['wall_seconds']:.2f} | {r['energy_max_increase']:.6g} |")
lines+=['','Map amplitude1.1 is an explicit10% well-scale buffer, not a fitted trajectory; strict1.0 is a supported entry with measured range-only lower-bound phase error.3358% pooled/.4573% terminal versus FEM. Implicit scalar mass correction and cancellation-free tanh/Jacobian derivatives are independently validated. Mu remains exactly eliminated.','', 'Tanh alone gives no convincing phase improvement; r/J before the existing spatial metric worsens the strong recipe. Lowering modes5 helps modestly in that Jacobian recipe and improves the retained integral recipe in this run. No multi-seed ranking or universal tanh/CH failure claim. Lowest pooled phase now17.9854%, terminal29.3681%, but native mu129.29%/836.60%, phase[-1.34285,1.24822] and energy rise.007078 remain unacceptable. No SOTA/default promotion.','', 'Fourier features use pi*k*z, not2*pi*k*x. Modes8 minimum wavelengths.125/.10, about11.14/8.91 median nearest-neighbor spacings; modes5 .20/.16. Initial basic coordinate waves reach5. Spatial MLP uses no temporal Fourier embedding; time is rank24 log-Chebyshev. Measured native A² gain of mode8 relative5 is2.46--3.30 for the four axis sine/cosine probes. This is not a Nyquist theorem for irregular grids or a guarantee that nonlinear MLPs generate no higher harmonics.','', 'All final/best CPU replays and exact chemical/mass audits pass. Trained phase time derivatives independently match NumPy Chebyshev/Brent chains (max2.3e-12). Both new mathematical tests pass. The transformed constant mode is only.052%/.020% of final Jacobian-control loss; it does not explain all degradation. Dense midpoint objectives differ by less than1%; no gross time-node aliasing appears in this check. No Jacobian underflow (min~.043) or hidden floor.','', 'Canonical evaluate.py replay/physical boundary-normal diagnostic also passes for the mapped model; native wall normals remain local-limit diagnostics. Source/initial parity is in chtanh-provenance.json; map controls have identical51-source snapshots. The first unmapped control differs only in unused map/diagnostic helper updates; its active network/training/operator/optimizer code agrees. All3 modes8 initial parameter files are identical. Modes5 changes feature dimension/initialization, so this is not parameter matched.','', 'Exact follow-up is in the active task. No Git operation, reference/geometry change, supervised initialization or frozen time windows.']
(OUT/'chtanh-results.md').write_text('\n'.join(lines)+'\n')
for name in names:
    if name.startswith('20261028-chtanh-'):
        path=Path('/tmp/'+name+'.log')
        if path.is_file():shutil.copyfile(path,OUT/(name+'.txt'))
