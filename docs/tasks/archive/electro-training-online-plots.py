"""Same-stream online local-input ablation convergence and physical fields."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out=Path('docs/tasks/archive');root=Path('/root/autodl-tmp/tf-logs/electropolish2d')
base=root/'20261028-electro-operator-online-b8-adam-01'
run=root/'20261028-electro-operator-online-global-adam-01'
fig,axes=plt.subplots(1,2,figsize=(10.5,3.8),layout='constrained')
for label,path in [('Image + query-local IC',base),('Image only',run)]:
    h=[json.loads(line) for line in (path/'metrics.jsonl').read_text().splitlines()]
    for ax,key,title in zip(axes,('test_pooled_rel_l2_phi','test_terminal_rel_l2_phi'),('Pooled time','Terminal t*=1')):
        ax.semilogy([e['step'] for e in h],[100*e[key] for e in h],label=label)
        ax.set(xlabel='Training update',ylabel='Mean unseen phase relative L2 (%)',title=title)
        ax.legend(fontsize=8);ax.grid(alpha=.2)
fig.suptitle('Online B8 / same40000profiles / uniform time / 5000 Adam')
fig.savefig(out/'electro-training-online-convergence.png',dpi=180);plt.close(fig)

with np.load(run/'prediction.npz') as f:a={k:f[k] for k in f.files}
k=int(np.flatnonzero(a['condition_indices']==16)[0])
p,r=a['prediction'][k,...,0],a['target'][k,...,0]
ids=[int(np.argmin(abs(a['time']-t))) for t in (0,.05,.25,.5,1.)]
error=abs(p-r);emax=float(error[ids].max())
fig,axes=plt.subplots(3,5,figsize=(13,5),layout='constrained')
for col,k in enumerate(ids):
    for row,values in enumerate((r[k],p[k],error[k])):
        im=axes[row,col].imshow(values,origin='lower',extent=(0,1,0,.5),vmin=0,
            vmax=1 if row<2 else emax,cmap='viridis' if row<2 else 'magma')
        axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
        if col==0:axes[row,col].set_ylabel(('Reference','Prediction','Absolute error')[row])
        if row==0:axes[row,col].set_title(f't* = {a["time"][k]:.2f}')
fig.colorbar(im,ax=axes[2,:],orientation='horizontal',shrink=.65,pad=.025)
fig.suptitle('Image-only online model — unseen interface16 / phase only')
fig.savefig(out/'electro-training-online-fields.png',dpi=180);plt.close(fig)

audit=json.loads((out/'electro-training-online-audit.json').read_text())
old=json.loads((out/'electro-online-balance-diagnosis.json').read_text())
early=json.loads((out/'electro-online-balance-early.json').read_text())
new16=next(r for r in audit['conditions'] if r['index']==16)
old16=next(r for r in old['fields'] if r['index']==16)
fig,axes=plt.subplots(1,2,figsize=(10.5,3.8),layout='constrained')
axes[0].plot(a['time'],[r['roughness_prediction'] for r in old16['by_time']],label='Image + local IC')
axes[0].plot(a['time'],new16['roughness_prediction'],label='Image only')
axes[0].plot(a['time'],new16['roughness_reference'],color='black',linestyle=':',label='Reference')
axes[0].set(xlabel='Normalized time',ylabel='Std. interface height',title='Unseen16: retained roughness')
axes[1].semilogy(a['time'],[100*s['phi_predicted_spread_over_reference'] for s in early['spread']],label='Image + local IC')
axes[1].semilogy(a['time'],[100*s['phi'] for s in audit['spread']],label='Image only')
axes[1].axhline(100,color='black',linestyle=':')
axes[1].set(xlabel='Normalized time',ylabel='Predicted / reference input spread (%)',title='Four unseen inputs: retained differences')
for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.2)
fig.savefig(out/'electro-training-online-response.png',dpi=180);plt.close(fig)
print('plots complete')
