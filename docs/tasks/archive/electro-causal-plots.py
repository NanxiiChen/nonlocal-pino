"""Matched baseline/causal convergence, physical fields and causal activity."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out=Path('docs/tasks/archive');root=Path('/root/autodl-tmp/tf-logs/electropolish2d')
runs={'Baseline':root/'20261028-electro-operator-online-b8-adam-01',
      'Causal':root/'20261028-electro-operator-online-b8-causal-01'}
fig,axes=plt.subplots(2,2,figsize=(10.5,7),layout='constrained')
for label,run in runs.items():
    h=[json.loads(l) for l in (run/'metrics.jsonl').read_text().splitlines()]
    for row,metric in enumerate(('pooled','terminal')):
        for col,field in enumerate(('phi','c')):
            axes[row,col].semilogy([e['step'] for e in h],
                [100*e[f'test_{metric}_rel_l2_{field}'] for e in h],label=label)
            axes[row,col].set(xlabel='Training update',ylabel='Mean unseen relative L2 (%)',
                title=f'{field}: '+('pooled time' if row==0 else 'terminal t*=1'))
            axes[row,col].grid(alpha=.2);axes[row,col].legend()
fig.suptitle('Matched online B8 / 5000 Adam — physical error, 4 unseen inputs')
fig.savefig(out/'electro-causal-convergence.png',dpi=180);plt.close(fig)

run=runs['Causal'];h=[json.loads(l) for l in (run/'loss.jsonl').read_text().splitlines()]
steps=[e['step'] for e in h]
fig,axes=plt.subplots(1,3,figsize=(13,3.7),layout='constrained')
for i,name in enumerate(('AC','CH')):
    axes[0].semilogy(steps,[e['unweighted_parts'][i] for e in h],label=name+' raw')
    axes[0].semilogy(steps,[e['parts'][i] for e in h],linestyle='--',label=name+' causal')
    axes[1].semilogy(steps,[e['causal']['eps_used'][i] for e in h],label=name)
    axes[2].plot(steps,[np.mean(np.array(e['causal']['weights_'+name.lower()])[:,-1]) for e in h],label=name)
axes[0].set(title='Physical vs causal PDE loss',ylabel='Mean absolute residual')
axes[1].set(title='Adaptive eps',ylabel='eps (shared min used)')
axes[2].set(title='Last sorted time weight',ylabel='Mean weight across batch',ylim=(0,1.05))
for ax in axes:ax.set_xlabel('Training update');ax.legend(fontsize=8);ax.grid(alpha=.2)
fig.savefig(out/'electro-causal-weights.png',dpi=180);plt.close(fig)

with np.load(run/'prediction.npz') as f:a={k:f[k] for k in f.files}
index=int(np.flatnonzero(a['condition_indices']==16)[0])
p,r=a['prediction'][index,...,0],a['target'][index,...,0]
ids=[int(np.argmin(abs(a['time']-t))) for t in (0,.05,.25,.5,1.)]
error=abs(p-r);emax=error[ids].max()
fig,axes=plt.subplots(3,5,figsize=(13,5),layout='constrained')
for j,k in enumerate(ids):
    for row,values in enumerate((r[k],p[k],error[k])):
        im=axes[row,j].imshow(values,origin='lower',extent=(0,1,0,.5),vmin=0,vmax=1 if row<2 else emax,
            cmap='viridis' if row<2 else 'magma')
        axes[row,j].set_xticks([]);axes[row,j].set_yticks([])
        if j==0:axes[row,j].set_ylabel(('Reference','Prediction','Absolute error')[row])
        if row==0:axes[row,j].set_title(f't* = {a["time"][k]:.2f}')
fig.colorbar(im,ax=axes[2,:],orientation='horizontal',shrink=.65,pad=.025)
fig.suptitle('Causal online B8 — unseen interface16 / phase only')
fig.savefig(out/'electro-causal-fields.png',dpi=180);plt.close(fig)

audit=json.loads((out/'electro-causal-audit.json').read_text())
old=json.loads((out/'electro-online-balance-diagnosis.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(10.5,3.8),layout='constrained')
for label,records in [('Baseline',old['fields']),('Causal',audit['conditions'])]:
    record=next(e for e in records if e['index']==16)
    rough=[v['roughness_prediction'] for v in record['by_time']] if label=='Baseline' else record['roughness_prediction']
    axes[0].plot(a['time'],rough,label=label)
ref=next(e for e in audit['conditions'] if e['index']==16)
axes[0].plot(a['time'],ref['roughness_reference'],color='black',linestyle=':',label='Reference')
axes[0].set(xlabel='Normalized time',ylabel='Std. interface height',title='Unseen16: retained roughness')
early=json.loads((out/'electro-online-balance-early.json').read_text())
axes[1].semilogy(a['time'],[100*s['phi_predicted_spread_over_reference'] for s in early['spread']],label='Baseline')
axes[1].semilogy(a['time'],[100*s['phi'] for s in audit['spread']],label='Causal')
axes[1].axhline(100,color='black',linestyle=':')
axes[1].set(xlabel='Normalized time',ylabel='Predicted / reference input spread (%)',title='4 unseen inputs: retained differences')
for ax in axes:ax.legend();ax.grid(alpha=.2)
fig.savefig(out/'electro-causal-condition-response.png',dpi=180);plt.close(fig)
print('plots complete')
