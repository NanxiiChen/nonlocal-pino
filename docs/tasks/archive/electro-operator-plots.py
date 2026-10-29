"""Scientific field and convergence figures from direct saved pilot artifacts."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out=Path('docs/tasks/archive');root=Path('/root/autodl-tmp/tf-logs/electropolish2d')
runs={kind:root/f'20261028-electro-operator-{kind}-01' for kind in ('cross','film')}
fig,axs=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
for kind,run in runs.items():
    history=[json.loads(s) for s in (run/'metrics.jsonl').read_text().splitlines()]
    steps=[m['step'] for m in history]
    for ax,field in zip(axs,('phi','c')):
        ax.semilogy(steps,[100*m['test_pooled_rel_l2_'+field] for m in history],label=kind+' pooled')
        ax.semilogy(steps,[100*m['test_terminal_rel_l2_'+field] for m in history],ls='--',label=kind+' terminal')
        ax.set(xlabel='Training update',ylabel='Mean held-out relative L2 (%)',title=field)
        ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.savefig(out/'electro-operator-convergence.png',dpi=180);plt.close(fig)
for kind,run in runs.items():
    with np.load(run/'prediction.npz') as f:
        pred,target=f['prediction'],f['target'];times=f['time'];indices=f['condition_indices']
    k=2 # first held-out interface; previous two are training-monitor conditions
    selected=[int(np.argmin(abs(times-t))) for t in (0,.25,.5,.75,1.)]
    fig,axes=plt.subplots(3,5,figsize=(13,5),layout='constrained')
    error=abs(pred[k,...,0]-target[k,...,0]);emax=max(float(error[selected].max()),1e-12)
    for j,i in enumerate(selected):
        for row,field in enumerate((target[k,i,...,0],pred[k,i,...,0],error[i])):
            im=axes[row,j].imshow(field,origin='lower',extent=(0,1,0,.5),cmap='viridis' if row<2 else 'magma',vmin=0,vmax=1 if row<2 else emax)
            axes[row,j].set_xticks([]);axes[row,j].set_yticks([])
            if j==0:axes[row,j].set_ylabel(('Reference','Prediction','Absolute error')[row])
            if row==0:axes[row,j].set_title(f't* = {times[i]:.2f}')
        if j==4:
            fig.colorbar(im,ax=axes[2,:],shrink=.65,pad=.025,orientation='horizontal')
    fig.suptitle(f'{kind}: unseen interface {indices[k]}, phase only')
    fig.savefig(out/f'electro-operator-{kind}-fields.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(3,4,figsize=(11,4.5),layout='constrained')
    emax=float(np.abs(pred[2:,-1,...,0]-target[2:,-1,...,0]).max())
    for j,k in enumerate(range(2,len(indices))):
        for row,field in enumerate((target[k,-1,...,0],pred[k,-1,...,0],abs(pred[k,-1,...,0]-target[k,-1,...,0]))):
            im=axes[row,j].imshow(field,origin='lower',extent=(0,1,0,.5),cmap='viridis' if row<2 else 'magma',vmin=0,vmax=1 if row<2 else emax)
            axes[row,j].set_xticks([]);axes[row,j].set_yticks([])
            if j==0:axes[row,j].set_ylabel(('Reference','Prediction','Absolute error')[row])
            if row==0:axes[row,j].set_title(f'Unseen interface {indices[k]}')
    fig.colorbar(im,ax=axes[2,:],shrink=.65,pad=.025,orientation='horizontal')
    fig.suptitle(kind+': all held-out interfaces at t*=1 (5s)')
    fig.savefig(out/f'electro-operator-{kind}-terminal.png',dpi=180);plt.close(fig)
with np.load('/root/autodl-tmp/data/electropolish2d/operator_random16_v1/conditions.npz') as f:initial=f['initial']
fig,axes=plt.subplots(4,5,figsize=(12,5),layout='constrained')
for i,ax in enumerate(axes.flat):
    ax.imshow(initial[i,:,0].reshape(51,101),origin='lower',extent=(0,1,0,.5),vmin=0,vmax=1,cmap='viridis')
    ax.set_xticks([]);ax.set_yticks([]);ax.set_title(('Train ' if i<16 else 'Unseen ')+str(i),fontsize=9,color='black' if i<16 else 'tab:red')
fig.savefig(out/'electro-operator-initial-family.png',dpi=180);plt.close(fig)
print('plots complete')
