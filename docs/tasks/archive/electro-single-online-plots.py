"""Condition-specific optimizer curves and unseen continuous-online fields."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out=Path('docs/tasks/archive');root=Path('/root/autodl-tmp/tf-logs/electropolish2d')
runs={'16-input Adam':root/'20261028-electro-operator-cross-01',
      '1-input Adam':root/'20261028-electro-operator-single-cross-01',
      '1-input SOAP':root/'20261028-electro-operator-single-soap-01'}
fig,axes=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
for label,run in runs.items():
    entries=[json.loads(l) for l in (run/'metrics.jsonl').read_text().splitlines()]
    for ax,key in zip(axes,('pooled_rel_l2_phi','terminal_rel_l2_phi')):
        ax.semilogy([e['step'] for e in entries],
            [100*next(m for m in e['conditions'] if m['condition_index']==0)[key] for e in entries],label=label)
        ax.set(xlabel='Training update',ylabel='Same-interface phase relative L2 (%)')
        ax.set_title('Pooled time' if key.startswith('pooled') else 'Terminal t*=1')
        ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.savefig(out/'electro-single-convergence.png',dpi=180);plt.close(fig)
run=root/'20261028-electro-operator-online-b8-adam-01'
history=[json.loads(l) for l in (run/'metrics.jsonl').read_text().splitlines()]
fig,axes=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
for ax,field in zip(axes,('phi','c')):
    for prefix,label in (('pooled_rel_l2_','Pooled'),('terminal_rel_l2_','Terminal')):
        ax.semilogy([e['step'] for e in history],[100*e['test_'+prefix+field] for e in history],label=label)
    ax.set(xlabel='Training update',ylabel='Mean held-out relative L2 (%)',title=field)
    ax.grid(alpha=.2);ax.legend()
fig.savefig(out/'electro-online-convergence.png',dpi=180);plt.close(fig)
for suffix,k,label in [('single-cross-01',0,'Fixed interface / Adam'),('single-soap-01',0,'Fixed interface / SOAP'),
                        ('online-b8-adam-01',2,'Online B8 / unseen interface16')]:
    with np.load(root/('20261028-electro-operator-'+suffix)/'prediction.npz') as f:
        p,t,ts=f['prediction'][k],f['target'][k],f['time']
    ids=[int(np.argmin(abs(ts-v))) for v in (0,.25,.5,.75,1)]
    error=abs(p[...,0]-t[...,0]);emax=float(error[ids].max())
    fig,axes=plt.subplots(3,5,figsize=(13,5),layout='constrained')
    for j,i in enumerate(ids):
        for row,values in enumerate((t[i,...,0],p[i,...,0],error[i])):
            im=axes[row,j].imshow(values,origin='lower',extent=(0,1,0,.5),vmin=0,vmax=1 if row<2 else emax,
                                 cmap='viridis' if row<2 else 'magma')
            axes[row,j].set_xticks([]);axes[row,j].set_yticks([])
            if j==0:axes[row,j].set_ylabel(('Reference','Prediction','Absolute error')[row])
            if row==0:axes[row,j].set_title(f't* = {ts[i]:.2f}')
    fig.colorbar(im,ax=axes[2,:],orientation='horizontal',shrink=.65,pad=.025)
    fig.suptitle(label+' — phase only')
    fig.savefig(out/('electro-'+suffix+'-fields.png'),dpi=180);plt.close(fig)
# Display randomly drawn profiles from four actual optimizer updates (all sixteen panels differ).
coeff=np.load(root/'20261028-electro-operator-online-b8-adam-01/online_coefficients.npy')
x=np.linspace(0,1,500);fig,axes=plt.subplots(4,4,figsize=(11,6),layout='constrained')
for row,step in enumerate((0,999,2999,4999)):
    for col in range(4):
        a,f,phase=coeff[step,col].T;angle=2*np.pi*x[:,None]*f+phase
        y=.33+np.sum(a*(np.sin(angle)+np.cos(angle+.3)),axis=-1)
        ax=axes[row,col];ax.plot(x,y);ax.set_ylim(.1,.5);ax.set_title(f'Update {step+1}, item {col}',fontsize=9)
        ax.grid(alpha=.2)
fig.savefig(out/'electro-online-profile-stream.png',dpi=180);plt.close(fig)
print('plots complete')
