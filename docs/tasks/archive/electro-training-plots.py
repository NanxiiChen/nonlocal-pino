"""Same-input fitting, condition-path sensitivity and initial-distance plots."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out=Path('docs/tasks/archive');root=Path('/root/autodl-tmp/tf-logs/electropolish2d')
runs={'Single input':root/'20261028-electro-operator-single-cross-01',
      'Two inputs':root/'20261028-electro-operator-pair-adam-01',
      'Two inputs: image only':root/'20261028-electro-operator-pair-global-adam-01',
      '16 inputs':root/'20261028-electro-operator-cross-01'}
fig,axes=plt.subplots(1,2,figsize=(10.5,3.8),layout='constrained')
for name,run in runs.items():
    entries=[json.loads(l) for l in (run/'metrics.jsonl').read_text().splitlines()]
    for ax,metric in zip(axes,('pooled_rel_l2_phi','terminal_rel_l2_phi')):
        ax.semilogy([e['step'] for e in entries],[100*next(v for v in e['conditions']
            if v['condition_index']==0)[metric] for e in entries],label=name)
        ax.set(xlabel='Training update',ylabel='Same input0 relative L2 (%)',
            title='Pooled time' if metric.startswith('pooled') else 'Terminal t*=1')
        ax.grid(alpha=.2);ax.legend()
fig.suptitle('Uniform time / same initialization / 2000 Adam — batch/exposure differ')
fig.savefig(out/'electro-training-input-count.png',dpi=180);plt.close(fig)

r=json.loads((out/'electro-training-conditioning.json').read_text())
probe=next(v for v in r['swaps'] if v['model']=='online_final' and v['interfaces']==[16,17])
fig,axes=plt.subplots(1,2,figsize=(10.5,3.8),layout='constrained')
time=[v['time'] for v in probe['by_time']]
for key,label in [('image_swap','Swap image only'),('local_swap','Swap query-local IC only'),('joint_change','Swap both')]:
    axes[0].semilogy(time,[100*v['phi'][key] for v in probe['by_time']],label=label)
axes[0].axhline(100,color='black',linestyle=':')
axes[0].set(xlabel='Normalized time',ylabel='Change / true input difference (%)',title='Online model: input16 vs17')
axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
x=np.arange(3)
for shift,probe in [(-.18,r['gradient_probes'][0]),(.18,r['gradient_probes'][1])]:
    axes[1].bar(x+shift,[100*probe['equations'][name]['retention'] for name in ('AC','CH','IC')],.36,label=probe['times_kind'])
axes[1].set(xticks=x,xticklabels=('AC','CH','IC'),ylabel='Norm(mean grad) / mean(norm grad) (%)',
    title='Online model: across8initial conditions',ylim=(0,105))
axes[1].legend();axes[1].grid(axis='y',alpha=.2)
fig.suptitle('Frozen-checkpoint diagnosis — mixed-input swaps are sensitivity probes')
fig.savefig(out/'electro-training-conditioning.png',dpi=180);plt.close(fig)

with np.load(out/'electro-training-distance.npz') as f:a={k:f[k] for k in f.files}
fig,axes=plt.subplots(1,3,figsize=(12,3.7),layout='constrained')
xx,yy=np.meshgrid(np.unique(a['points'][:,0]),np.unique(a['points'][:,1]))
for ax,key,title in zip(axes,('phi_window','phi_global','phi_vertical'),
                        ('Existing33segment search','Global1999segment distance','Vertical gap only')):
    ax.contourf(xx,yy,a[key].reshape(xx.shape),levels=np.linspace(0,1,21),vmin=0,vmax=1,cmap='viridis')
    ax.plot(a['curve_x'],a['curve_y'],color='red',linewidth=.8)
    ax.contour(xx,yy,a[key].reshape(xx.shape),levels=[.1,.9],colors='white',linewidths=.6)
    ax.set(xlabel='x*',ylabel='y*',title=title,ylim=(.2,.43))
fig.suptitle('Initial interface16: same sign; different distance magnitudes')
fig.savefig(out/'electro-training-distance.png',dpi=180);plt.close(fig)
print('plots complete')
