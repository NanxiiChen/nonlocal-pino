"""Scientific convergence, prescribed3Dshapes and physical phase snapshots."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from examples.icemelting3d.ellipsoids import rotation

parser=argparse.ArgumentParser();parser.add_argument('run');args=parser.parse_args()
run=Path(args.run);out=Path(__file__).resolve().parent
config=json.loads((run/'config.json').read_text())
manifest=json.loads((Path(config['operator']['dataset_path'])/'manifest.json').read_text())

def save(fig,name):
    fig.savefig(out/(name+'.png'),dpi=180,bbox_inches='tight');plt.close(fig)

history=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
monitor=[item for item in history if item['split']=='monitor']
fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
for key,label in (('pooled_phi','Pooled phase'),('terminal_phi','Terminal phase')):
    axes[0].semilogy([r['step'] for r in monitor],[100*r[key] for r in monitor],label=label)
for key,label in (('pooled_solid','Pooled solid indicator'),('terminal_solid','Terminal solid indicator')):
    values=[np.mean([p[key] for p in r['conditions']]) for r in monitor]
    axes[1].semilogy([r['step'] for r in monitor],100*np.asarray(values),label=label)
for ax in axes:
    ax.set(xlabel='Optimizer updates',ylabel='Relative L2 (%)');ax.grid(alpha=.25);ax.legend()
fig.suptitle('Two fixed monitor ellipsoids; physical fields / native graph reference')
save(fig,'melting-operator-convergence')

index=manifest['test_indices'][0]
with np.load(run/f'prediction_{index:03d}.npz') as stored:
    times=stored['time'];target=stored['slice_target'];pred=stored['slice_prediction'];points=stored['slice_points']
n=int(round(len(points)**.5));assert n*n==len(points)
err=abs(pred-target);vmax=float(err.max())
fig,axes=plt.subplots(3,len(times),figsize=(14,8),constrained_layout=True)
for row,(values,label) in enumerate(((target,'Reference'),(pred,'Prediction'),(err,'Absolute error'))):
    for col,t in enumerate(times):
        im=axes[row,col].imshow(values[col].reshape(n,n).T,origin='lower',extent=(0,1,0,1),
            cmap='magma' if row==2 else 'coolwarm',vmin=0 if row==2 else -1,vmax=vmax if row==2 else 1)
        axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
        if row==0:axes[row,col].set_title(f't*={t:g} / {config["nondim"]["T_c"]*t:g}s')
        if col==0:axes[row,col].set_ylabel(label)
    fig.colorbar(im,ax=axes[row,:],fraction=.017,pad=.01)
fig.suptitle(f'First predeclared test ellipsoid (index {index}), z={points[0,2]:.4f}; phi only')
save(fig,'melting-operator-fields')

fig=plt.figure(figsize=(14,7),layout='constrained')
longitude,latitude=np.meshgrid(np.linspace(0,2*np.pi,41),np.linspace(0,np.pi,21))
sphere=np.stack([np.cos(longitude)*np.sin(latitude),np.sin(longitude)*np.sin(latitude),np.cos(latitude)],-1)
for k,index in enumerate(manifest['test_indices']):
    p=np.asarray(manifest['parameters'][index]);surface=(sphere*p[3:6])@rotation(p[6:],np).T+p[:3]
    ax=fig.add_subplot(2,4,k+1,projection='3d')
    ax.plot_surface(surface[...,0],surface[...,1],surface[...,2],cmap='viridis',linewidth=0,alpha=.95)
    ax.set(xlim=(0,1),ylim=(0,1),zlim=(0,1),title=f'Test {k}; axes '+','.join(f'{a:.2f}' for a in p[3:6]))
    ax.set_box_aspect((1,1,1));ax.set_xticks([]);ax.set_yticks([]);ax.set_zticks([])
fig.suptitle('Eight predeclared held-out 3D initial ellipsoids (prescribed geometry)')
save(fig,'melting-operator-initial-shapes')
print('Three scientific figures exported')
