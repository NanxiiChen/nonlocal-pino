"""Plot the independent reference at five representative deformation times."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
OUT=ROOT/'reports/twophase-vortex-pilot'
with np.load('/root/autodl-tmp/data/twophase2d/results_periodic_vortex.npz') as d:
    phi=d['phi_local']; points=d['points']; times=d['time']
X,Y=points.T.reshape(2,64,64)
ids=[0,5,10,15,20]
fig,axes=plt.subplots(1,5,figsize=(15,3.4),layout='constrained')
for ax,i in zip(axes,ids,strict=True):
    image=ax.pcolormesh(X,Y,phi[i],shading='nearest',cmap='viridis',vmin=min(-1,float(phi.min())),vmax=max(1,float(phi.max())))
    ax.contour(X,Y,phi[i],levels=[0],colors='white',linewidths=1)
    ax.set(title=f't = {times[i]:.2f}',xlabel='x',aspect='equal',xlim=(0,1),ylim=(0,1))
axes[0].set_ylabel('y')
fig.colorbar(image,ax=axes,label='Reference phi (no clipping)',shrink=.85)
fig.suptitle('Periodic reversed vortex: stretch, reverse, return; independent local CH reference')
fig.savefig(OUT/'reference-evolution.png',dpi=180)
metrics=dict(local_return_L2_to_initial=float(np.linalg.norm(phi[-1]-phi[0])/np.linalg.norm(phi[0])),
    local_midpoint_L2_to_initial=float(np.linalg.norm(phi[10]-phi[0])/np.linalg.norm(phi[0])),
    times=times[ids].tolist(),binary_droplet_axis_ratios=[])
for i in ids:
    xy=points[(phi[i]<0).ravel()]
    eig=np.linalg.eigvalsh(np.cov(xy.T))
    metrics['binary_droplet_axis_ratios'].append(float(np.sqrt(eig[1]/eig[0])))
(OUT/'reference-geometry.json').write_text(json.dumps(metrics,indent=2,allow_nan=False)+'\n')
print(json.dumps(metrics,indent=2))
