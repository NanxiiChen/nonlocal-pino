"""Compare final physical fields against one common local reference, no clipping."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
OUT=ROOT/'reports/twophase-vortex-pilot'
manifest=json.loads((OUT/'manifest.json').read_text())
if manifest['status']!='completed':raise ValueError('Incomplete cohort')
with np.load('/root/autodl-tmp/data/twophase2d/results_periodic_vortex.npz') as d:
    ref=d['phi_local'];points=d['points'];times=d['time']
fields=[ref];labels=['Reference (local CH)']
for r in manifest['runs']:
    with np.load(Path(r['run'])/'prediction.npz') as d:
        np.testing.assert_array_equal(d['points'],points)
        np.testing.assert_array_equal(d['time'],times)
        fields.append(d['phi'])
    labels.append('PINN '+r['method'])
X,Y=points.T.reshape(2,64,64)
ids=[0,5,10,15,20]
vmin=min(-1,min(float(a.min()) for a in fields));vmax=max(1,max(float(a.max()) for a in fields))
fig,axes=plt.subplots(3,5,figsize=(14,8),layout='constrained')
for row,(phi,label) in enumerate(zip(fields,labels,strict=True)):
    for col,i in enumerate(ids):
        ax=axes[row,col]
        im=ax.pcolormesh(X,Y,phi[i],shading='nearest',cmap='viridis',vmin=vmin,vmax=vmax)
        ax.contour(X,Y,phi[i],levels=[0],colors='white',linewidths=.7)
        if row:
            ax.contour(X,Y,ref[i],levels=[0],colors='orangered',linewidths=.7,linestyles='dashed')
            error=np.linalg.norm(phi[i]-ref[i])/np.linalg.norm(ref[i])
            ax.text(.02,.02,f'Local-target L2={100*error:.2f}%',transform=ax.transAxes,fontsize=8,
                bbox=dict(facecolor='white',alpha=.9,edgecolor='none'))
        ax.set(aspect='equal',xlim=(0,1),ylim=(0,1))
        if row==0:ax.set_title(f't = {times[i]:.2f}')
        if row==2:ax.set_xlabel('x')
        if col==0:ax.set_ylabel(label+'\ny')
fig.colorbar(im,ax=axes.ravel().tolist(),label='phi (full range; no clipping)',shrink=.8)
fig.suptitle('Reversed vortex: final5000 updates; white=field interface, dashed red=common local reference')
fig.savefig(OUT/'common-reference-fields.png',dpi=180)
