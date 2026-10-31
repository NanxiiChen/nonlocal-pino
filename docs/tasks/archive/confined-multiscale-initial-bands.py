"""Analytic prescribed IC split into low/mid/high bands; no network/solver."""
import argparse,json
from pathlib import Path
import numpy as np
from ml_collections import ConfigDict
from examples.confined_ch2d.operator_problem import fixed_initial
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
c=ConfigDict(json.loads((a.dataset/'config.json').read_text()));m=json.loads((a.dataset/'manifest.json').read_text())
i=m['test_indices'][0];record=m['conditions'][i];holes=np.asarray(record['holes'])
x,y=np.meshgrid(np.linspace(0,1,500),np.linspace(.07,.93,400));points=np.column_stack((x.ravel(),y.ravel()))
mask=np.sum(((points-np.asarray(c.geometry.center))/np.asarray(c.geometry.axes))**2,axis=1)>=1
for cx,cy,r in holes:mask|=np.sum((points-[cx,cy])**2,axis=1)<=r*r
wave=np.asarray(c.initial.waves);band=np.max(abs(wave[:,:2]),axis=1);components=[]
for k in (1,2,3):
 cc=ConfigDict(c.to_dict());cc.initial.waves=tuple(tuple(row) for row in wave[band==k])
 components.append(fixed_initial(points,cc,xp=np))
full=fixed_initial(points,c,xp=np);np.testing.assert_allclose(np.sum(components,axis=0),full,atol=2e-16,rtol=0)
values=np.stack((full,*components));bound=float(abs(values[:,~mask]).max());fig,axes=plt.subplots(1,4,figsize=(15,3.7),layout='constrained')
for ax,field,title in zip(axes,values,('Full 24-mode initial field','Large scale: 4 modes','Middle scale: 8 modes','Small scale: 12 modes'),strict=True):
 im=ax.imshow(np.ma.array(field.reshape(x.shape),mask=mask.reshape(x.shape)),origin='lower',extent=(0,1,.07,.93),cmap='RdBu_r',vmin=-bound,vmax=bound)
 for cx,cy,r in holes:ax.add_patch(Circle((cx,cy),r,fill=False,edgecolor='black',linewidth=.6))
 ax.set(title=title,aspect='equal',xticks=[],yticks=[])
fig.colorbar(im,ax=list(axes),shrink=.8,label='Phase contribution (shared scale)')
fig.suptitle('Same analytic ambient function for all geometries; exact band sum, no evolved labels')
fig.savefig(a.output/'initial_bands.png',dpi=200);fig.savefig(a.output/'initial_bands.pdf');plt.close(fig)
report=dict(dataset=str(a.dataset),geometry_index=i,modes_per_band=[int(np.sum(band==k)) for k in (1,2,3)],
 display='Exact analytic IC on a dense visualization grid, not finer solver quadrature',
 maximum_band_sum_error=float(abs(np.sum(components,axis=0)-full).max()),
 fluid_rms_per_panel=[float(np.sqrt(np.mean(field[~mask]**2))) for field in values])
(a.output/'initial_bands.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report),flush=True)
