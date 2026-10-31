"""Physics-only preview on existing meshes; not a validated reference bank."""
import json
import argparse
from ml_collections import ConfigDict
from pathlib import Path
import time
import numpy as np
from scipy import sparse
from examples.confined_ch2d.configs.train_operator_multiscale import get_config
from examples.confined_ch2d.configs.train_operator_richer import get_config as old_config
from examples.confined_ch2d.operator_problem import fixed_initial,case_config
from examples.confined_ch2d.native_reference import native
from examples.confined_ch2d.operator_plotting import SmoothMeshDisplay
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--request',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--receipt',type=Path,required=True)
args=parser.parse_args()
c=ConfigDict(json.loads(args.request.read_text()));old=old_config()
source=Path(old.operator.dataset_path);manifest=json.loads((source/'manifest.json').read_text())
output=args.output
output.mkdir(parents=True,exist_ok=False)
(output/'config.json').write_text(c.to_json(indent=2)+'\n')
records=[];arrays=[];start=time.perf_counter()
for index in (16,20,23):
    with np.load(source/manifest['conditions'][index]['file'],allow_pickle=False) as f:d={k:f[k] for k in f.files}
    x,v=d['points'],d['volumes'];nb,co=d['neighbors'],d['coefficients'];n=len(x)
    off=sparse.coo_matrix((co.ravel(),(np.repeat(np.arange(n),nb.shape[1]),nb.ravel())),shape=(n,n)).tocsr();off.eliminate_zeros()
    A=sparse.diags(np.asarray(off.sum(axis=1)).ravel())-off
    result=native(case_config(c,d['holes'],index),x,v,A,c.reference.dt,1e-8)
    phi=result['phi']
    def stats(values):return dict(pure_area_fraction=float(v@(abs(values)>=.8)/v.sum()),
        mixed_area_fraction=float(v@(abs(values)<.3)/v.sum()),minimum=float(values.min()),maximum=float(values.max()))
    record=dict(index=index,old_terminal=stats(d['phi'][-1]),new_terminal=stats(phi[-1]),
        initial=stats(phi[0]),seconds=result['seconds'],nodes=n)
    records.append(record);arrays.append(dict(d,phi=phi,time=np.linspace(0,c.physical.terminal_time,c.reference.snapshots)))
    np.savez_compressed(output/f'preview_{index:03d}.npz',phi=phi,time=arrays[-1]['time'],points=x,volumes=v,holes=d['holes'],mesh_points=d['mesh_points'],mesh_cells=d['mesh_cells'])
    print(json.dumps(record),flush=True)
fig,axes=plt.subplots(3,5,figsize=(14,8),layout='constrained')
bound=max(float(abs(d['phi']).max()) for d in arrays)
for row,d in enumerate(arrays):
    display=SmoothMeshDisplay(d['mesh_points'],d['mesh_cells'],d['volumes'],d['holes'],extent=(0.,1.,.07,.93),resolution=(400,320))
    values=display(d['phi'])
    for col,t in enumerate(d['time']):
        ax=axes[row,col];im=ax.imshow(values[col],origin='lower',extent=display.extent,cmap='RdBu_r',vmin=-bound,vmax=bound)
        for cx,cy,r in d['holes']:ax.add_patch(Circle((cx,cy),r,fill=False,edgecolor='black',linewidth=.6))
        ax.set(aspect='equal',xticks=[],yticks=[])
        if row==0:ax.set_title(f't = {t:g}')
        if col==0:ax.set_ylabel(f'Geometry {records[row]["index"]}')
fig.colorbar(im,ax=list(axes.ravel()),label='Phase phi',shrink=.8)
fig.suptitle(f'{len(c.initial.waves)} fixed ambient waves, T = {c.physical.terminal_time:g}: native preview (not refinement-validated)')
fig.savefig(output/'reference_preview.png',dpi=180);plt.close(fig)
d=arrays[0];display=SmoothMeshDisplay(d['mesh_points'],d['mesh_cells'],d['volumes'],d['holes'],extent=(0.,1.,.07,.93),resolution=(400,320))
values=np.stack((fixed_initial(d['points'],old,xp=np),fixed_initial(d['points'],c,xp=np)))
rendered=display(values);bound=float(abs(values).max());fig,axes=plt.subplots(1,2,figsize=(8.5,3.7),layout='constrained')
for ax,field,title in zip(axes,rendered,('Previous 8-mode initial field',f'New {len(c.initial.waves)}-mode initial field'),strict=True):
    im=ax.imshow(field,origin='lower',extent=display.extent,cmap='RdBu_r',vmin=-bound,vmax=bound)
    for cx,cy,r in d['holes']:ax.add_patch(Circle((cx,cy),r,fill=False,edgecolor='black',linewidth=.6))
    ax.set(title=title,aspect='equal',xticks=[],yticks=[])
fig.colorbar(im,ax=list(axes),label='Phase phi');fig.savefig(output/'initial_comparison.png',dpi=180);plt.close(fig)
report=dict(scope='Unrefined preview using already validated geometry/operator; new trajectories require subsequent independent refinement checks',config='examples/confined_ch2d/configs/train_operator_multiscale.py',records=records,seconds=time.perf_counter()-start,output=str(output))
args.receipt.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps(report,indent=2),flush=True)
