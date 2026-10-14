"""Read-only CPU TensorBoard scalars and physical fields from saved load states."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from ml_collections import ConfigDict
import numpy as np
from tensorboardX import SummaryWriter

from pinn.utils import log_config_text


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def endpoint_fields(path,row,*,nodes,dofs):
    """Endpoint figures must use synchronized q/d at the reported physical load."""
    if row['q_d_synchronized'] is not True:
        raise ValueError('TensorBoard endpoint is not mechanics/damage synchronized')
    if sha(path)!=row['fields_sha256']:raise ValueError('TensorBoard endpoint field hash differs')
    with np.load(path,allow_pickle=False) as data:
        fields={k:data[k].copy() for k in ('q','d','H')}
    for key,shape in [('q',(dofs,)),('d',(nodes,)),('H',(nodes,))]:
        v=fields[key]
        if v.shape!=shape or v.dtype!=np.float64 or not np.isfinite(v).all():
            raise ValueError(f'invalid monitored field {key}')
    return fields


def reference_row(curve,load):
    matching=[r for r in curve if abs(r['load']-load)<1e-14]
    if len(matching)!=1:raise ValueError(f'no unique reference at monitored load{load}')
    return matching[0]


def scalar_values(row):
    return {'loss/damage':row['full_loss'],'diag/inner_AT2':row['inner_AT2'],
        'diag/damage_change_rms':row['damage_rms'],
        'diag/displacement_change_rms_over_load':row['displacement_rms_over_load'],
        'diag/damage_min':row['damage_min'],'diag/damage_max':row['damage_max'],
        'train/load':row['load'],'train/outer_iteration':row['sweep'],
        'train/optimizer_count':row['optimizer_count'],'train/learning_rate':row['learning_rate'],
        'diag/mechanics_free_residual':row['mechanics']['free_residual_norm'],
        'diag/stage_reaction_for_previous_damage':row['reaction']}


def publish_endpoint(writer,root,out,tri,xy,row,curve,refroot,refcurve,*,dpi,dofs,volumes):
    folder=root/f"state_{row['index']:04d}"
    fields=endpoint_fields(folder/'fields.npz',row,nodes=len(xy),dofs=dofs)
    ref=reference_row(refcurve,row['load'])
    path=refroot/f"state_{ref['index']:04d}"/'fields.npz'
    with np.load(path,allow_pickle=False) as data:reference=data['d'].copy()
    if reference.shape!=fields['d'].shape:raise ValueError('reference grid differs from monitored material grid')
    rel=float(np.sqrt(volumes@(fields['d']-reference)**2/(volumes@reference**2)))
    step=row['index'];load=row['load'];out.mkdir(exist_ok=True)
    maps=[('damage',fields['d'],'inferno'),
          ('ux',fields['q'].reshape(-1,2)[:len(xy),0],'coolwarm'),
          ('uy',fields['q'].reshape(-1,2)[:len(xy),1],'coolwarm'),
          ('history',fields['H'],'viridis')]
    for name,values,cmap in maps:
        fig,axs=plt.subplots(1,2 if name=='damage' else 1,
            figsize=(10.2,4.8) if name=='damage' else (6.2,5.2),dpi=dpi,squeeze=False)
        axes=axs.ravel()
        minimum,maximum=(0.,1.) if name=='damage' else (float(values.min()),float(values.max()))
        datasets=[('Reference',reference),('PINN',values)] if name=='damage' else [('PINN',values)]
        for ax,(label,v) in zip(axes,datasets):
            image=ax.tripcolor(tri,v,shading='gouraud',cmap=cmap,vmin=minimum,vmax=maximum,rasterized=True)
            ax.set_aspect('equal');ax.set_xlim(0,1);ax.set_ylim(0,1)
            ax.set(xlabel='x',ylabel='y',title=f'{label}: {name}, U={load:.4f}')
            fig.colorbar(image,ax=ax,label=name)
        fig.tight_layout()
        fig.savefig(out/f'{name}_load_{step:04d}.png',dpi=dpi)
        writer.add_figure(f'train/{name}_field',fig,global_step=step,close=True)
    fig,axes=plt.subplots(1,2,figsize=(10.5,4),dpi=dpi)
    rows=[r for r in curve if r['index']<=step]
    refs=[r for r in refcurve if r['load']<=load+1e-14]
    for ax in axes:
        ax.plot([0]+[r['load'] for r in refs],[0]+[r['reaction'] for r in refs],'-',label='Direct nonlocal damage reference',color='#1f77b4')
        ax.plot([0]+[r['load'] for r in rows],[0]+[r['reaction'] for r in rows],'.-',label='PINN completed loads',color='#d62728',markersize=3)
        ax.set(xlabel='Applied displacement',ylabel='Reaction');ax.grid(alpha=.25);ax.legend(fontsize=8)
    axes[0].set_title(f'Loading through{load:.4f}')
    axes[1].set(xlim=(.0049,max(.0053,load)),title='Critical-load region')
    fig.tight_layout();fig.savefig(out/f'reaction_load_{step:04d}.png',dpi=dpi)
    writer.add_figure('train/load_reaction_curve',fig,global_step=step,close=True)
    for tag,value in {'train/reaction':row['reaction'],'train/max_damage':row['damage_max'],
        'diag/physical_AT2':row['physical_AT2'],'diag/field_stable':int(row['field_stable']),
        'diag/certified_path':int(row['certified_path']),'eval/direct_nonlocal/rel_L2_d':rel,
        'eval/direct_nonlocal/reaction':ref['reaction']}.items():
        writer.add_scalar(tag,value,step)
    writer.flush()
    return dict(index=step,load=load,fields_sha256=row['fields_sha256'],reference_load=ref['load'],
        reference_fields_sha256=sha(path),relative_damage_L2=rel)


def main(a,config_path):
    if (any(type(v) is not int or v<1 for v in (a.scalar_every,a.flush_seconds,a.figure_dpi,a.cpu_threads))
            or not np.isfinite(a.poll_seconds) or a.poll_seconds<=0 or a.log_source_endpoint is not True):
        raise ValueError('invalid explicit TensorBoard monitor settings')
    if not Path(a.event_dir).resolve().is_relative_to(Path(a.log_root).resolve()):
        raise ValueError('TensorBoard event directory must be inside the explicit log root')
    allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,allowed[-a.cpu_threads:])
    root=Path(a.solution_root);suite=root.parent
    while not (root/'preflight.json').exists():
        if (suite/'exit.json').exists():raise RuntimeError('solver exits before TensorBoard monitor initialization')
        time.sleep(a.poll_seconds)
    identity=json.loads((root/'identity.json').read_text());source_index=identity['source_index']
    c=identity['config'];n=c['full_loading'];refroot=Path(n['reference_root'])
    refcurve=json.loads((refroot/'result.json').read_text())['curve']
    with np.load(Path(c['trial']['geometry_dir'])/'mesh.npz') as mesh:
        nodes=int(mesh['n_material']);cells=int(mesh['n_material_cells'])
        xy=mesh['coordinates'][:nodes].copy();tri=Triangulation(xy[:,0],xy[:,1],mesh['triangles'][:cells])
        volumes=mesh['volumes'].copy();dofs=len(mesh['load_direction'])
    events=Path(a.event_dir);events.mkdir(parents=True,exist_ok=True)
    if any(events.glob('events.out.tfevents.*')):raise ValueError('refusing an existing TensorBoard event run')
    writer=SummaryWriter(str(events),flush_secs=a.flush_seconds)
    log_config_text(writer,ConfigDict(dict(driver=identity['driver_config'],monitor=a.to_dict()),allow_dotted_keys=True))
    writer.add_text('provenance','Inherited fine prefix through0.0052; endpoints provisional; reference is evaluation-only.',0)
    out=root/'tensorboard_figures';out.mkdir(exist_ok=True)
    published=[];done=set();scalar_done=set();seen=0
    initial=json.loads((root/f'state_{source_index:04d}/result.json').read_text())
    while not (root/'result.json').exists():time.sleep(a.poll_seconds)
    try:
        while True:
            result=json.loads((root/'result.json').read_text());curve=result['curve']
            for folder in sorted(root.glob('state_[0-9][0-9][0-9][0-9]')):
                index=int(folder.name[-4:])
                if index<=source_index:continue
                for path in sorted(folder.glob('sweep_[0-9][0-9][0-9][0-9].json')):
                    if str(path) in scalar_done:continue
                    row=json.loads(path.read_text());scalar_done.add(str(path))
                    if row['sweep']!=1 and row['sweep']%a.scalar_every:continue
                    step=(row['optimizer_count']-identity['source_optimizer_count'])//n['inner_updates']
                    if step<=seen:raise ValueError('TensorBoard optimizer-derived outer steps are not increasing')
                    seen=step
                    for tag,value in scalar_values(row).items():writer.add_scalar(tag,value,step)
            for row in [initial]+[r for r in curve if r['index']>source_index]:
                if row['index'] in done:continue
                published.append(publish_endpoint(writer,root,out,tri,xy,row,curve,refroot,refcurve,
                    dpi=a.figure_dpi,dofs=dofs,volumes=volumes))
                done.add(row['index']);print(json.dumps(dict(event='TensorBoard_load_published',**published[-1])),flush=True)
            writer.flush()
            atomic_json(root/'tensorboard_status.json',dict(complete=result['complete'],event_dir=str(events),
                log_root=a.log_root,published_loads=published,last_outer_step=seen,
                monitor_config_sha256=sha(config_path),monitor_source_sha256=sha(__file__)))
            if result['complete']:break
            if (suite/'exit.json').exists():raise RuntimeError('solver suite exits before monitoring completes')
            time.sleep(a.poll_seconds)
    finally:writer.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    main(runpy.run_path(args.config)['get_config'](),args.config)
