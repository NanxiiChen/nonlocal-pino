"""Common-reference summary and unclipped before/after fields for all six controls."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
OUT=ROOT/'reports/twophase-vortex-residual'
with np.load('/root/autodl-tmp/data/twophase2d/results_periodic_vortex.npz') as data:
    ref=data['phi_local'];points=data['points'];times=data['time']
rows=[];predictions={};histories={}
for report,representation in [('twophase-vortex-residual','additive'),('twophase-vortex-latent','leaky_tanh')]:
    analysis=json.loads((ROOT/'reports'/report/'analysis.json').read_text())
    if analysis['status']!='passed':raise ValueError('Incomplete analysis')
    for item in analysis['runs']:
        label=representation+'/'+item['mode'];run=Path(item['run'])
        with np.load(run/'prediction.npz') as data:phi=data['phi']
        predictions[label]=phi
        histories[label]=[json.loads(s) for s in (run/'metrics.jsonl').read_text().splitlines()]
        interface=[]
        for a,b in zip(phi,ref,strict=True):
            mask=np.abs(b)<.9
            interface.append(float(np.linalg.norm((a-b)[mask])/np.linalg.norm(b[mask])))
        rows.append(dict(label=label,run=str(run),midpoint_phi_common_local=item['per_time_phi_relative_L2_to_local'][10],
            terminal_phi_common_local=item['terminal_phi_relative_L2_to_local'],
            terminal_phi_own_reference=item['terminal_phi_relative_L2'],
            terminal_interface_phi_common_local=interface[-1],per_time_interface_phi_common_local=interface,
            terminal_width_proxy=item['geometry_to_common_local'][-1]['width_proxy'],
            terminal_width_proxy_relative_error=item['geometry_to_common_local'][-1]['width_proxy_relative_error'],
            terminal_contour_mean_distance=item['geometry_to_common_local'][-1]['symmetric_mean_contour_distance'],
            max_soft_area_drift=item['max_gas_area_relative_drift'],
            terminal_q_own_reference_rms=item['terminal_q_rms_error'],trainer_seconds=item['trainer_wall_seconds'],
            compiled_step_seconds=item['mean_compiled_step_seconds']))
# Explicit final5000 checkpoints only: representation-only candidate has lowest
# terminal common-local interface-band error among the three mapped controls.
best=min([r for r in rows if r['label'].startswith('leaky_tanh/')],key=lambda r:r['terminal_interface_phi_common_local'])
if best['label']!='leaky_tanh/original':raise ValueError('Revise comparison selection/report around changed results')
labels=['Common local reference','Additive + original residual','Leaky tanh + original residual']
fields=[ref,predictions['additive/original'],predictions[best['label']]]
ids=[0,5,10,15,20];X,Y=points.T.reshape(2,64,64)
vmin=min(-1,min(float(p.min()) for p in fields));vmax=max(1,max(float(p.max()) for p in fields))
fig,axes=plt.subplots(3,5,figsize=(14,8),layout='constrained')
for row,(phi,label) in enumerate(zip(fields,labels,strict=True)):
    for col,i in enumerate(ids):
        ax=axes[row,col]
        im=ax.pcolormesh(X,Y,phi[i],shading='nearest',cmap='viridis',vmin=vmin,vmax=vmax)
        ax.contour(X,Y,phi[i],levels=[0],colors='white',linewidths=.7)
        if row:
            ax.contour(X,Y,ref[i],levels=[0],colors='orangered',linestyles='dashed',linewidths=.7)
            error=np.linalg.norm(phi[i]-ref[i])/np.linalg.norm(ref[i])
            ax.text(.02,.02,f'L2={100*error:.2f}%',transform=ax.transAxes,fontsize=8,bbox=dict(facecolor='white',alpha=.9,edgecolor='none'))
        if row==0:ax.set_title(f't={times[i]:.2f}')
        if col==0:ax.set_ylabel(label+'\ny')
        if row==2:ax.set_xlabel('x')
        ax.set(aspect='equal')
fig.colorbar(im,ax=axes.ravel().tolist(),label='phi (full range; no clipping)',shrink=.8)
fig.suptitle('Phase representation comparison, final5000 updates; dashed red = common local interface')
fig.savefig(OUT/'representation-fields.png',dpi=180);plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
for label in ['additive/original','leaky_tanh/original','leaky_tanh/map_transport','leaky_tanh/map_both']:
    h=histories[label]
    axes[0].semilogy([r['step'] for r in h],[r['terminal_rel_l2_phi_local'] for r in h],label=label)
    axes[1].plot(times,[np.linalg.norm(a-b)/np.linalg.norm(b) for a,b in zip(predictions[label],ref,strict=True)],label=label)
    axes[2].plot(X[48],predictions[label][-1,48],label=label)
axes[2].plot(X[48],ref[-1,48],color='black',linestyle='dashed',label='Common local reference')
axes[0].set(xlabel='Updates',ylabel='Terminal phi relative L2 to common local reference')
axes[1].set(xlabel='Physical time',ylabel='Phi relative L2 to common local reference')
axes[2].set(xlabel='x at y=0.75, t=2',ylabel='phi (no clipping)',xlim=(.2,.8))
for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=7)
fig.savefig(OUT/'combined-curves-profiles.png',dpi=170);plt.close(fig)
summary=dict(status='passed',all_full_updates=5000,all_seed_as_provenance=42,
    reference='same independently refined periodic-vortex reference; all field/interface metrics use common local target',
    width_reference_terminal=.10917907563644604,
    selected_interface_candidate=best['label'],selection_scope='lowest terminal interface phi error among three final5000 mapped models; no checkpoint/seed search',
    conclusion='Representation improves interfaces. Exact inverse-map scaling trades some interface quality for lower full-field/q/mass errors; bounded physical-field weights worsen all main fields. No default promotion or robustness claim.',
    rows=rows)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
print(json.dumps(summary,indent=2))
