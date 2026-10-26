"""Compare direct-path SOAP/Adam checkpoints at the revised common budget."""
import json
import inspect
from importlib.metadata import version
from datetime import datetime
from pathlib import Path
import re

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ml_collections import ConfigDict
import numpy as np
import optax

from examples.icemelting3d.accuracy import _crossing
from examples.icemelting3d.train import _build_phase_network

jax.config.update('jax_default_matmul_precision','highest')
output=Path('reports/icemelting-soap-adam-5s-20261024')
launch=json.loads((output/'launch.json').read_text())
assert launch['status']=='complete_at_revised_budget'
archive=Path('docs/tasks/archive')
ref_path=Path('reports/icemelting-weak-prior-20261024/reference.npz')
with np.load(ref_path,allow_pickle=False) as data:
    ref={key:data[key] for key in data.files}
assert json.loads(str(ref['identity']))['nondim']['T_c']==5.
n=128
fold=(np.abs(np.arange(n)-(n-1)/2)-.5).astype(int)
truth=ref['phi'][-1][np.ix_(fold,fold,fold)].swapaxes(0,1)  # Production meshgrid uses xy ordering.
axis=np.asarray(jnp.linspace(0.,1.,n))
predict=eqx.filter_jit(lambda net,x:jax.vmap(net,in_axes=(0,None))(x,jnp.float32(1.))[:,0])
report={'physical_terminal_s':5.,'updates':5000,'lr_init':.001,'seed':42,
        'reference_path':str(ref_path.resolve()),'runs':{},
        'scope':'Configured SOAP versus configured Adam at shared lr, not separately tuned optima',
        'budget_note':'SOAP was stopped after a user budget reduction; only its5000-step checkpoint/history/timing are compared.'}
report['optax_version']=version('optax')
report['adam_defaults']={key:inspect.signature(optax.adam).parameters[key].default
                         for key in ['b1','b2','eps','eps_root','nesterov']}
configs=[]
initial_models=[]
fig,axes=plt.subplots(1,2,figsize=(11,4.5))

def timing(log_path):
    start=None
    end=None
    clock={}
    records=[]
    for line in log_path.read_text().splitlines():
        match=re.match(r'^[IWEF](\d{4}) (\d\d:\d\d:\d\d\.\d+)',line)
        if not match:
            continue
        stamp=datetime.strptime('2026'+match[1]+' '+match[2],'%Y%m%d %H:%M:%S.%f').timestamp()
        if 'Initial L2:' in line:
            start=stamp
            clock[0]=stamp
        m=re.search(r'L2 update (\d+):',line)
        if m and int(m[1])<=5000:
            clock[int(m[1])]=stamp
        m=re.search(r'\]\s+(\d+): total=',line)
        if m and int(m[1])<=5000:
            records.append((int(m[1]),stamp))
        if 'Checkpoint saved:' in line and 'ckpt_005000.eqx' in line:
            end=stamp
    assert start is not None and end is not None
    seconds=[(t2-t1)/(s2-s1) for (s1,t1),(s2,t2) in zip(records,records[1:])
             if s2-s1==20 and s1%200!=0]
    return {'seconds_to_5000_checkpoint_after_initial_eval':end-start,
            'median_update_ms_excluding_eval_boundaries':float(np.median(seconds)*1000),
            'elapsed_to_eval_s':{str(step):value-start for step,value in clock.items()}}

for spec,color in zip(launch['runs'],['#1565c0','#e87817']):
    name=spec['optimizer']
    directory=Path(spec['run_dir'])
    c=ConfigDict(json.loads((directory/'config.json').read_text()))
    assert c.nondim.T_c==5. and c.train.lr_init==.001
    configs.append(c.to_dict())
    history=[json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines()]
    history=[row for row in history if row['step']<=5000]
    endpoint=next(row for row in history if row['step']==5000)
    assert endpoint['full_grid_points']==n**3
    template=_build_phase_network(c,jax.random.split(jax.random.PRNGKey(c.train.seed))[0])
    initial=eqx.tree_deserialise_leaves(directory/'model_initial.eqx',template)
    assert eqx.tree_equal(initial,template)
    initial_models.append(initial)
    checkpoint=directory/'ckpt_005000.eqx'
    model=eqx.tree_deserialise_leaves(checkpoint,template)
    pred=np.empty(n**3,dtype=np.float32)
    for start in range(0,n**3,32768):
        ijk=np.stack(np.unravel_index(np.arange(start,start+32768),(n,n,n)),axis=1)
        points=axis[ijk[:,[1,0,2]]]
        pred[start:start+32768]=np.asarray(predict(model,jnp.asarray(points)))
    pred=pred.reshape(n,n,n)
    error=pred.astype(np.float64)-truth
    rel=float(np.linalg.norm(error)/np.linalg.norm(truth))
    assert np.isclose(rel,endpoint['graph_terminal_full_rel_l2'],rtol=2e-3,atol=2e-7),(name,rel,endpoint)
    k=n//2
    rays=[pred[k,k:,k],pred[k,:k,k][::-1],pred[k:,k,k],pred[:k,k,k][::-1],pred[k,k,k:],pred[k,k,:k][::-1]]
    reference_radius,reference_width=_crossing(ref['axis'],ref['phi'][-1,:,0,0])
    radii,widths=zip(*[_crossing(ref['axis'],ray) for ray in rays])
    assert all(radius is not None for radius in radii)
    full=[row for row in history if row['full_grid_points'] is not None and row['step']>0]
    times=timing(Path(spec['stdout']))
    row={'run_dir':str(directory),'checkpoint':str(checkpoint),'config':c.to_dict(),
         'history':history,'endpoint':endpoint,'timing':times,
         'independent_terminal_graph_rel_l2':rel,
         'phase_max_error':float(np.max(np.abs(error))),
         'interface_mae':float(np.mean(np.abs(error[np.abs(truth)<.9]))),
         'solid_fraction_rel_l2':float(np.linalg.norm(error)/np.linalg.norm(truth+1.)),
         'radius_axes':radii,'width_axes':widths,'reference_radius':reference_radius,'reference_width':reference_width,
         'max_radius_relative_error':float(np.max(np.abs(np.asarray(radii)/reference_radius-1))),
         'last_five_full_graph_rel_l2':[r['graph_terminal_full_rel_l2'] for r in full[-5:]],
         'initial_checkpoint_matches_fresh_builder':True}
    tail=[r['graph_terminal_rel_l2'] for r in history if r['step']>=4000]
    row['last_1000_updates_sample_graph_median']=float(np.median(tail))
    row['last_1000_updates_sample_graph_range']=[float(min(tail)),float(max(tail))]
    row['last_five_full_graph_median']=float(np.median(row['last_five_full_graph_rel_l2']))
    report['runs'][name]=row
    for ax,kind in zip(axes,['graph','analytic']):
        ax.semilogy([r['step'] for r in history],[r[f'{kind}_terminal_rel_l2'] for r in history],
                    color=color,label=f'{name.upper()}: 8000 nodes',lw=1.5)
        ax.scatter([r['step'] for r in full],[r[f'{kind}_terminal_full_rel_l2'] for r in full],
                   edgecolors=color,facecolors='none',s=50,zorder=4,label=f'{name.upper()}: full grid')
    print(name,'graph L2=',rel,'timing=',times['seconds_to_5000_checkpoint_after_initial_eval'],
          'max radius error=',row['max_radius_relative_error'],flush=True)

def differences(a,b,prefix=''):
    result=[]
    for key in a:
        path=prefix+'.'+key if prefix else key
        if isinstance(a[key],dict):
            result.extend(differences(a[key],b[key],path))
        elif a[key]!=b[key]:
            result.append(path)
    return result

changed=differences(*configs)
assert set(changed)=={'train.optimizer','train.iterations','log.run_name'},changed
assert eqx.tree_equal(*initial_models)
report['config_differences']=changed
report['initial_models_identical']=True
report['configured_update_limits_note']='SOAP saved its original20000-update limit before interruption; both selected states are at5000; the schedule does not depend on that limit.'
report['adam_over_soap_terminal_graph_l2']=report['runs']['adam']['independent_terminal_graph_rel_l2']/report['runs']['soap']['independent_terminal_graph_rel_l2']
ref_axis=ref['axis']
radius=np.sqrt(ref_axis[:,None,None]**2+ref_axis[None,:,None]**2+ref_axis[None,None,:]**2)
radius0=c.physical.R0/c.nondim.L_c
speed=c.physical.lam*c.nondim.T_c/c.nondim.L_c
analytic=np.tanh((radius0-speed-radius)/(np.sqrt(2)*c.physical.eps_star))
norm_ref=8*np.sum(ref['phi'][-1]**2)
norm_analytic=8*np.sum(analytic**2)
bias=8*np.sum((ref['phi'][-1]-analytic)**2)
report['graph_reference_versus_analytic_rel_l2']=float(np.sqrt(bias/norm_analytic))
for row in report['runs'].values():
    graph_sse=row['endpoint']['graph_terminal_full_rel_l2']**2*norm_ref
    analytic_sse=row['endpoint']['analytic_terminal_full_rel_l2']**2*norm_analytic
    row['error_cosine_with_reference_minus_analytic']=float((analytic_sse-graph_sse-bias)/(2*np.sqrt(graph_sse*bias)))
for ax,title in zip(axes,['Same-graph numerical reference','Approximate analytic profile']):
    ax.set_title(title)
    ax.set_xlabel('Completed optimizer updates')
    ax.set_ylabel('Terminal phase-field relative L2')
    ax.grid(True,which='both',alpha=.18)
    ax.legend(fontsize=8)
fig.suptitle('SOAP versus Adam: physical t=5 s, initial lr=0.001, seed42')
fig.tight_layout()
fig.savefig(archive/'icemelting-soap-adam-5s.png',dpi=180)
plt.close(fig)
(archive/'icemelting-soap-adam-5s.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print('Adam/SOAP endpoint L2 ratio:',report['adam_over_soap_terminal_graph_l2'],flush=True)
