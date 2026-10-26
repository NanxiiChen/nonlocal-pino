"""Archive production cold-start histories and independently reload final fields."""
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from ml_collections import ConfigDict

from examples.icemelting3d.accuracy import _crossing
from examples.icemelting3d.train import _build_phase_network

jax.config.update('jax_default_matmul_precision', 'highest')
root = Path('/root/autodl-tmp/tf-logs/icemelting3d')
names = ['20261024-ic-latent-coldstart-01', '20261024-ic-latent-default-coldstart-01']
directories = ['20261024-ic-latent-coldstart-01-sota', '20261024-ic-latent-default-coldstart-01']
archive = Path('docs/tasks/archive')
report = {'physical_terminal_s': 4., 'formulation': 'ic_latent', 'runs': {}}
initial_models = []
fn = eqx.filter_jit(lambda net, x, t: jax.vmap(net, in_axes=(0,None))(x,t)[:,0])
fig, axes = plt.subplots(1,2,figsize=(11,4.4),sharey=True)
for ax, name, directory_name in zip(axes,names,directories):
    directory = root/directory_name
    c = ConfigDict(json.loads((directory/'config.json').read_text()))
    history = [json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines()]
    stored_summary = json.loads((directory/'summary.json').read_text())
    summary = {field: stored_summary[field] for field in
               ('updates', 'training_wall_seconds', 'final_metrics', 'cold_start')}
    summary['model_path'] = str(directory/'model_final.eqx')
    assert summary['updates'] == history[-1]['step'] == 2000
    builder = _build_phase_network(c,jax.random.split(jax.random.PRNGKey(c.train.seed))[0])
    initial = eqx.tree_deserialise_leaves(directory/'model_initial.eqx',builder)
    assert eqx.tree_equal(initial,builder)
    initial_models.append(initial)
    model = eqx.tree_deserialise_leaves(directory/'model_final.eqx',builder)
    with np.load(c.eval.graph_reference_path,allow_pickle=False) as f:
        axis = f['axis']
        target = f['phi'][16]
    h = 1./127
    rays = np.full((6,len(axis),3),.5+h/2)
    for i in range(6):
        rays[i,:,i%3] = .5 + (1 if i<3 else -1)*axis
    fields = np.asarray(fn(model,jnp.asarray(rays.reshape(-1,3),dtype=jnp.float32),jnp.float32(1.))).reshape(6,-1)
    radius_ref, width_ref = _crossing(axis,target[:,0,0])
    radii, widths = zip(*[_crossing(axis,row) for row in fields])
    phi_initial = np.asarray(fn(model,jnp.asarray(rays.reshape(-1,3),dtype=jnp.float32),jnp.float32(0.)))
    phi_initial_before = np.asarray(fn(initial,jnp.asarray(rays.reshape(-1,3),dtype=jnp.float32),jnp.float32(0.)))
    assert np.array_equal(phi_initial,phi_initial_before)
    row = {'run_dir':str(directory),'config':c.to_dict(),'summary':summary,'history':history,
           'reference':{'path':str(Path(c.eval.graph_reference_path).resolve()),
                        'physical_time_s':4.,'snapshot_index':16},
           'initial_model_path':str(directory/'model_initial.eqx'),
           'checkpoint_reloaded':True,'initial_checkpoint_matches_fresh_builder':True,
           'initial_field_unchanged_after_training':True,
           'source_dir':str(directory/'source'),
           'geometry':{'radius_axes':radii,'width_axes':widths,'reference_radius':radius_ref,
                       'reference_width':width_ref,'max_radius_relative_error':float(np.max(np.abs(np.asarray(radii)/radius_ref-1))),
                       'ray_max_phase_error':float(np.max(np.abs(fields-target[:,0,0][None,:])))}}
    report['runs'][name]=row
    if directory_name != name:
        row['original_run_dir'] = str(root/name)
    steps=[r['step'] for r in history]
    for ref,color in [('graph','#1565c0'),('analytic','#e87817')]:
        ax.semilogy(steps,[r[f'{ref}_terminal_rel_l2'] for r in history],color=color,lw=1.7,label=f'{ref}: 8000 nodes')
        full=[r for r in history if f'{ref}_terminal_full_rel_l2' in r]
        ax.scatter([r['step'] for r in full],[r[f'{ref}_terminal_full_rel_l2'] for r in full],
                   facecolors='none',edgecolors=color,s=65,linewidths=1.5,zorder=4,label=f'{ref}: full grid')
    ax.set_title(f"K_core={c.sample.K_core}, Nt={c.sample.Nt}, lr={c.train.lr_init}")
    ax.set_xlabel('Completed optimizer updates')
    ax.grid(True,which='both',alpha=.18)
    ax.legend(fontsize=8,loc='upper right')
    print(name,json.dumps(row['geometry']),flush=True)
axes[0].set_ylabel('Terminal phase-field relative L2')
fig.suptitle('Production IC-only latent residual: cold starts at physical t=4 s')
fig.tight_layout()
fig.savefig(archive/'icemelting-production-latent.png',dpi=180)
plt.close(fig)
assert all(eqx.tree_equal(initial_models[0], model) for model in initial_models[1:])
(archive/'icemelting-production-latent.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
