"""Three stronger-density common-parent probes; no old-run writes or full trajectory."""
import gc
import json
import os
from pathlib import Path
import shutil
import time
import equinox as eqx
import jax
jax.config.update('jax_enable_x64', True)
jax.config.update('jax_default_matmul_precision', 'highest')
import jax.numpy as jnp
import numpy as np
from examples.fracture2d.configs.train_core_halo import get_config
from examples.fracture2d.loading import save_checkpoint, validate
from examples.fracture2d.losses import residual_metrics
from examples.fracture2d.mesh import load_geometry
from examples.fracture2d.monitoring import Monitor
from examples.fracture2d.stagger import committed_history, field_changes, field_stop, mechanics_first_sweep
from examples.fracture2d.storage import atomic_json
from examples.fracture2d.train import DamagePINN

ROOT = Path('/root/autodl-tmp/tf-logs/fracture2d/core_halo_more_tip_20261015')
PARENT = ROOT.parent/'pinn_core+halo/state_0009'
LOAD = .0010
if ROOT.exists():
    raise RuntimeError('refusing to overwrite existing probe root')
ROOT.mkdir()
shutil.copyfile(__file__, ROOT/'probe.py')
shutil.copyfile('/tmp/fracture2d_more_tip_density_stats_20261015.json', ROOT/'density_statistics.json')
c0 = get_config()
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
assert jax.devices()[0].platform == 'gpu'
m, operator, geometry = load_geometry(c0)
D = operator.to_scipy()
tri, area = m.triangles[:m.n_material_cells], m.areas[:m.n_material_cells]
solve_q = m.gpu_solver(tol=c0.validation.cg_rtol, atol=c0.validation.cg_atol,
                       maxiter=c0.validation.cg_maxiter)
with np.load(PARENT/'fields.npz') as f:
    q0, d0, H0 = [jnp.asarray(f[key]) for key in ('q', 'd', 'H')]
parent_row = json.loads((PARENT/'result.json').read_text())
prefix = json.loads((PARENT.parent/'result.json').read_text())['curve'][:9]
assert prefix[-1]['load'] == parent_row['load']
q_trial, _, _ = solve_q(d0, LOAD, q0)
H_trial = committed_history(H0, jnp.asarray(m.driver(np.asarray(q_trial))))
summaries = []
for label, boost, tip_boost, interval in [
        ('dense2', 36., 144., 1), ('dense4', 72., 576., 1),
        ('dense8', 144., 2304., 1)]:
    c = get_config()
    c.sample.boost, c.sample.tip_boost, c.sample.resample_every = boost, tip_boost, interval
    c.output.root = str(ROOT/label)
    c.output.event_dir = '/root/autodl-tmp/tf-logs/pinn_core+halo_more_tip/'+label
    c.output.checkpoint_sweeps = [1, 8, 32, 64, 128]
    validate(c)
    folder = Path(c.output.root)
    folder.mkdir()
    for source in Path('examples/fracture2d').glob('*.py'):
        dest = folder/'code_snapshot'/source.name
        dest.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, dest)
    atomic_json(folder/'config.json', c.to_dict())
    atomic_json(ROOT/'progress.json', dict(complete=False, case=label, sweep=0, summaries=summaries))
    started = time.perf_counter()
    pinn = DamagePINN(c, m, operator)
    build_seconds = time.perf_counter()-started
    pinn.restore(PARENT, parent_row['optimizer_count'])
    assert pinn.count == parent_row['optimizer_count']
    atomic_json(folder/'sampling_pool.json', pinn.pool_info)
    atomic_json(folder/'fork.json', dict(parent=str(PARENT), parent_load=parent_row['load'],
        parent_optimizer_count=pinn.count, trial_load=LOAD, initialization='restored_network_and_adam',
        first_slot=(pinn.count//interval) % c.sample.pool_size,
        finite_pool_cycles=True, source_geometry=geometry))
    # Pure compilation preflight: discard all candidate updates and retain state.
    context = pinn.context(H_trial, jnp.ones(m.n_material, jnp.float64))
    started = time.perf_counter()
    jax.block_until_ready(pinn.step(pinn.net, pinn.state, pinn.sampled_context(context), jnp.asarray(LOAD)))
    jax.block_until_ready(pinn.image(pinn.net, context))
    jax.block_until_ready(pinn.value(pinn.net, context, jnp.asarray(LOAD)))
    preflight_seconds = time.perf_counter()-started
    monitor = Monitor(folder, c, m, label)
    q, d, H = q0, d0, H0
    consecutive = 0
    journal = []
    started = time.perf_counter()
    try:
        for sweep in range(1, c.loading.prefix_outer_steps+1):
            def damage(history, old, carry):
                image, value = pinn.solve(history, old, LOAD)
                m.factors(np.asarray(image))
                _, relative = residual_metrics(np.asarray(image), np.asarray(history), D,
                    gc0=c.physical.Gc/c.physical.l0, gc1=c.physical.Gc*c.physical.l0)
                return image, carry, dict(full_loss=value, AT2_relative=relative)
            stage, _ = mechanics_first_sweep(q, d, H0, LOAD, None,
                mechanics=solve_q, driver=m.driver, damage=damage, omega=c.loading.omega)
            q, d, H = [stage[key] for key in ('q', 'd', 'H')]
            changes = field_changes(q, stage['q_previous'], d, stage['d_previous'], tri, area, load=LOAD)
            stop = field_stop(changes, displacement_limit=c.loading.displacement_tolerance,
                damage_limit=c.loading.damage_tolerance, sweep=sweep,
                minimum_sweeps=c.loading.minimum_sweeps, consecutive_passes=c.loading.consecutive_passes,
                previous_consecutive=consecutive)
            consecutive = stop['consecutive']
            row = dict(index=10, load=LOAD, sweep=sweep, outer_cap=c.loading.prefix_outer_steps,
                **changes, field_stable=stop['converged'], field_stopping=stop,
                full_loss=stage['inner']['full_loss'], inner_AT2=stage['inner']['AT2_relative'],
                reaction=stage['reaction'], damage_min=float(d.min()), damage_max=float(d.max()),
                optimizer_count=pinn.count, q_d_synchronized=False, mechanics_calls=sweep,
                parent_path=str(PARENT/'fields.npz'), elapsed_seconds=time.perf_counter()-started)
            journal.append(row)
            if sweep in c.output.checkpoint_sweeps:
                save_checkpoint(folder/'state_0010'/f'outer_{sweep:04d}',
                    dict(q=np.asarray(q), d=np.asarray(d), H=np.asarray(H), H_parent=np.asarray(H0)),
                    dict(row), pinn)
            if sweep == 1 or sweep % 8 == 0:
                atomic_json(ROOT/'progress.json', dict(complete=False, case=label, sweep=sweep,
                    current=row, summaries=summaries))
                atomic_json(folder/'journal.json', journal)
                print(json.dumps(dict(event='outer', case=label, **row)), flush=True)
                monitor.stage(row, 9*c.loading.outer_steps+sweep)
            if stop['converged']:
                break
        q_sync, reaction, mechanical = solve_q(d, LOAD, q)
        H_sync = committed_history(H0, jnp.asarray(m.driver(np.asarray(q_sync))))
        force = m.matrix(np.asarray(d))@np.asarray(q_sync)
        assert np.linalg.norm(force[m.free]) <= mechanical['convergence_limit']*(1+1e-3)
        np.testing.assert_allclose(force@m.load_direction, reaction, rtol=1e-9, atol=1e-6)
        _, physical = residual_metrics(np.asarray(d), np.asarray(H_sync), D,
            gc0=c.physical.Gc/c.physical.l0, gc1=c.physical.Gc*c.physical.l0)
        report = dict(row, reaction=reaction, physical_AT2=physical, q_d_synchronized=True,
            mechanics_calls=sweep+1, mechanics=mechanical, certified_path=False,
            status='field_stable' if stop['converged'] else 'budget_exhausted',
            timed_solve_seconds=time.perf_counter()-started)
        fields = dict(q=np.asarray(q_sync), d=np.asarray(d), H=np.asarray(H_sync), H_used=np.asarray(H),
                      H_parent=np.asarray(H0), d_parent=np.asarray(d0))
        save_checkpoint(folder/'state_0010', fields, report, pinn)
        atomic_json(folder/'journal.json', journal)
        atomic_json(folder/'result.json', dict(complete=True, curve=[report], parent=str(PARENT),
            scope='one_load_probe', requested_load=LOAD, preflight_seconds=preflight_seconds,
            pool_build_seconds=build_seconds))
        monitor.endpoint(report, fields, prefix+[report])
        summary = dict(case=label, **report, pool_build_seconds=build_seconds,
            preflight_seconds=preflight_seconds, pool_capacity=pinn.pool_info['node_capacity'])
        summaries.append(summary)
        atomic_json(ROOT/'progress.json', dict(complete=False, summaries=summaries, current=summary))
        print(json.dumps(dict(event='probe_complete', **summary)), flush=True)
    finally:
        monitor.close()
    del pinn, context, monitor
    gc.collect()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=150)
for row in summaries:
    history = json.loads((ROOT/row['case']/'journal.json').read_text())
    x = [v['sweep'] for v in history]
    axes[0].semilogy(x, [v['damage_rms'] for v in history], label=row['case'])
    axes[1].semilogy(x, [v['full_loss'] for v in history], label=row['case'])
axes[0].axhline(c0.loading.damage_tolerance, color='k', linestyle='--', label='stop tolerance')
axes[0].set(xlabel='Outer iteration', ylabel='Damage field change RMS')
axes[1].set(xlabel='Outer iteration', ylabel='Full-domain L1 loss')
axes[2].bar([r['case'] for r in summaries], [r['timed_solve_seconds'] for r in summaries])
axes[2].set(ylabel='Timed solve seconds (compilation excluded)')
axes[2].tick_params(axis='x', rotation=25)
for ax in axes[:2]:
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(ROOT/'comparison.png')
plt.close(fig)
atomic_json(ROOT/'result.json', dict(complete=True, scope='stronger_tip_density_probe', parent=str(PARENT),
    trial_load=LOAD, summaries=summaries, first_slots_follow_global_optimizer_count=True,
    stopping_tolerances_unchanged=True, new_full_loading_trajectory=False))
atomic_json(ROOT/'progress.json', dict(complete=True, summaries=summaries))
