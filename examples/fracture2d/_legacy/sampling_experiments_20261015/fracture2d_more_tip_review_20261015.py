"""Evaluate stronger densities and stored controls at the identical load."""
import csv
import json
from pathlib import Path
import shutil
import numpy as np
from examples.fracture2d.stagger import p1_rms

ROOT = Path('/root/autodl-tmp/tf-logs/fracture2d/core_halo_more_tip_20261015')
OLD = ROOT.parent/'core_halo_tip_hold_20261015'
result = json.loads((ROOT/'result.json').read_text())
assert result['complete'] and len(result['summaries']) == 3
old = json.loads((OLD/'result.json').read_text())
assert old['parent'] == result['parent'] and old['trial_load'] == result['trial_load']
controls = [row for row in old['summaries'] if row['case'] in ('original_k1', 'refined_k1')]
assert len(controls) == 2
stats = json.loads((ROOT/'density_statistics.json').read_text())
stats_by_case = {row['case']: row for row in stats}
stats_by_case['refined_k1'] = stats_by_case['previous']
previous_stats = json.loads((OLD/'density_statistics.json').read_text())
c = json.loads((ROOT/'dense2/config.json').read_text())
with np.load(Path(c['mesh']['prepared_dir'])/'mesh.npz') as mesh:
    V = mesh['volumes'].copy()
    xy = mesh['coordinates'][:len(V)].copy()
    tri = mesh['triangles'][:int(mesh['n_material_cells'])].copy()
local = xy[tri]
a, b = local[:, 1]-local[:, 0], local[:, 2]-local[:, 0]
areas = np.abs(a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0])/2
tip = np.sum((xy-np.array([.5, .5]))**2, axis=1) < .024**2
segment_x = np.clip(xy[:, 0], .5, 1.)
distance2 = (xy[:, 0]-segment_x)**2+(xy[:, 1]-.5)**2
offband = distance2 > .048**2
assert tip.any() and offband.any()
refroot = Path(c['output']['reference_root'])
refs = [row for row in json.loads((refroot/'result.json').read_text())['curve']
        if abs(row['load']-result['trial_load']) < 1e-14]
assert len(refs) == 1
ref = refs[0]
with np.load(refroot/f"state_{ref['index']:04d}"/'fields.npz') as fields:
    dr = fields['d'].copy()
with np.load(Path(result['parent'])/'fields.npz') as fields:
    parent_damage = fields['d'].copy()
rows = []
for source_root, case in [(OLD, row) for row in controls]+[(ROOT, row) for row in result['summaries']]:
    name = case['case']
    cfg = json.loads((source_root/name/'config.json').read_text())
    for key in ('physical', 'geometry', 'nl', 'data', 'mesh', 'arch', 'features', 'loss', 'train', 'loading', 'validation'):
        assert c[key] == cfg[key], (name, key)
    assert cfg['sample']['resample_every'] == 1
    with np.load(source_root/name/'state_0010/fields.npz') as fields:
        d = fields['d'].copy()
        np.testing.assert_array_equal(fields['d_parent'], parent_damage)
    assert d.shape == dr.shape and np.isfinite(d).all()
    delta = d-dr
    relative = lambda mask: float(np.sqrt(V[mask]@(delta[mask]**2)/(V[mask]@(dr[mask]**2))))
    history = json.loads((source_root/name/'journal.json').read_text())
    tail = np.array([row['damage_rms'] for row in history[-32:]])
    if case['sweep'] == 128:
        with np.load(source_root/name/'state_0010/outer_0064/fields.npz') as f:
            d64 = f['d'].copy()
        with np.load(source_root/name/'state_0010/outer_0128/fields.npz') as f:
            d128 = f['d'].copy()
        path_length = sum(row['damage_rms'] for row in history if 65 <= row['sweep'] <= 128)
        net_distance = p1_rms(d128-d64, tri, areas)
        net_over_path = net_distance/path_length
        assert 0 <= net_over_path <= 1+1e-12
    else:
        net_over_path = None  # The 64->128 diagnostic requires all 128 sweeps.
    draws = previous_stats['original']['expected_core_tip0024'] if name == 'original_k1' else stats_by_case[name]['expected_tip_cores']
    pool = json.loads((source_root/name/'sampling_pool.json').read_text())
    rows.append(dict(case=name, source=str(source_root/name), segment_boost=cfg['sample']['boost'],
        tip_boost=cfg['sample']['tip_boost'], expected_tip_draws=draws,
        global_damage_L2=relative(np.ones(len(V), bool)), tip_damage_L2=relative(tip),
        offband_damage_L2=relative(offband),
        offband_damage_RMSE=float(np.sqrt(V[offband]@(delta[offband]**2)/V[offband].sum())),
        offband_damage_mean=float(V[offband]@d[offband]/V[offband].sum()),
        sweep=case['sweep'], field_stable=case['field_stable'], damage_rms=case['damage_rms'],
        tail_damage_rms_mean=float(tail.mean()), full_L1=case['full_loss'], physical_AT2=case['physical_AT2'],
        net_over_path_change_64to128=net_over_path,
        timed_solve_seconds=case['timed_solve_seconds'], reaction=case['reaction'],
        reaction_relative_error=abs(case['reaction']-ref['reaction'])/abs(ref['reaction']),
        pool_capacity=pool['node_capacity']))
summary = dict(complete=True, scope='single_load_density_sweep_with_stored_controls',
    parent=result['parent'], trial_load=result['trial_load'], reference=str(refroot),
    reference_load=ref['load'], reference_reaction=ref['reaction'], tip_radius=.024,
    offband_distance_to_segment=.048, rows=rows,
    all_other_numerical_settings_equal=True, previous_controls_not_rerun=True)
(ROOT/'review.json').write_text(json.dumps(summary, indent=2)+'\n')
with (ROOT/'comparison.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=150)
x = [row['expected_tip_draws'] for row in rows]
axes[0, 0].plot(x, [100*row['global_damage_L2'] for row in rows], 'o-', label='Global')
axes[0, 0].plot(x, [100*row['tip_damage_L2'] for row in rows], 'o-', label='Tip radius 0.024')
axes[0, 0].set(ylabel='Relative damage L2 error (%)')
axes[0, 0].legend()
axes[0, 1].plot(x, [row['offband_damage_RMSE'] for row in rows], 'o-')
axes[0, 1].set(ylabel='Off-band damage RMSE')
axes[1, 0].plot(x, [row['tail_damage_rms_mean'] for row in rows], 'o-')
axes[1, 0].axhline(c['loading']['damage_tolerance'], color='k', linestyle='--', label='Stop threshold')
axes[1, 0].set(ylabel='Mean damage field change, last 32 outers')
axes[1, 0].legend()
axes[1, 1].plot(x, [row['timed_solve_seconds'] for row in rows], 'o-')
axes[1, 1].set(ylabel='Timed solve seconds (compilation excluded)')
for ax in axes.ravel():
    ax.set_xlabel('Expected tip draws per 2048-core batch (with replacement)')
    ax.grid(alpha=.25)
fig.suptitle('Same parent U=0.0009 -> U=0.0010; all cases resample every Adam update')
fig.tight_layout()
fig.savefig(ROOT/'density_comparison.png')
plt.close(fig)
lines = ['# Stronger tip-density sweep', '',
    'Same actual0.0009parent ->0.0010;core2048,pool2048,k1,inner128,outercap128.',
    'Priororiginal/refinedk1controls are reused from their unchanged savedruns.',
    'Errors use the same physical-load nonlocal-direct reference,with nodal',
    'volumeweighting. Tipradius0.024;offband means distance tosegment>0.048.', '',
    '| Segment/tip boosts | Expected tip draws | Global L2 | Tip L2 | Off-band RMSE | Outer | Seconds |',
    '|---|---:|---:|---:|---:|---:|---:|']
for row in rows:
    lines.append(f"| {row['segment_boost']:g}/{row['tip_boost']:g} | {row['expected_tip_draws']:.1f} | {100*row['global_damage_L2']:.3f}% | {100*row['tip_damage_L2']:.3f}% | {row['offband_damage_RMSE']:.6g} | {row['sweep']} | {row['timed_solve_seconds']:.2f} |")
lines += ['', 'Times excludepoolconstruction,compilation andendpointfigures.',
    'The tip region has91nodes;drawcounts include duplicates. Positivebaseline,',
    'detachedimportance correction andcompletehalo rows are unchanged.',
    'This is one parent/seed/load,not a full trajectory or a universal optimum.', '']
(ROOT/'review.md').write_text('\n'.join(lines))
shutil.copyfile(__file__, ROOT/'review_probe.py')
print(json.dumps(summary, indent=2))
