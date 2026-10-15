"""Compare actual synchronized early endpoint with the strict stored endpoint."""
import json
from pathlib import Path
import shutil
import numpy as np

ROOT = Path('/root/autodl-tmp/tf-logs/fracture2d/core_halo_tol2e5_20261015')
OLD = ROOT.parent/'core_halo_tip_hold_20261015/refined_k1'
NEW = ROOT/'refined_tol2e5'
c = json.loads((NEW/'config.json').read_text())
r = json.loads((ROOT/'result.json').read_text())
assert r['complete'] and r['summaries'][0]['field_stable']
with np.load(Path(c['mesh']['prepared_dir'])/'mesh.npz') as mesh:
    V = mesh['volumes'].copy()
    xy = mesh['coordinates'][:len(V)].copy()
tip = np.sum((xy-np.array([.5, .5]))**2, axis=1) < .024**2
refroot = Path(c['output']['reference_root'])
matched = [v for v in json.loads((refroot/'result.json').read_text())['curve'] if abs(v['load']-.0010) < 1e-14]
assert len(matched) == 1
reference = matched[0]
with np.load(refroot/f"state_{reference['index']:04d}"/'fields.npz') as fields:
    dr = fields['d'].copy()
rows, damages = [], []
for label, folder in [('strict_1e-5', OLD), ('relaxed_2e-5', NEW)]:
    report = json.loads((folder/'result.json').read_text())['curve'][-1]
    cfg = json.loads((folder/'config.json').read_text())
    assert report['load'] == .001 and report['q_d_synchronized']
    with np.load(folder/'state_0010/fields.npz') as fields:
        d = fields['d'].copy()
    delta = d-dr
    rows.append(dict(case=label, damage_tolerance=cfg['loading']['damage_tolerance'],
        sweep=report['sweep'], field_stable=report['field_stable'], damage_rms=report['damage_rms'],
        global_damage_L2=float(np.sqrt(V@(delta**2)/(V@dr**2))),
        tip_damage_L2=float(np.sqrt(V[tip]@(delta[tip]**2)/(V[tip]@dr[tip]**2))),
        global_damage_RMSE=float(np.sqrt(V@(delta**2)/V.sum())),
        reaction=report['reaction'], reaction_relative_error=abs(report['reaction']-reference['reaction'])/abs(reference['reaction']),
        physical_AT2=report['physical_AT2'], full_L1=report['full_loss'],
        timed_solve_seconds=report['timed_solve_seconds'], optimizer_count=report['optimizer_count']))
    damages.append(d)
summary = dict(complete=True, parent=r['parent'], load=.001, reference=str(refroot),
    reference_reaction=reference['reaction'], rows=rows,
    speed_ratio=rows[0]['timed_solve_seconds']/rows[1]['timed_solve_seconds'],
    early_vs_strict_damage_RMS=float(np.sqrt(V@(damages[1]-damages[0])**2/V.sum())),
    one_load_test_only=True, public_defaults_changed=False)
(ROOT/'review.json').write_text(json.dumps(summary, indent=2)+'\n')
# Correct inherited report-only labels; actual solver config was 2e-5 throughout.
shutil.copyfile(ROOT/'result.json', ROOT/'result_as_launched.json')
r['stopping_tolerances_unchanged'] = False
r['changed_stopping_control'] = dict(damage_tolerance=[1e-5, 2e-5], other_stopping_controls_unchanged=True)
r['metadata_correction'] = 'Copied driver initially retained the old descriptive flag; numerical config and stopping used 2e-5.'
(ROOT/'result.json').write_text(json.dumps(r, indent=2)+'\n')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
history = json.loads((NEW/'journal.json').read_text())
shutil.copyfile(ROOT/'comparison.png', ROOT/'comparison_as_launched.png')
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=150)
axes[0].semilogy([v['sweep'] for v in history], [v['damage_rms'] for v in history], 'o-')
axes[0].axhline(c['loading']['damage_tolerance'], color='k', linestyle='--', label='Actual stop threshold 2e-5')
axes[0].set(xlabel='Outer iteration', ylabel='Damage field change RMS')
axes[0].legend(fontsize=8)
labels = ['1e-5 (128-budget exit)', '2e-5 (field-stable)']
axes[1].bar(labels, [row['timed_solve_seconds'] for row in rows])
axes[1].set(ylabel='Solve seconds (compilation excluded)')
x = np.arange(2)
axes[2].bar(x-.18, [100*row['global_damage_L2'] for row in rows], width=.36, label='Global')
axes[2].bar(x+.18, [100*row['tip_damage_L2'] for row in rows], width=.36, label='Tip')
axes[2].set_xticks(x, ['1e-5', '2e-5'])
axes[2].set(ylabel='Damage relative L2 error (%)')
axes[2].legend()
fig.suptitle('Same U=0.0009 parent -> U=0.0010; density18/36, resample every Adam step')
fig.tight_layout()
fig.savefig(ROOT/'comparison.png')
plt.close(fig)
lines = ['# Damage-change threshold comparison', '',
    'Same0.0009parent->0.0010,segment18/tip36,k1,inner128. Only damageP1RMS',
    'tolerance changes;u/load1e-4,min3/consecutive3 unchanged.', '',
    '| Threshold | Outer | Seconds | Global damage L2 | Tip L2 | Reaction |',
    '|---|---:|---:|---:|---:|---:|']
for row in rows:
    lines.append(f"| {row['damage_tolerance']:.0e} | {row['sweep']} | {row['timed_solve_seconds']:.2f} | {100*row['global_damage_L2']:.3f}% | {100*row['tip_damage_L2']:.3f}% | {row['reaction']:.8f} |")
lines += ['', 'Errors arevolumeweightednodalrelativeL2againstthe samephysical-load',
    'nonlocal-directreference;tipradius.024. Strictcaseexhausts128budget;',
    'relaxedcasepassesfieldstop,neitherisPDEcertified. Timingexcludesbuild/',
    'compilation/endpointfigures. Publicdefaultsunchanged;nofulltrajectory.', '',
    'Report-onlycorrection:inheriteddriver descriptiveflag/plotline retained',
    '1e-5labelsalthoughactualconfig/solverused2e-5. Originals preservedin',
    'result_as_launched.json/comparison_as_launched.png;correctedoutputs',
    'derivefromactualconfigandjournal,withoutchangingfieldsortraining.', '']
(ROOT/'review.md').write_text('\n'.join(lines))
shutil.copyfile(__file__, ROOT/'review_probe.py')
print(json.dumps(summary, indent=2))
