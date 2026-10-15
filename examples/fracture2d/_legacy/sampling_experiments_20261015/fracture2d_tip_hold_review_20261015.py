"""Matched-load evaluation of the completed finite sampling probe."""
import csv
import json
from pathlib import Path
import shutil
import time
import numpy as np

ROOT = Path('/root/autodl-tmp/tf-logs/fracture2d/core_halo_tip_hold_20261015')
deadline = time.monotonic()+120
while not (ROOT/'result.json').exists():
    if time.monotonic() > deadline:
        raise RuntimeError('probe has not finished; check its process receipt')
    time.sleep(1)
r = json.loads((ROOT/'result.json').read_text())
assert r['complete'] and len(r['summaries']) == 4
c = json.loads((ROOT/'original_k1/config.json').read_text())
with np.load(Path(c['mesh']['prepared_dir'])/'mesh.npz') as mesh:
    V = mesh['volumes'].copy()
    xy = mesh['coordinates'][:len(V)].copy()
refroot = Path(c['output']['reference_root'])
refs = [row for row in json.loads((refroot/'result.json').read_text())['curve']
        if abs(row['load']-r['trial_load']) < 1e-14]
assert len(refs) == 1
ref = refs[0]
with np.load(refroot/f"state_{ref['index']:04d}"/'fields.npz') as fields:
    dr = fields['d'].copy()
tip = np.sum((xy-np.array([.5, .5]))**2, axis=1) < .024**2
rows = []
for row in r['summaries']:
    with np.load(ROOT/row['case']/'state_0010/fields.npz') as fields:
        d = fields['d'].copy()
    delta = d-dr
    roi = lambda mask: float(np.sqrt(V[mask]@(delta[mask]**2)/(V[mask]@(dr[mask]**2))))
    history = json.loads((ROOT/row['case']/'journal.json').read_text())
    tail = np.array([v['damage_rms'] for v in history[-32:]])
    rows.append(dict(case=row['case'], sweep=row['sweep'], field_stable=row['field_stable'],
        timed_solve_seconds=row['timed_solve_seconds'], damage_rms=row['damage_rms'],
        full_L1=row['full_loss'], physical_AT2=row['physical_AT2'],
        damage_relative_L2_reference=roi(np.ones(len(V), bool)),
        tip_relative_L2_reference=roi(tip),
        tail_damage_rms_mean=float(tail.mean()), tail_damage_rms_std=float(tail.std()),
        reaction=row['reaction'], reaction_relative_error=abs(row['reaction']-ref['reaction'])/abs(ref['reaction']),
        pool_capacity=row['pool_capacity']))
summary = dict(complete=True, trial_load=r['trial_load'], parent=r['parent'],
    same_parent_optimizer_count=14592, tip_evaluation_radius=.024,
    reference=str(refroot), reference_load=ref['load'], reference_reaction=ref['reaction'],
    rows=rows, scope='one_load_2x2_probe_not_a_full_loading_trajectory',
    full_default_audit=json.loads((ROOT/'full_default_audit.json').read_text()))
(ROOT/'review.json').write_text(json.dumps(summary, indent=2)+'\n')
with (ROOT/'comparison.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
lines = ['# Tip density and batch hold comparison', '',
    'All four cases start from the same saved0.0009 network,Adam andphysical',
    'state and solve only0.0010. Budgets:128inner/128outer,unchangedfieldstop.',
    'Times exclude poolconstruction/compilation andendpointfigures.', '',
    '| Density / hold | Outer | Seconds | Damage RMS change | Full L1 | AT2 relative peak | Damage L2 vs reference | Tip L2 vs reference |',
    '|---|---:|---:|---:|---:|---:|---:|---:|']
for row in rows:
    lines.append(f"| {row['case']} | {row['sweep']} | {row['timed_solve_seconds']:.2f} | {row['damage_rms']:.4g} | {row['full_L1']:.4g} | {row['physical_AT2']:.4g} | {100*row['damage_relative_L2_reference']:.3f}% | {100*row['tip_relative_L2_reference']:.3f}% |")
lines += ['', 'Originaldensity usessegmentboost9/tipboost0. Refineddensity uses',
    'segmentboost18/tipboost36,width.024,power2. Positivebaseline anddetached',
    'importancecorrection retained. Same2048entrypoolperdensity,completehalo',
    'padding24787(original)/22450(refined),persistentAdam throughout.', '',
    'Sampling followsglobaloptimizercount,soinitialslot256(k1) and1459(k10)',
    'differ. This is one empiricalpool/parent/seed comparison,not a universal',
    'ranking. No threshold or PDE changes. Fullsampling defaults unchanged.', '',
    'Artifacts: review.json,comparison.csv,comparison.png,and percase saved',
    'fields,network/Adam,configuration,sourcecopies,journal andTensorBoard.', '']
(ROOT/'review.md').write_text('\n'.join(lines))
shutil.copyfile(__file__, ROOT/'review_probe.py')
print(json.dumps(summary, indent=2))
