# Fracture2D artifact locations

All paths below are ordinary directories, with no compatibility symlinks or
SHA-based lookup. Numerical results and TensorBoard event contents are retained.

| Role | Canonical directory |
| --- | --- |
| Accepted MLP PINN trajectory | `/root/autodl-tmp/tf-logs/fracture2d/pinn_mlp_baseline/` |
| Provisionally accepted history-conditioned PINN | `/root/autodl-tmp/tf-logs/fracture2d/pinn_history_conditioned_baseline/` |
| Accepted cold history-conditioned FEM-energy PINN with scaled stopping | `/root/autodl-tmp/tf-logs/fracture2d/pinn_history_fem_scaled_stop_baseline/` |
| Matched local P1 AT2 direct reference for the FEM-energy baseline | `/root/autodl-tmp/tf-logs/fracture2d/runs/fem_damage_reference_full/` |
| Historical trials, diagnostics and plots | `/root/autodl-tmp/tf-logs/fracture2d/diagnostics/` |
| Default outputs of new solver runs | `/root/autodl-tmp/tf-logs/fracture2d/runs/` |
| Current local FEM comparison | `/root/autodl-tmp/data/fracture2d/fem_reference/` |
| Current nonlocal damage direct reference | `/root/autodl-tmp/data/fracture2d/nonlocal_reference/` |
| Shared prepared mesh and nonlocal operator | `/root/autodl-tmp/data/fracture2d/geometry_assets/` |
| Older point-anchor FEM and nonlocal-mechanics references | `/root/autodl-tmp/data/fracture2d/legacy_references/` |

Each accepted PINN directory contains its own `events/`. Diagnostic event trees
are likewise attached to their diagnostic run, where an existing run could be
identified. Older pre-consolidation event trees are under
`diagnostics/legacy_events/`. No fracture2d events belong directly under the
parent `/root/autodl-tmp/tf-logs/` directory.

The current three-way comparison uses local P1 mechanics/local gradient AT2
for `fem_reference`, and local P1 mechanics/nonlocal AT2 direct damage for
`nonlocal_reference`. The original nonlocal PINNs use the latter model with
neural damage. The accepted `pinn_history_fem_scaled_stop_baseline` instead
uses local P1 damage energy and must be compared with the matched reference
`runs/fem_damage_reference_full`.
`legacy_references/fem_point_anchor` is an older independent FEniCS mesh;
`legacy_references/nonlocal_mechanics` uses the older nonlocal mechanics route.
These older archives are not replacements for the current comparators.
The public `solver.py` remains the independent FEniCS entry and writes new
outputs under `data/fracture2d/runs/fem_point_anchor/`.

A `baseline` name records the user's provisional engineering acceptance. It
changes neither saved convergence flags nor the actual completed load range.
Run-specific limitations are in the baseline's `baseline.json`/`README.md`
and the [active task](../../docs/tasks/nonlocal-pinn-architecture.md).

The 2026-10-16 migration map, metadata backup and verification receipt live in
`tf-logs/fracture2d/_layout_migration_20261016/`. Current configs and checkpoint
path metadata use canonical locations. Old TensorBoard text/config payloads,
frozen source snapshots, and dated decision history retain their original
provenance; use the migration map to resolve their pre-cleanup paths. Historical
standalone scripts are not maintained entry points after relocation.

## Current FEM reference boundary and resolution

The saved `fem_reference/identity.json` declares engineering volume grips,
shared with the current nonlocal direct and PINN mechanics. Material occupies
0 <= y <= 1. Undamaged elastic grip strips extend to y=-0.048 and y=1.048,
with spacing0.004. Every added lower-grip node has uy=0; every added upper-grip
node has uy=U. Horizontal displacement is fixed only at material node0,
(x,y)=(0,0); other horizontal degrees of freedom are free. The material/grip
interfaces themselves are connected FEM nodes, not directly prescribed uy.
Other exposed boundaries, including notch faces, carry natural zero traction.
Local AT2 damage is solved only on material cells, with natural zero-flux
boundaries; the geometric notch is not a prescribed d=1 damage strip.

The actual material mesh is `cloud_h004_api.npz`: 71,307 nodes and 141,366
linear triangles, versus18,367 nodes in the older h=0.008 cloud. Its embedded
Gmsh metadata sets both h_far and h_near to0.004: this is a globally fine mesh,
not special crack-band refinement. Including grips gives77,331 nodes and
153,366 triangles. Physical l0=0.024 gives approximately six nominal mesh
spacings. All three current methods share the material nodes/triangles.
This is a finer discrete FEM comparator, not a demonstrated mesh-converged
continuum reference. This FEM path uses uniform load increments1e-4; the
nonlocal reference's additional critical-load steps are loading refinement,
not additional spatial refinement.

## Optional FE-energy damage diagnostics (2026-10-18)

`tf-logs/fracture2d/diagnostics/history_fem_energy_20261018/` contains the
bounded fixed-H conditioned-network experiment, its inputs, source snapshots,
direct evaluation fields and event files. It is a diagnostic, not an accepted
baseline or a coupled cold trajectory. `postfracture/` extends the warm fit to
the old local FEM's genuinely spanning H at U=0.0055. The initial detached
launch's incomplete directory is retained as `postfracture_attempt0001/`.

`tf-logs/fracture2d/runs/fem_damage_reference_pilot_20261018/` and
`tf-logs/fracture2d/runs/pinn_history_fem_cold_pilot_20261018/` are the new
matched direct/neural 0→0.0010 pilots, completed serially after that fit.
The neural pilot starts with entirely fresh weights and Adam; it does not
inherit the fixed-H fit. Check each result.json completion/status before using
it. Both have ten saved endpoints; this short elastic pilot is not a completed
fracture trajectory. Comparison details are saved in the diagnostic root's
`cold_comparison.json`. Existing accepted nonlocal/reference names are unchanged.

## Full FE-energy cold trial (2026-10-18)

`tf-logs/fracture2d/runs/fem_damage_reference_full/` and
`tf-logs/fracture2d/runs/pinn_history_fem_full/` are the new matched0→0.0060
direct/neural trial. The direct arm runs first; a fresh neural cold arm follows
only after its successful completion. Both retain the short pilot's numerical
settings, with only the terminal load/output paths changed. They are
experimental runs, not newly accepted baselines. Event files live in each
root/events. Prior pilots and fixed-H data remain separate.

`tf-logs/fracture2d/diagnostics/history_fem_full_launch_20261018/` stores the
serial job receipt, exact configs, launcher and per-arm logs. Use job.json
and each run's result.json to distinguish a queued arm, active loading,
budget failure and completed trajectory. No full completion is implied by
the directory name.

The direct arm completed0→0.0060. The original neural arm exhausted its
outer budget at0.0049; see the separate refinement/continuation below.

## FE-energy continuation after low-LR finishing

`tf-logs/fracture2d/runs/pinn_history_fem_lr3e5_continuation/` inherits the
accepted prefix0→0.0048 and the verified0.0049 refinement, then continues
0.0050→0.0060 at fixed Adam3e-5. Its `fork_manifest.json` records the manual
learning-rate change, retained Adam56704/bank98, replaced endpoint-H snapshot
and exact state checks. It is a continuation of a cold-origin trajectory;
it is not a new all-low-rate or automatic-schedule cold run.

The original full run and finishing control remain separate. Load0.0049's
trajectory count is136outers, including the original128 plus8finishing;
the extra32-outer control arm is diagnostic work. New TensorBoard events are
in the continuation root/events, with refined0.0049 fields and subsequent
load fields/curves. `continuation_job.json` and `result.json` record progress;
the new run reuses `fem_damage_reference_full` for same-load evaluation.

This continuation subsequently exhausted1024outers at0.0055. It saved a
synchronized spanning endpoint but did not pass fieldstop or reach0.0060.

## FE-energy finishing controls

`tf-logs/fracture2d/diagnostics/fem_energy_lr_finish_0049_20261018/` and
`tf-logs/fracture2d/diagnostics/fem_energy_lr_finish_0055_20261018/` hold the
completed matched same-state LR controls. Each contains source_config.json,
optimizer_equivalence.json, seed_receipt.json, job/result.json, run.py,
comparison.png, damage.png and per-arm events/fields/net/Adam checkpoints.
They do not advance loads or overwrite source trajectories.

At0.0055, both1e-5 and3e-6 pass original fieldstop after4extraouters from
the1024outer seed;3e-5 stillfails after32. Their field errors remain about
6.58% despite close synchronizedreactions. Final manifests explicitly remain
diagnostic: a later continuation must reconstruct the normal endpoint and
replace the historybank's lastendpointH, preserving allolderentries/moments.

## Fresh cold path with128outer rate blocks

`tf-logs/fracture2d/runs/pinn_history_fem_cold_budget128/` starts a new physical
and neural trajectory fromzero, using `configs/train_history_fem_cold.py`.
It targets0.0060; its name does not imply completion. Unlike earlier manual
continuations, it restores no oldloadcheckpoint. It uses128outer blocks,
lowers Adam1e-4→3e-5→1e-5 after failedbudgets, and preserves the rate/moments
across loads. Finalrate budget failure stops advancement. This is a trial of
the budget policy, not a newly accepted completed baseline.

Exact launch config, wrapper, initialization verification and job.json are in
`tf-logs/fracture2d/diagnostics/history_fem_cold_budget128_launch_20261018/`.
The zero endpoint has Adamcount0/bank0/q=d=H=0; its network/Adam arrays match
the original fresh seed. Physics, architecture, bootstrap, inner updates and
fieldstops match the original FE fulltrial. Event files live in runroot/events;
`train/outer_iterations` and `train/learning_rate` report endpointwork/rates.

This cold trial stopped at0.0055 after256outers (two remainingrateblocks),
with status`budget_exhausted`, reaction328.097 and damageL2error51.79%.
It completed fieldstable endpoints through0.0054 and did not reach0.0060;
the final saved field must not be presented as a converged fullfracture result.

## Fixed-load progress-block continuation

`tf-logs/fracture2d/diagnostics/fem_energy_progress_blocks_0055_20261018/`
continues the incomplete cold0.0055 endpoint with preserved Adam105344,
historybank110 and preceding0.0054historyfloor. It retains1e-5 while the
own synchronizedreaction drops by morethan1% per128outerblock; a plateau
lowers to3e-6. This is an empirical localdiagnostic with atmost4blocks, not
a new global schedule. The original fieldstop still decides convergence;
reference fields do not choose rates. No load is advanced by this driver.

Each block saves network/Adam, `raw_fields.npz` (actual staggercarry) and
`fields.npz` (independently synchronized diagnosticfields). Boundary mechanics
does not overwrite the rawcarry or commit newhistory. Check job/result.json
for execution status AND convergence; diagnostic completion can still mean
budget exhaustion. TensorBoard lives in this root/events; policy.json,
checks.json, source_config.json and run.py retain exactprovenance. The source
run and historybank remain untouched. Finaldiagnosticmanifests cannot be
used as normal loadendpoints without explicit state/bank reconstruction.

The trial completed4blocks/512extraouters, reaching totalouter768 at0.0055.
Finalsynchronizedreaction1.54923 vsreference1.53605, damageL2error6.663%,
with a straightspanningband. Originalfieldstop stillfails; reason is
`diagnostic_budget_exhausted`. Allblocks retained1e-5, so3e-6 was not tested
in thistrial. The saved finalcheckpoint is `outer_0768/`; no newloadsadvanced.

The user subsequently accepted the physical crack morphology/path andforce
of `outer_0768/` as an acceptable engineering result. Its strict fieldstop
status and incomplete0→0.0060trajectory remain recorded. The separate
`stopping_tolerance_audit.json` replays saved257–768metrics at candidate
tolerances; it neither creates an accepted695checkpoint nor changes a run.

## Forward validation of damage-scaled stopping

`tf-logs/fracture2d/diagnostics/fem_energy_damage_scaled_stop_0055_20261018/`
tests the first actual stop under `(dmax+.01)*1e-3`, retaining displacement
limit1e-4 and threeconsecutivepasses. It restores rawouter384/net/Adam121728
from the progress-block trial, uses the original bank110/0054floor and keeps
1e-5/128innerupdates. It does not advance loads or initialize a new coldpath.
Atmost128additionalouters; the predicted468candidate is not forced.

The final `stop_XXXX/` checkpoint stores rawcarry and synchronizedphysical
fields, network/Adam and an explicitlydiagnosticmanifest. Rootdamage.png
compares directFE, candidate anduseraccepted768. Exact policy, checks and
seedreceipt are retained with run.py/job/result.json; events live inroot/events.

Completed at the firstactualcandidate, outer468 (84newouters). Storedstate
`stop_0468/` has synchronizedreaction1.83023 vsreference1.53605 anddamage
relativeL2error10.36%. Crackisstraight/spanning, withmorebanderror than the
accepted768state. Materialedge diagnostic is in connectivity_check.json.
This is a faster, lessaccurate stoppingcandidate; no newcoldpath ornormal
0055endpoint was committed. Training/forceverification receipts remain separate
from acceptance of a fulltrajectory.

## Fresh damage-scaled cold experiment

`tf-logs/fracture2d/pinn_history_fem_scaled_stop_baseline/` is the accepted
independent 0→0.0060 baseline using `configs/train_history_fem_cold_scaled.py`.
Userformula applies at everyouter, withu1e-4/3passes. Ownreactionprogress
is checked every128outers; retain the rate whiledecreasingmorethan1%,
otherwise lower1e-4→3e-5→1e-5. Atmost4blocksperload; finalrateplateau and
budgetfailure stop, never advance silently. The rate andAdam survive across
loads. This accepted optional profile does not replace the nonlocal default.

Launcher/config/job/initialization_verification.json live in
`tf-logs/fracture2d/diagnostics/history_fem_cold_scaled_launch_20261018/`.
Verified actualzero physicalfields, Adam0/bank0 andsamefreshnetwork/state;
no468warmstart. Runroot/events records fields, curves, active stoppingtolerance,
learningrates andper-loadoutercounts. Check job/result.json foractualstatus.

Completion audit (2026-10-18): all 60 cold-start endpoints through 0.0060 are
`field_stable`; loading/reporting wall 3423.28s, total 930 outers and 119936
Adam updates. The critical 0.0055 step took 482 outers and produced a straight
spanning band at the same load as the matched FE direct reference. Damage
volume-weighted relative L2 errors are 1.359% at 0.0055 and 1.130% at 0.0060.
Tail reactions are 1.76914 vs 1.53605 and 1.05458 vs 1.17955 respectively,
so their relative errors remain 15.17% and 10.59%. All rates stayed at 1e-4;
the run does not demonstrate a benefit from rate reduction. The user accepted
it as a baseline after completion; it does not replace the public default.
Figures and TensorBoard events remain in this run root; no further loads are
queued. Detailed endpoint comparisons are in the active architecture task.

Accepted-baseline promotion (2026-10-18): the original directory
`runs/pinn_history_fem_cold_damage_scaled` was moved to
`pinn_history_fem_scaled_stop_baseline`, with no compatibility symlink.
Path-bearing JSON, including endpoint parents and replay stores, now uses
the canonical directory. Original JSON backup and validation receipt are in
`diagnostics/fem_history_baseline_promotion_20261018/`. Event files, arrays,
network/Adam states and frozen source snapshots were preserved; frozen source
and raw run.log retain their historical path text. Baseline acceptance,
configuration characteristics and limitations are recorded in the run's
`baseline.json` and `README.md`.
