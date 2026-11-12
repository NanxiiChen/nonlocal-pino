# Unify rolling core/halo sampling across the five PINOs

User goal: provide a successful rolling core/halo route for every principal
example, including those where earlier sparse sampling underperformed full
coordinates. Investigate whether parameterized geometries benefit from two
strata: domain interior and near-boundary cores.

Acceptance: audit the actual five defaults and previous controlled evidence;
design source-traceable sampling/inclusion weights that preserve the intended
PDE objective, distinguishing wall-adjacent volume rows from exact boundary
loss queries; fixed-shape complete stencils and rolling pools; matched cold
accuracy/time/memory tests on the current card; transparent retained failures
and no unmeasured default promotion. Reuse approved local reference cohorts
where possible; numerical source/provenance and reproduction paths required.

Constraints: pure physics training; no reference-label adaptation. Preserve
the original geometry/IC priors and hard/soft BC semantics. Current GPU must be
remeasured, historical timing is not comparable. Do not resume the paused old
high-mobility process618216. No broad cleanup, GPU change or research publication
is requested. User-authorized master-figure commit/push is a separate release.

Context: `docs/tasks/archive/pino-spatial-sampling-study.md`, particle online
and full-grid studies, current rolling Melting/Electropolish defaults, accepted
Spinodal/channel full-coordinate recipes and recent master figure.

Status: implementation and serial experiments in progress. Blocker:none.
Current resume point is the latest dated section below; earlier queues and
plans are preserved as history, not as live process claims.

## 2026-11-12 audit and first implementation

Publication is complete: contentfd7ed85/mainf88709c, verificationmain5db8745;
Pages43181750462 succeeded. No new research files have been committed.
Current GPU: RTX4090,49140MiB, UUIDa7b60c6d-62ad-e7f5-f91f-b750cc853621;
17GiB free disk at start. No cleanup performed.

Existing accepted routes: Electropolish rolling32/K256, Melting rolling32/K256,
and ceramic rollinggeometry/8spatialsets/K256. Channel/full193x81 and
Spinodal/full1536padded with dense residual metric need separate qualification.
The full geometry encoder contexts and BC loss rules are already retained in
channel compact mode. Lost PDE coverage is distinct from lost conditioning.

New opt-in sampler code:
- shared `training/spatial_sampling.py`: CPU inverse-CDF strata and actual
  segment-distance bands; no operator/kernel/physical fields changed.
- Channel `train_stratified_spatial.py`: small mapped local groups, inverse-CDF
  stratification of group area, optionally wall/bulk. All nonzero derivative
  and BC supports retained. Existing full/scatter/tile paths preserved.
- Ceramic `train_physical_spaced_w256_stratified.py`:64wall+192bulk cores,
  probability strata in physical-coordinate order. Known input generation
  remains online; source banks must explicitly match the sampler selector.
  No training launched yet; measured old50/50 failure retained.

Measure identity: for stratum S with mass M_S, draw groups by their mass/mass(S),
then estimate each group's area mean. Average all group slots and multiply by
M_S/M_total. Equal-probability CDF strata preserve this group expectation when
all slots are retained. Full-size tiles use their exact nodal weights; short
or clipped tiles draw the fixed row count by nodal mass. Per-stratum weight
sum equals M_S/M_total; these are volume-loss weights, never boundary measures.
Legacy tile estimates retain their existing inverse-inclusion weights and are
not renormalized by a random sampled sum. No reference fields enter choices.

Support preflight: `docs/tasks/pino_unified_sampling_preflight.py`, receipt
`/root/autodl-tmp/tf-logs/pino-unified-core-halo-20261112-01/support-audit.json`.
32saved initial/final geometries,16draws each,7schemes;672operator-row checks
are bitwise exact, unchanged boundary supports/context checked. Median halos:
scatter51214638.5; oldtile5124972.5; stratified4x4bulk5126753;
stratified4x4wall5126599; stratified4x4wall10249172.5. These exclude padding.
Synthetic localized quadrature RMSE decreases fromoldtile.0410to4x4bulk.01535
or4x4wall.01613; NOT a solution-error result or proof of a wall benefit.
Full-domain mass/resolvent dependencies prevent claiming sparse Spinodal
network evaluation without a formulation change. Its exact row-only benchmark
is still to implement; do not quietly truncate the dense metric.

Six focused CPU tests passed (full/compact physical loss+parameter gradients,
legacy sampling, polygon bands and random quadrature checks). Additional19ceramic/shared-sampler/online checks passed after its new opt-in
path. Four Spinodal row/legacy temporal tests pass, including exact full
parameter-gradient identities for a genuinely dense off-diagonal metric.

## Active finite resource queue

Owner583327, child first583332; launched2026-11-12T08:36:58UTC via
`docs/tasks/pino_unified_sampling_queue.py` under `.venv`, detached nohup.
Journal `/root/autodl-tmp/tf-logs/pino-unified-core-halo-20261112-01/probe-journal.json`.
Five serial60-update channel probes: full, oldtile512, strat4bulk512,
strat4wall512, strat4wall1024. Same42cold state,B8/Nt36/block6,0.90allocator.
Explicit capacities15633/8192/10240/10240/12288; fail on overflow, no resampling.
Journal/logs retain command, PID, exit, timing and peakmemory. No long accuracy
run has yet been queued. After completion: inspect actual resource/parity,
select matched longer controls; prepare ceramic banks only with fixed IC/BC
parity and test allocation effect separately. Keep complete same-card baselines.

## Exact resume after this session

The first shell-detached owner583327and its child disappeared before producing
a run directory or any training log; journal01is explicitly preserved as
interrupted_before_training. No numerical result is inferred from it. Relaunch
uses subprocess start_new_session, owner584272/session584272, child584279.
Active journal now:
`/root/autodl-tmp/tf-logs/pino-unified-core-halo-20261112-02/probe-journal.json`.
Support receipt was copied unchanged. Verified owner survives tool sessions.
Full60-update probe has completed compilation and reports a finite first loss
23.466957; five-stage queue remains running. Initial GPU compilation includes
a2minGEMM-autotune warning; report synchronized warm updates separately.

Spinodal diagnostic prototype: `operator_spatial.residual_rows` attaches
unbiased freshly sampled residual row IDs/weights. `operator_training.make_loss`
selects rows AFTER the dense metric, preserving all physical nodes and exact
mass. No canonical trainer samples these rows by default. Four CPU checks pass
for complete coverage and the exact weighted expectation of all one-row
losses/parameter gradients. `docs/tasks/pino_spinodal_row_probe.py` provides
CPU preparation and separate GPU60-update timing at rows0/full,256,512. It is
NOT launched or queued yet, and its fixed16input benchmark explicitly excludes
the full rolling bank memory. Run serially after the channel probe owner exits.

Next steps:
1. Inspect journal02and eachsummary, confirm all5probes and exact cold states;
   check fixedpoint-capacity overflow, timing, producer wait, finite weights.
2. Use those measurements to choose channel cold longer full/sparse controls
   with the same original equation/IC/BC/context and sampled time/geometry
   streams. Existing2k results are screening, not a converged ranking.
3. Prepare ceramic stratified inputs with seed1307500965and compare every
   fixed IC/BC/context array to the existing64bank. Add no-wallCDFcontrol to
   isolate coverage from allocation; all3cold20k runs should share seed/runtime
   streams, with existing64local evaluation at final weights if feasible.
4. Run Spinodal exact row timing before deciding whether any long row-only
   control is warranted; do not describe its global dense support as compact.
5. Regular Melting/Electropolish rolling paths are already accepted; retain
   their defaults and reuse existing accuracy evidence, remeasure timings
   only for a final common-card table. No default/README promotion before results.

Session changes: shared sampler; channel opt-in stratified pools/preset and
source snapshot; ceramic opt-in rule/preset/schema/provenance; Spinodal exact
row prototype; tests and finite resource scripts. Other pending research is
preserved. Focused tests and changed-module syntax passed; gitdiff--check passes.
Memory updated: this task, PROGRESS and append-only DECISIONS. Goal remains
active; no research Git operation, cleanup or local/FEM solve performed.

## 2026-11-12 completed probes and active followup

All five journal02 channel60-update arms exited0; owner584272 is finished.
Warm step including batch preparation / peak active GiB:
full3.27739/35.71445; oldtiles5121.39342/20.73908;
strat4bulk5121.67371/26.41939; strat4wall5121.67645/26.41982;
strat4wall10241.9777/29.938. Cold model and complete optimizer bytes match.
No accuracy ranking follows from these probes. Long paired accuracy runs are
still required; preserve all encoder context and original BC losses.

Followup owner600703 is live and started the fixed16-input Spinodal full/256/512
row diagnostics after the resource owner finished. Root and journal:
`/root/autodl-tmp/tf-logs/pino-unified-followup-20261112-01/journal.json`.
`docs/tasks/pino_unified_sampling_followup.py` runs serially and stops on an
explicit failed stage. After the row probes it prepares two64geometry banks
and verifies fixed-input parity, then cold20k ceramic area/CDF/wallCDF controls,
each with online-stream audit, native/local4 and final-only64local evaluation.
Runtime seed42; preserved Adam20k/B4/Nt32 and original physical configuration.
Near-wall allocation64of256, versus the old failed128of256. A CDF-only arm
separates spatial coverage from wall allocation. Existing64local inputs are
excluded from training. Reference labels never enter rolling builders/losses.
New final64 metric is per-channel Euclidean space-time norm, then channel and
sample means; physical-volume/time-weighted scores remain supplementary.
Thirteen online-validator tests passed after conditional source-schema support.

Spinodal compact candidate: `operator_potential.py` uses the independently
derived conservative auxiliary coordinate (see DECISIONS). Its2CPUtests passed
in69.259s, including full/compact neural fields, parameter gradients, hardIC,
volume mass and gauge invariance. The old session result could not be recovered;
the recorded successful rerun is `/tmp/pino-conservative-potential-tests.log`.
`operator_core_halo.VolumeGroups` builds input-only Cartesian groups with
optional actual contour-distance wall bands and volume-correct CDF strata.
Two additional tests pass: a nontrivial9-node chain needs only3query nodes for
its first core, and independent Monte Carlo means match the full volume norm.
Real16geometry support audit is running through
`docs/tasks/pino_spinodal_support.py`; result is under the followup root.
This is an experimental output/loss change, not an old-metric truncation or a
validated SOTA. Canonical training integration and accuracy remain outstanding.

Next: inspect real support audit and ongoing diagnostic stages; prepare matched
channel long controls; integrate/test a useful compact potential path only after
the support and mathematical checks. Keep the complete five-case goal active.
No further Git operation or cleanup. Current disk has14GiB free; CPU RAM ample.

## Latest resume: canonical compact CH implementation and queued accuracy

This goal turn made implementation/validation progress and checked live handles;
the preceding publication-only turn did not advance sampling. Do not treat the
research goal as complete. Remaining acceptance includes actual trained
accuracy, long-run complete support, common-card regular-case timing, and only
then choosing/documenting successful five-case rolling defaults.

All3Spinodal row-cost probes passed: full0.099812s/7.750GiB,256rows0.080076s/
7.351GiB,512rows0.079467s/7.351GiB. They still query the full grid and include
only16fixed known-input geometries, excluding the1024rolling-bank cost.
`spinodal-two-hop-support.json` under followup01 records4096payload checks:
16real domains ×16sampling settings ×16draws. Independent NumPy reconstruction
of phi and mu rows is bitwise identical; mass and physical weight sums pass.
For128cores/two64-row groups/cellwidth.12/wall_groups1, median query576nodes
(46.90%of physical nodes), maximum815in this finite audit. Do not interpret
this as a bound over the entire prior or as neural accuracy. Explicit pilot
capacities inner512/query1024 raise on overflow rather than bias/truncate.

Canonical `examples.confined_ch2d.train` now accepts two opt-in presets:
`train_operator_potential.py` (full) and `train_operator_potential_spatial.py`
(rolling128cores). Both preserve the exact original specification and online
1024geometry prior; coldSOAP/B16/Nt16+0/width256,5000pilot updates. They explicitly
select formulation=conservative_potential and metric=identity. Legacy metric
branches and SOTA defaults are unchanged. Physics remains the same regional
CH system; the output coordinate and residual norm change as documented above.

The potential loader/workers omit the dense metric. Decoder queries include
every two-hop dependency; full domain/boundary encoder tokens are retained.
Background spatial pools hold4sets per cached geometry slot, refresh by50-step
epochs, prefetch the next deterministic batch and invalidate on exact online
geometry replacement. Fixed shapes, weights and core IDs are recorded. There
is no prediction-based sampling and no global phase mean in this branch.
Off-graph phase evaluation explicitly rejects missing physical support.
Continuation for this experimental coordinate is explicitly unsupported until
a paired spatial recovery implementation is validated; pilots are cold only.

Focused checks: original2potential algebra/gradient tests;2initial grouped
sampling tests;3pool/support/quadrature tests plus4legacy online tests passed.
The fifth legacy test initially failed solely on an altered exception message;
restore its eight-input wording, then that test and the new canonical potential
dispatch check both pass (38.524s). No unchanged expensive suite repeated.
Live CPU draw0 independently builds old/new online inputs: all seven shared
arrays exactly equal; no new PDE/FEM solve.1245physical nodes; four exact
supports use557–670queries. Receipt:
`/root/autodl-tmp/tf-logs/pino-unified-accuracy-20261112-01/fresh-input-parity.json`.

Followup owner600703 is confirmed live, currently ceramic area20k training.
Both new64geometry banks built, and all fixed arrays/padded shapes match the
area bank exactly (`ceramic-input-audit.json`). The first baseline has reached
early development evaluations only; no new final64accuracy result yet.

Accuracy owner632166 is confirmed live/waiting for followup01 to complete.
Journal `/root/autodl-tmp/tf-logs/pino-unified-accuracy-20261112-01/journal.json`;
source `docs/tasks/pino_unified_accuracy_queue.py`. Planned serial stages:
1. Canonical full/bulk/wall potential60-update probes at production batch/shapes.
2. Cold5k full and wall potential pilots, native8test/16training monitors;
   exact cold-state/geometry/time/reference pairing audit.
3. Cold20k channel bulk512,wall512,full, with original LR and6-time blocks;
   each online-input audit and3reserved local-FEM evaluation, then strict pair.
The channel stages alone are roughly37hours at measured warm costs, plus
compilation/evaluation. This is a time estimate, not completed work.
No GPU overlap, no restart on observation timeout, no implicit failed-stage
retry. Inspect the live journal/handle before action. The owner stops on error;
completed stages and partial artifacts remain for diagnosis. Ordinary regular
case60-step timings and any promotion are still outstanding.

Session diff: experimental potential/group modules and presets, canonical
loader/field/loss dispatch, optional online CPU-install callback, timing and
spatial provenance; tests, real-support script and dependent accuracy owner.
Changed-module syntax and gitdiff--check pass. No Git mutation or cleanup;
all unrelated research is retained. Memory:PROGRESS,DECISIONS,this task.

Final resource audit also passes for all five completed probes: byte-identical
cold network/full optimizer and exact indices/times/draw_ids/amplitudes arrays.
Receipt:`pino-unified-core-halo-20261112-02/resource-parity.json` under tf-logs.
Both subsequent owner command lines were rechecked live before yielding.

## Latest resume: support bounds, lazy pools and authorized source checkpoint

The user now explicitly authorizes commit and push of the current work. This
supersedes the earlier no-publication constraint for this checkpoint only.
The five-case research goal remains active; no new default is promoted.

Support validation covers1032existing plus64fresh input-only geometries. The
maximum complete two-cell union is428inner/952query nodes for bulk and419/940
for wall-split groups, inside the declared512/1024capacities. This is a finite
cohort bound; future overflow still raises. Receipt:
`/root/autodl-tmp/tf-logs/pino-unified-accuracy-20261112-01/two-hop-capacity-bound.json`.
Lazy pool construction now prepares only actually selected deterministic sets
and reuses geometry partitions.90payload arrays match the frozen eager source
across spatial epochs and input installation; CPU batch preparation falls from
0.23–0.25s to0.029–0.054s. This is not an end-to-end GPU speed claim. Receipt:
`/root/autodl-tmp/tf-logs/pino-unified-accuracy-20261112-01/lazy-spatial-validation.json`.
The real heterogeneous two-stream encoder also passes full/compact parameter-
gradient equality, with nonzero gradients through both input streams.

Followup owner600703 completed area20k and online/native/local64 audits. The
4development physical-weighted phase error is0.32376467; final64Euclidean
space-time phase mean0.31597469, channel mean0.32434125. This is a failed neural
control, not an accuracy improvement. A historical same-seed42 run at
`/root/autodl-tmp/tf-logs/particle_corrosion2d/pino-rolling-ad-cold-20261111-01-particle_corrosion2d-s42-nonlocal`
has identical initial network/optimizer bytes and4case phase0.03180205. Its
config additionally enables `operator.method_comparison=rolling_spatial_v1`.
The reason for the divergent trajectory is not established. Next diagnosis:
compare actual saved core/time/geometry payloads and first loss, then inspect
the comparison branch's added `pde_core_points` input and frozen source. Do not
attribute this to seed variability or alter live sources mid-pair.

The CDF arm is running under the same followup journal; wallCDF follows.
Accuracy owner636249 waits at
`/root/autodl-tmp/tf-logs/pino-unified-accuracy-20261112-02/journal.json`.
The old owner632166/journal01 was superseded while waiting with no child or
completed stage; retain that receipt and do not restart it. Queue02 adds two
regular-domain60-update resource probes before the planned Spinodal potential
probes/full+wall5k pilots and channel bulk/wall/full20k controls. The paused old
high-mobility processgroup618216 remains untouched.

Checkpoint scope: pending study implementations/presets/tests, scalar provenance
and reproduction scripts, approved cost-error assets and example documentation.
No training data/checkpoints are added. Memory updated:PROGRESS,DECISIONS,this
task. Release validation and exact subsequent action are recorded below.

### Source-checkpoint validation

Static validation passes for61changed/new Python modules and11JSON records;
local Markdown destinations resolve, and the existing report builds all9pages.
`git diff --check` passes. The focused CPU run exercised58tests:57passed and
one FP32-overflow fixture initially failed after a preceding canonical test
enabled global FP64. The isolated test passes. Add an explicit temporary FP32
context to that fixture, then rerun all6extreme-training tests under an outer
FP64 context: all pass, and outer FP64 is restored. Production numerical code
is unchanged by this test correction. Combined coverage therefore passes all
58distinct checks; no new GPU job or reference solve was used for publication.

Commands use `JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` and
`PYTHONPATH=tests .venv/bin/python -m unittest -v` with the changed tests plus
legacy online validators. Full output: `/tmp/pino-research-checkpoint-tests.log`;
focused precision check: `/tmp/pino-overflow-precision-regression.log`; website
build: `/tmp/pino-checkpoint-site-build.log`.

The requested checkpoint retains the completed-study artifacts and approved
PNG/PDF, with no weights/raw fields/logs in Git. The active experiment remains
incomplete. Next: diagnose the same-seed area discrepancy and inspect the
existing serial owners; do not restart or promote a run from its timing alone.
The Git history and remote main ref identify the published checkpoint.

## Latest resume: exact-input audit and queued GPU execution replay

The preceding checkpoint turn made progress: publishedmain833880e, targeted
regressions, documentation and verified continuation of existing jobs. This
turn adds authoritative CPU evidence and a serial GPU diagnostic; goal remains
active and no default is promoted. No new Git publication is authorized.

`docs/tasks/pino_ceramic_repeatability.py` now reconstructs the exact first batch
and compares old successful/current failed area runs. All39saved arrays agree:
3sampling arrays,17initial arrays,2online identities and17final arrays. Cold
model/optimizer bytes and399physical refresh events agree, excluding worker
timing/cache data. Actual old core coordinates/weights match new initial-bank
replay for the first50updates. Source identity holds for network/loss/online/
geometry/mesh/operator/shared neural/optimizer/runtime functions. The extra
method-comparison coordinate does not change the full-shape CPU StableHLO:
959374bytes in both cases, identical. First loss differs by at most3.0518e-5.
This does not establish the GPU cause or equate journal identity to observation
of every intermediate payload. Receipt and first_batch.npz/MLIR live at:
`/root/autodl-tmp/tf-logs/pino-unified-repeatability-20261112-01/`. CPU job23577
finished0; log `/tmp/pino-repeatability-audit.log`. The script never trains.

`docs/tasks/pino_ceramic_update_replay.py` will execute the identical cold update
four times per plain/marked/plain-again group, retaining initial model, complete
optimizer, weights, actual batch and times each time. Ordinary and deterministic
policies run in separate processes. Compare every output leaf, loss component
and adaptive weight; this is one-update repeatability, not a full-run cause
certificate. No GPU execution of this diagnostic has occurred yet.

Verified empty waiting owner636249 had no stages/child and was deliberately
terminated; journal02 records superseded_before_any_stage and successor03. This
was a plan revision prompted by new evidence, not a retry after observation
timeout. New owner646658 is confirmed live/waiting at:
`/root/autodl-tmp/tf-logs/pino-unified-accuracy-20261112-03/journal.json`.
It depends on the same live followup owner600703. The two replay stages precede
the unchanged regular resource, Spinodal probe/5k and channel20k stages. Preserve
journal01/02 as history; do not restart them. Child639573 is the current CDF
training process; at10k its4case physical-weighted phase error is0.04297983, versus
area0.310570at10k. This remains a monitor result, not the final64case ranking.

`docs/tasks/pino_spinodal_replacement.py` compares frozen eager/current lazy
pools using genuinely different saved geometries/operators: physical node
counts1162/1275replace with1214/1205. Replace after prefetch, include repeated
slots and50-step epoch changes. All90payload arrays and complete encoder inputs
match, with correct version increments. Receipt:
`/root/autodl-tmp/tf-logs/pino-unified-spinodal-replacement-20261112-01/validation.json`.
Both source versions are frozen alongside it. CPU job93326finished0. No PDE or
FEM solve was required. This strengthens earlier initial-field-only replacement
checks and does not establish neural accuracy.

Changed files this turn: three diagnostic/reproduction scripts, the waiting
accuracy queue stage plan, PROGRESS,DECISIONS,and this task. CPU audit/replacement
executions, changed-script syntax and diff checks pass; the output-comparison
helper also passes identical/different synthetic-array checks. No production
loss/model/sampling source changed during the live matched ceramic trials.

Next: inspect live CDF/wall20k stages and their64case final evaluation; then the
queued same-state GPU replay. Use its evidence to qualify same-seed variability.
Follow through on Spinodal full/compact5k and channel full/bulk/wall20k accuracy;
only promote routes after complete support, physical metrics and costs agree.
The original five-case objective remains incomplete.

## Latest resume: common regional64 diagnostics

The previous goal turn made progress through exact-input/lowering audits and
real geometry replacement, then verified live training. Current turn adds a
geometry-only regional evaluation, not a new training measure.

`docs/tasks/pino_regional_metrics.py` reads saved predictions/references and
partitions all spatial nodes by distance<=delta(.026)to the actual reference
polygonal particle walls. Bulk is the exact complement. The same physical
geometry and mask are used across schemes. Report per-channel space-time norms,
then channel/sample means, with both Euclidean and physical-volume/time norms
and additive squared-error contributions. Three independent unit checks pass
for reduction order, weighted integrals and disjoint complete partitions. All
64global scores recover the existing evaluator. No mask uses predictions or
evolved reference fields. No inference or PDE/FEM solve was performed.

For failed area20k, Euclidean phase near-wall/bulk means0.31224443/0.31728146.
The wall band has mean13.2974%physical volume and12.7925%of physical squared
phase error; this failure is not isolated to the boundary. Receipt:
`/root/autodl-tmp/tf-logs/particle_corrosion2d/pino-unified-followup-20261112-01-area/evaluation64/regional_metrics.json`.
The explicit-schema replay in `/tmp/pino-area-regional-replay.json` exactly
matches all64records/statistics.

Historical successful same-seed area20k has Euclidean64phase0.04120417overall,
0.05753922near-wall and0.02870420bulk; channel-mean overall0.04554058matches the
published violin. Receipt:
`/root/autodl-tmp/tf-logs/pino-unified-repeatability-20261112-01/historical_regional_metrics.json`.
Its explicit historical_step20000schema uses existing direct prediction paths
and rechecks physical-weighted scores. This motivates near-wall coverage but
does not prove the new wall allocation helps.

Future CDF/wall final64evaluations append this CPU diagnostic after publishing
the unchanged original headline summary/timer. Source change is limited to the
evaluation helper; live training/math/inputs are untouched. Existing area and
historical outputs remain intact. The new option uses explicit schema names;
no silent fallback or change to the requested headline metric.

Disk check:14GiBfree; a current full channel resource run is519MiB and completed
area20k/evaluation156MiB. No cleanup performed. Regular-probe config/log schemas
are checked through actual dispatch: Melting uses spatial_training, whose
timing/memory keys match the waiting queue; the older fixed-pool trainer has
different keys and is not the selected route.

### Completed CDF20k and64case results; wall split started

CDF child639573 completed20000steps and final4case phase0.03177130. Its online
audit, native/local evaluation and final64evaluation all passed. The final64
Euclidean space-time means±sampleSD are phase0.04069381±0.00574889,
concentration0.04923459±0.00692743,channel mean0.04496420±0.00633714.
Near-wall/bulk phase means0.05733179/0.02784196. Exact receipt paths are in
`/root/autodl-tmp/tf-logs/particle_corrosion2d/pino-unified-followup-20261112-01-cdf/evaluation64/summary.json`
and adjacent `regional_metrics.json`. Compared with the historical successful
area64model, phase improves on38/64samples overall,32/64near-wall and48/64bulk;
mean overall change is only-0.00051036. Do not compare only against the failed
new area run or claim a robust sampler advantage from this single trajectory.

Training wall time is2050.59s (~34.2min), versus area2040.09s; warm update means
94.27vs93.93ms. The same-state/input/draw audit for the completed two arms passes:
`/root/autodl-tmp/tf-logs/pino-unified-followup-20261112-01/area-cdf-paired.json`.
Original helper also reports synchronized update+sampling+online costs and
active GPU memory. No training-budget/default change follows.

Followup owner600703 now runs wallCDF20k child650532, verified by its live
command line. Accuracy owner646658 remains waiting atjournal03. No process was
restarted this turn. After the wall final64 result, the existing queue performs
three-arm pairing and the short GPU replays before the remaining geometry
pilots. Goal incomplete. Memory changed:PROGRESS,DECISIONS,this task. New code:
regional metric helper/tests and evaluation-only hook. CPU checks, saved-norm
replays and `git diff --check` pass. No new Git action or reference solve.

## 2026-11-12 — User-authorized diagnostic checkpoint

Scope is exactly ten source/test/memory files: the three existing memory files,
accuracy queue and final64 evaluator, four diagnostic helpers, and regional metric
tests. No production model/loss/default, figure, website or external artifact
changes are included. The earlier master-figure publication is already complete.

Validation for this checkpoint: all three `test_pino_regional_metrics` checks
pass on CPU; AST parsing passes for all seven changed/new Python files; diff
whitespace checks pass. Existing full input/lowering/replacement and saved-norm
receipts above remain applicable; expensive checks are not repeated. The staged
files contain no checkpoints, raw arrays or logs. Read-only remote inspection
confirms origin/main is833880e before publication. Follow repository workflow:
commit on dev, regular merge to main, push main only, verify the remote ref and
clean worktree. Git history records the resulting exact revisions.

Live jobs verified without intervention: followup owner600703, wall training
child650532, and waiting accuracy owner646658. After this checkpoint, resume at
the wall final64 evaluation/three-arm pairing and queued GPU repeatability checks,
then the Spinodal and channel accuracy trials. Goal remains incomplete; no
new sampler is promoted. Memory updated:PROGRESS,DECISIONS,this task.
