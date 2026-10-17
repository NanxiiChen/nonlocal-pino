# Project Progress

> Short dashboard. Current work is self-contained in the one active task;
> completed experiments and superseded process observations are archived.

## Active task

[Nonlocal PINN history-field learning and architectures](tasks/nonlocal-pinn-architecture.md).
Opened 2026-10-15 after the user closed the fracture2d feasibility stage.

2026-10-18: user agrees SINGLE geometry first, retaining cross-geometry
architecture potential. Current question is FEM mechanics-vs-damage cost.
Typical hybrid fixed-H AT2: vector elasticity is usually costlier than scalar
damage, especially with crack-induced stiffness contrast, but project-specific
split is not measured consistently. Mechanics has a synchronized solve timer;
damage direct helper lacks matching timing. PINN damage training cost is a
different quantity. No benchmark/new run or code change under this discussion.

Latest user priority: cross-GEOMETRY generalization is the final objective.
Mesh differentiation is compatible: per-geometry FE matrices/quadrature can
vary while one geometry/history-conditioned network shares weights. Proposed
direction: geometry/boundary set encoder + multiscale H encoder + coordinate
decoder, trained by each geometry's FE energy. No fixed-node learned output
vector or node-ID parameterization; distinguish remeshing, unseen shape and
topology tests, plus zero-shot vs fine-tuning. Current history model has
partial geometry information in patch positions/masks, but no validated
cross-geometry training. Discussion only, no new implementation/GPU run.

2026-10-18 discussion: user proposes replacing damage nonlocal derivatives
with shape-function derivatives on the existing mechanics mesh. Most promising
new engineering candidate: neural nodal damage + material-P1 interpolation +
matched lumped-FE energy training; nonlocal energy retained as same-model
control. Current FEM reference already uses S=integral gradN.gradN and lumped
reaction/history. Cached localS496651nnz vs nonlocalD1138075; full neural
forward node count unchanged, so no claimed total speedup. P1 has zero interior
second derivatives: use weak/energy form, not pointwise Laplacian. New local
operator changes finite-horizon damage model; no code/default/run changed.

Latest discussion: user asks for an overall remedy, then the exact energy
functional. Candidate next fixed-H comparison: existing L2 residual vs the
equivalent volume-weighted quadratic damage functional. Checked reciprocal
positive graph/volume symmetry and directional derivative on actual operator
(relative mismatch2.13e-11). This is a proposed training objective only; no
loss/code/default change or new GPU run. It is a frozen-H damage functional,
not a joint energy proof for history-field stagger. Details in active task.

Sparse-probe screening COMPLETED, 2026-10-17 23:44:45Shanghai.
Root `tf-logs/fracture2d/diagnostics/sparse_response_lr_20261017` (autodl-tmp).
All4arms exit0; former7468/17367gone,GPUidle,no queuedcold or extension.
qualification.json eligible=false; newcoldroot absent. Oldcold stayscancelled.

- At0053/512,baseline8 vs event32 bothspan. DamageL2error21.564/10.342%,
  syncR1.23188/1.42092 vsreference1.57096. Wall34.66/31.26min excluding
  warmup:9.80%less fornewarm atfixedhorizon, withbetterfinalfield. Bothstill
  fail1e-4fieldchange at512 (2.676e-4/1.580e-4); no certifiedconvergence.
- Newfirstconfirmedshadowstop409:damageerror17.619%,syncR1.28275, despite
  dchange9.165e-5. Fieldchangepasses do not guarantee targetfieldaccuracy.
  Baselinehasnocandidate. Keepthisfirstcandidate, nooracle replacement.
- Elasticwindows baseline/event85.83/170.48s;0018outers14/39. Thisregression
  andmissingbaselinecandidate blockpredeclaredfullcoldgate. Currentvariant
  useful forcriticalpropagation butnotqualifiedasoverallcoldstrategy.
- Same-sourcebaseline rerun didnotreproduceprior8.77%quality; causeunresolved.
  Exactnext: inspectreproducibility andfinishing/stopadequacy beforeanother
  qualifiedpolicy/fullcold. No newnumericalcode/defaults or Gitoperations.
  NineCPUchecks plusmigrationregressionremainvalid; details in active task.
  Taskincomplete, finalMLPvscompletedadaptivecold two-curveplotstillpending.

Matched cold branches128→512 continuation COMPLETED (2026-10-16).
Root: `/root/autodl-tmp/tf-logs/fracture2d/diagnostics/cold0052_nonmonotone_lr_to512_20261016/`.
Both512exit0, allformer60313/60340/64187exited; GPUidle,no queuedextension.
TwofocusedCPUcontinuationchecksPASS32.098s remainvalid; bothclouds inspected.

- Adaptive/fixed3e-5 BOTHspan. Firstlogged stageR<3 at376/432outers,
  correspondingcumulativewall32.18/30.40min. Feweriterations is not less time.
- Final512adaptive/fixed: syncR1.43456/1.05498 vsreference1.57096;
  damagefieldL2error8.768%/20.631%,ownHerror8.342%/20.694%.
  dRMS5.844e-5/2.100e-3: adaptivepasses1e-4(10consecutive),fixedfails.
  PhysicalAT2.19854/.11718; thismetric alone doesnotrank fieldaccuracy.
- Cumulativefirst128+continuation43.01/35.98min, executed79744/65536updates;
  fixed16.33%lesswalltime but poorerunsettledfinalfield. No matched-quality
  speedupproof. Supports adaptivefinishing quality; trialcost remains open.
- Raw512checkpoints remainin_load/bank105 atfixed.0053. No loadadvance or
  fullcoldcompletion claimed. Finalcomparison.json/closeout.json recordscope.
- User agrees longerhorizon. Restore each arm's OWN rawouter0128 q/d/H,
  net/Adam86272,bank105,controller643/probephase/flags. H_parentchecked
  against actualcold0052H. Add384outers×128updates each, fixedload.0053,
  no fieldstop/loadadvance or policychange, originalcold remainscancelled.
- Checkpoint129/every16; physicalsync/validation at192/256/320/384/448/512.
  Each arm in separate serialprocess; adaptiveguardfailure is explicit and
  fixedarm stillruns independently. No fallback or rate-rule relaxation.
  Track cumulativeprior128+newtime/work, never label diagnostics as fullcold.
- Initial128comparison (completed, twoCPUchecksPASS17.201s) established:
- Adaptive vs fixed3e-5: synchronizedR552.86 vs554.60; both frontx.54386
  (notch tipx.5),same-load damageerror74.63% vs76.42%. Neither spans or
  satisfies fieldstopping. Reference0053R1.57096 is the fullyspanning target.
- Walltime644.35s vs538.09s (10.74 vs8.97min); fixed16.49%less time,
  executed16384 vs19584updates. Saves candidatework without meaningful
  crack-progress improvement within128. OwnHerrors11.66% vs11.52%.
- Matched old/new FIXED3e-5 audit resolves apparent contradiction: at128
  stageR555.72old vs553.60new,frontx.54386395 both. Oldfixedrapidunload
  occurred later:256R365.92,344R1.19. Newtest ended128, so no evidence
  it lost oldfixedrate benefit. Comparestage-to-stage, not synced-to-stage.
  Separate oldwarmADAPTIVE at128R447 used1e-4 too; not fixed3e-5control.
  Receipt matched_horizon_audit.json. User subsequently authorized extension above.
- Bounded result does not establish final breakthroughtime or prove all
  nonmonotone strategies ineffective. No newcontroller/default adopted;
  longer-horizon evaluation has completed as recorded above.
- User authorized multistagger evaluation allowing temporary loss growth,
  then explicitly stopped the prior full cold. Old43941/43968 verified exited;
  `adaptive_cold_launch_20261016/job.json` now cancelled_by_user. Saved52
  endpoints through.0052,lastscalar0053outer264R481.29,latestfullcheckpoint256.
  Stoprequest/verifiedreceipts preserve state. Do not resume or claim0→.0060.
- New comparison restores EXACT actualcold0052 network/Adam/bank/q/d/H:
  count69888,bank104,globalcontrollerouter515. Arm1savedadaptivepolicy;
  arm2fixed3e-5 without lossdescent veto. Fixedarm is a diagnostic, not a new
  adaptivecold replacement. Each128outers×128innerupdates; no early stopping
  or loadadvancement during fixed observation horizon.
- Evaluate synchronizedfields/force/H/AT2,ownHdirecterror,same-load reference
  error and connectedcrackfront at8/32/64/128. Additional reporting mechanics
  do not feed back. Same physics,Hfloor,full-domain L2/Adam/replayOFF;
  reference never used in training or choosingrates.
- Latest cold behavior beforestop: lower-rate choices accompanied slow propagation;
  one128update same-Hloss increase alone does not establish physicalfailure.
  OuterHtarget changes can legitimately raise loss. This is the motivation
  for a multiouter comparison, not an adopted new onlineLRscore.
- Earlierfirst14cost:71vsMLP132outers but557.94svs288.62s; bootstrap and
  heaviermodel/trials matter. Timing is historical,not hardwarecontrolled.
  Saved auditreceipts in oldlauncher; detailed provenance in active task.
- Requested final plot remains ONLYtwo plaincurves:MLP and a completed actual
  adaptivecold. Oldcold was interrupted; do not splice warm diagnosticarms
  into it or label new fixed-rate diagnostic as cold. Finaldeliverablepending.
- TensorBoard6007 continues; oldcold events preserved, newdiagnostic events
  live within this root's adaptive/events and fixed_3e5/events when produced.

Exact resume: completedcomparison/closeout and perarm512raw/validation/figures.
No liveGPUjob or automaticextension; do not signal formerPIDs. Nextresearch
choice is reducingadaptive trialcost and assessing actualfullcold trajectory,
not silentlyresumingcancelled oldrun. Publiccode/defaults unchanged; noGitops.
FinalMLPvscompletedadaptivecold two-curve deliverable remains pending.

## Fracture2d artifact cleanup completed — 2026-10-16

- All 201 fracture2d TensorBoard event files now live below
  `/root/autodl-tmp/tf-logs/fracture2d/`. Accepted directories are
  `pinn_mlp_baseline/` and `pinn_history_conditioned_baseline/`, with their
  own `events/`; trials are under `diagnostics/`, new defaults under `runs/`.
- Current comparison data are `/root/autodl-tmp/data/fracture2d/fem_reference/`
  and `nonlocal_reference/`; prepared inputs are `geometry_assets/`.
  Older models are distinguished in `legacy_references/`.
  Current FEM uses the shared globally fine h=0.004 mesh and volume-grip BCs;
  it is not a demonstrated mesh-converged reference (details in artifact index).
- [Artifact index](../examples/fracture2d/ARTIFACTS.md) records canonical paths.
  Full inventory, metadata backup, migration map and verification receipts are
  in `tf-logs/fracture2d/_layout_migration_20261016/`.
- Verified 77,543 files, 2,183 rewritten JSONs, unchanged accepted fields/curves
  and seven presets' numerical settings. Both PINN checkpoints restore on CPU;
  history buffer restores 106 snapshots. TensorBoard6007 discovers both runs.
  Numerical experiments remain ON HOLD; no Git operations or training started.

## Accepted fracture2d stage

- [Stage closed and archived](tasks/archive/fracture2d-nonlocal-pinn.md),
  2026-10-15. Accepted engineering result uses local P1 direct mechanics,
  nonlocal AT2 PINN damage and maximum-history stagger. Same-model direct
  damage and independent local FEM remain comparators.
- Accepted `pinn_mlp_baseline` saves 0 -> 0.0057, spans by 0.0053; the user
  interrupted 0.0058. It is not a completed 0 -> 0.0060 run. Field accuracy,
  path smoothness, convergence and efficiency remain improvement targets.
- Default: full domain, L1/Adam/sigmoid/FP64, du=1e-4, 128 inner updates,
  outer caps 128/1024, damage-change tolerance 1e-5. No symmetry constraints.
  Optional core+halo keeps a fixed 2048-pattern exact-support padded pool.
- Measured insight: cheaper core+halo iterations needed more outer work;
  matched early-load total time was 3.35 times full. Keep full as default.
- Consolidation and publication complete: final main b395e48 independently
  verified on origin. Four public configs tracked. All _legacy/ directories
  and docs/reports/ are local-only/ignored; 324 withdrawn files preserved.

## Old GPU baseline stopped by user

Saved-config repeat pinn_baseline_gpu_rerun_20261015 is CANCELLED after
through-fracture at explicit user request. Verifiedowner31071/worker31076 exited.
54completed endpoints through0.0054 preserved,partial0.0055outer600 and named
checkpoint512 retained. TensorBoard6007 continues. External stoprequest/verified
receipts and archived stage record exact state; target0.0060wasnotcompleted.
This cancellation is separate from the current short history-operator probe.

## Other retained state

fracture1d remains an accepted engineering baseline. Older fracture2d
nonlocal-mechanics, warning-accepted direct runs and detailed diagnostics are
historical controls, not the public main route; see stage archives.
[DECISIONS](DECISIONS.md) retains append-only rationale. Example READMEs
contain stable formulations and usage.
