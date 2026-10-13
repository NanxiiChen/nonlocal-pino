# Project Progress

> Dashboard, not changelog. Detailed checkpoints and earlier experiments live in the active task.

## Current focus

- Active task: [`docs/tasks/fracture2d-nonlocal-pinn.md`](tasks/fracture2d-nonlocal-pinn.md).
  Goal: verified fracture2d using nonlocal differentiation; current branch uses
  direct local P1 mechanics, nonlocal AT2 damage and committed maximum history.
- User initially accepts current fracture2d results (2026-10-13), with
  substantial numerical improvement and code simplification/consolidation/
  organization still required. Dated stage report: [initial acceptance](reports/fracture2d-initially-accepted-20261013.md).
  Current-turn commit authorization covers source/docs/text evidence only;
  slides and binary figures/PDF/weights excluded.
- Fresh L1 random-Adam own-zero loading COMPLETE (2026-10-13). User defers
  explicit front/high-residual weighting. Grid5e-4 through.0050 then1e-4
  through.0060; fresh sigmoid weights, fixed spatial basis with recomputed
  latent RMS scales, physical H features/no direct labels, Adam1e-5 constant,
  persistent moments/count,128x512 caps and core2048/halo32768/pool256/resample1.
  Former supervisor91533/worker91534 and CPU audit95575/figure96334 exited. First
  .0005 finishes128/65536 in651.01s: R66.97442 vs same-load direct66.96340,
  damageL2 18.758%/RMSE7.633e-5, physicalAT2.11424; provisional, not ready.
  Independent native/CSR/history/BC/host FEM/Adam/core replay passes20
  completed loads through.0060 and all5120 seeded support blocks. .0056/.0057
  R312.10/157.43,damageL2 to matched direct53.13%/37.75%, not spanning.
  First confirmed spanning load.0058: outer64 R.90331 vs direct1.19457,
  d>=.9 tip-connected band reachesx1.0; firstR<2 at38,<1 at42.
  Final.0058 outer128 R.88857,damageL2 32.56%,physicalAT2.09228;
  provisional/unready. Reviewed cloud shows slightly curved/wider band.
  Force-displacement four-way full/critical/residual-force figure reviewed:
  FEM(local damage), refined nonlocal-direct, current L1/random Adam and
  old sigmoid L16/full L-BFGS. Both neural paths unload gradually to near
  zero around.0058; L1 has higher force at.0053..0057. Different early
  grids/history/scales prohibit isolated loss-effect claims. Plot/CSV/source
  manifest: independent_review/force_displacement_20261013T120154Z.
  Final .0060/128 R.84173694 vs same-load direct1.07800577, physicalAT2
  .07145288,damageL2 32.2983%,own-H raw error3.1436%; all positive loads
  provisional, strict certified prefix remains zero. Adamcount1310720.
  Root and full independent audit complete; worker exits0 at12:15:31UTC,
  supervisor restores fine18811. CPU audit/figures finished; full source,
  history, BC, host FEM, optimizer and sampling checks pass.
  Force plot retains its capture120154UTC (L1 complete.0058, saved.0059).
  Same model/stagger map does not imply equal finite-budget trajectories:
  fixed-H volume-SPD damage solution is unique; finite NN/optimizer accuracy,
  128-vs-2048 caps and provisional committed histories remain different.
  Direct at.005275 needs1575 outers;128 stillR580.54.
  root l1_damage_loading_from_zero_20261013 archives source/config/checkpoints/
  state/reports/audits/plots. First launch's report-type error archived and
  fixed; three focused unittest regressions and git diff --check pass.
  Fine18811 restored, observer20748 alive, coarse10891/13274 remain stopped.
  Next review completed full path and plan a common-parent critical-load
  control separating inner precision/outer convergence. Additional weighting
  remains deferred; extension beyond.0060 unlaunched. Exact resume in task.
- L2 Adam GPU-pool and frozen-H adaptation diagnostics COMPLETE
  (2026-10-13). Equal16x512 L1/L2 R534.0006/534.0703, physicalAT2
  .11365/.11226, matched16-direct damageL2 1.8401%/2.3947%; own-H raw
  errors1.7389%/2.2699%. L2 takes83.46s vs L1 87.46s; neither is ready.
  Extra frozen-H4096 at1e-6/1e-5 retains Adam state/count12288->16384:
  rawAT2 .07429 ->.07333/.06199, own-H damageL2 2.2699% ->2.2298%/2.0421%.
  Full MSE4.06e-7 ->3.41e-7/1.96e-7; original inner.001 target unmet.
  Rate/budget help average accuracy but do not remove peak imbalance.
  All16 physical sweeps and four final/best frozen checkpoints replay
  native/CSR/H/BC/host FEM/LR/state/identical cores; direct own-H oracles
  are evaluation only, no physical update in adaptation trials. Roots
  low_order_l2_pool_20261013 and low_order_l2_inner_adapt_20261013
  archive config/scripts/source/reports/fields/plots/audits. Supervisors
  83767/85593 exited0 and restored fine18811; observer20748 alive, coarse
  10891/13274 stay stopped. Latest fine.005275 outer442,R481.03,AT2.04208.
  Broad task remains active. Next read original fine512/completed load;
  explicit detached residual weighting near the active front is deferred
  by the user: current1/(N*p) correction restores uniform expected mean,
  so sampling density alone does not increase its objective weight.
- Low-order/SOAP and sampling-speed diagnostics COMPLETE (2026-10-13).
  Scope clarification: actual GPU-pool coupled training covers L1/L2+Adam.
  The four L1/L2 x Adam/SOAP fixed-H controls used the earlier fresh-core
  sampler (resample5); SOAP coupled pool controls remain unrun; no full four-combination matrix.
  Shared SOAP uses SOAP_JAX0.2.2/a7c92f17; nine focused CPU tests (6+2+1),
  source verification, dependency-lock check and git diff --check pass.
  Same-H original/L1randomAdam/fullL2Adam damageL2=4.34%/2.01%/2.20%;
  SOAP1e-5 final L1/L2=2.85%/3.51% despite increased MSE/max residual.
  Identical-source16x512 L1 resample5/resample1/GPU-pool reaction534.0308/
  534.0037/534.0006, physicalAT2 .11405/.11361/.11365: similar, unready.
  Raw-inner .07574/.07653/.07651 remains high near the active crack tip;
  fixed-H field improvement does not establish coupled convergence.
  Resample1 alone provides little accuracy change; not proof of overfitting.
  User correctly identifies elasticity prebuilt pools. New immutable support
  row cache removes full-CSR copies/per-core loops (64 actual batches exact,
  15.46ms->1.37ms packing). New256-entry GPU support blocks retain all8192
  fresh draws and count4096->12288, with dynamic batch selection/current
  detached H/features. Total resample1 time248.21s->138.19s(cache)->87.46s(pool),
  including all32 fresh pool preparations. Old resample5 costs99.58s.
  Native/CSR/history/BC/host FEM/optimizer/seeded sampling replay passes for
  all controls; plots inspected. Exact support parity is separate from
  cross-run trajectory parity: strict1e-10 test fails with small deviations
  and retained failure log; no bitwise numerical-trajectory claim.
  Roots low_order_damage_20261013, low_order_l1_coupled_20261013,
  low_order_l1_resample1_20261013, low_order_l1_cached_20261013 and
  low_order_l1_pool_20261013 contain configs/scripts/snapshots/reviews.
  All diagnostic workers exited0 and restored fine18811; observer20748 alive,
  coarse10891/13274 remain stopped. Fine resumes .005275 outer422,R487.44,
  physicalAT2 .04311, still unready. Broader fracture task remains active:
  next inspect original fine512/finished load, then separately address inner
  adaptation/learning rate/mean-vs-peak residual before accepting low-order paths.
- Frozen-H precision diagnostic COMPLETE (2026-10-13): fine55/.005275
  checkpoint256 independently replays. Same-H original/tight raw sigmoid
  damageL2=4.3365%/3.9979%, AT2=.00099804/.00070042; strict1e-4 target is
  not reached after16384 extra updates/~380s. First round stops14675 on
  implicit SciPy maxfun15000; archived, then remaining1709 updates use
  explicit maxfun65536 with a recorded optimizer restart. Same-half-relaxed
  one-update FEM R: original539.44048/tight540.26325/same-H direct539.48879;
  no improvement towards complete unloading. These are frozen-H diagnostics,
  not converged-load reference states. Moving-tip error.0669% ->.0839% remains
  small; source residual decomposition is dominated by relaxation/history
  update terms rather than the inner residual. Evidence is checkpoint-local;
  cumulative neural effects remain open. Matched outer280 reference/PINN
  R519.27/532.18 and coupledAT2.04150/.04121 are both unready; reference1024
  still R293.90, then1575 reaches1.59 (whole-load290.13s). Direct final versus
  neural intermediate comparison cannot isolate neural accuracy. Direct acceptance
  requires current/fixed-H AT2<=.001 plus all update/physical/force gates for
  three consecutive sweeps; final55 checks pass1573..1575 (details in task).
  Final native sigmoid/CSR/history/
  half-relaxation/host-FEM replay passes; clouds/force plots inspected.
  Root sigmoid_damage_inner_precision_20261013 contains config/scripts,
  snapshots, two-round logs/reports, oracle, audit and plots. No coupled tight
  continuation launched: target unachieved and immediate force benefit absent.
  Fine18811 restored by supervisor; observer20748 alive; coarse10891/13274
  remain stopped. No protected module/model/source loading state changed.
- Quarter-load sigmoid branch (2026-10-13): user requests smaller increments.
  From certified physical52/same-head adapted52 network, load .005225/.005250/
  .005275/.005300 with2.5e-5 steps, original model/gates,1024/4096 per-load
  outer/inner caps and explicit user-provisional ancestry. GPU18811 and CPU
  observer20748 run detached; coarse10891/observer13274 reversibly suspended
  at completed148,R564.78 until the fine worker exits, then observer resumes
  them. First fine load .005225 stabilizes atR607.443 versus matched fine
  reference607.684 (about.0396%); AT2=.0009713 passes, but healing.0048306
  exceeds.001 and prevents certification. First step caps1024/8832 total
  outer/inner in587.89s (user_provisional); damageL2=4.51%. Second .005250
  passes all local gates in146 outers/9216 inner updates, R607.72055 versus
  reference607.67474; damageL2=4.63%, still uncertified ancestry from the first
  step. Current .005275 completed422/1024 outers, R487.44, physicalAT20.04311
  versus inner0.00099514; critical coupling remains unready.
  Fine GPU and observer are alive after the precision diagnostic;
  coarse jobs remain SIGSTOP suspended. Independent first-sweep/native-
  sigmoid/CSR/history/FEM/same-load replay passes; stable128 and completed
  first/second loads independently replay.
  Two focused schedule/history tests and actual source52 GPU preflight pass.
  Figures/audits: sigmoid_damage_load_refined_20261013/independent_review.
  More available total work is recorded separately from load-refinement effect.
  Neural parameters warm-start across sweeps and loads; each SciPy L-BFGS
  fit restarts curvature history while reusing the cached JIT functions.
  Current inner loss is squared nodal L16 of AT2 residual normalized by
  2*RMS(H), floor1e-15, all71307 material nodes, detached H/features and
  sigmoid output; no explicit band weighting/reference/energy loss.
  Inner-accuracy attribution remains open: coarse fixed-H full8192 still
  has5.70% damageL2 at AT2.001095; residual pass alone does not certify
  field accuracy. Fine checkpoint256 same-H accuracy is measured above.
  Exact source/control/parent/labels/resume in the same active task.
- Critical-step accuracy experiment (2026-10-13): isolated sigmoid52->53
  controls complete and independently replay. At identical ownH/network,
  full512/full8192 damageL2=10.51%/5.70%, AT2=.002099/.001095, times13.02/202.78s;
  sampledAdam8192 plus full4096 at1e-4/1e-5 gives11.28%/6.36%, .002037/.001657,
  times200.54/205.56s. Full8192 wins these controls; no generic sampling claim.
  Same-load continuation starts from unchanged53physical128/committed52 H,
  up to1024 total outers/4096 inner, original gates. Audited129: inner.0009981
  passes; physical.07377, R573.78, tipx.5196/not spanning, final-reference
  damageL2=78.62%, still unready. Latest live report132: R573.67, inner.0009948,
  physical.07658; coupled residual is not monotone. Latest completed148:R564.78,
  physical.07058. GPU10891/supervisor10890 and CPUobserver13274/supervisor13273
  are reversibly suspended for fine-load GPU priority; observer restores them
  after the fine worker exits. Five focused tests and actual GPU source/
  exact-halo preflight pass; all five fixed states and physical129 replay.
  Plots/audit: sigmoid_damage_accuracy_20261013/independent_review. Other
  interrupted loading paths are not restarted. Exact receipts/resume in task.
- Bounded-output direct loading (2026-10-13): user requests actual sigmoid/
  exp-square coupled solutions rather than another residual screening. Two
  fresh no-z0/no-old-identity-weight52 fits4096 each are saved; extra8192
  adaptation continues same-head weights only. Direct mode then loads53..60
  with original128/512 outer/inner budgets,omega.5/history/gates and explicit
  provisional labels. Root bounded_damage_loading_20261013; former GPU149948
  and completion observer152318 are now absent, without final exit receipts.
  Interruption cause is unknown; user defers recovery while reviewing plots.
  Screening145226 stopped after its two52 fits; unrun identity/own-H60 cases
  canceled. Four focused tests pass; newtwo rerun after direct-mode change.
  Actual CUDA same-head restore/source/initial-gradient parity passes; initial
  CPUsource/snapshot replay passes. Both52 adaptations finish8192 extra updates;
  sigmoid53/.0053 completes128,R571.91,not spanning,AT2.08229 (provisional).
  Latest completed sigmoid57/.0057 is provisional; saved58/.0058 outer99
  has R.88513 and is not a completed load. Exp-square adaptation52 completes,
  but its loading has not started. Independent CPU replay passes adapted52
  heads and completed sigmoid53..57; separate saved58outer99 replay passes.
  Staggered local FEM mechanics/nonlocal PINN damage remains unchanged.
  Requested55 cloud milestone completes: reference/identity/sigmoid outside-
  band means .02910/.11376/.03129; sigmoid reduces identity mean by72.49%,
  while differing crack-growth stages and unready coupling limit attribution.
  Figures/audit: bounded_damage_loading_20261013/review_0055_background.
  Current three-way reaction,3x5 damage and2x5 actual-range displacement
  figures are generated and inspected in comparison_current_sigmoid_0058_20261013.
  Columns50/53/55/57/58 are user-selected; PINN50 inherits certified prefix,
  53/55/57 are provisional128 states,58 is saved outer99. No recovery or new
  load solve was started for this comparison; exact sources/resume in task.
  Same-H CPUdiagnostic60 outside-band mean .03570 versus PINN.11984 shows
  substantial neural background error; bounds alone do not enforce symmetry
  or zero background. Exact current adaptation/load state in active task.
- Active128-outer restart (2026-10-12): user selects full loading53..75
  from certified52/.0052 fields/network. Former CUDA120695 and completion
  observer124279 are now absent without final exit receipts; latest completed
  load67/.0067, pending68/.0068 outer66,R.46753. Recovery is deferred by user.
  Same512 full cached L-BFGS inner budget,
  omega.5/original model/ranges/gates; user-provisional advancement at128 if
  unready.7 focused tests and actual-source CUDA value/gradient/source preflight
  pass; independent53 complete plus8/32/64/128 and54outer8 replay.
  First53 completes128 in793.15s: R534.08 versus outer8R613.79, dmax.9972,
  no through-band; innerAT2.01598/own-H.13356 and same-H neuralL2 error26.82%
  remain unready. First spanning band at58/.0058outer12R7.54249;
  final58R.66940/own-HAT2.03382,59R.67606/AT2.02819;60/.0060 completes128
  with R.65427/AT2.02604, band spanning. Target75 remains incomplete.
  Independent replay passes complete53..60 and saved
  checkpoints. Same-load reference damageL2 errors58/59/60=144.35%/123.55%/
  115.80%; all53..60 locally unready. Requested FEM/reference/PINN reaction,
  3x5 damage and2x5 PINN displacement comparisons through.0060 are saved under
  restart_damage_loading_128_20261012/comparison_through_0060_20261012.
  Same mesh/volume grips; FEM has local lumped P1 damage, reference nonlocal
  direct damage. PINN crack bends upward and background damage is elevated;
  low reaction/spanning band do not establish accurate equilibrium fields.
  Immutable8/32/64/128
  checkpoints isolate within-load outer progress. Root
  restart_damage_loading_128_20261012; exact receipts/resume in active task.
- Dated stage record (2026-10-12): [provisional higher-load experiment](reports/fracture2d-provisional-loading-20261012.md)
  preserves settings, matched-load controls, conclusions/limits and three figures.
  Audited results and hash manifests are copied into repository report assets;
  large fields/weights remain in the original archives. Documentation only;
  no new calculation. Current status and exact resume stay in the active task.
- Latest result (2026-10-12): own-zero linear neural damage + separately enforced
  range accepts and independently replays all52 loads through.0052. Original
  AT2<=.001, range1e-10 and three consecutive coupled-equilibrium gates remain.
  R606.551297 versus matched direct606.590692 (difference.006495%); maximum
  prefix reaction discrepancy.047835%. No direct targets, old-head weights,
  reference output, projection or nodal damage correction.
- User accepts microunit negative damage (2026-10-12): separate same-load
  recovery branch sets damage-range tolerance2e-6 consistently in inner,
  physical/current-H NN and FEM validation. All original residual/history/
  force/healing/change/consecutive gates remain. Old strict root stays intact.
- Tolerant continuation completed (2026-10-12): PID30710 exited after264s.
  Resumed inner123 passes with0 updates; physical min-4.40e-7 retained, not
  clipped. New inner124 stops after four2048 stages atAT2.00101803>.001;
  its damage is nonnegative, so range is no longer the stopping reason.
  Initial and final independent signed-FEM/own-H/NN/ancestry audits and2
  focused tests pass. No new accepted53; physical AT2.085455 remains unready.
  This worker exited; its final124 checkpoint seeds the new budget branch
  below. Range2e-6 stays; final equilibrium still required.
- Earlier strict worker16781 stopped at123 only on four negative NN points;
  accepted52 plus rejected53 independently replays. Scalar-bias probe is
  retained as evaluation evidence, not implemented into the new optimizer.
  Exact new config/source pins/log and resume point are in the same task.
- Output conditioning: matched sigmoid/exp-square/identity controls all failed
  initially. Same-H identity inequality augmented-Lagrangian continuation reaches
  .000807004/min-5.21e-11, passing original gates without clipping. NumPy
  network/operator/own-H/dual replay and two focused AL/zero-field tests pass.
- Distinguish physical activity from equation imbalance: report absolute and
  normalized AT2, forcing/component scales, damage mean/L2/active fractions.
  Positive-H d=0 has full residual-2H and original relative1; only d=H=0 has
  zero absolute residual. Across accepted loads absolute residual increases
  while normalized balance passes. Plot/report: range_damage_report_20261012.
- Current physical-field maps (2026-10-12): physical_field_maps_20261012
  compares accepted52/.0052 and unready53/completed outer123. Actual relaxed
  d,ux,uy,H on material triangles share color scales; H logarithmic, signed d
  retained. Damage zoom shows tip localization/rightward extension; not full
  fracture certification. Script/source hashes/ranges in fields_report.json.
- Matched final-reference comparison (2026-10-12): accepted.0052 reaction
  error.006495%, volume-weighted nodal damage L2 error1.4283%,
  damageRMSE.000642996/max absolute.001895629, material
  displacement.03219%. Current.0053 outer123 reaction570.539 versus direct
  outer123570.793 (difference.04452%); final direct outer1296 is1.57856
  with a damage band across the remaining ligament. Current-to-final damage
  L2 error78.6663%, displacement33.7364%; intermediate progress, not final
  equilibrium. Inner124 stops only1.803% above its.001 threshold: investigate
  the rigid training-budget stop separately from coupled acceptance. No gates
  changed. Checks/plots: physical_reference_comparison_20261012.
- Additional-budget recovery (2026-10-12): owned GPU PID40800,
  linear_damage_budget_continue_20261012. Restore failed-inner124 NN at exact
  own H/completed physical123, accepted52 parent retained. Only maxiter/stage
  2048 ->8192 (early exit) and explicit new output/recovery metadata differ.
  Original residual.001/range2e-6 and all final coupled gates unchanged.
  Two focused tests and actual-source CPU ancestry/force/config preflight pass.
  Independent initial restore replay passes. User then requests negative
  tolerance1e-3; worker stopped withSIGTERM before a new physical update.
  Unsaved stage2 residual.00070518/min-2.32555e-6 retained as scalar evidence;
  this partial branch is not a final numerical rejection or accepted state.
- User-requested full-path attempt (2026-10-12): former GPU PID45229,
  linear_damage_relaxed_continue_20261012, monitored session47821. Explicit
  asymmetric bounds[-.001,1+2e-6] in inner/physical/current-H NN/signed FEM.
  Restore certified failed124 weights/completed123 fields with accepted52
  parent; four8192-stage budgets and all equation/history/coupling gates stay.
  Four focused asymmetric-range/config/FEM/inner/outer tests pass. Target53..75
  ends.0075. Independent recovery passes: inner124.000997809 after1440 updates,
  physical124AT2.0769857/R569.84265/min-4.596e-6. Both worker/observer later
  disappear without final numerical report; last physical126 persists. No
  accepted53 yet. Interrupted source and all prior receipts remain preserved.
- Pending126 recovery (2026-10-12): independent NN/own-H/half-relaxation/
  signed-FEM replay passes, R568.74079/physicalAT2.0767984/min-1.9535e-5.
  New linear_damage_pending_continue_20261012 restores saved126 NN/fields,
  starts127, retains accepted52 parent and identical numerical controls.
  Two focused recovery tests pass. Double-fork worker51924 and CPUobserver54698
  run under PID1-owned supervisors in independent sessions, with explicit
  log/exit receipts. Initial and first127 recovery replays pass: innerAT2
  .000999834 after7072 updates, physicalAT2.0777684/R568.12846/min-3.1012e-5.
  Direct transient127R568.50577 (difference.06637%). Latest inspection151
  independently replays: physicalAT2.0653540/R554.88559/min-2.0689e-5.
  Direct transient151R556.06628 (difference.21233%). Latest saved152
  R554.36868/physicalAT2.0664127/min-1.8500e-5; next inner153.
  GPU/observer remain alive across turns after74min. Accepted parent still52,
  no accepted53. Recent10 inner solves average165.35s, optimization cost remains
  high. Full target53..75 remains; next final accepted/rejected/reference audit.
- Live continuation/review plan (2026-10-12): GPU51924/observer54698 remain
  live after102/100min; physical158 completed, inner159 starts, accepted52
  unchanged. Physical157 R551.66508/AT2.0674618/min-1.9546e-5; reaction and
  damage activity advance, coupled residual is not monotone. Recent10 inner
  solves average233.48s. Next diagnostic around200: same-load field error,
  identical-H neural error and crack extent;200 is no acceptance/abort gate.
  Existing max2048 and all original physics/coupled criteria remain.
- Detached weighting pilot (2026-10-12): three same-network/current-ownH153
  CUDA fits2048 iterations complete; uniform/static/damage fullAT2
  .00125161/.00161168/.00187068, same-H damageL2 about4.73..4.75%. No benefit
  demonstrated for boost9/width.024/power2; five meaningful focused tests and
  independent NumPy audit pass. Weights detached and frozen perL-BFGS fit.
  Exact core1024 halo13515..14567 of71307 preserves original rows; actual
  sampled training remains next, no speed claim. Pilot exits0; main worker
  continues through physical164/inner165 with no numerical change. Artifacts:
  weighted_damage_trial_20261012/comparison_audit.json.
- User switches to core+halo (2026-10-12): authorized stopping slow original
  continuation at complete physical192/.0053 (R535.96631, AT2.05529124).
  Main51924/final observer54698/outer200 observer79842 exit bySIGTERM with
  receipts; exact192 NN/fields preserved, accepted52 unchanged. Wait-for200
  is superseded. Original full-graph, cached full-graph and uniform/crack-biased
  core+halo fixed-H controls now run separately; exact support/gradient/global
  normalization and detached sampling checks pass8 focused tests. Whole-field
  acceptance unchanged; no new coupled load or sampling speed claim yet.
- Fresh random sampling (2026-10-12): first isolated four-case core-halo pilot
  completed and independently replays. Cached full36.27s/AT2.0010803;
  old crack sampling19.84s/.0050128, uniform.88285, all unready. User selects
  geometric crack-centered probabilities with bilateral inverse-power decay
  and fresh resampling, replacing the unrun mandatory-peak proposal. Nine
  focused tests pass; actual fresh2048 cores every8/32 L-BFGS updates complete:
  31.94s/AT2.00280961 and24.48s/.00216050, both above.001. Independent
  probability/seed/all-halo/NN/CSR audit passes.32-step sampling is better here;
  no certified convergence speedup yet. Complete halo/detached probabilities/
  importance correction/global gates unchanged. Original main stays stopped
  at192; accepted52 unchanged. New GPU93645 exits0, no worker or promotion.
  Next same-H cached full polishing and total-time-to-gate comparison. Figures:
  core_halo_damage_resample_20261012/{sampling_density,resampling_convergence}.png.
- Optimizer clarification (2026-10-12): elasticity Adam preserves moments when
  switching sampled subgraphs; fracture pilot restarts SciPy L-BFGS curvature
  history per block while retaining NN weights. Reset is a conservative method
  choice, not a sampling requirement; existing timing conflates sampling and
  restart cost. Next consider fresh random cores with persistent Adam state,
  optionally fixed full L-BFGS polish. No new training or numerical change.
- Persistent Adam trial authorized (2026-10-12): same audited192/currentH,
  three sampled rates1e-3/1e-4/1e-5 with2048 updates and fresh core every5,
  moments/count preserved. Full cached Adam best-rate control and full L-BFGS
  polishing completes/replays. Sampled/fullAdam/fullLBFGS final fullAT2
  .00107347/.00107517/.00104459, route56.17/57.57/72.02s; all above.001.
  Three-rate search59.16s aggregate (selected19.82s already in sampled route).
  11 focused tests and independent NN/CSR/
  probability/1230 fresh draw+halo/saved-state audit pass. CUDA99571 exits0,
  no worker/physical promotion or accepted53. Sampling benefit versus fullAdam
  remains limited; common late residual plateau needs further same-H analysis.
  Figures: core_halo_adam_trial_20261012/{adam_convergence,adam_polishing}.png.
- Polished Adam physical diagnostic (2026-10-12): hold sampled_polish signed
  damage and solve own P1 FEM at.0053: R535.10159; original omega.5 candidate
  R535.53651, saved192R535.96631, converged same-load reference1.57856.
  OwnH based on committed52; whole physical AT2.05444/.05484 remains unready.
  Host force/BC/history/PDE/reaction/source checks pass, clouds/reaction figure
  inspected. Evaluation only, not accepted53/completed193; no live worker.
  Output: adam_polished_physical_diagnostic_20261012. Metric-cost profiling
  deferred (global checks every128,16 inside2048; single-check cost unmeasured).
- Higher-load experiment COMPLETE (2026-10-12): separate user-provisional
  PINN path advances53..75 with8 alternating updates/load and512 cached full
  L-BFGS updates/inner; GPU107561 exits0, no live worker. Final.0075R267.65834,
  tip-connectedd>=.9 frontx.85217, not through-cracked. Own-H AT2.34650 and
  damage healing fail original gates; strict prefix52 untouched. Full53..75
  independent NN/FEM/history/ancestry/PDE replay passes,7 focused tests pass.
- Matched-budget direct damage control completes and independently replays:
  same source/model/8 outers/load fractures at.0074, final.0075R.18216.
  Fully converged same-model reference fractures at.0053 and finalR.55497.
  Thus capped coupling itself delays fracture and NN error adds further lag;
  no assertion of a verified material critical-load shift or optimizer
  impossibility. At matched.0075 NN damageL2 to reference64.25%. Curves and
  physical fields inspected in provisional_damage_loading_20261012/review_0075.
  Next possible step: bounded fixed-load75 continuation; exact sources/parent
  history, checks and restart requirements in active task. Broad task remains open.
- Same-model direct baseline completes75/75; local-gradient controls and load
  refinement are available. Spatial/horizon convergence and full critical-load
  neural certification remain open. Current evidence establishes feasibility
  of the accepted prefix, not neural speed or superiority. Earlier experiments,
  failed heads and accepted old52 ancestry remain preserved in the task record.
- Damage error maps (2026-10-12): captured physical153 and converged direct
  reference both use load step53/.0053, not terminal75/.0075. Volume-weighted
  nodal relative L2/RMSE/Linf are77.2400%/.100630/.965564. Current crack is
  near the tip; this same-load reference has crossed the remaining ligament.
  Saved inner NN versus original direct AT2 at identical H_used gives
  4.7182%/.002978/.008468; CPU CG116 iterations, AT2 residual6.25e-9.
  Distinguish uncompleted coupled evolution from fixed-history neural error.
  Pinned sources, clouds and centreline profiles: damage_error_maps_20261012_outer0153.
  Evaluation only; no training targets, clipping, source/gate change or accepted53.

## Direct nonlocal reference (adopted)

- Scheme: original BA/Amor/AT2 equations on the 3395-node `delta=0.048` cloud,
  `h_g=0.02`, `l0=0.024`, `Gc=2.7`, `E=210e3`, `nu=0.3`, `k_res=1e-8`, cubic
  kernel, fixed damped-Picard `omega=0.5`, `du=1e-4`, `du_min=6.25e-6`, at
  most 144 sweeps, `stabilized_warning` acceptance, load path `0 -> 0.0075`.
- Artifacts: fresh compact solution
  `.../engineering_grip/direct_nonlocal_final/solution_u0p0075_fresh_v1.npz`
  (SHA256 `5f3bc12b...`); research archive SHA256 `9545bed7...`.
- Response: peak `641.34017` at `0.0061` (FEM `642.48573` at `0.0057`);
  terminal `69.15426` at `0.0075` (FEM `3.97718`); 76 states = 65 strict + 10
  warning; drift points `2 -> 31` from `0.0062` to `0.0075`.
- Per-state labels: strict prefix through `0.0059` (59 accepted states);
  `0.0060` and `0.0061` are drift-free warnings, while `0.0069`, `0.0070`,
  and several later drifting states have strict acceptance labels. Classify
  by archived flags and load, not by a contiguous 65/10 partition.
- Accepted limitations: warning-only acceptance,
  `field_equations_validated=false`, terminal reaction mismatch, particle
  drift, and no completed strict `0 -> 0.0075` path. All are documented in
  `examples/fracture2d/README.md`.

## Current state

- `fracture1d` is an accepted engineering baseline; default PINN dtype is
  float32 with `spatial_emb_dim=32` and `spatial_scale=16`.
- The local FEM point-anchor warning-mode archive
  (`/root/autodl-tmp/data/fracture2d/local_fem_point_anchor/results_point_anchor.npz`)
  remains the separate continuum comparator.
- Parked and ignored: monolithic/arc/reduced research bundle, durable
  `engineering_grip_workflow.py`, and the stage-specific fracture2d test suites
  (moved to `_legacy/`). They remain in git history and on disk for reference.
- Memory: `docs/DECISIONS.md` is the append-only rationale log;
  `docs/tasks/archive/` holds the closed direct-stage records.
