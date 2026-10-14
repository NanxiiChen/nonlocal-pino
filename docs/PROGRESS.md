# Project Progress

> Dashboard, not changelog. Detailed checkpoints and earlier experiments live in the active task.

## Current focus

- Active task: [`docs/tasks/fracture2d-nonlocal-pinn.md`](tasks/fracture2d-nonlocal-pinn.md).
  Goal: verified fracture2d using nonlocal differentiation; current branch uses
  direct local P1 mechanics, nonlocal AT2 damage and committed maximum history.
- User initially accepts current fracture2d results (2026-10-13), with
  substantial numerical improvement and code simplification/consolidation/
  organization still required. Dated stage report: [initial acceptance](reports/fracture2d-initially-accepted-20261013.md).
  Stage commit db4dddc on dev includes source/docs/text evidence only;
  slides and binary figures/PDF/weights excluded.
- Source checkpoint (2026-10-14): user authorizes a local dev commit of the
  current stagger/all-node training, comparison drivers, tests, monitoring and
  Markdown records. Slides, binary figures/PDF/weights and external run data
  remain excluded. Numerical improvement and code consolidation remain open;
  the 1024-cap results are preserved after the user-requested stop.
  See the task checkpoint record.
- Publication (2026-10-14): checkpoint dca037e is saved on dev; user now
  authorizes publication through a regular dev-to-main merge and origin/main
  push. Use an isolated main worktree to preserve the live numerical source
  directory. Outgoing changes contain no slides or binary result artifacts.
- Current own-zero stage provisionally ACCEPTED by user (2026-10-14):
  preserve0..0057,spanning0053 and matched-load FEM/direct/PINN figures.
  User authorizes a local dev checkpoint of current Python source/configs/tests
  and Markdown, saved in dev8a46afe. User now authorizes a regular main merge
  and origin/main push. Equation accuracy and substantial code simplification
  remain open; no training restart requested. Binary results,slides,weights
  andevents excluded;exact stage and acceptance in the active task.
- Main-route preference (2026-10-14): retain L1/Adam/all-node. User explicitly
  rejects symmetry constraints and prescribed straight-path losses; prioritize
  training accuracy and early loading history without added model structure.
  Own-H direct oracle at all-node644 retains near-identical drift (.496..5070
  vs NN .496..5075), so investigate its origin before a late readout correction.
  Fresh L1 used du5e-4 below .005; older straighter trajectory inherited du1e-4.
  At .0052 matched-load damage L2 errors are6.9831% versus old1.4283%;
  optimizer/head/sampling/prefix accuracy also differ,so step-size causality
  remains unproven. Paired cold all-node coarse/fine controls now COMPLETE;
  training-budget attribution and exact supervision/resume are recorded in task.
- Fresh own-zero PINN trajectory STOPPED BY USER (2026-10-14): user accepts
  developed fracture and cancels remaining0060 loading. All57positive loads
  through0057 are saved and independently replayed; spanning at0053 persists
  through0057, R1.59124 ->1.26834. Matched damageL2 at0057 is27.6345%.
  Interrupted0058outer582; latest fully verified named checkpoint isouter512.
  Supervisor21331/GPU28996/audit28995/monitor21339 are all stopped; GPUfree.
  Preserve cold_full_pinn_loading_0060_20261014,fine state0000..0057,
  partial0058 and57 published TensorBoard endpoints. Server6007 remainslive.
  user_stop_request/user_stop_verified distinguish cancellation from numerical
  failure;requested0060 trajectory incomplete. No automatic restart or new
  loading authorized. Next review accepted stage and remaining code/accuracy
  improvements at user direction; exact receipts/resume in active task.
- Matched comparison figures GENERATED (2026-10-14): current stopped own-zero
  PINN versus same-mesh local-damage FEM and nonlocal-direct reference.
  Reaction curves use completed0..0057;damage3x5 andPINNux/uy2x5 useactual
  loads0051/0052/0053/0054/0055. Individualactual displacement extrema,
  shared damage0..1,PNG/PDF/CSV/sourcehashmanifest inrunroot/
  comparison_0051_0055_20261014. No training resumed;details inactive task.
- Best-current .0053 continuation STOPPED by user replacement (2026-10-14):
  prior terminal.0065,
  inherit audited own-zero fine prefix0..0052;rerun.0053 anddu1e-4 to.0065.
  L1/Adam1e-5/all71307/sigmoid/FP64,warm net+Adam384000,omega1,
 128inner/1024outer cap eachload,same P1 field-change early stop,no symmetry.
  User stopped4096 branch at.0053outer314;newrestartfrom0052 usesroot
  best_full_pinn_loading_1024_tbroot_20261014/fine,supervisor15207/GPU15212/
  CPUaudit15210/field_monitor15211. Prior10578 logger waskilledbyuser's
  TensorBoardserver restartcommand,causingcontrolledGPUstop atouter196;
  currentrecoveryagainrestores0052. Restore/count/CSR loss andcopied
  prefix checks pass; .0053 now completes1024 and spans at d>=.9, with
  synchronized R1.59543 versus same-load direct1.57096 (1.56% difference).
  Matched damage L2 remains27.89%, same-H neural L2 is1.677%; field-stability
  and residual gates remain unmet. Preserved0053 and pending0054 when user
  requested a fresh complete trajectory through0060. Original
  source/configs preserved.
  CPU TensorBoardX monitor publishesouterloss/residual/change scalars and
  damage/ux/uy/H/reaction images atEVERYcompletedload;baseline0052 already
  published. Existing6007 service HTTP discovers19scalar/5image tags for
  fracture2d/events/best_full_pinn_loading_1024_tbroot_20261014. User's
 6007server reads/root/autodl-tmp/tf-logs directly;obsoletealiasremoved.
  Fieldloggerargv omits'tensorboard' toseparateitfromserverrestart. Event consumer
  smoke test passes;no numerical driver edit. Automatic finalfields retained.
  No certification from budgets/field stability;complete curve includes
  inherited provisional ancestry. Full cap13loads ~7..8h;early stable loads
  reduce cost. Exact state and resume in active task; source checkpoint authorized below,
  with no slides changes.
- Cold load-step suite COMPLETE (2026-10-14): full_damage_load_steps_20261014,
  coarse13/fine53 loads to.0053; identical cold network/scales/Adam policy,
  pre-.005 du5e-4 vs1e-4 only. L1/Adam1e-5/all71307 nodes/sigmoid,
  omega1/128inner,prefix128outer/critical644outer,field-change early stop;
  residual diagnostic and all loaded endpoints provisional. Detached
  all stages/endpoints in declared audit coverage replay,shared initialization
  matches; worker/supervisor/audit exit0 at04:23:32 Asia/Shanghai Oct14.
  Actual duration coarse55.66min/fine121.07min,total~2h57min. At.0052
  matched-load damageL2 coarse.6098%/fine.4099%;at.0053 coarse55.0456%/
  fine45.3335%,R266.52083/231.02961 versus reference1.57096.
  Neither spans or becomes field-stable after644 outers. Same-developed
  intervalx.52..82 pathRMS drift .01173/.00583;fine reduces drift and advances
  crack farther but also uses466432 versus213760 Adam updates. No isolated
  step-size causality or convergence claim. Scientific clouds/path/cost
  records in comparison_review;no new training during result review.
  Old fine18811 was restored on exit;currently it and observer20748 absent,
  old quarter-step result incomplete. Do not claim that branch remains live.
- Coupled L16 diagnostic COMPLETE (2026-10-14): same own-L1 .0052->fixed
  .0053,all-node/L16/L-BFGS/omega1/128inner/644outer,warm network.
  Final synchronized R593.24538,physicalAT2.0140915,damageL2 versus common
  parent direct81.7705%; no span or field stability and no load advancement.
  Transverse peaks d>=.5 reachonlyx.54 (3cuts),insufficient long-path evidence.
  Root norm_optimizer_coupled_L16_20261014;all native snapshots/final replay
  pass,worker exits0. Supervisor restored fine18811 at17:04:29UTC Oct13;
  fine and observer20748 currently alive,trial/audit/figure PIDs exited.
  Preserve completed diagnostic; future main route remains L1/Adam/all-node.
- L1/L16 optimizer control COMPLETE (2026-10-14): same own-L1
  .0052 network,first .0053 frozenH,all71307 nodes; L1/Adam,L16/Adam,
  L16/L-BFGS,4096 budgets. Both Adam start zero moments and retain states;
  stable old squaredL16 formula,unchanged physical model/sigmoid/operator.
  Same-H direct evaluation only. L1/Adam4096:fieldL2 .448677%,maxAT2
 .00784379,45.608s;L16/Adam4096:44.8637%,.0284471,45.847s (oscillating).
  L16/L-BFGS4096/4711eval:3.00209%,.00114290,105.896s,budget exit.
  Better peaks but worse fieldL2 for L16/L-BFGS; neither L16 meets declared
  joint eligibility. No coupled branch launched,no loading advancement.
  Root norm_optimizer_damage_20261014;all9 snapshots/native/source/CSR/
  history/oracle/Adam/selection replays pass;2 focused tests pass. Scientific
  clouds/cost plots inspected;PIDs159741/159744/159742/161848 exited.
  Fine18811 restored00:15:57 Asia/Shanghai Oct14;observer20748 alive,
  coarse10891/13274 stopped. New files/README/memory uncommitted,no slides.
  Next distinguish mean-versus-peak compromise at later H or a declared
  combined objective before choosing a coupled route; no universal ranking.
- Damage-weighted frozen-H L1 diagnostic COMPLETE (2026-10-13), root
  weighted_damage_fixedH_20261013. Same all-node644 source net/Adam868864,
  synchronized .0053 H,71307 nodes,4096 Adam1e-5 updates each; weights
  detached/mean-normalized/frozen from source d. Same-H direct oracle only
  evaluates accuracy; no coupled state promotion/loading. Initial same-H
  damageL2 4.4874%; uniform ends4.1598%,reciprocaleps.001 40.2891%,cap50
 33.3165%. Weighted objectives decrease but strong weights broaden the
  high-damage band and worsen uniform loss/field error. Twelve checkpoint
  native/CSR/weight/H/oracle/Adam replays pass; two focused tests pass.
  Retained Adam moments and one frozen-H/source/budget limit attribution;
  this is not a rejection of every weighted L1 scheme. Diagnostic errors
  differ from coupled endpoint error42.4923% to common-parent direct.
  Supervisor154474/worker154477/auditor154476 exited; fine18811 restored
  at15:54:40UTC,observer20748 alive,coarse10891/13274 stopped.
  Scientific fields/curves and comparison JSON inspected/archived; source,
  config/backend/driver/auditor/tests and README/memory changes uncommitted.
  Latest recommendation (2026-10-14): prioritize a same-parent/frozen-H
  L16 control, distinguish loss from optimizer with L1/Adam,L16/Adam and
  L16/L-BFGS, then validate coupled .0052->.0053 if useful. Gentler bounded
  damage weighting is secondary. Recommendation only; no new trial launched.
  Broad task active; exact plan and limits in the active task.
- All-node L1/Adam morphology control COMPLETE (2026-10-13): same own-L1
  .0052 parent,fixed.0053,omega1/sigmoid/Adam1e-5,all71307 nodes/128 inner/
 644 outer; no symmetry/straight labels/extra weights/load advance. Same-H
  pilot selected Adam under full L1 loss rule; both screenings replay.
  Final R228.51959,tipx.827920/no span,physicalAT2.240949,fullL1 8.87615e-5,
  damageL2to common-parent direct42.4923%,Adam868864,18.06min.
  All saved stages/final independently replay. Field unstable/uncertified.
  Random128 R382.70689/15.34min;random512 R227.14351/52.47min,tipx.831384.
  Full128 improves crack progress versus random128 and nearly matches
  random512 force/length with fewer Adam updates; peak residual is worse
  than random512. Final cloud/path waviness persists, no cleaner-path claim.
  Off-band means full.02986/direct.02918, so nonzero background alone is
  not evidence of neural error. Same-overlap transverse-peak diagnostics
  remain P1/mesh sensitive. Final scientific clouds/path inspected; paired
  comparison JSON in full_damage_trial_20261013/independent_review.
  Worker147035/supervisor147033/CPUaudit147034/figure148905 exited;
  supervisor restores fine18811 at15:28:02UTC (23:28:02 Asia/Shanghai).
  Fine latest539/.005275,R448.20890;observer20748 alive,coarse10891/13274
  stopped. No new code/training/test/commit/slides change this status turn;
  PROGRESS/task updated,DECISIONS unchanged,git diff --check passes.
  Broad task active: next review representation/inner precision/physical
  accuracy rather than attribute path waviness to random batches alone.
- Equal-direct-outer unrelaxed PINN controls COMPLETE (2026-10-13): same
  own-L1 .0052 parent/floor,fixed.0053,omega1,L1/Adam1e-5/sigmoid/core+halo,
  total644,no early stop/load advancement.512-inner R227.14351,tipx.831384,
  no span,physicalAT2.121181,dL2to common-parent direct41.8324%,52.47min.
  128-inner R382.70689,tipx.696284,no span,AT2.331119,dL2 60.9365%,15.34min.
  Neither field stable/equation certified. Direct644 R1.58091/spans,2.51min.
  Both finals/snapshots and1032/322 new blocks independently replay; GPU/
  CPU workers and figure/final-review watchers exited successfully. Suite
 136211 restores fine18811 at14:29:45UTC (22:29:45 Asia/Shanghai);
  observer20748 alive,coarse10891/13274 stopped. Fine latest508/.005275,
  R458.98055,original scheme unchanged. All equal644 results/curves/cloud
  under staggered_damage_644_suite_20261013/independent_review, inspected.
  Equal extra82432 Adam updates:128-inner644 R382.71 vs512-inner161
  synced-stage R519.88; first322 core streams match exactly. Baseline161
  uses scalar journal only, no native161 field replay; efficiency clue,
  not an accuracy/convergence certificate.128-inner can be worth studying
  at larger outer counts rather than discarded from equal-outer comparison.
  No new trial/loaded source change/tests/commit/slides edit this turn.
  PROGRESS/task updated;DECISIONS unchanged;git diff --check passes.
  Broad task active: next review completed controls and plan equal physical
  accuracy/cost or inner precision checks. Exact sources/resume in task.
- Mechanics-first stagger/omega1/field-change diagnostic COMPLETE
  (2026-10-13), root staggered_damage_trial_20261013. Common own-L1 .0052
  parent/count786432/RNG, fixed.0053; model/history unchanged. One mechanics
  per outer plus one endpoint sync, no budget load advancement. Exact P1
  RMS stopping (d1e-5,u/U1e-4,3passes); residual is diagnostic.
  Both direct controls finish/replay: omega.5 1294/294.23s,R1.58108,
  physicalAT2.001165; omega1 644/150.33s,R1.58091,AT21.11e-6. Both span;
  matched reference dL2.7447%, mutual dL2.0006453%. Removing damping roughly
  halves work. Loose1e-3 checks first pass5 while cracking still advances.
  Both neural controls complete128x512/count851968/~628s: omega1 R539.46480,
  dmax.997653,tipx.554256,physicalAT2.187472,dL2to reference75.7237%;
  omega.5 R602.87087,dmax.916859,tipx.5,AT2.063904,dL281.1664%.
  Neither spans or meets field-change stopping; no load advancement.
  Same-budget direct omega1 (after128 damage updates) R515.30860 also
  remains before final unloading. New omega.5 vs old L1 R604.80677 differs
  .3201%; damage fields1.7213%, not bitwise identical. Both new512-block
  core streams match original L1 after parent count; original sources pinned.
  Independent native/CSR/H/BC/P1/force/Adam/core replay passes all4 finals
  and snapshots; worker/CPU audit/figures exited. Supervisor exits0 and
  restores fine18811; observer20748 alive, coarse10891/13274 stay stopped.
  Two pre-iteration failures archived/fixed;4 new focused tests plus3
  unchanged history/grid tests pass. Completed plot and comparison JSON
  under independent_review reviewed; no continuation of new128 states yet.
  Next consider fixed.0053 additional outer work and inner adaptation under
  new omega1 scheme. Call counts: old one initial plusN outer; new N outer
  plusone endpoint synchronization. Both N+1, not one fewer per outer.
  Root preparation also compiles one shared mechanical preflight, excluded
  from per-case counters/timers; no speedup claim from changing order alone.
  Broad task remains active; exact original fine resume in task.
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
