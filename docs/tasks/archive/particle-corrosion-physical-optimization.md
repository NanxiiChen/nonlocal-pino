# Physical particle-corrosion optimization

Goal (user-revised and now accepted): choose a dimensionally consistent
particle-corrosion regime while keeping substantial evolved corrosion and
visible particle-induced interface changes;then solve it by pure-physics PINO.
The user now authorizes parameter changes. Retain the old high-mobility case
and failed controls as separate evidence,not as a newly solved old problem.

Acceptance:normalizedtime0..1;explicitSIcoefficients/scales;reference morphology
showsrealgrowth/particleinteraction;trainingusesnoevolvedfieldlabels;report
nativeandlocalerrorsseparately. Near2%remainsanaccuracyaspiration,buttheold
strictfixed-physical-parameterrequirementwasexplicitlyrelaxedbytheuser.

Context: archived physical pilot gives 35.42% phase pooled/46.88% terminal;
production-native first-case 33.61%/46.24%. Both particle/electro use originalL1,
joint Adam/unit scales/no causal. Particle data concentration normalization,
LR schedule, grouping and initial transient differ. Prior turn made diagnostic
progress; no live job at start (GPU/ps verified).

Status: complete for the user-accepted finite study and README publication.
The user accepted the width256 fields, cleared the earlier app goal, and asked
for the example README to be updated. The near2% local aspiration was not
achieved and is not reclassified as achieved. No more optimization is scheduled.

The README now describes the six-particle50x30um/350s problem, original joint
AC/CH scheme and explicit width256 preset, and shows three layouts/five times
from the actual20k endpoint with direct-native reference contours. Local FEM
and native metrics remain separate. Root overview, PNG/SVG provenance and
asset index match. The numerical train_default selector is unchanged and is
clearly identified as the separate high-mobility5s research case.

Blocker: none. Resume artifacts, if new work is requested:
`/root/autodl-tmp/data/particle_corrosion2d/20261107-spaced-six-width256-02/assessment.json`
and `capacity_comparison.json`. New owners/evaluators are terminal. The old
high-L horizon group618216 remains explicitly paused, with its recovery record
retained below; do not automatically resume or discard it.

Width256 final local phase2.933% overall/5.757% terminal; native1.450%/2.781%.
Width192 local3.091%/5.985%; native1.637%/3.110%. The same four development
geometries and actual input/time streams are used; parameter shapes and initial
Fourier realization differ. These results are not a cross-seed robustness proof.
No Git operation, cleanup, numerical-default promotion or extra experiment.

## Historical investigation log

The entries below are chronological. Their live/queued statuses describe past
stages; use the current status and resume artifacts above for ongoing work.

First checks: 11 existing loss/dimensional tests and 2 initial-lifting/metric
checks pass (CPU). Slow-LR run is live; at4k phase32.41%, not a solution.
Independent production-mesh direct native vs local FEM:6.5794%phase pooled,
8.3338%terminal. Interior trace second moment3.93458 versus4, so a gross kernel
normalization error is not the main bias. A ~2%local target requires eventual
operator refinement as well as neural convergence. Audit/script/results in
/root/autodl-tmp/data/particle_corrosion2d/20261106-physical-optimization-01.
The first queue launch failed before any child (psutil absent); replaced only
process observation with standard /proc identity checking, preserving error log.
Serial hard-IC,physical-data metric and hard-IC/log-time controls are queued.

Slow-LR5000 complete: dev4phase33.5169%pooled/45.1978%terminal (fresh seed;
not an exactly paired attribution). Hard-IC/uniform-time5000 is running next.
Detected FP32 hard-lifting mouth endpoint subtraction misclassification; use
cast endpoint comparisons and add a focused endpoint test (passes). Existing
hard run imported the earlier source; retain that distinction. Its IC mismatch
is ~1e-6 MSE localized to these endpoints, not evidence of interior hard-IC
failure. Subsequent queued runs use the corrected source. Two log-time sampling
tests and two hard-map/full-JVP tests (including log decoder/t=0) pass.

Independent half-horizon/finer mesh check completed:delta.013,h.0075,13282nodes,
phase4.0787%pooled/5.0685%terminal vs local at identical points; c4.8235%/6.0042%.
This confirms reduction versus production6.58%, but does not establish a2%floor.
Hard-IC/uniform-time5000 completed; metric5000 is now live under serial owner.
Subsequent hard/log config uses4losses with no redundant IC query/penalty;
first hard run's five-term source stays frozen and must not be conflated.

## Current resume point (task-end, incomplete)

SI physics unchanged in all5candidate presets (explicit config-invariants check).
Completed fresh-seed5000development4geometry screenings:

| Candidate | Phase pooled | Phase terminal | Concentration pooled |
|---|---:|---:|---:|
| Slower LR |33.5169%|45.1978%|33.0937%|
| Hard IC,uniform time,tau.01,old5loss source |33.2921%|43.3517%|33.6722%|
| Slower LR,physical concentration data metric |29.6848%|34.8031%|30.2997%|

These have not reached the goal; they are exploratory fresh-seed candidates,
not a strict paired causal attribution. No reserved8new assessment or promotion.

The newly introduced tau.01 initial lifting itself prevents representing some
correct initial derivatives: phi0+tau*F0 falls outside[0,1]in14330/131072initial
core rows across64geometries. Tau.001 has0violations in this bank. This finding
concerns the new hard-map trial, not the original soft-IC failure. The first
old-tau log trial was deliberately terminated before an update; preserved
hard_log_aborted.json and queue_status.json(exit-15). No data/log was deleted.

Live corrected serial owner PID308948, child308950 at last /proc check:
/root/autodl-tmp/data/particle_corrosion2d/20261106-physical-optimization-01/queue2.py
and queue2_status.json. Running corrected hard_log5000-02 uses tau.001,
32global log-stratified times including0,log decoder scale.01,4losses,noICsoft
query. Automatically next: hard_log_l25000-01, same setup with squared PDE.
Both are cold pure-physics runs; Adam,space nonlocal,timeJVP,rolling inputs.
Read actual queue/process handles before resuming; never restart just on timeout.

Validation:11existing loss/dimensional checks,4new initial-map/endpoint/physical
oracle checks and2sampling checks pass in focused batches. One new test first
failed on attempting in-place normalization of a read-only NumPy view; corrected
only its oracle code and reran successfully (log /tmp/particle_hard_oracle_test.log).
Default uniform random stream is exact; initial identity/log-time JVP matches
finite differences; squared physical AC/CH oracle and4lossIC-independence pass.
Restored original division arithmetic in the absent-data-metric branch.
Config invariants,focused source review and git diff --check pass. No Git action.

Changed this session: optional training maps/loss metric/time sampling and5
research presets,focused tests,example README/development docs; memory files
PROGRESS,DECISIONS,this task and adjacent JSON. Prior uncommitted work retained.
Next action: inspect corrected log-time runs and fields, then extend successful
scheme. If still wrong, diagnose early residual and causal/stagger/L2 behavior;
if neural error falls, reduce finite-horizon bias before claiming2%local error.

Continuation: prior turn made concrete progress (three screenings,17focused
checks, native bias and initial-tangent audit). Corrected hard/logL1 finishes:
dev4phase30.0881%/40.7417%,c30.6886%; still wrong. L2control follows in queue2.
Add a qualified latent_additive/zero-increment-head/exact-mouth candidate to
avoid having the free neural phase reconstruct phi0 at the initial derivative.
Known z0 is only the fixed initial tanh coordinate, with no prescribed velocity.
The exact mouth mask sets phi=s=0 at reservoir nodes as in reference BDF;
only two PDE losses remain. Focused CPU identity/mouth/JVP/two-loss test passes.
Serial latent owner waits for queue2 L2 completion; direct paths/handles in
study/queue_latent_pid.json and latent_status.json. More refined direct native
and local solves also running on44518nodes,h.00375,delta.0065 using SciPy only
(no FEniCS MPI): PID319527,study/refine_native.log.

## User-directed simplification

User paused automatic goal pursuit and redirected this turn to original output
and established strategy diagnosis. All prior jobs now terminal; no GPU process.
Latent/exact-mouth trial ends31.3727%phase/43.1750%terminal, not a remedy.
Refined native first-case file is complete (study/refined_native.json); retain
it as discretization evidence, not the cause of the much larger neural error.
This turn: compare old/new kinetics and early AC/CH, then bounded matched cold
stagger/causal/CH-priority controls with the original ansatz. No new exotic maps.


## Focused strategy diagnosis, current user request

Broad automatic goal remains paused in the app; the current explicit request
authorizes finite original-output strategy diagnostics. Prior custom-map jobs
are terminal. Original phase/KKS outputs,softIC/BC,original32uniform times,
architecture,physical coefficients,optimizer and rolling data are retained.

Independent first-geometry old/new native early solves0..05both grow, without
the neural collapse. Effective(Nch,N1,N2,N3):old80/48.636/8/.004444,new1.7/1070/
176/.103. Bulk double-well relaxation scale.0625->.00284091; width-based CH
scale6.70527e-5->.00332395. Their ratio changes1090.59times (scale estimate,
not full coupled spectrum). Source/script/results:study/early_physics.py/.json.

Original20kfirst-geometry near-reference-interface audit at0: predictedphi_t
+83.8587,ownAC RHS-44.7832,directnativephi_t-12.9997. Thus it violates even its
own AC evolution direction. phiabsoluteRMS.0146664,cRMS.00121154 coexist with
qRMS/cLe.450496. Replacing q alone diagnostically lowersACRMS74.50->54.76,
not to0. At.05 the originalPINOcorrosionfraction.8528%vsnative5.4143%; early
jump enters an incorrect low-residual branch. These are one-geometry diagnostics,
not whole-population proof. Frozen original source and whole4080node graph used.
No reference fields are added to training. Data:study/early_network.json/.npz.
Figure/reproduction script in the visualizations directory,plot_corrosion_early_diagnosis.*.

New source options: established stagger objective mask (AC/CH alternating25
updates for first2000,IC/BCalways on); adaptive per-time causal with the shared
CausalWeightor controller,one common stopped prefix weight for summedAC/CH
segments and unweightedIC/BC; post-gradient-balancing ACpriority10. Pure raw
PDEpre-scales can largely cancel under inverse-gradient balancing; do not
mislabel post-bias as a changed physical coefficient or proven optimum.

Three5000controls are serial under study/original_strategy_owner.py,
ownerPID348640,causalchild348641 at last /proc check. Status journal:
study/original_strategy_status.json. Order:causal,its early audit,stagger,audit,
ACpriority,audit. Each uses --strategy-control pointing to original20krun.
The exact model_initial/optimizer_initial file bytes match; all5000actual
condition/pool/time arrays match source prefix after independent fullsource
RNGreplay. Physics and all non-strategy settings are checked explicitly.
Rollinginput seed/cadence retained. A single paired initialization is not a
cross-seed stability study. Scalar causal state and active selectors are logged.

Verification:3new strategy tests pass (completeAdam masks,data terms,physical
causal NumPy prefix oracle,adaptive controller/fullAdam update).2focused old
loss/completeAdam regressions pass. ActualcausalGPUcoldstart and prefix checks
pass; git diff --check passes. No Git operation,cleanup,default promotion or
new custom output map in these controls. Read current handles/journal before
continuing; no restart on observation timeout. Need inspect completed accuracy
and early signed dynamics,then decide whether initial-time sample coverage must
be strengthened (91.3%of32uniform updates miss t<the newreactionscale).

Causal5000finished:dev4phase35.69399%pooled/47.00101%terminal,no accuracy recovery.
Controller epsilon adapted from1to~.01; this is not a fixed-epsilon trial.
The serial owner now advances through its frozen-model early audit then stagger
and ACpriority. Current actual stage/PID is saved in adjacent strategy-controls
JSON; do not rely on the historical child348641. The next decision must assess
initial-time coverage,not add output-map constraints.

## Reactivated goal and next controlled tests

User reactivates2%goal; retain original output route. Stagger5000complete:
phase35.1098%pooled/46.4912%terminal; no recovery. ACpriority10runs next.
Additional initial-layer rule replaces only8of32times (0plus7resampled strata
ending4/(2*kappa*N2)=.0113636); the24global source times and all geometry/pool
IDs stay exact. Expected nodes before the reaction scale increase to2.81/update
from0.091; first0node guaranteed. This declares a different temporal residual
measure,not importance-corrected uniform sampling or a decoder-coordinate map.

Equivalent ICmetric(phi,s) addresses qerrors hidden by physical(phi,c)loss;
s0=(c0-dc*h(phi0))/cLe is computed from known IC only. Zero set unchanged,
PDE/BC/architecture unchanged. The original normalizedphysical IC Jacobian has
large off-diagonal dc*h'/cLe, so natural variables improve local conditioning.
Two5kcontrols queued after the prior owner: causal_early_ic first,then
causal_early,each followed by frozen whole-graph early audit. Both cold states
and actual time/condition/pool traces pair exactly with each other; only the
ICmetric differs. Against original root,changed early times are explicitly
recorded(source_time_draws_identical=False).5focused sampling/ICtests pass;
actualcontrol-prefix generation validated. QueuePID in study/early_controls_pid.json.

Next combined conventional candidate is queued after the two early controls:
train_physical_causal_early_ic_warmup_l2.py,10000total=1000knownIC/BC+9000PDE,
MSEoriginal residuals and stronger adaptivecausal(min_mean.1). Standard model,
no custom phase output and no evolved labels. First5kcheckpoint remains distinct
from final10k. Source-control cold/prefix checks pass; warmup objective-mask/
epsilon-held fullAdamtest passes. Owner and exactrun in study/warmup_owner_pid.json.
Owner always performs frozen early audit; ifdevphase<15%,then8local-reference
assessment and fullonlineinput audit are automatic. Otherwise it records explicit
nonconvergence and defers extra reserved evaluation. No claim near2%yet.

The warmup/MSE/stronger-causal10000control finishes30.2678%phase/39.9598%terminal,
no solution. All its owned stages finish; extra reserved evaluation explicitly
skipped due poor development accuracy. New integral route is a left time-integral
preconditioner of the same IVP,not another output map.32endpoints+31midpoints,
1000data+4000PDEupdates,original joint Adam/GradNorm,causal off. Independent BDF
quadrature defects mean/max phase.0544/.1105%,c.1267/.2193%on32random grids.
Polynomial exactness,fullphysicalmatrixoracle,parameter-gradient and old strong
residual checks pass. One endpoint test initially comparedFP32time toFP64literal;
fixed its expected dtype and all4samplingtests pass. Six other checks had passed.
Integral owner/run/commands in study/integral_owner_pid.json and integral_status.json.
It always audits strong original PDE on fullgraph; ifdevphase<15%,8local-reference
evaluation and complete online audit follow automatically. Actual memory at first/
25th production updates is recorded. Goal remains unmet and active.

## Latest completed original-output controls

All percentages below use the same development4 local-reference geometries.
The paired cold model/Adam state and unchanged geometry/core draws are checked;
the new early-time measures are explicitly declared, not identical uniform draws.

| Control | Total updates | Phase overall | Phase terminal | Concentration overall |
|---|---:|---:|---:|---:|
| Original prefix | 5000 | 35.6374% | 46.8530% | 34.9565% |
| Adaptive causal | 5000 | 35.6940% | 47.0010% | 34.9598% |
| Stagger, then joint | 5000 | 35.1098% | 46.4912% | 34.5421% |
| AC priority 10 | 5000 | 37.5602% | 48.5322% | 97.2075% |
| Causal + initial-layer times | 5000 | 29.4567% | 34.6764% | 30.1209% |
| Above + phase/liquid IC metric | 5000 | 30.5665% | 36.3645% | 31.1686% |
| Warmup + MSE + stronger causal | 10000 | 30.2678% | 39.9598% | 30.9266% |
| Warmup + global Simpson balance | 5000 | 25.2032% | 18.1329% | 26.0234% |

The warmup cases contain 1000 IC/BC-only updates. Their PDE budgets are therefore
9000 and 4000, respectively. Natural IC improves initial liquid-driving accuracy
but worsens phase IC fit; it is not a demonstrated whole-trajectory remedy.
The integral case has visibly incorrect shrink/regrow oscillations despite the
lower terminal error. At first/25th updates its active peak is 17.91 GiB;
the allocator reserves 23.52 GiB. Owner PID433370, final-audit child443311 at
last observation; read the journal/process state before resuming. Prior owner
PIDs in historical sections are terminal, not current instructions.

Integral audit is now complete. First-geometry native errors22.52%overall /
14.31%terminal; same-mesh local26.02%/20.32%. Corrosion fraction grows to.0732
at.03,shrinks to.0241at.2,regrows to.1016at.7,and shrinks to.0363at.8.
Original whole-graph AC RMS remains6.3..22.6,CH1.6..14.2at audited times.
These are differently scaled physical residuals,not a loss-dominance ranking.
No further integral budget is justified by a claimed accuracy recovery.

Next conventional control uses `train_physical_causal_early_soap.py`: only
Adam->sharedSOAP,b1.95,b2.9,precondition frequency10,max dimension256,
original strongL1residuals,softIC/BC,32early/globaltimes,adaptivecausal.
Owner447605/child447606,study/soap_status.json and soap_owner_pid.json.
Strategy validation checks identical model and actual inputs but intentionally
uses each optimizer's own fresh state; runtime explicitly records the exception.
One focused test passes for cold-state pairing,trace replay,and rejecting an
undeclared learning-rate change. The original same-optimizer check is retained.

CPU-only independent diagnosticPID460101 reconstructs predetermined final
training slots62/63 from the completed Adam early run. It verifies conditioning and original graph coordinates,
then computes same-mesh direct-native/local fields,followed by frozen-model
inference. Files:study/training_geometry_audit.py,training_geometry_adam.log,
training_geometry_native2/manifest.json and eventualtraining_geometry_adam.json.
No training process reads these fields. This checks optimization vs geometry
generalization,not a new qualified local-FEM population benchmark.

The training-geometry diagnostic is complete. Important identity correction:
online initial bank IDs are negative (-1-slot); final positive draw62/63 were
installed during rolling replacement, not present for all5000updates. Actual
trace counts are62and63, recorded explicitly. They are reused training cases,
not untouched initial-bank geometries or fully converged per-case fits.
Phase errors versus native are26.7852%/26.1679%,versus same-mesh local
29.1875%/28.5419%,similar to development4's29.4567%. This supports a substantial
optimization failure rather than a large generalization gap in this small
diagnostic; it does not establish population-level equality. References live
only in the diagnostic directory and never enter training.

SOAP finishes all3owned stages. Pair receipt confirms model bytes and every
actual sampling/bank array match Adamearly;onlyoptimizerdiffers. At0 its
phiRMS.01128,qRMS/cLe.3375,predictedphi_t-.736vsownRHS+2.164/ref-13.00,
so smaller physical residuals coexist with wrong driving and trajectory.
Its30.1791%phase is not a recovery. Goal remains active and unmet.

Next `train_physical_causal_focus.py` retains all warmup/MSE/naturalIC/Adam
settings and changes only max_last_weight.9->.1,min_mean_weight.1->.05.
The old10kcontroller endedwithlastweight.746,mean.845,so its stronger-minmean
label did not imply strong eventualcausal focus. Explicit thresholdvalidation
and independent rootprefixreplay pass. Owner470246/child470602,
study/focus_status.json,focus_owner_pid.json. One10kglobalnetwork,no windows,
no reference targets;1000data+9000PDEupdates. Owner always performs original
whole-graphPDE and early audits and checks pairedcold/input arrays against
warmup_l2_causal10000. Reserved8onlyifdevelopmentphase<15%.

## Latest resume point: early dynamics, not more IC constraints

Strictcausal10k and both original-PDE/early audits finish successfully,accuracy
unmet. Development4phase27.9505%overall/38.7188%terminal,c28.7886%overall.
Its first-geometry nativephase25.3967%,same-mesh local28.7659%.
Final causal lastweight.00109,mean.1203,epsilon.31863:focus is actually strong.
At0,near-interfacephiRMS.006697,qRMS/cLe.005693,predictedphi_t-11.926vsownRHS
-10.685andnative-13.000. At.05,predictedphi_t+.446vsnative-2.426,so later
dynamics still fail. The5ksnapshot earlier had initialrate-9.765. Pair receipt
focus_pair_validation.json verifies model/Adam bytes and all actual input arrays.

Independent SciPy initial_sensitivity.py evolves the unchanged full native
4080node PDE from the final learned physical IC (mouth values projected to
their prescribed0). Initial whole-field errors are.2093%phase/.1911%c.
Resulting trajectories differ from exact-ICnative by:

| Perturbation | Phase overall | Phase terminal |
|---|---:|---:|
| Both learned fields |0.071996%|0.028920%|
| Learned phase only,natural liquid unchanged |0.073872%|0.031094%|
| Learned liquid only,phase unchanged |0.003161%|0.002342%|

These are one-geometry sensitivity diagnostics,not PINO accuracy. They disprove
remaining IC-error amplification as the explanation for this model's~28%error.
The six-loss IC-group proposal was implemented/tested but removed before any
training run on this evidence. Its test logs remain external; no live code,
preset,checkpoint or additional IC constraint is retained.

Raw-gradient audit (raw_gradient_audit.py/.json) on one original-bank geometry,
four times,and original20kcold/finalnetworks:physicalL1norms do not reach the
1e6norm ceiling;MSEevaluation can (initialAC2.680e6). Some boundary-weight
proposals reach the1000cap. Thus scale cancellation is conditional,not exact;
these are not raw gradient measurements of every causal-training batch and do
not prove clipping caused the failure. Natural ICcomponent diagnostic atfocus5k
has MSE7.27e-5/8.89e-5,norms.886/4.016;not a reason to override the independent
sensitivity result. No physics/data targets changed by these diagnostics.

New run `20261106-original-causal_log10000-01` uses
train_physical_causal_log.py:only arch.time_coordinate changes from linear to
the existing logarithmic option,tau=1/(2*kappa*N2)=.0028409091. The physical
reaction time maps toembedding.11815instead of.00284. Physical times/JVP,
original outputs,MSE,strictcausal,naturalIC,1000warmup and9000PDE remain.
Owner490104/child490105,study/log_status.json and log_owner_pid.json. The owner
checks exact coldmodel/Adam/input pairing against strictcausal10k,then always
audits full originalPDE and early dynamics. Reserved8/onlineaudit follow only
ifdevelopmentphase<15%. No promotion. A focused control test passes for the
allowed coordinate option and rejects changedwidth/LR. Earlier full log-time
JVP/oracle tests remain valid;no derivative implementation changed.

Diagnostic figure/reproduction:
visualizations/2026/10/24/01a16d71-2d47-7201-9441-ab74e0523152/
plot_particle_initial_sensitivity.py/.png. It compares prescribed/learned-IC
direct-native trajectories with PINO fields;it is not another training loss.

## Queued bounded KKS row-elimination control

Independent kinetic_coordinate_audit.py/.json checks the exact chain rule on
the native and frozen PINO fields. At0,the native narrow-interface averages are
phi_t-13.000,s_t806.946,c_t10.161;the two c_tcontributions are+28.779and-18.618.
Original20k givesphi_t+83.859,s_t-151.349,c_t114.390. Strictcausal10k gives
phi_t-11.926,s_t724.543,c_t8.839,showing genuine initial recovery while later
front propagation still stalls. This does not identify a full-network Hessian.

Atphi.5,the raw-rate Jacobian rows have normalized cosine.998786 and condition
40.58;bounded row elimination makes them diagonal before per-loss balancing.
The explicit numerical matrix check is time_rate_jacobian.json. This motivates
testing Rtilde_c=Rc-dc*hprime(phi)*Rphi with original Rphi,not claiming a
general convergence theorem. The coefficient is bounded by1.5dc. Evaluate
cle*s_t+Gc-dc*hprime(phi)*Gphi directly using raw s_tJVP;no division by the
phase Jacobian,no new hyperparameter and no new IC/output prior.

Implementation is optional loss.transport_residual=ac_eliminated in
train_physical_kks_eliminated.py. Five labels remain,with transformed transport
explicitly named ch_ac_eliminated. MSE,strictcausal,naturalIC,1000warmup and
10kbudget remain exactly those of the linear-time strictcausal control.
Two focused tests pass:independent NumPy physical-matrix/map oracle,zero-set
and phase-rate-decoupling checks;full network IC/BC/AC invariance and parameter
gradient vsfinite differences. No derivative implementation or output map was
changed on the default path. Log:/tmp/particle_kks_elimination_tests.log.

Queued owner/script/journal are study/row_owner_pid.json,row_owner.py and
row_status.json. It waits for log_owner to finish all audits;never starts a
second GPU training concurrently. It compares actual cold model/Adam and every
input array to strictcausal10k,then audits original full-graph PDE and early
dynamics. If the preceding log run reaches<5%development phase,it defers this
new test for accuracy review instead. No promotion or reference targets.

The log-coordinate run and all3ownedstages now finish. Development4phase
27.6840%overall/39.8073%terminal,c28.5475%overall;first-case native25.1345%,
same-mesh local28.5175%. Pairreceiptlog_pair_validation.json passes.
At0,it predictsphi_t-12.803versusnative-13.000,and near-interfaceqRMS/cLe.00400;
by.05phi_t+.333versusnative-2.426. Initial dynamics improved but later drift
persists. This does not qualify the physical model or warrant default promotion.

Row-elimination owner503373 has started child507923 as
`20261106-original-kks_eliminated10000-01`;1800updates at last observation.
Exact command and stage inrow_status.json. Read current handles before resuming,
not historical finished PIDs. The optional configuration passed problem/strategy
validation and rootprefix replay before launch;2new physical/gradient tests pass.
git diff --check passes. Default configuration and all physical coefficients
remain unchanged. No Git operation,external cleanup or source/checkpoint
migration was performed.

## Current single-geometry diagnostic

Row10k and its2audits finish:development4phase28.7734%overall/40.2496%terminal,
c29.5296%overall. Original-PDEfirstcase native26.2891%,same-mesh local29.5984%.
At0phi_t-11.58(ref-13),qRMS/cLe.00736,but later dynamics remain wrong.
Pairreceiptrow_pair_validation.json verifies exact coldmodel/Adam/input arrays.
The bounded row operation is algebraically valid,not an accuracy remedy here.

Stop adding residual variants. The next original-output strong-residual control
uses one fixed known geometry(previous training slot62),not an easier geometry
or changed material. `train_physical_single.py` keeps192-wide architecture,
Adam/MSE/strictcausal/naturalIC/1000datawarmup and10000totalupdates.
Batch1,core1024,IC4096,Nt32 preserve the previous per-update totalPDE/ICcounts.
There are32fixed complete-halo pools,with no rolling geometry or spatial refresh.
This changes sampling/geometry structure and uses a fresh runtime modelseed;
it is not a paired one-factor causal ablation or an operator accuracy claim.

Known-input preparation (study/prepare_single.py) takes1.824s. It uses4093nodes,
halo3968,40neighbors;bankshapesgraph(1,32,3968,2),core(1,32,1024),
IC(1,4096,2). Mesh points/volumes match existingtraining_geometry_native2/case62
exactly. Only known geometry/IC/operator arrays enter inputs.npz;the script
does not read evolved fields. load_bank/problem/strategy validation passes.
Bank:study/single_geometry_inputs. Input seed is provenance,not a tuned preset.

Run `/root/autodl-tmp/tf-logs/particle_corrosion2d/20261106-single_geometry10000-01`,
owner524606/child524607,study/single_status.json and single_owner_pid.json.
The training entry remains train.py. Its periodic development4 are distinct
geometries and are not the criterion for this single-case diagnostic.
After training,owner runs study/evaluate_single.py,which loads the frozen
network,compares against this reused training geometry's existingnative/local
fields,and audits originalAC/CH on its entire graph. Results/plot will be
study/single_geometry_final.json/.npz/.png. This evaluation never feeds gradients.
Local same-mesh error is not a newly spatially-qualified population benchmark.
No automatic SOTA/goal promotion follows even if this one geometry converges.

If this fixed single case succeeds,investigate geometry-conditioned optimization
and a physics-only geometry curriculum before changing the PDE. If it fails,
the basic neural PDE optimization/representation still needs diagnosis. Do not
substitute new loss tricks without inspecting this result. Goal remains active,
approximately2%onheld-out local-reference geometries is not yet achieved.

## Latest conventional fixed-case controls

Single10k and evaluation complete:trainingnativephase24.3833%overall/34.0594%
terminal,c25.4813%overall;local same-meshphase26.8538%/37.1231%. At0,original
whole-graphAC/CH RMS=.1274/.1514;laterCH~.3,and corrosionfraction stalls near.05.
At5k trainingnativephase25.1734%,so10k is not a clear convergence recovery.
Measured activepeak5.23875GiB,pool8.99805GiB,including periodic evaluation.

Two controls retain fixedgeometry/B1/core1024/IC4096/Nt32/pools32/MSE/strictcausal/
1000warmup/10kbudget. SingleSOAP changes only optimizer;singleCHpriority changes
only post-balancing gradnorm_bias[1]1->10. It is not a physical coefficient or
rawpre-scale. Unlike earlier5kSOAP,this tests naturalIC/warmup/strictcausal.
Both match the source coldmodel and actual draws. SOAP has its own factory-cold
state,explicitly different. No new residual or output map is introduced.

Presets:train_physical_single_soap.py/train_physical_single_ch_priority.py.
Strategy replay permits a fresh initial_layer source only with identical target
rule. Regenerate the fullsource schedule,then select a prefix without a second
time transform. A prefix test caught target-budget shape validation;fixed to
ConfigDict(prior),retest passes. Olduniformroot behavior retained. Runtime time
identity now compares actual source/target arrays. Bothreal10kdrawreplays pass.

Owner540161,children540163(SOAP)/540164(CHpriority),understudy/
single_controls_owner.py and single_controls_status.json. Runs overlap with
preallocationoff based on measuredmemory. Accuracy comparison only;walltimes
are not fair optimizer-speed benchmarks. After both trains,owner checks
coldstate/config/trace pairing and serially runs evaluate_single.py. Results:
single_soap_final.json and single_ch_priority_final.json/.npz/.png. These are
single-training-geometry diagnostics,not held-out operator/goal qualification.

CPU4loss_gradient_geometry.py(child548878)measures actual parameter-gradient
cosines on the baselinefinal model's last payload/time/epsilon,without reference
fields. core_coverage_audit.py checks never-sampled small-volume/boundary nodes
and their share of full-graph residual. Read JSON/logs before claiming gradient
dominance or sampling failure;rawloss size alone is insufficient.

## Current gradient-normalization control

Both parallel fixedcase controls and their evaluations finish. SOAPnativephase
24.0889%,c25.2223%;CHprioritynativephase23.9441%,c24.7983%. SOAPmouthcmax~2e-5;
CHprioritymouthcmax.00589and larger IC errors invalidate a claimed physics
improvement. CHresidual loss alone was reduced at the expense of data constraints.
Both pairreceipts pass;initialmodel/draws match,SOAPstateintentionally differs.
Parallel GPUmemory observed19768MiB. Their overlapping walltimes are not fair
isolated optimizer benchmarks.

loss_gradient_geometry.json:post-final baseline onexactlastsampled payload/
time/epsilon,AC/CHparametercos+.25855. WeightedEuclidean normsAC.609/CH.793/
IC20.14;combined20.80vsindividualsum25.00(cancellationratio.832). This one
post-update batch does not establish global causality. Existing inverseL1norms
plus25-update weightlag do not ensure equal Euclidean updates. FixedICgradient
can remain large evenwhen its value is small. No strongAC/CHgradientconflict
was measured in this snapshot,so do not assert that as the main cause.

core_coverage_audit.json:32768draws,4084free nodes,745nevercore(722boundary),
but only.9154%physicalvolume. At0,allpool/fullMSEratiosAC.794/CH.902;at1,.840/
.862. Missingnodes carry13.7–14.8%ofACsquaredresidual and1.9–6.6%ofCH atthese
endpoints. Sampling error exists,but this does not demonstrate that it causes
all24%trajectoryerror. No new sampler or full-grid API was introduced.

Optional loss.gradnorm_norm=l2 now uses stable scaled Euclidean accumulation;
missingkind preserves historicalL1summation. Both ordinary andcausalsteps use
it;nonfinite raw norms now trigger the finitefailureflag beforeclip can hide
them. Newtrain_physical_single_gn_l2.py changes only this metric and refresh25->1.
PDE/outputs/physicalcoefficients/knownICBC/Adam/lr/times/corepools remainfixed.
This is a combined normalization test,not separately attributed factors.

Threefocusedtests pass:zero/largefinite scalednorms,completeordinaryAdamupdates
for historicaldefaultL1andnewL2,completecausalL2Adamupdate versus independent
NumPy norms/updates. Log:/tmp/particle_gradient_norm_tests.log. Strategycontrol
permits explicitmetric/refresh changes and rejects invalid nonpositiveintervals;
updatedcontroltest is in /tmp/particle_norm_control_test.log. Exactsource replay
against completed singlebaseline passes. No broadunchanged suite rerun.

Newrun20261106-single_gn_l2_every1_10000-01;owner573847/child573848,study/
gn_l2_status.json and gn_l2_owner_pid.json. It starts after bothparallelcontrols
and audits end;no concurrentGPUload now. Owner checks identical coldmodel/Adam
and allactualdraws,then evaluates on the same4093node reusedtraininggeometry.
Resultssingle_gn_l2_final.json/.npz/.png. Still not held-out operatorqualification
or the localFEM2%goal. Read actualjournal/process before continuing.


## Causal-release versus joint optimization control

GN-L2/every1 finishes10k:single nativephase24.6386%overall/34.4487%terminal,
c25.7330%;local same-mesh27.0981%/37.4910%. Its whole-graph early AC improves
but late CH RMS stays~.3. At10k causal meanweight.3537,lastweight.03196.
Inverse-gradient balancing alone did not recover the trajectory;the gradient
snapshot hypothesis was insufficient. Ownergn_l2_status.json2passed,actual
coldmodel/Adam/input pairing passes. Preserve negative outputs.

Two new exact paired10kcontrols use this GNsource:jointlosses(no causal) and
causal until5000thenjoint. Both retain1000datawarmup,MSE,original boundedphase/
KKSliquid outputs,physicalcoefficients,network,Adam and all32resampledtime draws.
Newoptional loss.causal_steps names absolute steps,not a physical timewindow.
Afterrelease effectiveepsilon0 gives alltimeweights1;positiveepsilonstate is
frozen,not fed back from unweighteddiagnostics. Optimizer/model are retained.
This tests early causal learning followed by full-time fitting,not a new prior.

Focused physical-loss/fullAdam equality test passes against ordinarynoncausal
training,including refreshedL2normweights;controllerfreezes andweightsall1.
Schedulevalidation and existingcold/inputcontroltest also pass(3total)in
/tmp/particle_causal_release_retest.log. Initialtestharnesserrors(nonarray
Equinoxleaves andConfigDicttypecheck)werefixed;no failedtraining or numerical
fallback. Missingcausal_steps preservesoldstepbranch.

Ownerstudy/causal_release_owner.py PID591268;children591273(release)/591274(joint).
Preallocationoff,two jobs overlapbasedon measuredsinglecasepeak;do notinterpret
walltimes as isolatedspeedbenchmark. Allposttraining stages arefinite:check
actualcoldstate/config/drawpairs and loggedreleaseschedule,then independently
run evaluate_single.py. Readsingle_causal_release_final.json andsingle_joint_final.json.
No evolvedreferencefields are read duringoptimization. Neither singlecase
resultalone satisfies held-out localFEM~2%goal.


## Dense-time diagnosis on the same fixed geometry

CPU4 single_time_diagnosis.py completes;single_dense_native.npz independently
re-solves the unchanged4093nodegraph at60unique times(logarithmic early nodes
plus the11originalsnapshots). Original savednative snapshots match within
rtol1e-10/atol1e-12. No evolvedreference enters training. FrozenGN-L2 model
is evaluated with complete physicaltimeJVP and originalAC/CH. Files:
study/single_gn_l2_dense.json/.npz and single_dense_native_solver.json.

At tstar.00910 phaseL2 is.1409%;at.1,5.1521%;at1,34.4486%. The latest recipe
therefore starts correctly but fails aftertheinitiallayer. At.02947,mean
predictedphase rate on the reference.4<phi<.6front is-.699versusnative-2.702;
itsownACrhs-.870. Wholetrajectorycorrodedfraction eventuallystalls~.05,
while native reaches.17481. This is more specific than repeating that IC/BC
or time count are wrong. Original CH interfaceRMS rises near.03;do not infer
parameter-gradient dominance solely from these differently normalized rows.

q_driving_audit.py/.json keepspredictedphi fixed and substitutesnativeq only
as an evaluation sensitivity test. At.02947 the signeddrivingrate defect is
+1.305,comparable to the observedslowing;this is not a causal decomposition
of the complete trajectory. At.5 the predictedphi on the truefront is.9970,
and at1,.999875. The phase-mapJacobian there drops to.000249,so a small
physicalACresidual alone can coexist with a badly misplacedfront. Alltruefront
nodes atthose times have predictedphi>.99. Neither mask entersoptimization.

coefficient_scale_comparison.json independently recordslegacy->SI effective
AC1/AC2 factors22,ACgradient23.175,anddiffusion.02125. The reaction/transport
ratiochanges1035.29;localpurephase reactiontime.0625->.00284091. This is
not thefullcoupledeigenvalue spectrum and not the olderuniformspeed2test.
TheACq-sensitivity atphi.5 rises70.353->1547.759. Figure/script outsideGit:
/root/.codex/visualizations/2026/10/24/01a16d71-2d47-7201-9441-ab74e0523152/plot_particle_stall_diagnosis.py/.png.
Sevenaffectedlegacystrategy/normtests pass(81.93s)in
/tmp/particle_release_legacy_regression.log. Two newrunsremainactive;readjournal.


## Conditional global-time-coverage control

Prepared train_physical_single_horizon.py and two small NumPy sampling helpers.
The absent option keeps all historical sampling arithmetic/RNG unchanged.
Explicit sample.horizon_curriculum={kind:exponential,initial_fraction:.02,
ramp_steps:4000} requires originaljointstrongnonlocal fields and the existing
initial_layer rule. Datawarmup1000retainsfulltimequeries. Updates1001..5000
expand onlythe24globalqueries fromhorizon.02to1;the8earlysamples stayexact.
Updates5001..10000retainallsourcefulltimequeries. The5000thrampendpoint also
usesfulltime,so5001PDEupdates havefullcoverage. No coefficients,timeJVP,
coordinate normalization,outputmapping orknownfieldtargetsarechanged.

This is onecurrentnetwork over alltrainedtimes,not multiple timewindows or
frozenpredictedIVPs. Themeasurechanges intentionallyduringtheramp. Validate
againstthecompleteinterval;do notpresentearlydomainaccuracy asfinalsuccess.
Sevenfocused sampling/controltests pass(1.354s)in/tmp/particle_horizon_tests.log:
RNGunchanged,early/warmup/taildraws exact,independentexponentialmapping,
range/schema/no-tailrejections and savedsourceprefixreplay. Physicalconfiguration
validates withnch1.7/AC1070,176,.103. ExistingphysicaltimeJVPunchanged.

Ownerstudy/horizon_owner.py PID618216 waitsforbothcurrentcontrolsandtheir
independentnativeevaluations. Journalhorizon_status.json. It startsone10k
coldrunonlyifbothpriornativephaseerrorsremain>=10%;otherwiseitdefersforreview.
Sourceiscompleted single_joint10000-01,coldmodel/Adam/core/conditiondrawsidentical;
onlydeclaredtimecurriculummaychangeconfig. Thenindependentlyreplayactualtimes,
evaluate_single.py andsingle_time_diagnosis.py. Newrunwhenlaunched:
/root/autodl-tmp/tf-logs/particle_corrosion2d/20261106-single_horizon10000-01.
No automaticlaterexperiment,promotion,Gitoperation orcleanup.


## Task-end: user pause and requested current physical-field gallery

All4causalrelease/jointownerstagespass;pairvalidation confirmscoldmodel/Adam,
condition/core/actualtimedrawidentityandschedule. Finalsinglecausalrelease:
phase25.5116%overall/33.9550%terminal,c26.5429%. Finaljoint:
phase25.9306%/34.1529%,c26.9307%. These are one reusedtraininggeometry metrics,
not held-outoperator/generalization or local-referencegoalqualification.

The user explicitlypausedtheappgoal whileaskingforcurrentfields. The horizon
owner had just startedchild625528;SIGSTOPwas sentonlytoverifiedownedprocessgroup
618216,includingownerandchild. horizon_pause.json records this;bothstatesverifyT.
No process was killed,no new experimentwasstartedafterpause,no automaticresume.
Goalstatuswasreadbackpaused. Theoldjournalrunningstatusdoesnotmeanitistraining.

Afterpause,onlycurrentvisualizationandmemorywerehandled. ScriptoutsideGit:
/root/.codex/visualizations/2026/10/24/01a16d71-2d47-7201-9441-ab74e0523152/plot_particle_current_fields.py.
Inputs:study/single_joint_final.npz andtraining_geometry_native2/training_062.npz.
All4093pointsandfiveactualtimeidentitiesarechecked. No interpolationbetween
savedtimes,newreference solve ortrainingisused. Exactmeshtrianglesrespectholes;
Gouraudshadingonlysmoothstherender. Figuresat0,.1,.3,.6,1(labelactual0,.5,1.5,3,5s):
particle_current_phi.png,particle_current_concentration.png,particle_current_liquid.png.
Eachhasreference/prediction/absoluteerrorrows;predictionsshowdirectnativeinterface
white-dashedandpredictedinterfacedark-solid. Grayparticles/redmouthareknowninputs.

Provenanceandindependentlyrecomputedspace-onlyerrors:particle_current_fields.json
inthatvisualizationfolder. At.1/1currentphaseL2is5.9916%/34.1529%;liquid
s=cL/cLeerrors57.0214%/47.7473%. The latter is the KKSliquidvariable,not totalc.
Checkc=cle*s+dc*h(phi)passes. Bothphi/liquidimagesvisuallyinspected;noGitassets
orweightswereadded. Thisgalleryexplainsfrontstallandconcentrationmismatch;
itdoesnotestablishthatqerroralonecausesthecompletenonlineartrajectoryfailure.

Diff-level continuationchanges:thissessionaddsoptionalstableL2/every1normcontrols,
causalreleaseschedule,andoptionalglobaltimehorizoncurriculumwithfocusedtests,
configpresetsanduser/developerdocs. Numericalphysicalparametersanddefaultscheme
wereretained. Currentfocusedreceiptsare7horizon/controltests,7legacystrategy/norm,
3causalrelease/control,pluspreviousnorm/controltests. Gitdiffcheckisrequiredatend.
NoGitoperation,externaldeletion,reference-targettrainingorSOTApromotion.

Exactresume:remainpauseduntiluserrequestsoptimization. Thenverifytheownedgroup
andstoppedstateagainstpause/starttimeevidencebeforedecidingwhethertoresumeRAMstate.
Pendinghorizonownerwillfinishone10ktrainingpluspairaudit/singleeval/denseaudit;
noautomaticlaterexperiment. Readhorizon_assessment.jsononlyafteractualcompletion.
Theheld-outlocal~2%goalremainsunmet,andnative/local6.58%productionbiasstillrequires
separateeventualrefinementifneuraloptimizationrecovers.


## Task-end: user BC-normalization and sustained-stagger hypothesis

Scope: answer the new requested BC/CH diagnosis while the optimization goal
and horizon owner remain paused. No SIGCONT,new training,default change or
reference solve. A prepared config is not an executed experiment.

Current latestjoint frozenconfig has no loss.concentration_scale override.
losses.py153..161 therefore evaluates mouth mean((c/cLe)^2),withcLe=.0356643;
it is a soft Dirichlet loss. Mesh-aligned boundary sampling covers all9actual
mouthnodes with positive edge weights(endpointsincluded). Savedcurrentfields
show mouthcmax2.755e-5at0,1.324e-5at.1,1.154e-5at1. These errors must be
compared to actual chemical gradients;smallabsoluteerror alone cannot ruleout
boundarysensitivity. Gradient-balancing also means a constant rawpre-scale
is not automatically a guaranteed increase in relative boundary priority.

New reproducible CPU-only study/bc_ch_snapshot_audit.py/.json uses existing
4093nodefieldarrays and the sameproductionoperator. Exactmouthprojection sets
q_mouth=-cLe and leaves currentinteriorvalues/free time derivativesfixed.
At.1 originalCH_RMS=.3982663,projected=.3983347;boundarydefect contributionRMS
.0011203. At.03,.5161124->.5160841;at0,.2075157->.2079577. Thus presentwholegraph
CHnonconvergence is mainlyinterior in thissnapshotmeasure. This is not a
retrained/evolvedboundarysensitivityexperiment and cannot exclude cumulative
BCeffects. Originalunprojectedvalues independentlymatchtheexistingaudit.

At.1 nativeRHS concentrationrateRMS=.8819724,sothephysicalCHresidual is~45%of
that rate scale;at1,.1405849vs.3306592(~43%). This supports testing better
transportoptimization. Do not compare AC/CH rawlossvalues as gradientdominance.
Distinguish totalc,KKSliquidcL,andq=cL-cLe. Onreference.4<phi<.6front at.1,
meansarec=.51631,cL=.031760,q=-.003904. The1e-3quantitycanbethedrivingchemical
offset,notnecessarilythetotalcusedintheboundaryloss.

Latestjoint10k hasnostagger. Historicalstagger5k usedonlyfirst2000AC/CH
alternating25,then3000joint;it alsouseduniformtimes,MAEandtheolderIC/gradient
recipe. Itsnegativeoutcome doesnotruleout sustainedstagger onthelatestrecipe.
Preparedexamples/particle_corrosion2d/configs/train_physical_single_stagger.py:
unchanged1000datawarmup,then9000PDEupdatesalternating25AC/25CH. Bothsoftmouth
lossesandICalwaysactive;one sharednetwork,notfreezingvariables. No causal or
horizoncurriculum. Onlytwoconfigfieldsdifferfromcompletedjoint10k. Validation
passes,cold-controlinputreplaymatchesallactualsourcearrays,andmaskcensusgives
4500AC/4500CH/10000dataupdates. Recordstudy/single_stagger_preparation.json
explicitlystatesprepared_not_trained. READMEdocumentsitassuch.

Memoryupdates:this taskandJSON,PROGRESS,andappend-onlyDECISIONS. Diffcheck
requiredatend. Remainpausedpendinguserrequest;do notautomaticallyreplaceorresume
thealready-suspendedhorizonexperiment. Thelocalreference~2%goalremainsunmet.


## User-authorized sustained-stagger trial (in progress)

The user said to try the preparedscheme. Canonicaltrain.py now runs
configs/train_physical_single_stagger.py with --strategy-control pointingto
20261106-single_joint10000-01. Target20261107-single_stagger10000-01 under
/root/autodl-tmp/tf-logs/particle_corrosion2d. Same4093nodegeometry,32pools,
B1/core1024/IC4096/Nt32,originalMSE/fields/naturalIC,Adam/LR and32resampledtime
queries. Initialdatawarmup1000;then4500ACand4500CHupdates in25stepblocks.
BothmouthlossesandICarealwayson. Sharednetworkoutputfieldsarenotfrozen.
Onlyloss.stagger_steps9000/stagger_switch_every25changefromjointbaseline.
Previousconfig/rangemask/fullsourceinputreplaychecksareunchangedandreused.

Externalfiniteownerstudy/sustained_stagger_owner.py,PID630967;readjournal
sustained_stagger_status.json forcurrentchild. ItchecksoldhorizonpausedPID/start
identityandneverresumesit. Plannedstages:train,bytecoldmodel/Adam/config/actual
samplingandloggedmaskaudit,evaluate_single.py,single_time_diagnosis.py,and
compare_sustained_stagger.py. No evolvedreferencefieldsentertraining.

Expectedoutputs:single_sustained_stagger_final.json/.npz/.png,
single_sustained_stagger_dense.json/.npz,sustained_stagger_pair_validation.json,
sustained_stagger_assessment.json. Comparisonindependentlyrecomputesphase/cL2
fromfieldsandcomputesKKSliquidL2. FiguresoutsideGit:particle_sustained_stagger_fields.png
andparticle_sustained_stagger_dynamics.png invisualizationsfolder. Accuracy
mustincludeoriginalAC/CHandmouth/ICconstraints;lowerCHaloneisnotacceptance.
Thebaseline25.93%nativephaseisonesingletraininggeometry,notoperatorgeneralization.
Noautomaticbroadergoalresume,SOTApromotion,Gitoperation orcleanup.


## Task-end: sustained-stagger result and exact comparison

The finite10k experiment completes. Same one4093node traininggeometry and all
actualsourcecondition/core/timearrays;model_initial/optimizer_initial bytes
identical. Frozenmodel/network/sharedpointoperator/optimizer sourcebytesmatch;
losses.py numericalASTmatchesafterexcludingdocstrings. Theonlyconfigdifferences
areloss.stagger_steps9000andstagger_switch_every25. Loggedlossselectorsmatch
1000datawarmupand25AC/25CHblocks;IC/BCremainactiveall10000updates. This is
lossalternationononesharednetwork,not separately frozen physicalfieldsolves.

| Metric, direct native reference | Joint10k | Sustainedstagger10k |
|---|---:|---:|
| Phase pooled L2 |25.9306%|25.5867%|
| Phase terminal L2 |34.1529%|34.0070%|
| Total concentration pooled L2 |26.9307%|26.6002%|
| KKS liquid pooled L2 |52.4690%|38.6230%|
| KKS liquid terminal L2 |47.7473%|42.0177%|
| CH RMS at tstar.1 |.398266|.455347|
| AC RMS at tstar.1 |.111710|.559600|
| Max mouth concentration over16evaluationtimes |2.7546e-5|9.0733e-5|
| Terminal corroded volume fraction |5.2957%|5.3558%|

Nativecorrodedfractionat1is17.4809%. ThesmallphaseL2changeisnotrestoredkinetics.
Liquidfieldvalueaccuracyimproves,butCH/ACresidualsandmouthaccuracydegrade;
do notpresentthisasacorrectsolutionorpromotethedefault. Neitherthenegative
resultnoritsoneinitializationrulesoutotherstagger/fieldblockschemes. The~2%
held-outlocalreferencegoalremainsunmet. Local-same-meshmetricsaresavedseparately,
notmixedwithnativeoptimizationerrors. Existingproductionnative/localbiasstill
requiresseparateeventualoperatorrefinementifneuraloptimizationrecovers.

Owneroriginaljournalhastrain/evaluation/densepassedandcomparisonfailed. The
comparisonhadalreadywrittencorrectnumericmetrics,buta localdictnamedboundary
shadowedtheimportedgeometrycontourfunctionduringplotting. Renameonlythatdict
toboundary_metrics andrerunreportstageonce. Recoveryexit0,numericassessment
unchanged;originalfailurelog/journalretained. Extra sourceparitychecksaddedto
pairreceiptandassessmentmetadataafterward. No retrainingorhiddenfallback.
Sources:study/sustained_stagger_report_recovery.json/.log andcomparisonhelper.

Finalallgraphanalysis:single_sustained_stagger_final.json/.npz/.png anddense
single_sustained_stagger_dense.json/.npz. Numericalsummaryandliquidderivation:
sustained_stagger_assessment.json. Bothphase/cpooledmetricsindependentlyrecomputed
fromsavedfields;time/pointidentitieschecked,nospatialortemporalfieldinterpolation
usedformetrics. Liquidreferenceis(c-dc*h(phi))/cLe;itdiffersfromtotalc.
Externalfiguresvisuallyinspected:particle_sustained_stagger_fields.png
(reference/joint/stagger,0,.1,.3,.6,1)andparticle_sustained_stagger_dynamics.png
(corrodedfraction,liquidL2,originalfullgraphCH_RMS)invisualizationsdirectory.

Trainingwall987.718s(16.46min),firstupdatecompile227.633s,medianheldupdate65.247ms.
Ownactivepeak5.287GiB,pool8.998GiB;theolderpausedhorizoncontextremainsallocated.
Do notfair-speedcompareagainsttheearlierjointcontrolwhichoverlappedanotherjob.
NoFEniCS/MPI,newreferenceintegration,Gitoperationorcleanupoccurred.

CodechangesinthisexecutionturnareonlycorrectingthepresetdocstringandREADME
fromuntrainedtoexperimental;thepreparednumericalconfigwasalreadyvalidated.
Newexternalfiniteowner/comparisonhelpersarecompiled;fullreal10kandpairaudits
exerciseactualstaggerintegration. Previousfocusedstrategy/norm/samplingtests
areunchangedandreused. Run gitdiffcheckbeforefinish. Updatedmemoryfiles:
PROGRESS,DECISIONS,this taskandJSON. No newGitassetsorlargeweightscopied.

Exactresume:finiteexperimentcomplete;allnewchildrenareterminal. Remainpaused
forbroaderoptimization. Theoldhorizonprocessgroup618216isstillSIGSTOPandmust
notresumewithoutnewuserauthorizationandPID/starttimeverification. No other
experimentisqueued. Nextdecisionneedsuserdirection;do notautomaticallyextend
thisstaggertrialorchangephysicalparameters.


## Revised physical regime: reference-first parameter screen

Userexplicitlyallowslowerdifficultybutwantssubstantialcorrodedcoverageandclear
particle-inducedmorphology. Screen actualL/Tc combinations,not independentmanual
changesofcachedPDEgroups. KeepD8.5e-10,A5.35e7,omega1.76e7,alpha2.575e-5,
cLe5100/143000,cSe1,Lc50umandallgeometryparametersfixed. Thusprofilewidthand
mesh/operatorsareunchanged. Candidates:L2e-9/Tc250,350,450s;L5e-9/Tc140s;
hardcontrolL2e-6/Tc5s. ForinstanceTc250/L2e-9 givesNch85,AC53.5/8.8/.00515,
closetotheolderwell-trainedregimeinstead ofhard1.7/1070/176/.103. Thesearecase
designparameters,notclaimedcalibratedreal-materialmeasurements.

Threeknowngeometries:firstprevioussingletrainingcaseplusfirsttwooriginaldev4.
Theyareexplicitdevelopmentcasesforphysicalparameterselection,notnewreserved
populationtests. ParallelCPUworkersbuildthe4093-ishnodemesh/visibleintegral
operatoroncepergeometryandindependentlyBDFsolveeachcandidate. Save21snapshots,
physicalseconds,groups,area/netgrowth,particlewettedperimeters andsolverstats.
Noreferencetrajectoryisreadbytraining. MatrixdoesnotdependonL/Tc;allchanging
physicsisinexplicitderivedcoefficients. Oldhardcontrolincludedforcontext.

Study20261107-moderate-regime-01/screen_parameters.py,PID651430,threeworkers.
Readscreen.log,screen_setup.json,andpercasescreen_case_*.json,thenaggregate
screen_results.json. Do notselectonlyaprettypitwithunchangedarea:inspectnet
corrosiongrowthandnumberofparticlescontacted. Ifacceptable,prepareexplicit
newpreset/newmatchingbank/referencecohort andstartcoldjointPINO. KeepSIfirst
nondimensionalizationandavoidmoreoutputprior/staggercomplexity.


## Selected moderate SI regime and full PINO cold training

Reference screen complete:mean terminalcorrodedfractionmild18.50%,moderate24.35%,
longer33.56%,transport23.41%,hard17.97%. Moderatecontacts5/5/4particles;longer
interacts7/5/7butisneedlesslyharderasafirstrecovery. SelectL2e-9,Tc350s:
Nch119,AC1 74.9,AC2 12.32,AC3 .00721. SI reactionsare1000timesslowerwhile
physicaldurationis70timeslongerthanoldhardcase;allnormalizedACgroupsfallby
14.286anddiffusionrises70fold. Thischangesphysicalregime,notmerelyunits.
Itsreaction/transportbalanceisclosetotheoldsuccessfullytrainedactivationregime.
A,D,omega,alpha,cSe,cLe,Lc,pit,particlesandinterfacek29.22959arestillfixed.

Explicitpresets:train_physical_moderate.py/reference_physical_moderate.py.
Historicaltrain_physical.py anddefaultnotyetchanged. Canonicalprepare.py
rebuilds64knowninputsetsusingoldinputseed1307500965:41.03s. Allactualknown
arraysandallgeometriesmatcholdSIbankexactly;newmetadatahascorrectL/Tc/groups.
IndependentSIformulas,validate(),validate_dimensional()andload_bank()pass.
Inputanddimensionalreceiptinnewstudy/input_and_dimensional_validation.json.
NoPDEtrainingtargetsaregenerated.

FournewlocalFEMdevrefs:h.01/boundary256;fourproductionnativerefs:h.015/128;
firstlocalrefineh.0075/512plusseparatetimecheck(maxdt.003125,tightertolerance).
All10solvescompletewithvalidphaserange. PhysicaltriangulationP1interpolation
respectsallholes;alltargetpointsinsideactualrefineddomain,andcoincidentnodes
arematchedexactly(no extrapolation). Localphase spatialdifference.078978%,
temporaldifference1.93e-7%;c.108308%/2.31e-7%. Native/localphasebias2.043/2.130/
1.972/1.961%on4devcases. Receiptreference_validation.json;newreferencesarenot
old-fieldrelabeling. Thesearedevelopmentchecks,notreservedpopulationaccuracy.

Newrun/root/autodl-tmp/tf-logs/particle_corrosion2d/20261107-moderate-w192-cold20000-01.
PurephysicsoriginaljointMAE,softIC/BC(cLe-normalized),originaltanh/KKSoutputs,
Adam1e-3withoriginaldecay,width192,Nt32uniformrandomdrawsperupdate,B4/core256/
IC1024. No stage,causal,stagger,horizoncurriculumornewoutputprior. Geometryand
core/halopoolsrollwiththeexistingvalidatedcontroller. Freshtrainingseedisruntime
provenance,notatunedconfigurationfield. Thisisnewproblemtraining,nota paired
attributiontochangedphysicalparametersacrossdifferentinitializations.

Finiteownernewstudy/train_owner.py PID653217;readtraining_status.jsonforchild.
Stages:train20000,completeonline_validationwith4CPUworkers,thenfrozennetwork
complete-nativeevaluation(physicalJVPandoriginalAC/CHon4productiongraphs).
Nativeevalpadsindependentqueries4608withnoqueryselfattention;cutactualNafter
inference. Savesfields/raw/dt plusphase/c/liquidmetricsandmouth/earlydynamics.
Localdevelopmentmetricsarethecanonicaltrain.pyevaluationagainstindependentFEM.
Noautomaticdefaultpromotionorlatertraining;assessafteractualcompletion.

ReferencepreviewfigureandregenerationcodeoutsideGit:
/root/.codex/visualizations/2026/10/24/01a16d71-2d47-7201-9441-ab74e0523152/plot_particle_moderate_reference.py/.png.
Itshowsdirectnativefields,notPINO,at0,.25,.5,.75,1(0..350s)on3geometries.
Visuallyinspectedandshared. Oldhigh-Ltimecurriculumgroup618216stillpaused.


## Six farther particles: user steering and controlled new prior

Theuserexplicitlyasksfewerparticlesandgreaterinitialseparation. Keepmoderate
SIcase,butsetcount6,pitclearance.075(3.75um),pairclearance.055(2.75um),
near-pitdistance.16andminimumnearcount2. Radii/modes/initialpit unchanged.
Thefourselectednewdevgeometriesarefreshsamples,new64trainingbankisrebuilt
in18.68s. No old8-particleinputisrelabelled. Presets:
train_physical_spaced.py/reference_physical_spaced.py. Allnon-particleconfig
fieldsareexactlyunchangedfrommoderateeightcase.

Previousmoderateeightcold20kandall3ownerstagescomplete:localphase34.1193%
overall/42.5100%terminal;native33.4308%. Restoreddimensionalreaction/transport
ratioaloneisthereforenotdemonstratedasufficientremedy. ExistingoriginalSOTA
hasold8-times/midpointmouth anddifferentseed,amongthenowdocumenteddifferences;
no hiddenmathematicalbugwasfoundbyreadingsourcediffs. Numericalmodifications
areexplicitcontrols,notsilentlyreplacingtheoldsuccessfulresults.

Analyticinitialwallnormalauditon64layoutsforeachprior:oldminholegap.0306222,
minimumwallphi.856946;11/64havesomewallinthe.1..9phaseband,and40/64within
thatbandplusdelta. Newminholegap.0756957,minwallphi.988169;bothoverlapcounts0.
Solidside90%distance=.0375856,plusdelta=.0635856. Medianmax-per-particle
normalphiRMS.622365->.0607337;normalq/cLeRMS.112171->.00090677(~124fold).
Theseareinitialcompatibilitymeasures,notacleancausalityproof;normalderivatives
are reduced,notexactlyzero. Script/receiptinitial_geometry_audit.py/.json.

Newlocal4devreferencesandproductionnative4arecomplete,plusfirstspace/time
refinement. Localspatialphase.1006556%,temporal1.03e-7%;productionnative/local
phase1.735/1.719/1.782/1.797%. Allcomparisonpointsinsideactualhole-respecting
physicaltriangulations. Newinputload/range/dimensional/only-particle-change
validationpasses. Noreferencetrajectoryentersupdates.

Native4devmeanterminalcorrodedfraction29.4952%;individual28.72/31.61/29.39/
28.26%. Interacts4/4/5/3particles. Firstsnapshotwith>10%wettedperimeteronany
particleis.6/.6/.5/.4;notexactcontacteventtimes. Thus simplificationretains
substantiallateparticleeffects. Referencefig/scriptoutsideGit:
plot_particle_spaced_reference.py/.png invisualizationsfolder,first3devsamples
at0,.2,.5,.8,1(0..350s),visuallyinspectedandshown. reference_morphology.json
recordsall4andphysicalnetgrowth315.6..359.6um2.

Addedoptionaltrain.py --seed solelyforruntimecontrolledreplay. Defaultstill
freshsecrets32;seedisabsentfromstrategyconfigs. Explicitseedrejectsinvalid
integer/rangeandcannotoverrideanyoldcomparisoncontrol. runtime.seed_source
makesoriginexplicit. Threefocusedtests(pass.162s)verifyfresh/defaultbehavior,
explicitNumPyintegers/invalidtypesandcontrolconflictrejectionbeforeGPUsetup.
Log:/tmp/particle_runtime_seed_tests.log. README/DEVELOPMENTdocumentthatmatching
slot/pool/timeindicesacrosspriorsdoesnotmeanidenticalphysicalspatialqueries.

Newrun/root/autodl-tmp/tf-logs/particle_corrosion2d/20261107-spaced-six-w192-cold20000-01.
RuntimeCLIreplayseed17617494comesfrommoderateeightcontrol,notaseedsearch.
Beforeupdates,ownerverifiesexactmodel_initial.eqx/optimizer_initial.eqxbytes
andallcondition_ids/pool_ids/timesarrays;initialization_pair.jsonpasses.
Actualgeometricpayloadsareintentionallydifferent. Themodelarchitectureis
unchanged;encoderboundarypointcountchangeswithparticlecount,paddedshapesfixed
withinrun. OriginaljointMAE/Adam/width192/B4/core256/IC1024/Nt32/rollingprior.

Ownernewstudy/train_owner.py PID670127,child670128. After20k:repeatcold/draw
checks,permitonlyfiveparticlepriorconfigchanges,checkonlinelogicaldraw/revision
scheduleagainstsource,runcompleteonline_validation,thenoriginalnativefullgraph
PDE/field/liquidevaluation. Noothertrialautomaticallystarts. Parentoldmoderate
ownerhadcompletedbeforethisnewGPUjob. Oldhigh-Lhorizongroupremainspaused.
No defaultpromotion,Gitoperation,cleanupornewoutputansatz.


## Six-particle endpoint and task-end (finite trial complete)

All three owner stages exit 0: cold 20k, complete online replay, frozen native
field/time-JVP/PDE evaluation. Owner and children are terminal; no subsequent
job is queued. Old horizon group 618216 remains paused, with no resume.

| Reference, four development geometries | Phase overall | Phase terminal | Concentration overall | Terminal corrosion IoU |
|---|---:|---:|---:|---:|
| Independent local P1 FEM |3.0914%|5.9848%|3.7569%|94.7692%|
| Production direct nonlocal |1.6369%|3.1099%|1.9854%|97.9535%|

Native liquid-coordinate error is 5.5982% overall. Local minority-corroded
phase error is 7.3583% overall/9.2336% terminal; native 3.9566%/4.9078%.
Do not equate the smaller total-phi norm with all interface/transport metrics.
The native/local errors cannot be subtracted as scalar error contributions.

Actual cold model/full Adam bytes, all condition/pool/time arrays, logical
online geometry/revision traces and only-five-prior-field config checks pass.
Physical spatial queries intentionally change. Complete replay audits all
20k draws, 380 sampled geometry identities, 1900 sampled spatial versions,
316 installed new geometries, 1520 spatial refreshes and 64 final slots. No
reference overlap, evolved labels or training PDE solves. Invalid-candidate
regeneration retains its existing behavior.

All four local and four native physical prediction files independently
reproduce their reported phase/concentration overall/terminal metrics to
1e-12 absolute tolerance. Saved early probes do not show a decline below the
network's own initial corroded fraction; this is not a proof between probes.
Final PINO corroded fractions are 28.23/30.99/28.72/27.62%, compared with native
28.72/31.61/29.39/28.26%. Late growth is slightly underestimated, while visibly
retaining geometry-dependent interfaces. Three-layout/five-time native-overlay,
convergence and four-layout area-growth figures are visually inspected.
Regeneration: plot_particle_spaced_results.py in the visualization directory;
no generated figures/weights/raw fields copied into Git.

The matched eight-particle moderate control remains 34.1193% local phase and
33.4308% native phase. The six/farther prior is markedly easier for this one
initialization, consistent with reduced initial wall incompatibility. Count,
spacing and development geometries all change, so this does not isolate a
single cause or establish population/cross-seed robustness. It does not solve
the old high-mobility case. No new mapping, hard constraint, stagger, causal
schedule or optimizer change is required for this recovery.

Diff-level changes for this finite geometry trial: four explicit moderate/
spaced train/reference presets; optional validated runtime --seed replay,
three focused tests (pass, reused unchanged); README/DEVELOPMENT setup and
provenance guidance; current task MD/JSON, PROGRESS and DECISIONS. Earlier
uncommitted diagnostics are preserved. Input/reference dimensional, space/time,
geometry and all-run audits supplement the focused tests. No Git operation or
cleanup. Default remains separate; near-2% local aspiration is not achieved.
The broader task stays open with this completed endpoint as the exact resume
point. No automatic follow-up trial is authorized or scheduled.


## User-requested width256 capacity control (running)

Increase backbone192→256 and spatial Fourier directions96→128, keeping four
heads (48→64 features/head), conditioning depth2, decoder depth3, Gaussian
spatial scale2 and temporal bandwidth unchanged. Parameter count is
1,268,546→2,248,450 (actual CPU model construction). All other configuration
fields equal the completed six-particle width192 control. No evolved targets
or change to the equations, output map, loss/optimizer or sampling measure.

Preset train_physical_spaced_w256.py is imported by canonical train.py. Reuse
width192 inputs64/local_dev4/native_dev4 paths; do not duplicate or regenerate
qualified references. Runtime seed17617494 replays the same actual input/time
stream, not identical weights: changed shapes require fresh model/full Adam,
and the Fourier realization also changes. This is one controlled stream, not
a cross-seed stability study. --seed remains absent from strategy configs.

Study: /root/autodl-tmp/data/particle_corrosion2d/20261107-spaced-six-width256-01.
Run: /root/autodl-tmp/tf-logs/particle_corrosion2d/20261107-spaced-six-w256-cold20000-01.
Attempt01 owner689100/child689101; it later fails before the first update.
The separate successful attempt02 uses owner699494/child699495; both are now
terminal. Their own training_status.json/capacity_train.log remain authoritative.
First audit checks only two arch changes, exact initial known inputs and all
actual condition/pool/time draws, and identical numerical sources. Final audit
additionally checks every online identity/revision trace and complete final
input arrays; canonical complete online audit and frozen native evaluator follow.
No subsequent experiment is queued. Configuration and external owner/evaluator
syntax checks pass. First25-step production peak memory is pending; do not
silently reduce batch/core/time counts on failure. About27GiB free disk at start;
no cleanup is needed or authorized. Old horizon group618216 remains paused.


Width256 attempt01 fails before its first optimizer update, requesting one
20.52GiB buffer. It is terminal and preserved, not retried in-place. Explicit
attempt02 uses the same strategy and seed with preallocation=true,fraction0.90.
Read resource_retry.json and training_status.json under
/root/autodl-tmp/data/particle_corrosion2d/20261107-spaced-six-width256-02.
The initial memory hypothesis is allocation fragmentation/limit; do not claim
this proven until production updates run. No batch/core/time/physics change.


Attempt02 production first25 updates pass: active peak20.6774GiB, versus
width19215.6606GiB. Explicit90% pool reserves about29.5GiB on the card; distinguish
reservation from active peak. Held updates initially63–65ms versus baseline48ms.
Same config/model_initial/optimizer_initial bytes as failed01 are independently
verified (resource_retry_state_validation.json). This establishes numerical-state
parity for the resource retry; it does not isolate fraction versus preallocation
as the sole cause of the first allocation failure. New-vs-width192 known inputs,
all condition/pool/time arrays and shared source checks also pass. Estimated
full20k duration28–30min from early held timing; final timing remains pending.


## Capacity endpoint and task-end (requested trial complete)

Width256/spatial_features128 attempt02 finishes cold20k; train, complete online
replay and original full-native physical-field/time-JVP/PDE evaluation all exit0.
Exact config difference is only width192→256 and Fourier directions96→128.
Condition/core/time counts, SI physics, LR/Adam, loss normalization and network
depths stay unchanged. Fresh parameter shapes mean initial weights differ from
width192; the same runtime stream seed is solely an input replay device.

| Metric, means of the same four development geometries | Width192 | Width256 |
|---|---:|---:|
| Local phase overall / terminal |3.0914% /5.9848%|2.9330% /5.7567%|
| Native phase overall / terminal |1.6369% /3.1099%|1.4504% /2.7812%|
| Local concentration overall / terminal |3.7569% /7.1766%|3.5663% /6.8709%|
| Native concentration overall / terminal |1.9854% /3.6998%|1.7630% /3.2778%|
| Local terminal corrosion IoU |94.7692%|94.9929%|
| Native terminal corrosion IoU |97.9535%|98.1049%|
| Parameters |1,268,546|2,248,450|
| Canonical complete-run time |1400.78s|1708.08s|
| Held-update median, complete timing trace |49.061ms|62.941ms|
| Final active allocator peak |15.6606GiB|20.6900GiB|

Local overall/terminal phase error decreases5.12%/3.81% relatively; native
11.40%/10.57%. All four geometries improve in all four local phase/concentration
norms. Native liquid-coordinate error is5.4926%, versus5.5982% baseline, so
not all physical quantities benefit equally. Width256's lowest recorded local
overall error is2.8050% at15k (terminal5.3456%), with a real saved checkpoint;
width192's minimum is3.0303% at15k. These are development-monitor minima, not
independent generalization measurements or substitutes for the fixed20k table.
No extra15k native evaluation was done.

Final capacity_pair_validation.json passes exact condition/pool/time arrays,
initial known inputs, all rolling identity/revision traces, final input arrays
and unchanged numerical source. Canonical online audit passes all20k actual
updates,380sampled geometries,1900spatialversions,316new geometries,1520spatial
refreshes,64rebuiltfinalslots,zeroevaluationoverlap/evolvedtrainingfields.
Independent all-four-local/all-four-native physical-field metrics reproduce
the reported norms with maximum1.74e-18 difference. Qualifying local refinement
and native-bias receipts are reused unchanged; no new reference solve or broad
suite rerun. Saved early area probes increase from their own initial values;
no claim of continuous monotonicity between probes. Three-geometry evolution,
convergence and area-growth plots are inspected outside Git. Regeneration and
comparison receipts are in plot_particle_spaced_capacity_results.py and the
study's capacity_comparison.json/saved_field_metric_validation.json.

Resource attempt01 failed before updating while requesting20.52GiB; all files
and traceback remain. Attempt02 explicitly preallocates90%; its config/model/
Adam initial bytes match01. No batch/core/time or numerical change was made.
Canonical28.47min excludes the failed attempt's283.62s; the failed attempt is
not hidden in the accuracy/timing comparison. Total nvidia-smi use~29.5GiB
includes allocator reservation/overhead/old paused context, not active arrays.
Only first-resource-policy feasibility has been demonstrated, not a unique
attribution to fragmentation versus allocation limits.

Interpretation: modest capacity gain, about22% longer completed training and
32% larger active peak. The overall local error still lies near3%; this trial
does not justify a claim that widening alone reaches2% or that either model
is consistently superior across random initializations. Keep both explicit
presets and checkpoints; no default or historical SOTA promotion.

Diff-level work this turn: one explicit train_physical_spaced_w256.py preset;
README capacity/results/resource guidance; external finite owner/evaluator,
strict pairing and figure/metric/comparison scripts. PDE/train/network/shared
numerical source code is unchanged. Memory changes: PROGRESS, DECISIONS and
this task MD/JSON. Source/config/input/field verification, final diff whitespace
and documentation review pass. No Git operation, cleanup,
new automatic experiment or resume of old paused horizon group618216.


## Accepted README publication and task-end

Reorganized the example README around the accepted six-particle SI problem,
with consistent physical/dimensionless coefficients, original residuals and
softIC/BC, generic two-stream point conditioning, timeJVP, Adam/LR and rolling
geometry/core-halo settings. Commands explicitly select train_physical_spaced_w256;
train_default itself is not changed. Root overview and assets guidance match.
Historical dimensionless/high-mobility failures and controls are linked to
archived records rather than mixed into the current setup/results.

The centralized PNG now shows first3of4 development layouts at0,.2,.5,.8,1
(0,70,175,280,350s), model_final at20k, direct-native contours. The separate local
FEM metrics are reproduced from all4 saved local predictions. The CPU renderer
validates config/cohort/geometry/order/time/quadrature and all phase/concentration
norms before plotting. No inference, training, new reference solve or new labels.
The setup SVG uses exactly the same three prescribed six-particle geometries.
Only the particle PNG/SVG selections change; other five cases remain unchanged.

PNG210534bytes,SVG31753bytes; combined242287bytes (~237KiB). All6 PNGs total
2262158bytes (<2.5MiB); all6 SVGs total285557bytes (<320KiB). Save rawfields,
weights and complete galleries externally. Updated assets/pino... provenance
and artifacts/index.yaml name the explicit recipe and direct reference paths,
with no hashes and no inaccurate default/SOTA promotion.

Focused render commands completed; final AST/JSON/XML/link/source/budget and
other-case byte-parity checks are recorded in the adjacent JSON. No code tests
or numerical-core suites are required for this documentation/render-only change.
Changed memory files: PROGRESS,DECISIONS,this task MD/JSON (now archived).
All earlier uncommitted numerical research changes are retained. No Git action,
cleanup or resume of the old paused horizon job. Accepted finite task complete;
new investigation requires a new user request.

Final verification:146local links,79unchanged frozen numerical-source files,
all4local+4native saved-field metrics,18actual ceramic SVG paths and all image
budgets pass. Other10case images/metadata are unchanged. Both selected CPU
renderers execute successfully; JSON/AST/XML and git diff --check pass.
