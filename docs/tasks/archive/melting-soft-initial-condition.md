# Melting soft-initial-condition ablation

## Goal
Test whether the hard initial-field lift makes the 3D melting comparisons too
easy. Replace the exact initial output constraint by an explicitly weighted
physical phase IC loss, preserving tanh and the preconditioned PDE. Compare
Nonlocal/FiLM, Nonlocal/joint query, Local AD/FiLM and FEM PINO/FiLM from cold
initialization with the preceding 10k architecture/physics/sampling settings.

## Acceptance
- Explicit opt-in config; existing hard-IC defaults/checkpoints stay valid.
- No sample-specific initial latent field or t=0 vanishing gate in soft outputs;
  initial phase still legitimately enters the condition encoder and IC target.
- Initial targets are prescribed data, never evolved reference labels.
- Validate soft/hard behavior, local derivative formulas, gradients, masks and
  initialization/draw matching before full GPU work.
- Four cold 10k runs, 64 test samples at each evaluation; separate initial and
  positive-time/full-time errors. Keep native64 and accepted single local CG2
  case005 diagnostics distinct. Record complete cold parent time and exact
  cumulative observation timing for this new study.
- Report accuracy and full-time convergence against historical hard controls;
  inspect failure modes if soft IC does not converge. No default promotion.

## Context
GPU is the same RTX PRO 6000 Blackwell Server UUID
GPU-85fe6412-636e-5d3c-d5f0-d1e9523c3faa (97887 MiB, idle at start).
Prior method/time controls are complete and published; use their saved configs,
final weights and 64-case reference bank. Spinodal investigation stays deferred.
User has not authorized new Git/history operations in this turn.

## State
Stopped by explicit user request on2026-11-15 09:03:58UTC. The user abandoned
the soft-IC study and retained the previous hard-IC default. Nonlocal andLocalAD
completed10k; FEM was interrupted; joint query/final audit did not start.
The visualization goal was cleared. No further runs or rendering are scheduled.

## Blocker
None; cancelled by user, not an unsuccessful attempt to finish the full suite.

## Next action
None for this study. The user switched to a Ceramic Time-FiLM/joint-query study.

## Implementation and checks
The explicit configs are `train_operator_soft_initial{,_ad,_fem,_joint}.py`.
`operator.initial_condition={kind:soft_phase,norm:mse,sampling:pde_core,weight:1}`
selects direct raw latent output. Training/prediction share the same selector;
AD physical PDE/normal-flux derivatives exclude all known initial jets. The IC
is uniform over256core nodes, freshly paired with each condition draw, not halo
nodes. Residual preconditioning, PDE L1, Adam andLR remain unchanged. Initial
raw output head remains zero, exactly the preceding parameter initialization;
thus soft initial physical predictions start at0, unlike the hard-lift controls.

Twenty relevant unittest cases were exercised. Eighteen initially passed;
the new analytic boundary test needed explicitFP64 inputs for its2e-13assertion
(the production input isFP32). A preexisting FEM identity test compared the64
comparison cohort to the8case default; corrected its baseline to the matching
64case nonlocal control. All eight affected soft/FEM tests then passed, the
other12passed unchanged. New tests independently compare physical PDE/BC,
parameter gradients, masked initial loss, target-independent soft outputs and
inference. Changed modules compile and `git diff --check` passes. Pytest is not
installed; used the repository's unittest runner.

## Running owner (2026-11-15)
- PID702687, detached finite queue; no automatic retry or default promotion.
- Bundle `/root/autodl-tmp/tf-logs/melting-soft-ic-20261115-01`.
- Owner log `/root/autodl-tmp/tf-logs/melting-soft-ic-20261115-01-owner.log`.
- Four sequential real-size probes record initial PDE/IC gradient norms and
  12updates; active-memory ceiling70GiB. All must pass before four10k trainers.
- Runs `/root/autodl-tmp/tf-logs/icemelting3d/melting-soft-ic-20261115-01-`
  suffixed `nonlocal`, `local_ad`, `fem`, `joint_query`.
- Each evaluation directly records `cold_wall_seconds` relative to the parent
  process's monotonic launch origin. Final parent duration includes process exit.
- `docs/tasks/melting_soft_initial_audit.py` checks original hard-vs-soft configs,
  byte-equal cold model/optimizer per architecture, all actual sampling arrays,
  independent uniform-core replay, source bytes,41complete64case observations,
  and final metrics recomputed from fields. Joint andFiLM share matching backbone
  leaves but have different temporal parameterization as before.

## Exact resume point
Inspect journal after probe completion, then on stage completion or no more than
approximately15min during training. Do not edit running numerical sources.
After terminal audit, inspect `comparison.json`, losses and separate initial/
positive-time errors; report soft/hard method differences with the native64/
localCG2single-case qualification. Render curves if useful. No Git authorization.

## Preflight evidence and automatic result rendering
Nonlocal12-update probe passed: initial PDE L1=7.856742, IC MSE=0.930597;
parameter gradient L2 norms3.045706 and2.035564 (ratio1.49625), active peak
1.60063GiB. Local AD probe passed: matching gradient ratio1.49624, initial
Neumann penalty0, active peak1.70602GiB. All terms/gradients finite. The explicit
weight1 remains a controlled starting point, not a claimed optimum. FEM/joint
probes then run serially before full training.

CPU figure follow-up PID721595 waits on the owner's Linux pidfd, without
training/GPU polling; after successful final audit it renders actual41-record,
64-case initial/positive-time/full-space-time mean curves with the established
log-LOWESS and whole-condition bootstrap95%interval (2000resamples, seed20261115,
span0.22). Direct parent elapsed times include final process completion. Output
will be `figures/icemelting3d_soft_initial_convergence.{png,pdf}` in the bundle,
with direct-path provenance and raw plotted values. No publication/default
replacement is authorized.

Interpretation: this isolates an optimization/output-coordinate advantage; it
does not make the underlying PDE or initial-shape prior more physically complex.
If error is dominated by initial fitting, do not label that a spatial-operator
accuracy failure. Only prescribed initial targets enter training.

## Session end / exact state at2026-11-15 08:41UTC
All four preflights passed: Nonlocal116.74s/1.601GiB, LocalAD226.38s/1.706GiB,
FEM114.29s/1.453GiB, joint query177.50s/9.371GiB. Joint initial PDE/IC
gradient ratio0.23538; no numerical/gradient/memory failure. These qualification
costs are separate from cold training costs. Nonlocal full10k is now running;
remaining three trainers and the independent audit run automatically in order.
Figure worker721595 waits without polling and renders only after success.

Old hard-default checkpoint inference compatibility passed on128distributed
case005nodes at all5times, CPU versus stored GPU predictions: max absolute
phase difference3.8743e-6. Receipt:`hard_prediction_compatibility.json` in bundle.

Diff-level changes: shared opt-in raw-latent output; masked physical IC MSE;
ADsoft derivative/BC path without initial jets; soft-aware predictor; initial/
positive-time metrics and parent cold-time logging; four configs; independent
soft-IC tests and corrected old64caseFEM test baseline; finite probe/train/audit/
figure tools; experimental README section. No changed production defaults,
raw artifact additions to Git, Git operations or website publication.
Memory changed: docs/PROGRESS.md, docs/DECISIONS.md, this task record.
Next: wait for stage completion or user status request (roughly15min monitoring
interval), inspect journal/comparison/figure_followup. Do not retrain or change
weights while awaiting the preset budget. At completion report native64and
separate localCG2diagnostic, soft/hard initial and positive-time errors, and
complete cold elapsed times; visually inspect the generated PDF/PNG.

## Goal monitoring session (2026-11-15)
User activated the persistent goal to monitor this suite through completion and
finish visualization. Owner702687is verified live. Original figure observer
721595terminated: this Python has noos.pidfd_open. Retain its failed JSON/log.
Replacement observer738336uses an explicit60second process-only /proc identity
check, preserving start ticks and recognizing missing/zombie owners. Its record
is `figure_followup-v2.json`; no trainer was restarted or changed. Live/exit
identity checks passed.

First completed stage: Nonlocal10k, parent404.035799894s, final64native phase
mean0.1745852343; IC0.06086552913; positive-time0.1918982903; terminal0.02685105543.
PDE L1at final sampled batch0.09998366, IC MSE0.003201247. Active peak1.584GiB.
AD child738464then starts. Final cross-method audit still pending. Do not
interpret the single early result as a method ranking.

Additional post-completion visualization prepared: four hard/soft final64
violin panels and fixedcase005XYslice evolution, including reference, historical
hard-ICNonlocal, and four new soft methods. Physical panels will use the original
24cubeddiagnostic queries, equal aspect, coolwarm and dashed native-reference
phi=0contours; labels report original3Dpoint norms at each time. Figures stay
external and carry editable PDF/PNG plus direct provenance. A separateCPU
initial-latent diagnostic will quantify saturation in prescribed bulk ice
(referenceinitialphi>0.999), using direct decoder values rather than inverting
rounded tanh. It runs only after all training to avoid disturbing timing.

Visualization scope also includes a four-panel hard/soft full-space-time error
convergence comparison versus actual steps (same10kbudget); historical wall
clocks for all hard methods are not fabricated. The separate three-panel soft
method comparison retains directly observed cold wall time. All curves use
complete64case trajectories and final distribution uses only final10kweights.

## Goal-turn end at2026-11-15 08:58UTC
Progress: repaired the failed figure observer (not the trainer); independently
verified PID/start/exit handling; prepared final4panelhard/soft violins and
matched-stepconvergence, fixedcase005six-row physical slices, and a separate
CPU initial-latent/homogeneous-sensitivity diagnostic. Nonlocal andLocalAD10k
completed successfully. AD final means:full0.1671080775,initial0.05796668472,
positive0.1837409486,terminal0.02685098893; physicalNeumannMSE9.78636e-7,
PDEL1.11381239,ICMSE.00371665. Parent584.63400962s; activepeak1.70651GiB.
FEM livePID748199; owner702687live; replacement plot observer738336live.
Goal remains active: two trainers, complete audit and rendered visual review
are still required. No blocking condition, no training restart or Git changes.

After queue success:
```bash
PYTHONPATH=/tmp/pino-pdf-audit OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu .venv/bin/python -m docs.tasks.melting_soft_initial_diagnostics --bundle /root/autodl-tmp/tf-logs/melting-soft-ic-20261115-01 --output /root/autodl-tmp/tf-logs/melting-soft-ic-20261115-01/diagnostic-figures-v1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu .venv/bin/python -m docs.tasks.melting_soft_initial_latent --bundle /root/autodl-tmp/tf-logs/melting-soft-ic-20261115-01
```
Use a fresh output directory if a failed figure render already created one;
retain failure evidence. The original auto figure is in bundle/figures and its
observer record is figure_followup-v2.json. Inspect actual images before final
report. Archive the task only after all required deliverables pass inspection;
then fix its README link and update PROGRESS. Goal completion requires
update_goal(complete); no Git operation is authorized.
Memory changed this goal turn:PROGRESS,DECISIONS,this task. Diff checks pass.

## Final user-directed closure
Verified owner702687and its FEM child748199belonged to the dedicated experiment
process group; stopped that group and observer738336withSIGTERM. All three
process handles are absent or terminal; GPU compute-process query is empty.
Original running journals were preserved as `*-before-user-stop.json`; current
journal/observer status iscancelled_by_user. Receipt:`user_stop.json` in bundle.
No checkpoint/log/reference was deleted. The incomplete FEM is not a final10k
result. No partial plots or full-suite audit are claimed.

Actual default validation: train.py default train_base.py uses the exact
InitialPhase latent lift (phase.formulation=ic_latent); PINO train_operator.py
uses the historical hard latent lift, Adam10000, rolling_pool, with no soft
initial_condition selector. Validation passed. Existing SOTA remains selected.
The opt-in soft research configs remain for reproducibility, inactive.

The previous goal is cleared. Archive this task as stopped, not as a completed
four-arm experiment; explicitly switch PROGRESS to the user's new Ceramic
Time-FiLM versus joint-query goal. No Git/history operation authorized.

## Closure verification (2026-11-15)
Rechecked the user's instruction to abandon soft IC and retain the hard-IC
default. The stop receipt lists no remaining live owned processes; a fresh
process/GPU inspection also finds no Melting trainer or observer. The selected
PINO config remains Adam10000 with rolling core/halo sampling, and the output
selector retains the exact initial lift unless a soft research config is
explicitly supplied. README already records the incomplete experiment and
retained hard default. No numerical edits, new training or Git operation were
needed. `git diff --check` passes. Earlier running/resume instructions above
are historical only; this study has no pending action.
