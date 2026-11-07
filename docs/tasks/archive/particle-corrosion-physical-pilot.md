# Dimensional-first particle-corrosion case

Date: 2026-11-06. Status: complete; cold training did not converge accurately.

Completed follow-up: the user asks why changing time/material scaling breaks
convergence, and whether joint/stagger, AC/CH prescale, causal and other training
choices match electropolish. Source/saved-config comparison is complete below.
No new numerical settings, cold runs, Git actions or cleanup were performed.
The physical20k and all five final stages have now completed successfully as
executions, but the neural solution fails to converge: reused8phi35.4226%/
46.8794%,c34.7789%/45.6977%; native same-production first-casephi33.6062%/
46.2413%. Full-mesh diagnostics and completed report are available in the study.

## User authorization and goal

The user requests changing this case to the electropolish dimensional-first
scheme, keeping physical constants as close as possible, permitting a modest
mobility increase, adapting characteristic length/time and geometry, and
rerunning the case. This explicitly follows the completed dimensional audit.
No commit/push or deletion requested. Preserve earlier dimensionless runs and
label their results as a different physical regime.

## Initial candidate and reasoning

Use D8.5e-10m²/s,A5.35e7J/m³,alpha2.575e-5J/m,omega1.76e7J/m³,
cSe1,cLe5100/143000 from electropolish. Actual uniform matrix mobility
L2e-6m³/(J*s), twice electropolish; do not multiply it again by historicalkappa2.
Take Lc50um and domain50x30um,pit center25um/radius6um,mouth half-width2.75um.
Derived dimensionless initial coefficient is29.2295900, near the old30, keeping
the existing h.015/horizon.026 resolution relevant. Geometric particle prior
may retain its normalized shape distribution with explicitly documented
physical radii2.25..3.75um. Probe physical horizons before choosing Tc; tstar
always spans0..1. Candidate horizon1s, with an initial reference inspection up
to2s if needed. Do not assume larger L guarantees the desired corrosion extent.

## Implementation and validation plan

1. Introduce a dimensional physical preset/factory deriving all four PDE groups
   and IC width together, with explicit validation against stale/edited derived
   coefficients. Preserve loading of legacy dimensionless checkpoints/configs.
2. Match PIT/BC/geometry/initial fields between PINO and independent localFEM.
   Reuse known geometry identities where valid but regenerate physical inputs
   and references. Inspect representative evolution and qualify space/time
   accuracy at selected physical horizon before training.
3. Cold train canonical train.py with width192,B4,Nt32,32fresh random times
   each update, original losses, node-aligned mouth BC and rolling geometry/
   core-halo. Check memory if tensor shapes change; old Nt32peak15.68GiB fits5090.
   Physics/default definition may change under this explicit request; never
   relabel an old dimensionless accuracy figure as the new problem's result.
4. Complete online audit, reused8final evaluation, fields/convergence and early
   residual checks. Report accuracy honestly and preserve negative controls.

## Implementation and launch history (superseded by final outcome below)

Implemented dimensional.py,train_physical/reference_physical presets and strict
validation in problem/solver/bank/reference loaders; SI reference files store
seconds as well as normalized time. train_default/preview select the SI preset.
Original train_online and old presets remain unchanged; archived dimensional
audit explicitly imports train_online. README/root overview separate new
physics from historical1.98%phase figures. No loss/optimizer/architecture change.

Selected Tc5s after one-geometry local screen: corrosion fraction4.8043%initial,
12.3933%at2s,21.4433%at5s; interface visibly reaches several ceramics. Final
groupsNch1.7,N1=1070,N2=176,N3=.103,interfacek29.2295900,kappa1. Four initial
dimensional tests and19existing reference/input/online tests passed; added a
fifth independent SI-versus-production-PINO residual test pending execution.

External study root:
`/root/autodl-tmp/data/particle_corrosion2d/20261106-physical-pilot-01`.
Contains screen.py,screening_reference(2s),screening_5s,prepare_study.py.
Finite CPU preparation started via unified session14470/PID243263. It builds
dev4,reused8,first-case spatial_h0075,temporal_half,native_h01 and inputs64;
all except native_h01 have finished at the last inspection. Geometry identities
are reused, all physical inputs/trajectories regenerated. Initial pool replays
old input seed1307500965, not a tuned training seed. New train seed remains fresh.

Preparation finished and validation.json passes: local mesh phase difference
0.185103%pooled/0.222730%terminal; temporal differences<1e-6%. Direct native
h.01/256/delta.026 versus local gives7.29661%/9.11138%, so it is not a negligible
discretization regime. The check is on a denser mesh than training. An additional
production h.015/128 native reference is computing in native_training_mesh;
report.py will evaluate the final PINO against it on the same first geometry.
All24focused tests pass (19legacy+5dimensional); the new SI/PINO oracle uses
the installed JAX0.10 `jax.enable_x64` context. git diff --check passes.

Cold20k finite owner started as detached PID244468, child244470. Run:
`/root/autodl-tmp/tf-logs/particle_corrosion2d/20261106-physical5s-w192-cold20000-01`.
Owner record/log: study/owner_status.json and owner.log. Stages: train, online
replay audit, final reused8 evaluation, report/native same-mesh check, fullmesh
residual audit. Evaluation sibling:20261106-physical5s-w192-reused8-final-01.
Scripts are stored in the study directory; exceptions are recorded and stop
the owner without overwriting or silently restarting. Training uses fresh seed.

Blocker: none. Next: monitor first finite updates/memory and convergence, then
check every final stage and update this record with actual errors/timing.
Previous audit receipt: docs/tasks/archive/particle-corrosion-dimensional-audit.json.
All prior uncommitted code/docs/records are preserved. No external cleanup/Git action.

## Earlier session handoff (superseded by final outcome below)

Implementation and launch are complete; training/evaluation are still active,
so do not archive or claim final convergence. Last inspection:2400/20000updates;
2000-step dev4phase35.1592%pooled/46.3801%terminal, concentration34.5746%/45.3850%.
These early errors are high; no improved accuracy claim. Held update48.1ms,
measured active peak15.3868GiB. Production-native4080node reference is complete
(native_training_mesh), with the same SI parameters and graph as the PINO.
No source edits should be made during the run that invalidate checkpoint replay.

Finite owner continues automatically. Resume by reading owner_status.json;
if running, inspect train.log plus run metrics.jsonl/loss.jsonl. After all five
stages pass, inspect report.json, native same-mesh metrics, full_mesh_residual_audit.json,
online_validation outputs, and four-sample fields_first4.png. Never replace the
scheduled final checkpoint by the best reused8 result. If an owner stage fails,
preserve its log and repair only the failed postprocessing before rerunning it;
do not restart training silently. There is no exact online optimizer resume.

Diff-level changes: new dimensional factory/presets/tests; validation hooks in
PDE/reference/input paths; default/preview switched to SI; README/root overview
and artifact registry distinguish new physics from preserved old results.
Validation:24focused tests across recorded runs, changed-module/study-script
AST checks,130real file links, all particle artifact directories and git diff
--check pass. No new repository raster images, raw fields or weights.
Memory files changed this session: PROGRESS,DECISIONS,this active task; prior
completed diagnostic archives remain as recorded in earlier sessions.

## Reference gallery requested by the user

Rendered the first four reused local-FEM cases at exact stored indices0,2,5,8,10:
tstar0,.2,.5,.8,1 and physical seconds0,1,2.5,4,5. No temporal interpolation,
new reference solve, model inference or numerical-setting change. Uses existing
conforming triangles, smooth color shading, white phi.5 contours and gray
ceramic interiors; coordinate labels are physical micrometers. Verified source
backend/material/scales/time conversion and finite fields; inspected the image.

Figure:
`/root/.codex/visualizations/2026/10/24/01a16d71-2d47-7201-9441-ab74e0523152/particle-corrosion-physical5s-local-reference.png`.
Reproduction script and JSON provenance are adjacent, named
`render_particle_physical_reference.py` and
`particle-corrosion-physical5s-local-reference.json`. PNG934848bytes, outside Git.
First4terminal smooth corrosion fractions21.4433%,20.5475%,19.3795%,19.3320%
of non-ceramic area, including the initial pit. Owner still reports training;
keep the broader physical-pilot task active with its existing automatic stages.
This display request is complete. Memory changes: this task and PROGRESS only;
no additional numerical decision, Git action or cleanup. Final check: diff whitespace.

## Final outcome and strategy audit

All five owner stages passed: train, complete online audit, final8 evaluation,
saved-field/native report and full-mesh residual audit. Training wall1376.44s
(22.94min), median held49.04ms, active peak15.3945GiB. Fresh runtime seed,
not a paired-seed causality experiment. Audited all20000input draws,380consumed
geometries,316new installed geometries,1520spatial refreshes and all64final
slots; no evaluation overlap or future training fields. Neural convergence
failure is preserved; do not promote these weights as SOTA or replace the
physical definition with the easier old dimensionless regime without instruction.

| Final relative L2 (%) | Eight reused local-FEM cases | First case, exact production-native reference |
|---|---:|---:|
| Phase, pooled |35.4226|33.6062|
| Phase, terminal |46.8794|46.2413|
| Concentration, pooled |34.7789|33.0255|
| Concentration, terminal |45.6977|44.7988|
| Corroded field, terminal |96.7394|96.5882|

On the first geometry, predicted smooth corrosion fraction decreases from
4.6482%at0s to0.7041%at.5s and0.7459%at5s; local reference instead grows
4.8043%->6.4825%->21.4433%. This confirms wrong early shrinkage, not only a
finite-horizon/local-reference discrepancy. Actual mouth-node max errors over
the audit times are about2.36e-4inphi and8.36e-4in c/cLe. At tstar.003 the raw
full-mesh AC/CH RMS are27.5589/6.67784; at tstar1 they are6.89916/.0022365.
98.01%of late AC absolute residual is in predicted interface nodes occupying
2.86%of volume. These residuals have different physical scales; do not compare
AC and CH magnitudes directly as a claim about gradient dominance.

### Comparison to the successful electropolish PINO

Compared actual configs and frozen loss source from
`/root/autodl-tmp/tf-logs/electropolish2d/20261029-electro-operator-match-nonlocal-01`.
Current/frozen physical_fields,make_loss_components,make_loss_vector ASTs agree;
later spatial-sampling support and preset promotion changed surrounding files,
not the selected loss equations. The single-interface SOAP PINN is a separate
recipe and must not be substituted for this Adam operator comparison.

| Feature | New particle PINO | Successful electropolish PINO |
|---|---|---|
| AC/CH update |Joint, no stagger path|Joint, stagger_steps0; operator rejects nonzero stagger|
| Fixed PDE loss scales |AC1,CH1|AC1,CH1|
| Causal |Absent|Off; separate optional causal experiment exists|
| PDE reduction |Mean absolute original residual|Same|
| Phase/liquid output |tanh phi, unrestricted scaled liquid, algebraic KKS|Same|
| Jacobian-divided residual |No|No|
| Initial/boundary conditions |Soft|Soft|
| Loss grouping |AC,CH,mouth_phi,mouth_c,IC|AC,CH,top(phi+c),bottom(phi+c),IC|
| IC/BC concentration metric |Concentration error divided by cLe before square|Physical concentration error, no division|
| Gradient balancing |Inverse L1 parameter-gradient norms, every25, cap1000, .9new+.1old|Same norm/frequency/cap, fresh weights(alpha1)|
| Adam LR |1e-3*.5^(step/3000),floor1e-5|1e-3*.95^(step/1000),floor1e-5|
| LR at5k |3.1498e-4|7.7378e-4|
| Budget/batch/time |20k/B4/Nt32 uniform(0,1)|5k/B8/Nt4 uniform(.001,1)|
| Network |Two point streams,width192,4heads,2condition/3FiLM,96spatial directions|32x64image,4x8patch,width64,4heads,2encoder/3FiLM,32directions|
| Time features |8frequencies spanning1..4 times pi*tstar|8integer frequencies1..8 times pi*tstar|
| Spatial inputs |Rolling64geometries and8core pools/geometry,256cores,refreshed IC|8fresh ICs/update,32fixed core pools,256cores,1024fixed IC nodes|
| Diffusion |Same radial polynomial,delta.026,physical FE volumes,particle visibility|delta.02,uniform grid volumes,no holes|
| Time derivative/precision |Complete output JVP,ordinary FP32|Same|
| Driving BC |Small electrolyte mouth, other outer/particle walls closed|Entire electrolyte top and metal bottom Dirichlet; closed sides|

The particle concentration MSE multiplier is1/cLe²=786.1976 before dynamic
group balancing. This is not an AC/CH PDE prescale. Group-wise normalization
does not cancel the relative phi/c weighting inside one IC loss. The losses
are therefore not identical despite both having five terms. This difference
predates the SI recalibration and alone does not establish the failure cause.

The updated physics is also not merely a time stretch. Order below is
(Nch,effectiveN1,effectiveN2,effectiveN3), counting kappa once:

- Original successful particle:(80,48.63636,8,.00444444).
- Original speed2:(160,97.27273,16,.00888889): an exact factor2time extension.
- New SI particle:(1.7,1070,176,.103): relative to original,(.02125,22,22,23.175).
- Electropolish:(.425,535,88,.012875): new particle/electro=(4,2,2,8),
  consistently reflecting halfLc,twiceL and unchangedTc.

Uniformly extending physical duration narrows early dynamics in normalized
time and changes finite-network/sampled-loss optimization; equivalence of PDE
rescaling does not make training convergence invariant. Original speed2 failure
cannot be reinterpreted as a coefficient-ratio bug. For the new material regime,
the reaction/transport competition also changes substantially.

Saved Nt32 draws contain no tstar<.01 in72.935%of updates, with median earliest
.02158. However successful electro Nt4 misses that interval in96.9%, so these
counts do not prove that sample count causes failure. Full-mesh early residuals
locate an unresolved initial transition. The late q/cLe mean is about-0.9984,
consistent with an almost constant chemical driving field, and near-pure phi
reduces h',g' and the tanh Jacobian. Such bulk states can leave large localized
interface errors while averaged residuals fall. This is an evidence-supported
failure mechanism, not proof that one missing optimizer switch is the cause.

### Untested next controls and limitations

Keep the physical PDE fixed. Compare the existing recipe with slower electro
LR decay and with an explicitly matched physical-data IC/BC metric/grouping
as separate controls; then test early-time coverage/adaptive causal or stagger
individually. None has been launched. Constant PDE prescaling alone can be
mostly canceled by ideal inverse-gradient normalization: scaling loss_i by a_i
scales its norm by a_i, leaving relative normalized gradient contributions
unchanged up to a common factor, absent caps/EMA/clipping. Do not promise that
prescale adjustment alone fixes the observed dynamics.

Reproduction: study/strategy_audit.py; exact scalar/stream/source checks:
study/strategy_comparison.json. Final result/report/source links remain in
study/report.json,owner_status.json,full_mesh_residual_audit.json and run/
online_validation.json. Archived compact receipt is adjacent:
particle-corrosion-physical-pilot.json. No task jobs or pending required stages.
Completed task-end: numerical sources unchanged during this audit; README/root
overview,artifact index,PROGRESS,DECISIONS and this archived task now report the
negative result and comparison rather than a pending or validated accuracy claim.
Final verification: all8saved local-reference metrics independently recomputed;
the actual installed Optax schedule agrees with scalar values at0/3k/5k/10k/20k;
132file links, all particle artifact directories, receipt/source paths and
git diff --check pass. Prior24focused numerical tests were not needlessly rerun
for this read-only strategy comparison and documentation-only completion.
