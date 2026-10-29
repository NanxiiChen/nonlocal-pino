Latest directive (2026-10-28): user explicitly switches to electropolish neural-operator development. Default-window request is complete; remaining global/chemical accuracy questions are deferred. This task is inactive; resume only on explicit request.

# Confined CH — time windows are the default

Current state (2026-10-28): user explicitly makes verified original-complex-case
30time windows the default. Implemented canonical train.py default selection,
matching solver default and auto timestamp/nonlocal-windows name on a runtime
config copy. Exact historical strategy parameters retained: ten waves/T1.5,
5000SOAP+500LBFGS total, LR.001, ordinary FP64, phase-only/exact mu/hard mass.
Default switch is COMPLETE and checked without retraining. Original simple
train_base foundation and all failed global experiments remain explicit options.
The broader all-field accuracy limitations are retained/deferred; no jobs queued.
Current verification: real no-config CLI/dispatch/naming, explicit names/legacy
CLI,31schemas, strict reference, complete selected trajectory, docs/index/diff.
Earlier directives below are historical chronology, not pending instructions.


Current state (2026-10-28): requested shared two-head/two-loss global experiments
complete, all4cold starts and independent audits exit0. Fixed/gradient_norm/
stagger/raw-gradient common phase55.88/63.63/68.83/55.73%; no acceptable accuracy
or improvement over retained phi-only global/window baselines. First3matched
cohort has57identical sources, initial parameters/full time arrays; raw later
optional-normalization control preserves model/initialization and records source
deltas. Detailed record archive/chbalance-results.md, analysis.json, provenance
JSONs, four-way plots/raw logs/configs. All workers idle. Current sub-study done;
all-field/global accuracy remains OPEN. Selected existing window fallback and
simple default preserved. Earlier directives below are historical chronology.


Current directive (2026-10-28, supersedes the selection note below): user requests
fresh one-network global shared phi/mu heads with two scaled PDE losses and
GradNorm/stagger experiments on the original ten-wave/T1.5 case. Current work:
implement explicit optional train.loss_balance, test gradients/state/masks and
frozen refinement, then matched fixed/gradient_norm/stagger5000+500 cold starts.
No windows/propagated states/reference labels in these runs. Preserve existing
window fallback and original simple defaults. Acceptance: three complete runs,
final/best CPU replay, common/native phi+mu L2/energy audit, convergence/field
figures, initialization/source parity and direct-path records. Blocker none.
Next: implement balancing helper/CLI integration/config validation and tests.


Current directive (2026-10-28): user explicitly selects time windows again,
superseding the original no-window goal below. Use the validated original rich
T1.5 train_coarsening_windows strategy. Global and moderate studies stay as
historical experiments; no new adaptive threshold/retraining/default change is
requested by this explanation turn. All-field accuracy remains OPEN; workers
idle. Current acceptance: preserve validated window checkpoints/physics, make
switching and error propagation explicit, and assess chemical/physical accuracy
if further work is requested. Old goal/acceptance below are historical context.


Goal: user rejects sequential time windows; solve all[0,1.5] with one jointly
trained global model/checkpoint. User now retains phase-only
prediction with chemical elimination; mixed-field studies are historical controls. No intermediate PINN state
propagation/freezing or reference labels in physics runs. Preserve physical/
reference/native operator and original artifacts. No Git authorization.

Status: OPEN / requested tanh-frequency study completed. Five new full5000+500
cold starts and all CPU/derivative checks complete. Tanh alone gives no clear
phase gain, r/J worsens this strong recipe. Frequency-only integral M5 gives
current lowest phase17.99%/29.37%, but mu129%/837%, overshoots and energy rise
remain. No SOTA/default/Git change, all workers idle. Exact current resume is
at the end of this file; user's primary route is phi-only, not mixed fields.

Acceptance: one saved global model covers arbitrary times, same hard IC/mass,
reported FEM/native phase and chemical errors plus physical audits/CPU replay;
failed controls retained; no disguised time window or external target solver.
Compare existing44.79% global and3.9% window phase without promising equivalence.

Context: archive/confined-ch-optimization.md, chopt-analysis.json. Window solution
requires30 networks; user explicitly dislikes it. Existing global trials change
IC lift/log coordinate, early sampling/curriculum, resolvent and chemical
elimination; best modes8 logblend .01 gives44.79%/55.51% in short pilot.
Chemical elimination is acceptable separately; may keep or ablate. A full5000+
500 trial with all times jointly sampled is not yet done. Deferred SOAP side
work remains outside this task. No FEM rerun warranted.


Full prior5000+500 all-horizon baseline completed:20261027-chglobal-resolvent-01,
common phase51.007%/terminal61.252%,255.48s. Native phase50.404%/60.734%; native
mu71.65%/terminal260.97%. No empirical replacement for window accuracy yet.

Implemented optional space_time_resolvent in phase metric, same one global
network/AD evolution residual. At sorted times solve a block lower triangular
linear system on residual vectors only; no predicted field propagation, no
frozen intermediate state, no time-slab networks. Weighted eigenbasis of
A=-D_delta precomputed. K=M epsilon^2 A^2; h-dependent inverse is analytic in
that basis. Checked weighted symmetry before explicit roundoff symmetrization.
Loss=sum_i h_i||e_i||_V^2/T, includes terminal residual explicitly. All parameter
gradients traverse the whole loss. New test independently solves dense block
system, recovers original residuals, verifies gradients. A second test compares
actual physical loss to finite-difference phi_t/independent NumPy A/implicit
metric solve and parameter directional finite differences. Both pass; raw logs
/tmp/chglobal-spacetime-test.log and /tmp/chglobal-loss-test.log.

Full global seed42 run now /tmp/chglobal-spacetime.log, directory
20261028-chglobal-spacetime-01,5000SOAP+500LBFGS same modes8/log-blend. Canonical
new config train_coarsening_global.py; no curriculum (all[0,T] every update),
Nt8 log stratification,16 refinement points+explicitT. First verify final and
best phase/mu and CPU replay; if promising, frozen second seed validation.
Do not claim exact continuous zeros from finite collocation or hide chemical
errors. Previous baseline snapshot remains prior to metric branch changes.


Space-time differential metric full run fails:20261028-chglobal-spacetime-01,
common phase97.969%/99.847%,native97.984%/99.848%,218.74s. Keep failed control.
No stopping early or seed shopping. New hypothesis: integral formulation
R=phi-phi0+M A integral(mu) explicitly keeps IC rather than quadrature of phi_t.
Global log-Chebyshev matrix integrates chemical values from ONE network; no
training targets from another solver. Polynomial integration moments and scalar
linear IVP convergence directly checked (n16 error5.96e-5 ->n32 6.16e-9 for
rate100). New focused physical/parameter-gradient test underway. Loss preconditioner
(I+tM epsilon^2 A^2)^-1 defined explicitly. Unlike differential residuals,
chemical-time quadrature error is a numerical discretization to audit/refine.

Integral32 first full run completed; see summary/raw log. Inverse-Laplacian + initial-feature/increment full5000+500 run now active after focused derivative and weighted inverse/gradient tests pass2tests8.51s. /tmp/chglobal-inverse.log; retain each outcome.


Integral32 completed5000+500,286.72s: common phase84.264%/73.404%, not a useful
improvement. Independent native quadrature audit confirms approximation errors
small and refining32/64/128 converges; not a reference error. Native trajectory
matches existing reference1.90e-14 and is validation-only. Need optimize trial
space/optimizer, not increase quadrature blindly.

Additional global-basis model now implemented (one MLP returns24 spatial
coefficients of global log-time Chebyshev modes, hardIC+mass). All coefficients
train together over full time; no freezing/propagation. Uses80 integration
nodes so cubic chemical interpolation (degree72) is unaliased. Startup output
scaled.01; knownIC input explicit. New test covers arbitrary-time JVP, IC/mass,
independent Gaussian chemical integral, compiled nonzero finite parameter
gradients. Test underway; run only after passing and current inverse finishes.


Global inverse5000+500 completes276.21s, common phase55.84%/64.96%, native mu
91.51%/terminal337.73%. Global basis24 completes200.48s, phase35.61%/63.40%,
native mu678.27% with phase overshoot3.49: reject as physically accurate despite
lower phase L2. All first5 final/best CPU saved-field audits pass (basis maxmu
5.02e-12); code/serialization is not the fitting cause. Energy-balance variant
160nodes/weight1 full run active after independent energy variation/objective/
gradient check passes1test10.89s. It penalizes chemical dissipation errors,
not just phase appearance. /tmp/chglobal-energy.log.

Final planned comparison: normalize the same global integral residual by t,
R(t)/t, so early information is not weakened by automatic O(t) vanishing.
Same global basis24/N80, no energy, no time windows or future labels. New
normalization test compares dense inverse expected value; do not change case
source after starting that run until a frozen-seed comparison decision.


First6 global controls all retained. Energy term lowers native chemical error
but worsens phase to~89%; not selected. Final R/t time-average control active,
20261028-chglobal-average-01 /tmp/chglobal-average.log. Source frozen after
normalization check1test16.24s. No more numerical methods planned this task;
select a fixed global candidate, second seed if meaningful, or report limits.
Audit script chglobal-audit.py runs explicitly named complete global runs,
final+best CPU field replay, independent chemical law, hard mass, energy,
continuous AD evolution diagnostics and parameter counts. First5 pass with
max phi1.07e-13/mu5.02e-12. Energy audit running. Reported trainer wall times
exclude initial reference/operator/metric preprocessing; do not call them full
process end-to-end times. Physics/reference/IC preserved throughout.


R/t completed5000+500 in197.77s (trainer only), common phase21.5396%/43.1779%,
native mu258.07%; materially improved but inadequate all-field/late accuracy.
Phase extrema[-1.759,1.668], so cannot promote it as solved/SOTA. One final
optimizer-only experiment now runs5000SOAP+200 exact-HVP Newton-CG on the same
global objective (config train_coarsening_global_newton.py). No new ansatz/PDE
or reference data. Manufactured hard-IC y_t=2t solve recovers a=0,b=1, passes
1test.446s. Source frozen. Record HVP calls/actual wall rather than claiming
200Newton equals200LBFGS. Initial /usr/bin/time unavailable; used shell time
instead; first attempt never launched trainer and created no run directory.
Final experiment /tmp/chglobal-newton.log,20261028-chglobal-newton-01.
After it ends, independently replay, report all controls, and finish this task
honestly even if target accuracy remains unresolved. No further formulations.


Newton interrupted by SIGINT after554.979s total, in SciPy inner CG Hessian
product (trace retained). Only iteration1 logged, exact later counts unknown;
no completed200-iteration result. SOAP5000 checkpoint equals saved best leaves,
independent CPU replay passes phi9.55e-15/mu6.82e-14. This partial stage is
excluded from completed-final ranking. No Newton default promotion.

Physical plot revealed late grid-scale oscillations, phase[-1.76,1.67], native
mu terminal2716%, energy rise.0671 in best completed R/t model. This new concern
justifies one targeted existing-method recombination: same global basis24 and
fixed80 nodes, but instantaneous AD time PDE residual with the independently
checked Neumann inverse norm; include t0 evolution. Uniform weight over81 fixed
nodes, not physical-time quadrature; no reference labels/frozen states. New
config train_coarsening_global_strong.py;20261028-chglobal-strong-01,
/tmp/chglobal-strong.log. Source now frozen including README for a second seed
if useful. Do not edit example files before that decision. This is not a new
operator/physical formulation. Verify chemical and energy, not just phase.
All prior7 complete final/best CPU replays pass; all16 then-existing config
schemas/physical identities pass; two legacy mixed-network IC/JVP/AD gradient
tests pass53.67s. New strong config validates in the canonical trainer.


Strong global differential control completes5000+500,251.90s: common phase
57.841%/65.552%, native mu90.323%/terminal319.794%, phase[-1.025,1.008]. It
removes large phase overshoots but does not solve morphology. Full audit pending.
A single existing-method causal-weight control now runs: global_causal config,
strength1/min1e-8, every fixed time has positive weight, no state stepping.
Only SOAP uses stopped exponential prefix weights (Eq3.2 Wang etal2022);
L-BFGS explicitly removes causal record in copied config to optimize a uniform
consistent objective. Gradient-rule unit test and schema pass. New config
train_coarsening_global_causal.py, /tmp/chglobal-causal.log. Case source frozen.
No other formulations/optimizers planned. If causal unsuccessful, report all
single-network limitations and leave next-step representation diagnostic exact.


Causal full run completed5000+500,251.97s, common phase56.766%/64.463%, native
mu105.619%/terminal367.434%. Full final/best CPU replay passes. Exact adaptive
weight diagnostics: min/terminal weight atSOAP5000=.01964; final diagnostic
weight.15276 (LBFGS itself uniform). Do not call causal weighting converged.

Validation-only native free temporal projection (chglobal-representation-floor.py/
json, not a learned PINN or a physics-training target) gives rank24 phase
0.05746% pooled/.08887% terminal, mu.79325%/4.32166%, on independent dense time
queries. Rank48 improves further. Initial derivative RMS defect remains62.99,
so this does NOT rule out derivative-resolution issues or prove a unique
optimizer cause. It does show time trial functions can represent accurate
fields when spatial coefficients are unconstrained. Source/reference unchanged.

One final capacity-only control active: train_coarsening_global_wide.py uses
width128 instead of64, same global basis24, instantaneous Neumann-inverse norm,
81 fixed temporal queries,5000SOAP+500LBFGS, FP64/ordinary. Directory
20261028-chglobal-wide-01, log /tmp/chglobal-wide.log, session69236. Case source
(including README) frozen after this config was added; do not edit example
files before deciding whether a fixed2026 cold start is useful. No more new
PDE/loss/window/optimizer schemes. After run: full audit via chglobal-audit.py,
update plots/index/README; if still inaccurate, report measured limitation and
leave exact representation/optimization diagnostic resume point instead of
claiming a solved single-network case. All prior9 complete controls/audits are
retained; Newton is explicitly incomplete and excluded from rankings.


## Session outcome and exact resume point

Ten complete seed42 global controls have final+best CPU replay, independent
native chemical law, hard-mass, energy and continuous derivative diagnostics.
All prediction replays pass (max phi1.07e-13/mu5.02e-12); physical accuracy does
not. Best completed phase remains average R/t21.5396%/43.1779% common-FEM,
native mu258.072% pooled/440.915% positive-only/2716.044% terminal, energy
increase.06705 and phi[-1.759,1.668]. Terminal mu error RMS1.190 vs native
reference.043815, not merely a denominator issue. Best evaluated checkpoint is
final5500, not an untrained or label-fit model. Lower phase L2 is insufficient.
Width128 (57,240parameters vs16,344) ends65.635%/73.925%, mu59.396%/188.274%,
308.02s; changing width alone does not resolve this seed. No seed2026 run was
warranted to promote an already inaccurate method; no stability claims.
All full runs retain SOAP5000+LBFGS500, FP64/ordinary, unchanged physics/reference.

Canonical route: train.py with train_coarsening_global_average.py, single
model_final.eqx in20261028-chglobal-average-01. It is an experimental result,
not a recommended accurate solver. Strong, causal and wide configs also use
one checkpoint/all times; no frozen intermediate fields. Causal SOAP has
stopped prefix weights, but L-BFGS is truly uniform (saved metadata verifies).
Newton partial model_soap.eqx replays; the200-iteration stage never completed.
Interrupted in inner CG after554.979s total; exact counts beyond logged1 unknown.
Do not include it in completed final comparisons or silently retry unlimited CG.

Mathematical diagnostics: independent global integration converges32->64->128;
native IVP validation agrees with existing reference1.90e-14. Free nodal global
time coefficients rank24 fit phase.05746%/terminal.08887%, mu.793%/4.322% on
dense queries, BUT initial derivative residual62.99 RMS remains. These free
coefficients are not a neural solution/training target or evidence of optimized
PINN accuracy. A uniform leftover-IC explanation lacks simple correlation evidence.
Do not attribute all error to reference, a single gradient conflict or capacity.

Resume precisely:
1. Read chglobal-analysis.json, chglobal-results.md and representation-floor.py/
   json. Do not repeat completed expensive runs or revive SOAP side work.
2. Audit temporal approximation with a hard initial derivative constraint derived
   only from the known PDE/IC, versus the existing free coefficient field fit.
   Evaluate the actual81-node Neumann-inverse residual, not only phase L2.
   This distinguishes fast initial derivative resolution from accurate fields.
3. If needed, run a clearly marked supervised-only spatial/coefficient fit of
   the SAME global network to assess neural representability; never report it
   as a PINN or initialize physics experiments from those reference-trained weights.
4. Choose a general representation/conditioning change only after those probes.
   Preserve one global network and no intermediate reference/PINN state freezing.
   Keep initial/geometry/operator/reference identity unchanged. Do not simply
   stack more losses or choose seeds until one looks good. Match frozen configs
   before any valid accuracy promotion. User prefers around5000 training steps.

Diff-level changes: optional SpaceTimeMetric/NeumannInverseMetric, checked global
integration, global basis/increment/IC-feature architectures, full-horizon
sampling/causal weights, optional exact-HVP cost-control helper, configs through
train.py, strict replay/physical/postprocessing scripts and independent tests.
Neumann inverse now explicitly rejects extra disconnected null modes; this last
validation changes no connected-graph arithmetic. README/DEVELOPMENT/index state
research status; original simple default/reference/paths and historical windows
are preserved. Data/replay arrays stay in tf-logs, direct paths/no hashes.

Verification: separate focused tests cover weighted block inverse, actual AD
physical loss/parameter finite differences, global quadrature/manufactured IVP,
IC-increment chain rule, Neumann inverse/nullspace, global basis chemical
integration/mass/time derivatives, thermodynamic variation/energy loss, causal
stopped-gradient direction, and manufactured Newton PDE. Legacy IC/mass/JVP and
both local/nonlocal mixed parameter gradients pass2tests53.665s. Every full GPU
run and final/best CPU replay pass numerical execution; physical energy/error
checks explicitly fail, not hidden. All19 schemas/physical identities pass.
Raw logs and configs/summaries are chglobal-* in archive; figures are
chglobal-convergence.png, chglobal-fields.png and chglobal-errors-energy.png.

/task-end: leave this task OPEN for unresolved accuracy with the exact resume
point above. Update PROGRESS/DECISIONS; do not archive as solved. No pending
workers, blockers, Git history action, SOTA tag or default promotion.
Memory changed: docs/PROGRESS.md, docs/DECISIONS.md, docs/tasks/confined-ch-global.md.

Final verification: explicit disconnected-nullspace guard and weighted inverse regression pass2tests1.957s; all19 config schemas/physical identities, registered paths/JSON/new README/dashboard links/source whitespace and git diff --check pass. Scientific fields and convergence visually inspected. All training/FEM/audit workers idle. Only diagnostics/scripts/docs changed after the full runs; connected-graph arithmetic is unchanged by the final guard. No unchanged expensive suite repeated.

## 2026-10-28 user-directed mixed-field diagnosis (active)

User asks whether phi/mu use shared channels or separate networks, and whether
field scales/two loss equations were handled. Current best-phase global model
eliminates mu; old mixed model shares one trunk/two channels with no output scale.
Known-IC physical-volume RMS phi=.674445, mu=.488573, phase rate=99.6552;
validation-only terminal native mu=.043815. New controlled direction: jointly
learn both globally supported fields, output mu with declared scale .5, divide
both chemical-unit residuals (A^+r_phi/M, r_mu) by .5, and compare shared/separate
field trunks. No frozen time states, reference loss, changed PDE/reference or
Git action. Preserve original arithmetic/checkpoints. Next: implementation,
independent residual/gradient/IC tests, then cold-start paired controls.

Mixed control results: shared58.4902%/63.5036% common phi, native mu70.2777%/
301.4807%,265.58s; separate71.2418%/71.3012%, mu74.0590%/360.5670%,285.05s.
Both final/best CPU replays pass; shared/independent snapshots have identical
40 source files. Separate has more parameters, so this is a layout/capacity
control, not a parameter-matched ablation. Initial separate/block checkpoint
files match byte-for-byte (direct comparison, no hashes). Block run
20261028-chmixed-block-01 reached5000SOAP phi48.72%, refinement still running.

New validation-only chmixed-time-probe uses the same native IVP, replay1.90e-14.
Free rank24 phi/mu coefficients yield phi.07697%/mu.94542% on dense checks,
but strong evolution loss1.00657; fixing known initial phase rate and mu gives
phi.07372%/mu.73586%, strong loss.054208 (18.57x lower), chemical loss9.90e-6.
These are NOT neural/PINN results or proof that rank24 cannot fit PDE derivatives;
the fields were fitted, not the full strong residual. Initial-rate defect is
~3.75e-14. Next supervised SAME separate network5000SOAP audit tests neural
field expressivity, with explicit future targets and zero weight transfer.
Pending: finish block, CPU audit, representation diagnostic, summarize limits.

## CURRENT session outcome / exact resume — mixed fields (2026-10-28)

Status OPEN, no active trainers/solvers after final checks; no blocker. All three
new physics-only runs completed5000SOAP+500LBFGS, seed42, FP64/ordinary.
- 20261028-chmixed-shared-01: phi FEM58.4902%/63.5036%, native mu70.2777%/
  301.4807%,265.58s,17,904parameters.
- 20261028-chmixed-separate-01: phi71.2418%/71.3012%, mu74.0590%/360.5670%,
  285.05s,32,688parameters. Best phase at5000, final5500 explicitly reported.
- 20261028-chmixed-block-01: phi46.1064%/38.6016%, mu50.5093%/130.0648%,
  386.24s,32,688parameters, sampled energy max rise0, phase[-1.0444,1.0599].
  Final5500 is best phase. Known linear mixed Volterra block inversion uses
  scalesEphi1, Emu.5; no extra chemical-elimination or conservation loss.

Canonical train.py new configs train_coarsening_global_mixed, global_separate,
mixed_block, with direct-path whole checkpoints. Shared model has one trunk/final
48 coefficient channels (24 per field); separate has two MLPs/one global model.
No sequential temporal models/frozen states. Output and row scale are separately
explicit, rounded known-IC mu RMS.5. No reference loss in any physics run.
Original model/chemistry/template arithmetic retained. Block precomputes checked
weighted eigensystem and per-mode time-block inverse; fixed Nt80 resolves cubic
rank24 log-time interpolation. Independent full2x2 dense inverse and true nonlinear
parameter finite-difference test pass. Original two-layout inverse-A loss/gradient/
IC/mass test also passes. Historical R/t and strong checkpoints replay after changes.
Final/best all3 CPU replays pass max~3e-14; mu constitutive defects are measured,
not assumed zero in mixed models. No full unchanged expensive suite repeated.

New representation result is SUPERVISED ONLY, never a PINN score:
20261028-chmixed-representation-01, same separate net/byte-identical init,
5000SOAP/noLBFGS,93.04s after reference preprocessing. Native phase pooled.43846%/
terminal.42996%; mu2.7160%/5.16961%. Explicit scope.txt, separately named
supervised_model.eqx, targets_supervised_only.npz and source snapshot. No weight
transfer. CPU replay maxphi9.55e-15/mu2.08e-14. Actual physical loss on those
weights: strong inverse-A.459483 (phase.458010/chemical.00147347); mixed block
8.98496e-5 (phase1.66141e-5/chemical7.32355e-5), versus physics-trained block
.00276241. This is direct evidence of a feasible better joint approximation
under the block objective, not proof that field fitting alone controls derivatives
or that every failure is only optimization. Pure field neural capacity is now
isolated; do not rerun this same diagnostic or train a PINN from its weights.

Boundary audit uses the existing outer defining-function distance approximation
and exact circular distances, band<delta. Block band21.098% physical area carries
57.391% terminal native squared error; relative band60.849%, bulk28.474%.
Native/FEM operator gap remains3.779%, independently frozen/validated. No reference,
geometry, physics, time horizon, migration, default, SOTA or Git change.

Diff-level changes this session: GlobalMixedNetwork and factory dispatch;
optional output scaling / two chemical-unit strong losses; mixed block metric and
integral loss/validation; three configs; independent mixed and block tests;
scale/time/field representation diagnostics with explicit supervised provenance;
CPU/constitutive/mass/energy/boundary audits and scientific plots; README,
DEVELOPMENT, artifact index and memory. All original uncommitted work retained.
Raw configs, metrics, summaries and source paths live under the direct run dirs;
copied logs/results under archive/chmixed-*. No hashes.

Resume exactly:
1. Read archive/chmixed-results.md, chmixed-analysis.json and
   chmixed-representation-audit.json. The old ten global controls remain valid.
2. Use mixed_block as the physically healthier conditional baseline, not SOTA.
   Preserve best pooled R/t21.54% as a separate metric tradeoff, never erase it.
3. Next choose ONE reference-free optimization-path probe after deriving/checking
   it: candidate is continuation from a stable initial linear CH operator to the
   original nonlinear mixed model, or a linear-propagator IC lifting. Neither is
   implemented; the final physical PDE must
   be unchanged and train all times jointly. Quantify known IC linear/nonlinear
   rate split first; no reference-dependent future trajectory or frozen states.
   The current supervised diagnostic is evidence only, never an initializer.
4. Match5000+500 budget and cold-start provenance, compare phi/mu/energy/boundary
   errors; consider a second fixed2026 run only after a materially accurate method.
   Do not widen nets/retune seeds or reopen deferred SOAP investigation.

Reproduction: CPU chmixed-audit.py --append --names RUN..., then
chmixed-boundary-audit.py, then chmixed-plots.py. Capacity audit is
chmixed-representation-audit.py; never call its model a physics-only solution.
/task-end: remain OPEN. Memory: PROGRESS.md, DECISIONS.md, this task. No Git work.

Final verification for this session: all22 training config schemas pass; all3 new
configs match the frozen richer physical identity and omit strategy seeds.
Artifact JSON syntax/all4 direct run registrations pass. Source/initial-file/
physical sampling optimizer parity is persisted in chmixed-provenance.json.
Two new numerical tests pass (12.44s mixed,9.93s block), old R/t/strong and all3
new final/best CPU replays pass, supervised-only CPU replay/actual PDE losses
are independently checked. Changed/new source/log whitespace and git diff
--check pass. Scientific convergence and3x5 phase fields visually inspected;
energy consistency is assessed separately from execution/replay. No training/
FEM/audit worker remains. No Git history operation or default promotion.

## Active follow-up — tanh/Jacobian and frequencies

User now explicitly retains phi-only/eliminated mu and requests tanh residual
transformation plus Fourier-frequency audit. Four predeclared cold starts:
strong-m8 control; tanh-map-m8; tanh-jac-m8; tanh-jac-m5, each5000+500/seed42.
Use amplitude1.1, known IC latent lift, exact scalar mass-shift with implicit AD,
physical r/J before invertible(A+Pi0)^-1. No Jacobian floor or claimed cancellation
of CH diffusion. Strong instantaneous residual avoids tanh time-integral aliasing.
Reference overshoot~2.8% makes strict1.0 tanh restrictive; entry remains explicit.
Fourier modes8 wavelengths.125/.1; current global model uses no temporal Fourier.
Next implement/test actual map/mass/JVP/loss, freeze source and run4controls.
No Git authorization/default changes; preserve all prior mixed/phase artifacts.

Tanh implementation/tests pass: scalar mass shift independently matches Brent;
all-input implicit JVP, analytic latent/physical chain including alpha_t, native
mu law, transformed residual, independently assembled lifted inverse objective,
parameter directional finite differences, exact IC/mass, stable tanh derivatives
at|z|20 and second derivatives. M8 unmapped/mapped parameter leaves agree at
initialization. A cancellation-free tanh custom JVP was added while only the
unmapped control was running, before any mapped run: its physical unmapped
training path is unchanged. Do not claim byte-identical whole source snapshots
for that first control; compare its active functions/optimizer/operators exactly.
Mapped controls start with the final stable tanh implementation and fixed sources.
Next monitor4runs; CPU replay/Jacobian/energy/dense-time audit and source parity;
compare final and best phi/mu, map-only versus Jacobian and modes8 versus5.

Primary first3 complete: fresh strong-m8 phi59.012%/65.162%, native mu87.122%/
315.912%,253.53s; tanh-map-m8 phi58.563%/64.977%, mu73.263%/265.295%,276.76s;
tanh-jac-m8 phi73.073%/78.424%, mu162.598%/518.044%,302.69s. Jac-m8 best is
step1000 phi71.002%, final5500 worse. Jac-m5 is running (serialdriver42059).
Independent checkpoint1000 constant-mode contribution4.25%, dense midpoint
mean loss.6159 vs training.6167, so no basis to blame all failure on constant
mode or gross time aliasing. CPU final/best audits first2 pass; third in progress.
Add ONE planned frequency-only current-best R/t modes5 control afterprimary4:
.venv/bin/python examples/confined_ch2d/train.py
 --config=examples/confined_ch2d/configs/train_coarsening_global_average.py
 --config.arch.spatial_modes=5 --config.log.run_name=20261028-chtanh-average-m5-01
 --seed=42. Keep existing M8R/t historical result (not matched fresh rerun).
Do not add new operator ordering/continuation schemes or maps now.

## CURRENT end state / resume — phase-only tanh and frequency (2026-10-28)

Status OPEN: requested tanh/frequency experiments completed, high-accuracy global
solution still unresolved. Five new complete cold starts, no workers/blockers.
Every run5000SOAP+500LBFGS, seed42, FP64/ordinary; canonical train.py, no reference
labels/weight transfer/time windows or changed physical target/reference/operator.
Primary names under /root/autodl-tmp/tf-logs/confined_ch2d/:
- 20261028-chtanh-control-m8-01: original strong, phi FEM59.0121%/65.1624%, native
  mu87.1217%/315.9124%,253.53s,16,344parameters. Best phase at4000, table is final.
- 20261028-chtanh-map-m8-01: a1.1 tanh/original r, phi58.5632%/64.9772%, mu73.2626%/
  265.2954%,276.76s,16,344parameters. Best final5500.
- 20261028-chtanh-jac-m8-01: same map/rJ inverse before lifted metric, phi73.0732%/
  78.4242%, mu162.5980%/518.0439%,302.69s. Best step1000phi71.0024%, not final.
- 20261028-chtanh-jac-m5-01: same transform/band5, phi69.2708%/75.1005%,
  mu165.6864%/531.3104%,295.69s,15,576parameters. Best final5500.
- 20261028-chtanh-average-m5-01: UNMAPPED retained integral R/t recipe, band5,
  phi17.9854%/29.3681%, native mu129.2923%/836.5967%,190.78s,15,576parameters.
  Best final5500. Phase[-1.34285,1.24822], energy rise.007078; no SOTA promotion.
  Historical M8R/t phi21.5396%/43.1779%, mu258.072%/2716.044%, energy rise.067052.
  Historical M8 is not a fresh paired rerun; no stable/seed-independent ranking.

User retains ONLY phi prediction with exact mu elimination. Keep this preference
for subsequent primary controls. New explicit working config:
examples/confined_ch2d/configs/train_coarsening_average_lowfreq.py, equivalent to
completed train_coarsening_global_average.py --config.arch.spatial_modes=5.
Load its whole model_final.eqx directly with saved config and original reference.
Original simple pilot default unchanged. Tanh controls remain opt-in; neither
Jacobian nor map is promoted. Mixed results retained as historical diagnostics.

Implementation: TanhPhaseNetwork global latent IC lift, scalar implicit weighted
mass shift (64-step bracketed bisection, no differentiated branch decisions),
cancellation-free stable tanh/Jacobian custom JVP, exact alpha_t mass derivative;
actual r/J before full invertible(A+Pi0)^-1, no floor/clip/mean truncation.
New configs map/Jac/modes5; validation rejects mapped polynomial-integral
unaliasing assumptions. Canonical boundary normals use projected physical map,
not latent/unshifted gradients. Old unmapped constructors/arithmetic retained.
Mu reaction does not globally cancel J through nonlocal A; this is not melting's
local Allen-Cahn cancellation. Tanh was NOT previously tested in confined CH.
Strict amplitude1 remains an entry, with measured FEM range-only error lower
bound.3358% pooled/.4573% terminal; native/FEM slightly exceed the unit wells. Amplitude1.1 is a fixed10% buffer
about the potential wells, not a future-trajectory fit.

Frequency: sin/cos(pi*k*z), k1..8 or1..5, z=(x-center)/axes. M8 wavelengths.125/.10,
M5.20/.16. Median nearest spacing.01122; M8~11.14/8.91 such spacings. No obvious
input sampling violation; nonlinear activation/IC add higher harmonics. Native
A² feature gains mode8/mode5=2.46--3.30 for the four axis probes. Initial basic
wave coordinates reach5. NO temporal Fourier in this static spatial MLP; time
is rank24 log-Chebyshev. Lower input count/init means not parameter matched.

Verification:2new focused tests pass19.597s, including independent Brent mass
root, all-input JVP, small dense lifted metric/analytic CH chain, true parameter
finite differences, stable derivatives at|z|20/second derivatives, exact IC/
mass/mu. All5 final+best CPU replays pass (max phase~1e-14/mu~1e-13), historical
M8R/t also replays. Independently differentiated NumPy polynomial/Brent chains
at11 actual times match trained AD derivatives (max2.3e-12). Canonical evaluate.py
mapped replay/physical-normal/field plot completes. Mapped source snapshots are
identical51files; first unmapped control's inactive map/diagnostic helpers were
stabilized before mapped runs, but active baseline functions/training/operators/
optimizer sources exactly agree. All3M8 initial parameter files are identical.
No hash computation. Full provenance in chtanh-provenance.json.

Actual Jacobian final constant-mode loss fractions.052%/.020%; dense midpoints
versus training81 nodes differ<1%. Min J~.053/.043, no underflow. Thus neither
implementation bug, mean mode alone nor gross time aliasing explains the poor
Jacobian result; reweighting changes finite-optimization behavior. Do not claim
universal tanh failure or transfer mixed supervised capacity proof to this new
map. Tanh plus the best integral recipe was NOT trained/validated in this study.

Outputs: archive/chtanh-results.md, analysis.json, frequency-audit.json,
objective-audit.json, derivative-audit.json, provenance.json, processes.json,
convergence.png, energy.png, integral-m5-fields.png and raw config/summary/log/
metrics copies. Run-local arrays/checkpoints remain at existing tf-logs roots.
Artifact index now marks M5R/t current lowest-phase experiment and preserves the
oldM8 tradeoff/mixed controls. README and DEVELOPMENT document options/limits.

Exact next resume:
1. Read chtanh-results.md/analysis.json, compare current working M5R/t and retained
   M8R/t; recognize chemical/overshoot/energy defects despite phase improvement.
2. Keep phi-only/chemical elimination and no frozen time states. Avoid another
   blind map/network/seed sweep. No secondseed for SOTA while physics is poor.
3. A general next candidate (NOT implemented/tested here) is the full single-field
   linear space-time Volterra inverse E=(I+M eps² A² W)^-1 R, versus current
   per-time spatial(I+t M eps²A²)^-1(R/t). Derive/check its manufactured linear
   limit/IC/mass and gradients independently before training. The already tested
   MixedBlockMetric top solve with chemical residual zero gives this candidate;
   do not add its second chemical loss when mu is eliminated. No promise it fixes
   nonlinear global optimization. Alternatively a mapped integral control needs
   explicit nonlinear quadrature refinement (no finite-degree unaliasing claim).
   Neither candidate justifies changing physics/reference/defaults now.
4. Match5000+500/ordinaryFP64, verify common/native phi/mu/energy and actual
   checkpoint replay. Only materially accurate physics supports promotion/cohort.

/task-end: leave OPEN, completed sub-study recorded. No Git authorization/action.
Memory changed: PROGRESS.md, DECISIONS.md, this task. No debugging/dead code.

Final /task-end verification: all26 training schemas and new physical identities/
seed policy pass; the explicit low-band preset exactly matches its completed
saved strategy. Both1.0/1.1 map config entries validate. Artifact JSON/all5 run
paths and working checkpoint/config pass. Independently assembled actual4strong
objectives match trainer final losses within1e-8. New source/log whitespace and
git diff --check pass. Convergence, phase fields and energy figures visually
inspected; grid oscillations and late energy rise remain visible in M5R/t.
Shared phase colour scale clips at±1; true ranges are explicitly reported.
GPU and all task CPU/FEM workers idle. No unchanged expensive suite repeated.
Memory changed: PROGRESS.md, DECISIONS.md, this task. No Git history operation.

## 2026-10-28 — Active: intermediate-difficulty physical control

User requests reducing case difficulty. New train_moderate.py preserves geometry,
epsilon .02/delta .03/mesh .02 and saturated IC envelope; three low waves and
T .75 replace ten waves/T1.5. One global phi-only M5 R/t network; mu eliminated;
5000SOAP+500LBFGS, FP64/ordinary, runtime seed42. Rebuild80 nodes for new horizon.
Reference root: /root/autodl-tmp/data/confined_ch2d/moderate_multimode3.
Acceptance: new independent FEM/native reference passes unchanged phase1%/mu4%
mesh/time gates and mass/energy checks, full cold-start training, physical-field/
phi+mu L2 and energy audit, representative plots and checkpoint replay. Original
case/results/default retained. This is a bundled difficulty change, not a strict
IC-only/time-only ablation. Blocker: none. Next: generate new reference in
fenicsenv before JAX GPU work, then train via canonical train.py and evaluate.

Reference completed exit0: fixed max-time phase mesh/time .162673%/.015437%,
mu .185718%/.037107%; native time <6.1e-8; mass drift <3.6e-15, no energy rise.
Native/local phase gap2.09428%, mu14.7586%. Strict reference loader passes
(11,3682). Initial native rate RMS31.09 vs hard99.66, preserving epsilon/delta.
Preliminary reference morphology: ~19% to43% pure area, substantial evolution.
5000+500 canonical cold start started: 20261028-chmoderate-m5-01, seed42;
log /tmp/chmoderate-training.log. FEM process exited before GPU launch.
Next: finish training, final/best independent CPU replay, phase/mu/energy audit
and figures. Blocker none; no default or Git action.

### Intermediate-difficulty result / task-end

Completed reference, full SOAP5000+LBFGS500 and final/best CPU audits (all exit0).
Report archive/chmoderate-results.md and chmoderate-analysis.json are the exact
result record. Run20261028-chmoderate-m5-01:188.15s, best=final step5500. Common
phase5.1557%/9.7665%, native4.2891%/8.9375%, native mu45.0646%/156.4975%.
Dense151-time mass drift8.05e-16, chemical defect3.28e-15, CPU replay4.66e-15/
3.03e-14 pass. Phase[-1.1158,1.1736] and energy rise.00033249 persist, visually
confirmed late wall patches/grid oscillations. No SOTA/global-solved claim.
Reference phase itself changes60.8%, pure area18.75% to42.92%; frozen IC baseline
40.84%/47.65%. Simpler IVP is easier, but bundled IC/time changes and one seed
cannot isolate causes or explain away residual training defects.

Diff-level changes: train_moderate config; README stable case/config/validation;
new independent data/run registered in artifacts/index.yaml; archive audit,
reference checks/raw logs/config/metrics, fields and convergence-energy figures.
No numerical-core changes, no old reference/default/path or Git changes. All CPU/
FEM/GPU jobs finished. Current sub-study complete; global accuracy remains OPEN.
Blocker none. Exact next resume: read chmoderate-results.md/analysis.json, retain
single global phi-only chemical elimination. If further optimizing this moderate
benchmark, isolate spatial modes5 ->3 with unchanged IVP/reference/budget and
check mu/dense energy alongside phi; existing verified reference can be reused.
If instead attributing IC versus horizon effects, separately retrain ten-wave
T.75 with an explicitly validated reference prefix; not done here. No automatic
second seed/SOTA promotion while chemical/energy defects remain.
Memory changed: PROGRESS.md, DECISIONS.md, this task. New config schema/strict
loader/full reference+training/replay are proportionate verification; no repeated
unchanged tests or compile suite. Final index/path/whitespace checks below.

Final /task-end checks PASS: saved strategy matches train_moderate exactly after
removing runtime seed/run label; new seed policy/global nodes/index paths/strict
reference loader/text whitespace/git diff --check pass. GPU query empty; all
workers exit0. archive/chmoderate-final-checks.txt records verification.

### 2026-10-28 — Verified time-window cost in response to user question

Read-only experiment review, no retraining or scheme/default change. Original
10-wave T1.5 window runs42/2026 report281.2538/288.2304s; both actually complete
5000SOAP+500LBFGS TOTAL over30 windows, not that budget per window. Common-FEM
phase pooled3.90795%/3.86634%, terminal5.75378%/5.67185%; native pooled.72376%/
.67753%. First window80.53/85.23s, subsequent median6.92/7.03s; training timer
includes first JIT but excludes independent reference generation and setup before
start_clock. Rich global M5 R/t191s,17.985%/29.368% remains the lower-phase global
control; windows cost~1.5x in this comparison, not30x. Configs also differ in
architecture/loss/metric/reset etc, so evidence supports best tested combination,
not necessity or an isolated window-only causal advantage. Chemical errors and
window derivative/off-grid limitations remain as previously recorded. User asks
for explanation/cost, does not authorize switching back to windows or launching
jobs. All artifacts/defaults unchanged, workers idle, accuracy task OPEN; exact
next experiment remains above. /task-end: only task and PROGRESS memory updated;
verified archived summaries/config/timing code, git diff --check passes. No Git.

### 2026-10-28 — Selected route restored / switching and accumulated error

User authorizes time-window route again, overriding earlier one-global-model
restriction. README/DEVELOPMENT/artifact recommendation now identify the existing
30-log-window rich-case preset as selected; original simple default unchanged.
Read actual window_training.py/window_network.py/config plus both archived
summaries. Boundaries fixed before training: tau=.005, T=1.5, count30. First
[0,.001047681], second ends.002314890, tenth ends.028508797, twentieth ends
.219567895, last[1.239278557,1.5]. Fixed650+29*150SOAP,36+29*16LBFGS caps;
all30 L-BFGS statuses in both runs are iteration cap1, not convergence success.
No loss/L2/endpoint validation gate or adaptive window length. Nonfinite checks
are safety checks only. Reference diagnostics do not select/seed propagated states.
Initial=end[:,0] freezes the previous physical grid phase; increment hard IC and
mean projection preserve continuity/mass, mu follows the same exact operator law.
Parameter warm start, optimizer reset, no later joint correction; derivatives
can jump. Therefore fitting error is added to propagated earlier error; dynamics
can amplify it, hard mass does not protect interface/chemical accuracy. Native
terminal phase.809%/.748% already includes cumulative errors on testedT1.5, but
neither longer-time guarantees nor chemical resolution follow. Proposed future
acceptance may combine independent-time residual and extra-training endpoint
stability; not implemented. Keep chemical/energy/derivative diagnostics visible.

/task-end: documentation/metadata only; no formula, optimizer, reference, checkpoint
or Git change, no retraining. New directive/goal is explicit above. Sub-request
answered; all-field accuracy remains OPEN. Blocker none. Exact next resume: use
train_coarsening_windows as primary rich-case preset, load complete trajectory.json,
read existing chemical/derivative diagnostics. Further work should specify and
independently validate any adaptive acceptance criterion before changing runtime;
do not claim current budgets are such a criterion. Verify schema/index paths and
git diff --check only (no unchanged code suite). Memory changed: PROGRESS.md,
DECISIONS.md, this task. All workers idle.

Final verification PASS: unchanged window preset schema/5000+500 budget, updated
artifact index, both complete30-window direct-path checkpoint chains, modified
document whitespace and git diff --check. No numerical code changed/tests rerun.

### Global shared-head loss-balancing implementation and live resume

Added case-local loss_balancing.py with stopped inverse-L1-gradient weights/
explicit stagger masks, optional canonical train integration, frozen joint
refinement and stage/weight logs; three explicit config presets and validator.
Three focused numerical tests pass26.630s (two new tests plus existing dense
mixed residual/derivative test). Independent analytic toy Jacobians verify
refresh/state/stopped weighted updates; real mixed-PDE SGD verifies all stagger
stages and hard IC. Three configs/reference schemas pass; physical/architecture/
training settings identical except loss_balance; no strategy seed. README and
DEVELOPMENT describe exact algorithm/limits before all source snapshots.

Serial driver docs/tasks/archive/chbalance-run-controls.py active, session91886,
log /tmp/chbalance-runs.log, process state archive/chbalance-processes.json.
Fixed joint completed exit0: trainer265.60s, common phase55.8807%/62.6196%, native
mu67.7328%/277.327%; final L-BFGS loss.0509072, best phase checkpoint SOAP5000.
GradNorm running, stagger queued. No changes to example sources until all three
complete (matched source snapshots). CPU audits not yet run to avoid timing
contention. Next: finish serial controls, run chbalance-audit.py for three names,
then chbalance-provenance.py/chbalance-plots.py, inspect physics/common/native
fields and update README/index/results. Dedicated audit scripts preserve all
old studies. No FEM/new reference/Git work. Blocker none; accuracy OPEN.

Three-arm cohort finished exit0; independent CPU final/best replays and actual
frozen objectives pass (up to ~1e-14 field replay, scalar objectives ~1e-15).
All initial checkpoint bytes, full5000x80 times and57source files identical.
Inverse-gradient weights refresh200times with no norm-floor/ceiling or weight-cap
hits. Final fixed/GradNorm/stagger phase55.88/63.63/68.83%; native mu67.73/147.12/
68.14%, terminal277.3/476.0/348.7%. Stage counts exact2000/2000/1000 for stagger.
GradNorm normalized chemical loss falls .03038 ->.00002972 while evolution rises
.02053 ->.20032, demonstrating tradeoff rather than accurate dynamics.

Add ONE raw-residual/known-initial-scale guard control to separate previous
spatial metric from balancing: train_coarsening_mixed_gradnorm_raw, evolution100,
chemical.5, same global shared network/reference/seed/budget. Optional explicit
loss.evolution_scale implements raw normalization; legacy formula unchanged.
Independent raw PDE/parameter derivative and old mixed tests pass2tests18.054s.
Raw run20261028-chbalance-gradnorm-raw-01 active, log archive/chbalance-gradnorm-
raw-training.txt. This later control has a later optional-loss source snapshot;
do not extend first-three57-file parity claim to all four. train.py/network/
balancing helper unchanged. Next finish raw run, append independent CPU audit,
regenerate four-way plots, register final reports/index and retain failed results.

### Global mixed balancing / final task-end

All4 full original-complex-case5000SOAP+500LBFGS cold starts finish exit0.
Final common-FEM phase pooled/terminal:
- fixed joint55.8807%/62.6196%,265.60s;
- inverse-gradient balance63.6294%/71.5108%,293.13s;
- alternating losses68.8259%/67.2824%,282.42s;
- raw normalized gradient balance55.7275%/64.2271%,113.81s.
Native mu pooled/terminal67.73/277.33%,147.12/475.95%,68.14/348.65%,104.09/364.33%.
Best-phase checkpoints (5000/4500/5500/5500) also replay and do not rescue errors.
No current control beats retained phase-only global M5R/t17.985%/29.368% or
window~3.9%/5.7%. No stable ranking/universal global impossibility claim.

Independent CPU final/best physical replay max phi7.12e-15/mu1.89e-14; mass drift
<=3.61e-16. Actual independent SciPy A/lifted inverse/physical chemical law/time
JVP reconstruct all final weighted scalar objectives to~1e-15. Learned chemical
law defects are measured, not assumed zero. Dense151-time energy/phase ranges
and chemical errors remain poor. First-three initial bytes, all57sources and
5000x80time arrays identical; stage counts/200weight refreshes pass, first-three
bounds/caps inactive. Raw later source comparison verifies only README/losses/
problem plus its config differ; model/train/balancing/operators and initial
bytes/times identical. Raw phase weight cap3/200refreshes, no norm-bound hits.
All records remain direct paths/no hashes. Curves/energy/GradNorm fields visually
inspected; model retains incorrect small structures and insufficient coarsening.

Diff-level delivery: optional loss_balancing.py and canonical train integration,
explicit3cohort configs plus raw-normalized control; optional evolution_scale
and validation; two initial focused tests and a raw PDE derivative test; README/
DEVELOPMENT API/validation; new runs indexed; archive scripts/tests/raw configs/
metrics/balancing logs/provenance/plots/result report. No architecture parameter
or old checkpoint/template change, no PDE/reference/physical problem/default/
old path/Git change. Tests:3in26.630s then2in18.054s (raw change justifies repeating
affected mixed regression;4distinct numerical tests). No unchanged broad suite.

Exact resume: read archive/chbalance-results.md/analysis.json and source-delta
provenance. Requested trial complete; accuracy OPEN, blocker none, workers idle.
Existing selected window trajectory remains usable. Any further global study
must isolate representation/initial flux/temporal optimization rather than assume
loss weights alone are the remaining cause. No extra seed/step/tuning experiment
is queued or authorized as an optimizer investigation. Keep untested original
relative-rate GradNorm/frozen-field block methods distinct from what was run.
Final schema/index/source/whitespace checks below. Memory files changed:
PROGRESS.md, DECISIONS.md, this task. No Git authorization/action.

Final /task-end verification PASS: all31 training config schemas/runtime-only
seed policy; four-run index/checkpoint/report paths; current active Python
matches latest source snapshot; new-text whitespace/git diff --check; GPU query
empty. Both focused test rounds and all final/best replay/objective/provenance
records retained. Final four-control curves/energy/raw fields visually inspected.
archive/chbalance-final-checks.txt is the verification record. All workers exit0,
no jobs pending, no new Git action. Memory: PROGRESS, DECISIONS, this task.

### 2026-10-28 — Active default switch requested by user

User explicitly makes time windows default after four failed global mixed
controls. Set canonical train.py default to existing richer windows preset and
solver default to same config; leave train_base as legacy/foundation to preserve
all dependent presets. Resolve blank window log name in train.py via copied
runtime config, timestamp/nonlocal-windows; preserve explicit names and direct
window API. Sync project/example README, DEVELOPMENT, artifact default/legacy
metadata. No numerical changes/new training or Git. Acceptance: no-config CLI
resolves30windows/T1.5 and dispatches with generated name, explicit config/name
still honored, all schemas/reference/selected trajectory paths and diff pass.
Blocker none. Next implement minimal entry edits then focused dispatch checks.

### Default switch delivered / task-end

Minimal code diff: train.py config flag points to train_coarsening_windows and
fills blank window log name on a copied runtime config; solver.py default module
matches. No window engine/PDE/operator/network/budget/reference change. Project
README updated first, example README quick start/default formulation/legacy
commands and DEVELOPMENT exception synchronized. Artifact default_training now
points to rich window preset/reference/complete trajectory; old simple metadata
preserved in legacy_simple_training. All old paths/checkpoints unchanged.

Verification PASS: actual ml_collections/absl no-config resolution equals the
historical validated seed42 window strategy after removing runtime seed/name;
mocked window dispatch confirms automatic name/copy and explicit-name preservation
without starting training. Explicit legacy CLI/reference still work. Solver
parser AST selects same physical preset. All31 config schemas/seed policy,
strict default reference, all30selected model/state paths, index parity, modified
text whitespace and git diff --check pass. Archive/chwindow-default-check.py,
chwindow-default-check.txt, chwindow-legacy-check.txt and
chwindow-default-selection.json retain verification. No expensive unchanged suite,
FEM/reference regeneration or new PINN training. No Git authorization/action.

Current requested action COMPLETE. Broader chemical/global accuracy remains
unresolved but deferred by user's strategy choice; no worker or queued tuning.
Exact resume: default command is python examples/confined_ch2d/train.py; use
explicit configs for legacy/global experiments. Further training/chemical-error
or adaptive-window work should follow a new explicit user request; do not silently
resume discarded global sweeps. Keep known fixed-budget/propagated-error and
chemical/time-derivative limitations visible. Memory files: PROGRESS.md,
DECISIONS.md, this task. No blocker.
