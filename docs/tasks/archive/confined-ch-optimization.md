# Diagnose and improve richer confined CH PINNs

Goal: diagnose and substantially improve the failed ten-wave T1.5 PINNs without
changing physical IC/geometry/parameters, native operator or validated reference.
Use general representation/conditioning/temporal-training changes, not target
trajectory priors, seed tuning or reference labels in physics training.

Status: complete. Material phase improvement confirmed by two independent full
cold starts; complete CPU trajectories, parity and physical audits pass.
Chemical-potential accuracy remains an explicit research limitation, not solved
by phase L2. No blocker, active trainer or Git operation. Resume only on a new
request: prioritize high-frequency/chemical error and derivative consistency
before further phase-only tuning; do not rerun FEM or reopen SOAP side work.

Acceptance: direct evidence for key failure mechanisms with alternatives stated;
material common-FEM and native L2 improvement vs prior82.96%/94.55% nonlocal
pooled/terminal; actual fields/chemical potential/mass/energy checked; initial
compatibility and saved-model replay tested; canonical train.py options explicit
and documented. Retain failed controls and original defaults until evidence.

Context: archive/confined-ch-coarsening-training.md. Paired seed42 SOAP5000+
LBFGS500 failed (local AD also82.95%/99.41%). Same19 sources/initial leaves,
strict CPU replays pass. Both best models were untrained step0. Initial chemical
loss52.66%/39.46% of fixed17 loss; positive mu L2>119%, so not exclusively t0.
Pure fractions11%/8% vs46% reference; energy sometimes increases. Native/local
phase gap3.779% is insufficient to explain own-native83.02% error. Fields are
neither uniformly collapsed nor frozen. Capacity/lifting/optimization not
isolated. Deferred SOAP implementation work remains outside this task.

Current network phi=phi0+(t/T)*raw_phi, learned independent mu, hard weighted
mass. Spatial features have3 axial modes while physical IC uses up to5 diagonal
modes plus tanh harmonics. Uniform stratified8 times plus exactlyt0 are used.
Physical epsilon.02,T1.5,M.02,beta1; geometry.02/3682 centroids, delta.03.
Reference /root/autodl-tmp/data/confined_ch2d/coarsening_multimode10/reference.npz
already passes fixed gates. Do not rerun FEM or change it to ease fitting.

Potential representation change is phi=g(t)*phi0+(1-g(t))*raw with
g=exp(-t/tau): exact IC and arbitrary free fields for t>0, no encoded future
interface. It is a coordinate/lifting choice, not an assumption of physical
exponential decay. Must check IC/mass/JVP/AD derivatives and old saved templates.
PDE row weights are optimization metrics, not changed equations. Any temporal
sampling-density change must declare its measure/objective explicitly.

Probe archived chopt-probe.py/json: initial PDE time scale.006755, uniform
stratum width.1875. Final PDE gradient norms comparable (.232/.130) with
cosine-.826; positive-only cosine-.737. Additive raw target RMS6.22 at.075,
blend tau.15 target.797. Legacy final CPU replay passes1.53e-14 after new static
metadata. Blend-only PDE2000+100 pilot failed (91.53%/99.26%).

Representation-only supervised diagnostics (never PINN results): baseline
additive3000 SOAP fits native phi4.236%/terminal3.758%, mu3.992%; blend tau.15
fits phi2.617%/terminal2.382%, mu3.067%. Same64x4/modes3 architecture demonstrates
much better representability than failed physics fits; it does not prove a
capacity floor or unique cause. Outputs under20261027-chopt-representation-*
must be registered as supervised_diagnostic_only, never selected as SOTA.

Implemented optional log-time residual measure rho(t) proportional1/(tau+t)
and geometric expanding horizon, with fixed Nt and recorded actual times.
No inverse-density compensation: this deliberately changes the loss metric,
not the PDE roots. New c.sample.temporal fields explicit; historical sampler
expression/RNG preserved. Hard IC retained; causal presets omit the t0 atom
while resolving arbitrarily near-zero positive times. L-BFGS accepts explicit
ordered time_nodes to retain the same early-time focus; historical settings
retain exact original midpoint expression. Test both node-plumbing branches.

Current pilot: additive + log time tau.005, H0.03 ->T1.5 geometrically over1800
of3000 SOAP steps,100 L-BFGS on16 log-midpoint nodes. Session to monitor via
/tmp/chopt-causal-additive.log; then analogous blend tau.05 control. No field
labels in either PDE pilot. Blend IC/derivative/mass/compiled-gradient2 tests
pass21.65s. Temporal/explicit-node tests running. Need legacy checkpoint checks
and broad-enough affected refinement check; no unchanged full suite rerun.

Completed global PDE controls (all seed42, no reference labels):
- blend0152000+100: phi91.53%/terminal99.26%.
- additive/log sampling/expanding horizon3000+100:52.98%/63.95%.
- log-time additive coordinate3000+100:54.92%/62.25%.
- same plus frozen biharmonic metric3000+100:51.32%/61.45%.
- exact chemical elimination plus above3000+100: final metrics in run
  20261027-chopt-eliminated-01 (about47% before refinement).
- eliminated/log-input exponential blend tau.01/modes8:44.79%/55.51%.
  Early native phi at.075 improves to3.73% at step1500, then4.45% at final;
  this is evidence for partial early fit/forgetting, not proof of unique cause.

New optional single-field mode derives mu from projected phi, has one output
head and only one PDE loss. predict_fields requires explicit D in this mode;
old templates retain learned2-head arithmetic. Exact chemical relation/native
ODE and finite coupled gradients tested independently (12.98s). Native mu is
grid-defined; evaluate reports only meaningful phi wall normals. General
logarithmic additive/blend maps have independent time chain-rule tests. Temporal
CDF/RNG/horizon/node-plumbing2 checks pass. Legacy shared L-BFGS manufactured
PDE test passes, no repeat of unchanged expensive suite.

Current full-budget window run:
/root/autodl-tmp/tf-logs/confined_ch2d/20261027-chopt-windows-01,
log /tmp/chopt-windows.log, canonical train.py with train_coarsening_windows.py.
30 log-spaced windows, widths.001048 ->.26072; first SOAP650 then29x150=5000,
first L-BFGS36 then29x16=500 cap (maxfun72+29x32=1000). Weights warm-start,
optimizer including LR schedule resets per window; no reference-based endpoint
selection. Raw NN learns increments phi=phi_start+(t_local/width)*N; hard mass,
native mu eliminated; residual width*(phi_t+M*A*mu). Width-matched resolvents
are precomputed per window and passed as dynamic same-shape inputs. Fixed
shapes avoid recompiling for states/window factors; both SOAP and vectorized
Scipy objective are compiled once for their shapes. PDE targets only.

Window/metric3 focused tests pass13.40s: independent ODE residual/gradients,
time chain rule, two saved windows/manifest replay/state continuity and weighted
resolvent. New trajectory.json lists all window checkpoints and initial states
by direct relative paths, no hashes. A last-window checkpoint alone is not the
whole trajectory. CPU replay accepts only complete manifests and checks every
state link, phi/mu continuity, derivative jumps, full fields/mass/energy. Since
initial states are fixed grid arrays, no off-grid pointwise wall extension is
claimed. Local AD windows explicitly unsupported. Need full run/replay before
recommending; early prefix error is over a growing time set, not full-domain
convergence and must be labeled accordingly.

First30-window full run exits0: exactly5000 SOAP+500 L-BFGS,281.25s;
common-FEM phi pooled3.90795%/terminal5.75378%, native.72376%/.80887%.
Full CPU trajectory replay passes (phi6.94e-16,mu4.21e-15); state links phi
2.51e-15/mu1.22e-14, mass and monotone sampled energy pass. Native mu pooled
31.007%/terminal102.106% remains weak despite exact chemical relation, so
curvature/transport accuracy is an explicit limitation, not all-field success.

Frozen second seed2026 run exits0,288.23s: common phi3.86634%/5.67185%, native
mu29.986%. All outcomes retained; no seed tuning. Source should match first
run exactly except serialized runtime seed/name; verify before README edits.
Second CPU replay running via /tmp/chopt-windows-2026-replay.log.

Final allocation control running: 20261027-chopt-windows15-01, log
/tmp/chopt-windows15.log. Same5000+500 budget but15 windows, first800+14x300 SOAP,
52+14x32 LBFGS; maxfun104+14x64=1000. No source/physics/hyperparameter change
other than count and derived per-window budget. Test whether more optimization
per slab improves native mu without sacrificing phase; retain30-window baseline
regardless. No unrequested return to SOAP work or FEM.


## Final results and exact resume point

The15-window allocation control completes5000+500 in276.71s: common phase
4.008% pooled/5.936% terminal, native mu31.349% pooled/112.755% terminal.
Its complete CPU replay passes phi1.14e-15/mu5.61e-15, mass/chemical links and
sampled monotone energy. Keep30 windows: fewer windows do not improve mu.
Second30-window CPU replay also passes phi6.11e-16/mu3.99e-15. All31 saved
sources match between30-window seeds; configs differ only train.seed/log.name
(actual serialized field log.run_name). No reference values enter PDE training.

Final common-FEM phase results at T1.5:

| Control | Pooled | Terminal | Trainer seconds |
|---|---:|---:|---:|
| Original global | 82.960% | 94.545% | 174.64875842258334 |
| Exponential lift only | 91.528% | 99.264% | 96.54583871737123 |
| Early-time sampling/curriculum | 52.984% | 63.950% | 110.1149833239615 |
| Log input coordinate | 54.916% | 62.253% | 111.70429543405771 |
| Global resolvent | 51.315% | 61.453% | 160.23340655490756 |
| Exact chemical elimination | 47.243% | 57.083% | 147.53914684802294 |
| Elimination + blend + modes8 | 44.793% | 55.515% | 163.11226085945964 |
| 30 windows,42 | 3.908% | 5.754% | 281.25379383191466 |
| 30 windows,2026 | 3.866% | 5.672% | 288.2303547039628 |
| 15 windows,42 | 4.008% | 5.936% | 276.71324580162764 |

Global pilot budgets are not all equal: inspect each saved config/summary;
only the original and full-window runs have5000+500. Table records outcomes,
not isolated component effects. Supervised probes are deliberately excluded.
Native mu pooled31.007%/29.986% includes exactly correctt0; positive-time errors
52.975%/51.231% and terminal102.106%/97.865% reveal the unresolved issue.
Seed42 terminal mu error RMS0.04474 vs reference0.04381. Hard chemical relation
alone cannot prevent high-frequency phase errors amplifying through Aphi.
A resolvent weakens high-frequency residual penalties; this is a candidate
tradeoff, not an independently isolated explanation. No claim of global
conditioning optimality. Phase/mu joins are continuous; physical phase time
RMS derivative jumps range0.050–8.93 (42),0.046–9.44 (2026). State continuity
avoids delta-function jumps in this first-order evolution, but pointwise PDE
accuracy at joins is not established. No off-grid classical wall extension.

Diff-level delivery: optional phase lifts/log sampling/explicit refinement nodes,
frozen physical-volume resolvent, nonlocal exact chemical elimination, sequential
training and strict whole-trajectory evaluator; focused independent tests for
chain rules, actual residuals/parameter gradients, hard constraints and replay.
Legacy saved simple nonlocal/local AD templates replay; unchanged shared-helper
manufactured refinement passes. Final existing mixed local/nonlocal coupled
parameter-gradient regression passes1test39.67s. Earlier focused checks were
not rerun unnecessarily. Raw test logs, global outcomes, configs, summaries,
trajectory manifests and independent audit archived under chopt-*.

Current recommendation is configs/train_coarsening_windows.py; canonical
train.py, full checkpoints20261027-chopt-windows-01 and
20261027-chopt-windows-2026-01/trajectory.json. A final-window model alone is
invalid for the full trajectory. README documents formula, budget, replay and
limitations; DEVELOPMENT documents optional schemas and compatibility. Artifact
index registers all11 new directories, with supervised probes explicitly marked.
Figures chopt-before-after.png, chopt-optimized-fields.png, chopt-time-errors.png,
chopt-accuracy-energy.png; chopt-analysis.json has all exact metrics/replays.
Postprocessing chopt-plots.py; diagnostic/replay script chopt-audit.py.

/task-end: completed task archived; PROGRESS active task cleared and outcome
updated; DECISIONS records choices/failed alternatives; no histories changed.
Memory changed: docs/PROGRESS.md, docs/DECISIONS.md and this archived task.
All training/FEM workers idle. Future work on mu or matched local AD windows
requires a new task, not silently extending this completed phase diagnosis.

Final verification: all9 confined training config schemas pass; JSON index, all registered run/trajectory paths and new README/dashboard links exist. git diff --check passes. Optimized reference/prediction/error and energy figures visually inspected. No unchanged expensive tests repeated.
