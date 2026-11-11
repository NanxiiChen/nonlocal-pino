# Current PINO nonlocal/local-AD comparison inventory

## Goal

Honor the user's decision to stop the thin-interface investigation. Audit
existing derivative comparisons and identify fair, feasible additions for the
current five main PINO cases, especially parameterized Neumann geometries.

## Acceptance

- Distinguish current PINO evidence, historical PINO recipes and fixed-case
  PINN studies; retain negative results and finite-cohort limitations.
- Check actual boundary conditions and derivative implementations before
  proposing a control. A local reference is not a local-AD neural baseline.
- Specify common physics, initialization, queries, evaluation and budgets;
  permit explicit local boundary enforcement and reasonable tuning.
- Prioritize runnable controls with accuracy, time, memory and boundary-cost
  measurements. Do not choose controls to force a nonlocal victory.

## Context and scope

The32arm extreme-interface study is complete and archived. No new output-scale
study, production-default change, Git operation or cleanup is requested here.
The current request asks which other cases lack comparisons and how useful
experiments can be added. This session inventories code/evidence and prepares
concrete experiments; long cold-training matrices are not yet specified.

## Completed inventory

The five principal PINOs have two historical operator-learning AD comparisons
(Electropolish and particle corrosion). Three lack a complete matching PINO
control (random-ellipsoid Melting, geometry Spinodal, and coupled channel flow).
None of the five has a complete AD pair for its currently selected recipe.
Heterogeneous material corrosion is an additional PINO with a useful matched
Adam control already present. Do not discard it because local AD is more accurate.

| Case | Existing evidence | Missing current control |
|---|---|---|
|Electropolish|Historical width64/Nt4/B8/Adam5k: NL/AD phase4.427/8.595%, terminal3.176/5.498%, same64local FD101 inputs|Width96/Nt16/rolling/Adam10k|
|Particle corrosion|Historical eight-particle width192/Adam20k: NL/AD phase1.9587/97.2665%, terminal3.4236/100%, same8local FEM geometries|Selected six-particle350s/width256/rolling/Adam20k|
|Spinodal|Old simple fixed-geometry PINN: NL/AD phase4.1935/5.6639%, terminal4.9998/3.8354%; harder fixed PINNs both about83%pooled|Current random-geometry width256/rolling/SOAP50k; current complex targets are native nonlocal only|
|Full channel NS--CH|GMLS versus moment-corrected integral studies, plus older spatial AD first derivatives|Full local spatial-AD gradients and Laplacians for current random-wall PINO|
|Random-ellipsoid Melting|Local FEM assessment of a nonlocal-trained PINO on one input; recent single-sphere PINN derivative studies|A trained local-AD ellipsoid PINO with common local-reference assessment|
|Additional heterogeneous-material corrosion|Matched Adam10k: NL/AD phase1.227/.872%, terminal2.086/1.516%, four local-FV inputs|Cross-seed confirmation if needed; not a missing initial control|

Historical timings cannot be transferred to the current GPU. Examples of actual
past benefits: prescribed-flow CH diffusion PINN was4.38x faster by trainer
walltime (3.778%NL versus3.364%AD phase); multiple-precipitate PINN was3.68x
faster (8.030%NL versus12.898%AD phase). The former retains spectral advection,
the latter spectral elasticity. Neither is the current full NS--CH PINO.

Evidence:

- [Electropolish paired operator study](electropolish-operator-local-comparison.md).
- [Eight-particle operator study](particle-corrosion-pino.md).
- [Accepted current particle study](particle-corrosion-physical-optimization.md).
- [Current Spinodal operator](confined-ch-width256-online-cold50k.md).
- [Channel integral versus GMLS](channel-integral-nonlocal-control.md).
- [Ellipsoid local-FEM check](melting-operator-local-fem.md).
- [Material corrosion pair](heterogeneous-corrosion-local-ad.md).
- [Fixed CH pilot documentation](../../../examples/confined_ch2d/README.md).
- [Prescribed-flow CH control](../../../examples/twophase2d/README.md).
- [Precipitate control](../../../examples/precipitate2d/README.md).

## Boundary interpretation and fair comparison

For a conditioned geometry, the encoder's domain/boundary tokens are fixed
inputs with respect to query-coordinate differentiation. Cache their encoding
within each loss evaluation; retain parameter gradients through the encoder.
Differentiate the query spatial attention, decoder and known coordinate-dependent
lifting. Share spatial derivatives/features across time where possible and use
directional JVPs rather than materializing dense query Jacobians.

Existing point streams already contain boundary coordinates, normals and physical
arc-length measures. Local strong-form controls can therefore add normal-
derivative losses on positive times. Corrosion constrains phase and q (chemical
concentration deviation), not total concentration alone. CH constrains phase
and chemical potential. Normalize using the same physical length/field scales;
keep reservoir Dirichlet conditions and IC/BC intersections consistent.

A regional visible-pair graph closes exchange across holes without an independent
wall penalty in its discrete nonlocal model. This is not exact pointwise local
Neumann satisfaction at finite horizon. The channel integral route explicitly
retains wall phase/mu and outlet phase/mu/traction losses; its potential benefit
is derivative cost, not removal of those conditions. Weak-form and properly
constructed hard-constraint local neural methods can also avoid soft Neumann
penalties. Claims must identify the actual strong-form AD baseline.

Retain the original initial-field family even if it is incompatible with
pointwise Neumann at t=0. Do not impose both contradictory constraints at the
initial instant or modify the physical problem to favor either method. Report
early positive-time and boundary-strip errors separately. No neural residual
labels or future interface trajectories enter training.

Primary literature checked as background:

- Sukumar and Srivastava, geometry-aware exact boundary constraints:
  https://arxiv.org/abs/2104.08426 .
- Goeschel, Goetschel and Ruprecht, strong/weak Neumann/Robin PINO constraints
  and piecewise-smooth boundary limitations: https://arxiv.org/abs/2510.24557 .
- You et al., finite-horizon Neumann consistency requires a boundary formulation:
  https://www.sandia.gov/research/publications/details/an-asymptotically-compatible-approach-for-neumann-type-boundary-condition-o-2020-07-01/ .

## Prioritized implementation and experiment plan

### 1. Current six-particle corrosion

Use explicit `configs/train_physical_spaced_w256.py`, not `train_default.py`.
The README deliberately documents that no-config selection is the separate5s
research case. No code/docs inconsistency or default promotion is inferred.

The existing `local_ad.py` already implements both PDE Hessians with batched
coordinate JVPs and two explicit, cLe-normalized Neumann losses. Independent
CPU inspection changed only method and the5->7loss-bias declaration and passed
`validate`, `validate_online` and `validate_strategies` for the selected recipe.
The original five physical losses and network are shared; added boundary losses
are an intentional complete-scheme difference.

The missing engineering is strict rolling spatial-method pairing: train.py
currently rejects online plus `--paired-run`, while `--sampling-control` allows
fixed-to-rolling changes of the same derivative method. Add an explicit paired
rolling mode and verify complete cold parameters/optimizer plus actual geometry
identities, spatial revisions, core IDs, IC/BC draws and all physical times.
No hidden fallback, input-family change or use of old collapsed models.

First validate derivatives/boundary normals and probe full optimizer updates on
the current GPU. Then propose two paired cold seeds at20k for each method.
The existing four development local-FEM geometries can monitor convergence;
use a separate predeclared reserved cohort (e.g.8newgeometries) for a stronger
final claim, with reference refinement and native/local discrepancy checks.
If AD again fails, diagnose BC versus interior residuals and give it a bounded
validation-only tuning budget before treating it as a competitive baseline.

### 2. Current Electropolish

Reuse width96/Nt16/B8/Adam10k and the original non-Neumann-compatible initial
interface distribution. Bridge `spatial_sampling.validate_spatial` (currently
rejects local_ad) so the local arm consumes the same rolling core IDs without
halo evaluation. Preserve exact initial-interface, time and IC/Dirichlet draws.
The existing AD kernel already uses independent-query directional second JVPs.

Propose two paired seeds, four cold10k runs. Reuse the common64local FD101
cohort and existing four FD401 checks. These inputs have been reused for
configuration development; label that fact and add a fresh cohort before a
population-generalization claim. Tune local Neumann sampling/weighting on
separate validation data if necessary. Compare all-field, concentration,
early side-boundary and interface errors, not only global phase L2.

### 3. Spinodal geometry operator

This is a promising geometry/closed-wall study but requires more setup.
Current hard-IC/mass model predicts only phase; eliminating mu gives local AD
fourth spatial derivatives, and mu-normal boundary losses third derivatives.
Direct scalar AD is an honest same-representation control, but it should not
be the only local baseline if a mixed formulation is more competitive.

Implement a matched mixed pair with phase/mu heads for BOTH methods, explicit
field scales, two PDE losses, identical balancing/optimizer, and local wall
conditions. Preserve the existing single-head nonlocal SOTA as a separate
production-recipe reference. Do not attribute representation changes to the
spatial derivative. The graph resolvent is currently part of the objective;
if reused for both methods, call it a common graph-based loss preconditioner,
not a completely nonlocal-free AD baseline. Alternatively qualify the same
new mixed residual normalization on both arms.

The spatial derivative of the mass projection must hold the original geometry
quadrature fixed, as current `query_phase` does for off-grid values. Do not move
all quadrature anchors along with a query. Preserve hard-IC/time differentiation.
First qualify local FEM on a few representative10mode/T3geometries and quantify
the native/local gap. The current4.833%native score cannot rank an AD model
solving the local PDE. Only then specify a full training matrix and its budget.

### 4. Full channel, then ellipsoid Melting

Channel has full AD first-derivative helpers but still always forms its
Laplacian from neighbor weights. Add an explicit spatial-AD Laplacian path;
removing a derivative selector alone would leave a hybrid control. Retaining
the current streamfunction makes velocity Laplacians third derivatives of the
streamfunction. Retain the same hard lift, physical time JVP, spatial queries,
conservative-form transport and all nine original loss groups. First run a
small complete-gradient/memory probe and a phase-Laplacian-only diagnostic
control if useful, clearly distinguishing it from full AD. Do not immediately
launch full20k on the old96GB-specific execution preset.

Melting's current ellipsoid SDF is precomputed by fixed-iteration bisection,
not intended as an AD geometry primitive. An independent FP64CPU sphere check
at center(.5,.5,.5),axes(.3,.3,.3),query(.65,.62,.61) finds correct distance
.0786405638 but naive AD gradient(-.149185,-.119348,-.109402) instead of
(-.677631,-.542105,-.496929), and Laplacian-4.98071 instead of-9.03508.
The existing value-only nonlocal PINO is unaffected. A future local control
must first qualify analytic/implicit signed-distance gradient/Hessian rules
and include their contribution to the hard initial latent field. Do not simply
differentiate the bisection algorithm. Broader local references are also
needed; the single-sphere surrogate cannot evaluate rotated ellipsoid inputs.

## Measurements and acceptance for future runs

- Shared physics, initial fields, architecture (within each declared pair),
  optimizer, actual samples and final-budget evaluation; at least two paired
  runtime seeds initially, retained as provenance rather than preset fields.
- Same independent local reference for headline errors. Native references are
  supplementary model/discretization diagnostics, never a substitute local-AD
  target. Preserve unsuccessful runs and early/interface/boundary metrics.
- Plot error versus optimizer steps AND cumulative wall time, reporting time
  to a common accuracy threshold when reached. Separate compilation, warm
  updates, evaluation, rolling-input preparation and active memory.
- Current device inspected: RTX4090,49140MiB, idle; about20GiB disk free. Old
  RTX3090/5090/6000D/PRO6000 timings are historical only. Probe actual memory and
  use exact gradient accumulation if needed, with the same effective batch.
- A geometry/point trace can be shared for pairing, but AD should not evaluate
  halo points or be charged for unused nonlocal-operator construction when
  reporting an optimized end-to-end local pipeline. Report shared-harness
  overhead separately from method-specific preprocessing.
- Secondary BC study: vary local boundary quadrature density and normalized
  penalty weights on validation inputs, then lock the recipe before reserved
  evaluation. Report whether accuracy is sensitive to these choices; avoid
  selecting only a poor local weight. Keep comparable tuning budgets explicit.

## Completion and next action

Inventory and experiment design complete. No new training, FEM solve or
production behavior change was made. The recommended next implementation is
strict current six-particle rolling NL/AD pairing, followed by its focused
checks/resource probe; current Electropolish is the second inexpensive control.
Spinodal needs a qualified mixed formulation and common local targets first.
The user's thin-interface direction remains closed.

## Session-end record

Changed memory: this completed task, PROGRESS and DECISIONS. Read-only code/
artifact audits were delegated by case; CPU validators and the exact-sphere
SDF derivative diagnostic completed. Relevant historical tables/reference
qualification and actual derivative paths were checked. Local links and
`git diff --check` pass. No Git operations, data cleanup, GPU training, new
reference solve or default promotion. This is a plan, not completed new
accuracy or performance evidence.
