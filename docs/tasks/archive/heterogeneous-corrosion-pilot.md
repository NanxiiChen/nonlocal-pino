# Activation-controlled heterogeneous corrosion PINO

Date: 2026-11-05

## Goal and acceptance criteria

User authorizes trying the heterogeneous pitting operator and explicitly names
the first regime activation-controlled (活化控制). Implement a separate example
with fixed initial pit/outer boundary and positive finite-contrast L(x). Qualify
reaction versus transport sensitivity with direct numerical solutions, then
complete a cold physics-only PINO pilot, report unseen-map field/contour errors
and representative evolution. A working, measured pilot is required; no accuracy
target or SOTA promotion was requested.

## Context and scope

[Paper feasibility](heterogeneous-pitting-candidate.md), Mai et al2016
Figure9 kinetic heterogeneity, not Figure8 impermeable particles. Existing
KKS AC/CH free energy and electro patch encoder provide the base. All AC terms
scale pointwise with L(x); constant transport keeps the spatial graph common.
Known material maps and IC/BC drive training; evolved reference fields are
evaluation only. Keep all current examples/defaults and historical artifacts.
No commit, merge, push, deletion or history operation is authorized this turn.

## Plan

1. Add self-contained example/config and a declared smooth positive material
   prior, fixed pit and paper-inspired opening/zero-flux boundary setup.
2. Qualify independent local reference and nonlocal baseline, material effects,
   time/spatial sensitivity and activation control before claiming that regime.
3. Adapt existing conditional network with shared space/time preparation,
   nonlocal spatial derivatives and time JVP. Train cold on fresh material maps.
4. Evaluate held-out conditions, plot maps/fields/contours and convergence,
   document exact commands/results and leave a recoverable run record.

## State, blocker and next action

Complete: the first mild cold5000 pilot and user-requested stronger material
prior are implemented, independently qualified and trained. The stronger
width64/5000 control underfit; width96/10000 gives accurate material-dependent
shapes. Every run and reference is preserved. GPU JAX checks passed on RTX6000D
(85651MiB). No FEniCS jobs are running. About31GB disk space remained before
launch; no cleanup was performed. No current blocker.

## Implemented configuration and checks

`examples/heterogeneous_corrosion2d/train.py` is the canonical entry. Fixed
1x0.6 rectangle, pit radius0.12, opening half-width0.055, interface tanh k30,
dimensionless time0..1. Smooth13-coefficient material prior gives kappa1..3.
KKS coefficients nCH80, nAC1=24.31818, nAC2=4, nAC3=.00222222; whole AC drive
is multiplied by kappa. Original residuals, bounded phase output, stable KKS
liquid coordinate, soft IC/mouth BC, five inverse-gradient-normalized losses.
Width64 patch/cross-attention encoder-decoder, B8, Nt8 fresh random times and
materials each step, 81x49 grid, core256/fullhalo, fixed32 spatial batches.
FP32 ordinary compilation; Adam1e-3 decays by0.5 per1000 updates.

Nine focused tests passed (first8 together, ninth input validation separately):
independent energy/Jacobian/operator/flux checks, material maps, time JVP,
physical loss oracle and parameter gradients. No evolved labels enter the loss.

Local direct-solver D80->160 changes area advance0.7619% at kappa1 and3.3238%
at kappa3. Raising kappa1->3 changes area advance .0332555->.1477140 and depth
.1875->.325; qualifies activation-dominated rather than zero diffusion resistance.

Four predeclared held-out maps: native/nonlocal versus independent local FV
pooled phase error0.2723/0.4801/0.5458/0.3645%; terminal0.5037/0.9403/1.0993/0.6785%.
Local161x97 refinement on condition0 changes pooled phase0.11254%, terminal0.20490%;
tighter time tolerance changes pooled phase4.70e-9 (fraction). Neither is a FEM
claim or a full population/convergence study. Native graph edge weights were
checked bitwise identical on CPU and GPU.

## Direct artifact paths and exact resume

- Qualified evaluation bank: `/root/autodl-tmp/data/heterogeneous_corrosion2d/20261105-activation-pilot-02`.
  Includes native/local references, qualification JSON and reference evolution.
  Earlier `activation-pilot-01` is superseded because FP32 host coordinates
  changed IC by6e-7; rebuilt before training with shared FP64 host coordinates.
- Successful 50-step probe: `/root/autodl-tmp/tf-logs/heterogeneous_corrosion2d/20261105-activation-probe50-01`.
  Compile144.25s, warm update13.27ms, peak actual allocation2.649GiB. XLA reserved
  65% allocator pool; reservation is not actual model memory.
- Fresh5000 run: `/root/autodl-tmp/tf-logs/heterogeneous_corrosion2d/20261105-activation-cold5000-01`.
  Completed exit0; independent runtime seed, no probe continuation.

## Completed mild pilot

Mean native phase0.7950%,terminal0.8592%,concentration0.9599%, terminalcorroded1.9003%.
Independent localFV same predictions: phase1.0099%,terminal1.3977%,concentration1.2141%,
terminalcorroded3.0047%. Runwall236.85s, firstupdate139.64s compile, warm11.54ms/update
+2.02ms material sampling, peakactual2.649GiB. Network146050learnable scalars
+64fixed spatial Fourier coefficients; first audit's146114count included both.

Condition-centered phase error10.00%pooled/6.18%terminal, response amplitude97.60/99.15%
ofreference: materials are learned, not a common prediction. Closest material pair
has31.9%pooled/13.4%terminal difference error despite low totalfield errors.
Full-grid residual audit: positive-saved-time AC/CH meanabs .00750/.1730;
unseen-coreCH .2132 vsseen .1678. NaturalboundaryCH .5074; unseenboundary1.1759.
t0CH meanabs14.28. Thus early transport/near-opening flux remain imperfect.
Source backend-loader validation was tightened after run; a focused new test
passed .013s. Ten tests total across the two validation phases. Model numerics
unchanged. Full-grid audit and condition sensitivity are separate run subfolders.

## Stronger-heterogeneity extension requested by user

Screening complete: choose explicit `configs/train_stronger.py`, kappa0.5..4,
13 Gaussian coefficient scales(0,1.3x12). The latent constant mode is fixed0;
physical kappa mean is not strictly fixed after sigmoid. Baseline scalar
sampling is bitwise unchanged. Prior-validation regression +2relatedtests
passed1.932s (11 unique tests total now). No pit-aware forcing is added.

Same4standardized spatial draws: mean absolute dissolution-increment centroid
shift .0120->.0341; mirror asymmetry5.47%->15.15%. Two clear samples change
-.0266->-.0666 and+.0193->+.0647. Stronger finalareas .1145–.1269. Actualmaps
D80->160 areaadvance changes1.86–2.27%, lateralcentroid atmost .00125;
uniform kappa4 control has5.36% area sensitivity. Activation-dominated, not a
pure infinite-diffusion limit. Max5candidate rejected for near-sidewall approach.

Direct screening: `/root/autodl-tmp/data/heterogeneous_corrosion2d/20261105-stronger-screen-01`.
Qualified4mapnative/localbank: `/root/autodl-tmp/data/heterogeneous_corrosion2d/20261105-activation-stronger-reference-01`.
Pooled native/localphi .40–.64%; firstmaplocalrefinement .212%pooled/.404%terminal.
Qualify CLI now accepts an explicit config module; activation CLI uses that
config's material bounds (mild1/3 preserved).

Freshstronger5000 run: `/root/autodl-tmp/tf-logs/heterogeneous_corrosion2d/20261105-activation-stronger-cold5000-01`.
Completed exit0. Same network/trainingbudget as firstpilot; native meanphi5.49%,
terminal9.84%,local5.80%/10.43%. Material-centered response errors63.9%/67.5%
and fourthsample spuriousrightbias show underfitting; preserve this control.

Current run: `/root/autodl-tmp/tf-logs/heterogeneous_corrosion2d/20261105-activation-stronger-w96-cold10000-01`.
Completed exit0. Explicit `configs/train_stronger_wide.py`: width96,
spatialfeatures48, cold10000, LR decay every2000; same physics/prior/sampling/
losses/Adam. 20% allocator reservation. This changes both capacity and budget,
so their individual contributions cannot be isolated.
Final native meanphase0.86491%,terminal1.30622%,concentration1.04442%; localFV
phase1.22692%,terminal2.08613%,concentration1.48668%,terminalconcentration2.55096%.
Use fixed10000endpoint, not the lower9750 scalar checkpoint. Four cases only.
Material-centered errors9.457%/8.313%, responseamplitude98.48/98.51%; wrong-input
permutations strongly degrade accuracy. Actual reference/PINO mirrorasymmetry
15.0828%/14.9632%; same-area11.0794/11.1543%. Fourthcase referencecentroid+.00203,
PINO+.00354, vsunderfit5k+.03767. Direct reference families show shapechange
even atcommonarea (mild4.06%->stronger11.08%mirrorasymmetry).

Runwall380.26s, firstupdate176.00s compile, warm14.60ms, peakactual3.931GiB.
323522learnable scalars+96fixedFourier coefficients. Positive-time full-grid
AC/CH meanabs .00712/.08060; naturalboundaryCH .22252; unseenboundary .94782;
t0CH13.96. Keep the early transport/flux limitation explicit, and note only
3501/3960free nodes become cores in the fixedpool. Fullfield errors include all.

## Final artifacts, changes and verification

Strongwide run `diagnostics/stronger_comparison.json` records native/local,
centeredresponse and common-area shape checks; `capacity_budget_comparison.json`
verifies identical target/condition/query arrays across stronger5k/10k runs.
`full_grid_audit/report.json` and `full_grid_residuals.npz` contain physical
constraint/residual checks. `comparison.png`, `fields.png`,
`convergence_comparison.png` and their plotting scripts are external artifacts.
All images inspected; no checkpoints/rawfields/images were imported into Git.

New self-contained example: problem/configs/solver/qualify/network/losses/train/
evaluate/audit; relevant tests, README overview and explicit artifactindex.
Post-run source changes only tighten referencebackend validation, add diagnostic
CLI/options and newmaterialprior presets. No existing example orsharednumerical
module changed. Eleven unique focused tests passed incrementally; the three
prior-related tests passed after the final sampler change. Production cold runs,
independent references and full-grid audits completed; AST, artifact existence,
README local links and whitespace checked at task end. No Git history action.

No running/queued job or blocker remains. If continued: preserve the qualified
stronger material prior and10000preset, then assess additional unseen materials
or improve early-time transport/boundary-core coverage before larger contrasts.
