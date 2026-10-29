# Matched local precipitate controls

Goal: answer the user's request for local results for the current elastic
precipitate pilot. Keep the model, constrained network, initialization, sampled
times, FP64/ordinary policy, optimizer and budget fixed. Compare existing local
spectral diffusion and a mixed local-AD diffusion control with the prior
nonlocal run; mechanics remains exact differentiable spectral equilibration.

Acceptance: complete cold starts through train.py, focused local derivative
checks, exact initializer/config comparison, CPU checkpoint replay, common
local-reference errors and timing/field/convergence comparison. No defaults,
references, SOAP-stability work or Git history changes.

Context: nonlocal run is 20261027-periodic-elastic-pilot-01 under the existing
precipitate2d run root; native pooled phi0.0387%, common-local0.5420%,183.1s.
Its full numerical run completed but a CLI return-value defect yielded exit1;
the entry is now fixed and independently smoke-tested. One run is not a
robustness or optimizer-stability study.

Status: complete (2026-10-27). Both local runs and independent replays exit0.
Blocker: none.
Next action: none. No default promotion or stability extension requested.

New runs under /root/autodl-tmp/tf-logs/precipitate2d/:
- 20261027-periodic-elastic-local-spectral-01: exits0, CPU direct replay passes;
  best/final5500 pooled phi0.00059888077, terminal0.00052866830, pooled
  u0.00027815897/stress0.00051564738/q0.0727329863. Trainer183.078s,
  first update30.062s, mean after100=19.950ms. Initial q still less accurate.
- 20261027-periodic-elastic-local-ad-01: exits0, CPU direct replay passes;
  best/final5500 pooled phi0.0006765729200/terminal0.0005895238399,
  u0.0002991952244/stress0.0006219640269/q0.07278502637. Trainer680.059s,
  first update63.501s, mean after100=91.525ms. L-BFGS141.489s,500 iterations,
  budget stop status1. SOAP5000 phi0.0010435 is an intermediate result.

Added explicit train_local_spectral.py and train_local_ad.py presets that change
only nl.method. AD computes point-network Hessian traces for both mixed fields;
time JVP still differentiates the full-grid phase projection. Existing nonlocal
and spectral residual arithmetic ordering is preserved. A band-limited
manufactured two-field check verifies residual parts and parameter gradients
against analytic Laplacians, and AD/spectral agreement; passes21.756s CPU.
No reference/default/optimizer/Git change. Previous broad tests remain valid
for unchanged components; no need to rerun them without a relevant failure.

Initial .eqx files have been compared directly and are byte-identical across
all three runs. Saved config differences are exactly nl.method and log.run_name.
Analysis script additionally checks network/mechanics/shared optimizer/runtime/
refinement sources and identical sampling assignments by AST; actual full GPU
sampling arrays were not logged, so do not claim an independent input trace.

## Common-target results and interpretation

Native best and final are identical update5500 for all three runs. Against one
refined local reference (phase, q, displacement, stress):

| Method | Pooled phi | Terminal phi | Pooled u | Pooled stress | Pooled q |
|---|---:|---:|---:|---:|---:|
| Nonlocal | 0.005419979174 | 0.005592545548 | 0.001892368439 | 0.004762385179 | 0.07425660343 |
| Local spectral | 0.000598880770 | 0.000528668297 | 0.000278158966 | 0.000515647381 | 0.07273298633 |
| Local AD | 0.000676572920 | 0.000589523840 | 0.000299195224 | 0.000621964027 | 0.07278502637 |

Nonlocal retains0.000387170526 native pooled phi, but the native/local reference
gap is0.005377960601. Thus its excellent native training error is compatible
with a larger common-local error, primarily finite-horizon bias at this delta.
No horizon optimization was performed. Local spectral/AD are close in accuracy
in this single run, while AD costs3.71x trainer time and4.64x compiled SOAP step
versus nonlocal. Spectral and nonlocal have essentially equal wall time here.
No multi-seed robustness/general accuracy dominance claim. All mechanical
fields are solved by identical spectral equilibration, not separate networks.

q initial relative error is0.29978775 spectral/0.29979010 AD; terminal q
0.00446740/0.00364186. Initial chemical inconsistency remains an explicit shared
limitation. Mass drift2.22e-16 for each run; sampled energies decrease. For AD
the audited nodal local energy is evaluated with Fourier derivatives, not an
asserted exact discrete energy law for the AD collocation residual.

## Validation and deliverables

Focused test passes: manufactured two-field analytic Laplacians verify both
mixed residuals and their parameter gradient, including the uniform phase
shift, and spectral/AD agreement. Existing nonlocal/spectral arithmetic ordering
is retained. Initial .eqx byte equality, config-only method/name differences,
network/mechanics/shared architecture/optimizer/runtime/refinement source parity
and random sampling assignment AST checks pass. Canonical reference unchanged.

Two full train.py cold starts exit0; both actual best checkpoints replay on CPU
and match saved fields/metrics. Independent acoustic matrix solves reproduce
saved u/stress; common-reference fields, interface-band errors, mass and sampled
energy checks pass. Both L-BFGS controls stop at the500-iteration budget,
not their convergence tolerance. Time includes JIT/evaluation/refinement;
historical nonlocal timing and occasional CPU analysis activity are not a
dedicated fully controlled microbenchmark.

Created configs/train_local_spectral.py and train_local_ad.py; added explicit
AD path and validation/schema support, documented the projected spatial Hessian
convention, updated README and DEVELOPMENT. Comparison analysis script/JSON,
per-run config/summary/best/replay/metric/refinement records, a common-target
convergence plot and3x5 local AD phase/reference/error figure are archived beside
this file. Artifact index registers both new runs; existing defaults preserved.
Plots inspected and local links/git diff --check pass. No GPU/FEM worker remains.

Changed memory: PROGRESS.md, DECISIONS.md, DEVELOPMENT.md and this archived task.
No Git history operation authorized or performed this turn. SOAP stays deferred.
