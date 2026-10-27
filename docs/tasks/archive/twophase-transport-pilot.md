# Prescribed-flow phase-field pilot — 2026-10-26

## Goal, acceptance and completed status

User authorizes a simple first phase-field/nonlocal case to exercise the route,
explicitly requesting commit/push of previous results first, then dev development.
Completed canonical entry, exact documented problem, independent refined
references, paired5000-update local/nonlocal GPU cold starts, CPU saved-field
reload/plots and focused numerical validation. No blocker, worker or required
remaining action. Full NS/pressure/density/gravity coupling is outside this pilot.

## Git checkpoint before development

Before any new-case changes, previous172files were committed on dev as
`05780d4ff37371b0fb0c37f8b9e36ec5a29e5e60`. Regular main merge
`21fe6d9af9b60aa7cb463ec3c77c899b455400b3` was pushed successfully to
`origin/main`, then switched back to clean dev. No dev push or squash.
Staged syntax48Python/65JSON checks and314registered directories passed.
Generated SVG/HLO/code EOF whitespace was normalized for staged diff-check;
no numerical change or expensive unchanged-suite rerun was needed for that.
The new pilot remains **uncommitted on dev**; it was not added to the published
snapshot. This matches the user's request to save old results before development.

## Diff-level changes

- `examples/twophase2d/{train.py,solver.py,evaluate.py}`: canonical cold-start
  mixed PINN entry, independent CPU reference, direct-path CPU audit/plots.
- `configs/train_base.py`: every pilot field explicit, runtime-only seed,
  ordinary FP32/SOAP/lr1e-3/5000updates.
- `problem.py`: smooth periodic initial data, physical/reference identity and
  finite shape/time/point-order validation. The final point-order guard was
  added during the cohort; it changes validation only, not numerical training.
- `network.py`: periodic integer features, mixed(phi,q) output, prescribed IC
  hard constraint only, no learned field clipping or prescribed future motion.
- `operators.py`: case-local periodic finite-horizon cubic negative Laplacian,
  explicit discrete second-moment correction; no historical shared operator or
  established default changed.
- `losses.py`: same full-grid/time-JVP original mixed residuals for both methods;
  AD first spatial gradient retained in nonlocal mode (hybrid), local control
  uses Hessians rather than fourth-order AD. Mass penalty explicit1/1/1 weights.
- `tests/test_twophase_transport.py`:5focused numerical/reference checks.
- Root README updated first; example README gives formulation/usage/reference
  conventions, AGENTS example navigation and DEVELOPMENT case-layout description
  updated without changing policy. Artifact index registers sixth pilot case,
  reference and all3run directories. Prior cases unchanged.

## Exact problem and formulation

Periodic[0,1)^2,time0–1,dimensionless,constant prescribed velocity(.2,.1),
M=.002,epsilon=.05,quartic potential(phi^2-1)^2/4. Initial
s=sin^2(pi(x-.35))+sin^2(pi(y-.5)),phi0=tanh((s-.3)/.12),a smooth periodic
droplet-like shape. These are explicitly chosen pilot data, not a silent repair
of the Qiu2022 attachment. No wall/flux ambiguity or variable-density model is
introduced. This pilot is not an exact-circle or full-CHNS benchmark.

Local model: phi_t+v.grad(phi)=M Delta(q),q=phi^3-phi-epsilon^2 Delta(phi).
The nonlocal control replaces negative Laplacians with L_delta,while advection
uses AD gradients. Thus r_phi=phi_t+v.grad(phi)+M L_delta(q),
r_q=q-phi^3+phi-epsilon^2 L_delta(phi). Both residual signs are independently
checked against Fourier-mode derivatives. Original residuals, no inverse-Jacobian
weighting. Pure prescribed phase IC is embedded as phi=phi0+t*N_phi,with q
independently output; no advected-coordinate network input or reference targets.

Grid32x32,endpoints excluded,delta=.08; positive reciprocal offsets strictly
inside horizon. Cubic weights120h^2/(pi delta^4)(1-r/delta)^2 are multiplied by
2/sum(w*z_x^2). Isotropy gives both diagonal second moments2 and mixed moment0,
so Taylor expansion yields negative Laplacian consistency. Periodic differences
annihilate constants and conserve the discrete zero mode,with symmetric positive
quadratic form. This explicit normalization is a case-local derivative choice,
not a modification of established examples' stored graph coefficients.

Network3x48GELU,integer periodic modes1–3,temporal embedding8(scale1),twooutputs;
full1024points x4shared freshly sampled times/update. SOAP b1/b2=.9/.99,
refresh5,clip10,initiallr.001 continuous.95/1000decay,floor1e-5,5000updates.
Ordinary FP32,runseed42 provenance only. PDE/chemical/mass MSE weights1/1/1.
No causal/stagger/adaptive-interface/window scheme is needed for this first
simple case; it does not test their benefits.

## Independent reference validation

CPU solver uses NumPy/SciPy Fourier collocation,IMEX-BDF2 with implicit
biharmonic/advection,explicit nonlinear extrapolation and Euler starter.
This is an explicitly derived unstabilized pilot integrator; actual refinement,
not an unconditional stability theorem,is the acceptance criterion. No training
operator/JAX import. Nonlocal symbol is independently derived from specified
weights. Local solution64/128grid and dt2e-4/1e-4 checks; graph32grid,time refinement.
The refined local field is sampled onto the common32grid for evaluation.

| Check | Pooled phi difference | Terminal phi difference |
|---|---:|---:|
| Graph time refinement | 7.7368e-8 | 8.2998e-8 |
| Local time refinement | 6.7327e-8 | 7.1639e-8 |
| Local64->128grid | 1.4885e-10 | 2.0651e-11 |
| Graph versus local | 0.00899373 | 0.00819771 |

Reference mean-phase drift<=1.2e-16. AD versus Fourier first derivatives and
finite-grid interpolation remain part of target discretization; graph reference
is not a continuum-exact solution. Kernel/horizon errors are separate from PINN
error. No claim of spatial/horizon convergence from a single nonlocal grid.

Reference paths:
`/root/autodl-tmp/data/twophase2d/results_periodic_pilot.npz`,adjacentJSON.
[Archived checks](twophase-transport-pilot-reference.json).

## Completed training and independent CPU results

Both predeclared cold starts complete5000updates,exit0,serialized initial models
byte-equal,configs differ only nl.method and log.run_name. No seed search or
intermediate-checkpoint selection. Exact final5000models independently reloaded
on CPU and compared to saved GPU fields/metrics. All21times finite; final model
bytes equal checkpoint5000. Original train/network/loss/operator sources match
both run snapshots exactly. Later evaluate.py and point-order validation do not
change numerical training. No statistical stability claim from one paired seed.

| Method | Own-target final phi L2 | Common local-target final phi L2 | Own-target pooled phi L2 | Maximum area drift | Trainer seconds |
|---|---:|---:|---:|---:|---:|
| Nonlocal diffusion + AD gradient | 0.00229651 | 0.00854194 | 0.00184355 | 0.00263956 | 66.67 |
| Mixed local AD | 0.00512124 | 0.00512124 | 0.00316067 | 0.00897484 | 91.06 |

Nonlocal has lower own-target error,while local AD has lower common-local-target
error. These are different questions; **no intrinsic nonlocal accuracy advantage
is established**. Nonlocal finite-horizon target discrepancy is about.82%,larger
than its own-target training error. Measured interface-band final phi L2 is
.00545670/.00545836(nonlocal/local),chemical-potential RMS error.00615556/
.00988590. Last1000-update terminalphi medians.00256204/.00502362 and maxima
.00386961/.00712936. Curves retain all26evaluations without cherry-picking.

Nonlocal/reference phase ranges[-1.09178,1.01725]/[-1.09060,1.01617];
local/reference[-1.11091,1.02276]/[-1.11028,1.02130]. Excursions also occur in
refined physical reference (CH has no automatic maximum principle); field panels
use the full range rather than clipping them away. Predicted energy decreases
between all21saved times for both controls; this observation and the soft mass
penalty are not exact learned conservation/dissipation guarantees.

## Timing scope and direct runs

Trainer wall includes actual initial compilation,evaluation and saving but
excludes reference generation/process import. First synchronized updates take
48.52/67.31s. Mean synchronized compiled steps after100updates are2.164/3.180ms;
these include derivatives/backprop/SOAP,but exclude random-time sampling,
Python bookkeeping,evaluation and checkpoint saves. Thus not a generic full-workload
benchmark. Observed end-to-end subprocess times72.45/96.83s. No fair universal
speed claim,large-grid extrapolation or cross-hardware guarantee.

- `/root/autodl-tmp/tf-logs/twophase2d/20261026-periodic-transport-nonlocal`
- `/root/autodl-tmp/tf-logs/twophase2d/20261026-periodic-transport-local_ad`
- Startup2-update run: `20261026-pilot-startup-check` under the same root.

Each full run retains config/source snapshots,initial/final/1000-update models,
loss/metric JSONL,prediction.npz,summary.json. Explicit existing names fail.
Raw logs,final source snapshot,audit and figures:
`/root/nonlocal-pinn/reports/twophase-transport-pilot`.
[Manifest](twophase-transport-pilot-manifest.json),
[CPU audit](twophase-transport-pilot-audit.json),
[parity/source verification](twophase-transport-pilot-verification.json),
[convergence](twophase-transport-pilot-convergence.png),
[nonlocal fields](twophase-transport-pilot-nonlocal-phi.png),
[local fields](twophase-transport-pilot-local_ad-phi.png).
Plots show reference/prediction/absolute error at0,.25,.5,.75,1; visually inspected.

## Verification and exact resume point

- Five distinct focused checks: periodic operator independent Fourier/nullspace/
energy/transpose; exact periodic IC/parameter differentiability; analytic mixed
AD and strengthened nonlocal residual identities; independent constant reference/
invalid configuration; shuffled reference-node rejection. Original4checks pass;
modified mixed test rerun once,added guard test passes. No unchanged expensive
suite rerun. Logs retained at `/tmp/twophase-pilot-tests.log`,
`/tmp/twophase-pilot-residual-final.log`,`/tmp/twophase-pilot-reference-guard-test.log`.
- Actual default GPU2-update startup passes,then both5000-update GPU jobs pass.
- Independent CPU reload/field/reference/mass/energy/history/source/initial/config
checks pass. Both artifact roots and all3run dirs registered;6cases/319dirs pass.
- New Python AST/JSON/local-link/whitespace and git diff checks recorded in
`twophase-transport-pilot-validation.json`. No current-code compileall/full suite.

Task complete. No worker or blocker. Current branch dev,new pilot uncommitted.
Published old snapshot remains main21fe6d9. Next optional task could add a
spatially varying prescribed vortex,fully nonlocal first derivatives,or a small
matched-density CHNS droplet after deriving pressure/velocity/BC coupling.
None is required to finish this end-to-end pilot. Changed memory: PROGRESS.md,
DECISIONS.md,DEVELOPMENT.md and this archived task; AGENTS navigation list only.
