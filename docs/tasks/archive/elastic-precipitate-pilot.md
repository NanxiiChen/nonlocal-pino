# Elastic precipitate phase-field pilot

Goal: publish the current research snapshot first, then implement and run a
small two-dimensional elastic-precipitate phase-field PINN pilot.

Acceptance: authorized dev commit -> main regular merge -> push main -> return
to dev precedes new numerical implementation. New example uses train.py, an
explicit cited/derived two-way elastic/phase model, independent refined
reference, operator/energy checks, a cold-start training run and field/error
plots. Report phase and mechanical accuracy and all physical scope limits.

Context: existing transport pilot is closed; SOAP stability remains deferred.
No requirement to reproduce the full anisotropic/no-flux PFHub4 benchmark in
the first pilot. Preserve existing cases, defaults, references and log paths.

Publication complete: dev fbe98d0 -> main regular merge fcdfe91 -> origin/main;
returned to dev with a clean worktree before numerical development. Snapshot
checks: 22 two-phase unittest checks, 6 cases/345 registered directories and
staged whitespace check. Verbatim upstream audit sources retain exact bytes
using a narrowly scoped .gitattributes whitespace exemption.

Status: complete (2026-10-27). Blocker: none. No remaining required action.
Next action: none; larger deformation, heterogeneous moduli, fully nonlocal
mechanics or an independent mechanical PINN require a subsequent task.

Run: /root/autodl-tmp/tf-logs/precipitate2d/20261027-periodic-elastic-pilot-01.
Reference: /root/autodl-tmp/data/precipitate2d/results_periodic_smallstrain_pilot.npz.
Canonical misfit E_xx=0.01; lambda2450/mu1225. A separate preflight reference
results_periodic_pilot used E_xx0.35/lambda2/mu1; C*E^2 and phi evolution agree.
No training used the old preflight parameter set.

Validation so far: four focused unittest checks pass for the final parameter
set: independent acoustic solve/stress/equilibrium/translation gauge; energy
derivative/chemical identity and discrete dissipation; IC/mass/periodicity and
parameter gradients; reference mass/energy and early input validation. At all
saved times, canonical reference max relative refinement differences:
nonlocal dt phi3.07e-7/q6.07e-6/u1.32e-7/stress2.75e-7;
local dt phi5.49e-7/q9.24e-6/u2.49e-7/stress4.98e-7;
local grid127->255 phi2.10e-9/q5.62e-7/u6.42e-11/stress2.21e-9.
Mass drift<=4.44e-16; no sampled energy increase. Disabling eigenstrain changes
terminal native phi by2.5268%; native/local pooled phase gap0.5378%.

## Final trained results

SOAP5000+L-BFGS500 cold start:183.0866 s trainer time including JIT/evaluation/
refinement, first update31.354 s, mean SOAP step after100=19.704 ms.
L-BFGS consumed37.866 s and547 objective calls, stopped at its500-iteration
budget (status1), not a convergence tolerance. No reference fields in loss.

Best checkpoint is also final, update5500, selected by evaluated native pooled
phase relative L2. Independent CPU direct checkpoint replay matches stored
predictions and all four recorded pooled errors to1e-8 tolerance.

| Field | Native pooled relative L2 | Native terminal relative L2 |
|---|---:|---:|
| phi | 0.0003871705262 | 0.0003092430775 |
| u, solved | 0.0001848767837 | 0.0001520412862 |
| stress, component norm | 0.0003541471427 | 0.0002872036374 |
| network q | 0.03807002221 | 0.003162075125 |

Common refined local phi L2 pooled0.005419979174/terminal0.005592545548.
Native interface-band phase L2 pooled0.001145398727/terminal0.000860580296.
q at t0 is16.4204% inaccurate; the phase IC is exact but q's algebraic relation
is soft and t0 is absent from sampled PDE times. It dominates pooled q error.
Do not hide this under the excellent phi metric or assert all-field subpercent
accuracy. Only one seed was trained; no cross-seed reliability/SOTA claim.

Independent saved-field audit: stress/displacement match separate acoustic
matrix solves; max equilibrium-force norm scaled by stress/L=4.80e-14;
phase mass drift2.22e-16; no sampled predicted energy increase. Terminal zero
contour semiaxes: reference(0.19419794,0.20258587), PINN(0.19420568,0.20256016),
with x/y ratios0.95859568/0.95875557; uncoupled control ratio1. Deformation is
modest and this is a route-validation pilot, not a demanding shape benchmark.

## CLI completion correction

The first full numerical run wrote all final outputs and summary but exited1
because an absl lambda propagated train()'s Path return into sys.exit. Replace
it with _main that discards the return; no training math changed. A separate
small-network SOAP1+L-BFGS1 smoke through the actual train.py exits0 and writes
all outputs. Do not mislabel that smoke as another full accuracy run.
The first constant-state assertion additionally needed explicit NumPy
broadcasting; corrected test passes for local and nonlocal reference modes.
An attempted system-Python config check lacked ml_collections; rerun using
the project's .venv passes. No dependencies were installed or changed.

Source snapshots retain the exact full-run code; final code adds CLI return
correction, stronger early config validation and correctly centered periodic
plot coordinates. None change the saved training trajectory or model identity.

## Deliverables and validation

New examples/precipitate2d: train.py canonical entry, explicit train_base config,
problem/identity/range checks, original mixed loss, periodic constrained network,
differentiable mechanics, independent solver and direct replay/plots. Reuses
twophase2d's existing periodic operator and PDE-only refinement helper without
changing them. New tests/test_precipitate.py covers the numerical contracts.
Root README, AGENTS navigation, DEVELOPMENT and artifacts/index are updated;
seven cases/349 registered directories validate. Existing defaults unchanged.

Focused verification: four new tests pass on canonical parameters, plus the
extended constant-state/validation check and actual corrected CLI smoke. The
22 pre-publication two-phase tests already pass; reused code is unchanged.
All-field reference refinements, full numerical run, CPU checkpoint replay,
independent saved-field audit, viewed plots, direct links and git diff --check
pass. No background training/FEM process remains; GPU is idle.

Artifacts alongside this report (elastic-precipitate-*): summary, best metrics,
config, complete loss/metric histories, refinement summary, reference checks,
CPU replay, independent validation JSON and its executable verify script;
phi_fields.png, mechanical_fields.png and convergence_and_shape.png. Large
checkpoints/references remain at their direct external paths. New development
is uncommitted on dev; the requested pre-development snapshot alone was pushed.

Changed memory: PROGRESS.md, DECISIONS.md, DEVELOPMENT.md and this archived task.

Model scope: dimensionless 2D small-strain isotropic elasticity, fixed zero
macroscopic strain, zero-mean periodic displacement, linear coherent eigenstrain
E*(1+phi)/2, quartic phase energy and CH kinetics. Nonlocal approximation only
in phase gradient energy/mobility; mechanics is a differentiable spectral
subproblem, not a separately learned displacement network. Equal moduli permit
exact Fourier elimination. Uniaxial eigenstrain supplies physical directional
shape preference even with isotropic stiffness. This simplified periodic case
is not PFHub4's anisotropic, tenth-order-potential, traction-free benchmark.
