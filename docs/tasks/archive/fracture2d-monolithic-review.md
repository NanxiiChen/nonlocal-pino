# Fracture2D Local FEM Monolithic Review

## Goal

Assess whether the local FEM reference should gain an optional monolithic
quasi-Newton solution path before the direct nonlocal baseline proceeds. The
review must distinguish ordinary mixed Newton from the BFGS/quasi-Newton method
reported by Kristensen and Martinez-Paneda (2020), and must preserve the current
warning-mode archive until a replacement is validated.

## Acceptance criteria

- Verify the supplied paper's residual, history, BFGS, convergence, and adaptive
  increment choices against the current hybrid AT2/Amor formulation.
- Identify the actual source of the current staggered failure and whether a
  coupled solver addresses that source.
- Audit reusable residual/forms, FEniCS nonlinear interfaces, and required
  mixed-state/history changes.
- Classify implementation effort and expected convergence improvement for
  adaptive stepping, ordinary monolithic Newton, and quasi-Newton monolithic.
- Give a recommendation without changing the accepted local FEM solver.

## Context

- Current reference: `examples/fracture2d/solver.py` with plane strain,
  full-elasticity mechanics degraded by `g(d)`, Amor positive history, DG0
  history projection, CG1 damage, and staggered continuation.
- Current base configuration uses `gate_mode="warning"`; strict staggered
  convergence fails on the crack-growth transition and warning mode commits
  finite candidates for the working curve.
- The point-anchor archive reaches `u_bar=0.0075`, peaks near `u_bar=0.0057`,
  and records 13 gate warnings. At the transition, history and damage defects
  become large enough that the archive is not a strict reference.
- Direct nonlocal work was paused at Stage 0 during this review. The selected
  nonlocal physical route is unchanged and resumes after archival.

## Paper findings

- Kristensen and Martinez-Paneda (2020), Sections 3.2-3.4, solve a coupled
  displacement/phase-field residual in one vector. The initial tangent is block
  diagonal and BFGS secant updates recover the coupling.
- Their convergence result is for BFGS/quasi-Newton monolithic, not ordinary
  Newton. They explicitly report conventional monolithic Newton failing in the
  tension and shear examples, even with line search.
- Their adaptive rule restarts an increment when an integration point has
  `phi < 0.7` and `Delta phi >= 0.5`, reducing the increment to 10 percent. It
  controls resolution of sudden crack growth but does not change the inner
  staggered fixed-point map.
- The paper uses the same hybrid idea: full elastic energy in mechanics,
  Amor positive energy in the history field, and AT2 phase evolution. However,
  its history evaluation and integration-point implementation are not a
  drop-in replacement for the current projected DG0 history field.

## Current implementation findings

- Mechanics residual `F_u` and exact fixed-damage Jacobian `J_u` already exist
  in `solver.py`; the damage equation is already assembled as a linear form.
- The unresolved feedback is the load-step map
  `d -> u -> psi_plus -> H -> d`, not failure of either fixed subproblem.
- A mixed `u,d` solve with history frozen at the committed load-step value
  would remove some mechanics/damage alternation but would not remove the main
  `u -> H -> d` feedback.
- A paper-like monolithic residual must evaluate the current trial history, for
  example as a cellwise `max(H_previous, psi_plus(u))`, or introduce an
  active-set/semismooth history treatment. The max is nonsmooth and the current
  `project()` call cannot simply be hidden inside a standard UFL derivative.
- The FEniCS environment provides `dolfin 2019.1.0`, `PETScSNESSolver`, and
  `petsc4py 3.25.2`; a PETSc SNES quasi-Newton route is potentially reusable but
  requires a mixed residual smoke test before being treated as equivalent to the
  paper's BFGS implementation.

## Status

The review is complete as a read-only assessment. The user deferred the
monolithic implementation, so no solver source, default configuration, or
accepted archive was changed. The optional quasi-Newton path remains a future
follow-up rather than an active task.

## Recommendation

Do not implement ordinary mixed Newton as the first replacement. If a strict
local comparator is required, prototype an optional PETSc SNES quasi-Newton
path with the current boundary conditions and a clearly specified trial-history
rule. Keep adaptive stepping as a supporting safeguard, not as the primary
convergence fix. Compare against the existing warning archive and pre-
localisation strict path before changing any default.

## Outcome

Archive this review and resume `docs/tasks/fracture2d-direct-nonlocal.md` at
Stage 0. If the monolithic path is reopened later, start with a short
pre-localisation mixed residual test and verify residual, history monotonicity,
reaction/work balance, rollback, and MPI behavior before running the
crack-growth transition.
