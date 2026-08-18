# Fracture2D Local FEM Reference

## Goal

Implement an independent plane-strain local FEM history-field phase-field
fracture solver for the unit square with a rounded finite-width notch and top
displacement loading.

## Acceptance criteria

- Generate the rounded notch mesh from explicit configuration geometry.
- Use the paper-aligned hybrid AT2/history-field formulation and boundary
  conditions without hidden loss or clipping constraints.
- Solve incremental staggered mechanics/damage states with rollback/cutback,
  finite-value checks, and reproducible archives/plots.
- Validate mesh, load-step, stagger, damage-range, energy, and reaction outputs.
- Add focused geometry/configuration tests and pass compilation plus relevant
  FEM smoke verification.

## Relevant context

- The current 2D reference uses plane strain, bottom-edge `uy=0`, a pointwise
  `ux=0` anchor at `(0,0)`, top vertical displacement, and history-field
  irreversibility before any nonlocal or PINN
  implementation. The formulation is now aligned with the supplied paper's
  Amor-history/isotropic-mechanics hybrid.
- Domain is `[0,1] x [0,1]`; the notch centerline is horizontal at `y=0.5`.
- `w_notch=2*l0`, `r_tip=l0`; `(0.5,0.5)` is the foremost tip, so the
  semicircle center is `(0.5-l0,0.5)`.
- The implementation order is local FEM, direct nonlocal discretization, then
  nonlocal PINN. The accepted `fracture1d` material/load values are reused as
  explicit first-stage defaults for comparability.

## Status

Complete as the accepted warning-mode working reference: geometry, hybrid FEM
solver, configuration, tests, documentation, smoke runs, pre-localisation
baseline, and the warning-mode full curve are validated.
The active boundary mode is `uy=0` on the bottom edge plus `ux=0` at `(0,0)`;
the top edge has `uy=u_bar` and free `ux`. The final repository regression suite
passes 65 tests.

## Validated outputs

- The default mesh has `3395` vertices and `6284` triangles. Its measured
  material area is `0.9762509375`.
- The pre-localisation path to `u_bar=0.005` completed with 50 load steps,
  peak reaction `601.5453`, final maximum damage `0.40061`, and no negative
  healing diagnostic:
  `/root/autodl-tmp/data/fracture2d/local_fem_prelocalization/results_prelocalization.npz`.
- The point-anchor warning-mode curve reaches `u_bar=0.0075` in 126 accepted
  steps, has 13 gate warnings, and peaks at reaction `642.4857` at
  `u_bar=0.0057`:
  `/root/autodl-tmp/data/fracture2d/local_fem_point_anchor/results_point_anchor.npz`.
- The preserved full-clamp comparison has peak reaction `637.6893` at
  `u_bar=0.0056`, 17 warnings, and is stored at
  `/root/autodl-tmp/data/fracture2d/local_fem/results_local_fem.npz`.
- The point-anchor mode shifts the peak by `+0.0001` in `u_bar`, raises the
  peak reaction by about `0.75%`, and reduces warnings from `17` to `13`.
- Archives include triangle connectivity, an explicit zero-load `(u_bar,
  reaction)=(0,0)` point, and continuous CG1 representative field plots at
  initial, before-crack, crack-development, near-complete, and fully-open states.

## Curve-diagnostic limitation

- The 2D history gate uses absolute/normalized tolerances of `1e-5`/`1e-5`;
  damage uses `1e-5`, and reaction uses absolute/normalized tolerances of
  `1e-4`/`1e-6`. All defects are archived separately.
- Strict mode still stops near `u_bar=0.0055` at the unstable staggered
  transition. Warning mode commits finite candidates after the stagger budget.
- The latest point-anchor warning-mode curve has `max(d)=1.005288` and minimum
  healing `-1.063e-4` after crack opening. These are small but nonzero violations, so
  the archive is a response-curve diagnostic, not yet a strict
  irreversibility/damage-range reference.
- No clipping, damage-bound projection, adaptive damage-triggered step rule, or
  symmetry constraint is added.

## Verification

- The focused fracture2d geometry/configuration suite passes 5 tests in
  `fenicsenv`.
- The repository `unittest` suite passes 65 tests in `.venv`.
- `solver.py`, geometry, configuration, and the focused test compile cleanly in
  `fenicsenv`.
- A serial default run reaches `u_bar=0.0075` and regenerates the archive and
  plots above.
- A two-rank MPI smoke run reaches `u_bar=1e-4`, exits cleanly, and gathers the
  global archive shapes `(3395,2)` vertices, `(6284,3)` triangles, and
  `(2,6284)` DG0 history/stress fields. FEniCS is imported before `mpi4py` so
  automatic MPI finalization occurs after the DOLFIN communicator lifecycle.
- Mesh cache metadata is written beside the XDMF file at
  `/root/autodl-tmp/data/fracture2d/local_fem/rounded_slot.xdmf.metadata.json`.

## Next action

Treat the point-anchor warning-mode archive as the working local reference and
start the next task with a separate direct pure-nonlocal baseline. Strict
staggered convergence, mesh refinement, and a monolithic solver remain deferred
follow-ups if a strict local reference is later required.
