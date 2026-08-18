# Decisions Log

> Append-only. Newest entries at the bottom. Never edit past entries — if a
> decision is reversed, add a new entry that references the old one.

<!--
Template for each entry:

## YYYY-MM-DD — <short title>
- Decision: <what was decided>
- Why: <reasoning>
- Alternatives rejected: <what else was considered, why not>
- Related: <task file / PR / issue if applicable>
-->

## 2026-08-16 — Adopt repository memory protocol
- Decision: Use `PROGRESS.md` as the short project dashboard, `DECISIONS.md` as
  an append-only technical decision log, and one `docs/tasks/<task-id>.md` file
  per active subtask; archive completed task files.
- Why: The project now has multiple numerical branches and unresolved 2D
  formulation choices that cannot be safely reconstructed from chat history.
- Alternatives rejected: one growing changelog, a single large task document,
  and relying on long-lived conversation context.
- Related: `docs/tasks/memory-protocol.md`, `.opencode/command/task-start.md`,
  `.opencode/command/task-end.md`.

## 2026-08-16 — Prevent silent task switching
- Decision: A new task may not replace a different active task named in
  `PROGRESS.md`; the agent must stop and ask whether to resume, close, or
  explicitly switch tasks.
- Why: Overwriting the dashboard link would orphan the previous resume point
  and make parallel or interrupted work indistinguishable.
- Alternatives rejected: silently replacing the active link and relying on
  filesystem scans to reconstruct task ownership.
- Related: `AGENTS.md`, `.opencode/command/task-start.md`.

## 2026-08-16 — Layer agent policy and engineering references
- Decision: Keep `AGENTS.md` as a concise mandatory policy and index; move
  detailed development conventions, numerical-method rules, and JAX guidance
  into focused documents under `docs/`.
- Why: Stable rules and frequently changing implementation guidance have
  different maintenance lifecycles; mixing them makes the agent contract hard
  to audit and causes stale instructions.
- Alternatives rejected: keeping one large `AGENTS.md`, or splitting every
  subsection into a separate document with no clear index.
- Related: `docs/tasks/agents-refactor.md`, `docs/DEVELOPMENT.md`,
  `docs/NUMERICAL_METHODS.md`, `docs/JAX_GUIDELINES.md`.

## 2026-08-16 — Start pure-nonlocal geometric-notch fracture2d baseline
- Decision: Plan the next fracture example as a direct pure-nonlocal numerical
  baseline on `[0,1] x [0,1]`, with a geometric horizontal notch from `(0,0.5)`
  toward `(0.5,0.5)` and displacement loading on the top boundary; defer PINN
  and core+halo sampling until the discrete baseline is validated.
- Why: A geometric notch avoids a permanent phase-field precrack constraint,
  while pure-nonlocal mechanics and damage make free surfaces part of the
  domain/operator definition rather than local-AD boundary losses.
- Alternatives rejected: phase-field precrack for the first 2D example and
  starting with local-AD mechanics. The exact zero-thickness-notch topology and
  finite-horizon boundary model remain explicit open decisions.
- Related: `docs/tasks/fracture2d-notched-tension.md`.

## 2026-08-16 — Select first fracture2d continuum and support choices
- Decision: Use plane strain, clamp the full bottom edge, prescribe vertical
  displacement on the top, and use the Miehe spectral split with the existing
  history-field irreversibility approach. Interpret the requested horizontal
  notch line as the centerline of a finite-width slot rather than a
  zero-thickness slit.
- Why: These choices define a controlled mode-I numerical baseline, avoid a
  permanent phase-field precrack constraint, and keep irreversibility comparable
  with `fracture1d` while both spatial derivatives remain nonlocal.
- Alternatives rejected: plane stress, minimal horizontal support, strict
  incremental damage inequality, volumetric-deviatoric split, and a
  zero-thickness visibility-only slit for the first implementation.
- Related: `docs/tasks/fracture2d-notched-tension.md`; refines the open topology
  item in “Start pure-nonlocal geometric-notch fracture2d baseline.”

## 2026-08-16 — Stage fracture2d from local FEM to nonlocal PINN
- Decision: Define phase-field width as the full `d=exp(-1)` width, set the
  finite notch width to `2*l0`, use a semicircular tip of radius `l0`, and use a
  common first-study horizon `delta_u=delta_d=delta`. Implement an independent
  local FEM reference first, then the direct nonlocal discretization, and only
  then the nonlocal PINN.
- Why: The local FEM stage verifies geometry, boundary conditions, the
  plane-strain spectral split, history updates, and load continuation before
  adding finite-horizon, correspondence-stability, neural, or sampling errors.
- Alternatives rejected: beginning with the direct nonlocal solver, using
  separate initial mechanics/damage horizons, and proceeding directly to PINN
  training. Whether `(0.5,0.5)` denotes the arc center or foremost notch tip
  remains unresolved and must be fixed before geometry construction.
- Related: `docs/tasks/fracture2d-notched-tension.md`.

## 2026-08-16 — Fix rounded-notch endpoint convention
- Decision: Treat `(0.5,0.5)` as the foremost point of the semicircular notch
  tip. With `w_notch=2*l0` and `r_tip=l0`, place the tip center at
  `(0.5-l0,0.5)` and connect its left semicircle to the boundary-opening slot.
- Why: This preserves the originally requested total notch extent while giving
  the finite-width geometry a smooth tip with the selected radius.
- Alternatives rejected: placing the arc center at `(0.5,0.5)`, which would
  extend the notch to `x=0.5+l0`.
- Related: `docs/tasks/archive/fracture2d-notched-tension.md`.

## 2026-08-16 — Start fracture2d local FEM reference implementation
- Decision: Implement the local FEM stage before any nonlocal operator or PINN
  work. Reuse the accepted fracture1d first-stage defaults `E=210e3`,
  `Gc=2.7`, `l0=0.024`, `k_res=1e-8`, `L0=1`, `du=1e-4`, and
  `u_final=0.02` as explicit 2D baseline configuration values, subject to
  later load-step refinement.
- Why: These values preserve scale comparability with the accepted 1D baseline;
  a shorter initial increment than the 1D `1e-5` keeps the 2D FEM smoke and
  baseline runs tractable while cutbacks and refinement remain explicit.
- Alternatives rejected: starting with pure nonlocal mechanics, using local AD
  as the final reference, or silently inheriting 1D runtime settings without a
  2D configuration.
- Related: `docs/tasks/fracture2d-local-fem.md`.

## 2026-08-16 — Align fracture2d FEM equations with Kristensen 2020
- Decision: Replace the previously selected Miehe spectral mechanics split with
  the hybrid formulation used in Section 3.1 of the supplied Kristensen and
  Martinez-Paneda paper. Use the Amor volumetric-deviatoric positive energy for
  the history field, while the mechanics energy/stress uses the full isotropic
  elastic energy/stress multiplied by `g(d)`. Retain the staggered solver and
  exclude the paper's quasi-Newton and adaptive-step algorithms.
- Why: For fixed damage the mechanics solve remains linear, and for fixed
  history the AT2 damage solve is linear. This is easier to converge for the
  first local FEM reference and matches the paper's reported hybrid approach.
- Alternatives rejected: keeping spectral split mechanics, which introduces a
  piecewise nonlinear tangent near principal-strain switches; implementing the
  paper's adaptive step scheme before the baseline equations are validated.
- Related: `docs/tasks/fracture2d-local-fem.md`,
  `refs/2020 - Phase field fracture modelling using quasi-Newton methods and a new adaptive step scheme.pdf`.

## 2026-08-17 — Hold strict FEM at the unstable transition
- Decision: Do not widen convergence tolerances or add the supplied paper's
  adaptive step rule to conceal the current local FEM failure. Preserve the
  validated pre-localisation archive and stop loudly near `u_bar≈0.0055` until
  a solver strategy is explicitly selected.
- Why: The fixed-damage mechanics and fixed-history damage solves are finite and
  individually stable; the remaining error is the staggered history fixed point
  on the unstable crack-growth branch. Advancing an unconverged state would
  invalidate the direct reference used by later nonlocal comparisons.
- Alternatives rejected: treating the finite candidate as converged, relaxing
  `atol_history`, or silently adopting the paper's adaptive increment criterion.
 - Related: `docs/tasks/fracture2d-local-fem.md`.

## 2026-08-17 — Relax the 2D history fixed-point gate
- Decision: Set the explicit 2D local-FEM history gate to absolute `1e-5` or
  scale-normalized `1e-5`; use damage absolute `1e-5`, and reaction
  absolute/normalized `1e-4`/`1e-6`. Archive all defects. Keep finite checks
  hard and retain healing as an explicit diagnostic; do not add the paper's
  adaptive step rule.
- Why: The previous `1e-7` absolute gate was inherited from the 1D scale. Near
  the 2D notch transition, observed absolute defects up to `2.5e-5` were tiny
  relative to the history field scale, while damage and reaction defects were
  also below the relevant field scales. The paper uses normalized
  residual/correction criteria rather than these inherited absolute thresholds.
- Alternatives rejected: removing the history check, silently accepting a
  non-finite or mechanics/damage-unconverged state, or adding adaptive
  increments before measuring the relaxed fixed-point path.
 - Related: `docs/tasks/fracture2d-local-fem.md`.

## 2026-08-17 — Add explicit warning-mode FEM curve diagnostics
- Decision: Add `solver.gate_mode` with `warning` and `strict` options. Use
  `warning` in the 2D base config to commit finite candidates after the
  stagger budget, while archiving a per-step warning flag and absolute/
  normalized defects. Keep `strict` available for rollback/cutback validation.
- Why: The relaxed finite curve reaches `u_bar=0.02`, but strict staggered
  iteration remains blocked at the crack-growth transition. This matches the
  accepted 1D curve-diagnostic workflow without silently changing the solver
  policy or pretending the warning-mode result is strictly converged.
- Alternatives rejected: clipping damage into `[0,1]`, enforcing a second
  irreversibility constraint, suppressing warning flags, or adding adaptive
  increments before the curve is inspected.
- Related: `examples/fracture2d/solver.py`,
  `examples/fracture2d/configs/solver_base.py`,
  `docs/tasks/fracture2d-local-fem.md`.

## 2026-08-17 — Cap the 2D curve at the complete-opening load
- Decision: Set the base local-FEM endpoint to `u_final=0.0075` and choose
  representative field states from the initial, pre-crack, peak/crack-growth,
  near-complete, and final snapshots. Save mesh triangles and include the
  explicit zero-load `(u_bar,reaction)=(0,0)` point in response histories.
- Why: The warning-mode response already reaches complete opening near
  `u_bar≈0.0057`; extending the default curve to `0.02` only adds a fully
  separated tail. The original nonzero apparent stress at zero was caused by
  omitting the initial state from the reaction history, not by prestress.
- Alternatives rejected: treating the first nonzero load point as the origin,
  using scatter plots for CG1 fields, or extending the default run past the
  complete-opening regime.
- Related: `examples/fracture2d/solver.py`,
  `examples/fracture2d/configs/solver_base.py`,
  `examples/fracture2d/README.md`.

## 2026-08-17 — Use physical extrema for continuous field colors
- Decision: Normalize each continuous representative field plot from the
  actual values in that load state: `vmin=min(field)` and `vmax=max(field)`.
  Apply this independently to `u_x`, `u_y`, `|u|`, and `d`; only a tiny
  degenerate-range padding is used for exactly constant fields.
- Why: A shared displacement scale hid the local notch-tip displacement
  gradient and made the crack concentration difficult to inspect. Per-field
  physical extrema preserve the actual local contrast without clipping.
- Alternatives rejected: a common scale across load states, symmetric limits
  imposed on signed displacement components, or manual crack-region rescaling.
- Related: `examples/fracture2d/solver.py`,
  `examples/fracture2d/README.md`.

## 2026-08-17 — Use minimal tensile displacement constraints
- Decision: Change the active 2D local-FEM boundary condition to `uy=0` on
  the full bottom edge, `ux=0` only at the point `(0,0)`, and `uy=u_bar` with
  free `ux` on the full top edge. Record the mode as
  `bottom_vertical_point_horizontal`.
- Why: This removes the artificial in-plane lateral restraint of the previous
  fully clamped bottom while retaining a well-posed displacement problem. In
  the same warning-mode run, the point-anchor curve reached the endpoint with
  13 warnings versus 17 for full clamp, with a peak reaction change of about
  `0.75%`.
- Alternatives rejected: retaining full bottom `ux=uy=0` as the default, or
  releasing bottom `uy` and leaving only a single point support.
- Related: `examples/fracture2d/solver.py`,
  `examples/fracture2d/configs/solver_base.py`,
  `docs/tasks/fracture2d-local-fem.md`.

## 2026-08-18 — Remove the redundant fracture2d status file
- Decision: Keep `examples/fracture2d/README.md` for stable user-facing problem,
  formulation, usage, and validation guidance; keep the active numerical status
  and output metrics in `docs/tasks/fracture2d-local-fem.md`; remove
  `examples/fracture2d/STATUS.md`.
- Why: The example status file duplicated the active task's blocker, results,
  and next action, while creating a second source of truth for run-specific
  metrics. The project-level `PROGRESS.md` and append-only `DECISIONS.md` remain
  because they serve dashboard and history roles rather than example status.
- Alternatives rejected: merging all project memory into the example README,
  keeping two live numerical status files, or adding a separate results report.
- Related: `examples/fracture2d/README.md`,
  `docs/tasks/fracture2d-local-fem.md`, `docs/PROGRESS.md`.

## 2026-08-18 — Record repository documentation roles in AGENTS.md
- Decision: Add the documentation-role hierarchy to `AGENTS.md`: example
  READMEs are stable user documentation, `PROGRESS.md` is the dashboard,
  `DECISIONS.md` is the append-only rationale log, and active task files hold
  changing validation/status details. Disallow duplicate example-level status
  files.
- Why: Agents need the rule at session start, before choosing where to record
  new results; leaving it only in a fracture2d task would not govern future
  examples.
- Alternatives rejected: copying the full policy into every example README or
  keeping multiple live status sources.
- Related: `AGENTS.md`, `docs/PROGRESS.md`, `docs/DECISIONS.md`,
  `docs/tasks/fracture2d-local-fem.md`.

## 2026-08-18 — Clarify superseded task paths and active FEM handoff
- Decision: Mark the archived fracture2d planning task as superseded, make the
  point-anchor warning-mode FEM archive the working local reference, and make
  the active task's next action the direct pure-nonlocal baseline. Keep strict
  convergence and mesh refinement as explicit deferred follow-ups.
- Why: The user accepted the finite warning-mode result as sufficiently accurate
  for the working reference, while the old dashboard still said to implement
  the FEM and the archived plan retained superseded support/split choices.
- Alternatives rejected: claiming strict FEM completion, reopening the mesh
  refinement work before the nonlocal baseline, or leaving two conflicting
  current-status sources.
- Related: `docs/PROGRESS.md`, `docs/tasks/fracture2d-local-fem.md`,
  `docs/tasks/archive/fracture2d-notched-tension.md`.

## 2026-08-18 — Finalize fracture2d MPI after DOLFIN initialization
- Decision: Import FEniCS before `mpi4py` in the fracture2d solver and rely on
  mpi4py's automatic MPI finalization; do not call `MPI.Finalize()` from the
  solver after a DOLFIN run.
- Why: Importing mpi4py first or explicitly finalizing before Python destroys
  DOLFIN communicator objects caused `MPI_Comm_* after MPI_FINALIZE` errors.
  FEniCS-first import order passed the two-rank smoke run with a clean exit.
- Alternatives rejected: manually finalizing MPI in `__main__`, which ran
  before DOLFIN global communicator cleanup, and leaving MPI finalization
  disabled, which made `mpirun` report abnormal process termination.
- Related: `examples/fracture2d/solver.py`,
  `docs/tasks/archive/fracture2d-local-fem.md`.

## 2026-08-18 — Align archived DG0 fields with triangle connectivity
- Decision: Reorder history and cellwise stress values by their cell-centroid
  correspondence to the archived CG1 triangle connectivity, using exact
  rounded coordinates first and a nearest-centroid fallback only within
  `1e-9`.
- Why: Parallel and serialized DOLFIN degree-of-freedom orderings are not a
  plotting contract. The explicit alignment keeps `H` and stress fields tied
  to the saved triangles without changing the FEM solve.
- Alternatives rejected: assuming local DG0 ordering is globally identical to
  mesh cell ordering, or silently accepting unmatched coordinates.
- Related: `examples/fracture2d/solver.py`,
  `docs/tasks/archive/fracture2d-local-fem.md`.

## 2026-08-18 — Add the requested fracture2d field grid
- Decision: Save `selected_fields_grid.png` as a three-row by five-column
  figure for `u_x`, `u_y`, and `d` at `u_bar=0.005`, `0.0055`, `0.0058`,
  `0.0060`, and `0.0075`. Keep the existing per-panel physical color ranges
  and use only exact saved snapshots within `1e-10` of each requested load.
- Why: The requested columns are present in the default continuation archive,
  and per-panel ranges preserve the established continuous-field plotting
  convention. Short smoke archives do not fail when they do not cover the full
  requested load interval.
- Alternatives rejected: interpolating fields between load states, silently
  reusing the nearest distant snapshot, or imposing a shared manual color
  scale that hides each field's physical extrema.
- Related: `examples/fracture2d/solver.py`,
  `examples/fracture2d/README.md`,
  `docs/tasks/fracture2d-field-grid.md`.

## 2026-08-18 — Parameterize selected field-grid loads
- Decision: Read the field-grid columns from the explicit configuration field
  `c.output.field_grid_loads`; use the default ordered list
  `[0.005, 0.0055, 0.00560, 0.0065, 0.0070, 0.0075]`.
- Why: The requested comparison loads can change without editing solver code,
  while startup validation prevents ambiguous duplicate or unsorted columns.
  Runs shorter than the requested range continue to skip only this optional
  figure.
- Alternatives rejected: retaining hardcoded plot loads or validating the
  targets against `loading.u_final`, which would make short smoke overrides
  fail before solving.
- Related: `examples/fracture2d/configs/solver_base.py`,
  `examples/fracture2d/solver.py`,
  `docs/tasks/fracture2d-field-grid-parameters.md`.

## 2026-08-18 — Revise default field-grid load columns
- Decision: Set the default `c.output.field_grid_loads` to
  `[0.005, 0.0055, 0.0058, 0.0059, 0.0060, 0.0075]` while retaining the
  configuration override.
- Why: These are the requested load states for the default physical-field
  comparison, including the crack-growth transition at `0.0058` and `0.0059`.
- Alternatives rejected: changing the plotting code's parameterization or
  altering the solver load path instead of only changing selected snapshots.
- Related: `examples/fracture2d/configs/solver_base.py`,
  `examples/fracture2d/README.md`,
  `docs/tasks/fracture2d-field-grid-defaults.md`.

## 2026-08-18 — Shift default field-grid loads into crack growth
- Decision: Set the default `c.output.field_grid_loads` to
  `[0.0055, 0.0058, 0.0059, 0.0060, 0.0061, 0.0062]` while retaining the
  configuration override.
- Why: The requested comparison focuses on the post-peak crack-growth branch
  rather than the earlier `u_bar=0.005` state or the fully open `0.0075` state.
- Alternatives rejected: changing the continuation endpoint or changing the
  selected-load plotting implementation instead of the default list only.
- Related: `examples/fracture2d/configs/solver_base.py`,
  `examples/fracture2d/README.md`,
  `docs/tasks/fracture2d-field-grid-defaults.md`.
