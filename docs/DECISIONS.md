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

## 2026-08-18 — Do not cite PD-PFM papers as unchanged hybrid AT2 support
- Decision: Do not use Roy et al. (2017) or Bie et al. (2023) as direct support
  for the planned Amor-history AT2/correspondence coupling. Treat them as
  alternative fracture models unless their phase-field-driven bond failure and,
  for Bie, modified stable correspondence kinematics are explicitly selected.
- Why: Primary metadata, Roy's abstract, and the equation-level descriptions in
  the 2024 Dorduncu et al. review show that both models use the phase field to
  provide a peridynamic bond-breaking criterion. Bie additionally introduces a
  new deformation gradient, shape matrix, and force state to remove zero-energy
  modes. Those choices differ from retaining diffuse AT2 damage while degrading
  full isotropic correspondence energy by `g(d)`.
- Alternatives rejected: inferring an endpoint-averaged degradation force from
  the paper titles, citing intact Silling correspondence as a complete fracture
  coupling, or presenting inaccessible Roy/Bie equations from secondary
  descriptions as independently verified primary equations.
- Related: `docs/tasks/fracture2d-direct-nonlocal.md`.

## 2026-08-18 — Qualify Bie PD-PFM after full-text review
- Decision: Treat Bie et al. (2023) as a separate cohesive plane-stress
  PD-PFM, not as the direct nonlocal version of the current hybrid AT2 FEM. Do
  not add binary bond deletion unless a separate explicit rule is selected.
- Why: The paper derives energy-compensated correspondence and nonlocal phase
  flow, but uses `alpha(d)=2d-d^2`, rational degradation, tensile strength,
  stress-based history, and plane-stress coefficients. Its displayed equations
  continuously degrade point-centered force states and do not print a binary
  bond-status update. The phase-flow equations also contain coefficient and
  notation ambiguities that require independent resolution.
- Alternatives rejected: silently setting `b=l0`, reusing the plane-stress
  micro-modulus in plane strain, or interpreting “bond breakage” as an unprinted
  graph-edge deletion algorithm.
- Related: `refs/Bie-PD-PFM-fracture.pdf`,
  `docs/tasks/fracture2d-direct-nonlocal.md`.

## 2026-08-18 — Classify SGE as nonlocal phase-field, not peridynamics
- Decision: Treat Sinha et al.'s SGE route as a finite-neighborhood nonlocal
  gradient phase-field model, not as correspondence peridynamics. A direct SGE
  solver must be derived from the stated energy using the discrete gradient
  adjoint rather than copying the paper's local-looking residual diagnostics.
- Why: SGE estimates displacement and damage gradients through centered
  neighborhood moment inverses and inserts them into AT1/AT2 energy, but has no
  deformation state, force state, bond energy, or endpoint peridynamic balance.
  The paper demonstrates a PINN energy implementation but does not provide the
  complete adjoint-consistent direct algebraic system.
- Alternatives rejected: equating the SGE moment matrix with a peridynamic
  shape tensor solely because the algebra looks similar, or calling its printed
  local AT2 residual the variation of the full nonlocal penalized energy.
- Related: `docs/tasks/fracture2d-direct-nonlocal.md`.

## 2026-08-18 — Select stable correspondence with nonlocal AT2
- Decision: Build the fracture2d direct baseline with stable,
  energy-consistent correspondence mechanics; a per-particle-volume weighted
  `SparseD` nonlocal AT2 damage operator; Amor positive-energy maximum history;
  history-field irreversibility; and staggered mechanics/history/damage
  coupling. Solve the fixed subproblems with sparse or matrix-free linear
  methods. The later PINN will reuse the same strong-form nonlocal residuals,
  and the existing local FEM remains the independent reference.
- Why: This preserves the fracture law, history semantics, staggered workflow,
  and direct-baseline-to-PINN progression of `fracture1d` while extending both
  mechanics and damage spatial operators nonlocally. It changes fewer physical
  ingredients than Bie PD-PFM or SGE and therefore gives the clearest comparison
  with the current local hybrid AT2 FEM.
- Alternatives rejected: adopting Bie PD-PFM as the primary baseline, because
  it introduces cohesive strength/softening physics, rational degradation, a
  different crack density, and plane-stress-specific formulas; adopting SGE as
  the primary baseline, because it is a nonlocal gradient-energy model rather
  than peridynamic correspondence and changes mechanics splitting and
  irreversibility; and using conventional `ElasticD` with an ad-hoc endpoint
  damage average, because that is neither energy-consistent nor stable against
  correspondence zero modes.
- Related: `docs/tasks/fracture2d-direct-nonlocal.md`,
  `examples/fracture1d/nonlocal_baseline.py`, `src/pinn/operators.py`.

## 2026-08-18 — Reassess local FEM monolithic convergence before nonlocal work
- Decision: Pause direct nonlocal implementation long enough to assess an
  optional monolithic/quasi-Newton local FEM solver. Preserve the accepted
  point-anchor warning-mode archive and do not replace the local-FEM default or
  claim a stricter reference until the alternative solver is independently
  compared.
- Why: The current staggered fixed-point gate fails on the crack-growth branch,
  while Kristensen and Martinez-Paneda (2020) specifically report improved
  performance from BFGS monolithic coupling. The paper's result concerns the
  quasi-Newton method, not an ordinary mixed Newton solve, and its history-field
  semantics must be reconciled with the current DG0 history implementation.
- Alternatives rejected: further relaxing the existing convergence gates,
  silently committing more staggered candidates, or proceeding directly to the
  nonlocal baseline without first deciding whether the local comparator needs a
  strict monolithic reference.
- Related: `docs/tasks/fracture2d-monolithic-review.md`,
  `docs/tasks/fracture2d-direct-nonlocal.md`,
  `refs/2020 - Phase field fracture modelling using quasi-Newton methods and a new adaptive step scheme.pdf`.

## 2026-08-18 — Defer local FEM monolithic implementation
- Decision: Do not implement the optional local-FEM monolithic/quasi-Newton
  path now. Archive its review, keep the warning-mode local FEM archive
  unchanged, and resume the selected direct pure-nonlocal baseline at Stage 0.
- Why: The review confirms that BFGS/quasi-Newton could improve the staggered
  transition, but it requires a new mixed solver and an explicit trial-history
  treatment. That work is not needed to freeze and validate the selected
  nonlocal correspondence/AT2 formulation, and changing the current comparator
  now would delay the main line without a validated replacement.
- Alternatives rejected: implementing ordinary mixed Newton as a shortcut,
  promoting an unvalidated quasi-Newton curve to the local reference, or
  continuing to relax the current staggered gates.
- Related: `docs/tasks/archive/fracture2d-monolithic-review.md`,
  `docs/tasks/fracture2d-direct-nonlocal.md`, `docs/PROGRESS.md`.

## 2026-08-18 — Draft bond-associated energy-first nonlocal formulation
- Decision: For Sol review, specify the direct baseline with Chen-Spencer-style
  common-neighbor bond-associated deformation gradients, a project-owned
  normalized center-energy partition, full plane-strain elasticity degraded by
  `g(d)=(1-d)^2+k_res`, weighted nonlocal AT2 damage, Amor particle history, and
  reactions obtained from the constrained mechanics-energy derivative. Use a
  reciprocal fixed visibility graph, barycentric particle volumes, and
  horizon-thick top/bottom displacement collars.
- Why: The formulation removes the conventional `ElasticD` zero-mode shortcut,
  keeps every mechanics quantity as a derivative of one declared discrete
  energy, preserves the local hybrid/history fracture law, and makes all
  quadrature, visibility, collar, and reaction choices explicit for independent
  tests.
- Alternatives rejected: multiplying the existing `ElasticD` residual by a
  point damage factor, averaging endpoint damage factors without an energy,
  adding an uncalibrated hourglass penalty, or adopting Bie's plane-stress
  cohesive PD-PFM equations as unchanged AT2 mechanics.
- Review status: Draft only; pending Sol's independent review and finite-cloud
  patch, energy-derivative, tangent, reaction, and null-space checks. No
  production implementation is authorized by this entry.
- Related: `docs/tasks/fracture2d-direct-nonlocal.md`,
  `refs/Bie-PD-PFM-fracture.pdf`, `src/pinn/operators.py`.

## 2026-08-18 — Require revision of the Stage 0 nonlocal draft
- Decision: Do not implement or submit the current Stage 0 draft as accepted.
  First restore direct target-bond sensitivity in the bond-associated support
  and select a boundary constraint model that explicitly distinguishes an
  exterior fictitious interaction layer from an interior material grip.
- Why: The graph-neighbor intersection excludes the target endpoint because
  self-edges are absent, so it does not faithfully retain the Chen-Spencer
  stability mechanism. The proposed interior horizon strip also changes the
  deformable domain and is not the exterior fictitious-layer method cited as
  support.
- Alternatives rejected: treating a random-cloud energy/Hessian check as proof
  of BA stability, or silently describing an interior grip as a referenced
  nonlocal volume collar.
- Related: `docs/tasks/fracture2d-direct-nonlocal.md`, Chen and Spencer (2019),
  DOI `10.1002/nme.5973`, and Behzadinasab, Foster, and Bazilevs Part II, DOI
  `10.1007/s42102-020-00039-6`.

## 2026-08-18 — Restore target-endpoint sensitivity in BA support
- Decision: Define the Stage 0 discrete bond-associated support as
  \(\mathcal C_{i|j}=\{j\}\cup\{k\ne i,j:(i,k),(j,k)\in\mathcal E\}\),
  with the explicit \(\omega_{ij}V_j\) target contribution carried through
  the shape tensor, current moment, kinematic map, force, and tangent sums.
  Keep \(j\) as an endpoint quadrature term rather than adding a graph self-edge.
- Why: Chen and Spencer (2019), Section 3, Eqs. (16)-(17), constructs the
  deformation gradient for an individual target bond from an intersected
  bond-associated neighborhood. The previous discrete intersection omitted
  \(j\) solely because self-edges were absent, making \(\boldsymbol F_{i|j}\)
  blind to \(\boldsymbol u_j-\boldsymbol u_i\). The corrected float64 check
  gives rank two for every support, condition number at most `6.102414...`,
  nonzero target sensitivity, affine error `2.22e-16`, three rigid null modes,
  and first non-rigid eigenvalue `3.5559679e-02` on the manufactured cloud.
- Alternatives rejected: retaining the deficient common-neighbor set, adding
  a self-edge to the reciprocal material graph, or claiming the regular-cloud
  check proves rounded-slot/boundary stability.
- Related: `docs/tasks/fracture2d-direct-nonlocal.md`,
  `tests/test_fracture2d_stage0_ba.py`, DOI `10.1002/nme.5973`.

## 2026-08-18 — Keep the nonlocal boundary BVP as a Stage 0 gate
- Decision: Do not silently select an exterior fictitious interaction collar,
  an interior material grip, or point/boundary-only constraints. Keep the gate
  open and require the selected option's particles, graph, kinematic masks,
  energy-center rule, mixed component treatment, and work-conjugate reaction to
  be derived before production implementation.
- Why: Behzadinasab, Foster, and Bazilevs Part II, DOI
  `10.1007/s42102-020-00039-6`, Section 4 and Eqs. (8)-(9), uses a horizon-thick
  exterior fictitious layer with separate kinematic and stress-divergence
  neighborhoods and assumes essential or natural data at each boundary point.
  The current material-only energy with componentwise top/bottom displacement
  and a point horizontal gauge does not define that exterior construction.
  Relabeling the existing interior strip would change the BVP without deriving
  the missing extension.
- Alternatives rejected: calling the interior material strip an exterior
  collar, copying the source's natural-boundary treatment into the energy-first
  BA model, or choosing a point/boundary-only treatment as an equivalent
  shorthand. Recommendation to Sol: choose a new exterior derivation only if
  that physical BVP is required; otherwise explicitly approve the project-owned
  interior grip or another discrete BVP.
- Related: `docs/tasks/fracture2d-direct-nonlocal.md`, `docs/PROGRESS.md`,
  DOI `10.1007/s42102-020-00039-6`.

## 2026-08-18 — Require independent explicit mechanics residual checks
- Decision: Keep the Stage 0 mechanics energy/Hessian linear-map path separate
  from a residual path that recomputes each BA term's current moment,
  deformation gradient, plane-strain stress, and explicit source/target endpoint
  contributions using \(\beta\boldsymbol P\boldsymbol K^{-T}\boldsymbol\xi\).
  Use that residual for the energy directional finite difference and for the
  residual-based Hessian JVP finite difference; assert its consistency with
  \(\boldsymbol K\boldsymbol u\) separately.
- Why: Using `gradient = hessian @ displacement` for the energy check could make
  a shared transpose, endpoint sign, or index error self-consistent. The direct
  assembly independently exercises the Section 0.5 force direction and
  Kronecker endpoint/source terms.
- Alternatives rejected: differentiating the Hessian action itself for the JVP,
  treating residual/Hessian agreement as the only energy check, or relying on
  the previous shared-path gradient.
- Related: `docs/tasks/fracture2d-direct-nonlocal.md`,
  `tests/test_fracture2d_stage0_ba.py`.

## 2026-08-18 — Approve Stage 0 for geometry and quadrature work
- Decision: Accept the corrected target-inclusive BA equations, explicit
  nonlocal residual/tangent checks, weighted AT2 algebra, and boundary-particle
  hard-constraint BVP as the Stage 0 specification. Proceed to Stage 1 geometry
  and quadrature work, but do not yet implement coupled mechanics/fracture.
- Why: Independent residual, energy-derivative, tangent-JVP, rigid-mode,
  free-block SPD, constrained residual, reaction balance, and reduced-energy
  checks pass on the manufactured cloud. The boundary rule also reproduces the
  local FEM component constraints without an exterior collar or interior grip.
- Alternatives rejected: reopening the corrected BA endpoint rule, using
  automatic spatial derivatives, or treating the manufactured check as proof
  for the rounded-slot nonuniform cloud.
- Related: `docs/tasks/fracture2d-direct-nonlocal.md`,
  `tests/test_fracture2d_stage0_ba.py`, `docs/PROGRESS.md`.

## 2026-08-18 — Freeze boundary-particle hard constraints for fracture2d
- Decision: Freeze the first-study boundary BVP on the material-only graph as
  direct hard constraints on physical boundary representative particles:
  `uy=0` on the bottom boundary, `uy=u_bar` on the top boundary, and `ux=0`
  at the lower-left anchor. All other top/bottom horizontal DOFs remain free.
  Boundary particles remain graph nodes, BA support points, and energy centers;
  damage has no hard boundary condition. Use DOF elimination/affine lifting,
  not a penalty.
- Why: This is the user's selected discrete BVP and is directly compatible with
  the declared material-only energy. The independent 5x5 check gives a strict
  free block, free residual `1.5669856e-16`, top reaction/reduced-energy FD
  relative error `6.3111087e-13`, and separate bottom/anchor diagnostics.
  Production masks must come from retained Gmsh physical boundary tags, or from
  an explicit deterministic Stage 1 mesh/geometry construction; hidden
  coordinate tolerances are not permitted.
- Alternatives rejected: an exterior fictitious collar, because Behzadinasab
  Part II defines a different extended-layer/two-neighborhood BVP; and an
  interior material grip, because it constrains a material volume and changes
  the deformable BVP. Neither is represented by the selected hard constraints.
- Related: `docs/tasks/fracture2d-direct-nonlocal.md`, `docs/PROGRESS.md`,
  DOI `10.1007/s42102-020-00039-6`.

## 2026-08-18 — Preserve tagged Gmsh entities for fracture2d Stage 1
- Decision: Add an explicit Gmsh physical point named `anchor` and make the
  Stage 1 loader require the named material surface, all outer/slot boundary
  curves, and that unique point. Build the particle cloud directly from the
  tagged `.msh` mesh rather than from the local solver's tag-dropping XDMF
  conversion. Store reciprocal directed horizon edges after analytic
  rounded-slot visibility filtering.
- Why: The frozen boundary-particle BVP requires exact top/bottom masks and a
  unique lower-left anchor. A coordinate-tolerance fallback would make the
  production boundary condition depend on mesh placement rather than the
  declared physical entities. Keeping the original `.msh` also preserves the
  physical tags needed by later operators without coupling them to FEniCS.
- Alternatives rejected: inferring masks from coordinate extrema, deriving
  volumes from a bounding box or particle count, and reusing the XDMF pipeline
  after it discards line/point physical data.
- Related: `examples/fracture2d/mesh.geo`,
  `examples/fracture2d/nonlocal_geometry.py`,
  `docs/tasks/fracture2d-direct-nonlocal.md`.

## 2026-08-18 — Use mesh-conforming rounded-slot visibility
- Decision: Define the Stage 1 reciprocal graph against the polygonized material
  domain represented by the retained Gmsh triangles and tagged boundary edges.
  Retain tagged notch-tip chord edges as material-closure interactions, and
  treat the unmeshed segment across the left slot opening as void rather than a
  material boundary. Keep exact-CAD visibility only as a separately declared
  future sensitivity study.
- Why: Barycentric quadrature and physical masks already use the triangulated
  Gmsh domain. Exact-circle visibility removes all 20 legal notch-tip boundary
  chords because they lie inside the CAD circle, mixing two discrete geometries
  and increasing the observed worst BA condition number by about two orders of
  magnitude. Mesh-conforming visibility keeps graph, quadrature, and boundary
  topology consistent without hiding the CAD-to-mesh geometry error.
- Alternatives rejected: exact-CAD filtering in the production Stage 1 graph,
  which deletes mesh boundary interactions; and treating the left opening line
  as material closure, because no material triangles or tagged material
  boundary represent its open segment.
- Related: `examples/fracture2d/nonlocal_geometry.py`,
  `tests/test_fracture2d_nonlocal_geometry.py`,
  `docs/tasks/fracture2d-direct-nonlocal.md`.

## 2026-08-18 — Close fracture2d Stage 1 geometry gates
- Decision: Accept the tagged Gmsh nodal cloud, barycentric nodal volumes,
  mesh-conforming reciprocal visibility graph, and target-inclusive BA support
  conditioning on the fresh rounded-slot mesh for progression to the weighted
  scalar operator. Keep the horizon value open; checks at `delta=0.032`,
  `0.040`, and `0.048` are diagnostics, not a production-horizon selection.
- Why: All directed BA supports had rank two and affine reproduction errors at
  most `1.39e-14`. At diagnostic `delta=0.048`, the intact constrained
  matrix-free mechanics operator had converged extremal eigenvalues
  `2.8440031721752207e-05` and `1.3314397965182772`, with normalized Ritz
  residuals `2.54e-11` and `8.99e-14`; rigid-mode, symmetry, and independent
  energy/action checks also passed. Zero and `1e-12` visibility inputs produced
  identical fresh-mesh graphs and retained every tagged slot edge.
- Alternatives rejected: selecting a production horizon from one finite mesh;
  adding a kernel floor or deleting near-cutoff edges to increase target
  sensitivity; and replacing the sparse/matrix-free spectrum with a dense
  global construction.
- Related: `examples/fracture2d/nonlocal_geometry.py`,
  `tests/test_fracture2d_nonlocal_geometry.py`,
  `docs/tasks/fracture2d-direct-nonlocal.md`.

## 2026-08-18 — Accumulate explicit-volume scalar differences by edge
- Decision: Keep the explicit positive edge weights and diagonal/off-diagonal
  matrix data for export, but evaluate the JAX matvec by accumulating
  `J_delta(r_ij) V_j (d_i-d_j)` directly on each directed edge.
- Why: This is algebraically identical to the frozen diagonal-minus-off-
  diagonal matrix, while avoiding cancellation between separately accumulated
  float32 diagonal and neighbor sums. The full-square constant residual became
  exactly zero in the manufactured check without changing the operator's
  SciPy matrix or the frozen volume pairing.
- Alternatives rejected: changing the kernel or volumes, adding a numerical
  correction to the diagonal, or modifying `SparseD` to accommodate the new
  nonuniform-volume operator.
- Related: `src/pinn/operators.py`, `tests/test_scalar_operator_2d.py`,
  `docs/tasks/fracture2d-direct-nonlocal.md`.

## 2026-08-18 — Freeze the fracture2d direct baseline at float64
- Decision: Keep the fracture2d direct geometry, barycentric volumes, strict
  directed graph, serialized operator arrays, and operator inputs in float64,
  with JAX x64 enabled before constructing `ExplicitVolumeSparseD`. Require
  general operator coordinates and volumes to use the same dtype; matching
  float32 remains available only as a separately rebuilt general path.
- Why: The Stage 0/1 finite-cloud mechanics evidence and the direct solver's
  numerical role use float64. Because graph membership uses strict
  `r_ij < delta`, casting coordinates can change the graph, so dtype is part of
  the graph contract rather than a post-construction representation detail.
- Alternatives rejected: casting the float64 graph for a future float32 PINN,
  allowing mixed geometry/volume dtypes, adding a cutoff tolerance or kernel
  floor, or silently downcasting when JAX x64 is disabled. A future float32
  PINN must rebuild and revalidate its graph in float32.
- Related: `examples/fracture2d/nonlocal_geometry.py`,
  `src/pinn/operators.py`, `docs/tasks/fracture2d-direct-nonlocal.md`.

## 2026-08-21 — Keep Stage 3 mechanics as an independent direct path
- Decision: Implement intact fracture2d bond-associated mechanics in
  `examples/fracture2d/nonlocal_mechanics.py` with NumPy/SciPy float64 arrays,
  explicit target-inclusive supports, an energy-gradient residual, an analytic
  tangent action, and matrix-free hard-constraint solves. Do not extend the
  conventional `ElasticD` path and do not introduce JAX/PINN mechanics in this
  stage.
- Why: Stage 3 is the independent direct numerical baseline used to validate
  the frozen energy, force, reaction, and spectrum before neural representation
  error is introduced. NumPy/SciPy provides the required deterministic sparse
  and matrix-free checks while keeping the implementation separate from both
  the local FEM weak form and the later PINN operator.
- Alternatives rejected: modifying `ElasticD`, whose conventional center-
  correspondence residual is not the approved BA energy derivative; adding a
  JAX mechanics implementation before the direct reference passes; and adding
  damaged mechanics before the intact path is accepted.
- Related: `examples/fracture2d/nonlocal_mechanics.py`,
  `tests/test_fracture2d_nonlocal_mechanics.py`,
  `docs/tasks/fracture2d-direct-nonlocal.md`.

## 2026-08-21 — Approve Stage 3 intact BA mechanics
- Decision: Accept the independent float64 intact bond-associated mechanics
  implementation for progression to Stage 4 fixed fracture subproblems. Keep
  its energy, residual, tangent, hard constraints, reaction convention, and
  solver-only block-Jacobi preconditioner unchanged while adding damaged and
  fixed-history subproblems separately.
- Why: Independent support/energy/residual checks, affine and rigid patches,
  exactly three unconstrained rigid modes, tangent symmetry/PSD, force/moment
  conservation, strict-CG hard-constraint solves, block-diagonal verification,
  reaction-work finite differences, and the interior Navier-Cauchy density check
  pass. Fresh rounded-slot diagnostics have no rank failures and independently
  reproduce the prior matrix-free tangent and positive constrained spectrum.
- Alternatives rejected: promoting a diagnostic horizon to the production
  common horizon; modifying `ElasticD`; treating block-Jacobi as mechanics
  stabilization; and beginning coupled damage or PINN work before fixed
  subproblems are validated.
- Related: `examples/fracture2d/nonlocal_mechanics.py`,
  `tests/test_fracture2d_nonlocal_mechanics.py`,
  `docs/tasks/fracture2d-direct-nonlocal.md`.

## 2026-08-21 — Share immutable BA data in fixed-damage views
- Decision: Construct `FixedDamageBondAssociatedMechanics` from one accepted
  intact `BondAssociatedMechanics` instance and one fixed float64 damage field.
  Share its read-only geometry, graph, supports, shape inverses, and constitutive
  data, but use source factors `V_i*g(d_i)*eta_ij`, separate lazy tangent caches,
  and an object-state lock. Independently reconstruct and validate the frozen 2D
  cubic SciPy damage matrix before every fixed-history solve.
- Why: Rebuilding all target-inclusive supports for every fixed damage field
  made the rounded diagnostic exceed ten minutes and could let mechanics and
  damage graphs drift. Mutating intact factors or sharing its block-Jacobi cache
  would make residual/tangent data inconsistent. The explicit CSR audit closes
  the analogous risk for damage energy and the Euclidean-SPD system.
- Alternatives rejected: rebuilding BA kinematics for each staggered iterate;
  mutating the intact instance; sharing damage-dependent caches; accepting an
  arbitrary `ExplicitVolumeSparseD` dimension, kernel, or SciPy export; and
  silently renormalizing Amor weights.
- Related: `examples/fracture2d/nonlocal_mechanics.py`,
  `examples/fracture2d/nonlocal_baseline.py`,
  `tests/test_fracture2d_nonlocal_baseline.py`.

## 2026-08-21 — Approve Stage 4 fixed fracture subproblems
- Decision: Accept fixed-`d=0` mechanics, fixed-history nonlocal AT2 damage,
  fixed-damage BA mechanics, and the 3D plane-strain Amor particle driver for
  progression to Stage 5 strict staggered continuation. Keep all tested
  horizons diagnostic and leave production-horizon selection open.
- Why: Independent manufactured tests pass energy/residual/tangent, SPD,
  source-degradation, effective-stress, reaction-work, immutability, and loud
  validation gates. Fresh rounded-slot checks at `delta=0.032`, `0.040`, and
  `0.048` reproduce the low-load FEM displacement/reaction/stress trends;
  Amor-driven damage remains in range with strong residual infinity norm below
  `7.2e-15`, conservation defects are near float64 precision, and the
  `delta=0.048` reaction-work relative error is `9.24e-13`. Independent Sol
  review found no remaining Stage 4 issue.
- Alternatives rejected: starting coupled continuation before fixed-subproblem
  review; treating warning-mode FEM values as a strict converged target;
  promoting one diagnostic horizon to production; clipping damage; or adding
  stabilization/floors to improve diagnostics.
- Related: `examples/fracture2d/nonlocal_baseline.py`,
  `tests/test_fracture2d_nonlocal_baseline.py`,
  `docs/tasks/fracture2d-direct-nonlocal.md`, `docs/PROGRESS.md`.

## 2026-08-21 — Implement Stage 5 as immutable trial-only continuation
- Decision: Keep the Stage 5 controller's committed state as independent
  bytes-backed float64 arrays, hash dtype/shape/bytes before and after every
  attempt, rebuild fixed-damage mechanics views for both mechanics solves, and
  retain a reduced cutback increment after acceptance without an automatic
  growth rule. Trial histories are always recomputed from the unchanged
  committed history, and the final mechanics state is accepted directly after
  the complete gated sweep.
- Why: These rules make rollback auditable by value rather than object identity,
  prevent lazy fixed-view caches or trial history from entering committed state,
  and avoid inventing an adaptive load-growth policy not specified for Stage 5.
  A direct final mechanics/damage pairing also prevents a standalone damage
  re-solve from producing a state that was never mechanically checked.
- Alternatives rejected: writable read-only-flag arrays, retaining fixed views
  in accepted records, accumulating trial history across stagger iterations,
  adding a final damage-only solve, or silently growing a cutback increment.
- Review status: Implementation evidence only; rounded-slot pilot and
  independent Sol review remain pending. No production horizon or Stage 5
  approval is claimed.
- Related: `examples/fracture2d/nonlocal_baseline.py`,
  `tests/test_fracture2d_nonlocal_continuation.py`,
  `docs/tasks/fracture2d-direct-nonlocal.md`, `docs/PROGRESS.md`.

## 2026-08-21 — Make Stage 5 metadata and archive contracts explicit
- Decision: Build immutable `ContinuationMetadata` into every strict result and
  require `SupplementalArchiveMetadata` when writing an archive. Validate the
  frozen model/state pairing, accepted residuals, triangle indices, first
  moments, and all required provenance fields before serialization. Keep the
  archive numerical and JSON-compatible so it can be loaded with
  `allow_pickle=False`.
- Why: The controller cannot infer mesh triangles, quadrature provenance, or
  visibility documentation from an operator alone. Explicit metadata makes
  accepted states auditable and prevents a valid-looking archive from mixing
  fields from different clouds or configurations.
- Alternatives rejected: optional archive metadata, inferred/default
  quadrature or provenance, pickle/object-array serialization, and relying only
  on object identity or rounded residual checks for state pairing.
- Review status: Independent dense BA/damage/Amor clean-step comparison,
  constrained-tangent symmetry/positivity, force/moment rejection, and forged
  accepted-step tests are now present and pass on the manufactured cloud. A
  rounded-slot pilot and independent Sol review remain pending; no production
  horizon or Stage 5 approval is claimed.
- Related: `examples/fracture2d/nonlocal_baseline.py`,
  `tests/test_fracture2d_nonlocal_continuation.py`,
  `examples/fracture2d/README.md`,
  `docs/tasks/fracture2d-direct-nonlocal.md`.

## 2026-08-22 — Hold Stage 5 approval after independent audit
- Decision: Keep the fracture2d Stage 5 strict continuation pending independent
  review. Do not approve the implementation or run a rounded-slot pilot until
  accepted-history pairing, fixed-history residual reconstruction, mechanics
  relative-residual validation, and structural exception propagation are closed.
- Why: The CPU-only focused suite (`24` tests) and complete Stage 0-5 list
  (`130` tests) pass, but targeted forged-result probes accepted altered
  `AmorHistoryResult.previous_history`, altered fixed-history Euclidean and
  strong residuals with matching diagnostics, and an altered mechanics relative
  residual with matching diagnostics/audit. A diagnostic `ValueError` is also
  swallowed in `_failure_diagnostics`. These are audit-integrity failures even
  though the live force/moment gates and solver-time mechanics validator pass.
- Alternatives rejected: treating passing manufactured curves as proof of
  result integrity, accepting summary fields when their physical vectors are
  forged consistently, or masking the diagnostic structural error to preserve a
  cutback path. No production horizon is selected.
- Related: `docs/tasks/fracture2d-direct-nonlocal.md`,
  `examples/fracture2d/nonlocal_baseline.py:1724-1772`, `:1926-2022`,
  `:2528-2537`, `:2649-2663`, and
  `tests/test_fracture2d_nonlocal_continuation.py`.

## 2026-08-22 — Approve the Stage 5 controller implementation
- Decision: Close the independent Stage 5 implementation audit after adding
  authoritative mechanics replay from frozen metadata, exact Amor and
  fixed-history replay, bytes-backed nested results, complete attempt/state
  pairing, explicit graph/archive invariants, and structural exception
  propagation. Keep the rounded-slot pilot and production-horizon selection as
  separate deferred gates.
- Why: Every finding from the initial review and subsequent re-reviews is closed.
  Independent forged-result probes are rejected, archives remain compatible
  with `allow_pickle=False`, and the explicit Stage 0--5 CPU list passes all
  `134` tests. Changed-module compilation and `git diff --check` also pass.
- Alternatives rejected: approving from clean-path tests alone, weakening
  residual/conservation gates, treating structural `ValueError` as a numerical
  cutback, or using a diagnostic horizon as a production choice.
- Related: Resolves “Hold Stage 5 approval after independent audit” for the
  controller implementation only; `docs/tasks/fracture2d-direct-nonlocal.md`,
  `examples/fracture2d/nonlocal_baseline.py`,
  `examples/fracture2d/nonlocal_mechanics.py`, and
  `tests/test_fracture2d_nonlocal_continuation.py`.

## 2026-08-22 — Authorize a temporary rounded-slot Stage 5 pilot
- Decision: Run the first strict rounded-slot continuation only as a temporary
  pre-localization diagnostic at `delta=0.048`, from zero to `u_bar=1e-3` with
  initial increment `1e-4`, minimum increment `1e-5`, cutback factor `0.5`, and
  no increment growth. Keep all values outside a production configuration.
- Why: `delta=0.048` has the strongest existing rounded-cloud BA conditioning
  evidence, and `u_bar=1e-3` is the highest load already checked by the Stage 4
  fixed-subproblem diagnostic. Loading/cutback and CG values are inherited from
  existing fracture2d configurations; all convergence gates reuse the reviewed
  Stage 5 manufactured-cloud settings.
- Alternatives rejected: selecting a production horizon from Stage 1--4
  diagnostics, extending immediately into peak/crack growth, relaxing gates in
  response to a failed trial, or adding an adaptive growth rule.
- Related: `docs/tasks/fracture2d-direct-nonlocal.md`,
  `/tmp/opencode/stage3_cloud_float64.npz`, and
  `/tmp/opencode/stage4_rounded_diagnostic.py`.

## 2026-08-23 — Approve the rounded-slot pre-localization pilot
- Decision: Accept the temporary `delta=0.048`, `u_bar=0` to `1e-3` strict run
  as rounded-slot Stage 5 pre-localization evidence. Do not select a production
  horizon or infer peak/crack-growth, refinement, or PINN approval from it.
- Why: All ten increments were accepted without cutback, every strict gate and
  CG solve passed, damage/history were monotone and in range, conservation
  defects stayed near float64 precision, all 148 archive arrays loaded with
  `allow_pickle=False`, and independent replay found no numerical mismatch.
  Stage 4 and warning-mode FEM trends remained consistent.
- Alternatives rejected: treating the run as production validation, weakening
  the two closest gates (`0.7791` damage residual and `0.7442` history), or
  extending directly through crack growth without addressing the `6614 s` CPU
  cost and selecting an explicit staged strategy.
- Related: `/tmp/opencode/stage5_rounded_pilot_delta048.npz`,
  `/tmp/opencode/stage5_rounded_pilot_delta048.json`, and
  `docs/tasks/fracture2d-direct-nonlocal.md`.

## 2026-08-23 — Optimize direct mechanics before peak continuation
- Decision: Preserve the approved mechanics equations and strict gates while
  first adding validated CG warm starts and an initial-residual early exit,
  then vectorizing the BA hot paths and implementing a separate all-device JAX
  float64 mechanics/CG backend. Keep NumPy/SciPy as the independent reference.
- Why: The rounded pilot spent nearly all `6614 s` in 110 mechanics solves, each
  taking 243 zero-start CG iterations over 72,934 directed BA edges. Adjacent
  sweeps often solve the same fixed-damage/load system, so algorithmic reuse is
  required before hardware migration; GPU acceleration requires JIT-compiled
  tangent, preconditioner, and CG without per-iteration host/device transfers.
- Alternatives rejected: loosening CG or continuation tolerances, replacing the
  CPU reference, sending only one NumPy callback to GPU under SciPy CG, silently
  downcasting the float64 graph, or beginning a long peak run at the measured
  CPU cost.
- Related: `examples/fracture2d/nonlocal_mechanics.py`,
  `examples/fracture2d/nonlocal_baseline.py`, and
  `docs/tasks/fracture2d-direct-nonlocal.md`.

## 2026-08-23 — Approve the standalone JAX mechanics backend for integration review
- Decision: Accept the separate float64 JAX fixed-damage BA mechanics/PCG
  backend for a later, independently reviewed Stage 5 adapter. Keep the
  NumPy/SciPy implementation and fresh CPU fixed-damage replay authoritative for
  every acceptance gate and archived physical diagnostic; JAX iteration and
  field diagnostics remain non-authoritative execution data.
- Why: Manufactured CPU/GPU tests match the frozen energy, residual, tangent,
  exact block-Jacobi, constraints, and warm-seed policy. On the diagnostic
  `delta=0.048` rounded cloud, one fixed-shape compile serves changing damage,
  load, tolerances, and seeds; a 243-iteration RTX 5090 solve takes about
  `75 ms` after compilation and passes the strict fresh CPU residual replay.
  Independent review found no remaining standalone-backend blocker.
- Alternatives rejected: replacing the CPU reference; treating GPU diagnostics
  as authoritative; using an occurrence-count-scaled heuristic tolerance to
  compare cancelled reductions; silently downcasting; or directly inserting
  the backend into Stage 5 without adapter-level rollback, replay, and archive
  tests.
- Related: `examples/fracture2d/nonlocal_mechanics_jax.py`,
  `tests/test_fracture2d_nonlocal_mechanics_jax.py`, and
  `docs/tasks/fracture2d-direct-nonlocal.md`.

## 2026-08-23 — Approve CPU-replayed JAX Stage 5 pre-localization
- Decision: Accept the restricted `JaxStage5MechanicsAdapter` and its temporary
  full `delta=0.048`, `u_bar=0` to `1e-3` rounded-slot run as pre-localization
  evidence. Keep every physical gate and archive field based on a fresh CPU
  fixed-damage replay. Do not approve peak continuation or a production horizon.
- Why: The adapter receives immutable damage/seed copies, preserves rollback and
  structural-error semantics, records backend provenance, and passes CPU/GPU
  continuation tests. The full run reached its endpoint after one correctly
  audited rejection/cutback; all accepted gates passed, fresh CPU reconstruction
  exactly reproduced mechanics fields, and the result remained close to the
  prior CPU path while reducing wall time from `6614 s` to `430 s`.
- Alternatives rejected: allowing generic mutable solver hooks; archiving
  unlabeled GPU iteration counts; weakening the close history/damage-residual
  gates; discarding the rejected attempt; removing CPU replay to maximize speed;
  or extending directly into crack growth.
- Related: `/tmp/opencode/stage5_rounded_jax_pilot_delta048.npz`,
  `/tmp/opencode/stage5_rounded_jax_pilot_delta048.json`,
  `examples/fracture2d/nonlocal_baseline.py`, and
  `docs/tasks/fracture2d-direct-nonlocal.md`.

## 2026-08-23 — Add a separate all-gate warning diagnostic mode
- Decision: Add `run_warning_continuation` as an exploratory mode separate from
  strict continuation. After the full stagger budget, any failed finite
  post-sweep acceptance gate may be bypassed only when a complete finite
  mechanics-before/damage/mechanics-after/history candidate exists. Archive the
  failed gates and keep `diagnostics.passed=False` for warning-accepted states.
- Why: The user selected staged progression toward peak and explicitly allowed
  all acceptance gates to become warnings. A separate result/archive contract
  permits curve exploration without relabeling a non-strict state as accepted
  strict evidence. The first authorized endpoint is `u_bar=2e-3`.
- Alternatives rejected: changing the default strict controller; clipping or
  projecting warning fields; silently treating warnings as passed; warning-
  committing a missing/non-finite mechanics candidate; or extending directly to
  the known local-FEM transition near `5.5e-3` in one run.
- Related: `examples/fracture2d/nonlocal_baseline.py`,
  `tests/test_fracture2d_nonlocal_continuation.py`, and
  `docs/tasks/fracture2d-direct-nonlocal.md`.

## 2026-08-23 — Anchor warning evidence to dtype and solve provenance
- Decision: Require authoritative replay comparisons to preserve exact dtype,
  and bind every CPU/JAX `MechanicsSolveResult` backend label to a validated
  process-local provenance token that is checked against continuation metadata.
- Why: Independent adversarial review showed that post-construction float32
  diagnostic rebinding and coherent relabeling of metadata plus mechanics-result
  strings could otherwise survive relational value checks.
- Alternatives rejected: accepting numerically close float32 evidence, trusting
  mutually consistent backend strings without execution provenance, or treating
  arbitrary post-construction mutation as outside the archive contract.
- Related: `examples/fracture2d/nonlocal_mechanics.py`,
  `examples/fracture2d/nonlocal_mechanics_jax.py`,
  `examples/fracture2d/nonlocal_baseline.py`, and
  `tests/test_fracture2d_nonlocal_continuation.py`.

## 2026-08-23 — Approve the staged warning checkpoint through 2e-3
- Decision: Accept the fresh zero-start `delta=0.048` CPU-replayed JAX warning
  run through `u_bar=2e-3` as a temporary pre-peak diagnostic checkpoint only.
  Require explicit authorization before choosing a farther endpoint.
- Why: All 20 nominal `1e-4` increments actually passed every strict gate, all
  fields were finite and monotone, independent CPU replay reconstructed the
  mechanics exactly, and the 164-field schema-2 archive passed status, digest,
  energy, work, and `allow_pickle=False` audits. Final reaction was
  `253.5305191` and maximum damage was `0.0513580`; reaction remained monotone.
- Alternatives rejected: calling the run a warning-bypass demonstration (none
  occurred), inferring peak/crack growth, selecting `delta=0.048` for
  production, extending beyond `2e-3`, or beginning refinement/PINN work from
  this checkpoint.
- Related: `/tmp/opencode/stage5_rounded_jax_warning_delta048_2e3.npz`,
  `/tmp/opencode/stage5_rounded_jax_warning_delta048_2e3.json`, and
  `docs/tasks/fracture2d-direct-nonlocal.md`.

## 2026-08-23 — Use semantic names for persistent diagnostics and adapters
- Decision: Remove roadmap-stage and temporary-pilot labels from persistent
  identifiers. The JAX continuation adapter is now
  `JaxCPUReplayMechanicsAdapter`; the bond-associated algebra test is now
  `test_fracture2d_bond_associated_algebra.py` with class
  `Fracture2DBondAssociatedAlgebraTest`; and fracture1d pilot modules/functions
  now use `*_diagnostics` and `run_*_diagnostic` names.
- Why: `Stage 0`/`Stage 5` describe project history, while the stable runtime
  concepts are bond-associated algebra and CPU-authoritative JAX replay. The
  diagnostic tools are persistent utilities, so their names should describe
  their computation rather than their temporary experiment status.
- Alternatives rejected: leaving stage numbers in public API names, adding a
  permanent compatibility alias that preserves the misleading name, or
  mechanically renaming semantic `baseline`, `warning`, and `failure_stage`
  terms. Historical decision entries and existing temporary archive filenames
  remain unchanged for provenance.
- Related: `examples/fracture2d/nonlocal_mechanics_jax.py`,
  `examples/fracture2d/nonlocal_baseline.py`,
  `examples/fracture1d/fixed_history_diagnostics.py`,
  `examples/fracture1d/transition_diagnostics.py`,
  `examples/fracture1d/pseudo_time_diagnostics.py`, and the renamed tests.

## 2026-08-23 — Authorize the next staged checkpoint through 3e-3
- Decision: Continue the temporary `delta=0.048` warning-mode diagnostic with a
  fresh zero-start run through `u_bar=3e-3`, retaining nominal `du=1e-4`, all
  existing gates, and CPU-authoritative replay. Stop at the endpoint or any hard
  failure and audit before selecting another load.
- Why: The approved `2e-3` checkpoint remained monotone and pre-peak, with all
  steps passing strictly and maximum damage `0.0513580`. Advancing by one more
  `1e-3` checkpoint preserves staged observation without jumping to the local
  FEM transition region near `5.5e-3`.
- Alternatives rejected: resuming from an unaudited mutable state, changing the
  horizon/increment/gates, jumping directly to the local-FEM transition, or
  treating this diagnostic horizon as a production choice.
- Related: `docs/tasks/fracture2d-direct-nonlocal.md` and
  `docs/PROGRESS.md`.

## 2026-08-23 — Snap ulp-scale load remainders to the endpoint
- Decision: When the next nominal continuation target lies within eight ulps of
  `load_final`, use the exact endpoint before solving and record the resulting
  full nominal increment. This applies equally to strict and warning modes and
  does not change any physical or convergence gate.
- Why: Repeated addition of `du=1e-4` left a three-ulp remainder before `3e-3`.
  Treating that remainder as a separate accepted load step produced a
  meaningless reaction slope from a `1.30e-18` denominator despite an otherwise
  valid CPU-replayed solution.
- Alternatives rejected: accepting the duplicate endpoint solve as a physical
  increment, adding a broad relative load tolerance, changing the load grid, or
  filtering the bad slope only during plotting.
- Related: `examples/fracture2d/nonlocal_baseline.py` and
  `tests/test_fracture2d_nonlocal_continuation.py`.

## 2026-08-23 — Approve the staged warning checkpoint through 3e-3
- Decision: Accept the snapped fresh-zero `delta=0.048` warning run through
  exact `u_bar=3e-3` as temporary staged pre-peak evidence only. Require a new
  explicit endpoint decision before farther loading.
- Why: The rerun has exactly 30 nominal steps, positive reaction increments and
  work, exact CPU mechanics reconstruction, finite monotone damage/history, and
  an independently audited schema-2 archive. Two complete finite states were
  correctly marked `warning_accepted`; final reaction is `371.4762173` and
  maximum damage is `0.1204572` at the notch tip.
- Alternatives rejected: approving the prior unsnapped curve, calling warning
  trigger loads deterministic, inferring a peak or propagated crack, selecting
  `delta=0.048` for production, or continuing beyond `3e-3` without another
  checkpoint decision.
- Related: `/tmp/opencode/fracture2d_d048_warning_u3e3.npz`,
  `/tmp/opencode/fracture2d_d048_warning_u3e3.json`, and
  `docs/tasks/fracture2d-direct-nonlocal.md`.

## 2026-08-23 — Authorize a complete warning run to the fracture2d u_final
- Decision: Temporarily defer Stage 6 refinement and run a fresh zero-start
  `delta=0.048` warning continuation to the fracture2d default
  `loading.u_final=0.0075`. Keep CPU-authoritative JAX replay, all existing
  gates, fixed graph/operator data, and no clipping, projection, remeshing, or
  dynamic bond deletion. Stop at `0.0075` or any hard failure.
- Why: The user wants one complete load path before refinement, with special
  attention to the crack-growth regime and whether the frozen nonlocal operator
  remains numerically auditable as damage localizes and spreads.
- Alternatives rejected: silently using a different `u_final`, resuming from
  the `3e-3` state instead of fresh zero, changing the horizon/gates, replacing
  the fixed graph during loading, or treating warning-accepted states as strict
  physical approval.
- Related: `examples/fracture2d/configs/solver_base.py`,
  `examples/fracture2d/nonlocal_baseline.py`, and
  `docs/tasks/fracture2d-direct-nonlocal.md`.

## 2026-08-23 — Hold runner refactor after raw u_final field inspection
- Decision: Do not begin the strict/warning runner refactor or approve the full
  `u_final=0.0075` curve yet. Treat the retained per-step raw checkpoint as
  `validated=False` diagnostic evidence and investigate boundary-driven damage,
  healing, and the crack-growth operator response first.
- Why: The raw path peaks at `u_bar=0.0055`, but final maximum damage occurs on
  the loaded top boundary rather than the notch: 38 of 51 top particles have
  `d>0.5`, while notch-tip damage is about `0.587`. Healing occurs at
  `0.0056`--`0.0058`, and the final damage residual infinity norm is about
  `0.0608`. These are material physical/numerical concerns, not merely plotting
  noise.
- Alternatives rejected: proceeding with API cleanup before understanding the
  fields, presenting the raw checkpoint as a validated archive, interpreting
  the reaction drop as clean notch-crack propagation, or hiding healing and
  residual defects with clipping.
- Related: `/tmp/opencode/fracture2d_d048_warning_ufinal_0075.raw.npz`,
  `/tmp/opencode/fracture2d_ufinal_raw_overview.png`, and
  `/tmp/opencode/fracture2d_ufinal_raw_damage_evolution.png`.

## 2026-08-23 — Diagnose mechanics/history boundary concentration before changing damage BCs
- Decision: Investigate the nonlocal mechanics/history response at the loaded
  top boundary before proposing a new damage boundary condition or modifying
  the fixed graph. Keep the runner refactor held.
- Why: FEM and nonlocal mechanics impose the same hard displacement DOFs, and
  neither model hard-constrains damage. FEM uses natural zero flux; the nonlocal
  damage solve uses graph-truncated families. However, at `u=0.0056` the
  nonlocal top history jumps to about `3875` while FEM top-adjacent history is
  about `4.97`. Re-solving the nonlocal damage equation exactly with that raw
  history reproduces top `d_max=0.9974`, while a uniform-history solve remains
  constant despite truncation. This points first to mechanics/history boundary
  concentration and degradation feedback, with damage-family truncation and
  warning underconvergence as secondary contributors.
- Alternatives rejected: claiming the hard masks differ from FEM, imposing
  `d=0` at the top without derivation, blaming damage truncation alone, or
  interpreting the post-peak drop as notch-driven crack growth.
- Related: `examples/fracture2d/nonlocal_mechanics.py`,
  `examples/fracture2d/nonlocal_baseline.py`,
  `/root/autodl-tmp/data/fracture2d/local_fem_point_anchor/results_point_anchor.npz`,
  and `docs/tasks/fracture2d-direct-nonlocal.md`.

## 2026-08-24 — Reject the top-damage warning branch as converged evidence
- Decision: Do not approve the `delta=0.048`, `u=0.0056` warning-committed state
  or extend its current staggered branch by merely increasing the iteration
  budget. Require a sourced nonlocal Dirichlet/free-surface boundary choice
  before implementing the next counterfactual diagnostic.
- Why: An exact 20-sweep replay reproduces the raw endpoint and shows ordered
  positive feedback: top tensile strain/history overtakes the notch tip in
  sweeps 8--9 and top damage in sweep 10. Top `H_used` then grows `20.4x` while
  tip `H_used` grows `1.05x`, and the final `H_next` damage residual is `142.64`.
  Each fixed-`H_used` damage solve is accurate to about `4e-12`, so the branch
  is accelerating away from the coupled fixed point rather than converging
  slowly because of an inaccurate linear solve.
- Alternatives rejected: treating the 20th finite candidate as physical crack
  growth, increasing the same undamped stagger budget, imposing top `d=0`
  without derivation, changing the graph silently, or beginning runner/Stage 6
  work before resolving the boundary model.
- Related: `/tmp/opencode/fracture2d_warning_transition_sweeps.json` and
  `docs/tasks/fracture2d-direct-nonlocal.md`.

## 2026-08-24 — Do not guess a fictitious-collar fracture2d operator
- Decision: Do not implement the proposed Behzadinasab-style mechanics-only
  collar counterfactual for the current fracture2d boundary-value problem until
  its mixed-component boundary, degradation, energy, reaction, and quadrature
  choices are independently derived and approved.
- Why: Behzadinasab Part I/II sources separate kinematic and stress-divergence
  neighborhoods and zero-stress fictitious free-surface nodes, but do not define
  a node with essential `u_y` and free/natural `u_x`, the point `u_x` anchor,
  free displacement components through the collar, phase-field degradation of
  fictitious nodes/bonds, a conservative global discrete energy or support
  reaction, or exact rounded-slot collar particles and volumes. Filling these
  gaps silently would create a new project-owned model rather than reproduce a
  published boundary treatment.
- Alternatives rejected: assigning zero/affine tangential collar displacement,
  extending real damage to ghosts ad hoc, calling constrained strong residuals
  reactions without a work derivation, or adapting the paper's centroid mesh to
  Gmsh vertex particles without a quadrature study.
- Related: Behzadinasab--Trask--Bazilevs Part I (DOI
  `10.1007/s42102-020-00040-z`), Behzadinasab--Foster--Bazilevs Part II (DOI
  `10.1007/s42102-020-00039-6`), and
  `docs/tasks/fracture2d-direct-nonlocal.md`.

## 2026-08-24 — Authorize only the phase-measured frozen grip prototype
- Decision: Implement Candidate A2 as a separate NumPy/SciPy float64 mechanics
  diagnostic at `delta=0.048`. Preserve the original material cloud and raw
  field IDs exactly; add top/bottom elastic-grip triangle measures, use union
  volumes for BA kinematics, use phase-measured center weights
  `V_m*g(d_m)+V_g*g(0)`, constrain only outer-face `u_y` plus the original
  anchor, and compare at matched physical interface opening.
- Why: This project-owned variational extension resolves the earlier collar's
  interface damage/volume conflict while retaining an explicit conservative
  energy, symmetric fixed-damage tangent, work-conjugate reaction, and exact
  raw-field transfer. Two independent reviews found the corrected contract
  executable with hard geometry, reference-algebra, spectrum, work, and
  conservation gates.
- Alternatives rejected: Behzadinasab zero-stress ghosts, which do not define
  this mixed-component phase-field BVP; remeshing the material cloud; one damage
  value for a shared material/grip center volume; vertically fixing the full
  grip volume; or starting coupled continuation before the frozen diagnostic.
- Related: `docs/tasks/fracture2d-nonlocal-boundary-formulation.md`.

## 2026-08-24 — Complete Candidate A2 numerical diagnostic gates
- Decision: Retain Candidate A2 as a separate, nonproduction frozen-mechanics
  diagnostic. Mark its implementation numerically admissible but its boundary
  counterfactual diagnostically unsuccessful until an independent final Sol
  review records closure; do not promote it to coupled fracture or production
  mechanics.
- Why: The manufactured gates and both authorized union-cloud spacings passed.
  The final `h_g=0.02` replay has 3701 particles, 79060 directed edges,
  `R_eff=667.0949973747717`, a positive constrained spectrum, symmetric tangent,
  and reaction-work relative error `1.9522e-10`. The matched-opening transition
  keeps the reaction within the 10% FEM guard and the global driver near the
  notch, but reduces the material-top `H_used` error by only about 3.8%, below
  the required 50%.
- Alternatives rejected: treating the transition as physical validation,
  relaxing the top-driver criterion, changing the collar or damage model after
  the fact, or starting coupled continuation before the boundary formulation
  receives explicit final review.
- Related: `docs/tasks/fracture2d-nonlocal-boundary-formulation.md`,
  `examples/fracture2d/nonlocal_grip_diagnostic.py`, and the authorized
  `/tmp/opencode/fracture2d_candidate_a2_*_transition_authorized.*` artifacts.

## 2026-08-24 — Close Candidate A2 as a split-result counterfactual
- Decision: Accept Candidate A2's isolated fixed-damage implementation and hard
  numerical gates, but reject promotion to coupled fracture or production. Keep
  the predeclared frozen-`H_used` usefulness failure and separately record the
  strong instantaneous-mechanics improvement.
- Why: Independent review and corrected tests validate the phase-measured energy,
  union visibility, affine constraints, spectra, work, reaction balance, and
  artifacts. Both spacings reduce instantaneous material-top `Psi_plus` from
  `78.8966` to `5.6326--6.2479`. Nevertheless, transplanted top history is
  already `76.1040`, so `H_used=max(H_m,Psi_plus)` improves its FEM error by only
  `3.7654%`, below the declared `50%` criterion.
- Alternatives rejected: relabeling the failed `H_used` gate after seeing the
  instantaneous result, claiming the collar is physically validated, deleting
  the useful diagnostic module, or starting a clean coupled continuation without
  a separate material-damage/grip-mechanics formulation.
- Related:
  `docs/tasks/archive/fracture2d-nonlocal-boundary-formulation.md` and
  `/tmp/opencode/fracture2d_candidate_a2_canonical_20260824_r2/`.

## 2026-08-24 — Derive a clean-start A2 material-damage coupling
- Decision: Open a separate formulation task for strict clean-zero-start
  continuation using Candidate A2 phase-measured grip mechanics and the existing
  material-only AT2 damage/history graph. Do not implement until the stagger,
  matched-opening, reaction/work, rollback, and replay contracts receive an
  independent numerical-method GO.
- Why: The frozen A2 diagnostic nearly removes instantaneous top mechanics
  concentration, but inherited raw history prevents testing whether the collar
  would avoid contamination when present from the beginning. A clean-start path
  is the discriminating experiment.
- Alternatives rejected: reusing the contaminated raw history, silently
  generalizing the current material-only continuation metadata to unequal
  mechanics/damage clouds, enabling warning commits, or treating A2 as production.
- Related: `docs/tasks/fracture2d-a2-coupled-formulation.md` and
  `docs/tasks/archive/fracture2d-nonlocal-boundary-formulation.md`.

## 2026-08-25 — Tighten A2 pilot mechanics to its work-reference precision
- Decision: Rerun only the strict clean-start A2 endpoint `opening=1e-4` with
  primary mechanics CG `rtol=1e-10`, `atol=1e-13`, equal to the independent
  work-reference solves. Keep the `1e-7` work gate and every other equation,
  increment, rollback, and acceptance setting unchanged.
- Why: The first run correctly stopped after accepting `5e-5`; at `1e-4` its
  primary relative residual was `9.69e-8` and mechanics-after work mismatch was
  about `5e-7`. Independent same-ray/tight-solve diagnostics reduced the mismatch
  to about `1e-11`, showing solve precision rather than energy/reaction algebra
  caused the incompatibility.
- Alternatives rejected: relaxing the work gate, adding an underived
  residual-aware reaction tolerance, accepting the failed state, resuming from
  `5e-5`, or authorizing a higher endpoint.
- Related: `docs/tasks/fracture2d-a2-coupled-formulation.md` and
  `/tmp/opencode/fracture2d_a2_first_sweep_work_reconstruction.json`.

## 2026-08-26 — Use engineering volume grips for the direct crack reference
- Decision: Stop pursuing matched physical-opening equivalence for the direct
  reference. Use intact external top/bottom volume grips with hard vertical
  displacement, free lateral motion, and one horizontal anchor. Keep
  damage/history material-only and focus validation on the notch-region operator
  and solution.
- Why: The direct solver is needed as a crack-region nonlocal reference, not as a
  high-fidelity boundary-reaction replica of FEM. The notch is roughly ten
  diagnostic horizons from the grips, so a controlled boundary approximation is
  acceptable provided no artificial interface localization reaches the material
  solution.
- Alternatives rejected: continuing the expensive matched-opening controller,
  relaxing numerical equilibrium/damage gates, reintroducing material-boundary
  hard loading that caused top history concentration, or imposing damage on the
  grips/material interface.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md` and
  `docs/tasks/fracture2d-a2-coupled-formulation.md`.

## 2026-08-25 — Preserve the mechanics target used by replay
- Decision: Archive the target opening actually passed to each fixed-damage A2
  mechanics solve separately from the continuation attempt target. Derive the
  production and independent-replay opening gates against the attempt target,
  while reconstructing displacement, reaction, slope, and work primitives at
  the archived mechanics target.
- Why: A numerical opening-gate rejection must be independently replayable when
  a solver returns a finite, equilibrated response at a deliberately off-target
  physical opening. Replaying that response at the attempt target would compare
  different displacement fields and could only be made to pass by forging a
  gate flag.
- Alternatives rejected: storing only the opening pass boolean, flipping the
  pass flag in a test double, or treating the achieved opening as the mechanics
  target; each loses the primitive provenance needed to distinguish a physical
  opening miss from a forged classification.
- Related: `examples/fracture2d/nonlocal_a2_coupled.py`,
  `tests/test_fracture2d_nonlocal_a2_coupled.py`, and
  `docs/tasks/fracture2d-a2-coupled-formulation.md`.

## 2026-08-26 — Hold engineering-grip smoke on replay audit
- Decision: Do not approve or extend the engineering-grip smoke archive until
  replay handles the exact initial state and its independent mechanics
  diagnostics have a reviewed operation-order contract. Keep the production
  equilibrium, conservation, damage, history, and localization gates
  unchanged.
- Why: `/tmp/opencode/fracture2d_engineering_grip_strict_1e4.npz` contains a
  valid strict continuation result and a correctly paired accepted state, but
  replay applies the solved-mechanics gate to state index `0`, where the exact
  zero solution produces a `1.0514815413942755e-11` loop-order residual against
  a `1e-13` absolute floor. The accepted state independently passes its
  `7.0742933779381594e-09` replay limit. After exempting state zero, the
  recorded production free-residual norms still differ from the independent
  loop reconstruction by up to `3.1830686624979701e-13`, exceeding the current
  `1e-15` replay comparator floor. Static BA data, phase weights, constraints,
  and state/step digests are not defective.
- Alternatives rejected: weakening the production CG/equilibrium gate,
  accepting the archive from the production result alone, changing phase
  weights or grip constraints without evidence, or treating the zero-state
  roundoff as physical non-equilibrium.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md`,
  `examples/fracture2d/nonlocal_engineering_grip.py:3569-3697,4137-4192,5327-5347`,
  and `/tmp/opencode/fracture2d_engineering_grip_strict_1e4.npz`.

## 2026-08-26 — Hold after scalable engineering-grip replay review
- Decision: Keep the engineering-grip smoke at **NO-GO** and authorize no
  longer continuation until the replay contract resolves its work-check,
  graph-integrity, metadata, and attempt-audit gaps. Do not change the declared
  mechanics/damage equations or production acceptance gates as a replay fix.
- Why: The independent vectorized BA/tangent algebra, particle block-Jacobi
  formula, sparse AT2 volume weighting, constrained spectrum, and current
  manufactured/temporary smoke checks are numerically consistent. However,
  replay accepted a work-limit/config mismatch, malformed or inconsistent graph
  provenance, and unvalidated static metadata; it also omits production CG
  status from the reconstructed mechanics gate.
- Alternatives rejected: treating the 16 focused tests and temporary large
  replay as sufficient evidence, weakening the production residual gate, or
  authorizing a longer physical curve before the archive evidence contract is
  independently trustworthy.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md`,
  `examples/fracture2d/nonlocal_engineering_grip.py:3942-3953,3978-3989,6174-6255,6345-6359`,
  and `/tmp/opencode/fracture2d_engineering_grip_strict_1e4.npz`.

## 2026-08-26 — Keep forgery signals outside replay roundoff budget
- Decision: Set direct forged free-residual diagnostics to a perturbation well
  above the independent constituent-scale operation-order bound, while keeping
  sub-bound differences accepted by the replay comparator.
- Why: The independent loop reconstruction legitimately differs from the
  production reduction by up to the declared cancellation-aware float64 bound;
  changing a residual from its recorded value to `1e-300` can remain inside
  that bound and is not a meaningful forgery signal.
- Alternatives rejected: weakening the reviewed replay bound to reject every
  representational change, or treating all sub-bound operation-order drift as
  archive corruption.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md`,
  `examples/fracture2d/nonlocal_engineering_grip.py`, and
  `tests/test_fracture2d_nonlocal_engineering_grip.py`.

## 2026-08-27 — Make the cross-backend phase-weight test roundoff-aware
- Decision: Keep the JAX phase-center formula unchanged and compare its JIT
  result with the NumPy phase-measure reference using an explicit eight-unit-
  roundoff float64 bound. Also select the CPU test device lazily when no GPU is
  available.
- Why: CPU XLA fusion produced a maximum `6.94e-18` last-bit difference from
  the separately evaluated NumPy expression, while the GPU result and the
  algebraic formula remained correct. The test's eager `jax.devices("cpu")`
  default was also evaluated even when a GPU was present, so CUDA-only JAX
  environments failed before running any test.
- Alternatives rejected: changing the production energy/phase formula to force
  compiler-specific bitwise ordering, weakening the physical/replay gates, or
  treating compiler roundoff as a formulation discrepancy.
- Related: `tests/test_fracture2d_nonlocal_engineering_grip_jax.py` and
  `docs/tasks/fracture2d-engineering-grip-baseline.md`.

## 2026-08-27 — Validate the engineering-grip v2 smoke archive
- Decision: Record the exact `delta=0.048`, `h_g=0.02`, `u_bar=1e-4` run with
  the restricted CPU-replayed JAX adapter and independent v2 archive replay as
  passing implementation evidence only. Keep the formulation, production gates,
  nonproduction label, and no-longer-continuation hold unchanged.
- Why: The fresh `cuda:0` run accepted one strict step and three complete
  sweeps; independent replay reconstructed the static graph, production CG
  status, mechanics, work, conservation, damage/history, state pairing, and
  crack localization without the former v1/replay-contract failures.
- Alternatives rejected: treating the stale v1 archive as current evidence,
  accepting the GPU diagnostics without CPU replay, or extending the physical
  curve before primary review.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md` and
  `/tmp/opencode/fracture2d_engineering_grip_v2_smoke.npz`.

## 2026-08-27 — Approve the engineering-grip v2 smoke checkpoint
- Decision: Accept the exact `delta=0.048`, `h_g=0.02`, `u_bar=1e-4`
  engineering-volume-grip run and its v2 archive as a strict nonproduction smoke
  checkpoint. Keep the GPU fixed-damage solve as an accelerator and the fresh
  CPU evaluation plus independent archive replay authoritative. Authorize no
  larger endpoint or production horizon.
- Why: The run accepted one step in three complete sweeps with no cutback; the
  one-time work error was `1.2992513794712763e-12` against `1e-7`. Primary CPU
  replay completed in `220.11034105997533 s`, reproduced every strict gate, and
  confirmed crack-region damage/history maxima exceed outside maxima. The v2
  source-bundle hash, production `cg_info=0`, sparse damage replay, matrix-free
  mechanics replay, and meaningful-forgery checks pass. The test-only
  JIT-versus-NumPy phase-weight defect is bounded by eight float64 roundoff units
  and does not enter authoritative acceptance fields.
- Alternatives rejected: rerunning the hour-scale CPU production CG path,
  weakening numerical gates, accepting stale v1 archives, treating GPU reduction
  order as authoritative physics, or extending directly into crack growth.
- Related: `/tmp/opencode/fracture2d_engineering_grip_v2_smoke.npz`,
  `/tmp/opencode/fracture2d_engineering_grip_v2_smoke.json`, and
  `docs/tasks/fracture2d-engineering-grip-baseline.md`.

## 2026-08-27 — Authorize engineering-grip loading through 3e-3
- Decision: Run one fresh-zero strict engineering-grip continuation to exact
  `u_bar=3e-3` with nominal `du=1e-4`, preserving the validated v2 smoke
  formulation, GPU-production/CPU-authoritative mechanics split, and every
  numerical and localization gate.
- Why: The exact `1e-4` checkpoint and independent replay pass, and the user
  explicitly requested the next staged endpoint to assess the developing
  reaction and notch localization before attempting the full load path.
- Alternatives rejected: resuming from an unaudited mutable state, changing the
  horizon/increment/gates, using warning acceptance, or jumping directly to
  `u_bar=0.0075`.
 - Related: `docs/tasks/fracture2d-engineering-grip-baseline.md`.

## 2026-08-28 — Isolate the stabilized engineering-grip controller

- Decision: Implement the convergence-first experiment in
  `examples/fracture2d/nonlocal_engineering_grip_stabilized.py` with the public
  names `StabilizedEngineeringGripConfig`,
  `run_stabilized_engineering_grip`,
  `write_stabilized_engineering_grip_archive`, and
  `replay_stabilized_engineering_grip_archive`. Use the distinct archive schema
  `fracture2d-engineering-grip-stabilized-v1` and require an explicit workflow
  `--continuation-strategy stabilized` selection.
- Why: A separate module and schema prevent the new relaxation state machine
  from changing the audited strict/bounded-warning/unbounded-warning behavior
  or making an experimental result look like an existing validated archive.
  The archive will retain the base engineering-grip primitives plus exact
  relaxation/backtracking provenance and independently reconstruct each
  fixed-point damage map before checking the committed fields.
- Alternatives rejected: modifying `_run_engineering_attempt` in place,
  reusing a warning archive schema, treating the relaxed damage iterate as an
  unrelaxed fixed-history solution, or making stabilization the default CLI
  strategy.
- Relation: This is an implementation/API choice for the authorized
  convergence-first diagnostic; the approved energy, damage equation, history
  update, graph, boundary conditions, and existing acceptance policies are
  unchanged.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md`.

## 2026-08-27 — Approve the engineering-grip 3e-3 checkpoint
- Decision: Accept the fresh-zero strict engineering-grip path through exact
  `u_bar=3e-3` and its independently replayed v2 archive as nonproduction
  pre-peak evidence. Authorize no farther endpoint or production horizon.
- Why: All 30 nominal increments passed without cutback, damage/history remained
  monotone, reaction remained strictly increasing, and final maximum damage
  stayed at the exact notch tip. Crack-region damage/history exceeded outside
  maxima by factors `2.79` and `5.63`; interface maxima remained substantially
  lower. Independent replay reproduced every strict state and gate.
- Alternatives rejected: calling the monotone softening path a completed crack
  curve, skipping archive replay, relaxing the close mechanics CG gate, or
  extending directly to `0.0075` without another staged decision.
- Related: `/tmp/opencode/fracture2d_engineering_grip_v2_u3e3.npz`,
  `/tmp/opencode/fracture2d_engineering_grip_v2_u3e3.json`,
  `/tmp/opencode/fracture2d_engineering_grip_v2_u3e3_analysis.json`, and
  `docs/tasks/fracture2d-engineering-grip-baseline.md`.

## 2026-08-27 — Compare engineering grips and FEM at prescribed nominal strain
- Decision: Plot both response curves with the established nominal definitions
  `epsilon_nom=u_bar/L_y` and `sigma_nom=R_y/L_x`, using prescribed `u_bar` for
  engineering grips and FEM. Keep material-interface opening as a separate
  diagnostic rather than silently changing the engineering curve's strain axis.
- Why: This is the exact normalization archived by the accepted FEM reference
  and the declared continuation coordinate of the engineering-grip model. It
  gives a traceable like-for-like prescribed-load comparison while retaining the
  boundary approximation caveat.
- Alternatives rejected: using interface opening only for the nonlocal curve,
  comparing unnormalized quantities under a stress-strain label, or presenting
  the FEM warning-mode curve as a strict fracture reference.
- Related: `/tmp/opencode/fracture2d_u3e3_nonlocal_vs_fem_fields.png`,
  `/tmp/opencode/fracture2d_u3e3_nonlocal_vs_fem_stress_strain.png`, and
  `docs/tasks/fracture2d-engineering-grip-baseline.md`.

## 2026-08-27 — Authorize the full engineering-grip endpoint
- Decision: Run one fresh-zero strict engineering-grip path to the configured
  complete endpoint `u_bar=0.0075`, preserving the validated `3e-3`
  formulation, increment, gates, GPU/CPU authority split, and archive replay.
- Why: The user reviewed the pre-peak fields/curve and explicitly requested
  completion of the loading process. Fresh-zero execution preserves the current
  audited controller because no archive-resume contract exists.
- Alternatives rejected: manually resuming from serialized state, changing the
  increment or horizon, warning-committing a failed sweep, or accepting a result
  without independent v2 replay.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md`.

## 2026-08-27 — Hold completion at the 20-sweep budget
- Decision: Keep the repeated `u_bar=0.0049` state rejected and stop before
  changing solver settings. Ask explicitly whether to increase only
  `max_stagger_iterations` from 20 to 30 while preserving every equation and
  acceptance gate.
- Why: Two fresh-zero runs stop after the same `0.00485` committed state. The
  final failed attempt remains notch-localized and its history error decreases
  geometrically to `1.60` times the strict limit on sweep 20, indicating an
  iteration-budget limit rather than the prior loaded-boundary runaway.
- Alternatives rejected: warning acceptance, relaxing the history tolerance,
  manually committing sweep 20, unreviewed archive resume, or silently changing
  the stagger budget.
- Related: `/tmp/opencode/fracture2d_engineering_grip_v2_u7p5e3_failure.npz`
  and `docs/tasks/fracture2d-engineering-grip-baseline.md`.

## 2026-08-27 — Increase only the full-path stagger budget to 30
- Decision: With explicit user authorization, fresh-zero rerun the complete
  `u_bar=0.0075` engineering-grip path with `max_stagger_iterations=30`. Keep all
  equations, increments, tolerances, strict gates, and replay contracts fixed.
- Why: The repeated `0.0049` failure is geometrically converging and reaches a
  history gate ratio `1.60` at sweep 20, so a bounded iteration-budget increase
  is the minimal discriminating test.
- Alternatives rejected: relaxing the history gate, warning acceptance,
  committing the failed candidate, changing the increment, or manually resuming
  from the failure archive.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md`.

## 2026-08-27 — Hold after the 30-sweep 5.4e-3 stop
- Decision: Keep the `u_bar=0.0054` candidate rejected after the authorized
  30-sweep run and request a new explicit budget decision before continuing.
- Why: The run accepts through `0.00535`, remains notch-localized, and the final
  history/damage-residual errors continue contracting, but three strict gates
  remain failed at sweep 30. A 60-sweep ceiling is the next bounded test and is
  preferable to repeated small budget increases.
- Alternatives rejected: committing the failed state, relaxing damage/history
  gates, warning acceptance, or silently increasing the budget again.
- Related:
  `/tmp/opencode/fracture2d_engineering_grip_v2_u7p5e3_s30_failure.npz` and
  `docs/tasks/fracture2d-engineering-grip-baseline.md`.

## 2026-08-27 — Use durable engineering-grip tooling and evidence paths
- Decision: With explicit user authorization, use a 60-sweep ceiling for one
  final fresh-zero `0.0075` run. Before running it, consolidate reusable tooling
  under `examples/fracture2d/` and persist all engineering-grip inputs/evidence
  under `/root/autodl-tmp/data/fracture2d/engineering_grip/` with hashes and
  configuration metadata.
- Why: `/tmp/opencode` is system scratch and may be cleaned. Reproducible code
  belongs in the repository, while large numerical artifacts belong in the
  established external fracture2d data tree.
- Alternatives rejected: relying on temporary files as the only evidence,
  checking large binary archives into the repository, silently changing solver
  gates, or performing another run before its output path is durable.
- Related: `examples/fracture2d/README.md` and
  `docs/tasks/fracture2d-engineering-grip-baseline.md`.

## 2026-08-27 — Hold after auditing the 60-sweep contraction
- Decision: Keep the `u_bar=0.0059` trial rejected and preserve its durable
  failure evidence unchanged. Require an explicit user choice before another
  run; if only the iteration budget changes, use a bounded fresh-zero 72-sweep
  test rather than silently extending the failed trial.
- Why: The final damage, history, and post-history residual gate ratios are
  `1.748`, `6.974`, and `2.101`, but all three contract with a tail factor near
  `0.809`. Extrapolation predicts that history, the limiting gate, needs about
  10--11 additional sweeps. Displacement and reaction already pass, the fixed-
  history damage solve remains accurate, and rollback digests match.
- Alternatives rejected: warning-accepting the rejected state, resuming from
  its archive, silently raising the budget, weakening strict gates, or treating
  a smaller increment or nonlinear strategy as equivalent to a budget-only
  rerun.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md` and
  `/root/autodl-tmp/data/fracture2d/engineering_grip/engineering_grip_v2_u7p5e3_s60.failure.npz`.

## 2026-08-27 — Increase only the fresh-zero stagger budget to 72
- Decision: With explicit user authorization, run one fresh-zero engineering-
  grip continuation through `u_bar=0.0075` with
  `max_stagger_iterations=72`. Keep all equations, increments, tolerances,
  strict gates, rollback behavior, and provenance unchanged.
- Why: The 60-sweep rejected attempt contracts with a tail factor near `0.809`,
  and its limiting history gate is estimated to need 10--11 additional sweeps.
  A 72-sweep ceiling is therefore the smallest rounded budget that directly
  tests the observed convergence without changing the numerical path.
- Alternatives rejected: resuming the rejected state, warning acceptance,
  relaxing a gate, simultaneously changing the increment, or adopting an
  underived nonlinear strategy.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md`.

## 2026-08-27 — Hold after the 72-sweep 5.95e-3 stop
- Decision: Keep the `u_bar=0.00595` trial rejected and require another explicit
  numerical choice. Permit no automatic budget escalation; at most one bounded
  fresh-zero 80-sweep diagnostic remains justified by the current tail fit.
- Why: The 72-sweep run accepts the former `0.0059` blocker in 68 sweeps, but the
  next minimum-increment trial still fails history at `2.917` times its limit.
  Its tail contracts near `0.828` and predicts 6--9 more sweeps, while the
  accepted-step cost and contraction factor both worsened relative to s60.
- Alternatives rejected: warning acceptance, resuming the rejected state,
  treating the fitted estimate as a guaranteed convergence bound, repeatedly
  increasing the budget without review, or silently combining a budget change
  with a smaller increment or nonlinear strategy.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md` and
  `/root/autodl-tmp/data/fracture2d/engineering_grip/engineering_grip_v2_u7p5e3_s72.failure.npz`.

## 2026-08-27 — Study a smaller increment before another run
- Decision: Analyze a smaller minimum increment before authorizing another
  engineering-grip continuation. Do not run or silently select the new value
  until the retained s72 nominal and half-step failures support a bounded test.
- Why: The s72 history fixed point still contracts, but its rate and accepted-
  step cost worsened. The user selected increment refinement rather than another
  immediate stagger-budget increase.
- Alternatives rejected: automatically running 80 sweeps, changing both budget
  and increment together, resuming the rejected trial, or adopting an underived
  nonlinear strategy.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md`.

## 2026-08-27 — Make minimum-increment detection roundoff-safe
- Decision: Classify an engineering-grip trial as the configured minimum
  increment within eight float64 ULPs at the largest participating load or
  increment scale. Use the same rule in production and completed-archive replay.
- Why: The s72 audit contains two numerically identical `0.00595` retries because
  subtraction represented the `5e-5` interval microscopically above `du_min`.
  Retrying the same target cannot discriminate convergence and inflates the
  attempt audit.
- Alternatives rejected: changing the target itself, using an arbitrary decimal
  tolerance, keeping the duplicate as a cutback, or changing the physical gates.
- Related: `examples/fracture2d/nonlocal_engineering_grip.py`,
  `tests/test_fracture2d_nonlocal_engineering_grip.py`, and
  `docs/tasks/fracture2d-engineering-grip-baseline.md`.

## 2026-08-27 — Propose a bounded 2.5e-5 increment test
- Decision: Before another full-path attempt, propose one fresh-zero diagnostic
  through `u_bar=0.00595` with only `du_min` changed from `5e-5` to `2.5e-5`.
  Keep `du_initial=1e-4`, `cutback_factor=0.5`, `max_cutbacks=8`,
  `max_stagger_iterations=72`, and all equations and gates unchanged. Execution
  still requires explicit user authorization.
- Why: At the same committed `0.0059` state, halving the trial increment from
  `1e-4` to `5e-5` reduces final history ratio from `31.375` to `2.917` and
  improves the tail factor from about `0.8493` to `0.8281`. The next natural
  cutback tests `0.005925` without conflating increment and budget changes.
- Alternatives rejected: rerunning immediately to `0.0075`, changing the sweep
  budget simultaneously, inferring guaranteed convergence at `2.5e-5`, or
  resuming from the s72 archive.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md`.

## 2026-08-27 — Authorize a bounded engineering-grip warning path
- Decision: Add and run a separate fresh-zero warning-mode engineering-grip
  diagnostic through `u_bar=0.0075`. After 72 complete sweeps, warning-commit
  convergence-gate failures. Record every damage range violation and healing
  event; continue only while each absolute violation is at most `1e-3`.
- Why: The strict path repeatedly stops on a slowly contracting history fixed
  point, and the user wants the complete response for diagnosis. A bounded
  warning path exposes the later curve and fields without relabeling them as
  strict evidence.
- Alternatives rejected: modifying the strict controller, clipping or
  projecting damage, accepting non-finite/incomplete solves, silently ignoring
  warning causes, or allowing range/healing violations larger than `1e-3`.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md`.

## 2026-08-27 — Check warning safety bounds at the final sweep
- Decision: In the engineering-grip warning diagnostic, allow finite
  intermediate stagger sweeps to continue regardless of temporary damage range
  or healing violations. On the 72nd complete sweep, hard-stop unless raw range
  excess `max(0,-min(d),max(d)-1)` and healing magnitude
  `max(0,-min(d_next-d_committed))` are each at most `1e-3`. Treat final crack
  localization failure as warning-only and use a separate warning archive.
- Why: The user wants intermediate iterates to have an opportunity to recover
  before the warning candidate is judged, while retaining an explicit bound on
  the state that can actually be committed.
- Alternatives rejected: stopping on the first intermediate violation,
  subtracting strict tolerances from the `1e-3` raw bound, clipping/projecting
  damage, or modifying the strict v2 archive contract.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md`.

## 2026-08-27 — Hold warning continuation pending contract audit
- Decision: Do not execute the authorized full-path warning continuation yet;
  preserve the strict artifacts and audit the warning implementation first.
- Why: The declared project-owned warning contract permits finite raw lower-range
  damage violations within the explicit safety limit, but result construction
  rejects negative damage through the unconditional final localization diagnostic.
  The engineering-grip force and moment gates are also collapsed into one
  conjunction, so a warning archive can retain an incorrect failed-gate list.
  Independent replay additionally accepts a rejected complete attempt with fewer
  than the configured full sweep budget and does not enforce warning-specific
  incomplete-stage ordering.
- Alternatives rejected: running the long diagnostic and treating the resulting
  archive as trustworthy, clipping/projecting the lower-range candidate,
  weakening strict gates, or silently selecting one force/moment interpretation.
- Relation: These are warning/archive implementation-contract findings; the
  approved mechanics energy, material-only damage equations, strict gates, and
  fresh-zero provenance remain unchanged.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md` and
  `examples/fracture2d/nonlocal_engineering_grip.py`.

## 2026-08-27 — Preserve raw warning fields and gate provenance
- Decision: Permit signed damage extrema only through an explicit warning-mode
  crack-diagnostic opt-in; retain the default non-negative diagnostic contract
  for strict and standalone callers. Carry force and moment pass/fail values
  independently through production diagnostics and independent replay. Reject
  warning replay stages and rejected-complete attempt lengths that the warning
  controller cannot produce.
- Why: The authorized warning rule measures raw damage excess without clipping,
  while finite mechanics conservation has two separately configured gates. The
  archive must reproduce the controller's actual state machine rather than
  accepting a broader self-reported sequence.
- Alternatives rejected: clipping/projecting damage, changing strict gates or
  constitutive equations, replacing force/moment with one warning gate, or
  accepting `damage_acceptance` as a warning incomplete stage.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md`;
  `examples/fracture2d/nonlocal_engineering_grip.py`;
  `tests/test_fracture2d_nonlocal_engineering_grip.py`.

## 2026-08-27 — Bound residual reconstruction by AT2 constituents
- Decision: Compare redundant engineering-grip AT2 residual reconstructions
  componentwise using `512*eps64` times the sum of absolute constituent terms;
  multiply the bound by particle volume for the Euclidean residual.
- Why: Equivalent sparse-system and explicit-term evaluation orders differed by
  `2.3283e-10` during the warning run, within the independently established
  constituent bound. The generic absolute comparator falsely treated this
  cancellation noise as a structural error.
- Alternatives rejected: bypassing residual auditing, using an arbitrary larger
  absolute tolerance, or weakening the authoritative damage-residual gate.
- Related: `examples/fracture2d/nonlocal_engineering_grip.py` and
  `tests/test_fracture2d_nonlocal_engineering_grip.py`.

## 2026-08-27 — Stop warning continuation at the healing safety limit
- Decision: Keep the `u_bar=0.00675` warning candidate rejected and retain
  `u_bar=0.0067` as the last committed diagnostic state. Do not relax the
  user-authorized `1e-3` healing limit automatically.
- Why: The warning path exposes the intended peak and crack-growth branch, but
  the final candidate heals by `0.0065015` after 72 sweeps, over six times the
  hard safety limit. Range excess remains zero and rollback is exact.
- Alternatives rejected: clipping/projecting damage, committing the rejected
  state, silently raising the healing threshold, or resuming from the failure
  archive.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md` and
  `/root/autodl-tmp/data/fracture2d/engineering_grip/engineering_grip_warning_u7p5e3_s72.failure.npz`.

## 2026-08-27 — Separate FEM snapshot and response indexing
- Decision: Use snapshot indices only for FEM fields and independently locate or
  interpolate the FEM response at the requested load for stress reports.
- Why: At `u_bar=0.0067`, snapshot index 67 is not response index 67; the exact
  response index is 94. Reusing the snapshot index reported `535.5371` instead
  of the correct `5.5572326`, although the plotted full curve was correct.
- Alternatives rejected: assuming snapshot and response arrays share indexing
  or changing the FEM archive.
- Related: `examples/fracture2d/engineering_grip_workflow.py` and
  `tests/test_fracture2d_engineering_grip_workflow.py`.

## 2026-08-27 — Authorize unbounded finite damage warnings
- Decision: Add a separate explicit warning safety policy in which finite
  damage-range and healing violations are always warning-only, with no magnitude
  threshold. Run it fresh-zero through `u_bar=0.0075`; retain raw values and
  mark all outputs `validated=False`.
- Why: The bounded warning path already exposes the peak but stops at `0.00675`
  on healing `0.00650`. The user wants the complete diagnostic trajectory before
  choosing a solver or model correction.
- Alternatives rejected: clipping/projecting damage, silently encoding the
  policy as an infinite numeric threshold, changing strict behavior, resuming
  the rejected archive, or allowing non-finite/incomplete mechanics states.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md`.

## 2026-08-28 — Test stabilized convergence-first fracture continuation
- Decision: Add a separate experimental continuation path that keeps finite
  damage-range and healing violations warning-only, but never commits a state
  that fails the coupled displacement, damage, history, reaction, or damage-
  residual convergence gates. Use safeguarded damage under-relaxation and load
  cutback, then run a fresh-zero comparison only through `u_bar=0.0063` before
  considering a full path.
- Why: Sweep-level evidence at `u_bar=0.0062` shows monotone positive feedback
  localized at notch-tip particle 1869. The 72-sweep warning state is a budget
  cutoff with large coupled errors, not a converged solution; later increments
  inherit that state.
- Alternatives rejected: clipping displacement or damage, silently increasing
  `k_res`, accepting an unconverged state to preserve continuation, resuming the
  contaminated warning archive, or changing existing strict/warning semantics.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md`.

## 2026-08-28 — Persist stabilized physical states, not iteration vectors
- Decision: Replace the experimental stabilized archive's full per-sweep and
  per-backtracking vector evidence with a lean physical-state archive. Persist
  accepted `q`, `d`, irreversible `H`, reaction, load, geometry/source hashes,
  configuration, and compact scalar convergence/backtracking summaries only.
- Why: The user needs physical fields comparable in size and use to the FEM
  archive. Saving every intermediate trial displacement duplicated hundreds of
  transient fields and made a short continuation operationally impractical.
- Limitation accepted: accepted physical states can be independently checked
  against the governing residuals and gates, but the exact rejected trial path
  cannot be reconstructed from scalar summaries. Reports must distinguish
  field-equation validation from full path replay and must not claim the latter.
- Alternatives rejected: retaining multi-GB trial vectors, assuming all history
  is recoverable from the final displacement, or dropping convergence summaries
  and source/config provenance entirely.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md`.

## 2026-08-29 — Reuse selected mechanics result across damped sweeps
- Decision: Keep the selected `q_after` mechanics result only in transient
  memory and reuse it as the next sweep's `q_before` when load, boundary data,
  and `d_iter == d_trial` are identical. Invalidate it on load change, cutback,
  rollback, incomplete mechanics, or unselected candidates.
- Why: The next sweep otherwise solves the identical linear mechanics problem
  again. One union displacement vector is about 59 kB; retaining one immutable
  solve result is constant-memory and should nearly halve mechanics solves for
  fixed damping.
- Alternatives rejected: persisting sweep caches in the lean archive, caching
  across load/cutback boundaries, or changing mechanics tolerances/equations.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md`.

## 2026-08-29 — Emit flushed stabilized continuation progress
- Decision: The workflow passes a transient progress callback to the stabilized
  controller and emits structured `STABILIZED_PROGRESS` JSON lines with
  `flush=True` at attempt start, an explicit sweep cadence, acceptance, cutback,
  completion, and failure. The cadence is an explicit CLI argument.
- Why: Multi-hour fresh-zero runs previously printed only storage preflight and
  final status, so users could not distinguish forward progress from a stalled
  solve or repeated cutbacks.
- Alternatives rejected: persisting verbose logs in the lean archive, printing
  every internal linear iteration, or relying on buffered stdout.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md`.

## 2026-08-29 — Replace over-damped run with foreground `omega=0.5`
- Decision: Stop the background fixed-`omega=0.25`, 144-sweep run and launch a
  fresh-zero fixed-`omega=0.5`, 144-sweep cached run in the active shell. Keep
  all convergence gates and cutback behavior unchanged.
- Why: `omega=0.25` required about 130-140 sweeps per accepted softening step
  and reduced to `du_min`, despite bounded displacement amplification near
  `1.062`. The prior `omega=0.5` evidence was only modestly above the history
  and damage-residual limits after 72 sweeps, so the larger sweep budget should
  retain damping while avoiding systematic over-relaxation cost.
- Alternatives rejected: continuing the multi-hour background run, returning
  to undamped `omega=1`, or warning-committing unconverged states.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md`.

## 2026-08-28 — Close the stabilized final-trial replay gap

- Decision: Keep the stabilized controller isolated and strengthen its archive
  contract without changing the mechanics energy, fixed-history AT2 equation,
  Amor update, boundary conditions, or strict coupled gates. Complete relaxation
  trials must carry hard-gate/state provenance, selection must equal
  globalization, and independent replay must recompute and compare the final
  complete trial's merit and coupled status. Failure archives must pair the
  last committed load as well as its fields.
- Why: A forged final trial could previously alter its recorded merit or
  coupled-pass flag while leaving the independently replayed sweep accepted.
  The real tiny smoke archive now writes and replays independently after the
  caller explicitly enables JAX float64 mode.
- Alternatives rejected: changing the production `0.0063` path, modifying the
  existing strict/warning controllers, clipping or projecting raw fields, or
  silently accepting non-final trial provenance that has not been independently
  reconstructed.
- Related: `examples/fracture2d/nonlocal_engineering_grip_stabilized.py`,
  `tests/test_fracture2d_nonlocal_engineering_grip_stabilized.py`, and
  `docs/tasks/fracture2d-engineering-grip-baseline.md`.

## 2026-08-28 — Add conservative stabilized archive storage preflight
- Decision: Estimate the complete stabilized numerical archive before a
  continuation run from the validated model dimensions, continuation bounds,
  stagger budget, backtracking budget, and failure-attempt reserve. Require the
  output filesystem to have that archive upper bound plus the explicit
  `2 GiB` reserve before launching continuation work.
- Why: The stabilized archive contains fixed-shape state, sweep, trial, static
  operator, and canonical JSON records, and a rejected continuation can retain
  more attempts than a completed path. Compression is not a safe capacity
  contract, so the preflight uses uncompressed typed-member and JSON reserves
  and reports the bound before GPU continuation.
- Alternatives rejected: checking only the current archive size, relying on
  NPZ compression ratios, estimating from accepted states alone, or silently
  deleting failure/sweep provenance when the bound is exceeded.
- Related: `examples/fracture2d/nonlocal_engineering_grip_stabilized.py`,
  `examples/fracture2d/engineering_grip_workflow.py`, and
  `docs/tasks/fracture2d-engineering-grip-baseline.md`.

## 2026-08-29 — Replace stabilized v4 storage with a lean v5 field archive
- Decision: Make the experimental stabilized archive v5 persist only accepted
  physical states, accepted scalar diagnostics, scalar attempt/sweep/trial
  logs, work-check scalars, and explicit source/configuration provenance.
  Reconstructible geometry and operators are removed from the archive; the
  independent checker must reopen and SHA-verify the absolute source-cloud
  path.  Reports expose `field_equations_validated=true` separately from
  `path_replayed=false`; exact rejected-trial path replay is not claimed.
- Why: The v4 contract still retained static graph/support arrays and repeated
  trial vectors even after the accepted-state-only policy was selected.  That
  storage is redundant, makes the archive budget misleading, and conflates
  independent accepted-field checking with unavailable rejected-path history.
- Alternatives rejected: preserving v4 as the active writer, archiving all
  trial vectors for exact path replay, embedding a source-cloud copy, or
  silently accepting a missing/unverified source path.
- Related: `examples/fracture2d/nonlocal_engineering_grip_stabilized.py`,
  `examples/fracture2d/engineering_grip_workflow.py`, focused stabilized and
  workflow tests, and `docs/tasks/fracture2d-engineering-grip-baseline.md`.

## 2026-08-29 — Finalize stabilized v5 summary-only storage
- Decision: Use one compact `solver_summary_json` row for the initial state and
  each accepted physical state, retain typed accepted reactions and cumulative
  external work, and use `failure_summary_json` for the final failure audit.
  Remove v5 attempt/sweep/trial log fields and expose only `states`,
  `accepted_steps`, and `solver_summary_rows` as archive counts.
- Why: The accepted-state-only contract does not retain rejected numerical
  vectors, so separate attempt/sweep/trial rows add redundant provenance while
  obscuring the actual replay boundary. A per-state final summary keeps the
  accepted field-equation audit aligned with the physical rows and makes the
  storage estimate match the materialized archive.
- Alternatives rejected: retaining legacy log arrays in the v5 writer, deriving
  solver counts from omitted trial paths, or claiming exact rejected-path replay
  from scalar provenance.
- Related: `examples/fracture2d/nonlocal_engineering_grip_stabilized.py`,
  `examples/fracture2d/engineering_grip_workflow.py`,
  `examples/fracture2d/README.md`, and the focused v5/workflow tests.

## 2026-08-29 — Add explicit fixed damped-Picard relaxation policy
- Decision: Add `relaxation_policy=fixed` as a separate experimental stabilized
  policy. Each complete sweep evaluates exactly one candidate with
  `omega=omega_initial`; `omega_min` must equal it, `max_backtracks` must be
  zero, and sufficient-decrease controls must be absent. Candidate merit is
  retained as a diagnostic, not used as an acceptance condition.
- Why: The authorized convergence-first comparison needs fixed damped-Picard
  iteration to test whether the localized positive feedback is numerical
  fixed-point instability, while preserving the original equations and all
  coupled convergence gates.
- Alternatives rejected: changing the damage equation or constitutive law,
  clipping/projecting fields, weakening coupled gates, silently falling back to
  merit backtracking, or changing the existing `sufficient_decrease` policy.
- Related: `examples/fracture2d/nonlocal_engineering_grip_stabilized.py`,
  `examples/fracture2d/engineering_grip_workflow.py`,
  `examples/fracture2d/README.md`, and the focused stabilized/workflow tests.

## 2026-08-30 — Isolate minimum-increment sensitivity after fixed-damping stop
- Decision: Preserve the failed cached `omega=0.5`, 144-sweep run unchanged and
  do not resume its rejected `u_bar=0.0059875` candidate. Before considering a
  larger sweep budget, propose one fresh-zero foreground sensitivity stopping at
  the same target and changing only `du_min` from `1.25e-5` to `6.25e-6`; require
  explicit authorization before execution.
- Why: The last accepted state at `0.005975` is independently field-validated,
  notch-localized, and free of the undamped displacement outlier, while the
  rejected half-step still misses history and post-history damage residual by
  factors `3.2524` and `1.2618`. The preceding accepted steps already require
  135-141 of 144 sweeps, and lean v5 has no rejected sweep vectors from which to
  justify a reliable budget extrapolation.
- Alternatives rejected: warning-committing the failed state, resuming rejected
  fields, changing the sweep ceiling and increment together, returning to
  `omega=1`, or changing equations, tolerances, gates, clipping, or `k_res`.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md` and
  `engineering_grip_damped_cached_o5_u6p3e3_s144.failure.npz`.

## 2026-08-30 — Authorize fixed-damping half-increment sensitivity
- Decision: Run one fresh-zero foreground stabilized sensitivity through
  `u_bar=0.0059875`, changing only `du_min` from `1.25e-5` to `6.25e-6`.
  Preserve `omega=0.5`, 144 sweeps, mechanics cache, equations, tolerances,
  hard coupled gates, lean v5 provenance, and flushed progress output.
- Why: This is the smallest isolated test of whether the strict stop at
  `0.0059875` is controlled by load-step size before changing the sweep budget
  or nonlinear solver.
- Alternatives rejected: changing increment and sweep budget together,
  warning-committing the failed state, resuming rejected fields, or immediately
  implementing a new acceleration method without the bounded sensitivity.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md`.

## 2026-08-30 — Move from fixed Picard to safeguarded Anderson design
- Decision: Accept the completed half-increment path through
  `u_bar=0.0059875` as experimental strict evidence, but do not extend fixed
  Picard farther. Next derive and contract-test safeguarded Anderson
  acceleration for the same outer damage fixed-point map, retaining damped
  Picard fallback and every original unrelaxed acceptance gate.
- Why: Reducing the effective final increment from `1.25e-5` to `1.09375e-5`
  resolves the prior blocker, but the endpoint passes only on sweep `143/144`.
  This establishes load-step sensitivity while showing that further fixed-Picard
  continuation has effectively no iteration margin.
- Alternatives rejected: treating the exact endpoint as evidence for a long
  fixed-Picard path, increasing the sweep ceiling without acceleration analysis,
  warning acceptance, field clipping, or replacing the already accurate inner
  mechanics/damage linear solvers.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md` and
  `engineering_grip_damped_cached_o5_u5p9875e3_dumin6p25e6_s144.npz`.

## 2026-08-30 — Defer Anderson for a damped full-path warning diagnostic
- Decision: Defer Anderson implementation. Add a separate warning policy to the
  existing fixed `omega=0.5` stabilized controller and, after contract tests,
  run it fresh-zero through `u_bar=0.0075` with 144 sweeps and
  `du_min=6.25e-6`.
- Why: The user first wants to inspect the complete damped response and physical
  fields. The strict half-increment path is trustworthy through `0.0059875` but
  consumes nearly the full sweep budget, so a truthful warning diagnostic can
  expose later behavior without presenting it as converged evidence.
- Alternatives rejected: implementing Anderson now, modifying the existing
  strict stabilized policy, clipping raw fields, silently bypassing non-finite
  or mechanics/CG failures, or relabeling warning states as validated.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md`.

## 2026-08-30 — Define isolated stabilized-warning state and archive contract
- Decision: Keep `acceptance_mode=stabilized` and the existing v5 completed/failure
  schemas for strict stabilized runs. Add a separate
  `acceptance_mode=stabilized_warning` with distinct completed/failure schemas
  and completion/acceptance statuses. It is valid only for the explicit fixed
  damped-Picard policy; a complete final sweep may be committed when all finite
  and mechanics/CG hard gates pass, while any failed coupled/range/healing gate
  is retained as a warning. Incomplete sweeps, non-finite values, mechanics/CG,
  structural, and work-check failures remain hard stops.
- Why: The authorized diagnostic is intended to expose the full raw response,
  not to weaken strict stabilized convergence or turn warning states into
  validated evidence. Separate schema identity prevents old strict archives
  from being reinterpreted, and an explicit bypass list plus gate map makes each
  warning state's failed-gate provenance auditable.
- Alternatives rejected: changing strict v5 validation in place, accepting a
  warning before the final complete sweep, adding a safety clipping/projection,
  reusing legacy warning schemas, or allowing warning acceptance after a hard
  numerical failure.
- Related: `examples/fracture2d/nonlocal_engineering_grip_stabilized.py`,
  `examples/fracture2d/engineering_grip_workflow.py`, focused stabilized and
  workflow tests, and `docs/tasks/fracture2d-engineering-grip-baseline.md`.

## 2026-08-30 — Serialize accepted stabilized warnings as accepted rows
- Decision: Keep failed convergence provenance for a warning-accepted sweep in
  its gate map, canonical bypass list, and raw range/healing fields, but serialize
  the accepted v5 solver-summary row with empty `failure_stage` and
  `failure_reason` fields.
- Why: The row is an accepted physical state rather than the final rejected-target
  failure record. The existing summary validator requires accepted rows to have
  empty failure fields; retaining the text there made the new warning archive
  unreplayable even though its gate provenance was correct.
- Alternatives rejected: weakening the accepted-row validator, erasing the
  warning gate/bypass provenance, changing strict stabilized summaries, or
  relabeling the warning row as strict convergence.
- Related: `examples/fracture2d/nonlocal_engineering_grip_stabilized.py`,
  `tests/test_fracture2d_nonlocal_engineering_grip_stabilized.py`, and
  `docs/tasks/fracture2d-engineering-grip-baseline.md`.

## 2026-08-30 — Do not promote warning states to field-equation validation
- Decision: Continue independent numerical checks for warning archives, but
  report `field_equations_validated=false` whenever the archive contains a
  warning-accepted state.
- Why: The stabilized-warning path intentionally records a complete raw state
  that failed one or more coupled convergence gates. Its replay can audit the
  retained fields without presenting that warning state as converged evidence;
  strict stabilized archives retain the existing field-equation validation
  result.
- Alternatives rejected: reporting warning states as validated because their
  recomputed residuals happen to pass, skipping independent checks entirely, or
  changing the strict v5 replay result.
- Related: `examples/fracture2d/nonlocal_engineering_grip_stabilized.py`,
  `tests/test_fracture2d_nonlocal_engineering_grip_stabilized.py`, and
  `docs/tasks/fracture2d-engineering-grip-baseline.md`.

## 2026-08-30 — Retain the completed damped warning path as runaway evidence
- Decision: Preserve the completed `u_bar=0.0075` stabilized-warning archive as
  unvalidated diagnostic evidence and do not extend fixed Picard from it. Later
  strict-passing states do not erase inherited warning-accepted transitions.
- Why: Ten warning states carry the path through the peak and collapse. Reaction
  falls from `641.34` to `69.15`, crack-tip history reaches `2.168e6`, material
  displacement amplification reaches `18.95`, and healing reaches `0.02915`.
  Independent accepted-state replay is archive-bound but the static damage
  equation fails on warning states, so `field_equations_validated=false`.
- Alternatives rejected: relabeling the endpoint as strict evidence because its
  final sweep passes, clipping the runaway fields, resuming from a warning state,
  increasing the sweep budget again, or overwriting either strict artifact.
- Related: `engineering_grip_damped_warning_o5_u7p5e3_dumin6p25e6_s144.npz`
  and `docs/tasks/fracture2d-engineering-grip-baseline.md`.

## 2026-08-30 — Bound warning-v1 replay to retained accepted states
- Decision: Independently rebuild every retained accepted-state mechanics,
  reaction, conservation, history, and damage equation. Exclude only the
  transition-only `force_scale` and `moment_scale` summary scalars from the
  accepted-state replay claim because they combine pre/post candidate mechanics
  while warning-v1 intentionally stores only the accepted physical state.
- Why: Warning transitions can leave those combined scales materially different
  from the accepted-state scale even though independently rebuilt force/moment
  errors and hard conservation gates pass. Weakening physical tolerances would
  be incorrect; claiming reconstruction of omitted pre-candidate fields would be
  false.
- Alternatives rejected: loosening force/moment gates, silently using a 2%
  scalar-comparison tolerance, adding omitted trial vectors retroactively, or
  changing the archive/source hash.
- Related: `examples/fracture2d/engineering_grip_workflow.py` and the warning-v1
  replay report for the completed damped run.

## 2026-08-30 — Defer zero-stiffness drift and default only numerical values
- Decision: Record the suspected effectively zero-stiffness particle drift as an
  unresolved issue and defer its remedy. Make the current engineering-grip input
  paths, physical/discretization values, endpoint, load increments, 144-sweep
  ceiling, and fixed `omega=0.5` values CLI defaults, while keeping strict
  acceptance as the default and warning policy/storage/output choices explicit.
- Why: The values define the latest reproducible experiment, but the warning
  path is unvalidated and exhibits extreme displacement drift. Defaulting the
  warning state machine itself would make an unsafe diagnostic look normative.
- Alternatives rejected: default warning acceptance, change `k_res`, clip
  displacement/damage, remove particles, silently increase stiffness, or attempt
  a remedy without a separately checked derivation.
- Related: `examples/fracture2d/engineering_grip_workflow.py`, its README, and
  `docs/tasks/fracture2d-engineering-grip-baseline.md`.
