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

## 2026-09-17 — Implement safeguarded Anderson before fracture2d PINN
- Decision: Keep the validated direct nonlocal fixed subproblems and implement a
  separate safeguarded Type-II Anderson strategy for the outer material-damage
  fixed-point map. Use history depth `5`, mixing parameter `beta=0.5`, and a
  rank-revealing QR least-squares solve. Reject an accelerated candidate that is
  rank deficient, non-finite, outside the raw admissible damage range, or fails
  the existing sufficient-decrease merit check; restart Anderson history and
  fall back to the current `omega=0.5` damped-Picard candidate. Preserve all
  mechanics, AT2, Amor-history, rollback, cutback, archive, and final strict-gate
  equations. Do not run a long continuation until manufactured contract tests
  and independent review pass.
- Why: The fixed-history damage residual remains near float64 roundoff and the
  mechanics solves retain `cg_info=0`, while the strict damped path consumes
  `143/144` outer sweeps at `u_bar=0.0059875`. The completed warning diagnostic
  then develops large history and post-history residual errors together with
  near-zero crack-tip stiffness. This identifies the immediate solver failure as
  the outer coupled fixed point, not either inner linear solver. A PINN would
  inherit the same residual conditioning and add representation and optimization
  errors before the direct baseline is validated through peak load.
- Alternatives rejected: starting fracture2d PINN as a nonlinear-solver remedy;
  increasing the fixed-Picard budget again; weakening gates; clipping fields;
  changing `k_res`; or treating warning states as converged evidence.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md`,
  `examples/fracture2d/nonlocal_engineering_grip_stabilized.py`, and Walker--Ni,
  DOI `10.1137/10078356X`.

## 2026-09-17 — Pass post-fix Anderson implementation audit
- Decision: Mark the safeguarded Anderson implementation as GO for
  implementation and manufactured evidence only. Do not authorize a physical
  continuation or promote the stabilized path to production.
- Why: The read-only audit verified candidate-specific `H_next` map merit,
  structural-error propagation, numerical fallback/history restart, the
  original unrelaxed fixed-point and damage-equation gates, strict Anderson
  range/healing gates, and truthful `recorded_not_reconstructed` v5 provenance.
  Focused stabilized/workflow tests pass `57/57`, legacy engineering-grip plus
  JAX tests pass `60/60`, the related baseline suite passes `7/7`, compilation
  and `git diff --check` pass, and no continuation or durable artifact changed.
- Alternatives rejected: treating manufactured contracts as physical
  convergence evidence, weakening strict gates, reconstructing omitted rejected
  trials from scalar logs, or running a long continuation before primary review.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md`,
  `examples/fracture2d/nonlocal_engineering_grip_stabilized.py`,
  `tests/test_fracture2d_nonlocal_engineering_grip_stabilized.py`, and the
  Walker--Ni formulation decision above.

## 2026-09-17 — Authorize bounded Anderson physical checkpoints
- Decision: Run a fresh-zero strict rounded-cloud Anderson smoke to
  `u_bar=1e-4` with at most `20` sweeps. Only if its archive and independent
  accepted-field replay pass, run one fresh-zero strict comparison through exact
  `u_bar=0.0059875` with the existing `du_initial=1e-4`, `du_min=6.25e-6`, and
  `144`-sweep ceiling. Preserve `delta=0.048`, `h_g=0.02`, all equations,
  tolerances, gates, cutback rules, GPU/CPU authority, and existing artifacts.
- Why: The Anderson implementation/manufactured audit is GO, but no physical
  cloud evidence exists. The short smoke checks integration and provenance
  before the bounded endpoint directly compares against the fixed-Picard path
  that required `143/144` sweeps.
- Alternatives rejected: jumping directly to `u_bar=0.0075`, warning acceptance,
  changing the horizon or load step together with the nonlinear method,
  overwriting prior evidence, or treating smoke completion as crack-growth
  validation.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md`.

## 2026-09-17 — Hold Anderson at the bounded physical checkpoint
- Decision: Preserve the completed strict Anderson smoke and
  `u_bar=0.0059875` checkpoint, but do not authorize a farther continuation or
  promote Anderson to production. Any follow-up requires a new explicit
  decision; the smallest defensible proposal is a fresh-zero strict run only to
  `u_bar=0.0060` with all current numerical settings unchanged.
- Why: The bounded archive and independent accepted-field replay pass, and the
  endpoint agrees with fixed Picard. Anderson reduces accepted-step sweeps by
  `31.54%` at common loads, but almost all benefit is before `u_bar=0.0046`; the
  crack-growth tail improves only `0.97%`, the last ten retained rows all use
  insufficient-decrease Picard fallback, and runtime rises by about `48.7%`.
  Lean v5 also records rather than reconstructs Anderson selection/rank scalars
  and cannot replay intermediate or rejected trial paths.
- Alternatives rejected: jumping to `0.0063` or `0.0075`, resuming the completed
  archive, weakening strict gates, raising the sweep budget, interpreting early
  low-load acceleration as peak-region improvement, or using a PINN as the
  unresolved nonlinear solver.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md`,
  `engineering_grip_anderson_u5p9875e3_dumin6p25e6_s144.npz`, and the prior
  bounded-checkpoint authorization.

## 2026-09-17 — Authorize bounded opt-in Anderson scalar diagnostics
- Decision: On the same active engineering-grip task, add a minimal durable,
  opt-in scalar diagnostic stream for safeguarded Anderson. Bind its JSONL
  metadata to the source bundle, explicit configuration, and final archive
  hash; omit intermediate field vectors and preserve the lean v5 archive
  semantics. Fail loudly on diagnostic I/O errors. Preserve every equation,
  gate, acceptance decision, solver choice, legacy default, and fallback rule.
- Decision: First prove the structural merit mechanism with manufactured
  calculations and focused tests. Then run exactly one fresh-zero strict
  diagnostic with the prior Anderson command and parameters through exact
  `u_bar=0.0059875` or less, using a new durable run ID. Do not resume the old
  archive and do not run to `0.0060` or any higher endpoint.
- Why: The archived bounded run records selected scalar provenance but cannot
  reconstruct rejected accelerated trials. Per-sweep scalar components are the
  smallest durable evidence that can distinguish transition-update terms,
  candidate map residuals, sufficient-decrease rejection, and QR/history
  restart behavior without changing the physical method or storing field
  vectors.
- Alternatives rejected: changing the sufficient-decrease rule or gates,
  adding extra mechanics solves solely for Picard comparison, storing rejected
  field vectors, modifying the existing v5 archive schema, resuming the prior
  run, or extending beyond `u_bar=0.0059875`.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md`,
  `examples/fracture2d/nonlocal_engineering_grip_stabilized.py`,
  `examples/fracture2d/engineering_grip_workflow.py`, and the bounded Anderson
  archive manifest.

## 2026-09-17 — Keep Anderson scalar diagnostics outside v5 archives
- Decision: Implement the authorized diagnostic as an opt-in scalar JSONL
  writer and callback. Emit an explicit proposal record, one record per
  candidate trial, and one sweep record; bind the header to source/configuration
  provenance and bind the final record to the exact completed or failure archive
  SHA256. Reject numerical arrays and numerical lists, flush each record, and
  fsync the final record.
- Why: The existing stabilized dataclasses already retain the scalar values
  needed for candidate-specific merit and fallback analysis, while the v5
  archive intentionally omits rejected field vectors. Keeping the stream
  external preserves lean archive size and replay semantics without adding a
  second mechanics solve for a hypothetical Picard comparison.
- Alternatives rejected: adding diagnostic arrays to v5, reconstructing
  candidate fields from scalar rows, silently treating an unevaluated Picard
  comparison as zero, or changing the acceptance controller to produce logs.
- Related: `examples/fracture2d/nonlocal_engineering_grip_stabilized.py`,
  `examples/fracture2d/engineering_grip_workflow.py`,
  `tests/test_fracture2d_nonlocal_engineering_grip_stabilized.py`, and
  `examples/fracture2d/README.md`.

## 2026-09-18 — Apply the stabilized float64 contract to standalone reports
- Decision: Enable JAX x64 at the start of standalone stabilized archive
  analysis and plotting, using the same explicit helper as standalone replay,
  before any stabilized geometry/model reconstruction.
- Why: A real diagnostic archive analysis exposed the missing process-local
  x64 initialization and failed at `ExplicitVolumeSparseD`; replay already had
  the required initialization. Analysis and plotting must rebuild the same
  float64 coordinate/volume/operator contract rather than depend on the
  caller's process state.
- Alternatives rejected: relying on a previous replay in the same process,
  changing the operator's dtype requirement, or treating the analysis failure
  as an archive defect.
- Related: `examples/fracture2d/engineering_grip_workflow.py`,
  `tests/test_fracture2d_engineering_grip_workflow.py`, and the diagnostic
  task's standalone analysis verification.

## 2026-09-18 — Hold the bounded scalar Anderson diagnostic
- Decision: Preserve the fresh-zero scalar diagnostic at exact
  `u_bar=0.0059875` and do not extend or promote the experimental Anderson
  path. Treat its JSONL rejection statistics as review evidence, not as a new
  acceptance rule or physical-model change.
- Why: The completed archive and independent accepted-state replay validate,
  but `2,516/2,796` accelerated candidates fail sufficient decrease; `2,502`
  are dominated by history-transition merit. Only `46` accepted summary states
  select acceleration versus `20` Picard fallbacks, and the lean v5 contract
  still cannot reconstruct rejected trial vectors.
- Alternatives rejected: continuing above the authorized endpoint, weakening
  sufficient decrease or coupled gates, changing the history transition,
  storing rejected field vectors in v5, or interpreting scalar diagnostics as
  rejected-path replay.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md`,
  `docs/PROGRESS.md`, and the durable scalar stream under
  `/root/autodl-tmp/data/fracture2d/engineering_grip/`.

## 2026-09-18 — Qualify scalar accounting and make diagnostic I/O fatal
- Decision: Treat the scalar diagnostic as observational instrumentation only:
  callback-on/off execution must preserve manufactured accepted fields, path,
  selection, and solve counts, while callback/writer/archive I/O errors must
  propagate as `OSError` and never enter numerical fallback. Close the workflow
  writer in `finally` seams covering both completed and failure archive writes.
  Report the recorded workflow hash as historical provenance when it differs
  from the current post-x64-fix workflow bytes.
- Why: The independently checked stream contains `2,796 = 280 + 2,516`
  accelerated candidate outcomes; `2,445` rejected candidates improve the raw
  candidate map residual, `2,502` are history-transition dominated, and `2,152`
  also exceed the decrease limit in another component. The overlap means the
  rejection evidence must not be reduced to a history-only claim, and durable
  I/O failure must not be confused with a numerical candidate-map fallback.
- Alternatives rejected: adding an unevaluated Picard residual solve, claiming
  current workflow-byte identity for the older stream, swallowing writer errors,
  or changing any acceptance rule or physical continuation setting.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md`,
  `docs/PROGRESS.md`, `examples/fracture2d/engineering_grip_workflow.py`, and
  the Anderson scalar diagnostic tests.

## 2026-09-18 — Implement explicit committed-history residual Anderson policy

- Decision: Add a distinct `relaxation_policy=anderson_residual` while keeping
  historical `anderson` composite-merit semantics and old archive/configuration
  interpretation unchanged. The new policy compares only the normalized actual
  committed-history map residual `r(d)=G_c(d)-d`, with candidate/reference own
  fixed-point scales, the existing `GateTolerance`, and the explicit
  `(1-sufficient_decrease*beta)=0.995` threshold for `sufficient_decrease=0.01`
  and `beta=0.5`.
- Decision: Make finite, mechanics/CG-before/after, force, moment, raw
  range/healing-against-committed-state, and existing hard history-monotonicity
  checks prerequisites for accelerated and exact Picard selection. Numerical
  candidate-map failures restart history and fall back exactly to `omega=0.5`;
  structural and diagnostic I/O errors propagate. The fallback is not tested
  against the residual decrease threshold.
- Decision: Preserve every original final convergence gate, including the
  unrelaxed fixed-point and fixed-history damage-equation gates, transition
  displacement/damage/history/reaction gates, post-history residual,
  conservation, range, healing, globalization, and coupled acceptance. Bind a
  distinct merit identity through config, summaries, replay, CLI, and scalar
  diagnostics; old artifacts must not be reinterpreted.
- Why: Independent contract review approved the formulation, while scalar
  evidence shows that the historical composite merit rejects many candidates
  whose actual committed-history map residual improves. A separate explicit
  policy permits that numerical experiment without silently changing historical
  results or weakening final physical acceptance.
- Alternatives rejected: changing historical `anderson`, removing transition
  or final gates, comparing residuals with a shared normalizer, accepting a
  fallback without hard gates, adding a hypothetical Picard map solve, or
  launching a physical run before implementation review.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md`,
  `docs/PROGRESS.md`, `examples/fracture2d/nonlocal_engineering_grip_stabilized.py`,
  and `examples/fracture2d/engineering_grip_workflow.py`.

## 2026-09-18 — Tighten residual-policy validator contracts
- Decision: Permit historical `anderson` replay reports to omit only the two
  optional identity fields, while requiring any supplied identity to match the
  archive. Require residual-policy fallback summaries to keep selection-merit
  and candidate-map fields null; require complete accelerated summaries to
  retain finite, nonnegative merit/reference values and candidate-map metrics.
- Why: Independent review found that the controller and manufactured policy
  evidence were correct, but replay/summary validators accepted forged legacy
  identity combinations and incomplete residual selection fields. The validator
  must preserve historical artifact semantics without weakening residual-policy
  provenance checks.
- Alternatives rejected: changing controller numerics, reinterpreting legacy
  archives, accepting residual fallback merit values, or requiring candidate
  vectors that the lean v5 archive intentionally does not store.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md`,
  `examples/fracture2d/engineering_grip_workflow.py`,
  `examples/fracture2d/nonlocal_engineering_grip_stabilized.py`, and the
  focused validator regressions.

## 2026-09-18 — Hold the completed committed-history residual experiment
- Decision: Preserve the fresh-zero strict
  `engineering_grip_anderson_residual_u5p9875e3_dumin6p25e6_s144` artifact set
  as experimental evidence and authorize no further physical endpoint. Do not
  resume or overwrite it, and do not run `u_bar=0.0060` or any higher endpoint.
- Why: With the exact prior scalar configuration and unchanged physical/final
  gates, the residual policy reached `u_bar=0.0059875` in `60` accepted steps,
  `0` cutbacks, and `593` complete sweeps. Its production time was
  `1669.1594366817735 s`, versus documented production timings `13038.79 s`
  for historical Anderson and `8765.93 s` for fixed Picard. The endpoint is
  finite, monotone, notch-localized, and independently x64-replayed, but the
  scalar stream records `100` combined hard-prerequisite candidate rejections
  and `19` sufficient-decrease rejections, while rejected trial vectors remain
  unavailable under the lean v5 archive contract.
- Formulation provenance: Walker--Ni and Saad support the fixed-point/Anderson
  notation used by the implementation. The committed-history map residual
  `r(d)=G_c(d)-d`, candidate/reference-specific normalizers, `0.995` selection
  threshold, and exact `omega=0.5` fallback are project-owned numerical policy;
  no PDE, constitutive law, boundary treatment, physical gate, or legacy
  `anderson` semantics changed.
- Alternatives rejected: promoting the policy from one bounded endpoint,
  treating accepted-state replay as rejected-path replay, mixing replay time
  into production timing, relaxing hard/final gates, continuing above the
  ceiling, or changing the legacy artifacts.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md`,
  `docs/PROGRESS.md`, and the durable evidence under
  `/root/autodl-tmp/data/fracture2d/engineering_grip/`.

## 2026-09-18 — Retain residual policy with corrected timing and path interpretation
- Decision: Retain `anderson_residual` as a distinct experimental policy after
  independent result review. Correct the previous entry's historical Anderson
  production timing: `12703.620977 s` is production; `13038.790514 s` is the
  workflow total. Correct production speedups are `7.6108x` versus historical
  Anderson and `5.2517x` versus fixed Picard.
- Why: The bounded strict run and accepted-field replay pass with 60 steps,
  zero cutbacks, 593 sweeps, and maximum cost 47 sweeps. Endpoint differences
  include an independently located intermediate history maximum sampled by
  the old cutback paths but missed by the new path. Strict convergence gates
  alone cannot establish load-step convergence of irreversible history.
- Alternatives rejected: treating differing-path timings as controlled
  benchmarks, describing field differences as solver-tolerance identity,
  attributing combined hard rejections to a component not recorded, or promoting
  one checkpoint to a complete peak/post-peak validation.
- Related: the preceding completed residual-experiment decision and
  `docs/tasks/fracture2d-engineering-grip-baseline.md`. The next recommended
  study is an explicitly specified matched-load schedule within the same
  endpoint; no farther continuation is authorized here.

## 2026-09-18 — Contract matched-load schedules before any physical run
- Decision: Implement an opt-in workflow `--load-schedule-npz` only for
  stabilized strict `anderson_residual`. Require a numeric, no-pickle NPZ with
  one exact little-endian float64 one-dimensional nonzero target array and
  JSON-compatible binding metadata. Validate finite strict ordering, the
  explicit endpoint rule, source archive existence/SHA256, and bitwise equality
  between targets and the historical archive's `state_u_bar[1:]`; the reviewed
  canonical artifact has 66 targets and raw-target SHA256 `13d294...`, bound to
  the historical archive SHA256 beginning `4c9...`.
- Decision: In schedule mode prepend exact zero and attempt each target once;
  cutback count is always zero, with no adaptive `_next_target`, retry,
  insertion, or cutback. Required parser increment fields remain inactive
  configuration inputs and are not a minimum-increment guarantee. Strict
  failure rolls back durably and names the exact next schedule target.
- Decision: Use schedule-specific stabilized completed/failure schema and replay
  identity rather than reinterpreting legacy v5. Bind mode, canonical path,
  file hash, target hash/count/endpoint, and source path/hash through config,
  scalar header/final, summary, replay binding/report, and manifest. Preserve
  adaptive controller/archives and historical `anderson` semantics unchanged.
- Why: A matched load path separates solver-policy effects from irreversible
  history sampling while making the target sequence independently auditable and
  preventing adaptive cutbacks from silently changing the comparison. Distinct
  provenance prevents old adaptive artifacts from being relabeled as schedule
  runs.
- Alternatives rejected: embedding targets in code or JSON/pickle, inferring a
  schedule from an archive without source/hash binding, allowing schedule mode
  in warning/legacy/other policies, treating inactive increment fields as
  guarantees, reusing the adaptive retry state machine, or running the physical
  comparison before validator and replay review.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md`,
  `examples/fracture2d/nonlocal_engineering_grip_stabilized.py`,
  `examples/fracture2d/engineering_grip_workflow.py`, and the reviewed
  historical Anderson archive.

## 2026-09-18 — Make matched-load provenance an explicit workflow artifact
- Decision: Expose schedule extraction as a read-only workflow subcommand that
  creates a no-pickle NPZ through an exclusive destination operation. Keep the
  schedule archive, stabilized schedule archives, scalar diagnostics, replay
  bindings, and manifest bound to the same canonical target/source payload.
- Why: A Python helper alone is not a durable user workflow, and a
  check-then-replace write could overwrite a concurrently created artifact.
  Independent replay must also distinguish a complete matched path from a
  strict failure at the exact next target.
- Alternatives rejected: embedding the reviewed sequence in source code,
  allowing `--overwrite` for schedule extraction, treating a failure archive's
  final committed load as its next target, or reusing the historical adaptive
  v5 schema for schedule runs.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md`,
  `examples/fracture2d/nonlocal_engineering_grip_stabilized.py`,
  `examples/fracture2d/engineering_grip_workflow.py`.

## 2026-09-18 — Close the public schedule provenance seam
- Decision: Make `StabilizedLoadSchedule.__post_init__` perform one-time,
  no-pickle artifact validation of the schedule path/file digest, canonical
  target bytes, historical source path/digest/state sequence, and exact
  configuration endpoints. Revalidate that immutable binding at config and
  controller entry, outside all numerical/JIT loops. Add manufactured
  scheduled-controller success/failure archives and reject forged target,
  metadata, and next-target bindings.
- Why: The public frozen dataclass previously accepted arbitrary hashes and
  nonexistent paths; a caller could therefore construct a schedule that passed
  configuration and only failed later during archive/replay I/O. The reviewed
  canonical target/source contract must be established before a controller can
  run, while one-time filesystem reads do not alter the fixed-shape numerical
  path.
- Alternatives rejected: a caller-supplied provenance flag/token, trusting the
  dataclass's frozen fields without loading the artifacts, putting source I/O in
  stagger/JIT loops, or weakening schedule mode to accept legacy, warning, or
  adaptive policies.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md`,
  `examples/fracture2d/nonlocal_engineering_grip_stabilized.py`,
  `tests/test_fracture2d_nonlocal_engineering_grip_stabilized.py`, and
  `tests/test_fracture2d_engineering_grip_workflow.py`.

## 2026-09-18 — Retain the completed matched-load residual experiment

- Decision: Preserve the fresh-zero strict scheduled
  `anderson_residual` run as experimental, nonproduction evidence. Do not
  resume, overwrite, promote, or continue it beyond exact
  `u_bar=0.0059875`; the matched-load subtask is complete pending primary
  review of the durable comparison report.
- Why: The canonical 66-target schedule is bitwise equal to the historical
  Anderson `state_u_bar[1:]`; the new archive has 67 states, 66 attempts, zero
  cutbacks, 822 accepted-state sweeps, and a maximum of 43 stagger iterations.
  Independent CPU/x64 replay, analysis, and plotting all bind to archive
  SHA256 `c2565925e599678b38d105c63e5410aacd5d87146521d361b089bcc3ba0488ed`
  and report `field_equations_validated=true`.
- Result: At all 67 matching states, the maximum matched-versus-historical
  infinity-norm differences are `2.67634785861004e-10` in union displacement,
  `6.04185976804672e-08` in damage, and `6.821109599286501e-05` in history.
  Endpoint differences are respectively
  `1.960145956138304e-10`, `4.8201790447954806e-08`, and
  `6.17388192836188e-05`; maximum top/bottom/anchor reaction differences are
  `5.219860668148613e-06`, `6.384582547980244e-06`, and
  `6.354155860407995e-06`. The fixed-Picard archive shares 66 loads and
  differs from the schedule only by the additional matched target
  `0.005982812500000004`; matched-versus-fixed field differences remain of
  the same order.
- Result: The matched path is finite, damage/history/reaction monotone under
  the declared tolerances, crack-localized, and all retained final gate flags
  pass. Final damage/history are `0.7465332077673724` and
  `282.0311557111417` at `(0.5,0.5)`; final top reaction is
  `638.6926722766116`.
- Timing: Production completion elapsed is `2803.56144060893 s`, versus the
  retained historical Anderson production reference `12703.620977 s` and
  fixed-Picard `8765.930929828435 s`, giving `4.531244x` and `3.126713x`
  ratios. Replay, analysis, and plotting are excluded. The matched schedule
  controls the historical comparison; fixed Picard still lacks one matched
  intermediate state, so neither timing nor history evolution is a universal
  load-step-convergence claim.
- Solver diagnostics: The scalar stream records 756 accelerated candidate
  attempts, 490 candidate-map evaluations, 466 selected accelerated
  candidates, 266 combined hard-prerequisite rejections, 24 sufficient-
  decrease rejections, 356 Picard fallbacks, and no candidate-map failures.
  Hard rejection records retain aggregate failure status rather than a full
  per-gate causal attribution; rejected trial vectors remain unavailable.
- Provenance: The durable comparison is
  `/root/autodl-tmp/data/fracture2d/engineering_grip/engineering_grip_anderson_residual_matched_historical_u5p9875e3_s144.comparison.rerun.json`
  with SHA256
  `7d2a2b9854a789d44181890a46e101c649b06ad33af354aa12dfd8fc137bc0e5`.
  The schedule file SHA256 is
  `8c8fcc7f1ef5bcf0239fbd3fca160ea3e8ff9644f4b9b53cd81aaf6775b38e0a`, its
  target SHA256 is
  `13d294d9f3817bfbf0da5eec1622f278cc6217f0e4c528275fbdbfe1fbb10cb5`, and
  its historical source SHA256 is
  `4c9a6f7e68f26afe01ba3fcf15a44a7b588174671128001dab440dc6b08fcbdc`.
- Formulation provenance: Walker--Ni and Saad support the fixed-point and
  Anderson notation. The committed-history residual merit, candidate-specific
  normalizers, `0.995` sufficient-decrease rule, exact `omega=0.5` fallback,
  and matched schedule are project-owned numerical policy. No PDE,
  constitutive law, boundary treatment, physical gate, or legacy `anderson`
  semantics changed.
- Alternatives rejected: treating small matched field differences as bitwise
  identity, inferring rejected-path replay from scalar records, attributing
  combined hard rejections to one unrecorded gate, promoting the experimental
  policy from one endpoint, or running a higher load.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md`,
  `docs/PROGRESS.md`, and the immutable artifacts under
  `/root/autodl-tmp/data/fracture2d/engineering_grip/`.

## 2026-09-18 — Close the matched comparison with a separate audit correction

- Decision: Retain the completed matched-load residual run as valid
  experimental evidence. Preserve the original comparison report, but bind a
  separate correction audit because its reduced scalar-binding extraction
  falsely recorded `final_archive_hash_matches=false`; the scalar final record
  and archive SHA256 are exactly identical.
- Why: Independent review verified all report and manifest hashes, the exact
  66-target schedule, 67 retained states, strict gates, replay, monotonicity,
  localization, and matched field differences. Matching the historical load
  path reduces maximum displacement/damage/history differences to
  `2.676e-10/6.042e-8/6.821e-5`, supporting the load-sampling diagnosis while
  not claiming bitwise or load-step-converged identity.
- Alternatives rejected: overwriting the original derived report, ignoring its
  false binding flag, treating the historical production reference as currently
  durable in the artifact directory, promoting the solver to production, or
  running a farther endpoint/new schedule.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md`, the original
  `.comparison.rerun.json`, and the separate `.comparison.audit.json` artifact.

## 2026-09-18 — Authorize observational hard-gate attribution before fresh-zero run
- Decision: On the existing engineering-grip task, implement only scalar,
  observational per-component hard-rejection attribution for experimental
  `anderson_residual` before the separately authorized fresh-zero strict run
  capped at `u_bar=0.0061`. Do not execute that physical continuation in this
  session.
- Decision: Preserve all equations, hard/final gates, candidate evaluation
  order, fallback and numerical results, historical `anderson` semantics, old
  artifact interpretation, and archive hash/source/config binding. Record the
  already computed candidate booleans and normalized/raw metrics, a canonical
  ordered failed-gate list, explicit unavailable/null stage reasons, and scalar
  JSONL only. Add compatibility for old scalar streams without changing v5 or
  schedule accepted-summary meaning.
- Decision: Current diagnostic output must validate internal consistency and
  reject forged lists, booleans, and metrics. Analysis reports gate counts and
  overlaps rather than exclusive causal attribution. Verification must include
  callback parity, manufactured individual/overlap/missing-stage cases, forged
  scalar/parser checks, old-stream compatibility, no-extra-solve checks, and
  fatal I/O propagation. No new CLI setting or numerical/adaptive rule is
  authorized.
- Why: The completed matched-load stream has aggregate hard-prerequisite
  rejections but no per-component attribution. This instrumentation supplies
  the missing audit evidence without adding hypothetical solves or changing
  the experimental policy, while the pending fresh-zero run must consume only
  independently reviewed implementation evidence.
- Alternatives rejected: reinterpreting old rows, adding a Picard/map or
  mechanics solve for comparison, changing gate ordering or acceptance,
  storing vectors, weakening validation, or running the physical endpoint
  before implementation review.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md`,
  `examples/fracture2d/nonlocal_engineering_grip_stabilized.py`,
  `examples/fracture2d/engineering_grip_workflow.py`, and the immutable
  matched-load artifacts.

## 2026-09-18 — Keep residual hard-gate attribution scalar and marginal
- Decision: Represent residual-policy hard-gate attribution as optional scalar
  proposal fields in the existing Anderson JSONL v1 stream. Use the fixed order
  `finite`, mechanics-before/after, force, moment, lower/upper damage range,
  healing, and history ordering; retain null metrics and explicit stage reasons
  for unavailable quantities. Workflow analysis reports marginal failures and
  pairwise overlaps rather than exclusive causes.
- Why: The production diagnostic already computes the required booleans,
  residuals, scales, and extrema. A scalar snapshot preserves that evidence
  without retaining trial vectors or adding a solve/map/mechanics evaluation,
  while the split range bounds remove the ambiguity in the aggregate range
  flag.
- Alternatives rejected: changing the v5/schedule archive schema, adding a
  Picard comparison solve, reconstructing rejected fields, inventing normalized
  range/healing tolerances, or reinterpreting historical `anderson` rows.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md`,
  `examples/fracture2d/nonlocal_engineering_grip_stabilized.py`,
  `examples/fracture2d/engineering_grip_workflow.py`.

## 2026-09-19 — Bind both stabilized diagnostic archive outcomes

- Decision: Bind the scalar Anderson diagnostic header to both the completed
  archive and its possible `*.failure.npz` archive, while retaining the
  completed path as the primary path. Finalization and loading accept only
  those explicitly bound paths.
- Why: A continuation failure legitimately finalizes the same diagnostic stream
  against the failure archive. Binding only the completed path made valid
  failure analysis fail its own provenance check; the correction changes no
  equations, gates, candidate ordering, or numerical evaluations.
- Alternatives rejected: dropping archive-path validation, binding arbitrary
  paths, creating a second diagnostic stream for failures, or changing the
  historical scalar stream interpretation.
 - Related: `docs/tasks/fracture2d-engineering-grip-baseline.md`,
   `examples/fracture2d/nonlocal_engineering_grip_stabilized.py`,
   `examples/fracture2d/engineering_grip_workflow.py`, and the residual scalar
   diagnostic contract tests.

## 2026-09-19 — Pair scalar diagnostic archive identity fields
- Decision: Extend the current two-path scalar diagnostic binding with an
  ordered completed/failure table. Each entry pairs its exact path suffix
  (`.npz` or `.failure.npz`), archive kind, completion status, and archive
  schema. Finalization and loading must validate the stabilized NPZ identity
  and reject cross-paired paths, statuses, kinds, or schemas for both adaptive
  and scheduled archives.
- Why: Separate path, schema, and status lists admitted a completed path with
  a failure status/schema, and scheduled validation checked only the schedule
  payload rather than the archive's complete identity. The canonical table
  makes the accepted failure outcome explicit without changing numerical
  evaluation or archive layouts.
- Alternatives rejected: accepting any path from the legacy lists, checking
  only the archive hash, adding a new archive schema, or reinterpreting old
  scalar streams that lack the current binding table.
 - Related: `docs/tasks/fracture2d-engineering-grip-baseline.md`,
   `examples/fracture2d/nonlocal_engineering_grip_stabilized.py`,
   `examples/fracture2d/engineering_grip_workflow.py`,
   `examples/fracture2d/README.md`, and the scalar diagnostic contract tests.

## 2026-09-19 — Complete authoritative hard-rejection metric and parity contract
- Decision: Treat residual-policy hard-gate attribution as an observational
  serialization of the existing primitives, with no new solve, map evaluation,
  gate, or numerical rule. The canonical schema is
  `fracture2d-engineering-grip-anderson-residual-hard-gates-v1`, ordered as
  `finite`, `mechanics_before`, `mechanics_after`, `force`, `moment`,
  `damage_range_lower`, `damage_range_upper`, `healing`, and
  `history_ordering`.
- Contract: Each gate carries exactly availability/pass flags, raw and limit
  metrics, ratio availability/normalization, a canonical metric reason and
  stage reason, plus CG status/residual fields where applicable. Mechanics
  attribution uses the already-computed authoritative `cg_info`,
  `free_residual_norm`, and `cg_convergence_limit`; its pass is
  `cg_info == 0` together with the strict residual-limit check. Force/moment,
  range, healing, history-ordering, and finite retain their existing gate
  semantics, including null metrics for unavailable stages and no invented
  ratio for the zero history-ordering limit. `failed_hard_gates` is derived in
  canonical order and `hard_pass` must agree with it.
- Decision: Require strict validation of forged metric types, booleans, lists,
  reasons, raw/limit/ratio relationships, CG subfields, and residual-policy v5
  selection metrics. Preserve callback-on/off parity for fields, path,
  selection, and solve counts; hard rejection must retain zero candidate-map
  evaluations. Historical scalar streams without attribution fields remain
  readable, and historical `anderson` semantics/artifacts remain unchanged.
- Why: The preceding archive-pair table fixed path/kind/status/schema pairing
  but did not by itself close authoritative CG attribution, forged-metric
  rejection, or residual callback parity. The completed contract closes those
  audit points without altering numerics or physical evidence.
- Verification: Sequential CPU/x64 focused tests pass stabilized `59/59` and
  workflow `35/35`; compilation and `git diff --check` pass. No physical
  continuation or durable artifact change was performed.
- Alternatives rejected: weakening metric validation, deriving CG status from
  a proxy, adding comparison solves, changing the residual policy, or
  reinterpreting old scalar streams.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md`,
  `docs/PROGRESS.md`, `examples/fracture2d/nonlocal_engineering_grip_stabilized.py`,
  `examples/fracture2d/engineering_grip_workflow.py`, and the focused contract
  tests.

## 2026-09-19 — Make force/moment hard-gate attribution stage-consistent
- Decision: Extend the residual-policy hard-gate attribution with explicit
  `before` and `after` force/moment stage metrics. Each stage records the
  already-computed error, stage scale, tolerance limit, normalized ratio, and
  exact production pass flag.
- Decision: Keep the aggregate force/moment pass as the conjunction of the two
  stage flags and the aggregate normalized value as the worst stage ratio. The
  retained top-level `raw`, `scale`, and `limit` fields are a deterministic
  representative tuple from the worst-ratio stage (with `before` winning an
  exact tie), so top-level `raw / limit == normalized`; they are not a new
  cross-stage physical quantity. The attribution schema is versioned to make
  this contract explicit.
- Why: The prior max-error/max-scale pair could disagree with the production
  before/after conjunction when stage scales differed, making `passed` and
  `raw <= limit` contradictory. A worst-ratio representative preserves scalar
  compatibility while making every retained ratio self-consistent.
- Alternatives rejected: retaining the max-error/max-scale pair, which is not
  a metric pair for the aggregate ratio; adding a solver or comparison solve;
  changing the mechanics gates, evaluation order, candidate selection, or
  legacy `anderson` semantics; and inventing a summed or otherwise physical
  cross-stage force/moment quantity.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md`,
  `examples/fracture2d/nonlocal_engineering_grip_stabilized.py`,
  `examples/fracture2d/README.md`, and the stabilized/workflow contract tests.

## 2026-09-19 — Separate candidate-map failure and close v3 durable attribution
- Decision: Version the residual hard-gate attribution as v3. A numerical
  candidate-map failure is recorded with `hard_gate_status=candidate_map_failure`,
  `hard_pass=false`, and no invented failed physical gates; its proposal context
  must identify an incomplete accelerated candidate whose map status is failed
  at `anderson_candidate_damage`. Evaluated hard gates retain their ordinary
  ordered failed list.
- Decision: Permit production-valid zero force/moment limits. Zero error against
  zero limit passes with no normalized ratio; positive error fails. Aggregate
  severity is ordered as positive-error/zero-limit, finite ratio, then exact
  zero-error/zero-limit, with canonical before-stage tie-breaking.
- Decision: Allow schema-less attribution fragments only internally. Writer and
  loader boundaries reject any attributed durable record without the current
  scalar schema and context fields.
- Why: Independent review reproduced a callback-only failure of the intended
  Picard fallback and identified zero-limit and durable-record forgery gaps. The
  v3 contract fixes audit semantics without adding solves or changing numerical
  selection. Stabilized/workflow tests pass `98/98` with callback solve-count
  parity; compilation and `git diff --check` pass.
- Alternatives rejected: labeling numerical map failure as a physical gate
  failure, disallowing a configuration accepted by `GateTolerance`, weakening
  validation, or changing fallback/evaluation behavior.
- Related: `examples/fracture2d/nonlocal_engineering_grip_stabilized.py`,
  `examples/fracture2d/engineering_grip_workflow.py`, the fracture2d README, and
  the focused scalar-contract tests.

## 2026-09-19 — Stop the authorized fresh-zero continuation at exact 0.0061
- Decision: Accept the strict fresh-zero `anderson_residual` run through exact
  `u_bar=0.0061` as validated pre-peak evidence, but do not call it a peak or
  post-peak baseline and do not continue beyond the authorized endpoint without
  a new decision.
- Why: The run completed 61 accepted increments with zero cutbacks and maximum
  62 stagger sweeps; independent accepted-state replay passed. Top reaction was
  still monotone, increasing from `639.1241866042104` at `0.0060` to
  `641.3384950971831` at `0.0061`, while maximum damage reached
  `0.8202621313922084`.
- Evidence: archive SHA256
  `68c934008ccf95e71ac2d4ac78f1fa82fd16534cc6b2accba0b21d21c60021f8`;
  scalar stream SHA256
  `9d1c740d7565099f514ac0351fce61d960551461ddc9aff1f1dc68c7d9ebfb75`.
  All 817 proposals were attributed. The only hard-gate rejection was healing
  on 142 accelerated candidates; no mechanics, force, moment, range, history,
  or candidate-map numerical failures occurred.
- Alternatives rejected: silently extending to `0.0062` or higher, inferring a
  peak from the shrinking but still positive reaction increment, or treating
  accepted-state replay as rejected-path replay.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md` and the durable
  `engineering_grip_anderson_residual_peak_u6p1e3_s144` artifact family.

## 2026-09-19 — Authorize one fresh-zero extension to exact 0.0062
- Decision: Run one fresh-zero strict adaptive `anderson_residual` continuation
  to exact `u_bar=0.0062` with the independently approved v3 scalar diagnostic
  and all parameters unchanged from the completed `0.0061` run.
- Why: The `0.0061` reaction remained monotone but its final increment had
  narrowed to `2.214308492972691`; one minimum endpoint extension is needed to
  determine whether the first reaction decline occurs.
- Bound: Any strict or structural failure stops the run. Do not warning-accept,
  resume from `0.0061`, alter the increment policy, or continue to `0.0063`.
- Alternatives rejected: inferring a peak from a positive increment or making a
  wider unreviewed endpoint jump.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md` and
  `docs/PROGRESS.md`.

## 2026-09-19 — Bind gate-ratio report provenance to stored solver summaries
- Decision: Derive gate-ratio report locations from each diagnostic row's stored
  solver-summary `sweep_index` and authoritative load field, using
  `target_u_bar` only as a consistency alias for `load_target`. Reject missing,
  mistyped, inconsistent, non-finite, or out-of-range provenance rather than
  falling back to diagnostic-row enumeration.
- Why: The failed exact-`0.0062` archive reported gate locations at
  `sweep_index=63` even though its authoritative final failure summary stored
  `sweep_index=143`. The derived report therefore misidentified where the
  largest ratios occurred.
- Alternatives rejected: retaining enumeration-based locations, silently
  defaulting missing summary fields, changing solver numerics, or rewriting the
  physical archive.
- Related: `examples/fracture2d/engineering_grip_workflow.py`,
  `tests/test_fracture2d_engineering_grip_workflow.py`,
  `docs/tasks/fracture2d-engineering-grip-baseline.md`, and the failed
  `engineering_grip_anderson_residual_peak_u6p2e3_s144` artifact family.

## 2026-09-19 — Stop the exact-0.0062 attempt at the strict du_min failure
- Decision: Preserve the exact-`0.0062` attempt as a strict failure artifact and
  stop. The last accepted state is `u_bar=0.0061125000000000285`; the next exact
  target `0.006118750000000029` failed after 144 sweeps at
  `du_min=6.25e-6`. Do not use rejected-trial reactions to claim a peak.
- Why: Accepted reaction remained monotone, rising from `641.338492718448` at
  `0.0061` to `641.3882615552535` at `0.0061125`, a positive increment of
  `0.04976883680546962`. The final rejected sweep failed displacement, damage,
  history, reaction, damage residual, fixed-point, and coupled convergence even
  though finite, mechanics, force, moment, range, and healing hard gates passed.
- Evidence: failure archive SHA256
  `979f66ab19b9f74a52871cfeb56fa2f3efd57bd53197a40d9b516f159299c1ad`;
  scalar stream SHA256
  `1b9dc58a2eed2ee5ddf0fb063e339a55488949923995e57427c57670ace129d4`.
  Accepted-state replay passed, rejected-path replay remains unavailable, and
  all 9 manifest-listed artifacts pass hash/size audit after the derived report
  provenance correction. The corrected manifest SHA256 is
  `efd1970ead9b685d99980df66b46eba33e1f82c890536dd2e4e1bf9b578bbd5a`.
- Alternatives rejected: warning acceptance, treating the rejected final sweep
  as physical evidence, resuming the failed archive, silently lowering `du_min`,
  or continuing to `0.0063`.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md` and the durable
  `engineering_grip_anderson_residual_peak_u6p2e3_s144` artifact family.

## 2026-09-19 — Continue until a strict direct-solver final curve is obtained
- Decision: Continue the active engineering-grip task until the nonlocal direct
  solver has a strict, replayable peak/post-peak curve through the configured
  exact endpoint `u_bar=0.0075`. Intermediate bounded diagnostics and failed
  attempts may proceed without separate endpoint authorization, but every
  failure must stop, roll back, and remain auditable.
- Decision: Preserve the physical formulation and hard gates. Do not use warning
  acceptance, clipping, projected Anderson candidates, rejected states, or an
  unreferenced monolithic/arc-length equation to manufacture convergence.
- Why: The user explicitly requested uninterrupted continuation until the final
  nonlocal direct-solver solution is obtained. Existing task/config history
  defines the complete curve endpoint as `0.0075`.
- Alternatives rejected: treating `0.0061125` as final, accepting rejected
  reactions as a peak, or changing constitutive/discrete physics to cross the
  instability.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md`,
  `examples/fracture2d/README.md`, and `examples/fracture2d/configs/solver_base.py`.

## 2026-09-19 — Diagnose stronger damping before a new nonlinear policy
- Decision: First test existing fixed Picard with `omega=0.25` on a bounded
  one-step checkpoint from the independently replayed accepted state at
  `u_bar=0.0061125` to target `0.00611875`. This is a diagnostic, not a resumed
  production curve or permission to commit the state into historical evidence.
- Why: Scalar evidence shows the rejected map is marginally noncontractive:
  tail residual factor about `1.0073`, 81 of 143 ratios increasing, and Anderson
  history repeatedly collapsing to depth 1. Fixed-history damage, mechanics,
  finite, force, moment, range, and healing gates pass. Stronger damping is the
  smallest existing algorithmic change that may suppress the coupled
  history-feedback oscillation.
- Bound: Keep equations, committed history, target, strict gates, `du_min`, and
  144-sweep ceiling unchanged. Do not launch a full `0.0075` run unless the
  checkpoint demonstrates strict contraction and acceptance.
- Alternatives rejected for now: deeper Anderson while history is repeatedly
  reset, direct BFGS copied from a different local or PD-PFM formulation,
  clipping/active-set changes unrelated to the passing Picard hard gates, and
  arc-length without a derived continuation constraint.
- Related: the exact-`0.0062` scalar failure stream, Walker-Ni/Fang-Saad Anderson
  references, Kristensen-Martinez-Paneda (2020), and Bie et al. (2023).

## 2026-09-19 — Fixed-Picard omega-0.25 checkpoint remains rejected
- Decision: Stop after the single authorized fixed-Picard diagnostic. Do not
  promote its target, resume the source archive, or launch the `0.0075` curve.
- Why: The one-step attempt from the accepted `0.0061125` state completed all
  144 sweeps at `omega=0.25` but failed the strict coupled gates. Fixed-point
  error ratios ranged from `0.9857316138435906` to `1.075848052697296`; the
  final ratio was `1.075848052697296`, so the map was not strictly contractive.
- Evidence: Scalar diagnostic
  `engineering_grip_fixed_picard_omega025_one_step_u6p11875_20260919T072222Z.json`
  has file SHA256
  `2fe6924d8ac2e98f77e78a1b07e41ec7941d52e7c5eb6815185c3951b34cc82e`. The
  source failure archive hash remained
  `979f66ab19b9f74a52871cfeb56fa2f3efd57bd53197a40d9b516f159299c1ad` before
  and after the diagnostic.
- Alternatives rejected: treating the complete failed sweeps as a physical
  state, weakening gates, adding cutbacks/backtracking to this checkpoint, or
  selecting a new nonlinear method without a declared and independently checked
  discrete formulation.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md` and the bounded
  fixed-Picard scalar artifact.

## 2026-09-19 — Specify a distinct backtracked residual-Anderson controller
- Decision: Implement `anderson_residual_backtracking` without changing the
  historical residual policy. Build the same finite Type-II candidate with
  depth 5 and beta 0.5, then test
  `d(alpha)=d_iter+alpha*(d_AA-d_iter)` for exact
  `alpha=(1,1/2,1/4,1/8,1/16)`. No clipping or projection is permitted.
- Decision: For each alpha, evaluate existing mechanics/history diagnostics and
  hard gates first. A hard-gate failure rejects only that alpha and performs no
  candidate-map solve. After a hard pass, evaluate the exact committed-history
  map residual and accept the first trial satisfying
  `m_trial <= (1 - 0.01*0.5*alpha)*m_reference`. This is a project-owned bounded
  decrease rule, not a claim of standard Armijo theory.
- Decision: Candidate-map numerical failure is trial-local and proceeds to the
  next alpha. Valid stored base map pairs survive hard-gate, map, and merit trial
  rejection. Restart history only when proposal construction itself fails due
  to rank/QR/triangular/nonfinite data. After all alpha trials reject, evaluate
  the existing exact omega-0.5 Picard fallback once under unchanged semantics.
- Decision: Add explicit nullable `trial_alpha` provenance and a distinct merit
  identity; do not overload historical accelerated-trial `omega=None` or
  reinterpret old archives/scalar streams.
- Why: Both omega-0.5 and omega-0.25 Picard are noncontractive near
  `0.00611875`, while current Anderson loses useful secant history after
  trial-local healing/merit rejection. Backtracking the secant direction is the
  smallest map-compatible controller that can preserve admissibility without
  changing physics.
- Alternatives rejected: immediate restart for trial-local failure, evaluating
  candidate maps after failed hard gates, overloading `omega`, clipping,
  unbounded line search, or launching a full curve before a bounded checkpoint.
- Related: `examples/fracture2d/nonlocal_engineering_grip_stabilized.py`, the
  workflow CLI/replay contracts, and the active engineering-grip task.

## 2026-09-19 — Independently re-review backtracked residual Anderson fixes
- Decision: Recommend GO for the separately bounded physical checkpoint only;
  do not authorize a full physical curve from this implementation review.
- Evidence: v5 alpha provenance is checked against `backtrack_count`, scalar
  proposal provenance is checked against `trial_alpha_index`, finite trial
  attribution is explicit `not_evaluated`, and callback/no-callback solve
  counts agree. The summary excludes unavailable-stage rows from physical
  failure/overlap marginals while retaining unavailable counts.
- Verification: focused review tests `6/6`, stabilized/workflow `110/110`,
  legacy engineering-grip/JAX/baseline `67/67`, compileall, and
  `git diff --check`; no physical continuation was run.
- Source-backed versus project-owned: Walker--Ni/Saad support the fixed-point
  Anderson notation and Type-II least-squares construction. The depth, beta,
  alpha schedule, decrease rule, hard-gate ordering, fallback, and attribution
  schema are project-owned choices documented in the active task.
- Alternatives rejected: running the physical checkpoint before the
  provenance/parity review, treating unavailable gates as physical failures,
  or changing the historical `anderson` stream contract.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md`,
  `examples/fracture2d/nonlocal_engineering_grip_stabilized.py`, and the
  stabilized/workflow tests.

## 2026-09-19 — Prepare an isolated bounded backtracking checkpoint runner
- Decision: Keep the separately approved physical checkpoint in a new
  `/tmp/opencode/fracture2d_anderson_residual_backtracking_checkpoint.py` runner.
  It reconstructs the source archive's last strict accepted state, invokes the
  private stabilized attempt exactly once with
  `anderson_residual_backtracking`, captures scalar sweep/callback provenance,
  and writes only a timestamped exclusive-create JSON result.
- Why: The implementation review authorizes this bounded checkpoint only; the
  historical source archive and all production continuation paths must remain
  immutable, and the lean controller cannot claim rejected-field replay.
- Alternatives rejected: calling the fresh-zero continuation wrapper, resuming
  the rejected source target, retaining rejected trial vectors, overwriting the
  fixed-Picard provenance script, or launching a full curve.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md`,
  `/tmp/opencode/fracture2d_fixed_picard_omega025_checkpoint_20260919T072222Z.py`,
  and the bounded backtracking review entries above.

## 2026-09-20 — Hold backtracking checkpoint on target-contract mismatch
- Decision: Do not execute the scratch backtracking checkpoint until its target
  is reconciled with the active task contract. The runner encodes
  `0.0061125 -> 0.006115625` with increment `3.125e-6`, while
  `PROGRESS.md`, the active task, and the preceding bounded-checkpoint decision
  specify `0.0061125 -> 0.00611875` with increment `6.25e-6`.
- Why: These are distinct physical load checkpoints and produce different
  nonlinear maps; silently changing either the runner or the task memory would
  invalidate the authorization and provenance contract.
- Alternatives rejected: executing the runner as-is, silently retargeting it to
  `0.00611875`, or treating the half-step filename as sufficient numerical
  authorization.
- Related: `/tmp/opencode/fracture2d_anderson_residual_backtracking_checkpoint.py`,
  `docs/PROGRESS.md`, and
  `docs/tasks/fracture2d-engineering-grip-baseline.md`.

## 2026-09-20 — Retain strict `0.0062` evidence and hold nonlinear extension
- Decision: Retain the fresh-zero strict curve through exact `u_bar=0.0062`
  as replayed discrete-equilibrium evidence: 62 accepted steps, zero cutbacks,
  archive SHA256
  `95851472eb43eb2e24e4d26920d050ecf5ffc1aa83bfe0919f33bd7881cadada`, and
  scalar SHA256
  `14765f28710434b3ae13dc6a459b5afcdd15d2106a3164c69ae7c045297e6976`.
  Record the accepted `0.0061` reaction `641.3384921076807` and the
  post-peak `0.0062` reaction `631.4074774212969`, but retain the `k_res=0`
  crack-localized SPD conditioning result as a physical caveat.
- Decision: Preserve the accepted half-step checkpoint
  `0.0061125 -> 0.006115625` at `du_min=3.125e-6` and the failed same-target
  `0.0061125 -> 0.00611875` checkpoint as separate scalar evidence. Preserve
  the fresh-zero `0.0075` failure archive, rolled back at `0.006234375` after
  last accepted `0.00623125`; its failed gates are displacement, damage,
  history, reaction, damage residual, damage fixed point, and coupled. No
  rejected state is physical evidence.
- Why: The completed `0.0062` accepted states replay and satisfy field-equation
  checks, while the later strict attempt fails its coupled convergence gates
  even after the full 144-sweep budget. The conditioning study shows a valid
  discrete equilibrium but does not remove the physical caveat near the
  crack-localized mode.
- Blocker: No non-arbitrary regularization parameter/update rule is available
  for the proposed Tikhonov Type-II Anderson strategy; reducing `du_min` to
  force progress would require impractical microsteps. Do not guess `mu`,
  `k_res`, or arc length. The next action is a user/mathematical decision to
  approve a bounded regularization-parameter study/derivation or accept the
  strict curve only through `0.00623125` as incomplete.
- Verification: Existing records contain `110/110` relevant tests, `67/67`
  legacy tests from independent review, compilation, and `git diff --check`.
  No post-artifact test rerun is claimed beyond the noted independent
  replay/field-equation reviews.
- Alternatives rejected: warning-accepting or promoting the rejected state,
  treating the half-step as equivalent to the same-target failure, using an
  arbitrary regularization/floor/arc-length rule, or interpreting the
  conditioning result as proof of physical robustness.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md`,
  `docs/PROGRESS.md`, and the immutable engineering-grip artifacts.

## 2026-09-20 — Approve fixed-mu regularized residual-Anderson study policy
- Decision: Add the distinct experimental policy
  `anderson_residual_regularized_backtracking`. It inherits the exact
  `anderson_residual_backtracking` controller contract: committed-history map,
  consecutive-difference Type-II history, depth `5`, `beta=0.5`, alpha schedule
  `(1, 0.5, 0.25, 0.125, 0.0625)`, hard-gate ordering, merit threshold,
  trial-local rejection, history semantics, and exact `beta=0.5` Picard fallback.
  Only the proposal coefficient solve changes.
- Mathematical contract: with `f=d_solved-d_iter` and the existing `DF`
  consecutive-difference block, solve
  `gamma = argmin ||f-DF gamma||_2^2 + lambda ||gamma||_2^2` with
  `lambda = mu ||f||_2^2`, using an augmented rank-revealing QR solve of
  `[DF; sqrt(lambda) I] gamma ~= [f; 0]`. Normal equations are forbidden.
  `anderson_regularization_mu` is a required explicit finite positive
  configuration/CLI value; it has no default, adaptive update, or silent
  fallback. Proposal QR/rank/nonfinite restart semantics remain unchanged where
  applicable; structural augmented-solve failures restart history.
- Provenance contract: use a distinct merit identity and policy/config/archive/
  scalar identity. Proposal-level scalar diagnostics must retain `mu`, `lambda`,
  the unregularized least-squares residual norm, and coefficient norm. Lean v5
  accepted summaries/configuration must reject missing or forged `mu` identity,
  while genuine historical policy payload shapes remain readable unchanged.
- Study scope: this is an experimental bounded parameter study, not an Ouyang
  convergence claim. The approved study grid is `mu={0.25, 1, 2}`, corresponding
  to the published `mu0=1` and neighboring `eta2=0.25` / `eta1=2` updates; the
  implementation must accept any finite positive `mu` rather than hardcoding
  those three values.
- Source-backed versus project-owned: Walker--Ni/Saad support the fixed-point,
  Type-II consecutive-difference, least-squares, and QR notation; the
  regularized augmented-QR implementation and the Ouyang-linked bounded study
  values are the approved study contract here. The alpha schedule, gate order,
  merit threshold, fallback/restart semantics, provenance schema, and absence of
  any convergence claim are project-owned choices. No PDE, constitutive law,
  boundary treatment, history map, or physical continuation setting changes.
- Verification contract: add independent `np.linalg.lstsq` manufactured checks
  for full-rank, rank-deficient, zero/small-residual, and lambda-scaling cases;
  reject normal-equation implementations; verify unchanged candidate/history,
  CLI/config combinations, scalar/v5 forgery and missing fields, legacy payload
  compatibility, controller parity/solve counts, trial-local failure handling,
  restart/fallback behavior, and coefficient-norm reduction without clipping or
  projection. Do not run a physical checkpoint or create external artifacts.
- Alternatives rejected: changing historical policies, using normal equations,
  adapting or defaulting `mu`, hardcoding the three study values, clipping or
  projecting the candidate, claiming Ouyang convergence, or changing the
  committed-history/physical equations.
- Related: `docs/tasks/archive/anderson-residual-regularized-study.md`,
  `docs/tasks/fracture2d-engineering-grip-baseline.md`,
  `examples/fracture2d/nonlocal_engineering_grip_stabilized.py`, and
  `examples/fracture2d/engineering_grip_workflow.py`.

## 2026-09-20 — Complete fixed-mu regularized policy contract verification
- Decision: Regularized scalar proposal, sweep, and v5 records expose only the
  explicit fields `anderson_regularization_mu`,
  `anderson_current_residual_l2`, `anderson_regularization_lambda`,
  `anderson_unregularized_ls_residual_l2`, and `anderson_coefficient_l2`.
  Historical policy payloads retain their established shapes independently.
- Decision: The regularized controller is required to remain parity-equivalent
  to residual backtracking for alpha selection, hard-gate ordering, candidate
  map/fallback behavior, callback records, and mechanics-solve counts. The
  binding requires the fixed `mu` and the complete explicit field set.
- Verification: the expanded stabilized/workflow suites pass `118/118` in
  `431.568 s`; the affected legacy engineering/JAX/baseline suites pass
  `81/81` in `35.761 s`; `python -m compileall -q examples/fracture2d tests`
  and `git diff --check` pass. Expected argparse usage/error output is from
  negative CLI tests. No physical checkpoint or external artifact was run or
  created.
- Related: `docs/tasks/archive/anderson-residual-regularized-study.md`,
  `docs/PROGRESS.md`, and `examples/fracture2d/README.md`.

## 2026-09-20 — Prepare fixed-mu scalar sensitivity runner
- Decision: Prepare, but do not execute, a scalar-only bounded runner for
  `anderson_residual_regularized_backtracking` at the approved study values
  `mu={0.25, 1, 2}`. Bind the immutable failed-run source archive, its exact
  last accepted state, the production model/JAX adapter, and the historical
  strict base configuration; invoke one private target attempt only.
- Why: The requested sensitivity evidence must distinguish a bounded numerical
  diagnostic from continuation or physical evidence. Scalar callbacks, exact
  rollback/acceptance digests, and exclusive canonical JSON provenance preserve
  that boundary without serializing rejected field vectors.
- Alternatives rejected: running any physical attempt while preparing the
  runner, calling the continuation wrapper, changing the historical gates or
  load-step settings, or promoting a rejected state to physical evidence.
- Related: `/tmp/opencode/fracture2d_anderson_residual_regularized_sensitivity.py`,
  `docs/tasks/fracture2d-engineering-grip-baseline.md`, and `docs/PROGRESS.md`.

## 2026-09-20 — Bind source-code hash timing around the bounded attempt
- Decision: Capture the stabilized source-bundle and workflow hashes before the
  private target attempt, then recalculate them after it; archive, cloud, and
  runner hashes retain the same before/after contract.
- Why: A before/after provenance field must measure the attempt interval rather
  than two consecutive post-attempt reads, while preserving the no-mutation
  audit without executing a physical study here.
- Alternatives rejected: treating a post-attempt hash as the pre-attempt value,
  hashing field arrays, or running the target attempt merely to exercise the
  instrumentation.
- Related: `/tmp/opencode/fracture2d_anderson_residual_regularized_sensitivity.py`
  and `docs/tasks/fracture2d-engineering-grip-baseline.md`.

## 2026-09-20 — Complete negative fixed-mu sensitivity study
- Decision: Close the bounded scalar-only fixed-`mu` study as negative
  diagnostic evidence. The injected accepted state at
  `u_bar=0.0062312500000000345` was tested to exact target
  `u_bar=0.006234375000000035` with unchanged mathematics, `du_min=3.125e-6`,
  zero load cutbacks, and the full 144-sweep budget. All three attempts were
  rejected after 144 sweeps by the identical gates
  (`displacement`, `damage`, `history`, `reaction`, `damage_residual`,
  `damage_fixed_point`, `coupled`) and rolled back; no rejected state is
  physical evidence.
- Exact artifacts are:
  - `mu=0.25`: `/root/autodl-tmp/data/fracture2d/engineering_grip/engineering_grip_anderson_residual_regularized_backtracking_sensitivity_mu0p25_20260920T072155720312Z.json`, file SHA256 `a2d6afa5a5d86866d8de20ec37987873d243eb910941629869e363c31db787c6`, payload SHA256 `b5235a04b89ed84b1c63e92b3becf41930e1914945de829e512333e49eea8e46`.
  - `mu=1`: `/root/autodl-tmp/data/fracture2d/engineering_grip/engineering_grip_anderson_residual_regularized_backtracking_sensitivity_mu1_20260920T075026856378Z.json`, file SHA256 `c5f271504e3617b0e8d08683f958921fe045276a627739e3a444ddc86f5479ec`, payload SHA256 `31353edbc352f33fafd6fc67f9a063a29f45fb7c30786ffab085ef08c89e353a`.
  - `mu=2`: `/root/autodl-tmp/data/fracture2d/engineering_grip/engineering_grip_anderson_residual_regularized_backtracking_sensitivity_mu2_20260920T081716835642Z.json`, file SHA256 `c26a461ac64f7d54dc7d37b597f8064f9dedb69754d7394fd2dea67d9523d09a`, payload SHA256 `9b5685ba06c0742d8dc27256897349d2b2ae0289010f925fce90d4c25e6ea3cb`.
- Results: selected AA/Picard-fallback counts were `3/141`, `0/144`, and
  `0/144` for `mu=0.25`, `1`, and `2`; coefficient maxima were
  `0.7697/0.3656/0.2151`. Final fixed-point errors were
  `1.8632048281652658e-4`, `1.8608049513912928e-4`, and
  `1.8608033904232713e-4`, versus baseline
  `1.8391723755628586e-4`.
- Why: Increasing regularization suppresses AA toward Picard, and each tested
  value worsens the endpoint error relative to baseline. The study therefore
  does not justify expanding or adapting the `mu` grid. Source-bundle hashes
  differ by design because the new policy code changed, while source cloud,
  configuration, and rollback bindings match; comparisons are behavioral, not
  bitwise code-parity claims.
- Alternatives rejected: treating any rejected trial as an accepted physical
  state, expanding/adapting the tested `mu` grid from this negative result,
  guessing an arc-length rule, or changing `k_res` without a separate
  mathematical decision. The next decision is either unchanged mathematics
  with `du_min=1.5625e-6` (potentially hundreds of microsteps) or a materially
  new continuation/nonlinear formulation.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md`,
  `docs/PROGRESS.md`, and the archived implementation task
  `docs/tasks/archive/anderson-residual-regularized-study.md`.

## 2026-09-20 — Close smaller-du checkpoint and deeper-alpha audit
- Decision: Close the user-approved smaller-`du_min` checkpoint as a strict
  failure. Starting from the last accepted `u_bar=0.00623125` state with
  source digest
  `a5e0e6b78e1663d6e76262950730dcb5ee3a5a7bdc5c1f0d86f2b09adf7d9cd7`, the
  unchanged unregularized `anderson_residual_backtracking` policy targeted
  `u_bar=0.0062328125` with `du_min=1.5625e-6` and `144` sweeps. It failed
  strictly and rolled back; no rejected state is physical evidence.
- Evidence: the artifact is
  `/root/autodl-tmp/data/fracture2d/engineering_grip/engineering_grip_anderson_residual_backtracking_half_step_u6p2328125_20260920T094632Z.json`,
  file SHA256
  `28b3e0a751781b5f043ea2c2a2ed43852388a2b783e3d23c19bd9fb702cd0ba3`, and
  payload SHA256
  `e6b6434f6558afbc2143d58a3d7337ef04ffffd99eccc2b0d91f8845d3067159`.
  Final error was `4.779188545345203e-5`; the tail ratio was `1.0010643`, the
  last-ten maximum was `1.0012019`, and the same coupled gates failed:
  `displacement`, `damage`, `history`, `reaction`, `damage_residual`,
  `damage_fixed_point`, and `coupled`.
- Decision: Mark blanket deeper-alpha extension NO-GO. In the `alpha=1/16`
  audit, the last 30 trials were all hard-pass/map-complete, but only sweep
  `125` passed globalization; `29/30` directional slopes
  `(m(alpha)-m_ref)/alpha` were positive and the tail slopes were large.
  `alpha=1/8` was always worse.
- Why: The current controller path is exhausted: fixed Picard `omega=0.25`,
  unregularized alpha backtracking, fixed-`mu` `0.25/1/2`, the smaller-`du_min`
  checkpoint, and deeper-alpha evidence do not provide strict progress beyond
  `0.00623125`. Further damping or regularization would be guesswork rather
  than an independently derived numerical decision.
- Next decision: The user must select a materially new nonlinear formulation,
  such as an independently derived monolithic complementarity/semismooth-Newton
  method or a derived continuation constraint, versus stopping incomplete at
  `0.00623125`. Preserve the exact `0.0075` objective as unmet.
- Alternatives rejected: another guessed damping/regularization change,
  expanding/adapting `mu`, a blanket deeper-alpha extension, warning acceptance,
  using rejected fields as physical evidence, or inventing an arc-length rule,
  `k_res` change, or other continuation constraint without derivation.
- Verification: This was a memory-only update with no solver edits, tests, or
  physical runs; `git diff --check` passes.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md`,
  `docs/PROGRESS.md`, and the immutable checkpoint/failure artifacts.

## 2026-09-20 — Approve Phase 1 of the monolithic q-d-H engineering-grip formulation
- Decision: Implement Phase 1 only, as a private/experimental module
  `examples/fracture2d/nonlocal_engineering_grip_monolithic.py`: the exact
  code-compatible monolithic residual over the packed unknown
  `z=(q_free, d, H)` and its analytic generalized Jacobian action (JVP), with
  manufactured tests only. No Newton solver, GMRES, workflow/CLI, or physical
  run is authorized by this entry.
- Why: The user selected the pending choice in favor of an independently
  derived monolithic formulation, staged so that the residual/JVP contract is
  exactly derived and FD-verified before any solver exists. Phase 1 keeps every
  equation identical to the accepted staggered model: `Rq` is the existing
  `A2FixedDamageView` free-row residual at the affine hard-constraint target;
  `Rd = M rho` with `rho = Gc/l0 d + Gc l0 D d + 2 H (d-1)` (the existing
  material AT2 residual times particle volumes, no mechanics-energy term);
  `RH = H - max(H_committed, psi_plus(Q))` with the existing material-source
  embedded plane-strain Amor driver and the exact tie-committed strict-`>`
  active set. The analytic blocks are `Jqq` (existing tangent action on free
  DOFs), `Jqd` from the source center-weight derivative
  `dW/dd = -2 V_material (1-d)` times the existing endpoint edge-force
  assembly, `Jdd = M(diag(Gc/l0+2H)+Gc l0 D)`, `JdH = M diag(2(d-1))`,
  `JHq = -d psi_plus/dq` on active rows (Amor first variation
  `[K max(tr,0) I2 + 2 mu dev_2x2]:dstrain`, `K=lambda+2mu/3`, embedded 3x3),
  and identity in `H`. The packed Jacobian is nonsymmetric; exact adjoint
  actions are exposed only because they are implemented as exact transposes of
  the existing assemblies and tested against the dense transpose.
- Alternatives rejected: finite-differencing Jacobian blocks in any production
  API; adding a mechanics-energy derivative to `Rd` (changes the staggered
  damage equation); a damage obstacle/clipping/projection in the residual;
  smoothing or tolerance on the history active set; changing any authoritative
  primitive equation (BA kinematics, center weights, AT2 operator, Amor split);
  and exposing a transpose before it can be implemented and tested exactly.
- Review status: Phase 1 implementation and manufactured evidence only; the
  user-supplied mathematical contract is recorded here and any Phase 2 solver
  remains unauthorized.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md`,
  `examples/fracture2d/nonlocal_engineering_grip_monolithic.py`,
  `tests/test_fracture2d_nonlocal_engineering_grip_monolithic.py`.

## 2026-09-20 — Set the Phase-1 monolithic review-resolution evidence standard
- Decision: Resolve the independent Phase-1 review findings without adding
  any solver: (a) freeze every cached static datum in
  `MonolithicGripResidual` — the material AT2 CSR `data`/`indices`/`indptr`
  internals and the derived material-source mask are read-only, so in-place
  mutation attempts fail loudly and leave residuals bit-identical; (b) add a
  genuinely independent test-only packed-JVP oracle that rebuilds the BA
  edge gradient/tangent endpoint assembly from frozen supports, shape
  inverses, and weights, the explicit graph-volume damage action from a
  from-definition cubic kernel, and the Amor source accumulation from edge
  strains/delta gradients, with exact zero `Jdq`/`JHd` and identity `JHH`,
  and compare the production JVP against it over multiple states and
  directions while tying the production adjoint to it only through the
  bilinear identity; (c) cover active-set switching along a line that
  crosses the committed level (including an exact bit-for-bit median tie)
  with a 1-Lipschitz no-jump check, and demonstrate out-of-range `d`/`H`
  inputs are evaluated by the raw finite equations with no obstacle,
  clipping, or projection — only dtype float64, finiteness, and exact shapes
  are validated, by contract; (d) demote the dense Jacobian helper to
  private `_dense_jacobian`, documented as production self-data for
  structural nonsymmetry evidence only, never independent correctness
  evidence.
- Why: Cached mutable CSR internals and masks could be corrupted silently;
  the previous dense-transpose adjoint check reused production `jvp` columns
  and therefore could not detect a shared transpose/index error; and the
  switching/obstacle contract of the generalized derivative needed explicit
  manufactured proof.
- Alternatives rejected: eliminating the caches instead of freezing them
  (rebuilds per call add cost without improving the audit); building the
  oracle from production assembly methods (not independent); keeping a
  public dense Jacobian; and adding any range validation on `d`/`H`, which
  would violate the approved no-obstacle contract.
- Review status: Phase 1 remains implementation/manufactured evidence only;
  Phase 2 solver work stays unauthorized.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md`,
  `examples/fracture2d/nonlocal_engineering_grip_monolithic.py`,
  `tests/test_fracture2d_nonlocal_engineering_grip_monolithic.py`.

## 2026-09-20 — Make the Phase-1 cached CSR irrevocably immutable
- Decision: Build `MonolithicGripResidual`'s cached material AT2 CSR from
  independent bytes-backed copies of `data`, `indices`, and `indptr`, and reject
  construction if SciPy exposes any of those buffers as writable or allows
  `setflags(write=True)`. Keep the authoritative model operator untouched and
  preserve SciPy matvec/transpose use for the residual and generalized actions.
- Why: A reversible NumPy read-only flag does not protect an owning CSR buffer
  from later write-flag re-enablement. An immutable bytes owner makes both
  direct writes and re-enable attempts fail while retaining the sparse operation
  path needed by Phase 1.
- Alternatives rejected: mutating the authoritative operator's exported
  storage, relying on `setflags(write=False)` alone, rebuilding the damage graph
  for every residual/JVP/VJP call, or replacing the sparse path with a dense
  matrix.
- Review status: The module and VJP docstrings now identify the independent
  oracle bilinear identity as adjoint evidence rather than dense-transpose
  evidence. Focused tests cover bytes ownership, direct/re-enable failures,
  SciPy matvec/transpose behavior, and bit-identical residual/JVP/VJP outputs.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md`,
  `examples/fracture2d/nonlocal_engineering_grip_monolithic.py`,
  `tests/test_fracture2d_nonlocal_engineering_grip_monolithic.py`.

## 2026-09-20 — Authorize Phase 2A scaled semismooth Newton core
- Decision: Implement only a private/experimental, explicit-parameter
  scaled semismooth Newton/GMRES core around `MonolithicGripResidual`.
  The new frozen solver configuration requires
  `max_newton_iterations`, `gmres_rtol`, `gmres_atol`, `gmres_restart`,
  `gmres_maxiter`, `armijo_c`, `line_search_reduction`,
  `line_search_max_backtracks`, and `line_search_alpha_min`; no numerical
  defaults, workflow, CLI, continuation/cutback, physical run, or auxiliary
  strict sweep is added.
- Equations retained: the Phase-1 packed residual and generalized JVP remain
  exactly the approved code-compatible `q-d-H` equations, including the strict
  `psi_plus > H_committed` active set with ties committed, volume-weighted
  damage row, and raw unconstrained damage variable. At each iterate, the
  solver freezes the exact active mask and gate-derived positive row scales
  `Lq`, `Lrho`, and `LH` and applies the same row transformation to the JVP:
  `Rhat=(Rq/Lq, (M^{-1}Rd)/Lrho, RH/LH)` and
  `Jhat dz=(Jq dz/Lq, (M^{-1}Jd dz)/Lrho, JH dz/LH)`.
- Numerical choices owned by this project: the merit is
  `phi=0.5*||Rhat||_2^2`; GMRES is a nonsymmetric SciPy `LinearOperator` solve
  with an explicit installed-API `callback_type`, an explicit unpreconditioned
  residual check, and no fallback; the left preconditioner is block diagonal
  only, using the existing particle mechanics block-Jacobi inverse, a sparse
  strong damage `A_H` solve, and scaled identity history; and globalization is
  raw `z+alpha*delta` Armijo backtracking. A trial is eligible only when finite
  and `H>=0`; there is no clipping, projection, obstacle, or line-search
  physics. Accepted trials rebuild the active set.
- Acceptance semantics: scaled mechanics L2 and scaled damage/history Linf
  residual blocks must each be at most one, with finite values, a stable active
  set, and the existing displacement/damage/history correction-gate semantics.
  Initial exact roots are accepted without a GMRES call. GMRES status,
  unpreconditioned residual, callback history, Armijo budget, and minimum-alpha
  failures are hard failures.
- Source-backed versus project-owned: the residual/JVP, AT2/Amor equations,
  particle mechanics preconditioner, strong damage matrix, and gate-limit
  definitions are reused from the reviewed Phase-1/direct engineering-grip
  implementation. The scaled row transformation, semismooth Newton iteration,
  block preconditioner composition, Euclidean merit, and Armijo globalization
  are project-owned numerical choices explicitly authorized here; they are not
  presented as equations supplied by Kristensen/Martinez-Paneda or a
  peridynamic reference.
- Alternatives rejected: changing the physical residual or history law;
  finite-differencing the JVP; symmetric CG for the nonsymmetric packed system;
  unscaled GMRES; preconditioner fallbacks; clipping/projecting `d`; smoothing
  the active set; or adding continuation/cutback rules before the core is
  manufactured-tested.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md`,
  `examples/fracture2d/nonlocal_engineering_grip_monolithic.py`, and its
  focused manufactured tests.

## 2026-09-21 — Resolve independent Phase-2A review findings
- Decision: At every raw Newton iterate, define `H_used` as the current packed
  history and `H_next=max(H_committed,psi_plus(Q))`. Reproduce the existing
  direct-gate damage-residual scale with all six constituent terms from both
  histories, and reproduce the existing history scale with committed, used,
  and next histories. Store the paired histories and a state digest in the
  immutable scale bundle.
- Decision: The no-GMRES shortcut is permitted only when every packed residual
  entry is exactly numerical zero and the packed history is finite and valid.
  Tolerance-level nonroots must enter Newton so correction and active-set
  stabilization are audited.
- Decision: Diagnostics now name `linearization_scales` and
  `accepted_trial_scales`; convergence norms are evaluated with the latter.
  Scale/state digest and `H_used` pairing checks reject forged diagnostic or
  result chains. Every numerical failure retains an immutable
  `MonolithicNewtonFailureAudit` containing the stage, reason, current state,
  active set, available linear/GMRES/line-search evidence, and completed rows.
- Why: The prior implementation used only the current history in the damage
  scale, accepted small scaled residuals without auditing a Newton correction,
  conflated linearization and accepted-trial scales, and discarded current-step
  failure evidence. These changes make gate provenance and failure diagnosis
  reproducible without altering the approved residual/JVP or adding a physical
  workflow.
- Source-backed versus project-owned: the six-term gate scale and history
  semantics are reused from the reviewed direct engineering-grip diagnostics;
  state-digest pairing, immutable failure auditing, and exact-zero shortcut
  policy are project-owned implementation contracts.
- Alternatives rejected: retaining the tolerance shortcut, storing one
  ambiguous scale field, returning only prior diagnostics on failure, or
  silently converting nonfinite/raw-trial failures into a fallback step.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md`,
  `examples/fracture2d/nonlocal_engineering_grip_monolithic.py`, and
  `tests/test_fracture2d_nonlocal_engineering_grip_monolithic_newton.py`.

## 2026-09-21 — Independent Phase-2A re-review keeps integration unwired
- Decision: Keep the private Phase-2A core out of auxiliary strict-sweep or
  continuation integration until its failure-audit and fixed-shape scale
  contracts are tightened. The manufactured core remains usable for further
  review-only testing.
- Why: The focused and affected manufactured suites pass, but the initial
  `active_mask` call at
  `examples/fracture2d/nonlocal_engineering_grip_monolithic.py:1739` occurs
  before the solver's audit-protected initial evaluation block, so a numerical
  active-set/Amor failure escapes as a raw exception. Also,
  `MonolithicNewtonScales` only requires nonempty equal-length history vectors
  (`:1038-1045`), `_validate_scale_pair` checks only the packed tail
  (`:1085-1088`), and the preconditioner does not enforce the model's exact
  material count (`:1508-1514`); a same-digest one-entry history bundle was
  accepted by a direct probe.
- Alternatives rejected: Treating passing internally generated solver paths as
  sufficient for the private API forgery contract, or authorizing a physical
  continuation without structured provenance for every numerical failure.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md`,
  `examples/fracture2d/nonlocal_engineering_grip_monolithic.py`, and
  `tests/test_fracture2d_nonlocal_engineering_grip_monolithic_newton.py`.

## 2026-09-21 — Close Phase-2A initial-audit and scale-shape findings
- Decision: Run the initial active-set and residual evaluations under the
  existing immutable failure-audit contract, retaining a valid audit mask and
  the exact current packed state when an active/Amor, residual, or initial
  scale evaluation fails. Require every model-bound scale consumer to validate
  both history vectors against the exact `monolithic.n_material`; retain the
  state digest and `H_used` tail check where the scale bundle is paired to its
  linearization state.
- Why: A raw initial active-set numerical exception bypassed
  `MonolithicNewtonFailureAudit`, and a generic one-entry scale bundle could
  pass same-digest tail validation before reaching the preconditioner. The
  fixed-shape checks close the forged-bundle path without changing the
  approved residual, active-set, or solver equations.
- Alternatives rejected: using a zero active set to continue after evaluation
  failure, enforcing model size in the generic dataclass constructor (which
  would remove legitimate generic construction), accepting a digest as a
  substitute for exact model shape, or wiring the core into a workflow.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md`,
  `examples/fracture2d/nonlocal_engineering_grip_monolithic.py`,
  `tests/test_fracture2d_nonlocal_engineering_grip_monolithic_newton.py`.

## 2026-09-21 — Scope the Phase-2B verified monolithic attempt
- Decision: Add a private single-target verified-attempt layer around the
  manufactured Phase-2A solve. It must receive the existing engineering-grip
  model/configuration/committed state, an explicit target, explicit mechanics
  adapter, explicit packed `initial_z`, and an explicit
  `MonolithicNewtonConfig`; it must not create a warm start, continuation,
  cutback, workflow, CLI, archive, or physical-run path.
- Decision: Recheck every converged Newton candidate with one authoritative
  fixed-history engineering sweep: raw Newton `q/d/H` seed mechanics-before,
  committed-history Amor `psi_before`, `H_used=max(H_committed,psi_before)`,
  exact fixed-history damage, before-seeded mechanics-after, committed-history
  Amor `psi_after`, and `H_next=max(H_committed,psi_after)`. The canonical
  candidate is exactly `(q_after,d_solved,H_next)` and is accepted only after
  a fresh monolithic residual/scale/active-set check and all existing strict
  diagnostics gates pass.
- Why: The Phase-2A root is an experimental packed nonlinear solve, whereas
  the existing engineering controller's mechanics, Amor, AT2, reaction,
  conservation, range, healing, and history semantics are the authoritative
  direct path. The auxiliary sweep makes that distinction auditable without
  silently promoting raw Newton fields to a committed engineering state.
- Source-backed versus project-owned: all residuals, Amor/AT2 equations,
  fixed-history solve, mechanics adapter contract, and diagnostic gate
  definitions are reused from the reviewed engineering-grip implementation;
  the single-target orchestration, canonical-state pairing, immutable failure
  audit, and no-serialization contract are project-owned.
- Alternatives rejected: wiring Newton into continuation, using hidden direct
  warm starts, accepting the raw Newton state without the authoritative sweep,
  clipping/projecting damage, warning acceptance, or serializing rejected
  trial arrays as durable physical evidence.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md`,
  `examples/fracture2d/nonlocal_engineering_grip_monolithic.py`, and the new
  Phase-2B manufactured tests.

## 2026-09-21 — Strengthen Phase-2B audit payload pairing
- Decision: Require every populated auxiliary audit field to equal its
  corresponding authoritative diagnostic field, require nonnegative raw
  history, and require accepted attempts to carry and recompute canonical
  active/scaled-residual fields from the canonical packed state.
- Why: The Phase-2B result is an audit contract, not only a numerical return
  value. A forged or partially paired success payload could otherwise preserve
  the right digest while reporting a different field bundle or omit the
  canonical residual evidence.
- Alternatives rejected: trusting the constructor caller, checking only the
  packed-state digest, or adding a warning-only consistency path.
- Related: `examples/fracture2d/nonlocal_engineering_grip_monolithic.py`,
  `tests/test_fracture2d_nonlocal_engineering_grip_monolithic_verified.py`,
  and `docs/tasks/fracture2d-engineering-grip-baseline.md`.

## 2026-09-21 — Close Phase-2B audit-order and provenance findings
- Decision: Make the Phase-2B auxiliary result a fully paired audit object,
  carrying the exact model/configuration/committed-state/static identities and
  validating every raw, optional, mechanics-provenance, diagnostic, canonical,
  and Newton field against those identities. Recompute the strict aggregate
  from the individual diagnostic gate fields; never trust a stored aggregate
  boolean.
- Decision: Preserve the authoritative single-target operation order as
  mechanics-before, Amor/`H_used`, fixed-history damage/`rho_used`,
  mechanics-after seeded by the before solve, Amor/`H_next`/`rho_next`, then
  complete diagnostics including force/moment gates. The mechanics-after seed
  provenance retains the raw Newton damage digest because its seed displacement
  came from the raw-damage mechanics-before solve.
- Why: A forged aggregate, one-sided optional field, foreign auxiliary object,
  or mislabelled seed can retain numerically plausible arrays while severing the
  authoritative replay chain. Early gate short-circuits also discard later
  fields needed to distinguish complete strict rejection from incomplete
  numerical failure.
- Alternatives rejected: trusting `diagnostics.passed`, accepting equal-value
  foreign objects without identity checks, using the solved damage digest for
  the mechanics-after seed, or testing call order only through production
  self-consistency.
- Related: `examples/fracture2d/nonlocal_engineering_grip_monolithic.py`,
  `tests/test_fracture2d_nonlocal_engineering_grip_monolithic_verified.py`,
  and `docs/tasks/fracture2d-engineering-grip-baseline.md`.

## 2026-09-21 — Bind Phase-2B numerical failure stages and audit identity
- Decision: Store the exact `MonolithicNewtonFailureAudit` object separately on
  `MonolithicVerifiedAttempt` and require a rejected failure payload to reference
  that same object. Split auxiliary numerical exceptions after fixed-history
  damage into `rho_used` and `rho_next` stages so retained fields and failure
  classification identify the precise completed operation.
- Why: A failure payload that only carried an independently reconstructed audit
  could preserve equal values while severing object-level provenance. Combining
  fixed-history residual reconstruction with the damage stage also obscured
  whether `rho_used` or `rho_next` had been completed and paired.
- Source-backed versus project-owned: the fixed-history residual operations and
  existing diagnostic semantics are reused from the reviewed engineering-grip
  implementation. Exact in-memory audit identity and the finer failure-stage
  labels are project-owned Phase-2B audit contracts; they do not change any
  residual, constitutive law, or physical gate.
- Alternatives rejected: trusting equal-value foreign audit objects, collapsing
  all fixed-history residual failures into `damage`, adding a serializer for
  rejected arrays, or wiring the layer into continuation.
- Verification: focused Phase-2B tests pass `10/10`; the combined affected CPU
  unittest command passes `261` tests with two expected GPU-only skips;
  `compileall` and `git diff --check` pass. No workflow or physical run was
  performed.
- Related: `examples/fracture2d/nonlocal_engineering_grip_monolithic.py`,
  `tests/test_fracture2d_nonlocal_engineering_grip_monolithic_verified.py`,
  `docs/tasks/fracture2d-engineering-grip-baseline.md`, and `docs/PROGRESS.md`.

## 2026-09-21 — Authorize manufactured-only monolithic Phase-2C continuation
- Decision: Implement a private Phase-2C adaptive strict continuation around the
  approved Phase-2B single-target attempt. The continuation configuration must
  expose explicit `load_start`, `load_final`, `du_initial`, `du_min`,
  `cutback_factor`, `max_cutbacks`, and a required `MonolithicNewtonConfig`,
  while reusing the existing engineering-grip gate/work-check configuration.
  There is no CLI/workflow entry, archive writer, rejected-field serializer, or
  physical run in this phase.
- Why: The controller contract needs a deterministic, auditable load state
  machine without adding a second nonlinear map or hidden stagger warm start.
  Every attempt can therefore be represented by one immutable Phase-2B audit,
  and every accepted state is exactly its Phase-2B canonical state.
- Alternatives rejected: calling Phase2B more than once per target, seeding
  from a rejected trial, retaining trial fields as committed state, warning or
  fallback acceptance, clipping/projection, callback side effects, and durable
  serialization before the in-memory ordering/digest contract is tested.
- Relation: User-authorized Phase2C extension of the Phase-2B contract recorded
  immediately above; this does not select a physical formulation or endpoint.
- Related: `examples/fracture2d/nonlocal_engineering_grip_monolithic.py`,
  `tests/test_fracture2d_nonlocal_engineering_grip_monolithic_continuation.py`,
  and `docs/tasks/fracture2d-engineering-grip-baseline.md`.

## 2026-09-21 — Bind Phase-2C configuration and failure audit to Phase2B
- Decision: Make `MonolithicContinuationConfig` carry the explicit load
  schedule fields, the existing validated `EngineeringGripConfig`, and the
  required `MonolithicNewtonConfig`. The duplicated schedule fields must equal
  the corresponding engineering-grip fields exactly; all gate and work-check
  values therefore remain owned by the existing direct configuration.
- Decision: Represent each continuation attempt with one immutable
  `MonolithicContinuationAttempt` containing the authoritative predictor,
  load/increment/cutback metadata, model/static/config identities, and exactly
  one `MonolithicVerifiedAttempt`. A completed or failed run is represented by
  an immutable `MonolithicContinuationResult`; numerical endpoint failure is
  raised as `MonolithicContinuationFailure` carrying that partial result.
- Why: This makes rejected-target rollback and solver provenance inspectable
  without copying or serializing trial fields, while preserving the existing
  `_next_target`, `_is_minimum_increment`, and one-time work-check contracts.
- Alternatives rejected: deriving the schedule from hidden base-config fields,
  returning only the last accepted state, catching Phase2B structural
  exceptions as numerical failures, or adding a workflow/archive serializer.
- Related: `examples/fracture2d/nonlocal_engineering_grip_monolithic.py`,
  `docs/tasks/fracture2d-engineering-grip-baseline.md`, and the new Phase-2C
  manufactured tests.

## 2026-09-21 — Permit finite raw Newton trials and close Phase-2C audit gaps
- Decision: Keep the exact monolithic `R_H=H-max(H_committed,psi_plus)`
  residual, but remove the `H>=0` obstacle from intermediate raw Newton
  line-search trials. Any finite packed `d/H` trial may be evaluated without
  clipping or projection; successful Newton results and the canonical Phase-2B
  state must still pass explicit nonnegative-history checks.
- Decision: Make Phase-2C result validation recompute the exact work-check
  binding and ordered target/increment/cutback schedule from the existing
  `_next_target` and `_is_minimum_increment` helpers, with strict attempt/state
  object and digest pairing.
- Why: A semismooth raw line search must be allowed to cross an unconstrained
  intermediate history value, while irreversibility is enforced by the exact
  residual and final canonical state contract. Recomputing continuation metadata
  prevents a superficially valid immutable result from hiding a forged schedule,
  work-check configuration, or rollback pairing.
- Source-backed versus project-owned: the residual, exact history maximum,
  direct predictor, work-check calculation, endpoint snap, and minimum-increment
  helper are existing reviewed project equations/contracts. The finite raw-trial
  boundary, final/canonical history gates, and exhaustive Phase-2C audit checks
  are project-owned numerical/audit contracts.
- Alternatives rejected: retaining an intermediate history obstacle, clipping or
  projecting raw trials, warning-accepting a negative final history, replacing a
  direct predictor with a random seed, or relying on mocked/fabricated success
  results for the controller smoke path.
- Verification: CPU monolithic tests pass `54/54`; the combined
  engineering/JAX plus monolithic focused command passes `114` tests with one
  expected GPU-only skip; targeted compilation and `git diff --check` pass. No
  workflow, archive, or physical run was performed.
- Related: `examples/fracture2d/nonlocal_engineering_grip_monolithic.py`,
  `tests/test_fracture2d_nonlocal_engineering_grip_monolithic_newton.py`,
  `tests/test_fracture2d_nonlocal_engineering_grip_monolithic_verified.py`,
  `tests/test_fracture2d_nonlocal_engineering_grip_monolithic_continuation.py`,
  and `docs/tasks/fracture2d-engineering-grip-baseline.md`.

## 2026-09-21 — Close work-check and Newton residual-audit defects
- Decision: Require the three engineering work-check solves to share the
  validated constraints/static runtime identity and static, material-damage,
  and unit-system digests, while deliberately not requiring equality of their
  fixed-damage cache/runtime tokens. A replay adapter may construct a fresh
  fixed-damage view for each centered load.
- Decision: Keep `EngineeringGripWorkCheck` construction strict. The failure
  regression now obtains a genuinely valid `passed=False` check by running the
  real work-check solver with a controlled zero relative-error limit and then
  verifies continuation short-circuits before Phase-2B.
- Decision: Retain three immutable scaled residual vectors per Newton
  diagnostic (reference, accepted line-search at reference scales, and
  accepted trial at accepted-trial scales). The normal validator checks their
  algebraic merit/block/final-row pairing; explicit `validate_with_owner(owner)`
  independently replays all residuals through an in-memory authoritative
  `MonolithicGripResidual`.
- Why: Cross-solve cache identity is not a material/model invariant, an
  inconsistent forged work check must not be constructible, and a durable audit
  must remain self-paired even when a failure-orchestration test temporarily
  replaces an operator method. The real manufactured Newton path still receives
  owner-backed replay; only the Armijo orchestration test uses a deterministic
  method replacement.
- Alternatives rejected: requiring damage-cache Python identity, weakening the
  work-check constructor, silently omitting intermediate residual validation,
  or making all result validation depend on mutable monkeypatched methods.
- Verification: affected CPU unittest command passes `233` tests with one
  expected GPU-only skip; `python -m compileall -q examples/fracture2d tests`
  and `git diff --check` pass. No workflow, archive, or physical execution was
  performed.
- Related: `examples/fracture2d/nonlocal_engineering_grip.py`,
  `examples/fracture2d/nonlocal_engineering_grip_monolithic.py`,
  `tests/test_fracture2d_nonlocal_engineering_grip.py`,
  `tests/test_fracture2d_nonlocal_engineering_grip_monolithic_newton.py`,
  `tests/test_fracture2d_nonlocal_engineering_grip_monolithic_continuation.py`,
  and `docs/tasks/fracture2d-engineering-grip-baseline.md`.

## 2026-09-21 — Require owner-backed authenticity before archive integration
- Decision: Add explicit owner replay to the engineering work check, each
  Phase-2B verified attempt, and the Phase-2C continuation result. Replay uses
  the authoritative model/config and exact stored numerical settings, then
  compares all durable scalar, array, digest, reaction/energy, gate, state, and
  failure fields while ignoring only ephemeral runtime object identities.
- Decision: A non-complete continuation interval may not produce a failed result
  without at least one rejected attempt. The failed attempt must carry the exact
  reconstructed target and controller failure stage/reason; a work-check
  exception still propagates before a result exists, and a legitimately already
  complete interval remains a complete result.
- Why: Structural/value pairing can be forged coherently, including a mechanics
  residual, reaction, and energy. Fresh authoritative replay is required to
  bind durable evidence to the frozen model and equations before a future
  archive reader accepts it. Runtime cache and Python identity tokens are not
  cross-solve evidence.
- Source-backed versus project-owned: the zero-damage work identity, mechanics
  solve, monolithic residual, Phase-2B auxiliary ordering, and canonical state
  are existing reviewed equations/contracts. Deterministic model/config
  digests, exhaustive fresh-solve comparators, recursive continuation binding,
  and controller failure provenance are project-owned authenticity contracts.
- Alternatives rejected: trusting structural self-consistency alone, comparing
  fresh fixed-damage cache identities, accepting callback/mock output as replay,
  rerunning a full physical controller when per-attempt replay is sufficient, or
  starting archive work before owner validation is available.
- Related: `examples/fracture2d/nonlocal_engineering_grip.py`,
  `examples/fracture2d/nonlocal_engineering_grip_monolithic.py`,
  `docs/tasks/fracture2d-engineering-grip-baseline.md`.

## 2026-09-21 — Dispatch controller-only no-progress records without Phase2B replay
- Decision: In continuation owner validation, classify the no-PDE exception
  only as a terminal `MonolithicContinuationAttempt` whose Phase2B-shaped
  failure has `stage="controller"`, exact `load_target == load_start`, and
  exact `increment == 0`. Revalidate its schedule, canonical reason,
  predictor, model/config digests, committed digests, and empty PDE-audit
  payload, then skip numerical Phase2B replay.
- Decision: Every other attempt, including all positive-increment attempts,
  must use recursive `validate_with_owner` and authoritative Phase2B replay.
  Stage relabeling or target/increment/reason mutation cannot opt a real PDE
  attempt into the controller-only branch.
- Why: A floating-point no-progress dispatch is not a PDE attempt and cannot
  be replayed at a target equal to its committed load, while actual numerical
  attempts still require independent owner authentication.
- Alternatives rejected: replaying a controller record through the positive
  target Phase2B API, classifying by failure stage alone, or exempting all
  zero-increment records without terminal schedule/reason/digest checks.
- Verification: focused CPU engineering/Phase-2B/Phase-2C tests pass `77`, the
  expanded affected command passes `112`, and compileall plus `git diff --check`
  pass; no archive or physical execution was performed.
- Related: `examples/fracture2d/nonlocal_engineering_grip_monolithic.py`,
  `tests/test_fracture2d_nonlocal_engineering_grip_monolithic_continuation.py`,
  and `docs/tasks/fracture2d-engineering-grip-baseline.md`.

## 2026-09-21 — Bind owner replay to durable values and terminal attempts
- Decision: Keep structural constructors identity-strict for live in-memory
  pairing, but make explicit owner replay compare durable model/static/config
  digests and recursively compare arrays, scalars, diagnostics, reactions,
  energies, gates, and failure payloads. Fresh equivalent models and fresh
  fixed-damage views may therefore authenticate without sharing cache tokens.
- Decision: Preserve numerical exceptions from an authoritative work-check
  replay, while allowing a carried `EngineeringGripWorkCheckFailure.check` to
  be compared as a durable failed numerical result. A floating-point/solver
  exception without a result is not relabelled as a structural forgery.
- Decision: When continuation target rounding cannot advance, construct and
  retain one terminal controller-stage Phase-2B-shaped failed attempt at the
  exact current target with zero increment, rather than producing a result with
  no attempt record.
- Why: A coherent structural forgery can alter a residual, reaction, or energy
  while preserving local algebraic checks. Fresh CPU replay is the independent
  authenticity source; the continuation audit must also preserve exact
  controller provenance for future archive readers.
- Alternatives rejected: requiring fresh Python/cache identity, allowing the
  public replay entry point to be monkeypatched as an authenticity source,
  wrapping numerical failures as generic `ValueError`, or retaining arbitrary
  no-attempt failure metadata.
- Related: `examples/fracture2d/nonlocal_engineering_grip.py`,
  `examples/fracture2d/nonlocal_engineering_grip_monolithic.py`,
  `examples/fracture2d/README.md`, and the focused owner-replay tests.

## 2026-09-21 — Hold archive integration after independent owner-replay audit

- Decision: Keep durable archive schema/reader work **NO-GO** until the
  controller-stage no-progress continuation audit has a dedicated owner
  validation path. Do not weaken the requirement that archive readers
  reconstruct the owner and call `validate_with_owner`.
- Why: A real direct-predictor success authenticates against a fresh equivalent
  model, while a coherent residual/reaction/energy forgery is structurally
  accepted and owner-replay rejected, recursive work-check forgery is rejected,
  arbitrary no-attempt failure is rejected, and public replay-entrypoint
  monkeypatches are bypassed. However, a retained terminal controller attempt
  with `target <= committed.u_bar` is not a Phase-2B numerical attempt;
  `MonolithicContinuationResult.validate_with_owner` currently routes it to the
  unconditional Phase-2B replay and rejects the legitimate result.
- Source-backed versus project-owned: the CPU mechanics/Newton/Phase-2B replay
  equations and existing continuation schedule remain the reviewed contracts.
  The controller-stage branch, direct-predictor binding, and failure-provenance
  comparison are project-owned authenticity rules and require independent
  review before implementation.
- Alternatives rejected: silently excluding failed archives, treating a
  controller no-progress record as a valid Phase-2B solve, or approving archive
  integration from success-only tests.
- Verification: focused owner tests `123` with one expected skip; broader
  affected CPU suite `241` with one expected skip; compileall and
  `git diff --check` pass. No source-code edit, archive, workflow, or physical
  execution was performed during this audit.
- Related: `examples/fracture2d/nonlocal_engineering_grip_monolithic.py`,
  `examples/fracture2d/README.md`,
  `docs/tasks/fracture2d-engineering-grip-baseline.md`, and the continuation
  owner-replay tests.

## 2026-09-21 — Approve archive schema/reader design after controller-only replay fix

- Decision: **GO** for archive schema/reader design. The reader must reconstruct
  the authoritative owner and call the top-level
  `MonolithicContinuationResult.validate_with_owner(...)`; it must not replace
  that call with direct per-attempt replay for controller-only records.
- Why: The terminal zero-increment no-progress record now receives exact
  controller metadata/predictor validation without a PDE replay, while every
  positive-increment attempt still reaches the captured authoritative CPU
  Phase-2B entry point. Forged stage/target/increment/reason payloads are
  rejected at the owner boundary.
- Source-backed versus project-owned: existing A2/monolithic equations and
  Phase-2B CPU replay remain unchanged. The controller-only branch and its
  metadata/predictor provenance checks are project-owned archive-authenticity
  rules.
- Verification: targeted owner probes `3/3` passed; focused monolithic suite
  `62/62` passed; compileall and `git diff --check` passed. No source edit,
  archive, workflow, or physical execution occurred during confirmation.
- Related: `examples/fracture2d/nonlocal_engineering_grip_monolithic.py`,
  `tests/test_fracture2d_nonlocal_engineering_grip_monolithic_continuation.py`,
  and `docs/tasks/fracture2d-engineering-grip-baseline.md`.

## 2026-09-21 — Define a separate lean monolithic archive contract

- Decision: Add a private monolithic-continuation archive module with its own
  schema identifier.  Store canonical JSON metadata as UTF-8 `uint8`, accepted
  state arrays, work-check/result summaries, and ordered attempt scalar
  provenance; do not serialize rejected or intermediate `q/d/H` arrays.
- Decision: The writer must validate the live result with its owner before
  writing, reject object/non-finite archive fields, enforce an explicit
  uncompressed-byte budget before atomic replacement, and default to refusing
  overwrite.  The reader must enforce a fixed ZIP member allowlist and budget,
  reconstruct the two continuation configs, rerun the strict controller with
  the supplied owner, call top-level `validate_with_owner`, and compare every
  accepted state and ordered scalar/digest audit record.
- Why: The existing stabilized v5 archive is a different public contract and
  contains physical/workflow fields that are not appropriate for the private
  manufactured-only monolithic layer.  Owner replay is the authenticity source;
  archive bytes are evidence and must not become a second numerical solver.
- Source-backed versus project-owned: the monolithic residual, engineering
  configs, strict continuation schedule, owner replay, accepted state fields,
  work-check values, and gate/solver diagnostics are existing reviewed
  contracts.  The schema/member layout, canonical metadata encoding, byte
  budget, atomic persistence, and archive mutation rules are project-owned
  durability contracts.
- Alternatives rejected: reusing stabilized v5, serializing a pickle/object
  graph, storing rejected Newton/auxiliary vectors, trusting archive metadata
  without fresh replay, or adding a workflow/physical execution entry point.
- Related: `examples/fracture2d/nonlocal_engineering_grip_monolithic_archive.py`,
  `examples/fracture2d/nonlocal_engineering_grip_monolithic.py`, and the
  focused monolithic archive tests.

## 2026-09-21 — Bind archive replay below replaceable public callbacks

- Decision: Keep the public monolithic continuation entry point injectable for
  manufactured controller tests, but route the captured archive replay entry
  through a separate private implementation bound to the captured authoritative
  work-check and Phase-2B runners.
- Why: Capturing only the public controller function does not prevent its body
  from resolving caller-replaceable public runner attributes at replay time.
  Owner-authenticated archive replay must not be changed by monkeypatches or
  workflow callbacks while existing manufactured injection tests retain their
  explicit public seam.
- Alternatives rejected: capturing only the public controller, replacing the
  existing public test seam, or treating a callback/monkeypatch as authenticity
  evidence.
- Source-backed versus project-owned: the work-check, Phase-2B, and
  continuation numerical equations are unchanged; the dispatch boundary is a
  project-owned owner-authentication rule.
- Verification: focused continuation/archive tests `20/20` passed, including
  rejection of public controller, Phase-2B, and work-check callback use during
  archive replay.
- Related: `examples/fracture2d/nonlocal_engineering_grip_monolithic.py`,
  `examples/fracture2d/nonlocal_engineering_grip_monolithic_archive.py`, and
  `tests/test_fracture2d_nonlocal_engineering_grip_monolithic_archive.py`.

## 2026-09-21 — Harden the private monolithic archive boundary

- Decision: Keep the `fracture2d-monolithic-continuation-v1` member contract
  exact, and make ZIP inspection reject nonzero flags, data descriptors,
  extras/comments, special-file metadata, local-header mismatches, excessive
  compressed size, and decompression ratios. Parse each NPY member from the
  same opened archive file after requiring an exact C-order payload with no
  trailing bytes.
- Decision: Install archives with an atomic hard-link no-clobber path when
  `overwrite=False`; only an explicit `overwrite=True` may use replacement.
  Capture owner replay runners below the replaceable public/private callback
  attributes, while retaining the public manufactured injection seams.
- Decision: Treat unknown scalar-evidence objects and non-strict schema-version
  values as errors rather than recording an opaque `runtime_only` fallback.
- Why: The archive is evidence, not a solver graph. A path replacement between
  inspection and loading, a coherent ZIP/NPY forgery, a mutable callback alias,
  or an opaque omitted audit field could otherwise bypass the declared
  owner-authentication and accepted-state-only contract.
- Alternatives rejected: weakening the existing uncompressed budget, trusting
  `numpy.load` to enforce the raw NPY contract, using a check-then-replace
  no-clobber sequence, or changing the numerical equations/continuation gates.
- Source-backed versus project-owned: no PDE, constitutive, history, load, or
  gate equation changes; ZIP/NPY durability limits, atomic installation, and
  runner binding are project-owned archive-authenticity rules.
- Related: `examples/fracture2d/nonlocal_engineering_grip_monolithic_archive.py`,
  `examples/fracture2d/nonlocal_engineering_grip_monolithic.py`,
  `examples/fracture2d/nonlocal_engineering_grip.py`, and the focused archive
  tests.

## 2026-09-21 — Keep monolithic archive v1 ZIP32-only and name mechanics energy precisely

- Decision: Define `fracture2d-monolithic-continuation-v1` as ZIP32-only with a
  hard `500000000`-byte practical budget, reject every ZIP64 sentinel or extra,
  and require the canonical writer to use `allowZip64=False`. Rename the
  accepted energy member to `accepted_mechanics_energy` and the ordered attempt
  metadata to `attempt_scalar_evidence`; v1 has not been published, so no
  compatibility alias is retained.
- Decision: Keep crack-region strain, Amor driver arrays, and static operators
  out of the lean archive. The digest-matched owner and recursive
  `validate_with_owner` replay must recompute those fields and all canonical
  equations/gates before accepting the persisted `Q`, `d`, `H`, reaction, and
  A2 mechanics-energy arrays.
- Decision: For default no-clobber writes, fail closed when hard links are not
  supported. If directory fsync fails after link installation, attempt an
  identity-checked destination unlink and a second directory fsync, and report
  cleanup failure explicitly. Preserve controller-only no-progress failures as
  exact scalar provenance and validate them through the top-level replay branch
  without inventing a Phase-2B solve.
- Why: The practical archive is far below ZIP64 scale; explicit rejection is
  safer than partial compatibility. The former `total_energy` name overstated
  the retained A2 mechanics quantity. Accepted fields plus authoritative owner
  replay provide the required physical evidence without duplicating derivable
  arrays or persisting rejected Newton states.
- Alternatives rejected: partial ZIP64 parsing, a misleading coupled-total-
  energy label, serializing static solver graphs or rejected fields, portable
  check-then-replace no-clobber writes, or accepting archives without recursive
  owner replay.
- Verification: focused archive tests pass `15/15`; the full monolithic suite
  passes `77/77`; the final affected engineering/workflow/monolithic CPU suite
  passes `252` tests with one expected GPU-only skip. `compileall` and
  `git diff --check` pass. No workflow integration or physical run occurred.

- Related: `examples/fracture2d/nonlocal_engineering_grip_monolithic_archive.py`,
  `tests/test_fracture2d_nonlocal_engineering_grip_monolithic_archive.py`, and
  `examples/fracture2d/README.md`.

## 2026-09-21 — Require one canonical ZIP32 EOCD

- Decision: Parse the raw EOCD from the same archive descriptor within the
  legal ZIP comment search window and require exactly one `PK\x05\x06` record
  ending at EOF with a zero comment. Require single-disk fields, exact
  allowlist entry counts without `0xffff`, non-sentinel bounded central
  size/offset, and exact equality with the parsed central-record span/location.
- Decision: Reject preceding ZIP64 locator/end signatures and duplicate,
  ambiguous, or trailing EOCD content before accepting `zipfile` metadata.
- Why: `allowZip64=False` alone does not establish the v1 EOCD cardinality,
  single-disk, count, or central-directory-location contract. A valid ZIP32
  archive must be authenticated against the exact same descriptor and raw
  record layout used by the existing central/local parser.
- Alternatives rejected: trusting `ZipFile.start_dir`/`infolist()` without raw
  EOCD checks, accepting the final EOCD while ignoring preceding records,
  allowing comments/trailing bytes, or adding partial ZIP64 compatibility.
- Source-backed versus project-owned: ZIP32 EOCD field meanings and the legal
  comment search window are ZIP format rules; exact one-record, zero-comment,
  allowlist-count, and cross-record archive policy are project-owned v1
  durability rules. No numerical formulation changed.
- Verification: archive tests `16/16`, full monolithic tests `78/78`,
  compileall, and `git diff --check` passed. No workflow or physical execution
  occurred.
- Related: `examples/fracture2d/nonlocal_engineering_grip_monolithic_archive.py`,
  `tests/test_fracture2d_nonlocal_engineering_grip_monolithic_archive.py`,
  `examples/fracture2d/README.md`, and the active engineering-grip task.

## 2026-09-21 — Wire strict monolithic continuation as an isolated CPU workflow

- Decision: Add an explicit `--continuation-strategy monolithic` workflow path
  that constructs the existing engineering-grip owner, builds the required
  `MonolithicContinuationConfig` from explicit CLI Newton/GMRES/Armijo fields,
  runs `run_monolithic_strict_continuation(model, config, None)`, writes the
  v1 archive, and immediately owner-replays it with the same model and required
  ZIP32 budget. Keep legacy and stabilized dispatch unchanged.
- Decision: Reject warning acceptance, stabilized relaxation, Anderson,
  progress-callback, and matched-load options on this path. Do not load FEM or
  construct a GPU adapter. Structural exceptions remain ordinary exceptions;
  terminal numerical results are reported through completed/failure artifacts
  and numerical-failure exit status.
- Decision: Keep monolithic workflow summaries scalar-only while binding source,
  configuration, archive SHA256/budget/member contract, ordered state/attempt
  digests, accepted reaction/`accepted_mechanics_energy`, and failure endpoint.
  Standalone replay requires an explicitly reconstructed owner model and budget;
  no geometry, static-operator, or budget fallback is inferred from the lean
  archive.
- Why: The private controller and archive already have an independently checked
  owner boundary. The workflow seam can therefore reuse those authoritative
  equations and replay rules without introducing a second formulation or
  persisting rejected vectors. CPU-only integration is the approved bounded
  manufactured checkpoint; no physical rounded-slot run is authorized here.
- Alternatives rejected: routing monolithic through the GPU/stabilized adapter,
  loading the FEM comparison archive, silently accepting warning/relaxed
  controls, adding archive-embedded geometry/static arrays, or treating a
  structural exception as a numerical continuation failure.
- Source-backed versus project-owned: the monolithic residual, continuation,
  archive schema, and owner replay are existing project-owned implementations;
  the CLI dispatch, artifact naming, report fields, strict-option rejection,
  and CPU workflow policy are project-owned orchestration choices. No external
  source changes the PDE or constitutive equations.
- Related: `examples/fracture2d/engineering_grip_workflow.py`,
  `examples/fracture2d/README.md`, and
  `tests/test_fracture2d_engineering_grip_monolithic_workflow.py`.

## 2026-09-21 — Close strict monolithic workflow review findings

- Decision: Require `--max-stagger-iterations` to equal the documented
  `examples/fracture2d/configs/solver_base.py` compatibility value `100` on
  the monolithic path. The error explicitly states that this field is not a
  monolithic iteration budget; the explicit Newton/GMRES/Armijo controls remain
  the only monolithic iteration settings. Audit the remaining stabilized-only
  controls with the same reject-or-canonical-value rule.
- Decision: After a monolithic archive is written, a structural owner-replay
  failure removes the final archive/summary/replay/manifest only through
  identity-checked cleanup. The archive must match captured device/inode when
  available and SHA256/size; a replacement is preserved. Cleanup fsyncs the
  directory, and unlink/fsync failure raises a compound durability error
  chained to the original replay cause and carrying the archive path. A
  terminal numerical failure whose replay succeeds retains its failure archive.
- Decision: Define `field_equations_validated` as validation of retained
  accepted-step canonical fields by the owner replay, independent of completion
  status. Initial-state-only results are false; the field does not claim
  validation of omitted rejected vectors or a physical run.
- Why: The workflow must not expose a misleading stagger budget, leave an
  unverifiable final artifact after replay rejection, or overclaim what the
  lean accepted-state archive proves. These are orchestration/provenance rules;
  no PDE, constitutive law, continuation equation, or acceptance gate changed.
- Alternatives rejected: silently accepting arbitrary stagger values, unlinking
  a path after a check without identity verification, retaining stale final
  artifacts after replay rejection, treating completion status as field-equation
  validation, or claiming omitted rejected-trial/physical validation.
- Verification: New workflow tests cover canonical/noncanonical stagger values,
  normal cleanup, concurrent replacement preservation, unlink failure, fsync
  failure, and initial-state-only validation. The requested affected CPU test,
  compileall, and `git diff --check` commands remain to be run.
- Related: `examples/fracture2d/engineering_grip_workflow.py`,
  `examples/fracture2d/README.md`,
  `tests/test_fracture2d_engineering_grip_monolithic_workflow.py`, and the
  active engineering-grip task.

## 2026-09-21 — Verify strict monolithic workflow review fixes

- Verification: the new plus existing workflow tests pass `48/48`; the
  monolithic/newton/verified/continuation/archive command passes `78/78`; the
  legacy engineering-grip tests pass `50/50`; the stabilized tests pass
  `82/82`; `python -m compileall -q examples/fracture2d tests` and
  `git diff --check` pass.
- Scope: all verification is CPU/manufactured or regression coverage. No
  physical rounded-slot execution was performed or authorized.

## 2026-09-21 — Publish monolithic workflow results as one atomic bundle

- Decision: Publish every monolithic run result into one
  `<run-id>.monolithic/` directory containing exactly `archive.npz`,
  `replay.json`, `summary.json`, and `manifest.json`. Generate the archive and
  JSON members in a unique same-parent mode-`0700` staging directory, fsync each
  JSON, fingerprint the complete four-member directory, and fsync staging
  before publication.
- Decision: Use Linux `renameat2(RENAME_NOREPLACE)` through a ctypes wrapper for
  atomic no-clobber directory publication and fail closed when the syscall is
  unavailable. Fsync the parent directory after the rename. If that fsync
  fails, raise `MonolithicBundleDurabilityError` and retain the complete
  published bundle; before publication, structural failures may remove only the
  private staging directory. Reject monolithic `overwrite=True`.
- Decision: Report final bundle paths in all published JSON. Keep legacy and
  stabilized flat-artifact publication unchanged, and do not retain the former
  final-path replay cleanup path/error for monolithic bundles.
- Why: A directory rename makes the complete artifact set visible in one atomic
  namespace operation, while private staging prevents partial/failing replay
  work from touching a final path. No-clobber rename closes the concurrent
  publisher race; retaining a published bundle after parent-fsync failure is
  required because its durability is uncertain but its contents are complete.
- Source-backed versus project-owned: no PDE, constitutive law, history rule,
  continuation equation, or acceptance gate changed. The existing ZIP32 archive,
  owner replay, and scalar provenance contracts remain project contracts. The
  directory layout, staging permissions, rename/fsync ordering, fingerprints,
  path rewriting, and error/cleanup policy are project-owned durability rules.
- Alternatives rejected: check-then-replace or `os.replace` publication,
  portable rename fallbacks, deleting a published bundle after parent-fsync
  failure, retaining final artifacts after structural replay failure, and
  changing legacy/stabilized output contracts.
- Verification: workflow plus bundle seam `52/52`, full monolithic/newton/
  verified/continuation/archive `78/78`, legacy `50/50`, stabilized `82/82`,
  compileall, and `git diff --check` pass under `.venv`. No physical or
  rounded-slot execution was run.
- Related: `examples/fracture2d/engineering_grip_workflow.py`,
  `tests/test_fracture2d_engineering_grip_monolithic_workflow.py`,
  `examples/fracture2d/README.md`, and
  `docs/tasks/fracture2d-engineering-grip-baseline.md`.

## 2026-09-21 — Close primary monolithic archive/workflow review defects

- Decision: Read the bounded archive bytes once from the opened descriptor and
  inspect/decompress only that immutable snapshot. Detect ZIP64 through
  structural fields and the fixed locator position, not arbitrary payload-byte
  signature searches.
- Decision: Convert only explicit floating-point and linear-algebra Newton
  failures into immutable numerical audits. Propagate type, shape, identity,
  and API contract errors without cutback or archive publication. Reject
  non-default FEM/GPU options on the CPU-only monolithic path.
- Decision: Propagate published-bundle durability uncertainty separately from
  numerical continuation failure. Bind archive/replay/summary fingerprints and
  mark the manifest's own fingerprint scope as self-reference-excluded; a JSON
  document cannot recursively contain its stable SHA256.
- Why: Same-descriptor access alone does not prevent in-place inode rewrites,
  broad catches erase structural provenance, payload signatures are not ZIP
  records, ignored CLI options misstate execution, and durability failure is not
  a numerical result. Explicit self-reference scope avoids claiming an
  impossible manifest self-hash while retaining exact four-member validation.
- Alternatives rejected: post-read mutation detection after decompression,
  scanning all compressed payload bytes for ZIP64 signatures, wrapping
  structural exceptions as solver failures, silently ignoring CPU-inapplicable
  options, returning the numerical-failure status for fsync uncertainty, or
  adding a fifth sidecar solely to hash the manifest.
- Related: `examples/fracture2d/nonlocal_engineering_grip_monolithic.py`,
  `examples/fracture2d/nonlocal_engineering_grip_monolithic_archive.py`,
  `examples/fracture2d/engineering_grip_workflow.py`, focused monolithic tests,
  and `docs/tasks/fracture2d-engineering-grip-baseline.md`.

## 2026-09-21 — Authorize bounded CPU monolithic smoke

- Decision: Run one fresh-zero strict CPU-reference monolithic workflow from
  `u_bar=0` to exact `1e-4` with the reviewed rounded-cloud configuration,
  explicit Newton/GMRES/Armijo controls, ZIP32 budget, atomic bundle, and
  immediate owner replay. Stop at completion or any strict failure.
- Why: Primary implementation review has no remaining finding and all affected
  manufactured/regression suites pass; the bounded smoke is the next staged
  integration check.
- Alternatives rejected: GPU execution, a higher endpoint, warning acceptance,
  archive resume, changing gates or numerical controls, or proceeding without
  owner replay.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md` and the reviewed
  command in `examples/fracture2d/README.md`.

## 2026-09-21 — Stop the bounded CPU monolithic smoke on runtime grounds

- Decision: Treat the user-aborted `0 -> 1e-4` CPU-reference monolithic launch
  as an operational performance stop, not a numerical continuation result. Do
  not rerun the same physical-cloud command unchanged.
- Why: Runtime without progress was unacceptable to the user. Post-interruption
  checks found no solver process, published bundle, or private staging artifact,
  so there is no durable numerical result to replay or interpret.
- Alternatives rejected: allowing the CPU job to continue, relabeling the
  interruption as a strict failure, using partial state as evidence, or silently
  substituting the existing staggered GPU adapter for the CPU-authenticated
  monolithic owner.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md` and run ID
  `engineering_grip_monolithic_cpu_u1e4_20260921`.

## 2026-09-21 — Profile CPU monolithic cost before redesign

- Decision: Use bounded real-cloud setup and single-operation profiling to
  separate model construction, residual/JVP, scaling, preconditioner, and
  linear-solver costs before changing implementation or designing a GPU path.
- Why: The reviewed CPU smoke was operationally too slow, but its lack of
  progress output does not identify the bottleneck. Measured optimization is
  safer than changing numerical architecture speculatively.
- Alternatives rejected: rerunning the full CPU continuation unchanged,
  guessing the hotspot, weakening numerical checks, or immediately replacing
  the owner with an unreviewed GPU implementation.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md`.

## 2026-09-22 — Use sparse A2 algebra and exact mechanics preconditioning

- Decision: Represent the damage-independent A2 support gradient as a read-only
  CSR operator and reuse it for gradient, tangent, and endpoint-divergence
  actions. Freeze and cache identical monolithic damage views and JVP
  kinematics within one CPU owner. Replace the monolithic mechanics
  particle-block Jacobi preconditioner with an exact sparse solve of the same
  free-DOF fixed-damage tangent; retain the exact strong-damage and scaled
  history blocks and omit all coupling blocks as before.
- Why: Real-cloud profiling measured the original zero-damage work check at
  `207.35s`, one tangent action at about `0.46s`, and an initial GMRES that did
  not finish within `180s`. The CSR action is algebraically the same support
  gradient/divergence (real-cloud tangent relative defect `1.45e-15`). The
  assembled mechanics tangent matched the matrix-free action to relative defect
  `1.90e-15`; its sparse factorization took `0.676s`. With the exact mechanics
  block, the same initial GMRES fell from `94.3s/1927` callbacks after operator
  caching to `1.03s/16` callbacks while passing the unchanged unpreconditioned
  residual gate.
- Alternatives rejected: weakening tolerances or gates, replacing three
  independent work-check solves by load scaling, silently using the staggered
  GPU adapter, retaining the measured slow block-Jacobi preconditioner, or
  replacing the legacy deformation-gradient evaluation with `I + Dq` (which
  erased the audited tolerance-level nonroot branch at zero load).
- Related: `examples/fracture2d/nonlocal_a2_coupled.py`,
  `examples/fracture2d/nonlocal_engineering_grip_monolithic.py`,
  `tests/test_fracture2d_nonlocal_a2_coupled.py`,
  `tests/test_fracture2d_nonlocal_engineering_grip_monolithic_newton.py`, and
  `docs/tasks/fracture2d-engineering-grip-baseline.md`.

## 2026-09-22 — Authorize strict monolithic continuation to 0.0075

- Decision: Run one fresh-zero strict CPU-reference monolithic continuation to
  exact `u_bar=0.0075`. Change only `load_final` from the validated `1e-4`
  smoke configuration; retain `du_initial=1e-4`, `du_min=5e-5`, all cutback,
  Newton/GMRES/Armijo controls, physical gates, ZIP32 budget, atomic bundle
  publication, and immediate owner replay. Continue until strict completion or
  a terminal strict failure, even if the run takes hours.
- Why: The user selected the earlier engineering target after the bounded
  `1e-4` CPU integration completed and explicitly confirmed the unchanged
  numerical controls and long-running execution boundary.
- Alternatives rejected: resuming from the `1e-4` archive (unsupported by this
  workflow), changing the minimum increment, warning acceptance, GPU
  substitution, or stopping at an intermediate accepted state.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md` and the
  `engineering_grip_monolithic_cpu_u7p5e3_20260922` run.

## 2026-09-22 — Stop strict monolithic continuation at the configured minimum increment

- Decision: Treat the `u_bar=0.00615` line-search exhaustion as the terminal
  strict result of the authorized `0 -> 0.0075` run. Retain `u_bar=0.0061` as
  the last committed state and do not treat the rejected `0.00615` trial or the
  requested `0.0075` endpoint as reached.
- Why: The fresh-zero owner completed 61 accepted steps, then rejected both the
  `1e-4` and cut-back `5e-5` increments. At the configured minimum increment,
  Newton iteration 8 exhausted all 13 Armijo trials while GMRES itself returned
  successfully and passed its unpreconditioned residual gate. The published
  failure archive was immediately replayed with owner authentication and
  retained-state field-equation validation.
- Alternatives rejected: reducing `du_min`, increasing the line-search budget,
  accepting the closest insufficient-decrease trial, resuming from the failure
  archive, or changing solver/gate parameters without a new numerical decision
  and explicit authorization.
- Related: bundle
  `/root/autodl-tmp/data/fracture2d/engineering_grip/engineering_grip_monolithic_cpu_u7p5e3_20260922.monolithic`
  and `docs/tasks/fracture2d-engineering-grip-baseline.md`.

## 2026-09-22 — Diagnose the shared nonlinear difficulty before new trials

- Decision: Following the user's request, compare existing stagger and
  monolithic evidence and audit the coupled algebra before changing parameters
  or launching another continuation. Keep near-singular coupled feedback,
  active-set transitions, and actual-state derivative consistency as competing
  diagnostic questions rather than declaring a physical limit point.
- Why: Both routes encounter the same strongly damaged load region, but have
  different last accepted loads and histories. Accurate GMRES solves alongside
  growing raw Newton directions and shrinking line-search steps distinguish
  the current failure from simple linear-solver under-resolution.
- Alternatives rejected: blind step/budget tuning, inferring no equilibrium
  from nonlinear failure, or inferring snap-back solely from a falling reaction.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md`, read-only
  stagger/monolithic comparison following the `0.0075` attempt.

## 2026-09-22 — Identify the crack-tip near-unit coupled feedback mode

- Decision: Treat a crack-tip-localized near-unit fixed-active feedback mode as
  the demonstrated common nonlinear conditioning mechanism behind the stagger
  slowdown and monolithic large directions. Keep line-search budget exhaustion
  as the immediate monolithic stop, not the root physical conclusion. Do not
  declare the Schur operator singular, a physical fold, or absence of a root.
- Why: Exact failure-state replay reproduces iteration 8, validates all smooth
  JVP blocks to about `1e-8`, and gives a Newton directional derivative matching
  `-||R||^2` to about `6e-13` relative. The ideal fixed-active stagger map has a
  dominant real eigenvalue `0.9955678745`, concentrated at notch-tip particle
  `1869` with participation ratio `1.056`; its path grows from `0.6677` at
  `u_bar=0.0055` to `0.8922` at `0.0061`. The eigenpair implies
  `sigma_min(I-T) <= 0.00443213` and amplification of at least `225.6` along
  that direction, but nonsymmetry prevents equating this bound with the exact
  smallest singular value.
- Alternatives rejected: attributing failure to GMRES accuracy or a JVP sign
  error, equating nonlinear stagger norm ratios with eigenvalues, treating two
  extra successful Armijo backtracks as proof of eventual convergence, or
  launching parameter sweeps before selecting a bounded solver change.
- Related: exact read-only single-target replay and coupled-spectrum audit in
  `docs/tasks/fracture2d-engineering-grip-baseline.md`.

## 2026-09-22 — Authorize one bounded deeper-Armijo target replay

- Decision: Run exactly one in-memory verified monolithic attempt from the
  archived strict `u_bar=0.0061` committed state to exact `0.00615`, changing
  only `line_search_max_backtracks` from `12` to `14`. Preserve all equations,
  gates, predictor construction, load target, GMRES/Newton controls, and source
  artifacts. Do not run continuation or write a production archive.
- Why: Read-only probing at the deterministically reproduced iteration-8 state
  found the first Armijo-passing trial at `alpha=2^-14`; every configured trial
  through `2^-12` and the additional `2^-13` trial failed. This makes one
  bounded replay evidence-based rather than a budget sweep.
- Alternatives rejected: increasing any other budget, reducing the load
  increment, continuing toward `0.0075`, warning acceptance, or assuming that
  one newly accepted Newton trial establishes eventual convergence.
- Related: user authorization in the current session and
  `docs/tasks/fracture2d-engineering-grip-baseline.md`.

## 2026-09-22 — Reject deeper Armijo alone as the monolithic remedy

- Decision: Stop the bounded `line_search_max_backtracks=14` experiment after
  its exact-target failure. Do not increase line-search depth again or resume
  continuation. Any next solver proposal must separately derive and verify a
  step regularization/globalization method that addresses the demonstrated
  near-unit coupled mode while retaining the original residual as the final
  convergence contract.
- Why: At exact archived target `0.006150000000000033`, Newton iteration 8
  accepted the newly available `alpha=2^-14`, but iteration 9 then rejected all
  15 trials through `2^-14` and exhausted the enlarged budget. GMRES still
  passed with unpreconditioned residual `1.01793e-5` against limit
  `2.55837e-4`. The committed digest remained unchanged and no auxiliary or
  canonical acceptance stage was reached.
- Alternatives rejected: repeatedly extending backtracking, interpreting one
  accepted tiny correction as nonlinear progress, lowering the load increment,
  or proceeding toward `0.0075` under the same globalization.
- Related: bounded in-memory result in
  `docs/tasks/fracture2d-engineering-grip-baseline.md`.

## 2026-09-22 — Authorize globalization derivation and review only

- Decision: Derive and independently review a Schur-aware trust-region or
  pseudo-transient semismooth Newton step for the monolithic engineering-grip
  residual. This phase may inspect code and references and update design memory,
  but must not modify solver behavior or run physical continuation.
- Why: Deeper Armijo accepted one `2^-14` correction and immediately failed at
  the next Newton iteration, while the verified fixed-active operator has a
  notch-tip mode at `lambda(T)=0.9955678745`. A useful next method must control
  that weak coupled direction rather than uniformly shrinking every component.
- Alternatives rejected: further backtrack-budget changes, guessed fixed
  regularization, immediate implementation without a checked adaptive rule,
  changing physical residuals, or using a new physical run to tune the method.
- Related: user authorization in the current session and
  `docs/tasks/fracture2d-engineering-grip-baseline.md`.

## 2026-09-22 — Select full-space trust-region globalization for review

- Decision: Select a scaled full-space semismooth trust-region
  Gauss–Newton/Levenberg–Marquardt design as the first future globalization
  candidate. This selects a method family for manufactured implementation and
  review only; it does not authorize solver edits or a physical continuation.
  At a state `z_k`, freeze the strict active mask and row transform, define
  `F_k = W_k R(z_k)` and `J_k = W_k J_{A_k}(z_k)`, and use

  $$
  m_k(p)=\frac12\lVert F_k+J_kp\rVert_2^2,
  \qquad \lVert D_kp\rVert_2\leq\Delta_k.
  $$

  The equivalent damped KKT equations are

  $$
  (J_k^T J_k+\mu D_k^T D_k)p=-J_k^TF_k,
  \qquad \mu\geq0,
  $$

  with `mu` and the trust-region complementarity conditions determined by the
  subproblem, not guessed as a fixed physical regularization.
- Decision: Future code must apply the damped solve through the matrix-free
  augmented least-squares operator

  $$
  \begin{bmatrix}J_k\\ \sqrt{\mu}D_k\end{bmatrix}p
  \simeq
  \begin{bmatrix}-F_k\\0\end{bmatrix},
  $$

  or an algebraically equivalent Krylov formulation; it must not assemble
  `J_k^T J_k`. `D_k` is an explicit full-column-rank variable metric frozen by
  the model. Its blocks must map the mixed `q`, `d`, and `H` increment units to
  a dimensionless norm. The existing row scales are not a substitute for
  this metric.
- Decision: Use the unregularized frozen-scale model for globalization
  evidence. With `phi_k(s) = 1/2 ||W_k R(z_k+s)||^2`, record

  $$
  \operatorname{ared}_k=\phi_k(0)-\phi_k(p),
  \qquad
  \operatorname{pred}_k=-F_k^TJ_kp-\frac12\lVert J_kp\rVert_2^2,
  \qquad
  \rho_k=\operatorname{ared}_k/\operatorname{pred}_k.
  $$

  A non-finite or non-positive predicted reduction is a rejected model/inner
  solve, not an accepted step. Acceptance thresholds, radius shrink/expand
  factors, initial radius, maximum radius, and inner forcing tolerances remain
  explicit project-owned controls and are not selected here. Accepted trials
  reevaluate the exact max residual, then rebuild the active mask and scales;
  no frozen model is reused after an accepted active-set change.
- The symbolic controller contract to review is: choose explicit
  `0 < eta_reject < eta_expand < 1`, `0 < gamma_shrink < 1`,
  `gamma_expand > 1`, a boundary fraction `0 < theta_boundary < 1`, and a
  positive rejection floor `Delta_min`. Reject a finite trial when
  `rho < eta_reject`, keep `z_k`, and set
  `Delta <- gamma_shrink * Delta`; accept when `rho >= eta_reject`; and, only
  when `rho >= eta_expand` and `||D p|| >= theta_boundary*Delta`, set
  `Delta <- min(gamma_expand*Delta, Delta_max)`. A non-finite trial or invalid
  prediction follows the same rejected-model path. If a rejection would make
  `Delta < Delta_min`, emit a numerical failure audit. These symbols are an
  explicit adaptive rule; their numerical values are not chosen in this
  derivation. An inner Krylov solve that misses its declared forcing
  tolerance must not be silently accepted or replaced by the current GMRES
  path; its retry/failure budget must be a separate explicit control.
- Implementation constraint: `scaled_jvp` already exposes a frozen mask, but
  the current raw `vjp` recomputes `active_mask(z)` internally. A future
  LSMR/LSQR path therefore requires an explicit frozen-mask VJP and its row-
  scaled transpose action. The current block-diagonal object is a left
  preconditioner for GMRES; it is not automatically a right preconditioner
  for the augmented least-squares operator. Any right preconditioner must
  provide a checked transpose action and fixed-shape `D_k P_k` actions.
- Schur relation retained as a diagnostic/preconditioner identity only. For
  the raw blocks, `S = A - C P K^{-1} B = A(I-T)`. The observed
  `lambda(T)=0.9955678745` supports a near-unit coupled mode, but does not
  justify replacing the outer model by `S + mu A`; that shift is not the
  full-space Gauss–Newton/LM KKT system and does not define a global residual
  acceptance contract.
- The pseudo-transient alternative remains rejected as the first route, not as
  an invalid method. Its project-owned equation would be

  $$
  G_\tau(z;z_k)=F(z)+\tau^{-1}M(z-z_k)=0,
  \qquad
  (J_k+\tau^{-1}M)p=-F_k,
  $$

  where `M` must map mixed packed increments into scaled residual coordinates.
  This formulation needs an explicit metric, adaptive `tau` rule, and a
  separate descent/acceptance proof because the nonsymmetric shifted Newton
  step is not automatically a descent step for `1/2 ||F||^2`. No physical
  time/mass operator or adaptive rule is declared by this project.
- Source-backed versus project-owned: Conn–Gould–Toint,
  `https://doi.org/10.1137/1.9780898719857`, supports the trust-region model,
  ratio, and radius globalization family; Moré,
  `https://doi.org/10.1007/BFb0067700`, supports LM least-squares
  regularization; Qi–Sun, `https://doi.org/10.1007/BF01581275`, supports the
  generalized-Jacobian semismooth Newton setting; Kelley–Keyes,
  `https://doi.org/10.1137/S0036142996304796`, supports pseudo-transient
  continuation as a distinct steady-state globalization. The residual,
  strict tie-inactive active set, row scales, variable metric, parameter
  values, active-switch policy, final gates, and all displayed project block
  identities are project-owned. Brown–Hindmarsh,
  `https://doi.org/10.1137/0723039`, supports matrix-free Newton–Krylov as a
  solver pattern; the augmented least-squares operator and transpose API are
  project-owned algebraic implementation choices. DOI/Crossref records were
  available; the publisher full-text endpoints were access-restricted during
  this review.
- Manufactured algebra check: a deterministic random nonsymmetric block
  system verified augmented least-squares versus KKT, predicted-reduction
  identity, Schur factorization, and Schur back-substitution with maximum
  absolute defects `8.88e-16`, `8.88e-16`, `1.11e-16`, and `2.22e-16`.
- Alternatives rejected: another Armijo-depth increase, fixed guessed `mu`,
  direct `S + mu A` outer regularization, identity variable scaling, and
  immediate pseudo-transient implementation without a declared metric and
  adaptive rule. Related code is
  `examples/fracture2d/nonlocal_engineering_grip_monolithic.py`; the active
  task records the required manufactured validation before any physical run.

## 2026-09-22 — Prefer a right-preconditioned Krylov hookstep implementation

- Decision: Retain the full-space Gauss--Newton trust-region model as the
  mathematical contract, but select a right-preconditioned Newton--GMRES
  hookstep as the first manufactured implementation route. Freeze `z_k`, the
  strict active mask, row scales, metric, and block inverse `P_k`, define
  `F_k=W_kR(z_k)`, `J_k=W_kJ_A(z_k)`, and build Arnoldi data

  $$
  J_kP_kV_m=V_{m+1}\bar H_m,
  \qquad Z_m=P_kV_m.
  $$

  With `beta=||F_k||_2` and `v_1=F_k/beta`, solve the projected hookstep

  $$
  \min_y\frac12\left\lVert\beta e_1+\bar H_my\right\rVert_2^2
  \quad\text{subject to}\quad
  \lVert D Z_my\rVert_2\leq\Delta_k,
  \qquad p=Z_my.
  $$

  The plus sign follows from starting Arnoldi with `F_k`; the Newton equation
  remains `J_k p=-F_k`.
- Decision: Use the fixed reduced-coordinate dimensionless product metric

  $$
  \lVert Dp\rVert_2^2=
  \frac{\sum_iV_i^U\lVert(E_qp_q)_i\rVert_2^2}{V_Ul_0^2}
  +\frac{\sum_iV_i^M p_{d,i}^2}{V_M}
  +\frac{\sum_iV_i^M p_{H,i}^2}{V_M(G_c/l_0)^2}.
  $$

  Here `E_q` injects free increments into the full union vector with exact zero
  constrained increments, `V_U=sum(V_i^U)`, and `V_M=sum(V_i^M)`. All stored
  quadrature weights, `l0`, and `Gc` are strictly positive, so this is a
  full-rank norm on the packed reduced variables. It is a normalized continuum
  `L2` product norm, is dimensionless, and does not grow merely because the
  cloud is refined. Equal contribution of the three normalized fields remains
  an explicit project-owned choice requiring refinement tests.
- Decision: Form `B_m=DZ_m` explicitly by actions and use thin QR/SVD, not only
  the Gram matrix `B_m^T B_m`, so metric whitening does not square its condition
  number. Rank loss in `B_m` is an explicit numerical failure because `D`, `P`,
  and an orthonormal `V_m` are full rank in exact arithmetic; it must not be
  silently pseudoinverted. Rank deficiency of `bar H_m` is allowed and handled
  by the minimum-metric-norm least-squares solution for an interior step or by
  positive trust multiplier on the boundary.
- Decision: Production acceptance recomputes
  `j_p=scaled_jvp(z_k,p,scales,active_mask)` and uses

  $$
  \operatorname{pred}=-F_k^Tj_p-\tfrac12\lVert j_p\rVert_2^2,
  \qquad
  \operatorname{ared}=\phi_k(0)-\phi_k(p),
  \qquad
  \rho=\operatorname{ared}/\operatorname{pred}.
  $$

  The projected `bar H_m y` identity is checked diagnostically but is not the
  production acceptance value. Every nonlinear trial evaluates the exact max
  residual with frozen row scales. A rejected finite trial retains the exact
  state, mask, scales, `P`, `D`, Arnoldi basis, and projected matrices; after
  shrinking `Delta`, it resolves the small hookstep and recomputes `p`, `j_p`,
  prediction, and nonlinear residual. No basis is reused after acceptance.
- Decision: The controller must distinguish accepted outer corrections,
  rejected radius trials, Arnoldi/inner failures, and stationary nonroots.
  Invalid operator/preconditioner actions or missed inner tolerances are hard
  inner failures, not model rejections. `F=0` retains the exact-root shortcut.
  If the maximum allowed Krylov space has no positive model reduction while
  `F!=0`, report a Krylov-stationary nonroot failure; do not claim full-space
  stationarity without `J^TF`. Accepted active-set changes rebuild the complete
  model and cannot satisfy final convergence on that same unstable-mask row.
- Decision: The small trust problem must support the unconstrained
  minimum-metric-norm interior solution and a positive-multiplier boundary
  solution. Its secular solve, QR/SVD rank tolerance, Arnoldi dimension and
  orthogonalization/breakdown tolerances, trust-radius/rejection controls, and
  inner residual tolerances are required explicit configuration fields. Their
  numerical values are not authorized by this derivation. The symbolic radius
  rules selected above remain: rejected trials shrink and re-solve at unchanged
  state; accepted ordinary or excellent-interior trials retain the radius;
  only excellent boundary trials expand, capped by `Delta_max`; proposed shrink
  below `Delta_min` or rejection-budget exhaustion fails loudly.
- Why: This route directly bounds the large near-unit-mode Newton correction,
  reuses the already verified exact block inverse as a right variable map, and
  requires only existing frozen-mask scaled JVP actions. The previously selected
  augmented LSMR route requires a new frozen scaled VJP plus transpose-aware
  preconditioning and repeated full-space inner solves. Retain augmented LSMR
  as an independent manufactured reference and future fallback if the projected
  method is repeatedly Krylov-stationary or its metric basis is ill-conditioned.
- Source boundary: Bellavia--Morini,
  `https://doi.org/10.1137/040611951`, explicitly covers low-dimensional
  spherical/elliptical trust regions for large nonlinear systems with Krylov
  implementations. Viswanath, `https://doi.org/10.1017/S0022112007005459`,
  documents a Newton--Krylov/locally constrained optimal hookstep combination.
  Conn--Gould--Toint and Moré remain the trust-region/LM foundations. The exact
  packed metric, block preconditioner interpretation, active-set policy,
  projected rank/failure contract, and all numerical controls are project-owned.
- Alternatives rejected: explicit normal equations or metric Gram whitening,
  treating the projected method as full-space, reusing the current
  left-preconditioned SciPy GMRES audit unchanged, silently truncating a
  rank-deficient metric basis, and implementing full-space VJP/LSMR before the
  smaller projected contract is manufactured-validated.
- Related: `examples/fracture2d/nonlocal_engineering_grip_monolithic.py` and
  `docs/tasks/fracture2d-engineering-grip-baseline.md`.

## 2026-09-22 — Authorize manufactured hookstep core implementation

- Decision: Implement the first manufactured-only hookstep layer: explicit
  configuration and immutable audit types, the fixed quadrature product metric,
  right-preconditioned Arnoldi construction, and the QR/SVD projected
  interior/boundary trust-region solve. Keep it separate from the current
  Armijo Newton entry point, verified-attempt layer, workflow, archive, and
  physical continuation.
- Why: The derivation and two independent numerical reviews selected this as
  the smallest method that directly bounds the localized large Newton step
  while reusing the verified block inverse and frozen-mask JVP. Isolating its
  algebra first permits dense-oracle, rank, metric, and state-machine testing
  before it can affect accepted physical states.
- Alternatives rejected: replacing Armijo immediately, adding workflow flags or
  archive schema before the core audit is stable, physical tuning, hidden
  defaults, or implementing the larger frozen-VJP augmented-LSMR path first.
- Related: user authorization in the current session and the manufactured plan
  in `docs/tasks/fracture2d-engineering-grip-baseline.md`.

## 2026-09-22 — Complete and harden the isolated hookstep core

- Decision: The isolated manufactured hookstep core is complete. It exposes an
  explicit required configuration, the owner-paired quadrature product metric,
  right-preconditioned Arnoldi evidence, SVD-whitened projected interior and
  boundary solves, fresh full-action prediction checks, and a pure radius
  decision. It remains undispatched from Newton, Phase-2B, continuation,
  workflow, and archive code.
- Decision: Compute projected reduction with the cancellation-resistant form
  `-r^T(Hy)-||Hy||^2/2`, not a difference of squared residual norms. Retain a
  nonzero terminal Arnoldi vector even when a relative tolerance declares happy
  breakdown, so every stored Hessenberg column remains reconstructible.
- Decision: Treat the trust metric as model-owned numerical data, not merely a
  shape-compatible value. Arnoldi rebuilds the canonical metric and rejects
  same-owner forged diagonals or normalization scalars. Projected failures are
  rebound to the full owner/state/mask/scale audit at the direction API, and
  non-finite trial ratios or step norms are numerical radius failures.
- Why: Independent review found that squared-norm subtraction could erase a
  valid tiny predicted reduction, identity-only metric pairing admitted altered
  trust geometry, and tolerance-triggered breakdown could omit evidence needed
  for the final Arnoldi relation. These issues are now covered by focused
  manufactured regressions.
- Verification: the hookstep suite passes `18/18`; combined hookstep,
  monolithic residual/JVP, and existing Newton/preconditioner suites pass
  `56/56`; changed modules compile; `git diff --check` passes. Independent
  numerical re-review reports no remaining findings or blocking issue. No
  physical continuation or durable artifact was run or changed.
- Alternatives rejected: cancellation-prone prediction, accepting a forged
  owner-identical metric, discarding a nonzero terminal vector, weakening rank
  failures into pseudoinverse fallbacks, and wiring the uncalibrated core into
  the existing solver.
- Related: `examples/fracture2d/nonlocal_engineering_grip_monolithic.py`,
  `tests/test_fracture2d_nonlocal_engineering_grip_monolithic_hookstep.py`, and
  `docs/tasks/fracture2d-engineering-grip-baseline.md`.

## 2026-09-22 — Authorize manufactured-only hookstep outer controller

- Decision: Implement only an isolated outer controller for one frozen-model
  hookstep correction. It may consume `trust_radius_initial`, build one metric,
  preconditioner/Arnoldi model, repeatedly solve the projected problem after
  rejected radius trials, evaluate exact nonlinear trial residuals with frozen
  row scales, and return after the first accepted candidate.
- Decision: Rejected trials must preserve the input state, scale object, active
  mask, metric, and Arnoldi evidence exactly. Accepted output records whether
  the exact active mask changed and requires the caller to rebuild every
  state-dependent object; the controller must not perform a second Newton
  correction with stale data.
- Decision: A non-finite nonlinear trial is a rejected trial and follows the
  explicit shrink/floor/rejection-budget path selected in the trust-region
  derivation. A malformed or non-finite step, JVP, metric action, prediction, or
  projected solve remains a hard inner numerical failure. This distinguishes
  trial globalization from invalid inner algebra.
- Why: The user selected the manufactured-only outer-controller stage. This is
  the smallest next layer that validates rollback, radius re-solve, and rebuild
  boundaries without selecting physical controls or changing existing solver
  behavior.
- Alternatives rejected: dispatch from the current Newton solver, continuing
  through an accepted active-set change inside the same frozen model, treating
  a non-finite nonlinear candidate as an accepted value, adding workflow or
  archive fields, selecting physical parameters, or running continuation.
- Related: user authorization in the current session and
  `docs/tasks/fracture2d-engineering-grip-baseline.md`.
