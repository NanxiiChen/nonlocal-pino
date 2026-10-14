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

## 2026-09-22 — Complete the isolated hookstep outer controller

- Decision: Complete the manufactured-only outer controller as
  `run_monolithic_hookstep_correction(...)`. One invocation builds one canonical
  metric and one right-preconditioned Arnoldi model, starts from
  `trust_radius_initial`, re-solves only the projected trust problem after each
  rejected radius trial, evaluates the exact nonlinear trial residual under the
  frozen scales, and returns immediately after the first accepted candidate.
  Accepted output sets `rebuild_required=True` and reports whether the exact
  active mask changed; no second correction reuses a stale model.
- Decision: The controller derives its own scale bundle from an explicit
  `gate_config` instead of accepting caller scales. This removes the forged
  `Lq`/`Lrho`/`LH`/`H_next` path that survived shape-and-digest-only scale
  validation, and keeps the live numerics owner-computed.
- Decision: Preserve complete immutable terminal evidence. Unrepresentable
  finite ratios are recorded as a terminal `failed_ratio` trial; subnormal
  radius-shrink underflow, rejection-budget exhaustion, radius-floor crossing,
  trial `LinAlgError`, accepted-mask rebuild failure, and pre-model
  scale/active-mask failures raise audited failures retaining the triggering
  trial where one exists. A non-finite candidate, residual, or actual reduction
  is an explicit radius rejection, while a finite trial residual is retained
  even when the derived actual reduction overflows.
- Why: Independent review found that the prior controller could swallow a hard
  `LinAlgError` as a radius rejection, accept forged scale data, drop the trial
  that triggered a terminal ratio failure, and let a subnormal shrink bypass the
  audited floor failure. All are resolved and regression-tested.
- Verification: focused controller `16/16`, hookstep core `19/19`, and combined
  controller/core/residual/JVP/Newton `73/73`; changed modules compile;
  `git diff --check` passes. Two independent reviews of the completed controller
  report no remaining blockers.
- Alternatives rejected: caller-supplied scales, accepting forged scaling,
  continuing through an accepted active-set change, dropping terminal trial
  evidence, downgrading structural linear-algebra errors to globalization
  rejections, and wiring the controller into the existing solver.
- Related: `examples/fracture2d/nonlocal_engineering_grip_monolithic.py`,
  `tests/test_fracture2d_nonlocal_engineering_grip_monolithic_hookstep_controller.py`,
  and `docs/tasks/fracture2d-engineering-grip-baseline.md`.

## 2026-09-22 — Complete the trust-region Newton solver

- Decision: Complete the manufactured trust-region Newton solver
  `solve_monolithic_hookstep_newton(monolithic, initial_z, config, gate_config)`
  around the validated hookstep controller. `MonolithicTrustRegionConfig`
  requires `max_newton_iterations` plus an explicit `MonolithicHookstepConfig`.
  Each iteration freezes the state/scales/active set, takes one bounded
  correction, rebuilds all state-dependent data, and checks the same strict
  convergence gates as the Armijo core. The adapted radius is carried by
  rebuilding the frozen hookstep config with `trust_radius_initial=next_radius`.
- Decision: Numerical failures always surface as `MonolithicTrustRegionFailure`
  with an immutable partial audit. Bare hookstep failures and controller
  failures are both wrapped and retain their controller/inner-hookstep evidence;
  accepted-trial failures are audited against the valid reference
  state/scales; a mid-loop exact root is a dedicated `exact_root` stage instead
  of a crash. Results carry full relational validation and an owner-replay method
  (`validate_with_owner`).
- Why: Independent review of the first implementation found a blocker in the
  accepted-trial failure audit (scale/state pairing `ValueError`), an escaping
  `MonolithicHookstepFailure`, and a mid-loop exact-root `IndexError`, plus
  result-forgery gaps. These are resolved and regression-tested.
- Physical evidence (read-only, exploratory controls, no artifact): from the
  committed `0.0061` state the solver reaches `0.00611` in `6` iterations with
  final scaled block norms `(2.78e-5, 1.21e-8, 1.00e-6)`, the first monolithic
  advance past `0.0061`. Targets `0.00612`/`0.00615` stall within `40`
  iterations, showing the near-peak increment must be about `1e-5`.
- Verification: trust-region suite `23/23`; combined affected `96/96`; changed
  modules compile; `git diff --check` passes; two independent reviews report no
  remaining blockers.
- Alternatives rejected: changing the shared hookstep radius guard back to
  `metric_consistency_rtol`, weakening the strict convergence gates, and wiring
  the solver into the workflow/archive before the attempt-runner contract is
  designed.
- Related: `examples/fracture2d/nonlocal_engineering_grip_monolithic.py`,
  `tests/test_fracture2d_nonlocal_engineering_grip_monolithic_trust_region.py`,
  and `docs/tasks/fracture2d-engineering-grip-baseline.md`.

## 2026-09-23 — Hold formulation changes after canonical stall diagnosis

- Decision: Supersede the exploratory `0.006117` avalanche-barrier and
  full-stationarity interpretation. Keep the current `q-d-H` solver unchanged
  while requesting explicit authorization for any reduced formulation.
- Why: Canonical diagnostics show that the active mask becomes bitwise fixed,
  accepted corrections retain positive actual reduction, and K32/K64 projected
  gradients remain large. Block-level evidence instead identifies an explicit
  history-variable two-cycle: per-correction `H` changes are about `0.11528`,
  while two-correction net changes collapse below `4.5e-5`. A no-expansion test
  reaches K32-only false stationarity, and a direct K64 run still stalls.
- Alternatives rejected: labeling the state a physical fold, treating K32
  projected stationarity as full stationarity, increasing Krylov dimension as
  a complete remedy, suppressing radius growth until numerical failure,
  changing the trust metric without derivation, relaxing strict gates, or
  committing any scratch state.
- Relation: The smallest supported next formulation candidate is exact
  algebraic elimination of `H` while preserving
  `H=max(H_committed, psi_plus(q))`; implementation still requires explicit
  user authorization and independent derivation/testing.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md` and
  `/tmp/opencode/stall_diagnostic*.jsonl`.

## 2026-09-23 — Authorize manufactured exact history elimination

- Decision: Implement and independently test an isolated reduced `q-d` residual
  that substitutes the unchanged history equation
  `H=max(H_committed, psi_plus(q))` exactly. The selected generalized derivative
  retains the existing strict active rule `psi_plus > H_committed`, so exact
  ties remain inactive, and the reduced damage JVP must include the induced
  `-J_dH J_Hq` chain term.
- Decision: Keep this stage manufactured-only. Do not dispatch the reduced
  residual from the Newton, hookstep, verified-attempt, continuation, workflow,
  or archive layers, and do not run a physical continuation. Do not select a
  reduced trust metric or claim that the existing three-block preconditioner is
  a reduced solver contract.
- Why: Canonical `0.006117` diagnostics isolate a numerical two-cycle in the
  explicit history unknown after the active mask is fixed. Exact algebraic
  elimination preserves the physical history equation while removing that
  redundant unknown; independent derivations agree on the residual, signs,
  strict-tie semantics, and required chain rule.
- Alternatives rejected: dropping the chain term, smoothing or tolerancing the
  maximum, reusing the full `q-d-H` metric or preconditioner by implication,
  changing the trust metric, adding arc length, relaxing gates, or using this
  manufactured stage to authorize a physical run.
- Related: user authorization in the current session and
  `docs/tasks/fracture2d-engineering-grip-baseline.md`.

## 2026-09-23 — Complete the isolated reduced q-d residual contract

- Decision: Complete `ReducedMonolithicGripResidual` as an owner-bound,
  manufactured-only adapter. It packs only `q-d`, reconstructs canonical
  history from a zero-damage kinematic probe, embeds the state on the exact
  full history manifold, and evaluates the reduced residual through the
  authoritative full owner.
- Decision: Form the reduced generalized JVP by first recovering
  `delta_H=-J_Hq delta_q` under one explicitly frozen strict mask, then applying
  the authoritative full JVP once to `(delta_q, delta_d, delta_H)`. Require its
  history action to cancel exactly. This preserves canonical damage-row
  operation order and includes the required `-J_dH J_Hq` term.
- Why: Independent from-definition mechanics, Amor, and cubic-damage-graph
  oracles match the reduced residual and eliminated JVP; fixed-mask finite
  differences pass; exact ties remain inactive; and mixed active/inactive rows
  are covered without differencing through a kink.
- Verification: focused residual tests pass `29/29`; combined residual, Newton,
  hookstep, controller, and trust-region tests pass `88/88`; changed Python
  files compile; `git diff --check` passes; independent numerical and code
  re-reviews report no blocker.
- Alternatives rejected: separately adding already volume-weighted damage
  actions, manufacturing exact ties from an independent operation order, and
  adding reduced scaling, preconditioning, trust metric, solver dispatch, or
  physical execution in this stage.
- Related: `examples/fracture2d/nonlocal_engineering_grip_monolithic.py`,
  `tests/test_fracture2d_nonlocal_engineering_grip_monolithic.py`, and
  `docs/tasks/fracture2d-engineering-grip-baseline.md`.

## 2026-09-23 — Authorize the manufactured reduced solver contract

- Decision: Build the isolated reduced `q-d` trust-region solver with row scales
  obtained from the exact full lift and only `Lq/Lrho` applied to reduced rows;
  retain nonlinear reconstructed-history correction and active-set-stability
  gates despite eliminating the history residual row.
- Decision: Use the frozen differential pullback of the approved normalized
  `q/d/H` continuum metric through
  `delta_H=-J_Hq delta_q`. Implement its rectangular factor action without
  forming a Gram matrix, freeze it through rejected radius trials, and rebuild
  it after every accepted correction.
- Decision: Use an explicitly right-oriented lower-triangular preconditioner
  that exactly solves the mechanics and fixed-history damage diagonal blocks,
  retains the reduced lower-left history-chain action, and omits only the
  upper-right mechanics-from-damage block. Reuse the existing projected SVD,
  prediction, rejection, and radius-update policy with reduced-specific
  ownership and audit types.
- Why: Independent derivation and cross-review identify the pullback as the
  exact tangent restriction of the already approved full trust geometry. The
  triangular inverse preserves the new coupling exposed by exact history
  elimination and concentrates the remaining feedback in the mechanics Schur
  factor without introducing a tuning coefficient.
- Alternatives rejected: a simple `q-d` metric, nonlinear secant trust region,
  a primary two-block diagonal preconditioner that drops the history chain,
  weakening history correction gates, and adapting controls from a physical
  target.
- Scope: manufactured-only implementation and tests. No verified-attempt,
  continuation, workflow, archive, CLI, physical controls, physical run, or
  artifact is authorized.
- Related: user authorization in the current session and
  `docs/tasks/fracture2d-engineering-grip-baseline.md`.

## 2026-09-23 — Complete the manufactured reduced solver layer

- Decision: Complete the isolated reduced foundations and solver as
  `ReducedMonolithicNewtonScales`, `ReducedMonolithicTrustMetric`,
  `ReducedMonolithicLowerTriangularRightPreconditioner`,
  `run_reduced_monolithic_hookstep_correction`, and
  `solve_reduced_monolithic_hookstep_newton`. The reduced adapter freezes its
  owner binding; metric and preconditioner re-check construction-time owner
  pairing at apply time; `scaled_residual` supports frozen-scale nonlinear
  trial evaluation while other consumers keep strict reference-state pairing.
- Decision: Rejected radius trials reuse one frozen state, mask, scales,
  rectangular pullback metric, preconditioner, and Arnoldi model. Accepted
  corrections rebuild every state-dependent object and report the exact
  active-set comparison. Convergence keeps the reconstructed-history correction
  gate and active-set-stability requirement even though the history residual row
  is eliminated. Exact roots bypass metric/Arnoldi; non-finite trials are radius
  rejections that retain finite residuals, while inner algebra failures remain
  hard audited failures. `validate_with_owner` provides full owner replay.
- Why: Independent numerical and code reviews confirmed the projected
  orientation, whitening, signs, failure boundaries, and owner replay, and found
  no blocker. Their low-severity findings were closed before completion:
  accepted-correction metric state/digest/mask pairing, a discriminating
  radius-slack regression, and an exact-initial-root requirement for
  trial-free results.
- Verification: reduced foundation `7/7`; reduced trust-region `21/21`;
  combined affected battery `116/116`; changed-file `py_compile`;
  `git diff --check`. The shared projected-hookstep core was extracted
  textually unchanged and is covered by the existing full-solver tests.
- Alternatives rejected: inferred physical controls, durable integration,
  weakening history/active-set gates, reusing full audit types by shape
  compatibility, and changing full `q-d-H` behavior.
- Related: `examples/fracture2d/nonlocal_engineering_grip_monolithic.py`,
  `tests/test_fracture2d_nonlocal_engineering_grip_monolithic_reduced_newton.py`,
  `tests/test_fracture2d_nonlocal_engineering_grip_monolithic_reduced_trust_region.py`,
  and `docs/tasks/fracture2d-engineering-grip-baseline.md`.

## 2026-09-23 — Authorize the reduced physical path to full loading

- Decision: The user authorized continuing implementation until the full-load
  `u_bar=0.0075` final solve is completed. This authorizes (a) selecting the
  reduced solver controls by reusing the already explicit full-run controls
  where the dimensionless metric makes them equivalent, (b) implementing an
  isolated reduced physical attempt/continuation seam with durable artifacts,
  (c) bounded read-only/in-memory physical probes from the committed `0.0061`
  state, and (d) the full continuation run to exact `u_bar=0.0075`.
- Decision: Preserve every strict finite/residual/range/healing/correction gate,
  the no-warning-acceptance rule, and all existing failure/checkpoint artifacts.
  Do not guess new physics, arc length, regularization, or relaxed gates. A
  material mathematical ambiguity still stops the run and returns to the user.
- Why: The reduced formulation removes the explicit-history unknown whose
  two-cycle blocked the full `q-d-H` solver at `0.006117`, and it is now
  independently manufactured-verified; the remaining question is physical.
- Alternatives rejected: warning acceptance, gate relaxation, inferred
  physical controls from rejected states, overwriting existing artifacts, and
  declaring the objective complete from a nonconverged state.
- Related: user authorization in the current session and
  `docs/tasks/fracture2d-engineering-grip-baseline.md`.

## 2026-09-23 — Diagnose the reduced fold and validate the strict endpoint

- Decision: Keep every strict gate, equation, metric, and control unchanged.
  Record the fold evidence and the independently validated strict `0.0075`
  endpoint, and return the strict-continuation-across-the-fold decision to the
  user instead of guessing a new globalization.
- Why: From the authenticated strict state at `u_bar=0.00623125` the reduced
  trust-region stalls on the first increment (plateau `(1.12, 0.43)` at
  `du=2.5e-7`, `(13.8, 2.55)` at `du=1.25e-6`). Read-only controls do not
  change the plateau: K64 gives `(66.1, 5.50)`, the truncated direct `q-d`
  metric gives `(67.9, 5.82)`, and the monolithic predictor branch gives
  `(66--68, 5.5--6.6)`. Warning states are not near strict roots (damage norm
  `57.3` at `0.0063`) and diverge when strictly polished. Strict loads beyond
  the fold are history-path dependent: the `0.0069` strict seed has reduced
  damage norm `1.5e3` from the `0.00623125` committed history but `2.1e-4`
  from the warning-archive `0.0068` history.
- Decision: Independently validate the archived strict endpoint. From the
  `0.0074` stabilized-converged state, the reduced Newton at `0.0075` converged
  in `2` iterations with norms `(2.28e-5, 4.19e-11)`; the auxiliary sweep and
  all canonical gates passed, giving strict state digest `21f17cab...`.
  A strict equilibrium at the complete loading therefore exists; only the
  strict continuation across the fold interval is missing.
- Decision: Fix the shared `MonolithicHookstepDirection` radius guard to use
  the reviewed `(1 + secular_rtol) / (1 - metric_consistency_rtol)` envelope
  instead of `1 + secular_rtol`, removing a spurious bare `ValueError` on
  legitimately reconstructed steps. Affected battery `139/139`.
- Alternatives rejected: warning acceptance, gate relaxation, history
  injection or re-anchoring, treating a skipped-load state as path evidence,
  tuning controls on the physical stall, and declaring completion from the
  endpoint root alone without user confirmation of the completion semantics.
- Related: user authorization in the current session,
  `docs/tasks/fracture2d-engineering-grip-baseline.md`, and
  `/tmp/opencode/reduced_*` read-only diagnostics.

## 2026-09-24 — Authorize fracture-energy-controlled arc-length continuation

- Decision: The user selected a fracture-energy-controlled arc-length method to
  cross the strict reduced crack-growth fold. Augment the reduced unknowns with
  `u_bar`, use the repository's existing discrete nonlocal AT2 fracture energy
  as the path quantity, permit intermediate load reversal, and commit history
  only after an augmented equilibrium passes the existing strict canonical
  acceptance gates. After crossing the fold, continue to exact
  `u_bar=0.0075`.
- Why: The fixed-load reduced hookstep stalls despite step-size, Krylov-space,
  metric, and seed changes. Kelley--Keyes pseudo-transient continuation is a
  fixed-target root globalization, not branch following. Singh et al. (2016)
  and Bharali et al. (2022) instead support fracture-surface/energy control for
  phase-field snap-back by solving for the load factor together with the field
  variables. This is the closest source-backed method family for the observed
  crack-growth barrier.
- Scope: First derive the exact project-specific discrete energy constraint,
  load derivative, augmented metric, predictor orientation, correction system,
  adaptation/cutback policy, and immutable audit contract. Independently verify
  those on manufactured systems before physical execution. Preserve the reduced
  residual, exact history maximum and tie rule, physical equations, strict final
  gates, rollback, and all existing artifacts.
- Alternatives rejected: pseudo-transient as the selected first route because
  it does not track a fold; treating the already reduced exact-max equation as a
  new semismooth method; warning acceptance, history injection, gate relaxation,
  or importing a local-FEM fracture-energy formula without deriving its exact
  nonlocal discrete counterpart.
- Related: user selection in the current session,
  `docs/tasks/fracture2d-engineering-grip-baseline.md`, Singh et al. (2016),
  DOI `10.1016/j.finel.2015.12.005`, and Bharali et al. (2022), DOI
  `10.1016/j.cma.2022.114927`.

## 2026-09-24 — Implement the second manufactured arc foundation

- Decision: Keep the arc route isolated to one frozen augmented
  ``(q-d,u_bar)`` linearization, the exact surface-energy row, the full-union
  rectangular physical metric, and the bordered right preconditioner. Use the
  existing reduced residual, scales, load column, and lower-triangular right
  preconditioner without changing fixed-load solver APIs.
- Why: The selected contract requires independent manufactured evidence before
  any outer nonlinear solver, verified attempt, continuation controller,
  archive, or physical run. Frozen owner/state/mask/config identities and
  digests prevent trial or stale evidence from being reused silently.
- Numerical contract: ``LG`` is derived from the existing damage correction
  limit and a documented ``512*eps64`` local-plus-symmetric-graph constituent
  bound; the metric includes constrained load-direction DOFs; a scale-aware
  bordered denominator failure is hard, with no fallback. Intermediate signed
  trial loads use a private owner construction path while the public fixed-load
  constructor keeps its non-negative-load contract.
- Verification: the new focused arc suite independently checks augmented
  residual/JVP finite differences, energy evidence, metric action/rank, dense
  bordered action, forgery rejection, and immutability/signatures. No solver,
  workflow, archive, or physical continuation was added.
- Related: `examples/fracture2d/nonlocal_engineering_grip_monolithic_arc.py`,
  `tests/test_fracture2d_nonlocal_engineering_grip_monolithic_arc.py`, and
  `docs/tasks/fracture2d-engineering-grip-baseline.md`.

## 2026-09-24 — Implement the manufactured fixed-increment arc solver

- Decision: Extend the isolated arc foundation with an explicit
  `ArcTrustRegionConfig`, right-preconditioned augmented Arnoldi, projected
  hookstep radius re-solves, a frozen-model correction controller, and a
  fixed-`delta_G` nonlinear outer solve. Keep the result, diagnostics, and
  failure audits immutable and owner-replayable.
- Why: This is the next user-authorized layer and preserves the required
  separation between frozen trial algebra and accepted owner/history rebuilds.
  Rejected radii therefore reuse the same metric, bordered preconditioner, and
  Arnoldi evidence while evaluating the exact frozen nonlinear residual.
- Alternatives rejected: changing the existing fixed-load APIs, rebuilding
  Arnoldi for each rejected radius, falling back after a bordered-denominator
  failure, committing trial history/damage, or wiring continuation/workflow,
  archive, or physical execution before manufactured review.
- Verification: focused arc solver/foundation/residual tests pass `20/20`,
  affected reduced suites pass `51/51`, hookstep/controller suites pass `35/35`,
  and the changed arc module compiles under `py_compile`. No physical run was
  executed.
- Related: `examples/fracture2d/nonlocal_engineering_grip_monolithic_arc.py`,
  `tests/test_fracture2d_nonlocal_engineering_grip_monolithic_arc_solver.py`,
  and `docs/tasks/fracture2d-engineering-grip-baseline.md`.

## 2026-09-24 — Harden manufactured arc audit and owner boundaries

- Decision: Keep signed arc increments reversible while rejecting every trial
  whose absolute prescribed `u_bar` is negative. Classify nonlinear-trial
  `FloatingPointError` as an audited radius rejection, but classify trial and
  inner-construction `LinAlgError` as hard audited numerical failures; let
  structural `TypeError`/`ValueError` and API/pairing defects propagate.
- Decision: Require arc result and failure audits to bind model/static/path,
  `delta_G`, gate configuration, canonical active masks, state/radius chains,
  iteration counts, and stage-specific payloads. Replay every diagnostic row,
  correction, and final state through freshly constructed owner evidence.
- Decision: Build accepted-load arc linearizations from a load-only clone of the
  previous owner, share only immutable load-invariant CSR/gradient/static data,
  and cache the surface-energy Hessian absolute-sum scalar on the owner. Avoid
  repeating canonical reduced block-scale construction in the arc wrapper.
- Why: The prior implementation could relabel structural defects as numerical
  failures, shrink the radius after a hard trial linear-algebra failure, accept
  forged partial/result evidence, and rebuild large immutable operators during
  every trial or accepted outer rebuild.
- Alternatives rejected: allowing negative absolute loads, catching broad
  `ValueError`, treating trial `LinAlgError` as globalization evidence,
  rebuilding a fresh full owner after every accepted correction, and validating
  only the final residual.
- Related: `examples/fracture2d/nonlocal_engineering_grip_monolithic.py`,
  `examples/fracture2d/nonlocal_engineering_grip_monolithic_arc.py`, and the
  focused arc/monolithic regression suites.

## 2026-09-24 — Select metric-scaled arc preconditioner after bordered failure
- Decision: Make the parameter-free metric-scaled block-diagonal right
  preconditioner the only preconditioner used by the arc solver. Retain the
  bordered map only as manufactured diagnostic evidence, and normalize its
  augmented load block by the metric norm of the pure unit-load direction.
- Why: The recorded physical-shape probe at `u_bar=0.00623125` with
  `delta_G=0.00873355605` failed structurally in about 7 seconds because the
  bordered Schur quantity `c^T P b` was zero. The new map remains full rank
  without introducing a tunable denominator perturbation.
- Alternatives rejected: epsilon perturbation of the bordered denominator,
  silent bordered fallback, and changing the residual, metric, gates, or
  history semantics.
- Related: `examples/fracture2d/nonlocal_engineering_grip_monolithic_arc.py`,
  `docs/tasks/fracture2d-engineering-grip-baseline.md`, and the independent
  arc/preconditioner regression tests.

## 2026-09-24 — Equilibrate the physical arc Krylov coordinates

- Decision: Preserve the right-preconditioned Arnoldi span and its original
  ``J M``/rectangular-metric evidence, then build an immutable SVD-equilibrated
  basis from ``Z=M V`` using the singular values of ``D Z``. Re-evaluate the
  frozen JVP directly on each equilibrated column and use its projected action
  matrix for the arc hookstep solve and reconstruction.
- Why: The strict manufactured/diagnostic failure has accurate individual
  columns but a raw-coordinate norm ratio of about ``4.76e9``; the projected
  prediction defect is ``0.03766791043624996`` while the metric and Arnoldi
  identities remain accurate. This is cancellation in the projected
  combination, not evidence that the strict gate is too tight.
- Numerical contract: rank is tested against the largest physical singular
  value with the existing explicit metric rank tolerance; deficient spans,
  non-finite actions, metric orthogonality failure, and direct/transformed JVP
  disagreement are hard audited failures. No pseudoinverse, epsilon
  perturbation, silent fallback, or prediction-gate relaxation is introduced.
- Alternatives rejected: accepting the raw full prediction, widening
  `prediction_consistency_rtol`, transforming only the stored action columns,
  or rebuilding the physical continuation/probe before manufactured review.
- Verification: the manufactured arc/foundation/solver battery passes
  `24/24`; the monolithic hookstep/trust-region/controller battery passes
  `74/74`; the new ill-conditioned cancellation regression verifies direct and
  projected prediction agreement. No physical continuation or probe was run.
- Related: `examples/fracture2d/nonlocal_engineering_grip_monolithic_arc.py`,
  `examples/fracture2d/nonlocal_engineering_grip_monolithic.py`,
  `tests/test_fracture2d_nonlocal_engineering_grip_monolithic_arc_solver.py`,
  and the active engineering-grip task.

## 2026-09-24 — Refine the recomputed arc metric basis without loosening the gate

- Observation: A read-only physical probe after the initial equilibration
  reached Newton iteration 3 and then failed at `u_bar=0.0064136878883856953`
  because the recomputed metric defect was `4.8453109785517867e-07` against
  the unchanged `1e-8` gate. The residual norms were
  `(13654.191747629044, 400.53234921112465, 190888.96316738456)`, `Gerr` was
  `8.595017e-03`, and the radius was `.005`. No replacement physical probe is
  authorized by this decision.
- Decision: After the initial SVD transform, permit at most the explicit fixed
  `_ARC_EQUILIBRATION_REFINEMENT_PASSES = 3` passes. Each pass forms the
  symmetrized small Gram matrix `(D Q)^T(D Q)`, applies its symmetric inverse
  square root, composes the transform, recomputes `Q` from the original source
  span, and recomputes `D Q`. The final direct JVP is evaluated only after the
  final metric basis is accepted.
- Provenance contract: retain the source span, initial SVD transform, every
  polar refinement transform, final composed transform, metric-defect history,
  and scale-aware composition-defect history. Composition is checked
  relatively against the existing metric consistency tolerance rather than
  requiring bitwise associativity. A nonpositive/rank-deficient Gram matrix,
  composition defect, or final metric defect is a hard audited failure.
- Alternatives rejected: loosening the `1e-8` metric gate, silently accepting
  the first recomputed basis, using a pseudoinverse/eigenvalue clipping,
  adding an epsilon perturbation, or rerunning the physical probe before
  manufactured verification.
- Related: `examples/fracture2d/nonlocal_engineering_grip_monolithic_arc.py`,
  `tests/test_fracture2d_nonlocal_engineering_grip_monolithic_arc_solver.py`,
  and `docs/tasks/fracture2d-engineering-grip-baseline.md`.

## 2026-09-24 — Verify fixed-pass arc metric refinement

- Verification: the arc foundation/solver battery passes `25/25`; the final
  combined reduced/arc/trust-region/hookstep/controller battery passes
  `120/120`; the shared monolithic residual/Newton/continuation/verified
  battery passes `72/72`. Changed modules compile and `git diff --check` pass.
- The manufactured regression records an initial metric defect above `1e-8`,
  one polar refinement transform, and a final defect below `1e-8` while
  preserving the source span. No physical probe or continuation was run after
  the refinement change.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md` and the
  preceding metric-refinement decision.

## 2026-09-24 — Bind scale-aware composition evidence into arc replay

- Decision: store the existing metric-consistency tolerance with the
  equilibrated basis and validate transform/basis recomposition by relative
  Frobenius defect rather than requiring bitwise associativity. Preserve the
  same initial/refinement/final transform histories on direct-JVP and action
  consistency failures.
- Reason: floating-point reassociation may change composed coordinates without
  changing the represented source span at the existing strict tolerance; a
  bitwise-only check would reject valid evidence, while an unbounded check would
  hide a provenance error.
- Rejected: loosening the gate, accepting a fixed absolute tolerance, or
  recomputing the final basis from a different span.
- Verification: arc foundation/solver `25/25`, changed-module `py_compile`,
  and `git diff --check` pass; no physical probe was run.
- Related: `_ArcEquilibratedKrylovBasis`, `_ArcEquilibratedBasisFailure`, and
  `_validate_arc_arnoldi_replay` in
  `examples/fracture2d/nonlocal_engineering_grip_monolithic_arc.py`.

## 2026-09-24 — Replace source-recomposed polar passes with physical MGS

- Observation: the authorized read-only rerun still failed at Newton iteration
  3 after the three polar passes: the initial metric defect was
  `4.8453109785517867e-07` and the final defect was
  `9.0377926031948164e-08`, still above the unchanged `1e-8` gate. No further
  physical probe is authorized by this decision.
- Decision: when the initial SVD basis misses the metric gate, perform exactly
  two modified-Gram-Schmidt passes in the explicit `D` metric. Update the
  already formed physical columns `q_j`, recompute `D q_j` after every
  orthogonalization/normalization update, and accept only the final recomputed
  `D Q` gate. Do not rebuild the accepted basis from `Z @ T`.
- Provenance: update the coordinate transform in parallel as mathematical
  evidence only. Audit source/total recomposition with a documented
  scale-aware `gamma_k`/elementwise-product roundoff bound; the bound is not a
  replacement for the strict physical `D Q` orthogonality gate. The same
  bound is retained for transformed frozen-JVP action evidence when source
  action cancellation exceeds its ordinary consistency tolerance.
- Alternatives rejected: increasing the polar-pass count blindly, accepting
  the `9.0377926031948164e-08` plateau, loosening `1e-8`, or relying on
  non-portable extended precision.
- Related: `_ArcEquilibratedKrylovBasis` and
  `_build_arc_equilibrated_krylov_basis` in
  `examples/fracture2d/nonlocal_engineering_grip_monolithic_arc.py`, plus the
  manufactured source-recomposition plateau regression.

## 2026-09-24 — Verify physical-space arc reorthogonalization

- Verification: the arc foundation/solver battery passes `26/26`; the affected
  reduced/arc/trust-region/hookstep/controller battery passes `121/121`; the
  changed arc module and regression compile with `py_compile`; and
  `git diff --check` passes.
- The plateau regression demonstrates that the prior three source-recomposed
  polar passes remain above `1e-8`, while exactly two physical-space MGS passes
  satisfy the unchanged final `D Q` gate. No physical probe or continuation was
  run after the replacement.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md` and the
  preceding physical-MGS decision.

## 2026-09-24 — Prioritize the nonlocal PINN over strict direct-path completion

- Decision: The full strict direct-nonlocal continuation through `u_bar=0.0075`
  is no longer a prerequisite to pursuing the user's original nonlocal fracture
  PINN objective. Preserve the independently checked nonlocal operators and
  strict-state evidence; use the already completed stabilized-warning direct
  curve only as a clearly labeled qualitative diagnostic, with the local FEM
  curve as an independent, model-different comparator.
- Why: The existing fixed-`omega=0.5`, 144-sweep, warning-mode run already
  reached exact `0.0075` in 75 increments, including 10 warning-accepted
  states. Its accepted-state replay binds provenance, but reports
  `field_equations_validated=false`, with up to `0.0291481` damage healing;
  reconstructing the same curve under the same settings would not produce an
  independent strict reference. The remaining arc/globalization work is
  specific to the direct solver rather than a prerequisite for the PINN.
- Alternatives rejected: requiring a full strict direct curve before PINN
  work, treating warning fields as exact training labels, or rerunning an
  identical 3-hour warning solve without changing the question being tested.
- Related: `docs/tasks/fracture2d-engineering-grip-baseline.md` and
  `engineering_grip_damped_warning_o5_u7p5e3_dumin6p25e6_s144` under the
  durable engineering-grip data directory.

## 2026-09-24 — Resume staggered adaptive-step investigation

- Decision: At the user's request, pause the coupled/monolithic solver effort
  and the proposed PINN handoff, and make staggered nonlocal adaptive
  load-step research the active task. Audit the saved onset of localized
  drift against Kristensen and Martínez-Pañeda (2020), Section 3.6, before
  adopting or altering its one-time damage-event trigger.
- Why: The strict staggered path reaches a post-peak accepted state while a
  small notch-tip cluster drifts on warning paths; the distinction between
  step resolution and nonlinear fixed-point convergence must be measured.
- Alternatives rejected: assuming the monolithic paper's speedup transfers to
  staggered iteration, silently loosening strict gates, or replacing the
  historical archive with a warning-mode result.
- Related: `docs/tasks/fracture2d-stagger-adaptive-step.md`,
  `docs/tasks/fracture2d-engineering-grip-baseline.md`.

## 2026-09-24 — Do not import the paper's damage threshold unchanged

- Decision: Hold early-cutback implementation pending an explicit trigger and
  retry choice. Preserve the accepted strict path as valid discrete-equation
  evidence while investigating its sharply localized displacement mode.
- Why: In the saved strict staggered path, tip particle 1869 already has
  `d=0.8178902` before the `0.0061 -> 0.0062` increment and reaches
  `d=0.9999172` with a `16.92414` displacement/load ratio afterward.
  The cited paper's `d_before<0.7` and `delta_d>=0.5` conjunction is false
  at that onset; in warning archives its first true load is `0.0064`, when
  the drift already exists. The existing failure-only cutback cannot reject
  a converged-but-localized state.
- Alternatives rejected: copying the published threshold as a purported drift
  remedy, changing damage/residual gates to force a rejection, or inventing a
  universal replacement threshold without a declared diagnostic contract.
- Related: `docs/tasks/fracture2d-stagger-adaptive-step.md` and the immutable
  engineering-grip strict/warning archives.

## 2026-09-24 — Isolate the approved one-time staggered damage-event retry

- Decision: Implement the user's explicitly approved, data-specific
  `d_before<0.9`, `delta_d>=0.1`, one-time `retry_fraction=0.1` policy as a
  separate strict in-memory controller. Keep a converged-but-discarded trial
  distinct from a failed nonlinear solve and from committed accepted steps.
- Why: The existing stabilized attempt audit treats a passing diagnostic as
  accepted, and its legacy archive/continuation validator cannot truthfully
  describe a passing trial discarded solely for load resolution. A separate
  result retains the event evidence without weakening historical replay.
- Alternatives rejected: mutating the legacy attempt into a failed solve,
  silently committing the large increment, modifying physical damage or
  convergence gates, or claiming the paper's monolithic convergence gains.
- Verification: focused manufactured controller tests `4/4`, changed-module
  compilation, and `git diff --check` pass. Physical load-path validation and
  event archive/replay integration remain open.
- Related: `examples/fracture2d/nonlocal_engineering_grip_adaptive.py`,
  `docs/tasks/fracture2d-stagger-adaptive-step.md`.

## 2026-09-24 — Separate adaptive-event evidence from legacy replay

- Decision: Keep the experimental early-rejection path and its prospective
  resumable evidence in a distinct schema. A discarded trial that has passed
  the strict nonlinear gates is recorded as a solved candidate, not forged
  into the legacy failed-attempt format. Resume only from an authenticated
  committed prefix with the earlier one-time event explicitly accounted for.
- Why: The bounded physical probe confirms that the large `0.0062` trial
  converges but has tip amplification `16.924`, while the strict `0.00611`
  retry has amplification `0.747`. Repeating this 15-minute pair after every
  process timeout would make a full onset study impractical; the legacy
  stabilized archive cannot encode passing-but-discarded attempts.
- Alternatives rejected: serializing live objects with pickle, treating the
  `0.0062` candidate as a failed solve, reusing warning-mode archives, or
  claiming one strict retry proves that all later steps are drift-free.
- Related: `docs/tasks/fracture2d-stagger-adaptive-step.md`,
  `examples/fracture2d/nonlocal_engineering_grip_adaptive.py`.

## 2026-09-24 — Hold one-time staggered event path at its strict stall

- Decision: Preserve the strictly accepted `0.006115` scratch prefix and
  refrain from promoting the one-time damage-event controller to a completed
  direct solver under the current `du_min=3.125e-6` and 144-sweep limits.
- Why: Owner-authenticated probes from the same strict checkpoint reject
  `0.00612` and its configured minimum-step retry `0.006118125` after all
  144 sweeps with the same seven nonconverged gates. The damage fixed-point
  residual fluctuates, rather than showing a reliably contracting tail.
- Alternatives rejected: committing either rejected candidate, equating a
  successful first `0.00611` retry with full path convergence, or silently
  shrinking `du_min`, changing acceleration, or allowing repeated event
  triggers without a new validated numerical contract.
- Related: `docs/tasks/fracture2d-stagger-adaptive-step.md` and the unchanged
  historical strict failure archive.

## 2026-09-24 — Label adaptive checkpoints as identity-only evidence

- Decision: Store discarded solved trials and accepted prefixes under a
  separate, exclusive no-pickle adaptive checkpoint schema with source/model/
  config/rule bindings. Report `field_equations_validated=false` until an
  independent owner replay of the field equations is implemented.
- Why: The existing stabilized archive cannot represent a strictly converged
  trial rejected solely for event-resolution reasons. Array digests and
  scalar attempt ordering alone cannot certify nonlinear equilibrium, even
  though they suffice to reject mismatched or altered checkpoint fields.
- Alternatives rejected: reusing the legacy archive schema, trusting a
  scratch checkpoint as a production restart, or using pickle to preserve
  live nonlinear diagnostics.
- Related: `examples/fracture2d/nonlocal_engineering_grip_adaptive_checkpoint.py`
  and `docs/tasks/fracture2d-stagger-adaptive-step.md`.

## 2026-09-24 — Separate energy-weighted error from strict strong residual

- Decision: Study a separately labeled scale-aware and energy-weighted
  approximate tolerance for the adaptive staggered path, while retaining the
  existing strict strong-residual/field-equation classification. Require a
  load-refinement and FEM physical-comparison audit before treating any
  approximate candidate as a useful crack-growth state.
- Why: The rejected minimum-step candidate has an independently reproduced
  maximum AT2 residual `0.967899` (`8.624e-4` on its six-term scale), almost
  entirely at a volume-`5.402e-6` notch-tip particle. The exact frozen-history
  quadratic energy gap is `1.92981e-9`, but an owner mechanics/Amor/damage
  follow-up increases the damage-map correction from `0.000641849` to
  `0.000662110`. A tiny integrated energy error does not make the nonlinear
  staggered map contract or prove the crack path.
- Alternatives rejected: using the FEM post-peak warning curve as an exact
  per-particle target, equating Abaqus correction tolerance with nonlocal
  strong-residual normalization, or silently accepting the scratch state under
  the historical strict schema.
- Related: `docs/tasks/fracture2d-stagger-adaptive-step.md`,
  `/tmp/opencode/adaptive_stagger_min_target_candidate_v1.npz`.

## 2026-09-24 — Test a dual residual and FEM-envelope criterion separately

- Decision: Investigate an explicitly parameterized *diagnostic* acceptance
  audit with independently evaluated AT2 residual and fixed-history energy
  gap, correction relative to the actual load-step field increment, strict
  mechanics/history/irreversibility checks, and an empirical FEM/nonlocal
  displacement-amplification envelope. Continue labeling any state that fails
  the historical strict gates as approximate; do not edit the strict schema.
- Why: The authenticated minimum-step candidate has a tiny frozen-history
  energy gap (`1.93e-9`) but a locally amplifying staggered map. The warning
  curve at `0.0062` has a smaller six-term-normalized damage residual than
  the smooth rejected candidate (`4.9e-8` versus `8.6e-4`), yet its tip
  displacement amplification is `16.86` versus `0.79`. Current-history
  normalization alone would accept the known outlier as history explodes.
  Local FEM snapshots stay below `1.064` maximum displacement/load through
  the reference endpoint, and the smooth nonlocal candidate is `1.059`.
- Alternatives rejected: a single relaxed relative AT2 residual tolerance,
  declaring a small integrated energy gap sufficient for convergence, or
  replacing the historical strict pass flags with a FEM curve-distance gate.
- Related: `docs/tasks/fracture2d-stagger-adaptive-step.md` and the immutable
  FEM, strict, and warning archives.

## 2026-09-24 — Reject an eight-percent correction rule as a complete remedy

- Decision: Keep the `0.08` correction-fraction and `0.005` scaled-residual
  envelope diagnostic-only. Test a single half-minimum load increment from
  the last strictly accepted `0.006115` checkpoint with all original
  equation/correction gates unchanged before selecting any continuation rule.
- Why: The frozen-history strong residual of the rejected candidate is
  `0.9679`, but its integral energy gap is only `1.93e-9`; the normalized
  residual and FEM displacement envelope alone cannot distinguish the
  severely drifting warning state. Replaying one step from a source-bound
  approximate candidate accepted under an eight-percent trial criterion
  reaches `0.00612125` only as an unaccepted candidate after 144 sweeps;
  its correction/increment ratio rises to `0.718`, scaled AT2 residual to
  `0.00255`, and it still fails all seven historical coupled gates. The
  approximate criterion does not remain stable even for one step.
- Alternatives rejected: promoting the eight-percent threshold, broadening
  it again to pass the next failed state, or accepting a low energy gap as
  proof of a stable load-controlled crack path.
- Related: `docs/tasks/fracture2d-stagger-adaptive-step.md`,
  `/tmp/opencode/adaptive_stagger_tolerance_successor_v1.npz`.

## 2026-09-25 — Do not promote a one-percent tolerance without stable path evidence

- Decision: Keep the FEM-inspired one-percent correction study strictly
  diagnostic and retain all original strong-equation/strict-pass semantics.
  Prioritize a branch/globalization diagnosis over any further tolerance
  widening.
- Why: A half-minimum smooth candidate at `0.0061165625` satisfies an
  independent experimental audit with `0.675%` maximum correction/increment,
  small scaled residual/energy gap, and no particle outlier. Nonetheless its
  one-step physical continuation to `0.006118125` fails after 144 sweeps with
  `328.8%` correction/increment while hard mechanics/history gates pass. An
  apparently reasonable threshold at one step does not define a convergent
  solver path.
- Alternatives rejected: accepting the seed as a historically strict root,
  increasing a tolerance above `3.288` merely to mark the next step passed,
  or using FEM's warning post-peak curve as an exact target.
- Related: `docs/tasks/fracture2d-stagger-adaptive-step.md`,
  `/tmp/opencode/adaptive_stagger_onepercent_successor_v1.npz`.

## 2026-09-25 — Keep conditional tip crossing separate from full equilibrium

- Decision: Treat the source-bound tip-only fixed-point map scan as a
  diagnostic of the near-peak branch. Retain original strict stopping gates
  and do not use a conditional scalar zero to authorize whole-field acceptance.
- Why: Fixing all other material damage to the rejected smooth candidate,
  the tip map error is positive near `d_tip=0.85`, then changes sign between
  `0.999` and `0.9999`, where tip displacement/load exceeds `9`. Neighbor
  particle 1873 retains a `0.03084` map defect at the conditional crossing.
  This supports a fold/globalization concern but does not prove that the
  complete nonlocal equations lack another acceptable root.
- Alternatives rejected: promoting the conditional crossing as a strict
  solution, broadening tolerance to accept a `329%` correction, or declaring
  smooth-branch root nonexistence without full-system evidence.
- Related: `docs/tasks/fracture2d-stagger-adaptive-step.md`,
  `/tmp/opencode/fracture2d_tip_map_scan.py`.

## 2026-09-25 — Keep algebraic and physical crack-growth validation separate

- Decision: Continue reporting the high-damage-seeded stagger map as a
  distinct read-only branch diagnostic, never as an accepted adaptive step.
  Further numerical acceptance must both satisfy the original field
  equations and be compared with independently trustworthy FEM/material
  field evidence; the scalar threshold study remains isolated.
- Why: Borrowing a historical high-damage guess at the failed target makes
  the alternating damage map contract to `9.41e-9` by sweep 258, while
  the near-root tip displacement/load remains `14.77` and history at the tip
  reaches about `903640`. FEM snapshots have maximum displacement/load
  below `1.064`, and the smooth one-percent candidate's successor diverges
  from the proposed correction threshold. A root of the discrete numerical
  equations need not meet the requested smooth physically comparable branch.
- Alternatives rejected: calling a different-path, unvalidated initial guess
  a valid continuation; allowing 258 sweeps and ignoring field outliers; or
  marking the smooth candidate converged by further widening tolerances.
- Related: `docs/tasks/fracture2d-stagger-adaptive-step.md`,
  `/tmp/opencode/fracture2d_high_branch_stagger_probe.py`.

## 2026-09-25 — Define the user's two-outlier allowance without deleting physics

- Decision: Assess an experimentally converged high-damage branch by
  comparing all 3395 original material particles against FEM, declaring
  outliers through an explicit displacement/load envelope before comparing
  the remaining field. Exclusion is **only** from a reported physical
  comparison; original BA mechanics, history, AT2 residual, quadrature,
  conservation, and strict gates still include all particles.
- Why: The tip-only high-damage initial guess at `0.006118125` solves the
  original map to represented precision, and the FEM/nonlocal clouds have
  identical node ordering. Exactly two tip particles exceed the `1.08`
  envelope while the other nodes' normalized displacement weighted RMS
  differs by `1.72%` from the stage-near FEM `0.0058` snapshot. The complete
  FEM archive maximum is `1.06670`; prior `<1.064` comparisons included
  only snapshots at/after `0.005`. This correction does not retroactively
  make FEM warning snapshots or the nonlocal branch a validated full path.
- Alternatives rejected: deleting outlier particles or their contributions,
  choosing outlier indices to minimize FEM error, calling the different-load
  seed an accepted zero-start step, and using an FEM warning curve as an
  exact pointwise constitutive target.
- Related: `docs/tasks/fracture2d-stagger-adaptive-step.md`,
  `/tmp/opencode/adaptive_stagger_tip_only_seed_v1.npz`.

## 2026-09-25 — Reject isolated front corrections as a global solver

- Decision: Keep the 2- and 32-node secant/Newton crack-front scans
  diagnostic-only and do not replace the old staggered globalization with a
  local correction that is not a descent direction for the complete map.
- Why: From the strict post-switch `0.006218125` state the next three
  targets exhaust all 144 sweeps. A two-node owner finite-difference
  correction reduces residuals on those nodes but shifts a larger defect to
  node 2034 and raises both the global maximum and volume-weighted RMS.
  A 32-node projected Jacobian has a small singular value `0.00660` and a
  near-zero positive eigenvalue `0.00823`; its correction increases global
  error even at quarter step and larger steps violate irreversibility.
- Alternatives rejected: accepting a `1%`-eligible candidate that fails at
  the next load with `12.6%` fixed-point correction/increment, silently
  increasing a tolerance again, or claiming the restricted Jacobian proves
  full-system root nonexistence.
- Related: `docs/tasks/fracture2d-stagger-adaptive-step.md`,
  `/tmp/opencode/fracture2d_front_secant_probe.py`.

## 2026-09-25 — Pursue same-target self-seeded stagger restart

- Decision: For a separate experimental path, when a complete but
  nonconverged strict alternating attempt reaches its sweep ceiling, retain
  the final *uncommitted* damage iterate as an explicit, auditable initial
  guess for further fixed-omega alternation at the **same** load with the
  **same** immutable committed history. Keep original mechanics, Amor,
  AT2 equations and strict final gates; give the restart its own provenance
  rather than writing it as an ordinary legacy attempt.
- Why: A source-bound experiment at `0.006243125` starts from the legacy
  failed candidate, continues from its own final iterate (not a warning
  archive), and reaches a full damage-map defect `8.68e-9` at cumulative
  sweep 1030. Its current-history strong AT2 residual is `7.12e-6` versus
  old limit `0.03916`; displacement/history/reaction last corrections also
  pass. Independent owner recomputation gives exact trial Amor history,
  positive damage monotonicity, free mechanics residual `8.18e-8`, and
  force/moment defects near roundoff. Exactly two predefined tip nodes are
  displacement outliers; the other 3393 nodes' normalized displacement
  differs by volume-weighted RMS `2.05%` from the FEM `0.0059` warning
  snapshot, reaction `592.787` versus `589.764`. A separate warning-derived
  seed found a close physical state faster, but is not required and must not
  be a production solver dependency.
- Alternatives rejected: accepting the original 144-sweep candidate under
  a widened `1%` tolerance that fails its next step, using warning fields as
  accepted labels, discarding the two outlier particles from the equations,
  or applying a restricted Newton direction that increases global residual.
- Relation: experimental recovery of the approved adaptive stagger goal,
  `docs/tasks/fracture2d-stagger-adaptive-step.md` and
  `/tmp/opencode/adaptive_stagger_self_seed_extended_picard_v1.npz`.

## 2026-09-25 — Keep post-restart cutbacks strict and bounded

- Decision: Preserve the successful same-target restart as a local
  experimental method, but reject the proposition that one universally
  sufficient 1000-sweep recovery budget or a one-percent correction
  threshold completes the displacement-controlled crack path. Continue
  retaining only original-gate accepted states and explicit cutback/failure
  evidence.
- Why: From the strictly accepted `0.006243125` scratch state the next
  loads `0.006268125` and `0.006293125` pass in 32/30 sweeps, whereas
  `0.006318125` and `0.006305625` each exhaust 144 Anderson plus 1000
  same-load recovery sweeps. A smaller increment to `0.006299375` passes
  in 37 sweeps, still with only the same two predefined FEM comparison
  outliers. This is repeated near-peak step sensitivity, not permission to
  relabel failed solutions as converged or to silently enlarge the budget.
- Alternatives rejected: warning-committing either large-step failure,
  inferring post-peak path completion from one recovered target, or rerunning
  identical costly targets without a changed numerical question.
- Related: `docs/tasks/fracture2d-stagger-adaptive-step.md` and the
  source-bound `/tmp/opencode/adaptive_restarted_branch_step01_17e4e03461a9.npz`.

## 2026-09-25 — Investigate user-selected AT2-energy outer continuation

- Decision: At the user's explicit selection, investigate the *existing*
  surface-only discrete nonlocal AT2 fracture energy as an outer path
  coordinate around the repeated displacement-controlled stalls. Keep
  original fixed-damage mechanics, committed-history Amor, and fixed-history
  damage subproblems as the inner staggered map. First evaluate the scalar
  energy response at frozen damage and different nonnegative absolute loads;
  do not assume that every positive energy target has a bracketed load root.
- Why: Source-bound strict experimental states from `0.006115` to
  `0.006299375` have strictly increasing surface energy, but the required
  increment varies strongly (`0.02412`, `0.04118`, `0.12553`, `0.02161`,
  `0.02752`, `0.01546`), while two larger displacement targets exhaust
  144 Anderson plus 1000 same-load recovery sweeps. This calls for an
  explicit fracture-resolution coordinate rather than declaring the
  next large load step converged by a widened tolerance.
- Alternatives rejected: silently implementing an arc-length constraint
  from local FEM, equating elementwise irreversible damage with monotonic
  nonlocal surface energy without checking, or treating the prior coupled
  arc module as an already-validated staggered production solver.
- Related: `docs/tasks/fracture2d-stagger-adaptive-step.md`,
  `examples/fracture2d/nonlocal_engineering_grip_monolithic_arc.py` for
  the independently derived surface-only energy, and the preserved
  experimental strict-prefix snapshots.

## 2026-09-25 — Audit damaged-operator weakness before tuning thresholds

- Decision: Treat the user's repeated crack-tip displacement outliers as a
  possible damaged BA operator/discretisation weakness; run only
  source-bound frozen-field diagnostics before proposing any changed
  kernel, residual stiffness, BA support, or degradation law. Keep all
  particles in mechanics and damage equations.
- Why: Nodes 1869/1873 have ordinary target-inclusive support sizes (8–15)
  and shape conditions (up to about 6.2), not missing-bond/rank evidence.
  At the identical nodal damage field, their damaged BA local block
  minimum eigenvalues are `0.472/3.142`, versus exact-CG1-FEM blocks
  `1643/1069`; two-node observed-displacement virtual energy is `0.00628`
  for BA versus `32.13` for FEM under matching plane-strain material data.
  The intact two-node mode is already about 300 times softer in BA, and
  the damaged gap increases to about 5100. A strong-form tolerance change
  cannot restore this lost high-frequency mechanical stiffness.
- Alternatives rejected: deleting/reweighting one or two outlier particles
  inside the PDE, claiming source support rank is defective without evidence,
  or silently increasing `k_res` to force cosmetic FEM agreement.
- Related: `docs/tasks/fracture2d-stagger-adaptive-step.md`,
  `/tmp/opencode/fracture2d_damaged_operator_outlier_audit.py` and
  `/tmp/opencode/fracture2d_outlier_mode_energy.py`.

## 2026-09-25 — Residual-stiffness floor does not cure the local mode

- Decision: Do not tune `k_res` as an outlier cure on the existing BA
  mechanics. Defer any change to the declared variational operator or
  degradation integration until its formulation is explicitly selected and
  independently derived and reviewed under `docs/NUMERICAL_METHODS.md`.
- Why: At identical frozen damage, 1000-fold increase from `k_res=1e-8` to
  `1e-5` only changes node 1869 displacement/load `21.84 -> 21.12`, node
  1873 `6.67 -> 6.45`, and BA minimum local block `0.472 -> 0.502` at
  node 1869. Exact BA edge-energy decomposition of the two-point mismatch
  attributes `12.351` and `3.469` of `15.857` *intact* virtual energy to
  centers 1869/1873 (about 99.8%). At damage `0.999955/0.999637` their
  contributions collapse to `1.49e-7/4.91e-7`; the only substantive
  remaining virtual energy comes from less damaged neighboring centers.
  This is the prescribed `W_i=V^m_i g(d_i)+V^g_i g(0)` quadrature acting on
  BA gradients, not evidence of a missing edge or a linear-solver failure.
- Alternatives rejected: silently raising the residual floor, forced
  complete damage as an *unverified* stiffness cure, or removing these
  points from the force/damage balance. A genuinely new stabilization or
  degradation quadrature changes the declared fracture equations and
  requires a separate formulation decision.
- Related: `docs/tasks/fracture2d-stagger-adaptive-step.md`,
  `/tmp/opencode/fracture2d_outlier_kres_frozen_probe.py`,
  `/tmp/opencode/fracture2d_outlier_mode_energy.py`.

## 2026-09-25 — Require PINN-transferable equations for a direct reference

- Decision: Any experimental drift remedy must provide the **same explicit
  fixed-shape nonlocal graph, energy, mechanics residual, material history,
  and damage residual** to both the direct solver and the later PINN. First
  perform independently checked, source-bound frozen-field research; do not
  relabel it as a converged new coupled fracture solution or install an
  unreviewed physical model.
- Why: The user clarified that the direct nonlocal solver's primary purpose
  is the nonlocal PINN reference. Opaque node deletion, imposed damage
  overrides, or solver-only heuristic displacement clamps cannot meet this
  objective unless explicitly made part of one validated common formulation.
  A strictly positive, row-normalized graph kernel over the **existing**
  horizon is a provisional PINN-transferable alternative for spatially
  distributing source-center degradation; its corresponding *adjoint*
  positive-energy history driver and rollback remain to be derived and
  validated before coupled implementation.
- Alternatives rejected: editing the original BA formulation silently or
  copying the original pointwise history equation after changing which
  damage variables degrade a mechanical center's energy.
- Related: `docs/tasks/fracture2d-stagger-adaptive-step.md`,
  `docs/tasks/fracture2d-direct-nonlocal.md`,
  `/tmp/opencode/fracture2d_pinn_transferable_kernel_probe.py`.

## 2026-09-26 — Reject horizon-scale damage averaging as a FEM-like PINN reference

- Decision: Keep the separately authorized, symmetric volume-balanced
  horizon-kernel damage coupling as an experimental **algebraically strict
  counterexample**. Do not promote its full zero-to-endpoint path to a
  physically validated FEM-comparable nonlocal PINN reference merely because
  it eliminates crack-tip displacement outliers and passes the original
  numerical acceptance gates.
- Why: The candidate's fixed graph preserves constants under its forward
  and volume-adjoint actions within `7.4e-14`; five manufactured checks pass.
  Seventy-five positive `1e-4` steps starting at exact zero strictly reach
  exact `u_bar=0.0075` with no FEM-envelope displacement outlier. Each
  retained field was rechecked against candidate mechanics, adjoint-Amor,
  AT2, conservation and irreversibility, with cumulative identity-only
  scratch checkpoints. Yet same-load FEM at the endpoint reports reaction
  `3.977` versus candidate `729.195` (factor `183.34`), max damage
  `1.00454` versus `0.76607`, and nonlocal AT2 surface energy about
  `0.440` versus FEM `2.084`. The candidate never reaches a force peak or
  developing fracture branch in the tested span; its `u_bar=0.0062`
  reaction `669.85` differs from FEM `7.014` (post-peak working state).
- Alternatives rejected: claiming that strict residuals or elimination of
  one/two outlier particles alone validates a reference solution; altering
  original accepted-state or warning archive labels; tuning a tolerance to
  make the two models' distinct physical curves appear converged.
- Related: `docs/tasks/fracture2d-stagger-adaptive-step.md`,
  `examples/fracture2d/nonlocal_engineering_grip_kernel_damage.py`,
  `/tmp/opencode/fracture2d_kernel_damage_continue_probe.py`, and
  `/tmp/opencode/fracture2d_kernel_damage_fem_comparison.py`.

## 2026-09-26 — Investigate separate P1-consistent damage quadrature candidate

- Decision: At the user's explicit choice, investigate a **separate**
  mesh-derived P1 consistent-mass spatial degradation quadrature using the
  *existing* reference triangulation solely to produce fixed positive
  nodal-to-source weights. Retain BA displacement kinematics, nonlocal AT2
  damage graph and the mathematically paired volume-adjoint Amor history.
  Demand full strict zero-start path, field replay, FEM response and
  eventual mesh/horizon sensitivity before deciding if it can serve as a
  common direct/PINN nonlocal reference.
- Why: The horizon-sized averaging delayed fracture past the load endpoint.
  P1 consistent mass defines an independently known no-fit quadrature
  `M_ii=sum_T A_T/6`, `M_ij=sum_{T: i,j in T}A_T/12` for off-diagonals,
  `W=diag(V)^{-1}M` with `V_i=sum_{T:i in T}A_T/3`. It preserves constant
  primal and volume-adjoint fields while distributing tip degradation only
  over mesh-neighbor cells. In a *frozen old-model high-damage field* its
  mechanics solve reduces the two large point displacements to one
  envelope outlier (tip 1869 amplification `1.306`) and yields reaction
  `611.11` at load `0.006243125`; this is not a coupled new-model root.
- Alternatives rejected: promoting the broad-horizon nonfracturing branch,
  silently replacing the original model, or declaring a frozen mechanics
  comparison sufficient proof of the new candidate's path or PINN validity.
- Related: `docs/tasks/fracture2d-stagger-adaptive-step.md`,
  `/tmp/opencode/fracture2d_local_mass_quadrature_frozen_probe.py`.

## 2026-09-26 — P1 mass averaging does not eliminate the crack-tip weak mode

- Decision: Do not promote the isolated P1-mass degradation model to the
  physically validated nonlocal PINN reference. Freeze its experimental
  results for negative-control and model-design analysis. Revisit the
  correspondence high-frequency energy/kinematics rather than widening the
  physical outlier allowance or automatically averaging degradation over
  ever more neighbors.
- Why: Seven manufactured energy/adjoint/zero-step checks pass. The separately
  bound P1 candidate follows a zero-start, terminal-equation-checked path
  through load `0.0069`; it strictly passes a `0.00695` half increment in
  150 sweeps and a same-model `0.007` continuation in 172 sweeps after
  bounded extra iterations. However onset is late: FEM peaks near `0.0057`,
  while the P1 reaction remains `683.825` at `0.0068` and `675.319` at
  `0.007`. At `0.00705` and `0.0071` the first two FEM-envelope point
  outliers return; at `0.00715` a third node (1873) exceeds the allowance,
  and tip 1869 displacement/load amplifies to `16.38` even though the
  candidate's terminal mechanics/history/AT2 gates pass. The shorter
  averaging support reduces the broad-kernel physical lag but cannot
  eliminate the intrinsic near-zero local mechanical mode at full tip damage.
- Alternatives rejected: claiming the first frozen-field result proves
  pathwise success, labeling a three-outlier terminal candidate physically
  acceptable, forcing `d=1` as a stiffness remedy, or treating the two
  candidate models' different physics as numerical tolerances.
- Related: `docs/tasks/fracture2d-stagger-adaptive-step.md`,
  `examples/fracture2d/nonlocal_engineering_grip_p1_mass_damage.py`,
  `/tmp/opencode/fracture2d_p1_mass_after_cutback.py`.

## 2026-09-26 — Qualify the near-zero mode as damage-field dependent

- Decision: Evaluate candidate displacement drift with the **same nodal
  damage** in the local FEM virtual-energy oracle before attributing it
  solely to a BA implementation or stiffness bug. Continue to reject the
  P1 path for the three-point physical outlier and reaction mismatch.
- Why: At the new `0.00715` root, P1 tip/neighbor damage is about
  `0.9992–0.9999`. The observed three-node displacement mismatch has BA
  energy `0.00737` and local FEM energy `0.03889` under this *same* damage;
  FEM is also weak locally, unlike the former old-model two-node test in
  which FEM energy was `32.13`. This shifts the primary question to why
  damage localization/propagation differs so strongly from the FEM path,
  in addition to how BA treats high-frequency modes.
- Alternatives rejected: deducing a universal BA-only stiffness defect
  from local diagonal blocks or treating the P1 three-point mode as proof
  that any physically faithful stagger solver must delete nodes.
- Related: `docs/tasks/fracture2d-stagger-adaptive-step.md`,
  `/tmp/opencode/fracture2d_p1_mass_tip_mode_energy.py`.

## 2026-09-26 — Do not assume fluctuation-only BA stabilization cures tip drift

- Decision: Do not implement the parameter-free variance-only split of the
  existing BA energy as a supposed point-drift cure on the current cloud.
  Require a more effective and independently reviewed modification of
  high-frequency mechanics if this avenue is pursued.
- Why: Quadratic plane-strain energy exactly decomposes into energy of the
  `eta`-averaged BA strain plus nonnegative bond-associated variance.
  Replacing `g(d_i)` by the normalized-neighbor `gbar_i` *only on the
  variance* preserves affine energy and barely affects the original
  pre-event displacement's elastic energy (`+0.052%` with P1 support,
  `+0.156%` with horizon support). However the observed two-tip-node
  virtual mode has only about `2.48%` intact energy in that variance.
  Its actual damaged BA virtual energy rises only `0.00628 -> 0.00984`
  (P1) or `0.04002` (horizon); same-damage FEM is `32.126` in the earlier
  state. Thus the seemingly ideal affine-preserving split chiefly misses
  the near-zero source-averaged mode and is not justified as a cure by
  its exact variance identity alone.
- Alternatives rejected: an uncalibrated large penalty multiplier to force
  FEM agreement, or building another full zero-start path before confirming
  that the proposed term actually stiffens the observed weak mode.
- Related: `docs/tasks/fracture2d-stagger-adaptive-step.md`,
  `/tmp/opencode/fracture2d_ba_variance_split_frozen_probe.py`.

## 2026-09-26 — Keep same-mesh horizon comparisons source-bound and separate from acceptance

- Decision: Under the user's selected original-equation mesh/horizon
  sensitivity route, regenerate diagnostic `delta=0.040/0.032` graphs only
  after byte-equivalent reproduction of the original `delta=0.048` graph
  and exact equality of material vertices, triangles, volumes and boundary
  tags. Treat a re-solve at a frozen **other-horizon** damage field as an
  operator sensitivity, never an accepted continuation at that horizon.
  Advance a new-horizon path only from its own zero state with unchanged
  strict BA/Amor/AT2 equations and gates.
- Why: On the same 3395-node material mesh, shrinking horizon
  `0.048 -> 0.040 -> 0.032` raises the damaged 1869 local block minimum
  `0.472 -> 0.999 -> 2.803` and the old two-node virtual energy
  `0.00628 -> 0.01914 -> 0.04111`. The *frozen-damage* tip
  displacement/load falls `21.84 -> 10.76 -> 6.74` but remains well above
  the FEM-informed `1.08` envelope. The distinct `delta=0.032` zero-start
  strict original-model path progresses through `0.0058`, then repeated
  cutbacks/144-sweep attempts stall near `0.00583`; shrinking horizon
  changes the onset but does not yet cure continuation/drift.
- Alternatives rejected: interpreting a cross-horizon fixed-damage field
  as a coupled solution; selecting a single horizon solely to fit FEM;
  labeling a failed 144-sweep target as converged because its terminal
  residual decreases with load cutbacks.
- Related: `docs/tasks/fracture2d-stagger-adaptive-step.md`,
  `/tmp/opencode/fracture2d_horizon_same_mesh_graphs.py`,
  `/tmp/opencode/fracture2d_original_horizon_zero_path.py`.

## 2026-09-26 — Same-target self-seeded restart resolves the first d032 stall locally

- Decision: Preserve the strict `delta=0.032` `0.005834375` same-load
  recovery as an **experimental branch step**, bound to its actual last
  committed `0.00583125` state and explicit original-failure seed. Run a
  held-out next-step test before declaring a repeatable adaptive method.
- Why: The authoritative 144-sweep Anderson attempt rejected all seven
  coupled gates; its last damage iterate alone seeded a fixed-omega `0.5`
  continuation at the *same* target and *same* immutable history. After
  263 additional sweeps it passed every original strict gate. Owner
  recomputation gives AT2 strong residual `1.1714e-5`, map error
  `5.3766e-9`, free-force infinity residual `2.3952e-7`, reaction
  `621.179`, one displacement-envelope outlier (particle 1869), and
  exact committed-history Amor. The original failed candidate was not
  committed and the previous 256-extra-sweep run remains a separate
  honestly failed counterexample.
- Alternatives rejected: loosening the history/fixed-point gates, using an
  external warning field, erasing failure provenance, or claiming that one
  recovered root proves full-path continuation or absence of later drift.
- Related: `docs/tasks/fracture2d-stagger-adaptive-step.md`,
  `examples/fracture2d/nonlocal_engineering_grip_stagger_restart.py`,
  `/tmp/opencode/original_BA_d032_self_restart_u5p834375_v2.npz`.

## 2026-09-26 — Require a held-out step after the d032 own-seed root

- Decision: Treat the `0.005834375` source-bound own-failure-seeded root as
  usable **local continuation evidence** only after an independently
  replayed, strictly accepted next load. Preserve its experimental
  upstream parent chain; do not turn the scratch files into a certified
  zero-start production archive without exact attempt/rollback replay.
- Why: After owner revalidation of the parent state, the next original
  equations/strict gates at `0.0058375` pass in **17 ordinary Anderson
  sweeps**, without another restart. Owner recomputation gives free-force
  infinity residual `2.15e-7`, current-history AT2 strong residual
  `2.16e-6`, damage-map error `1.83e-9`, reaction `621.251`, and still
  exactly one diagnosed outlier (node 1869, amplification `4.262`). This
  gives one genuine held-out successor, but neither proves the later
  fracture-front path nor validates a load-increment growth rule.
- Alternatives rejected: equating the original 144-sweep failure with
  accepted convergence, relying on the previous external warning solution
  as a numerical seed, or labeling a single next-step pass a full
  PINN-reference fracture curve.
- Related: `docs/tasks/fracture2d-stagger-adaptive-step.md`,
  `/tmp/opencode/original_BA_d032_successor_from_own_restart_u5p8375_v1.npz`.

## 2026-09-26 — Separate strict field convergence from three-point physical rejection at d032

- Decision: Keep the d032 `0.0061125` roots as independently replayed
  **algebraic** evidence only; do not extend a physical comparison prefix
  past the last strict two-or-fewer-outlier state at `0.006109375`.
- Why: After strict own-seed restart at `0.005834375`, a next step and
  bounded `2.5e-5` increments reach `0.0060875` with one or two diagnosed
  outliers; a half step to `0.0061`, then another half to `0.00610625`
  and minimum `3.125e-6` step to `0.006109375` strictly pass. At target
  `0.0061125`, 144 Anderson plus 256 fixed-omega sweeps fail, whereas
  144+296 or 144+299 pass the original equations from **two distinct**
  strict smooth prefixes. Both converged roots exceed the user's
  observational budget with outliers `1868,1869,1873` and tip/load
  amplification about `10.09`; changing only the load cutback does not
  avoid the third anomalous point at this target.
- Alternatives rejected: treating successful strict algebraic gates as
  equivalent to FEM-like physical acceptance, deleting an outlier from
  equilibrium while still claiming unchanged BA/AT2 equations, or naming
  an experimental larger-increment path a production adaptive schedule.
- Related: `docs/tasks/fracture2d-stagger-adaptive-step.md`,
  `/tmp/opencode/fracture2d_d032_branch_growth_probe.py`.

## 2026-09-26 — Verify actual Gmsh h-refinement before using a mesh comparison

- Decision: Generate a separate mesh from an explicitly versioned
  `lc_near=0.002` geometry copy, and verify physical group cell counts and
  actual tip-region spacing before treating it as a refinement. Preserve
  the original `.geo`, FEM mesh/archive, and source cloud untouched.
- Why: `gmsh ... -setnumber lc_near 0.002` applied to the original `.geo`
  produced **exactly the same** 3395 material nodes/6284 triangles and
  `119/20/119` notch upper/tip/lower segments as the `0.004` mesh; the
  command-line override was ineffective for that `DefineNumber` invocation.
  The isolated `.002`-default `.geo` produced 3711/6634 material
  nodes/triangles and `238/38/238` tagged notch segments, with lower
  median near-tip nodal volume and element edge length. Rebuilding the
  material cloud at both diagnostic horizons now yields separately
  fingerprinted graphs (50128 edges at `.032`, 104678 at `.048`). This is
  a *near-tip mesh* diagnostic, not an established global h-convergence
  sequence or a valid source for the old node-index-specific crack state.
- Alternatives rejected: silently using the first, unchanged Gmsh output
  as a refined study, interpolating an old-horizon accepted state across
  the removed slot without a validated physical map, or running JAX GPU
  simultaneously with FEniCS mesh extraction.
- Related: `docs/tasks/fracture2d-stagger-adaptive-step.md`,
  `/tmp/opencode/fracture2d_tip_refined_lc002.geo`,
  `/tmp/opencode/fracture2d_refined_clouds.py`.

## 2026-09-26 — Separate PINN operator validation from fracture-path continuation

- Decision: At the user's request, re-prioritize a read-only root-cause
  diagnosis of the original nonlocal mechanics/damage/history operators
  and their discrete convergence, before more same-control crack-growth
  cutbacks. Test the direct/PINN common residual first on intact,
  fixed-damage, fixed-history, and strictly retained pre-instability
  states; treat full post-peak continuation as a separate validation
  milestone. No model equation or accepted-state status is changed.
- Why: independent fixed-subproblem checks pass, yet damaged BA
  source-center weights strongly soften some short-wavelength tip modes
  and coupled damage/history feedback approaches a noncontracting mode.
  A refined cloud passes one load at which the coarse cloud fails but
  rejects the next; successful full-load averaging suppresses fracture.
  Another solver-control scan cannot distinguish operator/discretization
  error from coupled branch instability.
- Alternatives rejected: further untargeted tolerance/damping/iteration
  tuning, equating a strictly solved but physically mismatched branch
  with operator validation, or assuming the post-peak FEM warning curve
  provides an exact pointwise target.
- Related: `docs/tasks/fracture2d-stagger-adaptive-step.md`,
  `docs/NUMERICAL_METHODS.md`.

## 2026-09-26 — Prioritize physical FEM similarity and permit qualified warnings

- Decision: Apply the user's revised acceptance priorities to the direct
  nonlocal reference: FEM-like solution-field and reaction-curve shapes with
  controlled peak-force discrepancy matter more than perfect agreement or
  historically strict convergence. Honest warning-mode candidates may be
  assessed physically. Diagnostic exclusion of drifting points from a field
  comparison is allowed; actual deletion from the equations is a separate
  permissible *research model* only if its recomputed fields and reaction
  meet the physical objectives and the PINN can use the same model.
- Why: local FEM also has warning snapshots; the validated old-model peak
  magnitudes are close, but the completed old warning path has a strongly
  discrepant final force. Reporting exclusion cannot change reaction, and
  deleting particles changes BA supports, forces, AT2 and continuation.
  Before judging success, an explicit peak-error tolerance and warning
  acceptance policy are needed; no new threshold is selected here.
- Alternatives rejected: requiring historical strict success for every
  snapshot, calling outlier-masked plots a newly solved crack path, or
  deleting nodes silently while labeling results original BA/Amor/AT2.
- Related: `docs/tasks/fracture2d-stagger-adaptive-step.md`,
  `docs/PROGRESS.md`.

## 2026-09-26 — Observational masking cannot validate the old warning curve

- Decision: Close observational-only outlier removal as a remedy for the
  user's full FEM-like reaction-curve goal. Retain the mask as a diagnostic
  field comparison, and require a separately selected, derived, and
  **recomputed** direct/PINN model for any actual material deactivation.
- Why: A hash-bound, exactly matched 3395-node read-only audit finds peak
  forces `641.340/642.486` nonlocal/FEM (only `0.1783%` discrepancy), but
  post-peak forces `631.672/7.014` at `0.0062` and `69.154/3.977` at
  `0.0075`. The prior `1.08` displacement envelope flags two points at
  onset and 31 along the crack at the endpoint. Excluding all 31 from
  reporting improves final normalized displacement RMS `0.06784 ->
  0.03595` but leaves the original force, severe load-shift, inherited
  warning path and damage healing `0.02915` untouched. Early crack-stage
  matching does improve masked displacement RMS to `0.01112`, so the
  physical failure is in later propagation/complete-opening response.
- Alternatives rejected: certifying the warning path from its peak
  magnitude alone, retuning the reporting mask to reduce a force computed
  elsewhere, or silently removing point/edge contributions from the old
  original-model force and calling it unchanged BA/Amor/AT2.
- Related: `docs/tasks/fracture2d-stagger-adaptive-step.md`,
  `/tmp/opencode/fracture2d_warning_fem_mask_audit.py`.

## 2026-09-27 — Return to original staggered nonlocal model

- Decision: At the user's explicit correction, do not investigate node
  deletion or another fracture-to-void model. Preserve the original
  BA/Amor/AT2 nonlocal operator, material particles, fixed graph and
  history semantics; investigate adaptive staggered load stepping with
  truthfully labeled, bounded warning acceptance as a *numerical* path.
  The peak force and approximate onset need no new model fitting; the
  remaining objectives are improved end reaction, post-peak response, and
  fewer drifting material points.
- Why: the already completed warning archive has an accurately sized
  peak, and read-only masked phase-aligned early-crack fields are close
  to FEM. The one-time strict event cutback temporarily removes visible
  tip drift but stalls just after; indiscriminate unbounded warnings
  accumulate up to `0.02915` damage healing and end at `69.15` versus
  FEM `3.98`. Neither direct masking nor a single cutback proves the
  original-equation target. A distinct coupled history path may depend
  on the load schedule, but its improvement has **not** been established.
- Alternatives rejected: the earlier proposed node-deletion experiment,
  reclassifying old warning fields as strict, blind tolerance expansion,
  or assuming load-step adaptation alone must recover the local FEM curve.
- Related: supersedes the deleted-point direction suggested at the end of
  `docs/tasks/fracture2d-stagger-adaptive-step.md`'s prior audit;
  `docs/PROGRESS.md` current focus.

## 2026-09-27 — Do not equate same-target damage-map recovery with late-curve success

- Decision: Keep the new original-equation `0.006118125` own-failure-seeded
  128-sweep follow-up as a failed, source-bound diagnostic. Do not turn a
  decreasing map residual into acceptance, assume a high-damage root has
  FEM-like fields, or select an unlimited restart budget as the adaptive
  staggered controller.
- Why: authentic strict `0.006115` parent and old failed-iterate NPZ were
  replayed/bound, then fixed `omega=0.5` at unchanged load/history reduced
  the last map defect to `7.1244e-5` and AT2 strong residual to `0.06079`,
  but seven strict coupled gates still fail after 128 more sweeps; max
  candidate damage is already `0.999884`. No accepted load-step response
  or altered end reaction exists. Prior approximately `1%`-eligible
  smooth candidates likewise failed a held-out successor. Physically
  useful step and warning controls require explicit rollback and a
  successor/curve check, not repeated error-tolerance scans.
- Alternatives rejected: promoting the failed candidate, deleting high-damage
  nodes, extrapolating a final reaction from the trial, or rerunning an
  unchanged 128-sweep experiment solely to collect more of the same data.
- Related: `docs/tasks/fracture2d-stagger-adaptive-step.md`,
  `/tmp/opencode/fracture2d_original_d048_ownseed_stall_probe.py`.

## 2026-09-27 — Separate empirical warning quality from original strict flags

- Decision: For the user's unchanged-equation staggered direction, add an
  isolated bounded warning-study controller that does not replace the old
  owner or archive. Explicit numerical/physical study envelopes screen a
  complete original fixed-omega trial *before* commit; rejected complete
  trials retain original warning/strict status and trigger a recorded
  cutback. Strictly solved study steps may grow up to `du_initial`; study
  warnings retain their increment. No provisional warning is labeled a
  production field-equation-validated state.
- Why: source-bound retrospective screening separates mild onset warnings
  from large map/healing/outlier defects; a new physical run from old
  warning `0.0062` provisionally passes one `0.006225` cutback but its
  held-out `0.00625` rejects. From original strict `.048` `0.0061`, a
  zero-outlier study window provisionally accepts `0.00611`, discards a
  `0.00612` two-outlier *warning* trial, and accepts `0.006115`, then
  `0.00611625` warning cutbacks with no outliers and reaction `641.368`.
  Its next `0.0061175` trial again has two diagnosed outliers, despite
  meeting the numerical warning envelope. Study limits (`map <=0.001`,
  absolute AT2 strong <=`1`, healing <=`2e-5`, and diagnostic amplification
  `1.08`) are **data-specific probes** selected for distinguishing
  archived states, not verified physical universal criteria. The full
  endpoint and tail-force improvement remain unproven.
- Alternatives rejected: requiring every accepted step to pass the
  historical strict gate, unbounded warning acceptance (the old full path
  heals `0.02915`), classifying early two-point high-history branches as
  complete physical agreement from a passing scalar envelope, or
  permanently reducing the load increment without a recorded growth
  choice.
- Related: `examples/fracture2d/nonlocal_engineering_grip_relaxed_adaptive.py`,
  `examples/fracture2d/README.md`,
  `docs/tasks/fracture2d-stagger-adaptive-step.md`.

## 2026-09-27 — Keep transient-warning recovery experimental after front failure

- Decision: Preserve all outcomes of the user's approved original-equation
  load-growth comparison as **provisional or rejected** research evidence.
  Do not reinterpret a warning that fails the previously declared scalar
  envelope as strictly converged, or approve the proposed one/two-step
  defect bridge as a complete adaptive staggered path. Retain the
  original BA/Amor/AT2 owner and material particles unchanged.
- Why: from a parent/owner-replayed zero-outlier `0.00611625` warning
  prefix, bounded growth and rollback reach provisional `0.00624375`
  with two point outliers and reaction `623.793`. A separate old strict
  source-bound near-load branch has a longer crack and reaction
  `592.787`; warning ease alone is not physical superiority. The
  `0.00624375 -> 0.00625` original candidate fails only an *empirical*
  absolute map/AT2 envelope (`0.00390/3.626`) yet improves reaction and
  crack shape; its two counterfactual successors at `0.006275/0.0063`
  recover that same envelope and reduce reaction to `581.655`, retaining
  two outliers. At `0.006325` the next *uncommitted* state nearly matches
  phase-aligned FEM reaction (`562.335` vs `563.115`) but acquires a
  third point outlier and map/AT2 defects `0.004576/4.288`. A held-out
  `0.00635` continuation acquires a fourth outlier, front jumps
  `0.6104 -> 0.6504`, and both scalar/field mismatch increase. A
  single half-increment target `0.0063375` has the same fourth point and
  near-identical front, so the transient-defect acceptance hypothesis
  fails its own physical follow-through in this window.
- Alternatives rejected: declaring the two-step counterfactual segment a
  certified new path, raising the absolute map/AT2 caps solely because a
  particular force matches FEM, deleting crack points, or rerunning
  unchanged small increments after the halved-step counterexample.
- Related: `docs/tasks/fracture2d-stagger-adaptive-step.md`, scratch
  `/tmp/opencode/fracture2d_relaxed_growth_crossing_probe.py`,
  `/tmp/opencode/fracture2d_relaxed_temporal_bridge_growth.py`.

## 2026-09-27 — Residual-plateau rule is an observation, not a convergence proof

- Decision: Add the user's stagnation criterion as an isolated read-only
  audit: `ResidualPlateauStudyControls(window_sweeps,
  relative_span_limit, absolute_scale_floor)` and
  `assess_residual_plateau(attempt, controls)` measure the full-window
  spread of raw damage-map defect, absolute AT2 strong residual, history
  correction, and reaction correction on real complete owner sweeps.
  A flat tail can be *reported* as a numerical plateau; it does not
  replace the historical strict/warning status, does not commit a state,
  and is not a certificate of equilibrium or of an FEM-like curve.
- Why: a continuing descent or a two-cycle must not be mislabeled as
  stagnation. On the best physical branch candidate `0.0063125` (two
  outliers, FEM-matching crack-front stage, reaction `573.6` vs `563.1`)
  the 12-sweep tail still rises (map `0.00124 -> 0.00203`, absolute AT2
  `2.92 -> 12.64`, history correction `~90 -> 914`), so the rule correctly
  declines acceptance; its tested successor at `0.0063375` then grows to
  four outliers and a worse field match. Manufactured flat/descending/
  cycling cases discriminate as intended (focused tests `5/5`).
- Alternatives rejected: accepting a rising or cyclic tail as "unchanged",
  using the observation to silently relax original gates, or claiming the
  candidate produces a solved post-peak path before a held-out successor
  and the full curve are checked under the user's physical criteria.
- Related: `examples/fracture2d/nonlocal_engineering_grip_relaxed_adaptive.py`,
  `tests/test_fracture2d_nonlocal_engineering_grip_stabilized.py`,
  `docs/tasks/fracture2d-stagger-adaptive-step.md`.

## 2026-09-27 — Adopt the complete warning solution and a drift-excluded plotting rule

- Decision: Pause blind direct-solver attempts. Adopt the best complete
  scheme from the research record as the direct nonlocal reference artifact:
  original BA/Amor/AT2 equations on the 3395-node `delta=0.048` cloud, fixed
  damped-Picard `omega=0.5`, `du=1e-4`, `du_min=6.25e-6`, at most 144
  sweeps per attempt, honest stabilized-warning acceptance, load path
  `0 -> 0.0075`. Provide one concise entry point
  (`examples/fracture2d/direct_nonlocal_solver.py`, commands `run`,
  `verify`, `plot`) and remove the experimental modules from this task.
  For field figures, compute the plotted color minimum/maximum over the
  non-drift material points only (`|u_i|/u_bar <= 1.08`), while keeping
  every drift point inside the solution and drawing it in the figure.
- Why: the archived damped-warning run is the only complete `0.0075`
  solution and already has the closest-to-FEM peak/onset behavior
  (peak `641.340` vs FEM `642.486`; state-wise warnings honestly labeled).
  Its terminal reaction (`69.154` vs FEM `3.977`) and 2-to-31 drift points
  remain real limitations, so it is a diagnostic warning reference, not a
  strict or field-equation-validated production archive. The plotting rule
  was requested by the user and only affects visualization scale, never
  the equations, particles, reactions, or acceptance labels.
- Alternatives rejected: further blind cutbacks/threshold scans, promoting
  the uncommitted counterfactual branch, deleting drift particles from the
  physics, or presenting the drift-inclusive extreme (`|u| = 0.142`) as
  the field range.
- Cleanup scope: removed `nonlocal_engineering_grip_adaptive*.py`,
  `..._kernel_damage.py`, `..._p1_mass_damage.py`,
  `..._relaxed_adaptive.py`, `..._stagger_energy.py`,
  `..._stagger_restart.py`, `..._tolerance_study.py`, their dedicated test
  file, and the 18 experimental tests plus imports/README paragraphs.
  The original model/solver/workflow modules and the paused monolithic
  family remain (the monolithic engine is still imported by tracked
  workflow/archive code).
- Verification: focused helper tests `4/4`, full stabilized battery
  `82/82`, `git diff --check` clean; adopted-solution figures generated at
  `.../engineering_grip/direct_nonlocal_final/`.
- Related: `docs/tasks/fracture2d-stagger-adaptive-step.md`,
  `docs/PROGRESS.md`.

## 2026-09-27 — Consolidate the direct solver into `src/pinn/nonlocal2d` and park research variants

- Decision: Move the adopted direct-solver implementation out of
  `examples/fracture2d/` into the shared package `src/pinn/nonlocal2d/`
  (`nonlocal_grip_diagnostic`, `nonlocal_a2_coupled`,
  `nonlocal_engineering_grip`, `nonlocal_engineering_grip_jax`,
  `nonlocal_engineering_grip_stabilized`). Keep exactly one direct nonlocal
  solver entry in the example folder, renamed
  `examples/fracture2d/nonlocal_baseline.py` (formerly
  `direct_nonlocal_solver.py`). Prune the moved code to the symbols reachable
  from the adopted entry by static reachability analysis, deleting
  `nonlocal_geometry.py` and `nonlocal_mechanics_jax.py` and merging
  `nonlocal_mechanics.py` and `nonlocal_baseline.py` into their only consumers.
- Why: the example folder still exposed the full 45k-line research stack after
  the initial move; the adopted path needs only a subset of it, and the PINN
  stage must import the same operators/energy/history/damage. A shared package
  is the only structure that keeps one visible solver file, avoids duplicating
  the validated numerics, and prepares the PINN handoff.
- Verification before adoption: the pruned package reproduces the original
  `model_digest`, `static_digest`, `material_hash`, and `union_hash` exactly; a
  CPU smoke of the merged package returns reactions `12.67494679` at `0.0001`
  and `25.34683561` at `0.0002`, matching the archived early states; the
  retained entry helper tests pass `4/4`; `verify`/`plot` reproduce the fresh
  compact solution (76 states, 10 warnings, peak `641.34017`).
- Parked (untracked + ignored, kept on disk under `_legacy/`): the
  monolithic/arc/reduced bundle, `engineering_grip_workflow.py`, and every
  fracture2d test file that targets removed variant APIs.
- Alternatives rejected: physically deleting the research code (history and
  local reference are still useful); keeping the stack in `examples/` with
  untracked modules (a fresh clone could not import the tracked entry); a
  full single-file rewrite of the 45k-line validated stack (large
  revalidation risk with no numerical benefit).
- Remaining work (same task): deep-trim
  `nonlocal_engineering_grip_stabilized.py` to the `fixed` +
  `stabilized_warning` path by removing the Anderson, matched-load schedule,
  and archive-budget machinery.
- Related: `docs/tasks/fracture2d-lean-direct-solver.md`,
  `docs/tasks/fracture2d-nonlocal-pinn.md`,
  `examples/fracture2d/README.md`, `docs/PROGRESS.md`.

## 2026-09-27 — Merge the direct solver back into one GPU-only example file

- Decision: Collapse the adopted direct nonlocal solver into a single
  self-contained file `examples/fracture2d/nonlocal_baseline.py` (merged
  sections use module-prefixed private names `_g_/_a_/_j_/_e_/_s_`), delete
  `src/pinn/nonlocal2d/`, and make the GPU adapter mandatory. The CPU CG
  reference solvers are removed; CPU replay evaluation of device results is
  retained because it is the CPU-authoritative basis of the acceptance gates.
- Why: the user requires one file in the example folder and GPU-only execution;
  the later PINN stage can import the same file, and the single file removes
  the cross-package boundary that made the previous lean pass hard to audit.
- Rejected alternatives: keeping the shared package (user decision), keeping
  the CPU solver as a fallback (user decision; a full CPU path would also take
  many hours), and rewriting the 45k-line history from scratch (unnecessary
  revalidation risk).
- Verification: `model_digest` and `static_digest` match the archived compact
  solution field-for-field; retained tests pass `10/10` (entry `4/4`);
  GPU CLI smoke `0 -> 2e-4` reports `25.34683561371731` at `0.0002` versus the
  archived `25.3468356139`; `verify` reproduces the full-solution statistics
  (76 states, 10 warnings, peak `641.34017`, terminal `69.15426`); `plot`
  draws the drift-excluded figures.
- Remaining optional trim: the Anderson / matched-load schedule /
  archive-budget machinery inside the single file; those policies are not
  reachable from the CLI but still occupy roughly 3,000 lines.
- Related: `examples/fracture2d/nonlocal_baseline.py`,
  `examples/fracture2d/README.md`,
  `docs/tasks/fracture2d-lean-direct-solver.md`,
  `docs/tasks/fracture2d-nonlocal-pinn.md`, `docs/PROGRESS.md`.

## 2026-09-27 — Remove the unreachable research options from the single-file solver

- Decision: Rewrite `StabilizedEngineeringGripConfig` to a fixed-only
  dataclass, freeze the policy flags in the continuation, and mechanically
  delete every Anderson / matched-load-schedule / archive-budget helper and
  dead branch by iterative constant folding plus reachability pruning. The
  file drops from `16,263` to `12,188` lines and non-`fixed` policies are
  rejected at configuration time.
- Why: those options were unreachable after the adopted-scheme consolidation
  but still occupied roughly a quarter of the solver and kept dead imports and
  branches alive.
- Verification: GPU strict `0 -> 2e-4` gives `25.3468356138` (archived
  `25.3468356139`); the one-sweep warning path is bit-identical in reactions
  and acceptance statuses to the pre-removal file with field differences
  `<= 5.7e-12`; `model_digest`/`static_digest` are unchanged; retained tests
  pass `10/10`; `verify` reproduces the archived solution statistics.
- Alternatives rejected: keeping the dead options as documented-but-unused
  code (the user requires the adopted scheme only) and rewriting the numerical
  core (unnecessary revalidation risk).
- Related: `examples/fracture2d/nonlocal_baseline.py`,
  `docs/tasks/fracture2d-lean-direct-solver.md`.

## 2026-09-27 — Close the direct stage: default behavior confirmed, lean task archived

- Decision: Accept the two-pass lean refactor as the final direct-stage code
  (`examples/fracture2d/nonlocal_baseline.py`, `11,374` lines, GPU-only),
  archive the lean-refactor task, and make the nonlocal PINN task the active
  next step.
- Confirmation: a fresh default run `0 -> 0.001` (10 strict steps) matches the
  archived fresh solution to `4.3e-19` on loads, with identical acceptance
  statuses and warning flags, reaction max relative difference `4.2e-10`,
  displacement `3.2e-13`, damage `4.3e-11`, and history `5.7e-9`;
  `model_digest`/`static_digest` are unchanged and retained tests pass `10/10`.
- Why stop trimming: a line-coverage trace of the real strict + warning +
  `verify` + `plot` paths reaches every top-level definition (`0` fully
  unexecuted). Further cuts would remove mandated validation or warning-audit
  provenance, or require a core rewrite with a full GPU revalidation.
- Related: `docs/tasks/archive/fracture2d-lean-direct-solver.md`,
  `docs/tasks/fracture2d-nonlocal-pinn.md`,
  `examples/fracture2d/nonlocal_baseline.py`.

## 2026-09-28 — Reuse device A2 and material operator for fracture2d PINN residuals
- Decision: Build the PINN residual bridge around the frozen
  `JaxA2UnionMechanics` A2 energy-gradient assembly and the model's existing
  `ExplicitVolumeSparseD`. Implement only the Amor material-source accumulation
  in JAX using the same plane-strain embedded-3D positive energy and eta weights
  as the direct NumPy oracle; check its output on identical fields before use.
  Prescribe collar and anchor DOFs by affine lifting, and evaluate the full
  graph at fixed float64 shapes. Keep committed history as an explicit input.
- Why: Direct NumPy evaluations cannot differentiate through a neural network;
  the maintained GPU backend already contains the differentiable A2 kinematics
  and energy residual. A JAX Amor expression is necessary for a coupled neural
  objective but must remain numerically equivalent to the frozen oracle.
- Alternatives rejected: re-deriving bond-associated mechanics, casting the
  float64 graph to float32, training on the warning tail as converged data, or
  treating CPU replay as a differentiable loss.
- Related: `docs/tasks/fracture2d-nonlocal-pinn.md`,
  `examples/fracture2d/nonlocal_baseline.py`.

## 2026-09-28 — Use per-state acceptance flags for the PINN comparison
- Decision: Classify the adopted compact archive by its actual per-state
  `warning`/`acceptance_status` flags. The 10 warning states occur at loads
  `0.0060` through `0.0068` and `0.0071`; 65 strict labels include six
  post-localization states. The pre-warning strict prefix is indices `1..59`
  (`u_bar <= 0.0059`); `0.0060` and `0.0061` are drift-free warnings. Restrict
  the initial single-state PINN training/comparison to the strict prefix and
  report the full warning/drift tail separately without deleting points.
- Why: Counting statuses without checking their load positions conflated
  drift-free states with strictly accepted states; using `0.0061` as a strict
  convergence target would promote a warning candidate to ground truth.
- Alternatives rejected: assuming the warning flags occupy the last 10 rows,
  calling every drift-free state strict, or omitting later strict labels from
  the diagnostic tail.
- Related: `docs/tasks/fracture2d-nonlocal-pinn.md`,
  `examples/fracture2d/README.md`, `docs/PROGRESS.md`.

## 2026-09-28 — Start with a full-graph single-state neural residual experiment
- Decision: Use a float64 Equinox 2D-to-4 MLP with exact affine lifting of
  collar/anchor DOFs and raw material damage/history outputs. Initialize the
  displacement around the affine `u_y=load*y` field, scale its trainable
  correction by `load`, and scale the damage/history network outputs by
  `E*load**2/(Gc/l0)` and `E*load**2`, respectively. Use explicit fixed
  optimization-only residual scales and Adam settings in `train_base.py`,
  report unscaled norms and material field errors, and train one archived
  strict-prefix load at a time with its *previous* archived history as context.
- Why: Full-graph residuals keep the actual graph/weights/BCs, while a
  load-scaled affine initialization avoids the unphysical all-node drift seen
  after the first unscaled network step. The output scales are parameter
  conditioning choices only; no PDE coefficient, graph, or physical gate changes.
  A single-state result is an experiment, not a coupled continuation solution.
- Alternatives rejected: assuming fixed loss weights are physical validation,
  silently advancing predicted history as if it passed direct-solver gates,
  treating warning-labeled history as exact, or adding adaptive weight updates
  before the raw same-field residual check succeeds.
- Related: `examples/fracture2d/train.py`,
  `examples/fracture2d/configs/train_base.py`,
   `docs/tasks/fracture2d-nonlocal-pinn.md`.

## 2026-09-28 — Approve separately rebuilt float32 fracture2d PINN graph
- Decision: At the user's request, implement a distinct float32 PINN graph and
  revalidate its BA supports/material operator against the frozen float64
  direct reference before using it for training. Keep the archived float64
  direct solution unchanged and explicitly report any graph/topology and
  field-residual discrepancies. Preserve the same collar/anchor hard boundary
  constraints, material particles, kernel, physical parameters and split.
- Why: On the installed GeForce RTX 4080, warmed-up JAX 3072-square matrix
  multiplication with highest precision took 1.92 ms in float32 and 67.8 ms
  in float64 (~35x, indicative, not a PINN-step speedup). Independently
  rebuilding the strict float32 union candidate horizon on rounded coordinates
  changes 209 undirected pairs in and 76 out versus the float64 reference;
  152 float64 visible directed edges are not strictly within the float32 cutoff.
  Merely casting the network would leave the expensive residual in float64;
  merely casting the old edge list would violate its own float32 horizon rule.
- Alternatives rejected: relabeling mixed-precision training as full float32,
  silently retaining the invalid float32-cast old graph, or changing the
  already-adopted float64 direct solver.
- Related: `docs/tasks/fracture2d-nonlocal-pinn.md`,
  `examples/fracture2d/README.md`, `docs/PROGRESS.md`.

## 2026-09-28 — Keep FP32 PINN operator separate from the direct FP64 model
- Decision: Rebuild the float32 candidate/visible graph on rounded union
  coordinates with the original polygon visibility predicate, recompute phase
  triangle quadrature, cubic BA target-inclusive supports, inverse shape
  tensors, normalized eta weights and `ExplicitVolumeSparseD`, and reuse only
  the existing JAX A2 residual/energy methods on these float32 arrays. Retain
  the exact original collar and point-anchor DOF masks and affine lift.
  Disallow the float32 residual backend's inherited direct solve/replay entry
  points; the adopted direct solver stays float64. Train with float32 network,
  graph, mechanics, Amor and damage residuals and check their dtypes at startup.
- Why: This implements the user-approved separate graph without passing a
  float32 field through the float64 PCG/graph contract. On the frozen cloud,
  float32 candidate pairs differ by +209/-76 and material directed edges by
  +76/-152; max BA condition is 20.24, float32 Amor/AT2 residuals at the first
  archived field remain small, and its top reaction is 12.548 versus 12.675
  under the original float64 model. The material particle set and BC masks
  remain identical; this response difference is reported as discretization
  difference, not fitted away.
- Alternatives rejected: using the frozen float64 graph with float32 weights,
  calling an FP32-cast source tensor an independently rebuilt BA model, or
  modifying the adopted direct reference to match the new graph.
- Related: `examples/fracture2d/fp32.py`, `examples/fracture2d/losses.py`,
  `examples/fracture2d/train.py`, `docs/tasks/fracture2d-nonlocal-pinn.md`.

## 2026-09-28 — Trial frozen initial-residual scales for FP32 PINN training
- Decision: Compare the existing fixed scalar loss scales with a single
  explicit `initial_driver_rms` optimization policy on the same first strict
  state, seed, FP32 graph, MLP, Adam step size, and iteration budget. Freeze
  once before training: mechanics scale = initial full free A2 residual RMS,
  history scale = initial material Amor driver RMS, and damage scale = twice
  that Amor-driver RMS (the declared `2H` AT2 driving term). Require every
  scale to be finite and strictly positive; never change the raw equations,
  BCs, reference labels, or residual acceptance gates. Report scaled loss
  components alongside raw residual maxima, field errors, and reaction.
- Why: The 200-step fixed-scale FP32 pilot reduced total loss `0.002608 ->
  0.000990` while reaction moved from `8.34` to `27.17` versus `12.67`
  reference, with material displacement relative L2 still `0.351`. Initial
  hard-coded scales `1/100/1` make damage and history terms negligible at
  `u_bar=1e-4`; initial equation-derived characteristic scales make a bounded
  experiment measurable without treating a lower weighted loss as a physical
  success. The scheme is a project-owned optimizer preconditioner only.
- Alternatives rejected: treating fixed-scale loss reduction as convergence;
  selecting scales from the held-out current reference fields; clipping damage
  or history; adaptive multi-loss weights before the frozen-scale comparison.
- Related: `examples/fracture2d/train.py`,
  `examples/fracture2d/configs/train_base.py`,
  `docs/tasks/fracture2d-nonlocal-pinn.md`.

## 2026-09-28 — Trial deterministic spatial features for the first FP32 load
- Decision: Evaluate an explicit optional multi-frequency sine/cosine coordinate
  embedding (`1, 2, 4, 8, 16` cycles per unit length) against the current
  plain-coordinate MLP, holding the FP32 graph, physical model, BC masks,
  seed, MLP hidden width/depth, Adam, initial-residual loss scales and load
  fixed. The embedding is an input representation only, not a nonlocal kernel
  or fracture-model modification. Keep the experiment's first-load reaction,
  drift count, raw residuals and field errors visible alongside total loss.
- Why: A 1000-step full-graph frozen-scale run decreased normalized total loss
  to `0.224`, while mechanics remained `0.219` and the first-load field error
  worsened to `u` relative L2 `0.433` with 9 drift points and reaction `22.76`
  versus `12.67`. More steps alone did not resolve the mechanics
  representation/optimization limitation near the rounded notch. A controlled
  basis comparison tests whether spatial representation is the bottleneck.
- Alternatives rejected: promoting the low weighted loss as convergence,
  training on the held-out reference fields, changing graph/constitutive law,
  or tuning multiple architecture and optimizer dimensions at once.
- Related: `examples/fracture2d/train.py`,
  `docs/tasks/fracture2d-nonlocal-pinn.md`.

## 2026-09-28 — Compare fixed-damage energy and strong-residual mechanics training
- Decision: Add a separately named, diagnostic `fixed_damage_energy` objective
  that minimizes the existing A2 energy with previous archived damage fixed
  and exact affine boundary constraints. On this objective do not train damage
  or history, do not claim a coupled fracture equilibrium, and report the raw
  free mechanics residual, material displacement error and reaction alongside
  energy. Compare to the coupled-strong-residual run on the same FP32 model and
  spatial network without introducing supervised current-state targets.
- Why: With 1000 steps and frozen initial-residual weighting, a Fourier input
  reduces total loss from 0.224 (plain) to 0.186 and the free A2 residual max
  from 0.256 to 0.171, but reaction remains 28.20 versus 12.67 and material
  displacement relative L2 remains 0.420. Fixed-damage A2 energy is the
  already validated convex-in-field mechanics functional; isolating its
  optimization tests whether a squared-strong-residual objective or neural
  displacement representation limits this load.
- Alternatives rejected: interpreting reduced weighted loss as convergence,
  adding reaction/field supervision, changing A2 physics or boundary masks,
  and simultaneously updating the damage/history network under an energy
  objective that does not include their correct governing laws.
- Related: `examples/fracture2d/train.py`,
  `docs/tasks/fracture2d-nonlocal-pinn.md`.

## 2026-09-28 — Stage the two governing equations with exact Amor history
- Decision: Add a separately named FP32 single-load stagger experiment using
  two independent neural fields: displacement on the full union and damage on
  material points. At fixed trial damage minimize the existing A2 free-row
  mechanics residual; set trial history to the exact maximum of the unchanged
  committed history and the material-source Amor driver; at fixed history
  minimize the existing material AT2 residual; under-relax the damage trial
  with the adopted `omega=0.5`; then re-solve mechanics and recompute history
  from committed history at the same load. Keep q/d networks independent, pass
  changing arrays as JIT inputs, report full-graph raw coupled defects and BCs,
  and do not accept or advance an unconverged state.
- Why: The model has two governing equations (A2 equilibrium and AT2 damage)
  plus one algebraic Amor maximum update. An earlier single-network three-loss
  1000-step experiment reduced its weighted loss to `0.224` (plain encoding)
  or `0.186` (Fourier) without reducing field/reaction mismatch; the extra
  history-output network makes consistency a third optimization target even
  though the direct model evaluates it exactly. Alternating fixed-field solves
  follows the verified direct stagger semantics without adding new physics.
- Alternatives rejected: treating history as an independent PDE, updating the
  committed history within a trial sweep, sharing trainable trunk weights
  between nominally frozen fields, or marking a PINN iterate accepted solely
  because its stage loss decreased. The energy-only diagnostic attempt was
  interrupted and supplies no numerical result.
- Related: `examples/fracture2d/stagger.py`,
  `docs/tasks/fracture2d-nonlocal-pinn.md`.

## 2026-09-28 — Share one FP32 Amor reduction across history and damage diagnostics
- Decision: Expose the exact trial maximum history from the same Amor driver
  evaluation in `FractureResidual.evaluate(q, d, H=None, H_committed)`. The
  stagger controller uses this returned history and damage residual together
  for the final state; an explicitly supplied `H` still returns its genuine
  history defect. Verify the exact maximum and zero same-pass history defect.
- Why: Repeating float32 GPU segment reductions on identical fields produced
  a history residual of order `2.7e-6` in the first short stagger pilot. That
  discrepancy reflected reduction-order noise, not a deliberate change to
  the history law. The single-pass form avoids counting the noise as a
  coupled-history defect without relaxing any physical gate.
- Alternatives rejected: raising tolerance until the repeated calculation
  passed, clipping or rounding the driver, or treating an unaccepted trial as
  a converged historical state.
- Related: `examples/fracture2d/losses.py`,
  `examples/fracture2d/stagger.py`,
  `tests/test_fracture2d_pinn_residual.py`.

## 2026-09-28 — Close the interrupted fixed-damage energy CLI trial
- Decision: Remove the unvalidated `fixed_damage_energy` option from the
  single-network training CLI and config after the user steered the active work
  to alternating A2 and AT2 governing equations. Keep the fixed-damage energy
  methods in the adopted operator for independent validation, not as a second
  undeclared production training path.
- Why: The only energy-only run was interrupted before an optimization result;
  retaining a non-stagger option with no successful diagnostics would make it
  too easy to mistake an isolated mechanics result for coupled fracture PINN
  convergence. The two-network stagger diagnostic still uses exactly the A2
  and AT2 strong residuals and unmodified hard boundary constraints.
- Alternatives rejected: documenting the interrupted trial as a successful
  comparison or leaving a dormant CLI branch without meaningful validation.
- Related: `examples/fracture2d/train.py`,
  `examples/fracture2d/stagger.py`,
  `docs/tasks/fracture2d-nonlocal-pinn.md`.

## 2026-09-28 — Measure long stagger training before attributing short-run failure
- Decision: Extend the unchanged FP32 two-network single-load stagger to a
  documented >=10,000-iteration budget per fixed mechanics and damage stage,
  with periodic raw-residual/reaction reporting and resumable optimizer/network
  checkpoints bound to the cloud, reference, graph and configuration identity.
  Keep the same Adam learning rate, loss scales, direct-model BCs, `omega=0.5`,
  and no acceptance/next-load advancement; do not infer physical convergence
  from stage loss alone.
- Why: The current `elasticity2d/configs/train_base.py` uses 40,000 training
  iterations for a much simpler hole-in-plate example (its README still says
  20,000). The earlier 100–1000-step fracture2d trials cannot establish a
  representational/conditioning ceiling. Durable checkpoints avoid losing
  lengthy FP32 training on interruption; full coupled diagnostics remain the
  acceptance evidence.
- Alternatives rejected: claiming the short pilots prove that additional
  optimization cannot help, copying elasticity's learning rate/numerical
  tolerances without a controlled comparison, or changing the material model.
- Related: `examples/fracture2d/stagger.py`,
  `examples/elasticity2d/configs/train_base.py`,
  `docs/tasks/fracture2d-nonlocal-pinn.md`.

## 2026-09-28 — Add TensorBoard scalars to durable long-run stagger diagnostics
- Decision: Log fixed-stage training losses, raw full-graph residuals,
  displacement error, reaction, range/healing and drift diagnostics through
  `tensorboardX.SummaryWriter` at explicit stage intervals. Use one global
  iteration coordinate across stagger phases and `purge_step` when resuming a
  bound checkpoint, plus explicit flush/close. Keep an atomic checkpoint of
  network, optimizer and trial-field states, not only a weights-only snapshot;
  bind it to the configuration, float64 reference and rebuilt float32 graph.
- Why: The user specifically requests TensorBoard logging in the style of
  fracture1d/elasticity2d. Those examples log scalar training and reference
  metrics, while a >10,000-iteration stagger additionally needs a recoverable
  fixed-history/trial state; weights alone cannot reproduce an interrupted
  Adam and committed-history continuation. The installed TensorBoardX API
  documents `add_scalar`, `flush`, `close` and `purge_step` for resumed runs.
- Alternatives rejected: outputting only occasional terminal JSON, losing
  optimizer moments on resume, quietly reusing an existing run directory, or
  treating TensorBoard curves as physical convergence gates.
- Related: `examples/fracture2d/stagger.py`,
  `examples/fracture2d/stagger_checkpoint.py`,
  `docs/tasks/fracture2d-nonlocal-pinn.md`.

## 2026-09-28 — Do not accept the first-load 12k-per-stage stagger candidate
- Decision: Leave the complete 36,000-update FP32 single-load stagger as an
  unaccepted diagnostic. Do not blindly repeat it with more sweeps or promote
  a low independently trained mechanics loss to coupled convergence. First
  diagnose strong-form mechanics conditioning/representation and the reference
  model discrepancy using the preserved TensorBoard trace and bound checkpoint.
- Why: At `u_bar=1e-4`, one sweep of `12,000/12,000/12,000` fixed-stage
  steps produced raw mechanics-before/free residual infinity norm `1.476`,
  fixed-history solved damage residual `1.308`, and final coupled mechanics/
  AT2 residuals `1.764/2.019`. Despite mechanics-after logged loss near
  `0.1766`, the material displacement relative L2 grew to `0.714`, reaction
  was `23.146` versus archived `12.675`, and `22` drift points appeared where
  the direct first reference has none. History is evaluated exactly (zero
  same-pass algebraic residual), so that third condition is not the blocker.
  The reference's 40,000-step elasticity budget justified testing a longer
  interval, but by itself does not guarantee the present two-equation map
  converges with this objective, network and optimizer.
- Alternatives rejected: warning-accepting this trial, attributing its
  divergence solely to having too few steps without examining measured raw
  residuals, changing damage bounds or the kernel to obtain a smaller loss,
  or hiding early-load drift from the diagnostics.
- Related: `docs/tasks/fracture2d-nonlocal-pinn.md`,
  `/root/autodl-tmp/data/fracture2d/pinn_stagger_12000_v1/`.

## 2026-09-28 — Put fracture2d TensorBoard events under tf-logs
- Decision: Follow the user's specified TensorBoard root
  `/root/autodl-tmp/tf-logs/fracture2d/`, with a separate child directory per
  data run name. Keep the numerical `config.json` and atomic checkpoint in
  `/root/autodl-tmp/data/fracture2d/<run-name>/`. Move the completed long-run
  and previous smoke event directories into the new TensorBoard root; do not
  rewrite the immutable numerical checkpoint or its config hash.
- Why: This matches the established example-specific TensorBoard layout and
  lets one TensorBoard invocation see all fracture2d runs without mixing
  TensorBoard events with numerical artifacts. Checking both directories
  before starting a run prevents accidental overwrite of unrelated data.
- Alternatives rejected: leaving old events under each data run, combining
  events from distinct runs in one directory, or editing a completed run's
  checkpoint solely to change its log location.
- Related: `examples/fracture2d/stagger.py`,
  `examples/fracture2d/configs/train_base.py`,
  `docs/tasks/fracture2d-nonlocal-pinn.md`.

## 2026-09-28 — Explicitly detach previously accepted Amor history in PINN
- Decision: Implement the same direct-solver history law
  `H_trial = max(H_committed, Psi_plus(q_current))`, where `H_committed` is the
  accumulated maximum from the last accepted load, not only that load's
  instantaneous strain energy. Apply `jax.lax.stop_gradient` to this fixed
  committed-history input and to the fixed damage/history arrays in the
  stagger subproblem steps. Preserve gradients through the current Amor
  driver wherever the neural history residual differentiates with respect to
  displacement; do not alter the forward maximum or the physical equations.
- Why: The current FP32 stagger already passes prior history as an archived
  constant and differentiates only the active network, so it did not optimize
  previous history. The explicit detachment makes that fixed-state gradient
  boundary auditable and prevents a future traced history input from
  accidentally backpropagating through the committed load path. A small
  manufactured JAX check verifies zero derivative with respect to previous
  history and the expected nonzero derivative on active current-driver rows.
- Alternatives rejected: taking the max against only the preceding
  instantaneous energy, stopping the gradient through the current driver, or
  changing the direct history's max/irreversibility semantics.
- Related: `examples/fracture2d/losses.py`,
  `examples/fracture2d/stagger.py`,
  `tests/test_fracture2d_pinn_residual.py`,
  `docs/tasks/fracture2d-nonlocal-pinn.md`.

## 2026-09-28 — Log each unaccepted coupled trial at stage reporting steps
- Decision: At every configured logging step evaluate a provisional full-graph
  A2/AT2/Amor trial from the current q/d networks, with one same-pass
  committed-history maximum, and write it to `coupled/*` tagged
  `coupled/provisional=1`. Replace the stage `loss/<phase>` logging value with
  the loss recomputed from that same *post-update* state and the correct fixed
  phase field/history and scale; retain the gradient-step's *pre-update* loss
  separately as `loss_preupdate/<phase>`. The final accepted-status diagnostic
  at each sweep endpoint uses `coupled/provisional=0` and
  `coupled/accepted=0`, without a duplicate coupled entry at the last after
  stage step. Do not rewrite historical TensorBoard event files.
- Why: The completed 36,000-update run has 24 records per stage-loss tag but
  only one `coupled/*` value at step 36,000. Its loss tags came from before
  the optimizer update while their same-step raw residual/reaction tags came
  from after, so within-step comparisons were misaligned. During the damage
  stage its optimized residual uses fixed `H_used`, while the provisional
  coupled residual must instead use the current q-driven history; reporting
  both separately preserves the distinction without changing either equation.
- Alternatives rejected: retroactively forging earlier coupled traces from a
  final checkpoint, labeling intermediate fixed-subproblem fields as accepted
  equilibrium, or dropping the pre-update value without naming the change.
- Related: `examples/fracture2d/stagger.py`,
  `docs/tasks/fracture2d-nonlocal-pinn.md`.

## 2026-09-28 — Define a separate provisional coupled diagnostic loss
- Decision: At each stage logging point, evaluate current displacement and
  damage with a single exact committed-history maximum and record
  `coupled/loss = mean((r_q/s_q)^2) + mean((rho/s_d)^2)`, alongside the raw A2,
  AT2 and history residual metrics. Freeze `s_q` at the initial free-row A2
  RMS and `s_d` at twice the initial current-history RMS; validate both are
  positive and log the scales at step zero. Use the same diagnostic definition
  at the final sweep endpoint. It is not an optimizer objective or acceptance
  gate, and fixed-history `loss/damage` remains a separate stage loss.
- Why: The earlier `coupled/*` group contained only terminal raw diagnostics,
  not a combined loss curve. A fixed normalization makes future interim
  coupled values comparable across the before/damage/after phases without
  conflating the damage stage's fixed `H_used` with its q-driven provisional
  history or adjusting physical equations to fit the training loss.
- Alternatives rejected: summing incomparable raw residual units, changing
  diagnostic scales every stage, or treating a lower coupled scalar as proof
  that the unaccepted state satisfies all physical gates.
- Related: `examples/fracture2d/stagger.py`,
  `docs/tasks/fracture2d-nonlocal-pinn.md`.

## 2026-09-28 — Rename stagger TensorBoard tags to the repository convention
- Decision: Replace the ad-hoc `coupled/*`, `residual/*`, `scales/*` and
  `loss_preupdate/*` groups with the convention used by `corrosion2d` and
  `fracture1d`: `loss/*` for the three stage objectives
  (`loss/mechanics_before`, `loss/damage`, `loss/mechanics_after`), `eval/*`
  for reference comparisons (`eval/rel_L2_u`, `eval/rel_L2_d`, `eval/rel_L2_H`,
  `eval/reaction`, `eval/reaction_reference`), and `diag/*` for residual and
  state diagnostics (`diag/mechanics_residual_inf`, `diag/damage_residual_inf`,
  `diag/history_residual_inf`, `diag/coupled_loss`, `diag/drift_count`,
  `diag/damage_min/max`, `diag/damage_healing_max`, `diag/mechanics_scale`,
  `diag/coupled_damage_scale`, `diag/accepted`). Every `loss/*`, `diag/*` and
  `eval/*` value at a reporting step is computed from the same post-update
  state via one Amor reduction. Do not retro-rename v1/v2 event files; a new
  `2/2/2` smoke run verifies the new tag/step contract.
- Why: The user found `coupled/*` confusing and inconsistent with the
  repository, and it mixed loss-like, error-like and residual-like quantities
  under one unclear prefix. The renamed groups make the semantics explicit
  and let TensorBoard (and future scripts) treat losses, reference errors and
  diagnostics uniformly. `diag/coupled_loss` remains a diagnostic, not the
  optimizer objective, and the damage-phase `loss/damage` (fixed `H_used`)
  can differ from `diag/damage_residual_inf` (current q-driven history).
- Alternatives rejected: keeping aliases for the old names (duplicates the
  same curve), renaming values inside completed event files, or logging the
  pre-update loss again under a nonstandard prefix.
- Related: `examples/fracture2d/stagger.py`,
  `examples/fracture2d/README.md`,
  `docs/tasks/fracture2d-nonlocal-pinn.md`, `docs/PROGRESS.md`.

## 2026-09-28 — Build fracture2d PINN networks through pinn.arch.make_pinn
- Decision: Replace the bespoke `eqx.nn.MLP`/`_FourierNetwork` construction in
  `examples/fracture2d/train.py` with `_build_network(config, key, out_dim)`
  calling `pinn.arch.make_pinn`, following the `fracture1d`/`elasticity2d`
  structure. Expose `arch.type`, `hidden_dim`, `num_layers`, `activation`,
  `use_spatial_emb`, `spatial_emb_dim`, `spatial_scale`, `multi_head`,
  `tensor_split`, `film_hidden_dim`, `film_num_layers` in
  `configs/train_base.py` and validate them in `_validate_config`; remove the
  custom `--encoding plain|fourier` CLI. After construction, zero the final
  field projection (`output`, `out_heads`, or `trunk_output`) so the initial
  fields remain the affine lift (`u_y = load*y`, `d = H = 0`). `train.py`
  builds one four-output network (`u_x, u_y, d, H`); `stagger.py` builds
  independent two-output (displacement) and one-output (damage) networks from
  the same helper.
- Why: `make_pinn` with explicit `arch.*` fields is the repository standard;
  the custom network bypassed the shared Fourier embedding and head options
  and duplicated architecture code. The zeroed final layer preserves the
  load-scaled initialization and hard-constraint lifting contract. Parameter
  initialization changes (different key splits inside `pinn.arch`), so earlier
  recorded losses are not bit-reproducible; the physical model, data, BCs and
  loss definitions are unchanged.
- Alternatives rejected: keeping the bespoke Fourier embedder, changing the
  physical model or training data to accommodate the refactor, or changing
  field semantics (damage/history stay material-only).
- Related: `examples/fracture2d/train.py`,
  `examples/fracture2d/stagger.py`,
  `examples/fracture2d/configs/train_base.py`,
  `examples/fracture2d/README.md`.

## 2026-09-28 — Pass the fracture2d PINN config via ml_collections and widen the network
- Decision: Convert `train.py` and `stagger.py` to the repository script
  pattern: `train.py` defines one
  `config_flags.DEFINE_config_file("config", configs/train_base.py)` flag and
  `stagger.py` reuses it by importing `train`, so both run through
  `absl.app` and every run parameter (state index, stage budgets, output
  directory, log/checkpoint interval, resume) lives in the ml_collections
  config and is overridden with `--config.<path>=<value>`. Remove the
  argparse entry points. Enlarge the default network from `hidden_dim=32,
  num_layers=2` to `hidden_dim=128, num_layers=4` (still one four-output net
  in `train.py`, two single-field nets in `stagger.py`, zeroed final output).
- Why: The user requires the same config mechanism as the other examples and
  judged the previous `32 x 2` MLP too small for the nonlocal fracture fields.
  A single shared flag avoids an absl duplicate-flag conflict between the two
  modules (and with `solver.py`, which is not imported alongside them).
- Alternatives rejected: keeping argparse defaults outside the config file,
  registering two separate `config` flags, or silently changing the physical
  model/losses.
- Related: `examples/fracture2d/train.py`,
  `examples/fracture2d/stagger.py`,
  `examples/fracture2d/configs/train_base.py`.

## 2026-09-28 — Record the 128x4 first-load run as improved but unaccepted
- Decision: Keep the wider-network (`128 x 4`) v3 first-load run as an
  unaccepted diagnostic, and do not advance or accept its state. Use its
  curves to target the next diagnosis: the fixed-history damage solve now
  converges well and displacement drift disappears, so the remaining failure
  is the coupled A2/AT2 state after the mechanics re-solve.
- Why: In the identical `12k/12k/12k` budget the v3 run has 0 drift points
  (v2: up to 22) and `loss/damage` `0.9379 -> 0.00916`, but
  `diag/coupled_loss` only dips to `0.358` at step 24,000, then returns to
  `3.72` after the mechanics-after stage and ends at `6.00` with
  `diag/mechanics_residual_inf=0.9755`, `eval/reaction=33.00` versus FP64
  `12.67`. The training loop runs in about 83 s, so step budget is not the
  bottleneck; blind re-runs with more steps are not justified.
- Alternatives rejected: accepting the low damage-stage loss as convergence,
  reverting to the smaller network, or changing tolerances/kernels to make
  the coupled numbers smaller.
- Related: `docs/tasks/fracture2d-nonlocal-pinn.md`,
  `/root/autodl-tmp/data/fracture2d/pinn_stagger_12000_v3/`,
  `/root/autodl-tmp/tf-logs/fracture2d/pinn_stagger_12000_v3/`.

## 2026-09-28 — Keep the verified PINN operators; blame the mechanics optimizer, not the residual
- Decision: Do not change the A2/Amor/AT2 operators, BC lifting, reaction or
  the FP32 graph in response to the rising `loss/mechanics_after`. The audit
  against `examples/fracture2d/nonlocal_baseline.py` found the PINN residuals
  and operators equivalent to the adopted direct ones (rebuilt FP32 data with
  the same formulas; inherited `JaxA2UnionMechanics` assembly; the same
  `ExplicitVolumeSparseD` damage operator; the same `AffineConstraints` and
  reaction selector). Next work targets the inner mechanics optimization and
  GPU assembly determinism instead.
- Why: an exact free-DOF solve of the FP32 mechanics operator (frozen v3
  damage) reaches objective `8.8e-6` with reaction `12.664` versus the FP64
  direct `12.675` and displacement rel-L2 `0.12%`, and the direct reference
  field scores `2.2e-5` in the same PINN objective. Continued training from the
  v3 checkpoint still raises the loss (`0.349 -> 0.382` over 3,000 Adam steps
  at `lr=1e-4` and `0.349 -> 0.357` at `lr=1e-5`), so the failure is in the
  optimizer, not in the equations. `optax.lbfgs` is not a drop-in fix: its
  eager first step is good (`0.349 -> 0.217`) but the same update under
  `eqx.filter_jit` raises the loss to `4.45` (mixed float64/float32 scatter in
  its line search). The FP32 GPU assembly is also nondeterministic at `~1e-4`
  relative (`jax.ops.segment_sum`), and `--xla_gpu_deterministic_ops=true`
  makes build/compile impractically slow.
- Alternatives rejected: hunting an operator bug, more Adam steps or
  learning-rate sweeps, enabling `xla_gpu_deterministic_ops`, or accepting the
  low fixed-history damage loss as convergence.
- Related: `docs/tasks/fracture2d-nonlocal-pinn.md`,
  `examples/fracture2d/nonlocal_baseline.py`,
  `examples/fracture2d/fp32.py`, `examples/fracture2d/losses.py`.

## 2026-09-28 — Move the mechanics stage to L-BFGS on a deterministic FP32 assembly
- Decision: Keep the stagger structure and the verified operators, but replace
  Adam in the mechanics stages with a curvature-aware L-BFGS optimizer, and
  replace the GPU-atomic `jax.ops.segment_sum` assembly with a deterministic
  segmented reduction. Adam stays in the damage stage for now.
- Why: The audit showed the objective is a stiff least-squares (condition
  `~2.5e4`) with a float32 floor of `8.8e-6`, while Adam plateaus near `0.25`
  and then rises even at `lr=1e-5`. The eager L-BFGS first step is excellent
  (`0.349 -> 0.217`), and the run-to-run assembly noise (`~1e-4` relative) is
  an obstacle for any function-value-based optimizer; both are fixable in this
  example without touching the validated equations.
- Alternatives rejected: accepting Adam plateaus as convergence, GN/LM as the
  first step (kept as fallback if L-BFGS still stalls), and enabling
  `--xla_gpu_deterministic_ops` (impractically slow build/compile).
- Related: `examples/fracture2d/fp32.py`, `examples/fracture2d/stagger.py`,
  `docs/tasks/fracture2d-nonlocal-pinn.md`.

## 2026-09-28 — Fix the jitted lift with a mask blend; pin matmul precision and assembly
- Decision: `FractureResidual.lift` imposes the collar/anchor values with the
  precomputed select blend `jnp.where(free_mask, raw_q, prescribed)` instead of
  `prescribed.at[free_dofs].set(raw_q[free_dofs])`; `train.py`/`stagger.py`
  set `jax_default_matmul_precision = "highest"`; the FP32 mechanics assembly
  uses deterministic per-segment scans instead of atomic `jax.ops.segment_sum`.
  A regression test compares the jitted and eager lift composition.
- Why: the fused scatter compiled to a different displacement field under jit
  (`2.07e-5` max difference; the jitted field exceeded the prescribed
  top-collar value), which made the training objective and the eager reports
  different functions (`0.3490` vs `7.2514` on the same parameters) and
  invalidates the pre-fix optimizer traces. The select blend is numerically
  identical eagerly and agrees with jit to `6.4e-9`. Default TF32 matmuls
  contributed another `~6e-5` of eager/jit forward mismatch, and the atomic
  assembly made repeated residuals differ by `~1e-4` relative; both are
  removed. Post-fix tests `9/9`.
- Alternatives rejected: keeping the scatter with `optimization_barrier`
  (still wrong by `2.07e-5`), relying on `--xla_gpu_deterministic_ops`
  (impractically slow), or accepting eager/jit drift as rounding noise.
- Related: `examples/fracture2d/losses.py`, `examples/fracture2d/fp32.py`,
  `examples/fracture2d/train.py`, `examples/fracture2d/stagger.py`,
  `tests/test_fracture2d_pinn_residual.py`.

## 2026-09-28 — Keep L-BFGS as an unintegrated work item; the remaining gap is optimizer/representation
- Decision: Keep `examples/fracture2d/lbfgs.py` (deterministic host-side
  two-loop L-BFGS with Armijo backtracking) as an unintegrated work item, and
  do not yet rewire the stagger to it. The next step is a decision between
  matrix-free Gauss-Newton/Levenberg-Marquardt, a stronger curvature method,
  or network-representation changes.
- Why: Adam rises on the corrected objective (`0.349 -> 0.392` over 5,000
  steps), and the L-BFGS step descends `0.349 -> 0.065` over ~700 iterations
  but plateaus far above the `8.8e-6` exact-solve floor; a supervised L-BFGS
  fit of the network to the direct free-DOF displacement also stalls near
  `13%` relative error. The optimizer validates on an ill-conditioned
  quadratic (and scipy needs 500 iterations there too), so the plateau is not
  a gross implementation bug.
- Alternatives rejected: more Adam budget; integrating a plateauing L-BFGS
  into the stagger before the bottleneck is understood.
- Related: `examples/fracture2d/lbfgs.py`,
  `docs/tasks/fracture2d-nonlocal-pinn.md`.

## 2026-09-28 — Mechanics plateau is an energy-norm wall; test Gauss-Newton and restate the floor
- Decision: The next mechanics experiment is a matrix-free damped
  Gauss-Newton/Levenberg-Marquardt step (JVP/VJP plus truncated CG on the
  normal equations) for the frozen-damage objective, and the acceptance
  criterion is restated in field/reaction terms plus a residual reduction
  relative to the loading scale. Reach for exact-solve distillation only if
  Gauss-Newton also stalls.
- Why: Clean probes show the architecture fits the reference displacement to
  `0.7%` material relative L2 under supervised Adam (`lr=3e-4`, 3,000 steps),
  but the same field has a free-DOF residual `1.7e4` times the reference
  field's, and A-optimal linear readouts of random/adapted trunks bottom out
  at `4.2e3`/`1.7e3`. The mechanics objective is the A-norm of the field
  error, so the wall is high-frequency residual sensitivity, not L2
  representation. The `8.8e-6` exact-solve value and the `2.2e-5` reference
  value are float32 assembly-rounding floors of an essentially exact
  solution; demanding them from a neural approximation is not meaningful.
  Adam rises on the stiff objective and L-BFGS plateaus at `0.0653`, so a
  method that uses the exact tangent (Gauss-Newton) is the principled next
  step, and the frozen-damage block is linear in the displacement.
- Rejected alternatives: further L-BFGS tuning (Wolfe/memory), larger Adam
  budget, and representation changes justified by the earlier `13%`/`81%`
  probe numbers — those were probe bugs (one coefficient vector shared by
  both DOF components; a misplaced square) and are not evidence.
- Related: `docs/tasks/fracture2d-nonlocal-pinn.md`,
  `examples/fracture2d/losses.py`, `examples/fracture2d/fp32.py`.

## 2026-09-28 — Gauss-Newton implemented and ruled out for the raw residual; precondition the residual next
- Decision: Keep `examples/fracture2d/gauss_newton.py` (matrix-free damped
  Gauss-Newton with truncated CG and host Armijo backtracking) as a validated
  optimizer for well-scaled objectives, but do not use it on the raw
  mechanics residual. The next mechanics experiment is a preconditioned
  residual objective `||B R(q)||^2` with `B` an approximate inverse of the
  frozen tangent operator, plus a field/reaction-based acceptance criterion.
- Why: On the frozen first-load mechanics objective the first GN step from
  the affine base field is excellent (`1.0 -> 0.155`), but the line search
  cannot find descent afterwards: at the stuck point `||grad|| ~ 6.7e4`, a
  GN direction has norm ~`80`, and the objective is flat to `~1e-5` for all
  step sizes in `[1, 2^-45]` before exploding; one further accepted step
  needs `step_size = 3.7e-9` and then damping rises to `1e14`. The residual
  is the A-norm of the field error (`kappa(A) ~ 2.5e4`), so the
  parameter-space landscape is numerically jagged at the `1e-5` parameter
  scale in float32. The same mechanism explains the L-BFGS plateau: its
  `0.065` residual field has `rel_L2_u ~ 0.5` because the residual is nearly
  blind to near-rigid free-region modes. Preconditioning by `B ~ A^{-1}`
  keeps the exact solution as the unique zero but turns the landscape into
  the supervised L2 problem that Adam solves to `0.7%` in 3,000 steps.
- Rejected alternatives: more GN damping/backtracking range (already tested
  to `1e14` and `2^-45`), Adam/L-BFGS retuning, treating the L-BFGS `0.065`
  residual as convergence despite the `50%` field error.
- Related: `examples/fracture2d/gauss_newton.py`,
  `docs/tasks/fracture2d-nonlocal-pinn.md`.

## 2026-09-28 — Float64 evaluation removes the FP32 jaggedness; distillation is the accurate mechanics path
- Decision: Record that the mechanics optimization noise is a float32
  evaluation artifact and that exact residual preconditioning is equivalent
  to supervised fitting of the exact linear solve. Suspend the
  preconditioned-residual work and put the choice to the user: residual-only
  float64 mechanics (honest, slow field convergence) versus float64 exact
  solve as the mechanics target (distillation, accurate field) with the
  damage stage still a residual PINN. Continue the FP32 path only where its
  noise does not gate optimization.
- Why: On the float64 backend the reference-field residual is `8.2e-10` and
  raw Adam decreases monotonically `1.0 -> 0.0267` over 20,000 steps,
  whereas FP32 rose and Gauss-Newton could not step. But the residual is
  nearly blind to near-rigid modes, so at loss `0.0267` the field is still
  `32%` off with reaction `19.7` vs `12.7`. Preconditioning by `A^{-1}`
  turns the objective into the field-error fit (distillation), and truncated
  CG preconditioning needs ~200 iterations per evaluation here. Direct
  distillation reaches `0.4-0.5%` field error and reaction `12.5/11.3` in
  about 100 s per state.
- Alternatives rejected: FP32 residual optimization (jagged by evaluation
  noise), Gauss-Newton without preconditioning (cannot step), host L-BFGS in
  float64 (slower than Adam here), and further preconditioner approximations
  that converge back to distillation at this problem size.
- Related: `docs/tasks/fracture2d-nonlocal-pinn.md`,
  `examples/fracture2d/gauss_newton.py`, `examples/fracture2d/lbfgs.py`.

## 2026-09-28 — Target-free low-mode fixes fail; the linear block needs the solve in the loop for field accuracy
- Decision: Do not add affine ansatz parameters or a low-rank spectral
  preconditioner to the mechanics loss. If field/reaction accuracy is
  required, the linear mechanics block must use the linear solve inside the
  loop (distillation target or Newton-preconditioned gradient, which are
  equivalent); if the acceptance is residual-based with strain-sensitive
  coupling, the pure float64 residual stage can proceed end-to-end.
- Why: both target-free fixes fail empirically. The affine-augmented ansatz
  reaches raw loss `0.051` at 20k steps with material rel-L2 `48%` (plain
  float64 Adam: `0.027`, `32%`). The 48-mode Lanczos spectral preconditioner
  (`lambda_1 = 19.2`, `kappa ~ 1.4e4`, symmetry `1.6e-15`) reduces its own
  objective `1.0 -> 6.5e-3` while the raw loss stalls at `0.30` and the field
  degrades to `60%`: a low-rank inverse re-balances modes, it does not
  equalise their convergence. Only `A^{-1}` balances all modes, and that is
  the exact solution.
- Rejected alternatives: treating the low-rank preconditioned loss as the
  acceptance metric, using the reference displacement as a training target
  without an explicit user decision.
- Related: `examples/fracture2d/losses.py`, `examples/fracture2d/fp32.py`,
  `docs/tasks/fracture2d-nonlocal-pinn.md`.

## 2026-09-28 — FP64 is affordable for the end-to-end path; adopt option A
- Decision: Proceed with option A (end-to-end residual mechanics, no external
  target) on the float64 model, and start porting the stagger to float64.
- Why: the actual workload is edge-assembly dominated. Warmed timings on the
  RTX 4080: mechanics forward `0.60 -> 1.88 ms` (3.1x), mechanics Adam step
  `2.70 -> 5.10 ms` (1.9x), damage Adam step `0.59 -> 2.77 ms` (4.7x), while
  the raw `3072^3` matmul ratio is `37x`. A `12k/12k/12k` sweep costs about
  `2.6 min` in FP64. Float64 also removes the optimization noise that broke
  Adam in FP32 and uses the reference graph directly (no candidate-edge
  mismatch).
- Rejected alternatives: staying in FP32 for the mechanics optimization
  (jagged objective), or mixing FP32 and FP64 graphs in one stagger
  (inconsistent discrete operators between stages).
- Related: `docs/tasks/fracture2d-nonlocal-pinn.md`,
  `examples/fracture2d/stagger.py`.

## 2026-09-28 — Stagger ported to float64; first end-to-end state-1 sweep runs bounded and monotone
- Decision: Make `train.dtype = "float64"` the stagger default on the adopted
  reference graph (precision-specific checkpoint schema
  `fracture2d-fp64-stagger-checkpoint-v2`), keep the rebuilt float32 path
  behind `train.dtype = "float32"`, and keep the residual-only end-to-end
  objective. Continue with a larger stage budget before any acceptance
  judgement.
- Why: the FP32 path cannot optimize the stiff residual (jagged by float32
  evaluation noise), while the measured FP64 workload is only `2-5x` slower
  and the `12k/12k/12k` state-1 sweep runs in about five minutes. In that
  sweep the `after` mechanics loss falls `0.052 -> 0.019` (still falling),
  drift goes `20 -> 0`, reaction `23.9 -> 15.0` vs reference `12.675`, and
  `u_relative_l2` `0.471 -> 0.207`; damage reaches `0.0028` with no healing.
  The FP32 v3 run (lift-bug era) had `mechanics_after -> 156`.
- Rejected alternatives: keeping float32 for the mechanics stage, mixing
  FP32/FP64 graphs in one stagger, and reverting to a solver-target
  distillation (user preference is end-to-end).
- Related: `examples/fracture2d/stagger.py`,
  `examples/fracture2d/stagger_checkpoint.py`, `docs/tasks/fracture2d-nonlocal-pinn.md`.

## 2026-09-28 — Float64 budget comparison passes the physical gates; positivity and acceptance remain
- Decision: Keep the float64 end-to-end residual stagger and treat the two
  state-1 budget runs as the new baseline. The next items are damage
  positivity control and explicit acceptance gates before any state is
  accepted.
- Why: the `48k/48k/48k` single sweep reaches `u_relative_l2 0.0278` and
  reaction `12.811` vs `12.675` with drift `0`; three `12k` sweeps reach
  `u_relative_l2 0.0427`, reaction `12.845`, coupled loss `0.0316` (from
  `0.170` after one sweep), `d_relative_l2 0.229`, and a positive damage
  minimum with no healing. Both are far better than the FP32 lift-bug v3
  baseline (`mechanics_after -> 156`, reaction `33`, drift `22`). The long
  run's `damage_min = -2.6e-8` shows the AT2 least-squares does not enforce
  `d >= 0`, which the direct scheme does.
- Rejected alternatives: judging acceptance on the raw residual against the
  float32 rounding floor, and extending to more loads before the damage
  positivity and gate definitions are settled.
- Related: `examples/fracture2d/stagger.py`,
  `docs/tasks/fracture2d-nonlocal-pinn.md`.

## 2026-09-28 — Damage positivity via a smooth one-sided floor
- Decision: Parameterize the phase-field damage output through
  `d = 0.5 * (raw + sqrt(raw^2 + floor^2))` with an explicit
  `physical.damage_floor = 1e-8` config value, validated in `_validate_config`
  and covered by tests. The reference damage is strictly nonnegative (bulk
  minimum `2.3e-9` at state 1), so an unconstrained least-squares output is
  not acceptable.
- Why: the float64 long run reached `damage_min = -2.6e-8` and the negative
  bulk values dominated `d_relative_l2`. With the floor the three-sweep run
  gives `damage_min = 2.29e-7`, `u_relative_l2 0.0269`, reaction `12.770`
  vs `12.675`, coupled loss `0.0316`, no healing. The floor equals `raw`
  above the floor and keeps a finite gradient at zero, so the positive part
  of the solution is unchanged.
- Rejected alternatives: a hard `relu` (zero gradient region freezes the
  damage where it must activate), a clamped/straight-through output, and an
  exterior penalty with a tuned weight.
- Related: `examples/fracture2d/train.py`, `examples/fracture2d/stagger.py`,
  `examples/fracture2d/configs/train_base.py`,
  `tests/test_fracture2d_pinn_residual.py`.

## 2026-09-28 — Damage floor validated; damage solve matches the reference maximum
- Decision: Keep the smooth one-sided damage floor and move to defining the
  acceptance gates.
- Why: post-fix runs give a strictly nonnegative damage field
  (`damage_min 2.3e-7` after three sweeps, `3.9e-11` after the long sweep)
  while the damage stage solves `damage_max = 1.2095e-4` against the
  reference `1.225e-4` (1.3% low). The three-sweep mechanics also improves
  to `u_relative_l2 0.0269` and reaction `12.770` (0.75% high). The final
  reported damage is the `omega = 0.5` relaxation of the solved field, as in
  the direct scheme, so a single sweep reports half the solved damage by
  construction.
- Rejected alternatives: treating the relaxed `d_trial` maximum as a damage
  solve failure, and gating on the bulk `d_relative_l2` while the reference
  bulk values are `~1e-9`.
- Related: `docs/tasks/fracture2d-nonlocal-pinn.md`,
  `examples/fracture2d/stagger.py`.

## 2026-09-28 — State 1 accepted once; cross-process training still has last-bit backward nondeterminism
- Decision: Keep the deterministic forward/assembly and the explicit gate
  logging, but treat the trained-state acceptance as statistically spread:
  two identical `12k x 3` gate runs gave `(u 0.0286, reaction 12.667)` and
  `(u 0.0364, reaction 13.099)`; only the first passed every gate. Use gates
  with margins consistent with that spread (or converge further) before
  judging acceptance, and document the spread in the task record.
- Why: the forward path is bitwise reproducible (network, residual, FP64
  backend, deterministic operator matvec), but the reverse-mode graph is not
  run-reproducible on this GPU stack. Isolated primitives (segmented-scan
  transpose, explicit broadcast contraction, plain gradients) are
  deterministic; only their composition is not, which points at XLA fused
  reduction codegen rather than the model math. `CUBLAS_WORKSPACE_CONFIG`
  and `--xla_gpu_autotune_level=0` do not fix it, and
  `--xla_gpu_deterministic_ops` is unusably slow (a single probe exceeded
  900 s versus ~30 s). The stiff residual amplifies last-bit differences
  into `~1-3%` reaction spread over 36k updates.
- Rejected alternatives: claiming bitwise run-to-run reproducibility of the
  trained state; treating the single passing run as a stable acceptance;
  accepting the `1e5 x` slowdown of deterministic XLA ops.
- Related: `examples/fracture2d/fp64.py`, `src/pinn/operators.py`,
  `docs/tasks/fracture2d-nonlocal-pinn.md`.

## 2026-09-28 — Allow five percent reaction margin for end-to-end continuation
- Decision: Raise the explicit stagger reaction-relative gate from 2% to 5%,
  keep the displacement-relative gate at 5%, and advance strictly labelled
  loads only after the final sweep passes every gate. Carry the accepted
  networks, optimizer states, displacement, damage, and cumulative Amor history;
  a rejected trial never overwrites the committed state.
- Why: Two same-configuration state-1 FP64 runs produced reaction errors 0.06%
  and 3.35% from last-bit reverse-mode differences amplified by the stiff
  objective. The user selected this margin and end-to-end multi-load work.
- Alternatives rejected: retaining the 2% gate with run-dependent acceptance,
  training on archived current-load fields, and treating warning-labelled loads
  as strictly converged.
- Related: `examples/fracture2d/stagger.py`,
  `docs/tasks/fracture2d-nonlocal-pinn.md`.

## 2026-09-29 — Stop strict PINN continuation at state-2 healing and mechanics gates
- Decision: Preserve the user-approved 5% reaction and 5% displacement gates,
  require every other existing gate, and stop at the first rejected load.
  State 1's accepted snapshot and optimizer state stay committed; no accepted
  state-2 snapshot is written. Keep the damage history-only formulation while
  asking how to resolve its approximate healing, and review mechanics-scale
  normalization before changing the next experiment.
- Why: a small two-load smoke proved owner-paired history, atomic acceptance,
  rejection rollback, and idempotent restart. In the first full-budget run,
  state 1's final reaction error was 7.88%, so nothing was committed. In the
  second independent full-budget run state 1 passed (u 2.40%, reaction 1.97%);
  state 2 failed healing `5.97e-6 > 1e-10` and normalized mechanics loss
  `0.826 > 0.01` despite u 2.94% and reaction 0.23%. Warm starting the second
  load decreases its initial mechanics residual scale from `0.05103` to
  `0.005606`, making a fixed normalized gate ~83 times stricter in absolute
  residual. Resume verified that the rejected completed load cannot be
  skipped and state 1's accepted digest is unchanged.
- Alternatives rejected: automatically accepting the numerically plausible
  state 2, raising the healing tolerance without approval, imposing a new
  `d >= d_prev` inequality on the history-only law without a model decision,
  or silently reusing archived state-1 fields as training context.
- Related: `examples/fracture2d/continuation.py`,
  `docs/tasks/fracture2d-nonlocal-pinn.md`.

## 2026-09-29 — Permit up to 1e-3 damage healing as an explicit acceptance tolerance
- Decision: On the user's request, set `stagger.healing_atol=1e-3` and use it
  solely for the healing gate and accepted-successor comparison. Preserve the
  separate `stagger.damage_range_atol=1e-10`, strictly monotone cumulative Amor
  history, raw healing logs, and the unchanged history-only AT2 equation.
- Why: The first state-2 PINN attempt had healing `5.97e-6`, which the user
  considers acceptable. The adopted direct warning workflow also reports
  nonzero healing instead of masking it. This is an explicit tolerance on the
  accepted damage path, not a proof of exact damage irreversibility: at the
  first few loads `1e-3` can exceed the magnitude of the damage field.
  State 2 would still fail the independent normalized mechanics gate
  (`0.826 > 0.01`), so no previously rejected run is reclassified as accepted.
- Alternatives rejected: broadening the damage range along with healing,
  silently clipping the damage field, or changing the AT2/history equation.
- Related: `examples/fracture2d/configs/train_base.py`,
  `examples/fracture2d/stagger.py`, `examples/fracture2d/continuation.py`,
  `docs/tasks/fracture2d-nonlocal-pinn.md`.

## 2026-09-29 — Freeze the mechanics normalization at the first load
- Decision: On the user's selection, normalize every continued mechanics
  residual by the FP64 free-row RMS for the first strict load's zero-output
  affine field and archived zero-load damage. Recompute this identity-bound
  scalar from fixed model inputs at each load rather than carrying a mutable
  training value. Keep raw force residuals, the `0.01` normalized mechanics
  gate, and the A2 equations unchanged.
- Why: warm-starting state 2 from accepted state 1 reduced the per-load
  initial residual scale from `0.05103` to `0.005606`, which made the same
  squared normalized gate about 83 times stricter in absolute residual.
  A common denominator gives the same absolute residual threshold for all
  strict loads. It is a deliberately absolute path-wide criterion, not a
  claim of equal relative accuracy at larger loads.
- Alternatives rejected: silently treating an old per-load checkpoint as
  compatible with the new normalization or changing the raw PDE residual;
  the per-load warm-start and per-load affine normalizations were offered as
  separate explicit options.
- Related: `examples/fracture2d/stagger.py`,
  `examples/fracture2d/configs/train_base.py`,
  `docs/tasks/fracture2d-nonlocal-pinn.md`.

## 2026-09-29 — Permit online acceptance during the final mechanics-after stage
- Decision: If the online-acceptance flag is set, evaluate all existing gates
  at each complete reporting point of the **final** stagger sweep's
  mechanics-after stage. On the first simultaneous pass, stop and atomically
  commit that exact network/optimizer/field/history state, recording its
  actual optimizer step. If no point passes, retain final-step rejection.
  Do not search past curves for a favourable point.
- Why: Under the approved first-load normalization, a state-2 after-stage
  report at step 8,000 passed the mechanics loss (`0.009967`), displacement
  (`1.46%`), reaction (`1.29%`), healing (`6.07e-6 < 1e-3`), range, drift
  and history gates; additional Adam steps raised the final mechanics loss
  to `0.01045`. Online first-passage stopping avoids losing an already gated
  state solely by continuing noisy updates.
- Alternatives rejected: retroactively accepting that archived intermediate
  report without its network checkpoint, using a partial before/damage stage,
  loosening the `0.01` mechanics threshold, and changing the physical law.
- Related: `examples/fracture2d/stagger.py`,
  `examples/fracture2d/continuation.py`,
  `docs/tasks/fracture2d-nonlocal-pinn.md`.

## 2026-09-29 — Extend only the continuation endpoint, preserving warm starts
- Decision: Treat increasing `data.end_state_index` as an explicit resume
  operation. Verify every previously stored per-load config, accepted state
  and predecessor digest against its original numerical identity; permit
  changes to the endpoint only and keep old accepted files unchanged. The
  successor continues from the last accepted displacement/damage networks,
  their separate Adam states and the committed q/d/H fields.
- Why: binding the original endpoint into the root config made requesting
  state 3 after completing a two-state run look like a numerical-policy
  change, unnecessarily forcing a new root and re-training the first two
  loads. The endpoint specifies *how far to run*, not the numerical problem
  at a previously accepted state. Parent-chain validation still prevents
  mixing models or silently substituting archived previous fields.
- Alternatives rejected: overwriting old accepted checkpoints with new
  endpoint identities, dropping configuration checks for continuation, or
  reinitializing the neural networks/Adam states at each load.
- Related: `examples/fracture2d/continuation.py`,
  `docs/tasks/fracture2d-nonlocal-pinn.md`.

## 2026-09-29 — State 3 remains unaccepted under the fixed first-load mechanics gate
- Decision: Stop physical continuation at state 3 without changing the
  user-selected first-load normalization or `0.01` mechanics-loss limit.
  Preserve the accepted state-1 and state-2 snapshots and report the state-3
  raw diagnostics. Ask for a new explicit gate/budget decision before rerunning.
- Why: using only the permitted endpoint increase resumed the actual two-load
  root, verified its unchanged SHA256 parent chain and warm-started state 3
  from state 2. After three `16k/12k/16k` sweeps, state 3 passed displacement
  (`3.51%`), reaction (`1.97%`), damage range, healing (`4.65e-7`), history,
  finite and drift gates, but failed the absolute first-load-normalized
  mechanics loss (`0.01984 > 0.01`). Its directory has no accepted snapshot;
  a subsequent resume refuses to skip it. This is not evidence that the
  current mechanics normalization is appropriate across all 59 strict loads.
- Alternatives rejected: silently increasing the loss limit, changing to a
  load-relative scale after looking at state 3, or committing the state from
  good field/reaction metrics while its mechanics gate is red.
- Related: `examples/fracture2d/continuation.py`,
  `docs/tasks/fracture2d-nonlocal-pinn.md`.

## 2026-09-29 — Normalize later mechanics residuals by prescribed-load amplitude
- Decision: On the user's approval, add an explicit mechanics scale policy
  `first_load_times_load_ratio`: the adopted first-load zero-damage affine
  free-row RMS multiplied by the current prescribed displacement divided by
  the first prescribed displacement. Use the same scale for the training
  objective, reports and acceptance. Continue recording raw residuals and
  retain the predeclared normalized loss limit of `0.01`. Fork into a new
  identity-bound continuation root from the verified old accepted prefix.
- Why: a fixed first-load denominator is an equal-*absolute*-residual gate
  even though both the applied displacement and elastic force scale with
  load in the linear fixed-damage limit. A load-ratio scale compares residuals
  relative to the imposed amplitude; it does not change A2/AT2 or claim exact
  invariance once damage evolves. The existing state-3 trial failed the
  absolute gate alone, so its archived result cannot be retroactively
  accepted under the new policy.
- Alternatives rejected: rewriting the old accepted snapshots, silently
  relaxing `0.01`, retrospectively reclassifying a rejected checkpoint, or
  using current-load reference displacement as a training target.
- Related: `examples/fracture2d/stagger.py`,
  `examples/fracture2d/continuation.py`,
  `docs/tasks/fracture2d-nonlocal-pinn.md`.

## 2026-09-29 — Prioritize reaction curve and damage-field errors in continuation review
- Decision: On the user's clarification, present reaction versus load and
  damage-field comparisons before displacement error. Keep all previously
  authorized gates unchanged while the user selects a revised acceptance
  contract; neither infer a damage-field threshold nor silently remove the
  displacement gate.
- Why: the load-ratio fork accepted loads 3–6, but at load 7 all gates except
  displacement passed: reaction error `0.685%`, damage relative L2 `5.49%`,
  displacement relative L2 `5.23% > 5%`. Existing acceptance records damage
  range and healing, not reference-field damage error; relative L2 at load 1
  was `22.33%` and is not directly interchangeable with later damage errors.
- Alternatives rejected: relabelling the completed rejected state 7 as
  accepted, quietly increasing displacement tolerance, or imposing a new
  damage L2 gate without an agreed threshold and early-load interpretation.
- Related: `examples/fracture2d/stagger.py`,
  `docs/tasks/fracture2d-nonlocal-pinn.md`.

## 2026-09-29 — Report absolute damage RMSE and peak error at every load
- Decision: On the user's request for a better small-load damage metric,
  make dimensionless material-node `d_rmse = sqrt(mean((d_PINN-d_ref)^2))`
  the primary damage comparison and also report the maximum absolute error
  to reveal notch-localized discrepancies. Preserve the old relative L2
  only as an auxiliary trend; add both absolute metrics to per-stage
  TensorBoard evaluations, progress reports and the final result. This
  changes reporting only: the AT2 training objective and all existing
  acceptance gates remain as before pending a chosen damage-error threshold.
- Why: the first-load reference damage norm is just `9.011e-4`, amplifying
  a `2.013e-4` field error into `22.33%`; the damage is dimensionless and
  `0..1` bounded, so absolute nodewise error has defined units even when
  the reference field vanishes. On the first accepted load, RMSE is
  `3.454e-6` and maximum error `3.287e-5`; a peak-error companion avoids
  concealing the localized error in the whole-field mean.
- Alternatives rejected: relying on relative L2 alone at zero/small damage,
  normalizing by an arbitrary reference-damage floor, or retroactively
  assigning an unapproved damage-field acceptance tolerance.
- Related: `examples/fracture2d/stagger.py`,
  `examples/fracture2d/README.md`,
  `docs/tasks/fracture2d-nonlocal-pinn.md`.

## 2026-09-29 — Gate absolute damage RMSE, keep displacement diagnostic
- Decision: On explicit user confirmation, remove displacement relative L2
  from the acceptance gates (continue logging it), add the full material
  damage-field RMSE gate with limit `1e-2`, and keep the maximum absolute
  damage error diagnostic-only. Keep reaction relative error at `5%`,
  normalized mechanics loss at `0.01`, range, healing, history, drift and
  finite gates. From the accepted state-6 network/optimizer/q/d/H snapshot,
  train load 7 afresh in a separate identity-bound root; preserve the
  previously accepted and rejected checkpoints byte-for-byte.
- Why: the user's desired quality measures are the reaction curve and damage
  field; state 7 failed only the old `5%` displacement gate. The explicit
  `1e-2` RMSE value is the user's chosen limit, not one inferred from
  previously accepted data. Since its early-load values are orders of
  magnitude smaller, retain raw per-load RMSE/max error for assessing how
  discriminating this limit is as damage progresses.
- Alternatives rejected: accepting the old rejected state 7 under new rules,
  applying the `1e-2` limit to both damage metrics, or using reference damage
  as the PINN training target.
- Related: `examples/fracture2d/stagger.py`,
  `examples/fracture2d/continuation.py`,
  `docs/tasks/fracture2d-nonlocal-pinn.md`.

## 2026-09-30 — Stop strict PINN acceptance at the last strict reference load
- Decision: Continue under the committed, unchanged FP64 load-ratio and
  reaction/damage gate policy through reference state 59, then stop strict
  acceptance. Preserve the identity-bound state-1–59 snapshot chain. Do not
  promote `0.0060/0.0061` or the later mixed-warning/drifting reference tail
  to strict acceptance merely because the PINN passes pre-peak gates.
- Why: all 59 pre-warning strict loads passed and replayed from checkpoints,
  but the next two reference loads are explicitly warning-accepted even
  though drift-free; the post-peak part contains drifting and mixed-warning
  states. Reaction/damage acceptance against that archive would no longer
  establish the same strict validation contract. This boundary is a
  reference-validation limitation, not an observed failed PINN gate.
- Alternatives rejected: silently widening the strict prefix, asserting
  post-peak convergence by extrapolating the 59 pre-peak acceptances, or
  treating warning-labelled current-load fields as strict training targets.
  Any diagnostic post-peak continuation needs a separate, explicit policy.
- Related: `examples/fracture2d/continuation.py`,
  `docs/tasks/fracture2d-nonlocal-pinn.md`.

## 2026-09-30 — Explore the warning-region PINN only as a separate diagnostic
- Decision: The user selected exploratory post-peak PINN continuation from
  accepted state 59 rather than waiting for a new strictly validated direct
  reference. Keep the strict 59-state root unchanged. Any post-59 results
  must be explicitly labelled diagnostic; archived warning-reference fields
  and reactions are comparisons only, never training targets or strict
  acceptance evidence. Specify diagnostic advancement and physical-stop
  semantics before enabling a new run beyond 59.
- Why: the first post-59 reference states `0.0060/0.0061` are warning-labelled
  despite zero drift, and later reference states have mixed warning labels
  and drift. Retrospectively extending the strict gate contract would
  misrepresent post-peak validation. A separate lineage can retain the
  accepted state-59 networks, optimizers and q/d/H without rewriting them.
- Alternatives rejected: silently relaxing the existing strict-prefix
  loader, treating warning fields as ground-truth supervision, or claiming
  post-peak strict convergence from pre-peak acceptance alone.
- Related: `examples/fracture2d/continuation.py`,
  `docs/tasks/fracture2d-nonlocal-pinn.md`.

## 2026-09-30 — Advance diagnostic post-peak trials only across existing physical gates
- Decision: For the user's separate exploratory branch, retain the adopted
  A2/Amor/AT2 equations, hard displacement BCs, three-sweep FP64 budget,
  load-ratio mechanics normalization, and the existing finite, normalized
  mechanics, damage-range, healing, history-monotonicity and zero-drift gates
  as **trial-advancement** checks. Log but do not gate on warning-region
  reference displacement, reaction or damage. An eligible trial creates an
  identity-bound `diagnostic.ckpt.zip` with a separate schema and its
  predecessor digest; it is never called `accepted.ckpt.zip` or strict
  convergence. A failed physical gate stops without a successor trial.
- Why: warning-labelled and drifting direct-reference states cannot certify
  strict field/reaction accuracy, while reusing predeclared physics and
  numerical-bound checks avoids inventing a new PDE or quietly accepting
  nonfinite/healing/drifting fields. The conservative zero-drift stop may
  prevent reaching the post-peak tail; that failure must be reported rather
  than automatically weakening the gate.
- Alternatives rejected: copying strict accepted snapshots into warning
  directories, interpreting reference comparison tolerances as validated
  warning-region gates, and suppressing drift to force continuation.
- Related: `examples/fracture2d/stagger.py`,
  `examples/fracture2d/diagnostic_continuation.py`,
  `docs/tasks/fracture2d-nonlocal-pinn.md`.

## 2026-09-30 — Preserve the failed terminal diagnostic instead of claiming fracture convergence
- Decision: At the terminal load `0.0075`, retain the completed trial and its
  failed drift gate; leave state 74 as the last usable diagnostic parent.
  Report that loading trials reached the endpoint but the PINN did not
  reproduce the archived post-peak unloading or establish coupled AT2/stagger
  convergence. Do not loosen drift, relabel diagnostic snapshots as accepted,
  or retry a different optimizer/budget inside the existing root.
- Why: state 75 has 1 drift node, reaction `620.165` vs archived `69.154`,
  damage RMSE `0.1113`, committed AT2 residual infinity `104.092` and large
  stagger changes despite normalized mechanics loss `1.85e-5`. The earlier
  diagnostic-usable states passed the chosen advancement checks only; those
  checks do not establish the omitted coupled damage/stagger convergence.
- Alternatives rejected: declaring the full fracture path converged based
  solely on a small mechanics loss, hiding the terminal drift, or attributing
  the entire post-peak damage mismatch to the reference displacement outliers.
- Related: `docs/tasks/fracture2d-nonlocal-pinn.md`,
  `examples/fracture2d/diagnostic_continuation.py`.

## 2026-09-30 — Isolate outer stagger budget at fixed load 0.0060
- Decision: Following the user's approval, restart from the identical
  verified accepted state-59 networks, separate Adam states and q/d/H,
  hold prescribed displacement at `0.0060`, run a predeclared 12 sweeps
  of `16k/12k/16k` with unchanged `1e-4` learning rate and `omega=0.5`,
  and disable first-passage acceptance. Save every completed sweep's
  coupled AT2 residual, field changes, reaction and reference diagnostics.
  Use a new output directory and separate diagnostic snapshots, not a
  modification to old accepted or completed trial files.
- Why: the final AT2 residual and stagger changes remained large despite
  low fixed-history subproblem residuals and mechanics loss. This experiment
  tests outer iteration insufficiency before changing the optimizer/network.
  Warning-reference comparison remains diagnostic-only, and a full budget
  does not by itself prove coupled convergence or correct branch selection.
- Alternatives rejected: multiplying every subproblem's step budget at once,
  retrying from the accumulated inaccurate terminal diagnostic state, using
  current-load reference fields as training targets, and accepting intermediate
  reports after retrospectively searching their curves.
- Related: `examples/fracture2d/fixed_load_sweeps.py`,
  `examples/fracture2d/stagger.py`,
  `docs/tasks/fracture2d-nonlocal-pinn.md`.

## 2026-09-30 — Place new PINN artifacts and events under tf-logs
- Decision: Follow the user's directory rule: `/root/autodl-tmp/data/`
  holds material clouds and independent reference data, while new PINN
  checkpoints, numerical reports, figures and TensorBoard events all use
  `/root/autodl-tmp/tf-logs/`. Give new run artifacts and events separate
  sibling paths to avoid collisions. Copy the completed fixed-load experiment
  (including figures and bound checkpoint) into its existing tf-logs run
  directory and verify matching hashes before providing download paths.
- Why: splitting PINN checkpoints/plots under `data` and event files under
  `tf-logs` made the Jupyter events directory appear to contain no figures.
  Historical accepted checkpoint configs and SHA256 parent chains bind the
  old absolute data paths. Preserve original bytes and leave migration of
  those historical roots unresolved until a verified relocation contract
  is explicitly chosen; copied artifacts alone do not make the archive
  resume-compatible at its new physical path.
- Alternatives rejected: overwriting checkpoint metadata to make a moved
  path appear identity-compatible, silently deleting historical roots or
  user-created files, and treating a server-only `/root/...` path as the
  sole download link.
- Related: `examples/fracture2d/configs/train_base.py`,
  `examples/fracture2d/README.md`,
  `docs/tasks/fracture2d-nonlocal-pinn.md`.

## 2026-09-30 — Relocate historical PINN runs with identity-preserving path resolution
- Decision: On the user's explicit request, move the historical
  `data/fracture2d/pinn_*` run directories to
  `tf-logs/fracture2d/pinn_runs/<original-run-name>` without editing
  checkpoint bytes or stored config JSON. Resolve old absolute PINN paths to
  that physical archive at filesystem boundaries while preserving the
  original recorded strings for config digests, accepted SHA256 identities,
  parent-chain validation and interrupted-run resume. Never redirect
  independent reference/cloud paths or unrelated directories.
- Why: the user requires `data/` to contain only reference-related data and
  also requires old checkpoint paths to keep working. Rewriting archived
  configs would invalidate snapshot hashes, while compatibility symlinks
  would leave the PINN directory names visible under `data/`.
- Alternatives rejected: editing checkpoint metadata, leaving the bulk of
  historical PINN artifacts in `data/`, silently deleting unfamiliar
  notebook/run files, or sacrificing old accepted-chain verification.
- Related: `examples/fracture2d/pinn_paths.py`,
  `examples/fracture2d/continuation.py`,
  `examples/fracture2d/diagnostic_continuation.py`,
  `docs/tasks/fracture2d-nonlocal-pinn.md`.

## 2026-09-30 — Run a controlled 36-sweep fixed-load coupling diagnostic
- Decision: On the user's approval, use the same accepted state-59 parent at
  fixed `0.0060`, FP64, `16k/12k/16k`, Adam `1e-4`, `omega=0.5`, and 36 full
  sweeps with first-passage stopping disabled. Save per-sweep physical fields
  and explicit AT2 term/decomposition diagnostics in a new tf-logs root.
  Keep committed history/healing context tied to state 59 throughout.
- Why: 12 sweeps still decreased the coupled defect, while the fixed-H solved
  defect was much smaller. This isolates a larger outer budget with the same
  inner-stage settings. Extra sweeps also increase total optimizer work.
- Diagnostics: Normalize AT2 infinity residual by the sum of infinity norms
  of its three equation terms; no denominator floor. Report successive-sweep
  q changes relative to prescribed displacement, H changes relative to the
  larger previous/current infinity norm, absolute damage changes, and reaction
  changes relative to the larger previous/current magnitude. Zero scales fail
  loudly. Record physical arrays and residual-max location for independent
  checks and later threshold selection.
- Alternatives rejected: inventing convergence thresholds to terminate this
  comparison, changing relaxation and budget simultaneously, accumulating
  trial history as committed history, or treating warning-reference agreement
  as coupled convergence. This diagnostic introduces no new acceptance gate.
- Related: `examples/fracture2d/fixed_load_sweeps.py`,
  `docs/tasks/fracture2d-nonlocal-pinn.md`.

## 2026-10-01 — Probe FP64 inner residual convergence before post-peak extension
- Decision: Following completion of 36 sweeps and the user's sustained-research
  request, freeze the bound state-60 diagnostic damage/history and test 100
  Gauss-Newton steps per network, 200 CG iterations per direction, with the
  existing validated damped residual optimizer. Use only the physical residual,
  same FP64 network ansatz and exact hard BCs, and separate probe artifacts.
- Why: Final AT2 inf is `0.43069` but reaction still fluctuates by about 1% and
  fixed-H solved damage inf is `0.03089`. Establish whether inner optimization
  can reduce the remaining noise before spending many hours extending loads.
  Earlier Gauss-Newton failure was on FP32; it does not establish FP64 failure.
- Alternatives rejected: reference distillation, changing the frozen physical
  solver/model, treating a usable diagnostic as coupled-converged, or promoting
  the probe's altered networks to the existing checkpoint chain.
- Related: `examples/fracture2d/inner_probe.py`,
  `examples/fracture2d/gauss_newton.py`,
  `docs/tasks/fracture2d-nonlocal-pinn.md`.

## 2026-10-01 — Compare fixed-subproblem energy optimization without reference targets
- Decision: Probe 3000 FP64 L-BFGS-B iterations per frozen subproblem from the
  same bound snapshot. Mechanics uses the existing A2 energy. Fixed-H damage
  uses the independently checked volume-weighted quadratic whose derivative
  is volume times the existing AT2 residual. Check both field gradients against
  the declared residual before optimization and report raw residuals afterward.
- Why: FP64 residual GN stabilizes reaction but 200-step normal-equation CG
  remains far from its tolerance and residual improvement plateaus. Energy
  avoids squaring the field-operator conditioning; fixed subproblems retain
  their physical roots. This is an optimization experiment, not joint energy
  minimization of the hybrid history model or a replacement fracture law.
- Alternatives rejected: training to archived fields, claiming energy descent
  alone proves residual convergence, or silently changing accepted training.
- Related: `examples/fracture2d/inner_probe.py`,
  `docs/tasks/fracture2d-nonlocal-pinn.md`.

## 2026-10-01 — Test target-free local RBF enrichment of the existing neural fields
- Decision: Use frozen inherited MLPs plus trainable Gaussian hidden-unit
  readouts (64 global centers, 225 local centers around the notch-tip region).
  Same graph, BCs, A2/Amor/AT2 laws and smooth nonnegative damage activation.
  Mechanics readouts minimize the existing restricted fixed-d energy by a
  projected tangent solve, with declared SVD relative cutoff `1e-12`. Damage
  readouts use fixed-H volume-weighted energy and L-BFGS-B, max 1000 iterations.
  No exact nodal solution or reference field is a network training target.
- Why: Both frozen-inner probes plateau on the original MLP. Local enrichment
  tests approximation-space limitations rather than adding more global MLP
  iterations. This is an RBF-enriched neural approximation with fewer weights
  than free nodal unknowns, not a direct solve relabeled as neural training.
- Experimental convergence contract: require three consecutive full sweeps
  satisfying term-normalized AT2 inf `<=1e-3`, absolute damage change `<=1e-4`,
  relative H/q changes `<=1e-3`, relative reaction change `<=5e-3`, existing
  finite/range/healing/history/zero-drift checks, mechanics normalized MSE
  `<=1e-2` and normalized infinity residual `<=0.1`. These are explicit research
  tolerances, not strict-reference certification. Fixed load starts in the
  original state-59 context; loads advance only after this new contract passes.
- Alternatives rejected: prescribed crack/damage targets, selecting features
  from reference errors, loosening drift to obtain a curve, or training a
  readout at every independent nodal DOF. Local centers enrich the representation
  without deleting edges, particles or changing the physical model.
- Related: `examples/fracture2d/rbf_stagger.py`,
  `docs/tasks/fracture2d-nonlocal-pinn.md`.

## 2026-10-01 — Refine the tip neural basis and restore strong-residual training
- Decision: Preserve the failed coarse-RBF loading trial at `0.0061`. New
  experiments add Gaussian units at material coordinates within radius `0.06`
  of the geometric notch tip, width `0.008`, and explicitly select fixed-d
  mechanics residual least squares and fixed-H damage residual least squares.
  Mechanics is restricted readout training with SVD cutoff `1e-12`; damage
  retains the existing positivity activation and uses its analytic Jacobian,
  at most 100 function evaluations. The same coupled/physical gates apply.
- Why: Coarse energy-trained readouts stall at a very stable but unbalanced
  state (`AT2 inf 5.469`, fixed-H solved `5.470` at `0.0061`); iteration
  stability alone fails to establish physical equilibrium. Largest defect is
  at the notch tip. Tip-centered units test local spatial resolution, while
  strong residual fitting prioritizes that physical defect over tiny energy
  differences. No reference field is used to choose centers or train weights.
- Alternatives rejected: accepting the stable plateau, weakening the AT2 or
  mechanics-infinity checks, repeating more unchanged outer sweeps, or using
  a nodal exact solve as a supervised training target.
- Related: `examples/fracture2d/rbf_stagger.py`,
  `docs/tasks/fracture2d-nonlocal-pinn.md`.

## 2026-10-01 — Resolve collar residual hotspots and condition the RBF readout
- Decision: New trial adds geometry-owned collar centers (every third union
  coordinate with `y>=0.96` or `y<=0.04`, width `0.016`) and orthonormalizes
  the damage feature matrix with SVD cutoff `1e-10`. Serialize the transform
  so off-grid Gaussian-network evaluation remains defined. Keep the same
  strong residual objectives and all declared coupled/physical checks.
- Why: Independent saved-field recomputation locates the largest mechanics
  defect at the material/top-grip interface, not the tip. Tip-only enrichment
  cannot fix collar field structure. Damage optimization terminates on small
  parameter changes despite residual `0.382`; conditioning the strongly
  overlapping Gaussian readouts tests that parameter-space limitation.
- Alternatives rejected: raising the `0.1` mechanics-infinity limit to pass
  the current `0.106` defect, calling least-squares optimizer termination
  physical convergence, or using reference fields for new feature centers.
- Related: `examples/fracture2d/rbf_stagger.py`,
  `examples/fracture2d/research_report.py`,
  `docs/tasks/fracture2d-nonlocal-pinn.md`.

## 2026-10-01 — Add particle-scale tip units after a converged readout plateau
- Decision: New multiscale basis adds width-`0.003` Gaussian units at material
  coordinates within `0.10` of the tip, alongside the existing broad/tip/collar
  units and declared damage whitening. Keep residual objectives and checks.
- Why: Whitening resolves the readout optimizer stopping issue (gradient
  tolerance now passes), and collar enrichment clears the mechanics-infinity
  check, but fixed-H AT2 still bottoms near `0.380` at the rounded tip. Narrow
  units test the remaining local representation floor without increasing outer
  iterations or changing physical operators or reference targets.
- Alternatives rejected: relaxing the normalized AT2 limit `1e-3` to accept
  the `0.001186` plateau, claiming optimizer convergence equals PDE convergence,
  or replacing the entire field by one independent nodal weight per particle.
- Related: `examples/fracture2d/rbf_stagger.py`,
  `docs/tasks/fracture2d-nonlocal-pinn.md`.

## 2026-10-01 — Test a bounded damage-network link after post-peak overshoot
- Decision: New RBF run uses the standard sigmoid damage-output link, which
  enforces `0<d<1` through the neural ansatz while leaving AT2/history and
  mechanics equations unchanged. Initialize its logit readout by transferring
  the existing PINN's own relaxed state-60 field, not the direct reference.
  Keep the inherited damage MLP as a fixed anchor-load base (`0.0060`) so
  warm-start readouts have the same parameterization at each subsequent load.
- Why: Multiscale residual RBF reached coupled-checked `0.0062/0.0063`, but
  the `0.0064` trial overshot damage range and then diverged within its full
  budget. A nonnegative-only output permits unphysical damage above one;
  bounding the representation tests whether preventing this optimizer escape
  can maintain the crack-growth iteration. Rejected values are never committed.
- Alternatives rejected: clipping damage arrays, suppressing range/drift/healing
  failures, changing AT2 degradation/history, or calling the divergent trial a
  successful fracture response. Sigmoid parameterization may itself stall;
  report raw residuals and require the same coupled checks.
- Related: `examples/fracture2d/rbf_stagger.py`,
  `docs/tasks/fracture2d-nonlocal-pinn.md`.

## 2026-10-01 — Preserve tiny bulk damage structure in the bounded-network base
- Decision: New bounded-link trial uses `log` of the inherited positive damage
  network as the fixed sigmoid base, with Gaussian logit corrections. Transfer
  the PINN's own previous logit field into correction weights. The final field
  remains sigmoid-bounded; the same equations and physical checks apply.
- Why: The first sigmoid trial bottoms at AT2 `1.317` and healing `0.00194`,
  with the maximum defect far from the tip near the collar. A raw-MLP sigmoid
  base starts bulk damage near `0.5` and spends the finite RBF space rebuilding
  the inherited near-zero bulk profile. A log-positive base preserves this
  structure and uses localized correction capacity for fracture evolution.
- Alternatives rejected: clipping the transferred damage, relaxing healing
  or residual gates, or confusing a stable sigmoid optimizer with equilibrium.
- Related: `examples/fracture2d/rbf_stagger.py`,
  `docs/tasks/fracture2d-nonlocal-pinn.md`.

## 2026-10-01 — Test joint neural residual optimization at the Picard instability
- Decision: After quarter-step continuation fails at `0.006375`, freeze the
  coupled-checked `0.00635` PINN parent history and jointly optimize the enriched
  displacement and bounded-damage readouts at `0.006375`. Use the unchanged
  A2/AT2 residuals and `max(H_committed, current Amor driver)` in one objective.
  Orthonormalize the mechanics readout basis (cutoff `1e-10`) and use matrix-free
  trust-region least squares with JAX JVP/VJP and LSMR (300 iterations, atol/btol
  `1e-6`), at most 100 function evaluations. Check eager/JIT and a directional
  finite-difference JVP before training. Fixed scales are declared from the
  load-ratio mechanics convention and parent-history RMS.
- Why: Load cutback reaches `0.00635` but does not remove the localized
  mechanics/history runaway at the next step. A joint solve can test whether
  the equilibrium branch exists but is unstable to outer Picard iteration,
  without inventing a new fracture energy or differentiating cumulative
  committed history. The hybrid model is not replaced by joint energy descent.
- Alternatives rejected: more unchanged divergent sweeps, using the failed
  state's inflated history, changing physical stabilization, or dropping
  drift/range/healing/physical residual checks. Optimizer success alone is
  not cracking success; report all physical defects independently.
- Related: `examples/fracture2d/joint_rbf.py`,
  `docs/tasks/fracture2d-nonlocal-pinn.md`.

## 2026-10-01 — Equilibrate the joint neural Jacobian and tighten the inner solve
- Decision: New joint probe from the same checked `0.00635` parent uses inverse
  initial Jacobian-column norms as explicit parameter trust-region scales, and
  increases LSMR from 300 iterations at `1e-6` to 2000 at `1e-9`. Preserve the
  residual scales, graph, fields, objective and physical checks. Serialize all
  column scales; zero/nonfinite scales fail loudly.
- Why: Joint v8 avoids Picard runaway and has zero drift, but terminates at
  AT2 relative `0.001556` and healing `0.0010078`, both outside the declared
  limits. The mechanics/damage parameter blocks have very different Jacobian
  magnitudes; a unit-scaled, truncated inner solve can falsely stall. Test
  conditioning and solve accuracy before changing the representation again.
- Alternatives rejected: loosening these nearly passing gates, claiming v8
  optimizer success is equilibrium, or committing its unaccepted trial history.
- Related: `examples/fracture2d/joint_rbf.py`,
  `docs/tasks/fracture2d-nonlocal-pinn.md`.

## 2026-10-01 — Test stronger AT2 weighting inside the same uncommitted joint load
- Decision: A separate joint trial at `0.006375` multiplies the damage residual
  block by explicit optimization weight 100, with the same raw physical gates,
  column equilibration and LSMR settings. Warm-start only its neural weights
  from v9; retain the original `0.00635` committed damage/history and verify
  source/basis identity. The v9 history is not promoted or used as committed.
- Why: v9 still decreases but misses AT2 (`0.001224`) and healing (`0.001332`)
  while mechanics loss is already small. An empirical, declared loss-weight
  experiment can prioritize the damage defect; it does not change the common
  zero of the coupled residual equations or widen any acceptance tolerance.
- Alternatives rejected: advancing v9, imposing a new damage inequality,
  using inflated unaccepted trial history, or presenting weight 100 as a
  physically derived universal method. Observe whether mechanics gates worsen.
- Related: `examples/fracture2d/joint_rbf.py`,
  `docs/tasks/fracture2d-nonlocal-pinn.md`.

## 2026-10-01 — Make the maximum-history generalized derivative explicit
- Decision: Keep the joint objective's exact `max(H_committed, driver)`, but
  build its Jacobian from the piecewise derivative (0 inactive, 1 active,
  midpoint 0.5 at numerical ties within 128 float64 ulps of the history scale).
  Freeze that mask only for each linearized trust-region solve and recompute it
  at every accepted optimizer point. Record the policy; verify same-point
  objective agreement, fixed-branch finite differences and JVP/VJP adjointness.
- Derivation: For scalar `max(a,x)` with fixed `a`, derivatives are 0/1 off
  the switch, and its generalized derivative at equality is the interval
  `[0,1]`; 0.5 is the symmetric directional-derivative choice. This changes
  only the selected linearization at unresolved ties, never the forward law.
- Why: v10b found three history switches at every finite-difference step from
  `1e-6` to `1e-9`, with ~9% disagreement. The warm optimizer lies on the
  nonsmooth history surface; silently treating its central finite difference
  as an ordinary smooth derivative or training after a failed check is invalid.
- Alternatives rejected: smoothing or replacing the maximum history law,
  removing history derivatives, bypassing derivative checks, or asserting the
  unselected central difference uniquely defines the nonsmooth Jacobian.
- Related: `examples/fracture2d/joint_rbf.py`,
  `docs/tasks/fracture2d-nonlocal-pinn.md`.

## 2026-10-01 — Continue only checked joint neural states into crack growth
- Decision: After joint damage-weight-100 v10c passes every raw physical check
  at `0.006375`, continue from that completed joint neural state, validating
  foundation identity, feature transforms and q/d reconstruction. Its q/d/H
  become the new research committed parent only for subsequent joint trials.
  Keep residual laws, weight 100, generalized derivative and physical gates.
- Why: v10c avoids drift and reduces AT2 inf to `0.07821` / relative
  `0.0002087`, healing `3.48e-5`, mechanics inf normalized `0.09512`. It
  crosses the load where outer Picard diverged. The model is still a research
  approximation, not a strict-reference accepted state or a fully open crack.
- Alternatives rejected: advancing v8/v9 or failed stagger trials, promoting
  unaccepted trial history, calling sigmoid saturation crack convergence, or
  changing the physics to match the warning reference curve.
- Related: `examples/fracture2d/joint_rbf.py`,
  `docs/tasks/fracture2d-nonlocal-pinn.md`.

## 2026-10-01 — Enrich displacement at physical-residual hotspots
- Decision: After joint loading passes `0.0064` but fails mechanics infinity
  at `0.0065` (`0.12094 > 0.1`), add 32 width-`0.003` Gaussian displacement
  units at distinct largest free-mechanics-residual nodes of the unchanged
  checked parent. Keep damage features/link, history context, joint objective
  and all physical gates. Transfer old neural coefficients with zero new-unit
  weights and explicitly reproject the same field into the orthonormal basis.
- Why: Joint damage residual is already `0.000225` relative with zero drift,
  but the displacement approximation limits further loading. Target-free local
  residual enrichment directly addresses that defect; it selects no reference
  errors and changes neither collocation quadrature nor the physical graph.
- Alternatives rejected: increasing mechanics-infinity tolerance, advancing
  the rejected `0.0065` state, retraining on reference displacement, or adding
  one independent displacement weight at every nodal DOF.
- Related: `examples/fracture2d/joint_rbf.py`,
  `examples/fracture2d/joint_loading.py`,
  `docs/tasks/fracture2d-nonlocal-pinn.md`.

## 2026-10-01 — Rebalance joint objectives after damage passes but mechanics fails
- Decision: Separate same-load `0.0065` trial uses damage weight 10 instead of
  100, with the same 32 new displacement units, `0.0064` committed parent,
  raw physical gates and generalized Jacobian. Warm-start only rejected v13
  weights, verifying unchanged context and transforms.
- Why: v13 damage relative defect is `0.0002158`, well inside `1e-3`, while
  mechanics infinity remains `0.1171 > 0.1`. Weight 100 may over-prioritize
  damage fitting relative to mechanics near this transition. A declared
  optimization-only rebalance tests the tradeoff without weakening either gate.
- Alternatives rejected: committing v13, treating reaction growth as cracking,
  adjusting residual gates to match this trial, or reusing its unaccepted H.
- Related: `examples/fracture2d/joint_rbf.py`,
  `docs/tasks/fracture2d-nonlocal-pinn.md`.

## 2026-10-01 — Prioritize remaining mechanics outliers with fixed positive row weights
- Decision: Separate same-load trial keeps the same committed parent and
  enriched ansatz, but freezes mechanics residual row weights at
  `sqrt(1 + (r_initial / rms_initial)^2)` from the warm PINN trial. Damage
  weight stays 10; all unweighted physical gates stay unchanged. Serialize
  the row weights; zero/nonfinite RMS fails loudly.
- Why: Weight-10 v14 reduces mechanics mean-square loss about threefold to
  `5.27e-5`, but its largest normalized defect remains `0.1185 > 0.1`.
  Positive fixed weights emphasize the sparse remaining outliers rather than
  spending more work reducing already small rows. They preserve the common
  residual zero and do not alter graph quadrature, equations or physical gates.
- Alternatives rejected: declaring average loss sufficient, loosening the
  infinity gate, hidden adaptive resampling, or using reference-field errors.
  This is a declared optimization experiment, not a physical weighting law.
- Related: `examples/fracture2d/joint_rbf.py`,
  `docs/tasks/fracture2d-nonlocal-pinn.md`.

## 2026-10-01 — Repeat fixed-load outlier reweighting without committing trial history
- Decision: Research loading may perform at most five separate same-load
  refinement trials after a rejected joint candidate. Each recalculates the
  declared positive row weights from its warm neural trial, but keeps the
  original checked load parent q/d/H, basis and physical gates. All trials get
  separate directories; only a fully checked completed trial advances the load.
- Why: v15 passes `0.0065` with mechanics normalized inf `0.0465`, but the
  first `0.0066` trial develops new outliers (`0.1835`) as the coupled fields
  change. Refreshing fixed row weights between optimizer solves addresses
  moving defects without changing quadrature or pretending their history is
  committed. Reuse the existing v16 failed trial read-only for the first round.
- Alternatives rejected: accepting the failed new load, hidden per-step weight
  mutation, rerunning an identical rejected candidate, or unlimited blind loops.
- Related: `examples/fracture2d/joint_loading.py`,
  `docs/tasks/fracture2d-nonlocal-pinn.md`.

## 2026-10-01 — Reprioritize AT2 at the next crack-growth candidate
- Decision: At the rejected `0.0067` candidate, test damage weight 100 again
  while retaining the successful mechanics-outlier weighting, inherited
  enrichment and checked `0.0066` committed context. Warm-start rejected
  trial weights only; at most 200 evaluations, same raw physical gates.
- Why: Three reweight rounds pass `0.0066`; `0.0067` reaches damage maximum
  `0.9937` and passes mechanics/zero-drift checks, but AT2 `7.164` and healing
  `0.00987` fail. The bottleneck has shifted back to coupled damage accuracy.
  Independent residual hotspots remain ahead of the tip. Weight 10 is not
  sufficient here; optimizer weighting is tested explicitly, not physical gates.
- Alternatives rejected: calling one high-damage node a converged crack,
  advancing the rejected candidate, dropping healing, or changing the AT2 law.
- Related: `examples/fracture2d/joint_rbf.py`,
  `docs/tasks/fracture2d-nonlocal-pinn.md`.

## 2026-10-01 — Select new displacement units from the current uncommitted residual
- Decision: New same-load `0.0067` trial appends 32 narrow displacement units
  at v18's current residual hotspots, keeping the original checked `0.0066`
  committed context. Preserve all existing trial units and transfer their
  neural weights into the enriched orthonormal basis, checking q/d field
  reconstruction. Use damage weight 100 and fixed outlier mechanics weights.
- Why: v18 repairs damage (`AT2 relative 0.000359`, healing `0.000129`) but
  shifts mechanics maximum to `0.1945`. Features selected only from the prior
  load miss newly emerging localization. Current residuals are a target-free
  training signal, not committed physical history or reference-field targets.
- Alternatives rejected: advancing v18, changing physical gates, discarding
  old neural functions, or adding nodal variables across the entire cloud.
- Related: `examples/fracture2d/joint_rbf.py`,
  `docs/tasks/fracture2d-nonlocal-pinn.md`.

## 2026-10-01 — Audit fixed neural fields with independent direct subproblem solves
- Decision: Independently solve mechanics at the frozen v18 damage field and
  AT2 at its frozen history using the adopted validated operators, reporting
  convergence, drift amplitude, reactions and field discrepancies. Recompute
  the direct-q driver against the original checked parent history and inspect
  its fixed-H AT2 solve. Keep these oracle fields diagnostic-only; do not
  use them as network training targets or committed load states.
- Why: Several joint weighting/enrichment probes trade damage and mechanics
  defects around `0.0067` without reproducing unloading. An independent audit
  distinguishes representation/optimization error from a near-singular or
  drifting physical fixed-field problem before more neural-budget changes.
- Alternatives rejected: silently distilling direct solutions into the PINN,
  changing the frozen solver, declaring the near-one damage maximum successful
  opening, or diagnosing model instability from neural optimizer failure alone.
- Related: `examples/fracture2d/frozen_field_audit.py`,
  `docs/tasks/fracture2d-nonlocal-pinn.md`.

## 2026-10-01 — Isolate the physical coupling branch with diagnostic-only exact subproblems
- Decision: Run 24 fixed-`0.0067` oracle sweeps from the same neural damage and
  checked `0.0066` parent context, using strict fixed-d mechanics PCG and exact
  fixed-H AT2 solves, original `omega=0.5` and unchanged committed-history
  maximum. Save all defects and drift amplitudes; never use these fields as
  neural targets, accepted states or advancing PINN history.
- Why: The fixed-field audit converges mechanics in 302 iterations to relative
  `9.26e-10`, but its exact field already has 2 drift nodes and displacement
  maximum `3.743` times load; fixed-H neural damage is accurate to `0.000521`.
  Determine whether the oracle's coupled branch unloads and whether localized
  drift persists before interpreting the neural failure as optimization alone.
- Alternatives rejected: guessing a stabilization law, overriding the zero-drift
  gate, calling an exact fixed-field solve a coupled fracture solution, or
  quietly distilling oracle fields into the neural model.
- Related: `examples/fracture2d/frozen_field_audit.py`,
  `docs/tasks/fracture2d-nonlocal-pinn.md`.

## 2026-10-01 — Compare intact and damaged tangent modes before proposing model changes
- Decision: Independently audit the existing constrained tangent at intact and
  v18 damage fields, using matrix-free extremal eigenpairs, explicit Ritz
  residual checks and nodal participation/localization. Save diagnostics only;
  do not change the operator or add uncalibrated stabilization.
- Why: Exact oracle sweeps reduce reaction `664.4 -> 614.3` but retain two
  drifting nodes whose maximum displacement grows to `21.12` times load,
  with history `2.08e6` and coupled AT2 inf `32.47` after 24 sweeps. Distinguish
  a soft localized damaged mode from neural conditioning/representation and
  from an actual negative/zero tangent mode before deciding the next formulation.
- Alternatives rejected: declaring every optimizer failure a model defect,
  inferring eigenvalues from CG iteration count, changing `k_res` or graph
  visibility without a separate physical decision, or hiding crack-core drift.
- Related: `examples/fracture2d/operator_spectrum.py`,
  `docs/tasks/fracture2d-nonlocal-pinn.md`.

## 2026-10-01 — User selects model/soft-mode review with current equations and checks retained
- Decision: The user explicitly chose "先审查模型与软模态（推荐）". Continue
  mathematical/source-backed review before any model or gate change. Compare
  both the Euclidean force-tangent spectrum and the natural union-volume
  displacement-norm spectrum (`A v = lambda V v`, implemented symmetrically
  as `V^(-1/2) A V^(-1/2)`). This is an audit metric, not added inertia.
- Why: Euclidean low eigenvectors alone can conflate tiny quadrature volumes
  with loss of physical energy control. Existing calculations show strictly
  positive damaged eigenvalues, so calling these true hourglass/zero-energy
  modes or adding stabilization would be premature. The current kernel,
  degradation/history laws, graph, reference and all-material drift gate stay.
- Sources: Kristensen/Martinez-Paneda (2020), pp. 4–5, Eqs. (23),(25),(30),(31),
  distinguishes fixed-d stiffness conditioning and residual/correction checks;
  Chen/Spencer (2019), OSTI accepted manuscript 1572458, Section 4,
  Eqs. (24)–(26), characterizes energy positivity, not a displacement envelope.
- Alternatives rejected: confusing a positive localized soft mode with a
  proven negative/zero mode, interpreting raw force norms as volume-weighted
  physical field norms, or changing the model to force the PINN curve to drop.
- Related: `examples/fracture2d/operator_spectrum.py`,
  `docs/tasks/fracture2d-nonlocal-pinn.md`.

## 2026-10-01 — Audit degradation positivity and transmission without adding stabilization
- Decision: Verify the project-owned A2 energy's directional Loewner bounds,
  decompose localized mode energy by source center, and probe the unchanged
  mechanics operator across prescribed fully damaged strips of widths
  `0.008/0.024/0.048/0.096/0.144` at load `1e-4`. These synthetic fields are
  not AT2 solutions, accepted states or training targets. Keep `k_res=1e-8`.
- Derivation: Write `A(d)=sum_i w_i(d) A_i`, with each fixed-geometry `A_i`
  positive semidefinite. The phase center ratio `alpha_i=w_i(d)/w_i(0)` is
  between `k_res/(1+k_res)` and 1, including intact grip contributions.
  Therefore `alpha_min A(0) <= A(d) <= alpha_max A(0)` in quadratic-form
  order. Strict intact free-block positivity implies fixed-d positivity.
- Why: Volume spectra remain positive and condition estimates grow only
  `34234 -> 36959 -> 43344`, even as crack-core displacement grows. Separate
  expected degradation/tiny-volume effects from true correspondence instability,
  and test whether fixed kinematic neighborhoods transmit across a failed band
  before introducing damage-aware graph/kinematic changes.
- Alternatives rejected: adding a penalty just because Euclidean modes are
  localized, calibrating `k_res` solely to the drift envelope, or treating a
  prescribed strip as a solved fracture profile.
- Related: `examples/fracture2d/degradation_audit.py`,
  `docs/tasks/fracture2d-nonlocal-pinn.md`.

## 2026-10-01 — User authorizes an independent scale-separation validation branch
- Decision: The user explicitly chose "验证尺度分离（推荐）" after the completed
  model review. Retain plane strain, Amor history, AT2 and the A2 energy form,
  but independently declare mechanics horizon, damage horizon, and physical
  grip extent. Initial pilot: material far size `0.008`, near size `0.004`,
  grip spacing `0.008`, mechanics horizon `0.016`, damage horizon `0.048`,
  grip extent `0.048`. Refined comparison: material/grip size `0.004`, mechanics
  horizons `0.008` and `0.016`, same damage horizon and grip extent. Keep all
  constitutive constants and current range/healing/drift checks unchanged.
- Why: The synthetic `d=1` band at width `l0=0.024` retains 86.86% of intact
  reaction with the old common horizon. Positivity/source audits do not show
  a true zero-energy instability, so scale separation/refinement is a more
  controlled next test than an arbitrary stabilization or new damage law.
  The pilot checks implementation/rank first; it is not claimed mesh-converged.
- Boundary discipline: retain original physical collar extent and all added
  collar vertical constraints. Refining the ghost spacing must not silently
  replace the engineering volume-grip BVP by the older outer-row-only BVP.
- Verification: common-horizon baseline equivalence; rank/affine patches;
  energy/force and reaction-work consistency; failed-band transmission; then
  independent new reference loading and PINN. Old solver, clouds, reference
  and accepted SHA chain remain immutable and cannot certify the new branch.
- Alternatives rejected: lowering a horizon on an unresolved cloud with a
  silent rank fallback, changing grip/loading geometry with horizon, treating
  Bie (2023) cohesive plane-stress laws as an unchanged AT2 fix, and adopting
  diagnostic synthetic damage as a PINN fracture state.
- Related: `docs/tasks/fracture2d-nonlocal-pinn.md`,
  `examples/fracture2d/mesh.geo`.

## 2026-10-01 — Build a strict independent reference with explicit damage PCG and larger outer budget
- Decision: Reference loading for the validated fine split model uses the same
  volume-multiplied AT2 SPD system, solved by Jacobi-PCG with explicit residual
  recomputation (rtol `1e-11`, atol `1e-14`, maxiter 2000), and strict FP64
  mechanics. Require three consecutive coupled-residual/field-change/physical
  passes; preserve original committed history through each load. Reference
  fields remain comparison-only, never neural training targets.
- Verification: Damage PCG matches the frozen independent sparse direct solve
  on a manufactured cloud, and deliberately exhausted CG fails loudly.
  Fine-model loads 1–40 passed strict reference checks. Load 41 (`0.0041`)
  exhausted 200 sweeps solely at coupled AT2 relative inf `3.88e-6 > 1e-6`,
  with zero drift and still decreasing small field changes.
- Follow-up: A separate read-only fork verifies all 40 committed field digests
  and raises the outer budget to 1000, leaving all tolerances/physics/load
  increment unchanged. Do not relabel or overwrite the completed rejection.
- Alternatives rejected: a sparse factorization of the 35-million-edge damage
  graph at every sweep, relaxing the nearly passing coupled threshold, copying
  rejected history into a successor, or pretending reference success is PINN.
- Related: `examples/fracture2d/split_reference.py`,
  `tests/test_fracture2d_split_reference.py`,
  `docs/tasks/fracture2d-nonlocal-pinn.md`.

## 2026-10-02 — Bound split-reference diagnosis and prioritize PINN damage/reaction
- Decision: The user authorizes repairing the current direct bottleneck and
  then connecting the split model to PINN, prioritizing damage fields and
  reaction curves. Other unmet checks may be retained as explicit warnings;
  record original values/limits and distinguish diagnostic advancement from
  strict acceptance. Nonfinite or malformed arrays remain fatal.
- Numerical experiment: compare unchanged damped Picard against an explicit
  undamped Picard retry at the same load, initialized from the SHA-verified
  uncommitted trial while retaining the original accepted state-41 history.
  This changes only the outer iteration, not equations or physical thresholds.
- Why: latest state-42 sweep 610 has converged mechanics and zero drift, but
  coupled AT2 relative inf 0.057585 and damage change 0.012071. There is no
  demonstrated implementation bug; first test whether slow crack-front
  propagation under fixed damping is the bottleneck. PINN integration need not
  wait for a complete post-peak reference path.
- Alternatives rejected: deleting the split branch, overwriting old rejection
  or trial records, promoting warning states to strict reference, importing
  trial history as committed history, or using reference fields as targets.
- Related: active fracture2d PINN task, split_reference.py, split PINN bridge.

## 2026-10-02 — Preserve the previous direct resume point during field publication
- Decision: New split-reference writes use generation-named trial NPZ files,
  then atomically publish their filename/hash in progress.json. Retain legacy
  fixed-name trial compatibility, with identity/digest validation in both cases.
- Why: An interruption between replacement of trial.npz and progress.json could
  leave the previous progress pointing to different field bytes. A manufactured
  interrupted-next-generation regression protects the previous valid snapshot.
- Alternatives rejected: bypassing SHA checks on resume or inferring committed
  history from the newest uncommitted file. No numerical equation changes.
- Related: split_reference.py, test_fracture2d_split_reference.py. The already
  running omega-one probe loaded the prior fixed-name writer; subsequent runs
  use the generation-pinned writer.

## 2026-10-02 — Compare neural energy minimization after split residual training stalls
- Decision: Add an explicit fixed-d A2 energy mechanics objective in a separate
  split PINN trial, keeping damage's fixed-H AT2 loss and every raw report/check.
  Normalize energy by E times prescribed displacement squared times union
  volume, a positive load-dependent constant. Residual-MSE remains selectable.
- Why: Real fine-cloud residual training (3 sweeps, 2000 updates per block)
  reduces mechanics MSE to 0.09324 but reaction error is 72.93% at the first
  load. The previously verified energy gradient equals assembled force; its
  fixed-d SPD stationary equation is the same constrained mechanics equation.
  This tests a different neural optimization metric rather than weakening the
  reaction comparator or adding supervised reference targets.
- Alternatives rejected: advancing the rejected first load, fitting reference
  displacement/reaction/damage, or labeling small absolute low-load damage RMSE
  as fracture success. No constitutive or history equation changes.
- Related: split_pinn.py and manufactured split energy/force checks.

## 2026-10-02 — Use explicit final-block first-passage comparisons for split PINN
- Decision: Preserve the completed rejected load 4; a new read-only neural
  prefix fork may inherit loads 1–3 and stop the final mechanics block at its
  first logged report satisfying damage RMSE and reaction limits. Keep every
  secondary warning and warning ancestry. Config binds this stopping policy.
- Why: Load 4's sweep 2 reaction error is 0.504%, but the fixed-step final sweep
  ends at 15.18%; constant-step Adam can leave an adequate reaction comparison.
  The project already used final-block first-passage stopping in the adopted
  PINN. Apply it online in a new trial, never retroactively promote sweep 2.
- Alternatives rejected: relabeling the completed rejection, changing the
  reaction tolerance, adding reference values to the objective, or importing
  direct committed fields instead of learned neural predecessor fields.
- Related: split_pinn.py and the earlier accepted first-passage policy.

## 2026-10-02 — Advance the split neural prefix and audit localized damage
- Decision: Continue unchanged energy/first-passage warning loading through
  state 20 after states 7–10 meet primary comparisons. Add comparison-only
  whole-field volume-weighted and tip-region damage diagnostics alongside
  existing unweighted RMSE, maximum error, and reaction-curve error.
- Why: State 10's global damage RMSE is 4.97e-5, but its maximum local error
  is 0.003345 against a neural damage peak of 0.02207. Localized errors need
  to remain visible as damage increases. The tip diagnostic region is explicitly
  a radius-l0 disk about the benchmark's notch tip (0.5, 0.5), not a new BC,
  loss, damage law, or acceptance gate.
- Alternatives rejected: treating low-load RMSE alone as crack accuracy,
  changing primary limits, adding reference supervision, or repeating the
  failed direct undamped budget study during this loading segment.
- Related: active fracture2d PINN task and split_pinn_report.py.

## 2026-10-02 — Benchmark exact analytic pullbacks before more fine-cloud loading
- Decision: User identifies fine-branch runtime as a concern. Pause loading at
  the verified state-20 prefix and benchmark optional analytic reverse rules
  for fixed-geometry A2 energy and material damage matvec, against unchanged AD
  at identical learned fields/networks. Keep FP64, clouds, equations, objectives,
  budgets, physical data and report thresholds unchanged.
- Derivation: A2 energy is sum_e factor_e*density_e; its q gradient is the
  previously verified assembled residual, and factor gradient is density_e.
  For the scalar operator, V D is symmetric, hence D^T z = V D(V^-1 z).
  These replace differentiation through large deterministic segmented scans,
  not the forward physical operator. Verify values, parameter gradients,
  independent finite differences/CSR transpose and single optimizer updates.
- Why: The fine cloud has 71307 material points, 34779392 damage edges and
  9011748 mechanics support occurrences. Generic reverse-mode through these
  scans can dominate update cost; synchronized benchmarks must establish any
  benefit before changing production training.
- Alternatives rejected: shrinking the validated horizon/cloud, silently using
  FP32, weakening gates, reference supervision, claiming a speedup without
  measured warmup-separated GPU times, or guessing an adjoint.
- API provenance: current JAX custom_vjp/defvjp and compiled.memory_analysis/
  block_until_ready documentation verified via Context7 /jax-ml/jax.
- Related: active fracture2d PINN task, planned split_pinn_gradients.py.

## 2026-10-02 — Test deterministic row-padded reductions for the same discrete operators
- Decision: After analytic scan pullbacks yield only 1.13x mechanics / 1.14x
  damage update speedup, benchmark a second explicit layout: fixed-shape padded
  row sums for the exact existing damage CSR entries, mechanics support moments,
  and occurrence-to-particle force assembly. No edges, coefficients, quadrature,
  dtype, constraints or physical equations change; padded entries contribute zero.
- Why: Global associative scans over 35 million scalar edges and 9 million
  mechanics occurrences remain costly. Short row reductions avoid their
  multi-level intermediates and synchronization. Unlike the first reverse-only
  option, this changes floating-point summation order, so compare primal values,
  analytic and AD gradients, and optimizer updates with declared tolerances.
- Alternatives rejected: atomic nondeterministic assembly, graph coarsening,
  hidden FP32, or treating a layout change as a new constitutive model. Retain
  original layout and snapshots; production adoption requires measured benefit.
- Related: split_pinn_gradients.py, split_pinn_benchmark.py, active PINN task.

## 2026-10-02 — Pair same-parent full and reduced budgets after kernel speedups prove modest
- Decision: Keep legacy training unchanged; add an explicit new trial combining
  analytic mechanics pullback and row-padded damage operator. Compare load 21
  from the same verified neural state 20 against the existing 2000-update
  blocks. A separate reduced-budget trial uses 200 mechanics / 500 damage
  updates per block, retaining 3 sweeps and all comparisons/warnings.
- Why: Measured fine-cloud kernel speedups are only 1.12x mechanics and 1.21x
  damage; dense FP64 network work remains substantial. Long blocks already show
  near-flat energy, so reducing redundant updates may provide greater benefit.
  Do not infer equal field quality from speed or gate passage: compare complete
  damage fields, local peaks, reaction errors and warning residuals at equal load.
- Alternatives rejected: claiming a large kernel speedup, silently lowering
  budgets on the old root, using FP32, or extrapolating a small-load budget to
  critical crack growth without validation. Failed trial histories never advance.
- Related: split_pinn.py, split_pinn_fast.py config and paired load-21 trials.

## 2026-10-02 — Retain a measured fast small-load continuation, with accuracy tradeoff explicit
- Decision: Retain the optional fast config (analytic mechanics, padded damage,
  200/500 block updates) for the tested small-load continuation. Its default
  read-only parent is the more accurate full-budget state 21, and it completed
  states 22–23. Keep the full-budget config and all comparison roots.
- Why: At identical load 21 and parent 20, load wall time fell 956.48 -> 321.01 s
  (2.98x), damage RMSE 1.331e-4 -> 1.357e-4, and damage fields differ by 0.1213%
  relative L2. Reaction error worsened 1.176% -> 2.790%, still below 5%; do not
  conceal that tradeoff. Cold model build remains ~8 min, so build-plus-load
  speedup is only 1.84x. Warm state 23 took 113.19 s and passed primary checks.
- Scope: FP64, graph, constitutive data and thresholds are unchanged. This is
  fewer optimization updates plus implementation optimization, not an improved
  fracture formulation or proof that the reduced budget works near failure.
  Same mechanics/AT2/outer-change warnings remain; critical direct load is unresolved.
- Alternatives rejected: a blanket multi-fold kernel-speed claim, claiming
  equivalent training trajectories after changing budgets, adopting padded
  mechanics despite no measured benefit, or loosening the reaction gate.
- Related: fast config, timing.json, paired benchmark/quality JSONs in tf-logs.

## 2026-10-02 — Prepare safeguarded Anderson for the unresolved split-reference critical load
- Decision: Keep mainline PINN loading active; prepare a separately identified
  Type-II Anderson residual strategy for the fixed accepted-history direct map
  G(d) = AT2(max(H_committed, Amor(u(d)))). Use pivoted QR, depth 5, beta 0.5,
  and normalized infinity map residual for candidate selection. Candidate
  backtracking interpolates from the ordinary Picard point toward the Anderson
  proposal, with alpha=1/2^j for j=0..4; never clip damage or accumulate trial H.
- Source: Walker/Ni (2011), DOI 10.1137/10078356X, p.1716 Eq.(1.2), p.1722
  Eq.(3.1) and the following Type-II update, pp.1725–1726 QR/rank discussion.
  The archived reviewed project implementation uses the same QR/update form.
  Backtracking, bounds/physics prerequisites and 0.995 merit factor are explicit
  project numerical safeguards, not a new constitutive model or convergence proof.
- Why: Fixed damped and undamped Picard both failed the split critical load;
  simply adding sweeps has not solved the propagating damage front. Candidate
  map evaluation must include strict mechanics, fixed-H damage and unchanged
  committed history; final coupled and physical reference gates remain intact.
- Alternatives rejected: weakening reference tolerances, committing trial
  history, clipping extrapolated fields, modifying the frozen baseline, guessing
  stabilization, or running another GPU solve concurrently with PINN loading.
- Relation: active fracture2d PINN task, split_reference.py, planned Anderson helper.

## 2026-10-02 — Refine mechanics after the fast mainline load 25 rejects
- Decision: Preserve the completed load-25 rejection in the fast root. Fork
  read-only from its last usable neural state 24 and raise mechanics block
  budget 200 -> 1000, leaving damage budget 500, three sweeps, operator mode,
  material data and all comparisons/warnings unchanged. First endpoint is 30.
- Why: Damage RMSE 0.00017594 and zero drift pass, but reaction error 11.18%
  fails the primary 5% criterion. Give the existing energy optimizer more
  online report opportunities before proposing a different neural objective.
- Alternatives rejected: warning-advancing a failed primary comparator, reusing
  rejected trial history, rewriting load 25, loosening 5%, or continuing the
  inadequate 200-step budget automatically.
- Relation: main PINN task and pinn_split_main_refine_from24_20261002.

## 2026-10-02 — Retain valid Picard secants when critical-load extrapolation is rejected
- Decision: Preserve the completed 120-sweep Anderson rejection. Add an explicit
  rank-only restart option: rejected accelerated trials are never stored, but
  valid evaluated Picard points retain their fixed-map secant history. QR rank
  deficiency still restarts. A new critical-load fork uses beta 1 and numerical
  sufficient-decrease coefficient 0.001, with the same final reference gates.
- Why: The pilot's late histories repeatedly have size 1 and select Picard;
  at sweep 80, an admissible alpha=0.0625 trial reduces merit 0.016031 ->
  0.015954 but misses the fixed 0.995 decrease factor. Sweep 120 remains
  coupled-unconverged (AT2 relative 0.0216895, reaction 363.148). Resetting all
  valid secants on every rejection prevents the declared depth-5 model forming.
- Source: Walker/Ni Eq.(1.2), Eq.(3.1), Section 4: retaining recent fixed-map
  evaluations and controlling QR conditioning. The rank-only restart and
  weaker numerical merit factor are declared safeguards, not physical relaxation.
- Alternatives rejected: overwriting the failed root, weakening PDE/physical
  acceptance, using rejected trial H as committed history, or further unchanged
  Picard-budget escalation.
- Relation: anderson.py, split_reference.py, critical reference retry.

## 2026-10-02 — Allow rank-conditioned longer secant windows at the localized crack front
- Decision: Add an explicit drop-oldest rank strategy, retaining recent
  independent secants instead of resetting the entire window when old columns
  are dependent. The next bounded critical-load trial may use depth 20 with
  rank-only restart and the already declared beta/decrease controls.
- Evidence: The retained depth-5 pilot reaches the nonlinear defect at x~0.672,
  y~0.5, not the original notch tip. Its recent residual-vector correlations
  are 0.968–0.999, with same-sign local growth and accepted extrapolations that
  can retreat the front. This is diagnostic evidence of a difficult coupled
  map, not proof of a zero-energy mode or a changed constitutive law.
- Source: Walker/Ni Section 4, pp.1725–1726 recommends discarding old columns
  to preserve conditioning and application-dependent longer windows. Check
  the drop against an independent reduced-window least-squares calculation.
- Alternatives rejected: retaining rank-deficient QR columns, clipping the
  candidate, committing trial history, or declaring a stable fracture branch
  from a small reaction change alone.
- Relation: anderson.py and reference42_secant_audit_20261002.json.

## 2026-10-03 — User returns the mainline to the old validated kernel
- Decision: Explicitly return the fracture PINN mainline to the original
  3395-node common-horizon A2/Amor/AT2 branch and its verified state-59 prefix.
  Preserve its geometry, physical constants, operator/residual implementation,
  FP64 subproblem kernels and all accepted snapshot bytes. Stop the owned
  split-reference GPU process and retain its generation-pinned progress.
- Why: User asks for minimal repairs on the old stable kernel rather than
  continued new-model development. The old fixed-load-60 36-sweep diagnostic
  already separates inner fixed-H error (0.03089) from relaxation/history lag
  (coupled AT2 0.43069, adjacent reaction change 1.11%). Focus on this concrete
  numerical bottleneck first, not another constitutive/discretization change.
- Scope: No remaining kernel bug is assumed. Missing coupled readiness and
  constant-step optimizer oscillation are audited as numerical limitations;
  fixed thin-band transmission remains a documented model/discrete-support
  property and cannot be silently corrected as a coding bug.
- Alternatives rejected: deleting split artifacts, editing the frozen baseline,
  importing split fields into the old cloud, reference supervision, or claiming
  that minimal numerical repairs guarantee traction-free fracture.
- Relation: active fracture2d PINN task; original stagger/fixed-load diagnostics.

## 2026-10-03 — Audit a force-dual norm without changing the old operator
- Decision: Compare frozen-d mechanics polishing of the same old load-60
  snapshot using original force RMS versus the volume-dual force norm, both
  target-free, FP64 and identical network/optimizer initialization. The norm
  follows virtual-work duality: ||R||_(V^-1)^2 = sum_i R_i^2/V_i with union
  nodal volumes repeated for two displacement components. Normalize each by
  its first-load affine RMS and the same load ratio. A final-quarter linear
  learning-rate taper 1e-4 -> 1e-5 is declared for both comparisons.
- Why: Exact fixed-d mechanics changes old trial-36 H by 15.4563 and the
  fixed-H damage discrepancy from 0.000682 to 0.013629, despite only 0.7006%
  displacement error. Raw integrated-force RMS may underweight tiny-volume tip
  DOFs; compare measured field/driver improvement before changing production.
- Alternatives rejected: importing exact fields as targets, assuming an operator
  bug, changing old geometry/horizon/material law, or using a new physical
  formulation to manufacture reaction decline. This is an optimizer diagnostic.
- Relation: legacy_kernel_audit_20261003.json and frozen mechanics norm probe.

## 2026-10-03 — Retain the old loss and isolate late-stage numerical refinement
- Decision: Do not adopt the tested volume-dual loss: it slightly reduces raw
  force maximum but worsens q/driver errors. Preserve the original force-RMS
  objective and add only an optional final-quarter update taper, plus a separate
  numerical-trial seed distinct from the accepted parent q/d/H context.
  Continue the completed old 36-sweep trial in a new fixed-load root, retaining
  the exact state-59 committed history and recording explicit coupled readiness.
- Why: Paired 5000-update probe ends with H error 15.39 for original loss versus
  15.95 for volume-dual. The tapered original-loss reaction settles 639.569 near
  exact fixed-d 639.417, whereas constant-step reports had swung 638.35–642.19.
  Taper improves numerical quietness, not demonstrated full-field accuracy;
  the coupled experiment must establish whether it helps convergence.
- Scope: Trial networks/optimizer/damage iterate are initial guesses only; trial
  H never becomes the committed floor. Readiness requires declared residual and
  adjacent-field/reaction checks and does not retroactively change old acceptance.
- Alternatives rejected: changing the stable operator, adopting a worse norm,
  committing an unconverged trial to resume it, or reporting optimizer smoothing
  as a corrected constitutive law or guaranteed crack convergence.
- Relation: stagger.py, planned legacy_coupling_refine.py, preserved old diagnostics.

## 2026-10-03 — Continue fixed-load old-kernel refinement after a measured improvement
- Decision: Preserve the completed 12-sweep refinement and start a separate
  36-sweep numerical continuation from its trial, retaining original state-59
  committed q/d/H. Keep the old objective/operators/stage budgets/omega; declare
  final-quarter taper ratio 0.01 (1e-6 terminal rate) instead of 0.1.
- Evidence: Reaction error 0.393% -> 0.0211%, adjacent reaction change
  1.1138% -> 0.002389%, damage RMSE 0.00089945 -> 0.00080242 and coupled AT2
  relative 0.00135370 -> 0.00085042. Readiness remains false for field/history
  changes and AT2; no trial history advances to a new load.
- Reason: Reduce remaining optimizer noise while continuing the same concrete
  residual/lag correction; do not promote reaction accuracy to full convergence.
- Alternatives rejected: loosening the newly declared readiness limits, rewriting
  the completed experiment, changing the physical kernel, or loading trial H
  as the committed lower bound.
- Relation: legacy_coupling_refine.py and old fixed-load numerical forks.

## 2026-10-03 — Test removal of residual relaxation lag on the same old fixed load
- Decision: Preserve the 36-sweep stronger-taper result, which remains not
  coupled-ready. Test an explicit numerical coupling omega 1 in a new 12-sweep
  fork from its complete trial, keeping original omega 0.5 as the legacy default,
  operators/objectives/stage budgets, accepted state-59 history and all limits.
- Evidence: Final AT2 inf 0.0762087 decomposes into relaxed fixed-H 0.0581883
  versus solved fixed-H 0.0222866 plus current-history shift 0.0180204. The
  damping lag still exceeds the inner fitting error. Reaction is already stable
  (relative reference error 0.00723%, adjacent change 0.002995%) with zero drift.
- Derivation: Nonzero omega changes the fixed-point iteration, not its stationary
  equation d=G(d). Omega 1 directly uses the neural fixed-H image; validate
  boundedness/history/field convergence instead of assuming stability.
- Alternatives rejected: relabeling readiness, clipping damage, committing seed
  H, changing the stable physical kernel, or continuing an unchanged budget.
- Relation: old stagger's optional numerical mixing selector and refinement wrapper.

## 2026-10-03 — Keep readiness limits while finishing the old fixed-load correction
- Decision: Preserve the completed omega-1 refinement and continue a new
  eight-sweep trial from it with identical numerical settings and original
  state-59 history floor; do not relax readiness or advance load 61 yet.
- Evidence: AT2 relative 6.6386e-5 passes 1e-4; damage/H/reaction changes and
  physical checks pass. Only q adjacent change 1.1144e-4 is above its 1e-4
  threshold. Reaction error is 0.0103%, damage RMSE 0.00074088, zero drift.
- Why: A short unchanged correction from a new numerical iterate tests the
  remaining near-boundary stationarity rather than redefining convergence.
- Alternatives rejected: post-hoc threshold relaxation, calling the whole
  fracture path solved, or committing trial H as the physical lower bound.
- Relation: old-kernel fixed-load finishing trial and readiness reports.

## 2026-10-03 — Stop mechanics-after only on same-pass coupled readiness
- Decision: Add an optional report-time stop for the mechanics-after inner
  block, requiring every predeclared coupled/adjacent-field check and the old
  physical/mechanics gates at that exact state. Keep all thresholds, objective,
  architecture and operators unchanged; convergence still requires three
  completed outer sweeps. This is a separately identified old-kernel trial.
- Why: The fixed-count finish misses readiness again: q change 1.0947e-4 and
  current-H AT2 1.5770e-4, despite fixed-H residual ~6.72e-5 and small d/H/R
  changes. Inner mechanics work can move the history after the damage solve;
  blind fixed-count endpoints are not a coupled stopping rule. Do not relax
  thresholds or call an earlier unrecorded endpoint converged retrospectively.
- Alternatives rejected: post-hoc selection, freezing the whole displacement
  field to manufacture zero changes, committing unready fields, altering the
  physical kernel, or unbounded unchanged sweep escalation.
- Relation: stagger.py optional after_stop callback and legacy refinement wrapper.

## 2026-10-03 — Validate the required three outer confirmations without changing limits
- Decision: After the coupled-report-stop trial ends with every readiness check
  passing for two consecutive outer sweeps, run a separate four-sweep continuation
  with the same settings. Four is the minimum existing experiment budget above
  the original three-sweep parent; the first adjacent check remains unavailable.
- Evidence: Latest AT2 relative 6.4320e-5, q change/load 7.4818e-5, damage change
  3.5183e-5, H change 1.0375e-4 and reaction change 2.6956e-5 all pass, but
  consecutive count 2 < 3. Preserve false readiness rather than changing counts.
- Alternatives rejected: importing a confirmation counter without predecessor
  binding, post-hoc acceptance, weakening three confirmations, or changing physics.
- Relation: final old-kernel coupled-confirmation run from numerical iterate 112.

## 2026-10-03 — Advance the old kernel only from verified coupled-ready diagnostics
- Decision: Implement a separately bound single-load warning continuation from
  ready state 60 to 61, reusing the original stagger core and physical operators.
  Recheck the bound diagnostic fields and its last three full-sweep readiness
  reports against unchanged limits before making its neural H/d the next-load
  history floor. A same-load numerical seed remains separate and cannot commit H.
- Why: The user requested the next step after the fixed-load correction reached
  three confirmations. Existing warning continuation allows physically usable
  but uncoupled parents; the fixed-load correction entry cannot advance loads.
- Alternatives rejected: retraining state 60, inheriting the old uncoupled
  warning path, using reference H/d as a floor, relabeling strict acceptance,
  changing the kernel, or advancing past a non-ready diagnostic.
- Relation: legacy_coupling_refine.py, diagnostic_continuation.py and the active
  fracture2d-nonlocal-pinn task. Use an explicit 16-sweep initial state-61 budget
  with the demonstrated quarter-tail ratio 0.01, omega 1 and same-pass after-stop.

## 2026-10-03 — Bound the measured load-61 outer correction at 32 further sweeps
- Decision: Preserve the completed 16-sweep state-61 experiment and use it only
  as a same-load numerical seed in a new 32-sweep root. Keep neural state 60 as
  the committed H/d floor and all operators, budgets, taper, mixing and limits.
- Evidence: AT2 relative fell 0.0146814 -> 0.00185121, damage RMSE 0.00413883
  -> 0.00137089, reaction error 0.8575% -> 0.2002%, with zero drift. Fixed-H
  residual is 0.02298 versus current-history 0.63112; history shift dominates.
  Observed contraction is about 0.87/sweep over 1..16, not a proof of future
  convergence. Thirty-two additional sweeps provide a bounded test of reaching
  the existing 1e-4 AT2/q/d and 1e-3 H/reaction limits with three confirmations.
- Alternatives rejected: advancing to 62 while unready, weakening thresholds,
  committing the seed's H, changing the kernel/loss, or open-ended budget growth.
- Relation: legacy_coupled_loading.py same-load seed and active load-61 task.

## 2026-10-03 — Test only the remaining load-61 displacement stationarity
- Decision: Preserve the completed 32-sweep correction and run one short
  eight-sweep same-policy confirmation from it, again with state-60 H/d floor.
  Do not advance to 62 or loosen the displacement change/three-pass criteria.
- Evidence: At cumulative sweep 48, AT2 relative 6.63504e-5, damage change
  4.97074e-5, H relative 3.85078e-4, reaction relative 5.30874e-8 and physical
  gates all pass. Only q change/load 1.65531e-4 exceeds 1e-4. This is now a
  single near-threshold field-stationarity deficit, rather than the prior
  order-of-magnitude outer/history residual. Reaction error is 0.10634%,
  damage RMSE 0.00105371, zero drift, readiness remains false.
- Alternatives rejected: treating the reaction/AT2 passage as full readiness,
  moving the q threshold, freezing displacement to manufacture zero change,
  inheriting trial H, or an unbounded unchanged-budget loop. If this bounded
  confirmation fails, record the actual field/optimizer blocker before another
  numerical policy is selected.
- Relation: legacy_coupled_loading.py state-61 finishing confirmation.

## 2026-10-03 — Stop load-61 outer budget growth at the localized q deficit
- Decision: Keep the completed 56-sweep state-61 checkpoint diagnostic-only and
  refuse advancement to 62. Next isolate fixed-d mechanics/driver stationarity
  with the same original network/objective and state-60 history floor before
  selecting another numerical correction. Preserve all current thresholds.
- Evidence: Final q/load 1.33009e-4 remains above 1e-4; the maximum is the
  notch-tip y displacement at particle 1868. Other readiness gates pass, AT2
  relative is 6.26948e-5, reaction error 0.10383%, damage RMSE 0.00104875,
  zero drift/H decrease. Tip changes keep one sign while some whole-field
  changes reverse; force loss continues decreasing. Fresh original-kernel
  checkpoint/physics replay succeeds and the next-load reader actually refuses
  the unready parent. No new kernel/BC/serialization defect is demonstrated.
- Alternatives rejected: moving the q tolerance, reducing/freezing updates
  simply to create small changes, another unchanged outer-budget escalation,
  committing unready H, or interpreting accurate scalar response as convergence.
- Relation: final state-61 parent_replay.json, displacement-change plots and
  docs/tasks/fracture2d-nonlocal-pinn.md exact fixed-d diagnostic resume point.

## 2026-10-03 — Resume with a bounded state-61 fixed-d mechanics diagnostic
- Decision: On the user's request to continue solving, restore the verified
  numerical iterate 56 and its Adam state, fix its damage, and run three
  original 16k mechanics blocks with the same residual objective, quarter-tail
  ratio 0.01 and state-60 history floor. Use an exact fixed-d solve only for
  diagnostics, never as neural training targets or an initial field.
- Reason: Separate slow inner mechanics evolution from changing-d coupling
  before choosing a local numerical correction for the remaining q criterion.
- Alternatives rejected: another blind outer-budget extension, changing the
  physical kernel/limits, committing trial H, or freezing q to fake readiness.
- Relation: active fracture2d-nonlocal-pinn task and original FP64 backend.

## 2026-10-03 — Compare fixed-d original-residual L-BFGS before integration
- Decision: Run a separate bounded FP64 L-BFGS-B probe on the same original
  normalized force-MSE objective at fixed state-61 damage, starting from neural
  iterate 56. Explicit controls: maxiter 1000, maxfun 5000, maxls 40, maxcor 30,
  ftol 1e-15, gtol 1e-10. Exact fields are comparison-only, no training targets.
- Evidence: Three original 16k Adam blocks improve loss only 4.46110e-6 ->
  4.40275e-6 (1.31%); q-vs-fixed-d-exact L2 remains 0.537%, driver inf error
  remains ~39.95. This isolates poor inner optimization/representation progress,
  not solely outer damage lag. A line-search numerical comparison is justified.
- Alternatives rejected: lowering updates to fake q stability, changing the
  kernel/objective, or deploying an untested optimizer into coupling. No L-BFGS
  result is accepted as a physical/coupled state by this probe.
- Relation: SciPy L-BFGS-B documentation verified via Context7 and official API;
  existing inner_probe.py already uses this API for a different objective.

## 2026-10-03 — User explicitly chooses warning continuation to inspect trends
- Decision: Add explicit --warning-continue to the old single-load entry and
  run diagnostic loads 62..65 with four original-budget sweeps each. The user
  selected warning continuation rather than requiring q stationarity first.
  Coupled/adjacent failures are retained unchanged as warnings; the original
  finite, mechanics, damage range/healing, history and zero-drift gates remain
  hard stops. All states and ancestry remain diagnostic-only, never strict.
- Reason: Obtain the requested damage/reaction trend from the corrected old
  neural prefix, while accurately disclosing uncoupled history transitions.
- Alternatives rejected: silently bypassing the ready-parent guard, relaxing
  thresholds, using reference H/d, adopting the inconclusive L-BFGS probe, or
  claiming strict fracture/unloading validation from these warnings.
- Relation: explicit user answer to the continuation-mode question, original
  diagnostic_continuation policy and legacy_coupled_loading.py. L-BFGS exhausted
  1000 iterations with only ~0.12% force-loss improvement; it is not adopted.

## 2026-10-03 — Continue warning loading to observe reaction decline
- Decision: On the user's explicit further-solving request, restore verified
  diagnostic state 65 and extend the same original-kernel warning path through
  states 66..75 or the first physical/mechanics hard stop. Keep four 16k/12k/16k
  sweeps, quarter-tail 0.01, omega 1, all physical gates and all warning labels.
- Reason: Determine whether the slight observed 0.0064->0.0065 reaction decline
  continues at higher prescribed displacement; provide actual trend/results.
- Alternatives rejected: tuning to force unloading, relaxing drift/bounds,
  adopting optimizer probes, changing the frozen kernel, or promoting warnings
  to strict fracture validation.
- Relation: active task, user-authorized --warning-continue contract and
  state-65 diagnostic SHA 189b7d068b4e7a006f63c6da6aa19bcaa52c351437e861b9314ce120730a7474.

## 2026-10-03 — Retain the original drift hard stop after observed softening
- Decision: Stop at rejected state 70 and retain state 69 as last usable warning
  parent. Do not continue 71..75, clip/remove its drifting particle, or reinterpret
  the authorization to warn about coupling as permission to relax physical gates.
- Evidence: Usable reaction falls 7.58054% from sampled 651.5889 peak to
  602.1949 at 69. Candidate 70 gives 582.4343 but one node has norm/load
  1.08363225 >1.08, independently reconstructed from the saved full field.
- Reason: Answer the user's reaction-decline question using usable warning
  observations, without promoting a hard-rejected endpoint or hiding drift.
- Alternatives rejected: threshold enlargement, skipping 70, committing trial H,
  calling the sampled decline validated fracture, or treating mixed/drifting
  archived reference labels as strict truth.
- Relation: legacy_warning_decline_20261003.json, rejected_field_audit.json and
  active task exact parent/retry point.

## 2026-10-03 — Diagnose the reaction gap before selecting a repair
- Decision: Prioritize common-parent, fixed-load diagnostics over further
  warning loading or reaction-curve tuning. Separate inner neural mechanics,
  fixed-history damage, outer/history lag, and drifting-reference/model effects.
  No new numerical/physical formulation is adopted in this analysis session.
- Reason: State-69 fixed-H AT2 inf 0.085985 versus current-H 46.663634 and
  history-shift 46.663074 directly identify a large outer mismatch. Fixed-d
  state-61 neural q L2 error ~0.537% coexists with H inf error ~40. The archive
  unloads through drifting/warning ancestry, so agreement is not strict truth.
- Alternatives rejected: treating peak d or small average force MSE as proof
  of fracture convergence, training on archived fields/reactions, attributing
  a same-kernel PINN/direct discrepancy solely to horizon physics, or assuming
  another Adam/L-BFGS budget solves the demonstrated plateau.
- Relation: active task, coupling_0004.json, fixed-d probe.json and original
  strip_transmission.json. Proposed exact/hybrid subproblem comparisons are
  diagnostic benchmarks, not neural supervision or an adopted hybrid solver.

## 2026-10-03 — Execute the common-parent subproblem diagnostic matrix
- Decision: On explicit approval, compare all four direct/neural q/d combinations
  at independent fixed loads 61/62 from verified neural state 60. Reset every
  case to the same networks/Adam and q/d/H floor. Use original damped mixing
  0.5 in all cases, eight matched outer sweeps; direct/direct has a separately
  labeled maximum 64-sweep convergence budget. Original neural 16k/12k/16k,
  quarter-tail 0.01, residuals and physical gates are retained.
- Reason: Isolate solver substitutions without confounding different committed
  histories, relaxation factors, geometry, or reference-supervised fitting.
- Direct equations: use original FP64 A2 mechanics and volume-multiplied fixed-H
  AT2 SPD equation; GPU Jacobi-CG uses tol 1e-11, atol 1e-13, maxiter 10000 and
  explicit residual verification (JAX cg info is None, not a convergence flag).
  Mechanics keeps 1e-9/1e-10/10000. A CPU sparse solve is a one-time independent
  validation only, never a numerical fallback or production field.
- Alternatives rejected: advancing 62 from case-dependent 61 histories,
  comparing unmatched final budgets as matched causality, oracle field labels,
  hidden CPU fallbacks, and changing the frozen kernel. A hard/numerical failure
  stops that case, not the other independently initialized cases.
- Relation: active task and approved plan; original AT2 derivation at
  nonlocal_baseline.solve_fixed_history_damage, JAX cg official API verified.

## 2026-10-03 — Prioritize outer path stability and neural mechanics after matrix evidence
- Decision: Do not adopt a solver/physics rewrite or target-fit the archive from
  these diagnostics. Next establish a converged direct-61 common-floor baseline,
  then genuine small-step sequential loading and bounded outer stabilization;
  address neural tip strain/driver accuracy alongside it. Fixed-H neural damage
  is not the leading demonstrated bottleneck at the tested states.
- Evidence: At matched sweep 8, direct-q/neural-d versus direct/direct reaction
  differences are 8.71e-6 / 3.38e-6 at 61/62 and damage image errors ~6e-5.
  Neural-q cases have q L2 ~0.62% but H errors ~23.17 / 35.87, primarily negative
  near the tip. Direct/direct 62 hits drift at sweep 29 with accurate inner
  solves and coupled AT2 relative 0.359. Direct/direct 61 does not drift within
  64 but remains above research readiness, so it is not a converged baseline.
- Limitations: Load 62 independently starts at the same neural state-60 floor;
  this diagnoses a specific damped-Picard path, not all possible physical loading
  paths or nonexistence of equilibrium. Matched 8-sweep data and longer direct
  endpoints remain explicitly separated. No strict/reference validation result.
- Alternatives rejected: blaming all differences on the damage network,
  attributing drift solely to NN optimization, declaring exact-inner solves to
  guarantee coupled convergence, or inferring a model/zero-energy defect without
  continuation/stability controls.
- Relation: subproblem_matrix.py and legacy_subproblem_matrix_20261003/matrix_report.json.

## 2026-10-05 — Tighten direct-61 and test genuine sequential step sizes
- Decision: On the user's continuation request, warm-start the saved direct-61
  sweep-64 q/d under the original neural state-60 H/d floor, with up to 192
  additional Picard sweeps (mixing 0.5). Require all declared AT2/d/q/H/reaction
  dimensionless defects <=1e-6 and original hard gates for three consecutive
  sweeps. This is a tighter diagnostic contract, not the complete frozen direct
  archive acceptance contract or validation of the neural history ancestry.
- Decision: Only if the baseline passes, independently start three genuine
  sequential paths from it to load 0.0062: du 1e-4, 5e-5, 2.5e-5; max 256
  sweeps per subload, commit each path's H/d only after the same convergence
  contract. Keep failed trial history uncommitted and all physics gates unchanged.
- Reason: Distinguish the earlier independent state-60->62 probe from actual
  converged loading, and test whether step size changes the near-critical failure.
- Alternatives rejected: silently adopting trial H at baseline 64, warning
  commits during this controlled study, changing drift/material/kernel, or
  calling a tight diagnostic baseline a fully validated reference.
- Relation: direct_path_probe.py, frozen A2/AT2 primitives and common-floor matrix.

## 2026-10-05 — Compare safeguarded Anderson at the finest failed subload
- Decision: After tight baseline 61 passes and all three Picard step sizes
  hard-stop on their first subload, add one bounded existing-Type-II-Anderson
  diagnostic at .006125 from the same converged baseline. Max 128 steps,
  depth 5, beta .5, sufficient decrease 1e-4, eight backtracks, reject_or_rank
  restart and drop_oldest rank strategy; all physical/convergence limits unchanged.
- Reason: Distinguish a damped-Picard map-iteration limitation from an absent
  admissible algebraic solution under the same original equations/history.
- Map: d -> direct q(d) -> max(H_parent,psi_plus(q)) -> GPU fixed-H AT2 image;
  candidates are never clipped/projected. Existing helper's explicit Picard
  fallback and candidate audits are part of the declared numerical method.
- Alternatives rejected: another smaller-step-only budget increase, relaxed
  drift bounds, new constitutive law, or treating an Anderson root as physical
  energy/path stability or full reference validation without further checks.
- Relation: anderson.py audited Walker/Ni formulation; direct_path_probe results.

## 2026-10-05 — Preserve converged baseline; do not promote stalled stabilized trial
- Decision: Retain the tight direct-61 diagnostic baseline as the only new
  convergence-gated parent. Do not commit any of the three failed Picard trials
  or the unready Anderson trial. Next investigate the coupled root/residual and
  local stability/branch behavior under unchanged equations before selecting
  another solver or model correction.
- Evidence: Baseline AT2 relative 6.36e-7 and all changes pass 1e-6 three times.
  Smaller Picard steps delay but do not remove one-node drift; all first targets
  fail. Anderson selects 56 accelerated/72 Picard steps, remains non-drifting,
  but residual plateaus near 1.98e-3 and does not meet the tight contract.
- Reason: Avoid interpreting finite stable-looking iterates or accurate scalar
  reactions as self-consistent fracture states. No evidence proves absence of
  equilibrium or a specific zero-energy/model defect from these controls alone.
- Alternatives rejected: warning-committing this controlled study, drift-limit
  enlargement, another unbounded unchanged-budget loop, or declaring Anderson
  a completed remedy without convergence/stability validation.
- Relation: legacy_direct_path_v2_20261005/replay_report.json and explicit
  Walker/Ni safeguarded-method control; original kernel remains frozen.

## 2026-10-05 — Reassess supplied Sinha SGE fracture paper against current blockers
- Decision: Treat the user's supplied full paper (JMPS 208, 106458,
  DOI 10.1016/j.jmps.2025.106458) as a relevant homogeneous nonlocal fracture
  comparator; no SGE, split, penalty, horizon or optimizer replacement is adopted.
- Evidence: The paper was already reviewed on 2026-08-18 (archived direct task,
  SGE model clarification). Sections 5.1/5.2 use uniform continuum parameters;
  explicit heterogeneous unit cells motivate/calibrate the earlier study.
- Reason: Equations 7–12 and 20 suggest auditing common-energy consistency and
  crack-side support before selecting a new method. Current mechanics degrades
  full elastic energy, damage uses the positive Amor maximum history, and the
  graph damage energy differs from SGE gradient-squared energy. Individually
  checked subproblem energies do not establish one joint variational potential.
- Caveats: Derive direct equations with quadrature-weighted adjoints; printed
  residuals 22/23 omit the full nonlocal adjoint/penalty and Eq.23 is AT2-like
  despite the SEN AT1 description. Radius below ell alone does not prove
  crack-side visibility. Eq.19 has tails outside [0,1]; log of total energy
  does not independently rebalance component weights or guarantee conditioning.
- Alternatives rejected: copying residuals/BCs or assuming published curves prove
  current gates, using horizon calibration to fit the warning archive, replacing
  history with a penalty as a supposedly equivalent numerical-only change, or
  declaring SGE free of zero modes without analysis.
- Relation: User paper inquiry; active fracture2d task. Recommended next audit
  distinguishes original-map numerical convergence from an explicitly reviewed
  alternative unified split-energy/irreversibility model; failed trial H remains
  uncommitted and the original frozen kernel remains unchanged.

## 2026-10-05 — Execute user-selected paired original A2 objective study
- Decision: At fixed neural state-61 damage, reset both arms to the SHA-bound
  iterate-56 network and fresh Adam. Compare normalized original free-force MSE
  against original A2 elastic energy normalized by fixed-d affine energy; one
  original 16000-update block, same learning rate and quarter-tail schedule.
- Reason: User explicitly selected audit/energy training before a new model.
  Fresh Adam in both arms avoids importing residual-specific moments into energy
  training. This changes the neural objective, not fixed-d physical equilibrium.
- Audit: verify field energy gradients, volume-weighted fixed-H damage energy,
  and mixed directional derivatives of original mechanics versus frozen/current
  history damage. Report both current driver and cumulative-history errors near
  the notch and globally. Direct fields remain comparison-only, never labels.
- Alternatives rejected: simultaneous optimizer/architecture/model changes,
  transferring trial H, cherry-picking matched endpoints, weakening drift gates,
  or interpreting fixed-d training as coupled fracture convergence.
- Relation: seed 4375c3519f07d6966e3683b756a9ea6c79c6400e587fa4d172bf430a09fab8f8,
  actual state-60 parent dda0d1cd2c79bf6a2770442f221f742e916d3c06d4502ecbe9179bd90084179e.

## 2026-10-05 — Retain partial energy-objective improvement without solver adoption
- Decision: Preserve the completed fixed-d pair as diagnostic evidence; do not
  promote its energy network to a coupled checkpoint or change the production
  objective/formulation. Any unified-energy branch requires explicit review.
- Evidence: Energy versus residual at matched 16k reduces q L2 error 47.96%
  and peak driver error 8.75%, but raises force MSE 22.07% and retains ~9.78%
  tip driver underprediction. Both endpoints are non-drifting; all 34 reports
  and final networks independently replay under the original physical backend.
- Audit evidence: Individual energy gradients match their equations, but the
  volume-weighted current-H tip cross derivative is -.0100190 versus mechanics
  +.0257525; independent finite differences agree. The original residual pair
  is not one differentiable u/d energy gradient at this state/physical pairing.
  This is not proof of the cause of drift or nonexistence of equilibrium.
- Alternatives rejected: using reaction error .001958% to claim accurate crack
  driving forces, comparing fixed-d AT2 values as coupled convergence, calling
  raw total-energy minimization equivalent to the original history formulation,
  or blindly increasing unchanged training budgets after modest local progress.
- Relation: legacy_objective_pair_20261005/replay_report.json, first-stage paper
  study and current task. Future original-residual polish needs a fresh optimizer;
  any trial neural H remains uncommitted. No numerical job remains running.

## 2026-10-07 — Distinguish horizon scale separation from fracture disconnection
- Decision: Interpret a mechanics horizon below a declared interface width as
  a scale-separation check, not a sufficient guarantee of traction-free crack
  faces or a remedy for drift. Distinguish horizon radius, nominal local AT2
  full width at d=exp(-1), actual weak/failed-band width, direct graph edges,
  shared BA support DOFs and residual-stiffness transmission.
- Reason: Original delta_u=delta_d=.048 equals the nominal width 2*l0=.048,
  rather than being smaller. Source-only degradation leaves failed-node
  displacements in intact-source kinematics. The saved coarse split probe has
  delta_u=.016 < fully damaged width .024 but reaction/intact=.01891997;
  width .032=2*delta_u lowers it to 5.81344e-7. These checked prescribed-field
  probes refute a universal no-transmission claim from delta_u < width alone.
- Derivation boundary: A positive geometric gap w between two sets prevents
  direct strict-radius edges if delta_u <= w. That does not remove intermediate
  material nodes or shared support variables. For an ideal planar fully failed
  separating band, opposite intact-source radius-delta_u neighborhoods cannot
  share a point when w >= 2*delta_u; this geometric overlap bound is not a
  universal phase-field convergence/traction law. The adopted k_res>0 also
  retains finite stiffness. Actual diffuse AT2 width is not a void/failed-band
  width; delta_d controls damage regularization, not the mechanics edge radius.
- Alternatives rejected: adopting delta<l0 as a universal theorem, equating
  nominal 2*l0 with a fully failed strip, claiming graph disconnection from
  small horizons, or attributing all observed drift to this ratio. No horizon,
  graph, support, solver or acceptance change is adopted.
- Relation: user principle inquiry; nonlocal_baseline.py::_center_weights,
  _edge_factor and _deformation_gradient; existing degradation/split transmission
  reports; docs/tasks/fracture2d-nonlocal-pinn.md.

## 2026-10-07 — Recommend an isolated damage-radius cost/accuracy study
- Decision: Recommend keeping l0=.024, the physical grip extent .048 and other
  constitutive/loading choices fixed. First compare delta_d=.024 to the existing
  .048 on the validated fine h=.004/delta_u=.008/h_grip=.004 geometry. This is a
  proposed new-model study, not adoption or authorization to run continuation.
  If its damage/fracture checks pass but cost remains high, a uniform h=.006,
  delta_u=.012, delta_d=.024 candidate is a separate unvalidated coarsening study.
- Reason: Fine mechanics has checked rank/energy/work and favorable prescribed
  strip transmission, whereas the fine damage graph contains 34779392 edges
  versus 1122228 mechanics edges. Halving delta_d on the same approximately
  uniform 2D mesh predicts about one quarter of damage edges, not a fourfold
  whole-solver speedup. It isolates one model parameter before changing the mesh.
- Cost derivation: fixed-area quasiuniform 2D clouds have N proportional to
  h^-2, neighbors proportional to (delta/h)^2 and scalar graph storage/work
  proportional to delta_d^2*h^-4. BA support-occurrence work additionally counts
  common-neighbor occurrences, typically proportional to N*z_u^2. These are
  geometric array-work estimates; CG/outer iteration counts, JIT/build/I/O and
  actual timing require measurements. Existing coarse/fine damage-edge ratio
  is 15.934, illustrating the cost of widespread refinement at fixed delta_d.
- Validation boundary: l0 and nominal 2*l0 are not actual failed-band width.
  The mechanics diameter must be assessed against measured weak-band/support
  transmission, not made acceptable by artificially thickening the damage field.
  Delta_u/h=2 has operator evidence on the current fine cloud, not a universal
  convergence guarantee. New delta_d affects the damage profile and fracture
  response; changed horizons require a fresh reference, and mesh convergence
  must be checked with horizons fixed to distinguish the two effects.
- Alternatives rejected: widening l0 for runtime, reducing delta_u below mesh
  resolution, using far-field h>=delta_u without BA checks, claiming a twentyfold
  total-runtime saving from an edge estimate, or reusing old accepted histories
  as a validated prefix on a changed model. No config/solver/gate changed.
- Relation: user accuracy/runtime inquiry, saved split validation/benchmark
  reports and docs/tasks/fracture2d-nonlocal-pinn.md.

## 2026-10-07 — Execute approved configuration B direct validation/loading
- Decision: User explicitly approves the proposed B study and full direct solve.
  Use the existing SHA-bound fine cloud, h=.004, delta_u=.008, delta_d=.024,
  l0=.024, h_grip=.004 and grip extent .048, unchanged material/kernel/BC/history
  equations. Validate operators/work, prescribed failed-band transmission and
  fixed-H damage first, then load from zero to .0075 with du=1e-4, Picard omega=.5,
  up to 1000 sweeps/load and existing strict three-confirmation gates.
- Reason: Isolate the approved damage-radius change and test an actual loading
  path rather than promoting prescribed-strip or old-model histories.
- Implementation choice: Existing split_reference uses GPU mechanics but SciPy
  CPU damage PCG. Add an explicit GPU damage selector reusing the independently
  validated volume-SPD CG helper; retain the legacy CPU path/identities. Verify
  the GPU result against an independent CPU solve as a diagnostic only, never a
  fallback. Stop and retain physical/numerical failures and uncommitted trials.
- Alternatives rejected: warning commits, reusing A/old neural history as B's
  committed floor, simultaneously coarsening to C, widening l0, concealed CPU
  fallback, or claiming full physical accuracy from operator gates alone.
- Relation: current user execution request, scale_separation_fine configuration,
  split_reference.py and subproblem_matrix.damage_solver; active fracture task.
- Implementation detail: Move the checked CG helper into lightweight at2_cg.py
  and re-export it from subproblem_matrix, avoiding PINN CLI flag registration
  when the direct validation entry imports it. Picard now stops immediately on
  the same existing physical failures, matching the documented hard-stop contract;
  no threshold changes. Ten focused CPU reference/CG/geometry tests pass.

## 2026-10-08 — Keep direct verification ahead of configuration B PINN
- Decision: User explicitly reiterates completing the direct verification first,
  then proceeding to PINN only after it passes. Continue the existing B GPU job;
  independently check saved fields/ancestry/residual/history/reaction/physical
  gates before using its results as a PINN comparator. A rejected/unready
  critical direct state cannot become a PINN reference or training history floor.
- Reason: Numerical model changes require a new checked direct path; successful
  operator or low-load checks do not establish full fracture accuracy.
- Alternatives rejected: concurrent PINN loading, transferring old A history,
  advancing PINN against a rejected B state, or treating warning continuation
  as fulfillment of the requested direct-first verification.
- Relation: current user follow-up and active B worker PID 4731; same active task.

## 2026-10-08 — Retain B Picard failure; bound one critical-load Anderson retry
- Decision: B Picard stops at load .0051/sweep574 for one drift node (ratio
  1.08003862 >1.08), with AT2 relative .0921154. Keep strict B states 1–50 and
  their independently replayed fields; no trial51 H enters the committed floor.
  User's continued direct-solving request justifies one existing safeguarded
  Type-II Anderson retry at the same .0051 target, at most 128 outer steps,
  beta .5, depth5, four backtracks, sufficient decrease1e-4, rank_only restart
  and drop_oldest rank strategy. All strict/physical thresholds remain unchanged.
- Initial guess: SHA-bound physically passing trial398 q/d only (ratio1.04129,
  relative AT2 .0850226), with actual committed state50 H/d and reset secants.
  Earlier physically passing iterate avoids using the rejected drift field.
  It is a numerical guess, not an accepted root or committed history.
- Reason: Accurate inner solves and reaction decline coexist with substantial
  moving-front defects. Test already-reviewed map acceleration rather than
  silently committing the softening path or enlarging the unchanged budget.
- Alternatives rejected: relaxing drift/coupling gates, skipping .0051, transferring
  trial H, switching material/horizons, launching PINN or claiming full direct
  completion. Retry success must be field-replayed before extending the endpoint.
- Relation: B result.json/loading_replay.json, audited anderson.py/split_reference
  implementation, explicit user request to finish direct before PINN.

## 2026-10-08 — Configuration B full-path verification remains failed
- Decision: Keep B's strict direct prefix 1–50 only. Both critical .0051 trials
  are rejected: Picard574 has one drift node and AT2 relative .0921154;
  safeguarded Anderson128 has zero drift but AT2 relative .0563813 and nonzero
  field changes. Do not advance to52, commit either trial H, or start B PINN.
- Evidence: All 50 accepted fields replay with SHA/ancestry/BC/history/reaction,
  AT2 and recomputed free-force checks. Independent CPU reconstruction of B's
  8585162-edge damage operator reproduces the two terminal residuals and the
  accepted state50 residual. Separate saved-array audit reproduces the Picard
  drift point35120. Both workers finished; no GPU job remains.
- Reason: Shrinking delta_d achieved smaller graph storage and accurate inner
  equations, but did not establish self-consistent high-damage fracture. High
  damage, falling reaction and non-drifting iterates are insufficient acceptance.
- Alternatives rejected: treating the small drift-envelope excess as the only
  issue, accepting the unready Anderson endpoint, repeating unchanged budgets,
  claiming a verified .0075 path or an equal-accuracy whole-solver speedup.
- Relation: B loading_replay.json, rejected_field_audit.json and independent
  terminal_AT2_replay.json; direct-first user instruction. Resume only from B
  state50 committed.npz, SHA26b355a8..., for coupled-root/support investigation
  before choosing a further reviewed correction.

## 2026-10-08 — Audit storage pressure before removing numerical artifacts
- Decision: Prioritize storage relief within the same active fracture task; no
  new numerical run until retention is resolved. Prepare a dry-run proposal,
  not deletion: keep every committed solution, current progress generation,
  first/last4/every50 trial fields and explicitly referenced diagnostic seeds.
  Keep JSON reports/configs/plots and original inputs. Ask user to select cleanup
  scope before deleting intermediate fields or clearing package cache.
- Evidence: Root filesystem59GiB is98% used with~1.4GiB available. B Picard root
  is4.4GiB (2070 trial fields), B Anderson861MiB (128 fields), old fine Anderson
  roots total~1.6GiB. Read-only proposal covers5 direct roots, protects50 committed
  SHAs and55 progress SHAs, and identifies2110 redundant trial files totaling
  5.758GiB allocated (B-only4.378GiB). No files have been deleted.
- Cache boundary: uv cache occupies~11GiB; current .venv has no symlinks into
  it, but shared installed files mean cache size is not all reclaimable space.
  Official uv docs distinguish cache prune/clean and forbid manual cache edits;
  use uv's own command only if selected, then check imports/GPU and actual df.
- Alternatives rejected: removing committed/reference/mesh/PINN histories,
  deleting whole run roots, lossy FP64 conversion, recompressing existing NPZ
  archives while nearly full, or promising11GiB reclaimed from shared cache.
- Relation: user's disk-space concern; /tmp/opencode/fracture_storage_plan_20261008.json;
  existing generation-pinned reference archives and uv cache documentation.

## 2026-10-08 — User selects B-only snapshot pruning and lightweight future saves
- Decision: Apply only the two B-root candidate sets from the verified dry-run
  manifest, 1865 files/~4.378GiB. Keep all committed fields, progress generations,
  first/final4/every50 snapshots and explicit critical diagnostics. Preserve
  older branches and package caches. Record pruning and recheck protected hashes.
- Future storage: fresh direct roots default to sparse retention (tail4, stride50);
  an explicit all mode remains available. Resume inherits stored retention, with
  absent legacy fields retaining the previous all behavior/identity. Prune only
  after the new generation's progress pointer is published. Storage controls
  are bound to new roots but do not change numerical equations or acceptance.
- Reason: User chose “只精简本轮 B”; large intermediate array histories, rather
  than final solutions or scalar logs, caused avoidable disk growth.
- Alternatives rejected: pruning the old A roots/caches, deleting reference
  commits or resume data, mutating old numerical policy/config bytes, reducing
  precision, or unlinking old generations before publishing a valid new pointer.
- Relation: user's explicit cleanup selection and fracture_storage_plan_20261008.json.

## 2026-10-08 — Execute requested B warning trend continuation
- Decision: User requests looser/warning loading to inspect the complete curve.
  Add a separate warning entry/root; preserve original strict thresholds and
  rejected results. Coupling, adjacent changes, drift-envelope and healing
  failures are explicit warnings; finite fields, exact BCs, damage range,
  maximum-history semantics and checked linear subproblems remain hard checks.
  No particle removal, clipping, horizon/material change or hidden solver fallback.
- Path: inherit verified strict states1–50; initialize .0051 from SHA-bound
  Anderson128 q/d only, holding actual state50 H/d floor. Picard omega=.5,
  at most32 sweeps/load, du=1e-4 through .0075. Warning endpoint q/d/H may advance
  the next diagnostic load with warning ancestry recorded, never strict status.
  Sparse field retention is used. A passing current state cannot erase ancestry.
- Reason: The drift excess alone is small, but AT2 relative .0564 and d change
  .0101 are not resolved by modest tolerance relaxation. Warning labeling can
  answer the trend question without recasting those defects as convergence.
- Alternatives rejected: silently widening original gates, promoting the old
  unready H to a strict parent, replaying strict stages1–50, expensive1000-sweep
  budgets at every warning load, or using this tail as a passed PINN reference.
- Relation: current user request and B control/strict-parent artifacts; same task.

## 2026-10-08 — B warning path completes; distinguish transition and terminal defects
- Decision: Preserve completed warning states51–75 as diagnostic-only, never
  strict references. Terminal .0075 reaction8.95342, damage peak .9996497 and
  AT2 relative4.69e-9 show near-complete unloading/current equation consistency.
  Raw drift count60/ratio1.22875 and warning ancestry remain visible.
- Evidence: All25 warning endpoints replay ancestry/BC/history/AT2/reaction/
  range/healing/drift. Actual completed resume preserves policy bytes; free-force
  checks pass for all25 states. Transition51–59 AT2 relative grows .0917 -> .6411;
  state60 healing .0505821 exceeds1e-3. Coupled equation passes from61 onward.
- Interpretation: terminal60 envelope-exceeding nodes have d>=.998749, whereas
  the d<.9 diagnostic subset has max displacement/load1.00710 and zero excess.
  This supports reviewing the all-node envelope's applicability in failed
  material, not asserting a universal physical bound or deleting/exempting nodes.
  No permanent damage-dependent gate or physics correction is adopted.
- Alternatives rejected: calling the whole path converged because the endpoint
  residual is small, dismissing substantial transition defects as tiny tolerance
  effects, silently filtering damaged nodes, or automatically starting PINN.
- Relation: user request to inspect warning results first; warning replay/force
  resume reports and summary.json; original strict state50 remains unchanged.

## 2026-10-08 — Audit FEM/old/B discrepancy without blaming nominal interface length
- Decision: Confirm matched E/nu/Gc/l0/k_res and nominal notch geometry; do not
  explain B's early unloading as an l0 typo or warning label alone. Treat radius,
  mesh/quadrature, boundary discretization and seeded unconverged path as distinct
  influences requiring controlled comparisons before claiming causality.
- Evidence: SHA-verified FEM/old archives and B saved data show <=.005 loading
  curve differences .1608% (B) versus4.258% (old) on common points. Sampled peaks
  FEM642.486@.0057, old641.340@.0061, B588.786@.0050. At .0051 FEM600.300,
  old577.997 and B380.702: B's large discrepancy starts in crack development.
- Scale/path distinctions: old delta_u=delta_d=.048 on3395 material nodes;
  B .008/.024 on71307, same l0=.024/grip extent .048, but different grip spacing.
  B warning path uses advanced same-load q/d seed and32 sweeps versus FEM100,
  undamped damage, adaptive softening steps. FEM and nonlocal both keep history
  maxima against prior-load cumulative H; do not invent a history-law mismatch.
- Interpretation: terminal x=.75 sampled absolute d=exp(-1) widths FEM.08474,
  old.08705, B.06961 differ despite identical l0. They are solution/profile and
  spatial-interpolation diagnostics, not prescribed widths or isolated causes.
  Same-fine fixed-H .048->.024 damage-radius probe reduces peak .525->.389, so
  shrinking delta_d is a model change, not merely storage/performance optimization.
- Alternatives rejected: attributing new-vs-old differences solely to FEM/grip
  BVP differences common to both, declaring smaller horizons inherently more
  accurate, fitting l0 to the curve, or treating peak agreement as full validation.
- Relation: user's original-curve screenshot/correction; FEM_old_B_comparison.json
  and reaction/damage overlays under warning_b_from50_20261008; same active task.

## 2026-10-08 — Diagnose B onset with identical-parent small-step controls
- Decision: User requests continued diagnosis/remedy of refined B premature
  fracture. Reuse existing unseeded .0051 scalar trajectory rather than repeating
  it. Test Picard omega=.5 from SHA-bound strict50 on two schedules, increments
  2.5e-5 and6.25e-6 to .0051, up to256 sweeps per target, three unchanged strict
  confirmations. Stop a schedule at its first nonpassing target. No rejected
  history advancement, clipping, material changes or curve-fitting correction.
- Reason: Unseeded sweep32 R594.145 differs strongly from seeded warning51
  R380.702, but unseeded sweep574 also reaches R383.443 with unresolved coupled
  residual. Initialization changes displayed timing; small steps test whether
  the same model supports a converged nearby loading branch.
- Alternatives rejected: another identical .0051 unseeded run, treating a finite
  iteration reaction as equilibrium, increasing l0 to fit FEM, and launching PINN.
- Relation: strict50 SHA26b355a8..., B strict/warning trajectories; same task.

## 2026-10-08 — Separate smooth consistency from narrow-crack regularization
- Decision: Record an independent straight-profile scalar-energy witness before
  choosing a B remedy. Integrate the existing zero-H AT2 residual to obtain
  Gc/2*(sum V*d^2/l0+l0*d^T V D d); for x-uniform 2D fields sum original cubic
  weights over x offsets. Minimize this quadratic energy with one center row
  d=1. This is a regularization diagnostic, not coupled Griffith toughness.
- Evidence: At delta_d=.024, h=.004/.002/.001, minimized energies are
  1.989155/1.399449/.878805 versus local lumped-P1 2.709359/2.702343/2.700586.
  Smooth second moments are2.000749/2.000226/2.000020. Doubling transverse extent
  changes h=.004 energy by6.28e-9. Actual 2D ExplicitVolumeSparseD cross-check
  differs by<=3.28e-11 in matvec; stationary free equations pass~1e-15.
- Interpretation: Finite integrable graph kernels permit low-energy sharp scalar
  profiles despite smooth-Laplacian consistency. At fixed damage horizon a
  vanishing-width damaged row has an O(width) energy upper bound. Whether such a
  row releases mechanical traction depends on fixed BA support and degradation;
  this witness alone does not prove zero coupled toughness or B onset causality.
- Candidate diagnostic: delta_d=.008 at h=.004 yields profile energy2.717997,
  second moment2.012862. It is a horizon comparator, not an adopted correction;
  fresh loading and traction/driver validation are required before promotion.
- Alternatives rejected: explaining the witness as a missing factor2, claiming
  input Gc changed, calling scalar profile energy effective coupled toughness,
  tuning Gc/l0 to compensate, or silently replacing the phase-field formulation.
- Relation: /tmp/opencode/b_crack_profile_probe_20261008.py and independent actual
  2D reduction verifier; B onset-control task and saved-gradient diagnostic.

## 2026-10-08 — Test a short-damage-horizon remedy as a separate fresh branch
- Decision: After the same-B small-step controls finish, run a diagnostic
  fresh-zero horizon comparator with only delta_d .024->.008; preserve delta_u
  .008, h=.004, l0=.024, Gc2.7, cloud/BC/history/equations and strict thresholds.
  Target .0057 (FEM sampled peak), du1e-4, Picard .5, cap512/target, sparse fields.
  It is a candidate remedy, not adoption or a relabeling of B's saved curve.
- Reason: Actual same-cloud/same-H state50 solve changes peak d .633199->.546152
  with checked volume residual2.46e-13. Independent scalar energy/moment witnesses
  favor a better-resolved near-local regularization at the shorter radius. A
  coupled fresh path is needed to test whether this delays early fracture.
- History constraint: The fixed-H comparator would heal by.09110 if substituted
  into B state50. Therefore do not resume it with B history/damage or use its
  fields as an accepted continuation. Fresh zero is mandatory for this comparison.
- Alternatives rejected: fitting Gc/l0, swapping phase operators midway through
  a loaded path, claiming the profile diagnostic predicts peak load, and changing
  the permanent formulation/acceptance policy before a controlled result.
- Relation: b_short_damage_fixedH_20261008/result.json, profile witness and user
  request to diagnose/remedy refined B; original B strict50 stays intact.

## 2026-10-08 — Check continuum comparator mesh error before attributing remaining onset gap
- Decision: After the GPU candidate and saved-field replay, run the existing
  independent FEM solver on B's exact SHA-bound71307-node material cloud/triangles.
  Preserve local hybrid Amor/CG1/DG0 formulation, material/geometry/material-boundary
  conditions, baseline warning policy,100-sweep cap and load-step/softening rules;
  endpoint .0057. Write separate mesh/config/archive, never replace coarse FEM.
- Reason: Short-damage candidate strictly passes1–52 (R598.319 at.0051,
  R605.313 at.0052) but.0053 again develops unresolved softening. The old FEM
  comparator has only3395 nodes with far h.02, whereas B is uniformly h.004.
  Independent-reference discretization error must be checked rather than assuming
  the old .0057 peak is mesh-independent or blaming remaining errors on delta_d.
- Environment: serialize GPU run/replay and FEniCS environment; no MPI/FEM and
  JAX GPU concurrency. Use exact cloud mesh conversion with validated signature,
  not a newly guessed geometry or remeshed node ordering.
- Alternatives rejected: claiming the short-radius candidate fully solves early
  fracture after only51/52, treating FEM warning as a strict reference, changing
  the FEM BVP to nonlocal collars in the same mesh-error control, or fitting l0.
- Relation: validation-order policy and B short-damage loading; same active task.

## 2026-10-08 — Allow uniform FEM target sizes without changing equations
- Decision: Change local FEM mesh ordering validation to lc_near<=lc_far
  (both remain explicitly positive), permitting the exact uniform B-cloud
  reference control. Reject inverted ordering. Document the allowed range.
- Reason: The prepared exact-cloud FEM control failed before any solve because
  the old validator required strict inequality, unnecessarily excluding uniform
  meshes. Do not falsify the recorded target sizes to bypass validation.
- Verification: two focused unittest cases pass in fenicsenv, exercising the
  actual rounded-slot config with h_near=h_far=.004 and rejecting inverted sizes.
  All53 candidate fields and5 B-control endpoints replay SHA/ancestry/BC/H/AT2/
  reaction/free force. Every stored q passes without additional CG iterations.
- Alternatives rejected: perturbing metadata sizes to force strict inequality,
  remeshing the exact cloud, altering FEM/PINN equations, or accepting unreplayed
  fields. Candidate strict prefix ends52; .0053 remains coupled-unready.
- Relation: examples/fracture2d/solver.py, focused mesh-config test, root/example
  READMEs, b_short_damage_replay_20261008 reports. GPU ended before FEM starts.

## 2026-10-08 — Retain matched-cloud FEM and partial horizon remedy without declaring resolution
- Decision: Use the completed exact-material-cloud FEM as a mesh-error comparator,
  retaining its warning/overshoot limitations. Preserve original B, short-radius
  candidate and coarse FEM separately. Do not promote short-radius loading beyond
  strict52 or interpret rejected53 as a validated reaction peak.
- Evidence: exact coordinate/triangle sets match B's71307 vertices/141366 cells;
  E/nu/Gc/l0/k_res/geometry/local BVP/history/gate metadata match coarse FEM.
  Fine FEM completes63 loads to.0057, sampled peak619.416286@.0054 versus coarse
  642.485729@.0057;4 transition warnings and dmax1.0031067 persist. Archive SHA
  b12121ac8bb1be5a9022d2e1a65f5fa471199f78a9aba7b186f36825f70d9dce.
- Interpretation: Both reference-mesh sensitivity and finite damage-horizon
  sensitivity are demonstrated. Candidate accepted51/52 R598.318778/605.313064
  differs from matched-cloud FEM598.138129/606.018407 by.0302/.1164%; however
  dmax at52 .580436 versus FEM .481472 and failed53 coupling remain unresolved.
  These results do not identify a single isolated cause or prove a complete remedy.
- Next: same-grid52/53 local driver/damage-operator and numerical branch controls;
  maintain l0/Gc, strict parent floors and original gates before choosing another
  correction. Every current worker is finished; no GPU/FEM/PINN job remains.
- Alternatives rejected: calling coarse FEM's peak mesh-independent, treating
  FEM's finite warning tail as strict validation, accepting candidate53, fitting
  l0/Gc, or automatically starting PINN.
- Relation: b_same_cloud_comparison_20261008/result.json and active task.

## 2026-10-08 — User approves candidate warning path after critical-load stall
- Decision: Run a separate candidate short-damage warning continuation from
  SHA-bound strict52, no unready trial seed, Picard omega.5, at most32 sweeps/load,
  du1e-4 to.0075. Original finite/range/BC/history/linear-solve hard checks and
  reported coupling/healing/drift/change thresholds retain existing semantics.
- Reason: User explicitly permits warning-mode trend inspection instead of
  indefinite convergence retries; candidate .0053 already stalled512 sweeps.
  An unseeded fork avoids conflating the plotted first drop with a deeply
  advanced failed same-load trial as in the earlier B warning comparison.
- Scope: Same model/config/l0/Gc/cloud/horizons; no strict-state promotion or
  PINN transfer. Warning H can advance this diagnostic path with recorded
  ancestry. Compare .0052/.0053 local damage/driver witnesses and fine FEM data.
- Alternatives rejected: another identical512-sweep retry, replacing the strict
  history with rejected53 H, accepting warnings as strict roots, fitting l0/Gc.
- Relation: current user instruction, candidate strict52 SHAb30a974a..., existing
  warning_reference.py policy and matched-cloud FEM archive.

## 2026-10-08 — Isolate fixed-history gradient and mass effects before more radius changes
- Decision: Assemble CPU-only local scalar witnesses from solver.py:954--961
  on the exact B triangles. Compare the actual graph with local P1 stiffness
  under identical nodal-H/lumped reaction terms, then exact consistent P1-H
  moments, and independently reconstruct actual FEM DG0-H weak-form solves.
  These are counterfactual scalar solutions, not admissible loaded-state swaps.
- Checks: K*1=0, affine gradient energy=material area, M*1=reference volumes;
  exact simplex degree3 moments reduce to P1 mass for constant H. FEM coordinate
  and cell transfer are permutations with distances<=1e-12. Every CG solve has
  an explicit residual check; no clipping or fallback. No FEniCS process runs.
- Evidence at52: stored graph dmax .580436; same-H solved graph .580437,
  local-gradient/lumped .543529, consistent P1-H .539479, actual FEM-DG0 H
  .481474 versus saved FEM .481472 (inf difference1.86e-6). Thus gradient/mass
  choices have measured effects but do not explain the entire coupled gap.
- Evidence at rejected53: even local/lumped fixed-H solve remains dmax .997547
  with the already large trial H. Swapping scalar stiffness cannot undo the
  displacement/history feedback. Same P1 strain estimator on saved candidate52
  q gives positive-energy cell max893.098 versus FEM353.856; this is a same-field
  diagnostic, not proof of a mechanics-operator error or BA-driver amplification.
- Alternatives rejected: assigning additive causal percentages to coupled
  differences, confusing cumulative BA H with instantaneous P1 driver, or
  silently adopting local/consistent scalar solutions as new history states.
- Relation: short_damage_local_witness_20261008/result.json and fields52/53;
  user-approved warning continuation is running independently.

## 2026-10-09 — Complete candidate warning trend; retain path and budget limitations
- Decision: Preserve all23 diagnostic states53–75 from the unseeded strict52
  fork as warning-only. Original model/config/strict fields/gates remain unchanged.
  Do not claim the early-fracture problem solved from peak proximity or a small
  terminal equation residual. No additional solve is scheduled this session.
- Evidence: peak610.649295@.0053 versus fine FEM619.416286@.0054; at.0056 candidate
  R541.033223 versus fine FEM2.752473, demonstrating delayed unloading. Terminal
  R2.679365, dmax .999724985, AT2 relative5.78437e-9. All23 SHA/ancestry/BC/H/
  range/healing/AT2/reaction/free-force checks and actual completed resume pass.
- Defects: transition AT2 relative reaches.537701, healing.00972475. Current
  coupling passes65–75 but raw displacement envelope fails; terminal67 excess
  nodes have d>=.998017 and max ratio1.33764. Auxiliary d<.9 region has no excess
  (max1.007277); no filtering, exemption or permanent gate change is adopted.
- Numerical confounder: across.0052->.0056 FEM used552 outer sweeps with full
  damage updates and softening du2.5e-5, candidate warning128/.5/du1e-4. These
  different maps/history parents prevent assigning all unloading differences
  to physics. Prioritize matched-parent budget/step controls before further
  horizon changes or a constitutive correction.
- Alternatives rejected: accepting warning75 as a strict/PINN parent, claiming
  the fixed-H witnesses explain the entire coupled difference, comparing the
  original advanced-seeded B tail as a pure radius control, or fitting Gc/l0.
- Relation: warning_short_damage_from52_20261008 summary/replay/plots, same-grid
  scalar witnesses and current user instruction to inspect warning results first.

## 2026-10-09 — User accepts direct engineering checkpoint and requests PINN transition
- Decision: Provisionally adopt the latest fine-cloud delta_u=delta_d=.008,
  l0=.024 direct result for engineering PINN comparisons. Preserve its original
  strict52 and diagnostic warning53–75 records and all numerical limitations.
  Add a durable public config matching the verified scratch config exactly.
- Authorization: User explicitly requests a direct-solver commit and transition
  to PINN. Commit the reviewed direct changes/docs/tests on dev; subsequent PINN
  implementation/training is a separate working-tree phase. No push/merge is
  requested in this turn.
- PINN contract: Fresh FP64 networks/model identity, no old-model checkpoint
  reuse, reference arrays for comparisons only, learned predecessor H for the
  cumulative history. Explicit SHA-bound accepted-warning comparison policy
  may load the direct tail without granting strict reference/neural status.
- Reason: User accepts current solver response temporarily and prioritizes PINN
  progress over additional direct convergence/curve-fitting studies.
- Alternatives rejected: relabeling warning states as strict, changing PDE/Gc/l0
  or numerical gates, silently loading warning comparisons under strict mode,
  and importing reference history into neural training.
- Relation: current user request, adopted short-damage artifacts and same active
  fracture2d task. Earlier direct-first strict prerequisite is superseded by
  this explicit engineering acceptance, not by a numerical validation claim.

## 2026-10-09 — Publish direct checkpoint and add explicit PINN warning comparisons
- Decision: Direct stage is committed on dev as b6dc8de. PINN changes remain a
  subsequent uncommitted phase. Add strict-default comparison compatibility
  plus explicit user_accepted_warning policy with required SHA, model/field/
  parent/status checks and reference provenance in new neural identities/reports.
- Reason: The accepted full direct curve contains a warning tail; existing
  strict-only loaders cannot use it without an explicit engineering contract.
  Preserve original labels and reference-only comparison semantics.
- Training: start fresh short-damage FP64 model with analytic energy/padded
  damage gradients; initial2 loads, existing3 sweeps/2000 per block, learning
  rate1e-4, unchanged5% reaction/.01 damage-RMSE primary gates. Smoke verifies
  warning ancestry and interrupted/completed resume before real-grid training.
- Alternatives rejected: removing strict-reader guards, inferring authorization
  from a field filename, reclassifying direct warnings, importing reference H,
  reusing old-model neural checkpoints or changing physical/numerical gates.
- Relation: split_pinn.py and split_pinn_short_damage.py; user-authorized transition.

## 2026-10-09 — Verify fresh PINN prefix and start endpoint-only continuation
- Decision: Retain fresh neural states1–2 as warning-useful research checkpoints,
  not strictly converged fracture solutions. Actual completed real-grid resume
  preserves both ZIP SHA/report bytes. Start endpoint-only extension to75 with
  identical architecture/model/optimizer/budgets/comparison policy.
- Evidence: state1/2 R13.432671/27.654670 versus direct13.343131/26.682664
  (errors.6711/3.6428%); damage RMSE2.1123e-6/6.1989e-6; displacement L2
  .2887/.1322%; zero drift/history decrease. Mechanics losses .07124/.06039
  and coupled AT2 .15449/.05637 remain above secondary limits.
- History: learned neural1 H supplies neural2 floor; reference H never enters
  losses. Neural warning ancestry persists independently of the strict direct
  reference statuses at these small loads. Explicit direct warning provenance
  will also propagate when comparison index reaches53.
- Alternatives rejected: claiming strict neural convergence from reaction error,
  resetting learned history during endpoint extension, silently skipping a
  future rejected load, or committing subsequent PINN work into the direct
  checkpoint without a separate publication request.
- Relation: actual resume_check.json, config split_pinn_short_damage.py, dev
  direct commit b6dc8de; PINN continuation is now the active task phase.

## 2026-10-09 — Audit elastic PINN comparison without attributing stopping differences to physics
- Decision: Answer the user's regression concern using saved same-load fine-grid
  reports/configs, distinguishing older coarse59-point success from the closest
  fine-grid energy branch. Do not infer a general PINN regression from two
  points or change the running training policy during this read-only audit.
- Evidence: old/new fine-grid reaction errors at1/2 are2.398/.671% and
  .406/3.643%. Damage RMSE is slightly lower in the new run at both loads;
  mechanics/coupled defects are broadly similar. Architectures/clouds match.
  Direct elastic reactions themselves change only.0000668/.0002637%.
- Confounder: old fine energy run used fixed2000 final-after updates; new
  final-block first-passage policy accepts <=5% reaction error and stopped at
  200 for both loads. Only this final block is10x shorter, not the entire run.
  Gradient implementation and damage horizon also differ. Additional updates
  are not guaranteed to improve reaction monotonically; a matched policy/control
  is required before a causal claim or an accuracy-restoration claim.
- Alternatives rejected: saying every new metric is worse, blaming direct
  elastic stiffness changes, equating5% research advancement with old~1%
  elastic curve quality, or claiming a budget-only cause without control.
- Relation: pinn_elastic_comparison_20261009.json; running PINN continuation
  remains active with its existing identity/budget/stopping policy.

## 2026-10-09 — Continue existing PINN loading and assess the elastic trend
- Decision: User asks to continue training and observe. Keep the already-running
  endpoint75 continuation with its declared policy; assess reaction/damage and
  coupled/mechanics defects across more loads before selecting a separate
  stopping/budget accuracy control.
- Evidence: completed neural3/4 reaction errors .7927/.2174%, damage RMSE
  1.0594e-5/1.4975e-5, zero drift. Coupled AT2 improves to.03874/.03067 but
  remains above1e-3; mechanics losses .05802/.05665 remain warnings.
- Alternatives rejected: restarting the network, silently tightening the active
  identity-bound policy, or claiming full convergence from sub1% reaction error.
- Relation: current user instruction; same live worker PID34813 and short-damage
  checkpoint root. Additional matched-policy control remains conditional.

## 2026-10-09 — Add a read-only continuous PINN observer
- Decision: User requests sustained observation. Start a CPU-only observer at
  60s intervals for the existing training PID34813; no training/policy changes.
  Track completed reports, checkpoint SHA/parent/model consistency, reaction
  curve L2, per-load errors, drift/history/healing and secondary failures.
  Record reaction-error increases, observed reaction declines and direct warning
  reference status as observational flags, not new numerical acceptance gates.
- Reason: Persist progress across the long-running GPU loading sequence without
  duplicate training or unsupported claims of convergence/monotone improvement.
- Evidence: prefix5 now complete; state5 R error2.723%, d RMSE1.9583e-5, zero
  drift/history decrease. Five-point curve relative L2 is2.1118%, not directly
  comparable to the old full59-point metric. Two timed observer reads confirm
  periodic status refresh and unchanged training process identity.
- Alternatives rejected: changing5% early stopping during the active run,
  treating monitoring flags as physical gates, or relabeling warning checkpoints.
- Relation: observation_status.json/observation_history.jsonl in the neural run
  root; observer PID44538, script /tmp/opencode/watch_short_damage_pinn_20261009.py.

## 2026-10-09 — Monitor live milestones and distinguish chat updates from persistent records
- Decision: User requests progress notifications. During this live session,
  wait for completed-load milestones and report new metrics promptly; the
  independent observer continues recording after the session but cannot send
  autonomous chat messages. Keep training/model/stopping policy unchanged.
- Evidence: neural7–10 complete with R errors3.607/.865/.651/.362%; zero drift,
  no history decrease. Prefix10 reaction curve L2 is1.7928%; no primary load
  rejection. AT2 at10 .016785 and mechanics .053556 remain secondary warnings.
- Interpretation: recent reaction points improve, but nonmonotone earlier
  errors and unresolved secondary equations prevent strict/full-curve claims.
- Alternatives rejected: promising out-of-session push notifications, reporting
  in-progress sweeps as completed loads, or suppressing the unresolved residuals.
- Relation: current notification request and live observer/continuation; same
  active task, no new training worker or numerical-policy change.

## 2026-10-09 — Confirm full neural warm start and identify the measured per-load budget
- Decision: Answer the user's speed/warm-start question by checking actual
  network/optimizer lifetime, checkpoint restore and saved phase timings. Keep
  the current run unchanged; speed changes require a separate measured control.
- Evidence: q_net/d_net and Adam states initialize before the load loop, persist
  through all updates and restore together from state.zip on resume. Learned
  q/d/H predecessor fields also carry forward; no per-load network reset exists.
- Cost: loads14/15 each use6000 mechanics-before+6000 damage+4400 mechanics-after
  updates (16400 total), taking863.95/864.33s (~14.4min/load). Early stopping
  applies only to the final sweep's after block. model_build_seconds201.24 is
  the one-time invocation build measurement repeated in reports, not a new
  model build at each load. Phase timings include logging/checkpoint overhead.
- Alternatives rejected: blaming absent warm start, interpreting the build
  metadata as per-load reconstruction, or changing budgets without an accuracy
  comparison while the existing identity-bound continuation runs.
- Relation: split_pinn.py network initialization/load loop/state restore, saved
  state0014/0015 timing.json; current user confirmation request.

## 2026-10-09 — Design adaptive inner stopping before implementation
- Decision: User requests thinking about two/five small-update stopping criteria.
  Recommend five consecutive post-update evaluation checkpoints, not five Adam
  steps, with field/objective stability and explicit subproblem residual tests.
  Distinguish converged block, stalled-warning block and budget exhaustion.
  This is a project-owned engineering proposal, not a convergence theorem or
  adopted numerical policy; the live run remains unchanged.
- Reason: Small parameter/loss changes can reflect a small learning rate,
  oscillation or neural tangent-space stagnation while physical residual remains
  large. The checked A2 energy gradient obeys grad_theta E=J_theta^T F. Actual
  load15 final-before energy changes only~1.77e-6 relatively from step200 to2000,
  with an intermediate spike, while reported mechanical residual loss remains
  .05197 >.01. Stability alone is not an accuracy certificate.
- Stage distinction: mechanics holds d fixed; damage tests unrelaxed network
  output against fixed H_used (d_iter is constant until the damage block ends);
  outer coupling tests relaxed d/current q/updated cumulative H separately.
  Do not mix fixed-H inner and moving-H coupled residuals or compare changing
  objectives across outer sweeps as if they were one stationary loss.
- Candidate control only: evaluate every50–100 updates, min~200 warm-start
  updates,5 stable checkpoints, existing2000 cap; keep first-load budget more
  conservative. Actual absolute/relative thresholds require same-parent paired
  timing/accuracy checks on elastic and near-critical loads before adoption.
- Alternatives rejected: stopping on two tiny raw weight updates, using a
  constant d_iter to measure damage-training progress, declaring a plateau
  converged, or silently changing the active checkpoint identity.
- Relation: user planning request, split_pinn.py:420–495, analytic energy/force
  reverse rule, current full-graph timing and persistent observation.

## 2026-10-09 — Implement isolated acceleration controls with explicit loading schedules
- Decision: User approves adaptive-stop direction and proposes coarse .001
  increments through.005 then .0001. Add an explicit ordered reference-index
  loading plan, separating neural ordinal and direct index; the example is
  [10,20,30,40,50,51..75],30 target loads. Preserve legacy uniform behavior,
  immutable schedule/resume identity and learned-history updates at actual loads.
- Reason: Current index*load_step simultaneously identifies neural checkpoints,
  physical loads, direct files and residual scale. Nonuniform loading must map
  each explicitly; .005 has substantial damage and large jumps change history.
  Existing uniform trajectory remains the accuracy/control record.
- Adaptive trial: post-update checkpoints every100, min200 warm-start updates,
  first-load min2000,5 stable comparisons, existing2000 cap; objective abs/rel
  1e-8/1e-4, residual spread1e-8/1e-3, q-change/load1e-4, d-change1e-6,
  reaction abs/rel1e-8/1e-4. Residual accuracy uses existing mechanical .01 and
  fixed-H relativeAT2 .001. Stable inaccurate blocks return explicit stagnation
  warnings only when enabled; no convergence theorem or default adoption.
- Validation: first isolate adaptive stopping on a uniform same-parent load;
  separately exercise schedule mapping/resume and retain the aggressive30-load
  example as experimental. No reference interpolation/history injection,
  gate loosening, active-run mutation or Git-history operation.
- Relation: current user approval, planning entry and warm-started PINN study;
  same active task. GPU controls must be serialized for meaningful timings.

## 2026-10-09 — Retain acceleration support but do not adopt the first unsuccessful stop calibration
- Decision: Schedule/adaptive-stop implementation and regression checks pass;
  keep the coarse30-load example experimental. Restore original uniform
  continuation after a serialized uniform load18 adaptive control from exact
  neural17 parent. Do not loosen gates/tolerances just to manufacture speed.
- Evidence:12 CPU tests pass (2 GPU tests skipped), new GPU schedule/adaptive/
  interrupted/completed-resume/extension test passes137.90s. Real30-state reader
  preflight selects7 strict+23 warning snapshots without interpolation.
- Same-parent result: archived baseline load18 uses16400 updates, R error.3984%,
  d RMSE6.1904e-5, wall865.40s. First adaptive calibration exhausts all9 blocks:
  18000 updates, R error2.2112%, d RMSE6.8142e-5, wall1179.99s. No stable window
  triggered; first adaptive measured load includes new JIT/check overhead and
  disabled reference-first-passage stopping. No steady speedup claim is valid.
- Interpretation: small scalar loss changes did not imply stable fields at the
  proposed tolerances. Example final q changes/load4.24e-4/9.13e-4 and unrelaxed
  d-change3.64e-5 exceed1e-4/1e-6. The first candidate is not a useful speed
  replacement; calibration/alternative monitoring requires further paired work.
- Resource workflow: preserve completed uniform prefix20; terminate owned
  original process after its checkpoint, run one isolated control, then restore
  uniform endpoint75 (new PID71732) and observer71855. Prefix20 ZIP SHA remains
  f2acb9fd554c47c44284df17fcb2e4fd7e5d51436a9bb9161a17ff238ead8d4b.
- Alternatives rejected: treating sparse loading as identical history, adopting
  an unhelpful stop policy, concurrent large GPU benchmarks, or discarding the
  existing trajectory. No commit/merge/push authorized in this turn.
- Relation: experimental configs, tests/test_fracture2d_split_acceleration.py,
  paired_load18_comparison.json and restored live observation status.

## 2026-10-10 — User authorizes full PINN warning loading without comparison rejection
- Decision: Current uniform PINN stopped at54 (reaction difference7.2117%,
  d RMSE.0100681). User notes the direct comparator's own defects and explicitly
  requests complete loading before judging results. Add reporting.comparison_gate
  =False in a new branch; preserve original comparison limits/failed checks as
  diagnostics, never relabel old rejected54 or claim strict neural acceptance.
- Path: read-only inherit usable neural1–53, retrain54 from learned53 weights/
  Adam/q/d/H, then uniform du1e-4 through.0075. Fixed3 sweeps/2000 per block,
  omega.5; disable reference-first-passage stopping as well. No adaptive stopping
  or coarse schedule adoption. Reference H/failed54 H never becomes a floor.
- Hard semantics: retain finite fields, exact constraints, model/checkpoint/
  reference identity checks and exact learned maximum-history update. Other
  numerical/physical defects retain existing warning reporting. A bad reference
  comparison alone no longer rejects load advancement in this explicit mode.
- Reason: comparison error against an unconverged direct warning state is not
  itself a certified physical error. Full curve/self-residuals must be inspected.
- Alternatives rejected: raising5% to an arbitrary new number, overwriting
  old rejection labels, in-place policy mutation or skipping54 with its unready H.
- Relation: current user request, baseline stop54_audit.json and new full-warning
  config. Entire usable prefix53 curve L2.0208153; no GPU worker currently runs.

## 2026-10-10 — Complete the full neural warning curve without equating completion to convergence
- Decision: Retain the completed75-load curve as the requested diagnostic result.
  All checkpoint parent/model links are verified; all states remain warnings.
  Do not resume new training or claim full fracture reproduction in this status
  response. Inspect self-residuals/coupling and damage propagation next.
- Evidence: neural sampled maximum728.222478@.0070 versus direct610.649295@.0053;
  terminal R320.883300 versus2.679365. Terminal AT2 relative .604938, mechanical
  loss .0196246, damage-reference RMSE.0896994/max error.951813, healing.0054631.
  Whole-path maximum healing.0312187; zero drift reports, zero strict neural states.
- Interpretation: comparison error alone is not physical error against the
  imperfect direct warning reference, but substantial own AT2/damage-change and
  healing defects independently prevent a convergence claim. A high local dmax
  does not certify through-going traction-free crack opening.
- Alternatives rejected: hiding full-curve mismatch, declaring convergence from
  reaching75, or attributing the discrepancy solely to direct errors without
  checking the neural equation/field defects.
- Relation: full_curve_summary.json/completed_checkpoint_chain.json, reaction
  curve PNG and state0075 damage plot. Terminal ZIP SHA
  1cbf7b279437a5cba2efa2244374f175907003cfb96eec1de7f096180b40eba7.

## 2026-10-10 — Diagnose which neural subproblem limits fracture response
- Decision: User asks whether mechanics or damage is harder. Run bounded
  frozen-field diagnostics at54/65/75 using saved final/progress checkpoints:
  same-H unrelaxed neural damage versus exact scalar CG, residual decomposition
  into image fit/relaxation/current-H change, and exact mechanics at the saved
  neural damage to quantify displacement/reaction/local driver errors.
- Reason: mechanics MSE and AT2 term-normalized infinity residuals are not
  directly comparable. Fixed-H damage losses are~1e-7 late while coupled AT2
  reaches.605, suggesting an outer/local-driver issue that requires verification.
- Contract: unchanged adopted model, actual prior neural cumulative-H floor,
  no reference history injection, new loss/physics/training or state promotion.
  Direct oracles measure frozen-subproblem errors, not a converged coupled root.
- Alternatives rejected: blaming the visibly larger AT2 number alone, equating
  small global mean damage loss with accurate crack-tip fields, or declaring
  mechanics solved because elastic reaction agreement was good.
- Relation: current user question and saved full-warning artifacts; same task.

## 2026-10-10 — Frozen oracles identify local mechanics and outer feedback as priorities
- Decision: Prioritize local mechanics/driver accuracy and outer coupling,
  while retaining late fixed-H damage fitting defects. Do not identify a single
  PDE bug or attribute the whole loading mismatch from frozen checks alone.
- Same-H image error versus exact CG (inf/RMSE) at54/65/75:
  .000672/4.17e-5, .003428/4.71e-4, .064570/.004834. Fixed-H mean loss is small
  late but local infinity accuracy is not uniformly good (75 relative.1363).
- Same-d mechanics: q L2 errors .06855/.18782/.28453%, but reaction errors
  .2070/2.0368/33.1420% versus exact frozen-d mechanics. At75 exact R479.947
  versus neural320.883; this exact fixed-d solve still does not yield the
  direct loading curve R2.679, demonstrating damage/history/coupling also matters.
- AT2 common-scale decomposition at65: neural image.03114, relaxed fixed-H
  .24330, updated-H difference term.24738, current coupled.47272. At75:
  .08312/.29335/.38297/.60494. These are infinity norms, not additive causal
  percentages; peaks/signs differ. All algebra/checkpoint/history relations
  and explicit exact-subproblem residual checks pass.
- Caution: maximum driver discrepancies often occur in already highly damaged
  core nodes; they show local strain error but do not alone establish crack-
  front causality. A same-parent four-way neural/direct q/d outer control is
  needed to isolate error transfer; no further training is scheduled yet.
- Alternatives rejected: declaring damage solved from1e-7 mean loss, declaring
  mechanics solved from sub.3% displacement L2, or comparing raw differently
  normalized loss magnitudes as a difficulty ranking.
- Relation: pinn_equation_diagnosis_20261010/result.json and verified54/65/75
  final/progress checkpoints. No PDE/model/state/optimizer change.

## 2026-10-10 — Adopt hybrid direct mechanics and nonlocal PINN damage
- Decision: User selects direct mechanics with nonlocal PINN damage. Add an
  explicit hybrid runner/config on the current FP64 fine-cloud short-damage
  model. Reuse FP64A2Backend.solve_strict for both fixed-d mechanical phases;
  train only the damage network on the existing fixed-H nonlocal AT2 residual.
- Reason: Frozen checks identify local mechanical gradients/driver precision
  as a priority. This removes neural mechanical error while preserving damage
  representation and outer-coupling errors for independent inspection.
- History: H_used=max(own preceding-load cumulative H, current direct-q driver).
  The final load history is recomputed after the relaxed damage and direct
  mechanics update. External direct reference arrays remain comparisons only.
- Implementation: separate checkpoint schema containing damage weights/Adam
  and actual q/d/H fields, including q in in-progress phase snapshots. Begin
  fresh rather than silently importing a full-neural history as hybrid history.
  Preserve the three-sweep/2000-damage-update/omega=.5 control for a bounded
  initial comparison; report fixed-H image and final coupled defects separately.
- Validation: small-grid numerical/history/phase-recovery checks, followed by
  fresh real-cloud loads1/2 and unchanged completed-checkpoint resume.
- Alternatives rejected: a dummy displacement network in hybrid checkpoints,
  reference H/d targets in damage training, replacing the mechanics BVP with
  local FEM, or assuming accurate mechanics guarantees coupled convergence.
- Relation: user's hybrid selection and frozen-subproblem diagnosis; same
  active task. No full75-load training or Git-history operation scheduled here.

## 2026-10-10 — Clarify that direct–PINN uses staggered subproblem solves
- Decision: User explicitly wants direct mechanics and nonlocal PINN damage
  solved by stagger, not a coupled/monolithic algorithm. Confirm and document
  the implemented sequence: freeze d for strict direct q, freeze H_used from q
  for damage-network training, then alternate at the load. No joint optimizer,
  combined u/d loss, or gradient through mechanics/history is used.
- Reason: Earlier wording "coupled residual" was ambiguous. The preserved
  report key coupled_AT2_relative only measures the damage equation on final
  staggered q/d/H, and does not name the solving algorithm.
- Evidence: small-grid numerical/history/interruption/resume/extension tests
  pass (2 tests,113.031s). Fresh real-grid two-load stagger completes and actual
  completed resume preserves both ZIPs. Independent CPU mechanics/history/AT2
  replay passes: R13.343210/26.683132, reaction differences .000596/.001755%,
  d RMSE2.21e-6/6.40e-6; final stagger AT2 .18529/.05551 remains warning.
- Alternatives rejected: interpreting final consistency checks as a simultaneous
  u/d solve, promising fracture convergence from elastic reaction agreement,
  or inferring steady timing improvement from an interrupted two-load trial.
- Relation: current user clarification, hybrid_pinn.py and resume_check.json;
  the existing implementation already matches the requested stagger semantics.

## 2026-10-10 — Continue the verified direct–PINN stagger loading path
- Decision: User asks to continue solving. Resume the existing hybrid path after
  verified own state2 and extend only end_state_index from2 to75 (load.0075).
  Retain uniform du1e-4,3 frozen-field stagger sweeps/load,2000 damage updates/
  sweep, omega.5, FP64/Adam1e-4 and strict mechanical CG checks. Preserve the
  current explicit warning progression/reporting policy without relabeling it.
- Reason: This follows the selected solver direction and user-requested loading
  continuation while retaining the verified prefix and its actual own history.
- Execution: preflight checkpoint/model/reference/config identity, then one
  background GPU worker and CPU-only read-only observation. Verify the first
  newly completed load and persist the exact command/process/resume point.
- Alternatives rejected: restarting networks, importing neural/direct reference
  H, changing sweep budgets during identity-bound resume, joint u/d training,
  or calling completion/stagger advancement a convergence certificate.
- Relation: current user request and hybrid_short_damage_first2_20261010;
  existing damage/stagger residual failures remain recorded as warnings.

## 2026-10-10 — Separate damage training budget, representation and stagger consistency
- Decision: Answer the user's training/capacity question from the live bound
  identity and actual hybrid objective/network implementation. Treat insufficient
  fixed-H training, neural representation/optimization and finite stagger budget
  as separate hypotheses; do not change the active continuation in this answer.
- Reason: A high local damage peak does not establish adequate crack extent or
  unloading. A small full-cloud mean loss does not establish local AT2 accuracy,
  and a large final stagger residual is not alone a neural-capacity diagnosis.
- Proposed control, not scheduled: freeze the same own H, compare the fixed-H
  direct damage solution, extend training of the existing network, then test
  representation changes under matched conditions. Inspect outer stagger
  consistency independently rather than modify several controls together.
- Alternatives rejected: declaring tanh MLP capacity insufficient from the
  loading curve alone, assuming2000 updates is convergence, treating the load
  prefactor as a damage upper bound, or silently changing the live run.
- Relation: current user question, hybrid bound identity/code and saved62
  fields/reports/training log. Same live direct–PINN stagger task.

## 2026-10-10 — Reference fracture1d Swish/Fourier method in a larger 2D network
- Decision: User requests fracture1d-style network settings and a larger2D
  network. Add a separate hybrid candidate with256-wide6-hidden-layer MLP,
  Swish and64 trainable Fourier frequencies at scale16, using the existing
  shared make_pinn builder. Preserve the current2D model/output/training policy
  so this candidate specifically tests a richer neural representation.
- Reason: fracture1d's current baseline is200x4 Swish,32 trainable Fourier
  frequencies at scale16, plus33 local hat coefficients. The existing2D plain
  128x4 tanh/no-embedding network is smaller and lacks its feature enrichment.
- Scope: construct/count/forward-check the candidate on CPU while the existing
  GPU continuation remains active. A new architecture requires fresh network/
  optimizer state and separate checkpoints; do not mutate the live bound config.
- Local enrichment: fracture1d hats are tied to a fixed1D grid/imperfection
  patch. Their2D advancing-crack counterpart needs separate basis/support
  design and validation; do not silently inject a guessed crack path here.
- Alternatives rejected: increasing width alone while omitting Swish/Fourier,
  reusing incompatible old-network checkpoints, treating increased capacity as
  proof of fracture accuracy, or copying all1D optimizer/PDE choices at once.
- Relation: current user direction, fracture1d/configs/train_base.py and shared
  pinn.arch/pinn.embedding; same active direct–PINN stagger task.

## 2026-10-10 — Stop old network and start the larger candidate at load .0040
- Decision: User authorizes stopping the old run and then steers the new path
  to start around.004 or.005 to save elastic-stage time. Terminate owned old
  worker58450/observer58514, retain all artifacts, and start the prepared larger
  Swish/Fourier candidate fresh at.0040, then du1e-4 through.0075.
- Loading identity: explicit reference_indices=[40..75],36 selected targets;
  initial end_state_index=2 validates.0040/.0041 before endpoint extension to36.
  Own H/d start fromzero and evolve through direct–PINN stagger; no old weights/
  Adam/history or external-reference H is imported into the new architecture.
- Reason: .0040 is the more conservative user-proposed starting point; .0050
  already has substantial damage in the archived model. Skipping prior loads
  changes the discrete history path and is not asserted equivalent to uniform
  loading or a pure architecture-only comparison.
- Alternatives rejected: running two large GPU workers, deleting the stopped
  comparator, treating a new-shaped network as an old-checkpoint resume,
  assuming .005 is damage-free, or hiding the changed loading schedule.
- Relation: current stop instruction and immediate loading-start steering;
  same active fracture task. New launch will target only the first two selected
  loads for physical/training validation; no monolithic u/d solve.

## 2026-10-10 — Interpret the new network's two-load endpoint accurately
- Decision: Explain the observed stop as reaching the configured selected-load
  endpoint2 (.0040/.0041), using completed reports/observer state/log/process
  checks. Retain warning status and assess reference agreement separately from
  fixed-H and final stagger consistency. Do not restart or change training in
  this status response.
- Reason: Both completed comparisons pass the declared reaction/damage research
  limits, but their own residual/change defects are unresolved. Two precritical
  points do not establish post-peak unloading or an architecture-only benefit
  when the loading path also changed.
- Verification scope: archive config/schedule/phase/payload/parent hashes,
  finite fields, own-history monotonicity, relaxation and report agreement;
  this does not substitute for independent PDE replay or actual completed resume.
- Alternatives rejected: diagnosing a crash from an idle GPU, calling small
  reaction error converged fracture, or automatically extending an inspection
  request into a full overnight training job.
- Relation: current user status question, completed new-network two-load
  artifacts and completion_audit.json; same active fracture task.

## 2026-10-10 — Continue Swish/Fourier stagger after the two-load validation
- Decision: User requests continuation. Resume the new network after its own
  completed .0041 state and extend only selected end_state_index from2 to36,
  reaching .0075 on the already bound reference_indices=[40..75] schedule.
- Reason: Follow the selected architecture/loading path and preserve completed
  network/Adam/q/d/H checkpoints. Existing fixed-H/final stagger defects remain
  explicit warnings; no physics, optimizer, sweep-budget or acceptance change.
- Execution: verify both completed SHA/config/parent/payload identities and all
  selected references, then one background GPU worker and CPU-only observer.
  Confirm actual completed-state restoration and .0042 damage training before
  reporting the continuation active.
- Alternatives rejected: restarting weights/history, using last-index75 for a
  36-target schedule, silently increasing training budgets during resume, or
  resuming the parked old network.
- Relation: current user continuation instruction, new-network completion_audit
  and selected-load config; same active direct–PINN stagger task.

## 2026-10-10 — Diagnose slow training from measured phases and hardware limits
- Decision: Use matched-load saved timing and a read-only GPU query to answer
  whether network size, FP64 hardware or other work causes slow continuation.
  Distinguish measured damage-phase dominance from architectural explanations;
  do not interrupt/benchmark alongside the live GPU worker or change precision.
- Reason: Old and new paths already use FP64 and6000 updates/load; unchanged
  mechanics timing and~6x neural-phase growth identify the representation change
  as the incremental slowdown. Weak RTX3090 FP64 throughput and full71307-node
  batches amplify absolute cost. Phase timing alone cannot separate GEMM,
  transcendental activation/Fourier, sparse-D or compiler inefficiencies.
- Reference: NVIDIA CUDA12.6.3 Programming Guide16.7.1 specifies128 FP32 and
  2 FP64 cores/SM for compute capability8.6, versus64/32 for8.0. The corresponding
  ~1/64 ordinary arithmetic ratio is a hardware limit, not a measured whole-
  program FP32 speedup. Source URL recorded in the active task.
- Alternatives rejected: blaming slow direct mechanics, assuming reinitialization
  or model rebuild every load, inferring achieved FLOPS from98% utilization, or
  silently casting the numerical core/network toFP32 without accuracy checks.
- Relation: user's performance question and matched .0046 timing.json reports;
  active new-network continuation remains unchanged.

## 2026-10-10 — Replace slow 256-wide run with 128x6 Swish/Fourier from .0048
- Decision: User explicitly stops the preceding run and requests width128,
  depth6, Swish/Fourier, starting directly at.0048. Terminate owned worker82609
  and observer82674 after command verification, retain artifacts, and launch a
  distinct fresh FP64 hybrid trial with128x6 and the existing64 trainable Fourier
  frequencies/scale16. Keep direct mechanics and the current damage/stagger
  optimizer/budgets; no mixed-precision change is requested.
- Loading: explicit reference_indices=[48..75],28 targets; initial selected
  endpoint2 validates.0048/.0049, full endpoint28 reaches.0075. New-shaped
  network/Adam and own H/d start fresh, without old/reference history import.
- Reason: Reduce dense neural work while retaining depth/nonlinear/features;
  both width and initial loading path change by user direction. Performance/
  accuracy must be measured, not inferred from parameter count alone.
- Alternatives rejected: continuing the unwanted slow worker, overwriting its
  checkpoints/config, using incompatible old network states, changing FP64 or
  training budgets implicitly, or treating fresh.0048 as equivalent old history.
- Relation: current user instruction and measured damage-phase bottleneck;
  same active direct–PINN stagger task.

## 2026-10-10 — Run the narrower trial on its full selected path
- Decision: Launch the user-requested128x6 trial with --last-index28, reaching
  .0075 from.0048. Keep the config's default2 for standalone initial diagnostics,
  but inspect.0048/.0049 within the running full path rather than automatically
  pause again after two points. Retain all strict mechanical/field identity checks
  and current warning reporting.
- Reason: Continue the fracture-loading trial without repeating the earlier
  unexpected two-point stop; actual bound endpoint and recovery command are
  explicit. Startup timing is only a short warm-window observation.
- Alternatives rejected: confusing28 selected ordinals with reference75,
  concealing the live CLI override, or claiming a full-load speedup/convergence
  from the first200-update interval.
- Relation: preceding user-requested narrower trial and active worker92655;
  use --resume --last-index28 only after that worker exits.

## 2026-10-10 — Diagnose missing unloading with frozen-H and matched-parent controls
- Decision: User requests cause diagnosis for the128x6 trial's rising reaction.
  At a completed .0055 checkpoint temporarily stop the owned loading worker,
  serialize one diagnostic GPU process, then restore the original continuation.
  Preserve all loaded checkpoints/history and numerical/acceptance policy.
- Controls: frozen own-H exact AT2 damage and image/relaxation/current-H residual
  decomposition at.0052/.0054/.0055; fixed-H clone of the saved .0055 network/
  Adam with4000 additional updates (measure at2000/4000); same preceding .0054
  q/d/H parent at.0055 with exact damage/direct mechanics for up to32 damped
  stagger sweeps, recording the3-sweep control and later contractions separately.
- Reason: Strict direct mechanics already removes neural mechanical fitting
  error. Reference mismatch alone, against warning post-peak references, cannot
  isolate neural optimization, outer consistency or inherited/loading-path errors.
- Contract: unchanged PDE/model/omega/learning-rate/objective; own previous-load
  cumulative H remains the floor. Exact damage is an oracle/control, never a
  production loss target or loaded parent. Additional training/direct trial
  states are diagnostic only; no history/network promotion after the controls.
- Alternatives rejected: blind budget/network enlargement, comparing different
  normalized residuals as if identical, running training and diagnostic GPU jobs
  concurrently, or treating a32-sweep failure as proof of no coupled equilibrium.
- Relation: current user investigation request and current hybrid trial; same
  active task. This is a bounded numerical-control design, not an adopted remedy.

## 2026-10-10 — Stop the loading trajectory directly while retaining the diagnosis
- Decision: User explicitly says the current result may be stopped directly.
  Terminate owned92655/92770 now, retain completed/progress artifacts, and
  continue the requested frozen-field/control diagnosis on saved .0055 state8.
  The former plan to automatically restore loading is superseded; no loading
  worker will be restarted as part of this diagnostic session.
- Reason: Follow the user's active steering and avoid further costly propagation
  of the unready trajectory. Completed .0055 already exists for the planned
  matched-parent controls, so diagnosis does not require more production loading.
- Alternatives rejected: waiting for extra loads, discarding saved results,
  silently restarting after the diagnosis, or using diagnostic history as a
  production parent.
- Relation: immediate user stop instruction; same active fracture diagnosis.

## 2026-10-10 — Respect aborted diagnosis and retain an unverified attribution
- Decision: The serialized diagnostic command was user-aborted during geometry
  construction. Reconcile process/output state: no GPU process, no result.json,
  empty diagnostic root and only the layout event in its log. Do not silently
  rerun the cancelled long command or restart production loading.
- Reason: No fixed-H oracle/extra-training/direct-stagger measurements exist
  from this attempt. Existing production residuals establish unready damage/
  stagger consistency, but do not isolate training budget versus representation,
  optimization or inherited/path errors.
- Alternatives rejected: reporting planned controls as completed, diagnosing
  numerical failure from the cancellation, attributing unloading failure solely
  to training, or replacing the stopped source trajectory with diagnostic states.
- Relation: user-aborted tool execution and verified stopped state; same task
   remains incomplete with the exact controlled-diagnosis resume point recorded.

## 2026-10-10 — Reduce the next diagnostic to one frozen-H no-training oracle
- Decision: User asks the next action. Execute only the smallest decisive saved
  .0055 control: same own H exact AT2 damage, image/relaxation/history residual
  decomposition, and strict direct mechanics at unrelaxed neural/exact images.
  Add explicit --frozen-only/--output to the scratch diagnostic so it performs
  zero neural updates and zero outer-control sweeps in a new output directory.
- Reason: Obtain evidence before choosing longer training or more stagger;
  the previously cancelled combined experiment produced no numerical results.
- Contract: production remains stopped, source bytes/history untouched, same
  physical/operator model and actual previous-load H floor. Instantaneous force
  corrections are not coupled convergence or proof of a whole-path cause.
- Alternatives rejected: repeating the long combined command unchanged,
  guessing that network size/2000-step budget alone causes missing unloading,
  or automatically resuming the stopped loading trajectory.
- Relation: current user's "next" question and saved state8/.0055; same active task.

## 2026-10-10 — Prioritize outer/history control after tiny same-H force correction
- Decision: Completed same-H oracle shows image inf/RMSE .001898/.000150 and
  reactions629.3168(neural image)/629.2826(exact image), versus saved relaxed
  630.4007. Run the next isolated same-parent exact-inner stagger at.0055 for
  up to32 sweeps, zero neural training, to observe outer contraction/unloading.
- Reason: Current-H neural image fitting error barely changes instantaneous
  force; more training of that frozen subproblem alone cannot explain the
  large response discrepancy. Outer feedback and inherited/path errors remain.
- Contract: actual own .0054 predecessor q/d/H floor, omega.5, unchanged PDE/
  strict mechanical and scalar solve checks. Diagnostic states are not loaded
  parents and production remains stopped.32-sweep nonconvergence is not a
  proof of no equilibrium or of a unique causal error fraction.
- Alternatives rejected: jumping directly to moreNN capacity/4000 extra steps,
  treating tiny force correction as proof that past neural errors never mattered,
  or promoting the direct control into the stopped path.
- Relation: frozen0055 result.json and current bounded next-step investigation.

## 2026-10-10 — Exact-inner controls identify outer consistency as the next priority
- Decision: Prioritize a fixed-load convergence-controlled stagger baseline and
  load-step/outer-stability controls before another full neural curve or network
  enlargement. Keep128x6 as the representation for the next controlled trial.
- Evidence/rationale: from the same .0054 parent at.0055, exact-inner3-sweep
  R630.3624 nearly equals neural3-sweep630.4007. Exact-inner16/32 reactions
  621.7064/604.8941 fall while damage peak grows.7913/.9744 without NN training.
  However final stagger AT2 grows to.19359 and d-change.01740, so32 sweeps
  is not a converged state or an adopted32-sweep fix.
- Boundary: same-H oracle and same-parent3-sweep agreement rule out current
  frozen-H neural fitting as the main instantaneous force discrepancy at this
  state; they do not rule out earlier neural/history/path/model effects or prove
  that any finite Picard budget converges through fracture.
- Alternatives rejected: increasing2000 inner steps blindly, relabeling32-sweep
  unloading as validated equilibrium, loading its H as a parent, or reducing the
  question to an additive percentage of differently normalized residuals.
- Relation: frozen0055 and direct_stagger0055 diagnostic results; all production
  checkpoints remain unchanged and no loading worker is restarted.

## 2026-10-10 — Implement convergence-gated stagger after bounded exact-inner controls
- Decision: User approves continuing outer-consistency work. Continue the same
  fixed-.0055 exact-inner diagnostic from sweep32 to a total128 while retaining
  the original .0054 H floor; if not ready, test independent.00545/.005425 loads
  from that same original parent with up to64 sweeps each. Numerical cases remain
  diagnostic, not converged production parents.
- Controller: add an explicit optional residual-gated outer policy to the hybrid
  runner, separate from legacy fixed budgets. Require fixed-H image accuracy,
  current AT2 and damage/adjacent q/H/reaction stability plus physical/mechanical
  checks for consecutive confirmations. Budget exhaustion stops load advancement;
  rejected trials never update the committed history or seed subsequent loads.
- Reason: Exact-inner3 reproduces neural3;32 shows damage growth but not readiness.
  Numerical/iteration gates must decide advancement rather than a fixed count.
- Identity/recovery: numerical policy changes use a separate config/root; preserve
  old mode/checkpoint layout and permit only explicit read-only matching prefix
  forks. Warning ancestors stay warnings, even if a current child is ready.
- Alternatives rejected: restarting the stopped warning path in place, resetting
  the H floor to diagnostic32, automatically accepting exhausted budgets, changing
  network/physics/precision, or claiming Picard/small-step convergence in advance.
- Relation: current user approval; same active fracture task. Controls are bounded
  engineering checks, not a convergence theorem or model-formulation correction.

## 2026-10-11 — Replace the failing scan-in-CG path with checked padded CSR rows
- Decision: The current runtime reports RTX4080 SUPER/JAX0.10.1 rather than the
  earlier RTX3090 environment. Extended controls fail scalar CG explicit checks.
  Audit identical saved-H data: standalone scan and padded matvec match SciPy,
  but scan-based JAX CG diverges (host norm91.97); padded CG matches CPU to
  4.44e-16 and passes3.48e-13 residual versus3.69e-13 limit. Use fixed padded
  CSR off-diagonal row sums in AT2 CG, with the same volume-SPD equation,
  diagonal preconditioner, precision/tolerances and explicit residual gates.
- Reason: This isolates an implementation/runtime-specific nested solve path;
  the failed first controls are not evidence of outer or small-step nonconvergence.
  Exact compiler/library root cause is not established from this audit.
- Alternatives rejected: loosening tolerances, clipping divergent damage,
  silent CPU fallback, altering the physical operator, or claiming the old
  scan CG was validated on this new runtime because its standalone matvec passes.
- Relation: at2_gpu_audit_20261011/result.json; official JAX CG documentation
  confirms SPD assumptions and info=None, requiring independent residual checks.

## 2026-10-11 — Keep the new gate while acknowledging the unresolved Picard baseline
- Decision: Retain implemented convergence gates/checkpoint recovery and stop
  unready loads, but do not call the nonlinear convergence problem solved.
  Bounded exact-inner128 and half/quarter64 controls all exhaust their budgets.
  RealNN4-sweep protocol validation likewise stops.0055 and leaves.0056
  unattempted with its source warning parent unchanged.
- Reason: Accurate inner solves and force reduction do not imply outer readiness.
  The tested small steps share an unready.0054 predecessor, so they cannot rule
  out small-step benefits from a genuinely ready earlier history. Further outer
  stabilization/history-path controls are needed before full loading resumes.
- Alternatives rejected: accepting lower reaction as convergence, promoting
  exhausted trial H, bypassing gates because comparisons are diagnostic, or
  rerunning a long neural curve before a usable accurate-inner baseline exists.
- Relation: padded outer-control results and real gated smoke budget_gate_audit;
  all tests/implementations remain uncommitted, source trajectories parked.

## 2026-10-11 — Progress review prioritizes a ready-history diagnostic
- Recommendation/scope: for the next numerical study, first verify padded AT2
  residual parity on the current runtime, then test a separately identified
  sequential small-step direct path from the same-model strict .0052 state.
  This is an assessment, not adoption of a new solver or a launch instruction.
- Reason: saved exact-inner128/half64/quarter64 controls remain unready but share
  a warning .0054 neural predecessor. Strict direct .0052 exists with matching
  model identity, checkpoint SHA and saved acceptance gates, providing a useful
  independent diagnostic parent. Current fixed-H neural force correction is tiny.
- Runtime evidence: live RTX5090/JAX0.10.1 differs from prior RTX4080 validation;
  GPU is idle and disk has~78GiB available. Historical timing/tests must retain
  their environment context; no broad test rerun is warranted for this review.
- Alternatives rejected: another unchanged512-sweep .0053 run, more full-curve
  warning training, larger networks as the first remedy, or direct-history
  injection into neural training. If a ready-parent direct path still fails,
  inspect the coupled map before selecting a checked stabilization/formulation.
- Relation: current user progress/difficulty inquiry and saved outer/CG/gate
  artifacts; same active fracture task, no production restart or Git operation.

## 2026-10-11 — Reframe fracture strategy around nonlocal differentiation
- User clarification: the overarching goal is to solve2D fracture using
  nonlocal differentiation. Existing A2 mechanics/history/Picard/PINN choices
  are implementation choices, not immutable requirements for the whole task.
  Broaden the existing task goal; retain earlier branch contracts and artifacts.
- Recommendation: first specify a separate unified-energy, irreversibility-
  constrained direct baseline using local mechanics and nonlocal scalar damage.
  Then assess energy-derived nonlocal mechanics and optional neural energy
  approximation. This review does not adopt a new PDE or launch implementation.
- Reason: accurate-inner controls still fail outer convergence; the older
  branch also has checked mixed-energy and crack-transmission issues. Combining
  all of these with NN training obscures validation of the differentiation idea.
  A scalar-only nonlocal stage proves only that component, not all derivatives.
- Alternatives/limits: current-model stabilization remains a diagnostic option;
  arc-length is conditional on path instability; dynamic/viscous evolution is a
  separately declared physical/path choice requiring rate/energy checks. No
  solver is claimed to guarantee global minimization or every critical state.
- Primary references: Farrell/Maurini2017 DOI10.1002/nme.5300 (composite
  alternate minimization/Newton and constrained stopping); Wambacq et al.2021
  https://arxiv.org/html/2011.10125v2 (irreversibility/interior-point methods);
  Bharali et al.2022 https://arxiv.org/abs/2202.10040 (history versus variational
  irreversibility) and https://arxiv.org/html/2111.10649v1 (energy arc-length);
  Zhang/Ren2022 https://link.springer.com/article/10.1007/s00366-021-01537-x
  (energy-derived NOM with stabilization); Zhuang/Ren/Rabczuk2021
  DOI10.1016/j.euromechsol.2021.104380 (explicit dynamic NOM);
  Manav et al.2024 https://arxiv.org/html/2404.13154v1 (Deep Ritz fracture
  using NN nodal fields/numerical spatial gradients; remaining representation,
  quadrature and optimizer limitations). Proposed nonlocal substitution in the
  latter is our research inference, not a reported result of that paper.
- Relation: current user global-strategy request; same active task, old solver
  unchanged. New formulations need a checked specification before coding.

## 2026-10-11 — Narrow the explicit-constraint recommendation
- Clarification: maximum-history treatment is an established mainstream phase-
  field fracture approach. Its difference from the original constrained energy
  problem is not evidence of invalidity or the cause of this project's failure.
- Recommendation correction: simplification may retain a standard referenced
  history formulation as the first baseline; exact irreversible inequalities
  are a separately identified comparison route, not a required replacement.
  If strict constrained energy minimization is selected, use its proper KKT
  conditions. No solver choice or physical formulation is adopted by this reply.
- Reason: previous recommendation over-weighted variational consistency without
  a matched history-versus-constraint failure control. Accurate-inner controls
  establish outer difficulty, not history-specific causality. Changing several
  formulation ingredients together would obstruct attribution.
- Alternatives rejected: treating wide adoption as universal equivalence or
  convergence proof; treating non-equivalence as proof that history is wrong;
  promising that an active-set replacement resolves nonlinear fracture.
- Sources: Miehe/Hofacker/Welschinger2010,
  https://www.sciencedirect.com/science/article/pii/S0045782510001283;
  Wambacq et al.2021 https://arxiv.org/html/2011.10125v2, brittle benchmark
  comparisons show similar responses with history and constrained approaches,
  with problem-dependent cost and differences in crack width.
- Relation: user's history-field adoption question; narrows the preceding
  strategy recommendation, preserving existing code/results and same task.

## 2026-10-11 — Separate paper contributions from hybrid feasibility
- Assessment: local mechanics plus nonlocal damage PINN does not by itself
  establish a PINN advantage over FEM or matched direct nonlocal damage. The
  current x-only/per-load optimization has no demonstrated solve-speed or
  unseen-parameter benefit;99329 NN parameters exceed71307 scalar nodal unknowns,
  so current parameter counts also do not establish compression.
- Recommendation: for the stated differentiation goal, prioritize a matched
  derivative/representation study with a direct discrete-model control and
  refined local FEM reference. Physics/model changes are distinct from NN
  effects. A reusable parameter/history-conditioned damage predictor with
  residual correction is a proposed separate route if many-query utility is
  desired, not an implemented ability of the current PINN.
- Evidence standards: accuracy and critical-path robustness before speed;
  report operator/build/JIT/train/inference/correction/mechanics costs and all
  failures. For reuse, include held-out complete cases and offline/online
  break-even cost. Do not compare warning/inaccurate outputs as equal solutions.
- Alternatives rejected: paper claims based on an attractive crack image,
  conflating nonlocal operator savings with NN advantages, claiming hybrid FEM
  is globally mesh-free, or declaring first/novel without a full literature audit.
- Sources: Haghighat et al.2021 https://arxiv.org/abs/2006.00446 already combines
  nonlocal differentiation/PINNs; Manav et al.2024
  https://arxiv.org/html/2404.13154v1 sec4.7 reports neural fracture costs above
  FEA and motivates parametric reuse; Goswami et al.2022 DOI10.1016/j.cma.2022.114587
  https://www.sciencedirect.com/science/article/pii/S004578252200010X studies
  physics-informed fracture operator learning. Search is preliminary, not an
  exhaustive novelty review; future-dated2026 issue entries were not used to
  establish current publication priority.
- Relation: user asks PINN utility and paper story; same task, no formulation
  adoption or numerical launch. Contribution statements remain hypotheses.

## 2026-10-11 — Execute the authorized local-mechanics isolation trial
- Scope: user approves trying the proposed changes. Implement a distinct P1
  local FEM mechanics backend, then direct/PINN nonlocal damage comparisons.
  Retain fine cloud, cubic damage delta.008, plane strain/material/l0, maximum
  previous-committed history, volume-grip geometry/constraints and omega.5.
  First path matches the current128x6 trial's fresh.0048 selected-load start.
- Formulation: reuse solver.py's documented Kristensen/Martinez-Paneda hybrid
  full-isotropic degraded elasticity and plane-strain Amor positive driver.
  Material P1 degradation is integrated with degree2 triangle quadrature;
  undegraded grips retain1+k_res. Cell-constant Amor is transferred to material
  nodes by the declared lumped-L2/area projection. Validate stiffness/driver
  against independent FEniCS/UFL before physical runs. This is local mechanics
  and a specified driver discretization change, not strict common-energy solving.
- Reason: distinguish local/nonlocal mechanics from damage-network fitting.
  Keeping volume grips avoids silently changing the loading BVP at the same time.
  Separate exact-damage and NN paths have their own q/d/H histories; direct
  states are comparisons, never neural parents or targets.
- Controls: require independent explicit inner residuals and outer convergence
  before advancing. Matched loaded-state/frozen-H comparisons distinguish
  full-history effects. Runtime checks are serialized; FEniCS and JAX GPU
  environments stay exclusive. Old accurate-inner warning controls are context,
  not automatically matched histories or continuum truth.
- Alternatives rejected: new irreversible-constraint physics, silently importing
  direct H into PINN, changing BC/collars with mechanics, or accepting exhausted
  iterations as improved accuracy. No unconditional convergence claim.
- Relation: current user execution approval; same active fracture task.

## 2026-10-11 — Extend the new-model critical control and isolate scalar gradients
- Evidence: independent UFL and current-GPU checks pass. New local-mechanics/
  nonlocal-damage path is ready through.0052 but.0053 exhausts128 sweeps, AT2
  .077822/d-change.008233. Local mechanics alone is not a demonstrated remedy.
- Diagnostic decision: continue this new-model same-load trial128 to total512
  using its actual ready.0052 floor; if unready, independent half/quarter targets
  from that floor up to256. Compare a standard local FEM gradient damage matrix
  using the same nodal mass/history on this conditional common parent. The latter
  is an operator comparison, not a certified history path of the changed model.
- Reason: distinguish transient propagation/budget from scalar operator effects.
  This is not a repetition of unchanged old-model Picard controls. NN training
  remains conditional until a useful direct baseline is identified.
- Validation: local scalar stiffness independently matches UFL to6.89e-16,
  constants defect1.55e-15. CPU/GPU cubic coefficients share exact CSR structure
  and differ relatively5.40e-16; compare using existing1024-epsilon coefficient
  bound rather than an overly strict absolute cache threshold. Solver and physical
  gates are unchanged. An initial JSON NumPy-scalar serialization bug was fixed;
  failed startup artifacts are retained separately from the v2 physical run.
- Alternatives rejected: accepting lower reaction as convergence, importing
  conditional-control history into PINN, or declaring the history field causal.
- Relation: authorized attempt; independent diagnostic cases, no old-state edits.

## 2026-10-11 — Distinguish a propagating crack from a stalled outer solve
- Evidence: new local-mechanics/nonlocal-damage full.0053 remains unready after
  512 (R417.15,AT2.04766), but d>.99 front advances from x~.513 at128 to~.644
  at512. Reaction continues falling with a moving front; no fixed-root stagnation
  or no-equilibrium conclusion follows from this budget exhaustion.
- Decision: one bounded continuation to total2048 using the same ready.0052
  floor and source512 q/d guess only. If ready, a separately identified fresh
  whole selected path can verify the larger-budget policy. No failed H advances.
- Other controls: sequential quarter steps pass.005225 and.00525 then reject
  .005275 at512. Conditional local-gradient damage contracts to4.15e-9 but
  heals relative to the different-model predecessor, so is not accepted.
  A fresh-own-history local-gradient path addresses that ancestry confound.
- Alternatives rejected: equating declining force with convergence, treating
  transient H as a committed floor, or silently raising old run budgets.
- Relation: authorized local-mechanics experiment, field evidence before the
  final bounded study; no neural training or formulation change implied.

## 2026-10-11 — Verify the successful direct branch and bound the PINN comparison
- Evidence: separate fresh.0048-.0075 local-mechanics/nonlocal-damage run with
  max2048 passes all28 loads, peak606.591@.0052,terminal.554980,worst AT2~4.16e-5,
  zero drift. Same-load control reaches readiness at1294,R1.57878. Larger budget
  and local mechanics both differ from earlier failed experiments, so no single-
  factor causal or continuum-accuracy claim is justified.
- Decision: serialize the existing128x6 own-history PINN on the same local model
  for4 outer sweeps×2000 updates, starting.0048; stop unready advancement. This is
  a matched short-budget diagnostic, not a full PINN convergence experiment.
  Then check the direct branch's complete0-.0075 own-history loading to remove
  the skipped-prefix limitation of the successful selected schedule.
- Numerical control: whole-path config lowers only CG absolute floor1e-10 to
  1e-13 for small-load volume-weighted damage RHS; relative1e-9 and all outer/
  physical thresholds unchanged. A smaller requested floor is explicit, never
  an accuracy relaxation or backend fallback. New config/root keeps old bytes.
- Alternatives rejected: calling selected loads a verified zero-start path,
  claiming PINN success from direct success, or reusing direct H as neural history.
- Relation: authorized attempt and completed selected-path evidence; no Git
  operations or constitutive/history changes.

## 2026-10-11 — Retain the verified direct path and separate the neural limitation
- Evidence: local-P1 mechanics/nonlocal-AT2 with the existing committed maximum
  history passes all75 loads from zero through.0075. Critical.0053 needs1296
  outer sweeps; peak606.590692@.0052, terminal.554966, maximum accepted current
  AT2.00024475, zero drift. Same-model128 budget fails. Independent CPU CSR
  residual replay differs from GPU reports by at most3.95e-15; UFL assembly,
  checkpoint ancestry and completed/rejected resume checks pass.
- Decision: keep this separate direct branch as the verified discrete-model
  baseline for the broader nonlocal-differentiation goal. Preserve history
  treatment and existing stopping thresholds; do not replace history solely
  because the earlier A2/neural paths failed. Close this bounded experiment,
  leave the broader task active, and record spatial/horizon/load sensitivity
  and accurate neural inner continuation as remaining work.
- Neural limit: own-history128x6 PINN4×2000 updates rejects its first.0048 load,
  image AT2.04080/current.14893. Direct success does not establish PINN success;
  increasing per-load training without a budget/residual policy is not adopted.
- Interpretation: field snapshots show a traveling crack at128/512 followed
  by through-crack convergence. This disproves permanent stall for this new
  discrete path within2048; it is not a global theorem or proof of the old A2
  cause. Mechanics/projection and budget both changed. Continuum accuracy,
  energy/work consistency and superiority to FEM remain unestablished.
- Alternatives rejected: unconditional convergence claims, relaxing gates,
  promoting conditional controls, restarting failed paths past rejected loads,
  or treating the selected.0048 schedule as the only loading verification.
- Relation: user-authorized attempt, completed zero-start and selected-path
  evidence; original checkpoint SHA remains unchanged, no Git operations.

## 2026-10-11 — Compare fixed-history neural optimization before more loading
- Authorization: user says continue execution after the successful local-
  mechanics/nonlocal-damage direct path and proposed PINN improvements.
- Decision: freeze the prior own-history PINN.0048 H_used and weights; compare
  extra Adam, residual L-BFGS and fixed-H damage-energy L-BFGS with explicit
  bounded budgets. Independently verify the latter's derivative is M times
  the existing AT2 residual for the reciprocal volume-symmetric graph.
  It is not a joint u/d energy or a new irreversible formulation.
- Reason: distinguish optimization conditioning from outer propagation before
  repeating2000 updates for potentially~1296 outer sweeps. Keep architecture,
  precision, geometry, history and acceptance thresholds fixed in these pairs.
- Alternatives rejected: new direct-supervised loss or direct-history seed,
  an unbounded full PINN run, or treating energy decrease alone as convergence.
  Algorithm controls are experiments, not universal optimizer guarantees.
- Relation: current authorized continuation; same active task, separate roots,
  source artifacts remain byte-identical and no Git operations authorized.

## 2026-10-11 — Validate loading sensitivity after bounded neural controls miss gates
- Evidence: from identical own-H/weights, extra Adam2000 leaves image AT2
  .039920, residual L-BFGS512 .035569, energy L-BFGS512 .038802; none reaches
  .001. Actual-grid energy derivative agrees to6.52e-10 and3 CPU tests pass.
  A modest optimizer improvement is not a converged PINN or a cost superiority
  claim. Source bytes remain unchanged; no diagnostic state is promoted.
- Decision: locate these neural residuals/field errors on the actual cloud,
  and check the successful direct model's full own-history loading with
  increments reduced1e-4 to2.5e-5 only between.0052 and.0056. Same2048 maximum,
  matrices/material/history and stopping gates; separate root/config87 loads.
- Reason: establish load-step sensitivity while representation/optimization
  remains unresolved, rather than claiming the direct curve is quantitatively
  validated or repeating expensive unrestricted per-sweep training.
- Alternatives rejected: full neural continuation with unready inner solves,
  changing physical parameters or gates, or importing direct targets.
- Relation: authorized continued execution, same active fracture task;
  spatial/horizon and physical-energy validation remain distinct future checks.

## 2026-10-11 — Test an explicit neural predictor with nodal residual correction
- Evidence: frozen-H CPU probe reduces energy-trained neural AT2.03880 to
  .000433 after16 damped-Jacobi updates, preserving bounds. Residual-trained
  candidate reaches.000493 after32. Zero initialization remains.000707 at512.
  This is a fixed-H probe, not evidence of whole-path or direct-CG superiority.
- Decision: implement a separately labeled PINN predictor plus nonlocal nodal
  Jacobi corrector; test fresh own-history selected loading after the current
  direct sensitivity worker exits. Train predictor once per load with bounded
  energy L-BFGS, then warm-start corrected fields between outer sweeps. No
  direct targets or imported history. Report raw neural and corrected residuals.
- Derivation: A=diag(gc0+2H)+gc1 D is the existing M-matrix. Update
  d+=-omega*r/diag(A), omega=.8. Reciprocal negative off-diagonals, zero row sums
  and gc0>0 imply a contraction and positivity/boundedness preservation for
  initial0<=d<=1. Check assumptions and independent CPU/JAX iteration parity.
  Explicit image limit1e-5, max2048/check16, unchanged outer gates; fail loudly
  if correction budget is insufficient. No global coupled-map guarantee.
- Interpretation: this is a hybrid nodal damage solve with a neural initial
  predictor, not a purely neural field solution. It bypasses the observed local
  residual wall without disguising direct numerical work or promising savings.
- Relation: user-authorized continuation and broader differentiation goal;
  model/history unchanged, separately identified artifacts and no state promotion.

## 2026-10-11 — Project only an inadmissible predictor initialization
- Evidence: hybrid selected loads.0048-.0053 pass outer gates; critical.0053
  takes1298,R1.578703. At.0054, once-per-load64-step neural energy predictor
  reaches max1.015180, so Jacobi correctly rejects its invalid initial field.
  No.0054 physical damage/history is saved or advanced.
- Decision: explicitly project the predictor initial guess to[0,1] before
  nodal iteration in a separately identified v2 projected root. Raw predictor
  and its residual remain separately reported. Do not clip the corrected field
  or loosen acceptance; the same checked AT2 fixed point governs readiness.
- Provenance: fork only the original6 ready own-history hybrid states, preserve
  payloads/parent/network SHA and check full model/history/solver config against
  the v1 identity. The unfinished state7 is not copied. Old root is retained.
- Reason: bounded initial damage is a premise of the explicit M-matrix
  correction. Projecting an iterative initial guess changes neither the
  physical equation nor the committed history; this is not a claimed NN cure.
- Alternatives rejected: ignoring the upper-bound violation, relaxing ranges,
  relabeling the original failed point, or replaying the expensive unchanged
  critical prefix when six verified own-history predecessors already exist.
- Relation: same authorized continuation; concrete failure repair, no direct
  data/foreign history or Git operations. Additional cost/success remains measured.

## 2026-10-11 — Close the continuation with a verified bypass and honest cost limits
- Results: refined own-history direct87/87 ready, peak/terminal changes
  +.1802/-.4837%. Projected hybrid selected28/28 ready, max current AT2~4.51e-5,
  zero drift; matched direct reaction error<=.049685%,d inf<=.00013703.
  CPU residual/ancestry/fork/source and actual completed resume checks pass.
- Decision: retain direct nonlocal damage as the practical lower-cost route.
  The neural predictor/nodal corrector is a validated discrete-model bypass,
  not pure-PINN convergence: raw terminal neural residual~.989. Recorded summed
  load compute549.56s versus direct295.75s, excluding hybrid pretraining/setup/
  retries, establishes no speed benefit. No additional run pending this session.
- Remaining: spatial/horizon/FEM quantitative and physical-energy/work checks;
  pure-neural representation needs a separate study if that goal is selected.
  Load-step sensitivity is not continuum validation. Broad task stays active.
- Alternatives rejected: calling nodal correction a NN solution, claiming
  savings from fewer training iterations alone, or treating partial v1 failure
  as a completed path. Both bound source versions/payloads are preserved.
- Relation: user-authorized continuation complete; four new focused tests pass,
  old runs intact, GPU idle, no Git history operations.

## 2026-10-11 — Prioritize fracture accuracy validation before further neural tuning
- Decision: recommend the converged local-P1-mechanics/nonlocal-AT2 direct branch
  with committed maximum history as the practical validation baseline. First
  complete a same-condition own-history local-gradient control and audit a
  comparable independently converged FEM fracture reference. Then study spatial
  refinement at fixed horizon and horizon reduction on adequately resolved
  clouds, holding the physical phase-field length fixed.
- Reason: discrete/load convergence is now demonstrated, but continuum accuracy
  is not. Corrected neural damage is a nodal solve and its measured cost exceeds
  direct solving; more optimizer iterations alone do not answer the user's
  broader nonlocal-differentiation goal. Existing UFL assembly checks do not
  certify a complete fracture path; warning FEM states are not reference truth.
- Alternatives deferred: further full-path PINN capacity/budget tuning, mandatory
  replacement of maximum history, returning immediately to A2 mechanics, or
  simultaneous grid/horizon/physical-length changes that obscure interpretation.
- Relation: same active fracture task; user asks for next-step judgment. This
  session records a recommendation only: no new PDE, configuration or numerical
  run adopted. Pure-neural representation remains an optional separate study.

## 2026-10-11 — Execute an own-history local-gradient damage control
- Authorization: user approves the recommended matched local-damage/FEM
  validation step. Retain the same active fracture task and preserve all roots.
- Decision: use standard material P1 stiffness S and D_local=M_lumped^-1 S
  for scalar damage, with the same lumped nodal reaction/history, local P1
  mechanics, volume grips, material, own-zero loading and outer acceptance gates
  as the completed nonlocal-damage branch. Rebuild/check S against its existing
  independent UFL assembly and bind input/runner SHAs in a separate runner/root.
- Reason: changing only the damage spatial operator isolates a meaningful
  discrete comparison. The earlier local control nearly contracted but exhausted
  512 sweeps; use the already verified 2048 bounded budget, not warning acceptance.
  Independently evaluate equilibrium/driver/damage on accepted fields in fenicsenv
  after JAX exits. Old solver.py uses boundary loading, DG0 history and consistent
  mass, so its warning curve cannot serve as an identical-model strict reference.
- Alternatives rejected: seeding local damage with nonlocal history, silently
  modifying the old SHA-bound runner, changing maximum history or loosening
  gates. This validates a matched lumped P1 reference, not continuum convergence.
- Relation: first priority of the approved validation plan; no PINN retraining,
  explicit inequality model or Git history operation is selected.

## 2026-10-11 — Match the existing critical-load refinement in the local control
- Evidence: own-zero local control is ready through54; near-critical.0054
  contracts to readiness at739 sweeps under unchanged gates, whereas the earlier
  512 trial exhausted its budget. At.0055 the crack is propagating, not yet ready.
- Decision: after this worker exits, run a separate own-zero local control on
  the existing87-load nonlocal quarter-step schedule. Same physics/operator/
  history/gates/2048 cap. Compare both schedules so a peak/jump discrepancy is
  not assigned to spatial operator before checking loading sensitivity.
- Reason: this is a matched accuracy control within the approved comparison;
  no neural training, different-model history import or warning advancement.
  Independently replay all completed fields in fenicsenv after GPU work exits.
- Relation: same active task; broader grid/horizon convergence remains deferred.

## 2026-10-11 — Audit sparse-LU differences against the declared CG norm contract
- Evidence: independent UFL replay reaches the uniform local terminal oracle,
  then the audit's extra hard-coded image-inf1e-7 assertion fails. Cached-S LU
  probe measures image-inf3.1282e-7; LU relative residual3.98e-16 and independent
  CG residual1.2009e-7 are below the declared CG limit1.3386e-7. Refined terminal
  is similar3.1270e-7. Existing coupled/image acceptance remains satisfied.
- Decision: retain actual field-inf differences as diagnostics. Verify the
  independent image residual against the original explicit CG norm tolerance,
  with a separately computed assembly-roundoff bound, and check the rigorous
  SPD image/LU error bound. Do not modify physical fields, CG settings or gates.
- Independently checked derivation: A=diag(M*(gc0+2H))+gc1 S (or reciprocal
  weighted graph MD), S positive semidefinite, H>=0, M>0. Therefore
  lambda_min(A)>=gc0*min(M), and ||image-LU||inf<=||image-LU||2
  <=||r_image-r_LU||2/(gc0*min(M)). Report this conservative bound and actual
  differences; graph assumptions/standard P1 weak-form positivity remain explicit.
- Reason: the audit's arbitrary1e-7 field threshold is not implied by a weighted
  CG norm contract. This is a correction to diagnostic interpretation, not
  warning advancement or a numerical fallback. Preserve the initial failed log.
- Relation: same matched-damage validation. Re-run the failed audit after this
  explained correction; all completed numerical roots remain byte-identical.

## 2026-10-11 — Close matched local damage validation with a persistent transition discrepancy
- Evidence: new own-zero local uniform75/75 and refined87/87 paths pass all
  original gates. Uniform/refined local peaks618.449649/619.799371, at.0054/
  .005375; terminal.603524/.606168. Independent UFL324-state/eight-LU checks,
  CPU fields/ancestry/final-three-gate replay, focused test and actual completed
  resumes pass. No worker remains; both old nonlocal paths are unchanged.
- Decision: regard the local-gradient/lumped-P1 path as a verified matched
  discrete control. Preserve maximum history and prioritize spatial refinement
  at fixed l0=.024 and horizon.008, then resolved horizon refinement. Do not
  resume whole-path neural training or claim a continuum reference yet.
- Reason: refined nonlocal peak is1.955% lower but unloading occurs earlier
  (.005275 vs.0054 for below1% peak), producing24.42% displacement-weighted
  curve-L2 discrepancy. Terminal profiles/path are similar and the width at
  x.75 is3.36% narrower nonlocally. Small load-step peak changes (.218% local,
  .180% nonlocal) do not eliminate this scalar-operator discrepancy.
- Alternatives rejected: quoting only~2% peak difference as full-response
  accuracy, reusing boundary/DG0 warning FEM as matched truth, claiming a new
  independent trajectory from UFL field replay, or changing history/gates.
- Relation: user-approved validation step complete, same broad active task;
  old artifacts and source snapshots intact, no Git operations authorized.

## 2026-10-11 — Respect the accepted fracture shift and distinguish the FEM controls
- User preference: a modest earlier nonlocal fracture is acceptable; asks if
  the current local control is the earlier FEM model. Do not turn removing
  that shift into a mandatory blocker for the broader differentiation goal.
- Decision: explain shared hybrid AT2/plane-strain/Amor maximum-history physics,
  while explicitly separating old boundary loading, DG0 history and consistent
  scalar mass from current volume grips, projected nodal history and lumped
  scalar mass. The current local control is itself standard P1 FEM.
- Reason: calling the two fully identical would conflate BVP and discrete
  history/reaction choices. Their matched nonlocal/local comparison isolates
  the scalar operator, while UFL replay does not reproduce the old trajectory.
  Abrupt unloading magnifies the curve-L2 metric when transition positions shift.
- Alternatives rejected: equating shared physics with identical numerical
  models, describing the current local field as a PINN, or insisting on zero
  shift contrary to the user's accepted tolerance. Existing results/limits stay.
- Relation: same active task, clarification only; no new run/formulation or Git
  operation. A concrete next implementation goal is not selected in this turn.

## 2026-10-11 — PINN experiments can resume without waiting for continuum refinement
- User asks whether PINN training attempts can begin, after accepting the modest
  earlier nonlocal fracture. Recommend yes on the verified local-FEM/nonlocal-
  AT2/maximum-history baseline; spatial/horizon validation is a separate study,
  not a mandatory prerequisite for neural representation experiments.
- Decision: first bounded fixed-own-H neural-damage representation study, then
  short own-history staggered loading. Prior unchanged Adam/L-BFGS policies
  missed the image residual gate, so geometry/local representation is the next
  candidate to evaluate, not another unrestricted full-path budget increase.
- Reason: the verified direct path provides a useful equation/solver comparator;
  it does not solve the already observed pure-network residual wall. Preserve
  direct mechanics, own neural history, evaluation-only direct fields and the
  distinction between pure neural damage and nodal-corrected hybrid fields.
- Alternatives deferred: whole-path training with an unchanged failed policy,
  waiting for a perfect FEM continuum curve, or silently calling correction
  a PINN cure. No specific architecture/budget or new run is adopted here.
- Relation: same active task; readiness advice only, no code or Git operations.

## 2026-10-11 — Explain the shared vertex-particle coupling and retain PINN agreement
- User agrees to the proposed staged PINN attempt, then asks about FEM-element/
  node correspondence. Retain that authorization for the bounded neural study;
  no repeat permission request is needed. No training launched in this answer.
- Evidence: current prepared cache has71307 material vertices/particles and
  141366 material triangles; union FEM adds6024 grip-only vertices. Material
  coordinates and indexing are shared exactly. Particle area weights equal the
  sum of neighboring material triangle areas/3 (checked to5.08e-21 absolute).
- Clarification: FEM triangle strains drive lumped node history projection;
  neural/direct nodal damage returns through P1 interpolation at mechanical
  quadrature. FEM uses triangle connectivity, nonlocal damage uses the fixed
  radius/initial-geometry visibility graph. PINN evaluates the same node points.
- Reason: this is a shared discretization coupling, not independently scattered
  particles requiring spatial transfer, and not one particle per triangle.
  The current implementation still uses triangles for mechanics, geometry and
  quadrature; do not claim it is globally mesh-free or damage-dependent graph
  cutting. Extra grips do not enter the damage operator.
- Relation: same active task, clarification-only memory edits; current physics,
  acceptance, old artifacts and Git history remain unchanged.

## 2026-10-11 — Start a geometry and own-history conditioned neural damage study
- Authorization: user explicitly asks to execute the agreed three-stage plan.
  Retain local FEM mechanics, nonlocal AT2, committed maximum history and gates.
- Decision: first test a separately identified neural damage representation
  taking coordinates/local geometry, the known own H and its fixed graph
  neighborhood averages. The input includes PDE coefficient information so
  mesh-scale driver variation need not be inferred from coordinates alone.
  Use the old own-neural frozen H checkpoint for stage1, never direct labels.
- Feature derivation: the existing reciprocal positive graph Laplacian has
  P_ij=-D_ij/D_ii for i!=j, P_ii=0. Nonnegative rows sum to one; P^k H are
  fixed known-coefficient neighborhood features, not AT2 damage iterations.
  Check the row/shape assumptions and independent CPU feature/gradient parity.
  Features and neural ansatz change; PDE/operator/history do not. No general
  reuse/speed or unseen-case claim follows.
- Plan: bounded fixed-H residual training and full-field range/residual gates;
  if ready, own-zero short staggered loading, then ready-state critical/full
  continuation. Precompute fixed graph geometry, keep JIT shapes/dtypes fixed
  and warm-start network parameters. Direct solve only after training as oracle.
- Alternatives rejected: further unchanged coordinates-only whole-path training,
  hiding nodal correction in a PINN output, or importing converged direct history.
- Relation: same active task, new explicit configs/roots, no Git authorization.

## 2026-10-11 — Add compact neural readouts after conditioned MLP improves but misses gate
- Evidence:96x3 geometry/history/neighborhood MLP on identical own-H lowers
  residual-MSE2048 to.023680, then residual-L4 to.009014 after2048 more; still
  above5e-4. Independent CPU residual replay passes; source bytes unchanged.
- Decision: retain that trained conditioned trunk and add continuous compact
  radial-hat hidden features phi=max(1-distance/radius,0) with trainable readouts.
  Geometry-owned coarse centers plus centers selected by this neural residual
  provide local detail. No direct fields/targets or nodal AT2 updates are used.
- Reparameterization: for fixed H, column scale s*sqrt(N)/||A_strong phi_j||2
  normalizes the diagonal of the residual-MSE Gauss-Newton matrix in the linear
  positive-field regime. This is a checked optimizer coordinate scaling, not
  a new PDE or a claim of nonlinear/global convergence.
- Reason: locally resolved continuous hidden units test remaining representation
  error with fewer readout variables than71307 nodal unknowns. Neural residual
  selection changes the ansatz only; original physical residual/range gates stay.
- Alternatives rejected: accepting .009 as converged, loosening PDE gates, or
  using a direct image as a neural label. Short/full loading remains conditional
  on successful inner training. Same authorized three-stage task.

## 2026-10-11 — Refine continuous readouts from remaining neural residual
- Evidence: first compact4624-readout experiment stops above the target;
  residual-MSE .006343 and residual-L4 .003406, independently replayed on CPU.
  Source trunk/history bytes remain unchanged. This is improved, unready output.
- Decision: run a separate explicit finer coarse grid plus8192 smaller local
  units, selecting centers by that uncorrected neural residual. Start readouts
  from zero on the original frozen conditioned trunk; use no oracle image.
  Selection history must match exactly and source geometry/physics are checked.
- Reason: residual plateau under a converged reduced optimizer warrants testing
  finer continuous representation, rather than increasing the same iterations.
- Alternatives rejected: accepting the .0034 field, importing direct history,
  weakening gates, or treating per-node numerical correction as neural training.
- Relation: stage1 of the same three-stage authorization; original artifacts
  retained and no physical load advanced. Later stages still require readiness.

## 2026-10-11 — Permit coarse/fine units at the same tip/corner position
- Evidence: first compact residual maxima occur at corners and near the tip;
  finer-grid18247 readouts still miss the gate (.005677). The center selection
  excluded adaptive units around every coarse center, preventing smaller units
  at precisely these positions. Increasing total features alone did not help.
- Decision: apply separation only among adaptive centers; permit a small radial
  unit and a coarse radial unit at the same coordinate. Test this behavior and
  run8192 adaptive units with the original coarse grid in a separate root.
- Reason: different radii are distinct continuous functions; coarse presence
  should not suppress finer representation. This is an ansatz selection fix,
  not a change to AT2, history or its acceptance thresholds.
- Alternatives rejected: physical advancement from these unready fields or
  judging capacity by feature count alone. Same authorized stage1 investigation.

## 2026-10-11 — Isolate positive-baseline readout conditioning
- Evidence: permitting multiscale center overlap improves fixed-H residual to
  .001945 but still misses1e-3/5e-4. The trained trunk has raw minimum-.010887;
  about17000 particles remain near zero under its1e-8 positive link, whose raw
  derivative is almost zero for these negative values.
- Decision: one controlled trial uses the frozen trunk's positive image as
  baseline before adding readouts and applying the same positive link. Keep
  centers/radii/objectives/budgets identical to the multiscale trial. The initial
  field changes by at most the declared link floor, while near-zero baseline
  derivatives become approximately one half; check this independently.
- Reason: distinguish saturated-output conditioning from remaining basis error.
  The PDE, history and physical range/residual gates stay unchanged; no direct
  field participates in the ansatz or training.
- Alternatives rejected: more unchanged iterations or proceeding at .001945.
  Same bounded stage1 study; later loading still conditional on readiness.

## 2026-10-11 — Preserve learned features when adding another local scale
- Evidence: positive-baseline control reaches .001942 versus .001945 with the
  raw baseline, so output saturation does not explain the remaining maximum
  defect on this feature set. It slightly improves field L2 error but is unready.
- Decision: preserve the trained10768 continuous columns/readouts and add4096
  smaller units selected from their own residual. Recompute explicit column
  scales and transfer coefficients inversely so the initial physical field is
  identical to the parent; verify old columns and predicted fields independently.
  Only neural weights/features transfer, never direct damage or physical H state.
- Reason: a nested representation can test the residual remaining between
  resolved scales while retaining the previous trained field. This avoids
  conflating extra capacity with replacement of useful old features.
- Alternatives rejected: declaring1e-3 convergence from .00194, using optimizer
  termination as a physical gate, or advancing short loads without stage1 readiness.
- Relation: final nested enrichment of this fixed-own-H study, same active task.

## 2026-10-11 — Confirm original gate and test a further local scale
- Evidence: nested14864-unit field reaches AT2 .000901082, independently
  matching CPU. It passes the original1e-3 residual criterion but not the more
  conservative fixed-H5e-4 training target chosen for this study.
- Decision: retain the learned field and add4096 units at1.05 times nearest
  spacing, with explicit radius bounds, in a new root. Keep the conservative
  target; this is one further continuous representation scale, not a changed
  PDE, relaxed gate, or per-node direct numerical update.
- Reason: original-gate success makes a short coupled neural test credible,
  but a tighter inner margin should reduce error accumulation when H changes.
- Alternatives rejected: calling fixed-H success a coupled/full-path result or
  silently lowering the configured target to move on. Same authorized task.

## 2026-10-11 — Emphasize residual peaks without relaxing the inner target
- Evidence: further18960-unit scale improves residual to.000766756, still
  passing original1e-3 but missing the study's5e-4 margin. L4 reaches a plateau.
- Decision: explicitly select a degree-two L16 objective
  (mean((r/(2RMS(H)))^16)+epsilon^16)^(1/8), preserving learned columns/readouts
  and adding1024 remaining-residual centers in a separate root. Validate its
  parameter derivative independently. Methods are explicit configured stages.
- Reason: full-field acceptance uses a maximum norm, so a stronger smooth peak
  objective tests whether a few extremes rather than average error block the
  conservative target. Physical equation, positive link and all gates stay.
- Alternatives rejected: reducing the declared inner target or further replacing
  the feature set. Same fixed-own-H stage, no direct target or physical advancement.

## 2026-10-11 — Finish the peak-margin study and prepare own-zero transfer
- Evidence: L16/additional1024 features reaches.000546388;2048 further units
  reaches.000509709, with the original1e-3 gate retained and no physical parent.
- Decision: keep the entire22032-unit learned field (zero added units) and test
  the degree-two L32 objective at the same5e-4 target. Evaluate power objectives
  with a detached maximum scale to avoid power overflow/floor underflow; this
  algebraically preserves the objective, checked by directional derivatives,
  including zero residual. No change to positive link or physical gates.
- Own-zero entry prepared: direct verified FEM mechanics and current-H-conditioned
  frozen trunk plus trainable compact readouts. Readout contribution scales with
  load squared. Seed must reconstruct and meet its declared residual target;
  physical q/d/H start at zero, independent of diagnostic seed arrays.
  Inner rejection stops before damage relaxation/commitment. Original outer
  checks plus raw-network AT2 at updated H determine physical readiness.
- Alternatives rejected: advancing the .000510 field under the5e-4 target,
  using optimizer termination as readiness, importing its H or a direct target.
- Relation: same three-stage task; loading is implemented/tested but not started
  until the fixed-H gate passes. No concurrent FEM/GPU workers or Git operations.

## 2026-10-11 — Start own-zero neural loading after fixed-H target passes
- Evidence: unchanged22032-unit basis with L32 reaches .000481981657 in32
  iterations, passing5e-4/range. Host residual matches to4e-19; evaluation-only
  direct oracle d inf error.00105093, relative L2 .0119778. Source bytes unchanged.
- Decision: start the own-zero two-load transfer test (.0001,.0002), inheriting
  frozen trunk/continuous basis/readouts only. All physical fields start zero.
  Dynamic-H/load gradients, source reconstruction, original gates and strict
  inner-before-relaxation rejection are checked. No direct damage solver runs
  inside this physical entry.
- Reason: fixed-H target is met, so the next uncertainty is adaptation to the
  coupled solver's own evolving H, not further unchanged checkpoint training.
- Alternatives rejected: treating fixed-H readiness as whole-path success or
  importing seed H as a physical history floor. Same three-stage authorization;
  critical/full continuation still requires ready own-history short states.

## 2026-10-11 — Test joint neural adaptation after own-zero frozen-trunk rejection
- Evidence: first own-zero .0001 load with frozen conditioned trunk and trainable
  readouts rejects its first inner solve: MSE512 .082463, L16 another512 .0367503.
  No inner sweep accepted, no damage relaxation, no committed load/history and
  no .0002 advancement. Independent host mechanics force passes.
- Decision: preserve this root; start a new own-zero control training both the
  conditioned trunk and readouts, with explicit2048 updates per objective.
  Seed reconstruction and all physical gates remain; both sets of parameter
  derivatives are independently checked. No direct targets or nodal correction.
- Reason: fixed-H representation success does not establish H adaptation. Frozen
  coefficient response and optimization budget are unresolved, so this control
  explores joint adaptation. Larger budget is a confound; improvement would not
  isolate freezing as the cause without a matched frozen-budget control.
- Alternatives rejected: advancing a .03675 candidate or treating the diagnostic
  .0048 H as a physical start state. Stage3 still requires own-history readiness.
- Relation: same authorized three-stage task; two-load joint retry now runs.

## 2026-10-11 — Normalize the known-H feature amplitude under changing load
- Evidence: joint MLP/readout2048 per objective first.0001 load still rejects;
  final L16 residual .003193623. No accepted inner sweep or physical commitment.
  This improves frozen512 control but does not establish the cause of improvement.
- Decision: same joint model/budget/objectives, new own-zero root, normalize only
  neural history features by (load_reference/load)^2. At the seed load this is
  identity; physical AT2 and maximum history always use unscaled own H.
- Derivation: fixed-d linear FEM displacement/strain scales with applied load,
  and its quadratic Amor driver scales with load squared. The neural output
  already carries load squared; leaving input H unscaled drives its coefficient
  features toward zero at low loads. Amplitude normalization tests this issue.
  Do not claim exact homogeneity along a path with changing d/committed history.
- Validation: reference identity, quarter-driver/half-load invariance, detached
  coefficient input and both model/readout parameter gradients pass on CPU.
- Alternatives rejected: altering the history equation, importing a direct H,
  accepting .00319, or claiming frozen architecture alone caused the failure.
- Relation: stage2 bounded same-budget feature-normalization control starts;
  all original physical gates and the5e-4 inner target remain unchanged.

## 2026-10-11 — Separate the added optimizer margin from original physical gates
- Evidence: same-budget load-normalized joint control reaches .000711931,
  passes the verified original1e-3 inner residual criterion, but fails this
  study's added5e-4 optimizer margin. Strict-margin root remains inner_rejected
  with no relaxation or physical commitment; CPU H/force/residual replay passes.
- Decision: create a separate own-zero two-load control with optimizer readiness
  tied explicitly to outer_stop.inner_AT2_relative_limit (original1e-3). Every
  original current-field/inner/mechanics/updates/range/history/healing/drift gate
  plus the updated-H raw-network check remains identical. Keep the5e-4 roots
  rejected under their declared policy; do not relabel or resume them as parents.
- Reason: determine whether the extra heuristic margin prevents evaluation of
  an otherwise original-gate-valid coupled solver. User's goal is a converged
  nonlocal fracture solve under the verified criteria, not a newly imposed
  optimizer margin. This control tests that margin explicitly and transparently.
- Alternatives rejected: editing/relabeling old strict-margin runs, lowering the
  original physical gates, or claiming coupling readiness from fixed-H accuracy.
- Relation: refinement of stage2 within the same authorization. Source seed still
  meets5e-4; physical fields start zero. Stage3 depends on accepted outer states.

## 2026-10-11 — Continue the accepted own-zero neural prefix toward fracture
- Evidence: separate original-gate neural/FEM control passes two loads .0001/
  .0002 with12/11 outer sweeps. Original gates and updated-H raw-network checks
  all pass. Final inner residuals .000874442/.000981411; host FEM force checks
  pass. No direct field/history/targets or nodal correction are used.
- Decision: resume exactly that accepted own-history prefix and request75 loads
  through .0075. Full config changes only end_state_index; canonical identity
  parity is checked, code/config/seed and state/network ancestry are enforced.
  Do not resume any rejected5e-4-margin root as a physical parent.
- Reason: stage2 has actual own-history coupling evidence under the verified
  original criteria, so stage3 can test critical/full loading. Fixed-H/small-load
  success alone does not imply critical fracture convergence or speed benefit.
- Alternatives rejected: claiming a completed fracture path from two small
  loads, resetting history at later loads, or silently skipping a failed state.
- Relation: same authorized three-stage task; one serialized full-prefix worker.

## 2026-10-11 — Fork the accepted neural prefix to prioritize the peak objective
- Evidence: original-policy full worker accepts17 loads through.0017. Many new
  loads exhaust2048 MSE iterations at~.002 maximum residual, then meet1e-3 with
  L16. Peak accuracy is the actual acceptance condition; MSE consumes most time.
- Decision: interrupt the worker intentionally (exit130), retain its17 accepted
  states and incomplete diagnostics, and fork only the accepted own-neural
  prefix into a separate root with L16 before MSE. No original numerical code,
  formulation, physical gate, model architecture or history is changed.
- Provenance: parent identity SHA/prefix count, state/network/report hashes,
  original-gate pass status, zero-origin and exact own-history ancestry are
  checked before copying immutable accepted fields. Fork config/manifest binds
  provenance and wrapper code; loader then uses its normal strict resume checks.
  Parent code snapshots match saved hashes. A focused copy/rejection/resume test
  passes. The interrupted pending state is not copied or advanced.
- Reason: continue the user's critical-loading goal without spending a full
  mean-loss budget before every peak-loss solve on warm-started neural states.
- Alternatives rejected: relabeling an unready state, importing direct history,
  silently editing the original run's optimizer identity, or relaxing gates.
- Relation: stage3 continues from the accepted own-zero17-state ancestry in the
  declared optimizer fork; one GPU worker, old root preserved read-only.

## 2026-10-11 — Preserve the31-load prefix and test L32 at the first rejected load
- Evidence: peak-first neural branch passes31 loads through.0031. At.0032 its
  first L16 inner2048 reaches .001001699 (best .001001427), missing1e-3;
  subsequent MSE2048 worsens the peak to.001553864. State32 is inner_rejected,
  no outer update or history commitment. Worker exits normally; no skipping.
  Independent31-state CPU gate/force/history/ancestry replay passes. Maximum
  matched direct reaction difference .016447%, d inf difference .000712027.
- Decision: fork only the accepted31-state own-neural prefix and test L32 before
  L16, retaining1e-3 original inner/current and every physical/outer gate. Extend
  the declared objective factory only; the image/PDE/operator/history loop is
  unchanged. Both model/readout L32 directional derivatives pass. No new nodes,
  direct targets or corrected fields are introduced.
- Provenance: archive parent numerical/fork code matching original hashes;
  fork manifest records old and new numerical-runner hashes, while new identity
  pins the new objective extension. Copy/rejection/resume test checks both code
  hashes. Accepted physical ancestry is preserved; rejected32 is not copied.
- Alternatives rejected: accepting the .0010017 near miss, silently editing its
  gate, relabeling the rejected state, or importing a direct damage/history floor.
- Relation: continued stage3 experiment in a separate L32 root; critical loads
  and the full75-state trajectory are still unverified. One serialized worker.

## 2026-10-11 — Record the32-load result and an operator-aware next candidate
- Evidence: L32 branch accepts.0032, extending own-zero ancestry to32 loads.
  First.0033 innerL32 best/final .001003165/.001003963, thenL16 .001053067;
  original1e-3 criterion fails. No outer update/33 commitment. Worker exits0;
  GPU idle. CPU field/force/driver/history/ancestry/gate replay passes. Matched
  direct reaction maximum discrepancy .0297092%, d inf .000849341 for32 loads.
- Decision: retain this stopped physical prefix and diagnose/train on its own
  rejected33 H before another physical continuation. Do not keep increasing
  identical budgets, relax gates or claim a critical fracture solution.
- Next candidate (derived, not implemented): with D_i=D_ii>0 and stochastic
  P=I-diag(D_i)^(-1)D, set C_i=gc1 D_i, a_i=gc0+2H_i+C_i,
  d_theta=(2H_i+C_i sigma_theta)/a_i. Then the original residual is exactly
  C_i(sigma_theta-Pd_theta). If0<=sigma_theta<=1,0<=d_theta<1, and the full-space
  auxiliary fixed-point map sigma=P(b+W sigma) has infinity norm bound<1.
  Known operator diagonal/H handles local response; the learned unknown is a
  neighbor-average field. This is an ansatz change, not a per-node correction
  iteration, direct target, changed PDE or irreversibility rule.
- Reason: test an explicitly operator-aware smoother unknown after raw-damage
  representation/optimizer plateaus. Actual failed33 H/operator algebra replay
  agrees to1.73e-12 and PW infinity norm .992280322<1; training improvement and
  coupled readiness remain unknown. Require independent parameter-gradient
  checks and fixed-own-H gates first.
- Alternatives rejected: promoting the near miss, feeding direct history, or
  treating32 subcritical loads as a completed fracture path. Same active task;
  exact diagnostic/accepted-parent paths and resume point are recorded.

## 2026-10-11 — Clarify bounded output versus operator-aware reconstruction
- Context: user asks whether the proposed diagonal reconstruction is an output
  constraint and suggests sigmoid(raw) or exp(-raw).
- Finding: current loading code uses the smooth nonnegative damage head from
  train.py; it has no upper-bound guarantee. The positive-baseline experiment
  was not a sigmoid experiment. No bounded-head training result exists yet.
- Clarification: sigmoid(raw) gives0<d<1. exp(-raw) is positive but exceeds1
  for negative raw; exp(-softplus(raw)) equals sigmoid(-raw). These range maps
  do not establish loadwise irreversibility or a converged AT2 residual.
- Candidate: using sigma=sigmoid(raw) in the previously derived output
  d=(2H+gc1 D_ii sigma)/(gc0+2H+gc1 D_ii) both bounds damage and incorporates
  known local coefficients. This is distinct from applying sigmoid directly
  to damage; the auxiliary field represents the nonlocal neighbor average.
- Reason: distinguish physical range guarantees from the unresolved neural
  optimization issue. Sigmoid derivatives vanish near either endpoint; better
  convergence cannot be inferred from boundedness or full-space contraction.
- Alternatives rejected: attributing failure to absent upper bounds without
  evidence, claiming sigmoid was tested, or changing the history treatment as
  part of an output-map clarification. Relation: same active task; controlled
  fixed-own-H plain-sigmoid/lifted-sigmoid comparison is a proposed diagnostic,
  not an implemented solver, adopted model or launched training job.

## 2026-10-11 — Authorize a controlled bounded-output experiment
- Decision: user explicitly authorizes execution and adds exp(-raw^2). Compare
  nonnegative, sigmoid, exp_square, lifted_sigmoid and lifted_exp_square at the
  same rejected33 own H before coupled loading. Retain original1e-3 gates.
- Derivation: exp(-z^2) lies in(0,1] and has derivative -2z exp(-z^2), zero at
  z=0. Inverse positive branch sqrt(-log(d)) initializes a partial-damage field
  away from that stationary full-damage state. Both bounded heads also apply
  to the auxiliary neighbor field in the previously derived exact decoder.
- Initialization: freeze the rejected own-neural predictor and form a common
  initial physical field by the declared diagonal/H decoder applied to that
  predictor's bounded image. Each head inverts this same field (or its
  auxiliary image); an identically initialized conditioned MLP/compact residual
  network supplies trainable latent increments. No direct labels, learned
  per-node freedoms, iterative numerical damage correction or history import.
- Parameterization: normalize compact tangent residual columns independently
  for each head and the MLP scalar gain by its initial decoder derivative RMS.
  This is explicit parameter scaling, not a changed PDE or adaptive loss.
- Reason: compare head optimization from matching physical initial values,
  rather than sigmoid .5 versus exp_square1. Frozen own-neural warm baseline
  increases effective representation versus the old model; the nonnegative
  control uses the same baseline/trainable architecture to expose this confound.
- Alternatives rejected: zero latent exponential initialization, unbounded
  exp(-raw), relabeling rejected33 as an accepted parent, relaxing gates, or
  treating fixed-H success as full fracture convergence. Relation: same active
  task; bounded fixed-H training first, certified coupled transfer afterward.

## 2026-10-11 — Certify direct bounded heads before short coupled comparison
- Evidence: on identical rejected33 own H/initial damage, direct sigmoid and
  exp_square both meet the original1e-3 residual/range gate in128 L32 updates,
  at .0009571631/.0009280543. Matching nonnegative control exhausts4096 updates
  and ends .0012092756. Operator-aware heads remain under evaluation.
- Decision: after CPU replay, compare the two certified direct bounded heads
  in separate short coupled branches at loads33/34. Inherit physical q/d/H
  only from accepted32, recompute own FEM H, and warm-start only certified new
  decoder parameters. Rejected33 is a neural diagnostic, never a physical parent.
- Reason: fixed-H feasibility is encouraging but cannot select a generally
  better head or certify changing-H coupling. Both128-step results are stopped
  at first passage, not equally optimized minima; the small final-residual
  difference alone does not establish exp_square superiority.
- Alternatives rejected: claiming universal optimality, accepting raw rejected
  fields/history, bypassing inner/outer gates, or promoting fixed-H success to
  full fracture convergence. Relation: authorized output-head experiment,
  original committed history and every physical gate retained.

## 2026-10-11 — Correct the extra history-replay precision check before loading
- Evidence: initial coupled startup stops before any neural optimization/damage
  update or physical commitment. Repeated same-parent FEM solve differs at two
  of71307 H entries by .1633/.1766e-6, missing the added per-entry1e-8/1e-7 check.
  Exact source hashes/accepted ancestry are unchanged; no physical gate failed.
- Decision: preserve the startup root/error/code snapshot. In a fresh v2 root,
  use an explicit initial-H infinity difference normalized by max(H), limit1e-6,
  and record both absolute/relative errors. Always use newly computed own FEM H;
  original1e-3 AT2 and all mechanics/outer/physical gates remain unchanged.
- Reason: a per-entry relative check around small H is not a consistent measure
  of repeated tolerance-limited FEM-driver precision. The extra replay check
  assesses initialization parity; it is not an AT2 acceptance or history rule.
- Alternatives rejected: importing diagnostic H to force equality, silently
  continuing the failed startup root, changing physical tolerances, or calling
  a pre-optimization startup failure a bounded-head convergence failure.
- Relation: same authorized short comparison; initial failed root is retained
  without any accepted physical state, v2 is a new declared numerical identity.

## 2026-10-11 — Prefer direct bounded outputs after successful short coupling
- Evidence: sigmoid and exp_square both accept loads33/34 from the same own-
  neural accepted32 parent, with all original gates and committed history.
  Sigmoid takes16/17 sweeps, exp_square17/17; each trains288 new L-BFGS updates.
  Current AT2 at34 is .0008187313/.0008657689. Fixed-H diagonal reconstructions
  did not meet1e-3, while both direct bounded heads passed in128 updates.
- Decision: use direct sigmoid as the provisional main continuation candidate
  and retain exp_square as a successful comparator. Deprioritize diagonal
  reconstruction under the tested baseline/normalization/optimizer controls.
- Reason: both heads are feasible and similar at first-passage/few-load accuracy;
  their residual/iteration differences do not establish a numerical winner.
  Sigmoid is monotone and has no finite latent stationary full-damage point,
  whereas exp_square has zero derivative at z=0. This is a mathematical
  selection rationale, not demonstrated superiority at critical fracture.
- Alternatives rejected: claiming exp_square is globally optimal from its
  slightly lower fixed-H residual, declaring diagonal lifting always ineffective,
  claiming bounds enforce irreversibility, or treating34 subcritical loads as a
  completed fracture path. Relation: output-head trial succeeds at its short
  scope; final independent CPU field/network/history/force replay is still
  pending before close-out, critical/full loading remains the broader task.

## 2026-10-11 — Close the bounded-head trial with independent replay
- Evidence: all four new coupled states replay original current/inner/raw-current
  AT2, physical/history gates, saved networks and forces on CPU. Independent
  NumPy reconstructs known-H graph features, Swish MLPs, compact readouts and
  both output maps from the saved weights. H equals max(parent H, own FEM
  driver) exactly; parent/source hashes remain unchanged. New-point matched
  direct reaction discrepancies .02085-.02647%, d inf<=.00104024.
- Decision: record the short comparison as complete and retain the broader
  task active. All workers exit/GPU idle; provisional full-path parent is new
  sigmoid accepted34. Its frozen own-neural baseline, basis, scales and decoder
  must accompany the learned residual weights in any declared continuation.
- Verification route: the initial large JAX CPU replay was explicitly
  interrupted during compilation (exit130). Replace this verification path
  with independent NumPy math; that completed replay passes. Do not count
  the interrupted attempt as a successful check. Training/physics are unchanged.
- Alternatives rejected: claiming full fracture success from34 subcritical
  loads, treating the new residual weights as the old decoder, silently reusing
  a startup-failed root, or describing the whole new architecture as a mere
  sigmoid substitution. Relation: same authorized head trial and active task;
  exact accepted34 ancestry/decoder/resume point is recorded at task-end.

## 2026-10-11 — Continue the certified sigmoid branch through later loading
- Decision: user authorizes continued loading. Prepare a separate sigmoid
  continuation from accepted physical34 to uniform35..75/.0075. Retain own
  q/d/H, frozen rejected33 neural predictor, continuous compact basis, fixed
  head parameter scales and learned residual-network weights as one decoder.
- Provenance: verify old accepted1..32, new sigmoid33/34, fixed-H and coupled
  audits, source/code/basis/parameter/field/network hashes, and parent gates.
  Preserve all old roots; rejected states are diagnostic only. Restart accepts
  only certified continuation states and never skips an unready load.
- Performance choice: compile parameter-space value/gradient and metrics once
  per objective for the fixed network layout. H/load remain explicit dynamic
  inputs. SciPy L-BFGS objectives, maxiter2048/check32 and all optimizer/physical
  controls stay unchanged; check numerical parity and changing-H behavior first.
- Reason: the existing short-control runner lacks continuation/restart, and
  rebuilding JIT closures at every inner call spends seconds on compilation
  even when no new training is needed. Function reuse improves execution cost,
  not the mathematical solver or acceptance criterion.
- Alternatives rejected: resetting history, omitting the frozen baseline/head
  scales, replacing damage with a nodal correction, importing direct states,
  relaxing gates or editing previously certified numerical source files.
- Relation: same active fracture task; full/critical convergence remains a test
  to perform, not an inferred consequence of the short bounded-head success.

## 2026-10-12 — Bound the frozen initializer in a declared continuation fork
- Evidence: original sigmoid continuation accepts35..52 with unchanged gates;
  at53 sweep78 the frozen positive-only predictor reaches1.0014326143. The
  initializer guard stops before that sweep's neural solve/physical update.
  This is an initializer-domain failure, not a certified damage-PDE rejection.
  Last committed physical parent is52; retain the uncommitted53 diagnostics.
- Decision: independently audit35..52, then create a separate branch from
  accepted52 q/d/H and its trained residual-network weights. Bound only the
  frozen initializer predictor by min(prediction,1-epsilon), using the existing
  epsilon1e-10. Preserve sigmoid unknown damage, original AT2 residual,
  committed maximum history, omega.5, loading and every acceptance threshold.
- Derivation: sigma0=epsilon+(1-2epsilon)*bounded_prediction lies strictly
  between0 and1; with H>=0, gc0>0 and C=gc1 D_ii>0,
  d0=(2H+C sigma0)/(gc0+2H+C) lies strictly between0 and1. Its logit is finite.
  Below the cap the old and new images are identical; require both saved52
  H_used/current-H image reconstructions and own-FEM history replay at startup.
- Provenance: archive original continuation source/hash and completed audit
  before changing the continuation orchestration. Inject explicit parent/setup
  functions into that runner; keep previously certified head factories unchanged.
  Record the initializer operation and wrapper code in the new run identity.
- Alternatives rejected: clipping the solved physical damage, silently editing
  the old root, importing uncommitted53 history/damage, resetting history,
  direct damage labels/correction, or relaxing residual/outer gates.
- Relation: same authorized fracture task. The cap repairs a finite-logit
  initialization domain; critical-path/full neural convergence remains to test.

## 2026-10-12 — Initializer-cap continuation rejects; isolate moving-anchor defect
- Results: archived original continuation35..52 independently replays; maximum
  new reaction difference .398571% and damage inf .00866445, both at52.
  Safe fork reconstructs both parent52 images and own H; actual GPU cached
  gradient parity2.04e-15/3.01e-15 passes. It completes77 uncommitted53
  sweeps, then inner78 exhausts L32/L16 budgets2048 each, final original
  residual .459156/.528856. No53 commitment and no54 attempt; GPU worker exits.
- Independent NumPy replay validates rejected fields/network/own H/force and
  unchanged source ancestry. Saved q/d/H are exactly the last completed77
  trial checkpoint: rejected inner78 performs no physical update. Physical
  parent remains52. Range success is not equation convergence.
- Diagnosis: predictor maximum .9961106->1.0012968 caps only one node.
  Initializer logit changes by.394113 while learned increment changes by
  .024211; warm image changes by.0840036. At the same new H, unchanged old
  image has AT2 .0103891, changing features alone .0110649, changing only the
  initialization anchor .4884898, both .4912349. These are diagnostic algebra
  interventions, not accepted solutions or newly adopted physical models.
- Decision: record this repair as a failed loading experiment and retain52
  as the exact physical resume point. Do not repeat the rejected branch with
  more iterations or weaken gates. Next candidate is a fixed own-neural latent
  reference anchor, keeping current H/load explicit in the learned increment;
  first validate/train at the failed own-H checkpoint, then declare any new
  continuation from accepted52 with a certified matching decoder.
- Limits: evidence isolates this warm-start discontinuity on this checkpoint;
  it does not establish that a fixed anchor solves the full critical path,
  that sigmoid itself fails universally, or that maximum history is at fault.
- Relation: same active task and authorized research continuation. Preserve
  old/safe roots, audits/scripts/diagnostic and no direct damage/history labels.


## 2026-10-12 — Clarify the persistent reference output offset
- Decision: explain "initializer base" as the persistent reference logit z0,
  rather than a one-time parameter initialization. Current sigmoid output is
  d_theta=sigmoid(z0(x,H,load)+Delta_theta(x,H,load)). Reference damage is
  reconstructed from the frozen own-neural predictor/current H/operator
  diagonal, then mapped to z0 by logit; it remains in every evaluation.
- Reason: user asks for clarification. Frozen predictor weights do not freeze
  its changing-H/load output. Updating H changes both z0 and learned features;
  the prior checkpoint decomposition isolates their effects.
- Alternative wording rejected: implying this is a FEM/nonlocal basis, merely
  initial weights, or that fixing the reference freezes physical history.
  A fixed-reference candidate would hold z0 while current H remains in the
  neural inputs and original damage equation; no such candidate is adopted
  or implemented in this explanatory turn.
- Relation: same active task, accepted physical52/rejected53 retained; no
  training launch or numerical-policy change. Persistent-reference wording
  replaces the ambiguous initializer-base shorthand in user explanation.


## 2026-10-12 — Optional reference head and staggered history differentiation
- Clarification: z0 is a chosen output parameterization, not a physical
  requirement. Direct d_theta=sigmoid(f_theta(x,H,load)) is valid for the same
  bounded unknown and original AT2 equation. The reference form supplied a
  spatial warm prediction and matching initial physical fields in output-head
  controls. No claim that it is necessary or generally superior.
- Candidate scope: include no-z0 direct sigmoid as a fixed-own-H comparison
  alongside the proposed fixed-reference head. Simply removing z0 from the
  current trained decoder changes its field; new initialization/training and
  certification are needed before a matching physical52-parent continuation.
  No candidate implementation or new physical run is adopted this turn.
- Current differentiation contract: outer mechanics updates H=max(previous
  committed load H, own current FEM driver); each damage inner solve fixes H
  and differentiates only theta_d. The predecessor committed H is input data,
  not a trainable state or an unrolled previous-network trajectory.
- Code evidence: LocalMechanics.driver uses host NumPy; output_damage_continue
  calls it outside the damage loss. make_cached_lbfgs.value_grad differentiates
  only parameters with H/load passed separately. history_feature_input already
  uses jax.lax.stop_gradient(H). Thus current gradients do not traverse earlier
  physical steps even without an explicit stop_gradient on every saved array.
- Semantics: explicit detaching of predecessor H/current H can document the
  staggered contract, but stop_gradient preserves forward values. Values may
  still update between outer iterations. It does not freeze z0 or repair its
  observed change. JAX primary API reference:
  https://docs.jax.dev/en/latest/_autosummary/jax.lax.stop_gradient.html
- Alternatives rejected: treating reference removal as an exact unchanged-
  weights transfer, assuming stop_gradient freezes forward history, or
  prescribing this detach rule for a future fully differentiable/inverse
  trajectory without defining that different objective.
- Relation: same active task/accepted52 parent, unchanged physics/gates/source
  and no training worker. The existing exact fixed-own-H diagnostic resume
  remains available; plain sigmoid is an additional proposed control.

## 2026-10-12 — Authorized no-reference sigmoid experiment with parameter warm start
- Decision: user explicitly favors testing no z0 because weight warm start is
  already available. Implement direct d=sigmoid(f_theta(x,H,load)) using the
  existing conditioned MLP/continuous compact readouts and parameter scales.
  No frozen predictor or diagonal/H anchor enters the new damage evaluation.
  Keep current H detached for damage training and committed own-load history,
  original AT2/local FEM, all optimizer and physical gates unchanged.
- Transfer: current parameters were trained as a latent correction, so deleting
  their old reference does not preserve damage. Retain trunk/readout weights,
  shift the existing output bias once to match mean own-teacher logit, then
  permit an explicit own-network logit-fit transfer phase. This is parameter
  conversion from prior own-neural outputs, not FEM/direct labels, a persistent
  reference field or per-node damage correction. The new model has the same
  trainable parameter layout/count; no extra fixed-output network remains.
- Validation: first use completed uncommitted53 sweep77 neural parameters and
  own H as diagnostic-only inputs; independently reconstruct the old image and
  test full original residual/gradient of the new decoder. Transfer at its old
  fixed H, then L32/L16 at failed H78 with unchanged2048/check32 controls.
  Final readiness is original AT2<=1e-3/range, not a transfer-fitting metric.
- If ready, separately transfer accepted52 parameters at its own current H,
  verify original source/accepted ancestry/physical fields and certify the new
  decoder before a declared original53..75 continuation from physical52.
  Do not import rejected53 trial q/d/H as a physical parent.
- Alternatives rejected: silently treating residual-head weights as an already
  equivalent direct sigmoid network, claiming removing z0 alone guarantees
  fracture convergence, changing maximum history or tolerances, or using
  direct labels/nodal correction. Old reference/cap roots remain preserved.
- Relation: same active fracture task; explicit no-reference comparison
  supersedes the unimplemented fixed-reference candidate as the next trial.

## 2026-10-12 — Preserve failed inherited scaling and test latent-basis normalization
- Evidence: first no-reference fixed-H trial independently replays as unready.
  Own-logit transfer2048 updates leaves maximum damage.00806 versus teacher
  .82519, inf mismatch.82420; physical L32 stops after2 updates at all-one
  saturation, L16 cannot move (original residual1). No physical state promoted.
  Inherited reference-head column scales span.766 to2.93e8. This trial did not
  produce an adequate warm-start conversion; it is not a test of a certified
  no-reference coupled path. Preserve its root/source snapshot/audit.
- Decision: new explicit v2 parameterization normalizes latent compact columns
  by their RMS: s_j=1/sqrt(sum_i B_ij^2/N). Validate finite nonzero columns.
  Convert beta_new=beta_old*s_old/s_new so B(s_new beta_new) equals the prior
  latent field. Keep the old trunk gain/weights and the same learned-bias
  recentering. This is checked parameter reparameterization, not a reference
  output, new basis, per-node damage solve or changed AT2 equation.
- Reason: the head-specific normalization previously compensated extremely
  small reference-head output tangents; inheriting it blindly makes the new
  plain-logit fitting badly scaled. RMS columns suit the explicit logit-fit
  conversion and stay fixed across loading. Original budgets/gates unchanged.
- Alternatives rejected: claiming the first failure establishes no-z0
  infeasibility, editing its root, accepting all-one saturation or changing
  residual limits. Relation: same authorized trial; v2 must verify initial
  field/parameter parity before training and independent output replay after.

## 2026-10-12 — User switches to fresh network training beginning at.005
- Decision: user steers away from old-head parameter transfer and requests a
  quick applicability test starting at displacement.005. Interrupt the owned
  latent-RMS transfer worker; retain first failed root and interrupted v2 log/
  snapshots. No old trained weights or physical q/d/H enter the new trial.
- New branch: generate zero physical state, impose.005 as the first selected
  load and train a freshly initialized direct sigmoid network at own FEM H.
  Keep original local mechanics/nonlocal AT2, committed previous-load maximum
  history, omega.5 and original physical/optimizer gates. First verify fixed
  own-H inner training, then couple this same load if ready. Endpoint is this
  single load for the requested quick test; later loading requires readiness.
- Representation: retain the pinned continuous compact basis geometry, but
  initialize all trainable MLP/readout weights anew. Use explicit latent RMS
  column scaling, one-time trainable bias for initial damage.01 and no frozen
  reference/teacher/diagonal-H output term. Fixed gain sets unit MLP latent
  factor at the first load; load-dependent scaling remains declared.
- Interpretation: jumping from zero to.005 is a new loading schedule, not
  continuation of the prior accepted52 ancestry. Compare direct fields only
  after training, and report this schedule difference. No direct labels,
  nodal correction, imported history, gate relaxation or Git action.
- Relation: same active task; the user's latest scope supersedes transfer
  trial and accepted52 continuation as the immediate next computation.

## 2026-10-12 — Cold.005 RMS control rejects; compare initial residual-Jacobian scales
- Evidence: fresh no-reference own-zero.005 first inner runs2048 L32 plus2048
  L16 and rejects at AT2.11145845. No damage relaxation/physical advancement;
  fixed-H/initial network/own H/force and rejection independently replay.
  This is an incomplete inner solve, not a coupled-path convergence result.
- Decision: keep that root and compare a separate fresh-seed trial using
  initial AT2 readout-column Jacobian normalization. At uniform d_init=.01,
  J_j=[diag(gc0+2H)+gc1 D]diag(d_init(1-d_init))B_j; use existing checked
  residual_column_scale to make RMS(J_j/(2 RMS(H)))=1. Scales are computed
  once at own initial H and stay fixed. Trunk latent factor remains one at.005.
- Reason: latent RMS normalization ignores stiff graph-operator column scales;
  residual Jacobian is the existing verified parameter-normalization principle
  applied to this new uniform sigmoid initialization. No new output field,
  teacher, nodal solve, representation column, physics or gate is introduced.
- Controlled comparison: identical fresh seed/bias, initial physical q/d/H,
  basis and initial neural damage.01. Both readouts start zero, so scale
  changes preserve the initial prediction exactly. Budgets2048/check32 and
  original residual/range/outer gates remain. Require independent Jacobian
  algebra test and host network/history/force replay before interpretation.
- Alternatives rejected: promoting the rejected.005 fields, resetting a failed
  physical state as accepted, or claiming no-reference infeasibility from one
  scaled optimizer. Relation: same user-authorized fresh.005 applicability test.

## 2026-10-12 — Close the fresh.005 quick comparison without physical promotion
- Results: same fresh42577-parameter/no-reference sigmoid at own initial FEM H,
  2048 L32 plus2048 L16 each. Latent-RMS final AT2.1114584507; initial AT2
  Jacobian normalization .01611090435 (~6.92x lower). Both exceed original
  .001, so neither enters outer relaxation or promotes physical.005.
  Saved physical d/H_parent remain zero; current physical H is own initial
  undegraded FEM driver. No.0051 attempted, all workers exited/GPU idle.
- Independent NumPy replay: initial uniform.01, fixed/final saved networks,
  original residuals, scaling algebra, own-zero history, force/reaction,
  source/code and rejection transactions pass for both completed cold roots.
  Source-stable first inherited-scale transfer rejects and its audit remains;
  RMS-transfer root was deliberately interrupted130 at user scope change.
- Evaluation-only oracle: same H volume-symmetric CG168 iterations, symmetry
  defect7.76e-17, original residual inf3.7973e-10 and direct d_max.25917.
  RMS/Jacobian neural d inf errors .149978/.0237642, relative L2 errors
  .820263/.442697. Exact H equality permits reuse of one oracle for both;
  it is produced after training and never supplies training labels/history.
- Decision: conclude the requested quick experiment as unready, keep broad
  task active. Removing z0 eliminates the previously diagnosed moving-reference
  mechanism, but these cold inner solves have not achieved original accuracy.
  Favor the fixed initial AT2 scaling for a next optimizer/representation study;
  do not restore z0 merely because this limited-budget training is incomplete.
- Limits: this is first-load fixed-H accuracy, not a converged coupled.005
  state, full loading path, spatially converged solution or speed comparison.
  Do not treat residual reduction/visual field agreement as readiness.
- Relation: same active task; resume only same-decoder neural training at this
  own H in a new declared branch, or a declared fresh optimizer control.
  Rejected physical fields may not be used to advance loading.


## 2026-10-12 — Recommend a short gradual own-zero prefix before full no-z0 rerun
- User asks whether directly solving again from zero is necessary. Recommend
  an initially short gradually loaded own-zero experiment with the new direct
  sigmoid head and weight warm start, rather than committing immediately to a
  full75-state rerun. The completed.005 cold tests already initialized physical
  q/d/H=0; they jumped straight to.005 with untrained parameters.
- Rationale: smaller positive loads may supply accepted new-head parameters
  for successive loads and distinguish cold-start difficulty from persistent
  inner optimizer/representation failure. This is a hypothesis, not guaranteed
  convergence. A gradual path also changes own damage/history versus the jump;
  compare physical ancestry explicitly rather than calling both paths identical.
- Suggested scope only: exact zero-load physical initialization, then original
  positive uniform loads1..10/.0001.. .001 with the same physical/range/inner/
  outer gates. Extend toward.005 only if accepted. No training at zero load
  with the current positive-load relative-residual/load-normalization formulas.
  Adapt initial sigmoid bias and parameter scaling for the small-load regime;
  do not blindly inherit the unit-at.005 raw-logit/load-squared initialization.
- Alternatives rejected: treating the.005 jump rejection as proof gradual
  loading cannot work, starting from old decoder's bare weights, retraining
  every load from scratch, accepting failed predecessors, or automatically
  launching a full rerun on an advice-only question.
- Relation: same active task; numerical model/history remains unchanged.
  No new configuration, implementation, worker or physical promotion. Existing
  exact same-H diagnostic resume remains recorded; this is a proposed next
  sequential-path control alongside the fixed-H optimizer study.

## 2026-10-12 — Authorized fresh gradual no-reference path
- User authorizes restarting with the new output layer and fresh weights.
  Preserve all prior roots/accepted52 ancestry; this path imports neither old
  parameters nor physical fields. Initialize q/d/H exactly zero and train
  only at positive uniform loads, starting.0001. Warm start from accepted
  new-head states only; reject immediately on unchanged inner/outer gates.
- Declare a separate gradual decoder: sigmoid of trainable MLP plus compact
  readouts, without multiplying the entire learned logit by load squared.
  That former scaling would multiply a negative bias by100 between.0001
  and.001 and artificially saturate sigmoid. Preserve the old decoder factory
  and archived single-load results. Use physical, detached H features rather
  than load-normalized H; no reference logit, baseline or new damage equation.
- Explicit initialization: uniform neural damage1e-5 (physical damage0), unit
  trunk gain and fixed initial-own-H AT2 readout-Jacobian normalization.
  Initial bias is trainable; column scales are fixed parameter coordinates.
  No direct labels, nodal correction, gate relaxation or history import.
- Run a short1..10 prefix first; continue the authorized full uniform path
  only after it passes. Save identities, code/source pins, zero state and
  per-load networks/fields with parent hashes. A rejected state cannot advance.
- Relation: same active task; replacing advice-only scope with actual execution.

## 2026-10-12 — Short-budget gradual first load rejects; test larger budget
- Fresh gradual load1/.0001 exhausts2048 L32+2048 L16, AT2.01136134216
  against.001. No physical damage relaxation or second load. Independent
  NumPy network/scales/own-H/force/source/transaction replay passes; all
  physical zero-parent fields remain unchanged. Same-H post-training CG168
  gives original residual inf1.95399e-13, d_max.0001298375; NN inf error
  9.48226e-6, relative L2.263690. This oracle never enters training.
- Decision: user-authorized new-head rerun includes a separate longer-budget
  control,8192 updates per L32/L16 stage, with exactly the same fresh seed,
  initial physical/NN fields, operator, fixed parameter scales and original
  gates. Late residual is still decreasing;4096 updates alone cannot
  distinguish insufficient training from representation limits.
- Preserve the completed short root. New config/output root only; no changes
  to pinned decoder/runner, no failed physical state advancement, no direct
  targets or extra baseline. The first10 milestone and eventual full path
  remain conditional on accepted-only advancement. If the longer first
  inner rejects, retain that diagnostic and report the unresolved blocker.
- Relation: same authorized fresh gradual path; optimizer budget control,
  not new physics or relaxation of acceptance.

## 2026-10-12 — Qualify reproducibility of the longer-budget comparison
- Initial_network.eqx and zero_fields.npz hashes match exactly between short
  and long gradual controls. Configs differ only in output root and maxiter.
  Independent GPU CG setup is tolerance-equivalent, not bit-identical:
  q inf difference1.06488e-13; H inf7.63740e-11 (relative2.18731e-9);
  derived initial column scales differ relatively at most4.46933e-10.
- Decision: describe matching prescribed physical initial conditions and
  solver tolerance, not exact identical H or a bitwise optimizer trajectory.
  Keep original equations/CG thresholds; do not overwrite own initial fields
  with another run to hide roundoff. Do not reuse the short same-H oracle
  unless exact H equality is established. Late L-BFGS trajectories can differ.
- Relation: precision qualification of the budget comparison, not a new
  physical policy or permission to advance rejected states.

## 2026-10-12 — User steers to saturation-controlled output comparison
- Longer fresh gradual first load rejects at AT2.0026763814097 after8192
  L32+8192 L16. No coupled update/load2. Independent NumPy replay passes;
  evaluation-only own-H CG gives neural relative L2.0583957. Budget helps
  accuracy but still does not meet original.001. Both workers completed.
- User identifies sigmoid saturation and proposes exp(-z^2) or removing the
  output constraint. Adopt a same-fixed-own-H, same-capacity comparison of
  sigmoid, exponential-square and identity outputs, freshly initialized.
  Use the already audited first-load own-zero H exactly, not a direct target,
  old trained weights or rejected physical damage. No physical promotion.
- Independently checked derivatives: sigmoid'=d(1-d); exp-square'=-2zd;
  identity'=1. Use exp-square initial z=-sqrt(-log(d_init)) so derivative is
  positive. All initial fields equal1e-5; fix initial readout Jacobian scales
  separately and choose trunk_gain=sigmoid_tangent/native_tangent, matching
  initial d-sensitivity of every trainable coordinate. Bias stays trainable.
- This removes arbitrary initial gradient-size differences. Neither bounded
  head eliminates small-damage saturation; measure final effective derivative
  distributions and residual/range readiness before attributing failure.
  Identity may leave [0,1] during training, but retains original range gate;
  never clip a field or accept invalid damage to claim convergence.
- Keep old factories/runner snapshots immutable. New diagnostic factory/config/
  driver, same residual objectives/2048-stage budget and original.001/range
  gates. A fixed-H pass is not an accepted coupled loading state.
- Relation: latest user steering extends the same active fresh-head task;
  physical model/max-committed history unchanged.

## 2026-10-12 — Close exact-own-H head comparison without promotion
- Matching initial field/parameter gradients, same42577 parameters and4096
  updates: sigmoid AT2.01134167, exp-square.01070668, identity.00243834;
  original limit.001. Identity minimum-1.77518e-6 also violates range1e-10.
  Same-H post-training relative L2.260735/.263139/.0581968 respectively.
  Neither any head nor either gradual sigmoid budget reaches readiness.
- Independent NumPy maps/MLP/basis/residual/range/own-H/source/transactions and
  initial Jacobian/L32-gradient algebra pass. Initial3 GPU hidden networks
  match exactly; CPU seeded reconstruction is checked at roundoff tolerance.
  Effective minimum MLP derivatives bounded heads~1.9e-7/1.6e-7 versus
  constant9.9999e-6 identity. Scientific plots inspected; all workers exited.
- Decision: retain identity as the promising optimization diagnostic, but do
  not promote its negative or unready damage. Exp-square does not substantially
  improve this matched control. Evidence supports output conditioning as a
  contributor, not proof sigmoid saturation is the unique/main cause.
- Next candidate: separately declared same-H linear-output study with range
  enforcement in optimization/penalty and improved conditioning; no algorithm,
  penalty weight, new run or physical policy adopted at this session end.
  A fixed-H pass must precede a matching-head own-zero coupled experiment.
- Relation: same active broad task remains incomplete; user-authorized fresh
  rerun and saturation experiments are concluded honestly at their gates.
  Preserve all roots/old52 ancestry; no Git history operation.

## 2026-10-12 — Continue linear-head range optimization and distinguish residual from damage activity
- User authorizes continuation and cautions that developing damage may raise
  residual, so a smaller residual must not be equated with a better fracture
  field. Keep physical activity, absolute equation imbalance and normalization
  separate, especially across changing own H/load states.
- Independently checked current equation: r=(gc0+2H)d+gc1 Dd-2H. At d=0,
  r=-2H; with positive H, original part-normalized residual is exactly1,
  not0. Only d=H=0 has zero absolute AT2 residual; relative0/0 is undefined.
  Audit zero/all-one/saved heads/direct evaluation at identical H and report
  part norms, forcing-normalized residual, volume-mean/L2 damage and activity.
- New diagnostic: warm-start the same rejected identity-head network at its
  exact own H. Compare additional2048 L16 residual-only updates with a
  separately declared inequality augmented-Lagrangian control for0<=d<=1.
  Native linear output/42577 parameters/history/PDE/gates remain unchanged.
  No network-output projection, direct labels or nodal damage correction.
- Define S=2 RMS(H)/gc0, g_lo=-d/S, g_hi=(d-1)/S. Loss is original L16
  plus mean(sum([max(lambda+mu*g,0)^2-lambda^2]/(2mu))) over both bounds;
  after each inner solve lambda<-max(lambda+mu*g,0). Duals/constants are
  detached fixed coefficients during training. Fixed declared penalties
  [1,10,100,1000],2048 inner iterations each, initial duals0. Independently
  check gradient algebra, active/inactive bounds, dual update and H detachment.
  This is an optimizer trial, not a modified physical AT2 energy/history law.
- Readiness requires original AT2<=.001 and range1e-10; also report absolute
  and forcing-relative residuals. Growing damage/transient residual is not
  itself failure, but original final physical/equilibrium gates stay intact.
  Only after a fixed-H pass may a matching own-zero coupled branch begin.
- Relation: same active task. Preserve certified sources and completed roots;
  no Git history operation. Penalty values are explicit experimental controls,
  not claimed physically universal constants or a guaranteed solver.

## 2026-10-12 — Range AL reaches equation gate; continue the remaining range violation
- Residual-only additional2048 updates end at AT2.00155947, minimum-7.70e-7.
  Range AL four2048-stage rounds improves AT2.0008753814090, passing original
  .001, while minimum-2.34196e-8 still fails range1e-10. Damage max.000129857
  and volume mean1.19611e-5 versus evaluation-only oracle.000129838/1.19966e-5.
  Lower residual is not due to disappearing damage. No physical promotion.
- Independent NumPy initial/final net/AT2/activity/source/AL dual-transition
  audit passes. Constant all-one diffusion is0 in pairwise GPU operator;
  assembled CPU D1 cancels within checked8*eps*gc1*||D||inf roundoff bound.
  This is an audit comparison tolerance only; no physical gate is changed.
- Decision: continue the same linear network AND updated nonnegative AL duals
  at identical H in a separate pinned extension root, penalties1e4,1e5,1e6,
 2048 iterations each. No repeats of the completed residual-only control.
  Keep original PDE, decoder/scales/capacity, range/AT2 thresholds and physical
  zero ancestry. Do not round/clip small negative damage into acceptance.
- Rationale: range violation is now the specific blocker, with continuing
  progress and a passing equation residual. A continuation of the declared
  optimizer is justified before changing the head/model or abandoning it.
- Relation: same current continuation authorization and damage-activity audit.
  The diagnostic remains fixed-H only; any physical loading needs a separate
  original-gate coupled certification after inner readiness.

## 2026-10-12 — Certified linear fixed-H readiness; authorize next own-zero coupled check
- Range continuation reaches original fixed-H gates at second added stage
  mu1e5,96 updates: AT2.0008070040086; minimum-5.20563e-11 is within original
  [-1e-10,1+1e-10]. No clipping. Independent NumPy networks/AT2/activity/
  inherited dual transitions/source replay passes; same-H relative L2.0156231.
  Damage mean1.2017157e-5 versus oracle1.1996579e-5 (~.1715% difference),
  max.000129640 versus.000129838. This is not damage collapse or coupled acceptance.
- Decision: progress to separately identified own-zero local-FEM/linear-neural
  damage loading. Seed only these certified own-first-H neural parameters;
  initialize physical q/d/H exactly0. Same identity decoder/physical H features,
  fixed scales/gain and42577 parameters; no new load-squared output scaling.
  No old-head physical ancestry or trained weights/direct labels are imported.
- Original uniform loads1..75, assess first10 then advance only from accepted
  parents. Original omega.5, inner/current-H/physical/outer consecutive gates.
  Each fixed-H inner uses the same L16 inequality AL with explicit penalties
  [1e3,1e4,1e5,1e6],2048 iterations each; reset duals to0 at each new fixed-H
  problem so no stale H-normalized multiplier is silently inherited. Neural
  parameters warm start. Physics readiness can stop an inner immediately.
- Report current absolute/forcing-relative residual and physical/NN damage
  activity alongside original gates. Allow transient outer residual growth;
  commit only after the existing three consecutive equilibrium confirmations.
- Relation: same user-authorized continuation; a fixed-H success warrants the
  next staged coupled test, not a claim full fracture is already solved.

## 2026-10-12 — Certify linear neural prefix; keep transient activity separate from final balance
- Own-zero linear neural/range-AL loading accepts52 original loads through
  .0052; independent NumPy state/network/driver/force/range/AT2/relaxation/
  ancestry replay passes. No physical equation, history law or original gate
  changed. No old-head weights, direct targets, reference output or nodal
  correction. The new-head first-load failure is resolved for this branch.
- R52=606.55129664 versus matched direct606.59069204 (difference.00649456%);
  maximum prefix reaction discrepancy.04783507%. Peak physical damage.62043
  and volume mean.030629 show nontrivial damage. Absolute physical AT2 grows
  from~1.12e-4 at1 to.51528 at52 while forcing grows to1044.48; original
  part-normalized final residual.00065020 still passes. Thus absolute growth
  across loads is not equated with worsening physical convergence.
- Decision: continue the already authorized original-gate worker toward75;
  critical53/.0053 is still uncommitted. Permit transient residual/damage
  growth during training and outer iterations, but reject final states that
  fail equation/range/mechanical/three-confirmation checks. Do not substitute
  a forcing-normalized diagnostic for the established acceptance metric.
- Rejected interpretation: zero damage is always a zero-residual solution.
  At positive own H full AT2 has source-2H, so d=0 has relative1. Zero gradient
  or initial d=H=0 does not demonstrate loaded fracture equilibrium.
- Relation: implements the user's physical-activity clarification on the same
  active task. Prefix feasibility does not establish full critical fracture,
  continuum accuracy, or PINN speed/superiority. Preserve all earlier branches.

## 2026-10-12 — Continue live critical loading and inspect a frozen diagnostic checkpoint
- User asks to continue. Existing PID16781 is still solving53/.0053; no
  duplicate/restart or numerical-source mutation. Accepted52 remains parent.
- Recent completed outer100: fixed-H NN AT2.00099954 passes, mechanical force
  passes, but physical updated-H AT2.10681, raw current-H NN.48562, dchange
  .01008 and Hchange.05138 fail final gates. Raw updated-H max1.00507 fails
  range, although trained fixed-H max.98795 passes. These are transient outer
  diagnostics, not an accepted or globally rejected load.
- Decision: copy one atomic completed pending generation into a separate
  evaluation-only diagnostic root. Independently replay NumPy NN/own driver/
  force and evaluate the same fixed-H AT2 equations before/after own H update
  with the already validated volume-symmetric CPU CG oracle. Do not feed
  oracle fields to training or promote physical pending states. Compare actual
  field/activity changes as well as residual/component scales.
- Rationale: distinguish inner representability, mechanical solve, physical
  history-driven field change and network sensitivity to updated H without
  perturbing ongoing training. Keep original acceptance and physics untouched;
  an expected transient jump is not by itself proof of a structural defect.
- Relation: same active task and ongoing continuation authorization. Final
  critical/full-path success remains unverified; no Git history operation.

## 2026-10-12 — Frozen critical diagnostic identifies physical evolution and neural H sensitivity
- Completed pending103 independently replays NumPy network, own H, FEM force
  and original AT2. CPU volume-symmetric CG solves identical before/after own H
  to residual~2e-11, symmetry defect<3.1e-17; both oracle fields stay in0..1.
  Diagnostic has no training labels, physical promotion or live-source edits.
- Fixed-H NN .0009984806 passes equation, but updated-H raw NN .4972834 and
  max1.004778 fail. Physical updated-H residual.106820; reaction583.567.
  True corresponding fixed-H fields change by.0100012 inf; NN output changes
  by.0163414 inf. NN errors are.00508725 inf at trained H and.0145481 at new H.
  Thus evolving damage and extra network H sensitivity both contribute.
- Decision: keep running the existing declared worker/gates. This observation
  does not prove ultimate divergence or justify relaxing acceptance. If the
  critical path rejects/stalls, the captured same-own-H checkpoint provides
  a concrete optimization/conditioning diagnostic; final physical coupling,
  range and healing still need certification before another load advances.
- Relation: user's residual/activity distinction is confirmed with independently
  computed same-model fields, not by attributing every residual rise to good
  crack growth. No physics/history law or optimizer policy is changed here.

## 2026-10-12 — User requests current status and notification when the run ends
- User asks for latest results and a report when calculation ends. PID16781
  still runs;52 accepted/audited loads remain. Completed outer118; inner119
  then120 continue neural range checks. No fresh run, model change or restart.
- Decision: keep this turn monitoring the existing worker until termination,
  then independently replay any new accepted/rejected final state and report
  completion versus successful full-path convergence distinctly. No available
  automation_update tool was found; no scheduled reminder is claimed created.
- Relation: notification request concerns the current experiment; broad task
  remains incomplete and all original acceptance thresholds stay intact.

## 2026-10-12 — Loading terminates at123; audit the exact blocker without changing acceptance
- PID16781 exits normally with experiment status inner_rejected at53/.0053,
  inner123 after four2048-update AL stages. Inner AT2.0009762949 passes.001,
  but four raw neural particles are negative, minimum-1.28320568e-6 versus
  allowed-1e-10. No failed-inner relaxation, physical53 promotion or54 attempt.
- Independent NumPy replay passes all accepted52 and rejected53, confirms own
  history/current fields/NN/source/parent hashes. Last completed outer122 remains
  physically unready: AT2.0916949, dchange.00877848, healing.00756736. The last
  outer report describes then-current NN; final inner123 has updated weights,
  raw fixed/current H are identical with.0009762949, but its range still fails.
- Decision: report experiment termination distinctly from full-path success.
  Preserve accepted52 as physical parent and all rejected fields as diagnostics.
  No automatic rerun or physical gate relaxation in this notification turn.
- Small evaluation-only probe is justified: the native linear head's trainable
  scalar output bias adds constant delta to d when bias increases delta/gain.
  Hence r(d+delta)=r(d)+delta*(gc0+2H+gc1 D1); independently recompute full
  metrics and ranges after the smallest positive range-restoring shift. This
  is a read-only parameter-feasibility diagnostic, not output clipping or an
  adopted optimizer, serialized new network, physical update or acceptance.
- Relation: user's request to be told at termination is now actionable; broad
  fracture task remains incomplete and physical damage/residual are separate.

## 2026-10-12 — Read-only scalar neural-bias feasibility probe succeeds
- At rejected inner123, delta=1.2833056783e-6 (negative minimum restored
  with1e-10 positive margin) corresponds to output-bias increase.12833185115
  because gain9.9999e-6. Full CPU residual recomputation gives AT2.00097622561,
  minimum1e-10/maximum.99602253124, both original fixed-H gates pass.
  Independently checked r(d+delta) algebra including D1 roundoff. No clipping.
- Decision: retain parameter-bias feasibility restoration as a concrete next
  optimizer candidate. No weights were changed/serialized, no solver fork
  launched, no failed outer state promoted and no threshold relaxed in this
  notification session. Artifact end_range_bias_probe.json is evaluation only.
- Rationale: the immediate four-node range blocker has a feasible point in
  the existing neural architecture. It need not require another bounded head
  or PDE change. This does not solve the remaining critical coupled equilibrium,
  own-H response or healing issue, nor prove a general scalar shift always works.
- Relation: report current run as stopped/unready, despite a promising local
  optimization repair. Later implementation requires a separate declared
  matching-head continuation with accepted52 as physical parent/provenance.

## 2026-10-12 — User accepts microunit damage-range error; continue same critical load
- User explicitly accepts negative damage on1e-6 scale. Decision: new identified
  continuation uses symmetric damage range atol2e-6 (covers observed-1.283e-6),
  reported explicitly as a numerical range tolerance. Preserve all original
  AT2/mechanics/change/history/healing/consecutive thresholds. No clipping,
  bias restoration, decoder change, labels or physical formulation change.
- Apply tolerance consistently to inner image, relaxed physical field, raw
  updated-H NN and host mechanics validation. Existing LocalMechanics hard
  checks1e-10; preserve certified numerical source and use a new explicit
  subclass overriding only damage range validation with identical quadrature.
- New declared branch resumes uncommitted same-load53 at completed outer122,
  with failed-inner123 neural parameters at the identical own H. Certified
  accepted52 remains physical parent. This is same-load solver recovery, not
  acceptance of53 or advancement from an unready physical loading parent.
  Pin identity/final audit/accepted52/pending122/failed123 and verify own-H,
  parent/BC/force, and equality of pending and rejected physical fields.
- First inner should pass new range/old AT2 with0 updates. Resume outer123;
  after three full confirmations may commit53 and continue54..75. Preserve
  original strict root/status. Record new-range policy and source ancestry;
  old numerical files remain immutable for reproducibility of prior audits.
- Relation: user's current steering supersedes strict1e-10 range on this new
  branch only; it does not relax physical coupled equilibrium or healing.

## 2026-10-12 — Microunit-tolerance same-load recovery passes independent certification
- New linear_damage_continue.py/config resumes pending122 physical fields
  and exact failed123 NN weights. Explicit range2e-6 in inner/raw/physical/FEM
  validation; all other original controls/config entries match strict source.
  New tolerant FEM subclass evaluates signed d with identical quadrature/law.
  No clipping, bias shift, direct labels or changes to preserved numerical files.
- Two focused tests pass: unchanged signed FEM factors + tolerance failures;
  same-load accepted ancestry and rejection of promoted/tampered pending state.
  New modules compile, configs parse and exact non-range config parity passes.
- Owned GPU PID30710 starts. Restored inner123 passes with0 training updates,
  AT2.0009762949/min-1.28320568e-6. Physical relaxed minimum-4.40144e-7 is
  actually retained and independently signed-quadrature assembled force passes.
  Physical current AT2.0854548, reaction570.5391: coupled step still unready.
- Independent NumPy own-H/initial NN/relaxation/source/force/current-residual
  replay passes; recovery_audit.json certifies initial recovery/outer123 only,
  not full-load53. First outer snapshot is preserved independently of rolling
  two-generation scratch. Existing accepted52 remains physical parent.
- Relation: user's range-tolerance preference is implemented consistently.
  Continue original-gate worker toward75; do not equate tolerance change with
  coupled convergence or relabel the earlier strict stopped experiment.

## 2026-10-12 — Tolerant branch clears range stop, then exhausts inner accuracy budget
- PID30710 exits normally after263.83s, status inner_rejected at53/inner124.
  Completed one new outer123; no accepted53 or54. Final damage min+3.3303e-7
  and max.996093 pass new range2e-6. AT2.00101802972 is1.803% above unchanged
  .001 gate after four2048 stages; this stop is equation accuracy, not range.
- Independent final NumPy/signed-FEM/own-history/network/source/transaction
  audit passes. Physical d remains exactly completed123, including actual
  minimum-4.40144e-7, current AT2.0854548 and R570.5391; still coupled-unready.
  Failed inner124 was not relaxed/promoted; accepted parent stays52 unchanged.
- Decision: retain user's2e-6 tolerance consistently for later matching-head
  experiments. Do not revert to1e-10 or keep proposing scalar bias correction
  for this micro-negative field. Next investigate extra fixed-own-H training/
  conditioning at the pinned final neural checkpoint; no new numerical budget,
  solver policy or worker is adopted at this session end. Other gates unchanged.
- Relation: current user steering is implemented and its effect verified;
  immediate range blocker cleared. Numerical budget exhaustion just above a
  threshold is not proof model divergence or a completed fracture solve.

## 2026-10-12 — Show accepted and current unready physical field maps
- User requests current physical-field clouds. Plot actual relaxed physical d,
  material P1 nodal ux/uy and own committed-floor H from accepted52/.0052 and
  latest completed outer123 at unaccepted53/.0053. Do not substitute the failed
  inner124 neural image or relabel diagnostic53 as a converged physical state.
- Use material FEM connectivity and undeformed coordinates (omit grip-only
  nodes), shared color scales for the pair, common logarithmic history scale
  labeled explicitly, and a separate same-scale damage zoom near the crack.
  Retain actual signed microscopic d values; no clipping, smoothing, source
  modification, new physical calculation or loading experiment.
- Verify source/mesh/state hashes and prior independent audit pass; inspect
  rendered figures. Record reproducible plotting script and scalar ranges.
- Relation: same active task remains incomplete; visualization is presentation
  of verified saved data, not new convergence certification or task switch.

## 2026-10-12 — Saved physical cloud plots inspected without solver changes
- Two verified source fields and same mesh render successfully: actual d peaks
  .6204257264/.9959308981. P1 damage/ux/uy plus logarithmic own H overview and
  crack-tip zoom use shared scales, original material connectivity and clear
  accepted/unready labels. Initial material notch is white; no physical data
  clipping, filters, deformation amplification or final NN-image substitution.
- Rendering inspected, source hashes/status/array ordering/ranges checked.
  Standalone plot_script.py andfields_report.json archived under
  physical_field_maps_20261012; no numerical state/worker/gate change.
- Relation: user's cloud request completed; broad task/inner accuracy and
  critical outer equilibrium blockers/resume point remain unchanged.

## 2026-10-12 — Compare physical clouds with matched final reference before changing convergence
- User asks whether criteria are too strict and requests reference comparison.
  First compare accepted NN52 and latest physical NN53 (outer123) to same-load
  final converged local-P1/nonlocal-AT2 direct controls, verifying exact shared
  geometry/physical/operator/loading definitions and source field receipts.
  Old local-FEM warning archive is a different boundary/discretization branch;
  do not silently use it as the matched solver-accuracy target.
- Report reaction, volume-weighted nodal L2 damage/material displacement,
  infinity error, damage activity and shared-color maps/centreline profiles.
  Supplement with direct scalar outer123 and full transient reaction history
  to distinguish reasonable intermediate progress from final equilibrium.
- Separate finite neural training-budget stop at.00101803 versus.001 from
  current physical coupled AT2.085455/outer changes. No gate/physics/field
  change, new loading or reference labels are adopted by this comparison.
  Plot current physical d/q/H, not failed-inner neural outputs.
- Relation: same active task; evaluate the user's hypothesis with matching
  models and final versus transient results before recommending a remedy.

## 2026-10-12 — Matched reference supports intermediate progress, not critical-load acceptance

- Same mesh/operator hashes and exact physical/geometry/nonlocal/load controls
  verified. Independent signed-quadrature FEM force/reaction, own committed
  H and full current AT2 checks pass for physical NN/direct52 and53. Direct
  final reports pass all gates; source/parent receipts match. Comparison is
  against this discrete model, not the older different-horizon warning archive.
- At.0052, reaction error.006495%, volume-weighted nodal damage relative L2
  1.42828%, material displacement.0321881%. At.0053, NN physical outer123
  reaction570.539 matches direct outer123570.793 within.0445184%, but direct
  final outer1296 reaction1.57856 and a damage band crossing the ligament are
  far from current near-tip state. Final-field damage error78.6663% and
  displacement33.7364%. Per-outer scalar similarity is not full-field parity;
  reference intermediate field arrays were not retained.
- The finite inner124 budget abort at.00101803 versus.001 is a separate
  efficiency obstacle, only1.803% above the gate. Recommend investigating
  explicit continuation/additional budget or staged inner accuracy while
  maintaining final physical equilibrium checks. No new tolerance or solver
  policy adopted in this evaluation session. Wholesale outer-gate relaxation
  rejected: current physical AT2.085455 and reference final difference remain.
- Relation: previous final-versus-transient comparison decision; active task
  and exact failed-inner124 checkpoint retained, no loading or state promotion.

## 2026-10-12 — Authorized additional-budget recovery at unchanged critical-load gates

- User approves continuing after matched-reference assessment. First attempt
  an explicitly declared same-load53 recovery from tolerant failed-inner124
  network and completed physical outer123, with accepted52 still physical parent.
  Increase each existing L-BFGS AL stage maxiter2048 ->8192 (at most32768 per
  inner call, early exit on original residual/range gates). This is an empirical
  optimization-budget experiment, not a change to physical accuracy or model.
- Keep four penalties1e3/1e4/1e5/1e6, residual.001, symmetric range2e-6,
  omega.5, all history/mechanics/coupling/healing/consecutive acceptance and
  network parameterization unchanged. Do not use direct labels or reference
  fields. Preserve both previous stopped roots and their protected code.
- New separate entry/config validates pinned tolerant final identity/audit,
  accepted-parent receipt, pending123 physical fields and failed124 weights;
  verifies exact config parity except declared budget/output/recovery metadata.
  Restart L-BFGS memory and AL duals explicitly from NN parameters, as existing
  recovery does; no saved optimizer state is claimed. Check frozen own H and
  train until unchanged gates before any new physical relaxation.
- If recovered inner124 passes, continue original coupled path53..75; only
  commit a load after original final gates. Record/audit first recovery and
  later accepted/rejected checkpoints. More budget may not suffice; do not
  silently loosen stopping criteria if it fails.
- Relation: previous comparison supports intermediate progress; this isolates
  finite neural training budget before considering staged accuracy policies.

## 2026-10-12 — Preserve interrupted launch and restart under a monitored session

- First detached PID38955 exits after the initial NN check, without traceback,
  final result or a completed physical update. Cause undetermined; cgroup
  oom_kill count0. Do not interpret this as numerical rejection or convergence.
- Preserve all partial files under
  linear_damage_budget_continue_20261012_interrupted_launch and add an explicit
  interruption receipt. Retry identical source/config in a monitored foreground
  tool session17665/PID40800, logging to the separate .monitored.log. No model,
  gate, optimizer or initial neural checkpoint change and no duplicate live job.
- Relation: authorized budget experiment; obtain actual optimizer evidence
  before attributing the launch interruption to the numerical method.

## 2026-10-12 — User requests negative tolerance1e-3 and full loading attempt

- New steering explicitly raises negative damage tolerance to1e-3 and asks to
  attempt the entire loading path. Apply asymmetric bounds d>=-1e-3 and
  d<=1+2e-6 consistently to inner, physical, current-H NN and signed FEM input.
  Upper tolerance is unchanged because only negative tolerance was requested.
  Keep AT2.001, force/history/healing/change/three-pass gates and physics.
- Stop owned monitored budget worker40800 withSIGTERM (session exit143),
  preserving its partial files and progress log; no completed124/checkpointed
  inner update or new accepted state had appeared. Its last scalar inner124
  stage2 at2816 hadAT2.00070518/min-2.32555e-6. This demonstrates residual
  progress but is not a serialized neural seed or final equation acceptance.
- New separate asymmetric branch restores the certified original tolerant
  failed124 NN/completed123 physical state. Keep authorized8192 per-stage budget
  and original four penalties; early inner exit uses explicit asymmetric bounds.
  Reuse pinned original source rather than pretending the interrupted optimizer
  weights were saved. Accepted52 remains physical loading parent until53 passes.
- Add dedicated asymmetric gate/FEM/optimizer code; preserve all older numerical
  sources and artifacts. This tolerance experiment establishes relaxed-range
  discrete feasibility only; independently compare all accepted outputs with
  same-model direct fields and report negative magnitudes/volume fractions.
- Relation: supersedes2e-6 lower bound for the new full-path experiment only;
  never label relaxed-range success as convergence at the old lower tolerance.

## 2026-10-12 — Asymmetric-range recovery passes independent replay and continues coupling

- Dedicated AsymmetricLocalMechanics, inner L-BFGS checks and outer range
  checks consistently use lower tolerance1e-3/upper2e-6, preserving the actual
  signed quadrature and all non-range/consecutive gates. Four focused tests
  pass, including rejection of excess upper damage and equation/healing/drift
  failures. Exact config parity excludes only declared budget/range/recovery.
- GPU PID45229 restores certified failed124 seed; independent initial NumPy
  replay passes. Inner124 passes after1440 updates atAT2.000997808646,
  minimum-8.75144e-6; physical outer124 keeps min-4.59579e-6, AT2.0769857360,
  reaction569.8426451. Independent NN/own-H/half-relaxation/source/signed-FEM
  force/current-H raw residual replay passes. This clears the previous inner
  stop but does not certify coupled53; raw updated-H AT2.32955 remains unready.
- GPU continues requested53..75. CPU-only observer48624 replays each newly
  finished load, checks parent/NN/own-H/PDE/signed force/acceptance and compares
  accepted fields to matched direct reference, with negative counts/volume
  fractions recorded. No reference fields enter training. Observer reports
  audit failure/worker interruption honestly; no automatic app notification
  scheduler is available or claimed. Source code remains pinned while live.
- Relation: user-requested full-path relaxed-lower-range experiment; old strict
  and tolerant certifications remain distinct, full fracture success pending.

## 2026-10-12 — Recover interrupted relaxed-range pending126 and detach long jobs from tool sessions

- User requests continued execution. Both GPU45229 and CPU observer48624 are
  absent; no final root/state53 result or numerical traceback, cgroup OOM count0.
  Last completed physical126 is persisted; inner127 progress to3424 iterations
  is unsaved. Do not label process interruption as a residual/range rejection.
- Preserve original interrupted root/code/config. Independently replay pending126
  NN/own-H/parent/BC/signed FEM force/AT2 receipts, then start a declared fresh
  same-load pending recovery at127 with its last serialized126 network. Accepted52
  remains physical loading parent, not pending126. Identical physics, parameter
  scales,8192-stage budget, lower1e-3/upper2e-6 and all final gates.
- New pending-recovery entry validates absence of finalized state/root receipts,
  pins identity/prior audits/parent/parameterization/pending/sweep, and explicitly
  restarts only L-BFGS memory/AL duals. No assertion that unsaved127 weights were
  recovered. Numerical behavior after recovery follows the same original loop.
- Use a double-fork/session-detached supervisor with redirected file descriptors,
  per-job worker/supervisor receipts and recorded exit code, instead of tying
  production/observer lifetime to monitored tool sessions. Test this mechanism
  with a harmless short subprocess before the GPU launch. Cause of earlier
  process exits remains undetermined; this isolates lifetime management.
- Relation: authorized full-path attempt continues autonomously; process recovery
  changes execution/provenance, not physical accuracy or loading acceptance.

## 2026-10-12 — Pending recovery starts independently of execution sessions

- Two focused pending-recovery tests pass: exact config parity and rejection of
  finalized/promoted/tampered source. Independent pending126 NumPy/NN/own-H/
  half-relaxation/signed-FEM replay passes. Saved NN86181055... and fieldsda35ea65...
  restored exactly; initial independent replay reproduces current-H AT2.28630148,
  without incorrectly requiring this initial NN to be a solved new fixed-H image.
- Double-fork supervisor self-test survives launcher exit and records expected
  child exit23. First actual launch used JAX_PLATFORMS=gpu and explicitly failed
  on unavailable rocm before numerical initialization; exit1 receipt retained.
  Correct backend selection to cuda (as listed by local JAX), without solver/
  config change. GPU51924/supervisor51923 and CPUobserver54698/supervisor54697
  are in separate sessions; supervisors reparented toPID1, logs/exit receipts
  redirected to files. No tool-session process owns their lifetime.
- Optional scalar optimizer-trajectory check at inner127 iteration3424 is not
  bitwise equal to the old interrupted run: relative residual differs by about
  .36%, although initial source/network/history replay matches. Do not claim
  identical restarted optimization trajectory or infer a formulation change
  from that proxy. Independent full PDE/NN/force replay remains acceptance
  evidence; no numerical relaxation or speculative explanation adopted.
- Relation: continue the same authorized full-path experiment with explicit
  interruption provenance; critical physical equilibrium remains unresolved.

## 2026-10-12 — Restored pending126 advances to127 with independent certification

- Detached GPU51924 completes inner127 after7072 updates, fixed-H AT2
  .000999834195/min-4.24891e-5. Physical127 has reaction568.1284613,
  min-3.10120e-5/max.99619111, current AT2.0777683852 and raw current-H
  NN AT2.2673470. Independent NumPy source/NN/own-H/half-relaxation/signed
  FEM force2.35343e-5/current residual replay passes. Loading parent remains52;
  no critical53 acceptance or next loading state yet.
- Matched direct transient127 reaction568.5057714, relative difference.0663687%.
  This supports intermediate reaction agreement; final reference remains1.57856
  after1296 outers and is not yet reached. No saved direct127 field arrays, so
  do not claim full-field transient equality or monotonic physical residual.
- Worker and observer remain alive after multiple launcher/inspection sessions
  finish; independent session/PID1 supervisor receipts preserved. Numerical
  source hashes remain unchanged. Continue requested53..75 automatically under
  unchanged relaxed-range contract; no additional numerical policy introduced.
- Relation: certified process recovery, not a completed fracture solution.

## 2026-10-12 — Status inspection confirms cross-turn background progress through151

- User requests current status. Detached GPU51924/observer54698 and supervisors
  remain live after about74/72minutes; process-lifetime repair has survived the
  preceding conversation turn. No final numerical report, acceptance53 or
  new numerical policy. Continue the existing authorized full-path job.
- Independently replay a captured immutable pending151 snapshot: NN/features/
  history/half-relaxation/signed FEM force/PDE/source/parent checks pass.
  R554.8855908 versus direct transient151556.0662846 (.21233% difference),
  physicalAT2.0653540, dmin-2.06886e-5. Inner equations and negative bounds pass
  for all completed127..151; physical changes/healing/equilibrium still unready.
- Recent10 inner optimizations average165.35s: neural optimization cost is the
  current practical bottleneck. Same-stage scalar agreement is not final field
  convergence; direct critical reference takes1296 outers. Do not provide a
  precise completion estimate or loosen accuracy based on a status question.
- Relation: verify ongoing progress, maintain lower1e-3/upper2e-6 and original
  physical gates, no restart or unchanged expensive test-suite replay.

## 2026-10-12 — Plot current damage errors with final-state and fixed-history targets distinguished

- User requests damage-field error and figures. Capture latest saved physical
  pending153 and preceding152 into immutable plotting artifacts before rolling
  checkpoint rotation, preserving actual signed d and declared status.
- Primary comparison: same mesh/operator/physics/load.0053 final converged
  direct153-load-index53 reference (outer1296). Report volume-weighted nodal
  relativeL2/RMSE/Linf, shared-scale clouds, absolute difference and centreline.
  No pointwise percent error near zero damage and no claim of transient field
  equality from same-stage reaction agreement (direct153 fields not retained).
- Add a CPU-only fixed-H diagnostic by checked original volume-weighted AT2
  linear solve at the saved H_used. Compare saved NN damage_image against that
  solution to isolate inner neural approximation from uncompleted coupling.
  Verify the original equation and transform, and report solver residual/ranges.
  This diagnostic is never used as a training label, physical state or loading
  parent; keep GPU continuation/thresholds/source code unchanged.
- Relation: current global final-field mismatch and fixed-H neural error answer
  different questions; figures must label targets/status explicitly.

## 2026-10-12 — Quantified same-load damage errors and clarified reference labeling

- Both physical fields use loading step53 and prescribed displacement.0053.
  Reference is that step's converged direct result (outer1296), not loading
  endpoint75/.0075. User requested explicit clarification; figures now state
  the load and same-load convergence status rather than ambiguous "final".
- Physical153 versus that reference: volume-weighted nodal relativeL2
  77.2400166%, RMSE.100629556, Linf.965563791. Crack localization/propagation
  differs strongly; this is not a pure neural approximation error.
- Saved inner NN versus original AT2 direct solution at identical H_used:
  relativeL2 4.71821184%, RMSE.0029778974, Linf.0084682641. Checked original
  volume-SPD equation, symmetry/sign/positivity assumptions and CG stopping
  criterion;116 iterations, full original AT2 residual6.24586e-9.
- Diagnostic only, original production model/training/acceptance unchanged.
  No new direct training labels, output clipping or physical-state promotion.
  Reject interpreting same-stage force agreement or similar maximum damage
  as full-field convergence. Relation: earlier matched-reference comparisons
  and today's final-state/fixed-history diagnostic decision.

## 2026-10-12 — Continue live critical-load PINN with a diagnostic checkpoint

- User asks what to do and whether computation continues. GPU51924 and
  observer54698 remain live under independent PID1 supervisors; logs advance
  through physical158 into inner159. Accepted loading parent is still52.
- Retain original model, trained output, asymmetric negative1e-3/upper2e-6
  and original coupled/inner acceptance gates. Keep authorized53..75 worker.
  Next inspect around physical200 using same-load final field error,
  identical-history neural error and spatial crack propagation, together with
  coupled residual/change/healing.200 is a diagnostic point, not an acceptance
  or termination threshold; current configured outer cap remains2048.
- Reason: physical127..157 reaction568.13 ->551.67, volume mean damage
  .0337645 ->.0346672 and damaged-node fraction increase, while own-H changes
  decline. Coupled residual oscillates and remains about.0675; these are
  progress indicators, not proof of eventual convergence or harmless NN error.
  Same-stage direct157 reaction553.22 differs.28114%, scalar comparison only.
- Recent10 inner solves average233.48s, last completed157285.17s. Some require
  two optimizer stages; accept their final passing residual, not an isolated
  stage's iteration-limit message as a numerical failure. Training cost rises.
- Reject declaring convergence from dmax/reaction alone or blindly assuming
  more iterations must succeed. If a later window shows stagnation, first
  isolate fixed-H approximation versus outer coupling before a separately
  recorded optimizer/coupling change. No change implemented in this session.
- Relation: earlier153 fixed-history diagnostic and matched-load error plots;
  sustain the existing experiment with evidence-based checks.

## 2026-10-12 — User-proposed detached residual weighting and core-halo pilot

- User proposes static crack-region weighting, damage-dependent weighting and
  nonuniform core+halo sampling to reduce neural cost; explicitly stop gradients
  through weights. Implement a separate fixed-H diagnostic, leaving pinned live
  production code and coupled loading unchanged. Same checkpoint/H/network and
  budget across unweighted/static/damage-dependent cases; no direct training labels.
- Existing objective is stabilized degree-two L16 residual power loss. Weight
  the residual amplitude by sqrt(w), with positive weights normalized to nodal
  mean1; w=1 recovers identical original loss/gradient. Experiment controls are
  explicit (boost9, damage power2, geometric band widthl0), not physical laws or
  generally validated optimizer settings. Static weighting uses known notch-tip
  to right-boundary segment, not the reference crack. Dynamic weights use the
  saved physical damage and remain fixed throughout a line-search/fit block.
  Apply stop_gradient to weights/H/AL buffers. Update weights only between
  outer iterations/optimizer blocks, not inside parameter-dependent line search.
- Weighting preserves the positive-weight residual zero set; it changes training
  norm/conditioning, not AT2. Acceptance and comparisons retain full original
  residual and asymmetric bounds. Do not infer speedup from weighted loss alone.
- Derive core+halo from exact nonzero CSR rows, retaining each core's full halo
  and original coefficients/normalization. Test row/gradient equality. Nonuniform
  sampling needs explicit importance factors if estimating the original nodal
  moment; oversampling without correction intentionally reweights training.
  No arbitrary operator truncation or sampled acceptance; fixed padded shapes
  and masks required before a production sampled optimizer.
- Relation: same-history diagnostic isolates representation/optimization error
  from outer coupling. Full-path worker remains active while experimental
  methods and meaningful focused tests are developed separately.

## 2026-10-12 — First detached-weight pilot does not improve full-field convergence

- Actual CUDA/x64 comparison completes normally (pilot76319 exit0). Same
  independently certified physical153 network, its current ownH (not prior
  H_used), load.0053 and2048 L-BFGS iterations, penalty1000, original full-domain
  AT2/range gates. Static/damage weights use boost9, bandwidth.024, power2,
  positive mean-one normalization and sqrt(w) amplitude in original L16.
- Original full residual after same budget: uniform.00125160964,
  static.00161167801, damage.00187067839. None passes.001 in this short pilot.
  Fixed-same-H diagnostic damage L2 errors4.72986%,4.74799%,4.74688%: no
  measured field improvement for these settings. Reject immediate adoption or
  claiming this disproves all weights/scales; objective and snapshot are specific.
- Existing L16 already emphasizes peak residuals. Simply emphasizing a predicted
  crack/damaged region need not improve global peak accuracy or conditioning.
  Weighting changes optimization priorities; it does not reduce full-graph
  per-evaluation cost. All five focused detachment/parity/gradient/support/
  importance tests and independent NumPy checkpoint/NN/PDE audit pass.
- Core1024 draws need uniform/static/damage halo14554/13515/14567 of71307nodes,
  retaining16315/16308/16339 coefficients of1138075 full coefficients. Exact
  original row equality passes. This is support-size evidence, NOT measured
  sampled-training speedup. It motivates the next sampled-network experiment,
  which must compute NN only on halo and preserve full precomputed H features;
  slicing a full network prediction would retain its primary cost.
- Shared GPU fit times59.82/60.02/59.75s exclude JIT warmup but are not an
  isolated speed benchmark. Main worker51924 remains live, physical164 saved
  and inner165 progresses; no protected numerical source/gate update.
- Relation: user-proposed acceleration, fixed-history pilot complete; actual
  sampled core+halo training remains the next experiment, not adopted production.

## 2026-10-12 — User defers acceleration until physical outer200 review

- User explicitly prefers waiting for original outer200 before deciding on
  further acceleration. Defer new weighting/sampled training; keep the existing
  authorized main53..75 continuation and all numerical settings/gates.
- Latest scalar physical170/load.0053 is unready, R545.557165/AT2.0625691;
  inner171 training. This is a checkpoint review target, not solver acceptance
  or an instruction to stop main production at200.
- Add only a read-only detached CPU checkpoint observer: capture exact200 NN,
  physical fields, preceding199 fields, metadata and source hashes before
  rolling rotation, then reuse validated CPU diagnostic/plots at same load.
  It cannot launch training, modify production, advance states or change gates.
  Report explicitly if state53 finalizes/exits before200 or200 was missed.
- Relation: user-selected observation checkpoint takes precedence over the
  proposed next sampled-core training; preserve evidence for that decision.
- Execution receipt: read-only CPUobserver79842/supervisor79841 started
  independently; exact-copy/hash-rejection fixture passes. Future output
  damage_error_maps_20261012_outer0200 holds figures/report; control
  /tmp/fracture_outer200_observer_daemon_20261012 records capture/analysis/exit.
  This preserves evidence locally; no app-level automatic notification promised.

## 2026-10-12 — User stops slow full-graph continuation and authorizes core-halo trial

- User finds crack growth/coupled convergence too slow and explicitly authorizes
  interrupting the current run and trying earlier core+halo sampling. This
  supersedes waiting for200 before acceleration; retain the same active task.
- Stop only owned main51924/final observer54698/outer200 observer79842 with
  SIGTERM after verifying command identities. Supervisors retain exit receipts.
  Save last complete pending fields/NN/preceding state/checksums; unsaved inner
  optimizer progress is not recovered or promoted. Accepted loading parent52
  remains unchanged. No change to physics, residual/range/coupled criteria.
- Implement sampled network evaluation on exact halo, full-graph known history
  features computed once and sliced, unchanged compact basis column scales.
  Sampling probabilities/importance factors detached and fixed within each
  L-BFGS block. Rebuild fixed-shape pools between blocks, resetting optimizer
  memory because sampled objective changes. Full-domain metrics determine
  original readiness. Keep both uniform and crack-biased sampled controls
  against full-domain training at same checkpoint/H, with fair compilation
  exclusion and now isolated GPU timing. No direct training labels.
- Relation: replace wall-time-heavy full-graph inner experiment with measured
  core-halo optimization; original193-era fields/plots/provenance remain.

## 2026-10-12 — Protect full-domain residual peaks in sampled training

- Four-case isolated GPU pilot exits0. Warm value+gradient times original
 15.135ms/cached8.662ms/uniform3.510ms/crack3.355ms.2048 fits60.23/36.27/
 28.45/19.84s, full original residual .0010764/.0010803/.88285/.0050128.
 Smaller subgraphs improve per-evaluation cost, but random cores alone miss
 governing peak rows and fail original whole-field readiness. Do not adopt
 their output or equate time per update with time to verified convergence.
- Test a new explicit stratified core: mandatory512 largest current full
 absolute residual rows, plus1536 random draws from the complementary nodes,
 with frozen crack-biased positive probabilities. Recompute only between
 optimizer blocks. Preserve complete original halo, fixed padded capacity
32768 with overflow failure, original forcing RMS and full-field gates.
- Independently derive original nodal moment estimator:
 (sum mandatory f_i + mean_random(f_i/q_i))/N. With C=M+K core rows,
 mean over core uses coefficient C/N on mandatory and C/(N*K*q_i) on
 random rows. Same coefficients apply to range-AL moment; all detached.
 Nonlinear L16 root still is not claimed unbiased. Test exact expectation
 and all-core objective/gradient parity before GPU training.
- Then full-cached polishing can reach original .001 accuracy. Compare total
 sampled+polishing time against the already measured cached baseline plus
 equivalent continuation/polishing. Preserve same checkpoint/H and penalization,
 explicit memory resets and early full gates, no direct targets or load update.
- Relation: user-authorized core-halo acceleration; preserve peak accuracy
 while separating reliable feature-cache savings from sampling limitations.

## 2026-10-12 — User selects geometric random density and frequent fresh resampling

- User explicitly requests fully random cores, probability highest near the crack
  and decreasing towards both sides, following elasticity sampling, with ongoing
  resampling. This supersedes the proposed mandatory top-residual512 strategy;
  no mandatory-point GPU trial was run and its unused helper is removed.
- Elasticity uses normalized 1+refine_weight/r^power near an elliptical hole.
  Adapt geometry only: rho=sqrt(1+(distance_to_segment/width)^2), raw=1+49/rho^2,
  then p=raw/sum(raw). Segment(.5,.5)->(1,.5), width.024. This is an explicit
  optimizer experiment, not a physical law; positive background covers all nodes.
  Probability/distance/importance weights are stop_gradient. No damage product.
- Compare fresh independent with-replacement2048 cores after each8 or32 L-BFGS
  updates at identical current ownH192/.0053/source NN and total budget2048.
  Keep each line search on one frozen sample, reset curvature memory on switching.
  New samples are generated throughout, not cycling a prebuilt pool. Complete
  original CSR halo, fixed padded32768 capacity (overflow fails), full knownH
  features, original coefficients/normalization/bounds/full-field gates remain.
- Original nodal moment uses 1/(N*p_i), with independently tested expectation;
  nonlinear L16 root not claimed unbiased. Include packing/resampling and full
  checks in fit timing. First isolated pilot independently passes NumPy/CSR/CG
  replay; its cached full baseline36.27s/AT2.0010803 is retained, not rerun.
- Relation: user-directed sampling experiment; accepted52 unchanged, no physical
  state promotion, direct training labels or coupled-load advance.

## 2026-10-12 — Frequent random resampling improves pilot but needs global polishing

- New inverse-power geometric random sampling2048 updates completes normally:
  every8/32 update refresh gives fullAT2.00280961/.00216050 in31.94/24.48s.
  All256/64 independent core draws replay from seed, complete CSR halos fit
 32768, NumPy NN/full residual/geometry/probability/CG audit passes.9 focused
 tests pass. Keep probabilities detached and freeze them inside line search.
- Prefer32 over8 for the next experiment: lower original residual and lower
 wall time here. Do not conclude resampling alone caused the improvement:
 density differs from old static-times-damage profile. All fixed-H gates still
 fail.001; fixed-H nodal damageL2 about4.93%, almost unchanged.
- Reject promoting sampled outputs or declaring certified speedup. Next full
 cached polishing on the audited32-case NN, compare total sample+polish cost
 against cached baseline+required polish. Physical coupling/ownH/accepted53
 remains separate. Accepted52 and original stopped physical192 preserved.
- Relation: first user-selected random-density/resampling experiment complete;
 broad nonlocal fracture task stays active, no GPU worker remains.

## 2026-10-12 — Distinguish Adam state persistence from L-BFGS block resets

- User correctly notes elasticity keeps optimizer memory when changing sampled
  subgraphs. Source confirms train_base.py default Adam, opt_state initialized
  once and threaded through each optim.update; pool switch every5 changes only
  collocation data. Fracture pilot instead calls SciPy L-BFGS-B minimize anew
  per8/32-update block; NN parameters warm-start, curvature history does not.
- L-BFGS secant y=gradient_new-gradient_old should describe parameter change
  on a consistent objective. Using gradients from different sampled batches
  also includes a sample-change term. Reset is a conservative deterministic
  solver choice, not a requirement of sampling or stop_gradient. Reject claiming
  that Adam must reset or that L-BFGS resets are the only valid stochastic
  quasi-Newton design. Shared-batch curvature reevaluation is a distinct method.
- Frequent resets lose curvature information and add startup/check cost; current
  full-versus-sampled comparisons do not isolate sampling from optimizer resets.
  Prefer testing continuous Adam state with fresh random cores for the next
  sampling-method control, optionally followed by fixed full-domain L-BFGS.
  This is a recommendation, not an executed optimizer replacement or a claim
  Adam necessarily improves fracture convergence. Existing32-block outputs and
  .001 full gates remain unready; no worker launched on this explanatory turn.
- Relation: user's optimizer-state question refines next experiment design;
  physics, source192, accepted52, audited results and all artifacts unchanged.

## 2026-10-12 — User authorizes persistent Adam random-core trial

- Test Adam with state initialized once per independent case, preserving first/
  second moments and schedule count across freshly drawn random batches. Same
  audited192 NN/currentownH/.0053, exact original AT2/basis/halo and detached
  probability/importance buffers. No state53 promotion or labels from reference.
- Explicit exploratory initial rates1e-3/1e-4/1e-5, each2048 updates; smooth
  exponential decay to10% across2048, Adam b1=.9/b2=.999/eps1e-8/eps_root0,
  clip global gradient norm1000 (no damage clipping). Reuse crack geometry
  inverse-power density49/width.024/power2 and2048 core/padded32768 halo;
  fresh random draw every5 updates as elasticity default cadence. Full original
  acceptance checks every128 and at end; save best checked network separately.
- Compare the best sampled rate with full cached Adam at identical settings and
  source to isolate sampling effects. Fixed full cached L-BFGS polish on best
  checked sampled/full-Adam outputs plus prior full-L-BFGS output is permitted
  to compare total cost-to-gate; budgets explicit and failures reported, not
  accepted by sampled loss alone. GPU isolated, compilation excluded, actual
  packing/sampling/check cost included, no nonfinite gradient sanitization.
- Verify persistence against direct sequential Optax update/reference reset and
  fullcore sampled/full objective parity before actual GPU integration. Shared
  audited problem setup is factored from existing diagnostic runner; protected
  production numerical source remains unchanged. No new physics or adaptive
  numerical law presented as established; these are optimizer experiments.
- Relation: user-approved next experiment after clarification that elasticity
  preserves Adam state while fracture block L-BFGS restarted curvature memory.

## 2026-10-12 — Persistent Adam trial completes; sampling advantage remains limited

- CUDA99571 exits0. Each sampled rate1e-3/1e-4/1e-5 runs2048 updates,
  fresh random core every5 (410 draws), state counters/moments persist.
  Final fullAT2.02217088/.00260810/.00207765; best checked.01526496/
  .00175885/.00189423. Select1e-4 best checked at1792 for polishing, not its
  final checkpoint; report this choice and all2048-step training cost explicitly.
- Full cached Adam at selected1e-4 gives.00139927 in22.28s, versus sampled
  19.82s with fluctuating full residual. Geometry/exact halo/H/features/AL/
  importance/ranges unchanged. Sampled Adam changes cadence to5 as configured;
  this is not an isolated comparison of preserving versus resetting moments.
- With2048 extra full cached L-BFGS updates each: sampledAdam route56.17s/
  .00107347, fullAdam route57.57s/.00107517, oldfullL-BFGS route72.02s/
  .00104459. All original .001 gates still fail. Route times exclude compile
  and additional rate-search/control costs; search itself59.16s. Do not claim
  verified time-to-convergence speedup or automatically adopt sampling.
- Same-H diagnostic damageL2 after polish4.92074/4.90599/4.91050%; very
  similar fields.11 focused tests and independent NumPy NN/CSR/CG, detached
  probability formula, all1230 random draw/halo replays and final/best saved
  Adam/schedule counter checks pass. Physical source192 and accepted52 stay.
- Prefer resolving the common late residual plateau under full gates rather
  than attributing it solely to memory reset. Further same-H polishing on the
  independently audited branches can establish actual time-to-gate; resume
  stochastic Adam with its matching saved state if used. No worker remains,
  no coupled load advanced, no direct labels or protected production edits.
- Relation: user-approved persistent-state experiment complete; broader nonlocal
  fracture task remains active and critical load53 unresolved.

## 2026-10-12 — Clarify Adam route versus rate-search cost accounting

- Sampled route56.16749s already includes selected-rate training19.82341s.
  Three-rate search aggregate59.16094s includes that same case; only unselected
  rates39.33753s are additional search overhead on top of the chosen route.
  Do not add59.16 to56.17 and double-count selected training. Full Adam control
  and other polishing branches remain separate experimental cost. Earlier
  wording about additional rate-search cost denotes experiment overhead, not
  the full search sum to add again. Under-gate convergence remains unverified.

## 2026-10-12 — Evaluate polished sampled-Adam damage with its own FEM fields

- User requests physical fields/reaction corresponding to sampled Adam+full
  polishing. Earlier request to time full metrics is deferred; no profiler
  or training was launched on that interrupted turn. Same active task remains.
- The trial contains only a fixed-H NN damage image, not a coupled displacement
  state. For visualization hold this raw polished damage, solve original signed
  plane-strain P1 FEM at load.0053, and compute reaction from original weak form.
  Separately evaluate original omega.5 candidate from physical192 as a clearly
  labeled unaccepted next-sweep diagnostic. Do not change accepted loading state.
- Diagnostic ownH=max(committed load52H, own FEM positive driver), as original
  scheme; frozen trainingH192 remains separate. Report AT2 both before/after
  this ownH change, healing and equilibrium, avoiding a false coupled-ready
  label based on fixed-H training residual. Preserve signed damage/no clipping.
- Same-model reference is converged load53/.0053, not terminal75/.0075.
  Pin source field/network/accepted parent/reference hashes, verify independent
  host stiffness residual/reaction and exact BC/history construction. Produce
  static scientific clouds/reaction comparison, no training or state promotion.
- Relation: user-selected output visualization; wider fracture task incomplete.

## 2026-10-12 — Own-FEM visualization shows small physical advance, no coupled acceptance

- Hold raw polished sampled-Adam damage and solve own signed original FEM:
  R535.101585394; original omega.5 candidate gives535.536511777 versus
  saved192535.966305815. Host stiffness equilibrium/BC/driver/reaction audit
  passes, no damage clipping. Both use load.0053 and ownH from committed52.
- Frozen-H rawAT2.00107347 does not imply coupled equilibrium: ownH raw/
  relaxed AT2.0544446/.0548413, healing~.0087>.001; no candidate promotion.
  Same-load final-reference nodal damageL2 about75.67/75.71%; this differs
  from same-H neural4.92%. Ref53 is converged at.0053,R1.57856, not final75.
- Six-panel signed cloud and same-load reaction comparison saved/inspected,
  with separate raw-output and omega.5-candidate labels. Crack near tip and
  centreline nearly unchanged from192. Do not interpret a single fixed-H fit
  as advancing many physical outers or reaching through-crack equilibrium.
- Relation: user-requested physical display complete; source/accepted52 intact,
  no numerical worker remains. Interrupted full-metric-cost timing still deferred.

## 2026-10-12 — User authorizes provisional loading past unready0.0053

- User accepts a possibly delayed crack and explicitly requests accepting current
  state provisionally and increasing load. Authorization overrides the earlier
  requirement to stop on failed inner/coupled readiness for this new experiment.
  Existing certified branch and original gates remain intact; do not relabel an
  unready state as a converged solution. Same active broad task, separate run.
- Start from audited omega.5 physical candidate at.0053 (R535.536511777),
  sampled-polish NN/source scales. Commit this as user-provisional53 and advance
  indices54..75 (.0054..0075). Original signed P1 FEM/nonlocalAT2/identityNN,
  omega.5 and committed previous-load history remain; no labels/projection.
- Cached full L-BFGS avoids stochastic noise for this loading-policy test.
  Each load max8 physical outer updates, each inner max512 updates; original
  residual/range objective and early full gate retained. If inner range fails,
  explicit original penalty sequence1e3/1e4/1e5/1e6 with dual updates can use
  another512 budget; stop loudly if still invalid. Numerical nonfinite/range/
  FEM equilibrium failures still stop. Finite range-admissible inner residual
  or coupled failure alone no longer prevents user-provisional load advancement.
- Keep original readiness diagnostics and consecutive3 checks; report certified
  versus provisional per state. On provisional advancement, report healing,
  residual, reaction/damage/history changes/drift and propagate source-provisional
  ancestry; a later locally-ready state cannot certify the entire loading path.
- Diagnostic crack extent uses thresholdd>=.9 connectivity along original P1
  mesh edges, from explicit radius.024 around tip(.5,.5) to rightedge x=1.
  This reports a connected highly damaged band, not a new physical acceptance
  rule. Save fields/net at each sweep/load, restart ancestry/code/source hashes.
- Relation: tests user's delayed-crack hypothesis; matching-model direct reference
  already fractures at.0053, so additional-load success alone does not establish
  a valid critical-load shift. No repeated approval needed for this experiment.

## 2026-10-12 — Same-budget direct-damage control separates loading lag from NN error

- Provisional NN loads through.0061 still lack a spanningd>=.9 band; reaction
 501.28/frontx.62354 after per-load8 outer/512-inner budgets. User's delayed
 fracture hypothesis needs a matched iteration-budget control, not comparison
 only with fully converged reference. Add separate same-source/same-model/
 same53 history/omega.5/54..75/8-outer direct AT2 CG control. No reference
 damage is fed to NN; direct CG is the actual independent control solver.
- Reuse previously validated volume-SPD at2_cg.damage_solver with original
 geometry/operator/material/FEM/committed-history and CG tolerances. Keep
 original diagnostic gates/user-provisional ancestry and range/FEM checks.
- Control can share JAX GPU with existing NN worker; never run FEniCS/MPI.
  Record GPU-shared status; compare force/crack/error only, not timings.
  Existing worker numerical code/config unchanged while live. New control
  is a reversible experiment in the same authorized loading investigation.
- Relation: user requests higher-load fracture exploration; causal control
  distinguishes NN approximation/training effects from deliberately limited
  coupling sweeps. No new physical formulation or supervised labels.

## 2026-10-12 — Direct damage also fractures late under the8-outer loading policy

- Same-source53, same original physical model and8 outers/load direct AT2 CG
  control completes54..75. First tip-connectedd>=.9 through-ligament band
  at74/.0074, R.208916; terminal75R.182161. Independent NumPy/CSR CG/own
  signed-FEM/history/ancestry audit passes. Shared-GPU timing is not compared.
- Fully converged same-model reference fractures at53/.0053. Therefore the
  apparent loading delay cannot be attributed solely to NN/Adam/L-BFGS.
  Capping coupled updates while committing unready states itself delays crack
  evolution. NN at71/.0071 still shorter (frontx.79674,R341.7246) than
  capped direct control (frontx.88335,R176.2795), so NN error also contributes.
- Relation: user accepts provisional higher-load experiment. Do not report
  higher-load fracture as a verified material critical-load shift or label
  this budget-limited branch converged; complete NN75 and matched-load plots.

## 2026-10-12 — Higher-load neural trial completes without through-fracture

- User-authorized provisional53..75 branch finishes exit0, all states locally
  unready. Final.0075R267.658338/frontx.852169, no spanningd>=.9 band;
  original own-H AT2.346499>.001, maximum load-to-load damage healing.07213.
  Independent all-load NN/FEM/history/PDE/ancestry replay passes; original
  certified52 branch unchanged. Seven focused tests pass; final plots inspected.
- Increasing load does advance the crack but does not produce immediate full
  opening under these8-outer/512-inner budgets. Matched direct8-outer control
  fractures at.0074 versus fully converged reference.0053: coupling budget and
  neural solve accuracy both affect the apparent lag. At same.0075 NN damage
  L2 error64.25%, capped direct38.24%; through-band alone does not certify
  an equilibrium path or establish a harmless small critical-load shift.
- Exact next possible experiment is fixed-load75 additional coupling from
  saved75 fields/net with74 committed parent history; a new restart must
  distinguish physical iterate H75 from committed H74. Do not silently change
  the equation by treating75 as its own parent. No new long run started here.
- Relation: finishes the user's higher-load request and leaves the broad verified
  nonlocal PINN fracture task active with self-contained source/next-action record.

## 2026-10-12 — Preserve higher-load findings in a standalone dated stage report

- User explicitly asks to record this interesting result before further work.
  Create docs/reports/fracture2d-provisional-loading-20261012.md with Chinese
  explanation, exact model/source/budgets, same-load comparisons, figures,
  residual/healing limitations, validation and possible fixed-load next test.
- Copy three figures and audited JSON/identity/exit/comparison evidence into
  repository report assets with source paths and SHA256 manifest, preserving
  readable stage evidence independently of temporary run directories. Large
  fields/weights/logs remain at their original referenced archives.
- A dated experiment report is a fixed research snapshot, not another active
  task or example STATUS file. Keep one active fracture task and its current
  status/resume responsibilities; link report from task, PROGRESS and README.
- Clarify interpretation: capped direct control differs from zero-start strict
  reference in history and submission policy as well as outer budget; do not
  claim an isolated8-update causal effect or verified critical-load shift.
- Relation: documentation-only completion of stage-preservation request;
  no new numerical run, code/config change or Git history operation.

## 2026-10-12 — Make documentation stage records visible to version control

- Final inspection finds the existing reports/ ignore rule also hides
  docs/reports. Add the narrow directory exception !docs/reports/ so the
  newly requested stage report and evidence appear as untracked documentation.
  Other reports directories remain ignored. This changes an ignore file only;
  no staging, commit or other Git history operation is authorized or performed.
- Relation: prevents the standalone stage record from being silently omitted
  when repository documentation is saved later. Attachment hashes unchanged.

## 2026-10-12 — User requests128-outer PINN loading from certified0.0052

- User proposes restarting the converged0.0052 PINN state and increasing
  per-load alternating updates8 ->128, then explicitly selects full loading
  through0.0075. Same active task, separate restart root; preserve dated8-outer
  report and original certified prefix. Source state52 physical fields/network
  and inherited parameterization are hash-pinned and independently audited.
- Keep original mechanics/nonlocalAT2/committed maximum history/identity NN/
  omega.5 and512 cached full-L-BFGS updates per penalty stage. Keep original
  diagnostics/ranges/numerical failure stops and authorized provisional loading
  after finite budget exhaustion. Stop early only on original consecutive
  coupled convergence. Save immutable8/32/64/128 checkpoints for each load.
- Comparison at first load53 outer8 versus outer128 has identical source and
  fixed previous-load H52. The old8-outer path started from unready53 after
  prior192 outers and polishing, so whole-path differences also reflect the
  changed initialization/history; do not call that a sole-factor comparison.
- Relation: replaces the previously suggested fixed-load75 experiment with
  the user's chosen certified52 restart. No direct labels, nodal corrections,
  output clipping or changes to physics/inner accuracy limits.

## 2026-10-12 — Validated certified52 restart launches with persistent completion audit

- Actual fields/net restore and original/dynamic objective-gradient parity pass
  on CUDA. Seven focused tests pass; original52 physical/current-H AT2 gates
  still pass. CPU JAX source preflight stopped for thread oversubscription,
  then same checks run on actual CUDA backend; no numerical policy change.
- Launch owned detached CUDA120695 and CPU completion observer124279 in
  independent sessions. Observer polls source receipts and saves progress;
  on exit performs independent NumPy/P1-FEM/AT2/history/checkpoint audit,
  scientific figures and completion summary. It does not alter GPU fields.
- First-load53 immutable outers8/32 independently replay; R613.7921/609.4937.
  Current outer73R602.0803. User selected full53..75, so keep numerical worker
  progressing autonomously rather than launch another optimizer/sampling test.
- Relation: separates the neural inner solve from finite-budget coupled evolution;
  keeps original8-outer dated report intact and exact restart52 provenance.

## 2026-10-12 — Evaluate fixed-H neural error separately during128-outer restart

- At first-load53 immutable outer8/32/64 checkpoints, solve the original
  volume-SPD AT2 system independently with SciPy CG and original explicit
  tolerances/preconditioner, verifying symmetry and weighted residual norm.
  Use each checkpoint's own frozen trainingH, not converged-reference history.
- Neural same-H volume-weighted damageL2 errors1.8599/2.7733/2.8748%,
  while frozen-H neural fullAT2~.00098..001 passes. Physical own-currentH
  AT2.02085/.01573/.03185 remains larger. Inner residual gate passing alone
  does not prove negligible neural error or coupled convergence.
- Direct solutions are evaluation only, never fed to training or applied as
  nodal corrections. Diagnostic scalar report and script saved in new run root.
  Relation: numerical evidence keeps both inner accuracy and coupled iteration
  effects in view while user-authorized full128-outer loading continues.

## 2026-10-12 — First128-outer load advances with larger damage but unresolved inner error

- From identical certified52 parent, load53outer8 ->128 R613.7921 ->534.0830,
  dmax.6596 ->.9972; high-damage tip component remains short/not spanning.
  Physical signed range/FEM pass, but own-H AT2.13356, frozen-H inner.015985,
  raw-currentH NN.60523/upper-range and healing.01445 fail original gates.
  Commit explicitly user-provisionally, keep target54..75 as user selected.
- Independent completed53/four53 checkpoints/54outer8 replay passes. At128
  exact same-H original AT2 CG gives neural damageL2 error26.8151%, versus
  earlier1.9..2.9%. More outer growth now exposes the fixed512 inner-budget
  limit; preserve that result, do not quietly enlarge inner budget or interpret
  reaction decrease alone as adequate convergence. Same-load final-reference
  damageL2 remains73.5857%, a different comparison from same-H neural error.
- Root review_0053 fields/outer evolution figures inspected; current54outer26
  R413.8536, CUDA120695 and observer124279 continue. Relation: controlled
 128-outer full-path experiment remains active, source52/old8 report intact.

## 2026-10-12 —128-outer path develops through-fracture at0.0058

- First tip-connectedd>=.9 spanning band appears58/.0058outer12, R7.54249;
  final128R.669395,59R.676059, current60R~.65. Independent NumPy/FEM/
  own-history/NN/CSR/connectivity replay passes completed53..59/checkpoints.
  Earlier8-outer NN path never spanned by.0075; new restart52/128 path does.
  Initial source/history also differs, so do not claim an isolated budget effect.
- Through-band and force drop support the user's delayed-opening exploration,
  but original residual/healing gates still fail. Same-load damageL2 errors
  at58/59 are144.35%/123.55%, despite reactions~.67; reference reactions
  are1.20045/1.13983. Do not equate small post-fracture force with accurate
  global damage field or a certified loading path. Investigate actual damage
  error and same-H neural error separately after completing requested75 target.
- Preserve all original numerical policies and live sources; user requests
  status only, numerical worker continues. Record milestone independently
  without replacing the previously preserved8-outer dated experiment report.

## 2026-10-12 — Inspect post-fracture path rather than force alone

- New review_0059 signed physical fields versus identical-load direct reference
  show a through-band bending upward and broader off-crack damage; reference
  band remains horizontal. This is observed numerical field/path discrepancy,
  not evidence of a verified physical symmetry-breaking mechanism.
- Relation: same-load damageL2>100% and failed residual/healing gates require
  retaining this limitation alongside the successful force drop/through-band.
  Continue the user-selected128-outer target75 without changing live sources.

## 2026-10-12 — Same-load FEM/reference/PINN comparison through0.0060

- User requests three curves,3x5 damage panels and2x5 PINN displacement
  panels after completed0.0060. Four listed loads plus five columns are
  interpreted as0.0050/0.0053/0.0055/0.0058/0.0060; this is stated to user.
- Use the existing independently UFL-validated matched local P1/lumped damage
  FEM path, local P1 mechanics/nonlocal direct reference, and certified PINN
  prefix through52 plus its128-outer provisional restart53..60. All have the
  same material mesh, volume grips, material and history projection. Do not
  substitute the older boundary-loaded DG0-history FEM archive or old fully
  nonlocal mechanics baseline; those change the comparison model.
- Plot saved relaxed physical d and mechanics q at identical loads. Damage
  uses a common actual signed range; each displacement panel uses its own
  actual material-node extrema, with no clipping or symmetric expansion.
  Mark PINN53..60 provisional; low reaction/through-band does not certify
  equilibrium or crack-path accuracy. Background loading continues to0.0075.
- Relation: presentation/evidence only; no source, training or gate change.

## 2026-10-13 — Fresh bounded-output morphology trials after diffuse128-outer path

- User requests sigmoid/exp constraints to address tilted crack and diffuse
  positive background damage. Implement no-z0 sigmoid(z) and exp(-z^2); bounds
  alone do not constrain positive background amplitude or spatial orientation.
- Independent same-H CPU CG at completed60 gives outside-band mean damage
  .0356974 versus physical PINN.1198432 (abs(y-.5)>3*l0); converged same-load
  reference.0290967. Most diffuse background is therefore neural imbalance;
  inherited history also differs. This mask is evaluation only, not a zero-
  damage condition, sparsity loss or prescribed horizontal crack path.
- Compare fresh sigmoid/exp-square/identity networks at the same own-H52 and
  own-H60, same geometry/basis and initially matched AT2 parameter gradients.
  Explicit constant initial damage.03 avoids repeating the first-load1e-5
  saturation setup; all bias/readout/trunk remain trainable. No old weights,
  frozen reference field, direct labels, clipping or new PDE/loss penalty.
- Separate coupled bounded-head branches begin with certified52 physical
  d/q/H, fresh networks fitted only to that own-H AT2 residual. Keep128 outer/
  512 inner,omega.5, committed previous-load maximum history with stop-gradient
  and original gates. Provisional advancement remains explicitly labeled.
  This is a physical-parent restart, not an independently own-zero bounded
  trajectory or a certified converted52 network. Existing128 path continues75.
- Relation: expands the prior head screening from small-load residuals to
  critical-region morphology. Do not prescribe crack cleanliness by setting
  off-band damage to zero; finite positive AT2 background is physical.

## 2026-10-13 — User bypasses head screening and requests direct coupled loading

- User clarifies that prior residual comparisons are sufficient and requests
  actual bounded-output coupled solutions, allowing additional training because
  the old52 identity weights need not apply. Stop owned screening145226 after
  its two fresh bounded52 fits complete; SIGTERM exit-15 recorded. Saved fits
  remain immutable. The unrun identity/own-H60 screens are canceled.
- Direct-loading mode preserves certified52 physical q/d/H but loads no old
  identity network. Reuse only each newly initialized and4096-trained network
  with the same bounded head and identical H52/scales, all files SHA-pinned.
  Allow up to8192 additional own-H residual updates, then couple53..60 regardless
  of head residual ranking. No diagnostic direct AT2 oracle is run in this mode.
- Keep physical equations/stop-gradient history/original convergence gates,
  omega.5 and128/512 per-load/per-inner budgets. Label unready advancement
  provisional. First comparison endpoint is.0060; original identity128 path
  independently continues its user-selected.0075 endpoint.
- Relation: supersedes the preceding screening-first plan. Initial adaptation
  is network warm-start preparation, not a head-selection gate, certified
  converted52 state, extra constraint or independent own-zero bounded path.

## 2026-10-13 — Completed55 damage-cloud milestone for sigmoid

- User requests damage clouds after sigmoid0.0055 finishes, specifically to
  inspect excessive positive damage away from the crack. Compare completed
  same-load reference, original identity128 and sigmoid128 physical d.
- Use a common actual signed full-field color scale plus a separate common
  actual outside-band color scale. Exclude abs(y-.5)<=3*l0 only in the
  background diagnostic; grey indicates excluded crack-band geometry. No
  clipping, output editing or training mask is introduced. Report lumped-volume
  weighted outside mean and fraction d>.1 alongside reaction/crack extent.
- Wait for immutable state55 fields/network/report plus root curve commit;
  independently replay the sigmoid prefix before plotting. A CPU-only
  persistent milestone observer leaves all GPU workers/settings untouched.
  Different crack-development states at the same load remain visible; lower
  background alone is not certified field accuracy or an isolated head effect.
- Relation: requested physical-field milestone, not new head screening or
  a different staggered solver. Exp-square remains queued in the existing run.

## 2026-10-13 — Review completed55 clouds and recover interrupted loading jobs

- Completed55 same-load clouds show sigmoid background mean.031293 versus
  original identity.113757 and reference.029096 (72.49% reduction); d>.1
  outside-band fraction4.79% versus51.38%/reference.61%. Sigmoid crack is
  straighter/shorter and still not spanning at55, R416.22 versus identity226.15
  and reference1.41. Lower background is observed improvement, not certified
  equilibrium or proof of an isolated output-map effect.
- Both active GPU workers and completion supervisors disappear without exit
  receipts. Last bounded state is sigmoid58 outer99, R.8851267; last original
  state is68 outer66,R.4675346. GPU idle; no normal numerical stop report.
  Treat as interruption of unknown cause, not a failed model or completed path.
- Resume from hash-checked completed physical updates/networks, preserving
  parent57/67 committed history, all128/512 budgets, operators, losses, output
  forms and original convergence labels. Preserve immutable snapshots and
  record recovery driver/control receipts separately; do not rerun/retrain52
  or silently discard previous physical history. New sigmoid next outer100;
  original next outer67. Exp-square remains queued after sigmoid60.
- Relation: authorized loading goals60/75 persist. Recovery changes control
  flow and checkpoint I/O only; numerical stage bodies remain original and
  are checked against the preserved sources before execution.

## 2026-10-13 — User defers recovery; current sigmoid three-way field comparison

- User requests plotting existing results first and defers resuming calculation.
  No recovery worker has been launched; leave interrupted solver roots intact.
  This supersedes the immediately preceding planned recovery action until
  the user requests continuation again.
- User explicitly chooses0.0057 as the fifth load, so columns are
  .0050/.0053/.0055/.0057/.0058. Use matched pure local FEM, local-mechanics/
  nonlocal direct reference, and current sigmoid PINN. At50 use its inherited
  certified physical prefix;53/55/57 are completed128-outer provisional states;
 58 is the saved physical99-outer candidate, not an advanced/completed load.
- Preserve same load, material triangles and physical d/q; displacement panels
  use individual actual extrema. Independently replay58 pending fields/network/
  own history/host FEM and record snapshot hash/source, without restarting or
  solving any additional load. Retain original-head prefix provenance explicitly.
- Relation: presentation only; comparison request and accepted fifth column
  replace the immediate recovery plan, not the underlying staggered formulation.

## 2026-10-13 — Isolate sigmoid critical-step training and extend same-load coupling

- User authorizes investigating52->53 with more inner/outer training, crack-
  biased fresh core+halo sampling, and other useful accuracy improvements.
  Preserve original physical equations, local FEM mechanics, sigmoid damage,
  stop-gradient committed52 maximum history and existing readiness gates.
- At existing53outer128, fixed-H inner AT2=.0021122 after512 updates; same
  network on updated ownH has AT2=.0239172, while relaxed physical AT2=.0822923.
  Both inner optimization and moving-history coupling need separate evidence.
- Restore independently replayed53outer128 physical/network checkpoint; first
  compare same-network/same-ownH full512, extended full8192, and fresh crack-
  biased persistent Adam8192 at1e-4/1e-5 plus full4096 L-BFGS polishing.
  Global metrics every128 updates, not every Adam step. Core2048, padded halo
  32768, resample every5; inverse-power geometry density and importance moment
  correction reuse the verified sampling method. All context/probabilities
  detached, exact original CSR supports retained, no off-band damage penalty.
- Choose the lowest whole-field AT2 among the saved initial and compared
  candidates; use it only as network warm start from unchanged physical128.
  Continue at fixed load53 through at most1024 total outers with up to4096
  L-BFGS inner updates and original early residual stopping; record128/160/
  256/512/1024 physical checkpoints, ancestry and failed readiness explicitly.
  Do not commit53 as new previous-load history or advance54 for this trial.
- Same-H direct damage is an evaluation-only independent CPU diagnostic;
  physical fields, reaction, background and same-load errors remain required.
  Other interrupted loading branches stay preserved and are not restarted.
- Relation: supersedes plot-only waiting for this authorized isolated accuracy
  experiment; does not restore the previous full-loading jobs or change model.

## 2026-10-13 — Extended full training wins the matched sigmoid fixed-H controls

- All methods restore the same53outer128 sigmoid network at its updated ownH.
  Independently replayed full512/full8192 residuals .00209914/.00109531;
  same-H volume-weighted damageL2 errors10.5094%/5.6980%, times13.02/202.78s.
  SampledAdam8192 plus full4096 polish at1e-4/1e-5 gives .00203733/.00165723,
  errors11.2800%/6.3614%, times200.54/205.56s. Both fresh sampling sequences
  and persistent8192 Adam counts independently replay. Smaller learning rate
  is better here; neither sampled route beats full8192 in residual or field
  error at roughly similar observed total time. No generic sampling claim.
- Choose full8192 as the network-only warm start for unchanged physical128
  continuation at load53. The original committed52 history, omega.5 and gates
  remain; other interrupted full-loading jobs remain untouched.
- Lower outside-band mean alone is not improvement: sampled1e-4 produces
  mean.0250283 versus full8192.0310868 yet larger same-H field error. Direct
  diagnostic remains isolated from training and is not the coupled final53
  reference. Reduced frozen-H error does not establish final crack growth.
- Relation: actual outcome of the preceding budget/sampling decision. Continue
  evaluating physical reaction/path and readiness at the larger outer budget.

## 2026-10-13 — Quarter-load sigmoid continuation from certified physical52

- User requests smaller load increments after inner-budget accuracy controls.
  Isolate a new branch from certified physical52 q/d/H and its saved adapted
  sigmoid52 network; no53 candidate or reference weight/field initializes it.
  Loads use integer209..212 times2.5e-5: .005225/.005250/.005275/.005300,
  versus original1e-4. Preserve mechanics, nonlocal AT2, sigmoid/no-z0, omega.5,
  cached full L16 training and stop-gradient previous-load maximum history.
- Keep the recent per-inner4096 budget and up to1024 outer updates per small
  load, original convergence gates/three confirmations. Per-load budgets are
  equal to the extended coarse continuation; total available work can increase
  fourfold, so report cumulative iterations/time rather than attribute every
  gain solely to step refinement. Retain the previously user-authorized
  provisional load advancement at a failed cap, with all ancestry/gate failures
  explicit; never certify or hide an unready small-step parent.
- Existing matched fine direct reference has R607.684/.005225,607.675/.005250,
 1.59387/.005275,1.57096/.005300, with108/257/1575/12 outers. Small steps do
  not remove difficult snap-through coupling. Comparisons use actual loads,
  not step indices (fine states53..56 differ from old coarse indices).
- Prior same-load GPU10891/CPUobserver13274 are reversibly suspended to give
  the new GPU job priority, preserving their in-memory optimizer and saved
  physical progress. A persistent new observer audits small-step states,
  records completion/failure, and resumes the original owned processes after
  the new worker exits. No Git history or existing numerical source alteration.
- Relation: adds the requested load-refinement control within the same active
  fracture2d task; previous fixed-H controls and coarse continuation remain
  intact and the latter resumes after exclusive GPU scheduling.


## 2026-10-13 — Retain inner damage accuracy as a competing cause of fracture delay

- User asks whether PINN damage accuracy causes the slow critical-step response.
  Existing fixed-network/fixed-own-H controls show full512/full8192 AT2
  .00209914/.00109531 but same-H damage relativeL2 errors10.5094%/5.6980%.
  A near-threshold normalized residual does not certify negligible field error.
- Reassess the preceding emphasis on outer coupling: both neural inner error
  and difficult outer evolution remain plausible contributors. Reject assigning
  primary causality from inner residual pass or final-reference mismatch alone.
  Current fine .005275 has no new same-H exact-damage accuracy measurement;
  earlier coarse fixed-H errors cannot be substituted for that measurement.
- Proposed next diagnostic, not executed or a change to the active worker:
  freeze one immutable current H_used, compare its raw PINN damage image to
  the independently solved identical nonlocal damage equation, inspect crack-tip
  errors and the resulting mechanics response. Compare tighter inner tolerances
  from the identical network/H before testing a separate coupled continuation.
  A useful continuation must measure force/fields/cumulative work, not just loss.
  Direct diagnostic fields never serve as training labels or loading states.
- Relation: sharpens attribution within the existing fracture2d task; preserves
  current staggered model, history policy, bounds and active-job configuration.


## 2026-10-13 — Execute the authorized frozen-H inner-precision diagnostic

- User authorizes the preceding accuracy diagnostic. Freeze fine-load
  .005275/state55 immutable outer256 network and its H_used, not the updated
  H or a later mutable pending checkpoint. Replay the physical source first.
- Independently solve the original volume-weighted SPD AT2 equation at that
  H_used, using unchanged checked CPU Jacobi-CG controls. Measure raw network
  image error globally, in the active/crack-tip band and background; all masks
  are diagnostics only. This oracle never enters the training objective.
- Reuse unchanged sigmoid network/features/full L16 cached L-BFGS objective.
  From the same saved network, tighten the inner target1e-3 to1e-4 with an
  explicit16384-update cap and checks every128. Record a cap/optimizer stop
  honestly if the tighter gate cannot be attained; no projection, labels,
  changed bounds, objective or physical parameters. Preserve initial baseline.
- Evaluate initial/trained/direct damage using the same omega.5 relaxation
  from saved d_previous, then identical local FEM mechanics at .005275, with
  source q_previous as warm start. Direct-damage response is a one-update
  diagnostic, not the final coupled direct reference or a new loading state.
- CPU source/oracle diagnostics may run while fine GPU loading continues.
  During the isolated GPU training, reversibly suspend only the owned fine
  worker18811 and automatically resume it on diagnostic exit; its observer
  remains alive and old coarse jobs remain suspended. Retain process identity
  and suspend/resume receipts. Protected numerical modules remain unchanged;
  run-specific scripts/config/reports are archived separately.
- Relation: tests the competing inner-accuracy explanation without overwriting
  the active quarter-load or coarse continuation branches. A coupled precision
  continuation is conditional on demonstrable accuracy/response improvement.


## 2026-10-13 — Make the precision-trial evaluation cap explicit and complete its budget

- Tight frozen-H trial stops at14675 updates/15001 evaluations on SciPy's
  implicit maxfun15000, not its16384-update cap or requested1e-4 residual gate.
  Residual.00069823/globalL2=4.00227%; direct same-H relaxed-update R539.48879
  versus original539.44048/trained540.23619. Initial and trained states and
  host mechanics independently replay; original fine worker resumes normally.
- Preserve the entire first round with reports/audits/plots/control receipts.
  Complete only the remaining1709 updates from its saved same-H network in a
  separately recorded optimizer restart, explicit maximum_evaluations65536,
  original loss/memory/gtol/ftol/line-search/ranges and target unchanged.
  A run-specific copy of the cached optimizer adds explicit maxfun validation;
  do not edit protected numerical modules while loading jobs are alive.
- Record restart and cumulative iteration/time budgets; do not describe this
  two-stage trajectory as uninterrupted L-BFGS or evidence of representation
  impossibility. Reaudit the final aggregate state and identical FEM response.
- Relation: fixes experiment execution accounting within the user's approved
  inner-budget diagnostic, without changing either active loading branch.


## 2026-10-13 — Frozen-H precision trial yields limited accuracy and no force benefit

- Complete16384 additional updates at unchanged fine256 H_used, with a
  documented1709-update top-up/restarted optimizer after the implicit SciPy
  evaluation cap. Total fit time379.98s excluding compilation; final AT2
  .0007004246 remains above requested1e-4. Never mark target attained.
- Global same-H damageL2 improves4.33654% ->3.99786%, Linf.00878738 ->.00628058.
  Initial-tip error.38806% ->.25571%, propagation-band1.86941% ->1.09178%;
  moving-tip-centred radius.024 error.06693% ->.08394% does not improve.
  Low relative global loss does not imply all regions improve.
- Original/tight/direct images under identical omega.5 relaxation and local
  FEM update give R539.44048/540.26325/539.48879, physicalAT2
  .04550554/.04530405/.04544049. Direct-at-own-H also has an unextended crack;
  it is not the fully converged same-load reference with R1.59387.
- At the source physical-residual maximum, the exact algebraic decomposition
  gives inner -1.24297, relaxation -39.57695 and history-update -39.09422,
  summing -79.91414; independent NumPy/CSR replay verifies equality. Sup-norm
  contributions are not generic additive percentages or global causality.
- Do not launch an expensive tighter coupled branch from these results: the
  target has not been reached, and immediate mechanics benefit is absent.
  Prioritize a future controlled outer-update experiment while keeping neural
  accuracy diagnostics. This does not exclude accumulated inner errors or
  other loading states/architectures; it addresses this immutable checkpoint.
- Final independent native sigmoid/CSR/frozen-history/half-relaxation/hostFEM
  replay and scientific plots pass. Owned fine worker resumes after both
  isolated GPU rounds; observer remains alive and old coarse jobs stay paused.
  Relation: completes the user's approved precision diagnostic; original fine
  loading continues with unchanged settings and protected numerical sources.


## 2026-10-13 — Compare critical-step reference and PINN at matched outer counts

- User asks why the same staggered model converges with direct damage but
  not PINN. Inspect actual fine55/.005275 scalar sweeps: reference128/256/
  280/512/1024/1575 reactions580.539/526.937/519.272/449.815/293.903/1.59387.
  At280, reference coupledAT2=.0414972 and PINN=.0412069; both unready.
  PINN256/280 reactions539.440/532.184, versus reference526.937/519.272.
- Reference is also slow in outer count and remains unready at1024, exceeding
  the current neural per-load cap before convergence1575. Its recorded whole
  critical-load wall time290.13s illustrates the cheap direct inner solve;
  equal outer counts and equal wall time are different comparisons.
- Retain two possible neural effects: greater inner cost limits attainable
  outer work, and repeated inner approximation errors can alter the coupled
  trajectory. Frozen-H one-update agreement does not prove negligible
  cumulative effects or eventual neural convergence. Preceding load fields/
  histories also differ; matched-count path comparison is not an identical-
  initial-state causal experiment. Reject either final-vs-intermediate disparity
  as intrinsic neural failure, or an unconditional claim of eventual success.
- Relation: refines the completed frozen-H diagnostic and forthcoming outer-
  update priority; no worker/model/gate/budget change in this interpretation turn.


## 2026-10-13 — User-authorized L1/L2 random Adam and repository SOAP controls

- User requests corrosion2d-style low-order loss/nonuniform fresh sampling and
  Adam or SOAP without L-BFGS, explicitly points to src/pinn/optimizers.py.
  corrosion2d uses mean absolute weighted PDE residual; transfer the reduction
  style, not its AC/CH equations, time-causal terms or BC penalties.
- From immutable fine55/.005275 outer256 original network and exact H_used,
  compare nodal normalized MAE and MSE, Adam and repository SOAP at1e-4/1e-5,
 4096 persistent updates per case. Add full-field L2 Adam1e-5 control. Every5
  updates draw2048 fresh crack-biased cores, padded full original halo32768,
  inverse-power profile width.024/boost9/power2 from segment[.5,.5] to[1,.5].
  Importance multiplier1/(N*p_i) is detached; for MAE/MSE its mean estimator
  and expected parameter gradient equal the original uniform nodal objective.
  No nonlinear Lp root or added interface weight. H/features/probabilities
  detached, full metrics every128 only; exact supports and sigmoid at every
  halo node; native bounds and physical equations/history unchanged.
- Reuse repository soap() directly, with all controls explicit. Inspection
  finds its exp_avg_sq uses sum(g_projected**2) broadcast across a parameter
  tensor, unlike author reference's elementwise square at
  https://github.com/nikhilvyas/SOAP/blob/main/soap.py (raw lines159-160).
  Label results repository SOAP, document this implementation difference,
  do not silently repair shared code or generalize to standard SOAP performance.
  Adam .9/.999; repository SOAP .95/.95/precondition every5/maxdim128/no1D,
  no decay penalty and corrected bias; exponential LR decay.1 per4096 with
  minimum.1*initial, clipping1000, epsilon1e-8. SOAP first call initializes
  preconditioners without a parameter update; count this explicitly.
- Same-H CPUdirect is evaluation only; no training targets. Save both final
  and best full-RMS checkpoints plus complete optimizer state/core sequences,
  independently replay native fields, exact CSR residuals and physical
  responses. Full-L2 residual selects a candidate across controls without
  oracle values entering selection. If a candidate improves that metric,
  test a short16x512-update staggered continuation versus identical-source
  direct control, preserving optimizer state across batches/outer updates;
  diagnose rather than advance a new accepted load or alter readiness gates.
- Isolate GPU work with reversible owned fine-worker18811 pause and supervised
  guaranteed resume; coarse jobs stay paused until original fine exits. Keep
  protected modules/shared SOAP unchanged; new helper/driver/tests only.
  Relation: adds the requested objective/optimizer comparison to the existing
  fracture2d task; earlier L16 controls and loading states remain intact.


## 2026-10-13 — Confirm user-specified SOAP_JAX provenance

- User identifies https://github.com/haydn-jones/SOAP_JAX as the SOAP source.
  Repository header matches. Inspect its current src/soap_jax/soap.py:
  line212 uses optax.tree_update_moment_per_elem_norm(...,2), whereas the local
  vendored implementation broadcasts sum(g*g). Author PyTorch source likewise
  uses elementwise projected-gradient square. Record source URLs and this
  verified distinction; no shared implementation update is implicitly made.
- Continue the already authorized L1/L2/Adam/repository-SOAP controls. Local
  source SHA identifies exactly what ran; do not label it current-upstream SOAP.
  Relation: provenance clarification, not a task switch or replacement optimizer.


## 2026-10-13 — Replace shared SOAP with user-authorized SOAP_JAX 0.2.2

- User explicitly authorizes switching to SOAP_JAX. Pin package soap-jax==0.2.2
  and verify installed optimizer source against upstream commit
  a7c92f174c49b13136294a94ae0d46266b4f671f. Replace the local vendored code
  with the upstream import, retaining public soap and make_optimizer exports.
- This supersedes the unchanged-repository-SOAP trial choice above. New controls
  use upstream elementwise second moments and functional NamedTuple state. Old
  SOAP optimizer states are structurally incompatible; initialize new optimizer
  state while retaining network warm starts. Test persistence and serialization
  before running new trials. Existing live jobs do not use SOAP and are unaffected.
- Preserve the authorized MAE/MSE, detached sampling weights, original AT2 model
  and staggered local direct mechanics/nonlocal PINN damage scheme. No LBFGS
  in these controls; no claim of convergence from a residual reduction alone.

- Package 0.2.2 is absent from the configured registry; use an explicit uv Git
  source pinned to the above commit rather than an unpinned main install.


## 2026-10-13 — Add equal-budget L1 random Adam staggered follow-up

- User asks specifically about L1 + random Adam after the first controls.
  It reduces same-H damageL2 from4.3365% to2.0112% at1e-5; original selection
  by full MSE favors full L2 Adam instead. Do not equate that selection with
  best physical-field accuracy or discard SOAP field improvement just because
  MSE rises. Last SOAP1e-5 L1/L2 field errors2.8536%/3.5126% also improve over
  the original, but maximum residual and full MSE worsen. Keep both last/best.
- Add L1 random Adam16x512 continuation from exactly the same physical
  fine55/outer256 source as completed direct/full-L2 controls. Restore the
  immutable L1 best network and optimizer count4096; keep moments/schedule
  across batches/outers, same core2048/halo32768/resample5/seed1729 and
  committed previous-load H floor. Save every physical sweep for independent
  replay. Reuse the already completed identical-source direct16 control.
- This is a diagnostic comparison, not a new accepted load. Limited512
  inner updates and a carried-over decayed learning rate can affect coupling;
  separate field improvement from readiness rather than certify by force alone.
  Relation: follows the authorized low-order objective/optimizer comparison.


## 2026-10-13 — User-requested every-update random resampling control

- User asks whether rising residual with changing H is overfitting and requests
  sampling on every optimizer step. Existing controls resample every5 updates.
  Separate minibatch overfitting under fixed H from changing damage equations
  across staggered sweeps and from mean-vs-maximum residual differences. No
  overfitting attribution is established by the current observations.
- Repeat L1 random Adam16x512 from the identical immutable selected network,
  optimizer count4096 and physical outer256 source, changing only resample_every
  from5 to1. Retain detached probabilities/importance weights, seed1729,
  core2048/full original halo32768, fixed-H inner solves and persistent
  optimizer moments/schedule. Full metrics and physics are still evaluated
  per outer sweep; this is not accepted loading advancement.
- Save every physical sweep and all8192 core draws. Exact halo packing each
  update can increase CPU cost; measure runtime rather than promise speedup.
  Relation: controlled follow-up of the completed L1 random Adam diagnostic.


## 2026-10-13 — Remove avoidable fresh-batch support construction cost

- User correctly recalls elasticity2d's prebuilt subgraph pool. Inspect
  elasticity2d/train.py:317..332 and397..401: supports are built once and
  switched during training. Current fracture trial instead calls build_core_halo
  for every fresh core set: it copies/canonicalizes/checks the full CSR, slices
  rows, unions halo IDs and remaps rows in a Python loop. This is an
  implementation cost, not inherent stochastic-Adam or resampling cost.
- Precompute immutable padded original operator rows once. Fresh samples
  gather those rows and bulk-remap exact support with np.unique(return_inverse),
  avoiding full-operator work and per-core loops. Preserve genuinely fresh
  with-replacement cores each update; do not silently switch to cycling a
  finite200-entry pool, which would change the sampling experiment.
- Validate exact packed support, loss/gradients and fail-loud capacities on
  small tests, measure host costs on actual cores, then repeat the same
  resample1 staggered control to verify unchanged physical fields/counts
  with lower cost. Existing protected modules and live loading source remain
  unchanged; use a new helper and run-specific driver.


## 2026-10-13 — Separate exact support parity from cross-run trajectory comparison

- Cached/old sampled core IDs and operator supports match exactly, but an
  initial1e-10/1e-9 cross-run physical-array test fails. Retain that failure
  log and report deviations rather than claim bitwise-equivalent trajectories
  or silently loosen the physics checks. Per-run independent native/CSR/
  relaxation/history/host-FEM/state replay uses unchanged tolerances.
- Across16 sweeps, largest damage difference7.49e-5, damage relativeL2
  difference7.30e-5; final force differs.002655 (about5e-6 relative). GPU
  scatter accumulation order and nonsmooth L1 are a possible explanation,
  not isolated by this experiment. JAX docs describe implementation-defined
  conflicting scatter order: https://docs.jax.dev/en/latest/_autosummary/jax.Array.at.html.
  Report numerical closeness and exact support parity separately.
- Relation: audits the support-cache optimization without treating a very
  tight cross-run trajectory threshold as a model-convergence certificate.


## 2026-10-13 — Prebuild fresh support blocks for GPU batch switching

- User expects low-cost switching of prebuilt subpools, as in elasticity.
  Caching individual support rows removes full-CSR work, but138s integration
  still includes52s of host context enqueue over8193 calls. Prepare256-entry
  exact-support blocks outside optimizer updates, transfer each block once
  and gather batch/H/features/basis rows inside the compiled objective.
- Draw every block freshly from the same detached probabilities, preserving
  all8192 core draws and seed1729 from the resample1 control. Each entry is
  used once; block size256 divides the512-update inner budget. Do not cycle
  a fixed finite pool or freeze history-dependent features across outer sweeps.
  Optimizer state/schedule and original residual/normalization remain intact.
- Verify pooled loss/parameter gradients against unpooled exact supports and
  repeat the identical-source16x512 diagnostic. Preparation time remains
  included in measured training time; do not move costs off timer and claim
  a speedup. Source/schema and all snapshots remain reproducible.


## 2026-10-13 — User authorizes L2 random Adam with prebuilt GPU pools

- User accepts the proposed next comparison. Repeat the existing16x512
  staggered protocol with L2 random Adam and identical fresh8192-core stream,
  pool256/core2048/halo32768/resample1, source physical fine55/outer256 and
  committed previous-load history floor. Preserve model, output sigmoid,
  readiness gates and optimizer state across batches/outers.
- Restore the independently saved L2Adam1e-5 best checkpoint after4096
  fixed-H updates, count4096. L1 control restored its own L1-trained count4096.
  Both pretrains start from the same original network/H_used and use equal
  budgets, but coupled starting neural weights differ. This compares their
  complete objective-specific warm-start protocols, not only loss changes
  applied to an identical network. Explicitly report that distinction.
- Initially retain the carried-over schedule (minimum1e-6),512 updates per
  outer and original seeds to isolate the protocol comparison. Record actual
  learning rate and field/residual/force results; diagnose final fixed-H
  accuracy before changing iteration budget or rate. No SOAP coupled trial
  or full loading advancement is implicitly started by this instruction.
- Use a new run-specific driver/config and supervised fine18811 pause/resume;
  protected/shared numerical modules and immutable L1 controls stay unchanged.


## 2026-10-13 — Diagnose L2 inner-budget versus learning-rate limitation

- Equal-budget L2 pool16 continuation remains unready like L1; actual
  optimizer LR is the1e-6 schedule floor. Independent same-H field comparison
  shows L2 final raw-image error2.2699% versus L1's own-H1.7389%; this does
  not improve coupling merely by changing the norm. The user-authorized plan
  includes checking iteration/rate limitations, so isolate a frozen-H test.
- From immutable L2 coupled16 network/count12288, freeze its exact H_used
  and retain all Adam moment arrays. Compare4096 extra updates at constant
  1e-6 and1e-5, with identical fresh4096-core streams and256-entry GPU blocks.
  Rebuild only the learning-rate callable; deserialize the entire existing
  state rather than restart moments/count. For1e-6 this reproduces the prior
  floor;1e-5 increases the rate tenfold. Neither changes physical state/load.
- Exact own-H CPU direct solve is evaluation only, loaded after training;
  select best checkpoints by full MSE, report last/best/max residual and
  native fields separately. Readiness threshold stays.001. This isolates
  fixed-target optimization, not new-H physical coupling or a certified path.
  Relation: follows the completed L2/L1 pool protocol comparison.


## 2026-10-13 — Fresh L1 Adam loading; explicit weighting deferred by user

- User requests a new solve from physical zero, with displacement increments
  5e-4 through .0050 then1e-4 through at least .0060. Explicit front/high-residual
  loss weighting is recorded as a deferred next step and is not implemented.
- Use the checked local P1 mechanics/nonlocal AT2 damage/committed maximum
  history stagger scheme, sigmoid output, physical detached history features,
  geometry-only compact columns normalized by latent RMS and fresh weights.
  Initial physical q/d/H are exactly zero; fresh network output1e-3 is an
  optimization initialization only. No loss evaluation at H=0, no old neural
  weights, direct damage targets, inherited fitted parameter scales or output
  reference field. The first positive-load elastic solve establishes own H.
- Reuse L1 mean residual with1/(N*p) importance correction, detached positive
  crack-segment probabilities, core2048/halo32768/pool256, fresh sampling every
  Adam update. Preserve moments and count across batches, outer and load steps.
  Set constant LR1e-5 explicitly: the recent floor1e-6 adapts slowly; this
  is a new loading protocol, not an isolated causal comparison of loss norms.
- Keep up to128 outer sweeps and512 inner updates per sweep, original full
  inner/current AT2 and physical/update gates and three consecutive passes.
  Run the requested whole grid even when a budget is exhausted, but preserve
  provisional ancestry and never call such a state certified. Save fields,
  neural/optimizer state, parent hashes, readiness and sampling reproducibility.
  Independently replay zero/positive-load fields and host FEM before reporting
  physics. Old fine job is reversibly paused under a supervisor and restored
  when the new GPU worker exits; no FEniCS work runs concurrently.


## 2026-10-13 — Clarify spatial basis provenance and preserve failed cold launch

- The fixed compact basis9b56ba9b is inherited as a spatial representation
  from earlier geometry plus neural-residual center selection, not strictly
  geometry-only. This run reuses those fixed centers/supports but no learned
  coefficients, weights, fitted history-dependent column scales or direct
  labels. RMS scales are recomputed from the fixed matrix. Correct the usage
  description; do not rebuild this representation mid-run or imply a fresh
  cold trajectory isolates loss norm alone. Relation: clarifies preceding entry.
- First cold launch reaches GPU preflight/one optimizer block but fails while
  serializing a NumPy boolean produced by a NumPy load scalar in the outer
  report. Supervisor restores the old worker. Retain failed_report_type root
  with its exact source/log/receipts; convert load scalars to Python float
  before forming diagnostics and restart from fresh zero under the original
  requested scheme. No numerical formulation or optimizer setting changes.


## 2026-10-13 — Save initially accepted stage; consolidation remains necessary

- User explicitly requests a commit of the current fracture2d stage and
  judges the result initially acceptable while still needing substantial
  improvement. Preserve numerical provisional/certification labels: user
  stage acceptance does not assert equation/outer convergence.
- Record that current experimental code requires substantial simplification,
  consolidation and organization. Prioritize shared stagger/history/restart/
  stopping/save flows, damage representations/loss/optimizer/sampling helpers,
  overlapping drivers/configs and concise documentation in later work.
  Preserve this reproducible snapshot before refactoring; no large cleanup
  or altered numerical method is performed in this commit.
- User narrows scope explicitly: do not commit slides. Stage only source,
  Markdown, configuration/lockfiles and textual JSON/CSV evidence; exclude
  slides and binary figures/PDF/weights. Keep original artifacts and hashes
  in manifests; image references use external saved archives.
- Commit locally on dev under current-turn authorization. No remote push
  or main merge requested. Relation: preserves completed own-zero L1 stage
  and earlier controls before further numerical and structural improvement.


## 2026-10-13 — Mechanics-first, unrelaxed stagger and field-change stopping

- User requests testing the standard fixed-load mechanics -> history ->
  warm-start damage -> field-change check loop, direct replacement of damage
  (omega1), and field-change rather than residual-based outer stopping.
  Reference: https://github.com/saugatsn/Phase-Field-Modeling/blob/main/Phase_Field_Simulation.py
  lines113..142 (solver_disp/solver_phi then L2 field changes). Borrow the
  loop structure only: their constitutive/history/BC/projection choices are
  not identical to this model and are not silently copied.
- Correct call-count premise: current L1 solver uses one initial mechanics
  call per load plus one per outer, not two per outer. Reordering at omega.5
  pipelines the same fixed-point map, apart from diagnostic timing; expect
  actual convergence changes mainly from omega1 and stopping accuracy.
- Create an isolated common-parent single-load control: own L1 physical
  .0052, exact same network/Adam moments/count786432, parameterization and
  next RNG state; target.0053. Compare direct omega.5/1 (up to2048) and
  L1 random Adam omega1/.5 (up to128x512); never advance on budget exit.
  The parent's physical path is provisional and remains so in all controls.
- Preserve original local P1 mechanics/nonlocal AT2/operator/material,
  H=max(committed preceding-load H,psi(q)), detached H/features/probabilities
  and warm persistent Adam state. No reference damage targets or correction.
- Use exact material P1 L2 field RMS: d tolerance1e-5, displacement RMS
  divided by current positive load tolerance1e-4, minimum3 sweeps and3
  consecutive passes. Also record reference-style looser1e-3 field checks
  without promoting them to exact-equilibrium certificates. All thresholds
  explicit. Residuals are diagnostics and never gate this control's outer
  stopping; finite/range/BC/mechanical validity still fail loudly.
- Each outer saves q at its mechanics-stage damage and d after damage update,
  explicitly marking the pair unsynchronized. At final/budget exit, perform
  one extra mechanics solve solely to synchronize final d/q/H/reaction and
  independently check physical fields. This is one per case, not per outer.
  Stop labels distinguish field_stable from equation-balanced/certified.
- Pause owned fine worker reversibly under a supervisor while this GPU trial
  runs, restore on exit/failure; protected live sources and committed stage
  artifacts unchanged. Focused tests guard pipeline parity, actual call count,
  P1 L2 quadrature and false field-stability under inexact inner solves.

## 2026-10-13 — Equal direct outer budget and reduced-inner PINN controls

- User authorizes unrelaxed PINN at the same644 outer updates taken by
  common-parent direct_full, and exploration of fewer inner training steps.
  Hold .0053 with the original own-L1 .0052 committed history floor. Run
  512-inner baseline by continuing its audited128 stage snapshot, and a
  128-inner control freshly from the identical .0052 physical/network/Adam
  parent. Both total644 outers; keep field checks diagnostic during this
  equal-count experiment rather than stopping one case early.
- Preserve L1, Adam1e-5, sigmoid, detached history/features/probabilities,
  core2048/halo32768/pool256 and resample1. Baseline retains moments/count
  851968 and RNG after256 new blocks; reduced-inner retains parent786432
  and its next RNG. No extra weighting/inner adaptation/load advancement.
- Restore the baseline's outer128 stage q/d (same d/net/Adam as synced
  endpoint), avoiding inserting its reporting-only endpoint mechanics solve
  into the continued fixed-point sequence. Never promote its .0053 H to a
  preceding-load history floor. Record actual reporting/preflight calls
  separately from the modeled one mechanics call per outer.
- Compare equal outer budgets and wall time; different inner-update totals
  are deliberate, not equal-work controls. Save native fields/network/Adam
  and exact sampling journals at129/136/192/256/384/512/644 (plus reduced
  control early checkpoints). Run GPU cases sequentially under owned fine
  pause/resume supervision; independent CPU audit checks each branch.

## 2026-10-13 — Full-domain damage training and optimizer screening

- User requests returning from random core+halo to all-node damage training,
  suspecting crack-path waviness, and delegates Adam versus L-BFGS choice.
  Same own-L1 .0052 physical/network parent, fixed.0053, mechanics-first
  omega1/frozen committed-floor H/warm-start damage; retain sigmoid/basis,
  node ordering and original local-mechanics/nonlocal-AT2 model.
- Preserve uniform nodal L1 objective and its2*RMS(H) normalization to isolate
  sampling where possible. Full domain means all71307 material nodes at
  EVERY training update; full original CSR support, no core sampling,
  importance-density weights, symmetry projection or straight-crack labels.
- Screen full Adam128 updates (parent moments/count,constant1e-5) and
  full L-BFGS128 iterations from exactly the same parent weights and first
  .0053 mechanics-generated frozen H. L-BFGS uses explicit memory20,
  maxls50,maxfun8192,gtol1e-12,ftol0; L1 is nonsmooth so successful empirical
  loss decrease is checked rather than assuming quasi-Newton superiority.
  Select the finite candidate with smaller exact full L1 loss; archive both
  candidates/status/time/metrics. Screening does not advance physical fields
  or consume the coupled initial state; selected coupled run starts the same
  parent again. A screening winner is not a universal optimizer guarantee.
- Coupled control uses128 inner updates/iterations per outer, up to644
  outer sweeps; existing exact P1 field criteria remain diagnostics for
  equal-count comparison, no early stopping or load advancement. Adam keeps
  moments/count; L-BFGS warm-starts weights but rebuilds curvature memory
  when fixed H changes. Record actual iterations/evaluations and failed
  line-search/budget statuses; no swallowed exception or silent fallback.
- Save stage/final q/d/H/network and applicable optimizer state, pin all
  source/config/checkpoints, independently replay native network/CSR/H/BC/
  P1/host FEM/loss/counts. Compare both equal-outer and matched crack-growth
  stages where available; a shorter crack can look straighter and lower
  residual alone does not establish an accurate physical endpoint.
- New driver/backend reuse existing preparation/objective/sweep/audit math;
  preserve old pinned modules/results and supervised original fine worker.
  Reversibly pause owned fine GPU job, restore exact identity on exit/failure.

## 2026-10-13 — Empirical full-domain L1 pilot selects Adam

- Same first .0053 H/parent weights,128 updates: full Adam L1.009146298,
  pointwiseAT2.02979498,time1.36864s; full L-BFGS L1.017523798,
  AT2.05676481,time3.25956s/133 evaluations. L-BFGS reports explicit128
  iteration-budget exit, not line-search failure. Both candidates independently
  replay. Select Adam under the previously declared minimum-full-L1 rule.
- Formal coupled run restarts original .0052 weights/physical state/Adam
  count786432; screening weights/moments do not enter it. This preserves a
  useful all-node-versus-random comparison at128 inner/644 outer counts,
  same L1/Adam1e-5/basis/sigmoid. No global superiority claim for Adam or
  attribution of crack waviness to sampling before morphology comparison.

## 2026-10-13 — Test detached damage-based L1 weighting

- User points out earlier L1/Adam omega.5 also produced bent cracks; relaxation
  is not a sufficient explanation. User now explicitly proposes weighted L1
  (e.g.1/(1-d+1e-3)) and allows other weights. This supersedes prior deferral
  of extra spatial weighting for the present declared experiment only.
- L16 emphasizes large residuals, whereas damage weighting emphasizes large d;
  they are different. Current all-node644 largest physical residual is in an
  already damaged node (.564649,.498),d.997555,raw proposed weight290.29.
  Raw weights span.9990..540.44,mean1.6232; large crack-wake weights can
  dominate, so compare exact proposal with raw cap50 as an explicit control.
- First isolate training at the audited all-node644 network/Adam868864 and
  its synchronized .0053 H, using all71307 nodes. Baseline uniform L1,
  proposed reciprocal with eps.001, capped reciprocal50;4096 Adam1e-5
  updates each from identical weights/moments/count. Use w=raw/mean(raw),
  detach weights and H/features. Freeze weights from source physical d
  throughout this fixed-H diagnostic (a declared reweighted subproblem).
  For a future coupled run weights would be recomputed from current damage
  at each outer and held fixed within its inner solve; not activated yet.
- Evaluate native outputs, uniform L1, weighted L1, full max residual,
  same-H unique direct nodal oracle (evaluation only), crack-region/global
  field error and damage morphology. Fixed-H fits do not promote q/d/H or
  advance load; no direct target enters loss and no straight/symmetry
  correction is imposed. Source model/previous-load history remains pinned.
- This is a weighting diagnostic, not evidence that L16 is decisive or a
  replacement of the constitutive/history formulation. Compare field error
  as well as loss: positive weights preserve exact residual-zero states,
  but finite-network/budget weighted minima and dynamics may differ.

## 2026-10-13 — Retain weighting as a diagnostic after reciprocal fits worsen accuracy

- All three declared4096-update fixed-H fits complete with independent native
  CPU replay. Initial same-H relative damageL2 4.4874%; uniform4.1598%,
  reciprocal40.2891%,cap50 33.3165%. Reciprocal weighted L1 decreases from
 .00351197 to.000547825 (84.4%), while unweighted residual and field errors
  rise and the inspected damage band broadens. This is an observed objective/
  accuracy tradeoff, not evidence that training failed to reduce its objective.
- Do not promote these diagnostic states or enable reciprocal weighting in
  coupled loading based on its weighted loss. Positive weights preserve exact
  zero-residual solutions, but do not guarantee finite-network/budget accuracy.
  L16 residual emphasis and high-d damage emphasis remain distinct.
- Same retained Adam moments, single source/frozen H and finite budget limit
  attribution; optimizer-transition and gentler bounded weights remain possible
  followups. Suggested1+4d^2 has raw range1..5, but is untested, not selected
  as a validated replacement. This does not reject all spatial weighting or
  identify L16/L-BFGS as the decisive cause of previous straight crack paths.

## 2026-10-14 — Recommend prioritizing a controlled L16 revisit

- User asks whether another test is worthwhile and whether to return to L16
  or pursue weighted L1. Recommend a short controlled L16 revisit first:
  prior all-node/L16/L-BFGS has useful morphology evidence, whereas declared
  reciprocal/cap50 damage weights worsened same-H field accuracy. This is
  priority setting, not proof of L16 superiority or rejection of gentler weights.
- Proposed frozen-H controls share physical/network parent,operator/head/
  normalization/history: L1/Adam baseline,L16/Adam to compare losses under
  Adam,L16/L-BFGS to compare optimizers under L16. Use common field-error/
  residual/time metrics; different objective values cannot rank candidates.
  Optimizer initialization/transition protocol must be explicit before launch.
- A favorable frozen-H result needs subsequent same-parent .0052->.0053
  mechanics-first coupled validation: fixed-H accuracy alone does not certify
  improved history/crack path/force evolution. Prefer .0052 restart to carrying
  the mature bent PINN history into a new physical trajectory. Gentler1+4d^2
  remains an optional small secondary test, not a long loading run first.
- No new implementation,training,solver interruption or Git operation is
  performed for this recommendation turn. Existing source/results protected.

## 2026-10-14 — Execute common-parent L1/L16 optimizer screen

- User explicitly authorizes the recommended three-way test. Use the own-L1
  .0052 parent network and archived first .0053 mechanics-generated H from the
  audited all-node screen, not mature bent .0053 history. Keep sigmoid,basis,
  features,AT2/operator/material and full71307 rows unchanged. No direct labels.
- Screen L1/Adam,L16/Adam,L16/L-BFGS from identical neural parameters. Adam
  starts zero moments/count in BOTH screen cases to isolate losses from inherited
  L1 moments; thereafter no resets. Constant1e-5,4096 updates; L-BFGS uses one
  continuous4096-iteration fit,maxfun65536,memory20,maxls50,gtol1e-12,ftol0.
  Report actual iterations/evaluations/time, not equal iteration costs.
- L16 reuses the old stabilized squared nodal L16 objective exactly,power16,
  floor1e-15,physical residual normalized by2RMS(H). Uniform L1 remains the
  previous mean absolute normalized residual. All contexts detached.
- Independent direct same-H volume-SPD solve is evaluation-only. Assess common
  field L2 error,unweighted MAE,max residual and inspected morphology/cost;
  do not rank different objectives by their numerical loss. Select an L16
  candidate for coupled validation only if its same-H fieldL2 AND max residual
  do not exceed the final L1/Adam baseline. If both qualify,choose smaller fieldL2.
- After independent screen replay, a qualifying candidate may restart original
  physical/network .0052 for current mechanics-first omega1 .0053 validation
  (128 inner,up to644 outer,unchanged field diagnostics/no load advancement).
  Screening weights are discarded; fresh Adam only at branch initialization,
  persistent thereafter,or warm network/fresh L-BFGS curvature per new H.
  This is a declared optimizer transition and current-route test, not bitwise
  reproduction of the old L16 loading history. Pause/restore owned original
  fine GPU solver under identity-safe supervision; pinned old modules unchanged.

## 2026-10-14 — Three-way result does not qualify L16 coupled selection

- At identical first .0053 H/own-L1 .0052 parameters,initial same-H fieldL2
 6.56785%. Final4096 L1/Adam .448677%/AT2 .00784379/45.608s;
  L16/Adam44.8637%/.0284471/45.847s with late oscillation; continuous
  L16/L-BFGS3.00209%/.00114290/105.896s,4096iterations/4711evaluations,
  explicit iteration-budget exit (not line-search failure or convergence).
- All9 saved states and eligibility independently replay. L16/L-BFGS lowers
  peak imbalance about6.9-fold versus L1/Adam but has about6.7-fold higher
  fieldL2 and2.3-fold cost. Do not equate its better peak metric with overall
  accuracy or infer that L16 alone explains prior cleaner coupled cracks.
- Neither L16 meets the predeclared joint fieldL2/max-residual criterion;
  selection=None. No selected coupled phase or new loading is launched.
  Do not overwrite the original trajectory with a diagnostic state. This
  stage is before substantial crack extension,so no universal later-H ranking
  or final crack-path/coupled convergence conclusion is supported.
- Retain both objectives as tools for different error emphases; later-H tests
  or a declared combined mean/peak objective are possible next steps,unrun.
  Original fine worker identity safely restored; current files/results pinned.

## 2026-10-14 — User prioritizes crack straightness over global field L2

- User clarifies that current concern is a straight crack path despite larger
  global L2 for L16/L-BFGS. The previous joint L2/peak screen is an agent-chosen
  criterion,not a physical requirement,and does not evaluate long-crack shape
  at the first frozenH. It must not veto the authorized coupled experiment.
- Resume L16/full-node/L-BFGS coupled .0052->fixed.0053 from the same own-L1
  physical/network parent as the completed all-node L1 control,discarding screen
  weights. Current mechanics-first omega1,128 inner iterations,644 outer budget,
  warm network/fresh curvature per newH,one mechanics/outer plus final sync.
  Preserve original operator/material/sigmoid/features/committed .0052 H floor;
  do not import bent mature .0053 history or add a symmetry/straightness loss.
- Primary evaluation: compare clouds and transverse peak paths at overlapping
  active crack lengths; record lateral drift/variation and local band width.
  Global L2,residual,force and field changes remain physical diagnostics and
  cannot be erased to present a straight-looking unbalanced state as converged.
  No budget-based advancement beyond.0053 and no screen-state promotion.
- This steering overrides only eligibility priority,not the archived numerical
  screen result (selection=None remains preserved). Common-parent coupled
  comparison tests optimizer/loss combination; it is not isolated L16 causality
  or exact reproduction of older relaxed loading. Preserve pinned old files;
  serialize GPU execution under identity-safe pause/resume supervision.

## 2026-10-14 — Retain L1/Adam/all-node and distinguish path drift from roughness

- User selects L1/Adam/all-node as the current main route and asks how to
  obtain a straighter,smoother damage crack. Do not infer that another output
  head,power norm or sampling change alone resolves this. No new training is
  launched for this advice turn;the previously supervised L16 diagnostic has
  its original644 budget unless user explicitly cancels it.
- Independently compare all-node644 physical d,current-H NN image and unique
  direct damage oracle at EXACTLY its currentH. P1 transverse maxima .52..85
  are almost identical:physical/NN y.496..5075,own-H direct y.496..5070.
  At this checkpoint path drift is present in the history-driven subproblem,
  not merely final NN output error. This does not prove when it arose or
  permanent within-load history locking:currentH=max(previous-load floor,
  current mechanics energy),not maximum of every prior outer iterate.
- Actual mesh is not exactly reflection-paired about y=.5:nearest mirrored
  node maxdistance.00240662,mean.0000236418,exact fraction18.7976%. Direct
  final peak also has mesh-scale deviations. Separate macroscopic drift from
  P1/mesh/basis roughness;measure shared developed intervals and widths.
- Suggested sequence,not implemented:find first path/history asymmetry in
  saved stages,restart before it and tighten/adapt L1 Adam inner accuracy at
  the moving front. A benchmark-specific weak reflection constraint is a
  candidate after verifying geometry/loading/history/discrete symmetry;it
  restricts asymmetric solutions and does not guarantee a single straight
  crack (paired branches can remain). Do not force a predetermined line.
- Current ansatz includes MLP plus compact radial hat readout. Smoother local
  basis functions and reflection-paired centers are representation controls,
  not changes to D or material length;their benefit remains untested. MLP
  smoothness alone does not remove hat-basis corners or history-feature noise.
- A general alternative retaining L1 is a carefully normalized auxiliary
  SAME-OPERATOR frozen-H AT2 energy. Checked on actual matrix:
  Phi=sum V*(.5*g0*d^2+H*(1-d)^2)+.5*g1*d^T*V*D*d,
  grad Phi=V*((g0+2H)*d+g1*D*d-2H). Volume-symmetry maxdefect1.11e-16;
  independently assembled symmetric-quadratic gradient relative discrepancy
 1.55e-13. This is the fixed-H damage functional,not a claim that the full
  hybrid/history mechanics has a joint variational objective. Auxiliary
  energy preserves the exact fixed-H nodal optimum but finite NN tradeoffs
  need testing;no guaranteed straightness or arbitrary smoothing penalty.
- Primary reference for neural variational fracture approach:
  Goswami et al.,https://arxiv.org/abs/1907.02531 (method inspiration only;
  no assertion their formulation is our nonlocal operator or proves its gain).
  Numerical evidence archived in full_damage_trial_20261013/independent_review/
  morphology_advice_20261014.json. Main route updated in memory;old sources,
  results and previous comparisons remain protected,Git/slides unchanged.

## 2026-10-14 — No symmetry constraints; investigate precritical load history

- User explicitly rejects soft/hard symmetry constraints and predetermined
  straight-crack targets. This supersedes the optional benchmark reflection
  suggestion above. Keep L1/Adam/all-node,existing sigmoid/architecture,
  local mechanics/nonlocal damage and mechanics-first unrelaxed stagger.
  Do not introduce basis changes or auxiliary regularizers for this hypothesis.
- Investigate whether coarse du5e-4 before .005 creates early training/history
  bias that later coupling amplifies. Existing finer-load experiment starts
  only after .0052 and cannot rule out this different hypothesis. Older
  straighter sigmoid branch inherited a certified du1e-4 prefix; fresh L1
  coarse prefix uses random sampling and budget-exhausted provisional states.
  At .0052 matched-reference damageL2 is6.9831% versus old1.4283%,with
  multiple confounders; at .0050 fixed-own-H neural error is6.6520%.
- Recommended causal control: two cold L1/Adam/all-node runs with identical
  initialization,head,features,scales,Adam policy,inner/outer stopping and
  budgets; only pre-.005 load step differs (5e-4 vs1e-4). Thereafter both
  use1e-4 and compare .0050/.0052/.0053. Old randomized prefix cannot serve
  as the sole coarse control for a pure step-size claim. Finer loading also
  adds training work; distinguish this with cost reporting and,if improvement
  appears,a matched-total-update coarse control before causal attribution.
- Smaller steps are a testable warm-start/continuation hypothesis,not a
  guaranteed cure or proof every crooked path is training error. Reference
  fields and direct same-H solves remain evaluation-only. Do not restart
  from the existing .0052 floor when testing pre-.005 history. No new GPU
  training or numerical source changes in this investigation turn.

## 2026-10-14 — Launch paired cold all-node loading in the background

- User authorizes the proposed coarse/fine control and background execution,
  requesting a completion estimate. Use shared cold network/Adam initialization,
  L1/Adam1e-5/all71307 nodes/sigmoid/current physical features and latent scales.
  Only case identity and pre-.005 increment differ:5e-4 versus1e-4; thereafter
  both1e-4 to.0053. No symmetry/straight-path constraints or oracle labels.
- Declare128 Adam updates per frozen-H inner solve,128 outer cap at prefix
  loads and644 at.0053. Mechanics-first omega1;exact P1 field-change stopping
  at dRMS1e-5,uRMS/load1e-4 for three passes/minimum three sweeps. Residual
  remains diagnostic. Endpoints have one mechanics sync; budget continuation
  stays provisional. Existing sampled coarse run is not the matched control.
- Serialize the13-load coarse and53-load fine branches on one GPU under a
  detached supervisor,temporarily pausing identity-checked original fine18811
  and restoring it on exit. A separate CPU auditor checks saved native network,
  full CSR loss/history/BC/force/P1/count and matched-reference/same-H errors.
  Preserve old pinned numerical modules and run receipts. No Git operation.
- Full caps total9480 sweeps/1213440 Adam updates. Earlier all-node644 cost
 1083.59s suggests roughly4.4h total before early exits and startup/audit
  variance. Initial user estimate3..5h;calibrate after actual new warm sweeps.
  Do not infer causality from finer-step improvement without accounting for
  added training work; a cost-matched control remains a followup,not launched.

## 2026-10-14 — Completed cold coarse/fine control improves drift, not full opening

- Both all-node L1/Adam cold paths finish.0053 and independently replay;
  identical configuration apart from case/precritical increment and identical
  serialized network/initial optimizer hashes confirmed. All worker/auditor/
  supervisor exits0. Total elapsed~2h57min,finishing04:23 Asia/Shanghai Oct14.
- At.0052,matched-reference damageL2 coarse.609786% versus fine.409928%;
  ownH neural errors.478757%/.380929%. At.0053 after644 same-load outers,
  coupled errors55.045639%/45.333508%,R266.52083/231.02961 versus matched
  direct1.57096;ownH errors4.739783%/4.350269%. Neither field is stable or
  through-cracked. Completing a diagnostic budget is not full convergence.
- Inspected P1 clouds at.0050/.0052/.0053 and transverse profiles at.0053.
  Compare the31shared active(d_peak>=.5) cuts x.52..82:peak-position RMS
  drift .01172879 coarse/.00583095 fine,maximum.0175/.0100;reference
  RMS.00114652/max.0020. Mean contiguous half-height width.04070968/
 .03911290 versus reference.0395. Fine path develops to x.85 at this
  profile threshold,coarse.82;d>=.9 tip-connected fronts .824456/.796743.
  Finer prefix reduces this measured drift but does not remove it. The
  profile is an evaluation diagnostic,never a symmetry or straight-line loss.
- Actual total Adam updates coarse213760/fine466432;pre-.0053 counts
 131328/384000. Initial loss/head/sampling/physics confounders are controlled,
  but added training work remains part of the finer-loading benefit. Do not
  declare early step size the sole cause of path bending or claim a permanent
  material critical-load shift. Retain finer prefix as promising and prioritize
  critical-load inner/coupled accuracy;cost-matched prefix control unrun.
- Review artifacts and source-field hashes:full_damage_load_steps_20261014/
  comparison_review/{comparison.json,damage_comparison.png,reaction_and_paths.png}.
  No new numerical implementation,training,Git history or slides change during
  this status review;current source and checkpoint ancestry remain preserved.

## 2026-10-14 — Best-current full PINN loading uses finer prefix and longer coupling

- User authorizes a complete best-current PINN trajectory and asks a short
  configuration description. Follow-up explicitly sets terminal displacement
 .0065 instead of.0075. Inherit the already audited own-zero du1e-4/all-node
  fine prefix through.0052,then rerun.0053 and advance by1e-4 through.0065.
  Do not reuse the unfinished/bent.0053 endpoint as a committed history floor.
- Preserve local P1 mechanics/nonlocal AT2,MLP96x3+compact readout,sigmoid,
  physical H features/latent scales,FP64,L1 normalized by2RMS(H),Adam1e-5,
  full71307 nodes,unrelaxed mechanics-first stagger and persistent weights/
  Adam384000 state. No symmetry constraints,straight targets,residual weights,
  randomized cores,new output head or optimizer transition.
- Keep128 inner updates;raise maximum outer sweeps per load to4096 with the
  same P1 field-change early stop(d1e-5,u/load1e-4,3passes/min3). Retain
  force/residual/history/healing/field/path diagnostics independently. Field
  stability and budget completion are not physical certification;any capped
  load remains explicitly provisional in the complete curve. Longer coupling
  is the new test,not a guaranteed cure or a claim4096 is already validated.
- Preserve existing pinned source/runs;add independent restore/loading driver
  and CPU native auditor. Serialize one GPU in a detached supervised run,
  verify actual restore/preflight and first saved stage before reporting launch.
  Shared prefix avoids repeating the approximately100-minute cold solve.

## 2026-10-14 — User caps outer sweeps at1024 and requests TensorBoard monitoring

- User explicitly requests stopping4096 and restarting with1024 outer cap;
  follow-up adds TensorBoard scalar/field/reaction-curve monitoring like
  fracture1d/train.py. Preserve inner128,du1e-4,L1/Adam/all-node,sigmoid,
  fine source0052/Adam384000,standard unrelaxed stagger andterminal.0065.
  No symmetry constraints or changes to physical equations/output.
- Identity-checked4096 supervisor/GPU/audit6147/6153/6152 stopped at saved
  .0053outer314. Originalrun preserved withuser_stop_request.json andSIGTERM
  exit receipt. This is user cancellation,not a numerical failure. New run
  uses a fresh output/config;restore0052 rather than keep training314 state.
- Reuse unchanged numeric driver;add a separate CPU TensorBoard monitor using
  tensorboardX SummaryWriter/add_figure (same writer asfracture1d). Read
  immutable endpoint fields andjournals;each completedload writesdamage,
  ux,uy,H andcomplete-so-far load/reaction images,with matching-load reference.
  Outer loss/residual/change scalars use optimizer-count-derived monotone
  steps;endpoint images use load index. Do not plot an unsynchronized outer
  reaction as an equilibrated new-damage endpoint or alteroptimizer state.
- Explicit monitoring cadence/settings andprovenance willbe saved withrun.
  Useexisting TensorBoard service/logroot if compatible;verify actual event
  files/tags/images andHTTP discovery after launching1024 run inbackground.

## 2026-10-14 — Canonical TensorBoard root and process-name separation

- User requires all logs under/root/autodl-tmp/tf-logs and has restarted the
  port6007server with thatlogdir. Actual eventfiles already livedthere;the
  previous/root/tf-logs path wasonlyan agent-createdsymlink foroldserver.
  Removeonlythatverifiedsymlink andemptyaliasdirectories;preserveeventfiles.
  Removealiassetting/creation andvalidateeventdir insideexplicitlog_root.
- Readback detectsoldGPUrun stoppedat0053outer196:theuser's server-reset
  commandgreppedtensorboard andkilledthelogger whoseargv contained
  tensorboard_monitor;supervisor thenstoppedGPU/audit asconfigured. Treat
  thisasexternal monitoringinterruption,notnumerical failure. Communicated
  explicitly. Preservefailedroot/eventsandrestoreverifiedfine0052 again
  innew1024_tbroot run tocontinueauthorizedterminal0065 workload.
- Startlogger viafield_monitor withbest_full_damage_fields.py (neitherargv
  contains'tensorboard'). TensorBoard library/moduleinternals remainreused,
  butserver restartbyprocessname no longeralso matches thefieldlogger.
  Numericdriver,1024outer/128inner,model,sampling/Adam andsourcepins stay
  unchanged. Newlogs/events/plots allunderautodl-tmp; no/root/tf-logs alias.

## 2026-10-14 — Local source checkpoint while best-current loading continues

- Decision: the user explicitly authorizes committing the current modifications
  on dev. Include fracture2d Python drivers/configurations, focused tests and
  Markdown documentation/memory; exclude slides, binary figures/PDF/weights
  and external numerical/event artifacts, consistent with prior user scope.
- Reason: preserve the explored standard stagger, full-node L1/Adam route,
  controlled comparison branches and canonical-root TensorBoard monitoring as
  a reviewable stage. This checkpoint does not certify the still-running path.
- Rejected alternatives: committing all untracked files, reorganizing numerical
  modules during a pinned live run, or merging/pushing without authorization.
- Relation: extends db4dddc. Substantial numerical improvement and source
  simplification/consolidation remain required; the active broad task stays open.
