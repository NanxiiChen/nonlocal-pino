# Fracture2D Engineering-Grip Direct Baseline

## Goal

Build a lean, trustworthy direct nonlocal reference focused on the rounded-notch
crack-region operator and solution. Treat top/bottom loading as an engineering
volume-grip approximation; do not require detailed boundary-field or reaction
agreement with the local FEM model.

## Acceptance Criteria

- Preserve the original material cloud, notch visibility, material `SparseD`
  damage operator, and crack-region BA supports exactly.
- Add external top/bottom collars only for mechanics displacement transfer.
- Prescribe top/bottom collar vertical displacement by exact hard constraints;
  leave lateral displacement free except one horizontal anchor.
- Keep grips intact and absent from damage/history equations.
- Use strict finite, mechanics residual, damage residual, range, healing,
  history, force, and moment gates; no warning acceptance or clipping.
- Validate energy-gradient/tangent/reaction algebra on manufactured clouds, but
  treat FEM reaction and near-grip fields as diagnostics rather than model gates.
- Archive material displacement, damage, history, crack-region strain/Amor
  driver, and enough static operator data for independent crack-region replay.
- Before a long curve, pass a strict rounded-slot smoke checkpoint and verify the
  maximum material damage/history remains in the notch region rather than at a
  grip interface.

## Formulation Decision

Use the existing Candidate A2 union geometry and phase-measured conservative
mechanics energy. The material cloud is unchanged. External collar-only
particles with `y<0` form the bottom grip and particles with `y>1` form the top
grip; shared material-interface particles at `y=0,1` remain free.

Hard constraints are

$$
u_y=0\quad\text{on the bottom grip},
$$

$$
u_y=\bar u\quad\text{on the top grip},
$$

plus one `u_x=0` anchor at the original material particle `(0,0)`. All other
`u_x` components are free. The continuation coordinate is the prescribed grip
displacement `u_bar`; physical material-interface opening is archived only as a
diagnostic.

The mechanics center weight remains

$$
W_i(d)=V_i^m g(d_i)+V_i^g g(0),
$$

with union-volume BA kinematics. Damage/history remain exactly material-only.
No opening-slope scaling or `R_eff` is used. The loaded response is the summed
top-grip constrained residual

$$
R_T=\sum_{i\in\mathcal C_T}r_{i,y}.
$$

This is an explicit engineering boundary approximation, not a claim of
pointwise equivalence to the local FEM top boundary.

## Status

Paused by the user's 2026-09-24 switch to staggered adaptive load-step research:
the engineering-grip diagnostic task retains the completed strict
checkpoints and immutable historical artifacts. The approved isolated CPU
`monolithic` strategy completed a physical smoke and a strict fresh-zero attempt
toward `u_bar=0.0075`; the latter committed through `u_bar=0.0061` and terminated
at its configured line-search/cutback limits. Canonical scratch reconstruction
now passes `0.00611`, `0.006115`, and `0.006116`, but these states are not wired
through verified-attempt, continuation, workflow, or archive contracts. Detailed
`0.006117` diagnostics supersede the earlier avalanche-barrier interpretation.
The authorized exact history elimination is now independently manufactured-
verified as an isolated reduced `q-d` residual/JVP contract; it has no reduced
solver, metric, preconditioner, or production dispatch.
The archive now uses an explicit
`accepted_mechanics_energy` field and `attempt_scalar_evidence` metadata,
rejects ZIP64 content under the practical ZIP32 budget, hardens Linux
no-clobber cleanup after directory-fsync failure, and authenticates accepted
`psi_after/H_next` and canonical equations/gates through owner replay without
persisting redundant crack/operator arrays. Focused archive verification is
`17/17`; the full monolithic CPU command is `82/82`; the strict ZIP32 EOCD
parser requires one zero-comment single-disk record ending at EOF, exact
allowlist counts, bounded central size/offset, and exact parsed-record/location
agreement; a structurally positioned ZIP64 locator and duplicate/trailing EOCD
records are rejected without payload signature scanning. The monolithic
workflow publishes one private-staged
`.monolithic` directory with four members using Linux
`renameat2(RENAME_NOREPLACE)`, fsyncs every JSON/staging/parent directory, and
retains the published bundle when parent durability is uncertain. The focused
workflow verification is `52/52`; legacy engineering-grip/JAX verification
passes `63` tests with one expected GPU-only skip; stabilized verification
passes `82/82`; the monolithic/archive command passes `82/82`. Compileall and
`git diff --check` pass. Primary Sol review is complete with all six
archive/workflow findings resolved.
The fixed-mu
`anderson_residual_regularized_backtracking` sensitivity study is complete as
negative scalar-only evidence: all three injected-state attempts rejected the
same target after the full 144-sweep budget, with no accepted state or
production continuation. The user-approved smaller-`du_min` checkpoint and
deeper-alpha audit are also complete as negative evidence: the checkpoint
strictly failed and rolled back, while the deeper-alpha evidence is NO-GO.
The prior implementation task is archived; this engineering-grip task remains
the single active memory task.

The second manufactured-only arc layer is implemented in the isolated
`nonlocal_engineering_grip_monolithic_arc.py` module. It binds one exact
reduced `q-d` owner/state/mask/config, derives the fracture-energy row scale
from the strict damage correction semantics plus the documented float64
constituent bound, evaluates changed-load trials with frozen equilibrium
scales, and includes constrained load-direction DOFs in its rectangular
metric. The old bordered map is retained for diagnostic evidence only; the
arc solver uses the immutable parameter-free metric-scaled block-diagonal
right preconditioner. A private signed-load owner path is used only for arc
trials; the public fixed-load constructor remains unchanged. The focused arc
suite is manufactured-only and does not add a verified attempt, continuation,
workflow, archive, or physical run.

The current workflow review corrections are tracked here: the shared
`max_stagger_iterations` field must equal the documented `solver_base.py`
compatibility value `100` and is never a monolithic iteration budget;
post-write replay failures must clean only the private staging directory;
publication uses no-clobber directory rename and reports parent-fsync
durability uncertainty without deleting a published bundle; and
`field_equations_validated` is limited to retained accepted-step canonical
fields owner-validated independently of completion status.

## Blocker

The exact `u_bar=0.0075` objective remains unmet. Stagger committed through
`0.00623125`; the durable monolithic archive committed through `0.0061`.
Canonical scratch states reach `0.006116`. At `0.006117`, the active mask settles
but the explicit history unknown enters a trust-region two-cycle. Exact history
elimination now removes that redundant unknown algebraically. The manufactured
reduced and arc solvers remain isolated, with no verified-attempt layer,
continuation dispatch, durable workflow, or physical execution. Manufactured
correctness does not establish that the reduced nonlinear problem converges or
reaches a farther physical state.

## Next Action

Keep every completed/failure archive and prior diagnostic unchanged. Do not
extend Armijo again, launch a broad parameter study, warning-accept, or treat
rejected fields as a physical state. The reduced and arc solvers remain
manufactured-only; any verified attempt, strict continuation, workflow/archive
integration, or physical continuation needs separate review and authorization.

## 2026-09-24 — Complete the second manufactured arc foundation

- Status: complete for the authorized isolated foundation scope. Added the
  frozen augmented residual/JVP, committed-energy target and strict-derived
  `LG` evidence, full-union rectangular metric, and bordered right
  preconditioner. Added independent manufactured tests for mixed active masks,
  changed-load trials, energy rows, metric rank/pure-load action, dense
  bordered action, stale/foreign forgery, and immutable explicit signatures.
- Scope guard: no outer nonlinear solver, verified attempt, continuation
  controller, archive, workflow dispatch, or physical run was added.
- Verification: the focused arc/reduced/monolithic battery passes `72/72` in
  the repository `.venv`; the changed modules/test compile with `py_compile`,
  and tracked plus new-file `git diff --check` validation passes. A forced
  `JAX_PLATFORMS=cpu` rerun exposed one pre-existing reduced exact-array
  two-ulp comparison difference; the default repository environment passes
  the same `72/72` battery.
- Blocker: primary Sol review remains required before selecting any arc controls
  or physical continuation semantics.
- Next action: review the frozen-owner identities, energy constituent
  roundoff-bound derivation, constrained full-displacement metric rows, and
  bordered denominator tolerance. Do not run a physical arc continuation.

## Frozen Smoke Contract

Collar-only masks are exactly nonmaterial particles with `y<0` and `y>1`; every
nonmaterial particle belongs to one mask, material/interface particles belong to
neither. All collar `uy` DOFs and only the material anchor `ux` are constrained.

The strict stagger retains the existing committed-history ordering and material
AT2 residuals. It compares direct top-grip reaction before/after and both union
mechanics residual/conservation states. A complete failed sweep may update only
the local damage iterate; incomplete failures roll back without local update.

Smoke values are `delta=0.048`, `h_g=0.02`, `u_start=0`, `u_final=1e-4`,
`du_initial=1e-4`, `du_min=5e-5`, `cutback_factor=0.5`, `max_cutbacks=8`,
`max_stagger_iterations=20`, and mechanics CG `rtol=1e-10`, `atol=1e-13`,
`maxiter=3000`. Standard gates are `(atol=1e-9,rtol=1e-8,scale_floor=1)`;
damage residual uses `(1e-10,1e-8,1)`; range/healing tolerances are `1e-10`.

Before continuation, exact-zero damage at `u_bar=1e-4` must pass one centered
reduced-energy/top-grip reaction check with `epsilon=1e-5`, relative limit
`1e-7`, and the same tightened mechanics tolerance. This check is not repeated
inside stagger sweeps.

The crack region is the material set within one `delta` of the unique tip
particle `(0.5,0.5)`. Smoke completion requires finite crack-region fields and
material damage/history maxima no larger outside that region than inside plus
`1e-10`. Material interface maxima and physical interface opening are archived
as diagnostics; FEM reaction and near-grip errors are not acceptance gates.

## 2026-08-26 — Diagnose smoke archive replay blocker

- The authorized rounded-slot smoke continuation and archive writer succeeded for
  `/tmp/opencode/fracture2d_engineering_grip_strict_1e4.npz`: two states
  (`u_bar=0` and `u_bar=1e-4`), one accepted step, zero cutbacks, and three
  complete sweeps. The archive code hash matches the current module hash.
- Independent replay fails first at state index `0`, the exact initial zero
  state. Its loop-order mechanics reconstruction gives
  `free_residual_norm=1.0514815413942755e-11`,
  `free_rhs_norm=1.0514815413942755e-11`, and
  `cg_convergence_limit=1e-13`; the computed relative norm is exactly `1`.
  The exact mathematical zero-load residual/RHS are zero. The observed value
  comes from `F=K K^{-1}` identity roundoff: replay has maximum `|F-I|`
  `1.6107018615023199e-15` and edge-strain magnitude
  `1.1102230246251565e-15`. The production vectorized path also has a
  zero-state free norm `5.121762785804757e-12`, so this is not a physical
  non-equilibrium or a replay-only phase/constraint error.
- The accepted state index `1` independently passes: replay gives
  `free_residual_norm=6.8128546030427255e-09`,
  `free_rhs_norm=70.742933779381588`, and
  `cg_convergence_limit=7.0742933779381594e-09`. The archived production
  after-sweep values are `6.8129677713986885e-09` and inferred RHS
  `70.742933779380877`, with limit `7.0742933779380883e-09`.
- Static BA data reconstructed bitwise, phase-center weights have zero defect,
  and the canonical volume-grip masks/DOFs are intact (153 top collar
  particles, 153 bottom collar particles, 307 constrained DOFs, 7095 free
  DOFs). State/step pairing is valid: the final state equals accepted
  `q_after`, `d_next`, and `H_next`; the attempt digests pair with the initial
  and final states. `q_before` and final-sweep `d_iter` are target-trial
  primitives and are not expected to equal the prior committed zero state.
- A replay-only fix must first exempt the exact initial committed state from
  the solved-mechanics gate; it has no production mechanics result. A second
  replay contract issue remains: production-versus-independent free-residual
  norm defects are `3.493047770578776e-15` (sweep 0 before),
  `3.1830686624979701e-13` (sweep 0 after), and about
  `1.13e-13` thereafter, while the current recorded-field comparator allows
  only `1e-15` absolute. Do not weaken the production equilibrium gate or
  approve/extend this archive until an independent replay tolerance or
  matched-operation contract is derived and reviewed.

### Exact resume point

Keep the engineering-grip formulation and production gates unchanged. Resolve
the two replay-contract defects above, add a rounded-cloud regression for the
exact zero state and operation-order residual comparison, then rerun the
archive audit before any longer continuation.

## 2026-08-26 — Complete strict replay forgery coverage

- The replay contract now validates canonical work-check settings, duplicate
  and reciprocal graph structure, positive material outgoing families, the
  candidate/visible/removed edge partition, static metadata/digests, archived
  production CG status, exact attempt scheduling, and stage-specific failure
  primitives. The production equations and strict acceptance gates remain
  unchanged.
- Derived top-collar spacing and row ordinates use the explicit float64
  roundoff bound based on coordinate, interface, and offset magnitudes. Other
  serialized coordinate, index, mask, digest, and pairing fields remain exact.
- Direct engineering-grip archive tests cover work-config, graph, metadata,
  CG-status, constituent-scale, schedule, and incomplete-stage forgeries. The
  focused suite passes `18` tests; the compact baseline/mechanics/geometry/grip
  suites pass `42` tests. The combined A2/continuation command exceeded its
  `600 s` timeout and is not part of the engineering-grip acceptance result.

### Exact resume point

Keep the engineering-grip smoke archive unapproved pending primary review of
the replay implementation and operation-order tolerances. No longer endpoint,
production horizon, or rounded smoke extension is authorized.

## 2026-08-26 — Implement scalable independent archive replay

- The independent replay now evaluates mechanics response and tangent actions
  with padded fixed-shape BA supports, NumPy tensor contractions, and indexed
  reductions. It retains the independently rebuilt union graph and does not
  call the production mechanics evaluator or solver.
- Replay direct mechanics uses a `scipy.sparse.linalg.LinearOperator`, an
  independently assembled particle-block Jacobi preconditioner, and explicit
  CG `rtol`, `atol`, and `maxiter` arguments. Nonzero CG status, non-finite
  values, and a failed explicit free-residual check raise loudly. The returned
  `tangent` field is the matrix-free operator, never a dense global array.
- Replay material damage now assembles the independently reconstructed
  directed graph as CSR with `coo_matrix`/`diags` and solves the
  volume-weighted AT2 system with sparse `spsolve`. It checks graph validity,
  finite data, volume self-adjointness, and sparse-solve failure; no dense
  `N x N` matrix or `np.linalg.solve` fallback remains.
- Exact zero displacement is handled analytically in replay response and Amor
  reconstruction, preventing rounded `F=K K^{-1}` identity defects from being
  treated as physical residuals. The production gates and cancellation-safe
  replay comparators are unchanged.
- Focused engineering-grip coverage now includes exact vectorized-vs-production
  mechanics primitives, sparse-vs-production damage equivalence, an anti-dense
  replay contract, and exact-zero analytic replay. The focused suite passes
  `16` tests.
- A current-code replay audit of the existing two-state rounded-slot archive
  passed after changing only a temporary copy's metadata code hash; the
  repository archive was not modified. It covers `3701` union particles,
  `79060` visible directed edges, and `7095` free mechanics DOFs in
  `213.86 s` including three centered work-check solves. One current replay
  direct solve took `64.80 s` and `280` CG iterations; this is a large
  improvement over the prior dense multi-hour path but remains a practical
  runtime risk for larger archives.

### Exact resume point

Keep the formulation, production solver, strict continuation gates, and archive
schema unchanged. Obtain independent review of the vectorized BA contraction,
particle-block preconditioner, sparse damage assembly, and operation-order
tolerances before authorizing any longer engineering-grip continuation.

## 2026-08-26 — Strict scalable-replay audit findings

- Review status is **NO-GO** for a longer continuation. No source or test files
  were changed. The current formulation, production solver, and production
  gates remain unchanged.
- The declared mechanics algebra is reproduced correctly on the manufactured
  cloud: replay tangent finite differences, exact replay block diagonals,
  constrained spectrum, translation/rotation null modes, and sparse AT2
  volume symmetry all pass. The focused engineering suite passes `16` tests;
  separate baseline, mechanics/algebra, and grip-diagnostic suites pass `7`,
  `13`, and `11` tests. The supplied large archive evidence remains the
  `213.86 s` temporary hash-adjusted replay over `3701` particles and `79060`
  directed edges.
- P1: work-check replay uses archived `work_check_u_bar`, `work_check_epsilon`,
  and `work_check_relative_limit` at lines `6174-6255` without requiring them
  to equal the canonical values in `config`. A forged archive with config work
  limit `0` and archive work limit `1` was accepted, despite the recomputed
  work error being nonzero.
- P1: graph/provenance integrity is incomplete. `_replay_ba` checks reciprocal
  and sorted edges but not duplicate directed edges (`3479-3487`), and
  `_replay_damage` does not require every material node to have a positive
  outgoing family as `ExplicitVolumeSparseD` does (`4124-4167`). A sorted
  duplicate union edge is accepted by `_replay_ba`; a material graph with one
  reciprocal edge and isolated nodes is accepted by `_replay_damage`. The
  archive replay also does not cross-check candidate/visible/removed edge
  partitioning (`5782-5838`); changing candidate edges and recomputing the
  self-reported union hash was accepted.
- P2: replayed mechanics gates do not retain or verify production `cg_info`.
  Production `_mechanics_gate` requires `result.cg_info == 0` (`1524-1528`),
  while `_replay_mechanics_at_state` derives `mechanics_pass` only from finite
  values and the residual inequality (`3978-3989`). The independent replay
  therefore verifies the stored state residual, but not this production solver
  status primitive.
- P2: operation-order tolerance for free residual norms uses
  `residual_reduction_scale`, not the identity-cancellation-aware
  `residual_constituent_scale` (`3942-3953`). This is safe for the current
  nonzero smoke state but can underbound replay-vs-production defects for a
  sufficiently small nonzero load.
- P2: metadata fields written as static provenance are not validated or used:
  `static_digest`, `visibility_rule`, `h_g`, and `visibility_tolerance` are
  accepted without canonical/range checks (`3283-3298`, `4781-4799`). Changing
  `static_digest`, `visibility_rule`, or these scalar declarations was accepted
  by the archive replay.
- P2: rejected-attempt replay checks only `target > start` and sequential
  cutback counters (`6345-6359`); it does not reconstruct `_next_target`, the
  configured increment/cutback schedule, or the maximum cutback bound. Incomplete
  failure stages also do not consistently require the stage-specific failure
  reason or prior damage range/healing gates (`5427-5497`).

### Exact resume point

Keep the current code and strict smoke archive unapproved. Before any longer
continuation, independently decide whether the P1/P2 replay-contract findings
are in scope, then repair or explicitly narrow the archive contract, add
malformed-graph/work-config/metadata/attempt-schedule regressions, and rerun
only the authorized `delta=0.048`, `h_g=0.02`, `u_bar=1e-4` smoke audit.

## 2026-08-27 — Revalidate the v2 smoke and replay contract

- The current `.venv` contains JAX `0.10.1`, SciPy, and a visible `cuda:0`
  device. The exact authorized rounded-slot smoke was rerun with
  `delta=0.048`, `h_g=0.02`, `u_bar=1e-4`, tightened mechanics CG
  `1e-10/1e-13`, and the restricted CPU-replayed JAX adapter. It accepted one
  step, used three complete sweeps, and reached exact `u_bar=1e-4`.
- The fresh archive
  `/tmp/opencode/fracture2d_engineering_grip_v2_smoke.npz` uses
  `fracture2d-engineering-grip-v2`, includes production before/after
  `cg_info=[0]/[0]`, has eight source-bundle files, and contains no object
  arrays. Independent replay passed all static, state, sweep, work, mechanics,
  conservation, damage, history, and crack-localization checks.
- Smoke values were finite and strict: final maximum material damage was
  `1.2254719686451387e-4`, final maximum history was `1.4428893342271215e-2`,
  and the one-time work-check relative error was
  `1.2992513794712763e-12` against the `1e-7` limit. The complete run plus
  independent replay took `258.7294680662453 s` on `cuda:0`.
- The final crack-region maxima were damage `1.2254719686451387e-4` versus
  outside `4.6276411633044296e-5`, and history
  `1.4428893342271215e-2` versus outside `2.8906847354762616e-3`; both
  localization flags passed. Material-interface damage/history remained
  diagnostic only, with top/bottom damage maxima below `2.78e-5` and history
  maxima below `1.78e-3`.
- The JAX contract test had two harness issues rather than formulation defects:
  GPU setup eagerly evaluated an unavailable CPU fallback, and CPU XLA fusion
  differed from NumPy center weights by at most `6.94e-18`. The test now
  selects the CPU fallback lazily and uses an explicit eight-unit-roundoff
  `float64` comparison for the algebraic phase measure.
- Verification after the test-only correction: engineering-grip CPU suite
  `25/25`; related baseline/mechanics/geometry/grip/algebra suites and scalar
  operator suite `61/61`; JAX engineering/mechanics suites on CPU `20/20`
  with two GPU-only skips; JAX engineering suite on GPU `12/12`; changed
  modules compile and `git diff --check` passes.

### Exact resume point

The v2 smoke/replay implementation evidence is passing, but it remains
nonproduction and no endpoint above `u_bar=1e-4` is authorized. Primary review
accepted the test-only eight-unit-roundoff cross-backend comparison: the observed
JIT-versus-NumPy phase-weight defect was `6.94e-18`, while all production fields
and acceptance gates remain CPU-authoritative.

Primary independent replay of
`/tmp/opencode/fracture2d_engineering_grip_v2_smoke.npz` completed in
`220.11034105997533 s`. It reproduced schema v2, one attempt, one accepted step,
three sweeps, zero cutbacks, strict status, and all crack-localization gates. The
archive SHA256 is
`f47220888afad18a23f2d673d0d516b1970428ffb0fdc5db0057b3a7d6da1314`; the
derived summary is
`/tmp/opencode/fracture2d_engineering_grip_v2_smoke.json`.

The exact resume point is to obtain explicit authorization for a larger staged
endpoint, then preserve the same GPU production/CPU replay split, strict gates,
`delta=0.048`, `h_g=0.02`, and v2 provenance contract. Do not select a production
horizon or infer crack-growth validation from the current smoke alone.

## 2026-08-27 — Authorize the strict checkpoint through 3e-3

- The user authorized one fresh-zero strict engineering-grip run through exact
  `u_bar=3e-3`. Preserve `delta=0.048`, `h_g=0.02`, nominal `du=1e-4`, the
  reviewed GPU-production/CPU-authoritative mechanics adapter, all existing
  gates, rollback/cutback semantics, and v2 provenance.
- Stop at the endpoint or any hard failure. Archive and independently replay the
  result before drawing conclusions from reaction or crack localization. No
  endpoint above `3e-3` and no production-horizon decision is authorized.

## 2026-08-27 — Validate the strict checkpoint through 3e-3

- The fresh-zero GPU-accelerated strict run reached exact `u_bar=3e-3` in 30
  nominal `1e-4` accepted steps. All 30 attempts passed without cutback; the
  maximum stagger count increased from 3 to 9. The one-time work check passed
  with relative error `4.61205510715101e-13`.
- The current v2 archive is
  `/tmp/opencode/fracture2d_engineering_grip_v2_u3e3.npz`, SHA256
  `43f0d1b993fa785fcad2c01e86e51d49dddef83ae5148bfb02f849951c40f0df`.
  Independent replay completed in `890.6306492816657 s` and reproduced all 30
  states, attempts, strict gates, and localization flags. Derived summaries are
  `/tmp/opencode/fracture2d_engineering_grip_v2_u3e3.json` and
  `/tmp/opencode/fracture2d_engineering_grip_v2_u3e3_analysis.json`.
- Reaction remained strictly increasing from `12.67494676595838` to
  `366.5621549864186`; the minimum increment was `11.351915421926606`. The
  incremental tangent decreased from about `1.26719e5` to `1.13519e5`, so the
  path is softening gradually but has not reached a reaction peak.
- Final maximum damage is `0.11866742277968424` at the exact notch tip
  `(0.5,0.5)`. Final crack-region/outside damage maxima are
  `0.11866742277968424/0.04256610372899043`; history maxima are
  `15.490055633305454/2.750838677110657`. Damage and history remain monotone and
  every localization flag passes.
- Final material-interface damage/history maxima are `0.02434403900921257` and
  `1.5812990406930494`, respectively `20.5%` and `10.2%` of the crack-region
  maxima. No loaded-interface takeover is present at this endpoint.
- The closest physical/stagger ratios were damage `0.7048`, history `0.5016`,
  and post-history damage residual `0.4966`. CPU-authoritative mechanics
  relative residuals approached their configured CG tolerance (`0.9997` before,
  `0.9969` after) but remained strictly below it and independently replayed.

### Exact resume point

Keep `3e-3` as the validated nonproduction pre-peak checkpoint. No farther
endpoint is authorized. If continuation is requested, select another staged
endpoint and preserve the same formulation, strict gates, GPU/CPU authority
split, and independent v2 replay; monitor the increasing stagger count,
mechanics tolerance margin, tangent softening, and interface/crack ratios.

## 2026-08-27 — Compare 3e-3 fields and nominal response with FEM

- Matched-load `d`, `u_x`, and `u_y` fields use the common 3395-particle material
  cloud and triangle set at prescribed `u_bar=3e-3`. The comparison figure is
  `/tmp/opencode/fracture2d_u3e3_nonlocal_vs_fem_fields.png`.
- The response comparison follows the existing FEM archive definitions
  `epsilon_nom=u_bar/L_y` and `sigma_nom=R_y/L_x`, with `L_x=L_y=1`, for both
  curves. The engineering material-interface opening remains diagnostic and is
  not substituted for prescribed strain. The curve figure and extracted data
  are `/tmp/opencode/fracture2d_u3e3_nonlocal_vs_fem_stress_strain.png` and
  `/tmp/opencode/fracture2d_u3e3_nonlocal_vs_fem_curve_data.npz`.
- At `3e-3`, engineering-grip nominal stress is `366.5621549864186` and the
  accepted point-anchor FEM warning-mode reference is `383.89724379713516`, a
  relative difference of `4.5156%`. Curve RMSE at the nonlocal load samples is
  `10.628864120430805`.
- Nonlocal/FEM final damage maxima are `0.11866742277968424` and
  `0.1064464912161516`. The corresponding field ranges and exact plotting
  provenance are in
  `/tmp/opencode/fracture2d_u3e3_nonlocal_vs_fem_plot_report.json`.

## 2026-08-27 — Authorize the complete strict path through 7.5e-3

- The user authorized the complete default load endpoint `u_bar=0.0075` after
  reviewing the validated `3e-3` checkpoint fields and FEM comparison. Preserve
  `delta=0.048`, `h_g=0.02`, nominal `du=1e-4`, all strict gates, the reviewed
  GPU-production/CPU-authoritative mechanics split, and v2 provenance.
- The reviewed controller has no archive-resume API, so the run must start from
  the exact zero state rather than manually injecting the `3e-3` checkpoint.
  Stop at `0.0075` or any hard failure; do not warning-accept, clip, or project a
  failed crack-growth state.

## 2026-08-27 — Hold the full path at the stagger budget

- Two fresh-zero reproductions stop identically at target `u_bar=0.0049` after
  accepting `0.00485`. The second retained `validated=False` evidence at
  `/tmp/opencode/fracture2d_engineering_grip_v2_u7p5e3_failure.npz` (SHA256
  `29a86301ca49dcdbe30206e0992a05033b80414ebed369c7ffea026061730446`) records
  49 accepted states, 52 attempts, all final-attempt sweep primitives, and the
  exact rollback state.
- The nominal `0.0049` attempt failed at `du=1e-4`; `0.00485` then passed at the
  single permitted cutback, and `0.0049` failed again at `du_min=5e-5`. The
  failure is the history gate after the configured 20-sweep budget.
- The last attempt is converging rather than running away: its history gate
  ratio decreases from `1.097e6` on sweep 0 to `3.781` on sweep 18 and `1.600`
  on sweep 19. Damage remains maximal at the exact notch tip. At the last
  committed state, crack-region/interface damage maxima are
  `0.3679529432351974/0.06121270244387817`, and history maxima are
  `61.87667177886805/4.052743520614918`.
- Reaction remains strictly increasing through the last committed load, reaching
  `556.2604873511528` at `0.00485`; no reaction peak has yet been accepted.
- Completion is blocked on an explicit decision whether to increase only
  `max_stagger_iterations` from 20 to 30. Do not change this budget, resume from
  the serialized state, or weaken any gate without authorization.

## 2026-08-27 — Authorize a 30-sweep full-path rerun

- The user explicitly authorized increasing only `max_stagger_iterations` from
  20 to 30 and fresh-zero rerunning through `u_bar=0.0075`. Preserve the
  formulation, `du=1e-4`, `du_min=5e-5`, every tolerance and strict gate, the
  GPU/CPU authority split, and v2 replay.
- Retain the 20-sweep failure archive unchanged. Write the 30-sweep run to a new
  artifact path and stop at the endpoint or any new hard failure.

## 2026-08-27 — Hold the 30-sweep path at 5.4e-3

- The 30-sweep fresh-zero run passed the former `0.0049` blocker and accepted 54
  states through `u_bar=0.00535`, then stopped at target `0.0054` after both
  nominal and minimum-increment attempts. The retained `validated=False` archive
  is `/tmp/opencode/fracture2d_engineering_grip_v2_u7p5e3_s30_failure.npz`,
  SHA256 `8566b84293956019dd45525b21b18b1ddc316fb3e20ac81488c9c35b3a846338`.
- The failed gates are damage, history, and post-history damage residual. The
  final attempt still converges: history gate ratio falls to `4.526` on sweep 28
  and `2.268` on sweep 29, while `rho_next_inf` falls to `2.2707e-6`.
- The last committed state remains notch-localized: damage maximum
  `0.48978478383367424` is at the exact tip, interface damage is
  `0.07350434714437187`, and crack/interface history maxima are
  `96.05929260318072/4.895407954387862`. Reaction remains monotone through
  `598.3479417559025` at `0.00535`.
- A further budget change requires explicit authorization. If requested, use a
  single `max_stagger_iterations=60` run rather than repeated 10-sweep increases;
  preserve every gate and stop if the fixed-point iteration ceases to contract.

## 2026-08-27 — Authorize 60 sweeps and durable artifact storage

- The user authorized `max_stagger_iterations=60` for one final fresh-zero run
  through `u_bar=0.0075`, preserving every equation, increment, tolerance,
  strict gate, and replay rule.
- The user also rejected `/tmp/opencode` as the sole location for reusable code
  or evidence. Move reusable runner/replay/analysis/plotting code into
  `examples/fracture2d/` and store input clouds, archives, summaries, plots, and
  hash manifests under `/root/autodl-tmp/data/fracture2d/engineering_grip/`.
  Temporary paths may be scratch only and must not be the sole provenance copy.

## 2026-08-27 — Hold the authorized 60-sweep endpoint after strict failure

- The fresh-zero `max_stagger_iterations=60` run preserved the approved
  `delta=0.048`, `h_g=0.02`, `du_initial=1e-4`, `du_min=5e-5`, strict gates,
  GPU-production/CPU-authoritative split, and v2 provenance. It reached 59
  accepted nonzero steps (60 accepted states including zero) and 62 attempts,
  then failed both the nominal and minimum-increment attempts at target
  `u_bar=0.0059000000000000294` after committing `u_bar=0.00585000000000003`.
- The durable audit is
  `/root/autodl-tmp/data/fracture2d/engineering_grip/engineering_grip_v2_u7p5e3_s60.failure.npz`
  with SHA256
  `20fe2edd834df78cb509aeeacb5456cb2d655b504b046d71eede357803fc93d7`.
  Its completion status is `strict_continuation_failure`, `validated=False`,
  and the failed gates are `damage`, `history`, and `damage_residual` after all
  60 sweeps. The matching summary, analysis, plots, and manifest are stored
  beside it under the durable engineering-grip data root.
- The last committed fields remain notch-localized: maximum damage
  `0.6739614816616006` and history `199.66938702316548` are both at
  `(0.5,0.5)`, versus outside maxima `0.24962440915933132` and
  `19.834826075727165`; interface damage/history maxima are
  `0.08662855789757251` and `5.803795156646753`. Damage, history, and reaction
  remain monotone in the committed states, but this does not make the rejected
  endpoint valid.
- Focused workflow tests (`5/5`), CPU engineering tests (`25/25`), changed
  module compilation, and `git diff --check` pass. A separate direct CLI
  analysis/plot attempt on the 1.57 GB failure archive exceeded the 120-second
  scratch validation timeout; the in-run durable analysis and plot reports are
  available and were not overwritten.

### Exact resume point

Keep the strict failure archive unchanged and do not warning-accept or resume
from it. Inspect the durable failure analysis if needed. Before changing the
stagger budget, increment, tolerances, gates, or formulation, ask for an
explicit numerical decision; otherwise this task is blocked at the rejected
`u_bar=0.0059` target.

## 2026-08-27 — Audit the final 60-sweep contraction

- The last minimum-increment attempt contains 60 complete sweeps at target
  `u_bar=0.0059`. At sweep 60, displacement and reaction pass with ratios
  `0.00479` and `0.55685`; damage, history, and post-history damage residual
  fail with ratios `1.74763`, `6.97383`, and `2.10106`.
- Over sweeps 35--59, log-linear tail factors are `0.8094` for damage and
  `0.8093` for both history and post-history residual. The three failed fields
  therefore remain geometrically contracting rather than plateauing,
  oscillating, or diverging. Extrapolation indicates approximately 3 additional
  sweeps for damage, 5 for residual, and 10--11 for history; this estimate is
  diagnostic and does not validate an uncomputed state.
- The fixed-`H_used` damage residual remains accurate (`rho_used_inf` about
  `4.55e-13` on the final sweep), so the blocker is the coupled history fixed
  point rather than the linear damage solve. The final rejected trial remains
  notch-localized, with damage/history maxima `0.6987217/223.4757580` at the
  exact tip, but those fields are not committed.
- The rejected attempt's before/after digests are identical, confirming strict
  rollback to the committed `u_bar=0.00585` state. All durable archive and
  derived-artifact SHA256 values match the manifest; the focused workflow suite
  passes `5/5`, and `git diff --check` passes.

### Exact resume point

Do not alter or resume the failure archive. Ask the user to select either a
fresh-zero 72-sweep budget-only rerun, a separately specified smaller-increment
study, a separately derived nonlinear strategy, or stopping at the last strict
committed `u_bar=0.00585` checkpoint. Preserve all strict gates unless explicitly
and independently reconsidered.

## 2026-08-27 — Authorize the 72-sweep budget-only rerun

- The user explicitly selected one fresh-zero 72-sweep run through
  `u_bar=0.0075`. Change only `max_stagger_iterations` from 60 to 72; preserve
  every equation, load increment, tolerance, strict gate, GPU/CPU authority
  split, rollback rule, and durable provenance contract.
- Retain the 60-sweep failure evidence unchanged. Use a new durable run ID and
  stop at the endpoint or any strict failure; do not resume from or commit any
  rejected trial.

## 2026-08-27 — Hold the 72-sweep path at 5.95e-3

- The fresh-zero run accepts 59 nonzero increments through exact
  `u_bar=0.0059`, including the former blocker in 68 sweeps, then rejects the
  minimum-increment target `u_bar=0.00595` after all 72 sweeps. The only final
  failed gate is history: its error/limit/ratio are
  `7.3970860e-6/2.5356861e-6/2.91719`. Damage, post-history residual,
  displacement, and reaction ratios are `0.70169`, `0.80142`, `0.00260`, and
  `0.01711`.
- Over the stable tail, damage/history/residual contract with factors near
  `0.8286`; history has one small late rebound (`1.0069`) but does not plateau or
  diverge. Fitted extrapolation gives 6 additional sweeps and a conservative
  pair-factor estimate gives about 9. The comparable accepted-step cost rose
  from 59 sweeps at `0.00585` to 68 at `0.0059`, and contraction slowed from
  about `0.8095` to `0.8281`.
- The last committed tip damage/history are `0.6987243342/223.4791594`, versus
  outside `0.2622925128/21.5063986` and interface
  `0.0879826246/5.8979300`. Reaction remains monotone through
  `634.9988773`. The rejected `0.00595` trial remains tip-localized but is not
  committed; before/after rollback digests match exactly.
- The durable failure archive SHA256 is
  `4373a9c47d8676ae5d702e259f5654849c305e01175b2dafcaaff0d977948af1`.
  All seven generated artifact hashes match the s72 manifest, the s60 hashes
  remain unchanged, focused workflow tests pass `5/5`, and `git diff --check`
  passes.

### Exact resume point

Keep the s60 and s72 failure evidence unchanged. Ask the user to select a single
bounded fresh-zero 80-sweep diagnostic, a separately specified smaller-increment
study, a reviewed nonlinear-strategy derivation, or stopping at the last strict
committed `u_bar=0.0059` state. Do not infer open-ended budget escalation from
the conditional tail estimate.

## 2026-08-27 — Select a smaller-increment study

- The user selected the smaller-increment study instead of another immediate
  stagger-budget increase. No continuation run is yet authorized.
- First compare the retained nominal and half-step s72 attempts and determine
  whether the natural next cutback `du_min=2.5e-5` is supported while preserving
  `max_stagger_iterations=72`, all equations, tolerances, gates, and fresh-zero
  provenance. Request explicit authorization after defining the bounded test.

## 2026-08-27 — Define the bounded 2.5e-5 increment test

- At `u_bar=0.0059`, the retained `1e-4` target `0.0060` finishes 72 sweeps with
  damage/history/residual ratios `7.142/31.375/7.721` and tail factor about
  `0.8493`. The `5e-5` target `0.00595` improves these to
  `0.702/2.917/0.801` with tail factor about `0.8281`. Halving the minimum
  increment is therefore the natural single-variable sensitivity test.
- The bounded proposal changes only `du_min: 5e-5 -> 2.5e-5`, keeps
  `du_initial=1e-4`, `cutback_factor=0.5`, `max_cutbacks=8`, and
  `max_stagger_iterations=72`, and stops at `load_final=0.00595`. If the `5e-5`
  final step fails, the controller tries `0.005925`; failure there stops
  strictly. This study does not claim the full `0.0075` path.
- Audit of the s72 schedule found a duplicate `0.00595` retry because float64
  subtraction placed the represented trial increment microscopically above
  `du_min`. Production and completed-archive replay now share an eight-ULP,
  scale-aware minimum-increment classifier. It changes no equations, gates, or
  accepted state and prevents only a no-progress duplicate retry. Focused
  engineering tests pass `27/27`.

### Exact resume point

The proposed fresh-zero run ID is
`engineering_grip_v2_u5p95e3_dumin2p5e5_s72`. Change only `load_final` to
`0.00595` and `du_min` to `2.5e-5` relative to s72; keep every other numerical
argument fixed. Obtain explicit user authorization before execution. Existing
s60/s72 artifacts remain immutable evidence tied to their recorded source
bundles and are not rewritten after the scheduling fix.

## 2026-08-27 — Authorize a separate full-path warning diagnostic

- The user superseded the pending smaller-increment run with a separate
  fresh-zero warning-mode diagnostic through `u_bar=0.0075`. Keep the strict
  controller and all strict artifacts unchanged.
- After 72 complete stagger sweeps, convergence-gate failures may be warning-
  committed and must retain their exact failed-gate list and errors. Any damage
  range violation or healing is also a warning. Continue only while the maximum
  absolute range excess and maximum healing magnitude are each at most `1e-3`;
  exceeding either threshold rolls back and stops.
- Do not clip, project, or alter the equations. NaN/Inf, linear-solver failure,
  incomplete sweeps, structural errors, and violations beyond `1e-3` remain
  hard failures. Mark every output `validated=False` and use it only to inspect
  the full response and field evolution.
- The user selected final-budget threshold timing: finite intermediate sweeps
  may exceed the range/healing bound and continue, because later stagger sweeps
  may recover. On the 72nd complete sweep compute raw
  `max(0,-min(d),max(d)-1)` and
  `max(0,-min(d_next-d_committed))`; either value above `1e-3` rolls back and
  stops. Final crack-localization failure is warning-only in this diagnostic.

## 2026-08-27 — Use constituent-scaled residual reconstruction bounds

- Redundant AT2 residual reconstruction now uses the reviewed componentwise
  bound `512*eps64` times the sum of absolute local, graph, history-damage, and
  history-constant terms. Volume residuals use the same bound multiplied by the
  positive particle volume.
- This accepts the observed `2.3283e-10` operation-order defect within the
  derived `2.7306e-10--4.0967e-10` range while tests still reject strong and
  volume perturbations above their constituent bounds. The authoritative
  damage-residual gate and all physical tolerances are unchanged.

## 2026-08-27 — Hold the warning path at 6.75e-3

- The fresh-zero warning run accepted 67 nonzero states through
  `u_bar=0.0067`: 59 were `strict_passed` through `0.0059`, and 8 were
  `warning_accepted` from `0.0060` through `0.0067`. The run then rejected both
  nominal `0.0068` and cutback `0.00675` attempts after 72 complete sweeps.
- The final candidate passes every recorded gate except healing. Its raw healing
  magnitude is `0.0065015065`, exceeding the authorized `1e-3` safety limit;
  range excess is zero. Rejected before/after digests match exactly.
- Reaction peaks at `641.3398032` at `u_bar=0.0061` and falls to `71.6055695`
  at the last committed state. Damage reaches `0.99996267` and forms a
  centerline crack from the notch tip toward the right boundary. Interface
  damage remains `0.09922`, so no grip-interface takeover occurs, but committed
  damage is not monotone and history reaches `2.1692e6` near the crack.
- The durable archive is
  `/root/autodl-tmp/data/fracture2d/engineering_grip/engineering_grip_warning_u7p5e3_s72.failure.npz`,
  SHA256 `dc052bad525acf35e2a53b772771ba7d5257548287d618ec6979245c7628d192`.
  All manifest-listed hashes match; every warning artifact is nonproduction and
  `validated=False`.
- Plot-report indexing is corrected for future generation: FEM field snapshots
  and response points now use separate indices. At `u_bar=0.0067`, snapshot
  index 67 and response index 94 correspond to FEM stress `5.5572326`; the
  already persisted run report's scalar `535.5371` is stale, while its full FEM
  curve arrays and plotted black curve are correct.

### Exact resume point

Inspect the durable fields and response through `u_bar=0.0067`. Do not continue
past the rejected `0.00675` state without an explicit decision about the
`1e-3` healing safety limit or a different nonlinear/increment strategy. Never
resume from the failure archive.

## 2026-08-27 — Authorize unbounded finite range/healing warnings

- The user explicitly removed hard damage-range and healing magnitude limits for
  a new full-path warning diagnostic through `u_bar=0.0075`. Any finite range or
  healing violation must still be recorded as a warning but must not stop or
  cut back the continuation.
- Preserve raw fields without clipping or projection. NaN/Inf, mechanics/CG
  failure, incomplete sweeps, structural errors, and work-check failure remain
  hard. Keep strict and bounded-warning code paths/artifacts unchanged, start
  from exact zero, and write a distinct `validated=False` evidence set.

## 2026-08-28 — Localized displacement outlier diagnosis

- The unbounded warning run completed 75 increments through `u_bar=0.0075` and
  remains `validated=False`. The raw endpoint archive is
  `engineering_grip_warning_unbounded_u7p5e3_s72.npz` with SHA256
  `0e505a2fb87aa59b0c712857bdd2d3beb2f5915573286156a97fc9e735e919da`.
- The displacement amplification starts at `u_bar=0.0062` and is confined to a
  connected cluster of 2-4 notch-tip particles. Particle 1869 at
  `(0.499704520, 0.503754427)` controls the maximum: `|u|=0.104554` at `0.0062`
  and `0.142118` at `0.0075`. At the endpoint only one particle exceeds ten
  times the prescribed displacement. The graph has no isolated particles.
- Sweep-level evidence identifies an undamped stagger positive feedback, not a
  CG, indexing, boundary-condition, or disconnected-row defect. During the
  `0.0062` attempt, particle 1869 progresses from `d=0.9063`, `|u|/u_bar=1.09`
  at sweep 7 to `d=0.99938`, `|u|/u_bar=11.0` at sweep 10. Its history rises
  from about `1.65e3` to `5.48e5`; at sweep 71, `d=0.9999166`, local
  degradation is `1.70e-8`, and `|u|/u_bar=16.86`.
- The `0.0062` final sweep is a budget cutoff rather than convergence:
  `e_u=2.32e-5`, `e_d=1.72e-4`, `e_H=358.4`, and `e_R=0.0124`. It bypasses
  displacement, damage, history, reaction, and damage-residual gates. All CG
  solves report success with relative residual about `1e-10`; near-singular
  damaged modes mean this does not bound displacement error.
- Next action: before interpreting the post-peak fields physically, test a
  convergence-first continuation that retains unbounded range/healing warnings
  but uses load cutback and a stabilized stagger strategy (for example damage
  under-relaxation). Do not clip the localized displacement or alter `k_res`
  without a separate numerical decision.

## 2026-08-28 — Stabilized short-path experiment authorized

- Implement an independent experimental controller; preserve all existing
  strict, bounded-warning, and unbounded-warning schemas and behavior.
- Finite damage-range/healing violations remain warning-only. Coupled
  displacement, damage, history, reaction, and damage-residual failures are not
  commit-bypassable: use safeguarded damage under-relaxation and load cutback.
- The relaxed iterate must converge to the original unrelaxed fixed-point
  equations; do not change the damage equation, constitutive law, `k_res`, graph,
  boundary conditions, or raw fields.
- Verify with manufactured positive-feedback tests and independent archive
  replay, then run fresh-zero only through `u_bar=0.0063`. Compare particle 1869,
  displacement amplification, convergence gates, reaction, and crack fields
  against the unbounded diagnostic before authorizing a full `0.0075` run.

## 2026-08-28 — Lean stabilized archive requested

- The first bounded stabilized launch was interrupted before producing an
  artifact. No process or partial run file remains.
- Persist only accepted physical states (`q`, `d`, irreversible `H`, reaction,
  load), geometry/source/config provenance, and scalar iteration summaries. Do
  not persist intermediate sweep/trial vectors or static operators that can be
  rebuilt from the source cloud.
- Independent verification must check accepted-state equations and convergence
  gates. It must explicitly report that exact rejected-trial path replay is not
  available from the lean archive.

## 2026-08-29 — Fixed damping result through `u_bar=0.00555`

- Fresh-zero fixed damped-Picard (`omega=0.5`, 72 sweeps) committed 62 steps
  through `u_bar=0.00555`, then rejected `0.0055625` at `du_min=1.25e-5` after
  72 sweeps. Only history (`1.81` times its limit) and post-mechanics damage
  residual (`1.40` times its limit) failed; displacement, damage update,
  unrelaxed fixed-point, reaction, CG, force, moment, range, and healing passed.
- The 63 retained states all pass independent accepted-field checks. The last
  committed raw ranges are `uy=[-2.943e-4, 5.845e-3]`,
  `ux=[-1.838e-3, 2.656e-5]`, and `d_max=0.552795`; no localized displacement
  amplification is present at this load.
- Lean failure archive:
  `engineering_grip_damped_u6p3e3.failure.npz`, 6,638,879 bytes, SHA256
  `ce4c22dbe7782445aa8c2daba7e7eaf8540ee3921601f14d6ae5142c7c611fb1`.
- Next action: run a separate fresh-zero `omega=0.25`, 144-sweep experiment
  through `0.0063`. Do not resume the failure archive.

## 2026-08-29 — Foreground cached `omega=0.5` run selected

- The cached `omega=0.25`, 144-sweep run was stopped by explicit user decision
  while progressing normally near `u_bar=0.005475`; no physical archive was
  written. It required roughly 130-140 sweeps per accepted softening increment
  and was operationally over-damped.
- Run fresh-zero with `omega=0.5`, 144 sweeps, mechanics cache, lean v5 archive,
  and flushed progress every 10 sweeps. Execute in the active shell rather than
  the background so analysis continues immediately when the command returns.

## 2026-08-27 — Read-only warning-mode contract audit

- No production continuation or source/test edit was performed. Focused warning
  tests pass (`6/6`), but the implementation is not yet ready for the authorized
  full-path diagnostic.
- **P1:** a finite lower-bound warning is allowed by the declared raw safety
  rule but cannot be returned. A manufactured `d_next[0]=-5e-4` with
  `warning_safety_limit=1e-3` reaches the warning acceptance path, then
  `EngineeringGripResult.validate()` unconditionally recomputes final crack
  diagnostics and `compute_crack_region_diagnostics()` rejects any negative
  damage (`nonlocal_engineering_grip.py:976-977`). The same rejection exists in
  archive state replay (`:7770-7795`). This violates the no-clipping warning
  contract for the lower side of `max(0,-min(d),max(d)-1)`.
- **P2:** force and moment are not retained as independent gate results.
  `_mechanics_gate()` computes separate pass booleans but returns only their
  conjunction (`:2059-2089`); both `_incomplete_diagnostics()` (`:2266-2267`)
  and `_full_diagnostics()` (`:2397-2398`) assign that conjunction to both
  `force_pass` and `moment_pass`. A manufactured force-only failure therefore
  reports `gate_passes={'force': False, 'moment': False}` and warning bypasses
  both gates. Replay mirrors the same collapse (`:6459-6460`, `:6995-7006`),
  so it does not expose the provenance error.
- Replay also does not require a rejected complete attempt to contain exactly
  `config.max_stagger_iterations` complete sweeps; it accepts any recorded
  nonempty count up to the configured maximum before classifying the final
  failed sweep (`:8181-8225`, `:8473-8514`). Warning-incompatible incomplete
  stages such as `damage_acceptance` are likewise admitted by the generic
  incomplete-stage validator (`:6732-6842`). These are replay state-machine
  risks, not changes to the declared equations or strict production gates.

### Exact resume point

Keep the strict s60/s72 evidence and all strict artifacts unchanged. Do not run
the authorized warning continuation until the lower-bound warning diagnostic,
independent force/moment gate provenance, and warning replay state machine are
resolved or explicitly narrowed and independently reviewed. The `pytest`
executable remains unavailable; use the focused unittest suites.

## 2026-08-28 — Stabilized short-path contract audit

- Added focused coverage for the separate stabilized controller: the relaxed
  fixed-point identity, a manufactured merit-backtracking/positive-feedback
  case, exact rollback in a rejected tiny attempt, numeric-only archive writing,
  and replay without the production controller, mechanics solver, or damage
  solver.
- A real fresh-zero manufactured-cloud smoke archive was written and replayed
  independently after enabling the required JAX float64 mode. It reached exact
  `u_bar=1e-4` with one accepted step, one sweep, and a passing work check;
  the result remains `validated=False` and experimental.
- Replay now verifies the final complete relaxation trial's recomputed merit,
  hard-gate status, and coupled-gate status. Complete trials also require their
  state/damage provenance fields, selection is paired with globalization, and a
  failure archive checks the last committed load as well as its fields.
- Verification: stabilized focused `5/5`; workflow `16/16`; legacy engineering
  `47/47`; JAX CPU engineering/mechanics `20/20` with two GPU-only skips;
  compileall and `git diff --check` pass. No production `u_bar=0.0063` run was
  performed and no durable evidence was changed.

### Exact resume point

This is implementation evidence only. Keep the stabilized path nonproduction
and obtain independent review of the final-trial replay contract before any
longer continuation. The current unresolved audit scope is provenance for
non-final complete backtracking trials, whose archive stores only their digest
and scalar audit rather than a full independently reconstructed candidate
state.

## 2026-08-29 — Complete lean v5 implementation verification

- The v5 stabilized writer, loader, source-cloud SHA256 binding, archive-budget
  preflight, workflow dispatch, analysis/plot provenance, and explicit CLI
  contract are implemented without changing the approved equations or legacy
  strict/warning paths. The README documents the exact short comparison and
  the `accepted-state-plus-final-solver-summary-v5` limitations.
- Focused stabilized archive coverage passes `7/7`; the workflow suite passes
  `20/20`. The combined CPU verification of stabilized archive/replay,
  workflow, legacy engineering-grip replay, engineering-grip JAX, and
  mechanics JAX suites passes `94` tests with two expected GPU-only skips.
- `python -m compileall -q examples/fracture2d tests` and `git diff --check`
  pass. No production `u_bar=0.0063` run was performed and no durable evidence
  was changed.

### Exact resume point

The remaining review point is unchanged: v5 scalar logs and digests do not
provide independent state reconstruction for non-final complete backtracking
trials. Keep the stabilized path experimental and do not authorize a longer
continuation until primary review accepts or narrows that limitation.

## 2026-08-29 — Finish summary-only v5 workflow migration

- The stabilized v5 archive now exposes one compact final solver-summary row per
  retained state, typed cumulative external work, and a compact failure summary;
  the old v5 attempt/sweep/trial archive fields are no longer written or used by
  the stabilized workflow.
- Workflow analysis, replay binding, storage-preflight reporting, failure
  summaries, focused fixtures, and the example README now use the v5
  `accepted-state-plus-final-solver-summary-v5` layout. Legacy archive branches
  remain unchanged.
- Verification so far: the stabilized archive and workflow unittest command
  passes `27/27`. Full repository compilation and the complete unittest suite
  remain to be run after this migration.

### Exact resume point

Run `python -m compileall -q examples/fracture2d tests`, `git diff --check`, and
the complete available unittest suite. Inspect any failures without running the
experimental `u_bar=0.0063` continuation or changing the approved equations.

## 2026-08-29 — Verify explicit fixed relaxation policy

- The stabilized controller now accepts the explicit project-owned
  `relaxation_policy` values `sufficient_decrease` and `fixed`. Fixed mode uses
  one `omega_initial` candidate per sweep, does not apply merit decrease, and
  retains the original hard/coupled convergence gates and raw fields. The v5
  metadata and per-state solver summaries persist the policy and use a null
  sufficient-decrease limit for fixed rows.
- The workflow CLI requires the fixed-policy constraints explicitly and rejects
  sufficient-decrease flags, warning acceptance, and warning safety settings for
  stabilized runs. Union-sized v5 displacement rows are reduced to the material
  prefix for field plotting.
- Verification: stabilized archive/workflow focused unittest command passes
  `31/31`; legacy engineering plus engineering/mechanics JAX CPU suites pass
  `67/67` with no failures; `python -m compileall -q examples/fracture2d tests`
  and `git diff --check` pass. A full unittest discovery run exceeded the
  1200-second timeout without producing a failure summary, so it is not claimed
  as complete.
- No production or diagnostic continuation was run, no durable evidence was
  changed, and the existing strict/warning/unbounded paths were not intentionally
  modified.

### Exact resume point

Keep the fixed-policy path experimental pending primary review of its
single-candidate globalization semantics and the existing lean v5 limitation
that non-final rejected trial vectors cannot be independently reconstructed.
Do not run `engineering_grip_damped_u6p3e3` or any continuation until explicitly
authorized.

## 2026-08-30 — Hold cached fixed damping at `u_bar=0.0059875`

- The fresh-zero foreground run
  `engineering_grip_damped_cached_o5_u6p3e3_s144` accepted 63 increments through
  `u_bar=0.005975000000000032` and rejected the minimum-increment target
  `0.005987500000000033` after all 144 sweeps. The run took
  `8738.572825920768 s`.
- Only history and post-history damage residual failed. Their final
  error/limit/ratio values are
  `9.1728565e-6/2.8203134e-6/3.2524246` and
  `7.1172698e-6/5.6406267e-6/1.2617871`. Fixed-history damage solve accuracy
  remained near roundoff with `rho_used_inf=3.4106051e-13`; displacement,
  damage, fixed-point, reaction, mechanics, force, moment, range, healing, and
  monotonic-history gates passed.
- The last committed state remains notch-localized. Tip damage/history are
  `0.7393086577/271.8526975`, outside maxima are
  `0.2849748008/24.7514462`, and interface maxima are
  `0.0900253661/6.0400474`. Material displacement ranges are
  `ux=[-0.0019629603,3.1982005e-5]` and
  `uy=[-0.0003113567,0.0062863299]`; maximum material displacement magnitude is
  about `1.060` times `u_bar`.
- Particle 1869 at `(0.4997045202,0.5037544271)` has
  `|u|=0.0038094734`, `d=0.7284607859`, and `H=254.5307348`; it does not exhibit
  the undamped warning path's localized displacement amplification.
- The lean failure archive is
  `/root/autodl-tmp/data/fracture2d/engineering_grip/engineering_grip_damped_cached_o5_u6p3e3_s144.failure.npz`,
  size `6,717,674` bytes, SHA256
  `9cea3fe7407f1229b4a0a6558191c564284ffb36e317deca367b0d543e31bb1c`.
  All manifest hashes match. Independent replay reports
  `field_equations_validated=true`, `path_replayed=false`, and
  `rejected_trial_path_available=false` for 64 accepted states.
- The last five accepted crack-growth increments required
  `141/135/140/139/141` sweeps. Because v5 stores only each retained state's
  final solver summary, it cannot support a sweep-level contraction fit for the
  rejected attempt.

### Exact resume point

Do not alter or resume the failure archive. The smallest bounded proposal is a
fresh-zero foreground sensitivity run stopping at `u_bar=0.0059875` with only
`du_min: 1.25e-5 -> 6.25e-6`; keep `omega=0.5`, 144 sweeps, every equation,
tolerance, hard gate, cache rule, and lean v5 contract fixed. Obtain explicit
user authorization before execution. If it fails, assess a separate budget-only
change rather than combining parameters.

## 2026-08-30 — Authorize bounded half-increment sensitivity

- The user authorized the proposed fresh-zero foreground sensitivity run through
  exact `u_bar=0.0059875`.
- Change only `du_min` from `1.25e-5` to `6.25e-6` relative to
  `engineering_grip_damped_cached_o5_u6p3e3_s144`. Preserve `omega=0.5`, the
  144-sweep ceiling, mechanics cache, all equations, tolerances, hard coupled
  gates, lean v5 provenance, and progress output every 10 sweeps.
- Use a distinct durable run ID. Stop at the endpoint or any hard failure; do
  not resume or rewrite the prior failure archive and do not commit a rejected
  candidate.

### Exact resume point

Preflight the GPU and durable output location, then launch the authorized run in
the foreground. After return, independently check accepted fields and compare
the result against the `du_min=1.25e-5` failure.

## 2026-08-30 — Complete half-increment sensitivity at `u_bar=0.0059875`

- The fresh-zero foreground run
  `engineering_grip_damped_cached_o5_u5p9875e3_dumin6p25e6_s144` completed 65
  accepted increments through exact `u_bar=0.0059875` in `8765.93 s`. It used
  three accepted cutbacks; the final effective increment was
  `1.09375e-5`, compared with the prior failed `1.25e-5` increment.
- The endpoint passed on sweep 143 of 144. Final fixed-point, history, and
  post-history damage-residual ratios are `0.9471547`, `0.6338103`, and
  `0.7842454`; mechanics, force, moment, reaction, range, healing, and all other
  coupled gates pass. The fixed-history residual remains near roundoff at
  `rho_used_inf=9.0949470e-13`.
- Final reaction is `638.6926773`; the accepted reaction path remains monotone.
  Tip damage/history are `0.7465331523/282.0310825`, outside maxima are
  `0.2893016363/25.4098249`, and interface maxima are
  `0.0903669682/6.0638237`. Maximum displacement amplification is `1.0599556`.
  Particle 1869 has `|u|=0.0038420690`, amplification `0.64168`,
  `d=0.7357945944`, and `H=264.1115747`; no localized drift is present.
- The completed lean archive is
  `/root/autodl-tmp/data/fracture2d/engineering_grip/engineering_grip_damped_cached_o5_u5p9875e3_dumin6p25e6_s144.npz`,
  size `6,916,149` bytes, SHA256
  `209959f2861c6c50c907d785bb154d9b1a474e16c98259f1524d1c4aeb1adfe6`.
  Independent replay validates all 66 accepted states and reports
  `field_equations_validated=true`, `path_replayed=false`, and
  `rejected_trial_path_available=false`.
- The solve completed before integrated plotting failed because the FEM archive
  has no exact `0.0059875` field snapshot. Workflow plotting now distinguishes a
  missing exact snapshot from malformed/duplicate snapshot loads, permits
  nonlocal-only fields, and keeps FEM response interpolation. The regenerated
  plot report uses linear interpolation between FEM response loads `0.005975`
  and `0.0060`. Workflow unittests pass `24/24`; compileall and
  `git diff --check` pass.

### Exact resume point

The smaller increment resolves the exact prior blocker but consumes `143/144`
sweeps, so do not infer that fixed Picard can continue to `0.0063`. Preserve the
completed archive and prior failure archive. Next derive and contract-test a
safeguarded Anderson acceleration with unchanged equations, raw fields, and
final unrelaxed gates before authorizing another long run.

## 2026-08-30 — Read-only Anderson contract audit

- The primary references were checked directly. Walker--Ni, *Anderson
  Acceleration for Fixed-Point Iterations*, DOI `10.1137/10078356X`, and Saad,
  *Acceleration methods for fixed point iterations*, arXiv `2507.11746`,
  support the fixed-point notation `f(x)=g(x)-x`, forward-difference blocks
  `X_j=[Delta x]` and `F_j=[Delta f]`, the least-squares problem
  `gamma=argmin ||f_j-F_j gamma||_2`, and the update
  `x_next=(x_j-X_j gamma)+beta*(f_j-F_j gamma)`. Saad Eq. 140 records
  Tikhonov-regularized normal equations as one implementation choice, while
  the same section identifies SVD and QR as alternatives. Saad also states
  that finite-window AA-TGS is not equivalent to standard finite-window AA.
- Project-owned derivation: take `x=d_iter`, `g(x)=d_solved`, and
  `f(x)=d_solved-d_iter`. The existing fixed policy is exactly
  `x_next=x+omega*f(x)`. An accelerated candidate must still execute the
  existing post-candidate mechanics/Amor evaluation and every original strict
  coupled gate; only the outer damage iterate may change.
- The following remain unresolved and were intentionally not guessed: standard
  Type-II versus Type-I/TGS, finite history depth, regularization/conditioning
  rule, coefficient or candidate-norm safeguard, fallback/restart details, and
  how the lean v5 archive should describe omitted accelerated trial vectors.
  The v5 contract currently reports accepted-state field validation but cannot
  claim rejected-trial path replay.
- Verification after the audit: `python -m compileall -q examples/fracture2d
  tests` passed; `git diff --check` passed; the focused command
  `python -m unittest tests.test_fracture2d_nonlocal_engineering_grip_stabilized
  tests.test_fracture2d_engineering_grip_workflow` ran `40` tests in
  `139.827 s` and returned `OK`. The argparse error text printed by expected
  negative parser cases was part of successful tests.
- No production or diagnostic continuation was run, no physical/archive
  artifact was changed, and no source or test implementation was added in this
  audit.

### Exact resume point

Ask for an explicit numerical selection of the Anderson variant, depth,
regularization/solve method, safeguard and fallback semantics, and archive
provenance contract. Then implement only that declared map and manufactured
contract tests; do not run `u_bar=0.0063` until those tests and independent
review pass.

## 2026-08-30 — Defer Anderson and authorize damped stabilized warning

- The user deferred Anderson implementation and selected a full fresh-zero
  `u_bar=0.0075` diagnostic using the current fixed `omega=0.5` relaxation with
  warning acceptance after the 144-sweep budget.
- Preserve `du_min=6.25e-6`, mechanics cache, all equations, tolerances, raw
  fields, and progress output. Coupled convergence, finite range, and healing
  failures may be warning-recorded after a complete final sweep; NaN/Inf,
  mechanics/CG failure, incomplete sweeps, structural errors, and work failure
  remain hard stops.
- Keep the strict stabilized schema and artifacts unchanged. The new policy must
  use distinct acceptance/schema provenance, mark outputs `validated=false`,
  retain every warning state's failed-gate summary, and never claim rejected
  path replay or converged field equations for warning-accepted states.

### Exact resume point

Specify and contract-test the isolated warning state machine and lean archive
semantics, then run the authorized path in the foreground with a new durable run
ID. Completion cannot be guaranteed across non-finite or mechanics hard
failures; stop loudly if one occurs.

## 2026-08-30 — Implement isolated stabilized-warning contract

- The authorized implementation contract is now explicit: strict stabilized
  mode remains `acceptance_mode=stabilized` with its existing v5 schemas. The
  warning diagnostic uses a distinct stabilized-warning mode/schema, fixed
  damped-Picard only, and accepts only a complete final sweep after the full
  144-sweep budget when finite and mechanics/CG hard gates pass.
- Warning-accepted states may retain failed coupled convergence, raw finite
  damage-range, and healing gates. Their exact failed-gate list, raw range
  excess, healing magnitude, and `passed=False` status must be archived.
  Incomplete/non-finite/mechanics/CG/structural/work failures remain hard and
  trigger the distinct failure archive. No fields are clipped or projected.
- Strict and warning replay/reporting must remain separate. Warning replay may
  check accepted fields but must report `validated=False`,
  `field_equations_validated=False` whenever a warning state is present, and
  `path_replayed=False` with no rejected-trial replay claim.

### Exact resume point

Implement and contract-test the state machine, warning schemas, workflow CLI,
and independent replay. Run focused tests and static checks only; do not launch
the long fresh-zero `u_bar=0.0075` diagnostic until those checks pass.

## 2026-08-30 — Contract-test stabilized-warning policy

- Added focused workflow coverage for the explicit stabilized-warning CLI
  contract: warning mode requires fixed relaxation, fixed `omega_min`, zero
  backtracks, explicit unbounded safety policy, and no numeric safety limit;
  missing/bounded/numeric-limit and sufficient-decrease variants are rejected.
- Added manufactured final-sweep classification coverage for canonical bypass
  ordering, raw range/healing values, incomplete sweeps, all-pass sweeps, and
  hard-gate failures.
- Added warning-v1 versus strict-v5 archive coverage, metadata and bypass-mask
  tamper rejection, and independent warning replay. Warning archives remain
  `validated=False`, while warning-accepted states do not claim field-equation
  validation or rejected-trial path replay.
- Focused coverage exposed two archive-reporting defects: accepted warning
  summary rows retained failure fields, and warning-accepted replay reported
  `field_equations_validated=True`. The serializer now keeps gate/bypass
  provenance while emitting empty accepted-row failure fields, and replay keeps
  its independent checks but reports warning states as unvalidated. Strict
  stabilized serialization/replay is unchanged.
- Verification: the requested stabilized/workflow unittest command passes
  `43` tests in `156.002 s`; the focused new policy tests pass `2` tests in
  `0.003 s`,
  `python -m compileall -q examples/fracture2d tests` passes, and
  `git diff --check` passes. No continuation run or durable evidence changed.

### Exact resume point

Keep the strict stabilized artifacts and warning/unbounded diagnostic artifacts
unchanged. The stabilized-warning contract is locally tested, but the controller
remains experimental and requires primary review of warning acceptance and the
lean v5 replay boundary before any fresh-zero continuation.

## 2026-08-30 — Recheck warning archive metadata tamper coverage

- Extended the warning archive forgery regression with a metadata gate-partition
  mutation, in addition to the bypass mask, warning-presence, and raw range
  mutations. The replay validator rejects all four forged payloads.
- Verification: the focused archive test passes `1` test in `26.967 s`; the
  stabilized/workflow command passes `43` tests in `156.515 s`;
  `python -m compileall -q examples/fracture2d tests` and `git diff --check`
  pass. The argparse usage/error text emitted by expected negative parser cases
  is not a test failure.
- No production or diagnostic continuation was run, and no durable evidence was
  changed.

### Exact resume point

Keep the strict stabilized artifacts and warning/unbounded diagnostic artifacts
unchanged. Obtain primary review of the warning acceptance and lean v5 replay
boundary before any fresh-zero continuation.

## 2026-08-30 — Complete damped stabilized-warning diagnostic

- The fresh-zero foreground run
  `engineering_grip_damped_warning_o5_u7p5e3_dumin6p25e6_s144` completed all 75
  nominal increments through exact `u_bar=0.0075` in `10811 s`, with fixed
  `omega=0.5`, 144 sweeps, and `du_min=6.25e-6`.
- Ten states were warning-accepted: `0.0060` through `0.0068`, plus `0.0071`.
  The other 65 states passed strict stabilized gates. Warning bypasses cover
  coupled transition convergence and four healing events; damage never leaves
  `[0,1]`. Maximum healing is `0.0291481` at `0.0068`.
- Reaction peaks at `641.3402` at `0.0061`, falls sharply through the warning
  interval, and ends at `69.1543`. The endpoint itself passes strict gates only
  on sweep `144/144`, but inherits the prior warning states and is not promoted
  to validated evidence.
- The final damage/history maximum is particle 1869 at
  `(0.4997045201795474, 0.5037544271277291)`: `d_max=0.99996296` and
  `H_max=2.167992e6`. Maximum material displacement is `0.1421214`, or about
  `18.95` times the prescribed endpoint. Interface damage remains `0.09924`, so
  this path exposes crack-tip feedback/runaway rather than the earlier
  top-boundary takeover.
- Archive SHA256 is
  `9545bed758c19c07297d1a0afdce9093caab7a646655eef6c6b2a82cb23ac4fc`.
  Independent accepted-state replay binds all 76 states and reports
  `field_equations_validated=false`; the lean warning-v1 contract does not claim
  rejected-path replay. Transition-only `force_scale` and `moment_scale` combine
  pre/post candidate states and are explicitly outside accepted-state replay;
  accepted mechanics, reactions, force/moment errors, and conservation gates
  are rebuilt independently.
- Durable `.npz`, summary, replay, analysis, fields, stress-strain, curve, plot,
  and manifest artifacts are under
  `/root/autodl-tmp/data/fracture2d/engineering_grip/` with the run stem above.

### Exact resume point

Keep this warning-v1 archive and both strict stabilized artifacts immutable.
Review the generated fields and response. Do not extend fixed Picard or treat
later strict passes as repairing inherited warning states; the next numerical
choice, if any, is the separately deferred safeguarded Anderson design.

## 2026-08-30 — Defer effectively zero-stiffness point drift

- The completed warning path visibly develops suspected effectively
  zero-stiffness particle drift after severe damage localization. This is
  consistent with the extreme displacement amplification and collapsed reaction,
  and remains unresolved.
- The issue is explicitly deferred. No clipping, stiffness-floor change,
  particle removal, or constitutive correction is introduced.
- The current input paths, physical constants, discretization, endpoint,
  increment, 144-sweep ceiling, and fixed `omega=0.5` values are now workflow
  parser defaults. Strict acceptance remains the default; warning mode,
  unbounded policy, storage budget, progress cadence, and output identity remain
  explicit.

### Exact resume point

Preserve all existing artifacts. If this issue is resumed, first isolate the
near-zero-stiffness particle mechanics and derive an auditable remedy rather
than extending the warning path or increasing the numerical budget.

## 2026-09-17 — Complete post-fix Anderson implementation audit

- The read-only review confirms the declared project-owned safeguarded
  Type-II path: depth `5`, `beta=0.5`, pivoted rank-revealing QR, raw range
  safeguard, sufficient-decrease rejection, history restart, and exact
  `omega=0.5` Picard fallback. Walker--Ni and Saad support the fixed-point,
  forward-difference, least-squares, and QR formulation; the depth, beta,
  fallback policy, candidate-map check, Anderson-only raw gates, and lean
  provenance semantics remain project-owned choices.
- Accelerated merit now uses one extra fixed-history damage map at the
  candidate `H_next`, with candidate `g(d_trial)-d_trial` and candidate
  `H_next` residual scaling. Structural candidate-map errors propagate;
  numerical failures fall back and restart history. The original unrelaxed
  fixed-point and damage-equation gates remain required.
- Strict Anderson acceptance and v5 replay require raw damage-range and healing
  gates. Lean v5 reports Anderson selection/fallback/restart/rank scalars as
  `recorded_not_reconstructed`; it does not claim rejected-trial path replay.
- Verification: stabilized/workflow unittest suites pass `57/57` in
  `208.712 s`; legacy engineering-grip plus JAX suites pass `60/60` in
  `19.701 s`; the related baseline suite passes `7/7` in `3.086 s`;
  `python -m compileall -q examples/fracture2d tests` and `git diff --check`
  pass. Expected argparse usage text from negative parser tests is not a
  failure. No physical continuation or durable artifact change was performed.

### Exact resume point

Implementation/manufactured evidence is independently GO, but the stabilized
path remains experimental. Keep all strict/warning artifacts unchanged and
obtain explicit authorization for one bounded fresh-zero checkpoint no farther
than `u_bar=0.0059875`; archive and review acceleration/fallback counts, sweep
reduction, strict gates, localization, and displacement amplification before
selecting any farther endpoint.

## 2026-09-17 — Authorize bounded Anderson physical checks

- First run a fresh-zero strict `delta=0.048`, `h_g=0.02` Anderson smoke through
  exact `u_bar=1e-4`, with `du_initial=1e-4`, `du_min=5e-5`, `20` sweeps,
  depth `5`, `beta=0.5`, sufficient-decrease coefficient `0.01`, and all
  existing CG/physical gates unchanged.
- Only after completed archive replay passes, run a distinct fresh-zero strict
  endpoint comparison through exact `u_bar=0.0059875`, changing only
  `du_min=6.25e-6` and the sweep ceiling to `144` relative to the smoke. This is
  the matched comparison against the retained fixed-Picard endpoint.
- Stop on any strict, archive, replay, or structural failure. Do not warning
  accept, overwrite prior artifacts, or continue above `0.0059875`.

### Exact resume point

Preflight the GPU, source hashes, durable storage, and unused run IDs. Execute
the smoke and replay first; proceed to the bounded comparison only after all
smoke checks pass.

## 2026-09-17 — Complete bounded Anderson physical checks

- The fresh-zero strict smoke
  `engineering_grip_anderson_u1e4_s20` reached exact `u_bar=1e-4` in three
  sweeps. Its archive and independent accepted-field replay pass; the accepted
  final row selects the accelerated candidate.
- The distinct bounded run
  `engineering_grip_anderson_u5p9875e3_dumin6p25e6_s144` strictly reached exact
  `u_bar=0.0059875` in 66 accepted increments and four accepted cutbacks. The
  maximum retained accepted-step cost is 142 sweeps. The completed archive is
  `/root/autodl-tmp/data/fracture2d/engineering_grip/engineering_grip_anderson_u5p9875e3_dumin6p25e6_s144.npz`,
  size `7,024,509` bytes, SHA256
  `4c9a6f7e68f26afe01ba3fcf15a44a7b588174671128001dab440dc6b08fcbdc`.
- Independent replay covers all 67 retained states and reports
  `field_equations_validated=true`, `path_replayed=false`, and
  `rejected_trial_path_available=false`. The one-time work check passes with
  relative error `1.6355e-13` against `1e-7`. Every strict physical and
  convergence gate passes; damage and history remain monotone and maximal at
  the exact notch tip.
- The endpoint agrees with fixed damped Picard: top reactions are
  `638.6926756694` and `638.6926773168`; maximum union-displacement, damage, and
  history differences are `6.76e-11`, `5.19e-8`, and `1.28e-5`. Anderson final
  tip damage/history are `0.7465331614/282.0310940`, interface maxima are
  `0.09036696855/6.063823704`, and displacement amplification remains about
  `1.05996`.
- Retained accepted-step summaries total 2,293 sweeps for Anderson versus 3,148
  for fixed Picard. At the 65 common loads the totals are 2,155 versus 3,148, a
  `31.54%` reduction, but the benefit is front-loaded: through `0.0045` the
  reduction is `75.23%`, while the common `0.0046`--endpoint tail improves only
  `0.97%`. The last ten Anderson retained rows all select Picard fallback after
  insufficient decrease. The endpoint itself takes 136 sweeps versus 143, but
  follows an extra accepted state at `0.0059828125`.
- Among the 66 accepted final-summary rows, 49 record `accelerated` and 17 record
  `picard_fallback`; all 17 fallback/restart reasons are
  `insufficient_decrease`. Recorded QR/history ranks are `1:22`, `2:8`, `3:1`,
  `4:5`, and `5:30`. These are accepted-state final-summary scalars only, marked
  `recorded_not_reconstructed`; lean v5 cannot count or reconstruct every
  intermediate/rejected accelerated candidate or rejected load attempt.
- Production runtime is `13038.79 s` versus `8765.93 s` for fixed Picard, about
  `48.7%` slower despite fewer accepted-step sweeps. Extra candidate-map work
  and late Picard fallback remove the practical benefit near the peak-entry
  region. Independent numerical review found no arithmetic discrepancy and
  recommends holding the path here.
- Final verification passes: stabilized/workflow `57/57`, legacy
  engineering-grip plus JAX `68/68`, related baseline `7/7`,
  `python -m compileall -q examples/fracture2d tests`, and `git diff --check`.
  Concurrent JAX tests printed GPU allocation warnings but completed `OK`; no
  physical run or durable artifact was modified during verification.

### Exact resume point

Preserve the smoke, bounded Anderson, fixed-Picard, and warning artifacts. Do
not continue above `u_bar=0.0059875` or promote Anderson to production. If the
user explicitly requests one more diagnostic, propose a fresh-zero strict run
only to `u_bar=0.0060` with all current settings unchanged; do not resume an
archive, relax gates, increase the sweep budget, or jump to `0.0063`/`0.0075`.

## 2026-09-17 — Authorize bounded durable Anderson scalar diagnostic

- User authorization supersedes the stale `0.0060` suggestion above: the
  approved endpoint is exact `u_bar=0.0059875`, not `0.0060`.
- Add only opt-in scalar JSONL instrumentation for the existing safeguarded
  Anderson path. Each record must bind to the source bundle, explicit config,
  and final accepted archive hash; it may contain no intermediate field
  vectors. Diagnostic I/O failures must be fatal. Legacy defaults, equations,
  gates, acceptance, fallback, and lean v5 archive semantics remain unchanged.
- Record per-sweep/proposal reference merit components, candidate normalized
  components including transition damage/history terms, dominant component,
  candidate map residual, sufficient-decrease limit, QR/history/coefficient
  information, restart/rejection, evaluation counts, and timing where
  available. Quantities that would require an extra mechanics solve solely for
  an unevaluated Picard comparison must be explicit `null`.
- First run manufactured structural calculations and focused tests. If those
  pass, run exactly one fresh-zero strict foreground diagnostic through no
  farther than `0.0059875`, using all prior Anderson parameters and a new
  durable run ID. Independently replay accepted fields and analyze the scalar
  rejected-candidate records. Do not resume the previous archive or change any
  physical/numerical setting.

### Exact diagnostic resume point

The opt-in scalar writer/callback, workflow flag, source/config/archive binding,
manufactured structural test, workflow binding test, and README contract are
implemented. The focused Anderson suite passes `33/33`, the focused workflow
suite passes `27/27`, and the new writer/binding subset passes `3/3`.
Rerun the two focused suites sequentially after the final diff review, then
execute the new bounded fresh-zero foreground run. If the foreground tool times
out, preserve and inspect the durable partial diagnostic/archive state; do not
background or extend the run.

## 2026-09-18 — Complete bounded Anderson scalar diagnostic and replay audit

- The fresh-zero strict diagnostic
  `engineering_grip_anderson_scalar_diag_u5p9875e3_dumin6p25e6_s144`
  completed exact `u_bar=0.0059875` with `66` accepted steps, `4` accepted
  cutbacks, and a maximum retained accepted-step cost of `143` sweeps. The
  completed archive is
  `/root/autodl-tmp/data/fracture2d/engineering_grip/engineering_grip_anderson_scalar_diag_u5p9875e3_dumin6p25e6_s144.npz`,
  size `7,023,659` bytes, SHA256
  `2332248643218cb8ff67440a5e4b803cb3745e025c7431083daa6b4c3fcadc3a`.
- The scalar stream has `11,116` records: `2,866` sweep records, `5,382`
  candidate records, and `2,866` proposal records, plus header/final records.
  It contains no numerical arrays/lists, and the final archive hash matches the
  final JSONL record. The stream's cumulative recorded solve counts are
  `mechanics_before=70`, `mechanics_after=5,382`, and `candidate_map=2,796`;
  no unevaluated Picard comparison was claimed.
- Of `2,796` accelerated candidates, `280` passed sufficient decrease and
  `2,516` were rejected in globalization. Every accelerated rejection had the
  same reason, `sufficient-decrease merit condition failed`; `2,502` were
  dominated by the normalized `history_transition` term and `14` by
  `candidate_next_damage_residual`. The rejected-candidate merit-ratio median
  was `15.3720` (minimum `1.01078`), while selected accelerated candidates had
  median ratio `0.260287` and maximum `0.989757`. The selected final summary
  rows contain `46` accelerated states, `20` Picard fallbacks, and the initial
  state; final ranks are `1:23`, `2:7`, `3:1`, `4:5`, and `5:30`.
- There were `2,800` failed intermediate sweeps, all at the coupled-gates stage;
  `66` final sweeps passed. The endpoint remains independently validated and
  notch-localized: final damage/history maxima are
  `0.7465331615/282.0310941`, reaction is monotone, and the largest replayed
  gate ratio is `0.9914108` for the post-history damage residual.
- The first independent replay report preserved the pre-fix standalone x64
  failure rather than being overwritten. A new report
  `/root/autodl-tmp/data/fracture2d/engineering_grip/engineering_grip_anderson_scalar_diag_u5p9875e3_dumin6p25e6_s144.independent.rerun.json`
  succeeds with archive/source binding valid,
  `field_equations_validated=true`, `66` accepted steps, `67` states, work-check
  relative error `1.63545e-13`, `path_replayed=false`, and
  `rejected_trial_path_available=false`. Standalone `analyze` initially exposed
  the same missing x64 initialization; the workflow now enables JAX x64 before
  stabilized geometry reconstruction, and the new
  `/root/autodl-tmp/data/fracture2d/engineering_grip/engineering_grip_anderson_scalar_diag_u5p9875e3_dumin6p25e6_s144.independent.analysis.rerun.json`
  succeeds with the same validation. The same standalone x64 contract now
  covers plotting; the independent plot rerun succeeds, uses the declared FEM
  response interpolation because no exact FEM snapshot exists at the target,
  and preserves the replay binding.
- Verification after the fix: stabilized unittest suite `33/33`, workflow
  unittest suite `27/27`, legacy engineering/JAX/baseline suites `75/75`,
  `python -m compileall -q examples/fracture2d tests`, and
  `git diff --check` pass. The expected argparse usage text from negative
  parser tests is not a failure.

### Exact resume point

Preserve the diagnostic archive, JSONL stream, replay reports, fixed-Picard
comparison, and warning artifacts. Do not continue above `u_bar=0.0059875` or
promote Anderson to production. Primary review should focus on the scalar
evidence that most accelerated proposals are rejected by history-transition
merit, the lack of tail benefit, and the lean v5 limitation that rejected trial
vectors remain `recorded_not_reconstructed`.

## 2026-09-18 — Qualify scalar evidence and close diagnostic I/O paths

- Independent scalar review fixes the accelerated-candidate accounting to
  `2,796 = 280 selected + 2,516 rejected`. Among the rejected candidates,
  `2,445` improve the raw `candidate_actual_map_residual_inf` relative to
  `reference_fixed_point_error`. `2,502` rejections are history-transition
  dominated, while `2,152` also have another merit component above the
  sufficient-decrease limit; history is therefore not the sole blocker.
- The `280` selected accelerated candidates contain `234` intermediate
  coupled-unconverged selections and `46` final load accepts. When a Picard
  comparison was not evaluated, its residual-comparison fields remain null and
  false; no hypothetical Picard residual is claimed and no extra solve is
  implied.
- The scalar stream binds its recorded workflow hash, but that hash differs
  from the current workflow bytes after the stabilized x64 fix. The stream's
  recorded provenance is retained; current workflow-byte identity is not
  claimed. The stale pre-x64 standalone failure report remains preserved.
- A manufactured Anderson run now has callback-on/off parity coverage for
  accepted fields, attempt path, candidate selection, and mechanics/fixed-
  history solve counts. Callback, writer write/flush/fsync, and completed or
  failed archive-write `OSError`s propagate as I/O failures and are not treated
  as numerical fallback. The workflow closes the diagnostic writer in a
  `finally` seam around both archive-write paths.
- Verification on CPU with JAX x64 enabled by the focused fixtures: stabilized
  suite `36/36` and workflow suite `29/29` pass. Expected argparse usage output
  from negative parser tests is not a failure. No physical continuation or
  durable artifact was run or changed.

### Exact resume point

Preserve all scalar and physical artifacts and do not continue above
`u_bar=0.0059875`. Review the qualified rejection overlap and recorded
provenance caveat; do not reinterpret the scalar stream as rejected-path replay
or as current workflow-byte identity.

Primary final verification: the combined CPU/x64 stabilized/workflow command
passes `65/65` in `267.018 s`; compilation and `git diff --check` pass.
This session added observational scalar diagnostics, diagnostic contract tests,
writer cleanup, and standalone x64 initialization. The diagnostic subtask is
complete; the broader engineering-grip task remains active. The next numerical
proposal is to derive a residual-based candidate globalization criterion
separately from final convergence gates, independently review it, and test it on
manufactured maps before any new physical run. This proposal is not approval to
change the current criterion or remove history/final acceptance gates.

## 2026-09-18 — Authorize explicit committed-history residual Anderson policy

- User review approved implementation of a new explicit
  `relaxation_policy=anderson_residual` on this same active task. The historical
  `relaxation_policy=anderson` merit, configuration, archive interpretation, and
  CLI behavior must remain unchanged.
- The new policy uses only the normalized actual committed-history map residual
  `r(d)=G_c(d)-d` for reference and accelerated selection merits. Reference and
  candidate use their own fixed-point scales and the existing `GateTolerance`;
  the explicit coefficient `sufficient_decrease=0.01` and `beta=0.5` give the
  selection threshold `(1-sufficient_decrease*beta)=0.995`.
- Accelerated candidates require explicit finite, mechanics/CG-before/after,
  force, moment, raw damage-range, healing-against-committed-state, and
  existing hard-history-monotonicity checks before residual selection. Numerical
  candidate-map failures restart history and use the exact `omega=0.5` Picard
  fallback; structural and diagnostic I/O errors propagate. The fallback is
  not subjected to the residual sufficient-decrease test.
- Every original final gate remains required, including the unrelaxed fixed-point
  and fixed-history damage-equation gates, transition displacement/damage/
  history/reaction gates, post-history residual, conservation, range, healing,
  globalization, and coupled acceptance. The new policy gets a distinct merit
  identity in configuration, summaries, replay, and scalar diagnostics; old
  artifacts are never reinterpreted.

### Exact implementation resume point

Audit every literal `anderson` dispatch in the stabilized controller,
workflow CLI, scalar diagnostic, v5 summary/replay, and tests. Add manufactured
coverage for residual improvement versus failed transition gates, the `0.995`
threshold, unequal normalizers, hard-gate rejection, exact fallback/restart,
structural exceptions, policy archive round trips/mismatch rejection, and
historical-policy preservation. Run focused CPU/x64 tests sequentially and
compile/diff checks only; do not launch a physical continuation until primary
implementation review authorizes it. The only later physical endpoint allowed
by this task is at most exact `u_bar=0.0059875`; do not run `u_bar=0.0060`.

## 2026-09-18 — Complete explicit residual-policy contract coverage

- The authorized `anderson_residual` implementation audit is complete for the
  manufactured/controller/archive/workflow paths. Historical `anderson`
  configuration, selection merit, scalar identity, archive interpretation, and
  CLI behavior remain covered separately and unchanged.
- Added focused contract coverage for the residual-only map merit, unequal
  reference/candidate normalizers, the exact `0.995` threshold, transition and
  coupled-gate rejection, hard-gated Picard fallback, exact `omega=0.5`
  fallback/restart provenance, structural candidate-map exception propagation,
  residual archive round-trip and policy-mismatch rejection, CLI configuration,
  and scalar diagnostic merit identity/null fallback fields.
- Verification passes sequentially with JAX x64 enabled: stabilized `44/44`,
  workflow `29/29`, and legacy engineering-grip/JAX/baseline regressions
  `67/67`. The workflow suite's argparse usage/error lines are expected
  negative-parser-test output. Compilation and `git diff --check` pass.
- No physical continuation or durable artifact change was performed. The
  `u_bar=0.0059875` ceiling and prohibition on `u_bar=0.0060` remain active.

### Exact resume point

Primary Sol review should inspect the residual-policy dispatch and the
manufactured threshold/fallback/archive tests, with particular attention to
the committed-history map identity, separate normalizers, hard-gate ordering,
and the fact that rejected trial vectors are not claimed to be replayed. After
review, decide whether this experimental policy is retained; do not launch a
physical continuation from this implementation-only evidence.

## 2026-09-18 — Complete residual-policy validator fixes

- Replay binding now permits the historical `anderson` report to omit only
  `relaxation_policy` and `selection_merit_identity`; if either is supplied it
  must agree with the archive. Residual-policy reports remain strictly bound to
  `anderson_residual`, and a legacy archive paired with residual identity fields
  is rejected.
- v5 solver-summary validation now rejects residual fallback selection merit or
  candidate-map fields, requires complete residual accelerated rows to carry
  finite nonnegative merit/reference and candidate-map metrics, and preserves
  null candidate fields for incomplete failure rows. Historical `anderson`
  summary semantics are unchanged.
- Added accepted/failure forged-summary regressions and the historical optional
  identity/reverse-mismatch replay-binding regression. No controller numerical
  path or physical artifact was changed.
- Verification completed sequentially with JAX x64 enabled: stabilized
  `46/46` in `315.674 s`, workflow `30/30` in `18.884 s`, and legacy
  engineering-grip/JAX/baseline regressions `67/67` in `32.555 s`. The expected
  argparse usage/error text from negative parser tests is not a failure.
  `python -m compileall -q examples/fracture2d tests` and `git diff --check`
  pass. No physical continuation was run.

### Exact resume point

The requested validator fixes are locally verified. Primary Sol review should
inspect the optional historical identity handling, residual fallback nulls,
complete accelerated metric requirements, and incomplete failure-stage nulls.
Preserve all artifacts and do not run above `u_bar=0.0059875` or run
`u_bar=0.0060`.

## 2026-09-18 — Complete authorized committed-history residual experiment

- The fresh-zero foreground command used the exact scalar-diagnostic base
  configuration from
  `/root/autodl-tmp/data/fracture2d/engineering_grip/engineering_grip_anderson_scalar_diag_u5p9875e3_dumin6p25e6_s144.manifest.json`,
  changing only the run ID and `relaxation_policy=anderson_residual`, while
  retaining `--anderson-scalar-diagnostic`, `delta=0.048`, `h_g=0.02`,
  `du_initial=1e-4`, `du_min=6.25e-6`, `max_stagger_iterations=144`,
  `anderson_beta=0.5`, `anderson_history_depth=5`, and
  `sufficient_decrease=0.01`. It started at exact zero, used `cuda:0`, and
  did not use `--overwrite` or archive resume.
- The durable run ID is
  `engineering_grip_anderson_residual_u5p9875e3_dumin6p25e6_s144`. The strict
  run completed exact `u_bar=0.0059875` with `60` accepted steps, `0` cutbacks,
  `593` complete sweeps (`533` failed intermediate sweeps, `60` passed final
  sweeps), and maximum retained accepted-step cost `47`. No load attempt failed.
  The production-only elapsed time is `1669.1594366817735 s`, taken from the
  durable `STABILIZED_PROGRESS` completion record; the workflow total
  `2004.9197795251384 s` includes its internal replay, analysis, and plots and
  is not used as the production timing comparison.
- The residual policy is the project-owned selection rule
  `r(d)=G_c(d)-d`, with candidate/reference-specific fixed-point normalizers
  and the threshold
  `m_candidate <= (1-sufficient_decrease*beta)m_reference = 0.995m_reference`.
  Finite, mechanics/CG-before/after, force, moment, raw range/healing, and hard
  history-monotonicity prerequisites remain mandatory; failed prerequisites
  use exact `omega=0.5` Picard fallback, while every original final strict gate
  remains unchanged. Walker--Ni/Saad support the fixed-point/Anderson notation;
  this committed-history residual merit and its globalization contract are
  project-owned. No physical equation, constitutive law, gate, or boundary
  treatment changed in this run.
- The scalar stream has `1900` records (`712` full proposal records, `593`
  sweeps, `593` Anderson-proposal records, plus header/final records). Of `533` accelerated
  candidate attempts, `433` map evaluations completed, `414` were selected,
  `100` were rejected by the combined hard prerequisites, and `19` were
  rejected by sufficient decrease. The `179` Picard fallbacks comprise
  `60` insufficient-history, `100` hard-gate, and `19` insufficient-decrease
  fallbacks. No candidate-map failure occurred; `415` of the completed maps
  reduced the raw map residual. The scalar contract records selection/rank/
  fallback data but does not reconstruct rejected trial vectors.
- Independent x64 replay in a separate process passes the v5 binding and field
  equations: `60` accepted steps, `61` states, `field_equations_validated=true`,
  `replay_binding_valid=true`, work-check relative error
  `1.635450717121044e-13`, `path_replayed=false`, and
  `rejected_trial_path_available=false`. The largest authoritative independent
  gate ratios are `damage=0.8178129`, `history=0.9024915`, and
  `rho_next=0.9048608`; mechanics ratios near `0.996` are explicitly CG
  diagnostic proxies, not changed gates.
- Endpoint fields remain monotone and localized: damage/history maxima are
  `0.746538065454265` and `282.03868468430056` at `(0.5,0.5)`, crack-region
  versus outside maxima are `0.746538065454265/0.2893057262509533` and
  `282.03868468430056/25.410601905284004`, interface maxima are
  `0.09036698209006554/6.063824704563838`, and material displacement
  amplification is `1.0599555669957506`. The final reaction is monotone and
  equals `638.6929499863869`.
- Against the original historical `anderson` archive, endpoint infinity-norm
  differences are `4.01081698604179e-08` in union displacement,
  `1.3353467154608722e-04` in damage, and `5.0520752803635105e-02` in history;
  reaction differs by `2.743169895893516e-04` (`4.2949763e-07` relative).
  Against fixed damped Picard, the corresponding differences are
  `4.014521834665857e-08`, `1.3353232294960327e-04`,
  `5.052074311669941e-02`, and `2.726695945511892e-04` (`4.2691830e-07`).
  These are small endpoint differences under unchanged strict gates, but the
  residual path is not bitwise the historical path.
- Production timing is favorable but remains an experimental comparison:
  `1669.1594366817735 s` versus historical Anderson production
  `12703.620977 s` and fixed-Picard `8765.930930 s` (respectively `7.6108x` and
  `5.2517x` speedups). Historical Anderson's `13038.790514 s` is its workflow
  total, not production time. The separate replay reports `308.6693260553293 s`; it
  is not mixed into those production comparisons. The lower cost comes with a
  different `60`-step path versus historical Anderson `66` and Picard `65`, so
  it does not by itself authorize production promotion.
- Principal durable hashes are: archive
  `285014fe493fa20dd1663c4c9a28393bb59afc6826a8dd49c74e927e971ca381`, manifest
  `938c83b73663863b575d7757640d2b5659428feeaaabd2d60ffb348580d07bcb`, scalar
  stream
  `f8fc4991838a176d4d1aebeabab6490ab59e0f5d7e4a7e5da26bc705a4d11e29`,
  independent replay report
  `838c3296182cf0c5d30f4c8111a008af9d84792fc0663ee384b29addb101bd0a`, and
  scalar analysis report
  `6d775c3754a1c3abb2f94ea40493eb5dfcd152b1cc88bd33475b0305b708e8e1`; the
  durable foreground run log hash is
  `ab44f223173ec58d651a1aa2b588f26c6f0b6036141c5b3934b5785c119c1c01`.
  The source cloud hash remains
  `fa2ebf165f2e89ec88c63116346401b5039263e8e4c1a1917f6f13a3f4541267`; the
  residual source-bundle hash is
  `7529fd97ea1c583b36d1592d295c91b388415d8d4bd40ef164ce518a0e35a750`.

### Exact resume point

Preserve the residual-policy archive, manifest, scalar stream, durable run log,
replay report, analysis report, plots, and all historical artifacts unchanged.
No further physical endpoint is authorized, including `0.0060`, `0.0063`, or
`0.0075`; do not resume or overwrite any archive. Primary Sol review should
decide whether the residual policy is retained based on the hard-prerequisite
fallback rate, the accepted-state-only replay limitation, and the small but
nonzero endpoint difference from historical Anderson/Picard. No source change
was needed for run integration; the only wrapper retry was a preserved
non-numerical `/usr/bin/time`-missing command failure.

### Post-documentation verification

- After the memory update, the legacy engineering-grip/JAX/baseline suites passed
  `67/67` in `35.853 s`; `compileall` and `git diff --check` also passed.
- A fresh rerun of the combined stabilized/workflow suite did not produce a
  test failure, but exceeded the tool timeouts at `120 s` (after `23` dots) and
  `300 s` (after `39` dots). The earlier validator verification recorded
  `76/76`; no source files were changed after that verification, only task
  memory was updated. No further physical run was started to investigate this
  test-duration issue.

## 2026-09-18 — Independently retain the bounded residual-policy evidence

- Read-only review verified all 20 checked manifest entries, framed solver
  source-bundle hash, configuration identity, archive/scalar binding, and the
  independent replay report. The production completion timestamp for historical
  Anderson is `12703.620977 s` in retained tool output
  `tool_0ae916165001YT9Ea6pgXtYrqm:523`; its `13038.790514 s` includes workflow
  postprocessing. Correct production speedups are `7.6108x` versus historical
  Anderson and `5.2517x` versus Picard. The existing scalar-analysis report's
  historical production timing label is stale; preserve it as recorded evidence
  and use this correction for comparisons.
- Endpoint cost is `47` sweeps, including `11` accelerated selections; the final
  sweep selects Picard. All 100 combined hard-prerequisite rejections occur at
  loads `0.0055` through the endpoint (counts `3,5,12,18,28,34`). Their component
  metrics are null, so the record cannot distinguish healing/range/conservation/
  history-ordering causes. Do not infer a specific cause from the combined flag.
- The maximum endpoint damage/history differences occur at material particle
  `1851`, approximately `(0.49541641,0.51410685)`. Historical Anderson samples
  `u_bar=0.005965625`, reaches history `36.2153660865`, and retains that maximum;
  the residual path goes from `0.0059` directly to `0.0059875`, ending with
  history `36.1648453337`. Thus history sampling explains a concrete component
  of the difference: this is not solver-tolerance identity, and strict field
  replay does not prove load-step convergence of maximum-history evolution.
- Retain `anderson_residual` as an experimental option with independently
  passing accepted fields. Final primary CPU/x64 stabilized/workflow tests pass
  `76/76` in `320.139 s`; compilation and `git diff --check` pass. This resolves
  the previous short-timeout verification gap. No source change was needed.

### Exact resume point

The requested feasibility assessment, implementation, bounded physical run,
and independent review are complete. Preserve all artifacts. A matched-load-
schedule study at the same endpoint is the next recommended numerical check;
first specify its explicit schedule and provenance contract. No higher endpoint
or schedule change is authorized by this completion. The broader baseline task
remains active; this subtask changes the distinct solver policy, tests, README,
  and the three project memory files, without altering the physical model.

## 2026-09-18 — Authorize explicit matched-load schedule contract

- User authorization is for implementation and contract tests only; no physical
  run is authorized. The schedule is opt-in through workflow CLI
  `--load-schedule-npz`, and is valid only for stabilized strict
  `anderson_residual`.
- The schedule artifact must be numeric NPZ without pickle: one exact
  little-endian float64 one-dimensional nonzero target array plus JSON-compatible
  scalar/string/array binding metadata for the historical source archive path
  and SHA256. Targets are finite, strictly increasing, begin above
  `load_start`, and end at the existing explicit `load_final` under its exact
  endpoint rule. The canonical schedule has 66 targets and raw-target SHA256
  `13d294...`; its source archive SHA256 begins `4c9...` and its accepted
  `state_u_bar[1:]` must match targets bitwise.
- Schedule mode prepends exact zero, makes one attempt per target with zero
  cutbacks, and removes adaptive target calculation/retry/insertion. Adaptive
  increment fields remain required parser inputs but are inactive and must not
  be described as a minimum-increment guarantee.
- Schedule-specific stabilized completed/failure schema and replay identity are
  distinct from legacy v5, with binding through configuration metadata, scalar
  header/final, summary, replay report/binding, and manifest. Replay must verify
  the exact zero-plus-schedule state prefix, accepted target sequence, and the
  exact next target for strict failure. Historical adaptive archives remain
  unchanged.
- Exact implementation resume point: audit every stabilized/workflow/archive/
  replay validator and add extraction, malformed-artifact, controller, rollback,
  round-trip/forgery, legacy-preservation, and CLI contract tests; update the
  example README. Run focused CPU/x64 stabilized and workflow suites
  sequentially, relevant legacy suites, compilation, and `git diff --check`.
  Do not launch a physical continuation.

## 2026-09-18 — Implement matched-load schedule artifact and workflow contract

- Added the reviewed schedule loader and exclusive `extract-schedule`/
  `extract_schedule` workflow command. Extraction reads the historical completed
  archive with pickle disabled, verifies its source hash and historical Anderson
  provenance, and writes only the exact `targets`/`metadata_json` NPZ contract;
  existing destinations are refused, including concurrent creators.
- Schedule configuration is accepted only for strict stabilized
  `anderson_residual`. The controller prepends exact zero, attempts each target
  once without adaptive cutbacks or insertion, validates the exact accepted
  prefix, and records the exact next target on strict failure. Schedule-specific
  completed/failure archive schemas, replay contracts, scalar diagnostic fields,
  manifest fields, and summary fields are bound to the canonical schedule/source
  payload. Historical adaptive and `anderson` paths retain their prior schemas.
- Added focused artifact and CLI parser tests, updated schedule/replay binding
  expectations, and documented the contract in the fracture2d README. No
  physical run or durable simulation artifact was created or modified.
- Verification in the JAX `.venv`: stabilized suite `47/47` in `307.361 s`,
  workflow suite `31/31` in `17.523 s`, focused extraction/parser tests pass,
  legacy engineering-grip/JAX/baseline regressions pass `67/67`,
  `python -m compileall -q examples/fracture2d tests` passes, and
  `git diff --check` passes. The system Python lacks JAX; its import failure was
  avoided by running the required suites in `.venv`.

### Exact resume point

Primary Sol review should inspect the canonical 66-target/source hash constants,
the no-pickle exclusive extraction write, schedule archive header/metadata
binding, exact-prefix/next-target validators, and preservation of historical
adaptive/`anderson` contracts. Do not extract the durable schedule or launch a
 physical matched-load run until that review authorizes it; no endpoint above
 `u_bar=0.0059875` is authorized.

## 2026-09-18 — Re-run schedule provenance verification

- Re-ran the complete stabilized unittest module in the JAX `.venv`: `52/52`
  passed in `344.178 s`, including the scheduled-controller success and strict
  failure rollback/archive/replay tests. The workflow module passed `33/33` in
  `18.986 s`; expected argparse usage output came from negative CLI tests.
- Sequential legacy engineering-grip/JAX/baseline regressions passed `67/67`
  in `29.775 s`. `python -m compileall -q examples/fracture2d tests` and
  `git diff --check` passed. No source, durable simulation artifact, schedule
  extraction, or physical continuation was performed during this verification.
- Diff-level changes in this session were limited to updating the active task
  verification record and the project dashboard counts; the schedule source,
  workflow, and test implementation remained unchanged after the prior review.

### Exact resume point

The implementation and focused verification are complete. Primary Sol review
must still inspect the canonical constants, temporary-fixture provenance
coverage, exclusive no-pickle extraction, schedule archive/replay bindings,
and preservation of adaptive/legacy contracts before authorizing extraction or
any physical matched-load run. No endpoint above `u_bar=0.0059875` is
authorized.

## 2026-09-18 — Add scheduled manifest binding regression

- Added `EngineeringGripWorkflowTest.test_write_manifest_binds_exact_scheduled_payload`.
  It invokes `_write_manifest` with a valid temporary scheduled config and
  artifacts, reloads the JSON manifest, and asserts exact equality of the
  top-level and nested `load_schedule` payloads plus schedule archive/replay
  identities. No separate manifest validator exists to forge; the manifest is
  derived directly from the validated immutable config binding.
- Verification: stabilized suite `52/52` in `350.722 s`; workflow suite
  `34/34` in `18.372 s`; the new regression alone passed; `compileall` and
  `git diff --check` passed. No production code changed, and no physical run
  or schedule extraction was performed.

### Exact resume point

Primary Sol review remains the only next action: inspect the canonical schedule
constants, temporary provenance fixtures, manifest/archive/replay bindings,
and legacy adaptive preservation before authorizing extraction or physical
matched-load execution.

## 2026-09-18 — Complete the authorized matched-load physical comparison

- The canonical schedule was extracted to
  `/root/autodl-tmp/data/fracture2d/engineering_grip/schedules/engineering_grip_historical_anderson_u5p9875e3_targets.npz`.
  It has 66 little-endian float64 targets, target SHA256
  `13d294d9f3817bfbf0da5eec1622f278cc6217f0e4c528275fbdbfe1fbb10cb5`, file
  SHA256 `8c8fcc7f1ef5bcf0239fbd3fca160ea3e8ff9644f4b9b53cd81aaf6775b38e0a`,
  and is bitwise equal to the historical Anderson source `state_u_bar[1:]`.
- The fresh-zero matched run used stabilized strict
  `anderson_residual`, preserved the physical equations and final gates, and
  completed exact `u_bar=0.0059875` with 66 attempts, 66 accepted nonzero steps,
  67 retained states including zero, zero cutbacks, 822 complete sweeps, and
  maximum 43 stagger iterations. Its archive is
  `/root/autodl-tmp/data/fracture2d/engineering_grip/engineering_grip_anderson_residual_matched_historical_u5p9875e3_s144.npz`
  with SHA256
  `c2565925e599678b38d105c63e5410aacd5d87146521d361b089bcc3ba0488ed`.
- Independent CPU/x64 replay, analysis, and plotting were performed in
  separate processes and pass archive binding and `field_equations_validated`.
  The archive remains `experimental=true` and `validated=false` by the
  stabilized experimental contract; this is not a promotion to production.
- State matching is exact for all 67 states against historical Anderson. The
  maximum matched-versus-historical differences are
  `2.67634785861004e-10` in `q`, `6.04185976804672e-08` in `d`, and
  `6.821109599286501e-05` in `H`; endpoint differences are
  `1.960145956138304e-10`, `4.8201790447954806e-08`, and
  `6.17388192836188e-05`. Maximum top/bottom/anchor reaction differences are
  `5.219860668148613e-06`, `6.384582547980244e-06`, and
  `6.354155860407995e-06`.
- The fixed-Picard archive shares 66 loads and omits only the matched
  intermediate state `0.005982812500000004`. Matched-versus-fixed maximum
  differences are `2.6755367691332077e-10` in `q`,
  `6.033989585763067e-08` in `d`, and `7.317767580161672e-05` in `H`.
- Independent array checks confirm finite fields, monotone damage/history and
  reaction, all final gate flags true, zero retained CG statuses, and crack
  localization. Final damage/history are
  `0.7465332077673724`/`282.0311557111417` at `(0.5,0.5)`; final top
  reaction is `638.6926722766116`.
- Production completion elapsed is `2803.56144060893 s`, compared with the
  retained historical Anderson production reference `12703.620977 s` and
  fixed-Picard `8765.930929828435 s`. Replay, analysis, and plotting are
  excluded. The resulting ratios are `4.531244x` and `3.126713x`; the fixed
  timing remains a one-state-different comparison.
- The scalar diagnostic records 756 accelerated attempts, 490 map evaluations,
  466 selected accelerated candidates, 266 combined hard-prerequisite
  rejections, 24 sufficient-decrease rejections, 356 Picard fallbacks, and no
  candidate-map failures. Rejected trial vectors are not reconstructable from
  the lean archive/scalar contract, and hard rejections are not fully
  component-attributed.
- Durable comparison report:
  `/root/autodl-tmp/data/fracture2d/engineering_grip/engineering_grip_anderson_residual_matched_historical_u5p9875e3_s144.comparison.rerun.json`,
  SHA256 `7d2a2b9854a789d44181890a46e101c649b06ad33af354aa12dfd8fc137bc0e5`.
  No source change was needed for run integration.
- Final session verification passed: the durable comparison assertions,
  `python -m compileall -q examples/fracture2d tests`, and `git diff --check`.
  No full test suite was rerun because solver source was not changed in this
  review session; the focused implementation/regression results above remain
  the authoritative source-test record.

### Exact resume point

Preserve every historical, fixed-Picard, residual-policy, warning, schedule,
matched archive, report, log, scalar stream, and plot unchanged. Do not run a
higher endpoint or a new schedule. Primary Sol review should inspect the
durable comparison report, exact field/reaction maxima, scalar fallback and
hard-rejection accounting, schedule/hash bindings, and accepted-state-only
replay limitation before deciding whether the matched residual policy remains
experimental evidence. If an integration defect is found, stop and report it;
 do not alter the physical solver under this task.

## 2026-09-18 — Complete independent matched-artifact audit

- Independent read-only review verified all `31` hashes listed by the durable
  comparison report, all `9` manifest-listed artifacts, ZIP integrity, numeric-
  only arrays, schedule/source bitwise identity, `66` one-attempt targets, zero
  cutbacks, `822` complete sweeps, maximum `43`, all retained final gates, and
  zero retained CG statuses. The archive contains `66` accepted nonzero steps
  and `67` states including exact zero.
- The preserved comparison report incorrectly records
  `anderson_scalar.final_archive_hash_matches=false` because its generator kept
  only reduced final/header binding fields. Direct scalar-final inspection gives
  `final_archive_sha256=c2565925e599678b38d105c63e5410aacd5d87146521d361b089bcc3ba0488ed`,
  exactly the archive SHA. The original report remains immutable. The separate
  `engineering_grip_anderson_residual_matched_historical_u5p9875e3_s144.comparison.audit.json`
  records and binds this correction, with SHA256
  `758b1848f93ba3f7db5a46c0d963188ed9604d81401599889e550ac32ede0b91`.
- Across all 67 exact historical loads, the maximum differences occur at
  `u_bar=0.0059546875` for displacement and at `0.00594375` for damage/history/
  top reaction. They are `2.67634785861004e-10`, `6.04185976804672e-08`,
  `6.821109599286501e-05`, and `5.219860668148613e-06`, respectively. Endpoint
  differences are `1.960145956138304e-10`, `4.8201790447954806e-08`,
  `6.17388192836188e-05`, and `3.392785743017157e-06`. This supports collapse
  of the prior path-sampling difference, not bitwise identity or universal
  load-step convergence.
- Matched production time is directly durable at `2803.56144060893 s` and fixed
  Picard at `8765.930929828435 s`. Historical Anderson's `12703.620977 s`
  production value is a retained external tool-output reference recorded in
  task memory; only its `13038.790514 s` workflow total remains in the durable
  historical summary. The `4.531244x` historical ratio is arithmetically correct
  but must retain that provenance qualification; the directly durable Picard
  ratio is `3.126713x`.

### Exact resume point

The matched-load implementation, physical run, replay, comparison, and final
artifact audit are complete. Preserve the original comparison report and its
separate correction audit. Retain `anderson_residual` as experimental,
nonproduction evidence; do not run a higher endpoint, new schedule, or resumed
archive under this authorization. The remaining limitations are accepted-state-
only replay and aggregate hard-rejection provenance.

## 2026-09-18 — Authorize observational hard-rejection attribution

- Scope authorization: on this same active task, implement observational-only
  per-component hard-rejection diagnostics for experimental
  `anderson_residual` before the separately authorized fresh-zero strict run
  whose maximum `u_bar` is `0.0061`. This authorization covers source/tests/
  documentation and verification only; no physical continuation is to be run
  in this implementation session.
- Preserve exactly the current equations, acceptance gates, candidate
  evaluation ordering, fallback/results, legacy `anderson` interpretation,
  old artifact interpretation, and archive hash/source/config binding. Do not
  add a solve, map evaluation, mechanics evaluation, CLI setting, or new
  physical/adaptive rule.
- Each accelerated candidate must record explicit boolean/pass and the already
  computed normalized/raw metric for every hard component, plus a canonical
  ordered `failed_hard_gates` list. Distinguish mechanics-before and
  mechanics-after; distinguish force, moment, range-lower, range-upper,
  healing, history-ordering, and finite. If a quantity is not available,
  record explicit availability/null and a stage reason. JSONL records remain
  scalar-only; summaries count gates and overlaps without causal attribution.
- Extend validation/parser compatibility so old scalar files remain readable,
  while current output rejects forged lists, booleans, and metrics and checks
  internal consistency. Keep v5/schedule accepted summaries compact unless
  required by the diagnostic contract.
- Required verification includes callback on/off parity for fields/path/
  selection/solve counts, manufactured single-reason and overlap cases,
  missing/unavailable stages, forged scalar validation/report parsing, old
  scalar compatibility, no-extra-map/mechanics-solve checks, and I/O
  propagation. Use sequential CPU/x64 focused tests, legacy compilation, and
  `git diff --check`; obtain independent implementation review before the
  later physical run.

### Exact resume point

Implement and review the observational schema/parser/report changes only.
Do not run the authorized fresh-zero strict `anderson_residual` continuation
to `u_bar=0.0061` in this session; retain the existing matched-load artifacts
unchanged.

## 2026-09-19 — Complete implementation-only hard-rejection audit

- The diagnostic contract audit found and repaired one provenance defect: a
  scalar stream created before a continuation failure was bound only to the
  completed archive path, although workflow failure handling correctly finalized
  it against `*.failure.npz`. The binding now retains the completed path as the
  primary path and explicitly allows exactly the completed and failure archive
  paths; writer finalization and archive loading reject any other path.
- Attribution validation now rejects hard-gate details and failed lists without
  the canonical attribution schema. Historical streams that omit all current
  attribution fields remain readable. No equations, gates, candidate ordering,
  fallback behavior, CLI setting, physical archive, or numerical evaluation
  was changed.
- Focused coverage now includes completed/failure path binding, forged-path
  rejection, missing-schema rejection, residual hard-rejection callback parity,
  cached mechanics-before count, one mechanics-after count, zero candidate-map
  count on hard rejection, and one candidate-map count on a completed
  accelerated candidate. The README records the two allowed final archive
  outcomes.
- Verification in `.venv` with CPU/x64, run sequentially: focused diagnostic
  selection/path/count checks `11/11` in `17.735 s`; stabilized suite `57/57`
  in `362.907 s`; workflow suite `35/35` in `13.368 s` (argparse usage text is
  expected negative-test output); legacy engineering-grip/JAX/baseline suite
  `67/67` in `27.769 s` with one expected skip; `python -m compileall -q
  examples/fracture2d tests` and `git diff --check` pass.

### Exact resume point

Implementation and non-physical verification are complete. Primary Sol review
should inspect the completed/failure archive-path binding, strict attribution
schema requirement, residual callback field/selection/count parity, and
workflow summary/analysis/manifest propagation. Preserve all historical and
matched-load artifacts; do not run the authorized fresh-zero strict
`anderson_residual` continuation to `u_bar=0.0061` in this session.

## 2026-09-19 — Close archive identity pairing audit

- The scalar diagnostic binding now carries an ordered completed/failure table
  pairing each exact path suffix, archive kind, completion status, and schema.
  Writer finalization and diagnostic loading validate the stabilized NPZ scalar
  identity and schedule metadata, so adaptive and matched-load paths cannot
  cross statuses, kinds, or schemas. Historical streams without this table
  remain readable.
- The final scalar record and analysis summary expose the paired suffix, kind,
  schema, completion status, and expected binding table. The README documents
  the contract; no v5/schedule archive layout, equation, gate, candidate order,
  or numerical evaluation changed.
- Verification after the audit: stabilized contract suite `58/58` in
  `341.977 s`, workflow suite `35/35` in `17.524 s`, legacy engineering-grip/
  JAX/diagnostic/baseline suite `78/78` in `33.272 s`; compilation and
  `git diff --check` pass. Negative CLI usage text is expected. No physical
  continuation or durable artifact was run or changed.

### Exact resume point

Implementation and non-physical verification remain complete. Primary Sol
review should inspect the canonical archive-binding table, NPZ identity checks,
adaptive/schedule cross-pair rejection, final/analysis propagation, and
historical-stream compatibility. Preserve every physical artifact and do not
run the authorized fresh-zero continuation to `u_bar=0.0061` in this session.

## 2026-09-19 — Complete authoritative hard-rejection metric/parity audit

- The prior archive-pair correction was incomplete for the diagnostic contract.
  The current implementation now closes the residual-policy attribution
  semantics without changing equations, gates, candidate ordering, fallback
  behavior, archive layouts, or numerical evaluations. The authoritative
  mechanics fields are the already-computed `cg_info`,
  `free_residual_norm`, and `cg_convergence_limit` for distinct
  `mechanics_before` and `mechanics_after` entries; `passed` is the conjunction
  of `cg_info == 0` and the strict residual-limit check.
- The exact optional attribution schema is
  `fracture2d-engineering-grip-anderson-residual-hard-gates-v1`, with canonical
  order `finite`, `mechanics_before`, `mechanics_after`, `force`, `moment`,
  `damage_range_lower`, `damage_range_upper`, `healing`, and
  `history_ordering`. Every entry contains exactly `available`, `passed`,
  `raw`, `limit`, `normalized`, `ratio_available`, `metric_reason`,
  `stage_reason`, `cg_info`, `cg_status_pass`, and `residual_limit_pass`.
  Force/moment use their existing gate error and scale; range lower/upper use
  their existing raw excesses and `damage_range_atol`; healing uses its raw
  excess and `healing_atol`; history ordering uses raw excess with zero limit
  and no ratio. Finite is boolean-only. Unavailable stages have false pass,
  null metrics/CG fields, and a canonical stage reason.
- `failed_hard_gates` is the ordered subset of entries whose `passed` flag is
  false, and `hard_pass` is its exact complement. Current validation rejects
  missing attribution schema, forged booleans/lists, non-canonical reasons,
  inconsistent raw/limit/ratio or CG provenance, unavailable numeric metrics,
  and forged v5 residual-policy selection metrics. Historical scalar streams
  that omit all attribution fields remain readable; historical `anderson`
  semantics and artifacts are not reinterpreted.
- Callback on/off manufactured parity is exact for accepted fields, attempt
  path, candidate selection, and mechanics/damage/candidate-map solve counts.
  A hard rejection retains the cached mechanics-before count, one
  mechanics-after evaluation, and zero candidate-map evaluations; a completed
  accelerated candidate records one candidate-map evaluation. Attribution is
  observational and adds no solve or map evaluation.
- Verification in `.venv` with CPU/x64, run sequentially: stabilized `59/59`
  in `347.750 s` and workflow `35/35` in `14.350 s`; expected argparse usage
  text is negative-parser-test output. No physical continuation or durable
  artifact change was performed; compilation and `git diff --check` pass.

### Exact resume point

Implementation and non-physical verification are complete and ready for
independent primary review. Inspect the authoritative CG/metric derivation,
strict forged-metric rejection, callback parity/counts, and old-stream
compatibility. Preserve all physical artifacts and do not run the authorized
fresh-zero continuation to `u_bar=0.0061` in this session.

## 2026-09-19 — Independent contract review follow-up

- The focused validator, old-stream compatibility, callback-parity, and
  no-extra-map/count tests were rerun after the implementation audit: `3/3`
  scalar-contract tests passed in `0.004 s`, and the two manufactured callback
  tests passed in `33.179 s`. `git diff --check` also passes.
- The README attribution paragraph was corrected to enumerate the exact eleven
  per-gate fields and the authoritative CG/raw-limit semantics; no numerical
  source or physical artifact changed.
- Review point for primary Sol: the existing force/moment hard-gate flags are
  conjunctions of the before/after stage gates, while their scalar attribution
  currently serializes the aggregate maximum error and aggregate maximum
  scale. Under unequal before/after scales, that conjunction can differ from
  `raw <= limit`, but the strict scalar validator requires those values to
  agree. No production mismatch was observed in the focused manufactured
  cases; primary review must decide whether the contract should retain the
  aggregate representation, serialize the worst stage ratio, or explicitly
  preserve the conjunction as additional provenance. No formulation or
  physical-run decision was made here.

### Exact resume point

Keep all source numerics and physical artifacts unchanged. Primary Sol should
resolve the force/moment aggregate-versus-stage metric contract, then either
approve the current implementation or route a narrowly scoped diagnostic-only
correction. Do not run the authorized fresh-zero continuation to
`u_bar=0.0061` in this session.

## 2026-09-19 — Implement approved stage-consistent force/moment attribution

- The approved observational fix is implemented only in the stabilized scalar
  `anderson_residual` attribution path. The production before/after mechanics,
  force, and moment conjunctions, solver ordering, candidate selection,
  fallback behavior, and legacy `anderson` semantics are unchanged.
- Attribution schema v2 records `stage_metrics.before` and
  `stage_metrics.after` for force and moment. Each stage serializes the already
  computed error, stage scale, tolerance limit, normalized ratio, and pass
  flag. The aggregate normalized value is the worst stage ratio, aggregate pass
  is the exact conjunction, and aggregate raw/scale/limit is the tuple from the
  worst-ratio stage with deterministic `before` tie-breaking. The validator
  enforces this self-consistent representative contract and rejects forged
  stage or aggregate values. Missing-attribution historical scalar streams
  remain readable.
- Focused manufactured coverage now exercises unequal stage scales, aggregate
  ratio/pass parity, callback/no-callback solve-count parity, unavailable-stage
  handling, and stage/aggregate forgery rejection. The README and decision log
  record the scalar contract. No physical continuation or artifact rewrite was
  performed.
- Verification in `.venv` with JAX x64 enabled, sequentially: stabilized and
  workflow suites pass `95/95`; the existing old residual scalar stream loads
  without attribution; `python -m compileall -q examples/fracture2d tests` and
  `git diff --check` also pass.

### Exact resume point

Implementation and non-physical tests are complete. Primary Sol should inspect
the stage metric derivation against `_mechanics_gate`, the worst-ratio
representative contract, strict validator/forgery coverage, and callback parity.
Preserve all physical and historical artifacts; do not run the authorized fresh
zero `anderson_residual` continuation to `u_bar=0.0061` in this session.

## 2026-09-19 — Close attribution review and run exact 0.0061 endpoint

- Independent review found and the implementation fixed three attribution-only
  defects before physical execution: force/moment stage-scale aggregation,
  callback handling of numerical candidate-map failure, and valid zero-limit
  force/moment gates. Schema v3 also binds candidate-map status to accelerated
  failure context and rejects schema-less attributed records at both writer and
  loader boundaries. No solver equation, hard gate, evaluation order, candidate
  selection, or fallback was changed.
- Final non-physical verification passed `98/98` stabilized/workflow tests,
  including callback/no-callback field and solve-count parity, zero-limit mixed
  stage ordering, forged context, and durable-boundary rejection. Compilation
  and `git diff --check` passed. Independent review returned GO.
- The authorized fresh-zero strict adaptive run then completed at exact
  `u_bar=0.0061` with `du_initial=1e-4`, `du_min=6.25e-6`, 144 maximum stagger
  iterations, residual Anderson history 5/beta 0.5, and scalar diagnostics.
  It accepted 61 steps with zero cutbacks, 654 total accepted sweeps, maximum 62
  sweeps at `0.0061`, and elapsed production time `2329.8819416849874 s`.
- Accepted-state replay passed with 62 states and 62 solver-summary rows. The v5
  archive SHA256 is
  `68c934008ccf95e71ac2d4ac78f1fa82fd16534cc6b2accba0b21d21c60021f8`;
  the v3-aware scalar stream SHA256 is
  `9d1c740d7565099f514ac0351fce61d960551461ddc9aff1f1dc68c7d9ebfb75`.
  Rejected field paths remain unavailable and were not claimed as replayed.
- The scalar stream contains 817 attributed proposals: 593 accelerated and 224
  Picard fallback proposals. Of the accelerated proposals, 430 were selected,
  142 were rejected only by the healing hard gate, and 21 were rejected by
  sufficient decrease. All 224 Picard fallback proposals were selected. No
  finite, mechanics-before/after, force, moment, damage-range, history-ordering,
  or candidate-map numerical failure was recorded.
- The physical curve remains pre-peak at the authorized boundary. Top reaction
  rose from `639.1241866042104` at `0.0060` to `641.3384950971831` at `0.0061`;
  maximum damage reached `0.8202621313922084`. The run therefore does not yet
  establish a reaction peak or post-peak branch.
- Durable artifacts are rooted at
  `/root/autodl-tmp/data/fracture2d/engineering_grip/engineering_grip_anderson_residual_peak_u6p1e3_s144`.

### Exact resume point

The exact `0.0061` authorization is exhausted. Preserve the completed archive,
scalar stream, manifest, replay, analysis, plots, and derived curve. Do not run
to `0.0062`, `0.0063`, `0.0075`, or any higher endpoint without explicit new
authorization. The next decision is whether to approve a narrowly bounded
fresh-zero continuation to locate and validate the actual reaction peak; the
current evidence remains monotone pre-peak.

## 2026-09-19 — Authorize exact 0.0062 fresh-zero extension

- The user explicitly selected exact `u_bar=0.0062` as the next endpoint.
- Run fresh-zero strict adaptive `anderson_residual` with the same source,
  physical parameters, tolerances, `du_initial=1e-4`, `du_min=6.25e-6`,
  `max_stagger_iterations=144`, Anderson depth 5/beta 0.5, and v3 scalar
  diagnostics used by the validated `0.0061` run.
- Stop on any strict/structural failure and do not proceed to `0.0063`.

### Exact resume point

Verify that the new run stem is unused, then execute the authorized fresh-zero
run to exact `0.0062`, independently replay it, and compare the final reaction
increment with the `0.0061` evidence. Preserve all earlier artifacts.

## 2026-09-19 — Repair failed-run gate-ratio provenance

- The authorized exact-`0.0062` fresh-zero run is exhausted by strict failure
  after committing `u_bar=0.0061125`; its physical NPZ and scalar diagnostic
  evidence remain preserved.
- The workflow analysis now uses each stored solver-summary row's
  `sweep_index` and authoritative `load_target`/`target_u_bar` pair for gate
  ratio and mechanics-proxy locations. It rejects missing, mistyped,
  inconsistent, non-finite, and out-of-range provenance instead of using the
  diagnostic-row enumeration.
- Manufactured completed and failure coverage uses stored sweep indices `7`
  and `13`; the failure archive analysis reports the authoritative final
  `sweep_index=143` and target `0.006118750000000029` rather than the prior
  enumerated `63`.
- The existing derived failure analysis and embedded summary reports were
  regenerated from the unchanged failure archive. No NPZ or scalar stream was
  overwritten.
- Verification in `.venv` with CPU/x64: workflow suite `36/36` in `18.781 s`,
  stabilized suite `63/63` in `358.597 s`,
  `python -m compileall -q examples/fracture2d tests`, and `git diff --check`
  pass. Expected argparse usage text appears in negative parser tests.

### Exact resume point

Implementation and non-physical verification are complete. Primary Sol review
should inspect the strict provenance validator, completed/failure regression
coverage, and regenerated derived report locations. Preserve the failed NPZ and
scalar evidence; do not run or authorize a continuation to `0.0063` or higher
without a new decision.

## 2026-09-19 — Audit and close the exact 0.0062 attempt

- The fresh-zero strict run attempted targets `0.0062`, `0.00615`, and
  `0.006125`, accepted the cutback target `0.0061125` after 71 sweeps, then
  rejected `0.006125` and the final `du_min` target `0.00611875`. It stopped as
  required with 62 accepted steps and 67 attempts; no warning state was
  committed.
- The last accepted top reaction is `641.3882615552535`, up
  `0.04976883680546962` from the accepted `0.0061` state. Maximum accepted
  damage is `0.8351609589466233` and maximum history is
  `487.68458516893867`. Every accepted reaction increment remains positive, so
  no validated peak or post-peak state exists.
- The final rejected target failed after 144 sweeps at `du_min=6.25e-6` on
  displacement, damage, history, reaction, damage-residual, fixed-point, and
  coupled convergence. Its finite, mechanics-before/after, force, moment,
  damage-range, healing, and history-ordering hard prerequisites passed. The
  rejected reaction is diagnostic only and is not accepted physical evidence.
- The scalar stream has 2,257 attributed proposals: 1,337 accelerated and 920
  Picard fallback. There were 555 accelerated hard-gate rejections, all solely
  healing; 256 additional accelerated candidates passed hard gates but failed
  globalization. Candidate-map statuses are 782 completed, 1,475 not requested,
  and zero failed.
- Independent accepted-state replay and field-equation checks pass, while
  `path_replayed=false` and `rejected_trial_path_available=false` remain
  explicit. Failure archive SHA256 is
  `979f66ab19b9f74a52871cfeb56fa2f3efd57bd53197a40d9b516f159299c1ad`;
  scalar stream SHA256 is
  `1b9dc58a2eed2ee5ddf0fb063e339a55488949923995e57427c57670ace129d4`.
- After correcting derived gate-ratio provenance, failure analysis and embedded
  summary use stored `sweep_index=143`. The manifest was rebound to those two
  regenerated files; all 9 listed artifacts pass hash/size audit. Corrected
  manifest SHA256 is
  `efd1970ead9b685d99980df66b46eba33e1f82c890536dd2e4e1bf9b578bbd5a`.
- Final verification passes 99/99 stabilized/workflow tests, compilation, and
  `git diff --check`.

### Exact resume point

The exact-`0.0062` authorization is exhausted by strict failure. Preserve the
failure NPZ, scalar stream, corrected derived reports, replay, plots, and
manifest. Do not resume this archive, change `du_min`, warning-accept, or run to
`0.0063` without a new explicit decision. The next technical decision is whether
to investigate a different nonlinear strategy near `0.0061125`; the current
accepted evidence does not establish a reaction peak.

## 2026-09-19 — Persistent final-solution authorization and nonlinear diagnosis

- The user now explicitly requests continued work until the nonlocal direct
  solver obtains its final strict solution. The task target is the configured
  exact `u_bar=0.0075` curve with a validated accepted peak/post-peak branch.
  This supersedes earlier per-endpoint authorization holds but does not weaken
  rollback, hard gates, replay, or evidence requirements.
- Read-only numerical diagnosis of the `0.00611875` failure found a marginally
  noncontractive sawtooth map rather than subproblem failure. The accepted
  `0.0061125` tail contracted by about `0.598`; the rejected target's tail factor
  is about `1.0073`, with 81/143 residual ratios increasing. Fixed-history damage
  and mechanics remain solved, while coupled transition/history quantities do
  not converge.
- At the final target, 80 accelerated candidates failed healing hard gates and
  58 failed sufficient decrease; only 5 accelerated candidates were selected.
  Anderson history stayed at depth 1 for 271/282 proposal rows, so increasing
  nominal depth is not evidence-based.
- External references support coupled BFGS only for separately declared coupled
  residual formulations. Neither Kristensen-Martinez-Paneda (2020) nor Bie et
  al. (2023) supplies a tangent or arc-length constraint for this repository's
  committed-history nonlocal AT2 map. Direct transplantation is therefore not
  authorized.
- The next bounded diagnostic is an existing fixed-Picard `omega=0.25` one-step
  checkpoint from the independently replayed `0.0061125` state to exact
  `0.00611875`, retaining 144 sweeps and every strict gate. It must not overwrite
  or promote historical artifacts.

### Exact resume point

Construct and independently review the bounded one-step checkpoint so it consumes
the replay-validated accepted state without mutating the source archive. Run
`omega=0.25` to exact `0.00611875`; record contraction and gate results. Only if
it strictly accepts should a distinct robust controller be specified and tested
for a fresh-zero full curve through `0.0075`.

## 2026-09-19 — Complete bounded fixed-Picard omega-0.25 checkpoint

- The authorized diagnostic rebuilt the production model and `cuda:0` JAX
  adapter, reconstructed the immutable accepted state at
  `u_bar=0.0061125000000000285`, and invoked exactly one private
  `_run_stabilized_attempt` to `u_bar=0.006118750000000029`.
- The fixed policy used `omega_initial=omega_min=0.25`, zero backtracks, no
  sufficient-decrease test, no cutback, and the unchanged 144-sweep ceiling.
  It produced 144 complete sweeps and was not accepted. The final fixed-point
  error was `0.0032474112686108336`; the consecutive error ratios ranged from
  `0.9857316138435906` to `1.075848052697296`, with the final ratio
  `1.075848052697296`. Neither the full sequence nor its last ten ratios was
  strictly contractive.
- The final failed gates were displacement, damage, history, reaction,
  damage-residual, damage-fixed-point, and coupled. Finite, mechanics-before,
  mechanics-after, force, moment, damage-equation, range, healing,
  history-ordering, and globalization gates passed. Final top reaction was
  `641.3531241760143`; final damage/history maxima were
  `0.8666784148051708` and `667.8849496022742`.
- The only new durable artifact is the scalar diagnostic
  `/root/autodl-tmp/data/fracture2d/engineering_grip/`
  `engineering_grip_fixed_picard_omega025_one_step_u6p11875_20260919T072222Z.json`.
  Its file SHA256 is
  `2fe6924d8ac2e98f77e78a1b07e41ec7941d52e7c5eb6815185c3951b34cc82e`; the
  result-payload SHA256 recorded inside it is
  `6c6c28e3d24b2427f89abc8fa88e32b5ae7fa9ed537c970bd649e8ad31e41495`.
- The source failure archive and material cloud were hash-identical before and
  after the attempt. No NPZ, continuation state, rejected field vector, or
  repository source file was written or modified. The full `0.0075` curve and
  any distinct robust controller remain unrun pending review of this negative
  checkpoint.

### Exact resume point

Review the scalar checkpoint and decide whether to stop the current nonlinear
strategy or authorize a separately derived/implemented controller. Do not infer
acceptance or a physical peak from this rejected diagnostic and do not resume
the source archive.

## 2026-09-19 — Approve implementation contract for backtracked residual Anderson

- The new policy is distinct from historical `anderson_residual`. It retains the
  existing Type-II depth-5/beta-0.5 proposal and exact omega-0.5 Picard fallback,
  but line-searches the accelerated direction at alpha
  `1, 1/2, 1/4, 1/8, 1/16` before fallback.
- Every alpha trial is finite-shape checked, then evaluated by the existing
  mechanics/history/hard gates. Failed hard gates skip candidate-map evaluation.
  A hard-passing trial computes the exact committed-history map residual and
  must satisfy `m_trial <= (1 - 0.005*alpha)*m_reference`.
- Hard-gate, map-evaluation, and merit rejection are trial-local and preserve
  valid base map pairs. Proposal-construction QR/rank/triangular/nonfinite
  failures restart history. Candidate-map numerical failure is recorded and
  continues to the next alpha. Structural or diagnostic-I/O failure propagates.
- Accelerated trials receive explicit nullable `trial_alpha`; historical
  `omega=None` remains unchanged. The new policy receives a distinct merit
  identity and strict CLI/archive/replay validation.

### Exact resume point

Implement the policy and manufactured tests without running a physical curve.
Require callback parity, exact solve counts/order, no candidate-map solve after a
hard rejection, retained-history behavior, proposal-construction restart,
trial-local map failure, fallback identity, forged-alpha rejection, old archive
compatibility, compilation, and `git diff --check`. Then obtain independent GO
before the one-step `0.0061125 -> 0.00611875` checkpoint.

## 2026-09-19 — Complete backtracked residual Anderson implementation verification

- The implementation-only verification exposed and repaired one unrelated
  schedule serialization regression: `StabilizedLoadSchedule.payload()` built its
  canonical mapping but did not return it. This caused scheduled scalar-final
  binding and scheduled completed/failure archive writes to receive `None`.
  The fix restores the existing canonical payload contract without changing any
  schedule or solver semantics.
- The three previously failing scheduled regressions now pass: scalar header/final
  binding, scheduled strict-failure rollback/archive writing, and scheduled
  completed archive round-trip.
- Sequential `.venv` verification passes: the stabilized module `72/72` in
  `400.715 s`, the workflow module `36/36` in `15.922 s`, and the combined
  stabilized/workflow command `108/108` in `411.310 s`. Expected argparse usage
  output is emitted by negative parser tests. `python -m compileall -q
  examples/fracture2d tests` and `git diff --check` also pass.
- No physical continuation, schedule extraction, archive overwrite, or durable
  simulation artifact change was performed.

### Exact resume point

Implementation and non-physical verification are complete. Primary Sol review
should inspect the backtracking alpha schedule/threshold, trial-local numerical
failure handling, retained-history and solve-count parity, explicit new-policy
alpha provenance, and preservation of historical archive payloads. Do not run
the one-step `0.0061125 -> 0.00611875` checkpoint or any physical curve until
that review gives independent GO.

## 2026-09-19 — Independent re-review of backtracked residual Anderson

- The two prior high-severity findings are closed. v5 summaries now require
  the exact alpha/backtrack pairing, and scalar proposal records require the
  exact alpha/index pairing. Finite candidate failures now emit authoritative
  `not_evaluated` attribution with zero mechanics-after and candidate-map
  counts; numerical map failures remain distinct and trial-local.
- `summarize_anderson_scalar_diagnostic` excludes `not_evaluated` rows from
  physical failure/overlap counts while retaining per-gate unavailable counts.
  Historical streams without attribution remain readable, and legacy
  `anderson` payloads remain unchanged.
- Verification in `.venv`, CPU/x64, without a physical run: focused review
  tests `6/6` in `51.599 s`; stabilized/workflow suites `110/110` in
  `454.069 s`; legacy engineering-grip/JAX/baseline suites `67/67` in
  `31.074 s`; compilation and `git diff --check` pass.

### Exact resume point

No implementation-level blocker was found. Recommend GO only for the separately
bounded physical checkpoint; do not infer approval for a full curve or alter
the existing physical artifacts. Preserve this review record and obtain the
primary Sol decision before execution.

## 2026-09-19 — Prepare bounded backtracking checkpoint runner

- Added the standalone scratch runner
  `/tmp/opencode/fracture2d_anderson_residual_backtracking_checkpoint.py`.
  It verifies the source archive/cloud hashes and immutable accepted-state
  digest, round-trips the historical base/stabilized payloads, constructs the
  exact new backtracking configuration, and calls
  `_run_stabilized_attempt` exactly once from the accepted
  `u_bar=0.0061125000000000285` state to
  `0.006118750000000029`.
- The runner does not call the fresh-zero continuation wrapper, does not
  resume or mutate the source archive, and does not serialize rejected trial
  field vectors. It captures every scalar callback event and compact scalar
  sweep/trial diagnostic, summarizes alpha selection, gate/map statuses,
  evaluation counts, and contraction, then uses an exclusive timestamped JSON
  output with a canonical payload digest.
- Static verification only: `python -m py_compile` passes; an AST check finds
  exactly one private attempt call and no continuation-wrapper call; archive
  metadata inspection confirms 63 states, 62 accepted steps, strict final
  source status, historical residual policy, and the 144-sweep base ceiling;
  `git diff --check` passes. No physical attempt or output JSON was created.

### Exact resume point

Primary Sol should review the scratch runner's source-state/hash checks,
backtracking configuration, callback/event pairing, scalar-only serialization,
exclusive output/digest contract, and the fact that it invokes exactly one
`_run_stabilized_attempt`. Only after that review should the runner be
executed; do not launch a full curve or alter any historical artifact.

## 2026-09-20 — Hold checkpoint pending target reconciliation

- Read-only primary review found a target mismatch in the scratch runner:
  `TARGET_LOAD=0.006115625000000029` and
  `EXPECTED_DIAGNOSTIC_DU_MIN=3.125e-6`, so it would test the half-step from
  `0.0061125`.
- The active task, `PROGRESS.md`, and the preceding checkpoint contract specify
  the one-step target `0.006118750000000029` from the same committed state. The
  latter increment is `6.25e-6`; these checkpoints cannot be silently treated as
  equivalent.
- Static review otherwise confirms one private attempt call, no continuation
  wrapper, source/archive/cloud hash checks, scalar-only output, callback/event
  pairing, and exclusive canonical JSON output. Backtracking-focused tests pass
  `10/10`; no physical attempt or output JSON was created.

### Exact resume point

Clarify whether the authorized checkpoint is the runner's half-step target
`0.006115625` or the active-contract target `0.00611875`. Do not execute the
runner or change its constants until that numerical target and increment are
explicitly reconciled; preserve all historical artifacts unchanged.

## 2026-09-20 — Close bounded checkpoints and strict fresh-zero runs

- The bounded same-target `anderson_residual_backtracking` checkpoint from
  `u_bar=0.0061125` to exact `0.00611875` failed after 144 sweeps. Its JSON
  artifact is
  `/root/autodl-tmp/data/fracture2d/engineering_grip/engineering_grip_anderson_residual_backtracking_one_step_u6p11875_20260919T145746Z.json`
  with file SHA256
  `a6eddf1b01fd56e0069b99e19e166b9b9881701309b2ee206753b3256b807c16` and
  payload SHA256
  `ff9e65973fbc15982bc66c74961c07396209701fae7cc4d152e9918ce93d2c85`.
  The final failed gates were displacement, damage, history, reaction,
  damage residual, damage fixed point, and coupled. Source hashes were
  unchanged; no rejected state is evidence.
- The bounded half-step from `0.0061125` to `0.006115625`, with
  `du_min=3.125e-6`, was accepted in 30 sweeps with every gate passing and a
  contracting tail. Its JSON artifact is
  `/root/autodl-tmp/data/fracture2d/engineering_grip/engineering_grip_anderson_residual_backtracking_half_step_u6p115625_20260919T161030Z.json`
  with file SHA256
  `0d01376680c4b58431bc0968bdbf2da60eddea80e580ba6104b9dc1a06e7e0eb` and
  payload SHA256
  `f2d2d31ba76ab42b5593e0b9dacae3346d105cd60ecaeec9c097acaf38459292`.
- The fresh-zero strict production run completed exact `u_bar=0.0062` with
  62 accepted steps and zero cutbacks. Replay and field-equation checks pass.
  The archive SHA256 is
  `95851472eb43eb2e24e4d26920d050ecf5ffc1aa83bfe0919f33bd7881cadada`; the
  scalar SHA256 is
  `14765f28710434b3ae13dc6a459b5afcdd15d2106a3164c69ae7c045297e6976`.
  Accepted peak reaction at `0.0061` is `641.3384921076807`, and the
  post-peak `0.0062` reaction is `631.4074774212969`. Final damage is
  `0.999917` at notch-tip particle `1869`, with history `1.1984e6`.
- The accompanying conditioning study at `k_res=0` remains a physical caveat:
  the free system is SPD with `lambda_min=0.2696879587`, its modes are
  crack-localized, and direct residual correction is tiny. This supports
  converged discrete equilibrium, not an unconditional physical
  interpretation of the near-singular crack-tip response.
- The fresh-zero strict `u_bar=0.0075` attempt stopped and rolled back at the
  `du_min` target `0.006234375` after 144 sweeps. The last accepted load was
  `0.00623125`, with 64 accepted steps and 70 attempts. Its failure archive
  SHA256 is
  `a1743d0e4565670243481109064bed6e3044c5c14a481dc97faaecda327f6700`.
  Final failed gates were displacement, damage, history, reaction, damage
  residual, damage fixed point, and coupled; no rejected state is evidence.
- Already-run verification records `110/110` relevant tests, `67/67` legacy
  tests from independent review, `python -m compileall -q examples/fracture2d
  tests`, and `git diff --check`. These tests are not claimed as rerun after
  the physical artifacts; only the noted independent replay/field-equation
  reviews are post-artifact checks.

### Blocker

There is no non-arbitrary regularization parameter or update rule for the
proposed Tikhonov Type-II Anderson path. Smaller `du_min` produces impractical
numbers of microsteps and is not a final robust strategy. Do not guess `mu`,
`k_res`, or an arc-length rule.

### Exact resume point

Obtain the required user/mathematical decision: approve a bounded
regularization-parameter study/derivation, or accept the current strict
discrete curve only through `u_bar=0.00623125` as incomplete. Preserve all
checkpoint JSON, completed `0.0062`, and failed `0.0075` artifacts unchanged;
do not use a rejected state as physical evidence.

## 2026-09-20 — Prepare fixed-mu scalar sensitivity runner

- Added the scratch runner
  `/tmp/opencode/fracture2d_anderson_residual_regularized_sensitivity.py`.
  Its required `--mu` CLI accepts only `0.25`, `1`, and `2`, binds the exact
  source archive/cloud hashes and last accepted digest at
  `u_bar=0.0062312500000000345`, round-trips the historical base and residual
  backtracking payloads, and changes only the regularized policy identity and
  explicit `mu`.
- Current bound hashes are archive
  `a1743d0e4565670243481109064bed6e3044c5c14a481dc97faaecda327f6700`, source
  cloud
  `fa2ebf165f2e89ec88c63116346401b5039263e8e4c1a1917f6f13a3f4541267`, source
  state digest
  `a5e0e6b78e1663d6e76262950730dcb5ee3a5a7bdc5c1f0d86f2b09adf7d9cd7`, and
  runner
  `d6a2744cdf77212d0bfc16a863096657a8c300b121e8d31ed6f362e7c2749d38`.
- The runner constructs the production engineering-grip model and restricted
  JAX adapter, invokes `_run_stabilized_attempt` exactly once at
  `0.006234375000000035` with zero load cutbacks, captures scalar sweep/callback
  regularization, alpha, gate/map, selection/rejection, and contraction data,
  and writes only an exclusive timestamped canonical JSON if explicitly run.
- Static/non-physical checks pass: source state/config/current-summary
  validators, Python compilation, and AST one-attempt/no-continuation review.
- A non-executing API/CLI/serialization review also passes; source-bundle and
  workflow hashes are captured before the private attempt and checked again
  afterward rather than being read twice after the attempt.
  No physical attempt and no sensitivity JSON output were created.

### Exact resume point

Primary Sol should review the runner's exact source digest/cloud/model/adapter
bindings, the policy-only configuration diff, scalar-only callback contract,
rollback/acceptance checks, and exclusive payload-hash output. Do not execute
the runner or claim the bounded study complete until that review is explicit.

## 2026-09-20 — Complete negative fixed-mu sensitivity study

- The scalar-only bounded study used the injected accepted state at exact
  `u_bar=0.0062312500000000345`, targeted exact
  `u_bar=0.006234375000000035`, kept `du_min=3.125e-6`, and allowed the full
  `144` stagger sweeps with zero load cutbacks. Each of the three attempts was
  rejected after all 144 sweeps by the same final gates:
  `displacement`, `damage`, `history`, `reaction`, `damage_residual`,
  `damage_fixed_point`, and `coupled`. Each attempt rolled back to the source
  accepted-state digest; no rejected field vector was serialized or promoted.
- Exact diagnostic artifacts and hashes are:

  | `mu` | JSON artifact | file SHA256 | payload SHA256 |
  | ---: | --- | --- | --- |
  | `0.25` | `/root/autodl-tmp/data/fracture2d/engineering_grip/engineering_grip_anderson_residual_regularized_backtracking_sensitivity_mu0p25_20260920T072155720312Z.json` | `a2d6afa5a5d86866d8de20ec37987873d243eb910941629869e363c31db787c6` | `b5235a04b89ed84b1c63e92b3becf41930e1914945de829e512333e49eea8e46` |
  | `1` | `/root/autodl-tmp/data/fracture2d/engineering_grip/engineering_grip_anderson_residual_regularized_backtracking_sensitivity_mu1_20260920T075026856378Z.json` | `c5f271504e3617b0e8d08683f958921fe045276a627739e3a444ddc86f5479ec` | `31353edbc352f33fafd6fc67f9a063a29f45fb7c30786ffab085ef08c89e353a` |
  | `2` | `/root/autodl-tmp/data/fracture2d/engineering_grip/engineering_grip_anderson_residual_regularized_backtracking_sensitivity_mu2_20260920T081716835642Z.json` | `c26a461ac64f7d54dc7d37b597f8064f9dedb69754d7394fd2dea67d9523d09a` | `9b5685ba06c0742d8dc27256897349d2b2ae0289010f925fce90d4c25e6ea3cb` |

- Selected AA/Picard-fallback counts were respectively `3/141`, `0/144`, and
  `0/144` for `mu=0.25`, `1`, and `2`. The maximum Anderson coefficient norms
  were `0.7697`, `0.3656`, and `0.2151`. Final fixed-point errors were exactly
  `1.8632048281652658e-4`, `1.8608049513912928e-4`, and
  `1.8608033904232713e-4`, compared with the baseline
  `1.8391723755628586e-4`.
- The source-bundle hashes differ by design because the regularized policy code
  changed. Source cloud, configuration, and rollback bindings match across the
  comparison, so the result is behavioral evidence rather than bitwise code
  parity. The study remains diagnostic-only and does not establish a physical
  accepted endpoint or a convergence claim.

### Blocker

Do not expand or adapt the `mu` grid. The regularization suppresses AA toward
Picard and worsens the endpoint. The exact next decision is either to test the
unchanged mathematics with `du_min=1.5625e-6`, which may require hundreds of
microsteps, or to derive a materially new continuation/nonlinear formulation.
Do not guess an arc-length rule or `k_res`.

### Exact resume point

Preserve all three sensitivity JSON artifacts, the source failure archive, and
all prior checkpoint evidence unchanged. Obtain the user/mathematical decision
between the smaller-`du_min` diagnostic and a materially new continuation or
nonlinear derivation. Do not run a physical solve, expand/adapt `mu`, resume an
archive, warning-accept, or treat any rejected state as physical evidence.

### Verification

This documentation-only update changes no solver or test files and runs no
physical solve. `git diff --check` passes; it is the only verification run for
this memory update.

## 2026-09-20 — Close smaller-du checkpoint and deeper-alpha audit

- The user-approved bounded checkpoint started from the last accepted
  `u_bar=0.00623125` state with source digest
  `a5e0e6b78e1663d6e76262950730dcb5ee3a5a7bdc5c1f0d86f2b09adf7d9cd7`, used
  the same unregularized `anderson_residual_backtracking` policy, targeted
  `u_bar=0.0062328125`, used `du_min=1.5625e-6`, and allowed `144` sweeps.
  It ended in strict failure and rollback; no rejected state was accepted.
- The diagnostic artifact is
  `/root/autodl-tmp/data/fracture2d/engineering_grip/engineering_grip_anderson_residual_backtracking_half_step_u6p2328125_20260920T094632Z.json`.
  Its file SHA256 is
  `28b3e0a751781b5f043ea2c2a2ed43852388a2b783e3d23c19bd9fb702cd0ba3`, and
  its payload SHA256 is
  `e6b6434f6558afbc2143d58a3d7337ef04ffffd99eccc2b0d91f8845d3067159`.
  The final error was `4.779188545345203e-5`; the tail ratio was `1.0010643`
  and the maximum of the last ten ratios was `1.0012019`. The same coupled
  gates failed: `displacement`, `damage`, `history`, `reaction`,
  `damage_residual`, `damage_fixed_point`, and `coupled`.
- The deeper-alpha `alpha=1/16` audit found that the last 30 trials were all
  hard-pass/map-complete, but only sweep `125` passed globalization. The
  directional slopes `(m(alpha)-m_ref)/alpha` were positive in `29/30` trials,
  with large tail slopes. `alpha=1/8` was always worse. This is NO-GO evidence
  against a blanket deeper-alpha extension.

### Blocker

The current controller path is exhausted: fixed Picard `omega=0.25`,
unregularized alpha backtracking, fixed-`mu` `0.25/1/2`, smaller `du_min`, and
deeper-alpha evidence all fail to provide a strict route past
`u_bar=0.00623125`. Do not guess further damping or regularization. The exact
next decision requires selecting a materially new nonlinear formulation (for
example, an independently derived monolithic complementarity/semismooth-Newton
method or a derived continuation constraint) or stopping incomplete at
`0.00623125`; the exact `0.0075` objective remains unmet.

### Exact resume point

Preserve the checkpoint JSON and every prior archive/diagnostic unchanged. Wait
for the user/mathematical selection of a materially new formulation versus
stopping incomplete. Do not run another damping, regularization, or blanket
deeper-alpha experiment, resume an archive, warning-accept, or treat the
rejected checkpoint as physical evidence.

### Verification

This memory-only update changes no solver or test files and runs no physical
solve. `git diff --check` passes.

## 2026-09-20 — Implement monolithic Phase 1 (residual + analytic JVP)

- The user selected the monolithic direction and authorized Phase 1 only: the
  exact code-compatible monolithic `q-d-H` residual for the engineering grip
  and its analytic generalized JVP, plus manufactured tests. The mathematical
  contract is recorded in `docs/DECISIONS.md` (2026-09-20 entry) and forbids
  any Newton solver, GMRES, workflow/CLI wiring, or physical run in this phase.
- Target module: `examples/fracture2d/nonlocal_engineering_grip_monolithic.py`
  (private/experimental). Target tests:
  `tests/test_fracture2d_nonlocal_engineering_grip_monolithic.py` on the
  existing 5x5 manufactured engineering-grip fixture.
- Required manufactured coverage: exact `Rq` versus the authoritative
  `A2FixedDamageView` free residual; independent dense `Rd` and zero residual
  at the existing fixed-history damage solve; `RH` below/equal/above branches
  with the exact tie committed to the history row; centered FD checks of
  `Jqq`, `Jqd`, `Jdd`, `JdH`, `JHq`, and the Amor first variation including
  trace `<0`, `>0`, `=0`; full packed JVP versus centered FD with a fixed
  active set plus explicit dense nonsymmetry evidence; and loud
  shape/dtype/nonfinite/mutation contract failures.

### Exact resume point

Implement the module and tests, then run the focused monolithic suite, the
affected engineering-grip suites, `compileall`, and `git diff --check`. Do not
implement a solver or run any physical continuation.

## 2026-09-20 — Complete monolithic Phase 1 implementation

- `examples/fracture2d/nonlocal_engineering_grip_monolithic.py` implements the
  approved Phase-1 contract as a private/experimental
  `MonolithicGripResidual` bound to one `EngineeringGripModel`, target
  `u_bar`, and committed history. It exposes `pack`/`unpack`/
  `reconstruct_displacement`, the packed residual `(Rq, Rd, RH)`, `psi_plus`,
  the exact strict `>` `active_mask`, the analytic block actions
  (`jqq_action`, `jqd_action`, `jdd_action`, `jdh_action`, `jhq_action`), the
  packed `jvp`, exact adjoint `vjp`, and a manufactured-only
  `dense_jacobian` helper. `Rq` reuses the authoritative `A2FixedDamageView`
  residual/tangent verbatim; `Jqd` reuses the exact `_endpoint_residual`
  assembly through a center-factor-only shim with the derived
  `dW/dd = -2 V^m (1-d)`; `JHq` uses the existing `_delta_gradient`
  kinematic linearization and the embedded Amor first variation
  `[K max(tr,0) I2 + 2 mu dev_2x2] : dstrain` (`K = lambda + 2 mu/3`), which
  is single-valued at `tr = 0`. The tie is committed (inactive), `Jdq = 0`,
  `JHd = 0`, and no damage obstacle/clipping/smoothing exists anywhere.
- `tests/test_fracture2d_nonlocal_engineering_grip_monolithic.py` covers, on
  the existing 5x5 manufactured engineering-grip cloud: exact `Rq` versus the
  authoritative residual and constraint reconstruction; `Rq`'s `Jqq` exact
  equality with the existing tangent action; independent dense `Rd` plus
  bitwise equality with `engineering._damage_residuals`'s volume residual and
  a zero `Rd` at the existing fixed-history damage solve; `RH` below/above
  branches and the exact tie committed; centered FD checks of `Jqq`, `Jqd`,
  `Jdd`, `JdH`, and `JHq` (fixed active set, mixed edge trace signs present);
  the elementary Amor first variation at trace `<0`, `>0`, and `=0` (linear
  FD convergence at the C^1 kink); the full packed JVP versus centered FD
  with a fixed active set; explicit dense nonsymmetry evidence (nonzero
  `(q,d)` block against an exactly zero `(d,q)` block, symmetric `Jqq`/`Jdd`
  diagonals); `vjp` versus the dense transpose; and loud
  shape/dtype/nonfinite/type failures plus input immutability.
- Verification: the focused monolithic suite passes `18/18` in `3.8 s`; the
  legacy engineering-grip suite plus monolithic passes `65/65` in `18.8 s`;
  the JAX engineering/mechanics suites pass `21/21` in `18.7 s`; the
  baseline/mechanics/grip-diagnostic suites pass `30/30` in `16.2 s`;
  `python -m compileall` over the changed modules passes; `git diff --check`
  passes. No solver, workflow/CLI wiring, or physical continuation was added
  or run, and no existing source module was modified.

### Exact resume point

Phase 1 is implementation/manufactured evidence only. The next gate is
primary review of the residual/JVP contract (especially the `Jqd` shim reuse,
the `JHq` active-set derivative at ties, and the `vjp` transposes), followed
by an explicit user decision on whether a Phase-2 solver formulation
(semismooth Newton/complementarity treatment of `RH`, linearization strategy,
globalization, and archive provenance) may be derived. Do not implement any
solver piece before that approval.

## 2026-09-20 — Resolve independent Phase-1 review findings

- Frozen caches: the material AT2 CSR `data`/`indices`/`indptr` internals and
  the derived material-source mask cached by `MonolithicGripResidual` are
  now read-only, matching the already bytes-backed volumes/phase volumes/
  graph arrays. `test_cached_static_internals_are_read_only` proves every
  in-place mutation attempt raises `ValueError` and that residuals/JVP/VJP
  are bit-identical after the attempts.
- Independent oracle: `_IndependentMonolithicOracle` (test helper only)
  rebuilds the packed JVP from frozen static data with independently written
  algebra — explicit per-edge loops/matmuls for the BA gradient, tangent and
  endpoint assemblies, a from-definition 2D cubic-kernel graph-volume damage
  action (validated once against the frozen operator export to `1.0e-12`
  relative), and its own embedded Amor accumulation — and never calls
  production `jvp`/`vjp`/`_dense_jacobian`, `tangent_action`, the production
  endpoint assembly, or damage-residual helpers. `test_full_jvp_matches_
  independent_oracle` compares the full production JVP over three states
  (all-active mix, half-committed mix at a different load, all-inactive at
  zero load) and four directions each (pure q/d/H plus mixed), all below
  `1.0e-10` relative. `test_vjp_bilinear_identity_against_oracle` ties the
  production adjoint to oracle actions through the bilinear identity (plus
  the production-production identity), replacing the previous dense-transpose
  check that reused production `jvp` columns.
- Active-set switching and obstacle contract:
  `test_active_set_switching_across_committed` sweeps a line that crosses a
  committed median level; the median of the odd material count ties one row
  bit-for-bit at `t=0`. Every strict straddle flips the branch, every flip
  straddles or ties the level, the `RH` row is 1-Lipschitz across switches
  (no residual jump), and the exact-tie state takes the committed branch
  with exactly zero `JHq` contribution.
  `test_out_of_range_damage_history_evaluated_without_obstacle` shows raw
  `d` entries `-0.25/1.3/2.7` and `H=-0.5` pass through unmodified with
  finite equations: `Rd` equals the explicit raw formula, `Rq` still equals
  the authoritative fixed-damage residual at the raw damage, and the
  production JVP matches the oracle. Validation intentionally rejects only
  non-float64 dtype, non-finite values, and wrong shapes — no range check,
  because the approved contract forbids a `d` obstacle (the phase center
  weight `g(d)=(1-d)^2+k_res` stays positive for all finite `d`).
- Dense Jacobian demoted: `dense_jacobian` is now private
  `_dense_jacobian`, documented as production self-data for structural
  nonsymmetry evidence only; the independent correctness evidence standard
  is the oracle plus centered FD.
- Verification: monolithic suite `23/23` in `4.4 s`; legacy
  engineering-grip plus monolithic `70/70` in `17.4 s`; JAX
  engineering/mechanics plus mechanics/baseline/grip-diagnostic suites
  `51/51` in `34.8 s`; `python -m compileall -q examples/fracture2d tests`
  and `git diff --check` pass. No Newton/GMRES piece was added and no
  physical state was run.

### Exact resume point

All four independent Phase-1 review findings are resolved and verified.

## 2026-09-20 — Authorize and scope monolithic Phase 2A

- The user authorized Phase 2A of the selected monolithic direction: an
  explicit-parameter scaled semismooth Newton/GMRES core around
  `MonolithicGripResidual`, with manufactured tests only. This is the same
  active engineering-grip task; it does not authorize workflow/CLI wiring,
  continuation/cutback, a physical rounded-slot run, or an auxiliary strict
  sweep.
- The required frozen solver fields are exactly explicit:
  `max_newton_iterations`, `gmres_rtol`, `gmres_atol`, `gmres_restart`,
  `gmres_maxiter`, `armijo_c`, `line_search_reduction`,
  `line_search_max_backtracks`, and `line_search_alpha_min`. Existing gate
  tolerances and mechanics CG tolerances are supplied explicitly through a
  separate frozen gate bundle; no solver or gate defaults are introduced.
- At each iterate the implementation will derive and freeze `Lq` from the
  existing mechanics CG RHS/limit, `Lrho` from the existing strong damage
  residual gate, and `LH` from the existing history gate. It will solve the
  row-scaled nonsymmetric system with the exact Phase-1 active set and check
  the unpreconditioned GMRES residual. The only preconditioner is the
  exact-existing particle mechanics block-Jacobi, sparse strong damage `A_H`,
  and scaled history identity diagonal.
- Convergence will require finite scaled mechanics L2 and damage/history Linf
  block residuals no larger than one, stable active-set status, and the
  existing displacement/damage/history correction gates. Raw Armijo trials may
  reject only non-finite or `H<0` states in addition to insufficient merit
  decrease; no damage obstacle, clipping, projection, or continuation rule is
  allowed.

### Exact resume point

Implement the Phase-2A core and manufactured tests in the private monolithic
module/test pair. Verify configuration failures, scale/JVP algebra, GMRES
callback and residual checks, the three preconditioner blocks, exact and
perturbed roots, active-set switching, Armijo and hard-failure paths, raw
out-of-range damage, and input immutability. Then update the README and this
task/dashboard with the focused/related test, compileall, and diff-check
outputs. Do not run any physical rounded-slot solve.
Phase 2 (any solver formulation) remains unauthorized pending primary review
of this resolution and an explicit user decision on the semismooth-Newton/
complementarity treatment, linearization strategy, globalization, and
archive provenance.

## 2026-09-20 — Close the irrevocable cached-CSR re-review blocker

- Rebuilt the `MonolithicGripResidual` damage CSR from independent
  bytes-backed copies of `data`, `indices`, and `indptr`. Construction now
  rejects any SciPy CSR storage that is writable or permits
  `setflags(write=True)`; the authoritative model operator is exported and
  copied, never mutated.
- Extended the cached-static regression to assert ultimate bytes ownership,
  direct-write and write-flag re-enable failures, working SciPy matvec and
  transpose operations, unchanged authoritative operator storage, and
  bit-identical residual/JVP/VJP outputs after every rejected attempt.
- Corrected the module and `vjp` docstrings: adjoint correctness is supported
  by the independent-oracle bilinear identity, not a dense transpose assembled
  from production JVP columns.
- Verification after the change: the monolithic suite passes `23/23`; related
  engineering-grip/JAX/mechanics/baseline/grip-diagnostic suites pass `90/90`;
  `python -m compileall -q examples/fracture2d tests` and `git diff --check`
  pass. No solver or physical run was performed.

### Exact resume point

The sole Phase-1 re-review blocker is closed. Phase 2 remains unauthorized;
obtain primary review and an explicit formulation decision before adding any
Newton, GMRES, workflow, or physical-run code.

## 2026-09-20 — Implement Phase-2A core and manufactured tests

- The current user authorization supersedes the earlier Phase-1 resume note:
  `examples/fracture2d/nonlocal_engineering_grip_monolithic.py` now contains
  the explicit no-default `MonolithicNewtonConfig`, explicit gate bundle and
  scales, frozen-active-set scaled JVP, SciPy GMRES solve with an explicit
  `pr_norm` callback and unpreconditioned residual check, the exact three-block
  preconditioner, raw Armijo globalization, immutable success diagnostics, and
  hard failure paths. No workflow/CLI, continuation/cutback, auxiliary sweep,
  or physical solve was added.
- `tests/test_fracture2d_nonlocal_engineering_grip_monolithic_newton.py` adds
  manufactured checks for strict configuration, scale/JVP algebra, active-set
  freezing/switching, all preconditioner blocks, exact and perturbed solves,
  GMRES audit data, Armijo rejection/backtracking, hard failures, raw damage,
  and immutable results. The Phase-1 plus Phase-2A focused command passes
  `32/32` in `6.339 s`; changed modules compile and `git diff --check` pass.
- The combined legacy workflow/stabilized command was still running beyond
  the `300 s` verification budget after reaching the long scheduled-controller
  portion; its visible tests had no failure. This is a verification runtime
  note, not a physical-run result.

### Exact resume point

Run any remaining targeted affected suites if needed, inspect the final diff,
and obtain independent review of the Phase-2A scaling, semismooth active-set
semantics, preconditioner blocks, and failure diagnostics. Do not wire the core
into continuation/workflow code or run a physical rounded-slot solve.

## 2026-09-21 — Resolve Phase-2A independent review findings

- Corrected scale provenance: `MonolithicNewtonScales` now records immutable
  `H_used`, `H_next`, and a packed-state digest. `block_scales` uses the six
  existing damage-residual constituent terms from both histories and the
  committed/used/next history scale. Diagnostics now explicitly carry
  `linearization_scales` and `accepted_trial_scales`; convergence norms use the
  accepted-trial bundle and structural pairing validation rejects forged chains.
- Removed the tolerance-level initial shortcut. Only an exactly zero packed
  residual with valid finite `H` returns without GMRES; the rounded zero-load
  nonroot probe now enters Newton and records an actual GMRES call.
- Added immutable `MonolithicNewtonFailureAudit` retention for preconditioner,
  GMRES/status/residual, raw nonfinite or `H<0` trials, line-search
  budget/min-alpha, accepted-trial, and maximum-iteration failures. No failure
  path discards the current-step evidence or replaces it with an empty prior
  tuple.
- Strengthened the manufactured tests with an independent Phase-1 oracle for
  mechanics block and strong-damage preconditioner checks, actual GMRES keyword
  assertions, forged `info=0` residual rejection, raw-trial failures, final
  result residual/active/scale/correction-chain recomputation, and failure-audit
  immutability/completeness. The focused Phase-1 plus Phase-2A command passes
  `33/33` in `8.113 s`.

### Exact resume point

Run the affected engineering-grip suites, full changed-module compileall, and
final `git diff --check`; inspect the final diff for stale scale names or
workflow/physical-run changes. Keep the monolithic core manufactured-only.

## 2026-09-21 — Complete Phase-2A final verification pass

- Added explicit forged-pair regressions using `dataclasses.replace`: a
  diagnostic cannot substitute its linearization scale bundle for its accepted
  trial bundle, and a result cannot substitute the final linearization bundle
  for its final accepted-state bundle. The preconditioner now also rejects a
  linearization bundle paired with a different packed state.
- The focused Phase-1 plus Phase-2A command passes `33/33` in `9.105 s` after
  the final contract checks. `python -m compileall -q examples/fracture2d
  tests`, `py_compile` for the changed monolithic files, and `git diff --check`
  all pass.
- Targeted affected suites pass: direct engineering-grip `47/47` in `18.294 s`;
  baseline, grip-diagnostic, and mechanics suites `30/30` in `18.635 s`.
  The unrelated long A2 coupled suite reached its third verbose test without
  a failure but exceeded both `240 s` and `600 s` budgets; no production or
  physical workflow was launched.

### Exact resume point

The private monolithic implementation and manufactured evidence are ready for
primary Sol review. Keep the core unwired and manufactured-only; do not infer
physical convergence from the timeout-prone legacy suites or run a rounded-slot
workflow.

## 2026-09-21 — Independent Phase-2A numerical re-review

- Verification rerun on CPU: the focused Phase-1 plus Phase-2A command passes
  `33/33` in `7.961 s`; direct engineering-grip tests pass `47/47` in
  `16.227 s`; baseline, grip-diagnostic, and mechanics tests pass `30/30` in
  `19.085 s`; targeted `compileall` and `git diff --check` pass. A combined
  legacy command exceeded its `240 s` wall-clock budget after emitting only
  passing test dots; no workflow or physical state was launched.
- The requested scale, exact-zero shortcut, accepted-trial pairing,
  independent preconditioner oracle, GMRES keyword/status/residual, Armijo,
  result-chain, and immutable failure-audit checks pass on the manufactured
  paths. Source-backed equations and gate semantics remain unchanged; row
  scaling, merit, preconditioner composition, and globalization remain
  project-owned Phase-2A choices.
- Two P2 contract findings remain before integration:
  1. `solve_monolithic_newton` computes `active = monolithic.active_mask(z)` at
     line `1739`, before the audit-protected initial residual block beginning at
     line `1771`. A manufactured `FloatingPointError` there escapes as a raw
     exception rather than an immutable `MonolithicNewtonFailureAudit`.
  2. `MonolithicNewtonScales` validates only nonempty equal-length history
     vectors at lines `1038-1045`; `_validate_scale_pair` validates only the
     packed tail at lines `1085-1088`; and the preconditioner invokes no exact
     `n_material` check at lines `1508-1514`. A forged same-digest scale bundle
     with a one-entry `H_used/H_next` was accepted by direct probing. This is
     not reachable from the solver's own `block_scales` path, but violates the
     fixed-shape/private-API forgery contract and lacks a regression test.

### Exact resume point

Keep auxiliary strict-sweep/continuation integration **NO-GO**. Add the two
contract checks/fixes and rerun the focused and affected suites before any
workflow wiring or physical state. Do not infer physical convergence from the
passing manufactured Newton solves.

## 2026-09-21 — Fix Phase-2A P2 audit-contract findings

- The initial solver active-set evaluation now runs inside the audit-protected
  initial-evaluation phase. Active-mask shape failures and active/Amor,
  residual, exact-root-scale, and initial-scale numerical exceptions become an
  immutable `MonolithicNewtonFailure` with stage/reason/current-state evidence;
  no numerical fallback or workflow path was added.
- `MonolithicNewtonScales` remains generically constructible for dataclass use,
  but every model-bound shape consumer now requires `H_used` and `H_next` to be
  exactly `(monolithic.n_material,)`. Model-bound state-pair consumers also
  validate the exact packed shape, digest, and `H_used` tail before using a
  bundle. The preconditioner and solver failure-audit path reject forged
  same-digest one-entry bundles without emitting a raw validation exception.
- Added two focused manufactured regressions: an initial active/Amor failure
  retains a complete immutable audit, and a same-digest one-entry scale bundle
  is rejected by the preconditioner, scaled operators, and solver.
- Verification after these fixes: Phase-1 plus Phase-2A manufactured tests
  pass `35/35`; direct engineering-grip tests pass `47/47`; baseline,
  grip-diagnostic, and mechanics tests pass `30/30`; targeted compilation and
  `git diff --check` pass. No workflow/continuation or physical run was
  performed.

### Exact resume point

The two requested Phase-2A P2 contract findings are fixed in the private,
manufactured-only core. Keep auxiliary strict-sweep/continuation integration
and physical execution unwired pending primary Sol review; do not infer
physical convergence from these tests.

## 2026-09-21 — Implement Phase-2B single-target verified attempt

- Added the explicit manufactured-only
  `run_verified_monolithic_attempt(model, config, committed, target, adapter,
  initial_z, newton_config)` API and immutable
  `MonolithicVerifiedAttempt`, `MonolithicAuxiliarySweep`, and
  `MonolithicAttemptFailureAudit` payloads. Structural type/shape/identity
  violations propagate; Newton numerical failures and strict auxiliary/canonical
  gate rejections return a rejected attempt with the committed-before digest
  unchanged. No rejected-array serializer or durable archive path was added.
- The authoritative auxiliary sweep uses the raw Newton `q/d/H` candidate only
  as audit input and explicit seeds: mechanics-before at raw `d`, Amor with the
  original committed history, `H_used=max(H_committed,psi_before)`, exact
  fixed-history damage, mechanics-after seeded by mechanics-before, Amor again
  with the original committed history, and
  `H_next=max(H_committed,psi_after)`. Existing strict diagnostics provide the
  finite, mechanics, correction/fixed-point, reaction, damage-equation, range,
  healing, history, force, and moment gates; no warning bypass is available.
- The canonical packed state is exactly
  `(q_after_free,d_solved,H_next)`. Fresh monolithic residual/scale norms,
  active-set consistency, exact constraints, finite values, state arrays, and
  digest pairing are checked before accepting an `EngineeringGripState`.
  Existing Phase-2A and continuation/workflow code remains unchanged and
  unwired.
- Source-backed versus project-owned: the engineering model/config/state,
  mechanics adapter identity, fixed-history AT2 solve, Amor/history equations,
  reaction/conservation diagnostics, and gate limits are existing reviewed
  implementation contracts. Single-target orchestration, canonical pairing,
  failure audit structure, aliases, and no-durable-rejection policy are
  project-owned Phase-2B contracts.
- Manufactured verification on CPU (`JAX_PLATFORMS=cpu`): the new Phase-2B
  suite passes `7/7`; the Phase-1
  plus Phase-2A suite passes `35/35`; direct engineering-grip passes `47/47`;
  baseline/grip-diagnostic/mechanics passes `30/30`; targeted compilation and
  `git diff --check` pass. No continuation, workflow, or physical run was
  performed.

### Exact resume point

Primary Sol should independently review the Phase-2B target-flow validation,
raw-candidate/auxiliary ordering, gate names and strictness, canonical
monolithic recheck, digest/state pairing, and numerical-failure versus
structural-exception boundary. Keep the implementation manufactured-only and
do not wire it into `_run_engineering_attempt`, continuation, or CLI code.

## 2026-09-21 — Close Phase-2B audit identity and stage contracts

- Rejected `MonolithicVerifiedAttempt` objects now retain the exact same
  `MonolithicNewtonFailureAudit` object referenced by their failure payload.
  Auxiliary numerical failures after fixed-history damage reconstruction are
  split into `rho_used` and `rho_next` stages, preserving the precise completed
  operation and paired fields. The mechanics-after seed continues to retain the
  raw Newton damage digest because its displacement seed came from the
  raw-damage mechanics-before solve.
- The focused verified-attempt tests now cover the Newton-audit identity and
  target/model/static/config/gate provenance in addition to the aggregate-gate,
  optional-field, canonical, target, auxiliary, array, digest, and seed
  forgeries. The focused Phase-2B suite passes `10/10`.
- A combined CPU unittest command covering the monolithic Phase-1/2A/2B,
  engineering-grip, JAX, stabilized, mechanics, baseline, grip-diagnostic, and
  workflow suites passes `261` tests with `2` expected GPU-only skips. Expected
  negative CLI/parser diagnostics are confined to successful tests. `python -m
  compileall -q examples/fracture2d tests` and `git diff --check` pass.
- No workflow, continuation, archive, or physical execution was performed.

### Exact resume point

Keep the Phase-2A/2B layers manufactured-only and unwired. Obtain primary Sol
review of the exhaustive pairing contract, authoritative call order, canonical
recheck, Newton-audit identity, distinct fixed-history failure stages, and the
structural-exception versus numerical-rejection boundary before any integration.

## 2026-09-21 — Authorize Phase-2C adaptive strict continuation

- The user authorized a manufactured-only Phase-2C adaptive strict continuation
  around the approved Phase-2B single-target attempt. The new frozen config must
  carry explicit existing load schedule fields and a required
  `MonolithicNewtonConfig`, with no defaults; the existing engineering-grip
  gate/work-check settings remain authoritative.
- Each target is attempted exactly once through Phase2B. Its predictor is rebuilt
  from the unchanged committed state using `engineering._direct_seed` for the
  full displacement and committed `d/H`; cutbacks never warm-start from a
  rejected candidate. Accepted Phase2B canonical states become the next
  committed state; rejection retains exact rollback digests.
- The one-time existing zero-damage work check runs before state attempts. The
  result/failure audit is immutable and in-memory only, preserving every
  attempt, load, increment, cutback, solver configuration, model/static identity,
  and exact state/attempt digest pairing. No callback, workflow, archive writer,
  rejected-field serializer, warning acceptance, clipping, projection, or
  physical run is authorized.

### Exact resume point

Implement Phase2C in the private monolithic module, add the seven requested
manufactured test groups on the existing 5x5 cloud, update the README and this
task memory, then run focused/affected suites, compileall, and `git diff --check`.

## 2026-09-21 — Implement Phase-2C adaptive strict continuation

- Added private `MonolithicContinuationConfig`,
  `MonolithicContinuationAttempt`, `MonolithicContinuationResult`, and
  `MonolithicContinuationFailure` contracts to
  `examples/fracture2d/nonlocal_engineering_grip_monolithic.py`. The config
  carries explicit schedule fields, binds them exactly to the existing
  `EngineeringGripConfig`, and requires one explicit `MonolithicNewtonConfig`.
- Added `run_monolithic_strict_continuation(model, config, adapter)` plus
  strict aliases. It calls the existing zero-damage work check once, builds
  each packed predictor with `engineering._direct_seed` and committed `d/H`,
  calls Phase2B exactly once per target, commits only the Phase2B canonical
  state, and reuses `_next_target`/`_is_minimum_increment` for endpoint and
  cutback behavior. Numerical endpoint failure carries a validated immutable
  partial result; structural Phase2B exceptions propagate.
- Added eight manufactured test cases in
  `tests/test_fracture2d_nonlocal_engineering_grip_monolithic_continuation.py`
  covering explicit API/config binding, endpoint snap, rollback/predictor
  rebuild, minimum-increment failure, cutback-limit failure, work-check
  short-circuit, structural exception propagation, and forged audit pairing.
  Updated `examples/fracture2d/README.md` with the private API and restrictions.
- Source-backed/project-owned boundary: all residuals, constitutive/history
  fields, direct predictor, work check, endpoint snap, minimum-increment rule,
  and gate/work-check settings are existing reviewed engineering-grip or
  Phase2B contracts. The continuation state machine, immutable wrappers,
  digest/order validation, and failure result are project-owned orchestration
  contracts; no physical formulation was changed.
- Verification on CPU with `JAX_PLATFORMS=cpu`: the combined monolithic
  Phase-1/2A/2B/2C command passes `54` tests; the engineering/JAX plus all
  monolithic focused command passes `114` tests with `1` expected skip.
  `compileall -q examples/fracture2d tests` and `git diff --check` pass. No
  workflow, archive, continuation production run, or physical execution was
  performed.
- A bounded manufactured probe of the actual Phase2C call with the required
  direct predictor and smoke Newton settings rejected both `u_bar=1e-4` and
  the `u_bar=5e-5` cutback at the Phase2A line-search backtrack budget. This is
  an unresolved Phase2B numerical risk, not a changed solver setting or
  physical convergence result; the controller tests intentionally exercise the
  immutable orchestration with manufactured verified-attempt outcomes.

### Exact resume point

Primary Sol should independently review the Phase-2C config-to-base schedule
identity, direct-seed predictor reconstruction, one-call Phase2B boundary,
rollback/cutback ordering, partial-result validation, and the decision to let
the existing `EngineeringGripWorkCheckFailure` propagate before attempts.
Keep the implementation manufactured-only and unwired pending that review.

## 2026-09-21 — Close raw-trial and Phase-2C audit hardening

- Removed only the intermediate raw Newton line-search `H>=0` rejection. Raw
  trial vectors are now accepted/evaluated whenever all packed entries are
  finite; no clipping, projection, or damage obstacle was added. The exact
  `R_H=H-max(H_committed,psi_plus)` residual remains unchanged.
- Successful `MonolithicNewtonResult` objects now explicitly require a
  nonnegative final history, and the Phase-2B canonical audit adds an explicit
  history-nonnegative gate alongside the exact `H_next=max(...)` consistency
  check. A manufactured Newton regression records an accepted intermediate
  negative-history trial, while a successful solve ends with `H>=0`; forged
  Newton/canonical negative histories are rejected.
- Hardened `MonolithicContinuationResult.validate()` to bind work-check
  `u_bar`, `epsilon`, `relative_limit`, and pass status to the exact
  `EngineeringGripConfig`, reconstruct the ordered `_next_target`/
  `_is_minimum_increment` schedule, enforce `cutback_count` and
  `max_cutbacks`, and require object/digest pairing for all accepted and
  rejected attempts and states. Added one-field work-check and cutback
  forgeries.
- Replaced the tautological continuation success helper with an actual
  zero-to-`1e-4` controller/direct-predictor smoke test; no `initial_z`
  replacement or result forgery remains. CPU verification passes the combined
  monolithic suite `54/54` and the engineering/JAX plus monolithic focused
  suite `114` tests with one expected GPU-only skip. Targeted compilation and
  `git diff --check` pass.
- No workflow, archive, rounded-slot physical run, or production continuation
  was performed. Source-backed equations and existing direct-engineering
  gates are unchanged; raw-trial policy, final-history checks, immutable
  continuation validation, and test orchestration are project-owned contracts.

### Exact resume point

Obtain independent primary Sol review of the finite raw-trial/final-history
boundary, the new canonical history gate, and the exact Phase-2C schedule and
work-check audit before any workflow integration or physical execution.

## 2026-09-21 — Complete work-check and Newton audit repairs

- Engineering work-check validation no longer compares the three
  `damage_runtime_identity` cache tokens. It still requires distinct nested
  result objects, shared constraints, shared static runtime identity, exact
  static/damage/unit digests, exact centered loads, zero-damage provenance,
  explicit CG settings, and recomputed reaction/energy-difference arithmetic.
  This matches the CPU-replayed JAX adapter, which intentionally builds a fresh
  fixed-damage view for each centered load.
- The continuation work-check failure regression now runs the actual centered
  solver with a controlled `work_relative_limit=0.0`, captures its valid
  `EngineeringGripWorkCheckFailure.check`, validates that immutable failing
  object, and proves Phase-2B is never called. The strict constructor remains
  closed to inconsistent `dataclasses.replace` payloads.
- Newton diagnostics now retain immutable reference-scale, accepted
  line-search, and accepted-trial scaled residual vectors. Value-only result
  validation reconstructs raw initial residuals within the explicit
  `512*eps64` scale round-trip bound and binds merits, block norms, accepted
  line-search algebra, convergence, and the final row without operator calls.
  `validate_with_owner(owner)` is the explicit in-memory authoritative replay
  and is invoked by the solver for returned production results. The Armijo
  orchestration test uses deterministic key-based manufactured outputs so this
  replay remains complete rather than skipped.
- One-field forgery coverage now includes work-check reaction, centered energy
  difference, relative error, pass flag, result digest, and Newton final
  residual, block norms, merit, active mask, status, and reason. README text
  documents the cache-token and owner-replay boundaries.
- Verification with `JAX_PLATFORMS=cpu`: the affected combined unittest command
  passes `233` tests with one expected GPU-only skip; expected CLI usage/error
  text is confined to negative parser tests. `python -m compileall -q
  examples/fracture2d tests` and `git diff --check` pass. No workflow, archive,
  or physical execution was performed.

### Exact resume point

Keep Phase-2A/2B/2C private, manufactured-only, and unwired. Primary Sol should
review the no-cross-solve-cache-identity work-check rule, the controlled valid
failure fixture, the `512*eps64` residual unscale bound, and the distinction
between algebraic result validation and explicit owner-backed replay before any
workflow integration or physical execution.

## 2026-09-21 — Implement owner-backed authenticity validation

- Status: in progress. Before any archive-reader work, add authoritative fresh
  replay for the work check, Phase-2B verified attempts, and Phase-2C results.
  Comparisons must bind durable model/static/config digests, arrays, scalars,
  reactions/energies, gates, states, and failure provenance while ignoring only
  fresh runtime identities and cache tokens.
- Required negative coverage: a self-consistent forged mechanics
  residual/reaction/energy must pass structural validation but fail owner replay;
  recursive continuation validation must fail for the same forgery; arbitrary
  no-attempt failures must be rejected; a legitimate direct-predictor result
  must pass owner replay.
- Constraints: use no callback or mock as an authenticity source; preserve
  numerical-failure typing, work-check exception propagation, and any legitimate
  already-complete interval behavior. Do not add archive/CLI integration or run
  rounded-slot/physical continuation.
- Exact next action: implement the durable comparators and controller failure
  state-machine repair, add focused manufactured tests, then run affected CPU
  suites, compileall, and `git diff --check`.

## 2026-09-21 — Close the continuation owner-replay dispatch boundary

- `MonolithicContinuationResult.validate_with_owner(model)` now recognizes a
  controller-only record only after structural validation identifies the exact
  terminal `MonolithicContinuationAttempt` with `failure_stage="controller"`,
  `load_target == load_start`, and `increment == 0`. It revalidates the
  schedule target, step/cutback metadata, canonical no-progress reason,
  committed-before/after digests, predictor, model/config digests, and the
  absence of every PDE audit field, then deliberately skips Phase2B replay.
- Every other attempt, including every positive-increment attempt, continues
  through recursive `MonolithicContinuationAttempt.validate_with_owner` and
  authoritative Phase2B owner replay. Controller-stage relabels, target or
  increment changes, noncanonical reasons, and retained PDE audits are
  rejected at the dispatch boundary.
- Added manufactured coverage for legitimate no-progress owner validation
  without Phase2B replay and stage/target/increment/reason forgeries. The
  controller remains private, manufactured-only, and unwired; no archive or
  physical run was performed.
- Focused CPU unittest verification passes `77` tests across the engineering,
  Phase-2B, and Phase-2C modules; the expanded affected monolithic/direct
  command passes `112` tests. `python -m compileall -q examples/fracture2d
  tests` and `git diff --check` pass. Primary Sol review remains pending.

### Exact resume point

Run the final affected CPU suite after this dispatch-only change, inspect the
diff for unrelated edits, and obtain primary Sol review of the controller-only
exception boundary before any archive-reader work.

## 2026-09-21 — Independent owner-replay audit: archive integration remains NO-GO

- Status: owner-backed replay is implemented and the manufactured audit was
  rerun without source edits or physical execution. A real 5x5 direct-predictor
  controller reached exact `u_bar=1e-4`; its continuation replay passed against
  both the original and a freshly rebuilt equivalent model.
- The prior self-consistent work-check forgery again passed structural
  validation (`relative_error=2.220118369977306e-16`, `passed=True`) but was
  rejected by `EngineeringGripWorkCheck.validate_with_owner` on the durable
  reaction mismatch. Recursive continuation forgery, arbitrary no-attempt
  failure, Phase-2B auxiliary forgery, Newton/Phase-2B public-entrypoint
  monkeypatching, and fresh runtime/cache identity behavior all passed their
  negative/positive checks.
- Blocker: a legitimate controller-stage no-progress result retained its
  required terminal controller attempt, but
  `MonolithicContinuationResult.validate_with_owner` rejected it with
  `verified monolithic target must exceed committed.u_bar`. The continuation
  method currently sends every attempt to
  `MonolithicVerifiedAttempt.validate_with_owner`, whose unconditional
  authoritative Phase-2B replay is invalid for a zero-increment controller
  attempt. The structural constructor already checks its exact direct
  predictor and failure provenance; owner replay needs a controller-stage
  branch rather than a Phase-2B numerical replay.
- Verification: the focused owner tests ran `123` tests with one expected skip;
  the broader affected CPU command ran `241` tests with one expected skip; both
  passed. `python -m compileall -q examples/fracture2d tests` and
  `git diff --check` passed. No archive, workflow, or physical run was made.

### Exact resume point

Primary Sol should review and authorize the controller-stage owner-replay
repair, including exact direct-predictor/model/config/failure provenance
comparisons and a legitimate no-progress owner-validation regression, before
archive schema/reader work. Keep the layer private, manufactured-only, and
unwired.

## 2026-09-21 — Confirm controller-only owner replay fix: archive gate GO

- The controller-only branch now authenticates the terminal zero-increment
  no-progress record through model/config digests, canonical stage/reason,
  unchanged before/after state digests, exact direct predictor, and absence of
  any PDE audit. It does not invoke Phase-2B replay.
- Manufactured probes passed for the legitimate no-progress result and rejected
  forged failure stage, target, increment, and reason. A positive-target direct
  predictor still invoked the authoritative Phase-2B replay exactly once and
  owner validation passed.
- Verification: focused monolithic suite `62` tests passed; targeted owner
  probes `3` tests passed; `compileall` and `git diff --check` passed. No source
  edits, archive, workflow, or physical execution occurred in this confirmation.

### Exact resume point

Archive schema/reader design may proceed, provided the reader reconstructs the
owner and calls `MonolithicContinuationResult.validate_with_owner(...)` at the
top level, preserving the controller-only branch and authoritative CPU replay
for every positive-target attempt.

## 2026-09-21 — Implement the lean monolithic archive/replay layer

- Added the separate private `nonlocal_engineering_grip_monolithic_archive.py`
  schema `fracture2d-monolithic-continuation-v1`. Its exact NPZ member contract
  contains canonical metadata plus accepted-state `u_bar`, full union `Q`,
  material `d/H`, accepted canonical reaction, and accepted canonical energy.
  Rejected/intermediate `q/d/H` arrays are not durable members.
- Metadata reconstructs the complete engineering, Newton, and continuation
  configs; model/static/source identities; work-check scalars/digests; failure
  provenance; state digests; and ordered scalar attempt evidence including
  Newton/GMRES/line-search, canonical norms/gates, and attempt digests.
  Canonical JSON rejects duplicates/non-finite/unknown fields and is stored as
  `uint8`; all numerical members are exact float64 and non-object.
- The writer validates `result.validate_with_owner(result.model)` before
  serialization, accepts only CPU-reference-authenticated attempts, enforces a
  required uncompressed byte budget before atomic fsync/replace, refuses
  overwrite by default, and cleans temporary files on failure. The reader
  checks the exact ZIP allowlist/budget, reconstructs configs, verifies current
  source identity, reruns the captured strict controller with the supplied
  owner, invokes top-level `validate_with_owner`, and compares every accepted
  array and ordered scalar attempt/work/failure record. Both completed and
  terminal failed results are supported; replay arrays are immutable.
- Added the captured authoritative strict-controller entry point, README
  documentation, and six manufactured archive tests covering API/budget,
  owner replay, no rejected vectors, mutation/truncation, atomic cleanup,
  immutability, overwrite refusal, and failed-result replay.
- Verification: the focused archive suite passes `6/6`; the combined affected
  engineering/stabilized/workflow/monolithic CPU command passes `236` tests
  (expected negative CLI diagnostics only); compileall and `git diff --check`
  pass. A direct 5x5 manufactured success and failed-result round trip also
  passed. No workflow, rounded-slot, or physical execution was performed.

### Exact resume point

Primary Sol should independently review the new schema/member names, scalar
evidence projection (especially omitted-array presence digests), accepted
reaction/energy source fields, ZIP budget/atomicity checks, and the captured
top-level controller plus recursive owner replay. Then run the full affected
CPU suite and archive the task only after review approval.

## 2026-09-21 — Harden captured archive-controller dispatch

- The archive replay alias now routes through a private continuation
  implementation bound to the captured authoritative work-check and Phase-2B
  runners. The public three-argument continuation entry remains injectable for
  manufactured controller tests; public controller, Phase-2B, and work-check
  attributes cannot alter archive owner replay.
- Verification after the dispatch refactor: the continuation/archive focused
  command passes `20/20`; compilation and `git diff --check` pass. No workflow,
  rounded-slot, or physical execution was performed.

### Exact resume point

Primary Sol should review the private dispatch boundary together with the
archive schema/member and scalar-evidence contract before task archival.

## 2026-09-21 — Begin independent archive-boundary hardening

- Status: in progress. Continue within the same active engineering-grip task;
  no formulation, workflow, CLI, rounded-slot, or physical execution change is
  authorized.
- Scope: fix the confirmed ZIP/NPY inspection and same-file loading gaps,
  atomic no-clobber race, mutable owner-runner dispatch, strict schema-version
  typing, and opaque scalar-evidence fallback. Add focused safe-forgery and
  race/format regressions.
- Exact next action: implement these durability-only changes, run the focused
  archive/continuation tests when the JAX environment is available, then run
  compileall and `git diff --check`; leave the task incomplete if dependency
  blockers remain.

## 2026-09-21 — Complete private monolithic v1 archive hardening

- The schema is now explicitly ZIP32-only with canonical `allowZip64=False`,
  complete ZIP64 sentinel/extra rejection, and a practical `500000000`-byte
  maximum. Inspection and NPY loading use the same file descriptor and enforce
  exact members, flags, local/central headers, compressed and uncompressed
  budgets, decompression ratio, C-order payloads, dtypes, shapes, and payload
  lengths.
- The unpublished v1 member is named `accepted_mechanics_energy`, accurately
  identifying mechanics-after A2 mechanics energy. Ordered audit metadata is
  named `attempt_scalar_evidence`. Rejected/intermediate `q/d/H`, Amor arrays,
  crack-region strain, and static operators remain absent; owner replay
  recomputes and validates these derivable quantities before accepting archived
  states.
- Atomic no-clobber installation uses a hard link and fails closed where it is
  unsupported. A post-link directory-fsync failure triggers identity-checked
  destination cleanup and a second fsync, with cleanup failures reported rather
  than hidden. Controller-only no-progress results now have an end-to-end
  archive round trip and forgery coverage.
- Verification passes archive `15/15`, full monolithic `77/77`, and the final
  affected CPU suite `252` tests with one expected GPU-only skip. `compileall`
  and `git diff --check` pass. No workflow, CLI, rounded-slot, or physical run
  was performed.

### Exact resume point

Obtain final independent GO/NO-GO on the ZIP32 raw-record parser, no-clobber
cleanup, precise mechanics-energy provenance, controller-only archive replay,
and owner-recomputed-field contract. If approved, integrate the private
monolithic strategy into the workflow/CLI with every Newton and archive
parameter explicit, test manufactured CLI/provenance paths, and only then run a
 bounded rounded-slot zero-to-`1e-4` strict smoke archive and replay.

## 2026-09-21 — Close the strict ZIP32 EOCD blocker

- Added same-descriptor raw EOCD parsing over the legal ZIP comment search
  window. v1 now requires exactly one `PK\x05\x06` record ending at EOF with
  zero comment bytes, disk number and central-directory start disk equal to
  zero, and entries-on-disk/total entries equal to the exact allowlist count
  without `0xffff` sentinels.
- EOCD central-directory size and offset must be non-sentinel, file/budget
  bounded, and exactly equal to the parsed central-record span and location.
  Preceding ZIP64 locator/end signatures and duplicate, ambiguous, or trailing
  EOCD content are rejected before `zipfile` metadata is trusted; the existing
  central/local raw-header checks remain in force.
- Added mutations for both count sentinels, disk fields, count mismatch,
  central size/offset mismatch, duplicate/trailing EOCD, and preceding ZIP64
  locator/end signatures. The canonical writer archive continues to load.
- This is a project-owned ZIP32 envelope/authenticity rule; no PDE,
  constitutive, history, continuation, or gate equation changed.
- Verification with `JAX_PLATFORMS=cpu`: archive suite `16/16`; full monolithic
  Phase-1/2A/2B/2C/archive suite `78/78`; `python -m compileall -q
  examples/fracture2d tests` passed. No workflow or physical execution was
  performed.

### Exact resume point

Primary Sol should review the EOCD search-window/uniqueness rule, raw EOCD to
central-record cross-checks, and ZIP64-predecessor rejection. Keep the private
archive unwired and do not run a physical continuation.

## 2026-09-21 — Close workflow review findings before physical authorization

- Decision: Treat `max_stagger_iterations` as a compatibility field only on the
  monolithic workflow. Require exactly the documented
  `examples/fracture2d/configs/solver_base.py` value `100`; reject any other
  value with an explicit statement that it is not a monolithic iteration
  budget. The monolithic Newton/GMRES/Armijo fields remain the only iteration
  controls.
- Decision: If owner replay fails structurally after a monolithic archive write,
  remove the final archive, summary, replay report, and manifest only when
  their captured inode/device and SHA256/size identities still match. Preserve
  concurrent replacements, fsync the output directory after cleanup, and raise
  a compound durability error chained to the original replay exception if
  cleanup or directory fsync fails. A terminal numerical failure whose replay
  succeeds retains its failure archive.
- Decision: Define `field_equations_validated` as owner validation of retained
  accepted-step canonical fields, independent of completion status; an
  initial-state-only result is false and no omitted rejected vector or physical
  execution is implied.
- Verification: workflow tests `48/48`, monolithic/archive tests `78/78`, legacy
  tests `50/50`, stabilized tests `82/82`, compileall, and `git diff --check`
  pass. Physical rounded-slot execution remains unauthorized.

### Exact resume point

Review the implementation and verification output for the three decisions
above, especially identity-checked post-replay cleanup under replacement and
cleanup-failure races. Keep the monolithic path manufactured-only and do not
run the bounded physical smoke until primary Sol explicitly authorizes it.

## 2026-09-21 — Publish monolithic results as one atomic run bundle

- Replaced monolithic flat/final-path publication with a single
  `<run-id>.monolithic/` directory containing `archive.npz`, `replay.json`,
  `summary.json`, and `manifest.json`. The archive and all JSON are generated
  in a unique same-parent mode-`0700` staging directory; every JSON is fsynced,
  the exact four members are regular-file fingerprinted, and the staging
  directory is fsynced before publication.
- Publication uses Linux `renameat2(RENAME_NOREPLACE)` through a ctypes wrapper
  and fails closed when unavailable. The final parent directory is fsynced
  after publication. A parent-fsync failure raises
  `MonolithicBundleDurabilityError` while retaining the complete published
  bundle; structural failures before publication remove only the private
  staging directory. Monolithic `overwrite=True` remains rejected.
- All summary/replay/manifest paths report final bundle locations, and the
  manifest binds the staged archive/replay/summary content before publication.
  Legacy and stabilized flat-artifact publication is unchanged. At this
  checkpoint the CLI reported parent durability uncertainty with the numerical-
  failure exit status; the primary-review correction below supersedes that
  classification.
- Source-backed versus project-owned: no PDE, constitutive law, history rule,
  continuation equation, or acceptance gate changed. ZIP32 archive/replay and
  owner-authentication contracts are existing project contracts; bundle layout,
  staging, no-clobber rename, fsync ordering, fingerprint validation, and
  durability/error semantics are project-owned publication contracts.
- Verification under `.venv`: workflow plus bundle seam `52/52`, full
  monolithic/newton/verified/continuation/archive `78/78`, legacy `50/50`,
  stabilized `82/82`; `python -m compileall -q examples/fracture2d tests` and
  `git diff --check` pass. No physical or rounded-slot execution was run.

### Exact resume point

Primary Sol should independently review the private staging cleanup boundary,
Linux `renameat2` wrapper/fail-closed behavior, manifest/content fingerprints,
parent-fsync durability error handling, and preservation of legacy/stabilized
flat outputs. Keep the monolithic path manufactured-only; do not run the
bounded physical smoke.

## 2026-09-21 — Resolve primary monolithic integration review

- Primary review found two high-priority boundary defects: inspection and ZIP
  reads shared a descriptor but not immutable bytes, so an in-place inode
  rewrite could occur between them; and broad Newton exception catches could
  convert type/shape/API defects into numerical rejection archives. Replay now
  reads one budget-bounded immutable snapshot before inspection/decompression,
  while Newton catches only explicit floating-point/linear-algebra failures and
  propagates structural exceptions throughout initial evaluation, GMRES
  residual evaluation, line search, and accepted-trial validation.
- ZIP64 detection now checks the structural locator position immediately before
  EOCD instead of scanning compressed payload bytes for coincidental signatures.
  The CPU-only path rejects non-default FEM/GPU options, parent-fsync durability
  errors propagate separately from numerical failures, and the manifest marks
  its necessarily nonrecursive self-fingerprint scope explicitly while binding
  archive/replay/summary fingerprints.
- Added regressions for in-place rewrite after inspection, payload signature
  false positives, structural active-mask failure, unused FEM/GPU options,
  durability classification, and manifest self-reference semantics. Focused
  archive/Newton/workflow verification passes `48/48` after follow-up
  line-search, preconditioner, and failure-audit structural-error regressions.
- Final CPU verification passes the full monolithic suite `82/82`, workflow
  suite `52/52`, legacy engineering-grip/JAX suite `63` tests with one expected
  GPU-only skip, and stabilized suite `82/82`. `python -m compileall -q
  examples/fracture2d tests` and `git diff --check` pass.
- No PDE, constitutive law, history rule, continuation equation, gate, rounded
  cloud, or physical state changed.

### Exact resume point

Primary implementation review and nonphysical verification are complete. Keep
the monolithic path experimental/manufactured-only and do not run the bounded
physical smoke without separate explicit authorization. The broader physical
objective remains blocked at strict `u_bar=0.00623125` pending a materially new
user-approved numerical formulation or a decision to stop.

## 2026-09-21 — Authorize bounded CPU monolithic smoke

- The user explicitly authorized one fresh-zero strict CPU-reference monolithic
  smoke from `u_bar=0` to exact `1e-4` using the reviewed workflow, ZIP32
  archive, atomic bundle publication, and immediate owner replay.
- Preserve the documented rounded-cloud parameters, explicit Newton/GMRES/
  Armijo controls, all physical gates, and the `500000000`-byte archive budget.
  Use a new run ID and stop after this endpoint or any strict failure.
- This authorization does not permit continuation above `1e-4`, warning
  acceptance, GPU execution, archive resume, or reinterpretation of a rejected
  state as physical evidence.

### Exact resume point

Verify the source cloud and output parent, execute the documented CPU command,
then inspect the published bundle, replay report, summary, and manifest before
recording the result.

## 2026-09-21 — Stop CPU monolithic smoke for unacceptable runtime

- The authorized fresh-zero CPU-reference command was launched with the exact
  reviewed `0 -> 1e-4` configuration and then explicitly aborted by the user
  because its runtime without progress output was unacceptable.
- Process and artifact checks after interruption found no remaining monolithic
  process, no published `.monolithic` bundle, and no private staging directory
  for run ID `engineering_grip_monolithic_cpu_u1e4_20260921`. The interruption
  is neither a numerical failure nor physical evidence.
- The current monolithic workflow is CPU-reference-only by contract and does not
  construct the existing GPU/JAX mechanics adapter. Do not rerun the same
  physical-cloud command unchanged; performance is now an explicit blocker.

### Exact resume point

Choose and review a performance route before another physical-cloud launch:
either profile/optimize the CPU owner on bounded manufactured and cloud setup
operations, or separately design and validate a GPU-capable monolithic residual,
JVP/preconditioner, and owner-replay contract. Do not silently route the current
CPU-authenticated archive through the staggered GPU adapter.

## 2026-09-21 — Select bounded CPU profiling

- The user selected CPU profiling before any GPU monolithic design or another
  physical-cloud continuation launch.
- Profile the real source cloud only through bounded setup and single-operation
  probes. Separate model construction, monolithic residual/JVP, scale,
  preconditioner, and linear-solver costs; do not run an unbounded continuation.
- Any optimization must preserve exact equations, float64 data, strict gates,
  immutable audit contracts, and owner replay. Add equivalence tests before
  another workflow launch.

### Exact resume point

Inspect the construction and monolithic call graph, run time-limited probes with
progress/timing output, identify the dominant cost, then implement only a
measured minimal optimization.

## 2026-09-22 — Complete CPU profiling, optimization, and bounded smoke

- Real-cloud setup was not the dominant cost: source load took `0.012s`, union
  geometry `8.39s`, and owner model construction about `10s` after geometry.
  The unchanged three-solve zero-damage work check took `207.35s`; stack samples
  were entirely in matrix-free A2 CG tangent actions. One tangent action took
  about `0.46s` over `79060 x 47` padded support data.
- Added one damage-independent read-only CSR support-gradient operator and used
  its transpose for endpoint divergence. Vectorized the direct-solve
  block-Jacobi application. The exact work-check contract, three independent
  solves, float64 equations, tolerances, and provenance remain unchanged. The
  real-cloud work check now takes `17.09s`, still uses `281/280/280` CG
  iterations, and retains relative work error `2.4384612234653235e-12`.
- Monolithic profiling then showed repeated identical view/static validation and
  JVP kinematics plus poor block-Jacobi convergence. Added exact-state read-only
  one-entry caches and assembled the exact free-DOF mechanics tangent from the
  same CSR gradient and isotropic constitutive coefficients. Its matrix action
  matched the existing tangent to relative defect `1.90e-15`; sparse assembly
  and factorization took `0.494s` and `0.676s`. The first real-cloud GMRES
  changed from `94.317s`, `1927` callbacks to `1.025s`, `16` callbacks, with
  unpreconditioned residual `4.348170e-3` below limit `1.001986`.
- The authorized optimized CPU run
  `engineering_grip_monolithic_cpu_u1e4_optimized_20260922` completed strict
  `0 -> 1e-4` in `275.51056486275047s`: one accepted step, zero cutbacks,
  reaction `12.6749467637548`, mechanics energy `6.337473381879628e-4`, damage
  maximum `1.2254719713851597e-4`, and history maximum
  `1.44288933743512e-2`. Immediate owner replay passed with
  `field_equations_validated=true`; result digest is
  `ef5f3a49c3b3b64f824f5695600bc1a1c932bc36df142cad159a62065a07c0e5`.
- Published bundle:
  `/root/autodl-tmp/data/fracture2d/engineering_grip/engineering_grip_monolithic_cpu_u1e4_optimized_20260922.monolithic`.
  It contains exactly `archive.npz`, `replay.json`, `summary.json`, and
  `manifest.json`. Archive SHA256 is
  `828e60ac8aa4b7290b8796d6d34ced9c346a8fb7abc1e2bbe1398a1f56407807`;
  replay and summary hashes are
  `82df981ac8fc5ba307e9ec5348ce05b13342b5ee55f0ec949f8adf32af676a12`
  and `d32eef996d5753219eaccea3feced25c671d0da6a7ff166717f87609280bdc65`,
  matching the manifest. The manifest uses the reviewed
  `self-reference-excluded` scope.
- Verification passes the focused A2 independent-equivalence/work tests and the
  full monolithic discovery command `82/82`; Python compilation and
  `git diff --check` pass. Two attempts to run the combined and then full A2
  legacy modules exceeded the `360s` tool timeout without a result, so they are
  not recorded as passes or failures.

### Exact resume point

Treat the strict `1e-4` bundle as the completed bounded CPU integration result.
Do not continue above `1e-4` without new user authorization. A follow-up review
should audit the sparse-gradient assembly, exact mechanics preconditioner, and
one-entry immutable caches; broader legacy suites may be rerun separately with
a larger explicit time budget if needed.

## 2026-09-22 — Authorize long strict CPU continuation to 0.0075

- The user explicitly selected the prior engineering target `u_bar=0.0075` and
  confirmed a fresh-zero strict CPU monolithic run that may continue for hours.
- Change only `load_final` from `1e-4` to `0.0075`. Preserve
  `du_initial=1e-4`, `du_min=5e-5`, `cutback_factor=0.5`, `max_cutbacks=8`, the
  validated Newton/GMRES/Armijo controls, all engineering gates, the
  `500000000`-byte ZIP32 budget, atomic publication, and immediate owner replay.
- Run ID:
  `engineering_grip_monolithic_cpu_u7p5e3_20260922`. Stop only at exact strict
  completion or terminal strict failure; do not reinterpret a rejected trial or
  intermediate accepted state as completion.

### Exact resume point

Verify the new run ID is unused, launch the documented workflow with only
`--load-final 0.0075` changed, then inspect and record the resulting bundle.

## 2026-09-22 — Record terminal strict monolithic failure before 0.0075

- The fresh-zero strict CPU run ended through the solver's terminal numerical
  path after `14779.77797342278s` (about 4 h 6 min), not through the external
  12-hour timeout. Completion status is `failed`.
- It produced 61 accepted steps, 63 attempts, 2 rejected attempts, and 1
  cutback. The last committed state is exact
  `u_bar=0.0061000000000000325`; its reaction is `641.3384929751363`, mechanics
  energy `1.9560824035741737`, damage maximum `0.8202621432308079`, and history
  maximum `432.1558022505133`.
- The first rejected target was `0.0062`. The cut-back target
  `0.006150000000000033` failed at Newton iteration 8 with
  `line search backtrack budget exhausted`. GMRES returned `info=0`; its
  unpreconditioned residual `1.2062573530759628e-4` passed the
  `2.562367167072153e-4` limit. All 13 raw Armijo trials failed sufficient
  decrease, so the configured `du_min=5e-5` made this terminal.
- Bundle:
  `/root/autodl-tmp/data/fracture2d/engineering_grip/engineering_grip_monolithic_cpu_u7p5e3_20260922.monolithic`.
  It contains exactly `archive.npz`, `replay.json`, `summary.json`, and
  `manifest.json`. Archive, replay, summary, and manifest SHA256 values are
  `f97c60ec0ddd8658cdf9e2c641ca78254849dbfc2bae99e8a1e5deaec00d7b75`,
  `8637e37bcf782e16f0adb4763b1d84dad1e22ffc73c73391934d9b0dddc56950`,
  `db4ba66e6b9f213afee6101e1caa95fa5428eef36702bb5234de2353fc719f81`,
  and `7b501f948d27039d60105497c5ebff2b5a466b8815347c6d917c45e205f2f40e`.
  The first three match the manifest, whose own scope remains
  `self-reference-excluded`.
- Replay reports `owner_authenticated=true`, `path_replayed=true`,
  `field_equations_validated=true`, and reconstructed rejected-trial paths
  without persisting rejected vectors. `validated=false` is expected because
  the requested continuation did not complete. Result digest is
  `d5dd141208a1d272bbed1cece7b7c31dc9f731baf9fd9c6b0f57f5b1b579e4ab`.

### Exact resume point

The requested `u_bar=0.0075` endpoint was not reached. Keep `0.0061` as the last
strict committed monolithic state and the `0.00615` fields as rejected-only.
Do not rerun with a smaller increment, altered Armijo controls, warning
acceptance, or another formulation without a new explicit numerical decision
and user authorization.

## 2026-09-22 — Read-only comparison of stagger and monolithic failure

- User requested careful causal analysis before further trials. This session
  only inspected existing scalar evidence and solver algebra; no continuation,
  parameter modification, or rejected-vector reconstruction was executed.
- Endpoints differ: an earlier stabilized stagger run committed `0.0061125`
  and failed at `0.00611875`; the later residual-backtracking stagger route
  committed `0.00623125`, including a strict post-peak `0.0062` state, and failed
  at `0.006234375`. Monolithic committed `0.0061` and failed at `0.0062` and
  `0.00615`. These indicate a shared difficult region, not one proven critical
  load. Different minimum increments and histories prevent a controlled
  solver-only comparison. The reactions at shared `0.0061` differ by relative
  `1.3526e-9`; this is scalar agreement, not a full-field parity proof.
- At the rejected monolithic `0.00615` attempt, accepted Newton alphas were
  `1, 1/4, 1/8, 1/32, 1/512, 1/1024, 1/4096`. Reconstructing raw direction
  infinity norms from correction/alpha gives, at iterations 2 and 7,
  displacement `7.0125e-4 -> 2.0696e-2`, damage
  `6.6640e-2 -> 1.3536`, and history `203.214 -> 6645.334`.
  Shrinking accepted updates therefore hide growing raw Newton directions.
  This is consistent with an ill-conditioned coupled linearization or a
  difficult basin/branch transition, but is not a measured singular value.
- At iteration 8, relative scaled linear residual was `4.7076e-11`; failure
  occurred in nonlinear globalization, not GMRES convergence. The final tested
  alpha was `2^-12 = 2.44140625e-4`, not the configured `alpha_min=1e-8`.
  Its merit exceeded the reference by `0.103346%`; reference merit was
  `3.2828627494446846e12`. This is not tolerance-level stagnation. Failure over
  this finite tested range does not establish a non-descent derivative at zero
  alpha, nor prove that further backtracking would solve the branch problem.
- Algebraic diagnostic (fixed active set, exact subproblem solves): writing
  the strong damage row as `A d - 2H`, with
  `A=Gc/l0 I + Gc*l0 D + 2 diag(H)` and `C=2 diag(d-1)`, the coupled Jacobian
  has blocks `[K B 0; 0 A C; -P 0 I]`, where `P` includes the strict history
  active mask. Eliminating mechanics/history gives
  `S=A-C P K^-1 B`. The corresponding ideal damage fixed-point derivative is
  `T=A^-1 C P K^-1 B`, so `S=A(I-T)`. An eigenvalue of this ideal map near +1
  can simultaneously slow Picard and make the coupled Schur complement nearly
  singular. Observed stagger norm ratios are not spectral measurements, and
  the accelerated implementation is not this ideal map.
- Strong damage/strain/history feedback near the notch is supported by the
  earlier strict `0.0062` state (damage maximum about `0.999917`, history about
  `1.1984e6`), versus monolithic `0.0061` maxima `0.820262` and `432.156`.
  Previous crack-localized mechanics modes are a discretization/conditioning
  caveat. SPD of a fixed-damage mechanics block does not prove invertibility
  or stability of the full nonsymmetric hybrid/history coupled Jacobian.

### Exact resume point

No root-cause verdict of physical snap-back, absence of equilibrium, or JVP bug
is established. Before another continuation, prioritize a matched-state
full-field/configuration comparison, then a bounded actual-state JVP/directional
merit consistency audit and coupled Schur/singular-mode diagnosis. Account for
history active-set crossings and distinguish reaction softening from a fold in
the prescribed-displacement branch. Do not guess smaller increments, larger
iteration budgets, residual stiffness, or a different loading constraint.

## 2026-09-22 — Complete matched-state, derivative, and coupled-mode diagnosis

- Exact artifact identities were checked before comparison. The strict stagger
  `0.0062` archive has SHA256
  `95851472eb43eb2e24e4d26920d050ecf5ffc1aa83bfe0919f33bd7881cadada`;
  its static/material/union hashes exactly match the monolithic failure archive.
- At their common accepted `u_bar=0.0061000000000000325` state, monolithic minus
  stagger relative infinity differences are `3.7395e-8` for union displacement,
  `2.2580e-8` for damage, and `1.4619e-7` for history. Absolute maxima are
  `2.3988e-10`, `1.8522e-8`, and `6.3176e-5`; reaction relative difference is
  `1.3526e-9`. Largest differences occur at particle `1869` or its y DOF. This
  establishes close common-state parity, not bitwise equality.
- A read-only predictor audit at target `0.00615` found valid mechanics/damage
  JVP blocks but more than one thousand immediate history active-set crossings
  for generic q perturbations. Its fixed-active ideal feedback eigenvalue was
  only `0.1129`, so the predictor itself was not near the later coupled mode.
- One deterministic single-target CPU owner replay, with the archived `0.0061`
  committed state and exactly unchanged Newton controls, reproduced the same
  iteration-8 line-search failure in `163.03s`. It wrote no state or bundle and
  did not invoke continuation. The in-memory failed state had 2383 active rows,
  no history margins within `1e-8`, and no active switches under the small
  finite-difference perturbations used below.
- At that failed state, centered finite-difference relative defects for every
  smooth q/d/H residual block were about `1e-8`. GMRES again passed. The
  computed Newton direction satisfied
  `R dot (J delta) / (-||R||^2) = 0.9999999999994263`; block infinity norms of
  the raw direction were `0.0437861` (q), `2.85569` (d), and `14091.68` (H).
  This rules out a gross JVP sign/scale defect at the actual failure state.
- Read-only merit probing reproduced all archived Armijo failures through
  `alpha=2^-12`; `2^-13` also failed. `alpha=2^-14=6.103515625e-5` was the first
  passing trial, and every tested smaller alpha through `1e-8` passed. There
  were no active-set switches from `alpha<=2^-10`. Thus the 12-backtrack budget
  is the immediate stop and `alpha_min` is not, but accepting one deeper trial
  is not evidence that later Newton iterations or continuation would converge.
- For the fixed-active ideal stagger derivative
  `T=A^-1 C P K^-1 B`, dominant real eigenvalues at accepted states were:
  `0.667669` (`0.0055`), `0.771443` (`0.0058`), `0.810339` (`0.0059`),
  `0.849293` (`0.0060`), and `0.892222` (`0.0061`). Eigen residuals were
  `O(1e-15)`. At the failed Newton state the dominant eigenvalue was
  `0.9955678745`; its eigenvector peaked at particle `1869`, coordinate
  `(0.4997045202, 0.5037544271)`, with participation ratio `1.0563`. The next
  eigenvalue was `0.884778`.
- Because `S=A(I-T)`, the real eigenpair gives
  `sigma_min(I-T) <= |1-lambda| = 0.00443213` and at least `225.6` directional
  amplification for the preconditioned Schur problem. It does not prove exact
  singularity or give the full nonsymmetric condition number. An independent
  numerical audit confirmed the Jacobian/Schur/fixed-point signs and these
  interpretation limits.

### Current diagnosis

The strongest supported explanation is a progressively strengthening,
crack-tip-localized damage-mechanics-history feedback mode. It approaches unit
gain, slows stagger convergence, and amplifies monolithic Newton directions;
the configured monolithic line-search budget then stops before its first
acceptable small step. This is a numerical coupled-conditioning diagnosis, not
proof of physical snap-back, loss of the prescribed-displacement equilibrium,
or a constitutive error.

### Exact resume point

Do not launch a broad parameter study. If the user chooses another solve, the
narrow evidence-based experiment is a single-target replay from the exact
`0.0061` committed state with enough line-search depth to include `2^-14`, while
retaining all equations, gates, load increment, GMRES controls, and rollback.
That experiment must be treated as a globalization diagnostic, not as an
assumed route to `0.0075`. A stronger algorithmic alternative would require a
separate derivation addressing the near-unit crack-tip feedback mode.

## 2026-09-22 — Authorize bounded exact-0.00615 Armijo replay

- The user authorized the narrow diagnostic above. Run one in-memory verified
  attempt from the archived exact `0.0061` committed state to exact `0.00615`.
- Change only `line_search_max_backtracks` from `12` to `14`; preserve the
  direct predictor, `30`-iteration Newton ceiling, GMRES controls, Armijo
  coefficient/reduction/alpha floor, equations, physical gates, and CPU owner.
- Do not invoke continuation, publish an archive, overwrite the failure bundle,
  or increase another budget if the attempt still fails. Record whether Newton
  and the subsequent auxiliary/canonical strict gates complete.

### Exact resume point

Verify the original bundle member hashes, execute the single verified attempt,
then verify the hashes again and record the complete immutable attempt result.

## 2026-09-22 — Complete bounded exact-0.00615 Armijo replay

- Pre/post SHA256 values for the original four bundle members are unchanged:
  archive `f97c60ec...d7b75`, replay `8637e37b...9f50`, summary
  `db4ba66e...f81`, and manifest `7b501f94...f40e`. No production artifact was
  written.
- The authoritative target was loaded from archive metadata as
  `0.006150000000000033` (`0x1.930be0ded28b3p-8`). The committed load was
  `0.0061000000000000325`, and its digest before/after the rejected attempt was
  exactly `49fefd99777f8b4cb9845f5fa92a4a8bb57e783d113310e652f8e010fdefa239`.
- The exact-target in-memory verified attempt took `188.19s`. Its first seven
  accepted corrections exactly followed the prior failure path. Iteration 8
  accepted `alpha=2^-14=6.103515625e-5` after 14 backtracks; merit changed from
  `3.2828627494446846e12` to `3.2827740645014287e12`. The active set was stable,
  GMRES returned `info=0`, and its residual remained below its explicit limit.
- Iteration 9 rejected all 15 Armijo trials from `1` through `2^-14` and failed
  with `line search backtrack budget exhausted`. GMRES returned `info=0` after
  113 callbacks; unpreconditioned residual was `1.01793244429057e-5` against
  limit `2.558372926293337e-4`. Newton therefore did not converge, and auxiliary
  and canonical gates were not evaluated; their failure placeholders must not
  be interpreted as physical gate measurements.
- During target reconstruction, an initial in-memory preflight used arithmetic
  `committed_u_bar + 5e-5`, which is one float64 ulp below the archived target
  (`0x1.930be0ded28b2p-8`). It independently reached the same iteration-9
  14-backtrack failure in `182.32s`. The authoritative result above uses the
  exact archive-metadata target; neither attempt wrote state.

### Current conclusion and exact resume point

The bounded result closes deeper Armijo alone as a remedy: it permits one tiny
correction but does not restore useful nonlinear progress. Stop here unless the
user authorizes a derivation/review phase for a Schur-aware trust-region or
pseudo-transient semismooth Newton step. Such a method must leave the original
residual and strict final gates unchanged, define its adaptive regularization
and acceptance rule explicitly, and pass manufactured tests before another
physical target attempt.

## 2026-09-22 — Authorize Schur-aware globalization derivation

- The user authorized derivation and independent review only. Do not edit the
  solver or run a physical continuation in this phase.
- Compare a scaled full-residual trust-region method against a pseudo-transient
  or Levenberg regularization derived from `S=A(I-T)`. Preserve fixed-active
  semismooth linearization and the original residual/gates as the only final
  convergence contract.
- The selected design must explicitly define the regularized linear system,
  scaling/norm, predicted and actual reduction, acceptance ratio, adaptive
  parameter update, rejection limit, and behavior at active-set changes.
- Require external-reference checking plus independent algebra/numerical review,
  then record a manufactured-only implementation and validation plan. No
  physical tuning values may be guessed.

### Exact resume point

Audit the current scaled Newton operator and preconditioner interfaces, derive
both candidate methods, and select one only after independent review.

## 2026-09-22 — Complete globalization derivation/review

- The first future candidate is now selected as a scaled full-space
  semismooth trust-region Gauss–Newton/LM method, with the strict active mask
  and row scales frozen during each inner model. Its model is

  $$
  m_k(p)=\frac12\lVert F_k+J_kp\rVert_2^2,
  \qquad \lVert D_kp\rVert_2\leq\Delta_k,
  $$

  where `F_k=W_k R(z_k)` and `J_k=W_k J_{A_k}(z_k)`. The damped KKT form is
  `(J_k^T J_k + mu D_k^T D_k)p = -J_k^T F_k`, with `mu` determined by the
  trust-region subproblem. The implementation route must be the augmented
  matrix-free least-squares operator `[J_k; sqrt(mu) D_k]`, not explicit normal
  equations.
- The acceptance contract is frozen-scale and model-based:
  `ared = phi_k(0)-phi_k(p)`, `pred = -F_k^T J_k p - ||J_k p||^2/2`, and
  `rho = ared/pred`, where `phi_k(s)=||W_k R(z_k+s)||^2/2`. Non-finite or
  non-positive `pred` rejects the trial. Thresholds, radius policy, initial and
  maximum radius, inner forcing, and the dimensionally valid full-column-rank
  `D_k` remain project-owned unresolved controls.
- The symbolic adaptive contract to review uses explicit
  `0 < eta_reject < eta_expand < 1`, `0 < gamma_shrink < 1`,
  `gamma_expand > 1`, `0 < theta_boundary < 1`, and `Delta_min > 0`:
  `rho < eta_reject` rejects and shrinks `Delta` by `gamma_shrink`;
  `rho >= eta_reject` accepts; and `rho >= eta_expand` plus
  `||D_k p|| >= theta_boundary*Delta` expands up to `Delta_max`. Invalid
  predictions/trials use the rejection path, and a radius below `Delta_min`
  is a numerical failure. The numerical values and the separate inner-Krylov
  forcing/retry budget remain unresolved project-owned controls; no failed
  inner solve may silently fall back to the current GMRES path.
- An accepted active-set change is evaluated by the exact max residual and
  causes a new active mask/scale/Jacobian model on the next iteration. The
  original residual and strict Phase-2B/canonical gates remain the only final
  convergence contract.
- The current `scaled_jvp` is suitable for a frozen mask, but the current raw
  `vjp` recomputes the mask. A future matrix-free augmented solver therefore
  requires explicit frozen-mask VJP and scaled-transpose APIs. The current
  GMRES block preconditioner cannot be reused as an augmented least-squares
  right preconditioner without a checked transpose action and fixed-shape
  metric composition.
- Pseudo-transient continuation was retained as a materially distinct
  alternative, requiring a project-owned residual-coordinate metric `M`,
  adaptive `tau`, and a separate descent/acceptance contract. It is not the
  selected first route because no physical time/mass operator or adaptive rule
  is declared. The Schur identity `S=A(I-T)` and the observed near-unit mode
  remain diagnostics/preconditioner information, not an outer `S+mu A` solve.
- Source-backed method families are Conn–Gould–Toint trust regions, Moré LM,
  Qi–Sun semismooth Newton, and Kelley–Keyes pseudo-transient continuation.
  Brown–Hindmarsh supports the matrix-free Newton–Krylov pattern. The project
  residual, scaling, active-set tie rule, metric, controls, augmented operator,
  and acceptance/gate contract are project-owned. Publisher full text was not
  reachable in this review; DOI/Crossref bibliographic records were checked.
- A deterministic manufactured block audit independently verified augmented
  LS/KKT equivalence, predicted reduction, Schur factorization, and
  back-substitution; maximum defects were `8.88e-16`, `8.88e-16`, `1.11e-16`,
  and `2.22e-16`. No repository solver behavior or physical state changed.

### Required manufactured validation before implementation or physical execution

1. Frozen-mask scaled JVP/VJP bilinear identity, including a deliberately
   supplied mask that differs from recomputed `active_mask(z)`.
2. Independent small dense comparison of augmented LS, KKT, and QR/SVD for
   multiple positive `mu`, full-column-rank metrics, and the `mu=0` interior
   case.
3. Schur signs, right-hand side, factorization, and back-substitution,
   including row-scaled versus raw block coordinates.
4. Trust-radius/secular solve monotonicity, predicted-reduction identity,
   actual-ratio acceptance, rollback, and explicit invalid-prediction failure.
5. Active-set crossing, exact tie-inactive behavior, accepted switch rebuild,
   fixed shapes/dtypes, and right-preconditioner/transpose equivalence.
6. Synthetic nonnormal `T` cases, including the observed `0.9955678745` mode,
   to verify bounded step behavior without claiming singularity or a physical
   fold.
7. Manufactured final residual and Phase-2B/canonical gate replay; no
   physical continuation until independent review approves the contract.

### Exact resume point

Primary Sol should independently review the trust-region KKT/metric units,
frozen-mask transpose API, augmented-Krylov preconditioning, and the explicit
acceptance/radius/inner-tolerance fields. Keep the solver unchanged and do not
run a physical continuation.

## 2026-09-22 — Complete independent review and select Krylov hookstep route

- Primary review confirmed the full-space Gauss--Newton/LM model, KKT signs,
  frozen row scaling, exact-max trial evaluation, and `ared/pred/rho` signs. It
  found the full-space augmented LSMR design blocked by an undefined variable
  metric, incomplete rank/retry semantics, and the missing frozen-mask scaled
  transpose API.
- The metric is now resolved as the fixed quadrature-normalized continuum `L2`
  product norm recorded in `docs/DECISIONS.md`: displacement uses union volumes
  and scale `l0`; damage uses material volumes and unit scale; history uses
  material volumes and scale `Gc/l0`. Because the packed q block contains only
  free DOFs, zero constrained increments do not create null columns. The metric
  is dimensionless and full rank; its equal normalized field weighting remains
  project-owned and requires two-cloud refinement evidence.
- Independent comparison selected a right-preconditioned Newton--GMRES
  hookstep for the first manufactured route. For `A=J_kP_k`, Arnoldi stores
  `A V_m=V_(m+1) Hbar_m` and `Z_m=P_kV_m`; the small problem minimizes
  `||beta e1+Hbar_m y||/sqrt(2)` subject to `||D Z_m y||<=Delta`, with
  `p=Z_m y`. This uses the existing validated frozen-mask `scaled_jvp` and exact
  block inverse without a new VJP or transpose preconditioner.
- Metric whitening must use thin QR/SVD of `D Z_m`; forming only its Gram matrix
  is rejected because it squares conditioning. The small solve supports a
  minimum-metric-norm unconstrained interior solution and a positive-multiplier
  boundary solution. Rank loss of `D Z_m`, invalid Arnoldi/preconditioner
  actions, and missed inner tolerances fail explicitly; rank loss of the
  projected Jacobian is handled by SVD and trust regularization.
- Rejected nonlinear trials preserve the frozen state/mask/scales/metric,
  preconditioner, and Krylov basis, then shrink the radius and resolve the small
  subproblem. They must recompute the candidate, full `scaled_jvp` prediction,
  and exact nonlinear residual. Accepted trials discard the basis and rebuild
  all state-dependent data. A maximum-space zero-prediction nonroot is reported
  as Krylov-stationary, not converged or globally stationary.
- External metadata/abstract checks confirm Bellavia--Morini
  (`10.1137/040611951`) for low-dimensional Krylov trust-region methods on large
  nonlinear systems and Viswanath (`10.1017/S0022112007005459`) for the
  Newton--Krylov/locally constrained optimal hookstep combination. The dedicated
  paper-reader failed twice on its own file parser; DOI/Crossref and SciPy LSMR
  documentation were read directly instead. No inaccessible page/equation
  claims are made.

### Revised manufactured validation contract

1. Verify right-preconditioned Arnoldi orientation and the full/projected model
   identity on independent nonsymmetric dense systems.
2. Verify the explicit volume metric, dimensionless units, positive/full-rank
   reduced action, exact constrained-DOF injection, and refinement stability on
   at least two manufactured clouds.
3. Compare QR/SVD-whitened projected interior and boundary hooksteps against a
   dense generalized trust-region oracle, including singular projected
   Jacobians, happy breakdown, and rank-deficient metric-basis hard failure.
4. Prove production `pred` comes from a fresh full `scaled_jvp`; check its
   equality to the projected prediction to the declared Arnoldi tolerance.
5. Cover rejected-radius re-solve with bitwise-identical frozen basis data,
   accepted-radius retain/expand cases, `Delta_min`, rejection exhaustion,
   invalid/non-finite trials, and inner failures that do not masquerade as
   model rejection.
6. Cover exact root, positive-prediction interior/boundary steps,
   Krylov-stationary nonroot, active-set crossing, exact tie-inactive behavior,
   and mandatory rebuild after accepted switches.
7. Keep a dense full-space augmented-LS/KKT oracle, including frozen scaled
   transpose and nonuniform damage-volume dual scaling, as independent evidence;
   it is not the first production inner solver.
8. Add separate hookstep configuration/diagnostic/archive names. Do not reuse
   GMRES/Armijo fields with changed semantics. Final Phase-2B/canonical gates and
   original residual remain unchanged.

### Exact resume point

The derivation/review phase is complete. No numerical radius, rank, Arnoldi, or
inner-tolerance values have been selected, and no solver edit or physical run is
authorized. If the user approves implementation, first define those explicit
manufactured-only controls and the new immutable audit schema, then implement
the metric and small dense hookstep independently of the existing Armijo path.

## 2026-09-22 — Authorize manufactured hookstep core implementation

- Implement only the isolated hookstep algebra and immutable audit contract.
  Do not dispatch it from `solve_monolithic_newton`, Phase-2B, continuation,
  workflow, or archive code, and do not run a physical target.
- Every numerical control is an explicit required configuration field. The
  repository may use test-specific manufactured values, but this phase does not
  select a physical-run configuration.
- Required implementation units are the fixed quadrature metric, right
  preconditioned Arnoldi basis, QR/SVD metric whitening, minimum-metric-norm
  interior solve, safeguarded positive-multiplier boundary solve, and immutable
  success/failure evidence.
- Verification must use independent dense references and include units/rank,
  interior/boundary, stationary nonroot, happy breakdown, non-finite action,
  and at least two-cloud metric-refinement cases.

### Exact resume point

Add the isolated APIs and focused tests first. Run only the smallest monolithic
test modules needed for this new behavior, then request independent code and
numerical review before any solver integration proposal.

## 2026-09-22 — Complete isolated manufactured hookstep core

- Added the required `MonolithicHookstepConfig` and immutable metric, Arnoldi,
  projected-step, direction, radius-decision, and numerical-failure evidence.
  No default numerical controls were introduced.
- The metric is the selected quadrature-normalized `q/d/H` continuum product
  norm and is bound to the exact model/static owner. Arnoldi additionally
  rebuilds and compares the canonical metric, rejecting same-owner forged
  diagonals or normalization values.
- `build_monolithic_hookstep_arnoldi(...)` constructs `J P V`, stores
  `Z=P V`, uses two-pass MGS, audits orthogonality, distinguishes exact root
  from numerical failure, and retains the terminal vector for both ordinary and
  nonzero tolerance-triggered breakdown.
- `solve_projected_monolithic_hookstep(...)` uses the thin SVD of `D Z`, fails
  loudly on metric-basis rank loss, admits rank-deficient projected Jacobians,
  and computes minimum-metric-norm interior or positive-multiplier boundary
  steps. Predicted reduction uses the stable inner-product expression rather
  than subtracting nearly equal squared norms.
- `compute_monolithic_hookstep_direction(...)` reconstructs the full packed
  step, reevaluates the full frozen-mask scaled JVP, enforces metric and
  projected/full prediction consistency, and rebinds projected failures to
  owner/state/mask/scale evidence. `decide_monolithic_hookstep_radius(...)`
  exercises the explicit acceptance, retain/expand, shrink, floor, rejection
  budget, and non-finite-trial contracts without mutating solver state.
- Focused tests cover all-required controls, exact root, metric formula and
  immutability, same/different-owner forgery, constant-field and displacement
  two-cloud refinement behavior, independent generalized-JVP Arnoldi
  orientation, terminal-vector reconstruction, exact and tolerance happy
  breakdown, interior and boundary KKT solutions, tiny positive prediction,
  singular projected Jacobian, metric rank loss, stationary nonroot,
  non-finite actions/trials, full/projected mismatch, and radius decisions.
- Verification passes: hookstep `18/18`; combined hookstep, residual/JVP, and
  existing Newton/preconditioner `56/56`; changed-module `py_compile`; and
  `git diff --check`. Independent numerical re-review found no remaining
  finding; independent code re-review found no blocker. The only retained low
  observation is that `trust_radius_initial` belongs to the deliberately absent
  future controller and is therefore not consumed by an isolated explicit-
  radius operation.
- No physical continuation was run, no artifact was written, and no call path
  from the existing Newton, Phase-2B, continuation, workflow, or archive layer
  reaches the hookstep core.

### Exact resume point

Stop at the authorized boundary. Preserve the isolated core and existing
Armijo behavior. Do not select physical hookstep controls or add solver,
workflow, archive, or continuation dispatch without a new explicit
authorization. If integration is authorized, design and manufactured-test the
outer rejected-radius re-solve/state-rebuild controller before any physical
target attempt.

## 2026-09-22 — Authorize isolated outer-controller implementation

- The user selected only the manufactured outer-controller stage. Solver,
  Phase-2B, continuation, workflow, archive, and physical execution remain out
  of scope.
- One call may build one frozen metric/Arnoldi model, start from the explicit
  configured initial radius, re-solve the projected problem after rejected
  nonlinear trials, and return the first accepted candidate with immutable
  attempt evidence.
- Rejection must preserve the exact state, scales, active mask, and Arnoldi
  model. Acceptance terminates the call and marks all state-dependent data for
  caller rebuild, including an explicit active-set-change comparison.
- Non-finite nonlinear trial evaluation follows radius rejection/shrink. Inner
  operator, step, metric, projected-solve, or prediction failures remain hard
  numerical failures and must not masquerade as model rejection.

### Exact resume point

Define the minimal immutable controller result/attempt contract, implement it
without dispatch, and manufactured-test rollback, basis reuse, radius floor and
budget, finite acceptance, non-finite trial rejection, accepted active-set
change, and mandatory rebuild signaling. Then run affected regressions and
independent numerical/code review.

## 2026-09-22 — Complete isolated manufactured hookstep outer controller

- Added `run_monolithic_hookstep_correction(monolithic, z, gate_config,
  active_mask, config)` plus `MonolithicHookstepTrial`,
  `MonolithicHookstepCorrection`, `MonolithicHookstepControllerFailureAudit`,
  and `MonolithicHookstepControllerFailure`. One call builds one canonical
  metric, one preconditioner/Arnoldi model, starts at `trust_radius_initial`,
  re-solves only the projected problem after rejected radius trials, evaluates
  exact nonlinear trial residual under frozen scales, and returns after the
  first accepted candidate with `rebuild_required=True` and an explicit
  `active_set_changed` comparison.
- The controller derives its own scales from the explicit `gate_config`; it
  does not accept caller-supplied scale objects, closing the forged-scale path.
  Rejected trials preserve the exact state, scales, active mask, metric, and
  Arnoldi arrays; accepted corrections never continue under a stale model.
- Terminal numerical evidence is complete: unrepresentable finite ratios,
  subnormal radius-shrink underflow, rejection-budget exhaustion, radius-floor
  crossing, trial `LinAlgError`, accepted-mask rebuild failure, and pre-model
  scale/active-mask failures all raise immutable audits retaining the
  triggering trial where one exists. Non-finite candidate/residual/actual
  reduction follows explicit radius rejection, while a finite trial residual is
  retained even when the derived actual reduction overflows.
- Numerical hardening: `_stable_l2_norm` gives `beta` and the reference-residual
  binding underflow safety; the projected reduction uses the stable
  inner-product form; and the infeasible-step guard uses the exact
  `radius + secular_rtol*radius` expression shared with the projected radius
  tolerance.
- Verification: focused hookstep controller `16/16`, hookstep core `19/19`, and
  combined controller/core/residual/JVP/Newton `73/73`; changed modules compile;
  `git diff --check` passes. Two independent reviews of the completed controller
  report no remaining blockers.
- No physical continuation was run, no artifact was written, and no solver,
  Phase-2B, continuation, workflow, or archive path dispatches the controller.

### Exact resume point

Stop at the authorized boundary. Preserve the isolated controller and existing
Armijo behavior. Do not select physical hookstep controls or add solver,
workflow, archive, or continuation dispatch without a new explicit
authorization. If integration is authorized, first design and manufactured-test
the outer multi-correction Newton loop (accepted-correction rebuild and
convergence gates) before any physical target attempt.

## 2026-09-22 — Read-only single-step hookstep probe at the failure iterate

- Only the authoritative failure bundle was read; no state or artifact was
  written. The exploratory hookstep controls are explicit dimensionless test
  values and are not an authorized physical-run configuration.
- Reconstructed the owner model from the manifest execution command, read the
  committed `u_bar=0.0061000000000000325` state (digest `49fefd99...`) directly
  from `archive.npz`, rebuilt the exact archived engineering/newton configs, and
  formed the authoritative predictor with `_continuation_expected_initial_z`.
- Reproduced the production failure in memory in `121.5s`:
  `solve_monolithic_newton` failed at stage `line_search` with
  `line search backtrack budget exhausted`, yielding the exact failure iterate
  `current_z` with scaled residual norm `2.562367e6` and `2383/3395` active
  particles.
- Single `run_monolithic_hookstep_correction` at that frozen iterate with
  `max_krylov_dimension=32` and `trust_radius_initial=1.0`:
  - radius `1.0`, `0.25`, `0.0625`, `0.015625` rejected with negative actual
    reduction (`rho` far below `eta_accept`);
  - radius `0.00390625 = 2^-8` accepted with `rho=0.913604943391082` and actual
    reduction `2.986334353397521e12`;
  - `active_set_changed=True`; controller returned `rebuild_required=True` and
    `next_radius=0.0078125` (`2*radius`, expanded).
- Conclusion: at the exact iterate where Armijo globalisation failed, the
  bounded right-preconditioned hookstep finds a positively reducing step and a
  high-quality model ratio. This is a positive signal that the trust-region
  route addresses the near-unit notch-tip mode; it is a single frozen-model
  correction only and does not by itself prove the `0.00615` target is
  reachable.

### Exact resume point

If authorized, implement and manufactured-test the outer multi-correction
trust-region Newton loop (accepted-correction rebuild, convergence/active-set
gates) and then run one bounded physical target from the committed `0.0061`
state. Do not select production hookstep controls or run a physical target
without a new explicit authorization.

## 2026-09-22 — Complete the trust-region Newton solver

- Added the fixed-load trust-region Newton solver
  `solve_monolithic_hookstep_newton(monolithic, initial_z, config, gate_config)`
  with `MonolithicTrustRegionConfig` (`max_newton_iterations` + hookstep
  controls), `MonolithicTrustRegionDiagnostics`,
  `MonolithicTrustRegionResult`, `MonolithicTrustRegionFailureAudit`, and
  `MonolithicTrustRegionFailure`. The accepted radius is carried across
  iterations by rebuilding the frozen hookstep config with
  `trust_radius_initial=next_radius`; the active set and row scales are rebuilt
  after every accepted correction, and convergence uses the same strict gates
  as the Armijo core (scaled block norms `<= 1`, active set stable, correction
  gates).
- Manufactured verification: `tests/test_fracture2d_nonlocal_engineering_grip_monolithic_trust_region.py`
  passes `23/23`, including perturbed-state convergence and radius reuse,
  max-iteration and controller-failure audits, exact root, accepted-trial and
  bare-hookstep-failure wrapping, and mid-loop exact-root staging. The combined
  affected command passes `96/96`.
- Read-only physical validation from the committed `0.0061` state (digest
  `49fefd99...`), exploratory controls, no artifact written:
  - target `0.00611` CONVERGED in `6` iterations (`47-69s`), final scaled block
    norms `(2.78e-5, 1.21e-8, 1.00e-6)`;
  - targets `0.00612` and `0.00615` did not converge within `40` iterations and
    hit the iteration budget, with the radius collapsing near the notch-tip
    near-unit mode.
  - This is the first advance past `0.0061` in the monolithic path, but it
    shows the peak-region increment must be small (`~1e-5`) rather than the
    configured `du_min=5e-5`.
- Two independent reviews found and closed three blockers (accepted-trial audit
  scale/state pairing, escaping `MonolithicHookstepFailure`, mid-loop exact-root
  crash) plus result/failure relational-validation gaps. The final reviews
  report no remaining blockers.

### Exact resume point

The numerical trust-region solver is complete and manufactured-verified, and
advances the real problem to `0.00611` in memory. It is not yet wired into
`run_verified_monolithic_attempt`, the continuation controller, the workflow,
or the archive. Before any physical/durable run: (1) integrate the solver as an
explicitly selected attempt runner while preserving the strict Phase-2B/canonical
acceptance and archive schema, or define a new schema; (2) select an explicit
near-peak increment schedule; (3) obtain explicit authorization for a bounded
physical continuation. Do not silently overwrite existing artifacts.

## 2026-09-23 — Monolithic avalanche barrier at 0.006116

- Adaptive-increment continuation from the committed `0.0061` state with the
  trust-region solver accepted `0.00611` (`du=1e-5`), `0.006115` (`du=5e-6`),
  and `0.006116` (`du=1e-6`). Attempts at larger increments (`1e-5`, `2e-5`) were
  rejected with the radius collapsing, and the accepted-state checkpoint is
  `/tmp/opencode/continuation_state.npz` at `u_bar=0.006116000000000032`.
- The barrier is not step size: from `0.006116` even `du=1e-6` to `0.006117`
  fails at the Newton budget with scaled block norms `(4.26e2, 1.83e1, 1.34e1)`,
  `rho=0.208`, radius `4.88e-6`. A single Newton trajectory at `0.00612` reaches
  `rho=1` with `norms=(2.60e3, 9.2e1, 9.2e1)`, merit constant to seven digits,
  and radius collapsing to `6.1e-7` — a least-squares/GN stationary point, not a
  root. The Armijo path plus the trust-region solver from its failure iterate at
  `0.00615` also stalls.
- Diagnosis: the monolithic residual's history block is the nonsmooth
  `H - max(H_committed, psi_plus)` complementarity. The frozen-mask Gauss-Newton
  model cannot cross the avalanche where many particles flip active; every
  finite trial in the GN direction increases the true residual while the model
  predicts a decrease, so the trust region collapses to a spurious local minimum
  of the least-squares merit. Equilibria at higher loads do exist (the exhausted
  stabilized path reached `0.00623125`), so this is a globalization defect of the
  monolithic Newton, not proof of nonexistence.
- This matches the task's prior conclusion that a "materially new nonlinear
  formulation (for example an independently derived monolithic
  complementarity/semismooth-Newton method or a derived continuation
  constraint)" is required, and that it must not be guessed.
- No repository or archive artifact was written; all runs were in-memory and the
  only outputs are `/tmp/opencode` logs and the checkpoint.

### Exact resume point

The monolithic trust-region route advances the strict-equilibrium state from
`0.0061` to `0.006116` and is then blocked by the avalanche. The next decision
is the mathematical one the task already flagged: approve and derive a
semismooth/complementarity active-set globalization (or an explicit
avalanche-jump/arc-length continuation constraint) for the history block, or
stop with the improved but incomplete endpoint. Do not guess the formulation;
do not silently relax the strict gates or reuse a rejected state as evidence.

## 2026-09-23 — Correct the 0.006117 stall diagnosis

- Reconstructed the chain from the authoritative committed `0.0061` state and
  ran the canonical auxiliary sweep and Phase-2B-equivalent gates after each
  accepted scratch solve. Targets `0.00611`, `0.006115`, and `0.006116` pass;
  raw-to-canonical `L-infinity` differences are respectively `4.718004e-12`,
  `1.080025e-12`, and `1.179501e-12`. The terminal scratch checkpoint is
  `/tmp/opencode/canonical_trust_state.npz`. It is not a verified-attempt,
  continuation, workflow, or archive artifact.
- A detailed K32 run at `0.006117` records positive actual reduction on every
  accepted correction. The active mask settles at `1887` particles by iteration
  17 and remains bitwise fixed, so the earlier claim that ongoing active-set
  flips prevent progress is false. At iteration 40 the scaled block norms are
  `(425.390, 14.191, 15.313)` and both K32/K64 projected-gradient ratios are
  large (`3.762`, `3.918`), disproving full Gauss-Newton stationarity there.
- Suppressing radius expansion is not a remedy. It drives the radius down to
  `9.5367431640625e-9` and terminates on projected/full prediction consistency.
  At that state K32 reports a projected-gradient ratio `5.835e-5`, while an
  independently rebuilt K64 space reports `0.3425`; the apparent K32
  stationarity is only subspace exhaustion.
- A direct 80-correction K64 run does not converge. It ends with scaled block
  norms `(411.715, 12.898, 13.717)` and large K32/K64 projected-gradient ratios
  (`3.974`, `3.977`). K64 improves the trajectory modestly but does not remove
  the stall, so Krylov dimension alone is not the remedy.
- Block diagnostics locate the fixed-mask stall in the explicit history
  unknown. Once the mask is stable, each accepted correction changes `H` by
  about `0.11528` in `L-infinity`, while the net two-correction history change
  falls below `4.5e-5`; corresponding `q` and `d` two-correction changes are
  about `1e-9` and `1e-8`. Residual vectors remain almost parallel (cosine above
  `0.99998`). This is a numerical two-cycle under the tested normalized
  continuum-product trust metric, not an active-mask cycle.
- The earlier section's claims of a step-size-independent avalanche barrier,
  full-space least-squares stationarity, and universally increasing finite
  trials are superseded by this evidence. No repository solver, physical
  equation, strict gate, durable archive, or authoritative data was changed by
  these read-only diagnostics.

### Exact resume point

The next materially different option is exact algebraic elimination of the
history unknown using the existing equation
`H=max(H_committed, psi_plus(q))`, followed by an independently checked reduced
residual/JVP derivation and manufactured-only tests. Because this changes the
declared monolithic unknown set even though it preserves the history equation,
obtain explicit user authorization before implementation. Do not select a new
trust metric, arc-length constraint, or relaxed gate by inference.

## 2026-09-23 — Complete manufactured exact history elimination

- Added the isolated owner-bound `ReducedMonolithicGripResidual`. Its state is
  exactly `(q,d)`; `history(...)` reconstructs
  `H=max(H_committed, psi_plus(q))`, `lift(...)` embeds that state in the full
  owner, and `residual(...)` requires the lifted history row to be exactly zero.
- The strict mask remains `psi_plus > H_committed`; exact ties are inactive.
  The reduced JVP recovers `delta_H=-J_Hq delta_q` under one frozen mask and
  applies the full owner JVP to `(delta_q,delta_d,delta_H)`, retaining the
  required `-J_dH J_Hq` damage-chain term and canonical full-row operation
  order. Its history action must cancel exactly.
- Independent test-only algebra rebuilds BA mechanics, Amor history, and the
  cubic material damage graph. It verifies the reduced residual, history,
  mixed active/inactive eliminated JVP, fixed-mask finite differences, exact
  tie behavior, raw out-of-range damage evaluation, shape/dtype failures, and
  input immutability. Independent and production operation orders differ in
  history by only float64 roundoff, so only the canonical owner is used to
  manufacture a bitwise exact tie.
- Verification passes: focused residual `29/29`; combined residual/Newton/
  hookstep/controller/trust-region `88/88`; changed-file `py_compile`; and
  `git diff --check`. Independent numerical and code re-reviews found no
  blocker.
- No reduced scale type, preconditioner, trust metric, nonlinear solver,
  verified-attempt/continuation/workflow/archive dispatch, physical run, or
  artifact was added.

### Exact resume point

Stop at the authorized manufactured residual/JVP boundary. A future reduced
solver requires a separately derived and authorized row scale, two-block
preconditioner, trust metric (simple `q-d` product versus the non-diagonal
history pullback are materially different), convergence gates, and immutable
audit schema. Do not infer those choices from the full `q-d-H` hookstep and do
not run a physical target before manufactured solver review and explicit run
authorization.

## 2026-09-23 — Authorize manufactured reduced solver implementation

- Select lifted full-scale construction with only `Lq/Lrho` acting on reduced
  mechanics/damage rows. The reconstructed nonlinear history correction and
  active-set-stability gates remain mandatory.
- Select the frozen differential pullback of the existing normalized `q/d/H`
  metric through `delta_H=-J_Hq delta_q`; use a rectangular factor action and
  rebuild after each accepted correction.
- Select an explicitly right-oriented lower-triangular preconditioner retaining
  the reduced history-chain lower-left block and omitting only the
  mechanics-from-damage upper-right block.
- Reuse the independently verified projected SVD, prediction, rejection, and
  radius policy with reduced-specific state pairing and immutable audits.
- Scope remains manufactured-only. Do not add durable integration, select
  physical controls, or run a physical target.

### Exact resume point

Implement reduced scaling/metric/preconditioner and independent dense evidence
first. Only after those pass, add the isolated reduced hookstep controller and
fixed-load Newton loop with owner replay and failure audits. Preserve all full
`q-d-H` APIs and behavior.

## 2026-09-23 — Complete the manufactured reduced solver layer

- Foundations: `ReducedMonolithicNewtonScales` (canonical exact-lift scales,
  only `Lq/Lrho` act on reduced rows), `ReducedMonolithicTrustMetric` (frozen
  rectangular differential pullback of the full normalized metric through
  `delta_H=-J_Hq delta_q`), and
  `ReducedMonolithicLowerTriangularRightPreconditioner` (exact `K`/`A_H` sparse
  factors, retained `C_A` history chain, omitted upper-right mechanics block).
  The reduced adapter freezes its owner binding, and metric/preconditioner
  re-check construction-time owner pairing at apply time. `scaled_residual`
  evaluates arbitrary nonlinear trial states under frozen row scales, matching
  the full owner; `scaled_jvp`, metric, and preconditioner keep strict
  reference-state pairing.
- Solver: `run_reduced_monolithic_hookstep_correction(...)` and
  `solve_reduced_monolithic_hookstep_newton(...)` with reduced-specific
  immutable Arnoldi, trial, correction, controller-failure, diagnostics,
  failure, and result types. Rejected radius trials reuse one frozen
  state/scales/mask/metric/preconditioner/Arnoldi model; accepted corrections
  rebuild all state-dependent data and require active-set comparison.
  Convergence requires both scaled norms `<=1`, active-set stability, and the
  displacement, damage, and reconstructed-history correction gates. Exact
  roots (initial and mid-loop) bypass metric/Arnoldi; failures distinguish
  radius rejection from hard inner algebra failure and retain partial audits.
  `validate_with_owner` replays final residuals, scales, masks, JVP prediction,
  and all correction metrics.
- Verification: reduced foundation `7/7`; reduced trust-region `21/21`;
  combined residual/reduced/newton/hookstep/controller/trust-region battery
  `116/116`; changed-file `py_compile`; `git diff --check`. Two independent
  reviews (numerical and code) found no blocker; their low-severity hardening
  findings were then closed: accepted-correction metric state/digest/mask
  pairing with a forgery test, a discriminating radius-slack regression, and
  an exact-initial-root requirement for trial-free results.
- Scope: the shared projected-hookstep core was extracted textually unchanged
  and is used by both full and reduced wrappers; no workflow, continuation,
  archive, CLI, verified-attempt, or physical-run dispatch exists for any
  reduced symbol.

### Exact resume point

The authorized manufactured reduced solver layer is complete and independently
reviewed. The user then authorized the reduced physical path to full loading
(`u_bar=0.0075`): explicit controls reused from the already recorded full-run
and exploratory physical probes, an isolated reduced physical attempt and
continuation seam, bounded read-only probes from the committed `0.0061` state,
and the full strict run with durable artifacts. Strict gates, no warning
acceptance, and all prior artifacts remain mandatory.
The full `q-d-H` solver, gates, archives, and authoritative artifacts remain
unchanged.

### Reduced physical path — 2026-09-23

- Authorized scope and control-selection rule are recorded in
  `docs/DECISIONS.md` (2026-09-23, reduced physical path to full loading).
- Initial controls reuse the previously recorded dimensionless exploratory
  values: `max_krylov_dimension=32`, `trust_radius_initial=0.01`,
  `trust_radius_min=1e-14`, `max_radius_rejections=20`, `gamma_shrink=0.5`,
  `secular_rtol=1e-12`, `max_newton_iterations=40`, with the archived strict
  engineering gates and schedule (`du_initial=1e-4`, `du_min=5e-5`,
  `cutback_factor=0.5`, `max_cutbacks=8`).
- First step is a read-only in-memory probe from the authenticated committed
  `u_bar=0.0061` state (digest `49fefd99...`) at `0.00611` upward to the former
  `0.006117` barrier and beyond, using the auxiliary-sweep canonicalization
  identical to the prior probes. No artifact is written by the probe.
- A durable reduced artifact layer is required before a physical run is
  claimed; its exact schema is selected only after the probe establishes that
  the reduced path advances.

### Reduced physical probe results — 2026-09-23

- Hot-path equivalence-preserving optimizations were required first: exact-state
  history memoization in `ReducedMonolithicGripResidual`, consumer scale checks
  reduced to identity/state pairing (canonical full-scale validation remains at
  construction), and the reduced history differential reuses the owner's cached
  Amor first-variation factor. On the physical cloud a warm `scaled_jvp`
  dropped from `2.59 s` to `0.061 s` and each trust-region iteration from
  minutes to about `9 s`. The affected manufactured battery still passes
  `116/116`.
- Read-only probe from the authenticated committed `u_bar=0.0061` state
  (digest `49fefd99...`), exploratory controls K32/`r0=0.01`/`rmin=1e-14`/
  `max_rejections=20`/`gamma_shrink=0.5`, strict archived gates, auxiliary-sweep
  canonicalization, no artifact written:
  - `0.00611`: reduced Newton converged in `6` iterations (norms
    `(3.09e-5, 1.69e-7)`), all canonical gates pass; the full trust-region
    solver also converged in `6` iterations here.
  - `0.006115`: converged in `7` iterations, all canonical gates pass.
  - `0.006116`: converged in `6` iterations, all canonical gates pass.
  - `0.006117` (`du=1e-6` from `0.006116`): the reduced trust-region loop
    accepted iterations 1--12 with growing radius rejections, then failed hard
    at controller iteration 13 with
    `reduced hookstep projected/full prediction exceeds its tolerance`. This is
    a different failure mode from the former full `q-d-H` history two-cycle.
- Diagnosis of the `0.006117` failure and of a half-step `0.0061165` target is
  in progress; no gate, tolerance, metric, or control has been changed.

### Barrier diagnosis above 0.006116 — 2026-09-23

- The reduced trust-region stall is not step size, Krylov dimension, or metric
  choice. From the committed `0.006116` state:
  - `du=5e-7` (`0.0061165`): radius collapses to `1e-14`, scaled norms plateau
    at `(67.9, 6.58)`, controller radius-floor failure; read-only.
  - `du=2.5e-7` (`0.00611625`): strictly converges in `6` iterations, all
    canonical gates pass (digest `39ebc9d4...`).
  - From `0.00611625`, the next `du=2.5e-7` (`0.0061165`) stalls at
    `(68.4, 5.78)` after the full `40`-iteration budget; read-only.
  - K64 at `0.0061165` stalls at `(66.1, 5.50)` and fails at iteration `34`;
    K32 with the truncated direct `q-d` metric stalls at `(67.9, 5.82)`.
- The authenticated strict stagger state at `u_bar=0.00623125`
  (digest `a5e0e6b7...`, source failure archive `a1743d...`) is a strict root
  of the reduced residual at its own load: scaled norms `(0.945, 2.0e-5)` and
  `H(q) - H_committed = 0` bitwise. The monolithic predictor branch and this
  stagger branch are joined by that authenticated state, not by a monolithic
  solve through `0.0061165`.
- Warm-starting the reduced continuation from that stagger state also stalls on
  the next increment: `du=1.25e-6` plateaus at `(13.8, 2.55)` after `40`
  iterations; `du=2.5e-7` plateaus at `(1.12, 0.43)` after `40` iterations.
  The near-gate plateau is a least-squares/globalization stall, not a radius or
  tolerance failure.
- The warning archive state at `0.0063` has reduced scaled norms
  `(0.966, 57.3)`: the warning branch is far from a strict root by the damage
  equation. Read-only strict reduced solves from the warning states at `0.0063`
  and `0.0075` are in progress to determine whether strict roots exist beyond
  the stall and whether warning states are usable initial guesses only.
- No gate, tolerance, metric, control, or artifact has been changed; all runs
  are read-only scratch under `/tmp/opencode`.

### Fold resolution and the strict full-loading endpoint — 2026-09-23

- Fixed one shared spurious guard: `MonolithicHookstepDirection.__post_init__`
  rejected a reconstructed metric norm using only
  `radius + radius_tolerance`, even though the reconstructed norm may legally
  carry the separately audited `metric_consistency_rtol` defect. It now uses the
  same `radius * (1 + secular_rtol) / (1 - metric_consistency_rtol)` envelope
  as the already reviewed radius decision. Affected battery after the fix:
  `139/139` pass.
- Strict-state status map of the damped stabilized-warning archive
  (`engineering_grip_damped_warning_o5_u7p5e3_dumin6p25e6_s144.npz`, 76 states):
  loads `0.0000`--`0.0059` are `stabilized_converged`; `0.0060`--`0.0068` and
  `0.0071` are `stabilized_warning_accepted`; `0.0069`, `0.0070`, `0.0072`--
  `0.0075` are `stabilized_converged`.
- The strict full-loading endpoint is independently validated. Committed
  `0.0074` state digest `70540ddc...` (stabilized_converged), seed `0.0075`
  state digest `ddb3942f...` (stabilized_converged): the reduced Newton
  converged in `2` iterations at `0.0075` with norms
  `(2.28e-5, 4.19e-11)`, the auxiliary sweep passed, and all canonical gates
  passed with norms `(2.29e-5, 1.07e-8, 0.0)`. The strict endpoint state digest
  is `21f17cab24426184e9b6a65f6bd1d97195401d8e1307f8cbf527b9fdbb077689` with
  `d_max = 9.999630e-01`.
- The strict path between the stagger root at `0.00623125` and the strict
  post-fold loads is blocked by history path dependence, not by the solver
  controls: at the `0.0069` strict seed, the reduced damage norm is `1.5e3`
  when committed from the `0.00623125` strict state, but `2.1e-4` when
  committed from the warning-archive `0.0068` state. A history jump cannot be
  made strict without resolving the intermediate loads.
- Warning states are not usable strict seeds: the `0.0063` warning state
  strictly polished from the `0.00623125` committed state diverged to
  `(37.6, 147.3)` after `60` iterations and was correctly rejected.
- Conclusion: a strictly gated *continuation* across the fold requires a
  materially new globalization/continuation derivation (pseudo-transient,
  arc-length, or semismooth complementarity). It is not reachable by the
  currently derived step-size, Krylov, metric, preconditioner, or seed
  controls, and no gate or equation was changed.

### Fracture-energy arc-length authorization — 2026-09-24

- The user selected fracture-energy-controlled arc-length continuation for the
  strict fold crossing. The augmented unknown is `(q, d, u_bar)`; intermediate
  load reversal is permitted. The path quantity must be the existing discrete
  nonlocal AT2 fracture energy, derived from the repository operator rather than
  copied from a local FEM expression.
- The reduced equilibrium residual, exact
  `H=max(H_committed, psi_plus(q))`, tie-inactive derivative, strict canonical
  gates, and commit/rollback semantics remain unchanged. Trial iterates do not
  update committed history; only a strictly accepted augmented equilibrium can
  commit it.
- Before physical execution, derive and independently manufacture-check the
  discrete energy increment and derivative, exact reduced load column,
  augmented metric and predictor orientation, bordered correction, step
  adaptation/cutback, active-set rebuild, and immutable audits. Existing
  fixed-load APIs and artifacts remain unchanged.

## 2026-09-24 — Complete the fracture-energy arc-length foundation

- Added the surface-only nonlocal AT2 energy and exact Euclidean gradient to
  `MonolithicGripResidual`, using validated directed operator edges and the
  existing material volumes. The quantity is independent of committed history
  and evaluates raw damage without clipping.
- Added the exact prescribed-load direction and the frozen-active reduced load
  derivative to `ReducedMonolithicGripResidual`. It uses strict
  `psi_plus > H_committed` masks, supports signed load increments, cancels the
  full history row exactly, and returns the mechanics/damage reduced column.
- Constructor validation now checks reciprocal directed edges, no self-edges or
  duplicates, finite negative off-diagonal operator coefficients, and
  volume-weighted self-adjointness before caching the surface graph data.
- Added six independent manufactured tests for directed energy/gradient
  formulas, directional finite differences, all-inactive/mixed/tie active sets,
  float64/scalar/shape validation, raw out-of-range damage, history
  independence, and input immutability.
- Verification passes: arc-foundation `6/6`, combined residual/foundation/
  reduced-Newton/reduced-trust-region battery `63/63`, changed-file
  `py_compile`, and `git diff --check`.
- Scope remains foundation-only: no augmented metric, predictor, bordered
  correction, step controller, nonlinear arc-length solver, workflow/archive
  integration, or physical continuation was added or run.

### Exact resume point

The authorized surface-energy and frozen-active load-column foundation is
complete. Primary Sol review must confirm the API names, directed-edge
operator extraction, and exact history-row cancellation before any augmented
metric/predictor or arc-length solver layer is implemented. Preserve all
existing fixed-load APIs, artifacts, and physical blockers.

## 2026-09-24 — Implement the manufactured fixed-increment arc solver

### Status

The next authorized manufactured-only layer is implemented in the isolated arc
module. It does not change the existing fixed-load solver APIs and is not wired
to continuation, workflow, archive, or physical execution.

### Implemented

- `ArcTrustRegionConfig` requires `max_newton_iterations`, the existing
  `MonolithicHookstepConfig`, and a scale-aware bordered-denominator
  `GateTolerance`.
- Augmented right-preconditioned Arnoldi starts from `F/||F||`, applies the
  bordered right preconditioner before the frozen JVP, and retains terminal
  vector, Hessenberg, preconditioned basis, rectangular metric basis,
  happy-breakdown, and orthogonality evidence.
- Rejected radii reuse one frozen metric/preconditioner/Arnoldi model, resolve
  only the projected hookstep, and evaluate the exact frozen nonlinear
  residual. Signed increments may reverse, but every absolute trial load must
  remain `u_bar >= 0`; negative absolute loads and non-finite candidates are
  explicit radius rejections without commit or fallback.
- Accepted corrections rebuild the reduced adapter, active mask, row scales,
  energy scale, metric, and bordered preconditioner from a load-only clone of
  the previous owner. Immutable static/CSR/gradient data and the cached surface
  Hessian absolute-sum scalar are shared; committed path damage/history remain
  unchanged.
- Numerical `FloatingPointError`/`LinAlgError` construction failures are
  retained as audited arc failures at their exact boundary. Nonlinear-trial
  `FloatingPointError` follows radius rejection; trial `LinAlgError` is a hard
  failure. Structural `TypeError`/`ValueError` and API/pairing errors propagate.
- The outer fixed-`delta_G` solve checks reduced mechanics/damage residuals,
  exact energy evidence, active-set stability, physical displacement/damage/
  history correction gates, and non-negative final load. Results, diagnostics,
  replay checks, and failure audits are immutable. Result construction requires
  contiguous state/radius chains, exact iteration pairing, canonical initial
  and final linearizations, and a converged final diagnostic row; failure audits
  require stage-specific payloads and canonical model/path/mask/delta/config
  pairing.
- Added manufactured tests for rectangular Arnoldi orientation, interior and
  boundary projected solves, rejected-radius reuse, fixed-increment convergence
  and full diagnostic replay, failure state retention/forgery rejection,
  required configuration, reversible increments, hard trial linear-algebra
  failures, and structural-exception propagation.
- Removed the redundant `solve_arc_hookstep_newton` compatibility alias so the
  trust-region solver has one canonical public name.

### Verification

- Arc residual/foundation/solver battery: `24/24` under `unittest` after the
  hardening regressions.
- Affected reduced continuation/Newton/trust-region/verified, full trust-region,
  hookstep, and controller battery: `95/95`.
- Monolithic clone/cache regression and immutable-static checks: `2/2` focused;
  full monolithic battery remains to be rerun after the shared CSR hardening.
- Changed modules pass `python -m py_compile`.
- `git diff --check`: pending final diff review.
- Physical execution, continuation, workflow, archive, and verified acceptance:
  not run.

### Exact resume point

Final review should cover the canonical `solve_arc_trust_region_newton` API,
owner-clone identity/equality, hard-vs-radius numerical boundaries, complete
result/failure audit replay, and the cached surface-energy scale. After that
review, any continuation or workflow/physical integration requires separate
authorization; do not run a physical cloud from this module.

## 2026-09-24 — Replace the arc solver preconditioner

### Status

The reviewed bordered map remains available as manufactured diagnostic evidence,
but the arc solver now explicitly uses only the parameter-free
`ArcMetricScaledRightPreconditioner`. No physical solve was run.

### Implemented

- The new immutable map binds the metric-scaled kind, frozen metric and
  linearization identities, the existing reduced lower-triangular right block
  `P`, the pure unit-load metric action, `s_u`, and its inverse scale. Every
  application rechecks those bindings and applies `(r,s) -> (P r,s/s_u)`.
- Augmented Arnoldi retains the `J M` actions and the rectangular
  `metric.apply_columns(M V)` evidence. Controller, correction/diagnostic
  audits, and owner replay require the metric-scaled kind and preserve one
  preconditioner object across rejected radii.
- Removed `denominator_tolerance` from `ArcTrustRegionConfig` and the arc
  solver/controller call path. The old bordered denominator contract is not a
  solver fallback.
- Independent manufactured coverage includes unit pure-load metric norm,
  dense orientation/SVD rank, all-inactive and mixed masks, forged kind/metric
  rejection, rejected-radius object reuse, and bordered diagnostic retention.

### Verification

- Arc residual/foundation/solver suites: `22/22` under `unittest`.
- Affected reduced, trust-region, hookstep, and controller battery: `123/123`.
- Changed arc/test modules pass `python -m py_compile`.
- `git diff --check` passes.
- The prior physical-shape probe at `u_bar=0.00623125` and
  `delta_G=0.00873355605` remains recorded as a structural `c^T P b=0`
  failure in about 7 seconds; no replacement physical probe was executed.

### Exact resume point

Primary Sol review must inspect the metric-scaled kind/identity bindings,
pure-load normalization, retained `J M` and rectangular metric actions, and
the replay checks. Do not apply an epsilon denominator perturbation or run a
physical target before that review.

## 2026-09-24 — Stabilize cancellation-prone physical arc coordinates

### Status

The strict arc hookstep gate remains unchanged. The raw physical Krylov
coordinates are now followed by an immutable, independently checked
metric-equilibrated basis; no physical continuation or probe was run.

### Implemented

- Added `_ArcEquilibratedKrylovBasis`, constructed from the retained
  `source_basis = M V` and `source_metric_basis = D M V` by the thin SVD of
  `D M V`. Its scale-aware rank threshold, condition number, metric
  orthogonality defect, transform, and direct/transformed action defect are
  immutable evidence.
- Re-evaluated `ArcFrozenLinearization.frozen_jvp` directly on each
  equilibrated column. The projected action matrix is formed from those direct
  actions in the retained Arnoldi residual basis; the original `J M` actions
  and Hessenberg relation remain available for provenance.
- Extended the shared projected hookstep core with explicit model overrides,
  while preserving the original monolithic/reduced path when overrides are
  absent. Arc reconstruction now uses equilibrated coordinates, not the
  cancellation-prone raw coefficients.
- Stored the complete hookstep direction in each arc diagnostic row and replay
  it by rebuilding the metric, equilibrated basis/actions, projected solve,
  and direct frozen prediction. Failure/controller audits retain the same
  immutable Arnoldi object and therefore its equilibration evidence.
- Added a manufactured ill-conditioned regression with a `1e8` physical-column
  norm ratio. Raw prediction cancellation exceeds `1e-7`, while direct and
  equilibrated projected predictions agree to `1e-12`.

### Verification

- Arc foundation/solver battery: `24/24` under `unittest`.
- Monolithic residual/hookstep/controller/trust-region battery: `74/74`.
- Final combined reduced/arc/trust-region/hookstep/controller battery: `119/119`;
  final shared monolithic residual/Newton/continuation/verified battery:
  `72/72`. JAX emitted CUDA allocation warnings during the first battery but
  all tests completed successfully.
- Changed modules pass `python -m py_compile`.
- `git diff --check` passes.
- No physical continuation, physical probe, workflow/archive integration, or
  verified attempt was run.

### Exact resume point

Primary Sol review must independently inspect the SVD transform orientation,
scale-aware rank boundary, direct-JVP/action consistency gate, projected RHS
and action rows, and diagnostic replay identity comparisons. Do not loosen
`prediction_consistency_rtol`, add epsilon perturbations, or run a physical
arc solve before that review.

## 2026-09-24 — Refine the recomputed physical metric basis

### Status

The prior read-only physical probe exposed a recomputed metric defect of
`4.8453109785517867e-07` against the unchanged `1e-8` gate at
`u_bar=0.0064136878883856953`. The requested correction is implemented as a
manufactured-only refinement; no physical probe was run after this change.

### Implemented

- Added the explicit fixed maximum
  `_ARC_EQUILIBRATION_REFINEMENT_PASSES = 3`. When the initial recomputed
  `D Q` misses the existing metric gate, each pass forms the symmetrized small
  Gram matrix, computes its symmetric inverse square root, composes the
  coordinate transform, recomputes the final basis from the original source
  span, and recomputes `D Q`.
- The final basis is accepted only when the existing metric tolerance passes.
  Nonpositive/rank-deficient Gram eigenvalues, non-finite values, composition
  defects above the existing metric consistency tolerance, and exhaustion of
  the fixed pass budget are hard failures; no clipping, pseudoinverse,
  epsilon perturbation, or silent fallback is used.
- `_ArcEquilibratedKrylovBasis` now preserves the initial SVD transform, every
  refinement transform, the final composed transform, metric-defect history,
  and scale-aware composition-defect history. Replay compares all of this
  provenance before re-solving the projected direction.
- Added a manufactured regression whose first recomputed metric defect exceeds
  `1e-8` but passes after one recorded polar refinement, while preserving the
  source span and final direct JVP contract.

### Verification

- Arc foundation/solver tests: `25/25` under `unittest` after the refinement
  regression.
- Final combined reduced/arc/trust-region/hookstep/controller battery: `120/120`;
  final shared monolithic residual/Newton/continuation/verified battery:
  `72/72`.
- `python -m py_compile` for changed modules and `git diff --check` pass.
- No physical continuation or probe was run after the refinement change.
- Final replay hardening stores the composition-consistency tolerance and uses
  relative transform/basis recomposition checks; direct-JVP/action failures
  retain the available equilibration transform history.

### Exact resume point

Primary Sol review must inspect the Gram inverse-square-root orientation,
fixed-pass/failure semantics, source-span recomposition, composition-defect
scaling, and replayed transform histories. Do not loosen the `1e-8` gate or
run a physical probe before that review.

## 2026-09-24 — Replace source-recomposed polar refinement

### Status

The physical read-only rerun still failed at Newton iteration 3 after the
three source-recomposed polar passes: initial metric defect
`4.8453109785517867e-7`, final defect `9.0377926031948164e-8`, strict limit
`1e-8`. The source-to-total-transform remultiplication is now removed from the
accepted physical-basis path. No physical probe was run after this change.

### Implemented

- Replaced the polar refinement loop with exactly two explicit modified
  Gram-Schmidt passes in the current physical `D` metric. For each column,
  every orthogonalization and normalization update is followed by a fresh
  `D q` evaluation; the final stored `D Q` is the recomputed physical basis.
- The accumulated coordinate transform and per-pass transforms remain
  mathematical provenance only. Local and source-to-total recomposition are
  audited with an explicit scale-aware `gamma_k`/elementwise-product roundoff
  bound, not bitwise associativity and not the physical `1e-8` gate. The same
  bound is retained for transformed frozen-JVP action evidence when source
  action cancellation exceeds its ordinary consistency tolerance.
- Added a manufactured regression in which the former three source-recomposed
  polar passes plateau above `1e-8`, while physical reorthogonalization passes
  the unchanged metric gate.

### Verification

- Arc foundation/solver battery: `26/26`.
- Final affected reduced/arc/trust-region/hookstep/controller battery: `121/121`.
- Changed arc module and regression compile with `py_compile`.
- `git diff --check` passes.
- No physical continuation or probe was run.

### Exact resume point

Primary Sol review must inspect the MGS coefficient/update ordering, fresh
`D q` calls, the roundoff-bound derivation, and transform/replay provenance.
Do not increase the pass count, loosen the `1e-8` gate, or run the physical
probe before that review.

## 2026-09-24 — Resume bounded read-only arc probe

- Following physical-space MGS, the full first-step read-only probe from the
  authenticated `u_bar=0.0062312500000000345` state and
  `delta_G=0.0087335560467672324` printed only its start line before the
  600-second shell timeout. The command was terminated; no accepted state or
  durable result exists.
- A scratch-only rerun with `max_newton_iterations=4` (all equations, strict
  gates, `delta_G`, and radius controls unchanged) ended after 193.3s at
  `stage=max_newton_iterations`, `iteration=4`. Its last uncommitted trial load
  was `0.0063551067031042798`; scaled block norms were
  `(89078.91261932276, 274.2681520427689, 117800.49673313084)`, energy
  error `0.005303475`, radius `0.01`. This is numerical diagnostic evidence,
  not a converged or physically accepted arc step.
- Exact resume point: inspect all four accepted *inner* correction diagnostics
  and the projected/full model behavior before deciding whether further
  physical iterations are justified. Keep history at its authenticated
  committed value, preserve strict gates, and do not infer a fold crossing from
  uncommitted trial loads. The verified-attempt, controller, and durable replay
  remain to be implemented after a core step converges.
- The reproducible four-iteration trajectory at the initial `delta_G` has
  trial loads `0.0062312500000365903`, `0.0064136878883856953`,
  `0.0063575788116558346`, `0.0063551067031042798` and energy errors
  `0.008733556`, `0.008595017`, `0.005425569`, `0.005303475`. Krylov
  dimensions are `2, 20, 32, 32`, with `0, 1, 0, 0` radius rejections.
  The mechanical residual grows from `0.94` after correction 1 to `89078.91`
  after correction 4; all inner acceptance ratios are positive (`0.79` or
  higher) but the strict augmented root is far away.
- One bounded, scratch-only `0.5` energy cutback
  (`delta_G=0.0043667780233836162`, four Newton corrections, same strict
  gates) also exhausts its budget after 196.0s; final uncommitted load
  `0.0064033184353316768`, energy error `0.003743927`, scaled norms
  `(29382.02970019228, 377.717739834127, 83148.1502566964)`.
  This is not proof that smaller `delta_G` fails at convergence. The next
  concrete action is to inspect the scale and rank of the initial projected
  energy direction and subsequent residual trajectory before a longer run.
- Initial linearization inspection at the original `delta_G` finds two
  metric-normalized projected-Jacobian singular values approximately
  `2.13529977e11` and `2.67360011e-6`. The explicit
  `projected_rank_rtol=1e-12` therefore retains rank one; both metric singular
  values are approximately one. The initial projected step is interior, with
  physical step norm `3.66e-14`, so the huge initial energy residual hardly
  changes. This explains the first micro-correction, but does not prove the
  full augmented Jacobian is singular. The current `LG` deliberately derives
  from the *strict damage correction gate*, as documented in the arc module;
  changing residual weighting requires a separate numerical derivation and
  validation, not a silent decrease of the relative rank threshold into
  roundoff. Resume by analyzing the full-dimensional weighted augmented
  Jacobian and separating nonlinear merit scaling from the unchanged strict
  physical acceptance limits.

## 2026-09-24 — Evaluate existing warning path for the PINN handoff

- User direction: prioritize the original nonlocal fracture PINN; the direct
  solver need not supply a complete strict post-peak reference. Review the best
  existing non-strict/warning route for the overall behavior first.
- The completed, immutable `omega=0.5`, 144-sweep stabilized-warning artifact is
  `/root/autodl-tmp/data/fracture2d/engineering_grip/engineering_grip_damped_warning_o5_u7p5e3_dumin6p25e6_s144.npz`.
  The archive SHA256 matches its manifest:
  `9545bed758c19c07297d1a0afdce9093caab7a646655eef6c6b2a82cb23ac4fc`.
  The replay/analysis hashes also match; no historical artifact was rewritten.
- All 75 nominal increments reach exact `u_bar=0.0075` with zero cutbacks:
  65 are `stabilized_converged`, 10 `stabilized_warning_accepted` (loads
  `0.0060`--`0.0068` and `0.0071`). The prior independent replay authenticates
  the 76 saved states and source/configuration, but explicitly reports
  `validated=false`, `field_equations_validated=false`, and no reconstructed
  rejected-path replay. Damage monotonicity fails; maximum healing is
  `0.029148102228035988`, while damage range excess is zero. The worst
  warning-gate ratios are approximately `2.29e6` for damage correction and
  `2121` for the post-history damage residual.
- The saved field and stress-strain figures show a crack growing rightward from
  the notch tip in both nonlocal and FEM results. The nonlocal response peaks
  around `u_bar=0.0061`, then drops to reaction `69.1542612196795` at `0.0075`;
  the local FEM warning comparator has reaction `3.977181129987526` there.
  Nonlocal damage maximum is `0.9999629621798838` near the notch tip and
  interface damage maximum stays `0.09923934584217217`. Nevertheless the
  endpoint nonlocal vertical displacement reaches `0.141990164738418` versus
  prescribed load `0.0075`, with local tip history `2.167992e6`; the warning
  segment is not a trustworthy quantitative fracture solution.
- Exact handoff: retain the existing diagnostic curve/plots as qualitative
  evidence, use strict pre-growth states and the existing operator tests as
  nonlocal checks, and use the local FEM solution as a separate continuum
  comparator. Do not add a new expensive direct warning run to replicate the
  same evidence. The next implementation effort should inspect and develop
  the nonlocal PINN, with acceptance evaluated by its own nonlocal residuals,
  history/irreversibility, range, and reaction/energy checks.

## 2026-09-24 — Paused after staggered adaptive-step task switch

- The user explicitly switched to
  [`fracture2d-stagger-adaptive-step.md`](fracture2d-stagger-adaptive-step.md)
  and paused coupled/monolithic solver work. Preserve this task's unfinished
  arc/globalization resume point and all unchanged strict/warning artifacts.
- The prior PINN handoff is also deferred by the new active priority; no
  physical continuation or coupled-solver changes belong to this switch.

## 2026-09-27 — Archived: engineering-grip baseline closed with the adopted solution

- Closed. This task produced the engineering-grip model, the strict/warning
  controllers, the stabilized controller family, and the durable workflow; its
  continuation search ended with the adopted stabilized-warning solution
  (`0 -> 0.0075`, peak `641.340` at `0.0061`, terminal `69.154`, `10` warning
  states) now documented in
  [`examples/fracture2d/README.md`](../../examples/fracture2d/README.md).
- The monolithic family and the arc-length/reduced research layers were
  untracked and gitignored on 2026-09-27; they remain on disk for local
  reference only and are not part of the tracked repository.
- Superseded by: `examples/fracture2d/README.md`,
  [`fracture2d-nonlocal-pinn.md`](../fracture2d-nonlocal-pinn.md),
  `docs/PROGRESS.md`, `docs/DECISIONS.md`.
- No further work is planned on this task.
