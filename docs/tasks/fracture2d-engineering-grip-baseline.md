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

Active: the authorized fresh-zero half-increment sensitivity strictly reached
exact `u_bar=0.0059875`. Anderson implementation is deferred by user decision;
the next phase is an isolated damped stabilized-warning policy and full
`u_bar=0.0075` diagnostic. All stabilized paths remain experimental and
nonproduction.

## Blocker

The half-increment endpoint passes only on sweep `143/144`, so fixed damped
Picard has effectively no remaining iteration margin for progression toward
`u_bar=0.0063`. The v5 archive independently validates accepted physical states
but intentionally cannot reconstruct rejected trial vectors.

## Next Action

Keep the completed and prior failure archives unchanged. Add a separate warning
policy around the existing fixed `omega=0.5` sweeps: convergence-gate failures
after the full budget may be warning-committed only from a complete finite
candidate with passing mechanics/CG and structural hard gates. Preserve raw
fields and truthful failed-gate summaries; independently test the policy before
the authorized full diagnostic.

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
