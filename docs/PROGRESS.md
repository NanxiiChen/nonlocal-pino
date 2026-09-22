# Project Progress

> Dashboard, not changelog. Keep this short — detail lives in task files / git history.

## Current focus

- The lean monolithic archive/replay layer is implemented at schema
  `fracture2d-monolithic-continuation-v1`: it stores accepted states and
  scalar/digest provenance only, names the canonical energy
  `accepted_mechanics_energy`, and uses precise `attempt_scalar_evidence`
  metadata. The writer/reader are explicitly ZIP32-only (`allowZip64=False`),
  reject ZIP64 sentinels/extras, enforce the practical `500000000`-byte cap,
  and use fail-closed Linux hard-link installation with identity-checked
  fsync-failure cleanup. Replay takes one bounded immutable byte snapshot from
  the opened descriptor before inspection/loading, invokes top-level owner
  validation (including the
  controller-only no-progress branch), and checks owner-recomputed `psi_after`,
  `H_next`, static operators, and canonical equations/gates without persisting
  redundant arrays. The raw ZIP32 EOCD parser now requires one zero-comment
  single-disk record ending at EOF, exact allowlist counts, bounded central
  size/offset, and exact parsed-record/location agreement; a structurally
  positioned ZIP64 locator and duplicate/trailing EOCD records are rejected
  without scanning payload bytes for signatures. Focused archive
  tests pass `17/17`; the full monolithic CPU command passes `82/82`; workflow
  tests pass `52/52`; legacy engineering-grip/JAX tests pass `63` with one
  expected GPU-only skip; stabilized
  tests pass `82/82`; compileall and `git diff --check` pass. The approved
  workflow now exposes an isolated
  strict `monolithic` strategy: explicit Newton controls are parsed, the CPU
  owner runs without FEM/GPU construction, successful or terminal numerical
  results are archived and immediately replayed with the same owner/budget,
  and scalar summary/manifest artifacts bind the source/configuration/digests.
  Structural exceptions, including Newton type/shape/API failures, do not
  create numerical archives. The workflow now
  rejects noncanonical `max_stagger_iterations` values because that shared
  field is only the documented solver_base compatibility value, stages the
  complete monolithic bundle privately, publishes it with Linux
  `renameat2(RENAME_NOREPLACE)`, and cleans only private staging after
  structural failure. A published bundle is retained when parent fsync
  durability is uncertain, and that durability error is not classified as a
  numerical continuation failure. The manifest explicitly excludes its own
  recursive hash while binding archive/replay/summary fingerprints. It reports
  `field_equations_validated` only for
  retained accepted-step canonical fields owner-validated independently of
  completion. The integration is still
  experimental/nonproduction. CPU profiling replaced the large temporary A2
  tensor actions with one read-only sparse gradient/divergence operator, caches
  frozen monolithic JVP kinematics, vectorizes direct-solve block Jacobi, and
  uses an exact sparse free-mechanics block in the monolithic preconditioner.
  Primary Sol review found and resolved the archive snapshot, structural-error,
  ZIP signature, ignored-option, durability-classification, and manifest-scope
  defects. The first authorized CPU `0 -> 1e-4` physical-cloud smoke was
  user-aborted before any artifact, then bounded profiling and optimization
  reduced the zero-damage work check from `207.35s` to `17.09s` and the initial
  GMRES from `94.3s/1927` callbacks to `1.03s/16`. The optimized strict rerun
  completed in `275.51s` with one accepted step, no cutbacks, immediate
  owner-authenticated replay, and validated field equations. Its final reaction
  is `12.6749467637548`, damage maximum is `1.2254719713851597e-4`, result digest
  begins `ef5f3a49`, and bundle run ID is
  `engineering_grip_monolithic_cpu_u1e4_optimized_20260922`.
- The subsequently authorized fresh-zero strict CPU monolithic run toward
  `u_bar=0.0075` terminated after `14779.78s` at the configured minimum
  increment. It committed 61 steps through exact `u_bar=0.0061`, then rejected
  targets `0.0062` and `0.00615`; the latter exhausted all Armijo trials at
  Newton iteration 8. The final committed reaction is `641.3384929751363`,
  damage maximum `0.8202621432308079`, and history maximum
  `432.1558022505133`. The four-member failure bundle is owner authenticated,
  path replayed, and validates retained-state field equations; completion and
  top-level replay `validated` are correctly false. Result digest begins
  `d5dd1412`. Reaching `0.0075` remains blocked under the confirmed controls.
  Read-only comparison locates both methods' difficulties near the strongly
  damaged peak/post-peak region, but stagger previously reached `0.00623125`;
  there is no common proven failure load. Monolithic shrinking line-search
  alphas accompany growing raw Newton directions despite accurate linear
  solves. The completed diagnosis reproduces the exact failure without writing
  state: smooth
  JVP blocks pass finite differences at about `1e-8`, Newton's merit derivative
  is correct, and the fixed-active feedback eigenvalue rises from `0.6677` at
  `0.0055` to `0.8922` at `0.0061`, then `0.995568` at failure. The dominant
  failure mode is almost entirely notch-tip particle `1869`. Armijo first passes
  at `alpha=2^-14`, two reductions beyond the configured budget, but this does
  not establish eventual convergence. The subsequently authorized exact-target
  in-memory replay with a 14-backtrack budget accepted that iteration-8 trial,
  then failed immediately at iteration 9 after all trials through `2^-14`.
  GMRES still passed and the committed state/artifacts were unchanged. Deeper
  Armijo alone is therefore closed as a remedy; no new continuation was run.
- The authorized isolated manufactured hookstep core is complete and remains
  undispatched. It implements the owner-paired normalized `q/d/H` metric,
  right-preconditioned Arnoldi with complete terminal-vector evidence,
  SVD-whitened interior/boundary projected solves, cancellation-resistant and
  independently checked predictions, immutable numerical audits, and pure
  radius decisions. Focused tests pass `18/18`; combined hookstep,
  residual/JVP, and existing Newton tests pass `56/56`; compilation and
  `git diff --check` pass. Independent numerical re-review found no remaining
  finding and code re-review found no blocker. No physical controls were
  selected and no solver/workflow/archive integration or physical run occurred.
- Completed the approved numerical hardening: raw Newton line-search trials
  accept any finite `d/H` without clipping or projection, successful Newton and
  canonical Phase-2B states explicitly require nonnegative history, and the
  Phase-2C result audit now binds exact work-check settings, reconstructs the
  target/increment/cutback schedule, enforces cutback limits, and pairs every
  attempt/state. The actual direct-predictor controller reaches exact
  `u_bar=1e-4`. The work-check contract now binds validated model/static/value
  provenance without requiring cross-solve damage-cache runtime identity, and
  Newton diagnostics retain algebraically paired residual vectors plus an
  explicit owner-backed replay method. CPU verification passes `233` affected
  tests with one expected GPU-only skip; compilation and `git diff --check`
  pass. The numerical layers remain manufactured-only; the workflow/archive
  seam is wired but no physical run was performed. Primary Sol review remains
  required.
- Implemented the authorized manufactured-only Phase-2B single-target
  `run_verified_monolithic_attempt(...)` layer. It preserves the raw Newton
  candidate, runs the authoritative mechanics/Amor/fixed-history damage/
  mechanics/Amor sweep with committed history, constructs the canonical
  `(q_after_free, d_solved, H_next)` state, and returns immutable numerical
  failure/rejection audits with unchanged committed digests. Auxiliary and
  canonical payloads now enforce exhaustive identity, optional-field, array,
  digest, and gate pairing; Newton rejection retains the exact same failure
  audit object; and `rho_used`/`rho_next` numerical failures retain distinct
  stages. The focused Phase-2B tests pass `10/10`; the combined affected
  command passes `261` tests with `2` expected skips. The layer remains
  private/manufactured-only and requires primary Sol review; the workflow seam
  is now wired, but no production/physical continuation run was performed.
- Implemented and review-corrected the authorized Phase-2A private scaled
  semismooth Newton/GMRES core around `MonolithicGripResidual`: required
  solver/gate parameters are explicit, `H_used/H_next` reproduce the existing
  six-term damage/history scale semantics, exact-zero-only shortcuts are used,
  diagnostics pair linearization and accepted-trial scales, and immutable
  partial failure audits retain current-step evidence. The focused Phase-1
  plus Phase-2A manufactured tests pass `35/35`; no workflow/CLI,
  continuation, auxiliary sweep, or physical run was added or performed.
- Resolved all four independent review findings on the Phase-1 monolithic
  engineering-grip contract without adding any solver: cached CSR internals
  are bytes-backed and irrevocably read-only (including failed re-enable
  attempts), derived masks are frozen read-only with loud mutation failures; a
  genuinely independent test-only packed-JVP oracle (rebuilt BA assemblies,
  from-definition cubic-kernel damage graph, independent Amor accumulation)
  now carries the correctness evidence over multiple states/directions and
  ties the production adjoint through the bilinear identity; active-set
  switching and exact-tie committed behavior are manufactured-tested along a
  crossing line with a 1-Lipschitz no-jump check; out-of-range `d`/`H`
  inputs are proven to evaluate the raw equations with no obstacle; and the
  dense Jacobian helper is demoted to private production self-data. Focused
  Phase-1 monolithic tests pass `23/23`; the Phase-2A solver evidence is
  tracked in the active task below.
- Implemented and manufactured-verified Phase 1 of the user-selected monolithic
  engineering-grip formulation: the exact code-compatible packed `q-d-H`
  residual and analytic generalized JVP/adjoint in the private/experimental
  `examples/fracture2d/nonlocal_engineering_grip_monolithic.py`. The
  formulation decision is recorded in `docs/DECISIONS.md` (2026-09-20); the
  Phase-2A core remains private/manufactured-only and is exercised only through
  the isolated workflow seam.
  Primary Phase-1 review findings are resolved (see the bullet above).
- Completed implementation and non-physical verification of the approved
  bounded-study `anderson_residual_regularized_backtracking` policy with
  explicit fixed `anderson_regularization_mu`, augmented-QR coefficients,
  strict provenance, final diagnostic names, and controller-parity coverage.
- Completed the scalar-only negative fixed-`mu` sensitivity study from the
  injected accepted `0.0062312500000000345` state to exact target
  `0.006234375000000035` for `mu={0.25,1,2}`. All three attempts ran all
  `144` sweeps and rejected the same gates
  (`displacement`, `damage`, `history`, `reaction`, `damage_residual`,
  `damage_fixed_point`, `coupled`). Selected AA/Picard-fallback counts were
  `3/141`, `0/144`, and `0/144`; final fixed-point errors were
  `1.8632048281652658e-4`, `1.8608049513912928e-4`, and
  `1.8608033904232713e-4`, versus baseline
  `1.8391723755628586e-4`. Maximum coefficient norms were
  `0.7697/0.3656/0.2151`.
- The study is behavioral diagnostic evidence only: regularization suppresses
  AA toward Picard and worsens the endpoint, so the `mu` grid must not be
  expanded or adapted. Source-bundle hashes differ by design because the new
  policy code changed; source cloud, configuration, and rollback bindings
  match, so comparison is behavioral rather than bitwise code parity.
- The user-approved smaller-`du_min` checkpoint used the last accepted
  `0.00623125` state (digest `a5e0...`), the same unregularized
  `anderson_residual_backtracking` policy, target `0.0062328125`,
  `du_min=1.5625e-6`, and `144` sweeps. It strictly failed and rolled back.
  Artifact:
  `/root/autodl-tmp/data/fracture2d/engineering_grip/engineering_grip_anderson_residual_backtracking_half_step_u6p2328125_20260920T094632Z.json`
  (file SHA256 `28b3e0a751781b5f043ea2c2a2ed43852388a2b783e3d23c19bd9fb702cd0ba3`,
  payload SHA256 `e6b6434f6558afbc2143d58a3d7337ef04ffffd99eccc2b0d91f8845d3067159`).
  Final error was `4.779188545345203e-5`; tail ratio/last-ten maximum were
  `1.0010643`/`1.0012019`, with the same coupled gates failing.
- The `alpha=1/16` audit found all last 30 trials hard-pass/map-complete, but
  only sweep `125` passed globalization; `29/30` slopes
  `(m(alpha)-m_ref)/alpha` were positive and tail slopes were large. `alpha=1/8`
  was always worse, so a blanket deeper-alpha extension is NO-GO. The current
  controller path is exhausted; do not guess further damping/regularization.
- The prior same-target `0.0061125 -> 0.00611875` backtracking checkpoint
  failed after 144 sweeps; the half-step `0.0061125 -> 0.006115625` with
  `du_min=3.125e-6` accepted in 30 sweeps with all gates and tail contraction.
- Fresh-zero strict production reached exact `u_bar=0.0062` with 62 accepted
  steps and zero cutbacks. Accepted reaction is `641.3384921076807` at `0.0061`
  and `631.4074774212969` at `0.0062`; archive/scalar hashes begin
  `95851472`/`14765f`. Replay and field-equation checks pass, with a separate
  `k_res=0` conditioning study recording an SPD crack-localized mode caveat.
- A fresh-zero strict `u_bar=0.0075` attempt rolled back at the `du_min` target
  `0.006234375`; the last accepted state is `0.00623125` after 64 accepted
  steps and 70 attempts. Its failure archive begins `a1743d`; rejected state
  fields are not evidence.
- The current fixed-`mu` study supplies only a bounded behavioral comparison;
  it does not select a production regularization rule or convergence claim.
  Do not expand/adapt the `mu` grid, guess `k_res`, or add an arc-length rule.
- Final non-physical verification is `118/118` expanded stabilized/workflow
  tests and `81/81` affected legacy engineering/JAX/baseline tests; both
  passed. `compileall` and `git diff --check` also pass. Expected argparse
  usage/error output is confined to negative CLI tests. This documentation
  update changes no solver/tests and runs no fresh-zero production solve.
- Active task: [`docs/tasks/fracture2d-engineering-grip-baseline.md`](tasks/fracture2d-engineering-grip-baseline.md).
  Prior fixed-`mu` implementation task archived at
  [`docs/tasks/archive/anderson-residual-regularized-study.md`](tasks/archive/anderson-residual-regularized-study.md).

## Current state

- `fracture1d` is an accepted engineering baseline; default PINN dtype is
  float32 with `spatial_emb_dim=32` and `spatial_scale=16`.
- The active direction is a pure-nonlocal 2D geometric-notch phase-field
  fracture example on the unit square under top displacement loading.
- Plane strain, bottom-edge vertical support with a lower-left horizontal point
  anchor, the paper-aligned hybrid/history formulation,
  `w_notch=2*l0`, a semicircular tip with `r_tip=l0`, and one common mechanics/
  damage horizon are selected. The foremost semicircular tip is `(0.5,0.5)`,
  so its center is `(0.5-l0,0.5)`.
- Implementation order is local FEM reference, direct nonlocal solver, then
  nonlocal PINN.
- The local FEM point-anchor warning-mode archive is the accepted working
  reference at `u_bar=0.0075`; its strict convergence and mesh-refinement
  limitations remain explicit follow-ups.
- The repository-local and external primary-reference reviews found no single
  source for the planned Amor-history AT2/correspondence coupling. Silling
  supplies the intact variational correspondence basis, and bond-associated
  papers supply stable alternatives, but their damage-aware combination still
  requires an independently checked discrete derivation.
- The full Bie (2023) PD-PFM paper is now reviewed. It supplies modified
  energy-compensated correspondence, nonlocal phase flow, dual-horizon
  residual/tangent assembly, and an implicit BFGS route, but uses cohesive
  strength/softening physics and plane stress. Its printed equations do not give
  a separate binary bond-deletion rule and contain phase-flow ambiguities.
- Sinha et al. (2026) SGE is a genuine finite-range nonlocal phase-field model:
  it inserts moment-fit neighborhood gradients into AT1/AT2 energy. It is not
  peridynamics, uses split mechanics and penalty irreversibility, and does not
  supply the adjoint-consistent direct equations needed here.
- The selected direct-baseline route is stable energy-consistent correspondence
  mechanics plus weighted `SparseD` nonlocal AT2 damage, Amor maximum history,
  history-field irreversibility, and staggered linear subproblem solves. The
  later PINN will reuse the same strong-form nonlocal residuals.
- The optional local-FEM monolithic/quasi-Newton review is archived and deferred;
  the selected direct-baseline model route is unchanged.
- The BA target-endpoint omission is corrected with a target-inclusive support;
  the float64 regular-cloud check now uses an independent explicit mechanics
  residual and passes rank/conditioning, target sensitivity, affine/rigid modes,
  energy-gradient FD, residual-based Hessian JVP/symmetry, and the
  three-mode/first-nonrigid spectrum gate.
- The Stage 0 boundary BVP is frozen as project-owned material-only
  boundary-particle hard constraints with direct DOF elimination/affine lifting;
  exterior fictitious collars and interior material grips remain rejected
  historical alternatives.
- Sol reviewed and approved the corrected Stage 0 equations and hard-constraint
  contract to proceed to Stage 1; the rounded-slot geometry/quadrature cloud is
  now implemented and its finite-cloud metrics are recorded in the active task.
- Stage 1 production visibility is mesh-conforming: the tagged slot-face and
  notch-tip line-cell chain plus the unmeshed opening closure define the polygon,
  with strict interior filtering, a declared scale-aware machine floor, and
  explicit boundary-coverage validation. The analytic-circle helper is no
  longer part of the production API; the public cloud field records input
  tolerance while the predicate uses its effective floor.
- Stage 1 finite-cloud BA gates pass at diagnostic horizons `0.032`, `0.040`,
  and `0.048`: no shape-rank failures occurred, affine reproduction stayed near
  float64 precision, and the constrained `delta=0.048` spectrum converged with
  a positive minimum. No production horizon has been selected.
- Stage 2 `ExplicitVolumeSparseD` is implemented with explicit target volumes,
  matching coordinate/volume dtype validation, deterministic edge normalization,
  fixed-shape JAX matvecs, and SciPy export. The fracture2d direct baseline is
  frozen to float64 geometry/operator data with JAX x64 enabled; strict-horizon
  graphs are dtype-specific and must be rebuilt for any future float32 PINN.
  Focused graph, dtype, SPD, and full-square triangular/barycentric checks pass.
  Rounded-slot values are independent two-environment review evidence, not
  persistent unittest coverage; no local-consistency gate or production horizon
  is selected. Sol approved Stage 2 after the dtype and test-isolation review
  fixes.
- Stage 3 intact bond-associated mechanics is implemented independently in
  NumPy/SciPy float64. Target-inclusive supports, explicit energy-gradient
  residuals, independent tangent actions, eta-averaged fields, immutable
  boundary-particle constraints, matrix-free CG, reactions, and conservation/
  patch/work diagnostics pass on the persistent manufactured cloud. Fresh
  rounded-slot evidence at `delta=0.032`, `0.040`, and `0.048` has no rank
  failures; its tangent action independently matches the prior matrix-free
  action. Sol approved Stage 3 after strict-CG, independent reaction-work,
  block-Jacobi, and nonconvergence review fixes. These horizons remain
  diagnostics only.
- Stage 4 fixed-history damage, immutable shared-data fixed-damage mechanics,
  and standalone Amor history APIs pass independent manufactured-cloud and
  fresh rounded-slot review. At diagnostic horizons `0.032`, `0.040`, and
  `0.048`, fixed-`d=0` displacement/reaction/mean-stress trends agree with the
  warning-mode FEM trend, fixed-history damage stays in `[0,1]`, and the
  `delta=0.048` fixed-damage reaction-work error is `9.24e-13`. Sol approved
  progression to Stage 5; these horizons remain diagnostics only.
- Stage 5 strict continuation is implemented around the approved fixed
  subproblems. It archives an immutable zero/accepted state sequence, uses
  trial-only committed-history maxima, fresh fixed-damage views, strict
  residual/range/healing/conservation gates, rollback, cutback, and SHA256
  attempt audits. Every result now carries validated immutable model metadata;
  archive writing requires explicit supplemental mesh/provenance metadata.
  Independent dense BA/damage/Amor reference, constrained-tangent,
  force/moment-gate, forged-step, and archive tests pass on the manufactured
  cloud. The 2026-08-22 independent review findings are resolved: result replay
  reconstructs mechanics from frozen metadata, recomputes Amor and fixed-history
  fields/energies/residuals, validates exact attempt/state pairing and immutable
  nested data, and preserves structural exception propagation. Sol approved the
  Stage 5 controller implementation after all `134` explicit Stage 0--5 CPU
  tests passed. No rounded-slot pilot, production horizon, or physical
  continuation approval is claimed.
- The authorized temporary `delta=0.048` rounded-slot pilot reached
  `u_bar=1e-3` in ten accepted increments with no rejection or cutback. All
  strict gates passed; the largest error/limit ratios were `0.7791` for the
  committed damage residual and `0.7442` for history. Final damage was
  `[2.43e-7, 1.2556e-2]`, and Stage 4/FEM trends remained consistent. Independent
  review approved this pre-localization evidence only. The `6614 s` CPU runtime
  exposes a practical blocker before a much longer peak/crack-growth run.
- Safe CPU warm starts, vectorized block-Jacobi, and the restricted float64 JAX
  Stage 5 adapter are independently validated. The full CPU-replayed GPU pilot
  reached `u_bar=1e-3` in `430 s` after one strict rejection/cutback, versus
  `6614 s` for the prior CPU run. All accepted gates passed; the closest ratios
  were `0.9814` for damage residual and `0.8308` for history. The 154-field
  archive records `jax_gpu_cpu_replay`, loads without pickle, and is approved as
  pre-localization evidence only.
- The separate all-gate warning mode is independently approved. Its first fresh
  `delta=0.048` checkpoint reached `u_bar=2e-3` in 20 nominal increments; every
  step actually passed all strict gates, so no warning bypass or rollback was
  exercised. Final reaction was `253.5305191`, maximum damage was `0.0513580`,
  and reaction remained monotone. The independently audited 164-field archive
  is pre-peak evidence only; no farther endpoint or production horizon is set.
- The next fresh `delta=0.048` checkpoint reached exact `u_bar=3e-3` in 30
  nominal increments after adding an 8-ulp endpoint snap that prevents a
  floating-point remainder from becoming a duplicate solve. Two states were
  warning-accepted for damage/history convergence gates; all fields remained
  finite and monotone, reaction increased to `371.4762173`, and maximum damage
  reached `0.1204572` at the notch tip. Warning trigger loads vary with the
  GPU/stagger path and are not claimed reproducible. This remains pre-peak,
  nonproduction evidence.
- A full raw path reached `u_bar=0.0075` and shows a reaction peak near `0.0055`,
  but it is not approved: final maximum damage is on the loaded top boundary,
  38/51 top particles have `d>0.5`, notch-tip damage is only about `0.587`,
  healing occurs at `0.0056`--`0.0058`, and the final damage residual infinity
  norm is about `0.0608`. The raw checkpoint is explicitly `validated=False`;
  strict/warning runner refactoring is held pending physical/operator review.
- Frozen transfer at `0.0056` shows that the first mechanics-before solve has
  top `H_used≈78.9`, and the exact fixed damage solve gives top `d_max≈0.414`;
  mapped FEM history in the same operator gives top `d_max≈0.081`, matching FEM.
  The raw warning state instead has top history `≈3875`, top damage `≈0.971`,
  and residual infinity norm `≈143`. The top runaway therefore develops across
  underconverged damage-degraded mechanics sweeps, rather than being created by
  one hard-BC solve or the fixed damage operator alone.
- The exact 20-sweep replay reproduces the raw endpoint. Top strain/history
  overtakes the tip at sweeps 8--9 and top damage at sweep 10. From sweep 10 to
  20, top `H_used` grows `20.4x` while tip `H_used` grows only `1.05x`; the
  `H_next` damage residual grows to `142.64`. Every fixed-`H_used` damage solve
  is accurate to about `4e-12`, so the failure is the staggered top-boundary
  feedback, not the linear damage solve.
- The boundary-source audit found no drop-in surface/collar correction for the
  current project-owned BA energy. The closest 2D correspondence method uses a
  Behzadinasab-style fictitious layer with separate kinematic and stress
  neighborhoods, but it is a different RK/GMLS strong-form operator requiring
  independent residual, energy, tangent, and reaction validation.
- Cancellation-aware replay now bounds redundant AT2 residual reconstruction by
  `512*eps64` times the sum of constituent-term magnitudes. It accepts the
  observed `1.421e-13` ordering defect while retaining strict rejection of
  meaningful changes; focused continuation tests pass.
- Part I/II do not specify the current same-node mixed displacement/natural BC,
  point anchor, fictitious-node phase-field degradation, conservative energy/
  reaction, or rounded-slot collar quadrature. Implementing the proposed
  counterfactual would therefore require unapproved mathematics.
- A separate project-owned Candidate A2 derivation now preserves the material
  cloud exactly, adds phase-separated material/grip center measures with union
  BA kinematics, applies outer-face component constraints, and compares at
  matched physical interface opening. Independent review approves only an
  isolated fixed-`d/H`, NumPy/SciPy `delta=0.048` prototype with explicit
  geometry, algebra, spectrum, work, and conservation gates.
- Candidate A2 is implemented in a separate diagnostic module with 11 focused
  contract tests. Manufactured gates and the actual `h_g=0.02` and `h_g=0.01`
  union-cloud hard gates pass, including independent graph/reference, spectrum,
  energy/JVP, symmetry, reaction/work, and conservation checks. Its authorized
  frozen transition remains nonproduction: the reaction guard passes, but the
  material-top `H_used` error reduction is only about 3.8%, below the required
  50% diagnostic-usefulness threshold. Primary review confirms that this failure
   is controlled by inherited top history `76.1040`; instantaneous top
   `Psi_plus` falls from material-only `78.8966` to `5.6326--6.2479`.
- The strict coupled continuation contract now has primitive-independent replay
  coverage for an actual off-target opening-gate rejection: production and
  archive replay compare retained mechanics primitives against the attempted
  opening, preserve rollback identity, and reject forged failure/pass
  classifications. The implementation remains nonproduction and no pilot is
  authorized.
- The first strict A2 coupled pilot accepted `opening=5e-5` but stopped at
  `1e-4` because its primary `1e-7` CG solve could not satisfy the independent
  `1e-7` work gate. Reconstruction found no energy/reaction defect; primary CG
  residual controlled the mismatch. Independent review authorizes one exact-zero
  rerun to `1e-4` with primary CG tightened to `1e-10/1e-13`; all gates and the
  endpoint remain unchanged.
- Persistent adapter and diagnostic utility names now describe their runtime
  semantics rather than roadmap stages or temporary pilot status: CPU-replayed
  JAX, bond-associated algebra, fixed-history diagnostics, transition
  diagnostics, and pseudo-time diagnostics.
- The superseded v1 engineering-grip smoke archive reached exact
  `u_bar=1e-4` with one strict accepted step, but its independent replay stopped
  at the exact initial zero state because loop-order BA identity roundoff
  produced `1.0514815413942755e-11` against the `1e-13` replay limit. The v2
  replay contract and fresh smoke evidence below supersede that artifact.
- The initial scalable replay audit found work-check, graph/provenance,
  metadata, and production-CG-status gaps while confirming the BA/tangent,
  block-Jacobi, sparse AT2, symmetry, and constrained-spectrum algebra. Those
  replay-contract gaps are now covered by the current v2 implementation and
  forgery tests; no longer continuation is authorized.
- The engineering-grip replay audit repairs are implemented without changing
  the formulation or production gates. Derived top-collar coordinate checks
  use a bounded float64 roundoff rule while serialized primitives stay exact;
  direct work-config, graph, metadata, CG-status, constituent-scale,
  scheduling, and incomplete-stage forgery tests are covered. The focused
  engineering suite passes `25` tests and primary review plus the no-further
  continuation hold remain.
- The exact authorized engineering-grip v2 smoke was rerun on `cuda:0` with the
  restricted CPU-replayed JAX adapter: one accepted step, three complete
  sweeps, exact `u_bar=1e-4`, and an independently passing v2 archive replay.
  Production CG status is archived as zero for both mechanics solves; this is
  still nonproduction evidence and does not authorize a longer continuation.
- The JAX engineering-grip test now lazily selects its CPU fallback and uses an
  explicit eight-unit-roundoff float64 comparison for JIT-versus-NumPy phase
  center weights. Engineering CPU `25/25`, related CPU `61/61`, JAX CPU
  `20/20` (two GPU-only skips), and JAX CUDA `12/12` checks pass.
- The authorized fresh-zero engineering-grip path reached exact `u_bar=3e-3`
  in 30 strict accepted steps with no cutbacks, and its v2 archive independently
  replayed in `890.63 s`. Reaction remains monotone, final maximum damage
  `0.1186674` is at the exact notch tip, and every crack-localization flag passes.
  The maximum stagger count rose to 9 and the path is gradually softening; this
  remains a nonproduction pre-peak checkpoint with no farther endpoint approved.
- Matched `3e-3` material-field plots show similar notch-localized `d`, `u_x`,
  and `u_y` patterns between engineering grips and the accepted point-anchor FEM
  warning reference. Using the existing nominal definitions, nonlocal stress is
  `366.56` versus FEM `383.90`, a `4.52%` endpoint difference.
- The authorized full engineering-grip run strictly stops at target `0.0049`
  after committing `0.00485`; two fresh-zero runs reproduce the stop. The
  retained failure audit shows notch-localized fields and geometric convergence,
  with the final history gate ratio falling to `1.60` at sweep 20. Completion is
  held pending explicit approval to raise only the stagger budget from 20 to 30.
- The authorized 30-sweep rerun passes `0.0049` but strictly stops at `0.0054`
  after committing `0.00535`. Damage/history/residual errors remain contracting,
  the tip remains the damage maximum, and reaction is still monotone. Completion
  now requires an explicit decision on a 60-sweep ceiling; no gate relaxation is
  proposed.
- The explicitly authorized fresh-zero 60-sweep rerun reaches 59 accepted
  nonzero steps (last committed `u_bar=0.00585`) and stops strictly at target
  `u_bar=0.0059` after both nominal and minimum-increment attempts. The failed
  gates are damage, history, and damage residual after all 60 sweeps. Its
  durable `validated=False` audit, summary, analysis, plots, and manifest are
  under `/root/autodl-tmp/data/fracture2d/engineering_grip/`; committed damage,
  history, and reaction remain monotone and notch-localized. No failed state was
  warning-accepted. The final attempt remains geometrically contracting with a
  tail factor about `0.809`; final damage/history/residual gate ratios are
  `1.748/6.974/2.101`, while displacement and reaction already pass.
- The authorized fresh-zero 72-sweep rerun accepts the former `0.0059` blocker
  in 68 sweeps, then stops strictly at `u_bar=0.00595` after all 72 sweeps. Only
  history remains failed, at `2.917` times its limit; damage, post-history
  residual, displacement, and reaction pass. The tail still contracts near
  `0.828`, but convergence has slowed and the required sweeps per accepted step
  are increasing. The last committed fields remain monotone and tip-localized.
- The selected smaller-increment study supports one bounded fresh-zero test to
  `u_bar=0.00595` with only `du_min` halved to `2.5e-5`; keep the 72-sweep budget
  and every gate fixed. A roundoff-safe minimum-increment check now prevents the
  duplicate minimum-target retry observed in s72; focused tests pass `27/27`.
- A separately audited warning workflow preserves truthful strict gates, allows
  finite intermediate range/healing recovery, and warning-commits only final
  complete sweeps whose raw range/healing excesses are at most `1e-3`. Warning
  archives use a separate schema and remain `validated=False`.
- The fresh-zero warning path reaches `u_bar=0.0067` with 59 strict and 8 warning
  accepted steps, exposing a reaction peak `641.34` at `0.0061` and a centerline
  crack. It hard-stops at `0.00675` because final healing `0.00650` exceeds the
  authorized `1e-3` limit. The last reaction is `71.61`; no interface takeover
  occurs, but damage is no longer monotone.
- The isolated stabilized controller now has focused manufactured backtracking,
  rollback, lean v5 archive, storage-preflight, and independent-replay coverage.
  Stabilized archive tests pass `7/7`, the workflow suite passes `20/20`, and
  the combined stabilized/legacy/JAX CPU verification passes `94` tests with
  two GPU-only skips. A fresh-zero `u_bar=1e-4` manufactured-cloud archive
  writes and replays successfully. The controller remains experimental, and
  scalar logs/digests cannot independently reconstruct non-final complete
  backtracking trial vectors, so exact rejected-path replay is not claimed.
- The fresh-zero cached fixed-damping run with `omega=0.5` and 144 sweeps
  accepted 63 steps through `u_bar=0.005975`, then stopped strictly at the
  minimum-increment target `0.0059875`. Only history and post-history damage
  residual failed, at `3.2524` and `1.2618` times their limits. The 64 accepted
  states pass the independent field-equation audit, remain notch-localized, and
  limit material displacement amplification to about `1.060`; rejected-path
  replay remains unavailable under the lean v5 contract.
- Halving only `du_min` to `6.25e-6` lets the fresh-zero fixed-damping path
  strictly reach exact `u_bar=0.0059875` in 65 accepted steps. The endpoint
  passes on sweep `143/144`, with fixed-point/history/post-history-residual
  ratios `0.9472/0.6338/0.7842`; accepted fields remain tip-localized and the
  displacement amplification is `1.05996`. Independent accepted-field replay
  passes. This confirms load-step sensitivity but leaves too little iteration
  margin for a longer fixed-Picard path.
- The separate fresh-zero damped stabilized-warning path completed 75 nominal
  increments through exact `u_bar=0.0075`: 65 steps were strict and 10 were
  warning-accepted. Warnings occur at `0.0060`--`0.0068` and `0.0071`; the
  response peaks at `641.34` near `0.0061` and collapses to `69.15` at the
  endpoint. Crack-tip history grows to `2.168e6`, displacement amplification
  reaches `18.95`, and maximum healing is `0.02915`. The maximum remains at the
  notch tip and interface damage stays below `0.10`, identifying a crack-tip
  feedback runaway rather than top-interface takeover. The warning-v1 archive
  is `validated=false`; accepted-state replay is bound and complete but reports
  `field_equations_validated=false` because warning states fail the rebuilt
  static damage equation.
- The post-fix Anderson implementation audit is locally GO for implementation
  and manufactured evidence only. The focused stabilized/workflow suites pass
  `57/57`, legacy engineering-grip plus JAX suites pass `60/60`, the related
  baseline suite passes `7/7`, compilation and `git diff --check` pass, and no
  physical continuation was run. Anderson replay provenance truthfully reports
  scalar selection/rank data as `recorded_not_reconstructed`.
- The authorized Anderson smoke passed, and the bounded fresh-zero physical run
  strictly reached exact `u_bar=0.0059875` in 66 accepted steps. Independent
  accepted-field replay passes with `field_equations_validated=true`; the final
  reaction, damage, history, localization, and displacement amplification agree
  with fixed Picard to the expected strict-gate accuracy. Across 65 common loads
  Anderson reduces retained accepted-step sweeps from `3148` to `2155`, but the
  reduction after `u_bar=0.0045` is only `0.97%`; its last ten retained rows all
  use insufficient-decrease Picard fallback. Production runtime rises from
  `8765.93 s` to `12703.62 s` (the historical Anderson workflow total is
  `13038.79 s`).
- The authorized fresh-zero scalar diagnostic also strictly reaches exact
  `u_bar=0.0059875` in 66 accepted steps. Its `11,116` scalar JSONL records
  contain `2,866` sweeps and `5,382` candidate records. Of `2,796` accelerated
  candidates, `280` are selected and `2,516` are rejected by sufficient
  decrease. Of the rejected candidates, `2,445` improve the raw
  `candidate_actual_map_residual_inf` relative to `reference_fixed_point_error`.
  `2,502` are history-transition dominated, but `2,152` also have another merit
  component above the decrease limit, so history is not the sole blocker. The
  `280` selected accelerated candidates comprise `234` intermediate
  coupled-unconverged selections and `46` final-load accepts. No residual
  comparison to a hypothetical Picard candidate is made when it was not
  evaluated; those diagnostic fields remain explicitly unevaluated. The archive
  hash binds successfully and the independent replay/analysis reruns pass with
  `field_equations_validated=true`; the stale pre-x64 failure report remains
  preserved separately. The stream binds its recorded workflow hash, which
   differs from the current workflow hash after the x64 fix, so current
   workflow-byte identity is not claimed.
- The separately authorized `anderson_residual` implementation has
  manufactured threshold, unequal-normalizer, hard-gate, fallback/restart,
  structural-propagation, archive-mismatch, CLI, scalar-identity, and
  residual-summary/replay-binding validator coverage. Post-run verification
  passes stabilized/workflow `76/76`, legacy engineering-grip/JAX/baseline
  `67/67`, compilation, and `git diff --check`.
- The fresh-zero residual-policy physical run completed strictly at exact
  `u_bar=0.0059875` in `60` accepted steps, with `0` cutbacks, `593` complete
  sweeps (`533` failed intermediate sweeps and `60` passed final sweeps), and
  maximum retained cost `47` sweeps. Production elapsed time was `1669.159 s`
  from the durable progress log, excluding the separate replay/analysis/plot
  work. Its endpoint is monotone and notch-localized with reaction
  `638.6929499864`, maximum damage `0.7465380655`, maximum history
  `282.0386846843`, and displacement amplification `1.0599555670`.
- The residual scalar stream records `533` accelerated candidate attempts:
  `433` map evaluations, `414` selected, `100` rejected by combined hard
  prerequisites, and `19` rejected by the `0.995` sufficient-decrease
  threshold. It records `179` exact Picard fallbacks and no candidate-map
  failure. The v5 archive/replay does not reconstruct rejected trial vectors.
- Independent result review supports retaining the distinct residual policy as
  experimental: production is `7.61x` faster than historical Anderson and
  `5.25x` faster than Picard in these differing-path runs. Maximum endpoint
  damage/history differences (`1.335e-4`/`0.05052`) include an intermediate
  history maximum sampled by the old cutback paths but missed by the new
  60-step path; strict gates do not establish load-step convergence. The 100
  hard rejections lack per-component attribution. Primary final focused tests
  pass `76/76` in `320.139 s`, resolving the earlier verification timeouts.
- The authorized matched-load schedule contract is implemented and its
  extraction/provenance bindings were verified before the physical run.
  Extraction is a read-only `extract-schedule` workflow command
  that creates an exclusive no-pickle NPZ; schedule loading, config, controller,
  archive, scalar diagnostic, replay, manifest, and summary bindings use
  distinct schedule identities and verify the canonical source/target payload.
  The stabilized schedule suite passes `52/52`, the workflow suite passes
  `34/34`, and legacy engineering-grip/JAX/baseline regressions pass `67/67`
  in the `.venv`; `compileall` and `git diff --check` also pass.
- The authorized fresh-zero matched-load residual run reached exact
  `u_bar=0.0059875` in 66 scheduled attempts with zero cutbacks, 822 retained
  accepted-state sweeps, and maximum 43 stagger iterations. Independent CPU/x64
  replay, analysis, and plotting pass with `field_equations_validated=true`.
  Across all 67 historical loads, maximum differences are
  `2.67635e-10/6.04186e-08/6.82111e-05` for union displacement/damage/history;
  the final fields remain finite, monotone, and notch-localized.
- Matched production elapsed is `2803.56144060893 s`, versus historical
  Anderson `12703.620977 s` and fixed Picard `8765.930929828435 s`; these are
  `4.531244x` and `3.126713x` timing ratios. The fixed archive lacks only the
  matched intermediate target `0.005982812500000004`, so the fixed comparison
  is field-matched at 66 common states but is not a perfectly identical path.
- The durable comparison report is
  `/root/autodl-tmp/data/fracture2d/engineering_grip/engineering_grip_anderson_residual_matched_historical_u5p9875e3_s144.comparison.rerun.json`
  (SHA256
  `7d2a2b9854a789d44181890a46e101c649b06ad33af354aa12dfd8fc137bc0e5`).
- Final read-only audit verifies all `31` report-listed hashes and all `9`
  manifest artifacts. It corrects one derived-report extraction error: the
  scalar final record's archive SHA matches exactly, although the preserved
  comparison report says `final_archive_hash_matches=false`. A separate audit
  correction binds both files without overwriting evidence. Historical Anderson
  timing remains a retained external production reference, not a value
  independently recoverable from its current durable artifact directory.
- The observational hard-rejection implementation and review are complete.
  Schema v3 binds completed/failure archive paths, authoritative CG metrics,
  before/after force/moment metrics, zero-limit semantics, and numerical
  candidate-map failure separately from physical hard-gate rejection. Durable
  records require the current scalar schema and consistent candidate context.
  Sequential CPU/x64 stabilized and workflow suites pass `98/98`; compilation
  and `git diff --check` pass.

## Next up

- The completed derivation/review phase retains the scaled semismooth
  Gauss--Newton trust-region model and selects a right-preconditioned
  Newton--GMRES hookstep as the first future manufactured implementation route.
  It defines a dimensionless quadrature-normalized `q/d/H` product metric,
  projected QR/SVD trust solve, exact full-JVP prediction check, rejected-radius
  basis reuse, and explicit stationary/rank/failure semantics. Full-space
  augmented LSMR remains an independent manufactured reference. Numerical
  radius/Krylov/rank tolerances remain required project-owned controls and are
  not authorized here. Do not edit the solver or run physical continuation
  without a separate implementation decision.
- The exact `u_bar=0.0075` objective remains unmet. Do not guess further
  damping/regularization, expand/adapt `mu`, guess `k_res` or an arc-length
  rule, use warning acceptance, or treat any rejected state as physical
  evidence.
- Preserve the smaller-`du_min` checkpoint JSON, the three fixed-`mu` JSON
  diagnostics, prior checkpoint JSON records, the exact `0.0062` archive/scalar
  evidence, and the `0.0075` failure archive unchanged.

## Recently completed

- 2026-09-22 — completed the independently reviewed globalization derivation;
  selected a quadrature-metric right-preconditioned Newton--GMRES hookstep for
  future manufactured-only implementation, with no solver edit or physical run
  (see active task)
- 2026-09-22 — completed the read-only common-state, exact failure-state JVP,
  deep-Armijo, and coupled-spectrum diagnosis; identified a progressively
  near-unit notch-tip feedback mode without running new continuation (see
  `docs/tasks/fracture2d-engineering-grip-baseline.md`)
- 2026-09-22 — completed the authorized exact-`0.00615` 14-backtrack in-memory
  replay; iteration 8 passed at `2^-14`, iteration 9 exhausted the enlarged
  budget, and original artifacts remained unchanged (see active task)
- 2026-09-21 — completed primary review of the manufactured-only monolithic
  archive/workflow seam and resolved immutable-snapshot, structural-exception,
  ZIP-signature, ignored-option, durability-classification, and manifest-scope
  findings; full monolithic verification is `82/82` (see
  `docs/tasks/fracture2d-engineering-grip-baseline.md`)
- 2026-09-21 — replaced monolithic flat/final-path publication with a private
  staged `.monolithic` directory, Linux no-clobber directory rename, complete
  member fingerprints, and explicit parent-fsync durability reporting; focused
  workflow verification is `52/52` (see
  `docs/tasks/fracture2d-engineering-grip-baseline.md`)
- 2026-09-21 — completed the work-check runtime-cache and Newton residual-audit
  repairs with one-field forgery coverage and a valid controlled failing-work
  check (see `docs/tasks/fracture2d-engineering-grip-baseline.md`)
- 2026-09-21 — implemented the manufactured-only Phase-2C monolithic adaptive
  strict continuation and immutable cutback audit; primary Sol review remains
  pending (see `docs/tasks/fracture2d-engineering-grip-baseline.md`)
- 2026-09-21 — fixed the Phase-2A initial-failure audit and model-bound scale
  shape/pairing contracts with focused regressions (see
  `docs/tasks/fracture2d-engineering-grip-baseline.md`)
- 2026-09-20 — made Phase-1 monolithic cached CSR storage irrevocably
  bytes-backed/read-only and replaced stale dense-transpose docstring evidence
  with the independent-oracle bilinear evidence (see
  `docs/tasks/fracture2d-engineering-grip-baseline.md`)
- 2026-09-20 — resolved all four independent review findings on the Phase-1
  monolithic contract (frozen caches, independent JVP oracle, active-set
  switching/no-obstacle coverage, private dense helper; see
  `docs/tasks/fracture2d-engineering-grip-baseline.md`)
- 2026-09-20 — implemented and manufactured-verified the Phase-1 monolithic
  `q-d-H` residual/analytic JVP contract (no solver, no physical run; see
  `docs/tasks/fracture2d-engineering-grip-baseline.md`)
- 2026-09-20 — completed the negative fixed-`mu` sensitivity study (see
  `docs/tasks/fracture2d-engineering-grip-baseline.md`)
- 2026-09-20 — closed the smaller-`du_min` checkpoint and deeper-alpha audit
  (see `docs/tasks/fracture2d-engineering-grip-baseline.md`)
- 2026-08-18 — completed and validated the fracture2d local FEM working
  reference (see `docs/tasks/archive/fracture2d-local-fem.md`)
- 2026-08-18 — added and regenerated the requested fracture2d three-by-five
  physical-field grid (see `docs/tasks/archive/fracture2d-field-grid.md`)
- 2026-08-18 — parameterized the fracture2d field-grid loads and regenerated
  the six-column default image (see `docs/tasks/archive/fracture2d-field-grid-parameters.md`)
- 2026-08-18 — revised the default field-grid columns to the requested
  crack-growth loads (see `docs/tasks/archive/fracture2d-field-grid-defaults.md`)
- 2026-08-18 — refined the default field-grid columns to
  `0.0055`–`0.0062` crack-growth states (see `docs/tasks/archive/fracture2d-field-grid-defaults.md`)
- 2026-08-18 — resolved the Stage 2 dtype/diagnostic review blockers and
  revalidated the explicit-volume weighted scalar operator (see
  `docs/tasks/fracture2d-direct-nonlocal.md`)
- 2026-08-21 — implemented, independently reviewed, and approved Stage 3 intact
  bond-associated mechanics (see `docs/tasks/fracture2d-direct-nonlocal.md`)
- 2026-08-21 — implemented, independently reviewed, and approved Stage 4 fixed
  fracture subproblems (see `docs/tasks/fracture2d-direct-nonlocal.md`)
- 2026-08-21 — strengthened Stage 5 metadata, archive validation, independent
  reference tests, and conservation-gate coverage (see
  `docs/tasks/fracture2d-direct-nonlocal.md`)
- 2026-08-22 — resolved the Stage 5 audit findings and received independent
  implementation-review approval with the rounded-slot pilot still deferred
  (see `docs/tasks/fracture2d-direct-nonlocal.md`)
- 2026-08-23 — completed and independently approved the temporary rounded-slot
  Stage 5 pre-localization pilot through `u_bar=1e-3` (see
  `docs/tasks/fracture2d-direct-nonlocal.md`)
- 2026-08-23 — replaced persistent stage/pilot identifiers with semantic adapter,
  algebra-test, and diagnostic names (see
  `docs/tasks/fracture2d-direct-nonlocal.md`)
- 2026-08-23 — completed and independently approved the snapped fresh-zero
  warning checkpoint through `u_bar=3e-3` (see
  `docs/tasks/fracture2d-direct-nonlocal.md`)
- 2026-08-18 — completed and archived the deferred local-FEM monolithic review
  (see `docs/tasks/archive/fracture2d-monolithic-review.md`)
- 2026-08-16 — planned the staged 2D notched-fracture baseline and froze its
  geometry/continuum choices (see `docs/tasks/archive/fracture2d-notched-tension.md`)
- 2026-08-16 — refactored `AGENTS.md` into a concise policy/index with focused reference docs (see `docs/tasks/archive/agents-refactor.md`)
- 2026-08-16 — standardized the repository agent memory protocol (see `docs/tasks/archive/memory-protocol.md`)

## Known issues / open questions

- Existing `SparseD` does not yet provide per-node quadrature volumes for
  nonuniform 2D point clouds.
- Existing fracture losses use full-batch fixed operator nodes; core+halo
  sampling is not yet wired into the fracture path.
- The selected damage-aware correspondence energy and force have a reviewed
  Stage 0 derivation and corrected regular-cloud algebra check. Physical tags,
  nonuniform volumes, mesh-conforming visibility, BA support conditioning, and
  the diagnostic constrained spectrum now pass Stage 1; production-horizon and
  refinement selection remain open.
- The local FEM strict staggered convergence, mesh refinement, and optional
  monolithic solver remain deferred follow-ups. The completed task archive
  records the accepted warning-mode limitations.
