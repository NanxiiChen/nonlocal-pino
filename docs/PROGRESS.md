# Project Progress

> Dashboard, not changelog. Keep this short — detail lives in task files / git history.

## Current focus

- Active subtask: Review the completed damped stabilized-warning engineering-grip
  diagnostic and decide whether any further nonlinear-method work is justified;
  keep all strict stabilized behavior and artifacts unchanged.
- Task file: [`docs/tasks/fracture2d-engineering-grip-baseline.md`](tasks/fracture2d-engineering-grip-baseline.md)

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

## Next up

- Keep both stabilized strict artifacts and the completed warning-v1 diagnostic
  unchanged. Inspect its fields and response before deciding whether to resume
  the deferred Anderson design or stop this fixed-Picard model path; do not
  interpret later strict passes as repairing the inherited warning states.
- The severe-localization path shows suspected effectively zero-stiffness point
  drift. This issue is recorded and deferred; the current physical,
  discretization, load-step, and fixed-damping values are parser defaults, but
  warning acceptance remains explicit and nonproduction.
- The warning exact resume point is the rejected `u_bar=0.00675` state after
  committed `0.0067`. Keep its durable failure audit unchanged and inspect the
  fields/curve before deciding whether to change the healing limit, increment,
  or nonlinear strategy. Do not resume the rejected state.
- Keep all Stage 1/Stage 2/Stage 3 diagnostic horizons separate from the later
  documented common mechanics/damage horizon selection.
- Keep the conventional `ElasticD` path separate; production mechanics must be
  assembled from the approved energy-first bond-associated model.

## Recently completed

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
