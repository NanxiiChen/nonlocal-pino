# Fracture2D Staggered Adaptive Load Step

## Goal

Develop a direct nonlocal reference that reproduces the local FEM solution-field
patterns and reaction curve for validating a nonlocal PINN using the same
declared BA/Amor/AT2 equations. The user accepts qualified warning-mode
continuation and adaptive load increments, but has explicitly **rejected
investigating particle deletion or another changed model**. The peak force
and approximate onset are already satisfactory; prioritize the final
reaction and the number of drifting points. Coupled/monolithic work is paused.

## Acceptance criteria

- Audit the historical accepted and warning paths around the onset of
  notch-tip particle drift; compare the paper's pre-increment damage and
  increment thresholds against actual saved states, without treating warning
  states as strict evidence.
- Distinguish an early damage-event cutback from the existing failure-driven
  cutback, and define deterministic retry limits and restart provenance before
  changing the controller.
- Preserve original fixed-point/field-equation gate values as separately
  reported diagnostics, hard finite checks, and the accepted-state/rejected-
  attempt audit contract; label physically assessed warning states honestly.
- Verify targeted manufactured triggers, no-trigger behavior, rollback, and
  cutback exhaustion before considering a bounded physical comparison.
- Calibrate and independently validate any *additional* scaled numerical
  criterion against FEM phase-aligned responses, full nonlocal residuals,
  step refinement, and a held-out successor step; never label a failure of
  the original gates as a historical strict pass.
- If a branch switch is needed, verify that it is reachable as a solver
  procedure from the retained committed state, that warnings are measured,
  and that the resulting full path satisfies the user's physical agreement
  criteria. Preserve a distinct original strict-path status for comparison.
- User-revised physical acceptance takes priority over historical strict
  gates: compare full reaction curve/peak force and crack/displacement/damage
  morphology against FEM at matching loads and also aligned crack stages;
  report warning residuals, healing, and irreversibility without relabeling
  them as strict convergence. A concrete peak-error tolerance still needs
  selection before running an acceptance study.
- The completed observational masking study is a diagnostic only: its
  unmodified final reaction differs from FEM. Preserve **all particles and
  original fixed graph, energy, force, Amor history, and AT2 equation** in
  subsequent staggered/adaptive/warning experiments.

## Relevant context

Kristensen and Martínez-Pañeda (2020), Section 3.6, propose restarting a
monolithic BFGS increment at one tenth its size when some integration point
has pre-step damage below 0.7 and damage increment at least 0.5. They limit
the event to once per simulation. Their convergence gains come from the
monolithic quasi-Newton scheme; the damage event is a step-resolution rule,
not evidence that a staggered fixed point converges.

The earlier unbounded-warning stagger path develops a 2--4 particle notch-tip
displacement outlier at approximately `u_bar=0.0062` (particle 1869). The
strict residual-accelerated stagger path reaches `0.00623125` but rejects
`0.006234375` after 144 sweeps. The existing stabilized controller halves a
rejected increment down to `du_min`; the warning archive's post-peak fields
do not validate the full path. See the paused task
[`fracture2d-engineering-grip-baseline.md`](fracture2d-engineering-grip-baseline.md)
for the original artifacts and the exact coupled-solver resume point.

## Status / blocker / next action

Active and incomplete. The user now prioritizes preserving the accepted
original BA/Amor/AT2 operator, peak and approximate crack onset, and using
stagger with adaptive load steps plus truthfully reported bounded warnings to
improve end reaction and local drift. Particle deletion and model changes are
explicitly out of scope. Do not assume that warning relaxation or hiding tip
nodes repairs the post-peak reaction curve.
An isolated one-time event controller and
identity-only checkpoint schema exist. Source-bound physical probes show a
tip-only damage *initial guess* can reach a low-residual two-outlier branch
at `0.006118125` and the next step `0.006218125` can pass every original
strict gate. The starting branch switch has not been integrated into an
audited zero-start continuation. At the next load, `0.006318125`, and
cutbacks `0.006268125` / `0.006243125`, the original 144-sweep stagger
algorithm rejects. Paper-like one-percent and project-owned eight-percent
correction tolerances both fail a held-out successor step; neither is a
validated replacement. A locally restricted Newton correction also moves
defect to previously inactive crack-front nodes. The FEM comparison is
physically encouraging only around the first switch and includes a warning
FEM snapshot at a *different* load. No full crack-growth path, owner-replayed
adaptive archive, production dispatch, or verified physically acceptable
endpoint exists. Exact next action: derive an auditable global branch-
following strategy or explicitly select a model sensitivity study before
attempting more expensive same-control cutbacks.

## 2026-09-24 — Read-only historical event audit

- Compared immutable accepted-state arrays in the strict residual-backtracking
  failure archive, the original unbounded-warning archive, and the damped
  warning archive. This is an accepted-state, not sweep-level, trigger audit;
  no rejected candidate is treated as an accepted field.
- For the strict residual-backtracking path, tip particle 1869 at
  `u_bar=0.0061` has `d=0.8178902` and `|u|/u_bar=0.72390`; at `0.0062` it has
  `d=0.9999172` and `|u|/u_bar=16.92414`. Its load-step damage increment is
  about `0.182027`. The path continues strictly through `0.00623125`, with
  that particle's displacement amplification growing to `18.33161`.
- The original unbounded-warning path has `d_1869=0.8178664` and
  `|u_1869|/u_bar=0.72386` at `0.0061`; at `0.0062` they become
  `0.9999166` and `16.86360`. The damped-warning path similarly jumps from
  `0.8178599`, `0.72385` to `0.9999165`, `16.86095` at the same loads.
- Evaluated the paper's literal event predicate over every adjacent pair of
  accepted states: `any((d_before < 0.7) & (d_after-d_before >= 0.5))`.
  It first fires at `0.0063 -> 0.0064` (four particles) in both warning
  archives; it does not fire at `0.0061 -> 0.0062`, and does not fire at all
  along the strict archive through `0.00623125`. The damage of the drifting
  particle is already above `0.7` before the onset, and its increment is less
  than `0.5`.
- The stabilized controller already rolls back numerically rejected steps and
  halves their increments, floored at `du_min` with a cutback budget
  (`nonlocal_engineering_grip_stabilized.py:9557-9634`). A converged step is
  committed immediately (`:9537-9556`), so detecting this *strictly accepted*
  spatial event needs a separate early-rejection rule. The saved strict path
  confirms that tight fixed-point gates alone do not prevent this localized
  displacement mode; whether that mode is a numerical artifact or a valid
  near-zero-stiffness equilibrium requires separate physical scrutiny.
- Blocker: the cited paper supplies no validated alternative threshold for this
  nonlocal particle cloud and cannot justify transferring its BFGS performance
  claims to staggered iteration. Any different trigger/retry parameters must
  be selected explicitly as a bounded diagnostic, retaining all strict gates
  and accepting only convergence-tested retries.

## 2026-09-24 — Bounded diagnostic controller implementation

- The user approved `d_before<0.9`, `delta_d>=0.1`, `retry_fraction=0.1`,
  `max_events=1` as a *data-specific test*, not a general physical threshold.
- Added `nonlocal_engineering_grip_adaptive.py`: evaluate only a strict
  converged candidate before commitment, compare against the immutable
  step-start damage, retain its original complete attempt and candidate in
  separate discarded-event evidence, and retry from the same committed state.
  Missing `du_min`/cutback room produces typed failure with the accepted
  prefix. Subsequent accepted steps still satisfy the original strict gates.
  A supplied schedule and warning acceptance are rejected early. No legacy
  stabilized result or archive claims to encode these new event trials.
- Focused tests reuse a real manufactured strict staggered step to check
  event/floor rollback, unchanged no-event behavior, the historical threshold
  distinction, and an actual tenfold retry that reaches the original target
  with all strict gates. `4/4` pass in `11.968 s`; changed modules compile and
  `git diff --check` passes. No fresh physical continuation or artifact was
  created; historical archives remain unchanged.

## 2026-09-24 — Physical crack-onset diagnostic (bounded prefix)

- Added explicit `start_state` and `max_accepted_steps` controls to the
  in-memory experimental controller. A nonzero start must be authenticated
  externally; its load and shapes are checked against the owner before solve.
  An early return is labeled `completed=False`, not an endpoint result.
- A read-only scratch runner at
  `/tmp/opencode/fracture2d_adaptive_stagger_onset_probe.py` checked the
  immutable strict archive SHA256 `a1743d0e...`, source cloud SHA256
  `fa2ebf16...`, config round-trip, every strict source-state digest through
  `0.0061`, model static hashes, and the exact owner state digest
  `75e40348...`; it changed only the experimental load endpoint to `0.0062`.
  No field archive was created, and no original source file was modified.
- On `cuda:0`, the `0.0061 -> 0.0062` strict candidate converged in `59`
  sweeps; its digest is `fd86836b...`, damage jump maximum `0.1820269957`,
  and the event includes particle 1869 plus ten other particles. This
  candidate was **discarded** before commit; its particle-1869 damage is
  `0.9999171543` and tip displacement/load ratio `16.92413617`, matching
  the historical strict outlier.
- Retrying the original committed state at `0.00611` converged strictly in
  `33` sweeps, digest `916f3270...`. At particle 1869 damage is
  `0.8307810206` and displacement/load ratio `0.74655147`. Exactly one
  accepted retry exists in this bounded scratch result. The run took
  `927.776 s` from constructed model through this accepted prefix; it did
  **not** reach `0.0062` and is not a validated full physical continuation.
- An earlier unbounded scratch probe with the same source produced the same
  discarded `0.0062` and an accepted `0.00611` before timing out while
  solving the next small step; it retained no state/archive. No inference
  about the next load is drawn from the timed-out attempt.

## 2026-09-24 — Longer onset continuation and scratch snapshots

- A repeat of the source-authenticated GPU probe used an exclusive no-pickle
  scratch snapshot callback after every event/accepted step. It stores full
  prefix states, discarded candidate fields, source/model/config/rule
  identifiers, and scalar attempt records. This is a temporary recovery aid,
  not an independently equation-replayed production archive.
- The `0.0062` solved trial was strictly converged in 59 sweeps and discarded
  for the approved damage event; `0.00611` accepted in 36 sweeps. The next
  `0.00612` target exhausted all 144 sweeps and was rejected by the unchanged
  convergence gates. Its failure-driven half-increment retry at `0.006115`
  accepted strictly in 38 sweeps, tip displacement/load ratio `0.76755826`,
  state digest `8bb66828...`.
- The next `0.00612` target from committed `0.006115` was still running at
  sweep 100 when the 5400-second command limit ended the process; there is no
  accepted field or failure verdict for that trial. The latest scratch prefix
  is `/tmp/opencode/adaptive_stagger_onset_step02_event1_8bb66828a264.npz`;
  its three state digests and discarded candidate digest were independently
  recomputed from its numerical arrays and match the recorded values. Its
  `completed` flag is false and the endpoint remains `0.0062`.
- This evidence separates two effects: the early event cutback initially
  suppresses the visible isolated-particle amplification, whereas even after
  that event the established fixed-point/cutback mechanism can still stall.
  The one-time event rule cannot be called a convergence solution on this path.

## 2026-09-24 — Strict fixed-point stall after the event

- The read-only scratch probe
  `/tmp/opencode/fracture2d_adaptive_stagger_stall_probe.py` checked the
  historical strict source SHA, checkpoint source/model/config/rule bindings,
  every retained state digest, event trigger and discarded candidate digest.
  Input checkpoint SHA256:
  `59ce6c9b1c6c65d58ab3717bc12d12a0bec35fe646d98580e3e27e18629cf5de`.
  It did not write accepted or rejected fields.
- From committed `0.006115` (digest `8bb66828...`), target `0.00612`
  rejected after all 144 unchanged Anderson-residual-backtracking sweeps
  (`2386.7 s`). Seven gates failed: displacement, damage, history, reaction,
  damage residual, damage fixed-point, and coupled. Last-ten fixed-point errors
  fluctuate around `0.00133`, with damage residual near `2.0` and intermittent
  history jumps above `12` rather than converging monotonically.
- The existing failure-driven half-step reaches the configured minimum
  `du_min=3.125e-6`, target `0.006118125`. An independent read-only
  one-attempt probe there (cutback count 1) also rejected after 144 sweeps
  (`2415.9 s`), with the same seven failed gates. Last-ten fixed-point errors
  remain near `0.00063`, damage residual near `0.94`, and occasional history
  spikes above `11`. This is not an accepted field, and no further cutback is
  permitted under the current minimum-increment contract.
- The smaller step reduces defects but does not pass strict equations before
  cutback exhaustion. Neither a second damage event nor a new relaxation,
  minimum increment, or globalization rule is justified by the FEM paper;
  they require a new explicit numerical decision and independent validation.

## 2026-09-24 — Separate numerical-prefix checkpoint identity

- Added `nonlocal_engineering_grip_adaptive_checkpoint.py` with a distinct
  no-pickle prefix/event schema. Exclusive writes retain accepted-state and
  discarded-candidate fields, source archive/cloud hashes, owner model/config/
  rule identity, and scalar attempt provenance. Loading validates shapes,
  digests, the exact damage event and attempt ordering. It explicitly reports
  `field_equations_validated=False`: no physical-equation replay has yet been
  added to this new schema.
- A focused manufactured round-trip test also rejects a forged discarded
  damage field and checks no-clobber writes. Six focused adaptive tests pass
  in `14.116 s`; changed modules compile and `git diff --check` passes. The
  physical `/tmp/opencode/adaptive_stagger_onset_...npz` snapshots use an
  older scratch schema and were not silently converted or promoted.
- Exact resume point: source strict state `0.0061`, retained strict prefix
  through `0.006115` in checkpoint SHA above, and two rejected targets
  `0.00612` / `0.006118125`. Select a new staggered numerical strategy
  explicitly before any further physical continuation. Complete independent
  owner field-equation replay and production workflow/resume before declaring
  the adaptive direct solver validated or complete.

## 2026-09-24 — FEM-scaled tolerance and nonlinear feedback investigation

- New objective: determine and validate scale-aware tolerances with FEM as a
  physical comparator, then obtain a genuinely converged adaptive-stagger
  nonlocal crack-growth path. Do not silently reclassify an approximate state
  as one passing the original strict strong-form gates.
- The FEM point-anchor archive has 56 warning-free nonzero steps through
  `u_bar=0.0057`; their largest damage correction is `4.2877e-6`, history
  relative correction `9.7803e-6`, and reaction absolute correction
  `1.1759e-4`. The FEM damage/crack-growth states from `0.0058` through
  `0.006125` include 13 warnings. The FEM/nonlocal force peaks are similar
  (`642.486` versus `641.338`) but occur at different loads (`0.0057`
  versus `0.0061`); comparing at the same load after either peak is not a
  convergence test (`0.0062` force `7.014` FEM versus `631.407` strict
  nonlocal).
- A pre-existing scratch candidate
  `/tmp/opencode/adaptive_stagger_min_target_candidate_v1.npz` from the
  strictly rejected `0.006115 -> 0.006118125` target binds source archive
  SHA256 `a1743d...` and checkpoint SHA256 `59ce6c...`. Its recorded
  owner scales yield residual error/limit ratios: damage strong equation
  `86237`, damage fixed point `63467`, damage correction `31734`, history
  correction `293556`, free displacement correction `497`, reaction
  correction `171`. The exact fixed-`H_used` damage subsolve passes at
  `6.1e-8` times its limit. Correction-to-increment ratios for union
  displacement/damage/history are `0.0324/0.0316/0.0335`, exceeding the
  paper's `0.01` correction comparison (whose Abaqus normalizer differs).
- Reconstructed the original rounded-notch owner from the source cloud and
  independently checked the saved candidate. The current-damage mechanical
  free residual infinity norm is `3.648e-8`; the Amor history is exactly
  `max(H_committed,psi_plus(q))` to float64 array equality. The stored
  post-history AT2 strong residual matches the owner exactly at maximum
  `0.967899`, with `rho_scale=1122.365`, giving the dimensionless ratio
  `8.624e-4`. History is nonnegative and monotone; material damage is
  `[6.55e-6,0.851153]`, and tip displacement/load is `0.7913`.
- Re-solving the discrete fixed-history damage equation produces a maximum
  `d` correction `0.000641849` at particle 1869. A second owner mechanics
  solve at that corrected damage changes the tip displacement by
  `1.033e-5`, the history at the tip by `3.384`, and the next damage-map
  correction by `0.000662110` in the same direction. This one-direction
  secant indicates local amplification, not a proven eigenvalue. It explains
  why a small load increment alone need not converge the staggered map.
- For the frozen-history quadratic energy
  `E_H(d)=Gc/(2l) sum(V*d^2) + Gc*l/2 sum(V*d*(D@d)) + sum(V*H*(1-d)^2)`, the
  checked gap to its exact minimizer equals
  `0.5*sum(V*rho(d,H)*(d-d_star)) = 1.92981e-9`, or `8.514e-9` times the
  candidate's fracture energy `0.226671`. The strong residual is localized
  at particle 1869, whose volume is `5.402e-6`. This is evidence that
  energy-weighted error and pointwise residual measure different defects;
  neither alone certifies path-wise physical accuracy or nonlinear convergence.
- Next: set up explicit, separately named numerical/physical tolerances and
  test them against *both* the FEM pre-peak envelope and nonlocal load-step
  refinement, current-history residual, irreversibility, and near-tip drift.
  Do not use FEM warning fields as exact training/acceptance labels.

## 2026-09-24 — Retrospective tolerance grid and one-step continuation

- A private owner-backed audit
  `nonlocal_engineering_grip_tolerance_study.py` computes the fixed-history
  quadratic energy gap, current-history residual with its explicit scale
  floor, correction relative to actual per-step displacement/damage/history
  increments, exact trial history, finite/range/healing/mechanics/force gates,
  and maximum displacement/load amplification. Historical strict status
  remains independent; an `eligible_approximate` flag is experimental only.
  Focused manufactured checks include a strict smoke state and a one-sweep
  candidate with tiny residual but correction/increment ratio `2`, which is
  correctly rejected by the experimental criterion.
- Authenticated physical one-attempt sensitivity from the strict `0.006115`
  checkpoint to `0.006118125`: all 144 sweeps run, strict status false,
  six-term scaled AT2 residual `8.39495e-4`, elastic/fracture-scaled energy
  gap `9.35302e-10`, maximum displacement/load `1.05893`, and *maximum*
  correction/increment ratio `0.06734`. Controls `0.01`, `0.03`, and `0.05`
  for correction all reject; a provisional `0.08` could pass that one state
  but is not from the paper and cannot be called a validated threshold.
- Source-bound physical successor probe seeded from an independent scratch
  candidate at `0.006118125` (SHA256 `14b77d3e...`, approximate status only)
  reached the next target `0.00612125` after 144 sweeps **without** passing
  strict or experimental criteria. Correction/increment jumped to `0.71836`,
  scaled AT2 residual to `0.002546`, frozen-history gap fraction to
  `8.530e-9`; tip displacement/load stayed `0.799`. The failed candidate is
  saved under `/tmp/opencode/adaptive_stagger_tolerance_successor_v1.npz`
  with its approximate seed identity. This falsifies the idea that the
  proposed eight-percent relaxation by itself gives stable continuation.
- The FEM snapshot maximum displacement/load across peak and post-peak
  remains below `1.067` across the full archive, while healthy nonlocal states/candidate are near
  `1.059`; the historical warning crack-tip runaway is `16.86` at `0.0062`.
  Conversely, its *six-term-normalized* AT2 residual is only `4.86e-8` at
  that load because tip history already exceeds `1.19e6`. Thus a relative
  residual/energy-only test would accept a known drifting state. The
  FEM-based displacement envelope is a **model-dependent diagnostic**, not
  an unqualified general constitutive constraint.
- Exact next action: test half the configured minimum step (`1.5625e-6`)
  from the retained strict `0.006115` state without relaxing any strict
  gates. Preserve all discarded/failed scratch candidates as unvalidated.

## 2026-09-25 — FEM-informed one-percent branch-following falsification

- Single-target strict probe from authenticated `0.006115` to
  `0.0061165625` with `du_min=1.5625e-6` and unchanged 144-sweep strict
  gates failed seven coupled gates after all sweeps; its scratch fields are
  `/tmp/opencode/adaptive_stagger_half_minimum_from_u6p115_v1.npz`.
  The owner-reconstructed field has AT2 current-history residual `0.084676`
  on six-term scale `1100.564`, fixed-point correction/increment `0.006868`,
  history correction/increment `0.003667`, fixed-history energy gap/fracture
  energy `6.83e-11`, and max displacement/load `1.058946`. This is not a
  historical strict acceptance.
- An independently rerun source-bound full-sweep audit at the same target
  (`/tmp/opencode/adaptive_stagger_half_minimum_from_u6p115_v2.npz`, source
  SHA256 `38780186...`) found all original hard gates true, Amor history
  bitwise exact, current-history scaled residual `7.77946e-5`, energy gap
  relative to larger elastic/fracture energy `7.97071e-12`, and the maximum
  of displacement/damage/history/fixed-point corrections divided by the
  corresponding load-step increments `0.0067525`. The proposed diagnostic
  `1%` correction, `0.5%` scaled residual, `1e-6` energy gap, and `1.08`
  FEM-informed maximum displacement/load envelope would accept **only an
  approximate, non-strict seed**. All old strict gate flags remain false.
- A subsequent single-target physical attempt from that experimentally
  approximate seed to `0.006118125` ran 144 sweeps and failed both the old
  strict gates and the experimental audit. The source-bound scratch output
  `/tmp/opencode/adaptive_stagger_onepercent_successor_v1.npz` records
  correction/increment `3.2880`, scaled AT2 residual `8.5504e-4`, fixed-
  history energy gap `9.2517e-10`, and maximum displacement/load `1.05893`.
  Hard mechanics/history/range gates remain true, but the per-step correction
  explodes from `0.00675` to `3.288` in one step. No successor is accepted.
- The historical warning path is an important negative control: at `0.0062`
  its scaled AT2 residual is only about `4.86e-8`, while tip displacement/load
  exceeds `16.86`; at `0.0063` history exceeds `2e6` and strong residual
  grows. A normalized residual/energy criterion by itself would accept a
  known outlier. The FEM nodal maximum displacement/load stays below `1.067`
  across the archived reference path. The `1.08` envelope is therefore an
  explicit data-specific guard, not a general physical law.
- Conclusion: neither the paper's `1%` correction sensitivity nor project-
  owned `8%` relaxation yields a verified stable staggered crack-growth
  branch. Next research must distinguish (a) an unstable/nonexistent smooth
  load-controlled root from (b) failure of the current alternating fixed-
  point globalization, then select an explicitly derived branch-following
  method. Do not increase the tolerance to classify a 329% correction as
  convergence.

## 2026-09-25 — Conditional crack-tip fixed-point map

- Bound the source/cloud and strict-prefix hashes, then held all material
  damage values except notch-tip particle 1869 at the rejected smooth
  `0.006118125` state. For each specified tip value, the unchanged owner
  solved fixed-damage mechanics, committed-history Amor, and fixed-history
  AT2 damage. The conditional map residual is
  `F_1869(d) - d_1869`; it does **not** certify a full vector equilibrium.
- Tip-map residuals are positive at `d_tip=0.8411168` (`0.001517`),
  `0.851153` (`0.000642`), `0.86` (`0.001438`), `0.90` (`0.02283`), and
  `0.999` (`0.000721`); at `0.9999` it is `-0.000177`. Tip displacement/load
  rises from `0.791` at the smooth candidate to `9.282` at `0.9999`.
  Another particle (1873) retains a damage-map defect of `0.03084` at
  `0.9999`, so the sign change in one coordinate is **not** a full root.
- This conditional fold-like map plus the failed one-percent successor
  supports a branch/globalization diagnosis rather than further tolerance
  widening. The script `/tmp/opencode/fracture2d_tip_map_scan.py` is a
  read-only physical probe; no candidate was committed or promoted.

## 2026-09-25 — High-damage-seeded alternate stagger branch

- Read-only owner experiment `/tmp/opencode/fracture2d_high_branch_stagger_probe.py`
  combined the *same* last strict low-history `0.006115` state and target
  `0.006118125` with the separately authenticated, high-damage seed from the
  historical strict `0.0062` archive. Every sweep solved the original fixed-
  damage mechanics, committed-history Amor, and fixed-history AT2 equations;
  only the initial damage guess came from a different load path. No state was
  committed or classified as a valid continuation step.
- With fixed damage damping `0.5`, the full-vector damage map defect
  contracted from `0.006328` (sweep 1) to `6.204e-6` (140), `8.081e-8`
  (220), and `9.413e-9` (258), crossing the original `1e-8` damage map
  limit in 258 sweeps. This does not prove that all original coupled gates
  pass: the experiment did not retain final displacement/history/reaction
  correction checks and used a different-load initial guess.
- The near-root has `d_1869=0.999884`, crack-tip history approximately
  `903640`, reaction approximately `633.01`, and tip displacement/loading
  ratio `14.769`, versus the FEM snapshot maximum ratio below `1.067` and
  the smooth rejected nonlocal candidate around `0.79`. The initial seeded
  sweeps healed up to `6.15e-5` relative to the committed state, although
  after 15 sweeps the raw healing minimum becomes positive. The high-damage
  branch is not promoted: an algebraically contracting map may violate the
  physical comparison target even if later gates become small.
- Together with the conditional tip map and two failed smooth approximate
  successors, this establishes a concrete **branch-selection and local field
  regularity** issue that scalar residual or correction tolerances cannot
  resolve. Next action: decide explicitly whether to seek a strictly
  continued alternate branch and accept the model's highly localized tip
  displacement, or to undertake a separately derived model/operator
  sensitivity study. Do not tune a numerical tolerance to turn the currently
  noncontracting smooth candidate into a supposedly converged root.

## 2026-09-25 — At-most-two-point physical comparison and tip-only restart

- The user accepted a candidate branch for further study **if no more than
  one or two localized displacement outliers are excluded**, the remaining
  whole-field solution resembles FEM, and the full discrete equations
  converge. This is a conditional physical comparison criterion, not
  permission to delete particles from the nonlocal graph or omit their
  equation residuals.
- A second scratch route uses the strictly retained low-history `0.006115`
  state and rejected smooth `0.006118125` candidate, setting **only the
  initial guess** at particle 1869 to `d=0.9999`. All subsequent mechanics,
  Amor, and AT2 equations are unchanged; no d/H clipping or physical
  parameter edit is made. Its 320-sweep fixed-omega map reaches a maximum
  damage fixed-point defect `1.11e-16`, AT2 strong residual `9.66e-17`
  relative to its scale, zero final q/H/reaction correction to represented
  precision, and positive damage/history gaps against the committed state.
  The full field is saved at
  `/tmp/opencode/adaptive_stagger_tip_only_seed_v1.npz` (SHA256
  `54fb7498ad200a6c412c36381419a9d1efb9112a448ce7b6e5b9ce5fcf0c9b97`).
  This is still experimental source-bound scratch, not a workflow/archive
  accepted step.
- The material cloud and FEM archive have **identical ordered 3395 nodes**.
  At `0.006118125`, exactly two particles (1869, 1873) exceed a declared
  maximum displacement/load ratio of `1.08`: ratios `14.769` and `4.955`;
  the maximum over the other 3393 nodes is `1.05827`. The full FEM snapshot
  collection has a maximum ratio `1.06670` (the previous `<1.064` statement
  omitted earlier low-load snapshots; it remains true only for snapshots
  from `0.005` onward). Excluding those **predeclared** two nodes, the
  volume-weighted RMS relative difference of normalized displacement from
  the FEM `0.0058` snapshot is `1.72%`; volume-weighted damage RMS
  difference is `0.00684`; reaction is `633.01` versus FEM `614.67`.
  FEM `0.0058` is warning-accepted and its peak loading is shifted from
  the nonlocal peak, so this is phase-aligned qualitative/quantitative
  similarity, not a strict same-load continuum validation. At the same
  nominal `0.0061/0.0062`, differences are substantially larger.
- From the tip-only scratch root, the **original** strict stabilized
  attempt at `0.006218125` accepted after `39` sweeps with no failed gates,
  finite/range/healing pass, maximum damage `0.999923`, reaction
  `630.087`, and the same exactly two displacement-envelope outliers.
  Source-bound scratch state:
  `/tmp/opencode/adaptive_stagger_high_branch_next_step_v1.npz`, SHA256
  `d2855ab006db56d5ed802ca5410045e8671566132869d813ec2df3bf3f7ef742`.
  Its parent was experimentally seeded and is not a certified original
  zero-start continuation prefix; no legacy strict archive is replaced.

## 2026-09-25 — Next crack-front stall and restricted Newton diagnosis

- Continuing from the strict post-switch `0.006218125` scratch state, nominal
  `0.006318125` and cutback `0.006268125` both reject after 144 sweeps;
  the second-cutback `0.006243125` also rejects after 144 sweeps. An
  additional 60 fixed-omega sweeps from its last candidate increase the
  full damage-map error from `0.000637` to `0.000774` with exactly the same
  two displacement outliers. No rejected state was committed. The paired
  failed candidate is
  `/tmp/opencode/adaptive_stagger_branch_cutback_target_v1.npz` (SHA256
  `a521ca2eaeb4102c387367a40829c3d29c72ffa074616f1b4e132db817ca8cab`).
- A provisional `1%` correction-to-actual-field-increment diagnostic would
  accept that failed `0.006243125` candidate (`max=0.00918`), but its
  one-step successor at `0.006268125` rejects after 144 sweeps with
  fixed-point correction/increment `0.12595` and all seven historical
  coupled gates failed. Thus simple threshold relaxation again fails a
  one-step stability validation, even though both fields have only two
  displacement outliers. Successor scratch is
  `/tmp/opencode/adaptive_stagger_high_branch_onepercent_successor_v1.npz`.
- Conditional scans of the next front particles 1977 and 2001 show each
  scalar map error changes from small positive near its current damage to
  negative by damage `0.8`, with no extra displacement outlier under a
  *single-node perturbation*. A restricted finite-difference Jacobian of
  the **full owner** map at the failed candidate has eigenvalues
  `(-0.455,-0.881)` for the two-node block; solving its 2x2 correction
  reduces those local errors to around `5e-7`, but the *global* map infinity
  error rises from `0.000637` to `0.001167` and volume-weighted RMS rises
  from `2.92e-5` to `4.19e-5` as the defect moves to node 2034.
  For the 32 largest residual nodes a near-zero eigenvalue `+0.00823`
  and smallest singular value `0.00660` appear; the unconstrained projected
  Newton step reduces damage by up to `0.0804`, increases the global error
  already at quarter step, and eventually violates irreversibility.
  This is evidence of a collective near-fold mode, not proof of full-root
  nonexistence.
- Exact resume point: keep the two-outlier rule as an *observational* guard,
  retain original field equations and strict flags, and derive/validate a
  branch-following staggered globalization or an explicitly selected model
  sensitivity study. Do not classify the nearby failed candidate as
  converged under the paper's `1%` correction rule; its immediate successor
  contradicts that classification.

## 2026-09-25 — Self-seeded same-target stagger restart

- A scratch extension of the failed `0.006243125` iterate with original
  fixed-damage mechanics/Amor/AT2 equations and fixed `omega=0.5` reaches
  the original scalar damage-map, strong-residual, displacement, history,
  reaction, and healing limits at cumulative sweep `1030`, without taking
  any warning archive field as a seed. Owner recomputation gives free
  mechanics residual `8.18e-8`, current-history AT2 residual `7.12e-6`,
  exact Amor committed-history identity, and force/moment defects near
  roundoff. A warning-derived damage initial guess found a nearby root
  faster, but is **not** a required dependency; the two scratch root fields
  differ by at most `4.66e-8` in q and `6.24e-7` in d.
- Added isolated `run_staggered_self_restart(...)`. It accepts only a
  paired, complete, strict `gates` failure. The failed final damage iterate
  becomes an explicit admissible restart initial guess at the **same load**
  and with the **unchanged committed state/history**. Fixed-omega and extra
  sweep budget are required parameters. Its separate immutable evidence
  binds original/recovery audits, seed digest, and accepted strict step; the
  old continuation/archive schema cannot record this new chain. Invalid
  float64 seed shape and inconsistent committed state fail loudly.
- A source-bound real GPU run repeats original 144-sweep Anderson failure at
  `0.006243125`, then accepts in **885** additional fixed-omega sweeps with
  *all* original strict gates passing (experimental result digest
  `7eaef42f...`); independent owner AT2 residual `7.55e-6`, free mechanics
  `6.79e-8`, and exact Amor history are paired. Scratch artifact:
  `/tmp/opencode/adaptive_stagger_self_restarted_physical_v1.npz`.
  With only the predefined particles 1869/1873 excluded **from physical
  comparison**, the other 3393 matched FEM nodes at FEM load `0.0059`
  (warning state) differ by `2.048%` in normalized displacement weighted
  RMS and `0.008065` in damage weighted RMS; reactions are `592.787`
  versus FEM `589.764`. This is not yet a full source-to-target validated
  adaptive continuation: the earlier branch switch remains experimental.
- One manufactured restart regression passes using a genuine one-sweep
  failure and verifies that the strict accepted recovery retains the
  original committed-state digest, explicit seed, and fixed policy. Changed
  modules compile and `git diff --check` passes. The next physical target
  after this accepted scratch step remains unverified.

## 2026-09-25 — Repeated post-restart displacement-step cutbacks

- The immediately following held-out `0.006268125` step from the strict
  self-restarted `0.006243125` scratch root passed all original gates in
  32 Anderson sweeps, with no new restart. Its source-bound scratch state
  digest is `f543e8d3...`, owner AT2 residual `9.34e-7`, free mechanics
  residual `4.73e-8`, and exactly tip particles 1869/1873 exceed the
  FEM-based `1.08` displacement comparison envelope.
- Bounded continuation from this strict scratch state accepted
  `0.006293125` in 30 Anderson sweeps (digest `392926643e3f...`, source
  checkpoint SHA256
  `bc17be7f18162661409fc653c5dcf6fa1d747625a5ebdc3af477f9711ec2c159`).
  The nominal next `0.006318125` and first cutback `0.006305625`
  each rejected after original 144 Anderson plus explicit 1000 same-target
  fixed-omega recovery sweeps. A read-only continuation then strictly
  accepted the next cutback `0.006299375` in 37 Anderson sweeps (digest
  `17e4e034...`, source-bound checkpoint
  `/tmp/opencode/adaptive_restarted_branch_step01_17e4e03461a9.npz`).
  Independent AT2 residual reconstruction at that state gives `1.69e-6`;
  the two displacement outliers are unchanged.
- The next attempt from retained `0.006299375` toward `0.006305625`
  was in progress when a separate 7200-second command limit ended the
  process; there is no accepted or terminal failure audit for that trial.
  All preceding rejected targets retain the earlier committed digest;
  none was warning-accepted. Repeated failures followed by smaller strictly
  accepted increments show the need for branch-aware increment control or
  a different globalization; simply increasing the inner sweep limit to
  1144 is not enough to finish the full load path.
- Exact resume point: inspect owner-validated accepted scratch state at
  `0.006299375` and compare any new branch/energy control against the
  unchanged original equations, the FEM phase-aligned field envelope, and
  the two-outlier count. Do not repeat the two unchanged long failed
  targets without a materially new numerical question. The full target
  `0.0075` and owner-replayed adaptive event/restart archive remain open.

## Objective-to-evidence checklist (still open)

| Requested deliverable | Current concrete evidence | Status |
| --- | --- | --- |
| Scale-aware FEM/nonlocal convergence study | FEM and nonlocal archives, source-bound AT2/operator recomputation, `nonlocal_engineering_grip_tolerance_study.py` | Investigated; no generally validated threshold |
| Tolerance with held-out crack-growth convergence | One-percent smooth and high-branch provisional states fail their *next* physical step; eight-percent smooth state also fails | Not achieved |
| Adaptive staggered direct solver reaches full crack-growth target | One-time event controller and self-seeded restart are isolated; post-switch strict scratch steps through `0.006299375`, but no audited full adaptive path | Not achieved |
| FEM-like field with at most two diagnosed displacement outliers | Identical material/FEM node ordering; only 1869/1873 exceed declared `1.08` envelope through experimental `0.006243125`; stage-aligned FEM `0.0059` warning-snapshot RMS comparison | Locally supported, full path untested |
| Faithful equations/irreversibility and honest archive replay | Original mechanics/Amor/AT2 unchanged; scratch fixed-point residuals and individual step gates checked; checkpoint writer is identity-only with `field_equations_validated=false` | Full accepted-path owner replay not achieved |
| Verification | Nine focused manufactured adaptive/tolerance tests pass in `13.992 s`, changed modules compile, `git diff --check` passes | Focused checks pass; full physical validation incomplete |

## 2026-09-25 — Energy-controlled staggered trial and new operator question

- The user selected fracture-energy-controlled outer continuation with
  unchanged staggered fixed-damage mechanics, committed-history Amor, and
  fixed-history AT2 subproblems. Added isolated
  `nonlocal_engineering_grip_stagger_energy.py`: discrete surface-only AT2
  energy and its original-damage-gate-derived roundoff/gradient/Hessian
  tolerance; explicit finite load bracket with completed trial evidence;
  alternating outer solve that invokes the authoritative
  `_full_diagnostics`, strict fixed-point and energy gates, plus a separately
  reported two-outlier physical criterion. The method has no production
  workflow, source-to-target archive replay, or accepted full path. Three
  focused manufactured algebra/bracket/rollback tests pass.
- From the source-bound strict scratch state `u_bar=0.006299375`, the
  previous accepted AT2 energy increment is `0.01545740144`. A frozen-damage
  bracket crosses its target energy `0.49652814267` between loads
  `0.00645` and `0.00650`; an independent owner trial selects
  `0.00647313232` with energy error `1.45e-8` under the derived limit
  `5.08e-8`. This is one scalar subproblem result, not a coupled solution.
- A bounded 96-outer-iteration GPU trial at the same target reaches all
  **original nonlocal field-equation/correction gates** plus the derived
  energy gate by outer sweep 89, load approximately `0.00629592645`.
  However the final field has **three** material particles above the
  explicitly selected FEM displacement/load envelope `1.08`
  (`1864,1869,1873`), not the user's allowed one or two; the candidate is
  rejected with `physical_comparison_pass=false`. The source-bound scratch
  file is `/tmp/opencode/adaptive_stagger_energy_point_from_u6p299375_v1.npz`
  (SHA256 `6ca39c9689abf08fb75c9389666e504aa47d489259f6d2d92f8bc0870804cd6e`).
  The first zero-load bracket attempt failed an explicit JAX PCG residual
  gate; selecting the preceding **accepted** positive load as the lower
  bracket endpoint made the valid scalar bracket observable. No failed
  candidate was committed.
- The user's new question is whether the *nonlocal operator design* causes
  recurring crack-tip outliers. Frozen-field owner audit on the same mesh
  finds ordinary target-inclusive supports and modest shape conditions:
  node 1869 has 14 source/target edges, source support sizes 8–14,
  largest incident shape condition `6.19`; node 1873 has 15 edges,
  support 8–15, max condition `3.65`. Both have small nodal barycentric
  volumes (`5.402e-6`, `1.359e-5`) and no grip-phase center measure.
  At the high-damage strict scratch field, their degradation factors are
  `1.205e-8` and `1.415e-7`; minimum diagonal BA tangent-block eigenvalues
  collapse from intact `1200.9/3179.5` to `0.4725/3.1416`. The energy-
  controlled third outlier 1864 has damage `0.99653`, local BA block
  minimum `0.9463`, while intact is `1391.6`. This supports a damaged
  localized weak-mode mechanism, not an omitted-neighbor or rank defect.
- Independent P1 CG1 plane-strain stiffness integration at exactly the
  same nodal damage field/geometry/material computes degradation using
  exact triangle average
  `[(sum(1-d_a))^2+sum((1-d_a)^2)]/12+k_res`. For nodes 1869/1873
  the corresponding FEM local block minima are `1643.1/1068.9` at high
  damage, versus BA `0.4725/3.1416`; the FEM/BA intact minima are
  `324209.6/1200.9` and `470730.2/3179.5`. The virtual energy of the
  **same** two-node nonlocal-versus-FEM displacement mismatch (FEM
  `0.0059` snapshot rescaled by applied displacement) is BA `0.006282`
  versus FEM `32.126` under identical damage, an energy ratio
  `1.955e-4`; intact ratio is `0.003313`. This is evidence of high-
  frequency mode softness intrinsic to the current source-weighted BA
  discretisation, further amplified by centered damage quadrature. It
  does not itself prove a coding bug or authorize changing the kernel,
  degradation law, or `k_res`.
- Exact next action: finish a **frozen damage only** k-residual-stiffness
  sensitivity, separating the effect of chosen `k_res` from BA quadrature;
  then derive/review any alternative damage-aware operator before changing
  the model or claiming full-path convergence. All particles remain in
  mechanics, AT2, history and force balance even when the FEM comparison
  reports outliers separately.

### Frozen-damage residual-stiffness falsification

- `/tmp/opencode/fracture2d_outlier_kres_frozen_probe.py` independently
  re-solves **the same high-damage field** at `u_bar=0.006243125` using
  `k_res` `1e-8, 1e-7, 1e-6, 1e-5`, rebuilding the full model and GPU adapter
  for each changed constant and checking the original strict seed hash. The
  full free-force residuals are all at most `4.4e-7`; these solves test only
  frozen mechanics, **not** coupled stagger/history/AT2 consistency under
  the modified models.
- Node 1869's displacement/load decreases only `21.8375 -> 21.1177`
  despite a 1000-fold increase in `k_res`; node 1873 decreases only
  `6.6675 -> 6.4514`. Local BA block minima move just `0.47249 -> 0.50245`
  and `3.14156 -> 3.24461`. All four runs have exactly two values above
  envelope `1.08`, and top reaction changes only `592.7872 -> 592.8492`.
  Consequently the local weak modes are **not mainly controlled by the
  residual stiffness floor**, and adjusting `k_res` in the tested range is
  not a solution to the drift. The dependency is principally BA's
  nonlocal shape/energy/quadrature with neighboring damaged source centers.
- Current goal was explicitly edited to prioritize a mathematically faithful
  remedy for these near-zero-stiffness point drifts and fully converged load
  path. Only **if strict resolution is genuinely impossible** should deleting
  drifted particles or forcibly setting their damage to 1 be investigated,
  with changed equations/force/energy/physical response tested openly. Do
  not assume either is equivalent to the present model or would stabilize it.
- Exact next action: inspect the approved BA kinematics and documented
  source-backed formulation for a remedy that retains variational mechanical
  consistency and the existing damage/history coupling. Distinguish
  intrinsic finite-horizon weak high-frequency modes from a damaged
  integration defect. A change to the declared operator requires an explicit
  formulation decision under `docs/NUMERICAL_METHODS.md` § Stop Conditions.

### Energy-source diagnosis and decision point

- The frozen virtual-mode energy decomposed by exact BA source centers:
  intact centers 1869 and 1873 contribute `12.35116+3.46873=15.81989` of
  the entire `15.85679`, or about `99.77%`. At the **identical** high-damage
  field they contribute only `1.49e-7+4.91e-7`; most remaining BA mode energy
  `0.006282` comes from less damaged neighbors 1858/1876/1912. Direct
  `A2FixedDamageView.energy` agrees with edge-energy decomposition within
  `1e-10`. All local blocks remain positive: "zero stiffness" denotes a
  near-zero localized mode, not literal loss of rank.
- The FEM P1 deformation gradient is elementwise constant and integrates
  `g(d_h)` across the neighboring triangles, rather than assigning a
  nearly zero degradation to the entire energy of a single source center.
  This explains how FEM remains stiff against the **same** two-node mode.
  Neither switching solver nor merely raising `k_res` addresses the
  discretisation distinction. For fixed accepted `d` and load, the
  positive-definite BA mechanics problem has a unique displacement, so
  demanding a different non-drifting displacement at that same `d` requires
  changing either the damage equilibrium branch, spatial discretisation,
  boundary treatment or the declared mechanical energy.
- Stop condition: picking an edge/interpolated degradation quadrature or a
  stabilized correspondence penalty changes the existing variational law;
  no unique reference-backed alternative preserving the declared coupled
  formulation has been selected. Ask the user to authorize the physical
  formulation direction before implementing a new operator. Do not treat
  the present evidence as proof that *every* faithful strict alternative
  is impossible or as permission for conditional point deletion/forced
  damage. Resume with that explicit decision, then test affine/rigid modes,
  energy gradients/tangents, AT2/history derivatives, FEM fields, and the
  complete strict load path under the declared new model.

### PINN-reference constraint and an explicitly *unapproved* kernel candidate

- The user clarified that the direct solver is foremost a **reference for a
  future nonlocal PINN**: the remedy must use equations and static graph data
  directly reusable by the PINN, not a nodewise solver-only correction.
  Preserve the original BA source-weighted model/archive as a distinct
  comparator while evaluating any new formulation.
- A provisional project-owned, **not source-backed as an existing published
  fracture model**, purely nonlocal graph-quadrature candidate uses the
  original positive compact-support kernel with a self weight: for material
  indices, set `A_ij=J_delta(|X_i-X_j|) V_j` on the original reciprocal
  material graph, plus `A_ii=J_delta(0) V_i`, and normalize each row
  `W_ij=A_ij/sum_k A_ik`. The graph and all row sums are fixed and checked.
  The proposed material-center elastic energy would be

  $$
  \Pi_{m,W}=\sum_i V_i^m\Psi_i(u)\bar g_i(d)
  +\sum_i V_i^g(1+k_{\mathrm{res}})\Psi_i(u),
  \quad\bar g_i(d)=\sum_jW_{ij}[(1-d_j)^2+k_{\mathrm{res}}].
  $$

  The mechanics force/tangent follow by applying the **existing** BA
  gradient and its adjoint with `V_i^m bar_g_i + V_i^g(1+k_res)`; affine
  mechanics at uniform `d` is unchanged because `W 1=1`. One matching
  history-field candidate retaining the existing fixed-history diagonal
  AT2 reaction term has a volume-adjoint positive-energy driver

  $$
  \widetilde\Psi_j^+(u)
  =\frac1{V_j}\sum_i V_i W_{ij}\Psi_i^+(u),
  \qquad H_j^n=\max(H_j^{n-1},\widetilde\Psi_j^+(u^n)).
  $$

  This follows exactly by differentiating the proposed elastic energy with
  respect to `d_j` and replacing its current tensile component by the
  historical maximum; it **changes** the original local-source history
  coupling, so existing strict snapshots cannot become accepted states of
  this candidate. No regularisation parameter beyond the previously
  declared horizon/kernel is introduced; nevertheless whether averaging
  degradation at this physical horizon is a defensible fracture model
  requires separate review and full coupled validation.
- The bound read-only frozen-mode probe
  `/tmp/opencode/fracture2d_pinn_transferable_kernel_probe.py` on the old
  strictly accepted `0.006243125` *damage field* preserves `W 1=1` and
  positivity, but replaces the old mechanics center factor only in a
  **nonaccepted** trial. Averaged elastic factors at 1869/1873 become
  `0.09453/0.07696`, versus original factors `1.20e-8/1.42e-7`. An original-
  source-seeded GPU fixed-damage mechanics re-solve has free residual
  `3.88e-7`; those node displacement/load ratios become `0.670/0.452`
  instead of `21.84/6.67`; **zero** nodes exceed `1.08` at this one
  frozen state. Reaction changes `592.79 -> 626.99` (about 5.8%). This
  demonstrates only that a **PINN-transferable energy modification might
  remove frozen-state tip drift**. It says nothing about a valid new
  history/damage fixed point, FEM convergence, origin-to-endpoint path, or
  physical acceptability of changing the nonlocal fracture-zone width.
- Next action: independently review the graph weights, volume-adjoint
  history and AT2 surrogate (including max/history rollback) against the
  intended nonlocal PINN residual representation, then obtain a clear
  formulation decision before production implementation. If approved,
  develop a separately named model/provenance, patch/energy-gradient/tangent
  tests, fixed-history equivalence and same-graph PINN residual contract;
  continue the entire path from zero using strict gates before declaring a
  reference. If not approved, continue original-BA branch/mesh studies
  without silently changing equations.

### Session handoff — goal execution limit

- The user explicitly approved researching the independently named,
  PINN-transferable candidate. Its first row-normalized W lacked a constant-
  preserving **volume adjoint** near truncated boundaries; replacing it with
  symmetric volume Sinkhorn balance yields relative primal/adjoint row defects
  below `7.4e-14` in 44 iterations on the production cloud. At the same
  frozen damaged state, its mechanics solve has zero nodes above the `1.08`
  envelope, but this is not a coupled state of the new model.
- Added `examples/fracture2d/nonlocal_engineering_grip_kernel_damage.py` as
  an isolated algebraic candidate with fixed positive volume-balanced graph,
  distinct model identity, effective-center-damage **computational proxy**,
  and volume-adjoint positive-history transform. Its accompanying manufactured
  test `tests/test_fracture2d_nonlocal_engineering_grip_kernel_damage.py`
  passes `4/4`, checking primal/adjoint conservation, numerical derivative of
  changed energy with respect to *physical* damage, BA energy/force/tangent,
  and invalid inputs. A test warning about a regex escape was corrected after
  that run; a clean rerun is still due. Example README documents research
  status. No full candidate stagger controller or new-model state archive
  exists. These additions do not validate any full-path convergence.
- The OpenCode goal state was reported as `usageLimited` (`max auto-continues
  reached (25)`), although the goal has not been completed or marked unmet;
  no more substantive execution is permitted in this goal state. Resume
  point once the goal can be active: rerun the focused test after the regex
  correction, independently validate the volume-balanced kernel on the
  production cloud, construct a separately provenance-bound zero-start
  mechanics/adjoint-Amor/AT2 stagger controller with **original explicit
  gates**, and attempt the complete load path before considering point
  deletion or imposed full damage. Do not treat effective mechanical center
  damage as physical damage or use old-model saved states as committed seeds.

## 2026-09-26 — Authorized graph candidate: strict full path, failed physical reference

- Goal resumed. The approved independent model now has an isolated
  `KernelDamageState` binding candidate identity to physical `q/d/H`;
  `solve_kernel_damage_attempt` retains the original owner _full_diagnostics
  gates **and** checks the raw unrelaxed damage-map error. The fixed-damage
  BA view contains an effective-center-damage computational proxy only; the
  state records unmodified physical `d`. The graph's immutable CSR buffers
  cannot have their write flags re-enabled. Five focused manufactured tests
  pass, including a genuine CPU zero-start strict step and exact adjoint-
  history replay; GPU production-cloud defects of forward and volume-adjoint
  constant preservation are `7.33e-14`/`7.31e-14` (44 balancing iterations).
- A source/archive/code-bound experimental GPU continuation starts at exact
  zero and accepts all 75 nominal `du=1e-4` increments through exact
  `u_bar=0.0075` with no nonlinear cutback. Its final step takes 73 sweeps;
  original explicit mechanics/AT2/history/correction/range/healing/force/
  moment gates and the extra raw-map gate pass. Full `q,d,H` scratch rows
  `/tmp/opencode/kernel_damage_candidate_zero_prefix_v1_step001.npz` through
  `_step075.npz` are exclusive writes and bind source/model/candidate code/
  parent/state digests; they still declare
  `field_equations_independently_replayed=false`, **not** a durable production
  accepted/replay schema. Every accepted step was followed by a separate
  same-model fixed-damage re-solve, exact adjoint-Amor max, fixed-history AT2
  recomputation, strict force/moment and no-healing check. Prefix steps were
  identity-chain checked on resumption; the recomputation is recorded in
  `/tmp/opencode/fracture2d_kernel_damage_continue_probe.py` outputs.
- The candidate removes point drift across this entire tested path: maximum
  displacement/load at the exact `0.0075` endpoint is `1.05606`, no node
  exceeds the declared diagnostic `1.08`; maximum physical damage is
  `0.7660688`, maximum history `252.489`, reaction `729.1946`, and current-
  history AT2 strong residual `6.24e-7`. This is strict convergence **of a
  different physical model**; it is *not* strict convergence of the original
  BA equations nor evidence that the target fracture event was resolved.
- The same-node, same-load FEM comparison is a decisive physical negative
  control: at `0.0057` (FEM warning-free) candidate reaction `632.698` vs
  FEM `642.486`, damage maxima `0.3697` vs `0.6138`; at `0.0062` candidate
  reaction `669.854` vs FEM `7.014` and damage maxima `0.45235` vs
  `1.00527`; at final `0.0075`, candidate reaction `729.195` vs FEM
  `3.977`, damage maxima `0.76607` vs `1.00454`, same-node normalized
  displacement weighted RMS error `0.24718`, and AT2 surface energies
  `0.44010` vs `2.08405`. Some FEM onset snapshots are warning-accepted;
  the post-peak `0.0062`/endpoint reactions are archived as warning-free
  steps but follow a warning segment. This candidate postpones fracture
  beyond the target span and is unsuitable as a matching PINN reference.
- An exactly-final-load snap (8 ulp) was necessary because nominal step
  `75*1e-4` is `0.007500000000000001` while the configured endpoint is
  exactly `0.0075`; an earlier 75th-step call raised a loud out-of-range
  error without a new state. Only the snapped, equation-checked endpoint
  was retained.
- Next: do not promote or repeat the unchanged horizon-averaged candidate.
  Compare a **more local, independently derived** degradation quadrature
  that retains constant/volume-adjoint conservation and a fixed graph/PINN
  residual with the same freeze/strict/physical tests, or obtain a user
  formulation choice if a mesh-based candidate changes acceptable physics.
  A zero-outlier but nonfracturing solution cannot close the goal.

## 2026-09-26 — Independently named P1-mass candidate falsified at crack growth

- Following the user's approval of a mesh-derived but fixed-shape and
  PINN-reusable quadrature research candidate, added
  `nonlocal_engineering_grip_p1_mass_damage.py`. Standard P1 consistent mass
  over the original material triangles is positive, symmetric in its
  volume-weighted form, reproduces constants with both primal/adjoint,
  requires every triangle edge to belong to the original nonlocal graph,
  validates barycentric nodal volumes and triangle areas, and binds a
  different SHA256 model identity. It reuses the separately derived common
  BA energy/force and volume-adjoint Amor/AT2 candidate. The fixed CSR is
  bytes-backed immutable. Seven focused candidate tests pass, including
  a real manufactured zero-start strict step and model-mismatch rejection.
- A source-bound frozen **old-model** high-damage `0.006243125` test first
  found P1 factor `0.009801/0.002911` at nodes 1869/1873, compared with
  original factors `1.205e-8/1.415e-7`; the same frozen field had only one
  node above `1.08` (1869 amplification `1.3065`) and reaction `611.11`.
  This was an initialization diagnostic, not an accepted new-model state.
- An independently started, identity-chain checked P1 candidate path
  accepted nominal loads `0.0001` through `0.0069` (69 steps) with terminal
  owner mechanics, volume-adjoint history, original AT2 strong residual,
  force/moment, physical-damage range/healing and raw map gates, plus a
  separate same-model field-equation replay after each accepted scratch
  step. Scratch checkpoint pattern
  `/tmp/opencode/p1_mass_nonlocal_candidate_v1_stepNNN.npz`, schema
  `p1-mass-nonlocal-candidate-scratch-v1`, is *identity-only* and not a
  production controller/replay. At `0.0068`: max damage `0.84103`, tip
  damage `0.83297`, reaction `683.825`, zero displacement-envelope
  outliers. At `0.0069`: max `0.88150`, tip `0.87936`, reaction `681.522`,
  zero outliers; 133 sweeps were required.
- A nominal `0.0069 -> 0.007` target **rejected** after 144 Picard sweeps:
  damage/history/reaction/strong residual gates failed; raw map correction
  was `5.86e-8`, last trial tip damage `0.93439`. Explicit half cutback to
  `0.00695` after 144 sweeps still failed only the history correction gate
  (`e_H=2.095e-5`), but a **separately declared, same equations/gates**
  288-sweep ceiling strictly accepted it at sweep 150, reaction `678.993`.
  A subsequent `0.00695 -> 0.007` strict attempt under the same explicit
  extended ceiling passed in 172 sweeps, reaction `675.319`. Both sources
  were recomputed before trial; accepted snapshots are
  `/tmp/opencode/p1_mass_candidate_adaptive_s69_u6p95_v1.npz` and
  `/tmp/opencode/p1_mass_candidate_adaptive_u7p0_v1.npz`.
- Half increments thereafter strictly accepted `0.00705` in 199 sweeps
  (outliers 1865/1869; peak displacement/load `1.321`), `0.0071` in 250
  sweeps (same two outliers, peak `1.982`), and **`0.00715` in 185 sweeps
  with three outliers `1865,1869,1873` and tip amplification `16.379`**.
  Tip damage reaches `0.999906` while reaction remains `650.945`; the
  terminal current-history AT2 residual `4.35e-6` and free mechanics
  residual `4.91e-7` pass. The last field
  `/tmp/opencode/p1_mass_candidate_after_u7p0_v1_step003.npz` is a
  valid candidate model terminal algebraic root but fails the user's
  **physical** allowance; no third particle was removed from equations.
  Full target `0.0075` remains untested in this model, and its earlier
  `0.0062` reaction `662.724` is already far above same-load FEM `7.014`
  (the FEM comparison includes warnings near onset).
- Important acceptance-contract gap discovered while auditing the research
  code: the isolated `solve_kernel_damage_attempt` verifies healing on the
  **final complete** diagnostic, but unlike the original strict controller
  it does not short-circuit every intermediate sweep with healing versus
  immutable committed damage. All archived candidate *final* fields pass
  no-healing and independently recomputed equations; therefore describe the
  paths as **terminal-equation strict experimental continuations**, not
  production-certified original-controller strict paths. Fix this contract
  with an explicitly versioned new source identity and fresh zero-start
  evidence before a trusted reference or completion claim.
- Exact next action: quantify high-damage BA/P1 local virtual-mode weakness
  on the new accepted third-outlier state; evaluate a source-backed,
  parameter-justified correspondence stabilization/kinematic alternative
  that **annihilates affine strain corrections** without shifting crack
  energy/history through broad averaging. A new force term must come from
  one declared energy and be portable to fixed-shape PINN residuals. Do not
  infer impossibility of every strict remedy, so conditional particle
  deletion or forced damage is premature.

### Same-damage P1 versus FEM local-mode audit at the three-outlier state

- `/tmp/opencode/fracture2d_p1_mass_tip_mode_energy.py` checked the source
  code hashes, strict-terminal scratch identity and identical material/FEM
  node order before perturbing exactly nodes `1865,1869,1873` by their
  observed P1-versus-FEM displacement mismatch. In the *P1 candidate's own*
  physical damage field at `0.00715`, those nodes have damage
  `0.999761/0.999906/0.999212` and averaged center factors
  `8.64e-6/6.05e-6/3.79e-4`; P1 BA block minima collapse to
  `0.977/0.409/5.213` versus intact `1195/1201/3179`.
- The same 3-node virtual mode at **identical damage** costs BA elastic
  energy `0.00737` versus exact-CG1 FEM `0.03889` (ratio `0.190`);
  intact energies were BA `14.70` versus FEM `2331.06` (ratio `0.00631`).
  Unlike the earlier original-model high-damage two-point mismatch (BA/FEM
  ratio `0.000196`), this new three-point patch is **soft in FEM too when
  given the P1 candidate's nearly complete damage**. Thus this state does
  not prove a new BA-only coding/normalisation defect: weak displacement
  modes may also express a differently located/nucleated fracture patch.
  The measured pathwise FEM reaction and spatial crack advance remain far
  apart; three outliers still fail the user's explicit comparison contract.
- Exact next action: separate the changed *fracture morphology* from local
  linearized stiffness by quantifying along-notch crack extension and
  comparing FEM field histories. Do not assert a universal isolated-node
  stiffness bug from a single local block eigenvalue. The user-approved
  separate P1 integration trial is now a documented negative physical
  control, not a production/PINN reference.

### A source-energy variance correction also misses the tip mode

- A new *read-only mathematical precheck*, without changing either
  production model, decomposes the existing quadratic plane-strain BA
  center energy exactly as `psi(mean_eta(strain)) +
  mean_eta(psi(strain-mean_eta(strain)))`. The maximum direct/partition
  identity difference on the real cloud is about `4.66e-10` in the
  high-tip perturbation. For an affine field the second term vanishes,
  so selectively averaging damage only in the variance seemed capable of
  preserving the original crack onset while damping nonaffine drift.
- It is **not a sufficient frozen-state cure**: the observed old-model
  1869/1873 displacement-mismatch mode has only `2.48%` of its intact
  material BA energy in the nonaffine variance. The split with P1 weights
  raises damaged virtual energy only `0.006282 -> 0.009841` (factor 1.57);
  with the full horizon weights, only to `0.040023` (factor 6.37), still
  far below same-damage FEM `32.126`. At the original pre-event `0.0061`
  field, total elastic energy changes by only `0.0523%`/`0.1563%`
  respectively. This falsifies implementing the variance-only split as a
  drift fix; it does *not* establish that every energy-consistent
  correspondence stabilization is impossible.
- Mesh-band check of crack morphology on matched nodes: at P1 load
  `0.00715`, among material points with `x>=0.5` and `|y-0.5|<=0.03`, the
  furthest `d>0.5` point is `x=0.59086` and furthest `d>0.9` is
  `x=0.52115`. FEM's warning snapshot at `0.0059` reaches `x=0.57180`
  for `d>0.5`, while its post-peak `0.0062` reaches the far boundary `x=1`
  for `d>0.5` and `d>0.9`. The P1 root is therefore stage-lagged, not a
  complete crack-growth trajectory at a matching load.
- Next: an actually effective modified high-frequency BA kinematic/energy
  formulation needs a cited source or an independently verified derivation,
  plus tests on the exact old-model weak mode **before** another costly
  zero-start path. Do not claim the user-approved fallback of deletion or
  forced `d=1` is warranted by the limited negative candidate grid.

## 2026-09-26 — Original BA same-mesh horizon comparison; active resume point

- At the user's choice, returned to the **unchanged original BA/Amor/AT2**
  equations. In `fenicsenv`, `load_gmsh_particle_cloud` on the tagged
  `/tmp/opencode/fracture2d_stage1_mesh_conforming.msh` reproduced the
  authoritative `d048` source cloud exactly: all 3395 ordered material
  nodes, 6284 triangles, barycentric volumes, top/bottom and anchor, and
  all 72934 directed material edges. The source cloud SHA256 is
  `fa2ebf165f2e89ec88c63116346401b5039263e8e4c1a1917f6f13a3f4541267`;
  the tagged mesh SHA256 is
  `c40a3d3ebc4a139968655d0735c73da41b15d312ae0b87ee0849ffba8bebc791`.
  Generated separate nonproduction `delta=0.040` and `.032` graph NPZs
  under `/tmp/opencode/material_cloud_diagnostic_delta040_v1.npz` and
  `...delta032_v1.npz` (59024/30258 directed material edges), preserving
  all material arrays bitwise. No simultaneous FEniCS and JAX GPU work.
- With **only frozen original `delta=0.048` strictly converged** damage
  `d` from the `0.006243125` high-tip state, rebuilt distinct A2 operators
  at delta `.048/.040/.032`. Tip-1869 local tangent minimum increases
  `0.47249/0.99950/2.80348`; the *identical* two-point extra-displacement
  virtual energy increases `0.006282/0.019141/0.041106`, still below
  same-damage FEM `32.126`. Fresh cold JAX fixed-damage solves yield tip
  displacement/load `21.837/10.764/6.741` respectively. At `.032`, only
  particle 1869 exceeds the comparison envelope; this is cross-model
  frozen-field sensitivity, not a new-horizon coupled root or a proof of
  eventual absence of drift. The union collar thickness also changes
  with horizon, so interpret reactions within each separately bound BVP.
- A distinct source/graph/model-identity-bound `delta=0.032` **original
  equations** GPU path starts at exact zero, accepts all nominal increments
  through `u_bar=0.0058` (58 steps, same 144-sweep policy) with strict
  original stabilized gates and independent final-field BA/Amor/AT2 replay.
  At `0.0058`, tip damage `0.795852`, max damage `0.800851`, reaction
  `625.451`, displacement/load maximum `1.06158`, no outlier. The next
  `0.0059` nominal and `0.00585` half-step fail after 144 sweeps; the
  former fails seven coupled gates, the latter has fixed-point defect
  `0.00763` and damage residual `19.40`. The old source `delta=0.048`
  reaches a substantially later strict `0.0061` onset; the smaller horizon
  shifts the difficult onset toward FEM (`0.0057` peak) but does not by
  itself establish convergence.
- From the authentic `.032` `0.0058` parent, an explicitly **second**
  nominal cutback to `0.005825` passes in 45 sweeps; corrected
  cutback-count-2 scratch snapshot is
  `/tmp/opencode/original_BA_d032_strict_cutback_0p005825_v2.npz`, digest
  `2ff93446217639fb284a219b8c5033f83467600ec0b65c262b3baa60c1149592`.
  A prior `_v1` file achieved a similar field but incorrectly labeled the
  cutback count as one; preserve it as diagnostic history, not as the
  procedural prefix. At the corrected root tip damage is `0.83522`,
  reaction `625.154`, no outlier.
- From this root, targets `0.00585` (144 sweeps, map defect `0.00698`),
  `0.0058375` (144, `0.00186`) reject; cutback to `0.00583125` accepts
  strictly in 25 sweeps, tip damage `0.85247`, reaction `624.938`, no
  outlier. Its identity-bound scratch is
  `/tmp/opencode/original_BA_d032_adaptive_followup_u0p00583125_v1.npz`,
  digest `39fbfa54e3d4d4f08f408738b4025802dfd379289eeb5b49ca49bf1ee75e852f`.
  From that state, `0.0058375` fails after 144 sweeps (map `0.001864`),
  and its `0.005834375` half-step at the configured `du_min=3.125e-6`
  **also fails** after 144 sweeps (map `0.000731`, residual `1.884`).
  No failed state was committed. This branch likewise exhibits a stagger
  fixed-point/globalization barrier; a few strict smooth increments do
  not prove a full or drift-free load path.
- Exact resume point: after goal-mode reactivation, retain the last strict
  `.032` state at `0.00583125` and independently replay its parent, then
  examine *same-target self-seeded* recovery or an explicitly derived
  fracture-energy continuation **without changing original equations**.
  In parallel perform the already-built `.040` horizon comparison and
  design an independently traceable refined-material mesh study; the
  horizon sequence on one mesh alone is not a joint `h,delta` convergence
  study. No user-conditional node removal or forced `d=1` is justified.
- The OpenCode goal again reported `usageLimited` after 25 auto-continues;
  this is a harness execution limit, not evidence that the scientific goal
  is complete or impossible. Preserve all candidate labels and source
  fingerprints before resuming any expensive run.

## 2026-09-26 — Ordinary-mode resume: d032 self-seeded strict local root

- The user cancelled OpenCode goal tracking and instructed continuing in
  ordinary mode from the preserved work. An authentic diagnostic-graph
  model-bound `delta=0.032` state at `0.00583125` (digest `39fbfa54...`)
  passes independent parent-to-state BA/Amor/AT2/irreversibility replay.
- The original 144-sweep strict attempt to `0.005834375` fails its seven
  coupled gates, with final fixed-point error around `0.0007243` and
  strong damage residual `1.9057`. From only this attempt's own uncommitted
  damage iterate, the same-target, same-committed-history, fixed-
  `omega=0.5` self-restart passes **all original strict gates** on additional
  sweep `263` (total `407`). A prior *separate* 256-extra-sweep probe
  exhausted its budget with only history/fixed-point/coupled failures;
  neither failure was promoted. The accepted scratch state is
  `/tmp/opencode/original_BA_d032_self_restart_u5p834375_v2.npz`,
  bound to source/cloud/graph/model/parent/failed-seed digests;
  state digest
  `01e1319fee96c3dcdc09ea8ddfa44227800634ce23e5082d7f9fdd152bd6d032`.
  Its current-history owner AT2 strong residual is `1.1714e-5`, fixed-point
  correction `5.3766e-9`, free mechanical residual infinity norm
  `2.3952e-7`, reaction `621.1787`, tip damage `0.998424`, and only node
  1869 exceeds `1.08` (tip amplification `4.240`). All nodes remain in
  the original equations. This is a **locally recovered experimental
  strict state**, not a certified zero-start production archive or full
  loading curve.
- Exact next action: reconstruct the same `.032` owner/adapter, authenticate
  and independently replay the accepted scratch from its saved parent and
  underlying source, then attempt a held-out strictly gated load increment.
  Keep a larger-than-controller increment explicitly labeled as a separate
  branch-following experiment; do not silently change a production cutback
  or initial-guess policy. The `.040` and mesh-refinement comparisons remain
  open; the full `0.0075` original-model path remains unachieved.

### 2026-09-26 held-out delta032 successor

- The distinct bounded probe `/tmp/opencode/fracture2d_d032_restarted_successor.py`
  authenticates the previous parent, saved failure seed and root identity,
  independently rechecks the `0.005834375` owner BA/Amor/AT2 fields, then
  tries unchanged `_run_stabilized_attempt` at `0.0058375`. It **strictly
  passes in 17 Anderson sweeps**, no second restart or threshold widening;
  recomputed AT2 strong residual `2.1626e-6`, damage-map correction
  `1.8341e-9`, free-force infinity residual `2.1455e-7`, reaction
  `621.2508`, tip physical damage `0.998447`, and only particle 1869
  exceeds `1.08` (amplification `4.26174`). The source-bound scratch
  successor `/tmp/opencode/original_BA_d032_successor_from_own_restart_u5p8375_v1.npz`
  has state digest
  `514c8dc2997a48176df1984fceaaa813ecaf020b0edc3fb3f11b34942f0f6bc6`.
- This makes the d032 own-seeded method locally predictive at **one
  independently tested next load**. Do not infer full `0.0075` success,
  production schedule provenance, or reliable refutation of later
  3+ point drift. Next test an explicitly labeled finite larger load
  increment as a branch-following experiment; if it fails, use the already
  authenticated `0.0058375` state and report rollback.

### Ordinary-mode bounded branch extension and initial mesh-refinement preflight

- After the held-out strict `0.0058375` root, an explicitly **experimental**
  `du=2.5e-5` load-growth probe retains only parent/state/model/graph-
  paired, owner-equation-rechecked strict states. One attempt needed
  same-target own-failure restart at `0.0059125` (144+133 sweeps); retained
  prefix continues to `0.0060875` with at most particles 1869/1873
  above the declared displacement/load `1.08` envelope, and reaction
  rises to `626.989`. Distinct explicit cutbacks accept `0.0061`,
  `0.00610625`, and `0.006109375`. Last two-or-fewer-outlier strict
  physical-comparison root is
  `/tmp/opencode/original_BA_d032_branch_growth_probe_u0p006109375_v1.npz`,
  state digest
  `9c18f945b26a0741208099ee8b4a7d30bab3913abce8796506b5436889f6acbc`;
  it has tip damage `0.99913956`, reaction `627.224`, one outlier
  (1869, amplification `4.8825`). This scratch branch is not a vetted
  automatic increment-growth schedule or complete production archive.
- From `0.00610625` and from its separately strictly accepted minimum-step
  successor `0.006109375`, target `0.0061125` rejects the original
  144-sweep attempt plus 256 own-seeded fixed-omega sweeps; extending the
  extra budget to 512 reaches a root after 296/299 extra sweeps respectively
  but the root has **three** material outliers `1868,1869,1873` and tip
  displacement/load about `10.09`. The two roots have parent-dependent
  digests, reproduce original BA/Amor/AT2 field equations, and are
  **physically rejected** under the user's comparison rule. One early
  exploratory script created a load-keyed scratch file for a three-outlier
  root; do not regard that filename or its `accepted=true` *numerical*
  flag as a physically acceptable continuation prefix. Later probes
  explicitly print a separate physical-rejection record and retain the
  previous strict one-outlier state. This is further evidence that smaller
  horizon alone does not guarantee fewer than three point outliers or
  reach exact `0.0075`.
- To start the user-selected mesh-resolution branch, Gmsh was first called
  with `-setnumber lc_near 0.002` on the original `.geo`. Its actual mesh
  was **identical to the original** (3395 nodes, 6284 triangles and
  `119/20/119` notch upper/tip/lower line segments). This artifact is
  invalid as refinement evidence and was not used as such. An isolated
  `/tmp/opencode/fracture2d_tip_refined_lc002.geo` copied the material
  geometry/physical tags but explicitly changed only the default near
  cell size from `.004` to `.002`; its tagged mesh at
  `/tmp/opencode/fracture2d_tip_refined_lc002_default_v2.msh` has 3711
  material nodes, 6634 triangles, `238/38/238` notch segments, 51 bottom
  material points, and area `0.9762482516431508`. Mesh SHA256 is
  `ba04d7d83df9c57427a3cbf75bbce986267603b480ffe69bcdf4d460b1b26b1d`.
  Within tip radius `.024`, actual median nodal volume decreases from
  `1.359e-5` to `5.223e-6`, and median triangle longest edge from
  `0.00857` to `0.00602`; the exact tip-volume changes similarly.
  Source-bound graph arrays for the **new material cloud** are stored in
  `/tmp/opencode/material_cloud_tiprefined_lc002_delta032_v1.npz` and
  `...delta048_v1.npz` with 50128/104678 directed edges. No cross-mesh
  solution was silently interpolated or treated as accepted. This is
  *localized* refinement, not a complete fixed-horizon or local-limit
  convergence study.
- Exact resume point: validate the new-cloud BA shape/patch/tangent and
  grip geometry, then perform a bounded **fresh zero-start** original-model
  strict probe at `.032` on the refined material mesh. Compare load,
  reaction, crack morphology, and outlier scaling at equivalent physical
  stages; do not compare material indices across meshes. If strict
  convergence again stalls, record the nearest committed state and failure
  before another bounded own-seed recovery.

### 2026-09-26 — Refined material-mesh original-equation result

- Rebuilt original `.032` BA/Amor/AT2 operators on the 3711-particle
  locally refined material cloud (old cloud: 3395). Static checks pass:
  affine-gradient error `1.87e-14`, rigid-rotation energy `7.41e-27`,
  tangent bilinear symmetry relative defect `2.06e-16`. New-cloud SHA256
  `84ed0ba6747bc736af58054f8b14643c4b835f6956de3f9b784b882ab9074d48`;
  model digest `cf6969735feba9a3b8269d64090cebf079d78543ff504c79f77f01a2e6cb17be`.
- A new zero-start, strict original-stabilized path reaches `u_bar=0.0059`
  in 59 nominal increments. The coarse `.032` material mesh had failed
  its `0.0059` target after 144 sweeps; the refined material mesh **passes
  it after 47**. At this accepted load: exact-tip particle **2188** (old
  1869 index is invalid across meshes) has `d=0.792636`; damage max
  `0.824396`, reaction `630.1343`, and material displacement/load maximum
  `1.06157` with **no outliers**. Independent final-field BA/Amor/AT2
  reconstruction and range/healing checks pass. Scratch prefix state:
  `/tmp/opencode/original_BA_tiprefined_lc002_delta032_v1_step059.npz`,
  digest `83b9046d3546c91947a8a514adddc42f942517fff852e541a7b992a837f9a65f`.
- Next nominal `u_bar=0.0060` is **not** accepted. A first process timed
  out without a verdict; the bounded independent rerun completes all
  144 sweeps in `1381s` and fails seven coupled gates, with damage-map
  error `0.010597` and uncommitted trial-tip damage `0.931207`. The
  immutable committed state remains `0.0059`.
- The next proposed `0.00595` half-step command was **aborted by the user
  before producing output**. It has no known numerical outcome and no
  accepted snapshot. Resume from the authenticated `0.0059` root and
  evaluate this cutback once with explicit rollback; do not infer that
  the aborted run either passed or failed. This local two-mesh check
  establishes a one-step improvement, not convergent full fracture or a
  production nonlocal PINN reference.

### 2026-09-26 — Operator-first root-cause reassessment (read-only)

- The purpose of the direct solve is to independently test the **same
  nonlocal mechanics/damage/history residual** intended for the PINN.
  Completing a fragile displacement-controlled crack-growth curve is a
  *distinct* test, not a prerequisite for verifying linear, smooth,
  frozen-damage, and pre-instability nonlocal operators.
- Two distinct supported mechanisms must be kept separate. (1) The
  source-centered `V_i^m g(d_i)` BA energy loses stiffness for modes whose
  intact energy lies chiefly at the damaged notch-tip centers; full
  BA supports/affine patches and SPD fixed-damage solves do not imply
  FEM-like high-frequency stiffness. The frozen two-point virtual mode
  has BA/FEM energy `0.006282/32.126` at one shared damage field, but a
  later P1-mass high-damage field also softens the FEM mode
  (`0.00737/0.03889`); do not assert a universal BA-only bug. (2) The
  hybrid committed-maximum Amor history has positive displacement/damage
  feedback near crack growth; fixed-damage mechanics and fixed-history
  AT2 solves remain accurate while the full stagger map may not contract.
  The separately diagnosed `.048` monolithic failure has fixed-active
  feedback eigenvalue `0.995568`, a near-singular *coupled* mode, not
  evidence of an incorrect fixed-subproblem linear solve or a proven fold.
- A local FEM warning-mode post-peak curve is not an exact nonlocal root:
  its peak is around `0.0057` versus original `.048` nonlocal `0.0061`.
  Full-horizon averaging reached `0.0075` strictly but suppressed fracture;
  local P1 averaging eventually recovered multi-point drift. These controls
  show that absence of solver warnings or tip outliers alone is not
  validation of the underlying nonlocal operator.
- Revised next action: no additional same-control 144-sweep cutback until
  a scale-separated *read-only* operator study is specified. On retained
  authenticated clouds compare BA and local P1 virtual-energy Rayleigh
  quotients on smooth/near-tip/localized modes for the **same** displacement
  and damage fields, before/after tip localization, with mode
  normalization and source-energy decomposition. Pair this with the
  separately existing pointwise/volume-weighted fixed-point Jacobian
  evidence; assess both `(h,delta,l0)` sensitivity and physical crack
  length/reaction before selecting any altered formulation. For a PINN
  verification baseline start with intact, fixed-damage/fixed-history, and
  pre-localization strict states; use independent operator oracles and
  residual replays, then reserve post-peak path for a separate physical
  validation milestone. All proposals to change the energy/history/BC
  remain open formulation decisions, not approved implementation.
- Session diff: updated this active task and the dashboard to document the
  user-requested operator-first priority; appended the separate-validation
  decision in `docs/DECISIONS.md`. No solver, equation, configuration, test,
  or physical run was changed. Status remains incomplete. Blocker: no
  fixed-horizon `h` convergence plus justified small-horizon/local-limit
  or trustworthy strict post-peak FEM path has been established.

### 2026-09-26 — Revised user acceptance and smallest useful intervention

- User's actual objective is approximate *physical* agreement: FEM-like
  displacement/damage fields and reaction curve, especially controlled
  peak-force error; a qualified warning path is allowed, and actual
  particle deletion may be considered if it improves that response. This
  supersedes treating the old strict gates, zero outliers, or reaching
  `0.0075` by themselves as success.
- Existing `.048` old-model warning curve peaks near `641.34` against FEM
  `642.49` (about `0.18%` force discrepancy), at shifted loads `0.0061`
  versus `0.0057`. Its endpoint remains `69.15` versus FEM `3.98`
  (roughly `17.4x`), with damage healing up to `0.02915` and large tip
  history; deleting point values only from a *plot/comparison* cannot fix
  that reaction. Full-horizon damage averaging makes the whole curve too
  stiff and suppresses crack extension, so it is not an easy remedy.
- Revised exact next action: a **read-only** comparison of the already
  archived warning curve and fields, before/after coordinate-based
  diagnostic masking, without modifying reaction, plus an explicit
  per-load warning/residual/healing report. Request the user's numerical
  peak-error allowance and desired post-peak scope before accepting a
  candidate. If the reporting-only comparison still fails, design a
  *separate* fracture-to-void transition: its activation rule,
  material/union particle participation, BA support/rank, free-surface
  balance, damage/history/AT2 energy and irreversible update must be
  independently derived and applied identically in direct solver and PINN.
  Only a recomputed from-zero path can test its force curve. No activation
  threshold, changed PDE, or direct solve was selected or run this session.
- Session diff: revised the active goal/acceptance and next action in this
  task; updated `docs/PROGRESS.md` to match the user's physical priorities;
  appended the rationale to `docs/DECISIONS.md`. Task is incomplete;
  blocker is the unmeasured physically masked warning comparison and an
  unspecified acceptable peak-error bound. The prior operator diagnosis
  remains evidence, not a mandated full-suite research prerequisite.

### 2026-09-26 — Completed read-only warning/FEM field and curve audit

- Reproducible scratch-only analyzer:
  `/tmp/opencode/fracture2d_warning_fem_mask_audit.py`; comparison figure
  `/tmp/opencode/fracture2d_warning_fem_mask_audit_v1.png` SHA256
  `d4d1eafe92957e99359728d65811fc1cb1ff4b2c13576a785be3a9c7a24a9452`.
  Original archive hashes: nonlocal warning `9545bed758c19c07297d1a0afdce9093caab7a646655eef6c6b2a82cb23ac4fc`,
  local FEM `e9634f1f8fdb21116b50ad453d37115746c4a5a09f828e76245a9c59f6107d17`,
  material cloud `fa2ebf165f2e89ec88c63116346401b5039263e8e4c1a1917f6f13a3f4541267`.
  All **3395 coordinates are bitwise identical**; both archives have 76
  field snapshots with loads matching within `1e-12`, independently checked
  reaction summary pairing and 10 nonlocal warning statuses. The FEM has
  additional cutback response states (127 total), of which four warning
  states lie at the exact 76 common sampled loads. The analysis does not
  modify either reference or exclude material nodes from any equation.
- Full peak: nonlocal `641.3401673883` at `0.0061` is itself **warning-accepted**;
  FEM `642.4857294552` at `0.0057` is warning-free, peak magnitude error
  `-0.17830%`. At the *same load* `0.0061`, FEM reaction `317.55285` while
  nonlocal is `641.34017` (FEM onset snapshot warning). At `0.0062`, FEM
  reaction `7.01396` is warning-free, nonlocal `631.67237` is warning; at
  `0.0075` FEM `3.97718`, nonlocal `69.15426`. The 76-common-load curve
  RMS error divided by FEM peak is `0.21346` overall and `0.42470` for
  loads above the FEM peak. These use *raw, unchanged* reactions.
- Diagnostic outliers use the previously declared `|u_i|/u_bar>1.08` and
  material barycentric volumes, with no choice based on FEM mismatch:
  counts `2` at `0.0062/0.0063`, `3` at `0.0064`, `5` at `0.0065`, `10` at
  `0.0066`, `29` at `0.0067`, `28` at `0.0068`--`0.0071`, and `31` at
  `0.0075`. Initial two are at coordinates `(0.49970452018,0.50375442713)`
  and `(0.5,0.5)`; by the final load many anomalous points form a line
  from the notch tip to the right free boundary. Thus this is not merely
  a persistent two-point display defect. Excluding up to two from the
  final comparison still leaves normalized pointwise displacement mismatch
  `7.235`; excluding all 31 reduces the full-field normalized displacement
  RMS `0.06784 -> 0.03595` and damage RMS `0.00836 -> 0.00697`, but retains
  `99.163%` of material volume and **does not change** the `69.15426`
  reaction or warning/history path.
- At load `0.0062` the two-point mask reduces same-load normalized
  displacement RMS only `0.24929 -> 0.24536` because crack propagation
  is *shifted in load*. A diagnostic crack front defined by the furthest
  material point with `d>=0.5`, `x>=0.5`, `|y-.5|<=0.048` is `0.541889`
  nonlocal at `0.0062` versus FEM `0.541889` at `0.0058` (FEM warning).
  On these phase-aligned snapshots, compare the fields normalized by
  their respective applied loads: masked displacement RMS `0.01112`,
  damage RMS `0.00586`, reactions `631.672/614.671` (~`2.77%` difference).
  Once both front indicators reach `x=1` they cease to distinguish crack
  opening, so an equal-front comparison there does not erase the remaining
  roughly tenfold reaction difference (`71.126` vs `7.014` at their first
  full-front states).
- Nonlocal maximum damage healing between adjacent retained states is
  `0.0291481`, versus at most `1.6715e-5` over the FEM response states
  at the common loads. The nonlocal `0.0067` warning has post-history
  strong damage residual `91.9646` on scale `4.33598e6`; its raw map
  defect is `0.04580`, so warning does not mean a uniformly mild defect.
  Later `0.0075` itself passes old terminal gates but depends on the
  non-strict/healing earlier committed history; independent replay still
  correctly reports `field_equations_validated=false` for the full path.
- **Decision for this evidence only:** reporting-only exclusion does not
  satisfy the user's *whole-curve* goal, despite encouraging phase-aligned
  early-crack fields and an accurately sized peak. The next physically
  informative experiment must change and **recompute** a fracture-
  propagation mechanism near onset, rather than hiding terminal point
  values. Exact blocker: deleting particles has multiple materially
  different possible equations (BA center/target/support, free surface,
  material damage/history/AT2, intact grip treatment) with no selected
  derivation, and no numerical acceptance bound for peak force/post-peak
  curve was given. Obtain that formulation choice before implementing
  a deleted-point direct/PINN operator. No new physical solve was run.
- Session diff: active task audit and dashboard/decision summary updated;
  source archives unmodified. Audit verified by re-running its hash-bound
  read-only analyzer and visually inspecting the plotted response. No
  source-code behavior or numerical settings changed.

### 2026-09-27 — User correction: original equations and adaptive stagger only

- The user explicitly rejects investigating node-deletion models. The old
  read-only audit is still useful as a physical response measurement, but
  its suggested changed-model next action is superseded. Fix BA kinematics,
  material and union graph, source damage degradation, Amor committed-history
  maximum, AT2 operator, and boundary conditions. Do not change the kernel,
  horizon, material data, damage law, or physical equations in an adaptive
  solver experiment. Peak force and rough crack onset already meet the user's
  goal; evaluate chiefly end reaction, post-peak curve and drift count.
- Existing strict solved original `.048` `0.0061 -> 0.0062` candidate jumps
  from notch-tip `d=0.818`, displacement/load `0.724` to `d=0.999917`,
  displacement/load `16.924`, but reaction still `631` vs FEM `7.01` at
  the same latter load. The isolated one-time event controller discards that
  candidate and strictly accepts `0.00611` with no tip drift; strict
  continuation then stalls at/near `0.00612`. Therefore an early cutback
  alone cannot resolve the fixed-point stall or full late-stage response.
  Conversely old full warning continuation has 10 warning accepted steps
  and a large `0.02915` healing event at `0.0068`, so simply accepting every
  finite 144-sweep candidate is not an adequate numerical policy.
- Proposed **numerical** study, without selecting new numeric thresholds:
  (1) retain exact old fixed-subproblem and committed-history equations and
  trial-only rollback, and examine load-step behavior around the *already
  authenticated* `.048` strict peak/prefix; (2) use observed physical
  per-step crack advance, reaction change, damage jump and drift counts to
  diagnose accepted-but-unphysical large steps, distinct from algebraic
  nonconvergence; (3) on failure, compare bounded same-target own-iterate
  restart, explicit cutback, and later explicit load-increment growth as
  separately labeled methods, since more shrinking alone has already
  stalled; (4) allow only **separately audited, limited warning** candidates
  with completed finite fixed solves, positive/ranged damage, preserved
  committed history, checked healing and strong residual *on an absolute
  as well as scaled basis*, unchanged original strict-pass flags, and
  independently checked following increments. A previously tested `1%`
  approximate candidate failed its immediate successor; it does not
  authorize a new warning limit. Every warning must be assessed by its
  subsequent response and must not silently seed an apparently strict
  terminal curve whose inherited history is wrong.
- Distinguish two outcomes: a path-dependent warning/step policy that
  measurably improves final reaction **and** drift under the same equations,
  versus multiple independently re-solved admissible histories that retain
  the ~`69` final-force plateau. The second is evidence that the remaining
  FEM/nonlocal discrepancy may be intrinsic to the finite-horizon model,
  and should be reported rather than retuned through the acceptance gates.
  No such new comparison has been performed; no achievable endpoint is
  claimed. Exact next execution point: design the bounded, immutable
  unchanged-equation candidate/rollback and warning audit around the
  authenticated original `.048` strict prefix, with an explicit recovery
  budget and physically meaningful acceptance targets, before launching a
  costly GPU attempt. Do not replay the previous identical control suite.
- Session diff: memory-only change to remove the deleted-point direction and
  record numerical-only acceptance boundaries. No source equation, numerical
  setting, archive, or physical solve has changed. Task remains active.

### 2026-09-27 — Existing-warning step chronology before selecting controls

- The read-only hash-bound diagnostic script
  `/tmp/opencode/fracture2d_original_adaptive_gate_audit.py` evaluated the
  original `.048` 76-state warning archive; it performs no solver call and
  leaves all materials and equations unchanged. Relative to the previous
  retained load, the **first** warning `0.0059 -> 0.0060` still has zero
  displacement outliers, no healing, max damage increment `0.05730`,
  original-map correction `1.0719e-7`, and post-history strong AT2
  residual `7.33e-5`. At `0.0061` (warning peak) these are `0`, `0`,
  `0.07453`, `3.4475e-6`, and `0.003706`; at `0.0062` the damage jump
  reaches `0.18206`, two outliers appear, the history maximum changes
  `432.11 -> 1.19e6`, and the map correction grows to `1.748e-4`.
- `0.0064`--`0.0067` warning steps have max per-step damage changes
  `0.571/0.685/0.694/0.927` and outlier counts `3/5/10/29`, while
  reaction falls `521 -> 419 -> 313 -> 72.86`; at `0.0067` the map defect
  is `0.04580`, absolute AT2 strong residual `91.965`. The apparently
  map-exact `0.0068` warning state has `0.029148` material damage healing
  relative to the `0.0067` warning predecessor, even though its fixed
  subsolves, map, and terminal mechanics pass: it re-solves against a
  different committed, already non-strict damage/history state. Subsequent
  steps can pass all terminal gates and still inherit that path.
- This falsifies two proposed shortcuts: one-time damage-event cutback
  alone (prior strict `0.00611/0.006115` then stall) and unconditional
  warning acceptance at the iteration ceiling. Neither would establish
  a better endpoint under the original model. In the next bounded method
  study, distinguish three candidate outcomes before commit: strictly
  converged but physically abrupt, incomplete fixed-point but otherwise
  admissible, and hard invalid (healing, nonfinite, failed fixed subsolve).
  Report raw absolute as well as scale-normalized AT2 defects because the
  history becomes `O(1e6)`. Keep unchanged original strict classification.
- The concrete **research protocol** (not a validated threshold/success):
  start from independently authenticated original `.048` strict `0.0061`
  state; use the existing original-equation per-target stagger solver,
  never carry uncommitted trial history into a retry. Retain the already
  observed `0.0062` abrupt-root rejection and `0.00611/0.006115` strict
  cutbacks as controls rather than rerunning them. At the first unseen
  post-event target, compare bounded own-iterate same-target continuation
  against one *separately declared* physically constrained warning attempt;
  require an independently solved successor and retain last committed
  prefix on failure. An adaptive *growth* probe after a committed smooth
  step must be explicitly recorded, since a branch jump might be reachable
  when an arbitrarily small load increment stalls; no jump may be accepted
  merely from its load-step size. The end-force/outlier goal requires
  comparison with FEM after a complete path, not extrapolation from one
  recovered step. Original numerical warning limits, growth schedule,
  cost ceiling and physical diagnostic envelopes have not yet been selected.
- Status: numerical strategy specified at the test-contract level, no
  code or new physical solve. Next action is to select explicit bounded
  controls and build an isolated provenance-bound multi-outcome controller;
  evaluate one held-out physical target before any full-run expense.

### 2026-09-27 — Bounded original-equation same-target own-iterate probe

- Ran `/tmp/opencode/fracture2d_original_d048_ownseed_stall_probe.py`
  only at original `.048` load `0.006118125000000033` from independently
  BA/Amor/AT2-replayed strict `0.00611/0.006115` accepted parents. Parent
  `0.006115` digest:
  `8bb66828a2643c255cbb5e813805528b60dbe01a9caccd3f6374c8bc7536132b`;
  parent replay strong AT2 `8.7924e-7`, damage-map correction `7.6587e-10`,
  free mechanics residual `4.7095e-8`, reaction `641.38098`. Source strict
  archive, cloud, event-prefix NPZ SHA256
  `59ce6c9b1c6c65d58ab3717bc12d12a0bec35fe646d98580e3e27e18629cf5de`,
  and 144-sweep uncommitted own-iterate NPZ SHA256
  `14b77d3e212e801a9d4117747a03ff57f40b05bf0d17f715489dcc6ad0a38f1d`
  were authenticated. The saved 144-sweep seed is numerically consistent
  but does not contain the full old failure-attempt object; its identity
  and last iterate were used only for this separately labeled scratch test.
- Holding the **same committed history and target**, original BA/Amor/AT2
  fixed subproblems were iterated for a new explicit 128-sweep `omega=0.5`
  budget after that own seed. The damage-map defect was nonmonotone:
  `9.3723e-3` at additional sweep 32, `2.6133e-3` at 64,
  `4.1419e-4` at 96, `7.12444e-5` at 128. All seven original coupled
  gates still fail; the final AT2 strong residual is `0.0607904` and
  the uncommitted damage max is `0.99988413`. Attempt elapsed `407.4s`.
  There is **no accepted successor, no final-force/point-count claim for
  this failed field, and no production artifact**; the strict `0.006115`
  parent remains last committed in this branch. An old exploratory script
  import had become stale; the scratch runner was corrected to authenticate
  its saved prefix directly before any physical iteration, with no solver
  change. Do not rerun the same 128-sweep experiment unchanged.
- New diagnosis: the own-iterate continuation heads toward high damage
  while decreasing a coupled map defect, but cannot yet establish a
  physically improved root. A long same-target restart is not selected as
  an adaptive-controller default or as evidence that the original path
  will end with lower reaction/fewer drift points. The next meaningful
  small experiment must *compare distinct load-step trajectories at the
  same original equations*, including a branch-aware accepted-step
  physical check and separately audited warning policy; it must stop
  rather than accept large healing or inherit an unqualified warning
  history. Full endpoint `0.0075` remains unachieved under that policy.
- Session diff: scratch-only source-aware bounded probe, with no code or
  numerical setting change inside the repository; updated active-task
  status and dashboard. The task remains active and incomplete.

### 2026-09-27 — Original-operator adaptive *warning* study (not strict completion)

- User confirmed verbatim direction: **original-equation stagger + adaptive
  load steps + auditable relaxed acceptance; judge by the post-peak curve
  and fields**. Historical strict gates are reported, not the only
  acceptance target. Aborted refined `.032` half-step `0.00595` from the
  prior session remains without a verdict and is not the current path.
- A source-SHA-bound read-only screen of the original `.048` 76-state
  warning curve (`/tmp/opencode/fracture2d_relaxed_warning_window.py`)
  distinguishes old mild steps from severe ones using *explicit,
  provisional* absolute map/residual, healing, and `1.08` point-count
  envelopes. For map `0.001`, AT2 strong `1`, healing `2e-5`, and at most
  two diagnosed points, the **first** historical rejection is `0.0063`.
  A less selective map `0.01`, AT2 `10`, and at most five points first
  rejects `0.0065`. These are retrospective discrimination checks,
  **not** demonstrated acceptable solution tolerances: the later fields
  would change immediately if a rejected step were rolled back.
- Isolated experimental module
  `examples/fracture2d/nonlocal_engineering_grip_relaxed_adaptive.py`
  performs the *same* fixed-omega stabilized-warning owner attempts on
  the original model. It rechecks complete candidate field pairing,
  exact constrained values, and Amor history against the committed parent;
  keeps finite/mechanics/force/moment/fixed-history damage/range/history
  monotonicity hard; labels original strict/warning status separately;
  checks explicit absolute strong AT2/map, bounded healing and material
  outlier count; and pre-commit rejects/cuts back on study failure.
  Only an original strict-plus-study accepted step can grow the next
  increment up to configured `du_initial`; warnings retain their accepted
  increment. Attempt and accepted-step budgets are required. The module
  has no production archive or authority to validate a full PINN path.
  A genuine complete manufactured warning candidate tests pre-commit
  rollback and a permissive separate warning decision; two real fixed
  subproblem attempts from the same manufactured zero state verify
  an actual halved-load retry. All three focused tests pass.
- Bounded physical scratch probes from the **original old warning**
  `0.0062` parent: direct `0.00625` (144 sweeps) fails the `0.001/1`
  study envelope with map/strong `0.00204/1.8977`; its `0.006225`
  cutback passes *that study envelope only* at `0.000114/0.07561`,
  exactly two >`1.08` nodes, reaction `629.154`, warning status.
  Independent owner recomputation of that provisional candidate confirms
  exact parent-conditioned Amor history, mechanics force `3.83e-8`, AT2
  strong `0.07561`, and map about `0.0001124`. Its held-out successor
  to `0.00625` **rejects** at map/strong `0.003104/2.9641`, two outliers.
  The provisional `0.006225` scratch is SHA256
  `c96e673b16824624b76a55ed3aec0d12dc20874d2fc477f6456e77f6ffb99c8b`;
  it is not a verified full continuation.
- From the old **warning** `0.0061` parent, smaller first targets
  `0.00615/0.006125` pass the same empirical two-outlier envelope in
  144 sweeps each, yet already have max displacement/load
  `15.53/14.93` and maximum histories `1.006e6/0.927e6`. Meeting only
  a *count*-of-outliers cap can still carry the localized large-displacement
  and history feedback, so these states remain exploratory, not a solved
  physically improved peak/post-peak curve.
- More informative physical window uses the authenticated **strict** `.048`
  `0.0061` parent (reaction `641.33849`, field replay AT2 `7.49e-7`, map
  `6.10e-10`, force `3.53e-8`) and the new relaxed controller at the same
  original model. With study max **two** outliers, three warning states
  `0.00611`, `0.00612`, `0.00613` can be provisionally advanced, but
  outliers `1869/1873` appear by `0.00612`. With study max **zero**
  outliers, `0.00611` passes as warning (reaction `641.3872`, no outliers);
  `0.00612` has two outliers despite map/strong `0.0002146/0.18162`
  and is rejected **before commit**; a `0.006115` cutback accepts as
  warning (reaction `641.3816`, damage max `0.84107`, history max
  `511.78`, no outliers). The accepted prefix is stored only as a
  separately labeled identity-only scratch, SHA256
  `8818427b36691318faaae18b524427ddcc6b040034b9afc2175fe59e64693104`,
  with `field_equations_validated=false`. It is not strict or complete.
- An independently reconstructed BA/Amor/AT2/history replay from that
  provisional `0.006115` prefix gives free-force `3.46e-8`, current-H
  AT2 strong `0.00499`, map `3.60e-6`; its held-out step to
  `0.0061175` rejects under the **study** criteria (two outliers,
  map/strong `0.001357/1.0891`). Half-increment `0.00611625`
  accepts as **warning** (reaction `641.3678`, dmax `0.84645`,
  Hmax `535.85`, **zero** diagnosed outliers; map/strong
  `1.235e-5/0.01756`). The *next* `0.0061175` again rejects on
  two outliers, even though map/strong `0.000363/0.305` satisfy the
  provisional numerical envelope. A new identity-only accepted-prefix
  scratch was saved, SHA256
  `19cc132b5904e227fbc1bd318bb3068b3704ec99de176d52e8b27b42ccaa171c`;
  it is **not** a verified production archive, finished loading, or
  evidence of lower endpoint reaction.
- Interpretation: relaxation plus pre-commit physical rejection does
  permit small, smooth original-equation *warning* steps beyond the
  earlier `0.006115` strict stall; it does not yet traverse the near-fold
  onset without outliers or predict the final load. Do not continue with
  blind halving, adjust threshold merely to label the known high-damage
  root accepted, or rerun identical expensive branches. Next distinct
  question: compare a specifically proposed load-growth/branch-switch
  schedule from the **audited warning prefix**, keeping both original
  residuals and the physically measured curve, before attempting an
  entire fresh-zero path. Confirm whether the chosen physical outlier
  allowance is zero, two, or a range for the complete crack branch;
  previous exploratory windows demonstrate that this choice changes
  branch reachability, not just plotting.
- Session diff: added only the isolated relaxed numerical controller,
  focused tests and its research-only README API note under the repo;
  scratch probes and two provisional prefix artifacts live under
  `/tmp/opencode`. Original solver, equations, historical archives and
  previous strict flags remain unchanged. Active task remains incomplete.

### 2026-09-27 — Explicit load-growth and recovery diagnosis from provisional warning prefix

- User authorized a bounded cross-transition **numerical** load-growth
  experiment on the original BA/Amor/AT2 model, using the already
  parent/owner-replayed `0.00611625` zero-outlier warning prefix. Research
  gates stay explicit (`damage-map <=0.001`, absolute strong AT2 `<=1`,
  healing `<=2e-5`, amplification `>1.08` at no more than two material
  points); these are candidate-screening values, not validated universal
  convergence conditions. Every call still solves the full original fixed
  mechanics/Amor/AT2 stagger map with committed-parent history.
- First growth schedule `du=2.5e-6,5e-6,1e-5` provisionally accepts
  `u_bar=0.00611875/0.00612375/0.00613375`, each **old warning** status
  with exactly nodes `1869/1873` over the diagnostic displacement envelope.
  At `0.00613375` reaction `633.0051`, tip history maximum `950088`,
  map/strong `5.74e-7/4.95e-4`, front indicator
  `x=0.541889`. The FEM `0.0058` *warning* snapshot at the identical
  front indicator has reaction `614.671`; comparing displacement fields
  per respective imposed load and excluding the two *reporting-only*
  anomalous coordinates gives RMS `0.01110`, damage RMS `0.00645`.
  Exclusive source/model/parent-bound provisional snapshots a01/a02/a03
  have SHA256 `a53f828b72f9e45b4946b9e0d0acb95ac72012aeb2e5a1971b3daf0ce06f41c4`,
  `8345b72eacfd98c3a07d0283220d697ecbd0c077119a4af2c0c9e60a2beb3985`,
  `ecc1c18f6ea99fecffd0d95e6d6b352b09f63ac14b1499329b74292bde2fe004`.
  They retain `field_equations_validated=false` and cannot replace strict
  fields in the historical source archive.
- The next bounded growth from the owner-replayed a03 prefix accepts
  `0.00615375` (`du=2e-5`, reaction `632.858`) and `0.00619375`
  (`du=4e-5`, reaction `631.796`); a larger `0.00627375` from the latter
  is **rejected** at map/strong `0.005795/5.4236`. Cutback `du=4e-5`
  accepts provisional `0.00623375` (reaction `628.188`, front
  `x=0.554222`, map/strong `0.00022475/0.15623`, two outliers).
  Scratch b01/b02/b04 SHA256 `6768e9a3ba1ece044057dde72d09e892c912bbd57ca45ddb1e5acb7e48234926`,
  `734682a431ba87038ae41ab70eccfd729004ad7da044e265e71d5bc8c13909dd`,
  `1b3980b39d6ef00b4e0c9498e332d3db78f8816fdd6575b58b1bfc4a69eae5f4`.
- Source-chain/BA/Amor/AT2 rechecks from b04 reject `0.00627375`
  (`du=4e-5`), `0.00625375` (`du=2e-5`), then accept one cutback
  `0.00624375` (`du=1e-5`, reaction `623.793`, front `x=0.557641`,
  map/strong `0.000747/0.6392`, same two outliers); saved c03 SHA256
  `0f10a366d4e2690feab6c2c61548c48472f71f3a2c8b77d5f272df58d5aaa7ba`.
  This is **less** advanced and has a higher reaction than a distinct
  earlier original-equation strictly solved scratch state at the nearly
  same `0.006243125`: front `x=0.580174`, reaction `592.787`, two
  outliers; the two masked branches differ by normalized displacement RMS
  `0.01648` and damage RMS `0.01366`. Early ease of accepting warnings
  has not selected the best physical crack-growth branch.
- From c03, explicit target `0.006299375` (matched to the earlier strict
  two-outlier root with reaction `577.873`) rejects at map/strong
  `0.001924/1.445`; two further cutbacks `0.0062715625` and
  `0.00625765625` also reject, with no healing and still only two
  outliers. The committed *study* prefix remains c03. These 144-sweep
  attempts have no accepted fields and cannot establish reaction or front
  comparisons merely from their residuals.
- To avoid mistaking a scalar warning envelope for the user's physical
  target, an **uncommitted** original-equation trial at `0.00625` from
  c03 recorded full fields and a separate old warning status. It fails
  only the experimental map/strong ceilings (`0.003903/3.626` on AT2
  constituent scale `3.6624e6`), but its mechanical free-force error is
  `4.46e-8`, H exactly matches committed-parent Amor, damage has no
  healing, exactly two outliers remain, and reaction drops to `601.724`.
  At the FEM `0.0059` warning crack-front stage the reaction is `589.764`,
  masked normalized displacement RMS `0.01192`, damage RMS `0.00578`.
  This *rejected* trial is saved SHA256
  `6208f899ad28c973f50361c10fbaf9d298ea850bdc4417ef479fd2c5db3dc508`
  with `committed=false`; it is NOT a historical accepted state.
- Two deliberate **counterfactual** original-equation successors from
  that rejected seed at `0.006275` and `0.0063` each recover the
  *unchanged* study map/absolute-AT2 envelope (`0.000431/0.325` and
  `0.000464/0.329`), while reactions descend `588.187 -> 581.655`,
  histories remain owner-paired, no healing occurs, and there are
  still exactly two outliers. At `0.0063`, front `x=0.590861` vs FEM
  `0.0060` warning-stage `x=0.599902`; reaction `581.655` vs
  `563.115`, masked normalized displacement RMS `0.01398` and damage
  RMS `0.00570`. Their exclusive artifacts SHA256
  `02884bdd4e299cef51776207efcdef9bb49d54191abe46ee21835ace399cb354`
  and `08977fe697313fd0816d98d9c920287bc1a93531b27d9b6a2138c27a38f30e3d`.
  This demonstrates a one-step transient defect can recover on a
  stage-useful branch, **not** approval to alter the old scalar gate or
  claim a validated continuation: the intervening parent was rejected.
- The next `0.006325` counterfactual step has reaction `562.335`
  versus FEM at near crack stage `563.115`, masked normalized displacement
  RMS `0.01475`, front `x=0.610439`, **but a third** anomalous material
  point `1864` joins `1869/1873` and map/strong rise to
  `0.004576/4.288`. The original two-outlier study rejects and rolls
  back to the last source-audited two-point candidate at `0.0063`.
  A separately labeled three-point research rerun saved the *same*
  candidate as **uncommitted**, SHA256
  `2567df8168198553d97aa37ccd4646d2153b330ac5d5481cf37678d94d7ba399`.
  Its `0.00635` successor has four outliers (new point 2001), map/strong
  `0.005251/4.633`, reaction `515.302`, and jumps front by `0.040` to
  `x=0.650419`, versus stage-near FEM `x=0.599902`; masked displacement
  RMS worsens to `0.03030`. A single **half-increment** counterfactual
  from the three-point seed to `0.0063375` still has the same four
  outliers, near-identical front `x=0.650419` and map/strong
  `0.004631/4.288`, so the study does not rescue that transition by
  simply halving the load increment. Neither successor was committed.
- **Current judgment/next step:** the original equations show an
  encouraging phase-aligned force/field counterfactual around
  `0.00625--0.006325`, but the tested one/two-step transient-defect
  bridge **fails its held-out post-`0.006325` validation** with growing
  outlier count and field mismatch. Do not retroactively relabel the
  rejected `0.00625` or `0.006325` states as accepted because one
  reaction happens to match FEM. The last internally screened two-point
  warning prefix is c03 at `0.00624375`, which itself is provisional and
  not a production archive; separately the original strict scratch
  branch reaches `0.006299375` at two outliers with reaction `577.873`
  but does not yet traverse the full loading path. Before another costly
  probe, formulate a **distinct** crack-front-aware acceptance/load-step
  choice with explicit physical error measures (rather than changing
  the BA operator), and require independent held-out propagation through
  the fourth-point transition. Complete `0.0075` response and its
  terminal reaction remain unverified. All paths retained every material
  particle and original graph/energy/history/AT2 law.
- Session diff: no repository solver implementation changed. New scratch
  source-bound read-only/physical growth experiments and uniquely named
  *provisional or rejected* NPZ evidence under `/tmp/opencode`; active
  task, decision log, and project dashboard updated. This task remains
  incomplete with the exact resume paths and source identities above.

### 2026-09-27 — Residual-plateau acceptance observation on a real branch

- User proposed two alternatives to per-step scalar thresholds: start from
  the current best candidate under an explicitly **loose** threshold, or
  accept when residuals stop changing across a sustained sweep window.
  Added read-only `ResidualPlateauStudyControls` /
  `assess_residual_plateau` in
  `examples/fracture2d/nonlocal_engineering_grip_relaxed_adaptive.py`.
  The statistic measures the **full-window spread** of the raw damage-map
  defect, absolute AT2 strong residual, history correction, and reaction
  correction on real complete owner sweeps. A flat tail passes; continuing
  descent and a 2-cycle fail. It never commits a state, never relabels the
  original strict/warning status, and is not a convergence certificate.
- Measured on the real counterfactual branch: from the uncommitted `0.0063`
  parent to `0.0063125`, the last-12-sweep tail is **not** stagnant:
  damage-map `0.00124 -> 0.00203` (relative span 0.389), absolute AT2
  `2.92 -> 12.64` (0.769), history correction `~90 -> 914` (0.902),
  reaction-correction span 0.072. Physically the candidate is promising:
  two `>1.08` points, no healing, front `x=0.599902` equal to the FEM
  `0.0060` warning stage, reaction `573.595` vs `563.115`, masked
  normalized displacement RMS `0.01348`, damage RMS `0.00567`, owner free
  force `3.12e-8`. Saved **uncommitted** SHA256
  `6037f3d26a185b950d434800f4e5d6ecb423b10a9b9131be63aa73be4e10daa1`.
- Its held-out successor `0.0063375` has FOUR outliers
  (`1864,1869,1873,2001`), map `0.00504`, absolute AT2 `4.49`, masked
  displacement RMS `0.02243`, front `x=0.639883`; its own 12-window map
  relative span `0.0552` also misses the declared 5% observation limit.
  Saved **uncommitted** SHA256
  `39ea811a80fc37ed4e2602e8ef21024255346fe9b9a26c8004087f7c9191082e`.
- Focused regressions 5/5 pass: genuine warning-trial rollback, real
  half-load cutback, invalid-limit rejection, plateau flat/descent/cycle
  discrimination, and owner-status retention on a real 12-complete-sweep
  warning attempt. No production archive, workflow dispatch, or
  field-equation validation is claimed.
- Decision point left explicit: (a) declare a very loose absolute
  acceptance (e.g. accept `0.0063125` at map `~0.002`, absolute AT2
  `~12.6`) and continue from it, knowing its tested successor already
  degrades, or (b) extend the same load target beyond 144 sweeps and
  accept only on a genuinely measured plateau. Neither option is a
  validated full `0.0075` path.

### 2026-09-27 — Adoption of the best complete warning solution, plotting rule, cleanup

- User decision: pause blind direct-solver attempts. Adopt the best scheme
  and produce the closest-to-FEM complete solution in non-strict warning
  mode; keep drift points inside the solution but compute plotted field
  minima/maxima from the non-drift points; then remove unused code and
  retain one concise direct solver.
- Adopted scheme (research-record best complete run):
  original BA/Amor/AT2 equations on the 3395-node `delta=0.048` cloud,
  fixed damped-Picard `omega=0.5`, `du=1e-4`, `du_min=6.25e-6`, at most
  144 sweeps per attempt, honest stabilized-warning acceptance, load path
  `0 -> 0.0075`. Solution archive:
  `/root/autodl-tmp/data/fracture2d/engineering_grip/engineering_grip_damped_warning_o5_u7p5e3_dumin6p25e6_s144.npz`
  SHA256 `9545bed758c19c07297d1a0afdce9093caab7a646655eef6c6b2a82cb23ac4fc`;
  76 states (65 strict + 10 warning + initial), peak reaction `641.34017`
  at `0.0061`, terminal reaction `69.15426` at `0.0075`. Independently
  verified drift counts: `0` through `0.0061`, `2` at `0.0062`, `3/5/10`
  at `0.0064/0.0065/0.0066`, `28-31` from `0.0067` on (declared envelope
  `|u|/u_bar > 1.08`). This remains a diagnostic warning path, not a
  strict or field-equation-validated production reference.
- New single-entry solver `examples/fracture2d/direct_nonlocal_solver.py`:
  `run` (fresh zero-start warning continuation with the adopted scheme,
  compact solution NPZ), `verify` (state/warning counts, peak/terminal
  reaction, per-load drift statistics, drift-excluded final ranges), and
  `plot` (ux/uy/|u|/d panels whose color minima/maxima are computed over
  non-drift points; drift points are still drawn as red circles).
  Focused helper tests `4/4`
  (`tests/test_fracture2d_direct_nonlocal_solver.py`) and the full
  `tests/test_fracture2d_nonlocal_engineering_grip_stabilized.py` battery
  `82/82` pass after cleanup; `git diff --check` clean.
- Final figures for the adopted solution:
  `/root/autodl-tmp/data/fracture2d/engineering_grip/direct_nonlocal_final/fields_drift_excluded_ranges.png`
  and `response_curve.png` (with warning states marked), plus
  `plot_summary.json`. At `0.0075` the drift-excluded damage range becomes
  `[6.57e-6, 0.98954]` and `|u|` range `[3.26e-7, 7.726e-3]`, versus the
  drift-inclusive extreme `|u| = 0.142`; drift points no longer set the
  visualization scale.
- Cleanup performed (removed from the repository):
  `nonlocal_engineering_grip_adaptive.py`,
  `nonlocal_engineering_grip_adaptive_checkpoint.py`,
  `nonlocal_engineering_grip_kernel_damage.py`,
  `nonlocal_engineering_grip_p1_mass_damage.py`,
  `nonlocal_engineering_grip_relaxed_adaptive.py`,
  `nonlocal_engineering_grip_stagger_energy.py`,
  `nonlocal_engineering_grip_stagger_restart.py`,
  `nonlocal_engineering_grip_tolerance_study.py`,
  `tests/test_fracture2d_nonlocal_engineering_grip_kernel_damage.py`,
  and the 18 experimental tests plus their imports in
  `tests/test_fracture2d_nonlocal_engineering_grip_stabilized.py`; the
  README sections documenting them were removed. Retained: the original
  model/solver modules (`nonlocal_engineering_grip*`, `nonlocal_a2_coupled`,
  `nonlocal_geometry`, `nonlocal_grip_diagnostic`, the JAX backend), the
  durable workflow, and the paused monolithic family (its engine is still
  imported by tracked workflow/archive code and was left untouched).
- Status: direct-solver exploration is paused at the user's request with
  the adopted warning solution above. Open work (not started): any later
  PINN handoff on the same graph/residual, and the optional removal or
  archival of the paused monolithic family.
- Smoke verification of the new entry point: a fresh
  `direct_nonlocal_solver run --load-final 2e-4` reproduced the archived
  early states exactly up to rounding (reaction `25.3468356139` at
  `0.0002`), and the compact-solution `verify`/`plot` paths were exercised
  on both the compact smoke and the legacy adopted archive.

## 2026-09-27 — Archived: direct exploration closed, user accepted the result

- Closed. The user ran the full fresh `0 -> 0.0075` solution with the entry
  point defaults (compact archive
  `direct_nonlocal_final/solution_u0p0075_fresh_v1.npz`, SHA256
  `5f3bc12b...`) and accepted it; all statistics reproduce the adopted
  research archive (`9545bed7...`).
- The optional monolithic removal listed under "Open work" is done: the
  monolithic/arc-length/reduced modules and their tests are removed from git
  tracking and listed in `.gitignore` (kept on disk for local reference).
- Superseded by: `examples/fracture2d/README.md` (formulation, parameters,
  accepted limitations),
  [`fracture2d-nonlocal-pinn.md`](fracture2d-nonlocal-pinn.md) (next stage),
  `docs/PROGRESS.md`, `docs/DECISIONS.md`.
- No further work is planned on this task.
