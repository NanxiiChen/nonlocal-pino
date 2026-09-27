# Fracture2D A2 Coupled Formulation

## Goal

Derive, review, and, only after approval, implement a clean-zero-start staggered
fracture2d diagnostic coupling Candidate A2 phase-measured grip mechanics to the
existing material-only AT2 damage/history problem. Determine whether preventing
the initial top mechanics concentration also prevents irreversible top-history
contamination through the crack-growth transition.

## Acceptance Criteria

- Preserve the archived Candidate A2 geometry, phase-measure, energy, outer-face
  constraint, matched-opening, and reaction contracts.
- Keep damage and history exclusively on the original material particles and
  existing material `SparseD` graph; grips remain explicitly intact and absent
  from damage/history equations.
- Define every stagger step, immutable committed/trial field, physical-opening
  target, mechanics scaling solve, Amor update, damage solve, gate, rollback,
  cutback, and archive field without hidden defaults.
- Derive the opening-conjugate reaction and mechanics work check when the outer
  displacement changes with the current fixed damage.
- Start from exact zero damage/history/displacement and use strict mode first.
  Warning acceptance is not authorized by this task unless separately reviewed.
- Preserve existing material-only BA and A2 frozen diagnostic behavior and tests.
- Validate manufactured algebra, zero-load state, pre-transition comparison,
  strict continuation gates, top/tip history evolution, and independent replay.
- Obtain an independent numerical-method GO before implementation and a second
  implementation review before interpreting results.

## Relevant Context

- Material-only BA develops a top-boundary positive feedback at `u=0.0056`.
- Candidate A2 passes all isolated mechanics hard gates and reduces instantaneous
  top `Psi_plus` at matched physical opening from `78.8966` to
  `5.6326--6.2479`.
- Its transplanted frozen test fails the predeclared `H_used` improvement gate
  because raw top history is already `76.1040`; a clean-start coupled path is the
  only meaningful test of whether that contamination can be prevented.
- Candidate A2 is project-owned and nonproduction. This task does not authorize
  a production horizon, PINN/JAX integration, or a change to the local FEM
  reference.

## Status

Active: strict coupled implementation and independent contract validation are
in the worktree. The actual numerical opening-gate rejection now has a
primitive replay/archive regression test; no pilot or physical interpretation
is claimed.

## Blocker

Independent numerical-method GO and final implementation review remain
required before any rounded-cloud pilot or coupled physical interpretation.

## Next Action

Primary Sol review of the strict implementation and all focused tests; keep
the A2 path nonproduction and do not authorize a pilot.

## Derived Coupled Contract

### Fields And Energy

The union mechanics particle set has `N_union` particles and retains the exact
`N_material`-particle material cloud as its prefix. Damage and history have shape
`(N_material,)`; displacement has shape `(2*N_union,)`. Grip particles never
receive damage/history fields.

For material damage `d_m`, mechanics uses the archived A2 energy

$$
\Pi_{A2}(q;d_m)
=\sum_{i\in\mathcal P_\star}
\left[V_i^m g((d_m)_i)+V_i^g g(0)\right]
\sum_{j\in\mathcal H_i^\star}\eta_{ij}^\star\psi_{i|j}^\star(q),
$$

where the material term is present only for `i<N_material` and
`g(d)=(1-d)^2+k_res`. Union volumes remain the sole quadrature in family masses,
`eta`, `beta`, and shape tensors. The material `SparseD`, volumes, graph, AT2
energy, and fixed-history solve remain exactly unchanged.

Compute the source-centered Amor driver from union edge strains/weights, then
retain only material source centers:

$$
\Psi^+_{m,a}(q)
=\sum_{j:(a,j)\in\mathcal E_\star}
\eta_{aj}^\star\psi^+_{a|j}(q),\qquad a<N_m.
$$

Material-to-grip target edges therefore contribute to material history, while
grip-source centers do not. No grip history is constructed.

### Opening-Controlled Mechanics

Continuation load is the physical material-interface opening

$$
\Delta_m(q)
=\frac{1}{L_T}\sum_{i\in\Gamma_T}w_i^T u_{i,y}
-\frac{1}{L_B}\sum_{i\in\Gamma_B}w_i^B u_{i,y},
$$

not outer displacement. For fixed `d_m`, linear homogeneous A2 mechanics gives

$$
q(U;d_m)=U\widehat q(d_m),\qquad
s_\Delta(d_m)=\Delta_m(\widehat q(d_m)).
$$

Require finite `s_Delta>0`, then impose target opening by

$$
U_{ext}=\frac{\Delta_{target}}{s_\Delta(d_m)},\qquad
q_{target}=U_{ext}\widehat q(d_m).
$$

Reevaluate the scaled displacement and require exact opening, authoritative free
residual, force, and reference-moment gates. Archive `U_ext`, `s_Delta`, and the
achieved opening separately.

The work-conjugate response is

$$
R_{eff}=\frac{d\Pi_{red}}{d\Delta_m}
=\frac{R_{ext}}{s_\Delta(d_m)}.
$$

Centered fixed-damage energy differences must independently validate both
`R_ext=dPi_red/dU_ext` and `R_eff=dPi_red/dDelta_m`. Mechanics-before and
mechanics-after generally have different `U_ext` and `s_Delta`; their reaction
gate therefore compares `R_eff`, never raw `R_ext`.

### Strict Stagger

For immutable committed `(Delta_c,U_c,q_c,d_c,H_c)` and target `Delta_t`:

1. Set `d_iter=d_c`; keep `H_c` immutable throughout the attempt.
2. Build a fresh shared-data A2 view at `d_iter` and solve mechanics-before at
   exact `Delta_t`.
3. Set `H_used=max(H_c,Psi_plus_m(q_before))`.
4. Call the unchanged material fixed-history damage solve with `H_used`.
5. Strictly reject raw range/healing failures before mechanics-after; never clip
   or project.
6. Build a fresh A2 view at `d_next` and solve mechanics-after at the same
   `Delta_t`.
7. Set `H_next=max(H_c,Psi_plus_m(q_after))`; do not accumulate `H_used`.
8. Evaluate all gates. Commit exactly `(Delta_t,U_after,q_after,d_next,H_next)`
   only if every gate passes.
9. If a complete sweep fails with budget remaining, update only local `d_iter`
   and the normalized mechanics seed; retain committed history.
10. At budget exhaustion reject the attempt, verify the committed-state digest,
    and cut back the physical-opening increment.

No standalone final damage solve, warning commit, damping, clipping, increment
growth, or hidden adaptive rule is authorized.

Required complete-sweep gates are: achieved opening for before/after; material
and full-union displacement changes; damage change; history change;
`R_eff` change; committed `H_next` material damage residual; damage range;
healing; history monotonicity; both mechanics CG/free-residual checks; positive
opening slopes; and both union force/moment checks. The committed residual uses
the cancellation-aware constituent-term scale already adopted by the
material-only replay.

### State And Seeds

The immutable state is

```python
@dataclass(frozen=True)
class A2CommittedState:
    physical_opening: float
    outer_displacement: float
    union_displacement: np.ndarray
    material_damage: np.ndarray
    material_history: np.ndarray
```

The exact initial state contains scalar and array zeros and must independently
replay to zero energy, residual, reaction, and Amor driver.

Mechanics seeds are accelerators only. Normalize accepted displacements by their
accepted nonzero `U_ext`; at zero use the outer-top load direction. Changed-
damage systems cannot early-exit from a seed. Cutback seeds derive only from the
unchanged committed state, never a rejected attempt. Store seed provenance and
initial/final residual diagnostics.

### Architecture

Implement a separate `examples/fracture2d/nonlocal_a2_coupled.py`; do not relax
or copy the existing material-only `run_strict_continuation` contract. Reuse

- `UnionGripGeometry`, `AffineConstraints`, and interface line quadrature;
- `GateTolerance`;
- `solve_fixed_history_damage` and `FixedHistoryDamageResult`;
- the existing history maximum and Amor algebra where shapes match.

Add A2-specific immutable state, opening-mechanics result, accepted-step,
diagnostics, metadata, attempt/result, failure, archive, and replay records.
Metadata stores material and union clouds separately and verifies exact material
prefix/induced-graph identity. Archive schema `fracture2d-a2-coupled-v1` stores
static union/material data once and per-step states/gates without repeating all
union edge tensors. It must load with `allow_pickle=False`, record
`raw_source_used=False`, and reject transplanted history.

The static union BA support must be constructed once. Add a shared-data
phase-measured fixed-damage view that recomputes only `W_i`, effective/edge
degradation, center factors, and caches. It must match the existing independent
A2 constructor before continuation uses it. Rebuilding roughly 79,000 supports
for every stagger solve is prohibited.

### Implementation Validation Order

1. Shared-data A2 view equivalence and independent phase-weight algebra.
2. Exact zero state and one manufactured strict step against a dense reference.
3. Opening scaling, unequal-slope `R_eff` gate, energy/work, seeds, rollback,
   cutback, and strict failure tests.
4. Separate material/union metadata and forged archive replay tests.
5. One short rounded-cloud pre-localization step only after all focused tests.
6. A clean zero-start path through `0.0055` and `0.0056` only after short-pilot
   review and explicit run authorization.

The interpretation criteria remain top `H_used` below tip, at least 50% FEM
top-error reduction, no artificial material interface/corner maximum, and
`R_eff` within the diagnostic 10% FEM guard. They do not alter strict numerical
acceptance.

## Review Closure Contract

The first independent review rejected implementation until the following
executable details were frozen. This section supersedes any less-specific wording
above.

### First Pilot Configuration

The first rounded-cloud pilot is one strict clean-start endpoint run:

- `delta_m=delta_d=0.048`, `h_g=0.02`;
- `opening_start=0`, `opening_final=1e-4`;
- `d_opening_initial=1e-4`, `d_opening_min=5e-5`;
- `cutback_factor=0.5`, `max_cutbacks=8`;
- `max_stagger_iterations=20`;
- `cg_rtol=1e-10`, `cg_atol=1e-13`, `cg_maxiter=3000`;
- standard displacement-material, displacement-union, damage, history,
  reaction-effective, force, and moment gates use
  `GateTolerance(atol=1e-9,rtol=1e-8,scale_floor=1)`;
- damage-residual gate uses
  `GateTolerance(atol=1e-10,rtol=1e-8,scale_floor=1)`;
- opening uses absolute tolerance `1e-12` with zero relative tolerance;
- `opening_slope_min=1e-6`;
- `healing_atol=damage_range_atol=1e-10`;
- fixed-damage work check uses centered `epsilon=1e-5` and relative limit
  `1e-7` with independently tightened CG `rtol=1e-10`, `atol=1e-13`.

Opening targets use the same eight-ulp endpoint snap rule as the material-only
controller. Continue to exact `opening_final=1e-4`; cutbacks may produce multiple
accepted increments before that endpoint. No endpoint above `1e-4` is authorized.
Final numerical-method GO plus focused phase-1--4 test passage authorizes this
endpoint run, subject to primary verification of those tests.

### Residual And Gate Definitions

Retain both material residuals

$$
\rho(d,H)=\frac{G_c}{l_0}d+G_cl_0D_md+2H\odot d-2H,
$$

$$
\rho_{used}=\rho(d_{next},H_{used}),\qquad
\rho_{next}=\rho(d_{next},H_{next}).
$$

`rho_used` validates the fixed-history solve; `rho_next` is the committed stagger
gate. Archive strong and volume-multiplied versions of both and their infinity
norms. The committed gate uses the configured strong infinity norm. Redundant
replay uses the componentwise float64 reconstruction bound

$$
512\epsilon_{64}
\left(
\left|\frac{G_c}{l_0}d\right|
+|G_cl_0D_md|+|2H\odot d|+|2H|
\right).
$$

Material displacement error is the infinity norm over the first `2*N_material`
DOFs. Union displacement error is the infinity norm over all union DOFs and
therefore includes `abs(U_after-U_before)`. Their scales are respectively the
maximum before/after material norm and target opening, and the maximum
before/after union norm, both outer displacements, and target opening, with the
configured scale floor. Damage/history/reaction use infinity/absolute errors and
the same explicit max-before/after scale pattern. Force uses the infinity norm of
the complete union resultant with scale
`max(abs(R_ext),abs(R_bottom),abs(R_anchor),1)`. Moment is the absolute reference
moment with scale `force_scale*geometry_length`, where `geometry_length` is the
maximum absolute union coordinate/range. Both
mechanics-before and mechanics-after must pass every mechanics/opening gate.

For both mechanics-before and mechanics-after, centered fixed-damage work checks
for `R_ext` and `R_eff` are hard gates. Each relative error is
`abs(R_fd-R)/max(abs(R_fd),abs(R),1)` and must not exceed `1e-7`; use the frozen
perturbation and tightened work-solve tolerances above.

Require finite `s_Delta>=opening_slope_min`, finite `U_ext`, exact constrained
values after scaling, and opening error at most `1e-12`. The statement that
opening scales linearly means algebraic linearity at fixed damage; every
CG-approximate scaled result is independently reevaluated.

### Fixed-Damage Identity And Seeds

Every static union BA object has two identities. Runtime identity is a
process-local immutable token plus the exact material-damage identity. Archive
identity is a deterministic SHA256 of static primitive shape/dtype/bytes and a
separate deterministic SHA256 of exact material-damage shape/dtype/bytes. A
process token is never serialized or used for cross-process replay.

The deterministic static digest covers coordinates, original material damage
volumes, union-padded material phase volumes, grip phase volumes, union volumes,
material prefix count/map, material/union graphs, support primitives,
constitutive parameters including `k_res`, constraints, and line quadrature.
Every unit mechanics result stores runtime identities and deterministic digests.
Scaling rejects any mismatch and accepts only a unit result with exact
`outer_displacement=1` from that same fixed-damage view.

A changed material-damage digest always performs a fresh unit solve with early
exit disabled. Same-damage early exit is allowed only when static token, damage
digest, target system, and explicit residual all match.

Construct every normalized unit seed from
`constraints.prescribed_vector(1.0)` and copy only
`q_previous[free_dofs]/U_previous` into free DOFs. Never divide/reuse constrained
entries. Validate exact constrained equality, dtype, shape, and finiteness before
the solve. Zero uses the prescribed unit vector. Mechanics-after and later
sweeps may use the preceding normalized result only as a seed for a fresh solve;
cutback seeds come only from the committed state. Archive seed kind, source
state/step/sweep, source damage digest, seed digest, initial residual, early-exit
eligibility/use, and final residual. Incomplete/rejected sweep seeds are never
committed.

### Shared Static Mechanics

Introduce a dedicated immutable `A2StaticMechanics` built once from the union
geometry. Its explicit shared whitelist is coordinates, union volumes, directed
edges, source/target/distance/kernel weights, family masses, `eta`, support
indices, `beta`, `xi`, shape inverses/ranks/conditions, plane-strain constitutive
constants including `k_res`, rank tolerance, particle/DOF counts, original
material damage volumes `(N_material,)`, union-padded material phase volumes,
grip phase volumes, material prefix count/map, and runtime/deterministic static
identities.

`A2FixedDamageView` may reference only those whitelisted fields. It constructs
fresh material damage, material/grip/effective degradation, `W_i`, edge
degradation, center factors, and fresh tangent/preconditioner caches. It cannot
copy an arbitrary `__dict__`, inherit another view's lock/cache fields, or mutate
the static object. Independent-constructor equivalence covers energy, residual,
tangent action, strain/stress, reactions, center weights/factors, and cache
isolation under at least two nonuniform damage fields.

### Amor Invariant

Filter union edges only by `source<N_material`; retain all such targets,
including grip targets. For every material source, independently require the
retained union `eta` sum to equal one within `512*eps64`. Accumulate undegraded
edge Amor energies with those original weights and output exactly
`(N_material,)`. Never filter by target, renormalize, pad union history, or call a
helper whose source-shape contract would admit grip history.

### Failure Semantics

Incomplete mechanics-before, damage, range/healing, mechanics-after, or history
failure performs no local `d_iter` update, creates no synthetic `H_next`, cannot
be accepted, preserves the committed digest, and enters typed strict cutback.
Only a complete finite failed sweep may update local `d_iter` while stagger
budget remains. Structural dtype/shape/metadata/provenance errors propagate and
are not converted to cutbacks.

Records and archives contain literal `acceptance_mode="strict"`; accepted status
is only `strict_passed`. Warning status, bypassed-gate lists, and warning loaders
do not exist. Minimum increment or cutback exhaustion raises typed failure with
the last immutable committed state and complete attempt audit.

### Unequal-Cloud Archive And Independent Replay

Static archive primitives include all material coordinates, original damage
quadrature volumes with shape `(N_material,)`, triangles, damage graph/operator
parameters, material masks/anchor; all union coordinates, union-padded material
phase volumes, grip phase volumes, and union volumes each with shape
`(N_union,)`, material/grip/union triangles, union
candidate/visible/removed graphs, supports inputs, interface/outer masks,
constraints, line indices/weights, `delta`, `h_g`, visibility tolerance/rule,
material-prefix map, material and union hashes, constitutive/fracture parameters,
config, schema, code hash, and `raw_source_used=False`.

Require exact equality between original damage volumes and the prefix of the
union-padded material phase volumes, exact zero material phase volume beyond the
prefix, and `union_volume=material_phase_volume+grip_phase_volume`. Interface
prefix particles may have both material and grip phase volumes positive.

Each accepted final sweep stores `damage_before`, `d_iter`, `d_next`, `q_before`,
`q_after`, `Psi_before`, `Psi_after`, `H_used`, `H_next`, both outer
displacements, the mechanics target used by each before/after solve, slopes,
achieved openings, `R_ext`, `R_eff`, both strong/volume
damage residuals, mechanics residual/conservation/work values, every gate
error/scale/limit/pass flag, and complete seed provenance/digests. Attempts store
committed before/after digests and explicit availability masks for incomplete
fields; no NaN or object arrays are permitted.

Every complete rejected sweep stores the same primitive before/after fields,
residuals, work values, gate values, and seed evidence as an accepted final
sweep, plus failure stage/reason and availability flags. Incomplete rejected
attempts store every primitive produced before failure, explicit unavailable
flags for later stages, exception type/message, and no fabricated later field.
This permits independent replay of failure classification, cutback choice,
unchanged committed digest, and every complete failed gate set.

Replay must not call production mechanics, opening-scaling, Amor, or
fixed-history helpers. From primitive arrays it independently reconstructs cubic
weights, family mass/eta, target-inclusive supports, shape tensors, phase center
weights, energy-gradient residual, tangent checks, constrained equilibrium,
opening/slope/reactions, material-source Amor, the material AT2 matrix/solve/
residual/energy, all final gates, state pairing, and rollback digests using direct
NumPy/SciPy loops. Exact zero initial fields and `raw_source_used=False` are hard
requirements; any transplanted nonzero initial history is rejected.

### Latest Contract Validation

- The work-gate regression is named explicitly as a work-gate test.
- The opening-gate regression solves the real fixed-damage mechanics problem at
  `target_opening + offset`, retains the returned displacement/reaction/slope/
  work primitives, and derives the production opening rejection against the
  attempted target.
- Independent in-memory and on-disk replay reconstruct the off-target solve
  from line weights and archived displacement primitives, reject the exact
  opening failure classification when forged, and reject a forged opening pass
  flag. The rejected attempt retains the committed-state digest and no
  post-gate seed commit.
- Verification: the A2 unittest file passed `29/29`; changed modules compile;
  `git diff --check` passes. No rounded-cloud pilot was run.

Cross-damage energy differences are never treated as conservative work because
the history stagger is not one joint potential. Work checks perturb only outer
displacement at one fixed material-damage digest and use the reaction sign
`R_ext=a_ext.T @ grad(Pi_A2)`.

### Pilot Precision Revision

The first authorized run with primary `cg_rtol=1e-7`, `cg_atol=1e-10` stopped
strictly at `opening=1e-4` after accepting `5e-5`: mechanics-after work errors
were about `4.83e-7` (`R_ext`) and `5.53e-7` (`R_eff`) against the unchanged
`1e-7` gate. Independent reconstruction showed the primary solve residual was
`9.69e-8`, while the work-reference solves used `1e-10`; a tight primary solve
reduced the derivative mismatch to approximately `1e-11`.

Independent review therefore revises only the first pilot primary CG tolerances
to the tightened work-solve values `1e-10/1e-13`. The work gate, equations,
opening endpoint/increments, rollback, and every other strict setting remain
unchanged. The failed run remains `validated=False` evidence at
`/tmp/opencode/fracture2d_a2_coupled_strict_opening_1e4.json`. Rerun from exact
zero only; no endpoint above `1e-4` is authorized.

## 2026-09-27 — Archived: A2 mechanics absorbed into the adopted direct scheme

- Closed. The phase-measured A2 union mechanics derived and validated here
  became the mechanics layer of the adopted engineering-grip direct scheme
  (`nonlocal_engineering_grip.py` / `direct_nonlocal_solver.py`); the separate
  opening-controlled strict coupled pilot was never continued past `1e-4` and
  is no longer planned.
- The adopted complete `0 -> 0.0075` reference is the stabilized-warning run
  documented in [`examples/fracture2d/README.md`](../../examples/fracture2d/README.md);
  the split between the opening-controlled A2 diagnostic and the volume-grip
  coordinate is recorded in `docs/DECISIONS.md`.
- Superseded by: `examples/fracture2d/README.md`,
  [`fracture2d-nonlocal-pinn.md`](../fracture2d-nonlocal-pinn.md) (new PINN
  stage), `docs/PROGRESS.md`, `docs/DECISIONS.md`.
- No further work is planned on this task.
