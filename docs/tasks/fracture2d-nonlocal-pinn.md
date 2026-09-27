# Fracture2D Nonlocal PINN

## Goal

Build the 2D nonlocal PINN stage on top of the adopted direct reference. The
PINN must reuse the **same** material graph, union collar geometry, A2
phase-measured mechanics energy, Amor history, and nonlocal AT2 damage residual
as `src/pinn/nonlocal2d/`; it must not re-derive a different operator or silently
change the model.

## Acceptance Criteria

- Strong-form nonlocal residuals (mechanics, Amor history, AT2 damage) are
  assembled from the same frozen graph/operator/parameters as the direct solver.
- Training uses the archived material cloud (`material_cloud_d048.npz`) and the
  adopted parameters (`delta=0.048`, `h_g=0.02`, `l0=0.024`, `Gc=2.7`,
  `E=210e3`, `nu=0.3`, `k_res=1e-8`, cubic kernel).
- Comparisons against the adopted direct reference clearly separate the
  pre-drift region (`u_bar <= 0.0061`, 65 strict states) from the warning/drift
  region; drift points are never hidden or deleted from the comparison.
- No model change (kernel, split, degradation, BC, particle set) is introduced
  without an explicit decision entry in `docs/DECISIONS.md`.
- Fixed shapes/dtypes in JIT paths; CPU/GPU parity checks where applicable.

## Relevant Context

- Adopted direct solution: compact `solution_u0p0075_fresh_v1.npz` (SHA256
  `5f3bc12b...`); research archive SHA256 `9545bed7...`; peak `641.34017` at
  `0.0061`, terminal `69.15426`, drift `2 -> 31`.
- Known limits of the direct reference: warning-only acceptance,
  `field_equations_validated=false`, terminal reaction far above FEM, and
  post-localization particle drift. The PINN stage must not treat the warning
  tail as converged truth.
- The adopted direct implementation is the single GPU-only file
  `examples/fracture2d/nonlocal_baseline.py` (union geometry, BA/A2 mechanics,
  Amor, AT2, JAX PCG backend, fixed damped-Picard continuation, and the
  `run`/`verify`/`plot` CLI). Reuse its functions/operators rather than
  re-deriving them.
- Reference documents: `examples/fracture2d/README.md`,
  `docs/NUMERICAL_METHODS.md`, `docs/JAX_GUIDELINES.md`.

## Status

Queued. The direct stage is closing; the PINN task becomes active after the
remaining lean-solver cleanup (see
[`fracture2d-lean-direct-solver.md`](fracture2d-lean-direct-solver.md)).

## Blocker

None. The adopted reference solution and lean operator library are available.

## Next Action

1. Freeze the PINN residual API against `src/pinn/nonlocal2d/` (no new math).
2. Write a manufactured check that the PINN residual matches the direct
   `A2StaticMechanics`/Amor/AT2 evaluations on the same fields.
3. Scale to the 3395-node cloud, then compare pre-drift states with the adopted
   reference before any post-peak claims.
