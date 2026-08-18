# Fracture2D Notched Tension

> Superseded planning record. Its original full-clamp and spectral-split
> choices are historical; the active implementation uses the paper-aligned
> hybrid formulation and bottom-vertical/point-horizontal support documented in
> `docs/tasks/archive/fracture2d-local-fem.md`.

## Goal

Plan a direct numerical baseline for a pure-nonlocal phase-field fracture model
on the unit square with a horizontal geometric notch and top displacement load.

## Acceptance criteria

- The domain, notch topology, loading, supports, and plane assumption are explicit.
- The mechanics and damage formulations are fully specified and traceable.
- Nonlocal quadrature, visibility, boundary treatment, and scale requirements are explicit.
- The plan separates direct numerical validation from later PINN and sampling work.
- Verification covers operator consistency, stability, refinement, energy, and fracture response.

## Relevant context

- `fracture1d` is the accepted engineering baseline.
- The project direction is a 2D notched phase-field fracture example.
- `SparseD` lacks per-node quadrature volumes for nonuniform 2D clouds.
- Damage-aware nonlocal mechanics and 2D core+halo fracture sampling are not implemented.
- Repository policy requires continuum and quadrature choices to be frozen before implementation.
- The user selected plane strain, a fully clamped bottom edge, and the Miehe
  spectral split with the existing history-field irreversibility approach.
- The notch is a finite-width slot centered on the requested horizontal line;
  its width is to match a declared phase-field profile width and exceed both
  active horizons.
- The phase-field width is the full width at `d=exp(-1)`, hence the nominal
  local AT2 value is `2*l0`; set `w_notch=2*l0` and use a semicircular tip with
  `r_tip=w_notch/2=l0`.
- Use one common horizon, `delta_u=delta_d=delta`, for the first controlled
  comparison while keeping the two roles distinguishable in the derivation.
- Implementation order is independent local FEM, direct nonlocal discretization,
  then a nonlocal PINN.

## Plan

1. Freeze nondimensional material data, loading conditions, and load increments.
2. Implement and verify an independent local FEM reference with the selected
   plane-strain spectral/history formulation and rounded slot geometry.
3. Define a fixed Cartesian particle domain with slot particles removed,
   explicit particle area, and separately audited mechanics/damage graphs.
4. Extend and verify scalar and correspondence operators for geometry masks,
   visibility, dtype, energy, conditioning, and finite-horizon boundaries.
5. Derive a plane-strain spectral-split discrete elastic energy and its
   energy-consistent nonlocal force; keep the evolving damage graph fixed.
6. Implement a direct staggered solver with immutable previous-step history,
   direct fixed-history damage solves, strict convergence, and rollback/cutback.
7. Validate geometry, operators, intact elasticity, fixed-history damage, and
   coupled fracture against independent local FEM and refinement studies.
8. Add a PINN and fixed-shape core+halo sampling only after the direct baseline
   passes all gates.

## Status

Completed: the staged FEM/nonlocal/PINN implementation plan and geometry
convention are recorded.

## Blocker

None for this planning task. Material/load parameters belong to the subsequent
local FEM implementation task; the nonlocal Dirichlet collar remains deferred.

## Next action

Open a local FEM implementation task. Use a slot centerline at `y=0.5`, width
`2*l0`, and a semicircular tip centered at `(0.5-l0,0.5)` so its foremost point
is exactly `(0.5,0.5)`.
