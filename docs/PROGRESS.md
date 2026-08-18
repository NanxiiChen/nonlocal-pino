# Project Progress

> Dashboard, not changelog. Keep this short — detail lives in task files / git history.

## Current focus

- Active subtask: _(none)_
- Task file: _(none)_

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

## Next up

- Validate a 2D scalar nonlocal operator and quadrature before coupling a PINN.
- Build and validate the direct pure-nonlocal staggered baseline before adding
  a PINN or core+halo sampling.

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
- 2026-08-16 — planned the staged 2D notched-fracture baseline and froze its
  geometry/continuum choices (see `docs/tasks/archive/fracture2d-notched-tension.md`)
- 2026-08-16 — refactored `AGENTS.md` into a concise policy/index with focused reference docs (see `docs/tasks/archive/agents-refactor.md`)
- 2026-08-16 — standardized the repository agent memory protocol (see `docs/tasks/archive/memory-protocol.md`)

## Known issues / open questions

- Existing `SparseD` does not yet provide per-node quadrature volumes for
  nonuniform 2D point clouds.
- Existing fracture losses use full-batch fixed operator nodes; core+halo
  sampling is not yet wired into the fracture path.
- The selected damage-aware correspondence energy and force still require an
  independently checked derivation and zero-energy stability validation.
- The local FEM strict staggered convergence, mesh refinement, and monolithic
  solver remain deferred follow-ups; the completed task archive records the
  accepted warning-mode limitations.
