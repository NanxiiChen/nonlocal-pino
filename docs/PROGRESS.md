# Project Progress

> Dashboard, not changelog. Keep this short — detail lives in task files / git history.

## Current focus

- Active subtask: _(none)_
- Task file: _(none)_

## Current state

- `fracture1d` is an accepted engineering baseline; default PINN dtype is
  float32 with `spatial_emb_dim=32` and `spatial_scale=16`.
- The next project direction is a 2D notched phase-field fracture example.
- The 2D continuum, notch representation, quadrature, and mechanics choices
  remain open and must be frozen before implementation.

## Next up

- Freeze the 2D continuum choices: plane assumption, split, irreversibility,
  and geometric notch versus phase-field precrack.
- Validate a 2D scalar nonlocal operator and quadrature before coupling a PINN.
- Build a direct nonlocal damage baseline before adding core+halo sampling.

## Recently completed

- 2026-08-16 — refactored `AGENTS.md` into a concise policy/index with focused reference docs (see `docs/tasks/archive/agents-refactor.md`)
- 2026-08-16 — standardized the repository agent memory protocol (see `docs/tasks/archive/memory-protocol.md`)

## Known issues / open questions

- Existing `SparseD` does not yet provide per-node quadrature volumes for
  nonuniform 2D point clouds.
- Existing fracture losses use full-batch fixed operator nodes; core+halo
  sampling is not yet wired into the fracture path.
- A 2D damage-aware mechanics formulation has not been selected.
