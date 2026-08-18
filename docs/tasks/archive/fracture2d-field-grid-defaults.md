# Fracture2D Field Grid Default Loads

## Goal

Change the default field-grid columns to `0.0055`, `0.0058`, `0.0059`, `0.0060`,
`0.0061`, and `0.0062` while retaining the configurable load-list parameter.

## Acceptance criteria

- Update the base configuration and focused configuration assertion.
- Update stable README wording and append the decision rationale.
- Regenerate `selected_fields_grid.png` from the default run.
- Verify all six requested snapshots are selected and the regression checks pass.

## Status

Complete: the requested default list is configured, regenerated, and validated.

## Next action

The default image is
`/root/autodl-tmp/data/fracture2d/local_fem_point_anchor/selected_fields_grid.png`.
Its columns select `0.0055`, `0.0058`, `0.0059`, `0.0060`, `0.0061`, and
`0.0062`, with maximum snapshot error below `3e-18`. The focused fracture2d
tests (5) and full repository unittest suite (65) pass; changed modules
compile and `git diff --check` passes.
