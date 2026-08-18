# Fracture2D Field Grid Parameters

## Goal

Make the selected-load field grid configurable and regenerate it for the
requested six loads: `0.005`, `0.0055`, `0.00560`, `0.0065`, `0.0070`, and
`0.0075`.

## Acceptance criteria

- Read the ordered load list from `c.output.field_grid_loads`.
- Validate the list explicitly as finite, non-negative, and strictly increasing.
- Generate the default six-column `selected_fields_grid.png`.
- Preserve safe short smoke behavior and pass focused/full regression checks.

## Status

Complete: the six-load default grid is configurable, regenerated, and
validated.

## Next action

The default image is at
`/root/autodl-tmp/data/fracture2d/local_fem_point_anchor/selected_fields_grid.png`.
The custom short run with `c.output.field_grid_loads=[1e-4, 2e-4]` also produced
a two-column grid, confirming that the plot columns are configuration-driven.

Verification: 65 repository `unittest` tests passed; five focused fracture2d
tests passed in `fenicsenv`; changed modules compile; and `git diff --check`
passes.
