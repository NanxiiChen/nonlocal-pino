# Fracture2D Requested Field Grid

## Goal

Add one large figure containing the requested `u_x`, `u_y`, and `d` fields in
three rows and five columns for `u_bar=0.005`, `0.0055`, `0.0058`, `0.0060`,
and `0.0075`.

## Acceptance criteria

- Generate the grid from the same saved CG1 triangle connectivity and snapshots
  as the existing fracture2d plots.
- Keep each panel's physical field range and use clear row/column labels.
- Save the figure beside the default point-anchor archive.
- Do not make short smoke runs fail when the requested load range is absent.

## Status

Complete: plotting implementation and default archive regeneration.

## Next action

The default archive contains exact snapshots for all five requested loads, and
the generated image is
`/root/autodl-tmp/data/fracture2d/local_fem_point_anchor/selected_fields_grid.png`.
