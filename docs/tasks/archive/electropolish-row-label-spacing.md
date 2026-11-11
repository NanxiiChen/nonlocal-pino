# Electropolish row-label spacing

## Goal and acceptance criteria

Move the four Sample labels closer to the first field column, with about 1 mm
clearance. Preserve the 170 x 85 mm canvas, fonts, other layout and data. Export
matching PDF/PNG and inspect actual spacing; no README promotion or Git action.

## Context

Small presentation follow-up to archive/electropolish-main-compact.md. Current
label center x=4.5 mm and first field begins at x=10 mm. Change label center to
7.8 mm only. Preserve the preceding preview and its exact configuration.

## Changes and validation

Changed only style.row_label_x_mm from 4.5 to 7.8 in the figure specification;
renderer code is unchanged. New PDF/PNG are under:
/root/autodl-tmp/tf-logs/electropolish2d/publication-main-20261112/compact-170mm-close-labels

The matching filenames remain electropolish2d_main_phase.pdf and .png.
PDF 274,869 bytes; PNG 329,170 bytes. Previous files remain in compact-170mm;
before_redraw preserves the immediately preceding specification and receipt.

Independent actual-PDF checks confirm all four labels moved exactly 3.3 mm
right; visible label-to-panel clearance is 1.1399 mm. Other text origins, font
sizes and names are identical. PNG pixels at and right of the first field's
left boundary are exactly unchanged. All 20 scores are unchanged; existing
PDF size, fonts, raster/vector, exponent and centered-colorbar checks pass.
PNG visually inspected; git diff --check passes. No training/inference/solve,
README promotion or Git operation. Other local work remains untouched.

## Status

Complete; updated preview ready for review.

## Blocker

None.

## Next action

Apply further layout feedback or promote the reviewed figure when requested.
