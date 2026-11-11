# Electropolish compact 170 mm main-figure revision

## Goal and acceptance criteria

User confirms 170 mm width (A4 with 20 mm side margins), coolwarm, no reference
legend, tighter layout, scientific errors with multiplication sign and true
superscripts, vertically centered colorbar and upright label above it. Keep
actual type sizes at 5–7 pt and editable; rasterize only field/colorbar colors.
Same model, samples, saved times, raw-node phase errors as the prior preview.

## Context

Resume archive/electropolish-main-figure.md. Sources/configuration are explicit
in artifacts/electropolish_main_figure.json. This remains a preview; no README
promotion or Git action requested. Other local work stays untouched.

## Changes and artifacts

The preview is now 170 x 85 mm, replacing the previous 180 x 104 mm layout.
All twenty panels keep their 2:1 physical aspect ratio. Coolwarm uses one shared
0..1 scale. Removed the reference legend, reduced gaps, and placed the colorbar
on the midpoint of the complete physical-panel extent. Its phi label is above
and horizontal. This large-figure width is also recorded in the plotting guide.

Error annotations use explicit editable text spans: Helvetica for text/numbers,
DejaVu Sans Oblique for L, and true 5 pt raised exponents. Original LaTeX
expressions, mantissas and exponents are saved in the receipt. Font advances are
measured in physical points, independent of screen/export DPI; a first render
revealed DPI-hinted overlap, corrected before accepting the final version.
The final ordinary text is 6 pt, headings 7 pt, superscripts 5 pt.

Renderer: scripts/render_electropolish_main_figure.py.
Specification: artifacts/electropolish_main_figure.json.
Receipt: artifacts/electropolish_main_figure_validation.json.

External directory:
/root/autodl-tmp/tf-logs/electropolish2d/publication-main-20261112/compact-170mm

- electropolish2d_main_phase.pdf: 274,870 bytes, 170 x 85 mm.
- electropolish2d_main_phase.png: 329,335 bytes, 2007 x 1003, 300 dpi.
- validate_preview.py: independent raw-data, PDF font/layout and editability audit.
- before_redraw/: prior specification, renderer and receipt. Previous PDF/PNG
  remain at the parent output directory.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/render_electropolish_main_figure.py
```

## Validation

All 20 raw spatial phase scores and numerator/denominator norms match the prior
preview exactly; independent math.fsum norms pass. Existing samples, physical
times, grids and ranges unchanged. Both PNG and actual PDF preview inspected.

PDF physical width/height checked from its page box, without tight cropping.
It contains 21 raster color layers (20 fields plus the gradient) and 20 dashed
vector contours. All text remains embedded, editable, 5–7 pt sans-serif. Negative
exponents are raised, have the correct sign and do not overlap their base 10.
The actual PDF colorbar image center agrees with the combined field-image
extent within raster rounding (<0.08 mm); phi is centered above it and horizontal.
Ordinary-label and phase-symbol edits on temporary PDF copies pass.

Original selected README PNG, README, gallery script and source manifest remain
byte-identical to HEAD. Python syntax and git diff --check pass. No numerical
changes, new training/inference/solve, Git action or project dependency change.
Memory updated: docs/PROGRESS.md, docs/DECISIONS.md and this task archive.

## Status

Complete; updated compact preview ready for review, not promoted to README.

## Blocker

None.

## Next action

Apply further user feedback or promote the reviewed figure when requested.
