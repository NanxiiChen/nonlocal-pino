# Electropolish publication main-figure preview

## Goal and acceptance criteria

Four samples by five times, predicted phase only, reference phi=0.5 interface
as dashed or dash-dot contours, one shared colorbar, individual instantaneous
phase-channel relative L2 below every panel. Follow plotting typography rules:
Helvetica 5–7 pt, dark editable text, no example title, case-bearing filenames.
Export PDF with raster field colors and vector text/contours, plus matching PNG.
Display interpolation must not change source-grid error metrics.

## Context

Use the README selected final rolling 10k checkpoint and samples 0/21/42/63,
predeclared refinement cases from the 64-input local-FD cohort. Five actual saved
times t*=0/.25/.5/.75/1 (0..5 s), no temporal interpolation. This is a preview;
retain current curated README image until the user reviews it. No Git operations
are authorized in this turn. Other unfinished Melting work remains untouched.

## Implementation and artifacts

CPU renderer: scripts/render_electropolish_main_figure.py.
Explicit source/style specification: artifacts/electropolish_main_figure.json.
Panel norms, original sample IDs, displayed labels and independent PDF audit:
artifacts/electropolish_main_figure_validation.json.

External output directory:
/root/autodl-tmp/tf-logs/electropolish2d/publication-main-20261112

- electropolish2d_main_phase.pdf: 285,319 bytes, 180 x 104 mm.
- electropolish2d_main_phase.png: 331,444 bytes, 2125 x 1228, 300 dpi.
- validate_preview.py: independent metric/PDF audit using PyMuPDF from
  /tmp/pino-pdf-audit (no project dependency change).

The five columns use normalized times 0/.25/.5/.75/1, physical times
0/1.25/2.5/3.75/5 s. Rows labeled Sample 1..4 map to original cohort indices
0/21/42/63, unchanged from the selected README gallery. Viridis has one shared
0..1 colorbar. All panels preserve the 2:1 domain aspect ratio. No example title.

Field colors use data-space bilinear display interpolation, with pixel centers
aligned to original nodes and clipped to the physical domain (no phase clipping).
The original reference nodes define the dashed phi=0.5 contours. Each panel's
raw relative L2 uses its original 5,151 spatial nodes at that actual saved time,
phase channel only; no channel or temporal average and no percentage scaling.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/render_electropolish_main_figure.py
```

## Validation

Selected checkpoint/configuration, grid/time/input identities and all four
pooled historical phase metrics checked before drawing. Independent float64
math.fsum calculations match all 20 panel numerator/denominator norms and
relative errors (2e-14 relative tolerance). Errors range from 0.00580917 to
0.0255463. Display interpolation never enters metric computation.

Both PNG and rasterized preview of the actual PDF visually inspected. Actual
PDF contains 20 raster field images plus one raster colorbar gradient, and
21 dashed vector paths (20 reference contours plus one legend). All labels,
colorbar ticks and phase symbol are editable Helvetica text, 6 or 7 pt, dark,
inside the page; actual TrueType glyphs are embedded. Ordinary-label and Greek
phase-symbol edits on temporary PDF copies round-trip successfully. Python
syntax and git diff --check pass.

The current curated Electropolish PNG, README, original gallery renderer and
source manifest remain byte-identical to HEAD. No training, inference, reference
solve, numerical/default change, Git/history operation or cleanup. Existing
Melting work and paused process group 618216 untouched.

## Status

Preview complete and ready for user review. Not yet promoted to the README.

## Blocker

None.

## Next action

Apply the user's layout feedback; if requested, replace the curated main image
and its README caption using these same declared samples and source fields.
