# Spinodal main-figure preview

## Goal and acceptance criteria

User requests the next publication gallery, Spinodal, with RdBu_r. Carry forward
four samples/five times,170mmwidth, phase-only prediction with dashed reference
interfaces, one centered colorbar labeled above, close sample labels and
instantaneous phase-node relative L2 in scientific notation. Equal spatial
scales, explicit holes and editable5–7ptHelvetica/sans-math text are required.
Use actual saved times, no temporal interpolation or new numerical solutions.

## Context

The selected model is confined_ch2d/20261031-confined-width256-online-cold50k-01-train.
Eight reserved development inputs exist; use the first four in saved order,
not the old error-ranked three-row selection. Native nonlocal reference only;
local FEM is not available for this harder case. Fields are cell-centered:
display reconstruction must preserve original physical triangles and holes,
while scores remain on original cell values with equal point weights.
No Git authorization, numerical default changes or curated-image promotion.

## Status

Completed preview,2026-11-12. No inference, numerical solve/training, PDE/default
change, curated README-image replacement or Git operation.

## Data and layout

- Checkpoint: `/root/autodl-tmp/tf-logs/confined_ch2d/20261031-confined-width256-online-cold50k-01-train/model_final.eqx`,
  fixed-budget50000steps, saved training seed42.
- First four reserved development geometries1024/1025/1026/1027 from the
  existing eight-input cohort; not a new blind test and not chosen by errors.
- Reference root: `/root/autodl-tmp/data/confined_ch2d/operator_geometry_smallscale10_1024_v1`.
  Reference is time-refined native nonlocal integration, not local FEM.
  Original cell centers/areas, mesh, holes, times and saved reference arrays
  agree exactly with the model's prediction exports.
- Five actual dimensionless times0/.75/1.5/2.25/3. No time interpolation.
-170x113mmPDF; each physical panel26.8x21.44mm, x0..1/y0.1..0.9,aspectequal.
  Shared RdBu_r[-1,1], black dashed native-reference phi0contours, white holes
  and thin gray vector boundaries. No main title/legend; colorbar label above
  the vertically centered bar. Close row labels and scientific error captions.
- Positive incident-cell area weights reconstruct vertex values for display,
  followed by Gouraud rendering on original physical triangles. Analytic
  circle masks exclude hole interiors. No re-triangulation, field clipping
  or numerical mesh refinement. This does not add physical resolution.
- Errors are unweighted spatial phase relative L2 on original cell-center
  values, matching the current point-norm convention. Historical area-weighted
  pooled norms are separately checked for source identity only.

## Per-panel phase relative L2

| Row / geometry |0|0.75|1.5|2.25|3|
|---|---:|---:|---:|---:|---:|
|1 /1024|2.52465e-8|0.0280448|0.0280817|0.0314019|0.0380011|
|2 /1025|2.53746e-8|0.0332988|0.0340501|0.0375427|0.0434388|
|3 /1026|2.47554e-8|0.0603749|0.0705027|0.0876046|0.1102541|
|4 /1027|2.54664e-8|0.0247521|0.0206189|0.0187124|0.0159614|

Initial errors reflect the existing hard IC and stored float precision, not
supervised initial-field optimization. The third row retains the relatively
large terminal error; no outlier or unfavorable geometry was excluded.

## Changes and reproduction

- `scripts/render_spinodal_main_figure.py`: CPU-only saved-cell renderer and
  source/mesh/range checks; shares scientific-label/node-norm helpers with
  the Electropolish renderer.
- `artifacts/spinodal_main_figure.json`: exact model/config/reference paths,
  runtime seed/step, selection, metric and style; no hash-based data loading.
- `artifacts/spinodal_main_figure_validation.json`:20scores, geometry, fonts,
  layout/layer records and independent PDF/mask/metric checks.
- DECISIONS records reference/reconstruction/selection conventions; PROGRESS
  marks the preview completed. Existing example README images stay unchanged.

```bash
.venv/bin/python -m scripts.render_spinodal_main_figure \
  --spec artifacts/spinodal_main_figure.json
```

External directory:
`/root/autodl-tmp/tf-logs/confined_ch2d/publication-main-20261112`.
Files: `spinodal2d_main_phase.pdf` and matching `.png`.
PDF1734310bytes; RGB PNG1322057bytes,2007x1334pixels. These full-resolution
previews remain external and do not change curated repository asset budgets.

## Validation

- Render and source/config/input/time matching pass. Cell centers and areas
  independently agree with the original physical triangles; hole centers
  have no containing physical triangle. Reconstructed values do not overshoot.
- All20instantaneous raw norms independently match Python math.fsum; four
  historical area-weighted pooled scores agree with the prior gallery.
- All40hole-center PNG pixels are white. No interpolation connects hole
  interiors. PNG and separately rasterized PDF visually inspected.
- PDF actual dimensions170x113mm;5/6/7ptdark editable text, Helvetica and
  DejaVu Sans Oblique TrueType embedded, no Type3, no clipped labels.
-21raster layers correspond to20fields+colorbar. Geometry/interfaces/text
  stay vector. All40PDFhole circles have equal width/height, and20outer
  ellipses measure26.8x21.44mm: equal scale independently confirmed.
- Actual sample-label and scientific-annotation edits pass on a temporary
  PDF copy; original output unchanged. JSON/AST/new-file whitespace and
  `git diff --check` pass. Unrelated work and paused process untouched.

## Blocker

None.

## Next action

User review of the preview. Future presentation changes or README promotion
can use the explicit specification without recalculating physical solutions.
