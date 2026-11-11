# Melting combined main figure

## Goal and acceptance criteria

User approves the proposed three-sample/five-time main figure: each sample
has a predicted 3D phi=0 surface row and an XY phase-slice row with reference
zero contours. Use current rolling10k checkpoint, fixed camera/equal scales,
170 mm width, editable 5–7 pt Helvetica text, one slice colorbar and per-time
three-dimensional phase errors. Record all direct source paths and reproduce.

## Context

Design: `archive/melting-main-figure-design.md`. Preselected random samples5/9/2,
actual dimensionless times0/.25/.5/.75/1, Tc=5s. Native nonlocal reference,
not the single-sample historical local-FEM validation. Reference has24^3volume
queries and128^2XYslices atz*=64/127; prior 3D meshes use an older checkpoint.
Current checkpoint: /root/autodl-tmp/tf-logs/icemelting3d/melting-pools-cold10k-20261111-01-rolling_pool/model_final.eqx.
Use dense128^3prediction inference only for surface extraction; reported norms
remain at original diagnostic points. No training/default/README promotion/Git.

## Status

Completed preview, 2026-11-12. Dense current-model inference, 15 actual zero
surfaces and paired slice panels rendered and independently validated. No
reference integration, training, numerical default or Git operation.

## Sources and numerical checks

- Model/config paths and runtime seed43 are in `artifacts/melting_main_figure.json`.
  It is the final10000-update rolling model, not the historical fixed5k model.
- Existing randomly selected held-out trio5/9/2, in that order. Five actual
  normalized times0/.25/.5/.75/1, corresponding to0..5s. Native128^3graph reference;
  not a local-FEM comparison and not an enlarged test cohort.
- Inference on128^3points per time, plus exact128^2saved XY-reference queries.
  Saved original24^3predictions replay with max error9.53e-7. Independent signed
  distance hard-IC checks max4.49e-6; all dense/chunk and dense/slice comparisons
  are exactly equal. Original checkpoint bytes unchanged.
- Actual phi=0 marching-cubes meshes:15closed manifolds, interpolation residual
  at vertices at most1.084e-6. No geometric smoothing or prescribed shape law.
  Mesh volumes decrease at every sampled time for each of the three inputs.
- All15instantaneous errors independently recompute using math.fsum on the
  original13,824volume queries. They are not slice errors or128^3dense-grid errors.

| Sample / saved ID |0|0.25|0.5|0.75|1|
|---|---:|---:|---:|---:|---:|
|1 /5|5.06793e-7|4.05349e-4|5.49198e-4|1.03371e-3|1.70706e-3|
|2 /9|3.15119e-7|4.25859e-4|5.63392e-4|9.55558e-4|1.87600e-3|
|3 /2|1.94412e-7|4.25141e-4|6.88530e-4|1.22786e-3|1.62369e-3|

## Figure and verification

170x166mmPDF and2007x1960PNG. Each sample has an upper true3Dsurface strip and
a lower XYslice strip. Orthographic elevation23/azimuth-57, equalXYZaspect,
fixed limits/zoom/light at all times. Matte ice-blue surface color is decorative
shading, not a second field variable. A dotted plane cue appears in each first
3Dframe; all slices are fixed atz*=64/127. The slice RdBu scale is shared[-1,1],
with dark dashed native-reference zero contours. Predictions are blue for ice.
Time labels use editable t* spans; a single scientific volume-error caption
sits beneath each paired panel. No main title or reference legend.

The first layout was192mmhigh; compacted unused 3D whitespace to166mm without
changing camera/object scale, slice size, data or typography. Independent PDF
checks confirm170x166mm,15square26.8mmslice frames,5/6/7ptdark text with embedded
TrueType fonts,31raster layers(15surfaces+15slices+colorbar),vector silhouettes,
reference curves and text. Temporary PDF text editing passed. PNG and separately
rasterized PDF visually inspected. JSON/AST/new-file whitespace/diff checks pass.

## Diff-level changes and reproduction

- `scripts/prepare_melting_main_figure.py`: explicit infer/meshes stages,
  source identity/replay/IC/range/slice checks and closed-surface validation.
- `scripts/render_melting_main_figure.py`: paired-row CPU renderer, shared
  publication typography helpers, fonts, metric/source checks and output receipt.
- `artifacts/melting_main_figure.json`: model/data/selection/style configuration.
- `artifacts/melting_main_figure_inference.json`: actual GPU inference provenance.
- `artifacts/melting_main_figure_meshes.json`: actual mesh paths and checks.
- `artifacts/melting_main_figure_validation.json`: raw errors, PDF/layer validation.
- PROGRESS and DECISIONS record completion/choices; this task is archived.

Reuse prepared fields/meshes to redraw:

```bash
.venv/bin/python -m scripts.render_melting_main_figure \
  --spec artifacts/melting_main_figure.json
```

To regenerate numerical display data, copy the specification and set a fresh
prepared_root and receipt paths (preparation refuses existing output folders),
then run these explicit stages:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
  -m scripts.prepare_melting_main_figure --spec /absolute/new-spec.json --stage infer
/tmp/melting-render-20261029/bin/python \
  -m scripts.prepare_melting_main_figure --spec /absolute/new-spec.json --stage meshes
```

The existing optional rendering environment supplies scikit-image for mesh
extraction; no dependencies were installed or changed. Inference runs in.venv;
rendering runs in.venvafter meshes exist. No FEniCS process is used.

Output directory:
`/root/autodl-tmp/tf-logs/icemelting3d/publication-main-20261112`.
`icemelting3d_main_phase.pdf`1117449bytes, matchingPNG686984bytes.
Prepared arrays are about34MBcombined, external; no weights or raw fields are
copied into Git. Existing historical README images remain unchanged.

## Blocker and next action

None. Preview delivered for user review. No pending GPU work, new training,
reference solve, cleanup or Git operation; paused process group618216 untouched.
