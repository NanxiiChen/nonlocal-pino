# Particle corrosion main-figure preview

## Goal and acceptance criteria

Render the next case requested by the user: ceramic-particle pitting corrosion.
Carry forward the approved Electropolish layout: four samples, five times,
predicted physical phase only, dashed reference phi=0.5 interface, one coolwarm
colorbar, instantaneous spatial relative L2 below each panel, scientific
notation, 170mm width and editable 5–7pt Helvetica text in the PDF.
Use actual mesh holes, no interpolation through ceramic particles. Preserve
geometric aspect ratio. Record direct checkpoint/data/configuration paths and
sample/time selection; inspect output and independently verify displayed scores.

## Context

Electropolish main-figure manifest/script provide the current style. Particle
README and pino_readme_figures.json identify the accepted six-particle350s,
width256/20k model. The derivative-comparison cohort has newer matched-run
results; do not silently substitute them for the selected main-figure model.
No Git operation authorized. Existing unrelated changes and paused jobs stay.
The user accepts leaving the Electropolish irreversibility diagnosis as is.

## Status

Completed preview, 2026-11-12. No inference, training, reference solve, numerical
change, README-image replacement or Git action. Electropolish display unchanged.

## Selected data and presentation

- Original accepted run: `/root/autodl-tmp/tf-logs/particle_corrosion2d/20261107-spaced-six-w256-cold20000-02`,
  `model_final.eqx`, fixed20000steps, runtime seed17617494.
- All four previously declared development inputs, indices0/1/2/3 in saved
  order. No new blind cohort or sample selection by score.
- References: `/root/autodl-tmp/data/particle_corrosion2d/20261107-spaced-six-01/local_dev4`,
  independent mass-lumped local P1 FEM, spacing0.01,256contour edges per particle.
  Predictions use these exact reference points. Saved geometry, SI parameters,
  runtime reference path, points and times match their manifests.
- Actual saved times0/.2/.5/.8/1, corresponding to0/70/175/280/350s. The existing
  eleven-snapshot arrays do not contain .25/.75; this deviation from the
  Electropolish columns was communicated. No temporal interpolation.
- PDF170x92mm; panels26.8x16.08mm preserve the1:0.6physical aspect, ydownwards.
  Coolwarm0..1predicted phase, gray ceramic, thin vector particle boundaries,
  dashed local-reference phi0.5contours, no legend/title. Colorbar centered
  on physical panels with its phi label above. Close sample labels retained.
- Original physical triangles, Gouraud display interpolation only; no new
  Delaunay triangulation, mesh smoothing or interpolation through particles.
- Each error is an equal-node-weight spatial phase norm at that sample/time.
  It is not the previous volume/time-weighted README metric or violin metric
  averaged across channels and complete time. The historical weighted phase
  scores are independently recomputed as an input-identity check only.

## Per-panel phase relative L2

| Sample |0|0.2|0.5|0.8|1|
|---|---:|---:|---:|---:|---:|
|1|0.00338168|0.00702168|0.03388119|0.05455992|0.06776222|
|2|0.00342799|0.00708583|0.02461277|0.04745365|0.09329989|
|3|0.00337696|0.00739358|0.02908149|0.06055061|0.08304789|
|4|0.00344501|0.00923957|0.03369984|0.05600179|0.06387460|

## Changes and reproduction

- `scripts/render_particle_corrosion_main_figure.py`: saved-array CPU renderer;
  reuses Electropolish scientific-label and node-metric helpers.
- `artifacts/particle_corrosion_main_figure.json`: explicit data/model/runtime,
  selection, reference, metric and style specification, no hash-based loading.
- `artifacts/particle_corrosion_main_figure_validation.json`:20rawscores,
  geometry identities, font/layout/layer records and independent validation.
- DECISIONS records the data/time/reference choices; PROGRESS closes the task.

```bash
.venv/bin/python -m scripts.render_particle_corrosion_main_figure \
  --spec artifacts/particle_corrosion_main_figure.json
```

External output directory:
`/root/autodl-tmp/tf-logs/particle_corrosion2d/publication-main-20261112`.
Files: `particle_corrosion2d_main_phase.pdf` and matching `.png`.
PDF1175016bytes; RGB PNG587598bytes,2007x1086pixels. External previews do not
change the repository's curated-asset budget or existing README image.

## Validation

- Actual render succeeds; all20norms independently match Python math.fsum.
- Four historical weighted phase norms match the saved training summary.
- Each mesh has six closed internal boundary components. Particle centers
  lie outside physical triangles; all120center pixels in20panels are gray.
- All phase values fall inside the declared0..1color scale; no field clipping.
- PDF actual size170x92mm; font sizes5/6/7pt, dark text, embedded Helvetica
  and DejaVu Sans Oblique TrueType, no Type3, all text inside page bounds.
- Exactly21raster layers (20fields+colorbar); interfaces, particles and text
  remain vector. Actual text-edit smoke check passes for a sample label and
  scientific numeric annotation on a throwaway PDF copy.
- PNG and rendered PDF visually inspected. Model/runtime/source paths and
  JSON parse correctly; `git diff --check` and new-file whitespace checks pass.

## Blocker

None.

## Next action

User review of the preview. Further layout adjustments or README promotion
can reuse the saved specification without recomputing numerical solutions.
