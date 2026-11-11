# Particle-corrosion method field gallery

## Goal and acceptance

The user requests representative-time physical fields with reference,
nonlocal and local AD together. Use the completed seed42/20k strict pair and
the first three of its four reused local-FEM geometries, without error-based
selection. Plot phase at five saved representative times with common geometry,
coordinates, scale and phase convention. Preserve particle holes and distinguish
reference interface contours from predicted contours. Read saved predictions;
no new inference, training, reference solve or Git operation.

## Context and inputs

Completed comparison: [paired study](pino-rolling-local-ad-controls.md).
Owner root: `/root/autodl-tmp/tf-logs/pino-rolling-ad-cold-20261111-01`.
Particle seed42 nonlocal/local-AD runs are recorded in its `journal.json`.
Both particle seeds pass strict input/state pairing; Electro43's unrelated
audit qualification is not applicable to these particle field plots.

## Status, blocker and next action

Complete. Three CPU-rendered comparison figures use actual saved normalized
times0,.2,.5,.8,1 (physical0,70,175,280,350s). Each is3rows(reference/nonlocal/
localAD) by5times, with original FEM triangles, common0..1viridis scale, gray
particle holes, actual-field black0.5contours and reference white dashed0.5
contours. Each predicted panel labels its actual spatial-volume-weighted phase
L2; pooled metrics are in the row labels. No training/inference/solve was run.

Reproduction:
```bash
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  -m docs.tasks.particle_method_field_gallery \
  --root /root/autodl-tmp/tf-logs/pino-rolling-ad-cold-20261111-01 \
  --seed 42 --samples 0 1 2 --output /path/to/new-gallery-directory
```

Actual output directory:
`/root/autodl-tmp/tf-logs/pino-rolling-ad-cold-20261111-01/particle-field-gallery-s42`.
Files: `geometry_000_phi.png`, `geometry_001_phi.png`, `geometry_002_phi.png`,
and `provenance.json` with direct source paths, all four geometries' recomputed
metrics, displayed times and field ranges. Raw arrays are neither modified nor
duplicated. All four saved-field metric dictionaries match final summaries;
the three figures were inspected visually and `git diff --check` passes.

The selected three geometries have NL pooled phase3.10/2.97/2.99% and terminal
5.96/6.27/5.71%; AD pooled33.40/33.88/33.41% and terminal45.78/49.49/46.92%.
Nonlocal follows the expanding front with late interface-position differences.
The AD pit stays near its original size and its metal phase stays below one;
already at t=0 its relative phase errors are13.74/13.73/13.73%, versus about
0.42%for nonlocal. This directly shows an initial-condition fitting shortfall,
not a proof that particle-wall Neumann enforcement alone causes the failure.
No method/default changes or follow-up runs are initiated.

## Session changes

Added `docs/tasks/particle_method_field_gallery.py` and the four external image/
provenance artifacts. Updated `docs/DECISIONS.md`, archived this task, and updated
`docs/PROGRESS.md`. No numerical/source-default, README asset, Git/history or
cleanup changes. No pending work remains for this gallery.
