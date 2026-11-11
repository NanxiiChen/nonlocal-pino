# Melting publication figure design

## Goal and acceptance criteria

Discuss a publication layout combining 3D melting interfaces and spatial
slices, consistent with the accepted 170 mm phase galleries. Identify source
model/reference and avoid confusing older rendered surfaces with current SOTA.
Recommend sample/time layout, slice convention, reference overlays and metrics.

## Context

Current default: rolling pool, cold 10000 updates. Existing curated 3D gallery
uses historical 20261029 Adam model and three samples 5,9,2. User is considering
presentation options, not requesting a new training or numerical reference.

## Status

Completed design discussion, 2026-11-12. Existing 3D gallery visually inspected;
current model and reference array shapes checked. No figure generated.

## Blocker

None for design. A future main figure needs dense current-checkpoint inference.

## Next action

Await the user's layout choice; no rendering, solve or training is scheduled.
No Git authorization.

## Recommendation and supporting evidence

Prefer three samples, five times, two rows per sample (3D then XY slice).
Four samples remain an option when an eight-row figure is acceptable at170mm.
Use the same camera, lighting, physical limits and equal XYZ scales. Render
actual predicted phi=0 isosurfaces with a matte blue surface. Slice colors
represent phi, with shared limits and dashed native-reference zero contours.
Only the first frame needs a subtle slice-location cue. Preserve the visible
late-time shrinkage; do not zoom the final ice body independently.

Annotate a single instantaneous volume-phase L2 per sample/time, using all
original24^3diagnostic points. Clearly distinguish rendering resolution from
metric sampling; reference here is native nonlocal, not a new FEM cohort.
Near extinction, whole-field L2 can hide minority-phase error; the slice and
surface provide complementary shape evidence. An optional volume-decay plot
belongs in supplementary material if needed, not the already busy main grid.

Current model:
`/root/autodl-tmp/tf-logs/icemelting3d/melting-pools-cold10k-20261111-01-rolling_pool/model_final.eqx`.
Reference bank:
`/root/autodl-tmp/data/icemelting3d/operator_ellipsoids_v1`.
Inspected conditions5/9/2/3: saved predictions shape(5,13824); references
include both(5,13824)volume queries and(5,16384)slice values. Times exactly
[0,.25,.5,.75,1]; slice is z*=64/127, not exactly0.5. All four inspected final
reference slices still contain positive phase. A single fixed near-midplane
can therefore use existing reference exports; no sample-centered plane is
assumed or silently substituted.

Historical images/96^3meshes use the20261029fixed5kcheckpoint and must not be
relabeled as current. Existing randomly selected trio5/9/2 is a sensible
predeclared choice if using three samples; final choice awaits user feedback.
No smoothing, future shape law or ellipsoid fitting should replace predictions.

## Task-end

Only PROGRESS, DECISIONS and this design record changed this turn. Source-array
inspection and existing image preview completed; docs/diff whitespace checked.
No numerical source, training, inference, README image or Git operation.
