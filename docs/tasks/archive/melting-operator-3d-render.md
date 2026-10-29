# Publication-style 3D melting interfaces

Date: 2026-10-29. Status: COMPLETE; no training/FEM/render worker queued.

## Request and delivered scope

User requests genuine3Dviews of shrinking random ellipsoids instead of2Dslices.
They reject the decorative ice-material preview and prioritize figures suitable
for a paper. Reuse previous random held-out IDs5/9/2 and existing Adam final
checkpoint. Dense true predictions at61times over0..5s, extract phi=0 surfaces,
then render scientific figures with fixed orthographic cameras and scale.
No training, physics/geometry/model/default change or Git history operation.

Recommended main figure: [five-time matte surface strip](melting-paper-3d-single.png)
([PDF](melting-paper-3d-single.pdf), [SVG](melting-paper-3d-single.svg)).
Sample2 has the largest initial semiaxis ratio among the existing random trio;
this is geometric illustration, not error-based sample/checkpoint selection.
[All three inputs](melting-paper-3d-snapshots.png)
([PDF](melting-paper-3d-snapshots.pdf)) use the same times0/1.25/2.5/3.75/5s.

Recommended compact companion: [time-colored3Dcontours](melting-paper-3d-overlay.png)
([fully vector PDF](melting-paper-3d-overlay.pdf), [SVG](melting-paper-3d-overlay.svg)).
Each time has the actual silhouette plus intersections with three orthogonal
planes through the prescribed initial center. These are curves embedded in3D,
not a substitute2Dcentral slice. [Nested cutaway](melting-paper-3d-cutaway.png)
is an optional alternative: remove a front quadrant for display to expose
nested different-time interfaces. It is not a physical hole or multiple
simultaneous phases; caption it explicitly if used. The matte surface/contour
figures are preferable to the decorative preview for the user's paper goal.

## Numerical source and geometry validation

Checkpoint: /root/autodl-tmp/tf-logs/icemelting3d/20261029-melting-operator-online-b16-01/model_final.eqx
Saved config in the same run. Dataset:
/root/autodl-tmp/data/icemelting3d/operator_ellipsoids_v1.
All new fields/meshes/frames/provenance are under:
/root/autodl-tmp/data/icemelting3d/operator_render3d_v1.

- Actual unchanged FP32/highest/ordinary checkpoint inference on96^3XYZgrid,
  times61, input encoder once per sample, shared query preparation/chunk1024.
  Full field costs42.00/10.08/9.12s for5/9/2 including probes/save/first compile.
- Original saved field replay max1.32e-6; independent prescribed hardIC max
  4.15e-6; chunked/direct dense query check max4.77e-7. Checkpoint raw bytes
  unchanged. No file hashes or source-path migration.
- Marching cubes uses actual phi=0, grid order(time,x,y,z), spacing1/95.
  All183meshes closed/manifold (every edge has2faces); normals do not change
  vertices. Maximum trilinearphi residual atvertices4.52e-6. No geometric
  smoothing, prescribed shrink law or initial-ellipsoid rescaling is applied.
- Enclosed interface volumes decrease for all61times. Initial/final volumes:
  input5 .1242297/.0007602;9 .1405429/.0013028;2 .0959649/.00004734.
  These are geometric phi=0 volumes, not diffuse solid integrals or new L2.
- Fixed paper camera elev23/azim-57, orthographic, equal XYZaspect, bounds
  .08..92, common lighting. Axis limits are display limits, not a changed domain.
  Surface layers matte with thin silhouettes, no specular highlights. 450dpi
  PNG, PDF/SVG vector labels/axes/curves; dense filled layers rasterized at high
  dpi. The contour overlay is fully vector. Do not call filled PDFs fully vector.

## Preview setup and issues retained

Training.venv lacks optional3Drender packages; created isolated temporary
/tmp/melting-render-20261029. Exact package versions in melting-render-packages.txt.
No pyproject/training environment dependency change. Initial offscreen VTK probe
exited139 because standard libEGL.so.1 was absent. Downloaded libegl1 Ubuntu
1.4.0-1 and extracted it only to /tmp/melting-render-libs; no system install.
Explicit EGL renderer works onRTX3090/NVIDIA. Probe sphere is renderer setup
only and never used as a scientific result.

First preview had changing verified meshes but stale framebuffer because
add_mesh(render=False) was followed by screenshot without an explicit render.
Caught by image inspection before final delivery; raw first media/script/logs
retained under artifact root/initial-cached-render. Corrected p.render perframe
and verified first/last raw images differ. Foreground pixel-area ratios end/start
.0448/.0555/.0166 for5/9/2. Corrected preview GIF has61unique frames,8100ms with
endpoint holds; H264video81frames@10fps,1440x576. User rejected its decorative
style, so retain as preview rather than recommended paper output.

[Preview animation](melting-render-3d.gif), [MP4](melting-render-3d.mp4),
[prediction checks](melting-render-prediction.json),
[mesh/preview provenance](melting-render-surfaces.json),
[paper provenance](melting-paper-3d-provenance.json).
Nested cutaway uses one shared polygon collection to depth-sort all triangles;
separate collections incorrectly hid inner times in its first draft. Final
layout also keeps axes backgrounds transparent to avoid neighboring labels
being clipped. These are display fixes, not field/model changes.

## Reproduction and closeout

All scripts/logs/media are archived alongside this record. With existing mesh
artifacts, from repo root in a renderer environment:
python -m docs.tasks.archive.melting-render-paper
Paper rendering is CPU Matplotlib plus CPU VTKgeometry filters; EGL is only
needed for the optional preview. Optional pip packages were isolated and their
versions frozen. Dense inference uses training.venv and melting-render-predict.py;
its output root must be new (fails on existing). Preview rerender can explicitly
use --reuse-meshes with a fresh frames directory, preserving existing data.

Source checkpoint/fields remain at direct original paths. Read README for
paper links/caption conventions. No expensive numerical tests rerun because
neural/physical solvers did not change; field parity, independent IC, actual
mesh consistency/closedness, media metadata and visual inspection provide the
relevant verification. Python AST, strictJSON, PNG/PDF/SVG signatures, diff
whitespace and idle workers are checked at closeout. Memory: PROGRESS.md,
DECISIONS.md and this archived task updated. Next action: none; await feedback
on paper figure style. No commit/push/merge.
