# Four PINO README overviews — completed

## Goal and outcome

User requested a concise project overview and setup/strategy/result summaries,
with one3--5sample evolution figure in each main PINO directory. Delivered root
README and all four example READMEs:3Dmelting/random initial ellipsoids,
2Delectropolishing/random initial interface,2Dspinodal/random holes and full
2DNS--CH/random channel walls. No numerical configurations, training, inference,
reference solves or Git history operations were changed/performed.

## Diff-level changes

- Root README now leads with the four operators, varying inputs, physical models,
  scoped phase metrics/reference types, encoder/strategy comparison and commands.
  Secondary PINNs/prescribed-flow pilots follow. No duplicate historical-score
  tables in the root; detailed evidence remains in the example documents.
- Each example now starts with PINO overview, physical setup/BCs, strategy table,
  actual final checkpoint, scoped metrics, multi-sample picture and launch/data
  paths. Original content after its old title is preserved exactly in a folded
  detailed-formulation/single-IVP/research section; existing heading anchors stay.
- Added exactly one compact assets/pino-evolution.png per example. Melting uses
  the actual saved3Dphi0isosurface meshes for inputs5/9/2; other cases show saved
  phase with dashed independent reference contours. Five actual saved times per
  row, no time interpolation, field smoothing or new solution generation.
- Added scripts/render_pino_readme_figures.py, executed all four renderers on
  existing direct-path artifacts. It checks reference coordinates/times/fields,
  recomputes sample metrics and verifies closed mesh edge pairs. First3Dlayout
  had crowded labels; corrected only margins/repeated labels and rerendered.
- Added artifacts/pino_readme_figures.json with exact cases, times, direct model/
  field/reference paths, selection protocol and byte counts. Channel's existing
  selected default was missing from the old artifact index; added documentation
  pointers/actual metrics plus explicit child inventories. Existing index values
  remain exactly equal; no checkpoint/reference was moved or rewritten.

## Figures and representative results

| Example | Display samples | Times | PNG bytes | Evaluation headline |
|---|---|---|---:|---|
| Melting |5,9,2,existing random held-out trio |0,1.25,2.5,3.75,5s |289226 |8native tests,phase0.2266% pooled/0.0922% terminal; localFEM check separate |
| Electropolish |0,21,42,63,all predeclared refinement cases |0,1.25,2.5,3.75,5s |399754 |64localFD tests,phase4.427%/3.176% |
| Spinodal |1027,1028,1026,existing low/uppermedian/worst |0,.75,1.5,2.25,3 |913776 |8reused native development geometries,phase4.8329%/5.1935% |
| Full NS--CH |0,1,2,all reserved cases |0,1.5,3,4.5,6 |275761 |3reused localFEM cases,phase1.8396%/2.2708% |

Four necessary README assets total1878517bytes=1.79MiB. Each<1MiB, total<2MiB;
no raw arrays, trained weights, PDFs, videos or repeated galleries added. PNGs
are scientific plots from existing values. Display interpolation/mesh-to-vertex
projection is explicitly separated from original nodal/physical-cell metrics.
Only melting is true3D; other three are2D. References/scopes are intentionally
not interchangeable: native nonlocal melting/spinodal versus localFD electropolish
and local mixed FEM channel. Chemical errors and small-cohort limitations remain
visible; the term complete example does not claim universal accuracy.

## Verification and task-end

[Compact verification](../pino-readme-overview-validation.json).
- Executed actual figure generator for all cases; inspected all four final PNGs.
  Recomputed10selected2Dsample errors against the saved reports; checked15actual
  3Dmesh snapshots, times and closed edge pairing. No new model inference.
- Root pooled/terminal percentages checked against the actual selected saved
  evaluations, including exact8/64/8/3input counts. Corrected a double-rounded
  spinodal root terminal entry to5.193% from direct5.193489...% (detailed4decimal
  presentation5.1935% retained).
-36new/overview local links/anchors, all launch config paths and image signatures/
  dimensions pass. Historical README bodies and all existing artifact-index
  values remain exactly equal. The new helper AST and JSON records parse.
- Read-only artifact check passes9cases/445registered directories. Repository
  content guard passes2134currently tracked files; separately checked the four
  new images against1MiB/file and2MiBtotal. Git diff --check passes.
- No numerical tests, expensive unchanged suites, training or FEM rerun for this
  documentation task. Existing validated numerical results are reused.

Memory updated: docs/PROGRESS.md, docs/DECISIONS.md, this archived task and its
compact validation JSON. Status:COMPLETE. Blocker:none. Required resume:none.

## Deferred user idea

The user proposes a later channel study to20000updates with a lower final LR,
explicitly without starting now. Exact LR and continuation/cold-start choice
remain undecided. Recorded in channel README and project deferred work; no
scheduled automation, new active training task or run is created.

## Subsequent asset organization

The user subsequently requested a single root asset directory and publication.
The four PNGs are now at assets/pino/<case>.png, with all current references
updated and original bytes preserved. This supersedes the per-example paths
described in the historical implementation above. See
[asset organization/publication](pino-readme-assets-publication.md).
