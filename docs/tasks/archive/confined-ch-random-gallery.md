# Random physical-field gallery for the confined CH geometry operator

Date: 2026-10-30. Status: COMPLETE. No Git history authorization.

## Goal and acceptance

Show five randomly drawn reserved geometry samples from the completed best30k
operator. Record the draw before reading scores. Plot phase only at
\(t=0,0.75,1.5,2.25,3\), with reference/prediction/absolute-error rows and shared
color scales. Verify unchanged physical arrays and independently reproduce
area-weighted relative L2. Keep generated images outside Git.

## Context

Completed study: [optimization](confined-ch-fourier-optimization.md).
Selected run:
/root/autodl-tmp/tf-logs/confined_ch2d/20261030-confined-geometry-b16-t16-wide96-30k-retry-01.
Its eight reserved development geometries are 512..519, previously reused for
development but absent from training draws. References are native graph/BDF,
not local FEM. Initial ambient field is fixed; hole geometry varies. Saved
fields have already passed independent CPU checkpoint replay.

## Display and verification plan

Uniformly sample five of eight without replacement using fresh random entropy;
record actual IDs before loading predictions or errors. No reselection.
Reuse original-mesh area-weighted display reconstruction with analytic hole
masks. Reconstruct raw absolute error directly, avoiding cancellation from
interpolating signed error first. Metrics always use unmodified cell fields.
No model, optimizer, training, reference, or default changes.

## Result and validation

Blind draw in original order: 514, 519, 516, 517, 518. No reselection after
reading scores. Geometry514 happens to be the worst of the eight reserved
geometries; it is retained in the requested gallery.

| Geometry | Phase pooled relative L2 | Terminal relative L2 |
|---|---:|---:|
| 514 | 17.2550% | 19.1956% |
| 519 | 14.6498% | 16.4438% |
| 516 | 7.7125% | 8.0864% |
| 517 | 13.6721% | 14.7357% |
| 518 | 4.7791% | 4.9224% |

Five PNG/PDF panels saved in the selected run's random_gallery_01 directory.
Each has3rows/5columns at the requested five times. Common phase scale
[-0.8459164816085255,+0.8459164816085255], error [0,0.38419043789135476], derived
from all eight reserved geometries. Original-mesh display resolution600x480;
no mesh refinement or numerical field modification. All five images inspected.
Different geometries show different late-time morphology. Larger-error cases
show deviations near holes and portions of evolving interfaces.

All eight saved point/area/hole/time/mesh arrays and embedded reference fields
match independent bank files exactly. Initial phase matches to1e-13. Original
area-weighted pooled/terminal scores independently match evaluation/metrics.json
to1e-13relative/1e-15absolute tolerance. All selected display fields finite;
positive raw-absolute-error reconstruction preserves nonnegative/shared bounds.
Script exits0. PNG total4.57MiB, PDF2.27MiB, all outside repository.

Reproduction script: docs/tasks/confined_ch_random_gallery.py. Receipt with
actual IDs, native reference paths, scores and figure paths:
docs/tasks/confined-ch-random-gallery.json. Sampling receipt is published before
field/error loading in external selection.json. To reproduce precisely, use the
recorded IDs/figures; a new invocation intentionally makes a new random draw.
Artifact index adds the gallery pointer; no large media are added to Git.

## Task-end changes, blocker, next action

Added gallery script and compact receipt; updated artifacts/index.yaml,
docs/DECISIONS.md and docs/PROGRESS.md; archived this completed task. No changes
to models, training, solvers, references, configs or defaults. No Git action.
Verification: actual complete gallery execution with identity/score/display
assertions, visual inspection of all five panels, JSON syntax and diff check.
Status: complete. Blocker: none. Next action: none required; show five panels.
