# Random held-out melting samples

Date: 2026-10-29

## Goal and acceptance

Show three randomly selected held-out ellipsoid melting trajectories from the
completed 3D neural-operator pilot. Draw before examining errors; preserve the
selection. Show reference/prediction/absolute phase error at all five saved
times, with common color scales and separately identified 3D diagnostic L2.

## Context

Run: /root/autodl-tmp/tf-logs/icemelting3d/20261029-melting-operator-online-b16-01.
Dataset: /root/autodl-tmp/data/icemelting3d/operator_ellipsoids_v1.
Eight held-out inputs, IDs 2 through 9. Saved fields and fixed final checkpoint
are used; no training, model selection, numerical changes or Git history work.

## Outcome and validation

Complete. Uniform system-entropy draw without replacement selected IDs 5, 9,
2, saved before opening scores or predictions. Frozen final checkpoint only.
Each image shows 128^2 points at z=0.503937, five dimensionless times
0/.25/.5/.75/1 (physical 0/1.25/2.5/3.75/5 seconds). Rows are reference,
prediction, absolute phase error; phase range [-1,1] and error [0,.1350976842]
are shared across every panel and case. Visual inspection passed for all three.

Independently recomputed 24^3 diagnostic-node relative L2 percentages:

| Dataset ID | Pooled phi | Terminal phi | Terminal solid indicator |
|---|---:|---:|---:|
| 5 | 0.238264 | 0.083434 | 1.876282 |
| 9 | 0.273115 | 0.125496 | 2.179775 |
| 2 | 0.200350 | 0.075706 | 4.455818 |

Solid indicator is (phi+1)/2. Whole-field phase error is background-dominated
as ice disappears, so it does not alone measure small late solid remnants.
Metrics are 3D diagnostic-node errors, not full128^3 grid or slice errors.
Finite arrays, explicit shapes/times, XYZ image transpose, manifest geometry,
exact saved reference-field/coordinate equality and saved-summary metric
replay all pass. No numerical source, optimizer, checkpoint or data changed.
No blocker, next action or queued training. Git history untouched.

## Archived artifacts

- [Selection](melting-random-samples-selection.json)
- [Plot replay](melting-random-samples-plots.py)
- [Metric replay and checks](melting-random-samples-metrics.json)
- [Sample 5](melting-random-sample-005.png)
- [Sample 9](melting-random-sample-009.png)
- [Sample 2](melting-random-sample-002.png)

Task-end: added only gallery artifacts and memory records; inspected plots,
replayed metrics, validated JSON and diff whitespace. No retraining or
unchanged expensive code suite was needed for this plotting-only task.
