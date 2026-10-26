# Corrosion representative physical fields — 2026-10-25

## Goal and acceptance

Show the user saved physical fields at representative times, comparing the
original phase/s_bc seed42 baseline with optional stable raw-KKS/edge seed42.
Use final5000-update models, unchanged50 s graph reference, consistent color
scales, interface overlays and quantitative per-time relative L2. Include
driving concentration to avoid hiding transport defects behind total c.

## Context and status

Completed: rendered saved arrays and CPU-reloaded both checkpoints from explicit
paths. Prior completed study: [transport optimization](corrosion-transport-optimization.md).
No retraining, PDE/BC/IC/reference/default changes. Existing unrelated workspace
changes are preserved. Blocker: none.

## Results and interpretation

Each plot has columns0/10/25/50 s and rows reference/original/stable KKS edges.
Black dashed contour: reference phi=0.5; red solid: PINN phi=0.5. Color scales
are fixed0..1 for phi/c and-1..0 for q/c_le. Signed-error panels use a shared
-0.05..0.05 scale (larger errors saturate). Coordinates are dimensionless
x/Lc,y/Lc; times are physical seconds. Pit mouth is at y=0. Baseline already
includes s_bc and matched nodal IC/BC. Stable is raw-edge/L1 with original phase
residual, not full inverse-Jacobian weighting or the favorable seed42 tail arm.

| Time (s) | Original phi L2 | Stable phi L2 | Original c L2 | Stable c L2 |
| --- | --- | --- | --- | --- |
| 0 | 0.0004574 | 0.0013368 | 0.0004723 | 0.0011136 |
| 10 | 0.0053002 | 0.0037351 | 0.0063704 | 0.0044698 |
| 25 | 0.0060888 | 0.0048975 | 0.0072893 | 0.0058477 |
| 50 | 0.0117707 | 0.0088353 | 0.0140207 | 0.0104255 |

Both models capture pit expansion; physical-field errors concentrate near the
interface and grow later. Stable is better at the three selected positive
times, while initial phi/c are worse. Initial driving q free-node L2 also
increases0.06039->0.26282; terminal q decreases0.03829->0.01372. This seed42
gallery does not establish every-time/every-seed transport improvement; the
paired seed7 regression in the prior study remains valid.

Artifacts: [phase](corrosion-physical-field-gallery-phi.png),
[total concentration](corrosion-physical-field-gallery-c.png),
[driving concentration](corrosion-physical-field-gallery-q_scaled.png),
[signed errors](corrosion-physical-field-gallery-errors.png),
[metrics and direct paths](corrosion-physical-field-gallery.json),
[reproducible CPU plotting script](corrosion-physical-field-gallery.py).

## Verification and task-end

Selected physical times match unique saved reference entries exactly. Grid/time
arrays agree between predictions/reference. CPU checkpoint fields reproduce
saved arrays within5e-6; terminal phi/c L2 reproduce summaries at1e-12 relative
tolerance. Raw-output KKS q uses float64 tanh, and its relative norm excludes
Dirichlet nodes. Figures are visually inspected; finite JSON and local paths
are checked.

Diff-level changes: add one plotting script, four PNGs, one metrics JSON and this
report; update PROGRESS/DECISIONS memory. No tests/training/reference solve or
numerical/default changes. git diff --check passes. No Git history operation.
Task archived; next action none; blocker none.
