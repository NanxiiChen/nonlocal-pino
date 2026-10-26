# Electropolish representative physical fields — 2026-10-25

## Goal and acceptance

Show representative physical fields after the corrosion gallery: original
versus transformed phase residual, same matched electropolish reference,
final5000-update seed42 models at physical0/1/2.5/5 s. Shared scales and
interface overlays; report per-time phi/c L2 and preserve extra phase bands.

## Context and status

Completed: inspected saved fields, CPU-reloaded both models and rendered static
scientific figures. Existing [paired study](coupled-latent-residual-experiments.md) reports terminal
phi/c0.28604/0.29443 original versus0.34398/0.37920 transformed, with extra
interfaces in both. This is the earlier original/latent pilot, not a later
corrosion raw-edge/s_bc experiment. No new training, PDE/BC/IC/reference or
default changes. Blocker none.

## Physical fields and metrics

Panels use columns0/1/2.5/5 physical seconds and rows reference/original/
transformed. Fixed0..1 scales; black dashed reference and red PINN phi0.5
contours. Domain axes are dimensionless x/Lc,y/Lc. The stable raw-edge transport
option and normalized s_bc data objective developed later for corrosion are
not present in these earlier electropolish pilots. No new electropolish
optimization was requested or performed for this gallery.

| Physical seconds | Original phi L2 | Transformed phi L2 | Original c L2 | Transformed c L2 |
| --- | --- | --- | --- | --- |
| 0 | 0.0021359 | 0.0024860 | 0.0037807 | 0.0032372 |
| 1 | 0.2891708 | 0.3070498 | 0.2995330 | 0.3338107 |
| 2.5 | 0.3140710 | 0.3917651 | 0.3251704 | 0.4288362 |
| 5 | 0.2860433 | 0.3439793 | 0.2944289 | 0.3792039 |

Reference remains a single rough surface that recedes/smooths. At all three
positive selected times both PINNs have three vertical phi0.5 crossings in
each of101 columns, versus one in reference; both initial fields have one.
Original develops a spurious low-phi band within solid below the main surface;
transformed develops a low-phi band and rebound nearer the reference surface.
The final x/Lc0.5 profile makes these extra transitions explicit. This is a
topology/field defect, not only a small interface-location discrepancy. The
images do not establish a unique cause or exclude gains from later retuning.

Artifacts: [phase fields](electropolish-physical-field-gallery-phi.png),
[concentration](electropolish-physical-field-gallery-c.png),
[signed errors](electropolish-physical-field-gallery-errors.png),
[terminal centerline profiles](electropolish-physical-field-gallery-profiles.png),
[metrics/direct paths](electropolish-physical-field-gallery.json),
[reproducible plotting script](electropolish-physical-field-gallery.py).

## Verification and task-end

Unique selected physical times and stored X/Y/time arrays agree exactly.
Both models have identical initial checkpoint bytes and use5000 updates,
seed42,5 s, unchanged physical parameters/reference and soft initial conditions.
CPU reload matches saved fields within1.82e-6 original and1.08e-6 transformed
(5e-6 acceptance). All eight per-time phi/c norms per model reproduce saved
summary entries at1e-12 relative tolerance. Reference/model crossing counts
are independently recomputed with a strict sign-change convention; no exact
phi0.5 nodes occur. Figures are visually inspected, finite JSON/images/local
links verified and git diff --check passes. The prior independent reference
time-refinement error1.52e-8 is retained; no new solver run is needed.

Diff-level changes: one script, four figures, one JSON, this archived report;
PROGRESS and append-only DECISIONS memory updated. No numerical runtime,
configuration/default or Git history change. Existing workspace changes are
preserved. Task complete; next action none; blocker none.
