# Electropolish best-checkpoint phi panel — 2026-10-25

## Goal and completion

Complete. User requested3 rows(reference/prediction/absolute error),5 columns
at nondimensional times0,.25,.5,.75,1, phi only. A single best saved checkpoint
is used across every column. No new training/reference/default changes.

## Selection and provenance

Ranked all45 saved checkpoints(1000,2000,3000,4000,5000 updates) across the
completed current9-run cohort by lowest logged terminal phi relative L2.
Selected original residual, physical data losses, random IC/BC, seed7,
checkpoint005000: terminal phi L2=0.0011378188435460965. This is the successful
random baseline, not the nodal-data candidate. Criterion is terminal phi only,
not pooled error, transport, or unsaved intermediate models. Historical SOTA
had worse independently audited final phi errors and different grid/budget;
it is not part of this current-cohort selection. No production promotion.

Checkpoint direct path:
`/root/autodl-tmp/tf-logs/electropolish2d/20261025-data-physical-icrandom_interior-bcrandom-phase-stagger0-seed7/ckpt_005000.eqx`.
Reference direct path:
`/root/autodl-tmp/data/electropolish2d/results_matched_graph_20261024.npz`.
Reference is the previously validated independent same-graph numerical solve,
not an analytic field. Selected times exactly match saved reference arrays;
physical times0,1.25,2.5,3.75,5 s, T_c=5 s. Grid101x51.

## Outputs and verification

- [PNG](electropolish-best-phi-panel.png), [SVG](electropolish-best-phi-panel.svg).
- [Ranked checkpoints, direct paths and metrics](electropolish-best-phi-panel.json).
- [Selected fields and pointwise absolute errors](electropolish-best-phi-panel.npz).
- [Reproduction source](electropolish-best-phi-panel.py).

Run from repository root:
`JAX_PLATFORMS=cpu .venv/bin/python -m docs.tasks.archive.electropolish-best-phi-panel`.
Independent CPU checkpoint reload differs from saved GPU phi by at most
1.6093254e-6, below5e-6 check; terminal CPU L2=0.0011378260 reproduces logged
score within2e-6 absolute tolerance. Reference loader verifies config/model
compatibility; X/Y/time arrays exactly agree. Shapes, finite values and
prediction0–1 range pass. Per-time phi L2:0.00166628,0.00244266,0.00198130,
0.00185429,0.00113783. Pointwise absolute error maxima:0.01384885,0.01740267,
0.01240005,0.01461333,0.00793884.

Common0–1 color scale for reference/prediction; common0–0.01740267 scale for
all absolute-error panels, no quantile clipping or per-time rescaling. Error
is concentrated near the interface. Figure layout visually inspected. JSON,
SVG/PNG integrity, NPZ error identity, script syntax, links and diff checks pass.

## Task end

Complete; blocker none; no pending action. Memory changed: PROGRESS.md,
DECISIONS.md and this archived task. Outputs/source retained for reproduction.
