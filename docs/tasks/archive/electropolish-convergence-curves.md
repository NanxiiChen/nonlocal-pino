# Electropolish convergence curves — 2026-10-25

## Goal and acceptance

Show training convergence for the current random-data baseline and full-grid
analytic IC/graph-BC/s_bc candidate, seeds42/7. Separate phi/c panels and
terminal5 s versus pooled21-time metrics, raw200-update evaluations, common
logarithmic scale, no smoothing. Both groups use the original phase residual.

## Completed outputs

- [Terminal5 s PNG](electropolish-convergence-curves-terminal.png) and
  [SVG](electropolish-convergence-curves-terminal.svg).
- [Pooled0–5 s PNG](electropolish-convergence-curves-pooled.png) and
  [SVG](electropolish-convergence-curves-pooled.svg).
- [Raw histories and direct source paths](electropolish-convergence-curves.json).
- [Reproduction script](electropolish-convergence-curves.py).
- [Completed optimization context](electropolish-field-optimization.md).

Terminal L2 uses all5151 spatial nodes at physical5 s. Pooled L2 uses all5151
nodes and all21 saved physical times, rather than averaging per-time L2 ratios.
The candidate uses more prescribed initial-value samples; equal5000 updates
do not imply equal compute. No positive-time reference supervision is added.

## Results and validation

Seed42 candidate terminal phi/c0.002106/0.002518 versus random0.290928/0.299013.
Seed7 candidate0.002389/0.002869 versus successful random0.001138/0.001358;
preserve this counterexample. Candidate seed42 reaches the10^-3 order by
about1400 updates, followed by visible nonmonotone oscillations. Seed7 both
converge. Pooled histories show the same mixed endpoint comparison.

Verified exactly26 records per run, steps0..5000 every200; all four metric
series per run are finite/positive and each final value matches its saved
summary at rtol1e-12. Both PNG layouts inspected; labels/titles visible.
JSON, SVG/XML and PNG integrity, local links, source syntax and diff checks
pass. No new training, checkpoint, reference, numerical or default changes.

## Task end

Complete; blocker none, no pending action. Memory changed: PROGRESS.md,
DECISIONS.md and this archived task. Plot source and six artifacts are retained.
