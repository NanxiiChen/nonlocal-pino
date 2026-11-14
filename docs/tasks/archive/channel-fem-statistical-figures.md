# Channel FEM comparison: statistical figure revision

## Goal and acceptance

User requests final-error violins and convergence mean with standard-deviation
shading. Preserve actual saved data, field-wise norms, Helvetica editable PDF,
170 mm width, NPG colors and automatic log-axis limits. Update local README/site
captions and reproducibility metadata to match the revised figures.

## Context

Previous publication: `archive/channel-fem-publication.md`, main4854922.
Comparison: `/root/autodl-tmp/tf-logs/channel-fem-full-20261114-01/comparison.json`.
The every-500-step logs monitor one development geometry, so cannot supply
between-geometry standard deviations. Both runs retain weights only at steps
0, 5000, 10000, 15000 and 20000. Reevaluate these on the existing same three
qualified reserved FEM geometries. No new training or reference solve.

Use arithmetic means and sample SD (ddof=1) across the three geometry errors;
the band is not a confidence interval. Final violins use log10-error KDE,
Scott bandwidth, three-bandwidth tails and internal quartiles. N=3 is explicitly
limited descriptive evidence; smoothed density does not add observations.

## Completed changes and validation

Completed locally. New `scripts/evaluate_channel_fem_checkpoints.py` evaluates
both methods at all five real checkpoints, using canonical geometry/network
helpers and the exact original query ordering. Thirty predictions are retained
externally; no training or additional reference solve ran. Whole evaluation took
85.44 s. Final fields replay prior outputs to max absolute 3.94e-6; displayed
final means retain the published rounding.

`scripts/render_channel_fem_comparison.py` now independently audits 144 field
norms (24 previous final, 120 reevaluated), derives arithmetic mean and ddof=1
SD, shades actual five-checkpoint curves and renders requested log-error
violins. Final phase mean/SD are nonlocal0.016435/0.002997 and
FEM0.046465/0.003067. Intermediate trajectories are not assumed monotonic.
PDF/PNG pairs under `assets/pino-studies/` total328990bytes, below512KiB.
Helvetica text is embedded/editable,6–7pt, dark; log limits remain automatic.
Both previews were inspected; PDF label edits and font checks passed.

Example README, assets README, site chapter/README, DEVELOPMENT publication
guidance and v2 figure manifest now match the cohort/violin semantics. The site
build passes for9pages/271local links; all4copied assets match byte-for-byte.
Changed Python modules compile; `git diff --check` passes. Existing unrelated
dirty code and partial DEVELOPMENT edits are preserved. No Git operation,
deployment, default change, queued training or live preview remains.

## Direct outputs and resume point

- Inference: `/root/autodl-tmp/tf-logs/channel_nsch2d/cohort-checkpoints-20261114/evaluation.json`
- Render outputs: `/root/autodl-tmp/tf-logs/channel_nsch2d/publication-fem-statistics-20261114/`
- Provenance/commands: `artifacts/channel_fem_comparison_figures.json`
- Local validation: `artifacts/channel_fem_statistics_validation.json`
- Publication assets: `artifacts/channel_fem_comparison_publication.json`

The earlier live publication is unchanged. Future commit/push requires explicit
authorization; publish only this revision's scoped files and preserve unrelated
pending numerical work. No numerical task is automatically resumed.
