# Channel convergence: local smoothing and approximate trend intervals

## Goal / acceptance

User accepts final violins but requests the original development convergence,
all solid lines, local smoothing and95%confidence intervals. Preserve the
approved distribution bytes, raw41records per method, font/export conventions
and automatic log limits; keep README/site captions and provenance aligned.

## Statistical scope

One development geometry and one training run per method. This cannot estimate
between-seed or population uncertainty. Fit local linear LOWESS to log10error,
show raw values underneath and label approximate pointwise95%trend intervals.
Use circular-block residual bootstrap to retain short-range step dependence;
declare span, block length, replicate count and RNG seed. The fitted-trend
model assumes locally smooth trend and approximately stationary short-memory
residuals in logerror. Smoothing bias and model misspecification are not covered.
Neither ensemble confidence nor simultaneous95%coverage is claimed.

References: statsmodels LOWESS example and arch time-series bootstrap docs.
https://www.statsmodels.org/stable/examples/notebooks/generated/lowess.html
https://arch.readthedocs.io/en/latest/bootstrap/timeseries-bootstraps.html

## Method / reproducibility

Span0.22 selects9nearest records; use tricube distance weights and no robust
residual reweighting. Dense401locations only display the fitted trend; they are
not additional observations. Block length4,2,000replicates, RNG seed20261114
are fixed across all eight curves and recorded in provenance. Each residual
bootstrap resamples contiguous circular blocks, not independent steps.

With log-errors z, local-linear matrices S at observed steps and H at display
locations, let m=Sz and r=z-m-mean(z-m). Bootstrap data are m+r*. Recenter
bootstrap fits by Hm, so deviations d*=Hr*. The basic pointwise log-interval is
[Hz-q97.5(d*), Hz-q2.5(d*)]; exponentiate endpoints. This removes artificial
double-smoothing drift, not the unknown original trend bias. It is a conditional
approximation; coverage for the learning process is not empirically established.

Sources additionally include the LOWESS API and bootstrap interval definitions:
https://www.statsmodels.org/stable/generated/statsmodels.nonparametric.smoothers_lowess.lowess.html
https://arch.readthedocs.io/en/latest/bootstrap/confidence-intervals.html

## Completed changes and checks

`scripts/local_trend.py` implements the checked local-linear/interval algebra;
the channel renderer now overlays solid raw/trend curves and the bootstrap band.
All328original monitoring values are retained. Accepted violin PDF/PNG bytes
remain identical. All41points participate, including initialization; no clip,
curve averaging, altered error metric or monotonicity constraint is applied.

Independent weighted least squares matches all3,208fitted values to1.56e-15
in log10error. Explicit bootstrap refits reproduce band endpoints to2.78e-16;
analytic log-linear signals reproduce exactly with collapsed residual bands.
Circular block ordering and deterministic repeated export inputs pass.
PDF embedded/editable6–7ptHelvetica, automatic limits and dark text checks pass;
the preview is visually inspected. Four selected assets total406845bytes,
below512KiB. README/site captions and the v3manifest match the statistical scope.
Build passes for9pages; local links and copied assets are checked in the receipt.
No new training/inference, default change, Git action or deployment.

## Outputs / resume

- External outputs and eight raw/trend/bootstrap array files:
  `/root/autodl-tmp/tf-logs/channel_nsch2d/publication-fem-lowess-20261114/`
- Exact inputs/commands: `artifacts/channel_fem_comparison_figures.json`.
- Validation: `artifacts/channel_fem_lowess_validation.json`.
- Earlier mean/SD outputs: `publication-fem-statistics-20261114` in the same parent.

Complete locally, no blocker or pending job. A future explicitly authorized
publication should include these scoped changes and preserve unrelated dirty
numerical work. Earlier live Pages deployment remains unchanged.
