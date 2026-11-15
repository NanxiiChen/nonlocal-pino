# Ceramic phi/concentration comparison figures

## Goal and acceptance

User asks for the three-method ceramic comparison in the approved channel style:
separate physical channels in final-error distributions and smoothed convergence.
Preserve Nonlocal PINO, Local AD PINO, FEM PINO labels and distinguish the common
FEM numerical reference. Editable Helvetica5–7pt PDFs and matching PNGs; NPG,
all solid convergence lines, actual raw records, automatic log axes, quartile
violins without dots. Record direct inputs/configs/checkpoints and metric scope.

## Context / selected data

Use exactly the three already published ceramic comparison arms in
`artifacts/ceramic_fem_comparison_publication.json`, not the failed first run or
a silently substituted newer nonlocal model. All are final20k seed42.
Final distribution:64shared localP1/lumpedFEM geometries, separate phi/c raw
space-time Euclidean relative norms. Convergence:21actual0:1000:20000 records,
mean of four development geometries, spatial-volume/time-trapezoidal weighted
norms. These two metric/cohort definitions must remain visibly qualified.
The later canonical-default nonlocal result is a separate retained run.

## Execution and boundaries

CPU-only rendering and saved-field checks. Channel sampled-AD queue37019 stays
running; its current journal is
`/root/autodl-tmp/tf-logs/channel-sampled-ad-20261114-02/sampled-journal.json`.
Do not modify its numerical sources or launch GPU inference/FEM work.
User explicitly switches foreground work to these plots. No Git authorization.

## Completed result

Completed CPU-only rendering from the retained final20k comparison models.
Repository outputs (each PDF/PNG, 170x60mm):
- `assets/pino-studies/particle_corrosion2d_component_convergence`
- `assets/pino-studies/particle_corrosion2d_component_distribution`

Final64case phase / concentration means:
- Nonlocal PINO: 0.038174886259585886 / 0.04620518849296874.
- Local AD PINO: 0.3220699166440732 / 0.3260596766870013.
- FEM PINO: 0.05553496385564092 / 0.0659368978302109.

No training, inference, numerical-reference solve or checkpoint substitution.
The already-published channel-mean table retains its values. New component
violins show all64per-channel errors, log10 Gaussian KDE/Scott bandwidth,
three-bandwidth tails and dashed quartiles. Ratios are not scaled by100.

Convergence preserves21actual records at0:1000:20000, arithmetic means of four
volume/time-weighted development errors. Thin raw and thick smoothed curves
are solid. Local-linear LOWESS operates in log10mean error with span0.30,
six nearest records, no robust iterations,401display points. All256ordered
resamples of four whole geometry trajectories give percentile2.5/97.5 bands.
These are approximate pointwise small-cohort intervals conditional on one
training trajectory, not cross-seed uncertainty, simultaneous/population
coverage guarantees, or a correction for smoothing/model-selection bias.
The display grid introduces no additional measured evaluations.

## Reproduction and verification

Renderer: `scripts/render_ceramic_component_comparison.py`, reusing the accepted
ceramic loader/PDF checks and `scripts/local_trend.py`. Direct run/config/model,
reference paths, all plotted values, exact command, metrics and compact checks:
`artifacts/ceramic_component_comparison_figures.json`.
Full raw-field audit, trend arrays and font evidence:
`/root/autodl-tmp/tf-logs/particle_corrosion2d/publication-component-comparison-20261114-01/provenance.json`.
Use a fresh output directory on regeneration; existing output is protected.

- Independently recomputed384final channel norms from192original saved
  prediction/reference pairs; actual coordinate/time arrays match exactly.
- Validated126raw development means from504case/channel observations.
- Exact empirical-bootstrap mean/variance identities pass. Independent
  analytic exponential-curve/convolution checks reproduce bootstrap quantiles;
  identical geometry curves collapse the interval to zero width.
- Inspected both previews. PDF text remains editable, Helvetica plus declared
  sans-serif mathematical font, embedded TrueType,6/7pt; no PDF raster images.
- All4repository assets match external originals and local built-site files
  byte-for-byte; total307994bytes below512KiB. PDF physical size170x60mm.
- Local website build succeeds for9pages;274local links resolve. PDF/PNG
  names, dimensions, metadata and local README paths agree. Diff check passes.

## Diff-level changes and completion boundary

Added the renderer, four curated image files and compact provenance. Updated
ceramic README, website content/build asset registration, assets index and
scoped DEVELOPMENT budget guidance. Prior published assets remain available;
unrelated pending work and training defaults remain unchanged.
Memory updated: this task, PROGRESS and DECISIONS. No commit, push or deployment
was requested or performed. Restore the still-running channel AD task as the
single active task. Plotting is complete; no blocker or automatic next action.
