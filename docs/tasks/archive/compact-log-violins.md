# Compact log-scale violin revision

## Goal and acceptance criteria

Follow the user's notebook example: two separate compact figures with automatic log y limits,
filled NPG violins, thin outlines, inner quartile lines, no sample dots or mean
annotations. Plot original relative error ratios, no multiplication by100 and
no percent sign in the ylabel. Retain Helvetica and5–7pt publication typography.
Preserve all256sample errors from the same final seed42weights and64inputs.

## Context

Previous publication redraw is complete, but user rejects its aesthetic.
Original font remains /root/.local/share/fonts/pino/Helvetica.ttf. Existing
renderer/provenance source is scripts/render_pino_derivative_comparisons.py and
artifacts/pino_derivative_comparisons.json. User correction supersedes the
plotting guide's mandatory point overlays and percentage-axis wording.

## Changes

- Latest steering is applied: two separate55x45.72mmfigures, automatic log y
  limits/tick locations, original unscaled ratios, no percent sign or mean/dots.
- Updated renderer to thin NPG filled violins and dashed25/50/75percentiles.
  Scott KDE now fits natural-log errors with three-bandwidth tail extension;
  support is exponentiated for raw-ratio ticks. Width is density per log-error,
  area-normalized within each case. This removes the observed-extrema hard cut
  that caused flat ends, without modifying any observed error or quartile.
- Source manifest records KDE coordinate/tails/normalization and automatic
  range mode. Receipt records actual limits as outputs plus density samples.
- Updated plotting guideline, both example captions and assets/README.md to
  reflect raw ratios, no point overlay, explicit log KDE and automatic ylim.
- Regenerated only two violin PNGs/PDFs; physical-field sheets byte-identical.
  Previous presentation is externally archived; no combined figure generated.
- Memory changed: docs/DECISIONS.md, docs/PROGRESS.md and this archive.

## Artifacts and reproduction

Root:
`/root/autodl-tmp/tf-logs/pino-derivative-eval64-20261111-01/compact-log-violins-20261112`

- `electropolish2d_distribution.pdf`:18191bytes.
- `particle_corrosion2d_distribution.pdf`:17863bytes.
- `before_redraw/`: prior renderer, provenance, guide, captions and images.
- `validate_pdf.py`: independent PDF/data/range audit; PyMuPDF1.26.7 in
  `/tmp/pino-pdf-audit`, without changes to project dependencies.
- Existing README PNG paths under assets/pino-comparisons replaced in place.
- Compact audit: artifacts/pino_violin_pdf_validation.json.

```bash
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/render_pino_derivative_comparisons.py \
  --case all --figures distribution
```

## Validation

Both renders and all saved-field/config/ref audits pass. All256raw error
observations, channel norms, summary statistics, grids and checkpoints match
the prior receipt exactly. Quartile values match direct raw-sample quantiles.
Every observation enters KDE; all extended support points remain positive and
are precisely the declared log-extrema±3bandwidths. Densities are not manually
set to zero at endpoints. Actual y limits independently match Matplotlib's
5%margin in transformed coordinates; autoscale_y is true for both panels.
Electro actual range0.00637496714..0.20438991390; particle0.02496883953..0.39250877495.
These are observed renderer outputs, not input limits.

Both images visually inspected. PDFs contain embedded Helvetica TrueType,
6–7pt dark text including the L² label, no raster images, no clipped labels,
no percentage/mean text; label edits on temporary copies round-trip correctly.
JSON/source syntax, documentation links and git diff --check pass. The link
scanner was corrected to exclude mathematical delimiters; no figure defect.
All four comparison PNGs total1098909bytes. Numerical/default/SOTA field assets
and paused process untouched; no new inference, training, solve or Git action.

## Status

Complete; both separate compact figures ready for user review.

## Blocker

None.

## Next action

None pending. Any further presentation revision uses the same immutable scores.
