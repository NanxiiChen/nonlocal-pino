# Publish approved statistical figures and record future templates

## Goal / acceptance

User approves the latest plots and explicitly authorizes commit/push/Pages.
Publish current four selected PDF/PNG assets, renderer/helpers, direct-path
provenance and case/site documentation. Record the approved convergence and
violin styles as future defaults in PLOTTING_GUIDELINES, including correct
statistical scope. Follow dev -> commit -> main -> regular merge -> push main.
Verify workflow success and live page/figure bytes. No new training/defaults.

## Context

Start main4854922, localdevf772345; originNanxiiChen/nonlocal-pino.
Approved figures: artifacts/channel_fem_comparison_publication.json(v3).
Statistical checks: artifacts/channel_fem_lowess_validation.json.
Earlier3case inference: artifacts/channel_fem_statistics_validation.json.
Prior archived tasks retain method and direct inputs. Distribution unchanged
from accepted violins; convergence is41development records, solid raw/trend
curves and conditional approximate pointwise95%bootstrap intervals.

## Status / blocker / exact next action

Active, no blocker. Preserve pending bytes under
`/tmp/channel-statistics-release-20261114/before`, update plotting guidance,
stage only publication scope (partial DEVELOPMENT section), validate candidate
build and assets, then commit/merge/push. Await Pages success and verify live
bytes before marking complete. Memory changes included; unrelated numerical
and stopped research edits must remain local.

## Candidate validation

The exact staged candidate builds9pages and passes271relative link checks.
Its CPU renderer reproduces all4selected PDF/PNG files byte-for-byte, total
406845bytes. Editable embedded Helvetica,6–7pt dark text, automatic axes and
statistical audits pass. Python syntax is checked; numerical training tests
are not rerun for this plotting-only release. All67unrelated pending files and
the full partial-staged DEVELOPMENT worktree remain byte-identical.

The future plotting templates now explicitly distinguish raw/smoothed curves,
conditional trend intervals and cohort/seed uncertainty; actual errors remain
the source of numerical conclusions. Ready to commit24scoped files and publish.
Receipt: `artifacts/channel_statistics_publication_validation.json`.
