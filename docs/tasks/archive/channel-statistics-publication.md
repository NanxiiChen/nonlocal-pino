# Publish approved statistical figures and record future templates

Status: completed and published,2026-11-14.

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

## Publication in progress

Content0d831f1regularly merged/pushed as mainf2b9127. Pages43556036346 build
passed; deploy is queued. The clean main release worktree is
`/tmp/channel-statistics-release-20261114/release-main`; the original dev
worktree retains all67unrelated files and partial DEVELOPMENT bytes unchanged.
Next: verify deployment/live9HTML+4assets, archive this task and push a
documentation-only receipt via the same dev->main workflow.

## Task-end: live verification passed

Pages43556036346 completed build and deploy successfully. At
2026-11-14T04:20:19UTC, all9liveHTML pages and4selected PNG/PDF files returned200
and matched the exact candidate build byte-for-byte. The source content commit
is0d831f1, regular main mergef2b9127. Publish the verification/archive in a final
documentation-only follow-up; website bytes are unchanged and no new Pages
build is needed for that metadata commit.

Diff-level result: approved final violins;41record solid raw/LOWESS convergence
with conditional pointwise95%trend intervals; CPU renderer/helper, retained
cohort inference entry, direct-path provenance/validation, updated case README,
website chapter/build and shared plot templates. PLOTTING_GUIDELINES now adopts
these styles for future figures without changing metric definitions or claiming
cross-seed uncertainty from one run.67unrelated files and partial DEVELOPMENT
are preserved. No training/default change or external data/weight deletion.

Checks: selected staged-tree render reproduces all4files;9pages/271relative
links pass; font/edit/statistical/syntax checks pass; diff whitespace clean;
live13URLs match. Changed memory files are PROGRESS,DECISIONS and this archived
task, with `artifacts/channel_statistics_publication_validation.json` recording
publication evidence. No remaining required work or experiment is queued.
