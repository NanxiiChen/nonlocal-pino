# Publish the channel FEM PINO comparison

Date: 2026-11-14. Status: complete. User authorizes convergence/distribution
figures, README/site updates, commit, main-only push and GitHub Pages publication.

## Goal and acceptance

- Show actual development-field relative-L2 convergence through 20k for nonlocal
  and FEM PINO; separately show all three reserved-geometry final errors.
- Four fields, common numerical FEM references; raw ratios/log axes, fixed
  method colors and styles, 170 mm editable Helvetica PDF plus matching PNG.
- Distribution uses three paired observations and mean bars, not an unsupported
  KDE estimate. No evolved reference labels enter training, and no new runs.
- Retain direct input/checkpoint/config/reference paths, all plotted numbers,
  exact reductions and figure validation. Inspect PNG/PDF and web rendering.
- Update case README/site and relevant overview; publish only related sources,
  compact records and selected figures via dev -> main regular merge -> push.
  Preserve unrelated dirty files byte-for-byte and verify live Pages deployment.

## Evidence and scope

Completed record: docs/tasks/archive/fem-weak-pino-comparison-plan.md.
Receipt: artifacts/channel_fem_comparison_validation.json.
External bundle: /root/autodl-tmp/tf-logs/channel-fem-full-20261114-01/.
Source runs and 3-case final scores are recorded in comparison.json. Single
initialization; historical accuracy GPU UUID differs from current resource
probes. Four weak losses/norm/natural BCs differ from nine nonlocal losses.
No claim of universal superiority. Production defaults remain unchanged.

Before edits, pending files were copied to /tmp/channel-fem-publication-baseline
for exact preservation checks. Starting main is76cd883; origin is the renamed
nonlocal-pino repository. Any partial source staging must be independently
checked in the candidate release tree before commit.

## Status and resume

Plotting/source/asset policies read. Implement figures and direct-path manifest,
then validate data/fonts/edits/layout, update docs/site, inspect scoped source
dependencies, commit/push and monitor Pages. No blocker; no active GPU work.


## Pre-publication checks

The 170 x 54 mm convergence and paired distribution figures are complete.
All328actual monitor values and24reserved-case norms reproduce; both PDF/PNG
pairs reproduce byte-for-byte and total265384bytes. Actual PDF text6–7pt,
embedded Helvetica TrueType, dark editable vector labels and independent
label-edit probes pass. PNG and rendered PDF previews were visually checked.

Nine-page site build and271relative links pass. Both images load without page
overflow or console errors at1440px and430px; screenshots inspected. The exact
staged candidate produces byte-identical site files. Nineteen focused FE,
integral, budget and online tests pass in41.556s on the staged tree. Eight
scoped core sources match the trained snapshot. All67unrelated pending files
retain their original bytes; unrelated DEVELOPMENT sections remain unstaged.

Scope also carries the completed comparison archive and its compact melting
receipt to keep the existing provenance links valid; melting and Spinodal
numerical changes remain local. No raw arrays, weights, fonts or full galleries
are staged. Exact scope and checks: artifacts/channel_fem_publication_validation.json.
Next: commit on dev, regular merge main, push main and verify Pages/live bytes.


Content commit daa9c86 was regularly merged as main81ec0c4 and pushed to origin
main. Pages43547238968 started; build passed. Post-merge preservation checks
pass for67unrelated files and the full partial-staged DEVELOPMENT document.
Exact next action: await Pages deployment and compare9liveHTML plus4new assets
with the tested build. Then record the deployment, archive this task and push
only the final documentation receipt via the same workflow.


## Task-end: published and verified

Content daa9c86 was merged/pushed as main81ec0c4. GitHub Pages43547238968
completed build and deploy successfully. At2026-11-14T02:37:37UTC, all9live
HTML pages and4new PNG/PDF files returned200 and matched the tested build
byte-for-byte. The local preview server is stopped. No training was started.

Diff-level result: two reproducible field-wise figures, direct-path provenance
and publication manifests, validated optional channel FE implementation/configs
and focused tests, case/root README, asset guide, overview/channel site content,
build asset checks and Pages triggers. No numerical default changed. All67
unrelated pending files and partial DEVELOPMENT content were preserved.

Memory updated: PROGRESS, DECISIONS, this archived task and the direct
publication-validation receipt. Nineteen staged-tree tests, nine-page build,
271relative links, desktop/mobile views, all24sample errors/328monitor values,
font/edit probes, four-export byte reproduction and diff checks pass. There is
no remaining required work. New training or follow-up diagnostics require a
new user instruction. A final documentation-only commit records deployment;
it leaves the verified website bytes unchanged and does not trigger Pages.
