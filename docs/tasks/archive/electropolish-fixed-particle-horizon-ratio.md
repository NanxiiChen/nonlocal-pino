# Fixed-particle Electropolish horizon-ratio control

## Goal and acceptance criteria

User requests an additional fixed-particle-count sweep of delta/h. Keep the
physical interface thickness, network, optimizer, sampling and evaluation
protocol of the completed horizon/density study. Final10000weights only;
64common local-FDinputs, per-channel whole-space-time relative L2, then equal
channel averaging and64sample mean/SD(ddof=1). Draw error versus delta/h.

## Context and declared matrix

Previous5151particle curve already covers ratios1.5/2/3/4. Add a finer fixed
201x101grid:20301particles, h=.005. Ratios1.5/1.75/2/2.25/2.5/3 correspond to
delta=.0075/.00875/.01/.01125/.0125/.015. Reuse the completed n201-d010run at
ratio2by direct path, not a new best-checkpoint search. Five other arms start
from the same cold state/seed42 and run10000updates. Preserve original cubic
normalization; record discrete moments and exact halo capacities.

Source baseline:
/root/autodl-tmp/tf-logs/electropolish2d/horizon-density-cold10k-20261112-02/runs/n201-d010

Use its saved configuration, not mutable production defaults. The largest
ratio3has approximately the same halo capacity as a prior29.2GiBprobe, but
perform50actual updates before launching the full queue. Do not retry the
known oversized ratio4on this grid. Existing train.py, explicit study monitor
mode and final64evaluator remain the numerical path.

## Status

Complete. The five cold10000 runs, five64-input evaluations and final audit
finished at2026-11-12T04:19:00UTC; the ratio2baseline is reused. All12stage exit
codes, including the preflight probe, are zero. The production queue took
67.585minutes excluding the probe and already-completed ratio2baseline.

| delta/h | Mean ± sample SD | Phase mean | Concentration mean | Warm ms |
|---:|---:|---:|---:|---:|
| 1.5 | 0.0505167 ± 0.0045517 | 0.0462050 | 0.0548283 | 34.02 |
| 1.75 | 0.0290676 ± 0.0114876 | 0.0265843 | 0.0315509 | 34.18 |
| 2 | 0.0172813 ± 0.0073978 | 0.0157545 | 0.0188080 | 45.03 |
| 2.25 | 0.0272266 ± 0.0059918 | 0.0248495 | 0.0296036 | 65.67 |
| 2.5 | 0.0234125 ± 0.0088989 | 0.0213570 | 0.0254680 | 65.84 |
| 3 | 0.0247605 ± 0.0073109 | 0.0225841 | 0.0269370 | 86.80 |

Ratio2has the smallest final mean for this fixed grid/thickness and training
budget. Increasing support beyond2costs more without improving this run.
The raw quadratic moment gain at1.5is0.8877; it exposes under-integration,
but the error curve includes both numerical bias and training effects.
All arms share initial model/fullAdam and actual physical coefficient/time/
core/IC/BC draw identities. Halo shapes intentionally differ, so this is not a
claim of identical FP32 fusion or encoder-image rounding. Single seed42only;
sample SD is neither seed variation nor confidence interval. Defaults unchanged.

The automatic final audit independently recomputed384sample norms from saved
prediction/reference arrays. This review additionally recomputed all per-channel
means and sample SDs from casewise records, checked complete stage exits/final
weights and inspected the PDF/PNG. The85×65mm curve has6–7ptembedded editable
TrueType text, vector-only PDF, NPG line/shading and automatic y limits. Its
x axis is delta/h, not training step. External outputs:

- figures/electropolish_fixed20301_horizon_ratio.pdf (20,663 bytes)
- figures/electropolish_fixed20301_horizon_ratio.png (50,882 bytes)
- report.json: complete paired-input/norm/plot audit.

Repository provenance artifacts/electropolish_horizon_ratio_study.json now
retains direct configurations, final weights, all384scores and validation.
The example README records the comparison and CPU reproduction command.

## Blocker

None.

## Next action

None for this experiment; archive after diff/link checks. Do not restart any
completed arm, resume the unrelated paused process618216, or promote a new
production default from this single initialization. No Git action authorized
in the results-review turn.

## Artifacts and implementation

Root: /root/autodl-tmp/tf-logs/electropolish2d/fixed20301-ratio-cold10k-20261112-01
Runner: docs/tasks/electro_horizon_ratio_study.py. Renderer:
scripts/render_electro_horizon_ratio.py. Repository provenance:
artifacts/electropolish_horizon_ratio_study.json. Source baseline stays in the
previous study root by explicit path. No weights/fields duplicated for ratio2.

Training code, PDEs and defaults unchanged. Shared result collector now accepts
an explicit normalization policy: old density policy still allows grid changes,
new policy rejects them and only permits delta/padding/outputpath changes.
Two focused tests verify strictness and no mutation. Final audit additionally
compares all10000core/time/IC/BC/condition-acceptance draws, plus cold weights/
optimizer and physical condition coefficients, with the reused baseline.

Actual quadratic moment gains for ratios1.5/1.75/2/2.25/2.5/3:
0.887704/1.047930/1.006431/0.945468/0.986370/1.000821. No correction is applied.
Expected halo capacities2304/2304/3328/5376/5376/7424. These are operator
diagnostics, not predictions of the final training error.

## Probe and launch verification

Maximum-ratio50updates completed, exit0; active peak26.3594GiB, warmmean82.31ms,
firstupdate100.645s. Initial model/complete Adam state bytes and first50actual
condition coefficient draws match reused ratio2exactly. Independent host replay
checks all6arms'fixedIC/BCnodes and core draws at0,32,9999against baseline.
Full10000draw checks subsequently passed in the completed final audit.

Owner PID498310(start2026-11-12T03:11:25UTC), detached session. Owner/journal/log
paths in root/owner.json. Five cold10k runs, five64evaluations, one final audit/
plot stage. Source and explicit configs frozen under root/source and configs.
Two focused config-policy tests pass, source/JSONsyntax and diff checks pass.
No production numerical change, new reference solve, cleanup or Git operation.

## Session handoff / task-end record

### User-requested figure/site work during the queue

[Melting arrow revision](melting-setup-arrows.md) is complete:
aligned short leaders, unchanged geometry/ice shading/type. V2exports and
provenance verified; no numerical or Git changes.

[Material-aware3Dmelting setup](melting-material-setup-preview.md)
is complete: actual prescribed ellipsoid, ice/water shading and cube Neumann
label. Geometry/font/export and visual checks pass; preview only, no numerical
or Git changes.

[Spinodal arrow revision](spinodal-setup-arrows.md) is complete;
short aligned leaders and compact canvas, unchanged geometry/initial texture.
No numerical or Git changes.

[Material-aware spinodal setup](spinodal-material-setup-preview.md)
is complete: same fixed ten-mode field with first prescribed two-hole geometry,
softblue/orange texture and hatched boundaries. Exports/provenance verified;
no simulation, canonical-asset replacement or Git action.

[Material-aware two-phase setup](channel-material-setup-preview.md)
is complete: generic blue/orange fluids, hatched solid walls and verified BCs.
Single existing random geometry, validated editable PDF/PNG/SVG. Presentation
preview only; no training or Git changes.

[Setup label simplification](setup-remove-concentration-labels.md)
removed c=h(phi) from both material previews. Exports and verification complete;
geometry, BCs, numerics and Git history unchanged.

[Ivory ceramic figure revision](ceramic-ivory-figures.md) is complete:
setup scope wording removed, main particles recolored to match ivory. Same
fields/scores, local README/site assets synchronized and validated. No Git or
numerical changes; this remains presentation work alongside the experiment.

[Material-aware ceramic-corrosion setup](particle-material-setup-preview.md)
is complete: same gray/blue palette plus ivory ceramic, actual six-particle
geometry and narrow mouth. 100×75mm/all7pt vector PDF/SVG and PNG validated.
Presentation preview only; numerical queue and existing published assets unchanged.

Subsequent user request: preview a material-aware Electropolish setup diagram
before extending the style to the other examples. The CPU-only preview is
complete; see [the preview record](electropolish-material-setup-preview.md).
Gray/blue materials and concise IC/BC labels, editable PDF/SVG/PNG validated.
No new Git action, figure promotion or numerical change.

The subsequent [PDF-annotation revision](electropolish-setup-annotation-revision.md)
is also complete: remove the small panels/title/prefix, use a two-line random
interface label and partial-derivative BCs. Preview only, no numerical changes.

[Compact typography revision](electropolish-setup-compact-type.md) is
complete at100×58mm/all7pt after the user found the previous figure too large.
Only presentation source/config/records changed; previous versions retained.

On 2026-11-12 the user requested ceramic restyling and website synchronization
with the five approved README main figures. The final approved style is gray
#A0A0A0 with dark #333333/0.45pt outlines; the darker preview was superseded.
This CPU-only side work completed under explicit publication authorization:
content6b7a889/mainmerge b85da60, Pages run43149212058success, all9liveHTML and
10PNG/PDFfiles byte-identical to the inspected build. Prior previews remain
external and all20panel errors are unchanged. See the archived
[presentation record](ceramic-main-website-sync.md). The ratio queue
remains the sole active task; no experimental source was committed.

The user subsequently approved promoting all five material setup schematics.
[README/site preparation](material-setups-readme-publication.md) is
complete: five final PNG/PDF pairs, matching README links and local website
captions, explicit manifest and reproduction index. Total 1.39 MiB; retained
font/size/byte checks and nine-page local build pass. No numerical changes,
commit, push or live deployment were performed by this documentation request.

The user then authorized publication. Main 3778e3d and Pages run43163004937
publish all five material setup pairs, their README/site records, and ivory
ceramic styling. Nine live pages and twenty PDF/PNG files match the inspected
build. See [the release record](material-setups-pages-release.md).
Only presentation changes were committed; this numerical task and source edits
remain local. The temporary publication worktree was removed.

Work is complete. New files: finite ratio runner, ratio renderer, two focused
policy tests, direct-path artifact manifest. Existing renderer collector now
accepts an explicit config-comparison policy; its previous density behavior is
retained. Memory changed: this task, PROGRESSandDECISIONS. Previous unrelated
Melting work and completed horizon-study edits remain untouched. Measured
queue67.6minutes; results and both figure formats were inspected and archived.
