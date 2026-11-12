# Main-figure publication and fixed-thickness horizon/grid study

## Goal and acceptance criteria

User explicitly authorizes commit/push of the accepted five main figures, then
requests a new PINO experiment varying nonlocal horizon and particle density
at fixed phase thickness. Use final-step weights and64common test samples;
plot relative L2 mean plus one sample standard deviation. Prior metric applies:
per-channel relative norm over all evaluation space-time points, channel mean
per sample, then mean/std over64samples. No percentage rescaling.

## Context

Five completed publication previews have direct specifications/receipts under
artifacts/*_main_figure*.json and external PDF/PNG outputs. Existing pending
Melting thin/extreme numerical code is unrelated to this figure release.
Git starts on main c48fa06, dev7b89e89, origin points to nonlocal-pino.
Required publication workflow: dev commit -> main regular merge -> push main.

Electropolish is the candidate because local-FD64targets already exist and
parameterized geometry does not confound mesh/horizon controls. Keep initial
interface distribution, physical coefficients/thickness, architecture, optimizer,
core count and training budget fixed. Audit quadrature/moment quality and actual
halo sizes before choosing affordable δ/h combinations. No new training labels.

## Status

Figure publication complete: dev2e1789c, regular main merge04ea141e92d0c792b41e377aae5fde9ba184bdda;
origin/main and remote ls-remote verified equal. Ten approved PDF/PNG files total
8,306,343bytes in assets/pino-main, with five example links and exact provenance.
Unrelated pending Melting numerical changes preserved outside the commit.

Completed: all9cold10000runs, all9final64evaluations and independent audit/render
exited0. All576sample scores independently replayed; cold model/optimizer bytes,
condition coefficients and time draws match. PDF/PNG visually inspected; all
text6–7pt, fonts embedded,170×59mm, all-vector PDF. No new training is pending.

## Experiment

Root: /root/autodl-tmp/tf-logs/electropolish2d/horizon-density-cold10k-20261112-02
Finite owner: docs/tasks/electro_horizon_study.py; training enters train.py.
Nine unique cold10000arms, seed42, width96/Adam/B8/Nt16/core256/rolling pool32.
Fixed h=.01: delta=.015/.02/.03/.04.
Fixed delta=.02: Nx101/129/151, Ny51/65/76.
Fixed delta/h=2: Nx51/101/151/201, Ny26/51/76/101.
All physical coefficients, interface profile thickness, initial-input distribution,
encoder sensor resolution and optimization settings unchanged. Particle count
means the full spatial discretization, not the number of PDE cores per update.

Final headline: same64local-FD101references, same101x51querypoints, same21uniform
inclusive0..1times, raw per-channel full-space-time relative L2 -> channel mean
-> sample mean and SD(ddof=1). Four FD201/401checks also retained. These are the
same21times used by published final violins; four extra early reference snapshots
are not included in either main metric. Small native6monitor bank is fixed and
not a same-graph reference for changed arms; never used in the objective.

Operator audit: actualFP32cubic kernel, no moment correction. Interior quadratic
moment gain at delta/h1.5 is0.887704; at2 is1.006431; at3 is1.000821; at4 is1.002944.
This directly documents a sparse-neighborhood quadrature confound rather than
silently changing normalization. Maximum20,301particles,48neighbors; largest retained padded halo7,424nodes. Trace/cold-state identity will be checked after all arms finish.

## Blocker

None. Complete.

## Next action

None required. Optional multi-seed/final-tail stability controls are not launched.
Current production default and selected publication main figures remain unchanged.

### Probe corrections

Real CLI locks ConfigDict: fixed explicit-monitor clone deletion under unlocked
context; focused test now covers locked config. First allocation probe reached
42.11GiBcompiled requirement and exceeded default75%JAXquota. All9arms now
explicitly use95%allocator with no preallocation, on idle48GiB4090. Failed logs
and partial run are retained. Second50step probe uses probe-fraction95.json.

### Memory-qualified matrix revision

Nx201/delta.02fails before update1 even at95%allocator (42.44GiBcontiguous step
allocation plus existing state exceeds available capacity). Preserve both failed
probes under ...-01; do not call this a training divergence. Replace that unrun
arm with Nx129/Ny65/delta.02before looking at accuracy. Fixed-delta particles
5151/8385/11476; keep Nx201/delta.01 in coupled refinement. New final root...-02.
Largest retained graph Nx151/delta.02is undergoing50updates. No batch, time count,
network, optimizer or core changes. No GPU swap needed if this probe passes.
Fourteen distinct focused unittest checks pass (10operator/horizon +4spatial);
pytest unavailable, so existing unittest framework used.

### Full queue launched

Owner PID423144, process-independent session, owner.json/log and journal.jsonl
at final...-02root. Probe Nx151/delta.02peak29.1987GiB, firstupdate95.949s,
warmmean79.35ms, held median75.05ms, GradNormrefresh243.13ms. All50updates finite.
Nine10k runs plus9final64evaluations and one independent audit/render stage.
Estimated total1.5–2hours on this card; no promised completion timestamp.
Training/checkpoints/raw fields remain external. Final plot170x59mmPDF/PNG,
Helvetica5–7pt/NPG, raw ratios, automatic ylimits, one sampleSD.

### Session handoff / task-end record

Changed publication assets/READMEs/provenance are committed and pushed. New
uncommitted numerical work: explicit horizon-study monitor module/config, small
training-grid/eval-grid separation, finite runner/evaluator, CPU audit/plotter,
4focused horizon tests and study manifest. No production default/PDE/kernel
change. Existing unrelated Melting edits preserved.14distinct focused tests,
AST/JSON/diff checks pass; largest retained50step probe completes. Existing
isolated PyMuPDF1.26.7at/tmp/pino-pdf-auditis an explicit manifest dependency;
no package installation. Full queue is still active, not a completed experiment.

Next reader: check owner PID423144, journal.jsonl, owner.log, then any current
logs/<arm>-train.log. Final report/figures appear only after9complete runs and
64-inputevaluations. Inspect real PDF/PNG and report, update README only with
qualified results, archive this task when finished. Do not treat native6monitor
metrics as the requested local64headline. Memory updated: PROGRESS, DECISIONS,
this active task. No second commit/push of unfinished experiment code.

## Completed results and verification — 2026-11-12

| Particles | Horizon | Horizon / spacing | Mean ± sample SD | Warm update (ms) |
|---:|---:|---:|---:|---:|
| 5,151 | 0.02 | 2 | 0.02316 ± 0.00613 | 42.8 |
| 5,151 | 0.015 | 1.5 | 0.04398 ± 0.00587 | 31.9 |
| 5,151 | 0.03 | 3 | 0.03691 ± 0.00653 | 61.3 |
| 5,151 | 0.04 | 4 | 0.05269 ± 0.00634 | 61.6 |
| 11,476 | 0.02 | 3 | 0.03161 ± 0.01010 | 82.3 |
| 8,385 | 0.02 | 2.56 | 0.04640 ± 0.01572 | 63.8 |
| 1,326 | 0.04 | 2 | 0.18194 ± 0.03158 | 19.3 |
| 11,476 | 0.0133333 | 2 | 0.02456 ± 0.00750 | 43.7 |
| 20,301 | 0.01 | 2 | 0.01728 ± 0.00740 | 45.0 |

Reference: all64local-FD101, same21uniform times, physical(phi,c)channel mean.
Runtime109.28min for19stages, all exit0. No GPU process remains from this queue.
Best n201-d010mean0.0172813,SD0.0073978 versus current cold baselinemean0.0231593,
SD0.0061297:25.38%relative reduction, warm updates+5.17%. Phase-only bestmean
0.0157545,concentration0.0188080. Four-caseFD101/401mean forbest0.0203576/0.0191092;
all four-case rankings unchanged under refinement. These are not64refined cases.

Fixed-delta density scan is nonmonotone and includes optimization differences;
Nx129native4monitor phase9k0.03592->10k0.05994. Keep finalweights; do not cherry-pick
9k or call density harmful in general. Whole-field results include graph boundary,
quadrature, representation, sampling and optimizer errors. One shared seed with
matched physical draws is not cross-seed stability; actual query/ICnodes differ.

Final report: root/report.json; figures/electropolish_horizon_density.pdf/png.
Repository compact record: artifacts/electropolish_horizon_study.json, now holds
all576per-sample scores, mean/std, configs/checkpoints/arrays/direct figure paths.
Example README adds a qualified results table, controls and reproduction command;
no default promotion, new reference solve, training rerun or Git operation.

Task-end: archived this completed task; PROGRESSandDECISIONSupdated. Retain all
previous source changes and unrelated Melting edits. Previously passed14focused
tests were not rerun for documentation-only completion; JSON/links/diff checks
and actual PDF inspection are the relevant current checks.
