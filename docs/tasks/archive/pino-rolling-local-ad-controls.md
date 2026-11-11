# Current rolling PINO nonlocal/local-AD controls

## Goal and authorization

The user authorizes executing the preceding comparison plan, with notification
if the current GPU cannot support a useful experiment. First execute the two
ready cases: selected six-particle350s/width256/Adam20k and current rolling
Electropolish width96/Nt16/B8/Adam10k, each with two paired cold seeds. Preserve
production defaults and the closed thin-interface work. No Git/history or
cleanup operations are authorized.

## Acceptance

- Explicit train.py paths for both methods; same physical problem, model,
  cold state, optimizer, actual initial/geometry/time/core/data draws per pair.
- Correct local Neumann conditions at positive times, existing physical
  normalization and efficient independent-query spatial JVPs.
- Focused tests and full-update memory/timing probes before the finite queue.
  Serialize GPU work and do not run FEniCS/MPI simultaneously.
- Evaluate common local references, record all failures and field/interface/
  boundary errors, plot error versus updates and wall time. Report comparison
  harness/operator-preparation overhead honestly.
- Bounded external artifact footprint, direct paths/source snapshots, no hashes.
  Keep all existing work and results. No new method/default promotion.

## Context

[Completed inventory](pino-local-ad-comparison-plan.md) identifies exact
implementation gaps and historical limitations. Particle online rejects its
old method-pair CLI; Electropolish rolling pool rejects local AD. Both kernels
already exist. Current references are reused development cohorts; do not claim
fresh blind generalization without a separate reference expansion. Spinodal
mixed formulation/common local references are a subsequent stage, not silently
treated as already implemented.

Device: RTX4090,49140MiB,UUID GPU-a7b60c6d-62ad-e7f5-f91f-b750cc853621,
initially idle. Disk about20GiB free at2026-11-11. External tf-logs/data paths
remain unchanged. Existing paused process group618216 is unrelated.

## Status, blocker and next action

Completed with an explicit pairing qualification. All eight cold trainings,
all local-reference evaluations and saved-field recomputations finished.
Three strict pair audits passed; Electropolish seed43 failed the actual-input
value gate. Its physical coefficients and complete actual draw streams match,
but FP32 encoder/IC witnesses differ slightly. The original journal remains
failed, and the unqualified final report was never issued. See the final
results and independent prescribed-input diagnostic below. No training is
pending; GPU is idle. Defaults remain unchanged. Any causal attribution or
fully bitwise-input rerun is a separate follow-up, not an automatic continuation.

## Implementation and launch (historical record)

Implementation and focused CPU verification complete. Particle adds strict
`--rolling-method-control`, actual core traces and independently replayed online
input audits. Electropolish enables the same rolling cores for local AD without
halo queries, with cold-state/input/source audit receipts. Both preserve the
production recipes; explicit research presets select the method comparison.

Canonical serial resource probes completed successfully from09:49:49UTC in
screen session `pino-rolling-ad-probe-20261111-01` (former owner PID292250):
`/root/autodl-tmp/tf-logs/pino-rolling-ad-probe-20261111-01/journal.json`.
All four arms and both strict method/input audits pass. Particle251updates cover the first
geometry replacement; Electropolish65cover spatial replacements and weight
refreshes. Explicit allocator policy is90%preallocation, with active peak
checked separately from reserved memory. Probe states are discarded; all
formal runs remain cold. No FEM/reference solve is launched.

The finite owner `docs/tasks/pino_rolling_ad_runner.py` serializes four resource
arms followed, once explicitly started, by eight full runs. Before each stage
it verifies direct source bytes, GPU identity, disk headroom and paired input
requirements. Each Electropolish arm evaluates5k and10k on the same64local
targets. `pino_rolling_ad_report.py` recomputes final field metrics and adds
positive-time interface/near-wall diagnostics and labeled update/time curves.
The particle curve uses its4local-FEM monitors; the dense Electropolish curve
uses4native monitors, with64local anchors evaluated separately. They must not
be mislabeled as the same evaluation measure.

Probe results (warm means start at update index25 and include gradient-weight
refreshes; active peak includes initial/final monitoring, not allocator reserve):

| Case | Method | Warm mean ms/update | Active peak GiB |
|---|---|---:|---:|
| Particle | nonlocal | 93.60 | 20.43 |
| Particle | local AD | 80.25 | 11.19 |
| Electropolish | nonlocal | 43.44 | 13.00 |
| Electropolish | local AD | 28.21 | 5.31 |

Current hardware supports all canonical shapes. Keep the model/batch unchanged;
no GPU switch is needed. These are short resource probes, not converged
accuracy comparisons. Local AD is faster and uses less active memory here;
do not presume a nonlocal speed advantage. Separate shared unused graph
preparation overhead from the local update cost. Probe receipt:
`/root/autodl-tmp/tf-logs/pino-rolling-ad-probe-20261111-01/resource-results.json`.

Formal eight-run cold queue launched at10:07:51UTC, owner PID308774,
screen session `pino-rolling-ad-cold-20261111-01`. Journal:
`/root/autodl-tmp/tf-logs/pino-rolling-ad-cold-20261111-01/journal.json`.
Initial child PID308778 is particle seed42 nonlocal20k. Order: particle seed42
NL/AD, particle seed43 NL/AD, then Electropolish seed42 and43 NL/AD. Each pair
is audited before advancing; all Electro5k/final64case evaluations and the
combined report are in the finite queue. Estimated wall time about3hours from
measured warm costs plus fresh compilations/evaluations. This is an estimate,
not a completion guarantee or an accuracy result.

Launch-session check confirmed the first formal run had completed at least500
updates, finite objectives/gradients, with active peak20.42GiB. Runtime receipt
confirms cold initialization, seed42, rolling inputs and no evolved-reference
targets in training. The first positive-time evaluation is scheduled at1000;
no formal accuracy comparison is available yet.

Exact formal owner command (already running; do not duplicate):
```bash
.venv/bin/python -m docs.tasks.pino_rolling_ad_runner queue \
  --output /root/autodl-tmp/tf-logs/pino-rolling-ad-cold-20261111-01 \
  --probe /root/autodl-tmp/tf-logs/pino-rolling-ad-probe-20261111-01
```

Next resume: read the formal journal and current stage log, report latest
completed local-reference metrics with steps, and check source/disk/failure
state. Let the finite queue finish; do not edit its frozen numerical/runner/
evaluator sources while active. On failure inspect the exact stage rather than
silently restarting or changing the setup. Final output is `results.json` plus
`convergence.png` in the formal owner root, generated only after all8arms pass.

## Interim status at 2026-11-11 10:45 UTC

One of eight full runs completed: particle nonlocal seed42,20k. Process wall
34.38min, warm mean93.48ms/update, active peak20.44GiB. Independent CPU
recalculation from all saved fields exactly reproduces the same4local-FEM
development-case metrics:

| Metric | Space-time mean | Terminal mean |
|---|---:|---:|
| Phase relative L2 | 3.1802% | 6.0974% |
| Concentration relative L2 | 3.8689% | 7.3071% |
| Corroded fraction relative L2 | 7.5709% | 9.4137% |

Terminal IoU94.6679%; individual pooled phase errors2.9672--3.6599%.
Pooled phase at5k/10k/15k/20k is10.8010/4.6429/3.5326/3.1802%.
Improvement continues with late fluctuations; this is not evidence of a fully
converged plateau. Independent online audit passes:316newgeometries,
380distinct consumed geometries,1520spatial replacements,64final slots exactly
reconstructed, no evaluation overlap or evolved training labels.

Current child313526 is particle localAD seed42, in first full-update compilation
at the status check. No failure or OOM; remaining6runs follow this arm.
The queue remains unchanged, running under owner308774. Formal AD/native-method
accuracy comparison is unavailable until that arm finishes. Do not substitute
probe accuracy or historical results. Approximate remaining wall2.5hours.
Interim convergence image:
`/root/autodl-tmp/tf-logs/pino-rolling-ad-cold-20261111-01/interim-particle-nonlocal-s42.png`.
Status-only session updated this task and `docs/PROGRESS.md`; no numerical code,
defaults, queue settings or Git history changed.

## Verification

### AD trend check at 2026-11-11 10:57 UTC

Particle seed42 AD has reached the6k evaluation while the20kqueue continues.
The paired cold model/optimizer bytes and all planned condition/pool/time arrays
were independently compared and match. Full consumed rolling/core audits still
await AD completion. The same4local-FEM cohort gives:

|6k metric|Nonlocal|Local AD|
|---|---:|---:|
|Pooled phase L2|9.2015%|33.4528%|
|Terminal phase L2|14.0101%|46.7024%|
|Pooled concentration L2|10.9533%|33.5908%|
|Terminal concentration L2|16.3935%|49.7043%|

AD pooled phase at1k/2k/3k/4k/5k/6k is32.2840/33.1050/33.5692/
33.3338/32.7090/33.4528%. It drops during the first1k, then fluctuates near33%
without sustained improvement. Nonlocal instead improves substantially after4k.
This is an interim field-error plateau, not numerical divergence or evidence
that local AD cannot solve the PDE. Retain the declared20k budgets and second
paired initialization; no optimizer/loss/BC modification made during the run.

Matched-through6k curve and exact source metric snapshot:
`/root/autodl-tmp/tf-logs/pino-rolling-ad-cold-20261111-01/interim-particle-pair-s42-through-006000.png`
and same stem`.json`. Both plot arms stop at6k; no mismatched-budget endpoint
comparison. Owner308774 remains active; remaining cases follow automatically.
Status session changes only this task and `docs/PROGRESS.md` plus the external
plot/snapshot; `git diff --check` and finite/order/paired-state checks pass.

### AD trend check at 2026-11-11 11:10 UTC

AD seed42 has completed the16k evaluation and at least16.3k actual updates,
remaining finite. It has not escaped the field-error plateau: pooled phase
at6k/10k/16k is33.4528/33.8770/33.2514%. At the same16k budget and same4local-FEM
development cases:

|16k metric|Nonlocal|Local AD|
|---|---:|---:|
|Pooled phase L2|3.4211%|33.2514%|
|Terminal phase L2|6.4257%|46.6720%|
|Pooled concentration L2|4.1629%|33.4065%|
|Terminal concentration L2|7.7105%|49.7851%|
|Terminal IoU|94.2678%|27.4288%|

The field-error gap persists in this paired initialization; its cause has not
been isolated to Neumann enforcement versus interior optimization. Keep the
original queue/budgets. Owner308774 is live with AD child313526; about5minutes
remain in this arm, then second particle seed and both Electro seeds follow.
No OOM/failure and about20GiB disk free.1/8full arms complete at the check.
Matched-through16k figure and metric snapshot:
`/root/autodl-tmp/tf-logs/pino-rolling-ad-cold-20261111-01/interim-particle-pair-s42-through-016000.png`
and same stem`.json`. The prior chat's6kimage URL omitted`-tmp`; use these exact
existing paths in subsequent responses. Status session changes only this task,
`docs/PROGRESS.md` and external chart/metric snapshot; numerical sources remain
frozen. Finite/order checks and `git diff --check` pass.

### First full pair timing at 2026-11-11 11:16 UTC

Both particle seed42 arms have completed20k and their separate online audits.
Measured synchronized warm means (index>=25, including weight refreshes):
nonlocal93.4795ms, localAD80.3344ms. Held-step means84.4472/70.1522ms;
refresh-step means310.2532/324.7067ms. First update including compilation takes
89.0399/299.4509seconds. Whole training child processes take2062.5264/2007.2127s
(34.3754/33.4535min), including setup, compilation, training, monitoring and
artifact output, excluding the subsequent independent audits. AD uses14.06%
less warm-update time, but only2.68%less cold-run wall time due to compilation.
Shared graph preparation remains an avoidable AD harness cost.

AD final same4local-FEM metrics: phi33.4302%pooled/46.7399%terminal,
c33.7391%/49.8812%, terminalIoU27.1218%. Nonlocal final phi3.1802%/6.0974%.
Thus this first pair shows a field-error/convergence difference, not a nonlocal
per-update speed gain. Final paired audit/report passed; saved-field comparison
is in`particle_corrosion2d-s42-comparison/comparison.json`under the owner root.
Particle seed43nonlocal is now running as child321895. Electropolish remains
pending. No changes to training settings.
Status/timing session changes only this task and `docs/PROGRESS.md`.

### Completed three pairs at 2026-11-11 13:01 UTC

Seven of eight training arms have completed; Electro seed43nonlocal is in its
5k/final64case evaluation stages. The last AD seed43arm follows. Owner308774
remains healthy; disk about19GiBfree. About15minutes estimated remain, including
the last compilation/training/evaluation and final aggregate report.

All four particle20k runs and both Electro seed42ten-thousand-step runs were
independently recomputed from saved physical fields with the common local
references. Both particle input/state/core audits and Electro seed42pair audit
pass. Phase relative L2 percent, pooled / terminal:

|Case, seed|Nonlocal|Local AD|
|---|---:|---:|
|Particle42,4localFEMcases|3.1802 / 6.0974|33.4302 / 46.7399|
|Particle43,same4localFEMcases|3.0211 / 5.6398|33.5883 / 44.8020|
|Electro42,64localFD101cases|1.8625 / 1.5630|4.2615 / 3.0505|

Particle concentration pooled/terminal: NLseed42 3.8689/7.3071%,
NLseed43 3.6795/6.7542%; ADseed42 33.7391/49.8812%, ADseed43
33.9961/47.7186%. Both particle AD optimizations plateau; the same ordering
recurs across these two initializations, without establishing the physical
cause or a population-wide theorem. Four reused geometries remain a limitation.
Electro concentration pooled/terminal: NL2.2309/1.8669%, AD5.0003/3.6058%.
Electro pooled phase at5k->10k improves4.3592->1.8625%(NL),9.0486->4.2615%(AD).
Unlike particle, this AD arm continues meaningful improvement; do not describe
all AD controls as failed or intrinsically inaccurate.

Particle process wall minutes NL34.375/34.390, AD33.454/33.584; warm mean
NL93.48/93.51ms, AD80.33/80.47ms. Electro seed42 wall10.436minNL versus11.512minAD,
while warm steps42.74/28.80ms: AD's larger cold compilation outweighs its faster
updates at this budget. Total excludes the separate64case post-evaluations and
pair audits and includes the explicitly shared graph-preparation harness.

Updated curve and direct metric snapshot:
`/root/autodl-tmp/tf-logs/pino-rolling-ad-cold-20261111-01/interim-completed-pairs-20261111-1301.png`
and same stem`.json`. Left uses the complete4local-FEM histories for both seeds;
right uses only actual64local-FD evaluations at5k/10k of the first pair. No native
monitor/local-cohort mixing. Status-only edits: this task and`docs/PROGRESS.md`,
external plot/snapshot. No frozen numerical/queue source or default change.

### Implementation verification

Particle original AD oracle/gradient/rolling regressions and10new pair-audit
tests pass. Electropolish10focused sampling/pairing/AD-Hessian/BC checks pass.
Common local-reference identity tests pass for both methods and reject changed
physics/horizon metadata. Root owner/resource/report contracts replay existing
completed20kparticle and10kelectropolish artifacts successfully, including all
64saved local-field norms. Changed modules compile and `git diff --check` passes.

## Final results and pairing qualification — 2026-11-11

All8training arms and all reference-evaluation stages finished successfully.
The finite owner ended at13:13:51UTC after186.0minutes, stopping at the last
pair audit. Three audits passed; Electro seed43 failed. Its raw journal/logs
and all frozen training, evaluation and audit sources are preserved unchanged.
The supplemental CPU report independently recomputed every final saved-field
headline norm and positive-time regional norm for all8arms. No new inference,
training or reference solve was needed.

Common local-reference results, mean casewise relative L2 in percent:

|Case / seed|NL phase pooled / terminal|AD phase pooled / terminal|NL concentration pooled / terminal|AD concentration pooled / terminal|
|---|---:|---:|---:|---:|
|Particle42,20k,4localFEM|3.1802 / 6.0974|33.4302 / 46.7399|3.8689 / 7.3071|33.7391 / 49.8812|
|Particle43,same cohort|3.0211 / 5.6398|33.5883 / 44.8020|3.6795 / 6.7542|33.9961 / 47.7186|
|Electro42,10k,64localFD101|1.8625 / 1.5630|4.2615 / 3.0505|2.2309 / 1.8669|5.0003 / 3.6058|
|Electro43,same cohort,qualified pairing|1.9256 / 1.3875|3.3320 / 2.3176|2.2908 / 1.6627|3.9340 / 2.7408|

Electro5k->10k pooled phase: seed42NL4.3592->1.8625%,
AD9.0486->4.2615%; seed43NL3.5822->1.9256%,AD5.6116->3.3320%.
AD still improves on Electro, while both particle AD histories plateau near33%.
Particle terminal corrosion-region IoU: NL94.67/95.15%,AD27.12/32.03%.
These finite-budget results do not establish intrinsic AD inaccuracy or
isolate boundary enforcement as the cause. Only two initializations and reused
development cohorts were evaluated; particle has only4reference geometries.

Same-current-device timing, averaged across the two seeds:

|Case|NL warm ms|AD warm ms|NL cold process min|AD cold process min|NL / AD active peak GiB|
|---|---:|---:|---:|---:|---:|
|Particle20k|93.50|80.40|34.38|33.52|20.44 / 11.21|
|Electro10k|42.77|28.54|10.46|11.47|13.00 / 5.31|

Warm updates include weight refreshes and are synchronized. Cold process time
includes setup/compilation/training/monitoring/output, excludes independent
post-evaluation/audits, and retains shared unused graph preparation for AD.
Electro AD's first update is320.39/322.20s versus about111s for NL; its faster
updates do not yield a shorter cold10kprocess. Do not claim a nonlocal per-step
or memory advantage. GPU is idle,49140MiB RTX4090; disk retains about19GiBfree.

Regional phase L2, positive saved times and physical spatial weights, averages
across cases within each seed:

|Case|NL near-wall seed42/43|AD near-wall seed42/43|NL interface seed42/43|AD interface seed42/43|
|---|---:|---:|---:|---:|
|Particle|4.482 / 4.268%|30.959 / 31.236%|17.458 / 16.473%|67.204 / 67.811%|
|Electro|3.011 / 2.972%|8.137 / 6.121%|8.609 / 9.008%|19.392 / 15.206%|

Interface masks use reference phase in(.1,.9), solely for evaluation. Boundary
strips have width equal to the configured horizon; particle uses distance to
sampled wall tokens excluding the reservoir opening, Electro left/right walls.
These region errors support the field-error ordering without proving its cause.

The failed Electro43pair gate is specifically `Paired witness/images values
differ`. All10000actual profile coefficients, accepted counts, times, rolling
cores and IC/BC coordinates, source snapshots and cold network/Adam bytes
match. Recorded image values atupdates0/32/9999 differ by at most8.9407e-7
(RMS3.2926e-8); initial targets differ by1.2517e-6(RMS5.4178e-8).
Dirichlet values and witness coordinates are exact. Independent NumPy FP64
evaluation of the same recorded profiles and windowed segment SDF places
both image witnesses within3.2804e-6 of the common reconstruction; IC gaps
are2.9201e-6(NL)/2.9797e-6(AD). This is compatible with FP32 arithmetic
differences, but their optimization effect has not been isolated. No post-hoc
tolerance increase or strict-pass relabeling was applied.

Supplemental reproduction, leaving strict sources and raw journal intact:
```bash
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  -m docs.tasks.pino_rolling_ad_qualified_report \
  --root /root/autodl-tmp/tf-logs/pino-rolling-ad-cold-20261111-01
```

Artifacts under that root: `qualified-results.json` (explicit qualified status,
all metrics/regions and input diagnostic) and `qualified-convergence.png`.
The figure uses full particle local-reference histories and ONLY the5k/10k
Electro64local anchors, with no mixing of native monitor and local metrics.
The original unqualified `results.json` and `convergence.png` remain absent.
All final norms recompute successfully, the final plotting path was inspected,
and the original journal still records its failure. Training defaults unchanged.

## Session-end record

Implemented explicit rolling derivative presets/audits in both examples,
strict shared local-reference evaluation, finite resource/training owner and
combined regional/convergence reporting. Example README research sections
distinguish this completed comparison from old results; defaults unchanged.
All four canonical probes passed; all8full trainings and local evaluations
completed. Added separate qualified-report code and actual-array FP64 input
diagnostic; preserved the three strict passes and fourth strict failure.
README study pointers reflect completion; no production numerical sources
changed during final reporting. Archived this task; updated `docs/PROGRESS.md`
and `docs/DECISIONS.md`. No Git/history operations or old-artifact cleanup.
