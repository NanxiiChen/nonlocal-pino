# Melting fixed/rolling spatial comparison

## Goal and acceptance

Compare fixed core+halo pool and background rolling core+halo
training for the canonical random-ellipsoid 3D neural operator. Keep the128^3
graph, physics, hard IC/latent residual, model, Adam/B16/Nt8 and online IC/time
streams identical. Use the original independent native reference inputs, report
pooled and terminal phi/solid metrics and synchronized cold/warm timing.
User explicitly cancels full training after its measured cost estimate. Retain
the completed3update full-gradient memory/throughput probe as performance only,
without a full5000accuracy claim. Preserve5000updates for both sampled controls.

## Context

Electropolish3matched5000runs complete; current default unchanged. Melting default
has2097152nodes,12484608directed edges, K256/pool32, only8172unique static PDE
cores. Fixed and rolling require safe1792node/1536edge caps. Direct full B16/Nt8
activation memory is large; measure a bounded exact chunked full path before
choosing launch orchestration. GPU RTX6000D85651MiB idle; disk9GiB free.
No deletion or Git history operation authorized. Existing data/tf-log paths
remain. Electropolish composed-attention CPU/GPU replay issue is recorded;
independently verify melting checkpoint replay without changing its architecture.

## Status / blocker / next action

COMPLETE with separately recorded query-chunk consistency limitation. Both
fixed/rolling5000runs exit0; actual IC/time/core/model+Adam parity, independent
saved-field metrics and fresh standard GPU replay pass. Full3update probe
exits0 (15.44s/step,38.314GiB) and user cancels full5000. An extra full-spatial
residual validation fails field-invariance2e-5gate; retain failure and report
only the verified sameGPU/1024query evaluation path. Root cause is deferred in
docs/issues/melting-query-chunk-consistency.md. No required follow-up/queued
training or blocker; all workers exit, GPU released. Defaults unchanged.

Initial focused FP32 check: complete rows/FIFO/overflow, weighted full loss and
gradients pass, but strict one-step Adam parameter2e-5level gap fails3e-6gate.
Likely near-zero cancellation-sensitive attention bias gradients; retain this
failure, use FP64 to independently isolate the exact mathematical accumulation
identity rather than loosen that gate. Production probes remain FP32.

## Planned validation

- Uniform independent core draws, complete6-neighbor halo, no truncation and
  deterministic background ordering; fixed/rolling initial32draws identical.
- One-hop core operator/residual/parameter gradient against full small graph;
  full chunks cover every grid node exactly once, weighted accumulation agrees.
- Same saved initial model/Adam, actual ellipsoid/time stream, held-out identities.
- Actual final GPU checkpoint field replay and independent NumPy physical L2.
- Separate native discrete-reference scope from previous one-sample local FEM
  check; no new training labels/reference trajectories required.
- Explicit mode config/README/API, compact receipts, scientific convergence and
  time/error charts external; git diff --check and task-end memory update.

## Completed changes and exact controls

Optional sample.spatial controls dispatch from canonical train.py through
operator_training to spatial_training.py; absent field retains the old path.
New spatial_sampling.py gathers all six-neighbor CSR edges with sorted-index
remapping and strict padding; no O(N)random permutation/remap per fresh draw.
Both modes use NumPyGenerator(draw_id).choice(N,256,replace=False), first32sets
identical; this deliberately differs from the historical RandomState stream.
Workers produce only NumPy metadata; FIFO draw identities are independent of
completion time. Fixed/rolling both pad1792queries/1536edges. No physical loss,
neural architecture, temporal strategy or optimizer formula changed. Canonical
base/operator config, physical loss and shared conditional module byte matches
HEAD; original make_step/predictor/metrics ASTs unchanged. Existing data/log and
checkpoint compatibility paths remain; no Git history action/deletion.

Full optional path accumulates core-count weighted gradient chunks before a
single update;2focused tests (49.084s) validate complete graph rows/FIFO/strict
overflow and independent FP64 full loss/gradient/Adam equivalence. Original
FP32 near-zero-gradient Adam failure remains recorded in validation JSON.
Full probe64chunks*32768cores, peak38.314GiB, warm15.44s/update; performance
only, no full5000accuracy. Scope cancellation came directly from the user.

## Measured results

Both5000Adam/B16/Nt8/ordinaryFP32, same128^3graph/epsilon/5physical seconds,
online80000distinct ICs,40000actual time values, zero future training labels.
Initial model and Adam byte-identical; all actual IC/time arrays and first32
core sets exact; every saved draw/selected core independently replayed. Original
eight evaluation ellipsoids remain disjoint from actual online draws. References
are independently time-refined native128^3solutions; physical L2measured only
at24^3diagnostic nodes/five times. No new reference/FEM solve.

| Measurement | Fixed | Rolling |
|---|---:|---:|
| Mean pooled phi relativeL2 |0.163060%|0.421714%|
| Mean terminal phi relativeL2 |0.221842%|0.251366%|
| Mean pooled solid relativeL2 |0.463445%|1.189876%|
| Mean terminal solid relativeL2 |9.992808%|8.117967%|
| Cold process |190.055s|190.650s|
| Warm synchronized effective update |12.7373ms|13.2472ms|
| GPU update alone |11.3551ms|11.2541ms|
| Online IC preparation |1.3298ms|1.2823ms|
| Spatial retrieval/transfer |0.0524ms|0.7108ms|
| Active GPU peak |1.49521GiB|1.49546GiB|
| Unique PDE cores |8174|958123|
| Core coverage |0.38977%|45.68686%|

Warm excludes first100updates and includes synchronized condition/spatial/GPU
work; cold includes process import/setup/JIT/evaluations/saving. Rolling total
producer wait0.02358s, transfer2.06039s;4.0033%warm overhead and negligible
starvation. Same-padding timing is not the tighter historical fixed-pool timing.
Rolling final pooledphi2.586times fixed; both below0.5%, late histories fluctuate.
No stable multi-seed ranking/fully converged floor or unique sampling-cause
claim. Two monitor curves differ from the eight final scores. No new default.

## Checks and material limitation

Fresh sameGPU/1024query replay on inputs2/5passes<=9.62e-7threshold2e-5.
NumPy per-input raw reference/field norm metrics and all aggregate values pass;
physical/reference/config/source/stream shapes and safe graph traces validated.
Both scientific PNG/PDF figures inspected; bar title/legend overlap corrected.

Supplemental full-spatial residual check fails because65536query physical
predictions differ0.005770858from standard1024query predictions at512identical
points (input2,4positive times). Repeat explicitencoded/noJVP, explicitencoded/
JVP, dynamic/capturedimage probes retain0.00576943..0.00576991. Input images and
initial latents exact. Cause not isolated, not silently called roundoff; no
production/compiler/model change. Failed gate retained, full-residual scalar
not used for ranking. Verified empirical comparison remains the identical
GPU/1024evaluation path. No CPU or query-partition independence claim.

## Artifacts and exact resume

Root: /root/autodl-tmp/tf-logs/icemelting3d.
Prefix20261031-melting-spatial-01, dirs-fixed_pool/-rolling_pool hold initial/
final model+Adam, config/source, actual sampling NPZ, per-input predictions,
metrics/timing/memory/spatial JSON. Each summary fixed5000point is retained.
Runner journal: prefix-pools-process.json; process logs retained. Comparisondir
prefix-comparison holds comparison.json, convergence.png/pdf, final_errors.png/pdf.
Performance run:20261031-melting-spatial-full-probe-01.

Repository scripts: melting_spatial_runner/finish/report/replay.py and explicit
failed melting_spatial_full_residual.py plus melting_spatial_chunk_probe.py.
Receipts: melting-spatial-results/validation/replay/finish/chunk-probe.json and
melting-spatial-full-residual-first-failure.json. Open issue records direct
reproduction. Large arrays/media stay external; no hashes or cleanup.

Task-end: focused tests/AST/default/source/actual-stream/field/replay/figure and
whitespace checks recorded. Memory: PROGRESS.md, DECISIONS.md, DEVELOPMENT.md
and this archived task; exampleREADME/artifact index synchronized. Full training
canceled, all processes exited, no follow-up automatically scheduled.
