# Electropolish NO full/fixed/rolling spatial sampling comparison

Date: 2026-10-31. Status: COMPLETE, GPU comparison with recorded CPU portability limitation. No Git history authorization.

## Goal and acceptance

User requests electropolish first: compare all spatial coordinates, a fixed
precomputed core+halo pool, and a background-generated rolling pool for training
time and accuracy. Implement optional strategies through canonical train.py;
preserve current defaults/physical model and historical saved-config loading.

Run three cold5000-update canonical NO controls, Adam/B8/Nt4/width64/ordinary
FP32, shared runtime seed42, identical initial weights, condition/time draws,
IC/BC coordinates, five losses and learning-rate schedule. Fixed and rolling
use the same safe padding; rolling starts from the same32subgraphs then consumes
new deterministic draws. Full uses every spatial free node. Evaluate the fixed
final models against the existing common64native/local-FD reference cohort.
Report phi/c pooled and terminal L2, training/setup/compile/update/producer wait
and transfer costs, peak memory and exact sampling coverage. No new training
labels, FEM jobs, default promotion, external cleanup or Git history operations.

Acceptance: focused strict-padding/producer determinism/operator-row/legacy
checks; actual-input and initialization parity; all3fullbudget runs complete;
saved checkpoint/field replay and independently recomputed common metrics;
comparison table and convergence figure; stable README/index/memory aligned.

## Context

GPU RTX6000D/85651MiB idle;25CPUbudget/~92GiB host RAM,9.2GiB diskfree. Existing
NO uses a fixed32entry K256pool on101x51grid;3938/4949free nodes ever become
cores. Current ICs/times are already online. Core+halo raw extraction is NumPy;
old _pad_entry can truncate and cannot be used for arbitrary future draws.
Electropolish SDF nearest lookup must follow changed query IDs. Need complete
one-hop phi/q graph dependencies, preserve native BC closure and loss reductions.

## Status / blocker / next action

COMPLETE. All3cold5000controls complete; all64native/local evaluations and actual
draw/config/initial model+Adam state parity PASS. Independent CPU field replay
fails with~0.3167absolute gap despite exact inputs/parameters and GPU checkpoint
replay <=2e-5. Current diagnostic isolates the first large gap to cross-attention
(encoded/features/projections agree~1e-6). A direct NumPy attention reference
matches CPU~1.4e-6 but differs from GPU~0.2088. Disabling cuDNNFMHA alone does
not change the GPU result. No production math/flags/model or trained results
changed. Do not silently relax the CPU threshold or call this audit passed.

The isolated exact-q/kv matrix probe agrees CPU/GPU within7.16e-7; the composed
path remains discrepant. NumPy attention matches CPU. The q/kv barrier and
Triton/cuDNN controls do not fix it. Exact cause is unresolved and explicitly
recorded in docs/issues/electropolish-attention-replay.md. Do not claim CPU replay
passed or interpret the small accuracy differences as backend-independent.
The requested GPU time/accuracy comparison is complete. No further experiment
or investigation is queued; no required resume for this task. Original failed
finish receipt/log retained alongside the later qualified validation receipt.

## Implemented and validated

Optional sample.spatial selects full/fixed_pool/rolling_pool via new explicit
configs and canonical train.py. Existing config absence keeps the legacy path.
Case-local spatial_sampling.py: strict complete-halo padding, one NumPy thread,
32prefetch entries, initial32draws shared then global-step-ID draws, main-thread
device transfer and SDF lookup refresh. Actual degree12 gives3328nodes/3072edges;
full uses5151nodes/60296edges,4949free cores. Model/PDE/loss/step function ASTs
unchanged and canonical config exact. Four focused CPUtestsPASS0.708s (complete
rows/SDFlookup, ordering, overflow, workerfailure). Independent64cached images
exactly reconstruct; initial/reference gap2.98e-8, native time gates<=4.60e-8.

## Exact artifacts and measured results

Run root /root/autodl-tmp/tf-logs/electropolish2d, prefix
20261031-electro-spatial-01. Suffixes full/fixed_pool/rolling_pool are run dirs.
Journal <prefix>-process.json status training_complete, runner65552exited.
Final64fields/results in each run's spatial_evaluation64.npz/json. Draw audit
docs/tasks/electro-spatial-run-audit.json passes all5000draws/all40000conditions/
ICBC/time/initialstate/source parity. Full and rolling cover4949/4949free nodes;
fixed covers3938/4949. No test checkpoint selection or reference solves.

Common64localFD101 mean pooledphi/terminalphi/pooledc/terminalc:
full4.4813/3.7741/5.2960/4.4908%; fixed4.9640/3.8333/5.8494/4.5412%;
rolling4.8578/3.6507/5.7312/4.3586%. Native ordering comparable. Cold process
wall4.10884/3.91536/4.09045min; warm effective step(after100, GPU+condition+
spatial)15.83361/13.21537/15.07792ms; peakactive3.74376/2.71917/2.72017GiB.
Rolling total producerwait0.02756s, so it is not producer-starved.

docs/tasks/electro-spatial-results.json and external <prefix>-comparison/
comparison.json contain exact results and independent64field metric replay.
Convergence.png/pdf uses the4training-time native diagnostic inputs, not64;
final_errors.png/pdf compares final64native/local. Both initially inspected;
bar legend/titles were refreshed for readability and visually reinspected.

## Audit/analysis orchestration

Research scripts docs/tasks/electro_spatial_{runner,finish,audit,evaluate,report}.py.
Finish93795exited on failedCPUreplay; docs/tasks/electro-spatial-finish.json
retains passed draw/3evaluation stages and failedCPUstage. Comparison report
was run separately and succeeds. CPUreplay log <prefix>-CPU_replay.log retained.
Temporary diagnostic scripts /tmp/electro-replay-{parts,spatial,matrices}.py,
with external comparison/replay-parts-{cpu,gpu,gpu-no-fmha}.npz,
spatial-parts-{cpu,gpu}.npz and attention-numpy-probe.json. Matrix probes saved
attention-matrices-{cpu,gpu}.npz; isolated outputs agree. Clean regeneration
script docs/tasks/electro_attention_replay_probe.py was executed onCPU and
reproduces0.316676field gap without modifying the checkpoint. No external deletion,
default promotion or Git history operation authorized/performed. Disk~9GiBfree.

## Final session / task-end

Diff-level changes: optional spatial schema validation and training dispatch/
timing/trace recording in operator_data/operator_training; new spatial_sampling
module and3research presets;4meaningful focused tests; reproducible serial
runner/evaluator/audits/report and a clean attention-issue reproducer. No network,
PDE, physical parameter, reference, optimizer or default numerical change.
All other existing model/loss/step functions match their pre-edit ASTs.

Stable example README documents explicit strategies, exact64native/local
metrics, timing scopes, matched safe padding and the recorded portability issue.
DEVELOPMENT, artifacts/index.yaml, PROGRESS and DECISIONS are updated. Receipts:
electro-spatial-{input-audit,run-audit,validation,results}.json. The validation
status explicitly says GPU-validated/CPUissue, retaining the failed2e-5gate.
Actual GPU checkpoint max gaps full3.52e-6/fixed0/rolling4.09e-6; all64field
metrics and statistics independently replay. No duplicate expensive training or
new reference solve. Python/JSON/links/whitespace checks pass, allworker/runner/
training/evaluation processes exit, GPU released. Generated media/arrays/model
states and copied pre-edit sources remain external; no cleanup or Git action.

Memory files changed: PROGRESS.md, DECISIONS.md, DEVELOPMENT.md and this archive.
Only optional future work is the separately recorded composed-attention issue;
no automatic job or new default is selected by this result.
