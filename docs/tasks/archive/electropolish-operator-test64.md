# Electropolish operator: 64 unseen interfaces

## Goal and acceptance

Answer actual training cost and evaluation size; expand frozen image-only
operator evaluation from four to 64 new independently prescribed interfaces.
Retain the continuous online distribution, height acceptance, original IC,
physical equations, graph and refined reference tolerances. Report per-input
and aggregate phi/c pooled/terminal L2, dispersion, and evaluation cost.
Inspect local AD support and explain amortized encoding versus coordinate AD;
do not invent a local training time or accuracy result.

## Context and scope

Parent: archive/electropolish-operator-training.md. Frozen final checkpoint:
/root/autodl-tmp/tf-logs/electropolish2d/20261028-electro-operator-online-global-adam-01/model_final.eqx.
Previous mean of four unseen phi pooled/terminal5.573/4.372%, training315.83s
for5000Adam updates/B8. arch.local_input=zero; defaults remain unchanged.
New test data is not training/model selection. User suggests64or100; choose64.
No optimizer sweep, local full training, default change or Git operation.
Also measure a case-local PDE-loss plus parameter-gradient benchmark on the
same B8/Nt4/free core, sharing encoded images and time-independent spatial work.
This excludes IC/BC/optimizer/adaptive balance and is not a full training ratio.

## Status

COMPLETE.64identities/coefficients declared before solving in the separate
dataset manifest; no overlap with old reference identities or actual40000
training profiles. All64coarse/fine references pass original gates, max4.60e-8.
Original prescribed IC and recorded old-bank graph arrays match exactly
(IC tolerance3e-7), all predictions finite. Frozen final checkpoint only.

| Relative L2 | Mean | Median | P90 | Maximum |
|---|---:|---:|---:|---:|
|phi pooled|4.506%|4.253%|6.909%|10.541%|
|phi terminal|3.502%|3.002%|5.850%|8.210%|
|c pooled|5.323%|5.060%|8.024%|12.086%|
|c terminal|4.163%|3.585%|6.887%|9.641%|

Bootstrap95%meanCI for phi pooled4.093..4.940%, terminal3.142..3.887%.
These describe test-profile sampling at one frozen model, not training-seed
variation. Worst index56. New64input mean must not be called a training gain
over the previous four-input mean. All use101x51points/21times/0..5physical s.
References wall796.89s/fourworkers; summed solver3091.54s. Cold GPU evaluation
25.99s, first JIT+inference23.924s, subsequent full-field inference2.235ms mean.
Training remains315.83s/5000Adam/B8/40000new profiles, including timed JIT,
sampling and periodic small-bank evaluation, excluding reference preparation
and earlier setup. Large test bank does not slow the existing periodic monitor.

PDE benchmark validates directional Hessian traces against single-query full
Hessians in promoted precision, maxgap3.30e-16. SameB8/Nt4/245free cores,
nonlocal2453halo queries versus local245queries: median7.620/6.992ms; all losses
and parameter gradients finite. No IC/BC/optimizer/balance/full local training
or local accuracy result. Nonlocal timing overlapped CPU reference workers;
local warm timing followed their exit. Do not infer a stable8%speed advantage.
Local compile wall283.27s includes a deliberate SIGSTOP/SIGCONT interval while
references finish; raw wall is retained, not a pure compilation comparison.
Nonlocal firstcall109.23s also ran under competing CPU work. Current production
operator supports only sparse mode; this benchmark is an archived probe.

Read policy/memory and inspected sampling, references and model.
Current operator trainer accepts only sparse nonlocal mode. Encoder can be
shared for local coordinate AD via encode/prepare_encoded; local derivative
work must include query cross-attention, unlike the nonlocal difference path.

## Artifacts and session end

Dataset: /root/autodl-tmp/data/electropolish2d/operator_test64_20261028.
Contains manifest.json,64condition references/coarse arrays/metadata,
preparation.json, evaluation.json and prediction.npz (images/IC/predictions).
Scripts: [test preparation/evaluation](electro-test64.py),
[local component benchmark](electro-local-cost.py).
Raw JSON: [metrics](electro-test64-results.json), [test identities](electro-test64-manifest.json),
[cost](electro-local-cost.json). Raw stdout: electro-test64-preparation.txt,
electro-test64-evaluation.txt, electro-local-cost.txt.

Changes limited to these reproducible probes/results, example README,
artifact index and PROGRESS/DECISIONS/task memory. No numerical production
code/config, checkpoint, default, prior artifact or Git history changed.
Executed both scripts; final strictJSON/AST, all64independent physical-field
metric replays and aggregate checks, finite prediction shape64x21x51x101x2,
saved/current numerical-source byte parity, direct-path/archive/activeNone
checks and git diff --check PASS. No unrelated expensive suites rerun.
No blocker, queued training or exact resume action; all workers complete.
