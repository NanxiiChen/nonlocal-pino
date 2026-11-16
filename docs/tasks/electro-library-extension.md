# Electropolish fixed32 and online repeat

## Goal and authorization

User defers Ceramic stability work to TODO and requests two further Electropolish
runs:32fixed initial profiles and a new continuous-online run, evaluated on the
same64test conditions. Preserve prior fixed64/128/256/512results and defaults.

## Acceptance and controls

- Both cold15ksteps, retaining a10kcheckpoint, with seed42 and identical network/optimizer initialization.
- Fixed32takes the first32entries of the original nested512profile library;
  online generates new profiles at every update from the unchanged distribution.
- Same frozen numerical source as the prior fixed-library sweep; same physical
  parameters, width96/B8,16random times, Adam, rolling256core/halo sampling,
  GradNorm and host_float64_shared prescribed-input computation.
- Same64local-FD test references,21regular evaluation times, monitoring every250
  updates including zero,10kand15k. Final-weight equal-point full-space-time norms
  perphi/cchannel, then arithmetic channel/sample mean; one sampleSD, ddof1.
- Independent final inference/norm checks, shared sampling/initialization checks,
  full cold process timing and explicit hardware provenance. No automatic retry
  or preferential seed/weight selection to make online appear best.

## Context and direct paths

Previous bundle: `/root/autodl-tmp/tf-logs/fixed-condition-library-20261115-01`.
Nested coefficients: `/root/autodl-tmp/data/fixed-condition-library-20261115-01/electropolish`.
Reference inputs: `/root/autodl-tmp/data/electropolish2d/operator_test64_20261028`.
Local FD references: `/root/autodl-tmp/data/electropolish2d/operator_local_test64_20261029`.
Prior matched online run: `/root/autodl-tmp/tf-logs/electropolish2d/electro-threeway-20261114-03-nonlocal`.

Current GPU: NVIDIA RTX6000D,85651MiB,UUIDf0c22930-0180-a523-898a-9fef5fec4eb9.
It differs from the previous RTXPRO6000UUID85fe6412-636e-5d3c-d5f0-d1e9523c3faa.
Both new arms use the same current device. Historical times are not a controlled
cross-arm comparison; retain all hardware details. User explicitly prioritizes
final errors, accepts hardware change and authorizes concurrent execution if
memory permits. Prior identical-shape peak12.02GiB per arm is below the current
40%device reservation (~33.5GiB) each; two processes reserve80%combined.
Training is parallel; full-process times are recorded as concurrent, not isolated.
Only32/online are authorized, no repeated old fixed arms or additionalCeramic work.

## Status and next action

The user increased the budget to15kand requested publishing existing results.
The just-started10kattempt was stopped with all owned processes; logs and its
initial/intermediate evidence remain under the original bundle. It is superseded,
not a completed10kresult. New parallel cold runs use the same source/seed/scheme,
with iterations15000and checkpoint_step10000. Optimizer decay remains exponential
1e-3×0.95^(step/1000), floored at1e-5; extending the budget does not rescale it.
Time draws use per-stepfold_in and spatial draws keep the same prefix.

Current bundle:`/root/autodl-tmp/tf-logs/electro-library-extension-15k-20261116-01`;
launch log appends`.launch.log`; ownerPID30224. Runs beneath
`/root/autodl-tmp/tf-logs/electropolish2d/` are
`electro-library-extension-15k-20261116-01-fixed32` and
`electro-library-extension-15k-20261116-01-online`.
Preflight and actual initial model/optimizer equality passed against both new
arms and the historical fixed64 run. Step-zero64case metrics also match exactly.
Both trainers have entered ordinary updates without reported OOM. Retain
15minute routine observations; no duplicate or extra experiment.

After both trainers finish, the owner evaluates saved10kand final15kweights on
the same64references and independently checks metrics, first10ksampling prefix,
paired initialization and library membership/online diversity. The older fixed
64/128/256/512and FiLM studies are not being rerun. A20kextension is undecided.

User explicitly authorized README/website updates andcommit/push in this turn.
Publish completed results and accepted ranked main galleries; clearly label15k
controls as pending. Followdev→commit→main→merge→pushmainand checkPages.

## Preparation and validation

`docs/tasks/electro_library_extension.py` freezes the prior numerical source,
adds only two experiment configs and controls preparation/launch/final audits.
Numerical model, residual, optimizer and defaults unchanged. Preflight uses the
same with_run_seed entry transformation as actual training: the initial check
correctly rejected comparing unresolved configs with a serialized resolved seed;
the harness was corrected before training, with no seed change or relaxed gate.
Both resolved configs match the prior64arm except run name and fixed32count or
removal of the fixed-library block. Original32prefix byte identity passes.

Online audit was exercised against the completed historical online control with
output capture only (no writes to old run). All64recomputed channel norms match;
mean0.02163787033, sampleSD0.00779638443;80000distinct profile draws and shared
initialization/spatial/time checks pass. Both final audits retain original strict
metric tolerances. Historical coefficient identity is reported separately because
hardware differs. No extra reference solve or training probe.

## Launch verification and exact resume — 2026-11-16 12:49 Asia/Shanghai

OwnerPID7034; concurrent trainers7043(fixed32) and7044(online), both started
12:47:48Asia/Shanghai. Both run directories and actual initial model/optimizer
exist. Byte equality passes between the new arms AND the prior fixed64initial
states; receipt:bundle/initial_pair.json. Each CUDA process reserves34580MiB;
combined69174MiB of85651MiB, with no OOM or reported numerical error.
These are reserved allocations, not measured active peaks.

Both all64initial evaluations at step0 completed and match exactly:
phi0.64474486861,c0.64762158046. First optimizer-step compilation is in progress;
no trained-error claim yet. Next routine observation no earlier than13:04
Asia/Shanghai, unless user asks or a failure is reported. Read existing journal
and each run/local_metrics.jsonl; do not launch duplicate jobs.

/task-end: changedTODO,PROGRESS,DECISIONS,this active task and the new finite
experiment owner/audit helper. Frozen numerical files and public defaults are
unchanged. Actual configuration/prefix validation, historical online audit
rehearsal, cold initialization identity and initial64case evaluation all pass;
git diff whitespace check passes. Training and final automatic inference/audits
remain in progress. No Git operation or website deployment.

## Publication checks — 2026-11-16

The completed eight-run library study and five accepted ranked main galleries
are prepared for the authorized commit/push. Includes the optional fixed-library
samplers, reproducing experiment owners/audits, figures, example/overview READMEs,
website chapters/build manifest, and Ceramic stability TODO. Unrelated stopped
Melting soft-IC and old core/halo experiments remain outside this commit.

Previously recorded15focused library/online tests passed in9.613s; numerical
source is unchanged since that verification. Local nine-page build passes.
Staged build, asset/link verification and live deployment are the next checks.
The active task stays open for15kcompletion and final10k/15kaudits.
