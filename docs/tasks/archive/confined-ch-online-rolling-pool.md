# Confined CH25k continuation with online rolling geometry inputs

Date:2026-10-31. Status:CLOSED_BY_USER_SWITCH_AFTER_HOST_INTERRUPTION. No Git history authorization.

## Goal and acceptance

User explicitly asks to stop192/1024cold50k, load its25kmodel and train with
background online generation plus a rolling sample pool. Restore complete model
and SOAP/schedule state at25000, continue global counters through total50000.
Preserve width192/H4/D48/Fourier192channels/FP32ordinary/B16Nt16/SOAPbeta2.9cap512,
physical T3/10mode fixed ambientIC, native regional graph, hardIC/mass and
eliminatedmu/stable residual metric. Only geometry sampling/feeding changes.
No reference supervision or new training PDE/FEM solves. Do not promote defaults.

Acceptance: strict direct-path paired25krestore and initial field parity; sampled
geometry prior/known inputs/operator/metric parity and no future labels; bounded
memory/queue/disk; deterministic refresh order independent of worker completion;
fixed padded shapes/dtypes and loud overflow errors; finite productionGPUstep/
memory check; measured producer/consumer times and same16/8L2 monitoring. Launch
canonical train.py only after focused tests and gates. Keep an exact journal,
recoverable checkpoints and validation outputs. Preserve all previous artifacts.

## Context and paths

Parent /root/autodl-tmp/tf-logs/confined_ch2d/20261031-confined-geometry-b16-t16-width192-bank1024-fp32-50k-01.
Explicit checkpoint_025000.json names ckpt_025000.eqx(8100740bytes) and
optimizer_025000.eqx(47918192bytes). Parent has no complete50ksummary.
At25000phase train16pooled6.0960%, validation8pooled8.2508%/terminal9.0060%.
Original192/512baseline remains a separate cold30krun, not a matched online
continuation control. Native graph reference only; localFEMdeferred.

Initial input bank /root/autodl-tmp/data/confined_ch2d/operator_geometry_smallscale10_1024_v1.
Same old16training monitors and8accepted draws16..23reserved references. New
input sampling is continuous from the same valid two-hole parameter prior.
Historical6worker stage estimate~5.3input/s versus GPU~100input uses/s motivates
reuse, not all-freshB16perstep. Container12CPUbudget/96visible logical processors.

## Current state and plan

Parent runner339235/trainer346501 stopped by SIGTERM2026-10-31T06:08:33.505821Z;
both exited, GPUidle1MiB, disk5.42GBfree. Lastlogged26900, evaluated26500phase
train5.9137%, validation7.9115%/terminal8.6213%. All artifacts retained. Stop
receipt docs/tasks/confined-ch-width192-bank1024-50k-stop.json. Previous active
task is closed by user switch and archived, no50kcompletion claim.

Explicit online preset: initialize1024slots from old inputs,6CPUworkers,
replace32slots each100updates, two refresh batches prefetched; deterministic
accepted geometry stream and FIFO consumption. Separate fixed evaluation inputs.
Choose padded shapes with an explicit overflow check, measure full-size peak/
producer cost before starting. Resume from paired25kfiles with identical model,
SOAP count/moments, globaltime stream and LR. No architecture or physics change.

Implemented explicit paired-marker continuation and optional online producer/
pool integration, frozen evaluation payloads, bounded regenerated pool-state
checkpoints, sampled geometry/time identity records and first25production-update
finite/precision/memory gates in one process. Preserve legacy complete-parent
imports. No numerical law or default changed. The shape envelope is1536/128.

Ten focused online/legacy-continuation/metric CPUtests PASS95.48s; additional
canonical online train/save/recover integration PASS41.55s. The latter exercises
recovery from an online intermediate checkpoint and frozen monitor fields after
pool replacement, not only the host controller. Initial extra fixture attempt
used a missing legacy precompute record; fixed the toy configuration, no change
to production physics.

Real CPUgate PASS2026-10-31T06:27:26Z: exact25kmodel/SOAP serialized bytes and
integer schedule counter25000;64production inputs/6workers in11.306s(5.661/s),
mean mesh0.2493s/assembly0.0794s/stablemetric0.5030s/padding0.0126s. MaxNx1360,
degree93, constant-mode error<=3.29e-14. Two old meshio/CAD route comparisons
preserve centroid order and agree to FP32rounding; repeated draw0all8arrays exact.
No future training labels, PDE/FEMsolves or persistent generated operator files.
The first pure .venv Gmsh attempt exposed its missing native-library symbol;
use the project's existing explicit meshing interpreter instead. Only Gmsh/
NumPy are imported in that isolated subprocess, no DOLFIN/FEniCSMPI solver or
second activated environment. No package or dependency changes. Worker config
explicitly sets JAX platforms=cpu even if JAX was already imported; tested with
JAX_PLATFORMS initially absent. Data/physics and geometry prior are unchanged.

Detached runner376702/trainer376704 launched2026-10-31T06:32:53Z after CPUgates.
Initial setup318.10s. Model/SOAP initial files exactly match25kparent bytes;
all96per-geometry/field score checks pass, max absolute gap2.2423e-7. Restored
native phase validation8.2508%, terminal9.0060%. First32fresh inputs install
before25001; initial collection6.399s and donatedGPUreplace0.2228s. First step
compiles once in the production process. Canonical command/config and absolute
paths in confined-ch-online-launch.json. Runner performs final independent CPU
input/state/field replay audit automatically after50000.

Actual25production-update preflight PASS at25025: model/SOAPfloating leaves all
finiteFP32, every loss/gradientfinite; meanlast20GPUupdates0.116166s. Peak active
20.4337GiB versus35.5616GiBallocator limit. Eight input payload keys verified,
no future labels. No separate throwaway GPUcompile or optimizer reset.

Checked2026-10-31T06:53:49UTC: healthy existing runner/trainer; latestlogged27600,
complete27500native phase originaltrain16monitor7.0530%/terminal7.5249%,
reserved8validation8.4610%/terminal9.0805%. Imported25k had6.0960%training
monitor/8.2508%validation/9.0060%validationterminal. First25500evaluation was
8.5972%/9.3325%; recent error fluctuates slightly above import, no accuracy-gain
claim. Keep the prescribed50kbudget rather than changing scheme/seed on early
development scores. Different padding can change FP32kernels; historical
fixed-bank trace is not a fully isolated online-sampling ablation.

27refreshes installed864fresh inputs; fixed1024pool/64prefetch limit unchanged.
Later collection+CPUstacking~0.2-0.3s and GPUreplace~0.085s per100updates;
do not call collection time pure producer starvation. Evaluation-inclusive
steady0.141653s/update over25500..27500, ~52.9min updates remaining plus audits.
First real26000model/SOAP/online-state marker complete; state has10refreshes and
320newdraw IDs, all1024active identities independently reconstructed exactly.
Live receipt confined-ch-online-live.json includes all early evaluation points,
memory/restore checks and checkpoint sizes. Parent and old controls preserved.

Run /root/autodl-tmp/tf-logs/confined_ch2d/20261031-confined-width192-online25000-rolling-01-train.
Journal /root/autodl-tmp/tf-logs/confined_ch2d/20261031-confined-width192-online25000-rolling-01-process.json.
Training log /root/autodl-tmp/tf-logs/confined_ch2d/20261031-confined-width192-online25000-rolling-01-training.log.
Runner log /root/autodl-tmp/tf-logs/confined_ch2d/20261031-confined-width192-online25000-rolling-01-runner.log.

Blocker:none. Next action: monitor only the NEW online journal and complete
metrics/refreshes/checkpoints; all initial state/field/actualGPUgates passed.
Continue through50000with automatic independentCPUdraw/time/state/field audit.
Do not duplicate launch, restart old controls, promote defaults or claim online
improvement from this early snapshot. Review accuracy once the fixed budget ends.

## Session changes and verification

Performance follow-up2026-10-31T06:57UTC: existing journal healthy, last29000
evaluation phase train7.3838%, validation8.3497%/terminal8.9985%.41refreshes
installed1312inputs; rolling training pool has fully replaced its original
1024entries while historical16/8evaluation payloads remain frozen. User asks
whether replacement is substantially slower. Existing40steady refreshes average
collect/CPUstacking0.23377s + GPUreplace0.08663s per100updates:0.003204s/update,
~2.26%of full online walltime. This is observed direct critical-path overhead,
not a controlled estimate of all indirect CPU contention. Initial one-off
collection+replace6.622s. Full online25500..29000=0.141826s/update versus stopped
fixed-bank500..26500=0.149411s/update. Different padding/compiledkernels prevent
attributing that small speed difference to online sampling. No new run, timing
intervention, source/config/refresh-frequency change or accuracy claim.
Receipt confined-ch-online-refresh-cost.json; JSON/numeric and diff checksPASS.
Task-end: task/PROGRESS/performance receipt updated only. Exact resume remains
the existing NEW online journal, continuing through50000and finalCPUaudit.

README and DEVELOPMENT document optional online inputs and direct-path paired
recovery. PROGRESS switches the single active task and old control is archived
by explicit user switch. Artifact index marks oldrunstopped/newsetup, keeps
default selection. DECISIONS records change in sampling and restored-state
policy. New compact stop/CPUgate/launch receipts; no media/banks added to Git.
AST and git diff --check PASS before launch and at session end. Final receipt
confined-ch-online-validation.json confirms ten original training/field function
ASTs and five numerical/network/optimizer source files exactly match the parent,
JSON/whitespace/single-active-task checks and all actual run gates PASS.
Do not overwrite parent outputs or claim new50kresults before completion.
No Git history operation authorized.

Task-end for implementation/launch session: changed geometry array-export option,
strict paired-checkpoint recovery, optional fixed-shape online pool/worker/loop,
new explicit preset, focused tests and research gate/runner; README/DEVELOPMENT,
DECISIONS/PROGRESS/task/index and compact receipts updated. Old control archived
by user switch. Source numerical kernel, network architecture and physical loss
unchanged. CPUtests/gates, actualGPUwarmup/restore/firstcheckpoint pass as above;
AST/JSON/numeric/link/whitespace checks PASS. Keep this task ACTIVE: only ongoing
training and final automatic audit remain. Exact resume is the NEW journal/run
listed above, latest known complete27500. No blocker or additional queued run.

Error-status follow-up checked2026-10-31T07:01:58.911405+00:00: runner/trainer healthy, logged31000/50000, complete31000phase originaltrain16monitor7.2102%/terminal7.7217%, validation8pooled7.4907%/terminal8.0465%. Validation per-domain pooled range2.28..13.70%. Imported25kvalidation8.2508%/9.0060%; raw intervening fluctuations retained. 61refreshes/1952installed online inputs. Recent0.143134s/update,~45.3min updates left plus audits. Bothsplit/fourfield aggregates independently reconstructedPASS. No training/source/config/Git change or attribution solely to online sampling. Task-end: active task/PROGRESS/live receipt/index updated; continue existing online50kand automaticCPUaudit.

Fixed-bank comparison follow-up2026-10-31T07:04:58.065329+00:00: at common30000/width192, fixed512validation8.9148%/terminal9.8017% versus1024pretrain25k+rolling5k7.8984%/8.4874%; relative reductions11.40%/13.41%. Latest online32000=7.2543%/7.7704%. Positive observation for the combined broader-input strategy, not proof of rolling sampler alone: pretraining bank and paddedkernel differ, no fixed1024same-budget endpoint; at26500online8.5274% exceeded fixed10247.9115%. Same16/8cohorts audited, no fresh-test/multiseed inference. Task-end: comparison receipt/task/PROGRESS updated, JSON/numeric/diff checksPASS; no training/source/config/Git change. Continue existing50kandfinalCPUaudit.

## Closure — fresh width256 online50k on the replacement GPU

2026-10-31: replacement GPU reports NVIDIA RTX6000D/85651MiB. Both old
runner376702 and trainer376704 are absent; their journal stopped07:06:24UTC.
Last logged32900, evaluated32500: historicaltrain16phi6.9842%, validation8
phi7.2507%/terminal7.7399%. Last complete paired model/SOAP/pool checkpoint32000
is retained. No50k endpoint or final audit exists; the old journal is stale.
User explicitly requests a fresh width256/online50k run, superseding this partial
continuation. Preserve every file in this online run and its parent25k pair.
Resume only if separately requested; do not relaunch this runner automatically.
All remaining work moves to confined-ch-width256-online-cold50k.md.
