# Width256 FP32 operator, memory gate and 50k cold training

Date: 2026-10-31. Status: CLOSED_BY_USER_SWITCH. No Git history authorization.

## Goal and acceptance

User requests another modest network enlargement and50ktraining, explicitly
asking to confirm GPU memory fits first. Test width256/fourheads(D64), spatial
128Fourier directions(256sin/cos channels), four cross-attention blocks and
three FiLM layers:3579649trainable parameters. Keep sameGPU/FP32/ordinary,
SOAPbeta2=.9/axis cap512, B16/Nt16, LR schedule, physics/input bank/metric,
hardIC/mass and eliminated chemical potential. No learned weights imported.
First validate config/sampling and storage estimate, then production-size GPU
25updates with refreshes, finite/dtype/IR/memory checks and independent CPU
replay. Only if passed, discard probe and cold-start canonical train.py for50k.

Automatically retain the30kcheckpoint/evaluation for comparison with completed
192/30k, and compare256/30k->256/50k separately from capacity. Native/dense61/
geometry/CPU/full-actual-draw checks and step/time curves at fixed final budget.
No best-heldout checkpoint choice, default promotion, reference-label training,
changed geometry density, GPU migration or concurrent GPU/FEniCS work.

## Context

Completed192/30k baseline:
/root/autodl-tmp/tf-logs/confined_ch2d/20261031-confined-geometry-b16-t16-width192-fp32-30k-01
Phase pooled8.9148%/terminal9.8017%, dense61phase8.3458%; train16phase3.9700%.
Warm119.429ms, cold4752.380s, activepeak9.382GiB. Single runtime seed/eight
reused native development geometries; all final audits passed. Prior task:
docs/tasks/archive/confined-ch-width128-refinement.md.
Dataset /root/autodl-tmp/data/confined_ch2d/operator_geometry_smallscale10_512_v1.
GPU reports RTX3090/48GiB, idle1MiB; disk~11GB before new work. Larger-network
memory/time are unmeasured until actual GPU preflight. Existing selected defaults
and all previous runs remain intact.

## Current state and plan

Live checked 2026-10-31T04:19:32.837010+00:00: runner297584/trainer304259 remain in stage=training,
latest closed loss39200/50000 finite (1.7407594e-05); complete39000evaluation
native reserved8phase pooled8.5060%/terminal9.2893%,
train16phase2.4572%/2.6050%, reservedmu
15.1987%/20.2321%. Only the16labeled training
monitors are scored, not all512training inputs. Original native reference unchanged.

User-requested train/validation convergence is frozen through38500. Native pooled
train16phase20k->30k->38.5k is5.6377->3.5476->2.4917%; reserved8phase
9.7161->8.8383->8.5000%. Terminal at38.5k2.6467%/9.3047%. Recent5000linear slopes
train/validation -0.09883/-0.04496percentage points per1000updates. Both decline,
validation more slowly and monitor/validation gap grows; no sustained validation
rebound or established unique cause. Do not call monitor performance the entire
training-bank error, declare capacity solved, or automatically grow the bank.
Raw L2 points are unsmoothed; only displayed PDE loss has an explicitly labeled
11record median alongside raw values. Full78split/field aggregate checks and an
independent train/validation replay PASS; PNG/PDF visually inspected. Receipt
confined-ch-width256-50k-convergence.json; script confined_ch_live_convergence.py.
Plot/raw frozen records external in run/convergence_live_20261031_0420.


Width192 convergence requested next: all61recorded evaluations through fixed30k
are plotted with the same4panel format and16training/8validation cohorts. Pooled
train/validation10k10.3865/13.2276%,20k6.0776/9.9760%,30k3.9700/8.9148%; terminal
30k4.2544/9.8017%. Both continue decreasing, validation more slowly; train/validation
gap is not unique to256. Separate192/256overlay clips both frozen histories at
30000: training3.9700/3.5476%, validation8.9148/8.8383%; validation curves close,
capacity-only accuracy gain modest. No192extension or256training modification.
New receipt confined-ch-width192-convergence.json; same plotting helper now
labels a completed budget explicitly and uses saved optimizer/compute dtype.
New confined_ch_convergence_comparison.py checks explicit common steps/cohorts,
physical/reference/optimizer/runtime configuration and unsmoothed metric curves.
All61split/field aggregates independently replayed; both PNG/PDF visually inspected,
AST/JSON checks PASS. Initial plotting-only guard assumed nonexistent phase key;
corrected to actualCHschema before any output, no numerical or source-run change.
Large plots/frozen rows external in width192 run/convergence_detail_20261031 and
convergence_common30k_20261031. Current task remains256fixed50k plus final audits.

The common30000budget is now complete: width192 phase8.9148%/9.8017% versus
width256 phase8.8383%/9.6682%, a small native-phase improvement (0.0765 and
0.1334percentage points). Train16 pooled3.9700%->3.5476%; mu pooled17.1891%->
15.8122%. Online elapsed4744.08s->6557.87s (includes setup/compile/periodic
assessment, not final audit). Thus current capacity-only phase gain is modest;
32000versus19230000 additionally includes more updates, not a pure capacity
comparison. Same single seed/eight reused development geometries; final actual
sampling/CPU audit and common-budget dense61comparison remain pending.

Recent evaluation-inclusive0.189772s/update, estimated34.2min remaining
updates plus final audits. Initial119leaf/source audit PASS. Live receipt
confined-ch-width256-50k-live.json includes fixed30k and recent evaluations.
No process/config/default changes; continue fixed50k and automatic final audits.
Do not select a best heldout intermediate checkpoint. No final50kresults yet.

Explicit preset is added. Research-only capacity_case metadata and generalized
probe/runner select --case=width256_50k; omitted case retains192/30k commands.
Shared network/PDE/training/optimizer/precision code unchanged. Syntax PASS.
CPU config/shape/sampling/resource audit PASS: exactly width192->256,
spatial_features96->128 and iterations30000->50000; cap512 and all other fields
unchanged. Both architectures have exactly one excluded matrix axis: temporal
modulation output1152/1536. Parameter count2021185->3579649; model14,319,620bytes,
optimizer84,944,240bytes at256. Fifty paired backups plus three spare pairs,
serialization/probe margin require6,779,322,628bytes (6.314GiB); disk currently
fits without cleanup. Expected first25geometry batches match saved192trace;
time draws match at1..25,10k,30k,50k. New old192compatibility execution also PASS;
original completed receipts retained. Receipt confined-ch-width256-50k-config.json.

Full production25update GPUpreflight and independentCPU replay PASS. All losses,
gradients/model/SOAP states finiteFP32, no nonscalarFP64 training IR tensors;
actual25geometry prefix and time draws exactly match completed192baseline.
Measured activepeak12.0887GiB, allocatorlimit35.5616GiB, remainingheadroom
23.4729GiB. Ordinary/refresh steps about.16/.23s; warm20mean174.562ms
covers fourrefreshes. Firstcompile673.269s, fullprobe908.808s. No batch/precision/
SOAP fallback; current48GiBcard fits this production workload. IndependentCPU
field gaps[4.172325134277344e-07, 1.1026859283447266e-06] below declared2e-5phi/2e-4mu.
CPU119initial arrays also match exact freshGPUinitial. Receipts:
confined-ch-width256-50k-preflight.json and confined-ch-width256-50k-cold-init-check.json.
Pure50kupdates estimate145.5min; allow about3hours from coldtraining start
for setup/firstcompile/online evaluation/finalaudits, not measured completion.

Probe discarded. Detached runner297584 now owns canonical cold50ktrainer
304259, stage=training; initial compilation is complete; current L2 is above.
Journal:
/root/autodl-tmp/tf-logs/confined_ch2d/20261031-confined-width256-fp32-50k-01-process.json
Probe:
/root/autodl-tmp/tf-logs/confined_ch2d/20261031-confined-width256-fp32-50k-01-probe
Active cold50krun:
/root/autodl-tmp/tf-logs/confined_ch2d/20261031-confined-geometry-b16-t16-width256-fp32-50k-01
Launch receipt confined-ch-width256-50k-launch.json. Initial production CPUaudit
now PASS; current checked state is recorded above. Fixed50knative/dense61, paired30kdense61,
geometry/history/CPU/fullactualdraw checks and step/time curves are automatic.
Expected compact receipts use prefix confined-ch-width256-50k-{initial-audit,
preflight,audit,results}.json. No competingGPU/FEniCS or default/history change.

Session task-end: added256/50kpreset and explicit researchcase metadata;
generalized existing researchprobe/runner with retained192CLI and explicit
legacy192storagepolicy. Added exact resource/sampler audits and30kcommonbudget
validation. Updated README/DEVELOPMENT/PROGRESS/DECISIONS/task/artifactindex and
compactconfig/launch/initial/preflightreceipts. Shared neural/PDE/training/
optimizer/precision code unchanged. Bothcase CPUconfig/shape/sample checks,
newGPU25updates/IR/finite/headroom/CPUreplay, module syntax/JSON/diff checksPASS.
No unchanged broad test suite rerun or default change. Task remains ACTIVE.

Blocker: none. Next action: inspect only NEW256journal and production initial
audit, then collect fixed30k/50kmetrics and automatic final audits. Keep this task active
while the authorized training and automatic final audits run. Do not launch
another job or overwrite old control artifacts.

Status-session task-end: read live heartbeat/process identities, closed loss and
fixed30k192metrics; added frozen192detail and common30koverlay to earlier256plots,
updated current task, PROGRESS, live receipt and index.
JSON and diff checks only, no unchanged expensive tests or numerical changes.
Exact resume remains existing NEW256journal and automatic30k/50kevaluations;
do not relaunch or add workers. Task ACTIVE, no blocker.

## User-directed stop and switch — 2026-10-31

User cancels remaining256training and requests width192 with an expanded geometry
bank, then explicitly sets a cold50kbudget. Runner297584/trainer304259 identities
checked and terminated with SIGTERM at04:25:01UTC; no forced kill. Latest logged
loss40900, last complete evaluation40500phase train16/validation8 pooled
2.3728/8.5276%, terminal2.5175/9.3328%. Last complete model/SOAP pair40000 is
retained with its marker; all data/logs/probe/partial artifacts retained. Journal
now stopped_by_user; no50kresult or automatic final audits claimed. Stop receipt
docs/tasks/confined-ch-width256-50k-stop.json. GPU1MiB idle after stop.

This task is closed by explicit scope replacement, not by achieving its original
50kobjective. Active work moves to docs/tasks/confined-ch-width192-bank-growth.md.
Existing512/19230kfinal and25630kcommon-budget observations remain valid within
their original audit scopes. Do not restart256or run its cancelled final pipeline.
