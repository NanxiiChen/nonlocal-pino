# Geometry-operator capacity, precision and spatial-refinement controls

Date: 2026-10-30. Completed: 2026-10-31. Status: COMPLETE. No Git history authorization.

## Final outcome and completed task-end

Authorized width192cold30k and all automatic native/dense61/geometry/CPU audits
completed; every subprocess exited0 and currentGPU is idle1MiB. No further jobs
or required work remain. No new larger/deeper/longer run or default promotion.
Saved preset/source/checkpoints/optimizer states and all historical partial runs
remain intact. No Git history operation, dependency or physical-model change.

| Fixed30k FP32 control | Width128 | Width192 |
|---|---:|---:|
| Trainable parameters |905089|2021185|
| Native8development mean phase pooled |9.7841%|8.9148%|
| Native mean terminal phase |10.5799%|9.8017%|
| Dense61time mean phase pooled |9.3192%|8.3458%|
| Training16monitor pooled phase |5.0891%|3.9700%|
| Mean native chemical potential pooled |19.3091%|17.1891%|
| Worst native geometry pooled phase |18.5475%|15.5329%|
| Cold training wall including setup/compile/online evaluation |3878.968s|4752.380s|
| Warm compiled updates |92.333ms|119.429ms|
| Active training peak |8.431GiB|9.382GiB|

Relative phase improvements8.8854%pooled/7.3561%terminal, coldwall+22.5166%.
Both are fresh30k FP32runs with identical actual30000geometry draw batches,
same temporal sampling/PDE/bank/metric and original unrounded native references.
Joint backbone/Fourier enlargement with SOAPcoverage preserved; one runtime
seed/eight reused development geometries, not an isolated head-dimension gain
or repeated-seed/general continuum validation. Six of eight dense61geometries
improve;518/519 worsen. Mean<10%does not imply all individual geometries<10%.
Last5k nativephase9.2922->8.9148%, with final500step rebound from8.7841% at29500;
use prescribed30000final rather than select the best heldout checkpoint. No
claim of full convergence; no additional training is queued.

Independent CPU24field replay maxphi6.258e-7/mu2.817e-6 below fixed2e-5/2e-4
thresholds. Exact initial gap0; finite model/state, no labels in step payload,
no reserved input draws, all512inputs visited836..1062times. Native reference
time refinementphi<=1.0812e-9/mu<=9.3878e-10. Prediction mass drift1.630e-8 and
sampled energyincrease0. Geometryresponseerror19.6560->19.0847%, amplitude
.954025->.956250, correlation.980859->.981956. These are aligned native nonlocal
references; local FEM remains explicitly deferred.

Final direct artifacts:
/root/autodl-tmp/tf-logs/confined_ch2d/20261031-confined-geometry-b16-t16-width192-fp32-30k-01
Journal20261031-confined-width192-fp32-01-process.json statuscomplete; no livePID.
Native fields/figures in evaluation/,61time results in dense_evaluation/,
geometry_effect/,training_report/,capacity_comparison.png/.pdf in the run.
Compact receipts docs/tasks/confined-ch-width192-results.json,
confined-ch-width192-audit.json and confined-ch-capacity-final.json.

Completion-session changes: verified closed result/CPU/input/physical receipts,
inspected final convergence figure; recorded final comparison and limitations,
archived this completed task and updated PROGRESS/DECISIONS/artifact statuses
and README completed-record link. No code/config/checkpoint/default changes.
JSON/schema/link and git diff checks only; unchanged expensive tests not rerun.
Exact resume: none required. Any future capacity/continuation/default choice
starts with these fixed-budget records and needs user direction.

The following active/launch observations are historical.

## Historical launch and live width192 FP32 capacity control

Live checked 2026-10-30T18:14:03.236330+00:00: runner260556/trainer266287 healthy,
stage training, latest closed loss2900/30000 finite(0.000592428376). Latest complete
cohort evaluation2500: reserved8mean phase pooled21.2044%/terminal
22.5539%, versus matched128baseline23.9724%/25.3003%. Training16monitor
18.7543%/19.7262% versus128baseline20.7822%/21.5752%. Early same-step
capacity gain, not final/multi-seed advantage. Reserved phase at500/1000/1500/
2000/2500:33.6937/28.0701/25.3126/23.2085/21.2044%, still falling. Native FP64
reference and reused reserved development cohort unchanged. Actual recent
periodic-evaluation-inclusive rate0.134823s/update (~2.25min/1000) predicts~60.9min
remaining updates plus final audits. Receipt confined-ch-width192-live.json.
Production initial audit PASS119leaves: exact fresh-seed/probeINITIAL equality,
shared network/training/optimizer/precision/runtime source bytes unchanged.
Final audit/results remain pending; no default/runtime/process changes.


User approves moderate enlargement after the head/cross-attention discussion.
New explicit preset train_operator_smallscale10_512_width192_fp32.py changes
width128->192, spatial Fourier directions64->96 and SOAPaxis cap256->512.
Four heads now have48features; four cross-attention blocks/three FiLM layers,
2021185trainable parameters. Same512/8bank, B16/Nt16, FP32/ordinary, beta2=.9,
LR, hardIC/mass, eliminatedmu, PDE/metric and fixed30k budget. New seed-policy
initialization, no learned weights/state import. This is a combined capacity
control, not an isolated head-dimension ablation. Existing defaults unchanged.

CPU configuration/abstract-shape audit PASS: exactly those three fields change,
baseline/candidate905089/2021185params, both retain all preconditioned matrix
axes except their temporal-output axis. On baseline128, cap256 and512 have
identical coverage. First25expected geometry draws match completed baseline,
times exactly match. Receipt confined-ch-width192-config.json.

Full-bank25update GPU preflight and independentCPU replay PASS. All losses,
gradients/model/SOAP states finiteFP32; no nonscalarFP64 tensors in compiled IR.
Actual25geometry prefix and alltimes exactly match completed128baseline.
Warm20updates121.803ms (includes fourSOAP refreshes), activepeak
9.3578GiB versus allocatorlimit35.5616GiB. First compilation
486.399s, fullprobe695.391s; alarms were slow compilation, no numerical failure.
IndependentCPU replay gaps[2.8312206268310547e-07, 1.0281801223754883e-06] below declared
phi2e-5/mu2e-4. IndependentCPU119initial leaves equal exact saved freshGPUinitial.
Receipts confined-ch-width192-preflight.json and confined-ch-width192-cold-init-check.json.
Historical128FP32probe94.688ms/8.337GiB gives~1.286xupdate time,
not a completedtraining wall ratio. Pure30kupdates estimate60.9min;
allow~1-1.5hours from training launch for coldsetup/compile/evaluation/audits,
not measured completion or promised error.

Probe states discarded. Detached runner260556 now owns canonical cold30k
trainer266287, stage=training. Journal:
/root/autodl-tmp/tf-logs/confined_ch2d/20261031-confined-width192-fp32-01-process.json
Probe:
/root/autodl-tmp/tf-logs/confined_ch2d/20261031-confined-width192-fp32-01-probe
Active cold run:
/root/autodl-tmp/tf-logs/confined_ch2d/20261031-confined-geometry-b16-t16-width192-fp32-30k-01
Launch receipt confined-ch-width192-launch.json. Current trained metrics and
production initial-audit are recorded in the live status above. Old FP32 runner completed all
stages/exited, GPU1MiB/idle before launch, disk~12GiB. No competingGPU/FEniCS.
New config/probe/runner syntax, explicit config/shape/cap/actualprefix/CPUreplay
and JSON/diff checksPASS. No unchanged broad suite repeated or default change.

Runner sequentially validates finite FP32/IR/no bulkFP64, actual25draw prefix,
independent CPU fields, then discards probe state and cold-starts train.py.
Production initial arrays must exactly equal probe INITIAL arrays/fresh seed,
and shared model/training/optimizer/runtime sources must match the128baseline.
On30k completion automatically run native, dense61, geometry response, history,
CPU/state/no-label/full-actual-draw audits and step/wall curves against completed
FP32baseline. Expected receipts confined-ch-width192-preflight.json,
confined-ch-width192-initial-audit.json, confined-ch-width192-audit.json and
confined-ch-width192-results.json. Do not launch a duplicate. Resume only this
new journal; collect measured probe timings before any ETA claim/default choice.

Session task-end: added width192 FP32 preset, capacity probe and durable runner;
updated stable README/DEVELOPMENT, DECISIONS/PROGRESS/current task/artifact index
and compact config/launch/initial/preflight receipts. Shared model/PDE/optimizer
sources unchanged. Production initialization CPU audit now PASS; final validation
will follow the fixed30k training. Task stays ACTIVE,
no blocker. Exact resume: only NEW width192 journal/PIDs/training log above;
collect initial-audit, final results/audit and comparisons. No duplicate launches,
oldcheckpoint imports, GPU migration or Git history operations.

Status-session task-end: read current process identity/heartbeat, closed loss and
cohort metrics, same-step128baseline and production initial audit; updated this
task, PROGRESS, compactlive receipt and artifact index. No code/config/training
changes or repeated expensive tests. JSON and git diff checksPASS. Resume NEW
width192 journal and final automatic receipts; do not start another worker.

## Completed width128 FP32 trial on the existing GPU

All NEW03 stages complete, including CPU24field replay/input/state audit and
step/time comparison curves. Final nativephase9.784127%/10.579941%, train16
5.089073%/5.419521%; mu19.3091%/25.3023%, worstphase18.5475%. Reference mass and
energy checks retained; predictionmassdrift1.483e-8, sampledenergyincrease0.
CPU/GPU maxphi5.364e-7/mu2.488e-6 below predeclared tolerances; full geometry
draw trace equals original global seed/completedcontrol and excludes reserved
draws; step payload has no future fields. Cold training wall3878.968s,
warmupdate92.333ms; final online elapsed3871.816s has a slightly earlier endpoint.
Geometryresponsecorrelation.980859/amplitude.954025. Final receipts
confined-ch-fp32-results.json and confined-ch-fp32-audit.json. Best96FP64 defaults
remain selected; prior FP64precision control stops at21k, so no30kprecision rank.
The following live observations and launch details are historical.

Live update checked 2026-10-30T17:37:47.946682+00:00: NEW03 training completed30000steps;
online evaluation over eight reused reserved development geometries gives mean
phase pooled9.7841% and terminal10.5799%, train16monitor
5.0891%/5.4195%. Training-stage elapsed3871.816s includes setup/compile/
online evaluations; not whole-pipeline wall. Runner234487 currentstage=geometry,
activechild=256311; final validation pipeline continues. No final
CPU/result receipt claim or default promotion yet. No matchedFP64after21k.
Original references unchanged; last verified matched21k FP32phase11.0565/11.8718%
versus FP64/beta09 11.2643/12.2732%. Exact resume: NEW03 journal/current stage,
then final native/dense61/geometry/CPU results and precision curves. Do not launch
a duplicate or larger-network run. Previous live receipt is a historical25k
snapshot until updated; final pipeline results are authoritative once complete.

User requests FP32 and explicitly keeps current hardware. Implemented explicit
float32/float64 config validation, model/bank/time casts and public field/query
precision. FP64 host mesh, graph, input-only frozen metric and original native
references remain unchanged. Initialization/time RNG first follows original
FP64 path, then rounds; no optimizer import, new architecture/seed or default
promotion. Optional preset train_operator_smallscale10_512_width128_fp32.py
changes only dtype versus width128/beta09, apart from runtime run name.

Preserved closed FP64 state checkpoint_021000.json and finite model/optimizer
payloads; both counters21000. Old/new FP64 field replay difference0 on actual
geometry512. Owned runner198579/trainer198583 intentionally stopped, all old
states/configs/logs retained and old journal marks intentionally_stopped_for_fp32.
Latest21k phase11.2643% pooled/12.2732%terminal, train16monitor6.7194/7.1634%.
This is a partial run, not completed30k. Stop receipt confined-ch-fp32-switch.json.

29focused CPU tests PASS: new precision/JVP/gradient/7SOAPstep refresh checks,
independent chemical-rounding check rejects real errors, old temporal and
complete continuation tests, base operator/logtime/initial-RHS regressions.
Pytest is not installed; used the repository unittest runner, no dependencies
changed. All changed modules compile. FP64 actual checkpoint/source path parity
PASS. FP32 native chemical validation uses gamma_(degree+16) with explicit local
absolute term scales; CPU/GPU field replay limits predeclared2e-5phi/2e-4mu.
Original FP64 physical/replay thresholds are retained.

Detached runner234487 owns the validation pipeline; current stage/PID at task top.
Completed production trainer was241188.
NEW journal:
/root/autodl-tmp/tf-logs/confined_ch2d/20261031-confined-width128-fp32-03-process.json
Probe:
/root/autodl-tmp/tf-logs/confined_ch2d/20261031-confined-width128-fp32-03-probe
Active canonical cold30k run:
/root/autodl-tmp/tf-logs/confined_ch2d/20261031-confined-geometry-b16-t16-width128-fp32-30k-03
Runner sequentially performs full-bank25step GPU probe, independent CPU replay,
then discards probe state and starts train.py cold. It checks actual production
initial weights against exact FP64rounded snapshot, then native/dense61/geometry/
CPU audits and precision-vs-step/time curves. Probe rejects nonscalar FP64 tensors
in FP32 training StableHLO (scalar configuration literals are counted) and nonfinite gradients/parameters/SOAP states. No
simultaneous GPU work or FEniCS. Warm20probe updates cover fourSOAP refreshes.

Attempt01 intentionally remains failed: its config equality compared tuple
preset fields with JSON lists, caught before bank/compile/updates. Fixed by
comparing normalized serialized configurations; exact dtype-only parity PASS,
same seed/numerics. Attempt02 caught transient FP64 SOAP update arrays during IR audit before
compilation/updates. Local precision adapter now supplies typed FP32 beta
scalars; shared/upstream optimizer code unchanged. Rerun precision2tests with
IR/7updates/refresh PASS. Scalar configuration literals may stay FP64 before
conversion; network/PDE/gradients/SOAP matrices and QR remain FP32. Attempt03
is a setup/precision correction, not a training-seed retry.
All01logs/journal/empty probe directory retained. Launch receipts:
confined-ch-fp32-launch.json, confined-ch-fp32-launch-02.json, and
confined-ch-fp32-launch-03.json; failed01/02 journals and IR retained.

Changes this session: operator_precision helper, operator_training compute
plumbing, operator_problem dtype validation/initial-function arrays, native
physical checks, FP32 preset, precision unit tests, probe/switch/runner scripts,
precision-aware audit/dense scripts, README/DEVELOPMENT/index and memory.
Measured full-bank FP32 probe PASS:25updates, all loss/gradient/model/state
finite, no nonscalarFP64 training IR tensors;32scalarF64tokens are configuration
literals/conversions. All initial arrays match exact rounded FP64checkpoint,
time draws match rounded originals, actual25geometry draws equal completed
width96prefix. Warm20updates94.6877ms, activepeak8.33735GiB, live4.655GiB;
first compilation425.833s, firstexecution.345s, fullprobe631.482s. CPU replay
phi/mu max differences2.384e-7/9.537e-7 PASS predeclared2e-5/2e-4 gates.
Receipt confined-ch-fp32-preflight.json. Historical same-size FP64probe575.272ms,
activepeak17.3539GiB gives approximate6.075x warm speedup; that older probe used
beta2=.99/12steps, not a new matched25step beta09 FP64measurement. Current
beta09 FP64 production was about.60s/update including evals. Do not compare
compilation/whole-training wall as though those scope differences were absent.

Independent frozen21k FP64model cast/evaluated inFP32 on the eight reserved
geometries: max phi8.912e-7/mu3.077e-6 difference. Mean phiL2 .112643194752 versus
.112643197879. This is inference rounding of an existing FP64-trained model,
NOT FP32-trained accuracy/stability. Receipt confined-ch-fp32-frozen-fields.json.

Probe state is discarded. Canonical cold30k training now owns GPU under the NEW03
runner and journal above; current trainer PID is recorded there. Production
initial-checkpoint CPU audit now PASS:119array leaves exactly equal the rounded
FP64initial snapshot; serialized config differs only in compute dtype/runtime
name, shared network/optimizer/runtime source bytes unchanged. Receipt:
confined-ch-fp32-initial-audit.json. Initial launch estimate was47.34min for
updates and1-1.5hours including cold setup/compilation/evaluation/audits; this
estimate is superseded by the live measured status at the top of this task.
Final old FP64control remains partial21k, no completed30k precision ranking.

Session task-end: precision plumbing/preset/physical checks/tests/analysis runner
and docs delivered.29focused tests PASS, and precision/IR/refresh2tests rerunPASS
after local SOAPbeta typing. Old/new FP64 actual field parity0; independent
rounding-bound chemical check rejects genuine errors. Changed modules compile,
JSON/config equality/source-preservation/diff checksPASS. No dependency/GPU/
sharedoptimizer/default/Git history change; old failed setup logs/IR retained.
Memory changed: PROGRESS/DECISIONS/DEVELOPMENT/this task plus compact receipts.
Exact resume: inspect only NEW03 journal/log/current PIDs; no duplicate jobs.
Collect production initial-audit, then final audit/results and native/dense61/
geometry/curves. Task remains ACTIVE with the authorized training, no blocker.

Status-session task-end: read live process identities, heartbeat, closed logs and
latest25000step and matched21000step FP64 metrics; no runtime/config/optimizer/
default change. Updated
this task, PROGRESS, artifact index and compact confined-ch-fp32-live.json.
Production initial audit already PASS. Training and automatic final evaluations
continue; inspect the NEW03 journal on resume, do not launch a duplicate run.

## Architecture clarification during FP32 training

Checked2026-10-30T17:33Z against actual NEW03 config and shared source:
heterogeneous boundary/function point operator, width128/fourheads (32features
perhead), four condition-reading blocks with256wide feedforward hidden layers,
three128wide FiLM temporal decoder layers. Spatial64frequency vectors yield128
sin/cos channels plus raw2D coordinates; spatial_scale2.0. Time32frequencies
spanning1..4 yield64sin/cos channels plus scalar time. Only the phase change
is learned; hardIC/mass wrapper and eliminated chemical potential are retained.
Two streams:192boundary points with(x,y,nx,ny,type), domain1351paddednodes with
(x,y,phi0). Spatial representation/condition reads are shared across all17times.
905089trainable parameters (frozen128frequency coefficients excluded).

CPU abstract-shape check PASS for capacity options that ALSO align Fourier
feature width to backbone:128/64freq905089params;192/96freq2021185params;
256/128freq3579649params. Original earlier fixed64freq counts1984321/3481345
remain valid for their different scope. Receipt confined-ch-fp32-capacity-options.json.
Increasing feature count leaves frequency distribution scale unchanged but is
not a width-only ablation. A moderate192width/fourheads option is a reasonable
future test; no claim of improved L2/fit/speed/peak memory, no larger run queued.
SOAPaxis cap256 would skip additional384wide FFNaxes; use an explicit512cap
control (128cap512 has unchanged coverage) when comparing broader architectures.

Current live evaluation28000: training16phase5.5855% versus reserved8
10.0812% pooled, terminal10.8993%. Nonzero training error allows a capacity
hypothesis; the generalization gap and optimizer convergence prevent diagnosing
capacity from this pair alone. Let the existing30krun and automated audits finish.
No architecture/optimizer/runtime/default/GPU/history changes. Task-end for this
clarification: task, PROGRESS and compact capacity receipt updated; CPU shapes,
JSON and diff checks only. Exact resume remains NEW03 journal/final audit outputs.

## Cross-attention and head-dimension clarification

Source confirms the four previously named condition-reading blocks ARE four
heterogeneous cross-attention blocks (net.cross). Query embeddings come only
from spatial coordinates; boundary and domain-function MLP streams supply
separate K/V projections/measures in every block. Each block averages the two
stream messages, adds the query residual and a feedforward residual. This
variant has no separate context/query self-attention encoder. Three FiLM layers
then decode all times from the shared conditioned spatial representation.

Current Q/K/V projections each output total width128 then reshape H4,D32;
D is not an independent config in this implementation. Width192/H4 givesD48,
width256/H4 givesD64. Width128/H8 would reduceD16, not increase per-head capacity.
Width128/H2 yieldsD64 with fewer heads and unchanged projection parameter count;
it is a different attention decomposition, not an overall-capacity increase.
Normalized integral attention stores per-stream per-head D-by-D feature moments;
D32 may limit an individual read but is not proven the cause of current error.
Prefer retaining H4 when comparing modest backbone widening, instead of claiming
more heads inherently increases expressivity. Decoupled Q/K/V head size would
require a source change and new validation; not implemented or requested here.

Task-end: code/config explanation only, no runtime/architecture/default changes.
Updated task and PROGRESS; source shape arithmetic and git diff checks only.
Existing authorized FP32 runner/final audits remain the exact resume point.

## Encoder capacity, computation and RTX5090 assessment

Cross-case FP64 runtime explanation: earlier two-phase/precipitate FP64 timing
results remain valid. Fast absolute runtime does not imply near-FP32 arithmetic
throughput or enhanced FP64 hardware. Precomputed spatial operators, shared
spatial preparation/time JVP and small point networks keep total work low;
real updates also include graph memory access, elementwise functions, dispatch
and optimizer operations, whose relative costs differ by workload.

Ordinary-policy corrosion full-update controls (same paired inputs) actually
measure12.487ms FP32 and35.132ms FP64, a2.8135x slowdown, still practical in
absolute latency. See archived local-ad-performance-graph-ordinary-float32/64
JSONs and local-ad-performance.md. The older203/227ms comparison under serial
deterministic graph lowering is a separate valid measurement; it is not a
universal11.5%precision cost. No new current-operator FP32 run exists.

Matched earlier nonlocal two-phase FP644x64/Nt8 gives16.732ms/compiledSOAP
update. Current neural operator has905089parameters, four condition-reading
blocks, B16geometries/17times/1351paddednodes; probe mean575.272ms includes
refreshes. It evaluates367472padded output/residual sites versus32768 for
one64x64/Nt8trajectory (11.214x more sites, not an operation-count ratio), and
requests30k rather than5kSOAP steps (6xupdate budget). Different PDE/architecture
and derivative paths prevent treating these as a controlled speed comparison.
Current pre-optimization loss/gradient dot trace232.058GFLOPs supports substantial
matrix work, not measured GPU utilization or an isolated FP64 bottleneck.
No precision change follows this explanation; current FP64 run continues.

Task-end for this clarification: old benchmark and computational-structure
records re-read; numerical arithmetic/JSON inputs checked, no new GPU tests,
process/production/default changes. This task and PROGRESS updated;
git diff --check passes. Resume existing30k journal and automatic final audits.


Device/archived-benchmark follow-up: user correctly recalls valid FP64 feasibility
measurement. Original corrosion steps203.262ms FP32/226.737ms FP64 (+11.55%) used
legacy deterministic policy; later HLO audit establishes serial graph lowering,
so do not infer a near-equal hardware throughput ratio or transfer that slowdown
to the current NO. Pure4096 FP64 GEMM was .636185TFLOP/s, versus FP32 2.39496 in
that JAX highest-precision kernel; neither FP32 result nor the task ratio is a
vendor peak measurement.

New metadata-only CUDA Driver API query PASS with no current CUDA context,
allocation or kernels:82SM,CC8.6, single/double precision performance ratio64.
NVML reports3090/49152MiB/PCI2204:10DE, currentSM1950MHz/max2100MHz/power350W.
CUDA-visible total47.415GiB is recorded separately from NVML48GiB. A3090-layout
clock estimate at1950MHz gives .6396TFLOP/s FP64, compatible with old .636185;
clock and benchmark were not simultaneous and this is not a new timing.
Standard3090 is24GB, so an expanded/platform-customized board is plausible,
but no software query confirms physical modification. No observed enhanced
FP64 ratio. Receipt docs/tasks/confined-ch-device-identity.json; old benchmark
retained unmodified. Task-end: identity/hardware receipts, this task, DECISIONS
and PROGRESS updated; strictJSON/diff pass, GPU run unchanged (journal18.5k).


Hardware clarification requested by the user: this machine still reports RTX3090,
CC8.6,49152MiB. Both3090/5090 have nativeFP64=FP32/64. Deriving reference peaks
from official CUDA-core counts and nominal boost gives3090 FP32/FP64=35.6864/.5576
TFLOP/s,5090 FP32/FP64=104.8832/1.6388TFLOP/s, approximately2.94x higher theoretical
FP64 for5090.
The earlier warning about5090 FP64 versus AI TOPS was not a claim that3090 is
faster; make the shared1/64ratio explicit. No5090training benchmark exists.
Updated hardware receipt/this task/PROGRESS; JSON arithmetic and diff check pass.
No runtime, numerical or process change; existing30k pipeline remains active.


User asks about increasing network size, encoder efficiency and an available
RTX5090. This investigation stays inside the active task, using CPU-only shape
traces so the authorized beta09 GPU training continues without contention.
No model/optimizer/PDE/default change or new GPU run.

Current heterogeneous model has 905089 trainable parameters, width128, four
heads, four spatial condition-reading blocks and three temporal decoder layers.
Boundary stream192points and domain stream1351paddedpoints retain physical
arc/area measures. Source confirms encode/prepare occurs once per geometry
outside the time-vmap/JVP. All17times share the spatial representation; the
learned values must still be recomputed after each optimizer update.
Attention forms per-head feature moments and associative contractions, never
an Nq-by-Ncontext matrix. Actual geometry response correlation .97716/amplitude
.93275 and wrong-condition phase errors49.8..80.6% from the selected width96
checkpoint establish condition dependence; they do not prove sufficient
fine-scale expressivity or an encoder-only bottleneck.

CPU abstract-shape dot audit at production B16/N17/Nx1351:

| Component | Traced forward dot GFLOPs | Share of counted dot work |
|---|---:|---:|
| Input lifting |1.645|2.10%|
| Shared spatial condition reads |26.517|33.90%|
| Temporal decoder and time JVP |49.070|62.73%|
| PDE/residual metric |.994|1.27%|

Full loss value/parameter-gradient trace232.058GFLOPs. These are pre-optimization
JAX dot counts, not GPU wall fractions; omit elementwise functions, graph
memory/gathers/scatters, compiler reuse/fusion, optimizer/QR and bandwidth.
A genuine stage timing needs an exclusive GPU probe after current training or
on the new machine. Input-encoder optimization alone has limited counted
matrix work; the decoder/JVP is a substantial computation target too.

| Width / four heads | Trainable parameters | Features per head | Skipped matrix axes with SOAP cap256 |
|---|---:|---:|---:|
|128|905089|32|1|
|192|1984321|48|9|
|256|3481345|64|9|

At192/256, eight added skipped axes belong to four condition MLPs, because their
hidden widths become384/512. The original skipped axis is the 6*width time-MLP
output. Source init_conditioner applies the cap independently per axis.
Explicit cap512 restores the same one-skipped-axis coverage; at128 its
preconditioner shape pattern/state bytes are unchanged. Optimizer state at192
is71.70MiB withcap256 /89.70MiB withcap512; this is not activation/peak memory.
A reasonable next capacity control is192/fourheads with explicitcap512, keeping
Fourier bandwidth, geometry bank, mesh, time samples and physics unchanged.
Use equivalentcap512 at128 if needed for a single explicit shared optimizer
configuration. No promised L2 improvement, bigger-network run or default
promotion. Width96->128 gains were small at10k/15k; further gains are unproven.

Clarification: h=.030 refinement has only been checked on two native references;
no denser512geometry operator has been trained. Current neural L2 is against
its aligned native reference, so denser discretization does not by itself
resolve representation/optimization error. At16.5k train16monitor7.82% versus
reserved8geometry12.29%; this is not all512training-domain error or a new test.

Official NVIDIA specifications show standardRTX5090 32GB and CC12.0; CUDA's
12.x throughput table lists FP32:FP64=64:1. Current card reports48GiB. Existing
128width measured active peak17.354GiB may fit a standard32GB device, but new
192width activations/workspace/allocator and driver support require a fresh
production probe. No speedup measured on5090; AI TOPS do not determine FP64
training throughput. Current JAX0.10.1/CUDA13 packages are recorded; do not alter
the active environment. Source links and limitations:
docs/tasks/confined-ch-encoder-hardware-assessment.json.

Audit script/receipt: confined_ch_encoder_capacity_audit.py and
confined-ch-encoder-capacity-audit.json. Actual CPU executions PASS, full forward
component dot sums match, independent3x4x5dot arithmetic check PASS; updated
module compilation/JSON/diff checks PASS. Records changed: this active task,
PROGRESS, DECISIONS, artifact index, script and two compact receipts. No reference
labels loaded, no GPU contention, history operation or additional training.
Exact resume remains current beta09 journal; complete its fixed30k and automatic
validations before scheduling a separate baseline/192 profiling/accuracy control.

## Current live assessment at 16000 evaluated steps

Observed 2026-10-30T15:19 UTC. Runner 198579 and trainer 198583 are live;
latest logged training step 16400/30000, latest complete evaluation 16000.
GPU utilization was 83% at the adjacent status check. Native phase relative L2
is 12.3449% pooled and 13.4190% terminal; 16 training monitors give 8.0431% and
8.5648%. Paired checkpoint marker checkpoint_016000.json is present. Final
30k native/dense61/geometry/CPU audits are still pending, no error reported.

At the last common budget of 15000 steps, beta2=.9 gives phase 12.9692% pooled
and 14.0109% terminal, versus .99 at 12.5810% and 13.6436%. The .9 run is slightly
worse there, by 0.3883 percentage point pooled. Its small 5k/10k advantage did
not persist; do not claim faster convergence or a stable optimizer ranking.
New pooled L2 at 5k/10k/15k/16k is 19.1404/14.8891/12.9692/12.3449%, with local
fluctuations (14.5k->15k) and continued overall decrease.

Recent evaluation-inclusive rate is 0.601103s/update, about
2.34 hours from the last evaluated step to30k, plus final audits.
Same eight reserved/reused development geometries, native graph/BDF reference
and one initialization; no fresh-test, FEM or repeated-seed accuracy claim.
Selected width96/30k result remains unchanged. Live receipt:
docs/tasks/confined-ch-beta09-live.json, with closed metrics/loss rows, matched
steps, process identities and timing calculation.

Session task-end: status inspection only; no process, numerical, optimizer or
default changes. Updated live receipt, artifact index, this task and PROGRESS.
Strict JSON and git diff --check pass. Exact resume: inspect the existing NEW
journal and collect its automatic final receipts; do not relaunch or add jobs.
Task remains ACTIVE; no blocker.

## Current user-directed beta2=.9 restart (supersedes prior live status)

User explicitly requests beta2=.9 and a direct cold rerun. The .99runner177950
and trainer184137 were intentionally terminated after a verified closed15k
model/optimizer pair, with exact command/session identity checked before signals.
CPU loading verifies both payloads finite and both optimizer counters15000.
No history/config/reference files removed. Stop receipt:
docs/tasks/confined-ch-width128-beta09-switch.json. Old journal now explicitly
intentionally_stopped_for_beta09, with its previous snapshot retained. This
is not a failure of the numerical method or a completed30k result.

At15k the .99control has pooled/terminal phase12.5810/13.6436%; training16monitor
9.0950/9.6764%. This provides a same-budget intermediate comparison; width96
fixed30kbest9.6609/10.5131% remains selected. Old run retains all paired states;
it can support explicit future state recovery, not silently count as complete.

New preset configs/train_operator_smallscale10_512_width128_beta09.py changes
only train.soap_b2=.9. Inherited shampoo_beta=-1 couples Gram EMA to .9. Same
width128/512inputs/B16/Nt16, Fourier arrays, LR, FP64ordinarySOAP, physical scheme,
runtime seed and stateless sample counters. Fresh parameters/optimizer, no old
learned checkpoint imported. Config dictionary parity and validation PASS.

New detached runnerPID198579/trainerPID198583; journal
/root/autodl-tmp/tf-logs/confined_ch2d/20261030-confined-width128-beta09-01-process.json;
run
/root/autodl-tmp/tf-logs/confined_ch2d/20261030-confined-geometry-b16-t16-width128-beta09-30k-01.
GPU was verified1MiB/idle before launch. No new fullbank mesh/preflight repeat
is needed for scalar beta change; production keeps finite-gradient checks.
CPU initial-model audit now PASS:119actual array leaves are bytewise identical
to the .99control (maximum difference0), initial cohort metrics match, saved
training/model/optimizer/runtime sources are identical. Receipt:
docs/tasks/confined-ch-beta09-initial-audit.json. Initial evaluation completed
at173.11s; first compilation is now complete (latest live assessment above).
Runner automatically completes30k/native/dense61/geometry/CPU
audits and same-step/elapsed-time curves against old partial .99and width96best.

Actual old .99draw list was not exported before intentional interruption
(production writes it at completion). Source/config determinism establishes
the shared expected stream; new actual full draw trace will be independently
checked against global keys and the saved width96prefix. Do not claim a direct
old/new actual-trace comparison unavailable from the interrupted run.

Exact resume: inspect NEW journal/PIDs/metrics above; do not relaunch .99 or
duplicates. Expected compact outputs: docs/tasks/confined-ch-beta09-initial-audit.json,
confined-ch-beta09-audit.json and confined-ch-beta09-results.json. No new final
L2 yet; intermediate metrics above, no default promotion or optimizer implementation/replication study.
Current session adds beta09preset/initial-audit/runner, stop receipt and README/
index/memory records. Changed modules compile, width128beta-only dictionary
parity and diff check pass. Task remainsACTIVE for the authorized run.

Session task-end: beta09 runner/trainer identities live, prior owned processes
absent; old15k finite model/state retained. Artifact index marks .99partial
intentionally stopped and records the active .9control. JSON/config checks and
git diff --check PASS; no expensive unchanged suite repeated, no Git history
operation. Resume solely through the NEW journal, then collect the automatically
written final receipts/plots and compare common evaluated steps before any
recommendation change. No blocker; completion pending the running30k pipeline.

## Earlier .99 live assessment (historical, stopped by user-directed restart)

Training remains healthy under runner177950/trainer184137; GPU100%. About14100
updates completed, latest full cohort evaluation14000/30000. Native online phase
13.1092% pooled/14.1400%terminal, train16monitor9.2773/9.8746%. Same14000step
width96 has13.4886/14.4831% and training10.5213%; width128has a modest0.3794
percentage-point pooled improvement, not a clear large accuracy gain. At10k,
width9615.3286/16.4277% versus12815.1687/16.2485%.

Current candidate decreases from15.1687% at10k to13.1092% at14k. Original
width96 fixed30kbest9.6609/10.5131% remains selected; comparing that final to
candidate14k alone is not a matched-budget ranking. No new final/dense/CPU
audit available yet. Recent periodic-evaluation-inclusive rate .60104s/update,
roughly2.67hours remaining training plus validation. Live receipt
docs/tasks/confined-ch-width128-live.json; same8reuseddevelopment cohort,
online metrics only. Continue fixed30k plan and automatic validations, no
early stopping or new jobs. Memory/index/PROGRESS updated; no numerical change.

## SOAP settings check during the running control

User asks about slow convergence and actual SOAP settings. Read the saved run
config, project factory and installed pinnedSOAP_JAX0.2.2 signature/source;
no optimizer change or implementation investigation. Settings: lr_init1e-3,
continuous Optax decay0.95 per1000calls, floor1e-5; b1=.9,b2=.99; inherited
shampoo_beta=-1 means Gram EMA usesb2=.99; basis refresh frequency5; max axis
dimension256; 1DpreconditioningFalse. Inherited eps1e-8, weightdecay0,
correct_biasTrue, moment/QR dtypes follow parameters. Global gradient clip10;
model/moment/preconditioner FP64, ordinaryGPU policy. First call initializes
basis without parameter movement, not a repeated per-refresh skip.

Optax exponential_decay default staircase=False is confirmed. Learning rate
is max(1e-5,1e-3*.95^(k/1000)), where k is the optimizer schedule count, not a
floor-divided staircase. At approximately14600calls it is4.729e-4; at30k about
2.146e-4. Floor is not reached in this run. Slow convergence alone does not
establish too-small LR, bad beta choice or faulty implementation.

CPU shape inspection confirms exactly one matrix axis above256:
.time.layers[1].weight has shape(768,128), axis0 is not preconditioned; axis1
remains active. 1Dparameters independently retain Adam-style updates. Gram
statistics update every gradient call; basis refresh occurs every5calls after
initialization. The256cap therefore matters to coverage, not just memory.
Potential later independent controls: b2/shampooEMA=.95 or cap1024; benefits
are unmeasured. Do not change the running width-only comparison or resume the
deferred SOAP repeatability/implementation branch from this parameter query.

Receipt docs/tasks/confined-ch-width128-soap-settings.json; saved configuration
and local factory are authoritative. Upstream pinned source checked at
https://github.com/haydn-jones/SOAP_JAX/blob/a7c92f174c49b13136294a94ae0d46266b4f671f/src/soap_jax/soap.py.
Task-end: add this parameter/shape receipt and note, dashboard link; JSON and
diff checks pass. Existing detached training/validation proceeds unchanged.

## Goal and acceptance

Execute the user-approved next steps from the density/capacity assessment:
increase width96->128 at unchanged h=.035 and numerical scheme, independently
check h=.035->.030 on two identical geometries, and determine measured memory,
speed and native phase accuracy. Preserve the selected best and all old files.
Keep the current512training/8reserved development bank, B16/Nt16, SOAP/FP64/
ordinary policy, Fourier frequencies/depth, hard IC/mass, eliminated mu and
stable-part residual norm. No reference-label training, seed tuning, custom
geometry bias, time windows or default promotion without completed validation.

## Baseline

Selected width96/30k run:
/root/autodl-tmp/tf-logs/confined_ch2d/20261030-confined-geometry-b16-t16-wide96-30k-retry-01.
Native phase9.6609% pooled/10.5131%terminal;61time9.1887%; train16monitors5.9243%.
512input bank:
/root/autodl-tmp/data/confined_ch2d/operator_geometry_smallscale10_512_v1.
Earlier matched20k width64->96 improves15.33->11.38%. Width128CPU preflight
905089parameters versus522145 at96, not yet GPU measured. System48GiB GPU,
baseline active peak15.72GiB, allocator35.56GiB; disk available14.33GiB after
authorized cleanup. No training running when task starts.

## Plan and gates

1. Explicit opt-in width128 cold30k preset, only backbone width changes.
2. Before GPU work, generate two h=.030 meshes and independent native/BDF
   references in a new small external directory. Same holes/IC/epsilon/delta.
   Compare on common physical queries and report reconstruction sensitivity;
   do not describe this alone as continuum/FEM convergence.
3. Full-bank width128 GPU probe using production loss/optimizer/sampling,
   finite gradients and several SOAP updates; measure active peak/warm time.
4. If feasible, detached fixed30k cold training through canonical train.py;
   paired checkpoint snapshots and durable process journal. Fixed final,
   compare10k/20k/30k milestones where available without selecting a best seed.
5. Native/dense61/geometry-response/CPU checkpoint-replay/input-stream audits;
   archive exact results, keep defaults unchanged unless a verified improvement.

Do not run FEniCS MPI concurrently with GPU training. Small mesh/native work
comes first. No full refined512bank before evidence/storage review.

## Earlier .99 launch status (historical)

Status: full-bank GPU preflight PASS; canonical cold30k training running.
RunnerPID177950/session
177950; journal
/root/autodl-tmp/tf-logs/confined_ch2d/20261030-confined-width128-02-process.json.
Intended run
/root/autodl-tmp/tf-logs/confined_ch2d/20261030-confined-geometry-b16-t16-width128-30k-02.
Refinement output
/root/autodl-tmp/data/confined_ch2d/operator_geometry_smallscale10_h030_check_02.
Blocker: none. Runner sequentially executes refinement, GPU12step probe, then
only on success cold30k and native/dense/CPU audits. Do not launch duplicates.

Setup attempt01 stopped before training because the JAX environment lacks
meshio. Preserve its failed journal/logs/partial mesh directories; no GPU update
occurred. Corrected mesh subprocess to the existing explicit fenicsenv Python,
as operator_precompute already does. Dependency check meshio5.3.5/gmsh4.15.2PASS;
no packages installed/changed. Setup attempt02 uses new output paths, same
numerical settings/runtime seed, not best-of-retries training selection.

Added explicit width128preset, mesh-check/probe/detached-runner scripts and
README/index pointers. Config validation and dictionary identity prove only
arch.width changes; seed/continuation absent from preset. Changed modules
compile, diff check passes. Next: inspect training progress through its journal,
then collect completed native/dense61/geometry/CPU-audit and comparison outputs.

## Completed spatial diagnostic

Two independent native h=.030 references PASS in about206seconds total, before
any GPU work. Source and generated arrays/configs remain separate. Same holes,
fixed IC, epsilon=.06, delta=.11; no localFEM or new training bank.

| Geometry | Original points | Refined points | P1 pooled/terminal difference | P0 pooled/terminal difference |
|---|---:|---:|---:|---:|
| 514 |1221|1596|1.7281/1.7173%|11.5692/10.8576%|
| 518 |1295|1685|1.3375/1.2005%|11.2732/10.6463%|

Common P1 support21966/23318 and23104/24390physical uniform probes; P0 common
support23294/23318 and24366/24390. Both exclude holes/invalid cross-hole
triangles. P1 initial reconstruction errors coarse3.85/3.74% versus fine2.88/
2.85%; P0 coarse/fine initial difference about21/20.5%. Thus meshfield comparison
is materially reconstruction-sensitive; neither column is a pure nonlocal
operator error or a formal continuum convergence certificate. No IC-error
subtraction. Continuous-field P1 differences are smaller than current NN errors
on these cases but do not establish a universal density bound near walls.

Time refinement phase<=1.061e-9, mu<=8.146e-10; massdrift<=2.385e-16, sampled
energyincrease0. Barycentric constant/coordinate reproduction and positivity
checks PASS on actual common probes. Compact receipt:
docs/tasks/confined-ch-width128-spatial.json. Priority remains the separated
width128control; no full refined512bank launch or default change.

## GPU feasibility and live cold run

Full520geometry bank, production B16/17time loss and12SOAP updates PASS. Fixed
Fourier arrays exactly match the saved96width cold model; first12actual geometry
draws match its saved stream; temporal draws match baseline. Expected eight-key
step payload has no future labels. All12losses/gradients finite; loss .0115883
to .00877716. This is only a feasibility probe, not an accuracy/convergence gain.

Measured activepeak18633626624bytes=17.354GiB (48.8% of allocator limit35.56GiB),
live9399835648bytes. First compiled update359.232s; mean10warm updates .575272s,
including SOAP preconditioner refreshes. Full probe wall503.966s. An XLA delay-
kernel autotuning timer warning occurred during initial compilation; no failed
update or silent policy change. Steady timing excludes compiled firstupdate.
12probe updates are discarded; production restarts from common fresh runtime
seed through canonical train.py. No batch/precision/optimizer fallback.

Production trainingPID184137/session184137 launched after the GPU probe exited;
runner remains177950/session177950. Run path above now ends width128-30k-02.
Estimated30k update time4.79hours from short probe; allow about5-6hours with
setup/evaluation/audits, not a measured full-training duration. No new final L2
available yet. Currentbest96width9.66/10.51% remains selected.

Runner automatically follows training with native saved-field metrics/figures,
dense61time evaluation, geometry response, history report and independent CPU
24field replay/input/no-label audit. Full final results receipt will be
docs/tasks/confined-ch-width128-results.json; existing best is not overwritten.
No automatic default promotion. Parent journal records each command/log/PID,
heartbeat and fatal errors; don't start a duplicate while its child is active.

## Session task-end and exact resume

Changes: explicit width128preset; three scoped analysis/runner scripts; README
and artifact-index experiment pointers; spatial/probe compact JSON receipts;
DECISIONS/PROGRESS/current task. No PDE/network/optimizer-core code change,
dependency installation, reference training labels, fullbank refinement or Git
history action. Real spatial and GPU-probe executions/identity/physical gates
passed; changed-module compile, config width-only parity and diff check pass.

Task stays ACTIVE while authorized detached training/validation continues.
Exact resume: read the journal path above, verify live PIDs, then inspect
20261030-confined-width128-02-training.log and run/metrics.jsonl. On completion,
inspect final/audit/dense results, compare fixed30k and10k/20kmilestones against
96width; retain negative controls, update documentation/status and archive.
No user input or additional GPU is needed for the running control.
