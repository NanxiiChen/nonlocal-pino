# Add full local AD to the current channel PINO comparison

## Goal and acceptance

User asks whether local AD exists for this problem and to include it in the
comparison. Audit history; add a physically/configurationally matched local
spatial-AD control for current random-wall full NS--CH PINO, keeping accepted
nonlocal/FEM results. Preserve network, hard lift, time JVP, SOAP/cold20k,
actual geometry/time samples, nine nonlocal loss groups and local FEM references.
Latest steering explicitly permits sampled AD queries and requires independent
boundary losses. Use unbiased uniform subsets; equal PDE query counts are not
required. Keep full encoder geometry and complete independent boundary points.
Compare final same-three reserved fields, development convergence, current-card
time and memory; use approved solid LOWESS/violin styles after completion.

## Historical distinction

`twophase-matched-local-comparison.md` is prescribed-flow CH PINN,
not full NS--CH PINO. It compares local AD diffusion while retaining spectral
advection. `pino-local-ad-comparison-plan.md` confirms the current
channel lacks full local spatial AD. Existing channel spatial AD first helpers
still combine with nonlocal Laplacians; deleting a selector is insufficient.
Streamfunction velocity makes its local Laplacian a third query derivative.

## Scope / validation

Current GPU: RTX PRO6000Blackwell96GB, idle at task start. First qualify analytic
derivatives and parameter gradients, BC normals and conservative transport, then
probe full optimizer updates. Execution chunking may change only evaluation
order with exact loss/gradient accumulation, not effective samples/weights.
Historical current nonlocal full20k reference run is recorded in
`artifacts/channel_fem_comparison_validation.json`; references remain the same.
Do not silently alter geometry, output parameterization, residual scales or
training budget. No nonlocal win is assumed, no training references in loss.
No Git operations authorized; preserve all unrelated dirty numerical work.

## Status / blocker / next action

### Complete — production, independent audit and local publication

Final20k run, validation, three-case inference and independent final audit all
passed; journal finished2026-11-14T18:30:31UTC. Owner81883/trainer145877 exited
successfully. Final reserved-three means: phase2.1540%,velocity2.0866%,
pressure8.5370%,chemical16.0556%, versus nonlocal1.6435/1.5172/5.2840/13.9541%
and FEMPINO4.6465/4.3892/7.1489/43.7908%. Pressure FEM mean beats AD;
AD's worst-case pressure15.7110% must not be hidden by its8.5370%mean.
Pairing of initial model/SOAP and all20k input/time/geometry draws passes;
117source files and independent saved-field norms pass. Reference/cohort and
physical metrics are unchanged. See direct comparison:
`/root/autodl-tmp/tf-logs/channel-sampled-ad-20261114-03/comparison.json`.

Cold process36555.53s(~10.15h), warm inclusive median1.73796s,active peak26.1901GiB.
2048PDEqueries/timeblock9,B8,Nt36,all192independent boundary queries. Current-card
NLprobe1.60160s; FEM earlier same-cardprobe1.41693s. Historical nonlocal long-run
wall used another physicalGPU; do not label it a current-card process comparison.

Three-method preview05 fixes the convergence legend margin and retains all raw
metrics, density arrays and bootstrap trend arrays exactly. Four curated assets
(510,654bytes) and compact provenance are installed; example/root READMEs and
local website show all three methods. Nine-page build,342local links, raw-table
values, PNG/PDF byte identity and editable-font checks pass. Existing legacy
assets and the former website anchor remain available. Source/UI changes only;
no selected default, numerical source, Git operation or deployment changed.
Receipt: `artifacts/channel_three_method_publication_validation.json`.
Figure inputs: `artifacts/pino_three_method_figures.json`.
This child task is complete; subsequent cases/time-fusion work remain active in
`docs/tasks/pino-three-method-suite.md`. Background training was not restarted.

### Historical recovery — sampled CUDA graph allocation

Read-only inspection at2026-11-14T07:13UTC found owner37019 had stopped at
05:52UTC, before any cold20k run. Its1024/block18 probe completed60updates:
warm inclusive median0.661504s, live-array peak23.2383GiB; the allocator pool
reserved approximately92.12GiB. First compiled update1064.481s; total probe child1179.358s.
Single-development60-step errors: phase33.4435%,velocity8.0159%,pressure90.6569%,
chemical332.7287%. These are initial optimization values, not final/generalization
accuracy. Phase at initialization was47.94095%.

The2048/block9 probe failed while instantiating a4713-node CUDA command buffer
(two graphs alive), explicitly reporting driver OOM. It did not finish even one
update and has no warm timing. Reserving97%of GPU memory leaves little space for
driver graph objects outside the allocator; this is the first recovery hypothesis,
not proof that all driver memory pressure is explained. Preserve failed logs.

The queue launcher now requires an explicit AD allocator fraction for sampled
runs and records each stage's environment. New owner81883 uses80%for both AD
probes/production and97%for the full nonlocal probe, ordinary policy and unchanged
command buffers. Same1024/block18 and2048/block9 probes, then current-card nonlocal,
pairing/resource gates, predeclared2048/block9 cold20k and final audits. No
scientific/training configuration or sampled counts changed. All117saved numerical
Python sources match the completed1024probe. Missing/out-of-range CLI checks and
changed-module syntax pass; existing13numerical checks remain valid.

Current journal:
`/root/autodl-tmp/tf-logs/channel-sampled-ad-20261114-03/sampled-journal.json`.
Owner log:
`/root/autodl-tmp/tf-logs/channel-sampled-ad-20261114-03-owner.log`.
Exact launch and old failure/short metrics: `artifacts/channel_local_ad_validation.json`.
Next: monitor this new finite queue. It automatically starts cold20k only after
resource/pairing success. A second failure stops subsequent stages; diagnose
before changing execution further. Do not resume the failed02or cancelled full
queues, and do not report the short1024error as the2048production result.
No Git, default change, deletion, or figure update in this status/recovery turn.
Memory changed: this task, PROGRESS and DECISIONS; launcher and validation receipt
record the narrow resource correction. Task remains incomplete with a live queue.

### Earlier sampled queue and original plan

Full-query resource owner15899 and waiting cold20k owner34407, including owned
worker groups, were stopped before any long run when the user requested sampled
AD. The full probe completed its first update with active85,548,212,224bytes
(79.67GiB), but has no qualified warm timings or completed60-step history.
Its journal records `stopped_for_sampled_ad`; never use it as a completed arm.

New finite owner37019, started2026-11-14T05:06:58UTC:
`/root/autodl-tmp/tf-logs/channel-sampled-ad-20261114-02/sampled-journal.json`.
Order: input-only screen;60updates AD1024/block18;60updates AD2048/block9;
current-card nonlocal60probe; cold/input/resource qualification; predeclared
AD2048/block9 cold20k; validation of actual queries/rolling inputs; same3reserved
FEM references; independent final metric/source audit. Failures stop later stages.
This queue **does automatically launch the cold run after successful probes**.
No architecture/BC/default/Git change. Config:
`examples/channel_nsch2d/configs/train_operator_local_ad_sampled.py`, with explicit
production overrides `sample.ad_sampling.core_count=2048,train.microbatch.size=9`.

Sampling draws K distinct uniform interior nodes from M=(Nx-1)(Ny-1) per geometry
occurrence, independently each step. Multiply each squared PDE residual by
M*m_i/sum(m) and average K; five boundary terms use unchanged full quadrature.
All36times share those spatial queries per geometry, preserving tensor splitting.
The query RNG namespace271828 does not alter condition/time/online geometry draws.
Actual IDs are stored int32 and audited in bounded16-step chunks. The first new
test exposed implicit int64 under test x64 mode; return dtype is now explicit
int32. Both sampling tests pass in96.003s; see
`/tmp/channel-local-ad-sampling-tests-qualified.log`. They independently check
the expectation across every possible subset, full-subset loss/parameter-gradient
equivalence, unchanged five BC terms with a smaller subset, and exact replay.
Next: inspect the finite resource/production queue; after
completion integrate actual AD results into the existing plots/tables. Do not
restart the cancelled full-query path or claim convergence from short probes.

### Earlier implementation qualification

Active, no blocker. Full local AD dispatch/config/payload and spatial jets are
implemented; the original shared loss retains all nine terms, physical SBP
quadrature and conservative product-rule phase flux. No field stencils are
constructed. Default nonlocal config/architecture/optimizer are unchanged.

11 focused tests pass (161.02s CPU): analytic jets including third potential
derivatives, nonzero-divergence flux product rule, independent single-point
network Hessians, finite-difference parameter gradient, full/block loss,
gradient and one SOAP update, historical encoder/quadrature bitwise identity,
invalid-config rejection, existing FE weak checks and time-batching checks.
Log: `/tmp/channel-local-ad-qualified-tests.log`. An earlier raw-bank comparison
incorrectly compared FP32-cast inputs with saved FP64 construction arrays; the
test now correctly checks raw/raw exact equality and separately validates casts.

The earlier resource-only owner15899 started2026-11-14T04:41:55UTC:
`/root/autodl-tmp/tf-logs/channel-local-ad-20261114-01/probe-journal.json`.
It screens all declared geometry inputs, runs60full AD updates (timeblock2),
then a current-card nonlocal60probe and cold/input/resource qualification.
That resource phase and its later waiting production owner were cancelled under
the latest user steering above. Preserve its partial logs; do not resume it.
The new sampled phase is the sole live queue. Keep code unchanged during
resource/production execution; the queue checks source identity before cold20k.

## Session changes and completion boundary

Added `local_ad.py`, explicit full/sampled AD configs and focused tests. Updated
the existing data/loss/config/online/validation dispatch and training resource
recording; sampled IDs use an independent namespace and the common BC code.
`docs/tasks/channel_ad_queue.py` and `channel_ad_support.py` own finite runs and
paired audits. README documents optional AD, sampling weights and independent
boundaries; `artifacts/channel_local_ad_validation.json` retains checks/paths.
Memory updated: this file, `docs/PROGRESS.md`, `docs/DECISIONS.md`.

At session close, sampled1024 probe37422 is in its initial GPU compilation after
the finite16+802input screen passed. Initial development evaluation is finite
and matches cold nonlocal within roundoff. All13focused checks pass in two
recorded groups; syntax and `git diff --check` pass. Queue37019 remains active
and will qualify all three resource arms before launching2048-query cold20k.
No new final accuracy or warm sampled timing is yet available. No figures,
website assets, production defaults, unrelated files or Git history were changed.

## Read-only check during ceramic figure completion

The foreground ceramic plotting task is complete and archived. No channel
numerical source or job was changed. At 2026-11-14T05:44:30.597982+00:00 the
sampled queue remains running. Stage status: screen passed, probe-ad1024 passed, probe-ad2048 running.
Resume by checking its journal; retain the existing finite automatic queue.

## Suite checkpoint at2026-11-14T07:40UTC

Owner81883 remains live;1024/block18 with80%allocator completed60updates,
median inclusive0.658360s,active peak23.134825GiB. Single-development phase/
velocity/pressure are33.4469%/7.9709%/90.5187%, still short-probe values.
The2048/block9 recovery probe is now running as104897. The broader five-case
suite owns the single active task; Electro owner104855 waits for this channel
queue to complete successfully, then runs three serial cold10k controls. No
channel source was changed by that implementation; all117frozen files match.

## Final session / task-end

Completed the genuine sampled local-AD cold20k control, independent common-case
audit and three-method local publication. Added final component figures and a
compact direct-path manifest; updated the README/site text, preserved old assets
and the prior chapter anchor, and verified the nine-page build. No browser GUI
QA was performed; source figures and all generated local links were inspected.
Full suite remains active. Updated memory: parent task, PROGRESS, DECISIONS,
this archived child and the direct-path suite inventory.
