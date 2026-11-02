# Random smooth-wall NS–CH operator — completed and selected default

## Goal and outcome

Implement geometry-only operator learning with independently randomized smooth
upper/lower walls, fixed initial droplet/materials/driving, online rolling
geometry inputs and independent local FEM evaluation. The user accepts the
final physical fields and authorizes selecting this result as the default and
committing source/records to local dev. All finite training/audit stages are
complete; no additional training, FEM, cleanup, merge or push is queued.

## Final configuration and artifacts

- Canonical entry: examples/channel_nsch2d/train.py.
- Default: configs/train_default.py -> train_operator_random_walls_width192_batch8_time20.py.
- Width192/4heads,2condition blocks/3decoder layers,64spatial Fourier directions.
- SOAP10000cold updates,lr1e-3,beta1=beta2=.9,precondition every5; ordinary FP32/highest.
- Domain192x80subdivisions/15633nodes;40fresh stratified continuous times per
  update, physical timeJVP. Two20-time gradients average before one SOAP update.
- Integral moment-corrected spatial PDE/BC operators: quadratic kernel,
  horizon.075,capacity128,degree3. AD streamfunction curl remains explicit.
- Original nine soft losses with fixed normalization; muoutput4/row1/BC4.
  No tanh transform, gradnorm, causal weighting or stagger in the selected run.
- 16candidate rolling geometry slots/B8with replacement; replace2every50updates,
  2CPU workers, continuous constrained16-coefficient random-wall prior.
- Run: /root/autodl-tmp/tf-logs/channel_nsch2d/20261102-pro6000-integral-random-walls-online-width192-b8-time20-10k-01.
- Checkpoint: same run/model_final.eqx, byte-equal to ckpt_010000.eqx.
- Nominal reference: /root/autodl-tmp/data/channel_nsch2d/random-walls-20261102-02/development.
- Reserved references: /root/autodl-tmp/data/channel_nsch2d/random-walls-20261102-02/unseen3/manifest.json (three cases).
- Owner journal: /root/autodl-tmp/data/channel_nsch2d/20261102-pro6000-integral-random-walls-online-width192-b8-time20-10k-01-process.json.
- Full [compact validation receipt](../channel-random-wall-pilot-validation.json).

Training completed2026-11-02T14:28:24UTC; all ten owned stages exited0 by
14:40:19UTC. Historical101source files matched before this authorized default
promotion. Fresh full model/SOAP, actual geometry/time/slot/draw streams, final
pool inputs, hard IC/wall/flux/divergence, replay and240fresh-time full-grid
canonical residual probes passed their declared consistency checks. These
checks do not assert a continuum residual or independent nonlocal convergence.

## Accuracy and limits

| Evaluation | Phase | Terminal phase | Velocity | Pressure | Minority field | Chemical potential |
|---|---:|---:|---:|---:|---:|---:|
| Development |2.6781%|4.0550%|1.6700%|5.8931%|7.0768%|14.1030%|
| Reserved3mean |1.8396%|2.2708%|2.0160%|5.4118%|4.8474%|15.9168%|

Each geometry has3840physical queries and13snapshots0:.5:6; flow/muexclude
unprescribed time0. Worst reserved pressure9.0316%, mu19.9950%. Final interface
areaIoU86.14/94.25/93.64%. Only three predeclared, reused reserved cases were
solved; population/cross-seed accuracy remains unverified. Geometry-centered
pressure-response error remains73.0% relative to the small between-case signal,
despite low overall pressure L2. Nominal scalar phase minimum2.0476% at8500was
not saved; do not present it as a recoverable best checkpoint.

The old128/B2/5000and current192/B8/10000comparison changes width,batch,budget
and GPU together. At common5000steps phase6.5285% ->3.6407%, velocity4.4152%
->2.8910%, pressure6.5547% ->4.7353%. It is not an isolated capacity ablation.
Final nominal and reserved physical figures are under
/root/autodl-tmp/data/channel_nsch2d/random-walls-field-preview-20261102-01/current-width192-step10000
and /root/autodl-tmp/data/channel_nsch2d/random-walls-reserved-preview-20261102-01/current-width192-step10000.

## Resources, default verification and session end

Qualified RTX PRO6000 96GB timeblock20peak88.0555GiB, explicit97%allocator.
Actual training warm1.8838s/update, wall19514.65s (~5.42h). README documents
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.97` and explicit block2(~19GiB)/block8(~40.2GiB)
alternatives. No automatic scheme/block fallback or fitted-state import.

New default exactly equals all fields of the saved successful configuration
after automatic run-name normalization and runtime seed resolution. All56
config factories imported from verified source paths: only train_default changed
its values; all55historical factories match the frozen pre-promotion sources.
Historical integral ancestor, geometry sketch and legacy-RNG test now import the
explicit stronger preset, avoiding a circular reference to the moving default.
Four focused capacity/legacy tests pass; earlier17unique geometry/operator/
rolling/canonical-training tests remain valid. No unchanged expensive suite rerun.
AST/JSON, local documentation links and git diff --check checked before commit.

Full raw5000/10000L2/loss curves, their exact data, plot source and verification:
/root/autodl-tmp/data/channel_nsch2d/random-walls-convergence-20261102-05.
Both comparison PNGs visually inspected; no smoothing/fitting. The canonical
recorded-history plotting helper was also rerun on the complete10000logs; its
width/optimizer title now follows the saved config, retaining all metric checks. Curves monitor one development
geometry; no multi-geometry convergence history exists. Phase/pressure oscillate
late while velocity and chemical error generally decrease.

Diff-level changes include shared16-parameter geometry/host-JAX boundary values,
fixed-shape data/rolling/cohort handling, matched FEM query safeguards,
capacity/resource/replay controls and finite plotting/training helpers. This
session promotes the default, pins historical imports, updates stable README/
development guidance, completes the compact receipt and archives this task.
Source and compact records only belong in the local dev commit; images, weights,
solver data and logs remain external. Git authorization does not include push.

Status: COMPLETED. Blocker: none. Next action: none; wait for a new user request.

## Historical implementation and session log

The entries below preserve observations at their original times. Their earlier
pending/default-preservation states are superseded by the completed summary above.

## Checkpoint and implementation

- Local dev checkpoint3ac7070 created; tree was clean before numerical changes.
  No push or merge was performed for this commit-only request.
- Explicit random_spline/C3 geometry with8coefficients/side, exact host polynomial
  bounds and shared NumPy/JAX values/slopes; fixed encoder structure retained.
- Generalized initial/rolling/reserved shape and direct-draw records to16values;
  separate cohort schema/common physical envelope, reference CLI and strict cold
  geometry-extension audit. Legacy absent-kind RNG/payload arithmetic preserved.
- First production nominal integral construction passes:21--113physical neighbors
  within padded128, condition250.93, polynomialerror2.95e-12,1.31sCPU.
- Four geometry/JAX/legacy/payload tests passed20.56s. Combined original16test
  invocation:15passed, one synthetic reporting fixture failed because phi=1 made
  its minority denominator zero. Fixture corrected tophi=.5; actual canonical
  six-update NS--CH/integral/JVP/SOAP/rolling/audit test then passed25.10s.
  Additional chord-domain preflight test passed. No production exception hidden.

## Reference qualification correction

Attempt01: /root/autodl-tmp/data/channel_nsch2d/random-walls-20261102-01.
Four predeclared geometries; legacy40x16coarse mesh puts diagnostic queries outside
its polygonal wall for reserved1/2 (clearance-.00146/-.00108). The two remaining
trajectories continue to finish; all files preserved. This attempt cannot qualify
training. No extrapolation or changed physical BC is used. Pure host preflight
now rejects such queries before FEM factorization.

Selected explicit64x24/96x36reference pair, unchanged dt.05/.025 and all old
acceptance limits;64interval wall clearance>.0067 for all four frozen cases.
Attempt02/data paths are selected in the opt-in preset. Prior identities/draws,
IC/flow/materials and NO scheme stay unchanged. Full planned input-only prefix
screen (16initial+204online) is running separately onCPU, zero PDE/FEM solves.
Next: finish/reuse valid64x24trajectory exports, qualify new reference pair,
inspect four shape histories, then finiteGPUresource probe and ONE cold5000.

## Current handoff / task-end

- Attempt02 reference process631148 (unified exec session90161), four CPU workers;
  two complete64x24/.025exports reused byte-exactly from attempt01 after identity,
  query, dt/grid and common-protocol checks. All four now reached96x36fine solves;
  nominal/reserved0 reached80/240 at the last observation, others just started.
- Detached screen `channel-random-walls-5000-20261102`, supervisorPID704766,
  waiting for the exact reference process identity to exit. Journal:
  /root/autodl-tmp/data/channel_nsch2d/random-walls-20261102-02/training-pipeline.json.
  Reference journal: same directory/progress.json. Supervisor source snapshot:
  same directory/training-pipeline-owner.py. No recurring monitor/notification.
- Planned run: /root/autodl-tmp/tf-logs/channel_nsch2d/
  20261102-5090-integral-random-walls-online-width128-01.
  Inner finite owner journal will be /root/autodl-tmp/data/channel_nsch2d/
  20261102-5090-integral-random-walls-online-width128-01-process.json.
- Production resource probe is same attempt02root/resource-probe; fitted probe
  weights discarded, peak>28GiBstops before the16-slot pool training. Source,
  initial model/fullSOAP, all actual time/slot/draw IDs and final physical inputs
  are independently checked against the completed integral single-bump control
  where appropriate. Geometry streams and reference resolutions differ explicitly.
- Input screen220constructions passed83.94s; maxcondition250.93,
  polynomialerror4.57e-12. Seventeen unique focused checks pass after correcting
  one zero-minority synthetic reporting fixture; actual canonical6update test
  includes SOAP/JVP/integral/rolling and full16-parameter input reconstruction.
  Legacy saved GMLS evaluation payload rebuilt exactly. All config factories,
  changed AST/JSON and git diff --check pass. Compact receipt:
  [channel-random-wall-pilot-validation.json](../channel-random-wall-pilot-validation.json).
- Source changes: new shared geometry family/config/audit; parameter-dimensional
  data/rolling/cohort handling; matched FEM/common-query/preflight exports;
  finite reference/resource/training orchestration, scientific response figures,
  focused tests and README/development guidance. No generated binary tracked.
- Current default and old artifacts retained. New implementation is uncommitted;
  the requested BEFORE-change local checkpoint is3ac7070. No push/merge/cleanup.

Next action: read supervisor and reference journals. If a gate fails, inspect its
specific log; do not bypass it or retry automatically. On completion inspect
reference-response/*.png, run metrics/shape/convergence and unseen3/evaluation.json,
update the compact receipt and report nominal plus individual/mean/worst held-out
errors and aligned-shape diversity. Archive only when the finite attempt has
completed and results have been reviewed. Python training sources are frozen
while the pipeline is active; documentation updates remain independent.

## Status follow-up 2026-11-02T06:23:09.153424+00:00

Reference owner631148 and supervisor704766 are both alive. No GPU training
has started and no model/loss/relative-L2 logs exist for the planned cold run.
- development: latest saved fine step240/240, Newton3.
- unseen3/case_000: latest saved fine step240/240, Newton3.
- unseen3/case_001: latest saved fine step160/240, Newton3.
- unseen3/case_002: latest saved fine step160/240, Newton3.

All four fine solves are actively using CPU, with recent log updates; no
new failure or blocker. Approximate remaining reference time6--8minutes at
the observed rate, conditional on normal completion; pass/fail gates are still
pending. GPU idle (5090,4MiB). Queued finite probe/cold5000/held-out evaluation
remains unchanged. Exact resume: read progress.json and training-pipeline.json
under the attempt02root; do not start a duplicate run. Status-only verification:
process/log/journal checks and git diff --check, no unchanged tests rerun.

Two newly completed reference trios (development,reserved0) now pass both own
and common physical query gates. Across those two, worst space/time snapshot
difference:phi1.3303%,velocity.4350%,pressure1.8837%. These are independent
FEM refinement differences, NOT neural prediction errors. Other two fine solves
reached160/240; full cohort gate and queued training remain pending.

## Training status 2026-11-02T06:50:16.273941+00:00

Four reference trios pass (own and common physical queries); elapsed1584.33s.
Across all4, largest reference space/time snapshot differencesphi1.3303%,
u.4350%,p1.9049%; chemical is reported separately and not covered by that gate.
Reference plots inspected: centroid-aligned shapes differ moderately during
transit(t3--4.5), become closer atT6. At4.5the largest pairwise aligned mean
contour distance is.075875R0; this is inter-geometry shape diversity, not a neural
error. Do not claim radically distinct terminal morphologies.

7-update resource probe passed, warm.407075s/update, peak7.643GiB;337.36s
first update includes compile. Fitted state discarded. Formal cold5000 started
at06:33:05UTC. Actual model/fullSOAP cold equality with integral control passes.
At500->1000updates, nominal relative errors:phi22.6979%->16.7529%,
terminalphi26.8567%->19.7707%,velocity8.1650%->6.3536%,pressure19.7634%->12.3990%,
minorityfraction59.9775%->44.2682%,chemical199.4776%->116.6299%. All remain
intermediate optimization results; phase/minority and chemical errors still high.
No unseen geometry neural evaluation yet; it follows the same finite run.
Observed500->1000rate=0.440167s/update including monitors, roughly29.3minutes remaining training at that rate, plus audits.
GPU active; nvidia-smi24.24GiB is reserved process memory, not measured active
peak. No source/config/optimizer or sampling changes this follow-up, no new
training/FEM/cleanup/Git operation. Validation: journals/logs/processes, source
freeze preserved, updated JSON parses and git diff --check; no test rerun.

## Training status 2026-11-02T07:01:29.682825+00:00

Same cold run is active; latest update log2800/5000, latest full L2
monitor2500. From1000->2500updates, nominalphi16.7529%->9.8236%,
terminalphi19.7707%->14.5121%,velocity6.3536%->5.3139%,pressure12.3990%->7.4806%,
minorityfraction44.2682%->25.9582%,chemical116.6299%->64.4394%. Globalphi below10%
does not imply accurate droplet shape; minority/chemical still need improvement.
Recent monitored rate0.441502s/update; approximate remaining training
16.2minutes from last logged update, plus existing audits/3held-out
evaluations. GPU active, same owner/child PIDs704766/758199/758202. No scheme,
source, sample/optimizer/default or run-budget changes and no duplicate run,
FEM, cleanup or Git operation. Status-only validation: journal/log/process checks,
JSON parse and git diff --check; no test reruns. Exact resume remains the
attempt02training-pipeline.json and the inner process journal listed above.

## Physical-field visualization request

User asks to see current physical fields. Inspection confirms saved
log.checkpoint_every=5000, so no intermediate trained checkpoint exists; model
initial is the only weight file before the endpoint. Continue the same authorized
run to its imminent5000endpoint, then visualize actual prediction.npz onCPU.
No source/config update or invasive state extraction from live JAX is performed.

New postprocess-only helper docs/tasks/channel_random_wall_gallery.py verifies
complete endpoint, exact saved FEM arrays/physical identity and all six metrics
against summary.json. It generates phi/speed at0,1.5,3,4.5,6,pressure at.5,1.5,3,4.5,6,
plus physical-position zero-contour overlays with a common zoom scale and saved
IoU/minority/contour diagnostics. Spatial interpolation is display-only; no time
interpolation or centroid correction. Helper AST passes; actual data/figures
will be checked when ready. Script resides outside frozen training modules.

A CPU-only watcher (unified exec63460) waits for a completed summary, then runs
that helper once. Target external directory:
/root/autodl-tmp/data/channel_nsch2d/random-wall-fields-5000-20261102-01.
Original finite training/audit/cohort supervisor remains active unchanged.
Latest observed loss step4100; full4000metrics:phi8.2296%,terminal10.6119%,
u4.5834%,p6.3468%,minority21.7461%,mu51.3215%. Final errors/plots pending.

## Endpoint physical fields 2026-11-02T07:20:14.038061+00:00

Cold5000complete, trainingwall2605.04s(~43.42min), warm.422691s/update.
Final nominalphi6.5285%,terminalphi7.6275%,u4.4152%,p6.5547%,
minorityfraction17.2511%,mu44.9585%; phase range[-1.10627,1.05545].
User visualization request now fulfilled from actual final prediction.npz, not
initial/older/extrapolated model state. No additional model/PDE/FEM solve.

Gallery output: /root/autodl-tmp/data/channel_nsch2d/
random-wall-fields-5000-20261102-01/ (phi-fields,speed-fields,pressure-fields,
droplet-contours,convergence PNG/PDF; gallery-validation.json,plot-source.py).
All4field/contour PNGs visually inspected. Helper checked exact reference arrays,
physical identity, snapshot selection and all6summary L2 values; metrics use
raw original queries, display uses mapped bilinear interpolation only.

Observed shape bias: t0/1.5close, t3/4.5PINOtoo flat, terminal droplet shifted
slightly downstream. Position has not been aligned. TerminalIoU75.6282%,
terminalminorityL2=23.3206%,meancontourdistance.02533975 (~11.52%R0). Thus global
phi6.53%still hides meaningful localized shape/position error. Pressure/speed
match bulk patterns more closely; no pressure plot att0(unprescribed IC).

Existing inner owner currently atvalidate_run; finite3held-out evaluation
remains queued after its audits. Task stays active. Changed this subrequest:
postprocess-only docs/tasks/channel_random_wall_gallery.py plus DECISIONS,
PROGRESS, task and compact receipt; frozen training modules unchanged. No
new experiment, budget/default/SOTA/Git or cleanup action. AST, actual plotting/
array+metric checks, visual inspection and git diff --check passed; no unchanged
expensive tests rerun. Exact resume: existing training-pipeline.json/inner
process journal, then unseen3/evaluation.json and physical/time diagnostics.

## Convergence visualization 2026-11-02T07:29:03.678516+00:00

User requests convergence curves. Added postprocess-only
docs/tasks/channel_random_wall_convergence.py; no frozen training source edits.
Full11L2records(0,500,...,5000) and51lossrecords(1,100,...,5000) verified for
steps, finiteness, terminalsummary parity and total=sum(nineparts). The first
plot attempt correctly failed its mistaken assumed lossstart0; corrected the
postprocess schema to the actual logged step1, preserving all raw records.

Generated/visually inspected relative-l2.png/.pdf (phase pooled/terminal,
velocity/pressure, minority/chemical) and sampled-losses.png/.pdf (grouped PDE/BC
and individual5BCterms) at /root/autodl-tmp/data/channel_nsch2d/
random-wall-convergence-5000-20261102-01/. Includes curve-data.json, exact plot
source and convergence-validation.json. Raw observations joined only by lines;
no smoothing/fitting. Fixed-development FEM L2 is distinct from randomly sampled
normalized training MSE; no train-set or unseen L2 curve claimed.

Last500updates:phi7.9448%->6.5285%(17.83%relative improvement),
terminalphi10.3780%->7.6275%(26.50%),minority20.9936%->17.2511%. Pressure
6.2855%->6.5547%slightly rises; velocity4.5900%->4.4152%. Phase has not shown
a clear plateau; further improvement is plausible, not guaranteed and no extra
training is started. Chemical44.96%and minority17.25%remain limitations.
Existing actual-run replay/hard constraints status=replay_and_hard_constraints_passed;
owned pipeline currentlyunseen_geometry_evaluation, reserved evaluations pending. Task
stays active; exact resume remains the two journals and unseen3/evaluation.json.
New-request changes: plotting helper and task/PROGRESS/compact receipt only;
actual plotting/data checks, AST/JSON and git diff --check pass. No code tests
needed for this log-only visualization; no model/PDE/FEM execution, fitting,
new experiment, source/config/default or Git/cleanup change.

## Larger-batch/capacity training handoff 2026-11-02T07:51:50.774183+00:00

User asks current batch, proposes more geometry samples/network width and10k;
latest message explicitly asks B8 or16. Original update was B2geometry x40times,
width128/4heads(head32), pool16; pool size is not batch. New opt-in preset
train_operator_random_walls_width192_batch8.py selects width192/4heads(head48),
1232004parameters, B8 x40times, cold10000. Unchanged depths/spatialfeatures64/
timefeatures8, geometry prior/operators/IC/physics/BCs, SOAP(beta2.9)/LR schedule
and fixed nine loss scales. Timeblock2 averages20same-parameter gradients before
one clip/SOAP update. Changes floating-point reduction order, not the full-time
mean. Slot sampling remains with replacement;8draws need not be8unique domains.
No B16 trial: this larger batch already increases the measured update cost.

- Original completed3held-out cases: meanphi6.0754%,terminal6.7074%,u4.6609%,
  p8.1366%,minority16.0155%,mu47.4554%; worstphi6.1134%,u5.1113%,p10.9148%.
  Common-physical geometry-response correlationsphi.9135,u.9580,p.4809;
  pressure response amplitude.5246, so geometry-dependent pressure variation
  remains underpredicted despite moderate absolute L2. Finite3draw estimate only.
- Extended input-only418constructions(16initial+402online) pass435.80s with4CPU
  workers concurrent with GPU compilation; maxcondition250.93/poly5.60e-12.
  ZERO evolved training labels/PDE/FEM solves. Input prior/gates unchanged.
- Complete7update production-grid resource/SOAP/gradient probe passes; all finite,
 16input pool held on GPU, nominal query finite. Warm4.52441s/update,
 peak20384015360bytes=18.984GiB, JAX reserved23.52GiB; first429.77s includescompile.
 Probe fitted state discarded. Unfitted initial model/fullSOAP exported and
 byte-matched to ACTUAL formal cold state after startup. Estimated pure10k
 compute12.57hours, plus compilation/monitor/audits; not a completion guarantee.
- New finite screen session: channel-random-walls-192-b8-10k-20261102.
  Supervisor PID804542, inner owner806685, canonical training child806694 at
  handoff. Training backendcuda and GPU visibility checked. Initial files/source/
  full model/SOAP already saved; step0L2 monitor published (untrained), actual
  cold-state audit passed in the owner and99live source-byte checks passed.
  First optimizer update still awaits its own compilation at handoff.
- Pipeline: /root/autodl-tmp/data/channel_nsch2d/
  random-walls-width192-b8-10k-20261102-01/training-pipeline.json.
  Probe/input receipts in the same root. Inner owner journal:
  /root/autodl-tmp/data/channel_nsch2d/
  20261102-5090-integral-random-walls-online-width192-b8-10k-01-process.json.
  Run: /root/autodl-tmp/tf-logs/channel_nsch2d/
  20261102-5090-integral-random-walls-online-width192-b8-10k-01.
- Finite owner runs ONE cold10000 then input/replay/hard, phase, shape/gallery,
  240fresh-time diagnostics and the SAME3reserved FEM comparisons/gallery.
  Resource and capacity config gates reject unrequested changes; future audit
  checks initial-bank equality and time-prefix equality, independently replays
  the different B8condition draws and checks frozen source bytes. No retries.
- Corrected only the waiting supervisor's inherited empty CUDA_VISIBLE_DEVICES
  before any training launch; stopped/restarted that waiting process, preserved
  its journal as training-pipeline-prelaunch-env-correction.json. Both resource/
  input computations continued unchanged; no GPU experiment was restarted.

Task-end: changed source/config is the explicit capacity preset/audit, finite
probe/owner extension and new capacity handoff helper; no primal PDE/operator/
network implementation changes. Added2focused guard tests, both pass. New AST/
config/schema/actual resource and input checks pass; git diff --check passes.
README documents optional capacity configuration; defaults remain unchanged.
Memory files changed: PROGRESS.md, DECISIONS.md, this task and its compact JSON
receipt. Task remains active with the exact NEW journals above. No Git/history,
cleanup or FEM activity. This combined experiment cannot isolate width/batch/
budget benefits; compare actual5k and10kmonitor results when available.

## Capacity status follow-up 2026-11-02T08:03:48.790427+00:00

Same authorized cold192/B8/10k is active, canonical child806694. At least
50updates completed (actual refresh/loss logs); GPU100% at observation.
Only step0untrained L2 exists; first trained monitor at500is still pending.
Do not report initialized L2 as trained accuracy or substitute the old5k endpoint
for this run. Latest sampled loss/log and exact time are in the compact receipt.
99frozen Python/config bytes remain identical.

No optimization/source/config/default/queue/FEM/Git/cleanup change. Status-only
verification: child identity, GPU, journals/logs, source-byte checks, JSON parse
and git diff --check. Memory updated: PROGRESS, task and compact receipt; no new
decision. Next read the SAME owner's 500-step L2 and compare with old128/B2 at
matching updates, retaining nominal/unseen distinction.

## Capacity runtime clarification 2026-11-02T08:10:32.282318+00:00

User asks whether500updates really require half an hour. Actual training has
completed150updates; step100sampled totalMSE1.14925 versusinitial23.4403. No
trained L2 yet (first500), so this is not an accuracy/generalization comparison.
Loss100flush to before-step151refresh mtime yields4.57719s/update over50updates,
consistent with4.52441sprobe; includes between-update overhead, excludes compile.
500updates ~38.14min; remaining350 ~26.70min at this observation;10000 ~12.71h
plus initialcompile/monitor/audits. Estimates are conditional on continuing rate.

Old128/B2warm.42269s versus192/B8 ~4.58s is ~10.83times per update. This combines
4times geometry-time samples, width128->192 and timeblock8->2; no isolated
architecture/kernel/microbatch profile is available. Current update computes
8x40x15633=5002560geometry-space-time combinations before boundary work. Do not
attribute the whole slowdown to SOAP or geometry construction; producer waits
are only.026s per50update refresh. GPU100%/~444W at observation. No scheme,
source, queue, budget, FEM, cleanup or Git change.

Task-end: status-only journals/log timestamps and GPU checks, memory JSON parse
and git diff --check. Changed PROGRESS, task, compact receipt only; no new
decision or unchanged test rerun. Exact resume: SAME inner owner/pipeline and
its first500step monitor; keep the current finite cold10000 running.

## RTX PRO6000 hardware timing started 2026-11-02T08:23:48.601236+00:00

User explicitly replaced hardware and requested a new timing measurement.
Verified NVIDIA RTX PRO6000Blackwell Server Edition GPU-65a147dd-84d3-89dd-c771-
f0d2eac3664f,97887MiB; no old training/supervisor or GPU process survives. Old
external journals now explicitly mark interrupted_by_external_hardware_change,
retain their prior status and all artifacts. Only old unfitted initial EQX model/
SOAP exists; no trained resume can be claimed.

New root: /root/autodl-tmp/data/channel_nsch2d/
random-walls-pro6000-timing-20261102-01.
Same-scheme11update probe(time2) active, unified exec60938, PID2868. New
--hardware-trial selector independently allows only timeblock2or8, same strict
capacity scheme/control. Full16bank retained and actual finite gradient/SOAP
updates synchronized; unfitted initial states saved separately, fitted states
discarded. Eleven updates cover two refresh cycles after the cold update.

Added explicit train_operator_random_walls_width192_batch8_time8.py.
Finite owner optional --hardware-trial checks actual physical GPU UUID plus
peak<=min(90%physicalmemory,95%JAXpoollimit) and matching probe configuration;
old32GB/micro2 control route/gates remain. Same input screen402actual draws
passes without rerunning its unchanged constructions. Three focused capacity/
hardware guard tests pass(0.201s); syntax passes. No shared model/operator/loss
implementation changes, FEM, Git/history or cleanup.

### Same-scheme PRO6000 result

Timeblock2production11updates complete. Cold first354.591s includes compile;
warm10mean4.704753s/update, first7alignedmean4.706008s versus5090probe4.524410s
(~4%slower). Peak18.8824GiB. Actual fresh initial model AND complete SOAP bytes
match the old5090unfitted probe exactly. Pure500updates39.21min/10k13.07h,
excluding compile/eval/rolling. Hardware replacement alone did not speed up
this particular FP32/highest-precision workload. No architecture/accuracy claim.

Timeblock8probe now active, unified exec85366, same root/time8; new configuration
uses the same effective40sampled times and B8, but5rather than20gradient blocks.
Read result.json when its exact process exits, compare actual memory/mean times
and choose the qualified faster variant for replacement training.

### Aggressive-memory steering 2026-11-02T08:47:33.612256+00:00

User explicitly asks to fill available memory aggressively. Timeblock8passed:
2.390149s/update, actualpeak40.1859GiB (not ~72GiBallocator reservation), pure10k
6.64h. Keep these completed results. Directly probe20times/block (two gradients
per update), explicit95%allocator reservation, effectiveB8/Nt40 unchanged.
Memory helper docs/tasks/channel_gpu_memory_probe.py lowers/compiles the actual
step first, records temporary/output/alias and existing-resident bytes and skips
an oversized candidate explicitly before running it; successful candidates get
11synchronized full-gradient/SOAP updates and nominal finite queries. Fitted
probe state never initializes training. Source snapshots/direct paths, no hashes.

Active time20helperPID36436/unifiedexec2281; root/time20.log and result.json.
Actual target~85--90GiB; gate<=95%physical and97%allocator for this explicit
aggressive mode. Existing hardware/32GB gates preserved. New time20preset
requires finite owner --hardware-trial --memory-fraction .95 matching its probe.
Execution comparison allows40-time divisors but preserves original strict
capacity gate and all other numerical options. Three focused tests pass0.291s;
changed syntax passes. PDE/network/loss/optimizer/time-sampler unchanged.
Next inspect actual compiler memory and update timing. Consider40if viable,
then launch ONE qualified faster replacement cold10000 with existing references
and input screen, not a recovery of old150updates. No FEM/Git/cleanup/default.

### Block20memory preflight and explicit allocator adjustment

First block20compile completes380.04s. Estimated resident+output+temporary
peak94319245389bytes=87.841GiB, allowed93969114955bytes=87.516GiB under95%
allocator reservation. Explicit skipped_memory_preflight, zero optimizer
updates; preserve root/time20/{result,memory-analysis}.json. Only.326GiBover
that chosen gate; actual physicalcapacity95.593GiB.

User's aggressive-memory authorization supports explicit97%allocator retry,
retaining95%physical/97%allocator peak headroom. Same block20/B8/Nt40/width192
and unchanged numerical scheme; currentPID51302/unifiedexec50061, files
root/time20-fraction97.log and root/time20-fraction97/result.json. No fitted
state or numerical fallback; discard this resource state too. Owner now accepts
explicit fraction<=.97and requires exact probe match. README uses the matching
.97flag. Next inspect completed memory/throughput, then ONE fresh10k run if
qualified; no further increase is required merely to fill reserved bytes.

## PRO6000qualification complete / replacement training handoff 2026-11-02T09:06:39.505067+00:00

- Same effective192width/B8/Nt40/FP32/highest/ordinary/SOAP/integral/JVP and fixed
  nine losses. Same prior/online16pool,2refresh/50updates, input screen402 and4FEM
  qualifications retained; zero newFEM or evolved training labels.
- Actual11update resource/synchronized full-gradient/SOAP and nominal-query
  probe at20-time blocks/97%allocator passes. Actual activepeak88.05549GiB,
  ~92.115%physical, allocator limit~92.12GiB. Warm1.886506s, compiler371.978s
  separately. All finite, fitted state discarded. Pure500=15.7209min/10k5.2403h.
- SameGPUblock2=4.704753s/18.8824GiB, block8=2.390149s/40.1859GiB.5090block2
  =4.524410s/18.9841GiB. Selected2.3983times faster than5090original; gain comes
  from using larger time blocks, hardware alone was slightly slower. These are
  resource timings, not accuracy/stability claims; new trained L2 not yet available.
- Exact comparison table: /root/autodl-tmp/data/channel_nsch2d/
  random-walls-pro6000-timing-20261102-01/timing-comparison.json. Successful
  aggressive probe root/time20-fraction97/result.json, actual config/state/source
  and compiler memory saved. First95%allocator skip preserved under root/time20.
  Block40notrun because20already meets target; no unsupported impossibility claim.
- New screen: channel-random-walls-pro6000-192-b8-10k. Current owner67622,
  canonical training child67633. Owner statustraining; CUDA backend and explicit
  XLA_PYTHON_CLIENT_MEM_FRACTION=.97verified; all original old PIDs absent.
- New run: /root/autodl-tmp/tf-logs/channel_nsch2d/
  20261102-pro6000-integral-random-walls-online-width192-b8-time20-10k-01.
  NEW owner journal: /root/autodl-tmp/data/channel_nsch2d/
  20261102-pro6000-integral-random-walls-online-width192-b8-time20-10k-01-process.json.
  This owner directly runs training then existing replay/input/hard, phase,
  shape/gallery,240fresh-time and SAME3reserved evaluations. No outer timing
  pipeline is needed; old5090pipeline remains interrupted, not live.
- ACTUAL formal initial model/fullSOAP byte-match the successful UNFITTED
  resource state independently.101live Python/config source files byte-match;
  no fitted probe state imported. Exact values and owner IDs are in compactreceipt.

Task-end changes: hardware execution config/gates and presettime8/time20, finite
resource timing extension, aggressive compiler-memory probe, owner explicit GPU/
allocator validation and expanded focused guard tests; no shared model/PDE/loss/
operator/optimizer implementation change.3focused guards pass; changed AST and
actual config/resource/input/cold-state checks pass. README/DEVELOPMENT explain
new explicit flags/provenance. Memory updated PROGRESS, DECISIONS, task and
compact JSON receipt. git diff --check passes; no Git/history or cleanup.
Task remains active until this authorized replacement10k/audits finish. Exact
resume: NEW PRO6000owner above, check latest loss/500-step L2 and eventual final
held-out evaluation; do not report old5090or untrained step0as new trained error.

### Actual production first update verified 2026-11-02T09:11:46.772831+00:00

Canonical train.py has completed its first REAL full update under block20/
97%allocator, finite sampled totalMSE23.4403076 logged atstep1, after its own
compile. Owner remains training, child67633. Resource headroom now exercised
in the actual production step, not only the finite probe. No trained L2 yet;
first monitor500. Keep estimated1.8865s/update/5.24h pure10k distinct from
actual formal aggregate timing, which is not yet logged. No source/config or
experiment changes; compact receipt/PROGRESS/task updated, diff check passes.

## PRO6000trained L2 status 2026-11-02T09:34:21.711955+00:00

Same authorized run active, at least700updates complete. Latest complete
monitor step500on identical development FEM queries. Old128/B2versus new192/B8:

| Relative L2 | Old500 | New500 |
|---|---:|---:|
| phi | 22.6979% | 15.2738% |
| terminalphi | 26.8567% | 17.7469% |
| velocity | 8.1650% | 6.9654% |
| pressure | 19.7634% | 12.7339% |
| minority fraction field | 59.9775% | 40.3600% |
| chemical | 199.4776% | 99.8577% |

Early same-update improvement, not a stable/population advantage or isolated
width/batch effect. Minorityfield40.36%/chemical99.86%remain high; shape
accuracy is not certified by globalphi. No trained snapshot exists until5000,
so no new physical-field/contour assessment is claimed.3reserved evaluations
remain scheduled after10k. Latest sampled MSE0.079258atstep700.

Task-end: status-only journal/actual child/log/FEM-target comparison and101
frozen-source byte checks; memory JSON and git diff --check pass. No source,
configuration, default, sampling, run-budget or extra experiment change; no
new decision/FEM/Git/cleanup/test rerun. Updated PROGRESS, task and compact
receipt. Exact resume: SAME PRO6000owner, next500-step L2 monitors; let the
finite10k/audit owner continue.

Timing clarification: the status read crossed the step700loss/refresh flush, so
that equal-step timestamp span was correctly unavailable. Use independently
verified stable metric500->loss700timestamps instead: 1.903038s/update
including intervening input/host work, excluding initial compile. Remaining~4.92h
training plus eval/audits. Latest actual progress >=700, latest L2 remains500.


## PRO6000trained L2 status 2026-11-02T10:44:40.598766+00:00

Same owner67622/child67633active; at least2950/10000updates complete.
Latest complete monitor2500, same development localFEM target/queries.

| Relative L2 | Old128/B2 step2500 | New192/B8 step2500 |
|---|---:|---:|
| phi | 9.8236% | 5.4105% |
| terminal phi | 14.5121% | 4.6733% |
| velocity | 5.3139% | 4.3290% |
| pressure | 7.4806% | 5.0639% |
| minority fraction field | 25.9582% | 14.2970% |
| chemical | 64.4394% | 36.2904% |

All six improve from monitor2000and beat the old5000nominal endpoint.
This is one development geometry, not a population/generalization or isolated
width/batch/hardware ablation. Minority refers to the (1-phi)/2field L2, not
integrated volume; chemical/minority errors remain material. No new saved
checkpoint/field-shape assessment before5000;3held-out evaluations pending.

Recent full monitor elapsed difference(2000->2500) gives1.907485s/update,
~3.74h remaining training plus finite eval/audits. Saved elapsed records
avoid file-flush timestamp races and exclude initial compile. GPU100%active;
95123/97887MiB reserved is not the measured88.0555GiBactive resource peak.

Task-end: status-only process/log/matched-reference and101frozen-source byte
checks pass; memory JSON/diff whitespace checked. Updated PROGRESS, task and
compact receipt. No source/config/default/budget/extra experiment/FEM/Git/
cleanup/test rerun or new decision. Exact resume: SAME finite PRO6000owner
and next500-step monitor; keep live source frozen, let10k/audits finish.


## Physical-field request / task-end 2026-11-02T10:50:01.928962+00:00

Current run still has no trained checkpoint: first scheduled ckpt_005000.eqx.
Monitors retain scalar L2 only; do not substitute initial weights or claim the
old final arrays are current predictions. Explicitly show OLD128/B2/5000on
the same nominal geometry, original saved13snapshots/96x40queries. Its actual
phi6.5285%,terminal7.6275%,u4.4152%,p6.5547%replay saved endpoint exactly.

Generated old gallery: /root/autodl-tmp/data/channel_nsch2d/
random-walls-field-preview-20261102-01/old-width128-step5000.
Files phi-fields/speed-fields/pressure-fields/droplet-contours.{png,pdf},
gallery-validation.json and plot-source.py. Inspected all four PNGs: old
droplet too flat/forward at3--4.5, later recovers height but shape/position
bias persists. Spatial display interpolation only, no temporal interpolation
or contour/centroid correction. Pressure begins0.5because initial FEM pressure
is unsolved. Images remain external, no tracked raster/data/weights.

One finiteCPU-only helper docs/tasks/channel_checkpoint_gallery_waiter.py
started in detached screen channel-fields-pro6000-5000, PID90219. Waits at most
7200sfor actual5000checkpoint AND a later flushed update/final summary, thus
not reading the non-atomic writer mid-save. Verifies owner/source bytes, then
uses existing partial_evaluate / evaluate / droplet_metrics, CPU affinity last4
cores and JAX_PLATFORMS=cpu. No new train/FEM/optimizer recovery or GPU work.
Status/output: same external preview root/current-width192-step5000/status.json.
Current status waiting_for_complete_checkpoint. Actual partial5000CPU metric
replay must pass existing2e-6tolerance before plotting; it has not run yet.

Current latest monitor details are in the compactreceipt; phi/velocity improve
from2500to3000but terminalphi/pressure rebound. No claim of improved new
droplet shape before a new checkpoint is available.

Task-end changes: one finite visualization helper and DECISIONS/PROGRESS/task/
compactreceipt. Helper AST/CLI and actual waiting branch pass; original saved
field metrics/reference/grid checks pass;101live trainer source-byte checks
unchanged. git diff --check passes. No trainer/source/config/default/budget,
extra experiment/FEM/Git/cleanup action. Task remains active. Exact resume:
read SAME training owner plus finite gallery status above; when completed,
show current5000phi/speed/pressure and droplet contours with correct provenance.


## Evaluation-set scope clarification / task-end 2026-11-02T10:59:45.039331+00:00

User questions the simplicity of the illustrated nominal reference and actual
L2 test-set size. Verified actual nominal+3reserved localFEM payloads: each
fields(13,3840,5), points(3840,2),13saved times0:.5:6. Continuous training
monitors pool unweighted phi L2 over ONE development geometry/49920space-time
queries. Terminalphi uses3840queries; velocity/p/mu use12positive times and
46080queries. Query count is not independent geometry count. This nominal
geometry is excluded from PDE training inputs but repeatedly monitored, hence
development validation rather than a broad independent test.

Only3predeclared random geometries have qualified references. Current trial
evaluates them after10k, reporting per-case then mean/p90/worst; the older
128/B2run already evaluated them, so they are a reused validation cohort. No
32/64/100geometry test set exists and no population accuracy is established.
Training remains online with16candidate slots/B8/2refresh per50, not fixed16
geometries or reference-label supervision.

Status-only clarification updates PROGRESS, task and compactreceipt; no new
decision/source/config/runtime/Git/FEM or experiment change. All4actual payload
shape/time checks pass; git diff --check passes. Task remains active; exact
resume unchanged: finite PRO6000training owner and scheduled5000CPUgallery,
then3-case evaluation. Any expanded reference cohort is future work and has
not been launched.


## Other-geometry snapshot evaluation / task-end 2026-11-02T11:13:09.611062+00:00

User requests evaluation on other geometries now. Current model still lacks a
trained snapshot until5000. Recomputed OLD128/B2/5000checkpoint on all3frozen
reserved cases using the new CPU-only docs/tasks/channel_snapshot_cohort.py.
No new FEM or training. Each geometry uses13times0:.5:6/3840own-domain queries;
flow/chemical excludes0. Same canonical decoder, reference gates and metric
functions as full cohort_evaluate. Original final audit remains authoritative.

| Case | phi | terminalphi | velocity | pressure | minority field |
|---|---:|---:|---:|---:|---:|
| 0 | 6.0378% | 6.9192% | 4.3730% | 6.0607% | 15.8672% |
| 1 | 6.0752% | 7.0971% | 5.1113% | 10.9148% | 16.1437% |
| 2 | 6.1134% | 6.1057% | 4.4984% | 7.4343% | 16.0355% |

Meanphi6.07545%,u4.66090%,p8.13658%; worstp10.91475%. Globalphi errors are
similar across these three, but terminal area-weighted minority errors are
21.0387/20.8556/18.6412%, and negative-phase IoU77.4778/80.6706/84.1108%.
Raw contour distance/R0 is10.9339/11.0187/9.2409%. Whole-domain phi therefore
still understates droplet-shape mismatch. These are old-model results, not
new192/B8accuracy or a broad random-prior guarantee.

Old CPU evaluation completed2026-11-02T11:09:10.712683+00:00,16.6s total
including3geometry reconstruction/model execution/plots. Every saved metric
matches original audited GPU cohort within4.945e-8. Hard IC/wall/5section-flux
checks all below2e-5(actualmaximum~1.46e-6). Original97source checks pass.
Artifacts: /root/autodl-tmp/data/channel_nsch2d/
random-walls-reserved-preview-20261102-01/old-width128-step5000.
Actual evaluation.json/status.json/evaluation.log and3case folders, each with
prediction.npz/config/phi-fields/droplet-shape. Inspected reserved-phi.png and
reserved-contours.png/PDF: wall/shape variation visible, common middle/late
flattening and forward-position bias. Gallery source stored beside figures;
spatial display interpolation only, no temporal interpolation or alignment.

Current finite screen channel-reserved-pro6000-5000, PID92372, same helper,
waits at most7200sfor actual ckpt_005000.eqx AND successful nominal partial
metric replay from existing waiter90219. Uses4CPUcores, actual affinity IDs200--203(last8:-4), leaving204--207to the
nominal gallery.
JAX_PLATFORMS=cpu,CUDA_VISIBLE_DEVICES empty; GPU process list remains only67633.
Current state waiting_for_snapshot_replay. Status/output: same preview root/
current-width192-step5000/status.json. This is snapshot validation; no complete
online-input audit or exact recovery claim. Final10kowner/cohort audit unchanged.

Task-end changes: one CPU snapshot-cohort helper plus DECISIONS/PROGRESS/task/
compactreceipt. Helper AST/CLI and actual complete3case execution pass; all
reference/metric/hard checks above pass, current101frozen trainer sources match,
git diff --check passes. No canonical source/config/default/budget/Git/cleanup
change. Task remains active. Exact resume: inspect both finite5000postprocessor
status.json files; after completion report NEW per-case metrics/contours and
compare with old table above. Do not substitute old results for new evaluation.


## Live convergence plot / task-end 2026-11-02T11:43:05.483781+00:00

User requests convergence curves. Read complete actual logs, no live-source
change. Saved old128/B2through5000and current192/B8through4500in external
/root/autodl-tmp/data/channel_nsch2d/random-walls-convergence-20261102-02.
relative-l2-comparison.{png,pdf} shows6separate metrics; sampled-loss-comparison
.{png,pdf} shows raw total loss and current phase/chemical/momentum/BCgroups.
curve-data.json preserves exact records/configs and plot-source.py the plotting
code; convergence-validation.json captures plotting scope/order/last-step data.
No smoothing, fitted curves, best-so-far replacement, unseen-test averaging or
interpolation of missing5000new results. Inspected both PNGs.

Same4500comparison old->new: phi7.9448->4.4115%,terminal10.3780->5.9088%,
u4.5900->3.1867%,p6.2855->4.5433%,minority20.9936->11.6570%,
mu48.5043->24.2592%. Current improves on old at matched steps, but from4000
to4500phase/terminal/minority/pressure rebound while velocity/chemical improve.
Sampled normalized loss still trends down with stochastic fluctuations; this
is not proof of monotone FEM-error descent or stable convergence. Metrics
remain ONEdevelopment geometry,13times/3840queries;3reserved errors have no
current convergence series. Both5000CPUpostprocessors still waiting.

Task-end: documentation/artifact-only. Checked full500step order, finite positive
6metric arrays, old completed endpoint agreement, same physical/reference config
and normalized9term sums. Captured complete lines explicitly; incomplete trailing
bytes would be recorded, no arbitrary exception fallback.101frozen trainer
source files still match; git diff --check passes. Updated PROGRESS/task/compact
receipt; no new decision, numerical/source/config/default/budget/Git/FEM/cleanup.
Task remains active. Exact resume: SAME training owner and two5000CPUstatus
files; report current reserved results once they actually exist.


## Current5000snapshot evaluations complete / task-end 2026-11-02T12:04:02.833173+00:00

User asks to continue completing the authorized training/evaluation. Training
healthy, first actual ckpt_005000.eqx saved. Nominal waiting PID90219absent,
no partial fields or proof, stale waiting status/dead screen. Exit cause is
unconfirmed; do not blame PDE or claim a diagnosed host timeout. Preserved
original journal as interrupted-waiter-status.json in the existing nominal
preview folder. Direct CPU recovery owner95416completed4existing stages with
exit0from11:58:57to11:59:10UTC; recovery-source.py/recovery.log/stages record
exact commands. No checkpoint/optimizer/trainer/source/config change.

Actual nominal5000CPU replay matches logged GPU phi/terminal/u/p within6.429e-8.
Proof is run/partial_005000.json and physical arrays run/partial_005000.npz.
Current nominal gallery under externaldata/channel_nsch2d/
random-walls-field-preview-20261102-01/current-width192-step5000:
phi/speed/pressure-fields and droplet-shape. Status now
partial_checkpoint_gallery_completed. NominalterminalIoU84.4509%,minority
area-weighted terminalL214.2919%,contour distance/R0=7.1060%.

Existing reserved evaluator92372consumed the proof and completed all3cases;
status snapshot_cohort_completed, no duplicate evaluator. Current results:

| Case | phi | terminalphi | velocity | pressure | minority field | terminalIoU |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 2.8202% | 3.1667% | 2.9865% | 4.7911% | 7.4114% | 85.2928% |
| 1 | 2.6256% | 2.4217% | 3.2210% | 10.5763% | 6.9771% | 93.3518% |
| 2 | 3.3595% | 4.3265% | 2.9904% | 5.8643% | 8.8122% | 87.8233% |

Meanphi2.9351%,terminal3.3050%,u3.0660%,p7.0772%,minority7.7335%,mu26.3024%.
Old meanphi6.0754%,u4.6609%,p8.1366%,minority16.0155%,mu47.4554%.
Terminal area-weighted minority errors9.5434/7.1000/13.1929%, versus old
21.0387/20.8556/18.6412%. Contour distance/R0=4.9677/3.4946/6.5889%.
Shape improves in each case, so the globalphi improvement is not only majority
phase dilution. Pressurecase1still10.5763%(old10.9148%); chemical remains high.
Only3reused reserved cases, not broader-prior evidence or complete input audit.

Artifacts: externaldata/channel_nsch2d/
random-walls-reserved-preview-20261102-01/current-width192-step5000:
evaluation.json/status.json/evaluation.log,3case saved predictions/config/
phi-fields/droplet-shape, independent-validation.json. New reserved-phi and
reserved-contours.{png,pdf}/gallery-source.py explicitly label CURRENT192/B8/5k.
Inspected both PNGs: much better middle/late contour agreement, some position/
flattening bias remains. Physical positions uncorrected, display interpolation
only. Updated both-run5000curves and exact data/source under
random-walls-convergence-20261102-03; inspected relative-l2-comparison.png.
These curves remain one development geometry;3case values are snapshots.

Independent saved-array checks: full actual qualified FEM target/query/time/
parameter equality and old/new evaluation-query parity; all6raw L2s recomputed;
physical area-weighted terminalIoU/minority independently recomputed to1e-12.
All3IC/wall/flux max gaps<=1.46e-6;101frozen trainer sources unchanged.
Task-end: DECISIONS/PROGRESS/task/compactreceipt and external recovery/plot/
validation artifacts. git diff --check passes. No numerical/source/config/
default/budget/new training/FEM/Git/cleanup or expensive test rerun.
Task remains active until the actual10k/final audits complete. Exact resume:
SAME67622owner/67633trainer and original process.json, next500-step monitors,
then saved final/online-input/capacity/240fresh-time/3reserved evaluations.
Do not restart a run or reuse either completed5000waiter as a live process.


## Continue unchanged to10000 / task-end 2026-11-02T12:06:22.863327+00:00

User explicitly says let the still-improving run finish. Same owner67622/
canonical trainer67633healthy; requested_steps10000unchanged, GPU100%active.
At least5500updates complete; latest monitor5500nominalphi2.6168%,
terminal3.3473%,u2.9312%,p5.5703%,minority6.9146%,mu21.4307%.
These are one-development-geometry metrics;5000three-case results remain a
separate completed snapshot. Recent1.902513s/update,~2.38h training left plus
final audits/evaluation. No assertion that every field decreases monotonically.

Task-end: status-only owner/process/log/GPU check and PROGRESS/task/compact
receipt update; git diff --check passes. No new decision, numerical/source/
config/default/budget/run/FEM/Git/cleanup or test rerun. Task stays active.
Exact resume: SAME finite owner process.json, subsequent500-step monitors, then
its prescribed final input/capacity/240fresh-time/3reserved/shape audits.


## Training status / task-end 2026-11-02T12:26:32.393512+00:00

Same67622owner/67633canonical trainer healthy, GPU100%, at least6150updates.
Latest complete development monitor6000:phi3.5037%,terminal5.1262%,
u2.7977%,p3.2045%,minority9.2583%,mu19.4399%.
Versus5500phase/terminal/minority rebound; u/p/mu improve. Do not describe
all errors as continuously decreasing or infer overfitting from one interval.
Three reserved results are still the completed5000snapshot, not updated6000
evaluations. Recent1.902536s/update;~2.03h training left plus final audits.

Task-end: status-only owner/log/process/GPU verification; updated PROGRESS/task/
compactreceipt; git diff --check passes. No new decision, source/config/default/
budget/run/FEM/Git/cleanup or test rerun. Task remains active. Exact resume:
SAME process.json, next500step monitors; original10000training and final
input/capacity/240fresh-time/3reserved/shape audits continue unchanged.


## Updated6000convergence figures / task-end 2026-11-02T12:28:05.352518+00:00

User requests latest convergence plots. Actual completed monitor records
old128/B2through5000and current192/B8through6000, on identical development
localFEM target/queries. Saved under /root/autodl-tmp/data/channel_nsch2d/
random-walls-convergence-20261102-04: relative-l2-comparison and sampled-loss-
comparison.{png,pdf},curve-data.json,convergence-validation.json,plot-source.py.
Both PNGs inspected; raw fluctuations retained, no extrapolation of old5000
curve or smoothing. Current3reserved results remain a separate5000snapshot.
The6000phase/terminal rebound and continued u/p/mu improvement remain visible.

Task-end: artifact/documentation-only. Plot checks full500step ordering,
finite positive6L2series, old endpoint parity, identical physics/reference
config, normalized9term-loss sums. Owner67622/trainer67633still healthy,
at least6200updates complete. Updated PROGRESS/task/compactreceipt;
git diff --check passes. No new decision/source/config/default/budget/run/
FEM/Git/cleanup or test rerun. Exact resume unchanged: SAME original10000run
and final audits; do not treat plotted6000monitor as a saved model snapshot.


## Training status / task-end 2026-11-02T12:48:37.702507+00:00

Same67622owner/67633trainer healthy; GPU100%, at least6850/10000updates.
Latest complete monitor6500:phi2.6287%,terminal3.1493%,u2.4929%,
p7.8936%,minority6.9462%,mu19.6945%.
Versus6000phi/terminal/minority/u improve, pressure rises3.2045->7.8936%,
chemical slightly19.4399->19.6945%. Do not hide that pressure rebound or
claim all errors decline. These are one-development-geometry monitors;
3reserved results remain the completed5000snapshot. Recent1.898668s/update;
~1.66h training left plus final audits. No change to user's original10kplan.

Task-end: status-only owner/log/GPU checks and PROGRESS/task/compactreceipt
update; git diff --check passes. No new decision/source/config/default/budget/
run/FEM/Git/cleanup or test rerun. Task remains active. Exact resume: SAME
process.json, subsequent500step monitors and planned final input/capacity/
240fresh-time/3reserved/shape audits.


## Current available physical-field display / task-end 2026-11-02T13:06:14.529285+00:00

User requests current physical predictions. Canonical training now at least
7400, latest L2 monitor7000; ONLYtrained saved model still5000.
Explicitly label all shown fields current192/B8/5000, not latest scalar-monitor
weights. Reuse verified3reserved phi overview; create case1speed/pressure
triptychs through existing evaluate.py, CPU-only, no model/reference solve.
Case1selected explicitly because largest5000pressure error in the cohort.

New PNG/PDFand gallery-validation.json under existing
externaldata/channel_nsch2d/random-walls-reserved-preview-20261102-01/
current-width192-step5000/case_001/flow-fields. Flowtimes.5/1.5/3/4.5/6;
rowsFEM/PINO/absolute error. Phi overview pairs FEM/PINO per geometry at
0/1.5/3/4.5/6. Both new flow PNGs inspected: constriction acceleration captured;
pressure larger bulk bias near1.5--3plus droplet-region discrepancies.
Case1whole-trajectory velocity3.2210%,p10.5763%,phi2.6256%; these are5000
metrics, not current development7000metrics. Raw saved arrays exactly match
qualified FEM and prior cohort values; recomputed flow L2to1e-12. Spatial
display interpolation only, no temporal interpolation or field correction.

Task-end: artifact/documentation-only; PROGRESS/task/compactreceipt updated,
git diff --check passes. No new decision/source/config/default/budget/run/
FEM/Git/cleanup or test rerun. Same10000trainer continues, next model snapshot
at10000; task active. Exact resume: SAME process.json and final evaluation
pipeline; do not claim a7000/7300field snapshot exists.


## Interface-overlay display / task-end 2026-11-02T13:08:23.527880+00:00

User requests interface overlap figures. Displayed the already validated actual
CURRENT192/B8/5000reserved-contours.png in the existing reserved-preview root.
Three rows=cases0/1/2; five columns=0/1.5/3/4.5/6. Black solid localFEM and
orange dashed PINO phi=0curves, equal physical zoom, actual predicted positions.
Terminal negative-phase AREA IoU85.2928/93.3518/87.8233%; do not describe
that as a line-length overlap metric. Latest saved model still5000, current
training at least7450. Source/PNG inspected, no new evaluation or solve.

Task-end: presentation/status-only; updated PROGRESS/task/compactreceipt and
git diff --check passes. No decision/source/config/default/budget/run/FEM/
Git/cleanup/test changes. Active10krun and final pipeline unchanged. Exact
resume remains same training owner, next monitors/final saved10000evaluation.


## Near-final training status / task-end 2026-11-02T14:18:37.573363+00:00

Same67622owner/67633trainer healthy,GPU100%, at least9700/10000complete.
Latest monitor9500:phi2.7236%,terminal4.0336%,u1.8135%,
p3.9135%,minority7.1969%,mu14.3912%.
Best scalar nominalphi2.0476%at8500; that is a monitor, not saved
best weights. Phase has rebounded recently, velocity still improves. All six
latest metrics remain better than this same model's5000monitor, but no stable
convergence/population claim follows. Final summary/saved10000not yet present.
Recent1.904151s/update,~9.5min TRAINING remains plus planned final audits.
Current3reserved results still5000snapshot; do not relabel as9500or final.

Task-end: status-only process/journal/log/GPU verification, PROGRESS/task/
compactreceipt updates; git diff --check passes. No decision/source/config/
default/budget/run/FEM/Git/cleanup/test change. Task active. Exact resume:
SAME owner process.json, imminent10000save and its final input/capacity/
240fresh-time/3reserved/shape audits. Inspect actual stage exit codes; if an
audit fails, diagnose it before claiming full completion.


## Actual10000physical fields available / task-end 2026-11-02T14:34:04.529750+00:00

User requests current fields at9900. Waited for real final snapshot rather than
redisplaying5000. Training exits0at14:28:24.660652UTC; model_final,ckpt_010000,
optimizer_final,prediction,summary and final pool/sampling artifacts saved.
Actual nominal L2phi2.6781%,terminal4.0550%,u1.6700%,p5.8931%,
minority7.0768%,mu14.1030%. Training done, audits separate.

Finite CPU3case preview owner107853completed14:28:50UTC, exact same declared
manifest/qualified localFEM. Actual10000results:

| Case | phi | terminalphi | velocity | pressure | minority field | terminalIoU |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 1.9429% | 2.7804% | 1.8752% | 3.8795% | 5.1059% | 86.1437% |
| 1 | 1.6519% | 1.5196% | 2.1335% | 9.0316% | 4.3897% | 94.2458% |
| 2 | 1.9240% | 2.5123% | 2.0394% | 3.3242% | 5.0467% | 93.6443% |

Meanphi1.8396%,terminal2.2708%,u2.0160%,p5.4118%(worst9.0316%),minority4.8474%,
mu15.9168%. End negative-phase AREA IoU86.1437/94.2458/93.6443%. Three reused
reserved cases only. Snapshot eval does not certify full final input audit or
continuous-prior/generalized accuracy. Formal official cohort evaluation will
still run through the original owner after prior stages; not replaced.

Artifacts under externaldata/channel_nsch2d:
- random-walls-field-preview-20261102-01/current-width192-step10000: original
  nominal prediction-triptychs/zero-contours/validation via saved final gallery.
- random-walls-reserved-preview-20261102-01/current-width192-step10000:
  status/evaluation/log,3case NPZ/config/phi/shape outputs, independent-validation;
  reserved-phi and reserved-contours PNG/PDF/source explicitly10000; case_001/
  flow-fields speed/pressure PNG/PDF at.5/1.5/3/4.5/6. Same case as prior display.
Inspected all4new overview/contour/flow PNGs; early/middle contours closer,
late case0position bias remains, case1pressure still most difficult. Actual
positions/no temporal interpolation; spatial display interpolation only.

Independent checks: all6L2 metrics recomputed from raw predicted/reference
arrays to1e-12; actual FEM points/times/parameters/fields exactly equal; terminal
areaIoU independently recomputed. IC/wall/flux gates<2e-5,101current source bytes
match; ckpt_010000bytes EXACTmodel_final. Nominal saved-gallery metrics match
summary and FEM arrays. No expensive suite rerun for unchanged canonical code.

Original owner67622now diagnose_phase, activechild111783; training stage exit0,
remaining official validation/input/capacity/fresh-time/cohort steps pending.
Task-end updates DECISIONS/PROGRESS/task/compactreceipt plus external preview
artifacts. git diff --check passes. No canonical source/config/default/SOTA/
budget/new training/FEM/Git/cleanup change. Task remains active, not archived.
Exact resume: SAME original process.json, inspect stage exits/receipts; finish
any failed required checks before reporting full completion.
