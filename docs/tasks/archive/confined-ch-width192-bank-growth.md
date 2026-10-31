# Width192 cold50k with1024input geometries

Date: 2026-10-31. Status: CLOSED BY USER SWITCH. No Git history authorization.

## Goal and acceptance

User explicitly stops width256 and requests width192 with a larger precomputed
training geometry bank; then explicitly requests a cold50kbudget. Double512->1024
training inputs, preserving original16training monitors and the same8reserved
physical geometries (accepted draws16..23; new bank indices1024..1031). Use same
width192/fourheads(D48)/96spatial directions/FP32/ordinary/SOAPbeta2.9/cap512,
B16/Nt16, fixed10mode ambientIC, T3, graph, hardIC/mass, eliminatedmu, frozen
stable metric, LR and runtime seed policy. Initialize fresh, never import learned
weights/state. Keep30kcheckpoint/metrics/dense61for common-budget comparison with
completed512/19230k. Assess fixed50k separately; do not conflate extra updates
with bank expansion. No default promotion or best-heldout checkpoint selection.

## Context and paths

Completed baseline:
/root/autodl-tmp/tf-logs/confined_ch2d/20261031-confined-geometry-b16-t16-width192-fp32-30k-01
Native phase train16pooled3.9700%, validation8pooled8.9148%/terminal9.8017%;
dense61pooled8.3458%. Same single seed/reused8development cases, no fresh-test/
continuum claim; localFEMdeferred. Existing bank1024is already prepared and
passed with512new unlabeled operator-only inputs, no newPDE/FEMtrajectories.
Original520records reused by hardlink; bank size~12.93GB, maxNx1353/degree97.

Source /root/autodl-tmp/data/confined_ch2d/operator_geometry_smallscale10_512_v1.
Target /root/autodl-tmp/data/confined_ch2d/operator_geometry_smallscale10_1024_v1.
Same accepted-draw prefix and reserved geometries must be independently verified.
Actual training geometry IDs change when randint upper bound becomes1024;
verify fresh draws against global keys, not identical512draw IDs. Initial model
arrays and time draws can remain exactly identical to baseline. GPU reports
RTX3090/48GiB, idle1MiB, disk6.45GiBfree after previous stop. No cleanup allowed.

Previous user-stopped256task:
docs/tasks/archive/confined-ch-width256-50k.md; stop receipt
confined-ch-width256-50k-stop.json. Old40900logs/40000fullstate pair retained,
no50kresults. New task is an explicitly authorized scope switch.

## Current state and plan

Checked 2026-10-31T05:45:49.959152+00:00: runner339235/trainer346501 healthy,
latest logged17800/50000finite loss0.00011336351; complete17500evaluation
phase train16pooled8.3314%/terminal8.9592%,
reserved8pooled9.6123%/terminal10.4011%.
Same-step512trainpooled6.8384%, validationpooled
10.1642%/terminal11.0275%. Recent0.151213s/update,
~81.2min updates remaining plus final audits. Same8native development
geometries/one initialization, no stable/final accuracy ranking. Live receipt
confined-ch-width192-bank1024-50k-live.json.

Updated convergence snapshot16000contains33common assessments. Three requested
curves:192/N512solid,192/N1024dashed,256/N512dotted; blue=train16monitors,
orange=validation8. Pooled train/validation at16000:192/512=7.2788/10.6793%,
192/1024=8.4047/9.9460%,256/512=6.9395/10.6207%. Validationterminal respectively
11.5998/10.7231/11.5425%. Expanded bank has lower validation at this commonbudget;
256lowers training monitors with little validation gain. This does not establish
multi-seed stability or a final50kranking. The256trace is from the historical
user-stopped run; no restart or new256job. First-to-third also changes spatial
Fourier count with backbone width, not a width-only ablation.

Fourpanel detail, two-bank and threeway overlays generated onCPU. RawL2 has no
smoothing; PDEloss raw plus labeled11record displaymedian. Snapshot was latest
complete step at capture, not minimum-error selection. Helper --third verifies
originalbank/cohorts and only declared architecture/budget/run-name differences;
all33step sequences align. Bank-growth view maps accepted draws and exactsix
reference arrays on24physical monitors. Independent99evaluation split/field
aggregate replays, AST/JSON and PNG/PDF PASS; pair/threeway visually inspected.
Canonical receipt confined-ch-width192-bank1024-50k-convergence.json. Final media
external in run/convergence_live_03, convergence_comparison_03 and
convergence_threeway_03; previous previews retained. No numerical/runtime API,
active training, seed, default, compiler or optimizer changes.

Full1032GPU25step feasibility and independentCPUreplay PASS: FP32finite losses/
gradients/model/state, zero nonscalarFP64IR, actual newdraws/globalkeys and no
reserved overlap checked. Peak active15.6119GiB versus
35.5616GiBallocator limit; warm20mean159.040ms/update.
CPUphi/mu maxgaps[2.5331974029541016e-07, 9.51811671257019e-07] below declaredbounds.
Production initial119arrays exactly match fresh seed/probe and original192initial;
shared numerical sources unchanged. No probe/trained weights imported.

User asks why startup is slow. Measured preflight compile541.642s (876.449s
wholeprobe), production setup312.921s, firstupdate including compile approx549.740s
from wall timestamps. Original192/512firstupdate504.223s. Thus fulltrainstep
compiles twice in independent processes; localJAX0.10.1cache directory isNone,
so cross-process executable cache not initialized. Log106slow alarms across52
uniqueGEMM fusionmodules; parallel alarms must not be summed, and no profile
isolates codegen versus autotuning or SOAP contribution. Fullnetwork/PDEJVP/
parametergradient/SOAP graph is compiled; paddedNx/degree1351/96->1353/97 also
changes shapes. Compilation ended05:01:30UTC and real updates now advance.
Compile diagnostic receipt confined-ch-width192-bank1024-50k-compile.json;
o currentjob/cache/compiler/optimizer change or restart. Future cache/shared
entry optimization is a proposal requiring actual HLO/cache-hit verification.

Blocker: none. Next action: monitor only existing NEW1024journal and closed
metrics, compare same steps through30k, then automatic fixed30k/50k/native/
dense61/geometry/globaldraw/CPUaudits. No duplicate launches, default change,
seed tuning, reference supervision, GPU migration or artifact cleanup.

## Online geometry feasibility discussion — 2026-10-31

User asks whether larger architecture plus broader geometry support may lower
validation error, but explicitly defers changes until the current run finishes.
The common16000curves support investigating complementary capacity/coverage;
the combination itself is unmeasured and no lower error is guaranteed. Current
fixed50k, runtime, default and physical/optimization scheme remain unchanged.

Generating continuous-prior inputs requires meshing, physical centroid volumes,
visible neighbors/coefficient checks, known IC/encoder features and the current
frozen per-geometry stable residual metric. No evolved fields, PDE trajectories
or FEM solutions are needed. Mesh generation imports Gmsh/meshio through an
isolated interpreter; it does not integrate the physical equations. The current
prepare/load route is offline and has no implemented concurrent online queue.

Historical target-bank512new input builds:129.5646s total with6workers,
mean worker stages mesh0.5488s, assembly0.0605s, old Cholesky metric0.2466s,
compressed write0.6258s. That old metric is replaced at training load; its timing
must not be presented as the current stable-metric construction cost.

Lightweight NumPy/SciPy-only probe at05:52UTC: production
linear_stable_resolvent on unlabeled cases512/513/514, three repetitions each,
OPENBLAS/OMP/MKL/NUMEXPR limits1, mean0.51621s/build. All finite and constant-mode
errors<=2.08e-14. No JAX backend, mesher, PDE/FEM solve or optimizer step started.
Initial optional threadpoolctl import was unavailable; removed that diagnostic
dependency and reran successfully without installing anything.

Combining historical mesh/assembly with that current metric gives an estimated
1.1255s/geometry without compressed disk writes. Six-worker idealized throughput
5.33inputs/s, B16production~3.00s, compared with measured GPUwarm0.1590s/update
(100.60geometry uses/s). These are stage-based estimates, not a full online
pipeline benchmark; communication, padding/transfer, CPU contention and startup
are excluded. Host exposes96logical CPUs but cgroup cpu.max1200000/100000 is a
12CPU budget; unrestricted worker scaling cannot be assumed.

Deferred proposal: asynchronously generate into a bounded refreshed pool,
reuse each physical input across updates, then evict it; distinct inputs can
continue growing without storing an unbounded disk bank. Keep16training and8
development reference monitors separate from unlabeled new inputs. Fix padding
envelopes/dtypes before compilation and explicitly reject/report overflow;
observed maxNx1353/degree97 over1024inputs is not a global bound on the prior.
Online input support alone cannot guarantee better validation or complete
optimization. Before any implementation after this run, benchmark the full
producer/consumer route and verify meshing environment/process isolation,
input identity, quadrature/operator/metric parity and reproducible draws.
Receipt: confined-ch-width192-bank1024-50k-online-cost.json.

Discussion-session task-end: only this task, PROGRESS, DECISIONS and compact
online-cost receipt updated; no production code/config/process/default change.
Exact resume remains the existing1024journal and automatic30k/50k audits.
Keep the task ACTIVE; blocker none. JSON/numeric/active-task checks and
git diff --check PASS; no repeat of unchanged expensive test suites.

## Launch and verification

Existing1024bank independently verified:520original records are hardlinks with
5816exactly equal arrays; original512training prefix,16monitor roles and reserved
draws16..23preserved. New512records have no future fields. Dataset contains
1024training/8validation, maxNx1353/degree97; no newPDE/FEMsolve. CPUconfiguration
and abstract model/state/storage gate PASS: only declaredbank/provenance/budget
changes, exact baseline192fresharrays and time draws; model2021185trainable
parameters,8085508bytes, SOAPstate47865200bytes. Fifty paired states plus declared
reserves need4483725572bytes(4.176GiB); available~6.45GiB, no cleanup required.
Receipts confined-ch-width192-bank1024-50k-{bank,config}.json. Old192CLI CPUconfig
compatibility PASS; two existing unittest bank-reuse/isolation checks PASS.
The attempted pytest invocation found pytest uninstalled; tests ran successfully
through their existing unittest entry, no environment/dependency changes.

Dense61reference view created at
/root/autodl-tmp/data/confined_ch2d/operator_geometry_smallscale10_dense_validation_1024.
Eight NPZhardlinks map original512..519 to1024..1031 after exact accepted-draw,
points/volumes/holes, five-time phase/mu anchor and times checks. Original refined
validation gates retained; no new solver/data modification. Regeneration helper
confined_ch_dense_bank_view.py, source validation path and mapping in newview.

Research-only capacity_case/probe/runner now explicitly accepts
--case=width192_bank1024_50k. Old default192and256cases preserve their commands.
New case audits changed geometry sampling against its own globalkeys rather
than baseline512IDs, and checks exact baseline initial arrays. Runner uses
saved dataset and explicit mapped denseview, bank-growth CPUaudit, and retains
30kcheckpoint/dense61comparison plus final50k. Shared network/PDE/training/
optimizer/precision files are byte-identical to completed192baseline. Five
changed/new Python AST checks and git diff --check PASS.

Detached runner339235 launched2026-10-31T04:32:15UTC; GPUprobe339238 healthy,
initial stage=gpu_probe at launch. That stage and independentCPUreplay have now
passed; productiontraining is running as recorded above. All probe weights discarded.
Journal /root/autodl-tmp/tf-logs/confined_ch2d/20261031-confined-width192-bank1024-fp32-50k-01-process.json
Probe /root/autodl-tmp/tf-logs/confined_ch2d/20261031-confined-width192-bank1024-fp32-50k-01-probe
Planned cold run /root/autodl-tmp/tf-logs/confined_ch2d/20261031-confined-geometry-b16-t16-width192-bank1024-fp32-50k-01
Runner log /root/autodl-tmp/tf-logs/confined_ch2d/20261031-confined-width192-bank1024-fp32-50k-01-runner.log
Launch receipt confined-ch-width192-bank1024-50k-launch.json. Gate failures stop
the pipeline; do not launch a duplicate or bypass/relax gates.

Task-end: old256closed by user switch and archived; new1024/19250kpreset,
researchcase gates/runner and denseview helper implemented, README/DEVELOPMENT/
DECISIONS/PROGRESS/task/index and compact stop/config/bank/launch receipts updated.
Only the ongoingtraining/final pipeline remains. No default, numerical-core,
training-reference or Git history change, no artifact deletion. Exact resume:
read existing NEW1024journal/probe logs and gate receipts; monitor progression to
probe_cpu then canonicaltraining and initial audit, followed by fixed30k/50k
validation. Keep this task ACTIVE; blocker none while automatic stages progress.

Status-session task-end: checked existing PIDs/heartbeat and complete loss/L2
records; updated current task/PROGRESS/live receipt/index. JSON/diff checks only,
no unchanged expensive tests, no numerical/config/process change. Next action
remains existing NEW1024journal, matched-step metrics and automatic30k/50kaudits.

## Closed by explicit user switch — 2026-10-31

Stopped runner339235/trainer346501 by SIGTERM06:08:33UTC at26900logged/26500evaluated;
lastphase train5.9137%, validation7.9115%, terminal8.6213%. Preserved all outputs
and paired25kstate. This fixed50kcontrol is incomplete; no50kresult or automatic
final pipeline claimed. User requests checkpoint25000 continuation with a
background online rolling input pool. New active task: ../confined-ch-online-rolling-pool.md.
Stop receipt: ../confined-ch-width192-bank1024-50k-stop.json. No Git history change.
