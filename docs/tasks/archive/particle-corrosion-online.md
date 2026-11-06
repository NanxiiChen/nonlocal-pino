# Rolling geometry and spatial pools for particle corrosion

Date: 2026-11-06 (Asia/Shanghai). Status: complete.

## Completed outcome and resume point

The requested rolling-geometry and independent core/halo/IC refresh route is
implemented, cold-trained20000updates, fully audited and selected through
configs/train_default.py. Normal invalid candidates are rejected/regenerated;
budget exhaustion continues the same RNG with explicit logs. Unexpected errors
remain visible. Run02 passed the exact formerly failing draw251 in production.
No jobs or required work remain. No new Git operation or deletion authorized.

Final same8reused-localFEM comparison (mean per-geometry physical L2):

| Metric | Historical fixed pool | Rolling default |
|---|---:|---:|
| Phase pooled / terminal | 1.958708% / 3.423560% | 1.982720% / 3.120851% |
| Concentration pooled / terminal | 2.351003% / 4.089795% | 2.396368% / 3.740510% |
| Corroded field pooled / terminal | 5.8353% / 7.5258% | 5.900376% / 6.859836% |
| Terminal IoU | 96.1171% | 96.2247% |
| Held-update median | 19.6905ms | 22.5625ms |
| Total training (timing.json) | 747.9236s | 959.0387s |
| Active peak memory | 4.1717GiB | 4.9282GiB |

Rolling dev4 phase2.050688%/3.509405%,concentration2.483359%/4.223271%.
The total959.088235s in summary includes a final0.05s beyond timing.json; use
comparison's timing.json口径 consistently. Online control/transfer164.3128s
includes158.8213sproducer wait, not additive. One paired initialization and
finite previously inspected cohorts, plus different padded execution shapes,
do not establish a universal generalization or speed gain. Fixed pooled phase
remains slightly lower; rolling terminal errors improve. The requested default
selects continuous input refresh, not a new all-metric precision SOTA. Preserve
fixed train_wide, its historical image/checkpoints and lowest-pooled record.

Actual399events installed316newgeometries and1520spatialrefreshes,1836payloads;
380distinct geometries and1900geometry/spatial versions were sampled. All20k
consumed identities and all64final slot arrays independently reconstruct exactly;
no evaluation overlap or evolved training labels. Cold model/fulloptimizer and
all common slot/pool/time draws match the fixed control. Geometry sampler repair
is the only documented shared numerical-source exception.33focused tests pass
(7geometry+8producer+13control+5comparison), plus the actual production audit.

Direct final outputs:
- Run: /root/autodl-tmp/tf-logs/particle_corrosion2d/20261106-online-geometry-spatial-w192-cold20000-02
- Final8evaluation: /root/autodl-tmp/tf-logs/particle_corrosion2d/20261106-online-w192-reused8-final-02
- Comparison/curve: /root/autodl-tmp/tf-logs/particle_corrosion2d/20261106-online-vs-fixed-w192-02

Diff-level changes: new online producer,preset,input audit and saved-field
comparator; canonical train integration and default alias; normal geometry
admission/retry instrumentation; focused tests; README/development/artifact and
memory records. No shared model, PDE or optimizer implementation changes and
no new curated image. Existing uncommitted earlier example work is preserved.

## Goal and acceptance

User requests fresh random geometries rolling through a bounded training pool,
and rolling core/halo spatial sampling, aligned with existing examples. Preserve
the eight-particle/four-near-pit prior and current physical/model/loss recipe.
Implement explicit background known-input generation without trajectory solves;
retain fixed JIT shapes, complete halos and physical-volume sampling. Old fixed
bank controls remain replayable. Validate replacements/reproducibility and run a
cold control against unchanged local references, measuring wait, timing and L2.
No commit/push/deletion requested. RTX5090 32GB; .venv only.

## Context

Selected fixed-bank run: /root/autodl-tmp/tf-logs/particle_corrosion2d/20261106-eight-nonlocal-w192-cold20000-01.
Input bank: /root/autodl-tmp/data/particle_corrosion2d/20261106-eight-input-bank64-01.
64geometries,8fixed core pools each,256cores;B4,Nt8,width192,Adam20k,FP32.
Reserved8 phase1.9587%/terminal3.4236%,c2.3510%/4.0898% versus localFEM.
Geometry and core pools were fixed. Actual perstep draws are replayable.
Reference banks development4 and reserved8 remain evaluation-only. Prior8
reserved cases have already been inspected; they are now a reused comparison
cohort, not a new untouched final test for this strategy.

## Original plan (completed)

1. Review existing geometry/spatial producer APIs and particle input construction.
2. Implement bounded asynchronous geometry replacement and independent core refresh.
3. Focused fixed-shape/operator/replay/refresh tests and production-size probe.
4. Cold nonlocal20k control, input audit and same-reference accuracy/cost comparison.
5. Select requested rolling default, document actual validation/limitations and
   archive task. Preserve all historical configs/checkpoints and current images
   unless newly evaluated fields justify their replacement.

Blocker: none. All five steps completed; no pending resume action.

## Implemented schedule and pretraining verification

Explicit online preset: workers4; geometry4slots/250updates; spatial4slots/
50updates; boundedprefetch4events; halo3072/neighbors64; workerLRU64geometries.
Geometry replacement rebuilds all known inputs atomically; independent spatial
updates regenerate8core sets and1024IC points on the retained physical geometry.
Same-tick overlap is merged into the new geometry. Pure stateless namespaces
and per-update geometry/spatial versions prevent scheduling-dependent selection.
The buffer anticipates about200updates to absorb the measured prior rejection
runtime tail; worker caches include geometry records, not just meshes/operators.

128freshgeometry/1024core-pool CPU screen:0overflow,4workers84.15s, maximumhalo
2522/neighbors46, pergeometrymedian2.05s/p955.41s,8core setsmean42.7ms.
Artifact: /root/autodl-tmp/data/particle_corrosion2d/20261106-online-input-feasibility128-01.
Six real-worker/halo/padding/cache/FIFO tests and12control-provenance tests pass.
Root integration uses donated fixed-shape slot replacement, complete actual
identity trace and initial/final payload snapshots. Old fixed route unchanged;
actual historical config/source matches except explicit operator.online.
Small-network12update integration probe is running at
/root/autodl-tmp/tf-logs/particle_corrosion2d/20261106-online-integration-probe-01.
This probes refresh/control wiring, not production accuracy or resource limits.

## First production job (subsequently interrupted)

Integration probe completed12steps; auditpassed5events,4newgeometries,
10spatialrefreshes,10rebuiltfinalslots exactand54unchangedinitialslots exact.
All consumed geometry/revision identities match. Noheldoutoverlap or labels.

Production PID113050:
/root/autodl-tmp/tf-logs/particle_corrosion2d/20261106-online-geometry-spatial-w192-cold20000-01
Console and PID use the same prefix in that run root. Started cold with
configs/train_online.py and --sampling-control pointing to oldfixedwidth19220k.
Both initial arrays/optimizer and all slot/pool/time draws reproduce the old
run; addedproducer uses independent namespaces. Runtime seed1607243318 is
provenance, not a tuned config. Additional --exclude-reference supplies the
previous reserved8 bank; development4is always excluded. NootherGPUjob.

Next: monitor first compiled updates and early replacements for resource/wait
cost, finish20k, run online_validation --run above --workers4 --control oldrun,
then evaluate final on reused8. Update requested rolling default/docs and exact
metrics, but preserve oldfixed presets and original comparison data. Do not
launch a duplicate job. No commit/push or cleanup authorized.

Production passed compilation and actual swaps; first1000phase5.8205%,c6.9563%,
ordinaryupdates~21--22ms, initialactivepeak4.9203GiB. Finalizer execsession72533
runs /tmp/particle_online_finish.py waiting onPID113050: finalCPU inputaudit4workers,
GPUfinalevaluation on reused8, then CPUcompare_sampling. No extra training.
Expected finalpaths: 20261106-online-w192-reused8-final-01 and
20261106-online-vs-fixed-w192-01 under the same runroot. It writes separate
20261106-online-final-{audit,evaluation,comparison}.console.log; failures propagate.
The helper only validates saved fields, does not modify either originalrun.
Additional5comparison tests pass, so newfocused count is23(6+12+5).

## Production interruption — must not report completed20k

PID113050 exited at before_step(15750): new online geometry generation raised
ValueError from final validate_geometry refined clearance/curvature checks after
coarse candidate admission. Last installed event313 at15700; next event314 at
15750 draws248..251. Last saved model15000; developmentphi1.9183%,c2.3256% at15k.
No finalsummary, finalpool or exact optimizer checkpoint exists. Preserve run01;
interruption.json records the actual outcome. Finalizer session72533 exited1
without launching postprocessing; do not wait on it or reuse its future names
without a new owner. No model/PDE failure indicated by this input error.

Trainingagent reproduces offending candidate and will align its coarse/fine
admission with unchanged declared constraints. Physicsagent independently checks
prior interpretation and other refinement consistency risks. Need regression on
exactdraw and strict full planned input stream before another cold20k run.
No default promotion yet. Do not skipdraw/change seed/weaken margins to hide the
failure. Intermediateweights alone cannot supply an exact optimizer resume.

## Invalid candidates are normal rejections

Latest user steering explicitly requests rejecting/regenerating bad geometries
without terminating training. Candidate admission now checks both512/1024point
curvature/clearance limits; complete layouts also satisfy existing refinement
tolerances. No physical bounds change. For exact failing draw251, coarse
curvature0.015003903 passed but refined0.014999393 failed; the next accepted
geometry has minimumradius0.015452631. Six geometry regressions pass.

Full planned20k CPU input preflight completed with the repaired sampler:
380geometries(64initial+316new),399events,1836payloads/14688core sets;0failures,
maxhalo2583/maxneighbors45 below3072/64;4workers234.435s. All248already-installed
positive draw geometries exactly match run01. Direct path:
/root/autodl-tmp/data/particle_corrosion2d/20261106-online-fullstream-input-preflight-01.
This freezes the admission repair before the subsequent diagnostics-only API;
validate unchanged successful RNG/array paths separately, do not claim identical
source bytes. Dedicated GeometrySamplingExhausted retries will preserve the
Generator state, log every budget restart and leave other exceptions visible.
Next: finish diagnostic/retry tests, launch fresh cold20k run02, then audit/eval.

## Restart after the validated rejection repair

All15 geometry/producer tests pass (7geometry+8online), including forced budget
exhaustion then same-Generator success, unexpected-exception propagation and
cached diagnostics.13control-provenance+5saved-field comparison tests also pass.
Production-sized draw0/revision0 and draw251/revisions0,1 match all17payload
arrays against the frozen preflight implementation exactly. Draw251 geometry
and subsequent32RNG values are exact; diagnostics record one refined rejection
then success,0budget restarts. Report:
/root/autodl-tmp/data/particle_corrosion2d/20261106-online-geometry-rejection-diagnostic-01/metadata_parity/report.json.

Completed cold production is run02, former PID126711:
/root/autodl-tmp/tf-logs/particle_corrosion2d/20261106-online-geometry-spatial-w192-cold20000-02.
Same control/config/seed/input/reference paths as run01; no resume or changed
thresholds. Launcher command/environment,console,PID stored beside run02.
The training agent owns finalizer for run02 only: completeCPU input audit,
reused8model evaluation at20261106-online-w192-reused8-final-02,then saved-field
comparison20261106-online-vs-fixed-w192-02. No duplicate training. Await actual
20kcompletion/audits before updatingdefault. Generated statistics copied through
cache refreshes must be aggregated by unique geometryID, not summed perpayload.

Run02 crossed the original failure point: event at15750 installed draw251 after
one refined-clearance/curvature rejection,1273candidate proposals across2layout
attempts,0budget restarts. Training continued beyond16800. This is actual
production evidence of normal rejection, in addition to the forced-budget unit
test. Finalizer session95233 remains owner of audit/evaluation/comparison02.
