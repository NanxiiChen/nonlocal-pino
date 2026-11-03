# Channel cold20k and four PINO setup schematics

## Goal
Start one fresh20k random-wall NS-CH PINO with a lower learning-rate endpoint in the background, then draw matching conceptual setup/parameterization diagrams for melting3D, electropolish2D, confined spinodal2D and channel NS-CH.

## Acceptance criteria
- Preserve the validated width192/B8/FP32/SOAP/integral/JVP/rolling-input physics. Original budget/LR extension is retained; the user now explicitly selects36random time samples in two18-time chunks after the restart/memory study.
- Verify cold model/full optimizer, matching existing local FEM references, known-input construction coverage, GPU/memory identity and sufficient disk before starting.
- A finite detached owner runs canonical train.py and existing final saved-field/input/cohort audits, with direct paths and a durable process journal; no automatic retries/FEM solves.
- Confirm the trainer is actually running, then deliver four consistent conceptual diagrams identifying fixed setup versus variable condition; no hand-drawn field is presented as a solved result.
- Insert four portable setup SVGs into the corresponding example READMEs, keeping assets centralized and recording reproducible known-input geometry rather than evolved fields.
- Record the phase/displacement boundary example as deferred and do not implement it. Preserve current numerical SOTA/default until new validation completes.

## Context
Existing10k channel used1e-3 initial LR, exponential0.7per2000, floor1e-5; at10k the LR is~1.6807e-4. The same96GB PRO6000 is idle, prior timeblock20peak88.06GiB and~1.88s/update. User now explicitly authorizes cold20k and a lower endpoint plus background execution. Existing modified memory files from the preceding feasibility/sketch tasks must be preserved.

## Status
Completed:20k training and all prescribed saved-model audits/evaluation finished. Nominal phi1.4496%, terminalphi1.9020%, velocity1.3312%, pressure4.6703%; three reserved means phi1.6435%, terminalphi2.1961%, velocity1.5172%, pressure5.2840%. Explicit completion record supplements the original owner's240/18count rejection; historical journal remains unchanged. Future owner18route now explicitly uses252probes. Current10kdefault remains unchanged; no remaining authorized work.

## Blocker
None. Original240/18postprocessing count rejection is preserved as history and corrected with explicit252probe completion.

## Next action
No remaining action for this task. Preserve final weights/paired references/comparison/provenance. Any new training, default promotion or Git operation is a separate user request; none is started automatically.


## Current replacement execution

- Run: `/root/autodl-tmp/tf-logs/channel_nsch2d/20261103-pro6000-integral-random-walls-online-width192-b8-time18-nt36-cold20k-lr1e5-04`.
- Journal: `/root/autodl-tmp/data/channel_nsch2d/cold20k-restart-nt36-time18-20261103-04/owner/process.json`.
- Screen: `56642.channel-cold20k-nt36-time18-20261103-04`; resource probe56644.
- Launcher: `/root/autodl-tmp/data/channel_nsch2d/cold20k-restart-nt36-time18-20261103-04/launch.sh`.
- Config: `examples/channel_nsch2d/configs/train_operator_random_walls_width192_batch8_time18_nt36_cold20k.py`.
- Owner flag: `--execution-time-block=18`; probe flags `--time-block=18 --time-samples=36`.
- SAME192width/B8/SOAP/online/integral/JVP/physical horizon6/20k/LR1e-3->1e-5; Nt36in two18-time chunks.
- GPU PRO600096GB UUIDc55d506d-0eea-666e-713c-3ca8e3d81c9f. Resource gate PASSED: peak83.28GiB, warm1.6031s/update; input screen and actual cold-state checks also pass. Trainer73588completed/exited0; original owner56643stopped at postprocessing count validation. Explicit completion record below finished all remaining audits.
-6focused tests/config validation/changed-module AST pass. Actual20kGPU time draws independently replayed and all36strata passed; original geometry-prefix audit also passes.


## User refinement — insert the setup diagrams into the example READMEs

The user accepts the schematic style and requests portable README insertion.
Export the same known-input geometry as lightweight static SVGs in assets,
with a reproducible CPU-only script and compact sample-parameter provenance.
Keep the existing four evolution PNGs and numerical metrics unchanged. Training
owner135660/trainer135888 continues independently; do not change its copied
Python sources. README edits are outside the saved Python source-byte check.


## Original background execution — interrupted

- Screen: `channel-cold20k-20261103-01`, server135659, owner135660.
- Trainer:135888, canonical `python -m examples.channel_nsch2d.train`.
- Run: `/root/autodl-tmp/tf-logs/channel_nsch2d/20261103-pro6000-integral-random-walls-online-width192-b8-time20-cold20k-lr1e5-01`.
- Owner journal: `/root/autodl-tmp/data/channel_nsch2d/cold20k-lr1e5-20261103-01/process.json`.
- Owner/source/stage logs and planned config are in that journal directory.
- Control: `/root/autodl-tmp/tf-logs/channel_nsch2d/20261102-pro6000-integral-random-walls-online-width192-b8-time20-10k-01`.
- Config: `examples/channel_nsch2d/configs/train_operator_random_walls_width192_batch8_time20_cold20k.py`.
- Explicit97%allocator, the SAME qualified96GB PRO6000physical UUID.

The screen server and trainer have independent sessions. The initial launcher
used screen's nonforking-Dm flag and waited; after verifying every PID/session,
only that finished launch-wrapper PID135658was terminated, leaving the screen
and finite owner alive/reparented. This was not a trainer/input-screen restart.
No user training was terminated.

Historical same-shape memory evidence is explicitly reused from the qualified
time20-fraction97probe, not relabeled as a new resource run. The resource guard
requires unchanged full numerical recipe except budget/LR/run-name, the same
physical GPU, and its documented headroom. Full actual model_initial.eqx and
optimizer_initial.eqx bytes, initial bank/evaluation/reserved arrays match the
old10k control. No fitted state is imported. The nine unchanged numerical source
files checked against the old frozen trainer match exactly.

All16initial and802online known-input constructions pass in143.894s using8CPU
workers, maximum condition250.9298356and polynomial error5.60352e-12. No input
operator arrays or future solution labels are stored by that screen;0FEM/PDE
solves. The production rolling pool remains16slots/B8,2workers,2replacements
every50steps. First compilation emitted preserved CUDA-autotuning/slow-op
messages, then actual optimizer updates completed; no numerical retry, altered
compilation policy or fallback occurred.

Four focused budget-validation tests pass: actual existing Optax endpoint,
forbidden physical/seed/shape changes, cold model/optimizer/input mismatch and
wrong-GPU/execution resource reuse. Native train.py, shared optimizer factory,
old capacity contracts and all historical presets remain unchanged.

Finite owner stages after20k: validate_run(including actual rolling stream),
diagnose_phase, same-input/budget-prefix audit,240fresh continuous-time losses,
nominal droplet/gallery, same3reserved FEM cases and their galleries, full raw
convergence plots and frozen Python-source check. No FEM, automatic retries,
additional experiments, SOTA promotion or monitoring automation is queued.

## Four accepted setup figures and README insertion

The requested matching inline diagrams are in the thread visualization root:
`/root/.codex/visualizations/2026/10/24/01a16d71-2d47-7201-9441-ab74e0523152/`:
`pino-melting-setup.html`, `pino-electropolish-setup.html`,
`pino-spinodal-setup.html`, `pino-channel-setup.html`. Each has3known inputs,
fixed/variable labels, physical BCs/process and encoder condition type.
The native JavaScript runs at320/360/736/1024widths(48panels), with finite shapes
and correct phase/variable marks. No browser engine was used for those HTML
checks; a geometry-only contact sheet was visually inspected.

For portable README insertion, added `assets/pino-setups/{icemelting3d,
electropolish2d,confined_ch2d,channel_nsch2d}.svg` immediately before Important
strategies in each Problem setup section. Exact known input parameters live in
`artifacts/pino_setup_schematics.json`, with CPU-only reproduction via
`scripts/render_pino_setup_schematics.py`. The4SVGs total199768bytes(195.09KiB);
the shared spinodal texture is defined once and reused under each domain clip.
Actual SVG files were rasterized with system librsvg/cairo and all4previews
visually inspected, including labels and clipping. Valid XML/accessibility,
no scripts/external image dependencies and22changed-overview local links pass.
All four original evolution PNGs match their Git HEAD bytes exactly.

Drawing details: melting is a fixed orthographic projection of valid rotated
ellipsoids; electropolish uses accepted6-mode interface samples; spinodal uses
one fixed10-mode ambient function with two valid random holes per geometry;
channel uses independently sampled qualified spline walls, fixed straight ends
and an identical initial droplet. Input schematics do not show predicted fields
or assert future morphology. No inference/reference solve was used to draw.

## Diff-level changes and task-end

Added the20k opt-in config, narrow budget/resource/cold-input guard, finite owner
and4focused tests. Added compact schematic parameters, CPU SVG renderer and4
portable SVGs; inserted them into the four READMEs. Updated root README, assets
policy, DEVELOPMENT and artifact ownership(9cases/447directories). Existing
index content guard passes; all new candidate files also satisfy the1MiBcap.
No unchanged expensive test suite rerun. Whitespace/AST/JSON/XML/link checks pass.
PROGRESS, DECISIONS, this self-contained task and compact validation receipt are
updated. Current user-authorized publication completed on devbaf211b and regular mainmerge4ab2d98; actual origin/main reference matches. All24prepared files, including only4new generated SVGs, were published. Staged guard:2159files/26.26MiB;9cases/447artifact directories;54overview links; AST/JSON/XML/whitespace and originalPNG parity pass. The unchanged4focused tests retain their earlier pass. No cleanup, restart or numerical change.

The task remains ACTIVE solely for the background20k run/final audits. Preserve
the current10k numerical default and all existing external artifacts.

## Original publication-time snapshot — historical

```json
{
  "status": "training",
  "runner_pid": 135660,
  "active_child_pid": 135888,
  "heartbeat_utc": "2026-11-02T17:09:53.493251+00:00",
  "latest_loss": {
    "step": 900,
    "total": 0.07301872223615646,
    "parts": {
      "phase": 0.0186283141374588,
      "chemical": 0.015538599342107773,
      "momentum_x": 0.014478139579296112,
      "momentum_y": 0.023580167442560196,
      "wall_phase": 0.00023523051640950143,
      "wall_mu": 0.00021585139620583504,
      "outlet_traction": 0.00014751387061551213,
      "outlet_phase": 4.39542272943072e-05,
      "outlet_mu": 0.00015095077105797827
    }
  },
  "latest_monitor": {
    "step": 500,
    "elapsed": 1387.4213997870684,
    "phi": 0.16045069044767718,
    "terminal_phi": 0.18404425036284416,
    "dispersed_phase": 0.42397919731182654,
    "velocity": 0.0684368066250908,
    "pressure": 0.12946496772193702,
    "chemical": 1.0542165944452593,
    "phi_range": [
      -1.0591129064559937,
      1.0482302904129028
    ],
    "reference": "/root/autodl-tmp/data/channel_nsch2d/random-walls-20261102-02/development/bdf-fine.npz",
    "scope": "One local-FEM geometry; not a generalization estimate"
  },
  "latest_refresh": {
    "completed_updates": 900,
    "batch": 17,
    "slots": [
      2,
      3
    ],
    "draw_ids": [
      34,
      35
    ],
    "wait_seconds": 0.027300778776407242
  }
}
```

## Publication and current status closeout

The user-authorized payload is published: devbaf211b, regular mainmerge4ab2d98,
verified actual origin/main4ab2d981091eb28f23523e1259062555187b43f6. This compact
closeout records that completed publication and current training observation;
the prior closeout receipt was also published at main57f617b. No Git history operation is authorized in this restart turn. No
weights, raw arrays/logs or temporary previews were committed. All103frozen
trainer Python files remain byte-identical after branch changes and merge.

At the latest shared500-step monitor, new/old phi16.0451/15.2738%, terminal
phi18.4044/17.7469%, velocity6.8437/6.9654%, pressure12.9465/12.7339%,
minority42.3979/40.3600%, chemical105.4217/99.8577%. This is one development FEM
geometry at an early step, not generalization or the20k endpoint. Training is
still reducing loss; no accuracy advantage has been established. Read the live
journal/JSONL for later values, not these dated snapshots.

Task remains ACTIVE for the same owner/trainer20k endpoint and finite audits.
Memory updated: PROGRESS, DECISIONS, this task and its compact JSON receipt.

## User-authorized shutdown replacement — 2026-11-03

Original owner135660/trainer135888 are absent and their screen socket is dead.
Original log ends at8800/latestmonitor8500 with no exception; PID1 restarted
recently. User confirms an accidental shutdown and requests a fresh run from0.
Original artifacts remain unchanged.

Replacement run: `/root/autodl-tmp/tf-logs/channel_nsch2d/20261103-pro6000-integral-random-walls-online-width192-b8-time20-cold20k-lr1e5-02`.
Replacement preparation root: `/root/autodl-tmp/data/channel_nsch2d/cold20k-restart-20261103-02`.
Replacement finite owner journal: `/root/autodl-tmp/data/channel_nsch2d/cold20k-restart-20261103-02/owner/process.json`.
Screen: `channel-cold20k-20261103-02`.
Use the same existing cold20k preset/runtime42, fresh complete SOAP and network.
The new GPU UUID is GPU-c55d506d-0eea-666e-713c-3ca8e3d81c9f; same96GB PRO6000
name/memory but old same-device evidence is inapplicable. Existing explicit
resource probe must qualify11updates/timeblock20/fraction.97 before the same
finite owner repeats802known-input constructions and launches canonical train.
The resource probe has no fitted-state exports. Exact command/source and logs
will be kept in the replacement preparation root; no Python trainer changes.

Replacement resource probe is actually running under detached screen3534,
launcher3536,probe3539. Exact finite shell chain: `/root/autodl-tmp/data/channel_nsch2d/cold20k-restart-20261103-02/launch.sh`.
`bash -n` and the existing narrow config comparison pass; all103frozen old
trainer Python files remain unchanged. Actual fresh probe model and complete
SOAP state bytes match the completed10k initialization before any probe update.
This is not yet a resource pass or actual replacement trainer initialization.
The shell stops on failure; no automatic retry or further experiment.

Replacement resource validation passed11complete finite-gradient updates plus
finite nominal field evaluation on the actual new GPU; peak92689859072bytes
(86.324GiB), warm1.896346s/update, compile372.312s. No fitted probe weights
exported. The existing owner3536has started input screen19812; its actual
prepared configuration equals the interrupted run except fresh run name.

## Replacement actual cold-state verification

Canonical trainer19929has written its actual fresh model and complete optimizer.
The unchanged owner independently verifies exact initial model/fullSOAP bytes,
initial geometry bank, evaluation inputs and reserved geometries against the
completed10k run. No fitted state is imported; the new initial(phi47.94%)
monitor is step0, not a trained result. All103copied Python sources match.
The production first-update compilation is now running after the finite input
screen; use the replacement journal above rather than the dead original owner.

## Canonical-step memory failure and explicit corrective execution

Attempt02owner/trainer exited1at the first update: allocation91.91GiB
failed under the unchanged97%allocator. Its finite probe had passed86.32GiB;
the different JIT auxiliary-output tree does not guarantee matching compiled
temporary memory. No completed optimizer update or trained L2from02. Preserve
all logs and actual cold-state proof. No automatic retry occurred.

Prepared a separate8-time-block cold20k preset and an explicit owner/guard
argument. The unchanged20route still rejects undeclared block changes. The new
route allows only8, retains all40times and sampling/input-prefix checks, and
reports changed reduction order/throughput.5focused tests pass in0.684s, and
changed modules parse/whitespace check. A new finite chain under screen36692
qualifies8block, then repeats input screen and cold canonical trainer.

## Timeblock8 actual resource qualification

Resource gate passed on the actual replacement GPU:11complete finite-gradient
updates plus nominal field evaluation, peak43147762688bytes(40.184GiB), warm
2.394457s/update, compile365.078s. No fitted state export. Finite owner36693
started input screen53040after the resource process exited. Actual prepared
configuration equals original01except the declared8time block and fresh name.
Expected20k raw-update time13.30h; production startup/final audits add time.

## User refinement —36random times in two18-time chunks

User rejects the40GiB/time8peak and explicitly clarifies Nt36/timeblock18,
not18geometry samples. Geometry batch remains8. The owned03owner36693was
terminated via its signal handler, which stopped its managed input child;
all artifacts remain. No unrelated process was touched. No16+16+8padding,
remainder or new accumulation implementation was added.

The new narrow route explicitly permits only36/18against old40/20. Final
geometry sampling/input prefixes remain equal; time-prefix equality is false
by design and is replaced with required independent36-stratum/global-key
replay proof.6focused tests pass in0.619s, including rejected missing sampling
replay and geometry-prefix changes. Existing numerical train/loss/operator/
time-accumulation code is unchanged. CPUpreflight covers every20k36-time draw
with finite ordered in-stratum values; no actual arrays are stored.

## Actual user36/18resource result

Eleven complete finite-gradient updates plus nominal field evaluation passed on
the actual PRO6000: peak89420045312bytes(83.279GiB), warm1.603065s/update,
compile369.834s. Fresh probe initial states match the old completed control;
no fitted probe state is exported. Finite owner56643and input screen73482are
running. Expected raw20k update time8.91h plus startup/evaluation/final audits.
This resource proof still requires actual canonical optimizer confirmation.

## Actual36/18canonical cold initialization

Canonical trainer73588is live under detached owner56643/screen56642. Its
actual saved model/full SOAP bytes and initial/evaluation/reserved geometry
inputs exactly match the completed control; no fitted state was imported.
All105frozen Python sources match current files. All16initial+802online
constructions pass in135.729s with0PDE/FEMsolves. Only initial step0metrics
are available at this snapshot; production first-update compilation is running.

## Restart session task-end — actual canonical update confirmed

Canonical train.py completed its first optimizer update with finite loss
23.4669589996and continues under detached owner56643/trainer73588/screen56642.
This confirms the user-selected36/18production executable passes the memory
allocation that failed in02. Warm resource timing is1.603s/update and probe
peak83.28GiB; do not relabel probe peak as externally measured production
active-memory peak. The105current frozen Python source files remain exact.

Diff-level changes: two opt-in execution presets(40/8and final36/18), explicit
owner/config/resource guard extensions, mandatory independent36-stratum/time
replay rather than false40-time-prefix parity, resource probe36/18CLI,6focused
regressions and matching README/DEVELOPMENT/artifact ownership. Numerical
train/loss/operator/equal-block accumulation code is unchanged. CPUall20k
36-time planned draws pass; actual finalGPU sampling still requires its audit.
AST/JSON/whitespace and9case/452directory artifact checks pass. No unchanged
expensive suite rerun, FEM, cleanup or Git operations. Current10k default stays.

Memory updated: PROGRESS, DECISIONS, this task and its JSON receipt. Task stays
ACTIVE solely for the already-running36/18cold20k endpoint and finite audits.
Exact resume journal is the04path at the top; earlier01/02/03attempts are
interrupted/failed/superseded history with artifacts preserved.

Final session observation: at least100actual updates, latest logged step100loss1.2066961526870728.
Initial step1loss23.466959has fallen to1.206696at100; owner/trainer remain
alive and actual rolling refresh100has completed. Latest L2is initial0only;
first trained monitor is scheduled at500. Leave the SAME04owner running.

## Training-cost explanation — read-only inspection

User asks whether the slower case comes from more points or more loss terms.
Measured actual canonical loss flush300->400:162.650005s/100updates,
1.626500s/update, excluding cold compilation and including host/rolling/log work.
Detailed read-only observation: `/root/autodl-tmp/data/channel_nsch2d/cold20k-restart-nt36-time18-20261103-04/performance-cost-observation.json`.
No extra GPU benchmark/profiler or training/source change was made.

Each geometry uses193x81=15633mapped nodes(15089interior PDE rows),
B8xNt36=4502304logical space-time-condition nodes/update. K128makes
576294912logical neighbor visits per operator application; multiple channels/
operators and reverse-mode buffers add work. Electropolish default uses
256PDEcores+halo and4times/B8, so default residual sample counts are very
different. These are logical counts, not a claim that all hypothetical
intermediates are materialized or a wall-time ratio.

Actual fields use spatial JVPs for the streamfunction curl, time JVP of the
physical fields(including mixed psi_xt/psi_yt), then parameter backprop. The
nonlocal PDE/BC gradients/Laplacians still gather neighbor values.9soft losses
share fields/derivatives and are summed before each parameter-gradient call;
two time-block gradients precede one SOAP update. Fixed weights, no9-way
GradNorm gradient evaluation. Geometry attention is associative linear integral
attention; spatial query preparation is shared across each block time axis.
Do not attribute a quadratic full spatial attention matrix to this model.

Interpretation: workload scale and derivative/neighbor computation explain
why this coupled full-grid case is heavier; exact component percentages require
profiling and were not measured. Prior startup delay included resource checks,
OOM and shape-triggered compilation. The20kbudget doubles update count versus
the completed10k run. Keep SAME04training/final audits active; no new tuning.

Task-end: only this read-only diagnostic and PROGRESS/DECISIONS/task/receipt
records updated in this follow-up. JSON/diff checks pass; no code test rerun
is needed. Exact resume remains the live04owner journal above.

## Live progress follow-up — 2026-11-03T14:16:21.479295+08:00

Owner56643/trainer73588and fresh journal heartbeat are normal; at least5900
updates, latest loss step5900=0.0065453229. Latest L2is monitor5500on
one reused development FEM geometry: phi2.81947%, terminalphi3.67511%,
velocity2.79624%, pressure6.03765%, minority7.45024%, chemical21.73298%.
Previous5000monitorphi3.37669%fell to2.81947%at5500; pressure fluctuates.
Oldrun same5500phi2.61677%,velocity2.93123%,pressure5.57026%; overall similar,
no established new accuracy advantage. The two runs have36vs40time sampling.

Last500-update interval1.624389s/update including evaluation/host work,
remaining training approximately6.362h; final audits add time. ckpt_005000.eqx
is saved; no new endpoint/cohort result yet. Keep SAME04job and final audits;
no extra run, tuning, source edits, cleanup or Git operations. Task-end: updated
PROGRESS, this task and JSON receipt; JSON/whitespace checks pass. No numerical
test rerun is needed for this status-only update.

## Same-interval yesterday/today timing comparison

User feels today is slower. Compare actual metric elapsed4000->5500on both
runs: old40/20time1.901588s/update, current36/18time1.624145s/update;
current update time is14.5901%lower, with10%fewer time samples per update.
Both use192width/B8but differ in sampling/execution/LR and physical GPU UUID,
so this is observed throughput, not isolated method/hardware speedup evidence.
Yesterday budget10000completed in5.4170h; today budget20000requires~9.02h
raw updates plus startup/final work. The doubled update count makes the full
budget~1.708times longer despite quicker steps. Today also spent extra startup
time on resource checks, one canonical OOM and changed-shape recompilation.
Actual canonical training started2026-11-03T03:27:50.995985UTC(11:27:51Shanghai).

Direct comparison file: `/root/autodl-tmp/data/channel_nsch2d/cold20k-restart-nt36-time18-20261103-04/yesterday-today-timing.json`.
At least5950updates now; latest development L2still5500. Keep SAME04
trainer/final audits running. Status/timing follow-up changes only PROGRESS,
this task and JSON receipt; JSON and git diff checks pass. No tests or Git
operations are needed/authorized. Task remains active at the same resume point.

## Current user subtask — live convergence and physical fields

Generate raw current/old convergence and use actual saved5000weights for
phase/speed/pressure reference-prediction-error panels and interface overlap.
CPU-only existing partial replay must match the logged5000metrics; GPU trainer
continues. Output request: `/root/autodl-tmp/data/channel_nsch2d/cold20k-restart-nt36-time18-20261103-04/live-gallery-20261103-143218/request.json`.
This plotting subtask is complete; resume SAME04background20k/final audits.
No FEM, training edits, cleanup or Git.

## Live-gallery task-end — 2026-11-03

CPU-only actual saved5000model replay passed; phi3.37669%, terminalphi4.03156%,
velocity2.93215%, pressure5.07420%. Largest absolute relative-L2gap to the
original GPU5000monitor is1.34645e-8. No fitted model reconstruction from logs,
optimizer import or new PDE/FEM solve. SAME04GPU trainer remains alive under
the same owner, at least6750updates with fresh heartbeat.

Reproducible CPU plotter: `docs/tasks/channel_live_training_gallery.py`.
External output directory: `/root/autodl-tmp/data/channel_nsch2d/cold20k-restart-nt36-time18-20261103-04/live-gallery-20261103-143218/figures`.
Raw current L2through6500and full old10000history appear in relative-l2.png;
sampled-losses.png preserves raw normalized MSE. phi-fields.png, speed-fields.png
and pressure-fields.png have three rows(reference,prediction,absolute error)
at five exact saved times. droplet-contours.png uses native phi0contours,
equal physical zoom and uncorrected positions. All five primary images were
visually inspected. Phase includes t0; pressure/velocity start at0.5because
pressure is not prescribed initially. No temporal interpolation or curve
smoothing; mapped bilinear interpolation is for field display only.

At6500the scalar monitor is phi2.74285%, terminalphi3.63954%, u2.63021%,
p8.66227%, minority7.24778%, mu20.49775%. Pressure fluctuates rather than
decreasing monotonically. Curves are one reused development FEM geometry;
the field panels are actual5000weights, not6500weights or a cohort result.
Exact raw curves, source copy, provenance and replay gaps are saved beside
the figures in curve-data.json and gallery-validation.json.

Verification: actual CPU replay and plotting commands exited0, saved native
field shapes/times/contour counts passed, source freeze/AST/JSON/whitespace
checks passed. No numerical training changes or expensive suite rerun.
Memory updated: PROGRESS, DECISIONS, this task and its JSON receipt. Keep task
ACTIVE at the SAME04owner journal; only the plotting subtask is complete.

## Live progress follow-up — 2026-11-03 19:38 Asia/Shanghai

Read-only live journal/log/process/GPU inspection confirms SAME04owner56643
and trainer73588remain normal, fresh heartbeat, GPU100%. Actual loss/rolling
logs confirm at least17800updates(89%of20k), loss0.00245255325. Saved
checkpoints5000/10000/15000exist; newest scalar monitor17500is not a saved model.
Monitor17500on the one reused development localFEM case: phi1.53663%,
terminalphi2.21572%, velocity1.37488%, pressure2.57058%, minority4.06043%,
chemical10.51766%. Recent16500->17500wall interval1.62570657s/update;
remaining2200updates~0.9935h plus finite final audits. Current LR~1.7e-5.

Previous completed10k same-development-case endpoint: phi2.67813%,
terminalphi4.05496%, velocity1.67001%, pressure5.89309%, minority7.07676%,
chemical14.10304%. Current17500monitors improve all six quantities against
that endpoint. This is a longer-budget comparison with36vs40time sampling
and altered LR/GPU; do not isolate LR effects, assert monotonic convergence,
claim reserved-cohort generalization or promote the current default yet.

Task-end: only PROGRESS, this task and its JSON receipt updated for the status
request. JSON/whitespace checks pass; no numerical code/tests, new plots, FEM,
cleanup or Git operation. Keep SAME04background job active at its existing
owner journal; inspect its final endpoint and every finite audit on completion.

## Current user subtask — inspect saved15000physical fields

User requests current physical fields after the17500status update. SAME04has
passed18100updates with monitor18000; actual latest saved model is15000.
CPU-only partial replay of ckpt_015000.eqx must match its original15000logged
metrics, then reuse the gallery renderer for phase/speed/pressure/error and
native zero-contour overlap. Keep the GPU trainer and finite owner untouched.
Label all field images15000, not the newer scalar monitor. One-line plotter
provenance correction replaces the stale hardcoded saved5000scope with the
explicit requested checkpoint step; frozen numerical source remains unchanged.
Next: save direct-path request, replay, render/inspect and deliver; resume
SAME04background endpoint/audits. No FEM, cleanup, default promotion or Git.

## Saved15000gallery task-end — 2026-11-03 19:50 Asia/Shanghai

Subtask completed. Actual saved15000CPU replay exits0 and matches its original
GPU monitor within6.65468e-8 absolute relativeL2: phi2.03711%, terminalphi2.91763%,
velocity1.44577%, pressure3.63953%. Explicit replay source/request/log and
plot source/raw data/provenance are preserved under
`/root/autodl-tmp/data/channel_nsch2d/cold20k-restart-nt36-time18-20261103-04/live-gallery-15000-20261103-194855`.

The four requested primary physical images(phi,speed,pressure,zero-contour
overlap) and additional raw current18000/old10000convergence were rendered
and visually inspected. Field panels use five exact saved times with common
reference/prediction scales. Query96x40; mapped bilinear interpolation for
display only; no temporal interpolation/smoothing or contour alignment.
Each reference/prediction has one zero contour at each chosen time.
Later scalar18000phi1.14466%,terminalphi1.27513%,u1.32866%,p2.03878%must not
be assigned to these saved15000field panels. One reused development geometry,
not an unseen-cohort assessment. Late front/rear outline differences remain.

SAME04owner56643/trainer73588and fresh heartbeat remain normal, at least18200
updates and loss0.00208632. Frozen105Python sources unchanged. Actual replay
and renderer commands exit0; dynamic provenance scope says saved15000,
JSON/native shape/time/contour/image/AST/source-copy/whitespace checks pass.
No training/numerical source changes, new tests/FEM/cleanup/Git operations.
Memory updated: PROGRESS, this task and its JSON receipt. Keep overall task
ACTIVE; exact next action is the SAME04owner endpoint and finite final audits.

## Live progress follow-up — 2026-11-03 20:29 Asia/Shanghai

Read-only journal/log/process/GPU checks show SAME04owner56643/trainer73588
alive and normal, fresh heartbeat, at least19650/20000updates(98.25%). Latest
logged loss19600=0.00295950007. No20k summary/final model or post-training audit
yet; final validation/reserved-cohort stages have not started. Do not call
this a completed run. Roughly350updates/~10min training remain, plus audits.

Latest scalar development FEM monitor19500: phi1.55617%, terminalphi2.20435%,
u1.35882%, p4.15090%, minority4.11207%, mu11.45026%. All remain below the old
completed10k endpoint on the same development case, but phi has risen from
1.14466%at18000and pressure from2.03878%; convergence is not monotonic.
The18000minimum was not saved and is not the final/current checkpoint.
Keep the finite20k endpoint and independent3reserved-case assessment before
judging generalization or promoting a default. No extra run/tuning is started.

Task-end: PROGRESS, this task and JSON receipt updated; JSON/whitespace checks
pass. No numerical tests/source edits, plot regeneration, new FEM, cleanup or
Git operation. Task stays ACTIVE at the SAME04owner journal for endpoint and
all prescribed final audits; do not interrupt or duplicate the job.

## Final20k comparison and task-end — 2026-11-03

Training completed/exited0at2026-11-03T12:38:35UTC(20:38:35Shanghai), with
32985.8068s(~9.1627h)training wall and1.60826s/warm update. Final model,
complete SOAP and final rolling inputs/pool are saved in SAME04run. Actual
production active-memory peak89653278720bytes, separate from the earlier
resource-probe peak. No fitted state import or supervised evolved samples.

Original saved-run validation/phase diagnostics exited0. Original owner stopped
before residual evaluation because240is not divisible by18; preserve its failed
journal and log. Current-turn explicit completion source/journal/logs:
`/root/autodl-tmp/data/channel_nsch2d/cold20k-restart-nt36-time18-20261103-04/final-completion-20261103-210103`.
Eight remaining stages exited0: same3reserved cases,252fresh-time diagnostics,
nominal shape/phi gallery, three reserved phi galleries and complete convergence.
Status `remaining_saved20k_audits_completed`;105frozen sources unchanged.
The future finite owner now records252for18blocks; legacy20/8retain240.
No trained numerical source/checkpoint/refinement gate was changed.

Independent NumPy replay from both saved endpoints passes with bitwise-identical
reference/point/time/geometry/common-query arrays and matching native field
metrics/mean/p90/worst. Source `comparison-source.py`, report
`endpoint-comparison.json`in the completion directory. Per geometry has3840
mapped query points and13times; phase includes hardt0, flow/mu use12positive
times. All errors below are relativeL2percent, old10k->actual20kendpoint.

| Quantity | Development | Reserved mean |
|---|---:|---:|
| phi |2.67813->1.44962|1.83960->1.64350|
| terminalphi |4.05496->1.90196|2.27076->2.19610|
| minority |7.07676->3.83052|4.84742->4.33056|
| velocity |1.67001->1.33121|2.01602->1.51721|
| pressure |5.89309->4.67028|5.41177->5.28405|
| chemical |14.10304->11.67444|15.91681->13.95413|

| Reserved case | phi | terminalphi | velocity | pressure |
|---|---:|---:|---:|---:|
|000|1.94292->1.98600|2.78040->2.86820|1.87520->1.41315|3.87953->4.71612|
|001|1.65191->1.42926|1.51962->1.67819|2.13351->1.56306|9.03157->6.58847|
|002|1.92399->1.51523|2.51225->2.04191|2.03935->1.57541|3.32420->4.54756|

Reserved mean relative reductions: phi10.6601%,terminalphi3.2878%,velocity
24.7426%,pressure2.3601%,minority10.6626%,mu12.3309%. Not every case improves:
case000phase slightly worsens;000/001terminalphase and000/002pressure worsen.
Worst pressure improves9.03157->6.58847%. Common-physical centered geometry
response errors(phi,u,p):18.9538/13.0948/72.9983%->21.3246/9.3906/67.2826%.
Thus lower total fields do not imply uniformly better geometry sensitivity.
One cold initialization, three reused predeclared cases,36vs40training times
plus changed LR/budget/GPU; not cross-seed/population/isolatedLR evidence.
Keep the10kdefault and curated README images unchanged.

252fresh-time diagnostics have finite nine-term MSE arrays on nominal/final
poolslot0, no reference labels; mean totals0.00435857/0.00194624. These are
normalized residual diagnostics, not continuum accuracy passes. Saved model
replay/hard constraints, actual36-stratum draws and all798installed rolling
inputs reconstruct successfully. Nominal terminal droplet areaIoU93.4280%,
arc-weighted contour mean distance0.00623902(~2.836%ofinitial radius).
The integral spatial operator remains approximate; physical AD-curl divergence
and nonlocal discrete divergence are distinct diagnostics.

Final verification: independent comparison, eight completion stage exit codes,
252probe strata/finite array shapes,105source bytes, original cold/prefix audits,
all saved final-state artifacts and complete41monitor/201loss records checked.
Four final phi galleries and convergence visually inspected. Owner CLI and
changed Python AST/JSON/artifact-root/whitespace checks pass; no unchanged
expensive suite rerun or new training/FEM/cleanup/Git operation. README and
DEVELOPMENT document the measured opt-in result and explicit probe-count fix.
Task archived; PROGRESS, DECISIONS, artifact registry and compact receipt updated.
No authorized work remains; no automatic SOTA promotion or further experiment.
