# Channel cold20k and four PINO setup schematics

## Goal
Start one fresh20k random-wall NS-CH PINO with a lower learning-rate endpoint in the background, then draw matching conceptual setup/parameterization diagrams for melting3D, electropolish2D, confined spinodal2D and channel NS-CH.

## Acceptance criteria
- Preserve the validated width192/B8/FP32/SOAP/integral/JVP/rolling-input physical recipe; change only explicit budget and LR decay in a new opt-in preset.
- Verify cold model/full optimizer, matching existing local FEM references, known-input construction coverage, GPU/memory identity and sufficient disk before starting.
- A finite detached owner runs canonical train.py and existing final saved-field/input/cohort audits, with direct paths and a durable process journal; no automatic retries/FEM solves.
- Confirm the trainer is actually running, then deliver four consistent conceptual diagrams identifying fixed setup versus variable condition; no hand-drawn field is presented as a solved result.
- Insert four portable setup SVGs into the corresponding example READMEs, keeping assets centralized and recording reproducible known-input geometry rather than evolved fields.
- Record the phase/displacement boundary example as deferred and do not implement it. Preserve current numerical SOTA/default until new validation completes.

## Context
Existing10k channel used1e-3 initial LR, exponential0.7per2000, floor1e-5; at10k the LR is~1.6807e-4. The same96GB PRO6000 is idle, prior timeblock20peak88.06GiB and~1.88s/update. User now explicitly authorizes cold20k and a lower endpoint plus background execution. Existing modified memory files from the preceding feasibility/sketch tasks must be preserved.

## Status
In progress: actual cold20k training is running in its detached owner. All four setup schematics are complete and inserted into their example READMEs. Only the20k training and its finite final audits remain.

## Blocker
None identified.

## Next action
Read the SAME owner process.json and live loss/metrics logs. Do not start a duplicate run or reinterpret initial metrics as trained error. Wait for the prescribed20k endpoint and inspect every final audit exit code/receipt, then compare against the old10k result before considering any default change. Publication is complete: devbaf211b -> regular mainmerge4ab2d98, origin/main verified equal. Keep the same trainer running; no further numerical changes, restart, default promotion or cleanup.


## User refinement — insert the setup diagrams into the example READMEs

The user accepts the schematic style and requests portable README insertion.
Export the same known-input geometry as lightweight static SVGs in assets,
with a reproducible CPU-only script and compact sample-parameter provenance.
Keep the existing four evolution PNGs and numerical metrics unchanged. Training
owner135660/trainer135888 continues independently; do not change its copied
Python sources. README edits are outside the saved Python source-byte check.


## Concrete background execution

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

## Latest observed live snapshot

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
use the same dev -> main -> origin/main workflow to publish the receipt. No
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
