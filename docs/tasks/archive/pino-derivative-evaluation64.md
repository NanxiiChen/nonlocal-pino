# Expand derivative-control evaluation to64inputs

## Goal and authorization

User requests64test samples in both Electropolish and ceramic-particle
corrosion, computing new references as needed and loading existing weights
for inference. User explicitly chose mean curve plus standard-deviation
shading. Apply the same cohort to both methods and both initializations;
use every available saved checkpoint without inventing intermediate model
states. Update README comparison results, assets and full direct-path
configuration/data/plot provenance. No new training, cleanup or Git action.

## Acceptance

- Same local physical reference per case/method/seed;64distinct inputs and
  documented test/train separation. Preserve source physics, nondimensional
  parameters, BCs and solver tolerances; qualify new particle references.
- Only actual saved checkpoint steps appear in local-reference convergence.
- Compute per-input relative L2, mean and sample standard deviation(ddof=1)
  across64inputs separately for each method/initialization. Shading is sample
  spread, not confidence in the mean. Keep terminal and concentration metrics.
- Exact saved config/source compatibility, finite output/shape/coordinate/time
  checks and independent recomputation; preserve original comparison failures.
- Respect exclusive FEM/JAX environments and available resources/disk. Avoid
  duplicate raw prediction storage; no deletion of existing artifacts.
- Update each README and curated provenance/plots; inspect figures and budgets,
  run focused behavior checks and `git diff --check`, archive upon completion.

## Context

Completed source study: `archive/pino-rolling-local-ad-controls.md`.
Publication: `archive/pino-derivative-readme-figures.md`.
Run root: `/root/autodl-tmp/tf-logs/pino-rolling-ad-cold-20261111-01`.
Electro already has64localFD101reference inputs and5k/final inference for all
four arms; its dense curves used a different4native monitor cohort.
Particle only4localFEM geometries so far; checkpoints expected at0/5k/10k/
15k/20k. Runtime seeds42/43 are initialization provenance. Electro43's strict
FP32input-value audit remains failed; publication must preserve that caveat.

## Status, blocker and next action

Complete. All8arms/32saved checkpoints evaluated on64common inputs per example;
reference/inference journals passed. No training, default change, cleanup,
Git/history operation or Pages update. No pending GPU job or blocker. Old
paused process group618216 remains untouched.

## Artifacts and exact provenance

- Source training study:
  `/root/autodl-tmp/tf-logs/pino-rolling-ad-cold-20261111-01`.
  The original raw journal remains `failed` because Electro43did not pass its
  strict input-value audit. The later qualified report is unchanged.
- New particle reference:
  `/root/autodl-tmp/data/particle_corrosion2d/local_test64_20261111-01`.
  `cohort.json` freezes64layouts and runtime seed3474072216 before any solve or
  inference. It excludes700unique initial/installed-rolling/old-development
  geometries; zero exact training or old4development overlap and no sampling
  rejection. Runtime seed is provenance, not a selected strategy parameter.
- Existing Electro reference:
  `/root/autodl-tmp/data/electropolish2d/operator_local_test64_20261029`;
  encoder/IC inputs and surface identities remain in
  `/root/autodl-tmp/data/electropolish2d/operator_test64_20261028`.
  This is the reused development cohort, not a newly claimed blind test.
- Evaluation root:
  `/root/autodl-tmp/tf-logs/pino-derivative-eval64-20261111-01`.
  `journal.json` retains all8exact commands, timings, status, log and source-run
  paths. Each case-s{42,43}-{nonlocal,local_ad}directory has full saved config,
  per-checkpoint64individual errors, mean/sampleSD, model path, training time,
  reference path and final summary. `training_performed=false` everywhere.
  Particle20k predictions retain all64cases; intermediate models retain only
  the first3fields. Electro5k/final64fields are reused without copying; initial
  fields for the first3are retained. All64initial errors are recorded.
- Full pre-update README/renderer/manifest/receipt/PNG copies are preserved in
  evaluation_root/`previous_readme`; original training artifacts are unchanged.
- `artifacts/pino_derivative_comparisons.json` schema v2 preserves all8saved
  configurations, source inventories, every checkpoint-evaluation path, field
  and reference paths, actual sample identities/times, qualification and plot
  settings. No hash-based lookup. The v2 validation receipt stores all actual
  curve mean/SDs and all512final per-input norms, not just rounded README values.

## Reference qualification

Independent Gmsh-conforming P1 FEM with lumped mass and SciPy BDF, using the
existing local physical solver and SI parameters unchanged. Base h=.01 and
256polygon nodes per hole; rtol2e-8, atol2e-10, max normalized step.00625.
Actual field times:11uniform points0..1, physical0..350s.64base solves and
5checks used8single-thread CPU workers and completed in188.126s before GPU
inference. No FEniCS environment or coupled FEM/GPU workload was used.

Spatial checks at predeclared indices0/21/42/63: h=.0075, same256-node polygons,
so the comparison isolates mesh refinement with fixed geometry. Pooled phase
sensitivity0.09646/0.09873/0.09030/0.10925%; concentration0.12228..0.14588%.
Index0time check halves max step and tightens rtol/atol to1e-8/1e-10: phase
1.31e-7%, concentration1.65e-7%. All pass predeclared0.5%spatial/0.01%temporal
phase thresholds. This checks4spatial/1temporal cases, not a continuum-error
certificate for all64cases. Interpolation uses physical FEM triangles only;
no extrapolation across holes or extrapolation outside the physical domain.

## Checkpoint evaluation and statistics

Particle steps0/5000/10000/15000/20000; Electro0/5000/10000. Every plotted dot
uses all64inputs; no four-input history is relabeled as a64input curve, and no
unsaved checkpoint is synthesized. Curves join actual dots. Bands are sample
SD(ddof=1) across inputs, not across seeds, SEM or confidence intervals. Both
initializations are kept in separate rows. Mean is the average of64individual
relative norms, not the norm of concatenated fields.

Particle uses physical lumped spatial volumes and trapezoidal11time weights.
Electro uses101x51nodes and21uniform saved times; extra early-time snapshots
in its25time field files are excluded from pooled norms. Endpoint metrics and
concentration statistics are retained as well as phase pooled errors.

All numerical model/source files match frozen checkpoint sources byte-for-byte.
Saved configurations remain authoritative. FP32ordinary/highest matmul
settings are preserved. Inference runs serially, completes in341.504s total,
and is distinct from original training timing. Particle fixed512point chunks
reuse compilation across all64cases/checkpoints. Each final model is replayed
on an old saved input: max absolute gap7.75e-7. Every Electro5k/final model is
replayed on input0: max gap7.45e-7; predeclared field tolerance2e-5. Existing
Electro64prediction arrays are reused after this check. All final512case norms
are independently recomputed from physical arrays during figure generation.

Final pooled phase mean ± SD, percent:

| Example | Initialization | Nonlocal | Local AD |
|---|---|---:|---:|
| Particle20k | 42 | 3.202254 ± 0.307664 | 33.344979 ± 0.438094 |
| Particle20k | 43 | 2.961316 ± 0.277513 | 33.518885 ± 0.400031 |
| Electro10k | 42 | 1.862481 ± 0.592381 | 4.261495 ± 1.710130 |
| Electro10k | 43 | 1.925559 ± 0.762312 | 3.331960 ± 1.403449 |

Electro43retains the original strict value-audit failure (max encoder8.94e-7,
IC1.252e-6); the64case reevaluation does not resolve that training-pair caveat.
Particle64means agree closely with earlier4development results; they support
the same finite-budget method ordering. No intrinsic AD failure, isolated
Neumann-cause diagnosis or nonlocal warm-step speed advantage is inferred.

## Publication and verification

Both example READMEs now show64case mean±SD tables/convergence and full metric
scope. Particle gallery uses the first3new64cohort entries, strict seed42final
models, actual times0/.2/.5/.8/1. Electro retains its first3original64cohort
entries and times0/.25/.5/.75/1; that field PNG is byte-identical to the prior
publication. Common phase scale0..1, actual physical triangles/nodes and
reference interface overlays are retained. Original SOTA assets/defaults are
unchanged. The four comparison PNGs total1,247,628bytes(1.19MiB), each<1MiB
and combined<2MiB. These compact assets and JSON receipts stay in the repo;
raw references/checkpoints/fields remain outside Git.

- Reference qualification passed before checkpoint inference.
- Statistics checked against the known mean32.5/sampleSDsqrt(64*65/12) of1..64,
  invalid counts/nonfinite errors rejected; known10%relative-field perturbation
  checked in both pooled and terminal metrics. Initial exact-float assertion
  was corrected to tolerance comparison for floating-point reduction rounding.
- `tests.test_particle_corrosion_reference`:4focused numerical tests pass
  (conforming holes/normals, boundary flux, local Jacobian and conservation).
- CPU renderer checks all8configs and frozen source inventories, all physical
  coordinate/time identities, all512final norms and all32mean/SD records.
- Two convergence plots and the updated particle field sheet visually inspected;
  unchanged Electro field sheet retains its previous inspection.
- AST/JSON checks and `git diff --check` pass. All4PNGs reproduce byte-for-byte
  in a second full CPU rendering pass. Frozen-cohort replay preserves exact
  geometry/config/seed before solves; duplicate replay geometries are rejected.
  External storage added:101MiB references and149MiB evaluations/old-publication
  backup, without duplicating existing Electro64field files.
- All263local Markdown targets in the touched overview/example/memory files
  resolve (math blocks excluded from link parsing). PNG integrity and separate
  image budgets pass; original failed training journal and passed evaluation
  journal retain distinct status. No live GPU work remains.

## Reproduction

Plot only, without inference/training/solves:

```bash
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/render_pino_derivative_comparisons.py --case all
```

All actual8inference commands are in evaluation_root/journal.json. For example,
use a new output directory to preserve the existing evaluation:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m docs.tasks.pino_checkpoint_eval64 \
  --case particle_corrosion2d \
  --run /root/autodl-tmp/tf-logs/particle_corrosion2d/pino-rolling-ad-cold-20261111-01-particle_corrosion2d-s42-nonlocal \
  --reference /root/autodl-tmp/data/particle_corrosion2d/local_test64_20261111-01 \
  --output /tmp/particle-eval64-replay
```

The original reference generation command had no --cohort and used a fresh
runtime seed. To rebuild the exact frozen geometry set rather than draw a new
one, pass its explicit cohort path and a new destination:

```bash
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m docs.tasks.particle_reference64 \
  --study /root/autodl-tmp/tf-logs/pino-rolling-ad-cold-20261111-01 \
  --cohort /root/autodl-tmp/data/particle_corrosion2d/local_test64_20261111-01/cohort.json \
  --output /tmp/particle-reference64-replay --workers 8
```

Both scripts reject existing output directories. The evaluator checks saved
model-source compatibility; reference replay rejects incompatible cohort configs
or duplicate/training geometries instead of silently changing inputs. The original executed script snapshots remain in the reference/eval
roots; the later optional --cohort replay entry adds no change to equations or
reference numerics. Use `.venv`; these solvers are Gmsh/SciPy, not FEniCS.

## Session changes

Added `docs/tasks/particle_reference64.py` and `pino_checkpoint_eval64.py`.
Updated the comparison renderer, two provenance receipts, both example READMEs,
root README and assets guide; replaced two convergence PNGs and the particle
field PNG. Original Electro field PNG remains byte-identical. Memory:
`docs/DECISIONS.md`, this archived task and `docs/PROGRESS.md`. No other pending
work is reverted, and no Git/history operation is authorized or performed.
