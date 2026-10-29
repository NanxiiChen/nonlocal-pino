# nonlocal-pinn

Private GitHub repository: https://github.com/NanxiiChen/nonlocal-pinn

Nonlocal spatial operators and physics-informed neural networks for melting,
electropolishing/corrosion and elasticity. Local AD/FD controls and independent
FEM references remain part of the validation workflow.

## Examples

- [icemelting3d](examples/icemelting3d/README.md): Allen–Cahn melting, nonlocal/local comparison; an opt-in 3D conditional neural operator for continuously sampled random initial ellipsoids, with measured training memory and independent full-cube references.
- [electropolish2d](examples/electropolish2d/README.md): Primary corrosion example: KKS electropolishing; opt-in random-initial-interface neural operators with shared-space/time conditioning, fixed-input controls and continuous online sampling.
- [elasticity2d](examples/elasticity2d/README.md): Nonlocal elasticity with an independent FEM reference.
- [corrosion1d](examples/corrosion1d/README.md): Earlier implementation, retained for reference.
- [corrosion2d](examples/corrosion2d/README.md): KKS pit corrosion with a validated matched-graph training preset.
- [twophase2d](examples/twophase2d/README.md): Periodic prescribed-flow Cahn–Hilliard pilots: constant transport and reversed-vortex deformation, with mixed local-AD/nonlocal controls; no coupled Navier–Stokes yet.
- [precipitate2d](examples/precipitate2d/README.md): Periodic coherent elastic-precipitate pilot with differentiable mechanical equilibration and nonlocal phase diffusion.

- [confined_ch2d](examples/confined_ch2d/README.md): Confined CH in an ellipse with impermeable holes; default ten-wave T1.5 case uses30 sequential PINN time windows and a validated independent reference. Legacy simple/local AD and global-time research controls remain explicit options.

The unimplemented [PINO elasticity design](docs/design/pino_elasticity2d.md) is
retained as a design document. Electropolish2d is the primary corrosion case;
corrosion2d supplies the pit-corrosion case; corrosion1d is retained for reference.

## Current phase-field defaults

`examples/confined_ch2d/train.py` defaults to
`configs/train_coarsening_windows.py`:30 logarithmic time windows, phase-only
prediction with exact chemical elimination and hard mass, SOAP5000+L-BFGS500
total, initial learning rate1e-3, ordinary FP64. Run it without `--config`;
run directories are named automatically. Its reference solver selects the same
complex physical case by default. The legacy simple case remains explicitly
available through `--config=examples/confined_ch2d/configs/train_base.py`.

The three established phase-field `train.py` entries load `configs/train_base.py` and now
explicitly use **ordinary GPU compilation** (`train.gpu_policy="ordinary"`).
Precision is selected per case from complete5000-update controls, rather than
assuming FP64 always improves a training trajectory. All use SOAP, initial
lr1e-3 and5000 updates. Physics, reference, sampling, residuals and loss scales
are unchanged by this policy promotion.

| Case | Default strategy / arithmetic | Physical time | Terminal phi / c relative L2, seeds42 and2026 |
|---|---|---:|---:|
| [corrosion2d](examples/corrosion2d/README.md#sota-default) | Original phase; raw KKS CH edges; s_bc; nodal data; **FP64** | 50 s | 0.002499–0.006833 /0.002944–0.008077 |
| [electropolish2d](examples/electropolish2d/README.md#sota-default) | Original residual; physical data; random IC/BC; **FP32** | 5 s | 0.003399–0.004757 /0.004118–0.005738 |
| [icemelting3d](examples/icemelting3d/README.md#sota-default) | IC-only latent residual; **FP32** | 5 s | Full-grid graph phi3.9053e-5 (new42),5.9857e-5 (earlier2026) |

```bash
source .venv/bin/activate
python -m examples.corrosion2d.train
python -m examples.electropolish2d.train
python -m examples.icemelting3d.train
```

Run these GPU jobs individually. If your shell previously forced deterministic
operations, prefix the command with `XLA_FLAGS=''`; ordinary mode explicitly
rejects a conflicting inherited deterministic flag. Unrelated flags are retained.
Use `--config.train.gpu_policy=deterministic` for a deterministic control; matched
KKS additionally accepts `--config.train.compute_dtype=float32` or `float64`.
Old local-AD and ordinary-FP32 research presets retain their previous choices.

Seed is run provenance: canonical configs omit `train.seed`; `--seed=2026`
selects another run, while the common untuned default is42. Resolved saved
metadata retains it for direct checkpoint loading. Two seeds are evidence,
not statistical proof or a guarantee of low error on every run. New directories
are fresh; checkpoints and reference arrays load by explicit path without hashes.

The [completed default validation](docs/tasks/archive/ordinary-policy-defaults.md)
includes six new full cold starts, independent CPU field/transport/topology
checks and complete convergence curves. Electropolish ordinary FP64 fails seed42
(phi0.2998), so its default stays FP32. Corrosion FP64 improves final fields
and worst-seed late median, but late spikes and free-node Dq errors1.36–1.38
remain; it is a speed/field-accuracy compromise. Historical individual best
checkpoints remain separate from the representative current-default runs in the
[artifact index](artifacts/index.yaml).

The [isolated steady-update measurements](docs/tasks/archive/phasefield-gpu-policy-audit.md)
give electropolish ordinary FP3210.98ms versus deterministic FP32197.73ms,
and melting26.62ms versus468.26ms. The
[corrosion precision/policy benchmark](docs/tasks/archive/local-ad-performance.md)
gives ordinary FP6435.13ms versus deterministic FP32229.31ms. These include
sampling/SOAP/GradNorm but exclude first compilation, evaluation and saving;
full-run times have different compiler-cache histories. Elasticity and1D corrosion
were inspected statically, without full accuracy/default promotion.

## Physics-only neural-operator defaults

Both opt-in `configs/train_operator.py` presets use the currently recommended
measured strategy, with initial images as conditions and no evolved-field labels:

| Operator | Default configuration | Validation |
|---|---|---|
| 3D melting | Online B16, Adam5000, lr1e-3, ordinary FP32/highest, hard IC + latent PDE | Eight native tests; one local-FEM check, Adam pooled phi0.1479% |
| 2D electropolish | Online B8, Adam5000, lr1e-3, ordinary FP32/highest, image-only condition, original AC/CH + IC/BC | Independent64-input local-FD mean phi pooled4.427% |

```bash
python -m examples.icemelting3d.train --config=examples/icemelting3d/configs/train_operator.py
python -m examples.electropolish2d.train --config=examples/electropolish2d/configs/train_operator.py
```

Run individually. These are the recommended NO presets; no-argument `train.py`
still selects each established single-IVP PINN default. Melting retains Adam
for the current local-physics/cost compromise; SOAP's lower native pooled error
does not guarantee lower local-FEM error. Electropolish disables query-local IC
features while preserving actual IC losses. Historical finite/online/causal
controls remain explicit, and saved configs govern old checkpoint replay.
Selection is not a claim of a global optimum or statistical seed stability.

## Setup and use

```bash
uv sync --frozen
source .venv/bin/activate
python -m examples.electropolish2d.train --helpshort
python -m unittest discover -s tests -v
```

Use this project's own `.venv`. The lockfile preserves the source dependency
versions and CUDA13 JAX choice. CPU checks use `JAX_PLATFORMS=cpu`.
FEniCS reference/checking commands run separately in `fenicsenv` with
`PYTHONPATH=.:src`; do not combine the two environments. Examples use explicit
configuration files; review their external paths before launching a new run.

## Structure

```text
src/nonlocal_pinn/
  operators/ geometry/ nn/ training/ baselines/ io/
examples/                 # case geometry, BCs, configs, CLI and formulations
tests/                    # existing focused numerical/restart regressions
artifacts/index.yaml      # external artifact ownership, JSON-compatible YAML
scripts/check_artifacts.py
docs/                     # rules, progress, decisions, deferred issues, tasks and migration
```

Shared code lives in the package; examples keep case definitions and stable
config entry paths. The project has no runtime import from the other project
or the source checkout. FEM references remain independent of PINN operators;
the [coupled residual experiment](examples/corrosion2d/README.md) additionally
uses direct integration of the same graph to isolate PINN optimization error.

## Artifacts and recovery

Existing `/root/autodl-tmp/data/<example>` and `tf-logs/<example>` paths are
retained. TF-log directories also hold fields, checkpoints, optimizer/replay
state and experiment scripts. New experiments use separate named outputs;
authorized historical run renames are recorded in the artifact index.
The [artifact index](artifacts/index.yaml) records current ownership; check it
with `python scripts/check_artifacts.py`.

Generated research figures, videos, experiment arrays/checkpoints and compiler
dumps under `docs/tasks` are **local artifacts, not Git content**. Git keeps
source, Markdown, configs, metrics and regeneration scripts. Figure links in
archived reports refer to optional local files; they can be restored on this
host from the direct-path [local artifact manifest](artifacts/local_files.json):

```bash
python scripts/restore_local_artifacts.py --list --all
python scripts/restore_local_artifacts.py --restore --all
python scripts/check_repository_content.py
```

Existing local files remain at their original paths. Restoration checks sizes
and exact bytes, uses no file hashes and refuses to overwrite different local
files. On another host, regenerate from the recorded data/checkpoints or make
the external archive available. Small migration fixtures remain in Git. Ignoring
and untracking outputs limits future growth; old Git history is retained and
therefore is not reduced by this cleanup.

Old run identity includes source/input/config paths and replay-store paths.
Changed package layout is a new run identity: historical in-place resume is
not promised. Use the original checkout for historical continuation, or design
an explicit checkpoint import into an isolated new run. Do not bypass checks.

Source/recovery baseline: `/root/nonlocal-KKS`, `dev@61352f1`. Verified full
history and working-tree backup: `/root/nonlocal-KKS-split-backup-20261024`.
To abandon the split, return to the original repository and its environment;
no reset, deletion or data restoration is needed. See [migration/recovery notes](docs/migration/README.md).

## Research records

[Progress](docs/PROGRESS.md), [decisions](docs/DECISIONS.md),
[development](docs/DEVELOPMENT.md), [numerical rules](docs/NUMERICAL_METHODS.md)
and [JAX rules](docs/JAX_GUIDELINES.md). Original history remains available in
Git and the source checkout; the new dashboard does not duplicate old logs.

Mesh generation additionally uses the explicit `mesh` extra:
`uv sync --frozen --extra mesh` (Gmsh4.15.2, matching the baseline).
