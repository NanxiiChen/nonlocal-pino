# nonlocal-pinn

Private GitHub repository: https://github.com/NanxiiChen/nonlocal-pinn

Nonlocal spatial operators and physics-informed neural networks for melting,
electropolishing/corrosion and elasticity. Local AD/FD controls and independent
FEM references remain part of the validation workflow.

## Examples

- [icemelting3d](examples/icemelting3d/README.md): Allen–Cahn melting, nonlocal/local comparison.
- [electropolish2d](examples/electropolish2d/README.md): Primary corrosion example: KKS electropolishing.
- [elasticity2d](examples/elasticity2d/README.md): Nonlocal elasticity with an independent FEM reference.
- [corrosion1d](examples/corrosion1d/README.md): Earlier implementation, retained for reference.
- [corrosion2d](examples/corrosion2d/README.md): KKS pit corrosion with a validated matched-graph training preset.

The unimplemented [PINO elasticity design](docs/design/pino_elasticity2d.md) is
retained as a design document. Electropolish2d is the primary corrosion case;
corrosion2d supplies the pit-corrosion case; corrosion1d is retained for reference.

## Current phase-field strategies and recorded best results

All three canonical training commands load `configs/train_base.py`. SOTA here
records the best verified final5000-update physical-field observation in the
recorded studies. Seed is not a strategy parameter: each canonical config
omits it; `--seed` controls a run (common untuned default42). Saved run config
metadata retains the resolved seed for replay. Single-run SOTA is not a
cross-seed guarantee.
KKS cases are ranked by terminal phi+c relative L2, melting by terminal phi
against the same-graph numerical reference. The KKS winners have independently
checked checkpoints; transport and seed limitations remain in their READMEs.

| Case | Default strategy | Time | Terminal phi / c relative L2 |
|---|---|---:|---:|
| [corrosion2d](examples/corrosion2d/README.md#sota-default) | Bounded phase residual, eta0.1; raw KKS CH edges; s_bc; nodal data | 50 s | 1.43305e-3 / 1.71859e-3 |
| [electropolish2d](examples/electropolish2d/README.md#sota-default) | Original residual; physical data losses; random IC/BC | 5 s | 1.13782e-3 / 1.35782e-3 |
| [icemelting3d](examples/icemelting3d/README.md#sota-default) | IC-only latent residual | 5 s | 6.10970e-5 / — |

```bash
source .venv/bin/activate
python -m examples.corrosion2d.train
python -m examples.electropolish2d.train
python -m examples.icemelting3d.train
```

Use `--seed=2026` to repeat a strategy with another random run. Historical
best values in the table describe individual observations, not default-run
guarantees; see each README for the full cross-seed results.

The [independent seed2026 validation](docs/tasks/archive/phasefield-seed-stability.md)
completed5000 updates per case: corrosion phi/c0.4070/0.4310 fails with extra
phase bands; electropolish0.002832/0.003427 succeeds but retains a historical
seed42 failure; melting full-grid phi5.99e-5 agrees with the earlier6.11e-5.
The corrosion best-single-run preset is therefore not a stable SOTA strategy.

These are separate GPU jobs; run them individually. All use SOAP, initial
lr1e-3 and5000 updates. New cold starts use fresh run directories; existing
checkpoints/reference arrays load by direct file path. Legacy configurations
remain explicit alternatives. [Artifact provenance](artifacts/index.yaml)
records the selected checkpoint/reference and exact metrics for each default.

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
docs/                     # rules, progress, decisions, tasks and migration
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
