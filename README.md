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
- [corrosion2d](examples/corrosion2d/README.md): Earlier implementation, retained for reference.

The unimplemented [PINO elasticity design](docs/design/pino_elasticity2d.md) is
retained as a design document. Electropolish2d is the primary corrosion case;
the earlier corrosion examples are preserved.

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
or the source checkout. Reference solvers remain independent of PINN operators.

## Artifacts and recovery

Existing `/root/autodl-tmp/data/<example>` and `tf-logs/<example>` paths are
retained. TF-log directories also hold fields, checkpoints, optimizer/replay
state and experiment scripts. Nothing has been moved, overwritten or rerun.
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
