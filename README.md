# nonlocal-pinn

Physics-only neural operators and PINNs for phase-field dynamics, using nonlocal
spatial operators and shared spatial/time network evaluation. The four main
PINO examples learn a family of solutions from known initial fields or geometry;
**positive-time reference solutions are used only for evaluation**.

Private repository: [NanxiiChen/nonlocal-pinn](https://github.com/NanxiiChen/nonlocal-pinn).

## Physics-only neural-operator defaults

| Main example | What varies between samples | Physical model / horizon | Representative phase relative L2 | Reference / evaluation scope |
|---|---|---|---:|---|
| [Ice melting 3D](examples/icemelting3d/README.md#pino-overview) | Initial ellipsoid axes, center and rotation | Driven Allen–Cahn, 5 s |0.227% pooled; 0.092% terminal | Native nonlocal, 8 held-out inputs; one separate local-FEM check |
| [Electropolish 2D](examples/electropolish2d/README.md#pino-overview) | Initial rough metal/electrolyte interface | Coupled KKS AC/CH, 5 s |4.427% pooled; 3.176% terminal | Independent local FD, 64 unseen inputs |
| [Spinodal 2D](examples/confined_ch2d/README.md#pino-overview) | Positions and radii of two impermeable holes | Confined Cahn–Hilliard, T=3 |4.833% pooled; 5.193% terminal | Native nonlocal, 8 reused reserved development geometries |
| [Two-phase flow 2D](examples/channel_nsch2d/README.md#pino-overview) | Independent smooth upper/lower channel profiles | Fully coupled incompressible NS–CH, T=6 |1.840% pooled; 2.271% terminal | Independent local FEM, 3 reused reserved geometries |

These are means of per-sample errors from the selected final checkpoints, not
one common benchmark. Targets, quadratures and evaluation populations differ;
the example READMEs give the definitions, auxiliary-field errors, limitations,
direct checkpoints and **3–4 sample evolution figures**. Small reused geometry
cohorts do not establish population or cross-seed accuracy. Spinodal remains
in the existing `confined_ch2d` directory; the full NS–CH operator is
`channel_nsch2d`, distinct from the prescribed-flow `twophase2d` pilot.

| Example | Condition encoder and queries | Selected training strategy |
|---|---|---|
| Melting |3D initial volume -> patch tokens -> query cross-attention | Adam 5000, B16; hard IC; tanh latent residual; fixed core+halo pool |
| Electropolish |2D initial image -> patch tokens -> query cross-attention | Adam 5000, B8; original AC/CH + soft IC/BC, five balanced losses; fixed core+halo pool |
| Spinodal | Boundary and domain-function point streams -> normalized cross-attention | SOAP 50000, width 256, B16; online rolling geometry pool; hard IC/mass; eliminated chemical potential and resolvent residual |
| Two-phase flow | Boundary and domain-function point streams -> normalized cross-attention | SOAP 10000, width 192, B8; online rolling geometry pool; hard flow/initial data; integral spatial derivatives, time JVP and fixed residual normalization |

Regular domains use patch tokens; irregular geometry uses physical point sets
with boundary arc-length/domain-area measures. Queries carry coordinates/time,
without query-local initial-field features. Each operator prepares spatial
features once and shares that work across times; temporal derivatives use JVP.
Online initial fields are fresh continuous draws. Geometry pools roll in newly
constructed **operators/known inputs**, without solving training trajectories.

Run from the repository root, one GPU job at a time:

```bash
source .venv/bin/activate
python -m examples.icemelting3d.train --config=examples/icemelting3d/configs/train_operator.py
python -m examples.electropolish2d.train --config=examples/electropolish2d/configs/train_operator.py
python -m examples.confined_ch2d.train --config=examples/confined_ch2d/configs/train_operator.py
XLA_PYTHON_CLIENT_MEM_FRACTION=0.97 python -m examples.channel_nsch2d.train
```

The channel default uses the qualified96 GB GPU preset (~88 GiB live memory).
Its README supplies explicit block 8/block 2 alternatives (~40/~19 GiB) for
smaller cards. The other three commands explicitly select PINO; their
no-argument `train.py` entries retain single-IVP PINN defaults. All launchers
create fresh runs. Existing evaluation banks and direct artifact paths must be
available; reference preparation commands are in each example.

Seeds are runtime provenance through `--seed`, not tuned strategy fields.
Defaults use ordinary compilation and reject a conflicting inherited
`--xla_gpu_deterministic_ops=true`. Saved configurations govern old checkpoint
replay; checkpoints/references load by direct paths, without file hashes.

## Other examples and single-problem PINNs

- [Corrosion2D](examples/corrosion2d/README.md): KKS pit corrosion, matched-graph/local-AD controls.
- [Corrosion1D](examples/corrosion1d/README.md): Earlier one-dimensional implementation.
- [Elasticity2D](examples/elasticity2d/README.md): Nonlocal elasticity with independent FEM.
- [Precipitate2D](examples/precipitate2d/README.md): Periodic coherent elastic precipitation with mechanical equilibration.
- [Two-phase prescribed flow](examples/twophase2d/README.md): Periodic CH transport/deformation with prescribed velocity, without coupled NS.
- The four main example directories also retain their original single-problem
  PINNs and research controls. In particular, confined CH's no-argument PINN
  uses 30 time windows; its geometry PINO uses one global-time operator.
- [Elasticity PINO design](docs/design/pino_elasticity2d.md) remains a design document.

Single-problem defaults, seed checks and local/nonlocal controls are documented
in the corresponding READMEs. Their error values should not be substituted for
operator errors over varying inputs.

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
assets/pino/              # four compact README evolution figures
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

Generated research galleries, videos, experiment arrays/checkpoints and compiler
dumps under `docs/tasks` are **local artifacts, not Git content**. Git keeps
source, Markdown, configs, metrics and regeneration scripts. The four compact
`assets/pino/*.png` figures are the deliberate documentation
exception: one representative multi-sample image per main operator, collected
in [assets](assets/README.md). Their direct
inputs, sample identities and regeneration command are recorded in
[figure provenance](artifacts/pino_readme_figures.json); raw arrays/weights stay
external. Figure links in
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
