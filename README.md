# PINN for Nonlocal Phase-Field Models

Physics-Informed Neural Networks with a **nonlocal peridynamic operator $D_\delta$**
that replaces the local Laplacian.  $D_\delta \approx -\nabla^2$ (as $\delta \to 0$)
with compact kernel support; Neumann BCs emerge automatically from interior-only
neighbour pairs — no BC loss terms needed.

## Examples

| Example | Physics | Highlights |
|---|---|---|
| `icemelting3d` | Allen-Cahn, ice sphere melting | Analytical solution, nonlocal vs local AD ablation |
| `electropolish2d` | KKS phase-field, electropolishing | 2-field AC+CH, rough interface, Neumann BC advantage |
| `fracture1d` | AT2 phase-field fracture | Analytical crack profile, staggered incremental loading |
| `fracture2d` | Rounded-notch AT2 phase-field fracture | Local P1 mechanics, nonlocal AT2 PINN/direct damage, FP64 loading |
| `fracture2d_shear` | Rounded-notch AT2 under horizontal shear | Material-only mesh, direct edge clamps, local FEM-energy history-conditioned PINN and matched FEM reference |
| `elasticity2d` | 2D elasticity | Nonlocal elasticity and boundary-aware operators |
| `pino_elasticity2d` | 2D elasticity PINN | Geometry-aware PINN reference for nonlocal elasticity |
| `corrosion1d/2d` | KKS pitting corrosion | Legacy; superseded by electropolish2d |

当前 fracture2d 从零求解的阶段结果由用户暂且接受（2026-10-14）：已保存 `0 -> 0.0057`，在 `0.0053` 出现贯通裂纹，反力与同荷载 direct reference 接近。损伤场精度仍需改善，主干代码已合并整理。本阶段已于 2026-10-15 收尾；结果、已知限制和后台复跑交接见[阶段归档](docs/tasks/archive/fracture2d-nonlocal-pinn.md)。下一步正在评估[点云网络架构](docs/tasks/nonlocal-pinn-architecture.md)，概览见[项目进展](docs/PROGRESS.md)。历史实验 `_legacy/` 与阶段报告 `docs/reports/` 仅在本地保留，已忽略且不纳入 Git 追踪。

Fracture2D now has two main entries: `examples/fracture2d/train.py` for
local P1 direct mechanics/nonlocal damage PINN, and `nonlocal_baseline.py` for
the same model with direct damage. Both use ml_collections configuration and
one standard stagger/loading loop. Defaults preserve the accepted full-node
L1/Adam/sigmoid/fine-load strategy; optional norms, exact core+halo sampling,
optimizers and heads are documented in the [example README](examples/fracture2d/README.md).
`configs/train_core_halo.py` selects a fixed 2048-batch random pool, padded to
its exact largest complete support, with TensorBoard run name `pinn_core+halo`.
The optional core+halo preset includes segment/tip boosts18/36 and switches
batches every Adam update. Full-domain sampling with damage-change tolerance
1e-5 remains the default. In the measured fracture2d comparisons, cheaper
random updates required more iterations, so total solve time did not improve.
One-off density/batch/tolerance studies are kept in `_legacy/`.
Fracture2D artifacts use `/root/autodl-tmp/tf-logs/fracture2d/`: accepted
`pinn_mlp_baseline/` and `pinn_history_conditioned_baseline/` each contain their
TensorBoard `events/`; historical experiments live in `diagnostics/`.
Current local FEM and matched local-mechanics/nonlocal-damage direct references
are named `/root/autodl-tmp/data/fracture2d/fem_reference/` and
`nonlocal_reference/`. Their engineering acceptance and remaining convergence
limits are documented in the artifact index and active task.
An optional `history_probe.py` evaluates a point-patch encoder with coordinate
cross-attention and fixed-shape history replay; it is a bounded frozen-field
experiment, separate from the accepted coupled loading route.
`history_conditioning.py` provides an experimental normalized-history encoder
and query-local history branch for controlled frozen-field comparisons.
`train.py --config=examples/fracture2d/configs/train_history.py` runs the optional
history operator from physical zero, accumulating only its own load-start/end
history snapshots. The accepted MLP remains the default.
`configs/train_history_fem.py` optionally replaces damage derivatives with
material-P1 shape gradients and trains the history-conditioned network by
lumped FE energy; `configs/fem_damage_reference.py` selects its matched direct
control through `nonlocal_baseline.py`. This local damage discretization is an
experimental alternative; the default remains nonlocal. See the example README.
`configs/train_history_fem_full.py` and `configs/fem_damage_reference_full.py`
extend this experimental configuration to a separate zero-to-0.006 trajectory;
compute the matched direct reference first.
`configs/train_history_fem_cold.py` tests a fresh cold path with128outer
blocks and budget-triggered Adam rate lowering, retaining moments acrossloads.
`configs/train_history_fem_cold_scaled.py` selects the accepted cold baseline
with damage-scaled stopping and progress-aware128outer checks; see the example
README for its explicit accuracy/budget tradeoffs. Its completed trajectory
and TensorBoard events are under
`/root/autodl-tmp/tf-logs/fracture2d/pinn_history_fem_scaled_stop_baseline/`.
`configs/train_history_fem_replay.py` selects an isolated cold replay trial:
five distinct H sampled from all completed loads, fixed task shapes and the
same FEM-energy/stagger scheme. Its coordinate query features are shared
across histories within each update. The accepted baseline stays replay-free.
`configs/train_history_conditioned.py` selects the experimental normalized/local
history model with a one-time L1-to-L2 bootstrap and later L2 fits for a short
from-zero loading pilot; replay training is disabled in this profile.
`examples/fracture2d/plot_sampling.py` plots the configured sampling density,
a reproducible core+halo batch, notch-tip zoom and probability profiles.
Historical nonlocal-mechanics experiments and complex diagnostics are preserved
in local ignored `_legacy/` directories. Public solver/configuration/test
sources remain tracked and do not require these archives.

## Core idea

Standard PINN: $\nabla^2\phi$ via `jax.hessian` on every collocation point — expensive
and requires explicit BC penalty terms.

Nonlocal PINN: precompute a sparse $D$ matrix from neighbour pairs on a fixed grid,
then $D \cdot \phi$ is a cheap matvec.  Points on Neumann boundaries only interact
with interior neighbours; the zero-flux condition is built into the operator.

```
D @ φ ≈ -∇²φ    (kernel: cubic / quartic / Wendland)
```

## Key design decisions

- **Nonlocal $D_\delta$ over autograd Hessian**: avoids per-point Hessian overhead
  and handles Neumann BCs implicitly
- **GradNorm over fixed weights**: per-term adaptive loss weights from gradient norms,
  EMA-smoothed
- **L1 loss** (MAE) for PDE residuals: $\nabla |r|$ eliminates gradient imbalance
  between stiff spatial terms and small temporal terms
- **Grouped separate gradients** (regular step): PDE terms sharing `_compute_all`
  stay in one backward pass; BC and IC get their own.  Important for `tensor_split`
  performance (see `benchmarks/TENSOR_SPLIT_RTX_PRO_6000.md`)
- **Hard constraint** on output: enforces $\phi \in [0,1]$ and KKS mixture rule at
  the architecture level

## Scaling

Most JAX/nonlocal examples use non-dimensional characteristic length and time
scales. The `fracture2d` local FEM reference instead uses explicit consistent
material units with `L0=1`; see its example README and configuration.
Its FEM mesh controls allow `0 < mesh.lc_near <= mesh.lc_far`, including
uniform-resolution reference studies.
The fracture2d direct nonlocal baseline and current PINN research keep geometry,
quadrature, graphs, and operator inputs in float64 with JAX x64 enabled. A float32 PINN must
rebuild and revalidate its strict-horizon graph in float32 rather than casting
the direct-baseline graph.

## Fracture2D archived experiments

Historical solvers, one-off sampling diagnostics and experimental presets may
be kept locally under `examples/fracture2d/_legacy/`; dated reports may be kept
under `docs/reports/`. Both are ignored and untracked. Current use is documented
in the [example README](examples/fracture2d/README.md); current work records
remain in the tracked `docs/PROGRESS.md`, `docs/DECISIONS.md` and `docs/tasks/`.

## Environment

```bash
# FEniCS FEM solver
conda activate fenicsenv
mpirun --allow-run-as-root -np 12 python examples/icemelting3d/solver.py

# JAX PINN training
source .venv/bin/activate
python examples/icemelting3d/train.py --config=examples/icemelting3d/configs/train_base.py

# IMPORTANT: deactivate one env before activating the other
```

## Source structure

```
src/pinn/
  operators.py        # SparseD, DenseD, ExplicitVolumeSparseD, build_D()
  fd.py               # Finite-difference Laplacian (ablation)
  causal.py           # CausalWeightor (per-time-step weighting)
  optimizers.py       # SOAP / Adam wrapper
  arch.py              # MLP, ModifiedMLP, and FiLM PINN trunks
  embedding.py        # Learnable Fourier feature embeddings
  constraints.py      # Hard output and KKS constraints
  mesh.py             # Gmsh node reader
  utils.py             # Shared training and logging utilities

benchmarks/
  RESULTS.md                      # B200 measurements
  TENSOR_SPLIT_RTX_PRO_6000.md    # RTX PRO 6000: gradient grouping analysis
  bench_step.py, bench_tensor_split.py, bench_operator.py
  verify_equivalence.py           # Correctness gate for optimizations
```
