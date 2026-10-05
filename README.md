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
| `fracture2d` | Rounded-notch AT2 phase-field fracture | Frozen direct baseline, independent-horizon validation, FP64 PINN loading |
| `elasticity2d` | 2D elasticity | Nonlocal elasticity and boundary-aware operators |
| `pino_elasticity2d` | 2D elasticity PINN | Geometry-aware PINN reference for nonlocal elasticity |
| `corrosion1d/2d` | KKS pitting corrosion | Legacy; superseded by electropolish2d |

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
The fracture2d direct nonlocal baseline and current PINN research keep geometry,
quadrature, graphs, and operator inputs in float64 with JAX x64 enabled. A float32 PINN must
rebuild and revalidate its strict-horizon graph in float32 rather than casting
the direct-baseline graph.

## Fracture2D current entries

`examples/fracture2d/nonlocal_baseline.py` is the frozen GPU direct baseline.
The current user-selected PINN mainline preserves its original common-horizon
kernel and refines numerical coupling through `legacy_coupling_refine.py`.
`legacy_coupled_loading.py` advances one load from a SHA-verified diagnostic
predecessor only after rechecking three coupled-readiness confirmations.
An explicit `--warning-continue` policy supports diagnostic trend studies with
uncoupled ancestry disclosed and the original physical/mechanics hard gates.
`subproblem_matrix.py` compares direct/neural mechanics and damage at identical
stored history floors to diagnose inner accuracy versus outer coupling error.
`direct_path_probe.py` tightens the common-floor direct benchmark and compares
convergence-gated sequential loading step sizes without changing the kernel.
`objective_pair_probe.py` audits subproblem energy gradients and coupled mixed
derivatives, then compares fixed-d neural force-residual and original A2 energy
objectives from identical networks with fresh Adam; direct fields are comparisons.
The independent-horizon research branch uses `split_reference.py` for strict
comparison states and `split_pinn.py` for FP64 neural loading. The PINN starts
from zero and uses its own learned predecessor history; reference fields enter
comparisons only. Damage-field RMSE and reaction error lead the research report;
other unmet checks retain their values, limits and warning provenance. Warning
advancement is not strict field-equation validation. See the
[example README](examples/fracture2d/README.md) for commands and the
[active task](docs/tasks/fracture2d-nonlocal-pinn.md) for current results.
The optional `configs/split_pinn_fast.py` continues a verified learned prefix
with measured operator optimizations and a reduced small-load update budget;
the example README documents its scope and the paired quality/runtime checks.

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
