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

## Non-dimensionalisation

All examples work in non-dimensional form.  Characteristic length $L_c$ and time
$T_c$ define non-dimensional PDE parameters (e.g. $N_1 = MT_c/L_c^2$, etc.).
Hyperparameter transfer between problems is a matter of matching non-dimensional
numbers, not physical units.

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
  operators.py        # SparseD, DenseD, build_D() factory
  fd.py               # Finite-difference Laplacian (ablation)
  causal.py           # CausalWeightor (per-time-step weighting)
  optimizers.py       # SOAP / Adam wrapper
  arch/
    pinn.py           # MLP + FiLM PINN with Fourier embeddings
    hard_constraint.py
    mlp.py, modified_mlp.py, film_pinn.py, split_pinn.py

benchmarks/
  RESULTS.md                      # B200 measurements
  TENSOR_SPLIT_RTX_PRO_6000.md    # RTX PRO 6000: gradient grouping analysis
  bench_step.py, bench_tensor_split.py, bench_operator.py
  verify_equivalence.py           # Correctness gate for optimizations
```
