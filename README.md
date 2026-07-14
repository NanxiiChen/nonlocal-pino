# PINN for Non-local KKS Phase-Field Corrosion Model (1D & 2D)

## Overview

Physics-Informed Neural Network (PINN) for the Kim-Kim-Suzuki (KKS) phase-field
pitting corrosion model. Spatial derivatives use a **non-local peridynamic operator
L_δ** (precomputed D matrix), avoiding spatial autograd. Only time derivatives go
through automatic differentiation.

The project covers **1D and 2D** geometries with FEM reference solutions for
validation. Two physical regimes: **activation-controlled** (L=1e-11, fast kinetics)
and **diffusion-controlled** (L=2.0, slow kinetics).

## Architecture

### Network types

| Architecture | Description | Status |
|---|---|---|
| Full MLP | `net(x, t)` → (φ, c). Fourier embeddings on x and t. | 1D ✓, 2D partial |
| Split-PINN | `F_x(x) · F_t(t)` factorized. O(N_x+N_t) cost. | 1D only |
| Modified MLP | MLP with residual connections. | 1D only |

Default: Full MLP with 4 hidden layers × 64 neurons, GELU activation, Fourier
embeddings (64-dim for both spatial and temporal).

### Non-local operator D

Precomputed once (not trained). Replaces local Laplacian Δ:

```
D @ u ≈ -Δu    (as δ → 0)
```

**Factory API**: `build_D(x, delta, method=None, kernel="cubic")`

| method | Implementation | Memory | Best for |
|--------|---------------|--------|----------|
| `"dense"` | `DenseD` — (N,N) jnp array | O(N²) | N < 2000 |
| `"sparse"` | `SparseD` — neighbor table + scatter-add | O(N·k) | N ≥ 2000 |
| `None` (auto) | dense if N<2000 else sparse | auto | default |

**Kernels**:

| Kernel | J(z) | Dim | Normalization |
|--------|------|-----|---------------|
| cubic | J₀·(1-|z|/δ)² | 1D | J₀ = 30/δ³ |
| cubic | J₀·(1-|z|/δ)² | 2D | J₀ = 120/(πδ⁴) |
| tophat | J₀ | 1D/2D | J₀ = 3/δ³, 8/(πδ⁴) |

### Output constraint (HardConstraint)

Enforces thermodynamic consistency at the output layer:

```
φ = 0.5·tanh(raw[0]) + 0.5          → φ ∈ [0, 1]
c = h(φ)·c_S + (1-h(φ))·c_L          → KKS mixture rule
```

with `h(φ) = -2φ³ + 3φ²`, equal chemical potential `c_S - c_L = Δc₀`.

### Loss function (4-term with GradNorm)

```
Total = w_ac·L_ac + w_ch·L_ch + w_bc·L_bc + w_ic·L_ic
```

**PDE residuals** (non-dimensional, non-local):

Allen-Cahn (φ):
```
r_φ = ∂φ/∂t* + N_AC3·D(φ) + N_AC2·g'(φ) - N_AC1·(c - h(φ)·Δc₀ - c_le)·Δc₀·h'(φ)
```

Cahn-Hilliard (c):
```
r_c = ∂c/∂t* + N_CH·D(c - h(φ)·Δc₀ - c_le)
```

**GradNorm**: per-iteration adaptive weights based on L1 gradient norms, EMA
smoothed. 4 loss components: AC, CH, BC (Dirichlet), IC.

**L1 loss** (default): `mean(|r|)` — gradient is `sign(r)·∂r/∂θ`, robust to
gradient imbalance between PDE terms (AC spatial terms ~10⁸× larger than temporal).

### Non-dimensional parameters

```
N_CH  = D·T_c/L_c²        N_AC1 = 2A·L·T_c
N_AC2 = L·ω_φ·T_c          N_AC3 = L·α_φ·T_c/L_c²
```

## Codebase structure

```
src/pinn/
  operators.py            # DenseD, SparseD, build_D() factory
  causal.py               # CausalWeightor (temporal chunked weighting)
  optimizers.py           # Adam / SOAP optimizer factory
  arch/
    pinn.py               # Full MLP PINN with Fourier embeddings
    split_pinn.py         # Split-PINN (DeepONet-style)
    film_pinn.py          # FiLM-conditioned PINN
    hard_constraint.py    # KKS thermodynamic output constraint
    dirichlet_bc.py       # Hard Dirichlet BC layer
    embedding.py          # FourierEmbedding
    mlp.py / modified_mlp.py

examples/corrosion1d/
  solver.py               # 1D FEM reference solver
  losses_full.py          # Full MLP loss functions
  losses.py               # Split-PINN loss functions
  train.py                # 1D training script
  configs/
    train_diffusion.py    # Diffusion-controlled config
    train_activation.py   # Activation-controlled config

examples/corrosion2d/
  solver.py               # 2D FEM reference solver (MPI, FEniCS)
  losses_full.py          # 2D Full MLP loss functions
  train.py                # 2D training script
  configs/
    solver_diffusion.py   # 2D FEM config
    train_diffusion.py    # 2D PINN training config

reports/                  # Analysis and investigation reports
```

## Environment setup

```bash
# FEniCS FEM solver (conda)
conda activate fenicsenv
mpirun -np 4 python examples/corrosion2d/solver.py

# JAX PINN training (venv)
source .venv/bin/activate
python examples/corrosion1d/train.py --config=examples/corrosion1d/configs/train_diffusion.py
python examples/corrosion2d/train.py --config=examples/corrosion2d/configs/train_diffusion.py

# IMPORTANT: deactivate one env before activating the other
# conda and venv have conflicting library paths
```

## Current issues

### 1. 1D: Full MLP converges, Split-PINN hits static interface

- **Full MLP**: With L1 loss + HardConstraint + GradNorm, full MLP reliably converges
  for both activation-controlled and diffusion-controlled regimes.
- **Split-PINN**: The factorized architecture (F_x·F_t) can learn the spatial IC
  well but the temporal branch becomes irrelevant — all time steps output nearly
  identical profiles. The spatial basis can represent any time step equally, so
  the optimizer finds no pressure to evolve the interface.

### 2. 2D: Static interface on all architectures

Both full MLP and Split-PINN produce a **static interface** — φ and c fields
show almost no time evolution, regardless of architecture or loss function
(L1/L2). This persists even with:
- Causal training (chunked temporal weighting)
- Increased IC bias / IC oversampling near pit interface
- Neumann BC on solid bottom boundary
- SOAP optimizer

**What's been ruled out**:
- Non-local operator normalization bug (fixed: cubic kernel was using 1D normalization in 2D, dV was incorrect for non-unit domains)
- PDE residual sign errors (verified against 1D/2D FEM weak forms)
- Dense matmul float32 precision (sparse D gives identical results)
- L1 vs L2 loss function (both fail equally)

**Current hypothesis**: The 2D geometry has a degenerate loss landscape where a
static interface with correct width is a deep local minimum. The interface only
occupies ~7% of spatial points; the bulk dominates the mean loss. The network
can satisfy the PDE almost everywhere without evolving the interface.

See `reports/static-interface-2d-2026-06-25.md` for detailed analysis.

## Performance

See `benchmarks/RESULTS.md` for measured numbers (B200).  Highlights:

- `nl.method="conv"` (ConvD): the non-local operator as a stencil convolution
  on structured grids — exact same weights/sums as sparse (verified to fp32
  round-off), 2.6x faster matvec, ~8% faster 2D steps.  Default for 2D.
- `arch.tensor_split` (mlp trunk): first layer split into x-only + t-only
  matmuls, hoisted out of the vmap(x)xvmap(t) tensor product — exact same
  math, 1.6-2x faster PDE loss+grad at FP32-highest.  FiLM nets already get
  this implicitly; ~neutral under TF32.  Off by default.
- `train.matmul_precision`: default "highest" (exact FP32).  TF32 measured
  ~1.2-1.5x faster with matching loss trajectories, but is opt-in only.
- `benchmarks/verify_equivalence.py` gates all of the above against the
  reference implementation (losses, gradients, GradNorm weights).

## Key design decisions

1. **Non-local over local**: D matrix avoids expensive spatial autograd (jacrev
   over 20K spatial points). O(N·k) matvec for sparse D vs O(N²) for dense.
2. **L1 over L2**: `sign(r)` eliminates 10⁸× gradient imbalance between AC spatial
   and temporal terms.
3. **GradNorm over fixed weights**: Adaptive per-loss weighting; no manual tuning.
4. **HardConstraint over raw output**: Enforces φ∈[0,1] and thermodynamic c-φ
   relationship at the architecture level, simplifying optimization.
5. **Soft BC over hard BC**: Boundary conditions enforced via loss terms, not
   architectural constraints.
