# PINN for Non-local KKS Phase-Field Corrosion Model

## Overview

We solve a 1D phase-field pitting corrosion model (Kim-Kim-Suzuki / KKS) using a Physics-Informed Neural Network (PINN). The twist: instead of the standard local PDE with Laplacian operators, we use a **non-local peridynamic formulation** where all spatial derivatives Δ are replaced by the integral operator L_δ. This eliminates spatial autograd entirely — only first-order time derivatives go through AD.

The non-local operator is precomputed as a dense matrix D such that `D @ u ≈ L_δ u`, using a top-hat or cubic kernel with finite horizon δ.

## Architecture: Split-PINN (DeepONet-style)

We use a split architecture where spatial and temporal dependencies are factorized:

```
Spatial branch:  x (dim,) → FourierEmbedding(scale=2.0) → MLP(64→64→...→2p)
Temporal branch: t ()      → FourierEmbedding(scale=2.0) → MLP(64→64→...→2p)

φ(x,t) = dot(F_x[:p], F_t[:p])        c(x,t) = dot(F_x[p:], F_t[p:])
```

- **rank p=50**: spatial + temporal basis functions per field (SVD shows rank 5 captures >99.9%)
- **Activation**: configurable (tanh/gelu/silu). Currently using gelu.
- **Motivation**: O(N_x + N_t) forward cost vs O(N_x·N_t) for full MLP (~100x speedup for 1000×100 grids)

Time derivative of fields comes from differentiating only the temporal branch:
```
∂φ/∂t = F_x @ jacobian(F_t, t)
```
No spatial autograd needed.

## Non-local operator D

Precomputed once (not trained). Two kernel options:

| Kernel | J(z) ∝ | Continuity | Notes |
|--------|--------|------------|-------|
| tophat | 1 | C⁻¹ | Simplest, causes boundary oscillations |
| cubic | (1-|z|/δ)² | C¹ | Smoother, better behaved |

Normalized so that ∫ J(z)·|z|² dz = 2d, guaranteeing L_δ → -Δ as δ→0.

D matrix: dense for N_x ≤ 2000, conv-based _LDelta class for larger grids. Currently using dense with N_x=1001.

## Loss function (4-term with GradNorm)

```
Total = w_ac·L_ac + w_ch·L_ch + w_bc·L_bc + w_ic·L_ic
```

### PDE residuals (non-dimensional, non-local)

**Allen-Cahn (φ):**
```
r_φ = ∂φ/∂t* + N_AC3·Lδ(φ) + N_AC2·g'(φ)
    - N_AC1·(c - h(φ)·Δc₀ - c_le)·Δc₀·h'(φ)
```

**Cahn-Hilliard (c):**
```
r_c = ∂c/∂t* + N_CH·Lδ(c - h(φ)·Δc₀ - c_le)
```

where `h(φ) = -2φ³ + 3φ²`, `g'(φ) = 4φ³ - 6φ² + 2φ`, `Δc₀ = c_se - c_le`.

Non-dimensional parameters (matching FEM reference):
- `N_CH = D·T_c/L_c² ≈ 8.5e3`
- `N_AC1 = 2A·L·T_c ≈ 1.07e2`
- `N_AC2 = L·ω_φ·T_c ≈ 1.76e1`
- `N_AC3 = L·α_φ·T_c/L_c² ≈ 1.03e-2`

Pre-scaling: `L_ac` and `L_ch` can be multiplied by fixed factors to bring them to O(1). Currently `ac_scale = 1e-2`, `ch_scale = 1/N_CH² ≈ 1.4e-8`.

### Boundary conditions

- **Neumann** (no-flux): naturally satisfied by horizon truncation of D near ∂Ω
- **Dirichlet** (φ=1,c=1 at x=0; φ=0,c=0 at x=1): only the two endpoint collocation points are excluded from MSE (mask_free). The PDE residual at nearby interior points couples to the boundary values through D @ u. No hard enforcement inside D — the network learns BC values implicitly through PDE physics.
- **BC loss** (`L_bc`): soft penalty term penalizing endpoint drift at all times

### GradNorm weighting

Per-iteration adaptive weights based on gradient norms:
```
weight_i = mean(||grad_j||) / ||grad_i||
```
Capped at 1000 to prevent IC weight from exploding. EMA smoothed (α=1.0 = no smoothing currently).

## Current issue: temporal branch freezing

**Symptom**: All time steps output nearly identical spatial profiles (the IC). PDE losses decrease, IC loss → 0, but the interface does not move over time.

**Diagnosis (working hypothesis)**: The spatial branch is trained by 4 loss terms while the temporal branch only receives gradients through `∂φ/∂t` and `∂c/∂t`. The spatial branch quickly learns to represent the IC well (rank ~21 at init, potentially collapses to lower effective rank during training). Any linear combination of similar spatial basis functions produces the same profile, making the temporal branch irrelevant to the loss.

**Attempted fixes so far:**
- Gelu activation (replaced tanh): no improvement
- BC loss term: prevents endpoint drift, doesn't fix temporal
- GradNorm weight cap at 1000: prevents IC domination, but temporal still freezes

**Proposed fix**: Two-phase training: Phase 1 (100 iters) IC-only to learn spatial basis; Phase 2 (900 iters) PDE-only to force temporal evolution.

## File structure

```
src/pinn/
  operators.py          # build_D, DWithBC, L_δ operator
  arch/
    embedding.py        # FourierEmbedding
    mlp.py              # MLP
    modified_mlp.py     # ModifiedMLP
    split_pinn.py       # Split-PINN architecture
examples/corrosion1d_activation/
  configs/
    solver.py           # FEM config
    train_pinn.py       # PINN training config
  fem.py                # FEM reference solver
  losses.py             # Loss functions + GradNorm
  train.py              # Training loop + eval + TensorBoard logging
```

## Run

```bash
# FEM reference solution
conda activate fenicsenv
python examples/corrosion1d_activation/fem.py

# PINN training
source .venv/bin/activate
python examples/corrosion1d_activation/train.py --config.train.iterations=1000
```
