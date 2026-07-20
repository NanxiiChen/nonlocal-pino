# PINO: Physics-Informed Neural Operator for Parameterised Elastic Plate Geometries

> **Status: Design Document.** No implementation exists yet. This describes the
> proposed architecture and design decisions. Last updated: July 2026.

## 1. Overview and Motivation

The single-geometry PINN in [`../elasticity2d/`](../elasticity2d/) solves a 2D
linear-elastic plate with **one fixed hole geometry** per training run.  If you
want to solve a *family* of geometries — different hole radii, different
positions — you would need to retrain from scratch each time.

A **PINO (Physics-Informed Neural Operator)** learns the solution **operator**
that maps geometry parameters directly to displacement and stress fields:

```
  geometry parameters θ               solution fields
  (hole radius a,                     (u_x(x,y), u_y(x,y),
   centre (cx, cy))          →        σ_xx, σ_yy, σ_xy)
```

A single training run covers the entire parameter space.  At inference time,
the network generalises to unseen geometries without retraining.

This document proposes a PINO design built on the ElasticD correspondence
peridynamics operator validated in `elasticity2d/`.

---

## 2. Problem Statement

Same physics as `elasticity2d/`: 2D linear elasticity, plane strain, static
equilibrium.  The **key difference** is that the geometry is parameterised:

| Parameter | Symbol | Range |
|-----------|--------|-------|
| Hole radius | \(a\) | \([a_{\min}, a_{\max}]\) |
| Hole centre x | \(c_x\) | \([c_{x,\min}, c_{x,\max}]\) |
| Hole centre y | \(c_y\) | \([c_{y,\min}, c_{y,\max}]\) |

Future extensions: multiple holes, elliptical holes, arbitrary cut-outs.

**Boundary conditions** (inherited from `elasticity2d/`):

| Boundary | Condition | Enforcement |
|----------|-----------|-------------|
| Left edge (\(x=0\)) | \(u_x = u_y = 0\) (clamped) | Hard constraint |
| Right edge (\(x=L_x\)) | \(u_x = u_{\text{load}}\), \(u_y = 0\) (displacement-controlled pull) | Hard constraint |
| Top, bottom, hole(s) | \(\boldsymbol{\sigma}\cdot\mathbf{n} = \mathbf{0}\) (traction-free) | ElasticD horizon truncation (automatic) |

**Nonlocal formulation**: ElasticD correspondence peridynamics (Silling et al.,
2007).  A single `elastic_matvec(u, lam, mu)` call computes the full
Navier-Cauchy residual.  Free boundaries are handled automatically by horizon
truncation — no explicit Neumann BC loss terms.

---

## 3. Design Space: Two Approaches

### 3.1 Option A: Mask with Weight-Zeroing (Fixed Grid + Shared COO)

Use a **fixed regular grid** covering the full rectangular domain.  Each
geometry is encoded as a binary mask \(m(\mathbf{x}) \in \{0, 1\}\) where
\(m = 0\) inside the hole and \(m = 1\) in the solid.

```
Full regular grid:         Mask for hole radius a₁:    Mask for hole radius a₂:
┌──────────────────┐       ┌──────────────────┐       ┌──────────────────┐
│··················│       │··················│       │··················│
│··················│       │·······○○○○○······│       │····○○○○○○○○○····│
│··················│       │·······○○○○○······│       │····○○○○○○○○○····│
│··················│       │·······○○○○○······│       │····○○○○○○○○○····│
│··················│       │··················│       │··················│
└──────────────────┘       └──────────────────┘       └──────────────────┘
  N_x × N_y points          · = solid (m=1)            ○ = hole (m=0)
```

**Core idea — share the COO sparsity, zero the weights**: Build the ElasticD
operator **once** on the full rectangle.  For each geometry, scale edge weights
by the product of endpoint masks.  Edges crossing into the hole get zero
weight but stay in the COO arrays — the sparsity pattern never changes.

```python
# ═══ Built once, shared across ALL geometries ═══
D_full = ElasticD(x_full, delta)      # on full rectangle
row_idx = D_full._row_idx              # (nnz,) — shared
col_idx = D_full._col_idx              # (nnz,) — shared
w_pos   = D_full._w_pos                # (nnz,) base weights
K_inv   = D_full._K_inv                # (N, dim, dim) — shared

# ═══ Per geometry: scale weights by mask ═══
def apply_mask(mask):
    """mask: (N,) float, m ∈ {0, 1}"""
    m_edge = mask[row_idx] * mask[col_idx]   # (nnz,) — 0 if either end in hole
    w_masked = w_pos * m_edge                 # zero out cross-hole edges
    return w_masked
```

Effect of mask on different edge types:

```
Edge type                      mask[i]·mask[j]   Effect
────────────────────────────────────────────────────────
i, j both in solid              1·1 = 1         Weight unchanged
i in solid, j in hole           1·0 = 0         Weight → 0 (horizon truncated)
i in hole, j anywhere           0·* = 0         Hole point excluded
```

The hole boundary automatically becomes a traction-free surface — the same
mechanism as physically truncating the horizon, but algebraically via
mask-zeroing.

| ✅ Pros | ❌ Cons |
|---------|---------|
| ElasticD + sub-graph pool shared across all geometries | Hole boundary staircasing (O(h) geometric error) |
| Each geometry only stores (N,) mask (~4 KB) | Grid must be fine enough to resolve minimum feature |
| Same point set → easy batching, JIT-friendly | |
| Naturally supports topology changes (multiple holes) | |

### 3.2 Option B: Coordinate Mapping (Reference Domain + Deformation)

Define a **reference domain** \(\Omega_0\) (e.g., rectangle with a fixed unit
hole) and map each physical geometry via a smooth coordinate transformation
\(\boldsymbol{\phi}: \Omega_0 \to \Omega_a\).

The PDE is **transformed** to the reference domain, introducing metric terms
from the Jacobian of \(\boldsymbol{\phi}\).  For the Navier-Cauchy equation
this adds first-order derivative corrections (the Laplacian in curvilinear
coordinates includes Christoffel-like terms).

**Two sub-options for the nonlocal operator**:

- **B1: Build ElasticD on physical points**.  Rebuild per geometry — negates
  the sharing benefit and is expensive for many geometries.
- **B2: Build ElasticD in reference space, transform the PDE**.  One fixed
  operator; metric coefficients become additional network inputs.

| ✅ Pros | ❌ Cons |
|---------|---------|
| Smooth hole boundary — no staircasing | PDE gets metric terms (chain rule explosion) |
| One mesh → one ElasticD (B2) | ElasticD approximates reference Laplacian, not physical |
| Mathematically clean | Mapping must be defined analytically per geometry family |
| | Mapping singularities for large deformations |

---

## 4. Recommendation: Mask with Weight-Zeroing (Option A)

For a proof-of-concept, the **mask + weight-zeroing approach** is the
pragmatic choice:

1. **Alignment with ElasticD's strengths**:  The nonlocal operator handles
   free boundaries via horizon truncation.  Mask-zeroing is the algebraic
   equivalent — `mask[i]·mask[j] = 0` is functionally identical to "no
   neighbour outside the domain."

2. **Maximal sharing**:  COO index arrays, base weights, shape tensor
   inverses, and sub-graph pools are built once and shared across **all**
   geometries.  Only the `(N,)` mask vector differs.

3. **Memory**:  1000 geometries → 1000 × N × 4 bytes.  For N = 16384
   (128²), that's ~65 MB total — negligible.

4. **Resolution concern**:  For a \(128 \times 128\) grid and hole radius
   \(a = 0.1L\), the circumference spans ~80 grid points → ~1% geometric
   error.  Acceptable for PoC; increase grid resolution as needed.

5. **Coordinate mapping can be added later** (Option B) for geometries
   requiring smooth boundaries.

---

## 5. Architecture Design

### 5.1 Shared Infrastructure (built once)

```
x_full (N, 2)           — fixed regular grid covering full rectangle
D_full = ElasticD(...)  — COO arrays (row_idx, col_idx, w_pos, xi, K_inv)
subgraph_pool           — pre-built shared pool (if minibatch is used)
```

### 5.2 Per-Geometry (sampled each training step)

```
a ~ U(a_min, a_max)              — geometry parameter (hole radius)
(cx, cy) ~ U(box)                — hole centre (optional)
mask = hole_mask(x_full, a, cx, cy)  — (N,) binary, 0 inside hole(s)
```

### 5.3 Per-Step Computation

```
1. w_masked = w_pos * mask[row_idx] * mask[col_idx]   # scale edge weights
2. u = net(x_full, geom_embedding, mask)              # network forward
3. residual = ElasticD.matvec_with_weights(u, w_masked, lam, mu)  # masked matvec
4. loss = MAE(residual[mask == 1])                    # PDE loss on solid only
```

### 5.4 Network Input Design

```
spatial_emb  = FourierEmbedding(x_full)      → (N, 2 × emb_dim)
mask_channel = mask[:, None]                 → (N, 1)
geom_params  = MLP_geom([a, cx, cy, ...])    → (geom_dim,)
trunk_input  = concat(spatial_emb, mask_channel, tile(geom_params))
             → FiLM-modulated MLP trunk
             → (N, 2)  (u_x_raw, u_y_raw)
             → Hard constraint → (u_x, u_y)
```

The mask is concatenated as an **input channel** so the network can learn
boundary proximity.  Geometry parameters are encoded through a small MLP
and injected via **FiLM** (Feature-wise Linear Modulation) at each hidden
layer — decoupling geometry processing from spatial processing.

---

## 6. Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Grid | Fixed regular \(N_x \times N_y\) | Avoid per-geometry meshing |
| Mask encoding | Binary channel concatenated to spatial features | Network learns boundary proximity |
| Geometry param encoding | Small MLP → latent vector → FiLM modulation | Decouples geometry from spatial processing |
| ElasticD per geometry | Shared COO, weights × mask | One build, one pool; ~4 KB per geometry |
| Training data | Randomly sample params each step | On-the-fly generation, no precomputed dataset |
| Dirichlet BCs | Hard constraint (same as elasticity2d) | Exact, no BC loss tuning |

---

## 7. What Is Shared vs Per-Geometry

| Resource | Shared across geometries | Per-geometry |
|----------|------------------------|--------------|
| Spatial points `x` | `(N, 2)` full grid | — |
| COO indices `row_idx, col_idx` | `(nnz,)` int32 | — |
| Base weights `w_pos` | `(nnz,)` float32 | — |
| Displacement vectors `xi` | `(nnz, dim)` float32 | — |
| Shape tensor inverse `K_inv` | `(N, dim, dim)` float32 | — |
| Sub-graph pool | K_pool entries (indices) | — |
| Edge weights | — | × `mask[row]·mask[col]` |
| Mask vector | — | `(N,)` float32 (~4 KB) |
| Geom embedding | — | `(geom_dim,)` per sample |

**Memory comparison** (1000 geometries, N = 120k, nnz ≈ 1.2M):

| Scheme | Per-geometry storage | Total |
|--------|---------------------|-------|
| Prune edges (different COO per geometry) | ~2 MB (COO arrays) | ~2 GB |
| **Mask zero-setting (shared COO)** | ~480 KB (mask vector) | **~480 MB** |

---

## 8. Relationship to `elasticity2d/`

The single-geometry PINN in [`../elasticity2d/`](../elasticity2d/) validates
the core components that PINO inherits:

- **ElasticD** correspondence peridynamics operator
- **Hard-constraint** Dirichlet BC enforcement (`ConstrainedForward`)
- **Non-dimensionalisation** scheme (steel plate, \(L_c = 2\) m, \(u_c = 2\) mm)
- **FEM reference solver** (FEniCS CG1)
- **Training infrastructure** (Adam, GradNorm, JIT, checkpointing)

PINO adds:
- Geometry parameterisation via binary masks
- Shared COO sparsity with per-geometry weight scaling
- FiLM conditioning on geometry parameters
- Random geometry sampling during training

---

## 9. Implementation Roadmap

| Phase | Task | Dependencies |
|-------|------|-------------|
| **1. Mask generation** | Implement `hole_mask(x, a, cx, cy)` for circular holes; extendable to multiple holes | — |
| **2. Mask-weighted ElasticD** | Extend `ElasticD` to accept per-geometry `w_masked` weights while reusing shared COO + K_inv | Phase 1 |
| **3. Geometry embedding** | Add small MLP to encode `[a, cx, cy]` → latent vector; integrate with FiLM trunk | — |
| **4. Training loop** | Random geometry sampling each step; mask input channel; PDE loss on solid-only points | Phases 2, 3 |
| **5. Benchmarking** | Accuracy vs grid resolution; generalisation to unseen geometries; comparison with single-geometry PINN | Phase 4 |

---

## References

1. Silling, S.A. et al. (2007). "Peridynamic states and constitutive modeling."
   *Journal of Elasticity*, 88, 151–184.
2. Li, Z. et al. (2020). "Fourier Neural Operator for Parametric Partial
   Differential Equations." *arXiv:2010.08895*.
3. Lu, L. et al. (2021). "Learning nonlinear operators via DeepONet based on
   the universal approximation theorem of operators." *Nature Machine
   Intelligence*, 3, 218–229.
