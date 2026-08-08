# PINO: Physics-Informed Neural Operator for Parameterised Elastic Plate Geometries

> **Status: Design Document.** No implementation exists yet. This describes the
> proposed architecture and design decisions. Last updated: July 2026.

## 1. Overview and Motivation

The single-geometry PINN in [`../elasticity2d/`](../elasticity2d/) solves a 2D
linear-elastic plate with **one fixed hole geometry** per training run.  If you
want to solve a *family* of geometries — different hole radii, different
positions, or arbitrary hole shapes — retraining a separate PINN for every
geometry is inefficient.

A **PINO (Physics-Informed Neural Operator)** learns the solution operator
that maps geometry directly to displacement and stress fields:

```
  hole geometry                          solution fields
  mask / SDF / parameters           ->   u_x(x), u_y(x), sigma_xx, sigma_yy, sigma_xy
```

The initial scope should extend the validated `elasticity2d/` example from one
ellipse, `(a_x, a_y) = (0.10, 0.15)`, to a family of elliptical and star-shaped
holes.  The long-term target is arbitrary cut-outs represented by geometry
fields such as masks or signed-distance functions (SDFs).

The design below keeps the key advantage of `ElasticD`: traction-free hole
boundaries are handled by horizon truncation, without explicit Neumann loss
terms.

---

## 2. Problem Statement

The physics is the same as `elasticity2d/`: plane-strain, static, homogeneous
linear elasticity on a rectangular plate under displacement-controlled tension.
The difference is that the hole geometry changes between training samples.

### 2.1 Phase 1: Elliptical Hole Family

| Parameter | Symbol | Range |
|-----------|--------|-------|
| Hole semi-axis x | `a_x` | `[a_x_min, a_x_max]` |
| Hole semi-axis y | `a_y` | `[a_y_min, a_y_max]` |
| Hole centre x | `c_x` | fixed first, variable later |
| Hole centre y | `c_y` | fixed first, variable later |

The first implementation should keep the hole centred at `(Lx/2, Ly/2)` and
vary one parameter, then two parameters:

1. vary `a_x`, keep `a_y = 0.15`;
2. vary both `a_x` and `a_y`;
3. optionally vary `(c_x, c_y)` after the centred case works.

### 2.2 Phase 2: Star-Shaped Holes

A smooth star-shaped hole centred at `(c_x, c_y)` can be represented by a
Fourier radius:

\[
r(\theta) = r_0 + \sum_{k=1}^{K} \left[A_k \cos(k\theta) + B_k \sin(k\theta)\right],
\qquad \theta \in [0, 2\pi).
\]

The hole interior is:

\[
\rho < r(\theta), \qquad
\rho = \sqrt{(x-c_x)^2 + (y-c_y)^2}, \qquad
\theta = \operatorname{atan2}(y-c_y, x-c_x).
\]

This gives a fixed-dimensional geometry code:

\[
\boldsymbol{\theta} = [r_0, A_1, B_1, \ldots, A_K, B_K].
\]

The same representation also gives a useful signed-distance-like field:

\[
\mathrm{sdf}(x,y) \approx \rho - r(\theta),
\]

positive in the solid and negative in the hole.

### 2.3 Boundary Conditions

| Boundary | Condition | Enforcement |
|----------|-----------|-------------|
| Left edge (`x=0`) | `u_x = u_y = 0` | Hard constraint |
| Right edge (`x=Lx`) | `u_x = u_load`, `u_y = 0` | Hard constraint |
| Top, bottom, hole boundary | traction-free | ElasticD horizon truncation |

The hard constraint from `elasticity2d/` should be reused for the left/right
Dirichlet boundaries.  The hole boundary remains a free boundary and should be
handled through the masked nonlocal operator, not through an explicit traction
penalty in the nonlocal PINO path.

---

## 3. Geometry Representation and Operator Sharing

### 3.1 Fixed Grid + Masked ElasticD Coefficients

Use a fixed regular grid covering the full rectangular plate.  Each geometry is
represented on this shared grid by:

| Field | Shape | Meaning |
|-------|-------|---------|
| `mask` | `(N,)` | `1` in the solid, `0` in the hole |
| `sdf` | `(N,)` | signed distance or signed-distance-like field |
| `normal` | `(N, 2)` | `grad(sdf) / |grad(sdf)|` near the hole boundary |
| `near_boundary` | `(N,)` | smooth feature such as `exp(-abs(sdf) / delta)` |

The neighbour graph is built once on the full rectangle:

```python
# Shared across all geometries.
row_idx = D_graph.row_idx       # (E,)
col_idx = D_graph.col_idx       # (E,)
xi = D_graph.xi                 # (E, 2), x_j - x_i
w_pos = D_graph.w_pos           # (E,), base kernel weights
```

For each geometry, the edge weights are masked:

```python
m_edge = mask[row_idx] * mask[col_idx]
w_masked = w_pos * m_edge
```

This keeps the COO shape fixed while removing edges that enter the hole.  That
is the algebraic equivalent of horizon truncation at the hole boundary.

### 3.2 Masked `K_inv` Is Geometry-Dependent

For correspondence elasticity, masking edge weights is not sufficient by
itself.  `ElasticD` uses a per-point shape tensor:

\[
\bar{\mathbf{K}}_i = \sum_j w_{ij}\,
\boldsymbol{\xi}_{ij}\otimes\boldsymbol{\xi}_{ij}.
\]

The deformation gradient and residual both use \(\bar{\mathbf{K}}_i^{-1}\).
When a hole removes neighbours near the boundary, the shape tensor changes:

\[
\bar{\mathbf{K}}_i^{g} = \sum_j
w_{ij}\,m_i\,m_j\,
\boldsymbol{\xi}_{ij}\otimes\boldsymbol{\xi}_{ij}.
\]

Therefore each geometry needs its own masked shape tensor inverse:

```python
def masked_coefficients(mask):
    m_edge = mask[row_idx] * mask[col_idx]
    w_g = w_pos * m_edge

    K_g = scatter_sum(
        w_g[:, None, None] * xi[:, :, None] * xi[:, None, :],
        row_idx,
        num_segments=N,
    )
    K_inv_g = safe_inverse_2x2(K_g, active=mask)
    return w_g, K_inv_g
```

This does **not** mean rebuilding the full `ElasticD` object per geometry.
The expensive and shape-defining pieces remain shared:

| Shared once | Per geometry |
|-------------|--------------|
| full-grid coordinates `x_full` | `mask`, `sdf`, `normal`, `near_boundary` |
| `row_idx`, `col_idx` | `w_masked = w_pos * mask[row] * mask[col]` |
| `xi` | masked shape tensor `K_g` |
| base weights `w_pos` | masked inverse `K_inv_g` |
| padded graph shapes | residual/loss mask |

In 2D, recomputing `K_inv_g` costs one edge scatter for a `2x2` tensor and one
small inverse per point.  This is much cheaper than redoing neighbour search or
building a new COO graph for every geometry, and it keeps JIT shapes fixed.

Hole points have degenerate shape tensors.  A robust implementation should set
`K_inv_g` to zero or identity on inactive points and exclude those rows from
both residual and loss.

### 3.3 Current PINN vs Future PINO

The current single-geometry `elasticity2d/` PINN is not affected by this
masked-`K_inv` issue, because it first removes the hole points and then builds
`ElasticD` on the already perforated domain.  Its precomputed `K_inv` is already
for that specific hole geometry.

The PINO design is different: it wants one fixed full-grid graph shared by many
geometries.  In that setting, the graph structure can be shared, but the
operator coefficients cannot be fully shared.  Every geometry has a different
masked `D` in the mathematical sense.

---

## 4. Geometry Injection Into the Neural Operator

There are two separate pieces of geometry information:

1. **Global shape identity**: which hole geometry is being solved.
2. **Local boundary awareness**: how a query point relates to the hole boundary.

A global parameter vector alone is not enough for arbitrary holes.  Elasticity
with holes is dominated by local boundary effects: distance to the hole, normal
direction, curvature, and near-boundary stress concentration.  A binary mask
alone is also weak: it says whether a point is solid, but not how close the
point is to the boundary or what the local boundary orientation is.

The recommended conditioned implicit operator is:

```text
geometry field [sdf_grid, mask_grid]
        -> geometry encoder
        -> global_code + local_feature_grid

query point x:
  [x, Fourier(x), sdf(x), normal_x(x), normal_y(x),
   near_boundary(x), mask(x), local_feature(x), global_code]
        -> coordinate decoder
        -> u_x(x), u_y(x)
```

### 4.1 Minimal Ellipse/Fourier Version

For low-dimensional geometry families, such as ellipses or low-order Fourier
descriptors, the first version can avoid a large image encoder:

```text
geometry code: [a_x, a_y, c_x, c_y] or [r0, A1, B1, ...]
point features: [x, Fourier(x), sdf(x), normal(x), near_boundary(x), mask(x)]
conditioning: FiLM or AdaLN
backbone: coordinate MLP / modified MLP
```

This is the shortest path from the validated single-geometry PINN to a working
operator model.

### 4.2 Arbitrary-Hole Version

For truly arbitrary holes, use an encoder over the geometry field:

```text
[sdf_grid, mask_grid] -> CNN / U-Net encoder -> feature map + global code
```

The decoder should not rely only on a single global latent vector.  Query
points should be able to read local geometry features, for example by bilinear
interpolation on the encoder feature map.

### 4.3 DeepONet vs PI-CViT

A classical DeepONet has the form:

```text
branch(geometry) · trunk(x)
```

It is a reasonable baseline, especially if the geometry is low-dimensional.
For arbitrary holes, however, a branch network that compresses the entire hole
into one vector may lose the local boundary details needed near stress
concentrations.

A PI-CViT or cross-attention model is more expressive: query points can attend
to geometry tokens.  It is also heavier, more complex to implement in JAX, and
likely harder to train in the first physics-only version.  The practical order
should be:

1. coordinate MLP + SDF/normal/mask + FiLM/AdaLN;
2. CNN/U-Net geometry encoder + coordinate decoder;
3. PI-CViT/cross-attention only if local feature interpolation is insufficient.

---

## 5. Recommended Architecture

The recommended first implementation is:

```text
fixed full grid
+ shared neighbour graph
+ per-geometry masked weights and masked K_inv
+ SDF/normal/mask point features
+ coordinate decoder conditioned by global and local geometry features
```

### 5.1 Shared Infrastructure

```text
x_full                 — fixed regular grid covering the full rectangle
row_idx, col_idx       — shared neighbour COO arrays
xi                     — shared displacement vectors x_j - x_i
w_pos                  — shared base kernel weights
subgraph_pool          — optional fixed-shape two-hop subgraphs
```

### 5.2 Per-Geometry Data

```text
geometry parameters or geometry field
mask                  — solid/hole indicator on x_full
sdf                   — signed distance or signed-distance-like field
normal                — boundary normal feature near the hole
near_boundary         — smooth distance-to-boundary feature
w_masked, K_inv       — masked ElasticD coefficients
```

### 5.3 Per-Step Computation

```text
1. Sample geometry.
2. Build mask/SDF/normal fields on the fixed grid.
3. Compute w_masked and K_inv_masked on the shared graph.
4. Encode geometry field or geometry parameters.
5. Evaluate u = net(x_full, geometry_features).
6. residual = ElasticD.matvec_masked(u, w_masked, K_inv_masked, lam, mu).
7. loss = MAE(residual on solid rows only).
```

### 5.4 Sampling Strategy

Start with full-batch or uniform sub-sampling.  The recent `elasticity2d/`
experiments showed that strongly biased hole sampling can change the effective
objective and produce visually poor global displacement fields, even when the
sampled PDE loss is low.

For PINO, there is no single fixed hole boundary that defines an optimal
sampling distribution across all geometries.  Uniform sampling is the safest
starting point.  If sub-sampling is needed, use fixed-shape two-hop core+halo
sampling and preserve the intended domain objective.

---

## 6. Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Grid | Fixed regular `Nx x Ny` | Avoid per-geometry meshing and keep JIT shapes fixed |
| Geometry fields | SDF + mask + normal + near-boundary | Gives local boundary information explicitly |
| Low-dimensional conditioning | FiLM/AdaLN | Efficient for ellipses and Fourier descriptors |
| Arbitrary-hole conditioning | CNN/U-Net encoder first | Local feature maps preserve boundary detail |
| PI-CViT | Later optional upgrade | More expressive but heavier and harder to train |
| ElasticD graph | Shared COO on full rectangle | One neighbour search and one static sparsity shape |
| ElasticD coefficients | Per-geometry masked weights and masked `K_inv` | Correct near-hole correspondence operator |
| Loss rows | Solid points only | Hole points are inactive |
| Dirichlet BCs | Hard constraint | Reuse validated `elasticity2d/` mechanism |
| Neumann BCs | Automatic via masked horizon truncation | No explicit traction-free loss in nonlocal mode |

---

## 7. What Is Shared vs Per-Geometry

| Resource | Shared across geometries | Per geometry |
|----------|--------------------------|--------------|
| Spatial points | `x_full` | — |
| COO indices | `row_idx`, `col_idx` | — |
| Edge geometry | `xi` | — |
| Base weights | `w_pos` | — |
| Shape tensor inverse | optional full-grid diagnostic only | masked `K_inv` used by residual |
| Geometry fields | — | `mask`, `sdf`, `normal`, `near_boundary` |
| Edge weights | — | `w_pos * mask[row] * mask[col]` |
| Encoder features | network parameters shared | feature map/code depends on geometry |
| Subgraph shapes | fixed padding / index templates | active rows and masked coefficients |

---

## 8. Relationship to `elasticity2d/`

The single-geometry PINN in [`../elasticity2d/`](../elasticity2d/) validates the
components that PINO should inherit:

- ElasticD correspondence elasticity;
- hard-constrained left/right Dirichlet boundaries;
- non-dimensionalised material parameters;
- regular-grid training path;
- FEM reference workflow for evaluation.

PINO adds:

- geometry sampling during training;
- geometry fields (`mask`, `sdf`, `normal`, `near_boundary`);
- shared full-grid neighbour graph;
- per-geometry masked ElasticD coefficients;
- geometry-conditioned coordinate decoder.

---

## 9. Implementation Roadmap

### Phase 1: Variable `a_x` Ellipse

| Step | Task | Notes |
|------|------|-------|
| 1.1 | Build ellipse `mask/sdf/normal` fields | Reproduce `(a_x, a_y) = (0.10, 0.15)` first |
| 1.2 | Add `ElasticD.masked_coefficients(mask)` | Compute masked weights and masked `K_inv` |
| 1.3 | Add `ElasticD.matvec_masked(...)` | Use shared COO with geometry-dependent coefficients |
| 1.4 | Add coordinate decoder inputs | `[x, Fourier(x), sdf, normal, near_boundary, mask]` |
| 1.5 | Train over `a_x` | Check interpolation and extrapolation against FEM |
| **Goal** | Minimal PINO that generalises across a 1-D ellipse family | |

### Phase 2: Variable Ellipse and Fourier Holes

| Step | Task | Notes |
|------|------|-------|
| 2.1 | Vary both `a_x` and `a_y` | Centred ellipse family |
| 2.2 | Add Fourier descriptor generator | Low-order star-shaped holes |
| 2.3 | Use FiLM/AdaLN global code | Geometry parameters modulate coordinate decoder |
| 2.4 | Keep SDF/normal point features | Do not rely on global code alone |
| 2.5 | Benchmark unseen shapes | Compare against per-geometry FEM references |
| **Goal** | Operator for low-dimensional smooth hole families | |

### Phase 3: Geometry-Field Encoder

| Step | Task | Notes |
|------|------|-------|
| 3.1 | Add CNN/U-Net encoder for `[sdf_grid, mask_grid]` | Produces global code and local feature map |
| 3.2 | Interpolate local features at query points | Decoder receives local geometry context |
| 3.3 | Compare against FiLM-only baseline | Determines whether encoder is needed |
| **Goal** | Support higher-dimensional arbitrary holes | |

### Phase 4: Optimisations

| Step | Task | Notes |
|------|------|-------|
| 4.1 | Geometry microbatching | Start with 1-2 geometries per step |
| 4.2 | Uniform two-hop sub-sampling | Preserve fixed shapes and correct halo stresses |
| 4.3 | Precompute coefficients for fixed geometry pools | Avoid repeated masked `K_inv` when geometries repeat |
| 4.4 | PI-CViT / cross-attention | Add only if CNN/U-Net local features are insufficient |
| 4.5 | Multiple holes | Use union of SDFs and the same masked-operator path |

---

## 10. References

1. Silling, S.A. et al. (2007). "Peridynamic states and constitutive modeling."
   *Journal of Elasticity*, 88, 151-184.
2. Li, Z. et al. (2020). "Fourier Neural Operator for Parametric Partial
   Differential Equations." *arXiv:2010.08895*.
3. Lu, L. et al. (2021). "Learning nonlinear operators via DeepONet based on
   the universal approximation theorem of operators." *Nature Machine
   Intelligence*, 3, 218-229.
