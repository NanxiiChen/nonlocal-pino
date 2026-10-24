# 2D Linear Elasticity — Rectangular Plate with Elliptical Hole

Single-geometry PINN for a 2D linear-elastic plate with an elliptical hole under
uniaxial tension.  Two formulations are implemented side-by-side:

| Mode | Physics | Neumann BCs |
|------|---------|-------------|
| **Nonlocal** | ElasticD correspondence peridynamics (Silling et al., 2007) | Automatic via horizon truncation |
| **Local** | Navier-Cauchy via per-point Hessian autodiff | Explicit traction-free penalty (3 surfaces) |

Dirichlet BCs are enforced via a **hard constraint** (not loss penalty) at
both clamped and pulled edges.  The reference solution comes from a **FEniCS
FEM solver** — there is no analytical solution (the Inglis solution for an
elliptical hole assumes an infinite plate, which this is not).

---

## 1. Physical Problem

### 1.1 Geometry

```
    y=Ly ┌──────────────────────────────┐
         │             free             │
         │      ╭───────────╮           │
         │      │  elliptical│ a_x=0.10 │
         │      │  hole     │  a_y=0.15 │
         │      ╰───────────╯           │
  x=0 ───│            free              │─── x=Lx
  clamped│                              │  u_x = u_load
         │                              │
    y=0  └──────────────────────────────┘
              Lx = 1.0,  Ly = 0.75
```

**Physical dimensions** (steel plate, non-dimensionalised):

| Quantity | Physical | Non-dimensional |
|----------|----------|-----------------|
| Plate length | 2.0 m | \(L_x^* = 1.0\) |
| Plate height | 1.5 m | \(L_y^* = 0.75\) |
| Hole semi-axis x | 0.2 m | \(a_x^* = 0.10\) |
| Hole semi-axis y | 0.3 m | \(a_y^* = 0.15\) |
| Applied displacement | 2.0 mm | \(u_{\text{load}}^* = 1.0\) |

The elliptical hole is **vertically elongated** (\(a_y > a_x\)), giving a
stress concentration factor \(K_t \approx 1 + 2 a_y / a_x = 4\)
at the top and bottom tips (vs \(K_t \approx 3\) for a circular hole).

Non-dimensionalisation uses two independent characteristic scales:

| Scale | Symbol | Value | Normalises |
|-------|--------|-------|------------|
| **Length scale** | \(L_c\) | 2 m (plate length) | Spatial coordinates: \(\mathbf{x}^* = \mathbf{x}/L_c\) |
| **Displacement scale** | \(u_c\) | 2 mm (applied displacement) | Displacement field: \(\mathbf{u}^* = \mathbf{u}/u_c\) |

These serve **different roles**: \(L_c\) makes the geometry dimensionless,
\(u_c\) makes the loading dimensionless.  In the homogeneous Navier-Cauchy
equation the common factor \(E u_c / L_c^2\) cancels out — the PDE itself
depends only on \(\nu\) through \(\mu^*, \lambda^*\).

### 1.2 Governing Equations

Plane strain, static equilibrium, non-dimensional (stars dropped):

\[
\mu \nabla^2 \mathbf{u} + (\lambda + \mu) \nabla(\nabla \cdot \mathbf{u}) = \mathbf{0}
\]

Lamé constants (plane strain, \(\nu = 0.3\)):

\[
\mu^* = \frac{1}{2(1+\nu)} \approx 0.3846, \qquad
\lambda^* = \frac{\nu}{(1+\nu)(1-2\nu)} \approx 0.5769
\]

Stress-strain (Hooke's law):

\[
\boldsymbol{\sigma} = 2\mu\,\boldsymbol{\varepsilon} + \lambda\,\mathrm{tr}(\boldsymbol{\varepsilon})\mathbf{I},
\qquad
\boldsymbol{\varepsilon} = \frac{1}{2}\left(\nabla\mathbf{u} + \nabla\mathbf{u}^T\right)
\]

### 1.3 Boundary Conditions

| Boundary | Type | Condition | PINN enforcement |
|----------|------|-----------|------------------|
| Left edge (\(x=0\)) | Dirichlet | \(u_x = u_y = 0\) (clamped) | Hard constraint |
| Right edge (\(x=L_x\)) | Dirichlet | \(u_x = u_{\text{load}}\), \(u_y = 0\) (displacement-controlled pull) | Hard constraint |
| Top edge (\(y=L_y\)) | Neumann (traction-free) | \(\boldsymbol{\sigma}\cdot\mathbf{e}_y = \mathbf{0}\) | Finite-horizon natural behavior (nonlocal) / MSE loss (local) |
| Bottom edge (\(y=0\)) | Neumann (traction-free) | \(\boldsymbol{\sigma}\cdot(-\mathbf{e}_y) = \mathbf{0}\) | Finite-horizon natural behavior (nonlocal) / MSE loss (local) |
| Hole (\(\frac{(x-c_x)^2}{a_x^2}+\frac{(y-c_y)^2}{a_y^2}=1\)) | Neumann (traction-free) | \(\boldsymbol{\sigma}\cdot\mathbf{n} = \mathbf{0}\), \(\mathbf{n} = -\nabla f/|\nabla f|\) | Finite-horizon natural behavior (nonlocal) / MSE loss (local) |

---

## 2. Two Formulations

### 2.1 Nonlocal: ElasticD Correspondence Peridynamics

The nonlocal formulation uses the **correspondence peridynamic model**
(Silling et al., 2007), implemented in `ElasticD` (`src/nonlocal_pinn/operators/__init__.py`).

**How it works** (single matvec call):

1. **Nonlocal deformation gradient**: \(\mathbf{G}_i = \left[\sum_j w_{ij} (\mathbf{u}_j - \mathbf{u}_i) \otimes \boldsymbol{\xi}_{ij}\right] \bar{\mathbf{K}}_i^{-1}\)
2. **Classical constitutive law**: \(\boldsymbol{\varepsilon}_i = \text{sym}(\mathbf{G}_i)\), \(\boldsymbol{\sigma}_i = \lambda\,\text{tr}(\boldsymbol{\varepsilon}_i)\mathbf{I} + 2\mu\,\boldsymbol{\varepsilon}_i\)
3. **Force residual**: \(\mathbf{r}_i = \sum_j w_{ij}\left(\boldsymbol{\sigma}_i\bar{\mathbf{K}}_i^{-1}\boldsymbol{\xi}_{ij} - \boldsymbol{\sigma}_j\bar{\mathbf{K}}_j^{-1}\boldsymbol{\xi}_{ji}\right)\)

where \(\bar{\mathbf{K}}_i = \sum_j w_{ij}\,\boldsymbol{\xi}_{ij} \otimes \boldsymbol{\xi}_{ij}\) is the shape tensor
and \(\boldsymbol{\xi}_{ij} = \mathbf{x}_j - \mathbf{x}_i\).

```python
# Single function call → full Navier-Cauchy residual
r = D.elastic_matvec(u, lam, mu)  # (N, 2) residual
```

**Key properties**:

- **Nonlocal free-boundary behavior**: Points on free boundaries have truncated
  horizons, giving the natural boundary behavior of the finite-horizon operator;
  it is not pointwise identical to classical traction-free stress at finite \(\delta\).
- **Dirichlet rows**: Exact left/right prescribed rows are excluded from the
  nonlocal equilibrium loss; a finite-width volume-constraint collar is not
  implemented.
- **N_LOSS = 1**: Only PDE residual loss.  Hard constraint handles Dirichlet BCs.
- **Full batch**: All ~100k grid points (after hole exclusion) per step.
- **δ-dependent solution**: Smaller δ → closer to classical Navier-Cauchy but
  harder to train; larger δ → stable training but different effective PDE.

### 2.2 Local: All-AD Navier-Cauchy

The local formulation computes the Navier-Cauchy residual entirely via
automatic differentiation:

```python
# Per-point Hessian AD
hess_ux = jax.hessian(lambda x: net_u(x)[0])(x)  # (2, 2)
hess_uy = jax.hessian(lambda x: net_u(x)[1])(x)  # (2, 2)
lap_ux = jnp.trace(hess_ux)
lap_uy = jnp.trace(hess_uy)
gd_x = hess_ux[0,0] + hess_uy[0,1]  # [∇(∇·u)]_x
gd_y = hess_ux[1,0] + hess_uy[1,1]  # [∇(∇·u)]_y
r_x = mu * lap_ux + (lam + mu) * gd_x
r_y = mu * lap_uy + (lam + mu) * gd_y
```

**N_LOSS = 4**: PDE residual + three explicit Neumann BC penalty terms
(top, bottom, hole).  Each BC term minimises \(\|\boldsymbol{\sigma}\cdot\mathbf{n}\|^2\)
MSE at 200 collocation points per surface.

**GradNorm** adaptive weighting (updated every 10 steps, \(\alpha = 1.0\))
balances the four loss terms.  PDE collocation uses LHS (Latin Hypercube)
sampling with 2000 points resampled each step.

### 2.3 Comparison

| Aspect | Nonlocal (ElasticD) | Local (All-AD) |
|--------|---------------------|----------------|
| Loss terms | 1 (PDE only) | 4 (PDE + top + bottom + hole) |
| Free boundaries | Automatic (horizon truncation) | Explicit \(\|\boldsymbol{\sigma}\cdot\mathbf{n}\|^2\) penalty |
| Collocation | Full grid (~100k points) | LHS sampling (2000 points) |
| Second derivatives | Nonlocal operator (O(N·k) matvec) | Per-point Hessian AD (expensive per point) |
| GradNorm | Not needed | 4-term adaptive weighting |
| Tuning burden | Minimal (δ only) | Weight schedule + BC sampling |

---

## 3. Hard Constraint (Dirichlet BCs)

Dirichlet BCs are **not** trained via loss penalty.  They are enforced
exactly by wrapping the raw network output in a post-processing transform:

\[
\begin{aligned}
u_x(x, y) &= \frac{x}{L_x}\,u_{\text{load}} + x(L_x - x)\,u_{\text{load}}\,N_x^{\text{net}}(x, y) \\[4pt]
u_y(x, y) &= x(L_x - x)\,u_{\text{load}}\,N_y^{\text{net}}(x, y)
\end{aligned}
\]

where \(N_x^{\text{net}}, N_y^{\text{net}}\) are the raw (unconstrained)
network outputs.  The prefactor \(x(L_x - x)\,u_{\text{load}}\) vanishes at
both Dirichlet edges (\(x=0\) and \(x=L_x\)):

| Position | \(u_x\) | \(u_y\) |
|----------|---------|---------|
| Left edge (\(x=0\)) | \(= 0\) ✓ | \(= 0\) ✓ |
| Right edge (\(x=L_x\)) | \(= u_{\text{load}}\) ✓ | \(= 0\) ✓ |

Implemented via `ConstrainedForward` (`src/nonlocal_pinn/nn/constraints.py`) with
`pass_input=True` so the constraint closure receives spatial coordinates.

---

## 4. Network Architecture

| Component | Choice |
|-----------|--------|
| Trunk type | MLP (4 hidden layers) |
| Hidden dim | 64 |
| Activation | GELU |
| Spatial input | \((x, y)\) coordinates |
| Spatial embedding | Fourier embedding, 64 dim, scale = 2.0 |
| Temporal input | **None** — steady-state (`static=True`) |
| Output heads | `multi_head=True` — separate `Linear(64, 1)` for \(u_x\) and \(u_y\) |
| Output constraint | `ConstrainedForward` with hard constraint (see §3) |

---

## 5. Spatial Discretisation

- **Grid**: Regular \(400 \times 300\) Cartesian grid on \([0, L_x] \times [0, L_y]\)
- **Hole exclusion**: Points where \(((x-c_x)/a_x)^2 + ((y-c_y)/a_y)^2 < 1\) are removed (\(\approx 100,\!000\) effective points after exclusion)
- **dV**: Original regular-grid cell area `dx*dy` for every retained point;
  deleted hole points contribute no quadrature weight (not barycentric)
- **ElasticD construction**: Grid neighbour search path (fast, O(N·k))
- **No minibatch**: \(K_{\text{core}} = 0\) → full batch matvec in nonlocal mode

---

## 6. FEM Reference Solver

**Solver**: `examples/elasticity2d/solver.py` — FEniCS with CG1 elements.

- **Weak form**: \(\int_\Omega \left[2\mu\,\boldsymbol{\varepsilon}(\mathbf{u}):\boldsymbol{\varepsilon}(\mathbf{v}) + \lambda\,\nabla\cdot\mathbf{u}\,\nabla\cdot\mathbf{v}\right] d\Omega = 0\)
- **Mesh**: Gmsh unstructured triangular mesh (generated from `mesh.geo`),
  converted to XDMF via meshio, refined near the hole
- **BCs**: Left edge clamped (\(u_x=u_y=0\)), right edge pulled (\(u_x=u_{\text{load}}, u_y=0\))
- **Solve**: Direct LU (no time stepping)
- **Output** (`results.npz`): `coords`, `u_x`, `u_y`, `sigma_xx`, `sigma_yy`, `sigma_xy`

During PINN evaluation (every 200 steps), the FEM solution is interpolated
onto the PINN grid and relative L2 errors are computed:

\[
\text{rel\_L2}(u_x) = \frac{\|u_x^{\text{pred}} - u_x^{\text{fem}}\|_2}{\|u_x^{\text{fem}}\|_2}
\]

---

## 7. Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning rate | \(2 \times 10^{-3}\) initial, exponential decay (0.95 / 2000 steps) |
| LR floor | \(1 \times 10^{-5}\) |
| Iterations | 40,000 (`configs/train_base.py`) |
| ElasticD horizon | \(\delta = 0.0075\) |
| ElasticD kernel | Cubic (\(C^1\) smooth) |
| Gradient clipping | 1000.0 |
| Matmul precision | `"highest"` (exact FP32) |
| GradNorm update | Every 10 steps, \(\alpha = 1.0\) (instantaneous) |
| Evaluation | Every 200 steps |
| Checkpoint | Every 200 steps |

### 7.1 Sub-sampling (nonlocal mode)

When `c.sample.K_core > 0`, the nonlocal PDE is evaluated on a random subset
of the grid rather than the full ~100k points. This speeds up each step and,
with non-uniform weighting, concentrates training signal on the hole region.

**Mechanism.**  Each step selects `K_core` core points via weighted random
sampling (without replacement).  The two-hop neighbourhood of those core points
(halo1 + halo2, needed by the correspondence peridynamic stress computation)
is added, giving a sub-graph of size `K_all ≥ K_core`.  The PDE residual is
evaluated on all `K_all` points but **only core rows contribute to the loss**
(halo rows are zeroed by `core_mask`).  All core points have **equal weight in
the loss** — the sampling weights only affect *which* points are selected, not
the loss computation itself.

Pre-built pool.  `n_pool` sub-graphs are pre-generated at startup via
`ElasticD.build_subgraph_pool`, all padded to identical shapes to avoid JIT
recompilation.  The training loop advances to the next pool entry every
`resample_every` steps.

**Sampling weights.**  Per-point selection probability follows:

\[
w_i \propto 1 + \frac{\gamma}{r_i^p},
\qquad
r_i = \sqrt{\left(\frac{x_i - c_x}{a_x}\right)^2 +
              \left(\frac{y_i - c_y}{a_y}\right)^2}
\]

where \(r = 1\) is the hole boundary and \(r > 1\) in the solid.  The two
parameters control the spatial concentration of core points:

| Parameter | Default | Role |
|-----------|---------|------|
| `refine_weight` \(\gamma\) | 0.0 | Contrast: 0 = uniform; larger → more points near hole |
| `weight_power` \(p\) | 2.0 | Decay rate: larger → refinement confined to a thinner shell |

For example, with \(\gamma = 4, p = 3\): points on the hole boundary
(\(r \approx 1\)) are ~3.3× more likely to be selected than points at
\(r = 2\).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `K_core` | 0 | Core points per sub-graph (0 = full batch) |
| `n_pool` | 100 | Number of pre-built sub-graphs in the pool |
| `resample_every` | 10 | Steps between switching pool entries |

---

## 8. File Structure

```
examples/elasticity2d/
├── README.md                  # This file
├── train.py                   # Training script (grid, network, ElasticD, training loop)
├── losses.py                  # PDE residuals (nonlocal/local), Neumann BC losses, GradNorm, loss factory
├── solver.py                  # FEM reference solver (FEniCS CG1)
├── mesh.geo                   # Gmsh geometry script (elliptical hole)
├── configs/
│   ├── train_base.py          # PINN training hyperparameters
│   └── solver_base.py         # FEM solver config
├── mesh.msh                   # Generated Gmsh mesh (committed for convenience)
└── mesh.xdmf / mesh.h5        # Converted XDMF mesh files
```

---

## 9. Usage

### FEM reference (run first)

```bash
conda activate fenicsenv
python examples/elasticity2d/solver.py \
  --config=examples/elasticity2d/configs/solver_base.py
```

### Nonlocal (ElasticD) mode

```bash
# JAX environment (uv venv)
python examples/elasticity2d/train.py \
  --config=examples/elasticity2d/configs/train_base.py
```

### Local (all-AD) mode

```bash
python examples/elasticity2d/train.py \
  --config=examples/elasticity2d/configs/train_base.py \
  --config.nl.method=local \
  --config.log.run_name=local_run
```

> **Note**: Config files hardcode output paths to `/root/autodl-tmp/`.  Adjust
> `config.output.dir` and `config.log.log_dir` for your setup.

---

## 10. Design Decisions

| Decision | Rationale |
|----------|-----------|
| **ElasticD over hybrid D_δ+AD** | Correspondence formulation gives the full Navier-Cauchy operator in one matvec; no mixing of nonlocal and local derivative computations |
| **Hard constraint over BC loss** | Dirichlet BCs are mathematically exact (no tuning), reduce loss terms, avoid PDE–BC gradient conflict |
| **Full batch over minibatch** | ~100k steady-state points fit in GPU memory for a sparse matvec; no sub-graph bookkeeping |
| **FEM over analytical (Inglis/Kirsch)** | FEM handles finite-plate effects; the Inglis solution for elliptical holes and the Kirsch solution for circular holes both assume an infinite plate and would introduce systematic error |
| **Regular grid over Gmsh unstructured** | Simplifies ElasticD construction (grid neighbour search path), uniform dV, no barycentric dV complexity |
| **Multi-head output** | Separate `Linear(64,1)` per output component gives each field its own final projection, improving expressivity for multi-scale problems (u_x ≫ u_y) |

---

## 11. Extension: PINO (Parameterised Geometry)

This directory implements a **single-geometry** PINN.  For solving a
**family** of parameterised geometries (variable hole radius, position,
shape), a PINO (Physics-Informed Neural Operator) design using mask-based
weight-zeroing with shared COO sparsity is proposed.

See [`../../docs/design/pino_elasticity2d.md`](../../docs/design/pino_elasticity2d.md) for the full PINO design document.

---

## References

1. Silling, S.A., Epton, M., Weckner, O., Xu, J., & Askari, E. (2007).
   "Peridynamic states and constitutive modeling." *Journal of Elasticity*,
   88, 151–184.
2. Silling, S.A. (2000). "Reformulation of elasticity theory for
   discontinuities and long-range forces." *Journal of the Mechanics and
   Physics of Solids*, 48(1), 175–209.
3. Bobaru, F., Foster, J.T., Geubelle, P.H., & Silling, S.A. (2016).
   *Handbook of Peridynamics*. CRC Press.


## Project split (2026-10-24)

This example belongs to `nonlocal-pinn`. Install the project and run from its
repository root; existing external data/output paths are retained. Shared
implementation paths changed during the split; see
[the source map](../../docs/migration/source-map.json) and
[migration notes](../../docs/migration/README.md).
