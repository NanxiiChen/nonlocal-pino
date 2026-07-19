# 2D Linear Elasticity — Irregular Plate with a Hole (Kirsch Problem)

This example demonstrates the **nonlocal operator \(D_\delta\)** for a 2D
linear-elastic solid mechanics problem.  A square plate with a circular hole
under uniaxial tension is chosen as the benchmark because:

1. The **hole boundary is a curved, traction-free (Neumann) surface** — the
   nonlocal operator handles this automatically, while a local PINN must
   explicitly penalise \(\boldsymbol{\sigma}\cdot\mathbf{n} = \mathbf{0}\).
2. The **Kirsch problem** has a closed-form analytical solution, giving us
   a clean validation target.
3. The hole causes a **stress concentration** (SCF = 3) that tests whether
   the PINN can resolve sharp gradients without a conforming mesh.

More broadly, this example tackles the three questions raised by irregular-
domain elasticity:

| Question | Answer |
|----------|--------|
| How to sample points on an irregular domain? | Gmsh unstructured triangular mesh |
| How to construct \(D_\delta\) without a regular grid? | KD-tree neighbour search (already implemented) + barycentric `dV` (new) |
| How to handle boundary `dV` for the nonlocal operator? | Per-node `dV` from barycentric dual area; horizon truncation at boundaries is the *mechanism* that yields free Neumann BCs |

---

## 1. Physical Problem

### 1.1 Geometry

```
      u = u_Kirsch (Dirichlet, prescribed from analytical solution)
 ┌────────────────────────────────────────┐
 │                                        │
 │                                        │
 │           elastic plate                │
 │           (plane strain)               │
 │                                        │
 │              ┌──────┐                  │
 │              │ hole │  ← traction-free │
 │              │ r=a  │     (σ·n = 0)   │
 │              └──────┘                  │
 │                                        │
 │                                        │
 └────────────────────────────────────────┘
      u = u_Kirsch (Dirichlet on all 4 outer edges)
```

- **Domain**: \(\Omega = [-L/2, L/2]^2 \setminus \{r < a\}\), where
  \(r = \sqrt{x^2 + y^2}\).
- **Plate size**: \(L\) (square side length), with \(L \gg a\) so that
  boundary effects at the outer edges are negligible.
- **Hole radius**: \(a\), centred at the origin.
- **Loading**: applied via **displacement control** — the analytical
  Kirsch displacement \(\mathbf{u}_{\text{Kirsch}}\) is prescribed on
  all four outer edges as Dirichlet BCs.  Physically, this corresponds to
  far-field uniaxial tension \(\sigma_{xx} = \sigma_\infty\).

By symmetry, we model only the **quarter-plate**:
\(x \in [a, L/2]\), \(y \in [0, L/2]\), with symmetry BCs on \(x=0\) (for
\(y > a\)) and \(y=0\) (for \(x > a\)):

```
  y ↑
    │
L/2 ┤  ┌──────────────────────────┐
    │  │                          │  ← outer boundaries:
    │  │      quarter-plate       │    u = u_Kirsch (Dirichlet)
    │  │                          │
    │  │    ╭───╮  ← hole        │
  a ┤  │    │   │    boundary    │
    │  │    ╰───╯    (free)      │
    │  │                          │
  0 ┤  └──┬───────────────────────┘
    │     │ symmetry line: u_y = 0 (Dirichlet)
    │     │                  σ_xy = 0 (free — D_δ handles)
    └─────┴──────────────────────────→ x
          0    a               L/2
```

**Alternatively**, we model the **full plate** to avoid implementing
symmetry BCs — this is simpler and the hole boundary is the only
traction-free surface we need to validate.

### 1.2 Governing Equations

**Dimensional** (plane strain, static equilibrium):

\[
\mu \nabla^2 \mathbf{u} + (\lambda + \mu) \nabla(\nabla \cdot \mathbf{u}) = \mathbf{0}
\qquad \text{in } \Omega
\]

where:

\[
\mathbf{u} = (u_x, u_y) \quad \text{(displacement field)}
\]
\[
\mu = \frac{E}{2(1+\nu)}, \qquad
\lambda = \frac{E\nu}{(1+\nu)(1-2\nu)} \quad \text{(Lamé constants, plane strain)}
\]

**Component form** (2D):

\[
\begin{aligned}
\mu\left(\frac{\partial^2 u_x}{\partial x^2} + \frac{\partial^2 u_x}{\partial y^2}\right)
+ (\lambda+\mu)\left(\frac{\partial^2 u_x}{\partial x^2} + \frac{\partial^2 u_y}{\partial x\partial y}\right) &= 0 \\[4pt]
\mu\left(\frac{\partial^2 u_y}{\partial x^2} + \frac{\partial^2 u_y}{\partial y^2}\right)
+ (\lambda+\mu)\left(\frac{\partial^2 u_x}{\partial x\partial y} + \frac{\partial^2 u_y}{\partial y^2}\right) &= 0
\end{aligned}
\]

Or more compactly using the **stress tensor**:

\[
\boldsymbol{\sigma} = 2\mu\,\boldsymbol{\varepsilon} + \lambda\,\mathrm{tr}(\boldsymbol{\varepsilon})\mathbf{I},
\qquad
\boldsymbol{\varepsilon} = \frac{1}{2}\left(\nabla\mathbf{u} + \nabla\mathbf{u}^T\right)
\]
\[
\nabla \cdot \boldsymbol{\sigma} = \mathbf{0}
\]

### 1.3 Boundary Conditions

**Loading is applied via displacement control**: the analytical Kirsch
displacement field is prescribed on all outer edges as Dirichlet BCs.
This avoids the need to compute boundary tractions \(\boldsymbol{\sigma}\cdot\mathbf{n}\)
or sample normals — the load is applied purely through the BC loss.

| Boundary | Type | Condition | PINN enforcement |
|----------|------|-----------|------------------|
| Hole (\(r = a\)) | Neumann (traction-free) | \(\boldsymbol{\sigma}\cdot\mathbf{n} = \mathbf{0}\) | **Automatic** via \(D_\delta\) — domain boundary truncates the horizon |
| Outer edges (all 4 sides) | Dirichlet | \(\mathbf{u} = \mathbf{u}_{\text{Kirsch}}\) | MSE loss penalty |
| Symmetry planes (optional) | Mixed | \(u_y = 0,\ \sigma_{xy} = 0\) on \(y=0\) | Dirichlet on \(u_y\) + free on \(\sigma_{xy}\) (D_δ) |

**Why the hole needs no BC loss**:  The nonlocal operator \(D_\delta\) is
built from neighbour pairs **entirely inside** the domain.  Points on
the hole boundary simply have no neighbours outside — their horizon is
naturally truncated.  This truncation is the **mechanism** that produces
\(\partial/\partial n = 0\) behaviour at the continuum limit, without
ever asking the optimiser to enforce it.

**Why the loaded edges also need no extra handling**:  The load is applied
via Dirichlet BCs (prescribed displacements from the analytical solution),
not via traction.  This means both the nonlocal and local PINN only need
a standard BC loss term on the outer edges — no stress computation, no
normal vectors, no traction penalty.  The only BC that differs between
the two methods is the **hole boundary** (free for D_δ, explicit Neumann
penalty for local AD).

For the local (autograd) PINN baseline, the hole boundary requires
explicit collocation points and a penalty term \(\|\boldsymbol{\sigma}\cdot\mathbf{n}\|^2\)
in the loss — exactly the same structural conflict documented in the
[electropolish2d README](../electropolish2d/README.md).

### 1.4 Non-Dimensionalisation

Choose characteristic scales:

| Scale | Symbol | Definition |
|-------|--------|------------|
| Length | \(L_c = L\) | Plate side length |
| Stress | \(\sigma_c = \sigma_\infty\) | Far-field tension |
| Displacement | \(u_c = \sigma_\infty L / E\) | Characteristic displacement |

Non-dimensional variables (starred):

\[
\mathbf{x}^* = \frac{\mathbf{x}}{L_c}, \qquad
\mathbf{u}^* = \frac{\mathbf{u}}{u_c}, \qquad
\boldsymbol{\sigma}^* = \frac{\boldsymbol{\sigma}}{\sigma_\infty}
\]

The Navier-Cauchy equation becomes:

\[
\mu^* \nabla^{*2} \mathbf{u}^* + (\lambda^* + \mu^*) \nabla^*(\nabla^* \cdot \mathbf{u}^*) = \mathbf{0}
\]

where the non-dimensional Lamé constants are:

\[
\mu^* = \frac{1}{2(1+\nu)}, \qquad
\lambda^* = \frac{\nu}{(1+\nu)(1-2\nu)}
\]

These depend **only on Poisson's ratio** \(\nu\) — the Young's modulus
\(E\) and applied stress \(\sigma_\infty\) cancel out (linear elasticity).

For plane strain with \(\nu = 0.3\):
\[
\mu^* \approx 0.3846, \qquad \lambda^* \approx 0.5769
\]

---

## 2. Nonlocal Formulation

### 2.1 Operator Definition

The nonlocal operator \(D_\delta\) approximates the Laplacian on an
arbitrary point set:

\[
(D_\delta u)(\mathbf{x}_i) = \sum_{j \in \mathcal{N}_\delta(i)}
J(\|\mathbf{x}_i - \mathbf{x}_j\|)\, (u_i - u_j)\, dV_j
\]

where:
- \(\mathcal{N}_\delta(i) = \{j : \|\mathbf{x}_i - \mathbf{x}_j\| \le \delta,\ j \ne i\}\) — neighbours within the horizon,
- \(J(r)\) is the kernel function (cubic, quartic, or Wendland C²),
- \(dV_j\) is the **per-node volume element** (area in 2D).

**Matrix form**: \(D_\delta \mathbf{u} = \mathbf{D} \cdot \mathbf{u}\) where
\[
D_{ij} = -J(r_{ij})\, dV_j \quad (i \ne j), \qquad
D_{ii} = \sum_{j \ne i} J(r_{ij})\, dV_j
\]

### 2.2 Application to Elasticity

The nonlocal Navier-Cauchy equation replaces \(\nabla^2\) with \(-D_\delta\)
(the negative sign because \(D_\delta \approx -\nabla^2\)):

\[
-\mu^* D_\delta \mathbf{u}^* + (\lambda^* + \mu^*) \nabla^*(\nabla^* \cdot \mathbf{u}^*) = \mathbf{0}
\]

Component-wise:

\[
\begin{aligned}
-\mu^* (D_\delta u^*_x) + (\lambda^*+\mu^*)\frac{\partial}{\partial x^*}(\nabla^* \cdot \mathbf{u}^*) &= 0 \\[4pt]
-\mu^* (D_\delta u^*_y) + (\lambda^*+\mu^*)\frac{\partial}{\partial y^*}(\nabla^* \cdot \mathbf{u}^*) &= 0
\end{aligned}
\]

**Important**:  Only the Laplacian part is replaced by \(D_\delta\).
The \(\nabla(\nabla\cdot\mathbf{u})\) term still requires first derivatives
of the divergence, which are computed via automatic differentiation (AD)
through the PINN.  This is a **hybrid** approach: nonlocal for the
isotropic second-derivative operator, AD for the mixed-derivative terms.

The full per-component PINN residual (non-dimensional, dropping stars for
brevity):

\[
\begin{aligned}
r_x &= -\mu\,(D_\delta u_x) + (\lambda+\mu)\left(
      \frac{\partial^2 u_x}{\partial x^2} + \frac{\partial^2 u_y}{\partial x\partial y}
      \right) \\[4pt]
r_y &= -\mu\,(D_\delta u_y) + (\lambda+\mu)\left(
      \frac{\partial^2 u_x}{\partial x\partial y} + \frac{\partial^2 u_y}{\partial y^2}
      \right)
\end{aligned}
\]

### 2.3 Boundary Handling

**Free boundaries (Neumann / traction-free)** — the key advantage:

Consider a point \(\mathbf{x}_i\) on the hole boundary.  Its horizon
\(H_\delta(\mathbf{x}_i) = \{\mathbf{x} : \|\mathbf{x} - \mathbf{x}_i\| \le \delta\}\)
is partially outside \(\Omega\) (inside the hole).  The neighbour set
\(\mathcal{N}_\delta(i)\) contains only domain-interior points — the sum
over \(j\) naturally truncates.  This truncated sum produces the correct
zero-flux behaviour at the continuum limit, **without any explicit BC
loss term**.

**Surface correction** (peridynamics "surface effect"):

The truncation error near boundaries is \(O(\delta)\) rather than the
interior \(O(\delta^2)\).  Several approaches exist:

| Approach | Complexity | Accuracy |
|----------|------------|----------|
| No correction | Trivial | \(O(\delta)\) error in boundary layer of width \(\sim\delta\) |
| Volume scaling: \(dV_i \leftarrow dV_i \cdot (V_{\text{full}} / V_{\text{actual}})\) | Light | Reduces surface error, simple to implement |
| Energy-based correction | Heavy | Best accuracy for peridynamic fracture |
| PDE-corrected weights | Medium | Matches local solution on a patch |

**For this example**: start with no correction.  The boundary layer is
\(\sim 3\)–\(5\) points thick (\(\delta/h \approx 4\)).  The global solution
error is dominated by the PINN's PDE residual, not the surface effect.
If SCF accuracy is insufficient, add volume scaling as a follow-up.

**Dirichlet boundaries** (outer edges):

Still require explicit MSE loss terms:
\[
\mathcal{L}_{\text{BC}} = \frac{1}{N_{\text{BC}}} \sum_{i}
\|\mathbf{u}_{\text{pred}}(\mathbf{x}_i) - \mathbf{u}_{\text{Kirsch}}(\mathbf{x}_i)\|^2
\]

---

## 3. Point Generation and dV Computation

### 3.1 Gmsh Mesh

An unstructured triangular mesh is generated by Gmsh.  The `.geo` script
specifies:

- Domain: square with a circular hole
- Physical groups: `hole` (traction-free), `outer` (Dirichlet), optionally
  `symmetry_x` and `symmetry_y`
- Mesh refinement near the hole (stress concentration region)
- Target: 2000–5000 nodes (small enough for full-graph \(D_\delta\);
  scalable to larger meshes via sub-graph sampling)

```
Mesh generation pipeline:
  mesh.geo  ──(gmsh)──▶  mesh.msh  ──(read_gmsh_nodes)──▶  x_points (N, 2)
                                  ──(read_gmsh_elements)─▶  tri_nodes (M, 3)
                                                          ▶  boundary_groups
```

### 3.2 Gmsh Geometry Script (`mesh.geo`)

```python
// mesh.geo — quarter-plate with hole, Gmsh script
L  = 1.0;    // plate half-size
a  = 0.1;    // hole radius
lc0 = 0.02;  // mesh size far from hole
lc1 = 0.005; // mesh size near hole

// Points: outer square
Point(1) = {a, 0, 0, lc0};
Point(2) = {L, 0, 0, lc0};
Point(3) = {L, L, 0, lc0};
Point(4) = {0, L, 0, lc0};
Point(5) = {0, a, 0, lc0};

// Points: quarter-circle hole
Point(6) = {a, 0, 0, lc1};
Point(7) = {a*Cos(Pi/4), a*Sin(Pi/4), 0, lc1};
Point(8) = {0, a, 0, lc1};

// Lines
Line(1) = {1, 2};  // bottom edge (symmetry)
Line(2) = {2, 3};  // right edge
Line(3) = {3, 4};  // top edge
Line(4) = {4, 5};  // left edge
Circle(5) = {6, 9, 7};  // hole arc
Circle(6) = {7, 9, 8};  // hole arc
// ... (simplified — actual .geo file will be complete)

// Physical groups
Physical Curve("hole", 10) = {5, 6};
Physical Curve("outer", 20) = {2, 3, 4};
Physical Curve("symmetry", 30) = {1, 7};
Physical Surface("domain", 100) = {1};
```

For the full-plate variant, the geometry is symmetric about both axes
and the hole is centred at \((L/2, L/2)\).

### 3.3 Element Parsing (`mesh.py` additions)

Gmsh `.msh` files contain `$Nodes` (already parsed) and `$Elements`
(new parsing needed).

**Format 2.x**:
```
$Elements
nElements
elm-number elm-type nTags tag1 tag2 ... node1 node2 node3 ...
...
$EndElements
```

**Format 4.x**:
```
$Elements
nEntityBlocks nElements minElementTag maxElementTag
entityDim entityTag elementType nElementsInBlock
elementTag1 node1 node2 node3 ...
...
$EndElements
```

Key element types:

| Gmsh type | Element |
|-----------|---------|
| 2 (format 2) / 2 (format 4) | 3-node triangle |
| 1 (format 2) / 1 (format 4) | 2-node line (boundary) |

We extract:
- **Volume elements** (triangles): `tri_nodes` → `(N_elem, 3)` array of
  0-indexed node indices.
- **Boundary elements** (lines): grouped by physical tag → node index
  arrays for each boundary (hole, outer, symmetry).

Function signatures:

```python
def read_gmsh_elements(path):
    """Parse $Elements section from Gmsh .msh file.

    Returns:
        tri_nodes: (M, 3) np.ndarray — triangle node indices (0-indexed)
        boundaries: dict[str, np.ndarray] — boundary node indices by
                    physical group name
    """
    ...

def read_gmsh_mesh(path):
    """Convenience: returns nodes, triangles, boundaries in one call."""
    nodes = read_gmsh_nodes(path)
    tri_nodes, boundaries = read_gmsh_elements(path)
    return nodes, tri_nodes, boundaries
```

### 3.4 Barycentric dV Computation

Each node's volume (area in 2D) is computed from the triangular mesh
using the **lumped-mass** (barycentric dual) formula:

\[
dV_i = \frac{1}{3} \sum_{\text{elements } e \text{ containing node } i} \text{Area}(e)
\]

```python
def barycentric_dV(nodes, tri_nodes):
    """Compute per-node area from triangular mesh.

    Args:
        nodes: (N, 2) node coordinates
        tri_nodes: (M, 3) triangle vertex indices (0-indexed)

    Returns:
        dV: (N,) per-node area
    """
    import numpy as np
    # Triangle vertices: (M, 3, 2)
    tris = nodes[tri_nodes]  # (M, 3, 2)
    # Area = 0.5 * |(v1-v0) × (v2-v0)|
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    cross = (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) \
          - (v1[:, 1] - v0[:, 1]) * (v2[:, 0] - v0[:, 0])
    areas = 0.5 * np.abs(cross)  # (M,)

    # Accumulate area/3 to each vertex
    dV = np.zeros(len(nodes))
    for k in range(3):
        np.add.at(dV, tri_nodes[:, k], areas / 3.0)
    return dV
```

**Note**:  `np.add.at` is used for unbuffered accumulation (multiple
triangles contribute to the same node).  For 2D meshes with ~5000 nodes,
this is fast enough to run on CPU during setup.

### 3.5 dV Ratios and Mesh Quality

For a uniform mesh, all `dV_i` are approximately equal (the ratio
`max(dV)/min(dV)` is close to 1).  For a refined mesh, nodes near the
hole have significantly smaller `dV` than far-field nodes.  This
heterogeneity is the **physical** reason why per-node dV matters:

```
Far-field node:  horizon contains ~50 neighbours, large dV each
Near-hole node:  horizon contains ~50 neighbours, small dV each
                 → smaller total stiffness → softer response locally
                 → stress concentration at hole is resolved correctly
```

With uniform `dV = vol/N`, all nodes have the same total stiffness,
which artificially stiffens the near-hole region.

---

## 4. Network Architecture

### 4.1 Design

| Component | Choice |
|-----------|--------|
| Trunk type | MLP (or ModifiedMLP for better expressivity) |
| Spatial input | `(x, y)` coordinates |
| Spatial embedding | Fourier embedding with multi-scale frequencies |
| Temporal input | **None** — steady-state problem |
| Output | `(u_x, u_y)` — displacement components |
| Output constraint | **None** — displacements are unbounded |
| Hidden layers | 4–6 layers × 64–128 neurons |
| Activation | `tanh` (smooth, needed for AD of second derivatives) |

### 4.2 Network Call Signature

```python
net(x) -> (2,)   # for a single point
net(x_batch) -> (N, 2)   # for batched points
```

To fit the existing `arch.py` interface (which always takes `(x, t)`),
we set `use_temporal_emb=False` and pass a dummy `t=0.0`:

```python
# In train.py
net = make_pinn(spatial_dim=2, out_dim=2,
                trunk_type="mlp",
                use_spatial_emb=True,
                use_temporal_emb=False,  # ← steady-state
                spatial_emb_dim=64,
                spatial_scale=(1.0, 3.0, 6.0),  # multi-scale
                hidden_dim=64,
                num_layers=4,
                activation=jnp.tanh,
                key=key)

# In losses.py — evaluation at t=0.0
def net_spatial(x):
    return net(x, jnp.array(0.0))  # (N, 2)

# or equivalently, vmap over points
u = jax.vmap(net, in_axes=(0, None))(x_points, jnp.array(0.0))
```

### 4.3 Gradient Computation

For the \(\nabla(\nabla\cdot\mathbf{u})\) term, we need second derivatives
of the network output w.r.t. spatial coordinates.  These are computed via
`jax.hessian` or `jax.jacfwd(jax.jacrev(...))`:

```python
def grad_div(u_fn, x):
    """Compute ∇(∇·u) at point x: mixed second derivatives.

    u_fn: callable mapping (2,) -> (2,)
    Returns: (2,) array [∂²u_x/∂x² + ∂²u_y/∂x∂y,
                         ∂²u_x/∂x∂y + ∂²u_y/∂y²]
    """
    # Hessians of each output component
    hess_ux = jax.hessian(lambda xx: u_fn(xx)[0])(x)  # (2, 2)
    hess_uy = jax.hessian(lambda xx: u_fn(xx)[1])(x)  # (2, 2)
    # ∇(∇·u) = [∂ₓ(∂ₓuₓ + ∂yuy), ∂y(∂ₓuₓ + ∂yuy)]
    #         = [hess_ux[0,0] + hess_uy[0,1], hess_ux[1,0] + hess_uy[1,1]]
    gd_x = hess_ux[0, 0] + hess_uy[0, 1]
    gd_y = hess_ux[1, 0] + hess_uy[1, 1]
    return jnp.array([gd_x, gd_y])
```

This is **per-point** and will be `vmap`'d over the batch dimension.

---

## 5. Loss Function Design

### 5.1 Loss Terms

**Nonlocal PINN** (2 loss terms):

| Term | Formula | Purpose |
|------|---------|---------|
| PDE | \(\mathcal{L}_{\text{PDE}} = \frac{1}{N}\sum_i \|\mathbf{r}(\mathbf{x}_i)\|_1\) | Enforce Navier-Cauchy in Ω |
| Dirichlet BC | \(\mathcal{L}_{\text{BC}} = \frac{1}{N_{\text{BC}}}\sum_i \|\mathbf{u}(\mathbf{x}_i) - \mathbf{u}_{\text{Kirsch}}(\mathbf{x}_i)\|^2\) | Fix outer boundaries |

**No Neumann BC loss** — the hole boundary is free, handled by \(D_\delta\).

**Local (autograd) PINN** (3 loss terms, for comparison):

| Term | Formula | Purpose |
|------|---------|---------|
| PDE | \(\mathcal{L}_{\text{PDE}} = \frac{1}{N}\sum_i \|\mathbf{r}_{\text{AD}}(\mathbf{x}_i)\|_1\) | All-AD Navier-Cauchy |
| Dirichlet BC | Same as above | Fix outer boundaries |
| **Neumann BC** | \(\mathcal{L}_{\text{hole}} = \frac{1}{N_{\text{hole}}}\sum_i \|\boldsymbol{\sigma}(\mathbf{x}_i)\cdot\mathbf{n}(\mathbf{x}_i)\|^2\) | Traction-free hole |

### 5.2 PDE Residual (Nonlocal)

For a batch of points \(\{\mathbf{x}_i\}_{i=1}^N\):

```python
def pde_residual(net, x_points, D, lam, mu):
    """Compute Navier-Cauchy residual using D_δ for Laplacian + AD for grad-div.

    Args:
        net: callable (N, 2) -> (N, 2)
        x_points: (N, 2) spatial coordinates
        D: SparseD or DenseD operator
        lam, mu: non-dimensional Lamé constants

    Returns:
        r_x, r_y: (N,) residual components
    """
    # Network prediction
    u = jax.vmap(net, in_axes=(0, None))(x_points, jnp.array(0.0))  # (N, 2)
    u_x, u_y = u[:, 0], u[:, 1]

    # Nonlocal Laplacian (per component)
    D_ux = D @ u_x   # (N,)
    D_uy = D @ u_y   # (N,)

    # Gradient of divergence via per-point Hessian AD
    def grad_div_single(x):
        ...

    grad_div_vals = jax.vmap(lambda x: grad_div(lambda xx: net(xx, 0.0), x))(x_points)
    gd_x = grad_div_vals[:, 0]  # (N,)
    gd_y = grad_div_vals[:, 1]  # (N,)

    # Residual: -μ D_δ u + (λ+μ) ∇(∇·u) = 0
    r_x = -mu * D_ux + (lam + mu) * gd_x
    r_y = -mu * D_uy + (lam + mu) * gd_y

    return r_x, r_y
```

**Choice of loss norm**:  L1 (MAE) for PDE residual, following the existing
examples.  L2 (MSE) for BC to strongly penalise boundary violations.

### 5.3 PDE Residual (Local / All-AD Baseline)

```python
def pde_residual_local(net, x_points, lam, mu):
    """All-autograd Navier-Cauchy residual."""
    def navier_single(x):
        u = net(x, jnp.array(0.0))  # (2,)
        hess_ux = jax.hessian(lambda xx: net(xx, 0.0)[0])(x)  # (2, 2)
        hess_uy = jax.hessian(lambda xx: net(xx, 0.0)[1])(x)  # (2, 2)
        lap_ux = jnp.trace(hess_ux)
        lap_uy = jnp.trace(hess_uy)
        gd_x = hess_ux[0, 0] + hess_uy[0, 1]
        gd_y = hess_ux[1, 0] + hess_uy[1, 1]
        r_x = mu * lap_ux + (lam + mu) * gd_x
        r_y = mu * lap_uy + (lam + mu) * gd_y
        return jnp.array([r_x, r_y])

    r = jax.vmap(navier_single)(x_points)  # (N, 2)
    return r[:, 0], r[:, 1]
```

### 5.4 Loss Factory

Follows the established pattern from icemelting3d and electropolish2d:

```python
def make_loss_fn(x_all, D, x_bc, u_bc, nd_params,
                 gradnorm_cap=1000.0, alpha=0.9, local_mode=False):
    """Factory returning a loss_fn(trainables, ...) -> (total, aux, grad, w).

    Steady-state → no t_batch, no IC.
    """
    N_LOSS = 3 if local_mode else 2  # PDE + BC [+ Neumann for local]

    def loss_fn(net, w_ema, bias, update_w,
                x_points=None, sub_data=None, core_mask=None):
        ...
        return total, aux, total_grad, new_w
    return loss_fn
```

**Key differences from existing examples**:
- No time dimension → no `t_batch`, no `jvp`, no `vmap` over time
- No IC → one fewer loss term
- Both modes share the same Dirichlet BC loss on outer edges (displacement-controlled loading)
- Local mode adds one extra term: Neumann BC loss on the **hole boundary only**
- `core_mask` used to compute PDE loss only on core (interior) points

### 5.5 GradNorm Weighting

Same pattern as existing examples: compute per-term gradients separately,
normalise via GradNorm, EMA-smooth the weights:

```python
w_new = grad_norm_weights([pde_grad, bc_grad])  # (2,)
w = alpha * w_new + (1 - alpha) * w_ema
w = w * bias
total = w[0] * pde_loss + w[1] * bc_loss
```

---

## 6. Training Strategy

### 6.1 Optimizer

| Setting | Value |
|---------|-------|
| Optimizer | SOAP (or Adam for smaller runs) |
| Learning rate | \(5 \times 10^{-3}\) initial, exponential decay (0.95 every 1000 steps) |
| Gradient clipping | 100.0 |
| Precision | `"highest"` (exact FP32 for D_δ matvec accuracy) |

### 6.2 Spatial Minibatch

For meshes larger than ~2000 points, use the existing `SparseD.sub_sample`
mechanism (core + halo pattern):

```python
# Pre-build sub-graph pool (same API as icemelting3d)
subgraph_pool = D.build_subgraph_pool(K_core=500, n_pool=200)

# Per step: cycle through pool entries
x_points, sub_data, core_mask = subgraph_pool[step % len(subgraph_pool)]
```

### 6.3 Training Schedule

| Phase | Steps | Description |
|-------|-------|-------------|
| Warm-up | 0–500 | High LR, GradNorm active |
| Main | 500–5000 | Decaying LR, focus on PDE residual |
| Fine-tuning | 5000–10000 | Low LR, refine BC accuracy |

### 6.4 Evaluation Metrics

Computed every `eval_every` steps:

| Metric | Description |
|--------|-------------|
| `rel_L2_u` | \(\|\mathbf{u}_{\text{pred}} - \mathbf{u}_{\text{Kirsch}}\|_2 / \|\mathbf{u}_{\text{Kirsch}}\|_2\) |
| `rel_L2_sigma` | \(\|\sigma_{\text{pred}} - \sigma_{\text{Kirsch}}\|_2 / \|\sigma_{\text{Kirsch}}\|_2\) |
| `SCF` | \(\max_{\theta} \sigma_{\theta\theta}(a, \theta) / \sigma_\infty\) at hole boundary (target: 3.0) |
| `loss/pde` | PDE residual MAE |
| `loss/bc` | Dirichlet BC MSE |
| `loss/neumann` | Neumann BC MSE (local PINN only, should → 0) |

---

## 7. Analytical Solution (Kirsch, 1898)

### 7.1 Stress Field (Polar Coordinates)

For an infinite plate with a circular hole of radius \(a\) under far-field
uniaxial tension \(\sigma_\infty\) in the \(x\)-direction:

\[
\begin{aligned}
\sigma_{rr}(r,\theta) &= \frac{\sigma_\infty}{2}\left[
    \left(1 - \frac{a^2}{r^2}\right) +
    \left(1 - 4\frac{a^2}{r^2} + 3\frac{a^4}{r^4}\right)\cos 2\theta
\right] \\[4pt]
\sigma_{\theta\theta}(r,\theta) &= \frac{\sigma_\infty}{2}\left[
    \left(1 + \frac{a^2}{r^2}\right) -
    \left(1 + 3\frac{a^4}{r^4}\right)\cos 2\theta
\right] \\[4pt]
\sigma_{r\theta}(r,\theta) &= -\frac{\sigma_\infty}{2}\left[
    1 + 2\frac{a^2}{r^2} - 3\frac{a^4}{r^4}
\right]\sin 2\theta
\end{aligned}
\]

**Stress concentration**: at the hole edge (\(r = a\), \(\theta = \pm\pi/2\)),
\(\sigma_{\theta\theta} = 3\sigma_\infty\) → SCF = 3.

### 7.2 Displacement Field (Plane Strain)

\[
\begin{aligned}
u_r(r,\theta) &= \frac{\sigma_\infty}{4\mu}\Bigg[
    r\left(\frac{\kappa-1}{2} + \cos 2\theta\right) +
    \frac{a^2}{r}\left[1 + (1+\kappa)\cos 2\theta\right] -
    \frac{a^4}{r^3}\cos 2\theta
\Bigg] \\[4pt]
u_\theta(r,\theta) &= \frac{\sigma_\infty}{4\mu}\Bigg[
    -r\sin 2\theta -
    \frac{a^2}{r}(\kappa-1)\sin 2\theta -
    \frac{a^4}{r^3}\sin 2\theta
\Bigg]
\end{aligned}
\]

where \(\kappa = 3 - 4\nu\) (plane strain).

### 7.3 Cartesian Conversion

For implementation convenience, convert to Cartesian:

\[
\begin{aligned}
u_x &= u_r \cos\theta - u_\theta \sin\theta \\
u_y &= u_r \sin\theta + u_\theta \cos\theta \\[4pt]
\sigma_{xx} &= \sigma_{rr}\cos^2\theta + \sigma_{\theta\theta}\sin^2\theta
               - 2\sigma_{r\theta}\sin\theta\cos\theta \\
\sigma_{yy} &= \sigma_{rr}\sin^2\theta + \sigma_{\theta\theta}\cos^2\theta
               + 2\sigma_{r\theta}\sin\theta\cos\theta \\
\sigma_{xy} &= (\sigma_{rr} - \sigma_{\theta\theta})\sin\theta\cos\theta
               + \sigma_{r\theta}(\cos^2\theta - \sin^2\theta)
\end{aligned}
\]

### 7.4 Finite Plate Correction

The Kirsch solution assumes an **infinite** plate.  For a finite plate
with \(L/a \ge 10\), the error in the stress field near the hole is
\(< 2\%\).  For higher accuracy, the analytical displacement field can
be applied as BC on the outer edges — the interior solution will converge
to the correct finite-plate result.

---

## 8. Validation Plan

### 8.1 Unit Tests

1. **dV correctness**: compare `barycentric_dV` against hand-computed
   areas for a known mesh (e.g., 2-triangle square).
2. **D_δ row-sum**: verify `D @ jnp.ones(N) ≈ 0` (to machine precision
   for interior points, within O(δ) for boundary points).
3. **Analytical residual**: verify the Kirsch solution satisfies the
   Navier-Cauchy equation to within truncation error.

### 8.2 Integration Tests

1. **Smoke test**: 100 training steps, verify loss decreases monotonically.
2. **SCF convergence**: train to 5000+ steps, verify SCF at hole
   converges to ≈ 3.0.
3. **Local vs nonlocal**: run both on the same mesh for 5000 steps each,
   compare:
   - Final relative L2 error in displacement
   - Final SCF accuracy
   - Wall-clock time per step
   - Number of loss terms and tuning burden

### 8.3 Expected Results

With displacement-controlled loading on outer edges, both methods share
the same Dirichlet BC loss.  The difference is solely in how the **hole
boundary** is handled:

| Metric | Nonlocal D_δ | Local AD |
|--------|-------------|----------|
| Loss terms | 2 (PDE + Dirichlet BC) | 3 (PDE + Dirichlet BC + Neumann BC on hole) |
| Hole boundary | Automatic (horizon truncation) | Explicit penalty \(\|\boldsymbol{\sigma}\cdot\mathbf{n}\|^2\) |
| Neumann BC tuning | None needed | Weight schedule required (bias early, ramp later) |
| rel_L2_u | < 2% | < 3% (tuning-dependent) |
| SCF | 2.8–3.2 | 2.5–3.5 (strongly tuning-dependent) |
| Per-step time | Faster (D_δ matvec for Laplacian) | Slower (per-point Hessian AD for all second derivatives) |
| Outer edge BC | Same Dirichlet loss for both | Same Dirichlet loss for both |
| Per-step time | Faster (no AD Laplacian) | Slower (per-point Hessian) |

---

## 9. File Structure

```
examples/elasticity2d/
├── README.md              # This file
├── configs/
│   └── train_base.py      # Training hyperparameter config
├── train.py               # Training script
├── losses.py              # PDE residual, BC losses, loss factory
├── analytical.py          # Kirsch analytical solution
├── mesh.geo               # Gmsh geometry script
└── mesh.msh               # Pre-generated mesh (committed for convenience)
```

### 9.1 `analytical.py`

```python
"""Kirsch analytical solution for a plate with a circular hole.

All functions work in non-dimensional coordinates (plate size = 1).
"""

def kirsch_stress(x, y, a, sigma_inf=1.0, nu=0.3):
    """Stress tensor at point(s) (x, y). Hole centred at origin.
    Returns: sigma_xx, sigma_yy, sigma_xy
    """
    ...

def kirsch_displacement(x, y, a, sigma_inf=1.0, E=1.0, nu=0.3):
    """Displacement at point(s) (x, y). Plane strain.
    Returns: u_x, u_y
    """
    ...
```

### 9.2 `losses.py`

See §5 for detailed design.  Exports:
- `pde_residual(net, x, D, lam, mu)` — nonlocal PDE
- `pde_residual_local(net, x, lam, mu)` — all-AD PDE
- `loss_bc_dirichlet(net, x_bc, u_bc)` — Dirichlet BC loss (outer edges, both modes)
- `loss_bc_neumann_hole(net, x_hole, normals, lam, mu)` — Neumann BC on hole (local mode only)
- `grad_norm_weights(grads, cap)` — GradNorm
- `make_loss_fn(...)` — loss factory

### 9.3 `configs/train_base.py`

See §6 for parameter values.  Main config structure:

```python
def get_config():
    c = ConfigDict()
    c.arch = ...    # architecture (type, dims, embedding)

    c.nl = ...      # nonlocal (delta, kernel, method)
    c.sample = ...  # spatial sampling (N_mesh, K_core, K_pool)
    c.train = ...   # training (iterations, lr, optimizer)
    c.loss = ...    # loss weights (gradnorm bias, alpha)
    c.log = ...     # logging (log_dir, log_every, eval_every)
    c.physical = ... # E, nu, sigma_inf
    c.geometry = ... # L, a, mesh_file
    return c
```

### 9.4 `train.py`

Follows the electropolish2d pattern (§2 of the exploration report).
Steady-state adaptations:
- No `t_batch` sampling — always evaluate at `t = 0.0`
- No causal weighting (only 1 time step)
- `_bc_info(config)` samples outer-edge points + evaluates Kirsch
  displacement
- `_eval_analytical(net, ...)` computes relative L2 errors and SCF

---

## 10. Usage

### 10.1 Generate Mesh

```bash
# Install Gmsh
apt-get install -y gmsh

# Generate mesh from .geo script
cd examples/elasticity2d
gmsh -2 mesh.geo -o mesh.msh -format msh2
```

### 10.2 Train Nonlocal PINN

```bash
# From project root
python examples/elasticity2d/train.py \
  --config=examples/elasticity2d/configs/train_base.py \
  --config.log.run_name=nonlocal_run
```

### 10.3 Train Local PINN (Baseline)

```bash
python examples/elasticity2d/train.py \
  --config=examples/elasticity2d/configs/train_base.py \
  --config.nl.method=local \
  --config.log.run_name=local_baseline
```

### 10.4 Compare Results

```bash
# In TensorBoard
tensorboard --logdir=/root/autodl-tmp/tf-logs/elasticity2d/
```

---

## 11. Extension: PINO with Parameterised Geometry

A natural next step is to upgrade from a single-geometry PINN to a
**PINO (Physics-Informed Neural Operator)** that learns the solution
operator for a *family* of geometries — e.g., varying hole radius \(a\),
hole position \((x_c, y_c)\), or hole shape (ellipse aspect ratio,
multiple holes).  The key design question:

> How to represent varying geometry so the same network works across
> all shapes?

### 11.1 Two Approaches

#### Option A: Mask with weight-zeroing (fixed grid + shared COO)

Use a **fixed regular grid** covering the full square domain.  Each
geometry is encoded as a binary mask \(m(\mathbf{x}) \in \{0, 1\}\)
where \(m = 0\) inside the hole and \(m = 1\) in the solid.

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

**Key insight — share the COO sparsity, zero the weights**:  Build
\(D_\delta\) **once** on the full rectangle.  For each geometry, scale
edge weights by the product of endpoint masks.  Edges crossing into the
hole get zero weight but stay in the COO arrays — the sparsity pattern
never changes.

```python
# ═══ Built once, shared across ALL geometries ═══
D_full = build_D(x_full, delta)     # SparseD on full rectangle
row_idx = D_full._row_idx           # (nnz,) — shared
col_idx = D_full._col_idx           # (nnz,) — shared
J_dV    = -D_full._vals             # (nnz,) base weights

# ═══ Per geometry: scale weights by mask ═══
def apply_mask(mask):
    """mask: (N,) float, m ∈ {0, 1}"""
    m_edge = mask[row_idx] * mask[col_idx]   # (nnz,) — 0 if either end in hole
    vals_m = -J_dV * m_edge                   # zero out cross-hole edges
    diag_m = mask * jax.ops.segment_sum(
        J_dV * m_edge, row_idx, num_segments=N)  # diag also masked
    return vals_m, diag_m
```

Effect of mask on different edge types:

```
边类型                      mask[i]·mask[j]   效果
───────────────────────────────────────────────────
i, j 都在固体内部              1·1 = 1        权重不变，正常传消息
i 在固体，j 在孔内             1·0 = 0        权重归零 → horizon 被截断
i 在孔内，j 任意               0·* = 0        孔内点不参与计算
```

孔边界自动成为 traction-free 面——和物理挖孔相同机制，但 COO index 数组不变。
Sub-graph pool 也完全共享：`mask_local = mask[all_idx]`，同理缩放 `sub_vals`。

**Memory comparison**:

| 方案 | 每个几何存储 | 1000 个几何总存储 |
|------|------------|-----------------|
| Prune 边 (每个几何不同 COO) | COO row, col, vals ~2 MB | ~2 GB |
| **权重置零 (共享 COO)** | mask `(N,)` ~4 KB | **~4 MB** |

**Network input**: concatenate `[x, y, m(x,y), a]` where `a` is the
geometry parameter.  A geometry-embedding MLP can encode `a` into a
latent vector that modulates the trunk (FiLM-style).

| Pros | Cons |
|------|------|
| D_δ + sub-graph pool 全几何共享 | 孔边界锯齿状 (网格分辨率) |
| 每个几何只存 (N,) mask (~4KB) | O(h) 几何误差 |
| 同一套点集 → 易 batch，易 JIT | 网格需足够细才能分辨最小特征 |
| 天然支持拓扑变化 (多孔) | |

#### Option B: Coordinate mapping (reference domain + deformation)

Define a **reference domain** \(\Omega_0\) (e.g., unit square with a
fixed unit hole) and map each physical geometry via a smooth
coordinate transformation \(\boldsymbol{\phi}: \Omega_0 \to \Omega_a\).

```
Reference domain Ω₀:                Physical domain Ω_a:
┌──────────────────┐                ┌──────────────────┐
│                  │                │                  │
│    ╭───╮         │   φ(x,y; a)   │    ╭─────╮       │
│    │ ○ │         │  ==========>  │    │ hole│       │
│    ╰───╯         │               │    │ r=a │       │
│                  │                │    ╰─────╯       │
└──────────────────┘                └──────────────────┘
```

The PDE is **transformed** to the reference domain, introducing metric
terms from the Jacobian of \(\boldsymbol{\phi}\).  For the Navier-Cauchy
equation this adds first-order derivative terms (the Laplacian in
curvilinear coordinates includes Christoffel-like corrections).

**For the nonlocal operator**, there are two sub-options:

- **B1: Transform D_δ to reference space**.  Build \(D_\delta\) on the
  physical points \(\boldsymbol{\phi}(\mathbf{x}_0)\), then pull back the
  operator.  The kernel weights \(J(\|\mathbf{x}_i - \mathbf{x}_j\|)\)
  depend on physical distances, which change with geometry — so D_δ must
  be rebuilt per geometry.  This negates much of the benefit.

- **B2: Build D_δ in reference space, transform the PDE**.  Build one
  fixed \(D_\delta\) on the reference point set, and absorb the metric
  factors into the PDE residual.  The Laplacian in physical space
  \(\nabla^2_{\mathbf{x}}\) is expressed as a combination of reference
  derivatives with metric coefficients.  These metric coefficients
  become additional inputs to the network.

| Pros | Cons |
|------|------|
| Smooth hole boundary — no staircasing | PDE gets metric terms (chain rule explosion) |
| One mesh → one D_δ (B2) | D_δ still approximates reference Laplacian, not physical |
| Mathematically clean | Complex for large deformations (mapping singularities) |
| | Mapping must be defined analytically per geometry family |

### 11.2 Recommendation: Mask with Weight-Zeroing

For this example, the **mask + weight-zeroing approach (Option A)** is the
pragmatic choice:

1. **Alignment with D_δ's strengths**:  The nonlocal operator handles free
   boundaries via horizon truncation.  Mask-zeroing is the algebraic
   equivalent — `mask[i]*mask[j] = 0` is functionally identical to "no
   neighbour outside the domain."

2. **Maximal sharing**:  The COO edge index arrays (`row_idx`, `col_idx`),
   the base weights (`J·dV`), and the sub-graph pool are built once and
   shared across all geometries.  Only the `(N,)` mask vector differs.

3. **Memory**:  1000 geometries → 1000 × N × 4 bytes.  For N = 16384
   (128²), that's ~65 MB total — negligible.

4. **Resolution concern**:  The staircasing error is \(O(h)\).  For a
   \(128 \times 128\) grid and hole radius \(a = 0.1L\), the circumference
   spans ~80 grid points → ~1% geometric error.  Acceptable for PoC.

5. **Coordinate mapping can be added later** (Option B) for geometries
   needing smooth boundaries.

### 11.3 PINO Architecture Sketch (Mask + Weight-Zeroing)

```
Shared (built once):
  x_full (N, 2)          — fixed regular grid covering full rectangle
  D_full = build_D(x_full, delta)  — COO arrays (row_idx, col_idx, J·dV)
  subgraph_pool           — pre-built shared pool

Per geometry (sampled each step):
  a ~ U(a_min, a_max)     — geometry parameter (hole radius)
  mask = hole_mask(x_full, a)  — (N,) binary, m=0 inside hole

Per-step computation:
  vals_m, diag_m = apply_mask(mask)   # scale edge weights by mask
  u = net(x_full, a, mask)            # network forward
  D_u = segment_sum(vals_m * u[col], row_idx) + diag_m * u
  loss = pde_loss(...) + bc_loss(...)  # PDE only on mask==1 points

Network input:
  spatial_emb = FourierEmbedding(x_full)    → (N, 2*emb_dim)
  mask_channel = mask[:, None]              → (N, 1)
  geom_emb = MLP_geom(a)                    → (geom_dim,)
  trunk_input = concat(spatial_emb, mask_channel, tile(geom_emb))
              → FiLM trunk with geom_emb conditioning
              → (N, 2)  (u_x, u_y)
```

### 11.4 Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Grid | Fixed regular \(N_x \times N_y\) | Avoid per-geometry meshing |
| Mask encoding | Binary channel concatenated to spatial features | Allows network to learn boundary proximity |
| Geometry param encoding | Small MLP → latent vector → FiLM modulation | Decouples geometry from spatial processing |
| D_δ per geometry | **Shared COO, weights × mask** | One build, one sub-graph pool; mask-only per geometry (~4KB) |
| Training data | Randomly sample \(a \sim U(a_{\min}, a_{\max})\) each step | On-the-fly generation, no precomputed dataset |

### 11.5 Summary: What Is Shared vs Per-Geometry

| Resource | Shared across geometries | Per-geometry |
|----------|------------------------|--------------|
| Spatial points `x` | (N, 2) full grid | — |
| COO indices `row_idx, col_idx` | (nnz,) int32 | — |
| Base weights `J·dV` | (nnz,) float32 | — |
| Sub-graph pool | K_pool entries (indices only) | — |
| Edge weights `vals` | — | × mask[row]·mask[col] |
| Diagonal `diag` | — | × mask + segment_sum |
| Mask vector | — | (N,) float32 (~4 KB) |

---

## 12. References

1. Kirsch, G. (1898). "Die Theorie der Elastizität und die Bedürfnisse
   der Festigkeitslehre." *Zeitschrift des Vereines deutscher Ingenieure*,
   42, 797–807.
2. Silling, S.A. (2000). "Reformulation of elasticity theory for
   discontinuities and long-range forces." *Journal of the Mechanics and
   Physics of Solids*, 48(1), 175–209.  (Peridynamic formulation of
   elasticity)
3. Bobaru, F. et al. (2016). *Handbook of Peridynamics*. CRC Press.
   (Chapter 2: surface correction methods)
4. Timoshenko, S.P. & Goodier, J.N. (1970). *Theory of Elasticity*,
   3rd ed. McGraw-Hill.  (§4.8: Kirsch problem)
