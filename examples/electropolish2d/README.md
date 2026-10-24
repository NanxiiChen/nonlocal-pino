# 2D KKS Phase-Field Electropolishing — PINN + FEM Reference

## Motivation

This example is designed as a **killer ablation** for the nonlocal operator
$D_\delta$ vs finite-difference (FD) / local AD. The left/right boundaries
have **Neumann (zero-flux) BCs** — no explicit Dirichlet constraints, no BC
loss terms. The nonlocal operator handles Neumann BCs automatically (simply by
not sampling *outside* the domain), while FD must truncate its stencil and a
local PINN must explicitly penalise $\partial\phi/\partial n$ and
$\partial c/\partial n$ in the loss function.

The initial interface is a **rough, randomly-generated surface** described by
a signed-distance function (SDF) to a piecewise-linear profile.

## Physical Setup

A metal workpiece is held between **insulating (inert) fixtures** on the left
and right sides; only its top surface is exposed to the electrolyte.  The
fixture walls are electrochemically inactive — no reaction, no current.

```
                   electrolyte  φ=0, c=0 (top, Dirichlet)
  ┌─────────────────────────────────────────────┐
  │                                             │
  │  electrolyte   φ≈0,  c≈0                   │ ← inert wall
  ├──────── rough interface ~~~~~~~~~~~~~~~~~~~─┤   ∂φ/∂n = 0
  │                                             │   ∂c/∂n = 0
  │  solid metal   φ≈1,  c≈1                   │ ← inert wall
  │                                             │
  └─────────────────────────────────────────────┘
                   solid metal  φ=1, c=1 (bottom, Dirichlet)
```

- **Top** ($y=0.5$): liquid electrolyte, $\phi=0$, $c=0$
- **Bottom** ($y=0$): solid metal, $\phi=1$, $c=1$
- **Left / Right** ($x=0$, $x=1$): inert insulating fixtures, $\partial\phi/\partial n=0$, $\partial c/\partial n=0$
- **Interface**: initial rough surface at $y\approx 0.33$; metal dissolves at the interface

The Neumann (zero-flux) condition on the side walls is physically justified:
the insulating fixture does not participate in the electrochemical reaction,
so there is no flux of either the phase-field or the concentration field across
the fixture surface.  This is the standard inert-wall assumption used in
phase-field corrosion and electropolishing models.

### Why nonlocal wins here

For a local PINN (standard autograd) or an FD-based discretisation, the Neumann
condition on the side walls must be enforced *explicitly* — by sampling points
on $x=0$ and $x=1$ and adding a penalty term $\|\partial\phi/\partial n\|^2 +
\|\partial c/\partial n\|^2$ to the loss.  This creates a structural conflict:

1. The **initial condition** (rough SDF profile) does **not** satisfy
   $\partial\phi/\partial n = 0$ on the side walls — the rough interface meets
   the walls at an angle.
2. The **Neumann loss** demands $\partial\phi/\partial n \to 0$ immediately.

The optimiser is caught between two competing objectives at early time: the IC
loss pulls one way, the Neumann loss pulls another.  Mitigating this requires
careful weight scheduling (bias IC early, ramp Neumann later), adding tuning
overhead and sacrificing boundary accuracy near $t=0$.

The **nonlocal operator $D_\delta$** avoids this entirely.  It is constructed
from neighbour pairs *inside* the domain only — there is no stencil outside,
no explicit boundary condition, and no BC loss term.  The condition
$\partial/\partial n = 0$ emerges automatically from the fact that points on
the boundary only interact with interior neighbours.  The IC is free to take
any shape; $D_\delta$ itself guarantees the zero-flux condition without ever
asking the optimiser to enforce it.  **The boundary condition and the initial
condition never conflict.**

## Governing Equations (Local / Classical)

2D KKS phase-field model for electropolishing. Domain: $\Omega = [0,1] \times [0,0.5]$ (non-dimensional).

With $\Delta c_0 = c_{Se} - c_{Le}$ and
$q = c - h(\phi)\Delta c_0 - c_{Le}$, the free-energy density is

$$f = A q^2 + \omega_\phi g(\phi) + \frac{\alpha_\phi}{2}|\nabla\phi|^2.$$

**Allen-Cahn:**

$$
\frac{\partial \phi}{\partial t}
= L\left[\alpha_\phi \nabla^2\phi
- \omega_\phi g'(\phi)
+ 2Aq\Delta c_0 h'(\phi)\right]
$$

**Cahn-Hilliard:**

$$\frac{\partial c}{\partial t} = D\,\nabla^2\!\left[c - h(\phi)\Delta c_0 - c_{Le}\right]$$

The constant $c_{Le}$ is inert under the CH Laplacian, but remains part of
$q$ in the Allen-Cahn driving force.

**Interpolation functions:**

$$h(\phi) = -2\phi^3 + 3\phi^2,\quad g(\phi) = \phi^2(1-\phi)^2$$

### Geometry & Boundary Conditions

```
y=0.5  ┌─────────────────────────┐  Top: φ=0, c=0 (liquid electrolyte)
       │                         │
       │   domain                │  Left/Right: Neumann (∂φ/∂n=0, ∂c/∂n=0)
       │                         │  → no BC loss terms — key ablation point!
       │  ~~~~ rough ~~~~        │  Initial interface (SDF-based tanh)
       │                         │
y=0    └─────────────────────────┘  Bottom: φ=1, c=1 (solid metal)
```

| Boundary | Condition |
|---|---|
| Top ($y=0.5$) | $\phi = 0,\; c = 0$ (liquid electrolyte) |
| Bottom ($y=0$) | $\phi = 1,\; c = 1$ (solid metal) |
| Left ($x=0$) | $\partial\phi/\partial n = 0,\; \partial c/\partial n = 0$ (Neumann) |
| Right ($x=1$) | $\partial\phi/\partial n = 0,\; \partial c/\partial n = 0$ (Neumann) |

**Initial condition** — SDF-based tanh profile on rough surface:

$$y_{\text{surf}}(x) = y_0 + \sum_{i=1}^{n_\text{modes}} a_i \bigl[\sin(2\pi f_i x + \theta_i) + \cos(2\pi f_i x + \theta_i + 0.3)\bigr]$$

$$\phi_0(x,y) = \frac12\bigl[1 + \tanh(k \cdot \text{sdf}(x,y))\bigr],\quad c_0 = h(\phi_0) \cdot c_{Se}$$

where $k = \sqrt{\omega_\phi / (2\alpha_\phi)}\cdot L_c$, and the SDF is the signed perpendicular distance to the piecewise-linear profile $y_{\text{surf}}(x)$ (positive below / solid side).

---

## Non-Dimensionalisation

Characteristic length $L_c = 100~\mu\text{m}$ and time $T_c$ define:

$$\bm{x}^* = \frac{\bm{x}}{L_c},\qquad t^* = \frac{t}{T_c}$$

Four non-dimensional parameters:

$$N_{\text{CH}} = \frac{D T_c}{L_c^2},\quad N_{\text{AC,1}} = 2A L T_c,\quad N_{\text{AC,2}} = L\omega_\phi T_c,\quad N_{\text{AC,3}} = \frac{L\alpha_\phi T_c}{L_c^2}$$

### Non-dimensional Allen-Cahn

$$\frac{\partial\phi}{\partial t^*} - N_{\text{AC,3}}\,\nabla^{*2}\phi + N_{\text{AC,2}}\,g'(\phi) - N_{\text{AC,1}}\bigl[c - h(\phi)(c_{Se}-c_{Le}) - c_{Le}\bigr](c_{Se}-c_{Le})h'(\phi) = 0$$

### Non-dimensional Cahn-Hilliard

$$\frac{\partial c}{\partial t^*} + N_{\text{CH}}\,\nabla^{*2}\!\bigl[c - h(\phi)(c_{Se}-c_{Le}) - c_{Le}\bigr] = 0$$

with $\Delta c_0 = c_{Se} - c_{Le}$.

---

## Nonlocal (Peridynamics-Inspired) Formulation

Replacing $\nabla^{*2}$ with the nonlocal operator $D_\delta$:

$$\frac{\partial\phi}{\partial t^*} + N_{\text{AC,3}}\,D_\delta\phi + N_{\text{AC,2}}\,g'(\phi) - N_{\text{AC,1}}\bigl[c - h(\phi)\Delta c_0 - c_{Le}\bigr]\Delta c_0\,h'(\phi) = 0$$

$$\frac{\partial c}{\partial t^*} + N_{\text{CH}}\,D_\delta\!\bigl[c - h(\phi)\Delta c_0 - c_{Le}\bigr] = 0$$

In the local limit $D_\delta \approx -\nabla^{*2}$. The nonlocal operator is defined by a compactly-supported kernel $J(r)$:

$$(D_\delta u)(x_i) = \sum_{j \in N_\delta(i)} w_{ij}\,(u_i - u_j),\quad w_{ij} = -D_{ij}$$

Edge weights pre-computed from the `SparseD` operator. Supported kernels: `cubic`, `quartic`, `wendland`.

### Finite Difference (Ablation)

Setting `--config.nl.method=fd` switches to a 5-point FD Laplacian on the structured grid. The FD stencil is truncated at left/right Neumann boundaries, while the nonlocal operator requires no boundary handling.

---

## Physical Parameters

| Symbol | Value | Unit | Description |
|---|---|---|---|
| $D$ | $8.5 \times 10^{-10}$ | m²/s | Diffusion coefficient |
| $A$ | $5.35 \times 10^7$ | J/(m³·s) | Reaction constant |
| $L$ | $1.0 \times 10^{-6}$ | m/(J·s) | Interface mobility |
| $\alpha_\phi$ | $1.03 \times 10^{-4} / 4$ | J/m | Gradient energy coefficient |
| $\omega_\phi$ | $1.76 \times 10^7$ | J/(m³·s) | Barrier height |
| $c_{Se}$ | 1.0 | — | Equilibrium concentration (solid) |
| $c_{Le}$ | $5100 / 1.43 \times 10^5$ | — | Equilibrium concentration (liquid) |

### Non-Dimensionalisation

| Scale | Value | Description |
|---|---|---|
| $L_c$ | $100~\mu\text{m}$ | Characteristic length |
| $T_c$ | 5.0 s | Characteristic time |

---

## Rough Surface Configuration

| Parameter | Default | Description |
|---|---|---|
| `y0` | 0.33 | Mean interface height |
| `n_modes` | 6 | Number of Fourier modes |
| `amplitude` | 0.03 | Max amplitude of roughness |
| `freq_lo` | 1.0 | Min spatial frequency |
| `freq_hi` | 5.0 | Max spatial frequency |
| `seed` | 42 | RNG seed for reproducibility |

The signed distance is computed via the perpendicular distance to each segment of the piecewise-linear profile, using a sliding window (`window=16`) over query points for efficiency.

---

## Project Structure

```
electropolish2d/
├── solver.py               # FEM reference solver (FEniCS, MPI)
├── train.py                 # PINN training (MLP / FiLM)
├── losses.py                # Loss functions (AC, CH, BC, IC)
├── configs/
│   ├── solver_base.py       # FEM solver config
│   └── train_base.py        # PINN training config
├── README.md
```

---

## Usage

### Run FEM Solver

```bash
conda activate fenicsenv
cd examples/electropolish2d

# Serial
python solver.py --config=configs/solver_base.py

# MPI (18 processes)
mpirun --allow-run-as-root -np 18 python solver.py --config=configs/solver_base.py
```

Output: `.npz` file at `/root/autodl-tmp/data/electropolish2d/results_rough.npz` with `X`, `Y` meshgrids, `time` array, and `phi`, `c` fields over time. Also saves snapshot plots (`snapshot_phi.png`, `snapshot_c.png`).

### Train PINN

```bash
# From project root
python examples/electropolish2d/train.py --config=examples/electropolish2d/configs/train_base.py

# FD ablation
python examples/electropolish2d/train.py --config.nl.method=fd --config.log.run_name=fd_run

# Spatial minibatch
python examples/electropolish2d/train.py --config.sample.K_core=400 --config.sample.K_pool=400
```

### Key Config Options

```bash
# Architecture
--config.arch.type=film          # or "mlp"
--config.arch.hidden_dim=64
--config.arch.spatial_scale=2.0   # single scale or tuple, e.g. (1.0, 3.0, 6.0)

# Nonlocal operator
--config.nl.delta=0.02
--config.nl.kernel=cubic
--config.nl.method=sparse         # "sparse" | "fd"

# Spatial minibatch (core+halo sub-sampling)
--config.sample.K_core=400        # 0 = full grid
--config.sample.K_pool=400

# IC sampling
--config.sample.N_ic_global=2000
--config.sample.N_ic_local=2000   # oversampling near interface

# Staggered training
--config.loss.stagger_steps=1000
--config.loss.stagger_switch_every=25

# Hard constraint (enforce c bounds)
--config.arch.use_hard_constraint=True

# Causal training
--config.loss.use_causal=True
--config.causal.n_chunks=0
```

### Key Design Decisions

- **Left/Right Neumann BCs are NOT enforced via loss terms** — this is the central ablation that distinguishes nonlocal (handles Neumann naturally) from FD (requires truncated stencil)
- **Rough initial interface** is defined by SDF to a piecewise-linear profile, not simple y-distance, for smooth $\phi$ at t=0
- **IC oversampling** near the interface (`N_ic_local`) ensures the sharp tanh transition is adequately resolved
- **Hard constraint** (`use_hard_constraint=True`) enforces $c \in [c_{Le}, c_{Se}]$ via sigmoid rescaling, improving physical consistency


## Project split (2026-10-24)

This example belongs to `nonlocal-pinn`. Install the project and run from its
repository root; existing external data/output paths are retained. Shared
implementation paths changed during the split; see
[the source map](../../docs/migration/source-map.json) and
[migration notes](../../docs/migration/README.md).
