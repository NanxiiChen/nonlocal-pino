# 2D KKS Phase-Field Corrosion — PINN + FEM Reference

## Governing Equations (Local / Classical)

2D KKS phase-field model for pit corrosion. Domain: $\Omega = [0,1] \times [0,0.5]$ (non-dimensional).

**Allen-Cahn:**

$$\frac{\partial \phi}{\partial t} = -L\left[\alpha_\phi \nabla^2\phi - \omega_\phi g'(\phi) - A (c - h(\phi)\Delta c_0 - c_{Le}) h'(\phi)\right]$$

**Cahn-Hilliard:**

$$\frac{\partial c}{\partial t} = D\,\nabla^2\!\left[c - h(\phi)\Delta c_0 - c_{Le}\right]$$

**Interpolation functions:**

$$h(\phi) = -2\phi^3 + 3\phi^2,\quad g(\phi) = \phi^2(1-\phi)^2$$

### Geometry & Boundary Conditions

Pit(s) are semicircular notches on the bottom boundary ($y=0$), defined by centre $(cx, cy)$ and radius $R$.

| Boundary | Condition |
|---|---|
| Pit mouth ($y=0$, $\|x-cx\| < R/2$) | $\phi = 0,\; c = 0$ (active dissolution) |
| $y=0$, elsewhere | $\partial\phi/\partial n = 0,\; \partial c/\partial n = 0$ |
| $x=0,\; x=1,\; y=0.5$ | $\phi = 1,\; c = 1$ (solid electrode) |

**Initial condition** — sharp tanh profile at pit interface:

$$\phi_0(x) = \min_{\text{pits}}\left\{\frac12\left[1 + \tanh(k\cdot(d(x,\text{pit}) - R))\right]\right\},\quad c_0 = h(\phi_0)\cdot c_{Se}$$

where $k = \sqrt{\omega_\phi / (2\alpha_\phi)}\cdot L_c$.

---

## Non-Dimensionalisation

Characteristic length $L_c = 100~\mu\text{m}$ and time $T_c$ define:

$$\bm{x}^* = \frac{\bm{x}}{L_c},\qquad t^* = \frac{t}{T_c}$$

Four non-dimensional parameters:

$$N_{\text{CH}} = \frac{D T_c}{L_c^2},\quad N_{\text{AC,1}} = 2A L T_c,\quad N_{\text{AC,2}} = L\omega_\phi T_c,\quad N_{\text{AC,3}} = \frac{L\alpha_\phi T_c}{L_c^2}$$

### Non-dimensional Allen-Cahn

$$\frac{\partial\phi}{\partial t^*} - N_{\text{AC,1}}\bigl[c - h(\phi)(c_{Se}-c_{Le}) - c_{Le}\bigr](c_{Se}-c_{Le})h'(\phi) + N_{\text{AC,2}}\,g'(\phi) - N_{\text{AC,3}}\,\nabla^{*2}\phi = 0$$

### Non-dimensional Cahn-Hilliard

$$\frac{\partial c}{\partial t^*} - N_{\text{CH}}\,\nabla^{*2}c + N_{\text{CH}}(c_{Se}-c_{Le})\,\nabla^{*2}h(\phi) = 0$$

equivalently:

$$\frac{\partial c}{\partial t^*} = N_{\text{CH}}\,\nabla^{*2}\!\bigl[c - h(\phi)(c_{Se}-c_{Le}) - c_{Le}\bigr]$$

---

## Nonlocal (Peridynamics-Inspired) Formulation

Replacing $\nabla^{*2}$ with the nonlocal operator $D_\delta$ (peridynamics style):

$$\frac{\partial\phi}{\partial t^*} + N_{\text{AC,3}}\,D_\delta\phi + N_{\text{AC,2}}\,g'(\phi) - N_{\text{AC,1}}\bigl[c - h(\phi)\Delta c_0 - c_{Le}\bigr]\Delta c_0\,h'(\phi) = 0$$

$$\frac{\partial c}{\partial t^*} + N_{\text{CH}}\,D_\delta\!\bigl[c - h(\phi)\Delta c_0 - c_{Le}\bigr] = 0$$

where $\Delta c_0 = c_{Se} - c_{Le}$, and in the local limit $D_\delta \approx -\nabla^{*2}$.

The nonlocal operator is defined by a compactly-supported kernel $J(r)$:

$$(D_\delta u)(x_i) = \sum_{j \in N_\delta(i)} w_{ij}\,(u_i - u_j),\quad w_{ij} = -D_{ij}$$

Edge weights pre-computed from SparseD operator. Supported kernels: `cubic`, `quartic`, `wendland`.

---

## Physical Parameters

| Symbol | Value | Unit | Description |
|---|---|---|---|
| $D$ | $8.5 \times 10^{-10}$ | m²/s | Diffusion coefficient |
| $A$ | $5.35 \times 10^7$ | J/(m³·s) | Reaction constant |
| $L$ | 2.0 | m/(J·s) | Interface mobility |
| $\alpha_\phi$ | $1.03 \times 10^{-4}$ | J/m | Gradient energy coefficient |
| $\omega_\phi$ | $1.76 \times 10^7$ | J/(m³·s) | Barrier height |
| $c_{Se}$ | 1.0 | — | Equilibrium concentration (solid) |
| $c_{Le}$ | 5100 / 1.43e5 | — | Equilibrium concentration (liquid) |

### Non-Dimensionalisation

| Scale | 1-pit | 2-pit | Description |
|---|---|---|---|
| $L_c$ | $100~\mu\text{m}$ | $100~\mu\text{m}$ | Characteristic length |
| $T_c$ | 50 s | 15 s | Characteristic time |

---

## Project Structure

```
corrosion2d/
├── solver_1pit.py         # FEM solver (single pit)
├── solver_2pit.py         # FEM solver (two-pit merging)
├── train.py               # PINN training (coordinate MLP / FiLM)
├── train_graph.py         # D-Graph FiLM PINN training (experimental)
├── losses.py              # Loss functions (coordinate MLP)
├── losses_graph.py        # Loss functions (graph model)
├── configs/
│   ├── solver_1pit.py           # FEM solver config (1 pit)
│   ├── solver_2pit.py           # FEM solver config (2 pit)
│   ├── train_1pit.py            # PINN training config (1 pit, FiLM)
│   ├── train_2pit.py            # PINN training config (2 pit, FiLM)
│   ├── train_1pit_graph.py      # Graph model config (1 pit)
│   └── train_2pit_graph.py      # Graph model config (2 pit)
├── results/               # FEM reference solutions (.npz)
└── plots/                 # Output plots
```

---

## Usage

### Run FEM Solver

```bash
# 1-pit
cd examples/corrosion2d
python solver_1pit.py --config=configs/solver_1pit.py

# 2-pit (merging simulation)
python solver_2pit.py --config=configs/solver_2pit.py
```

Output: `.npz` files with `X`, `Y` meshgrids and `phi`, `c` fields over time.

### Train PINN (Coordinate FiLM MLP)

```bash
# 1-pit (from project root)
python examples/corrosion2d/train.py --config=examples/corrosion2d/configs/train_1pit.py

# 2-pit with spatial minibatch
python examples/corrosion2d/train.py --config=examples/corrosion2d/configs/train_2pit.py \
    --config.sample.K_core=1200
```

### Key Config Options

```bash
# Architecture
--config.arch.type=film --config.arch.hidden_dim=64

# Nonlocal operator
--config.nl.delta=0.02 --config.nl.kernel=cubic

# Spatial minibatch (core+halo sub-sampling)
--config.sample.K_core=1200   # 0 = full grid
--config.sample.K_pool=200    # sub-graph pool size

# Staggered training
--config.loss.stagger_steps=10000 --config.loss.stagger_switch_every=25

# Temporal ramp (bias later times for merging problems)
--config.loss.t_ramp_steps=2000 --config.loss.t_ramp_early=0.3 --config.loss.t_ramp_late=3.0

# Causal training (per-t mode)
--config.loss.use_causal=True --config.causal.n_chunks=0
```

### Graph Model (experimental, on `dgraph-pinn` branch)

```bash
git checkout dgraph-pinn
python examples/corrosion2d/train_graph.py --config=examples/corrosion2d/configs/train_2pit_graph.py
```
