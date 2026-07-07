# 1D KKS Phase-Field Corrosion — PINN + FEM Reference

## Governing Equations (Local / Classical)

The KKS (Kim-Kim-Suzuki) phase-field model for diffusion-controlled corrosion:

**Allen-Cahn (phase-field evolution):**

$$\frac{\partial \phi}{\partial t} = -L\left[\alpha_\phi \nabla^2\phi - \omega_\phi g'(\phi) - A (c - h(\phi)c_{Se} - c_{Le}) h'(\phi)\right]$$

**Cahn-Hilliard (concentration evolution):**

$$\frac{\partial c}{\partial t} = \nabla\cdot\left[M\nabla\left(\frac{\delta F}{\delta c}\right)\right] \approx D\nabla^2\left[c - h(\phi)(c_{Se} - c_{Le}) - c_{Le}\right]$$

**Interpolation functions:**

$$h(\phi) = -2\phi^3 + 3\phi^2,\quad g(\phi) = \phi^2(1-\phi)^2$$

---

## Non-Dimensionalization

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

## Nonlocal Formulation

Replacing $\nabla^{*2}$ with the nonlocal operator $D_\delta$ (peridynamics style):

$$\frac{\partial\phi}{\partial t^*} + N_{\text{AC,3}}\,D_\delta\phi + N_{\text{AC,2}}\,g'(\phi) - N_{\text{AC,1}}\bigl[c - h(\phi)\Delta c_0 - c_{Le}\bigr]\Delta c_0\,h'(\phi) = 0$$

$$\frac{\partial c}{\partial t^*} + N_{\text{CH}}\,D_\delta\!\bigl[c - h(\phi)\Delta c_0 - c_{Le}\bigr] = 0$$

where $\Delta c_0 = c_{Se} - c_{Le}$, and in the local limit $D_\delta \approx -\nabla^{*2}$.

The nonlocal operator $D_\delta$ is defined by a kernel $J(r)$ with compact support $r \le \delta$:

$$(D_\delta u)(x_i) = \sum_{j \in N_\delta(i)} w_{ij} \, (u_i - u_j)$$

where edge weights $w_{ij} = J(|x_i - x_j|) \cdot dV$, and kernel functions include `cubic`, `quartic`, `wendland` (all C¹ or C² smooth).

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
| $c_{Le}$ | 5100 / 1.43e5 ≈ 0.0357 | — | Equilibrium concentration (liquid) |

### Non-Dimensionalisation Scales

| Scale | Value | Description |
|---|---|---|
| $L_c$ | $100~\mu\text{m}$ | Characteristic length |
| $T_c$ | 50 s | Characteristic time |

---

## Project Structure

```
corrosion1d/
├── solver.py          # FEM solver (FEniCS, Crank-Nicolson θ-method)
├── train.py           # PINN training script (equinox + jax)
├── losses.py          # PDE residual, BC, IC loss functions
├── configs/
│   ├── solver_base.py       # FEM solver config
│   └── train_diffusion.py   # PINN training config (full MLP)
├── results/           # FEM reference solutions (.npz)
└── plots/             # Output plots
```

---

## Usage

### Run FEM Solver

```bash
cd examples/corrosion1d
python solver.py --config=configs/solver_base.py
```

Output: reference solution saved to `results/`.

### Train PINN (Coordinate MLP / FiLM)

```bash
# from project root
python examples/corrosion1d/train.py --config=examples/corrosion1d/configs/train_diffusion.py
```

Key config overrides (examples):

```bash
# Adjust architecture
--config.arch.type=film --config.arch.hidden_dim=64

# Enable causal training
--config.loss.use_causal=True --config.causal.n_chunks=10

# Staggered AC/CH training in the beginning 2k steps
--config.loss.stagger_steps=2000
```
