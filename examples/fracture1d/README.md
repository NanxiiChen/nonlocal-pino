# 1D Phase-Field Fracture — PINN with Nonlocal Operator

## Motivation

Phase-field fracture is an ideal application for the nonlocal operator $D_\delta$:
- **Crack surfaces are Neumann boundaries** (free surfaces) — FD needs explicit BC
  handling; nonlocal handles them automatically
- **Notches are geometric discontinuities** — points across a notch must not be
  connected; nonlocal achieves this simply by not sampling inside the notch (as
  long as notch width $w > \delta$)
- **Crack propagation changes topology** — without needing to re-mesh or
  reconstruct stencils

We start with **1D** because it has analytical reference solutions (critical
stress, crack profile, homogeneous damage-strain relation). Once verified, the
same formulation extends to 2D single-edge notched tension.

## Governing Equations (Classical / Local)

### Phase-field (AT2 model, Miehe et al. 2010)

The regularized total energy functional:

$$E_\ell(u,d) = \int_\Omega (1-d)^2 \Psi_0(\varepsilon(u)) \,dx + G_c \int_\Omega \left(\frac{d^2}{2\ell_0} + \frac{\ell_0}{2}|\nabla d|^2\right) dx$$

The evolution follows from the variational derivatives:

**Elastic equilibrium** (linear momentum balance):

$$\nabla \cdot \big[(1-d)^2 \sigma_0\big] = 0$$

**Phase-field evolution** (damage, with irreversibility via history field):

$$\frac{G_c}{\ell_0}\big(d - \ell_0^2 \nabla^2 d\big) = 2(1-d)\,\mathcal{H}$$

where $\mathcal{H}(x,t) = \max_{\tau \leq t} \Psi_0^+(\varepsilon(x,\tau))$ is
the history of the positive part of the elastic strain energy density.
$\Psi_0^+$ uses a volumetric-deviatoric split to prevent damage under pure
compression:

$$\Psi_0^+ = \frac{K}{2}\langle\operatorname{tr}\varepsilon\rangle_+^2 + \mu\|\operatorname{dev}\varepsilon\|^2$$

**Constitutive law** (linear isotropic):

$$\sigma_0 = \lambda\operatorname{tr}(\varepsilon)I + 2\mu\varepsilon$$

### Irreversibility: history field $\mathcal{H}$

Damage cannot heal: $\partial d/\partial t \geq 0$. In the AT2 formulation this
is enforced by feeding the **maximum historical strain energy** into the
phase-field driving force — not via a penalty constraint on $\partial d/\partial t$:

$$\mathcal{H}_n(x) = \max\big(\mathcal{H}_{n-1}(x),\; \Psi_0^+(\varepsilon_n(x))\big)$$

For quasi-static incremental loading, $\mathcal{H}$ is an **external field**
updated after each load step, not a network output. This avoids:
- Penalty hyperparameter tuning (as with soft $\partial d/\partial t \geq 0$)
- Optimisation instability from comparing two network outputs across load steps

---

## 1D Reduction (for analytical validation)

In 1D, under uniaxial tension, the equations simplify dramatically:

**Equilibrium** ($\sigma =$ constant across the bar):
$$(1-d(x))^2 E\varepsilon(x) = \sigma$$

**Phase-field:**
$$\frac{G_c}{\ell_0}\big(d - \ell_0^2 d''\big) = 2(1-d)\,\mathcal{H},\quad \mathcal{H} = \max_{\tau \leq t}\;\frac12 E\varepsilon^2$$

### Analytical results

| Quantity | Expression | Notes |
|---|---|---|
| Homogeneous damage | $d = \frac{2\ell_0 \mathcal{H}}{G_c + 2\ell_0 \mathcal{H}}$ | Before localisation ($d''=0$) |
| Critical strain | $\varepsilon_c = \sqrt{\frac{G_c}{\ell_0 E}}$ | Onset of damage ($d \to 0^+$) |
| Crack profile (fully developed) | $d(x) = \exp\left(-\frac{\lvert x\rvert }{ \ell_0}\right)$ | Valid when $\sigma \approx 0$ |
| Characteristic crack width | $\sim 2\ell_0$ | $d$ decays from 1 to $\sim\! 0.14$ at $|x|=2\ell_0$ |

These reference solutions allow systematic verification of:
1. The nonlocal operator's accuracy: does $D_\delta$ reproduce the exponential crack profile?
2. Critical strain prediction: does the PINN capture damage onset at the correct load?
3. Irreversibility: does the history-field mechanism prevent damage healing?

---

## Non-Dimensionalisation

The original 2D FEniCS code and many literature benchmarks work in **dimensional
form** (mm–kN–N/mm), which makes hyperparameter transfer between problems
difficult. We adopt a consistent non-dimensionalization.

### Characteristic scales

| Scale | Symbol | Definition |
|---|---|---|
| Length | $L_0$ | Domain size (e.g. 1 mm for Miehe benchmark) |
| Stress | $E$ | Young's modulus |
| Displacement | $L_0$ | Same as length |
| Energy density | $E$ | Same as stress (strain is dimensionless) |

Non-dimensional variables (starred):
$$x^* = \frac{x}{L_0},\quad u^* = \frac{u}{L_0},\quad \varepsilon^* = \varepsilon,\quad \sigma^* = \frac{\sigma}{E},\quad \mathcal{H}^* = \frac{\mathcal{H}}{E}$$

### Non-dimensional parameters

$$N_\ell = \frac{\ell_0}{L_0} \qquad\text{(dimensionless internal length — controls crack width)}$$

$$N_G = \frac{G_c}{E L_0} \qquad\text{(dimensionless fracture toughness = Griffith length / domain size)}$$

For the Miehe benchmark ($E \approx 210$ kN/mm², $G_c = 2.7$ N/mm, $\ell_0 = 0.024$ mm, $L_0 = 1$ mm):
$$N_\ell = 0.024,\quad N_G = \frac{2.7\ \text{N/mm}}{210 \times 10^3\ \text{N/mm}^2 \times 1\ \text{mm}} = 1.286 \times 10^{-5}$$

### Non-dimensional PDEs

**Elasticity:**
$$\frac{d}{dx^*}\Big[(1-d)^2 \varepsilon^*\Big] = 0 \quad\Longrightarrow\quad (1-d)^2 \varepsilon^* = \sigma^*\ \text{(constant)}$$

**Phase-field** (after multiplying by $\ell_0/G_c$):
$$d - N_\ell^2 \frac{d^2 d}{dx^{*2}} = \frac{2 N_\ell}{N_G}(1-d)\,\mathcal{H}^*$$

**Constitutive** (1D):
$$\sigma^* = \varepsilon^*,\quad \mathcal{H}^* = \max_{\tau \leq t}\ \frac12 (\varepsilon^*)^2$$

### Non-dimensional critical strain

$$\varepsilon_c^* = \sqrt{\frac{N_G}{N_\ell}} \quad\left(=\sqrt{\frac{G_c}{\ell_0 E}}\right)$$

For the benchmark parameters: $\varepsilon_c^* = \sqrt{1.286\times 10^{-5} / 0.024} \approx 0.023$

---

## Nonlocal (Peridynamics-Inspired) Formulation

Replacing $\frac{d^2}{dx^{*2}}$ with the nonlocal operator $D_\delta$ (where
$D_\delta \approx -\frac{d^2}{dx^{*2}}$ as $\delta \to 0$):

$$d + N_\ell^2\,D_\delta^*\, d = \frac{2 N_\ell}{N_G}(1-d)\,\mathcal{H}^*$$

where $D_\delta^*$ is the non-dimensional nonlocal operator (built on $x^*$ coordinates).

In the nonlocal ("peridynamic") limit, the damage diffusion is mediated by
compact kernel interactions between neighbouring material points. The kernel
support $\delta$ should satisfy $\delta \lesssim \ell_0$ (i.e. $\delta^* \lesssim N_\ell$)
so that the nonlocal interaction zone matches the physical crack process zone.

**Key advantage in 1D:** No notch, no geometric discontinuities — all points are
connected. This is the cleanest test of whether $D_\delta$ correctly approximates
the Laplacian in the phase-field equation.

For the 2D extension with an initial notch: points inside the notch are simply
**not sampled** (方案 A from the main discussion). As long as the notch width
$w > \delta$, KD-tree neighbours naturally disconnect across the notch. This
handles the geometric discontinuity without modifying the $D$ matrix.

### Incremental quasi-static loading

The problem is solved incrementally in load steps, not as a one-shot
spacetime PINN. This is required because:

1. **Fracture is path-dependent**: the total energy is non-convex; the system
   must follow a specific loading path through metastable states
2. **History field update**: $\mathcal{H}$ is updated after each converged load
   step, providing the irreversibility mechanism
3. **Warm-start**: each load step initialises from the previous step's network,
   keeping the optimisation problem near-convex

```
for n = 1..N_load_steps:
    λ_n = λ_{n-1} + Δλ              # increment applied displacement
    H_prev(x) = H(x) from step n-1  # external field, not network output

    train PINN:
      inputs:  x*
      outputs: u*(x*), d(x*)
      loss:
        L_elastic = mean(|d/dx*[(1-d)² ε*]|)     # AD
        L_damage  = mean(|d + Nℓ² D*_δ d - RHS|)  # D_δ nonlocal
        L_bc      = boundary conditions on u*
        L_ic      = initial crack (for step 0 only)

    update H(x) = max(H_prev(x), (ε*(x))²/2)     # for next step
```

The $D_\delta$ matrix is built **once** (domain topology never changes — the
crack is a diffuse $d$-field, not a geometric cut). After the initial notch
sampling exclusion (方案 A), no further $D$ matrix updates are needed.

---

## Physical Parameters (Miehe Benchmark)

| Symbol | Value | Unit | Description |
|---|---|---|---|
| $E$ | $\sim 210 \times 10^3$ | N/mm² (MPa) | Young's modulus |
| $\nu$ | $0.3$ | — | Poisson's ratio |
| $\lambda$ | $121.15 \times 10^3$ | N/mm² | Lamé first parameter |
| $\mu$ | $80.77 \times 10^3$ | N/mm² | Shear modulus |
| $G_c$ | $2.7$ | N/mm | Fracture toughness |
| $\ell_0$ | $0.024$ | mm | Internal length (crack width) |

### Non-Dimensional Parameters

| Parameter | Expression | Value (for this benchmark) |
|---|---|---|
| $N_\ell$ | $\ell_0 / L_0$ | $0.024$ |
| $N_G$ | $G_c / (E L_0)$ | $1.286 \times 10^{-5}$ |
| $N_\ell/N_G$ | $\ell_0 E / G_c$ | $\approx 1867$ |
| $\varepsilon_c^*$ | $\sqrt{N_G / N_\ell}$ | $\approx 0.023$ |

Note: $N_\ell/N_G \gg 1$ means the elastic driving force is very strong relative
to fracture resistance — the RHS term in the phase-field equation is large,
which is physically correct (once the critical strain is exceeded, damage grows
rapidly).

---

## Boundary Conditions & Crack Initiation

### 1D bar under tension

```
   x* = -0.5                         x* = 0.5
     ├──────────────────────────────────┤
     │   (1-d)² E ε = σ = constant      │
     ├──────────────────────────────────┤
   u*(-0.5)=0                     u*(0.5)=u_bar*
```

No initial notch needed in 1D — damage nucleates homogeneously when
$\varepsilon^*$ exceeds $\varepsilon_c^*$, then localises at the weakest point
(or at the centre by symmetry in a PINN with no initial imperfection).

### 2D single-edge notched tension (extension)

```
   u_y = u_bar, u_x = 0 (top)
   ┌──────────────────────┐
   │                      │
   │     domain           │
   │                      │
   └──┐            ┌──────┘
      └─── notch ──┘          ← crack initiates from notch tip
   ┌──────────────────────┐
   │                      │
   u_y = 0, u_x = 0 (bottom)
       x=-0.5            x=0.5
```

The notch is a **physical slit** of width $w$ (chosen so $w > \delta$).
No sampling points are placed inside the notch (方案 A). The nonlocal operator
automatically disconnects points on opposite sides of the notch.

---

## Projected Structure

```
fracture1d/
├── solver.py                # 1D FEM reference (analytical benchmark)
├── train.py                 # 1D PINN training (incremental loading)
├── losses.py                # Elastic + damage + BC losses
├── configs/
│   ├── solver_base.py       # FEM solver config
│   └── train_base.py        # PINN training config
├── README.md
```

Later:
```
fracture2d/
├── solver.py                # 2D FEM reference (Miehe benchmark)
├── train.py                 # 2D PINN training
├── losses.py                # 2D losses (notch-aware sampling)
├── configs/
│   └── ...
├── README.md
```

---

## References

- Miehe, C., Hofacker, M., & Welschinger, F. (2010). A phase field model for
  rate-independent crack propagation: Robust algorithmic implementation based
  on operator splits. *CMAME*, 199(45-48), 2765–2778.
- Borden, M. J., et al. (2014). A higher-order phase-field model for brittle
  fracture: Formulation and analysis within the isogeometric analysis
  framework. *CMAME*, 273, 100–118.
- Pham, K., et al. (2011). Gradient damage models and their use to approximate
  brittle fracture. *International Journal of Damage Mechanics*, 20(4), 618–652.
