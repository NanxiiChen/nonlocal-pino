# Ice Melting 3D

A shrinking ice sphere driven by an Allen-Cahn equation with a curvature-independent
source term. Admits an exact analytical solution — ideal for benchmarking.

## Dimensional Governing Equation

$$\frac{\partial\phi}{\partial t} = M\left(\Delta\phi - \frac{F'(\phi)}{\epsilon^2}\right) - \lambda\frac{\sqrt{2F(\phi)}}{\epsilon}$$

with $F(\phi) = \frac14(1-\phi^2)^2$:

$$F'(\phi) = \phi^3 - \phi,\qquad \frac{\sqrt{2F(\phi)}}{\epsilon} = \frac{1 - \phi^2}{\sqrt{2}\,\epsilon}$$

- **Domain**: $\Omega = [-50, 50]^3$
- **IC**: sphere of radius $R_0 = 35$ at origin:

$$\phi(\bm x, 0) = \tanh\!\left(\frac{R_0 - |\bm x|}{\sqrt{2}\,\epsilon}\right)$$

- **Analytical solution**:

$$\phi(\bm x, t) = \tanh\!\left(\frac{R(t) - |\bm x|}{\sqrt{2}\,\epsilon}\right),\qquad R(t) = R_0 - \lambda t$$

---

## Non-Dimensionalisation

Map to unit cube $[0,1]^3$ and $t^* \in [0,1]$:

$$\bm x^* = \frac{\bm x + 50}{L_c},\quad t^* = \frac{t}{T_c}$$

with $L_c = 100$, $T_c = 5.0$ (total time).  The sphere centre is $\bm x_c^* = (0.5, 0.5, 0.5)$.

Non-dimensional parameters:

$$N_1 = \frac{M T_c}{L_c^2},\quad N_2 = \frac{N_1}{\epsilon^{*2}},\quad N_3 = \frac{\lambda^*}{\sqrt{2}\,\epsilon^*}$$

where $\epsilon^* = \epsilon / L_c$, $\lambda^* = \lambda T_c / L_c$, $R_0^* = R_0 / L_c$.

### Non-Dimensional Governing Equation

$$\frac{\partial\phi}{\partial t^*} = N_1 \Delta^*\phi - N_2\,(\phi^3 - \phi) - N_3\,(1 - \phi^2)$$

### Non-Dimensional Analytical Solution

$$\phi(\bm x^*, t^*) = \tanh\!\left(\frac{R_0^* - \lambda^* t^* - |\bm x^* - \bm x_c^*|}{\sqrt{2}\,\epsilon^*}\right)$$

with $R^*(t^*) = R_0^* - \lambda^* t^*$.

---

## Nonlocal Formulation

Replace $\Delta^*$ with the nonlocal operator $D_\delta$ ($D_\delta \approx -\Delta^*$):

$$\frac{\partial\phi}{\partial t^*} + N_1\,D_\delta\phi + N_2\,(\phi^3 - \phi) + N_3\,(1 - \phi^2) = 0$$

PINN residual:

$$r_\phi = \frac{\partial\phi}{\partial t^*} + N_1\,D_\delta\phi + N_2\,(\phi^3 - \phi) + N_3\,(1 - \phi^2)$$

---

## Parameters

| Symbol | Value | Description |
|---|---|---|
| $M$ | 0.1 | Mobility |
| $\lambda$ | 5.0 | Interface speed (dimensional) |
| $R_0$ | 35 | Initial sphere radius (dimensional) |
| $L_c$ | 100 | Characteristic length |
| $T_c$ | 5.0 | Characteristic time |
| $N$ | 64 | Mesh elements per direction ($64^3$ cells) |
| $h^*$ | $1/N \approx 0.0156$ | Non-dimensional grid spacing |
| $\epsilon^*$ | $\approx 0.0226$ | Non-dimensional interface thickness |
| $R_0^*$ | $0.35$ | Non-dimensional initial radius |
| $\lambda^*$ | $0.25$ | Non-dimensional interface speed ($\lambda T_c / L_c$) |
| $\Delta t^*$ | $0.001$ | Non-dimensional time step |
| $T^*$ | $1.0$ | Non-dimensional total time |

---

## Usage

```bash
conda activate fenicsenv
mpirun --allow-run-as-root -np 12 python examples/icemelting3d/solver.py
```
