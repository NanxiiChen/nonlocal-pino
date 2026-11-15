A fixed initial droplet is transported through a family of smooth constricted channels. The operator predicts the coupled phase, chemical potential, velocity and pressure.

## Setup and boundary data

The undeformed channel is \(4\times1\). Its two walls use independent sets of eight quartic B-spline coefficients and a compact \(C^3\) taper on \(x\in[1.15,2.75]\). Proposals preserve straight ends and a minimum aperture in \([0.55,0.70]\). The common initial droplet has center \((0.8,0.5)\), radius 0.22 and diffuse thickness parameter 0.05. Its phase is negative in the droplet and positive in the carrier.

The inlet supplies a ramped parabolic velocity and carrier phase. Solid walls are no-slip with neutral wetting and zero chemical flux. The outlet has zero traction and zero normal gradients of phase and chemical potential. The traction condition supplies the pressure reference.

<figure>
  <a href="assets/channel_nsch2d-setup.png"><img src="assets/channel_nsch2d-setup.png" width="1299" height="767" alt="Orange initial droplet in blue carrier fluid between independently randomized gray channel walls"></a>
  <figcaption>Independent wall profiles vary the constriction and asymmetry. The initial droplet and driving remain fixed; arrows indicate driving direction, not a predicted velocity field. <a href="assets/channel_nsch2d-setup.pdf">Editable PDF</a>.</figcaption>
</figure>

## Governing equations and scaling

The constant-density and constant-viscosity model couples incompressible Navier–Stokes and Cahn–Hilliard through capillary forcing and conservative phase advection.

$$
\begin{aligned}
\rho(\mathbf u_t+\mathbf u\cdot\nabla\mathbf u)
   &=-\nabla p+\eta\Delta\mathbf u-\phi\nabla\mu,\\
\nabla\cdot\mathbf u&=0,\\
\phi_t+\nabla\cdot(\phi\mathbf u)&=M\Delta\mu,\\
\mu&=A(\phi^3-\phi)-\kappa\Delta\phi .
\end{aligned}
$$

In the modeled units, \(A=3\sigma/(2\sqrt{2}\epsilon)\), \(\kappa=A\epsilon^2\), \(\rho=1\), \(\eta=0.2\), \(\sigma=0.2\), \(\epsilon=0.05\) and \(M=0.0005\). The inlet mean speed reaches 0.2 after a one-unit ramp. The simulation ends at \(t=6\).

## Nonlocal residual and loss

The streamfunction lift defines \(\mathbf u=(\partial_y\psi,-\partial_x\psi)\). Its prescribed components impose the velocity data, stationary-wall velocity and section flux. The curl uses spatial AD and gives solenoidal velocity in the continuous representation. The phase lift imposes its initial and inlet data.

Write \(G_\delta\) and \(L_\delta\) for the moment-corrected gradient and positive Laplacian described in [spatial calculus](spatial-calculus.html#moment-correction-in-the-channel). The interior phase and chemical rows are

$$
\begin{aligned}
r_\phi &= \phi_t+\operatorname{div}_{G_\delta}(\phi\mathbf u)-M L_\delta\mu,\\
r_\mu &= \mu-A(\phi^3-\phi)+\kappa L_\delta\phi .
\end{aligned}
$$

The momentum rows use the corrected gradients for advection, pressure and capillary forcing, and the corrected Laplacian for viscosity. Boundary-target stencils evaluate the explicit phase and chemical normal derivatives and outlet traction. The nine MSE terms consist of four interior rows, two wall rows and three outlet rows. The phase transport divisor is 0.2. The chemical-potential output amplitude is 4, its relation divisor is 1, and its wall/outlet divisor is 4. The remaining row divisors are 1.

## Field evolution

<figure>
  <a href="assets/channel_nsch2d.png"><img src="assets/channel_nsch2d.png" width="2007" height="696" alt="Predicted phase in four constricted channels across five normalized times with dashed local-FEM droplet interfaces"></a>
  <figcaption>The selected 20,000-step model on the development geometry followed by all three reserved geometries. Columns show t* = t/T = 0, 0.25, 0.5, 0.75 and 1, with T = 6 in model units. Colors show predicted phase; white dashed curves are independent local-FEM φ = 0 interfaces. Labels give instantaneous phase relative L² on all 3,840 original spatial queries. <a href="assets/channel_nsch2d.pdf">Editable PDF</a>.</figcaption>
</figure>

## References and comparisons

Independent local mixed FEM integrates the coupled system and is checked with spatial and temporal refinement. Each case is scored at 3,840 physical queries and 13 saved times. Velocity, pressure and chemical-potential errors exclude the initial time. The same three reserved geometries are used for the 10k and 20k comparison.

| Mean relative \(L^2\) against local FEM | Historical 10k | Selected 20k |
| --- | ---: | ---: |
| Phase, pooled | 1.8396% | 1.6435% |
| Phase, terminal | 2.2708% | 2.1961% |
| Minority phase | 4.8474% | 4.3306% |
| Velocity | 2.0160% | 1.5172% |
| Pressure | 5.4118% | 5.2840% |
| Chemical potential | 15.9168% | 13.9541% |

The budget, learning-rate schedule and number of time samples changed together. Mean improvements coexist with worse errors on some individual cases. The selected run's worst reserved pressure error is 6.5885%, and the between-geometry pressure response is harder to predict than the total field.

<span id="fem-pino-comparison"></span>

## Three-method comparison

The final comparison trains three PINOs with nonlocal spatial operators, local
spatial AD, or assembled finite-element weak residuals. All retain the same
width192 network, hard constraints, time JVP, SOAP20k schedule and rolling
geometry draws. Local AD uses2,048fresh interior queries and all192independent
boundary queries. FEM PINO interpolates phase, chemical potential and velocity
with P2 triangles on15,633neural nodes; pressure uses P1. Its four weak losses
use a consistent-mass-diagonal dual norm, with natural phase/chemical flux and
outlet traction. **FEM PINO is a trained network**; the **FEM numerical reference**
is an independently solved trajectory.

<figure>
  <a href="assets/channel_nsch2d_three_method_convergence.png"><img src="assets/channel_nsch2d_three_method_convergence.png" width="2007" height="708" alt="Nonlocal, local AD and FEM PINO relative-L2 convergence through 20000 updates for phase, velocity, pressure and chemical potential"></a>
  <figcaption>All 41 actual evaluations, every 500 updates, on one common development geometry. Thin solid lines: raw errors; thick solid lines: local-linear LOWESS in log10-error (span 0.22). Shading: approximate pointwise 95% fitted-trend confidence intervals from 2,000 circular-block residual bootstrap replicates, block length four. This assumes a smooth trend with approximately stationary short-memory log residuals; it does not measure cross-seed or test-cohort uncertainty or account for unknown smoothing bias. <a href="assets/channel_nsch2d_three_method_convergence.pdf">Editable PDF</a>.</figcaption>
</figure>

<figure>
  <a href="assets/channel_nsch2d_three_method_distribution.png"><img src="assets/channel_nsch2d_three_method_distribution.png" width="2007" height="708" alt="Final relative error violins on all three reserved geometries for Nonlocal PINO, Local AD PINO and FEM PINO"></a>
  <figcaption>Final20k weights on all three reserved geometries. Log10-error KDE uses Scott bandwidth and three-bandwidth tails; dashed lines mark quartiles. N=3 supports only a descriptive small-cohort display; smooth tails are not additional observations. Panels report individual fields, with a joint vector norm for velocity. Axes show raw error ratios with automatic log limits.</figcaption>
</figure>

[Distribution PDF](assets/channel_nsch2d_three_method_distribution.pdf) ·
[Figure inputs and reproduction](https://github.com/NanxiiChen/nonlocal-pino/blob/main/artifacts/pino_three_method_figures.json)

| Reserved-geometry mean relative L2 | Nonlocal PINO | Local AD PINO | FEM PINO |
|---|---:|---:|---:|
| Phase | 1.6435% | 2.1540% | 4.6465% |
| Velocity | 1.5172% | 2.0866% | 4.3892% |
| Pressure | 5.2840% | 8.5370% | 7.1489% |
| Chemical potential | 13.9541% | 16.0556% | 43.7908% |

Norms use equally weighted space-time queries: all13times for phase, and the12
positive times for velocity, pressure and chemical potential. Average the field
norms over the three samples; no channel averaging is used in these panels.
Nonlocal has the lowest mean in each field. Local AD improves phase and velocity
over FEM PINO, but has a higher pressure mean; its worst pressure error is15.7110%.
One initialization and three reused geometries do not establish a universal ranking.

| Same-card warm resource probe | Nonlocal PINO | Local AD PINO | FEM PINO |
|---|---:|---:|---:|
| PDE queries per geometry | 15,633 | 2,048 sampled interior | 15,633 |
| Time block / total times | 18 / 36 | 9 / 36 | 18 / 36 |
| Inclusive time per update | 1.6016 s | 1.7372 s | 1.4169 s |
| Active memory peak | 83.38 GiB | 26.19 GiB | 80.82 GiB |

All use B8 on the same RTX PRO6000 device for these short probes, including batch
preparation. The FEM probe is from the earlier same-device control; the other two
are later paired probes. Parent timers from before process creation through
exit give full cold20k costs of 10.156 h for AD and 8.032 h for FEM, including
initialization, compilation, in-training evaluation, final writes and teardown.
Historical Nonlocal has an approximate external interval of 9.179 h on another
physical GPU of the same model: its UTC timestamp starts immediately after
spawn and exit is observed by ten-second polling plus monitor work. It is not
a strict sub-second or same-device comparison; the earlier 9.163 h value was
trainer-internal. [Complete timing protocol](training.html#cold-training-wall-clock-cost).
Sampled AD uses much less active memory, but its measured update is not faster.

Initial model/optimizer states and all20k condition/time/geometry draws match;
independent saved-field norms and source checks pass. AD retains the nine
pointwise terms and estimates the same full spatial objective by uniform
interior sampling, with explicit independent boundary losses. FEM uses different
weak-loss norms and natural boundary enforcement; its interpolated velocity is
only approximately divergence-free. The nonlocal default remains unchanged.
[Full formulation and validation](https://github.com/NanxiiChen/nonlocal-pino/blob/main/examples/channel_nsch2d/README.md#full-coordinate-fem-weak-residual-control).

## Implementation and reproduction

| Component | Selected setting |
| --- | --- |
| Network | Width 192, four heads, two spatial cross-attention blocks, three time-FiLM layers |
| Derivatives | \(192\times80\) subdivisions, 15,633 nodes, horizon 0.075, capacity 128, cubic moment reproduction |
| Samples per update | 8 geometry draws, 36 stratified times accumulated as two 18-time blocks |
| Online pool | 16 slots, two CPU workers, replace two slots every 50 updates |
| Optimizer | SOAP, 20,000 cold updates, \(10^{-3}\) to \(10^{-5}\), ordinary FP32 |

The qualified RTX PRO 6000 96 GB run used 83.50 GiB peak active memory and 9.16 h of training wall time.

~~~bash
source .venv/bin/activate
XLA_PYTHON_CLIENT_MEM_FRACTION=0.97 python -m examples.channel_nsch2d.train
~~~

The default loads the prepared random-wall evaluation bank by its configured path. For reference preparation, match the saved wall coefficients and physical parameters, and verify FEM space/time refinement before training. See the [example and configuration guide](https://github.com/NanxiiChen/nonlocal-pino/blob/main/examples/channel_nsch2d/README.md).
