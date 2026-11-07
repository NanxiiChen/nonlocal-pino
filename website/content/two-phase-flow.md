A fixed initial droplet is transported through a family of smooth constricted channels. The operator predicts the coupled phase, chemical potential, velocity and pressure.

## Setup and boundary data

The undeformed channel is \(4\times1\). Its two walls use independent sets of eight quartic B-spline coefficients and a compact \(C^3\) taper on \(x\in[1.15,2.75]\). Proposals preserve straight ends and a minimum aperture in \([0.55,0.70]\). The common initial droplet has center \((0.8,0.5)\), radius 0.22 and diffuse thickness parameter 0.05. Its phase is negative in the droplet and positive in the carrier.

The inlet supplies a ramped parabolic velocity and carrier phase. Solid walls are no-slip with neutral wetting and zero chemical flux. The outlet has zero traction and zero normal gradients of phase and chemical potential. The traction condition supplies the pressure reference.

<figure>
  <a href="assets/channel_nsch2d-setup.svg"><img src="assets/channel_nsch2d-setup.svg" alt="Three smooth constricted channels with different upper and lower walls and the same initial droplet"></a>
  <figcaption>Independent wall profiles vary the constriction and its asymmetry. The upstream droplet and driving remain fixed.</figcaption>
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
  <a href="assets/channel_nsch2d.png"><img src="assets/channel_nsch2d.png" width="2325" height="720" alt="Predicted droplet motion through three independently constricted channels at five times"></a>
  <figcaption>All three reserved geometries at t = 0, 1.5, 3, 4.5 and 6. Dashed curves are independent local-FEM phase interfaces on the same physical axes.</figcaption>
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

The default loads the prepared random-wall evaluation bank by its configured path. For reference preparation, match the saved wall coefficients and physical parameters, and verify FEM space/time refinement before training. See the [example and configuration guide](https://github.com/NanxiiChen/nonlocal-pinn/blob/main/examples/channel_nsch2d/README.md).
