# Next phase-field and two-phase-flow cases — 2026-10-26

## Goal, acceptance and status

User requests further phase-field cases suited to nonlocal derivatives, then
asks specifically about two-phase flow. Completed source-based recommendations
and a checked constant-density CHNS formulation. No implementation/training,
physical/config/reference/default change or Git history operation. No blocker
or required next action; a later implementation task must choose an actual
model, boundary data, units and benchmark before code.

## Ranked candidates and implementation boundaries

1. Periodic2D Cahn-Hilliard spinodal decomposition: scalar conserved field,
   morphology/topological changes and coarsening; PFHub1 gives explicit inputs.
   Independent local reference may use refined Fourier integration; finite-delta
   graph reference must separately match its operator/periodicity. Assess phi,
   chemical potential/flux, mass, energy and structure length, not phi L2 alone.
2. Ostwald ripening, PFHub2: coupled CH/AC, close to current corrosion solver
   infrastructure but with different free energy/initial data to specify.
3. Thermal dendritic solidification, PFHub3: phase and temperature with latent
   heat. Full anisotropy has gradient-direction/divergence terms; current scalar
   Laplacian is insufficient. An isotropic pilot is a different case, not the
   full benchmark.
4. Elastic precipitate, PFHub4: chemical/elastic coupling with quasi-static
   mechanical equilibrium; CH evolution is fictitious relaxation towards shape
   equilibrium in that benchmark. Existing elasticity code suggests reuse, not
   a verified drop-in coupling.
5. CHNS two-phase flow: strongest next multiphysics option if fluid coupling
   is the desired direction. Recommend2D equal-density, equal-viscosity, constant
   mobility, no-gravity elliptical-droplet relaxation in a sufficiently large
   periodic box, with a static-droplet verification first. This is a proposed
   test specification, not a run whose accuracy/runtime is established.

## Standard matched-density model and independently checked signs

For positive constant rho, eta, M, beta, kappa, no forcing and periodic boundary
conditions, a standard quartic-potential CHNS system is

$$
\begin{aligned}
\phi_t+\nabla\cdot(\phi\mathbf u)&=M\Delta\mu,\\
\mu&=\beta(\phi^3-\phi)-\kappa\Delta\phi,\\
\rho(\mathbf u_t+\mathbf u\cdot\nabla\mathbf u)&=-\nabla p+\eta\Delta\mathbf u-\phi\nabla\mu,\\
\nabla\cdot\mathbf u&=0.
\end{aligned}
$$

The capillary force convention differs from +mu grad(phi) by a pressure gradient
in the continuous setting. This identity must not be assumed for arbitrary
discrete operators without a discrete product rule. For

$$
E=\int_\Omega\left[\tfrac\rho2|\mathbf u|^2+\tfrac\beta4(\phi^2-1)^2+\tfrac\kappa2|\nabla\phi|^2\right]\,dx,
$$

the phase advection contribution is +integral(phi u dot grad(mu)), cancelling
the momentum capillary contribution -integral(phi u dot grad(mu)). Periodic
integration by parts and div(u)=0 remove pressure/kinetic advection, giving

$$
\frac{dE}{dt}=-\eta\int_\Omega|\nabla\mathbf u|^2\,dx-M\int_\Omega|\nabla\mu|^2\,dx.
$$

Integrating CH gives constant integral(phi). Driven walls/external forcing need
the boundary/forcing work in the balance; do not demand monotone total energy
for sheared flows without accounting for this work. These checks are analytic
formulation checks; no discrete PINN conservation guarantee is claimed.

## Nonlocal implementation and validation recommendation

Current build_D / SparseD is a scalar negative-Laplacian operator in
src/nonlocal_pinn/operators/__init__.py. For L_delta approximating -Delta,
chemical potential becomes beta(phi^3-phi)+kappa L_delta phi, and diffusive CH
contribution -M L_delta mu. Viscous diffusion may act componentwise. A network
may output(phi,mu,u_x,u_y,p), avoiding direct fourth spatial differentiation;
local AD has the same mixed-form option, so it is not a unique nonlocal benefit.

A derivative-discretization experiment may first retain spatial AD for first
gradients/divergence and use nonlocal Laplacians. A fully nonlocal scheme must
construct compatible gradient/divergence/advection/capillary operators, enforce
pressure gauge, and verify their weighted adjoint and constant/nullspace
properties. Present operator diagonal roundoff/truncated boundary treatments
are not automatically conservative or periodic. Any exact discrete energy
claim requires a separately derived identity; quadrature/core sampling and
soft PINN residuals can spoil it even for compatible full-grid operators.

Native nonlocal CHNS instead derives chemical potential from a pair interaction
energy. Literature establishes its well-posedness and energy properties for
specified hypotheses; that is a distinct physical/modeling choice from using
a finite-horizon approximation to the local differential model. Separate
fixed-operator reference error, horizon/grid convergence and PINN optimization.

Recommended progression: static droplet (pressure/surface-force balance and
spurious currents); elliptical relaxation (nonzero flow and interface motion);
capillary waves or sheared droplet; then coalescence and variable-density rising
bubbles. Large density ratios require a specified thermodynamically consistent
variable-density formulation, not naive rho(phi) substitution into the above.

Independent references should solve the selected exact model/operator with
refinement checks. Monitor phi/u/p errors (fix pressure gauge), mass drift,
divergence, spurious currents, interfacial geometry and total energy/work, not
only phase-field L2. No speed/accuracy benefit is predicted without measurement.

## Primary sources consulted

- PFHub1: https://pages.nist.gov/pfhub/benchmarks/benchmark1.ipynb/
- PFHub2 / overview: https://pages.nist.gov/pfhub/benchmarks/benchmark2.ipynb/
  and https://pages.nist.gov/pfhub/benchmarks/
- PFHub3: https://pages.nist.gov/pfhub/benchmarks/benchmark3.ipynb/
- PFHub4: https://pages.nist.gov/pfhub/benchmarks/benchmark4.ipynb/
- Jacqmin1999: https://doi.org/10.1006/jcph.1999.6332
- Colli,Frigeri,Grasselli2011: https://arxiv.org/abs/1101.3906
- Zeng,Xie,Yang,Wang2023: https://doi.org/10.1016/j.cma.2023.116289
- Abels-Garcke-Gruen nonlocal variable-density analysis:
  https://doi.org/10.1016/j.anihpc.2020.08.005

Some publisher full-text opens returned403; their indexed primary abstracts
and the arXiv author manuscript establish the literature claims. No inaccessible
full text is represented as read. Recommendations and integration costs are
our inference from formulations and repository source inspection.

## Verification and changed memory

Documentation-only task: inspect current operator API and analytical signs/
energy cancellation, source attribution, Markdown and git diff --check.
No code tests or training warranted. Memory: PROGRESS.md, DECISIONS.md and this
archived task. Production/defaults/references/artifact paths unchanged.
