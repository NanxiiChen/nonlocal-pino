# Next phase-field benchmarks after the transport pilot

Goal: recommend a short list of phase-field/nonlocal examples using primary
sources, existing infrastructure and independent validation prospects.

Acceptance: explain physical interest, operator reuse, reference strategy and
development difficulty; distinguish nonlocal derivatives from nonlocal physics.
No implementation or training requested.

Context: user considers the present two-phase pilot complete. It prescribes
velocity rather than solving coupled NS. SOAP stability is explicitly deferred.

Status: complete (2026-10-27). Blocker: none. No implementation selected.
Next action: none; user can choose a subsequent implementation task.

## Recommendations

Priority reflects our judgment about this repository, not measured runtimes or
an established nonlocal accuracy advantage:

1. Elastic precipitate shape relaxation (PFHub4): strongest new coupling and
   connection between existing phase-field and elasticity code. Misfit strain
   and elastic energy compete with interface energy. Start with one precipitate,
   then elastic contrast or interactions. PFHub uses fictitious conserved CH
   relaxation towards equilibrium, not experimentally calibrated kinetics.
   Nonlocal phase diffusion and variational elasticity need a consistently
   derived phase-dependent energy/force; constant-coefficient ElasticD is not
   a drop-in implementation of this coupled anisotropic benchmark. Independent
   mixed FEM with spatial/time refinement is a plausible reference route.
   Evaluate phase/displacement/stress, shape, integral mass and total energy.
2. Thermal dendritic solidification (PFHub3): strong morphological evolution,
   coupled phi and temperature/latent heat. Scalar thermal diffusion is reusable;
   full anisotropic interface variation needs compatible gradient/divergence
   terms, not a scalar Laplacian substitution. Independent refined FD/FEM
   reference; assess phi/temperature, tip position/speed, solid fraction and
   interface contour. Isotropic preparation is a different case from PFHub3.
   Its listed reaction term factors (1-phi^2), making inverse-map residual
   scaling a testable control; thermal/anisotropic terms and field L2 benefits
   are not covered by that algebraic cancellation alone.
3. Quasi-static brittle phase-field fracture: AT2 notched plate under increasing
   displacement is an appealing mechanical extension, but irreversibility,
   tension/compression treatment and phase-dependent stiffness make it the
   riskiest first implementation. Existing scalar and elastic operators supply
   ingredients, not a validated coupled model. Use a separately specified
   variational formulation and independently converged FEM; compare crack path,
   reaction-displacement response, energy and damage irreversibility as well
   as physical-field error. Do not silently equate a history-field surrogate
   with the constrained original variational problem.
4. Periodic CH spinodal decomposition (PFHub1-inspired or exact benchmark):
   lowest development cost, reuses mixed phi/mu transport infrastructure with
   zero imposed velocity. Topology change/coarsening tests many interfaces and
   long-time evolution; less new coupling than the above. Use independent
   refined Fourier reference and compare phase/mu, mass, energy and structure
   length. Do not call a simplified periodic IC an exact PFHub reproduction.

For emphasizing physical nonlocality rather than derivative replacement, a
separate native integral-kernel CH model is another option: select a published
energy, kernel and boundary treatment, and assess finite-horizon effects and
nonlocal-to-local convergence separately from PINN error. Do not identify
finite-horizon quadrature, Ohta-Kawasaki long-range inverse-Laplacian coupling
and local derivative approximation as interchangeable physical models.

## Primary sources checked

- PFHub1, spinodal decomposition:
  https://pages.nist.gov/pfhub/benchmarks/benchmark1.ipynb/
- PFHub3, thermal anisotropic dendrites, explicit phi/U PDE and no-flux setup:
  https://pages.nist.gov/pfhub/benchmarks/benchmark3.ipynb/
- PFHub4, misfit/phase-dependent elasticity and fictitious CH evolution:
  https://pages.nist.gov/pfhub/benchmarks/benchmark4.ipynb/
- Bleyer/Helfer AT2 implementation, variational irreversibility and staggered
  formulation (its actual example is an inclusion plate, not the proposed notch):
  https://thelfer.github.io/mgis/web/mgis_fenics_phase_field.html
- Davoli/Scarpa/Trussardi, nonlocal-to-local CH convergence under specified
  kernel/viscosity/Neumann assumptions (not a theorem for our current operator):
  https://arxiv.org/abs/1908.00945

Existing implementation context read: examples/elasticity2d/README.md,
examples/twophase2d/README.md and the earlier archived recommendations.
The current two-phase pilot prescribes velocity; it is not a completed CHNS
implementation. Its closure does not remove the explicitly deferred SOAP issue.

## Validation and memory

Documentation-only review of source formulations, scope and attribution;
local task links and git diff --check passed. No numerical tests or training
required. Changed memory: this archive, PROGRESS.md and DECISIONS.md. No
production/default/reference/checkpoint/dependency/Git history changes.
