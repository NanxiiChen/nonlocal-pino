# Fully nonlocal spatial derivatives: feasibility assessment

Goal: answer whether all spatial derivatives can use finite-horizon nonlocal
operators in the current prescribed-flow CH case and a future coupled NS case.

Acceptance: distinguish space/time/parameter differentiation; inspect existing
scalar and elasticity operators; consult primary nonlocal vector-calculus/Stokes
sources; independently check proposed periodic gradient/divergence algebra and
state nullspace/stability limitations. No request for a full new NS solver or
training comparison is inferred from this feasibility question.

Context: current recommended CH path uses spectral flux, nonlocal Laplacians
and time JVP; it solves no NS. ElasticD has moment-corrected correspondence
gradients, but not a turnkey periodic flow gradient/divergence API.

Status: complete (2026-10-27).
Blocker: none.
Next action: none for this assessment. A future implementation should first
select and validate a stable periodic first-derivative pair, then add an explicit
advection option and a matched reference/control. Full NS remains a separate
modeling/implementation step.

Conclusion: yes for all spatial derivatives. Time and parameter differentiation
need not change: JVP handles physical time; AD differentiates the loss through
nonlocal matrix operations with respect to weights. All-spatial-nonlocal is
not all differentiation replaced, and does not itself introduce solved NS.

Current CH candidate:

$$
\phi_t+D_\delta(\phi\mathbf v)=-M L_\delta q,\qquad
q=\phi^3-\phi+\varepsilon^2L_\delta\phi .
$$

L_delta is the existing symmetric positive negative-Laplacian approximation;
the new divergence must annihilate total flux sums under periodic quadrature.
For a nodal gradient, a moment-corrected candidate is

$$
(G_\delta f)_i=K^{-1}\sum_j w_{ij}(f_j-f_i)\xi_{ij},\qquad
K=\sum_jw_{ij}\xi_{ij}\xi_{ij}^{T}.
$$

For uniform-grid Euclidean inner products choose D=-G^T. General quadrature
requires the corresponding weighted adjoint, not an unweighted transpose.
Then q^T D(phi*v)=-(Gq)^T(phi*v): coupling the future momentum equation
with -phi*Gq cancels this phase-advection work algebraically. This proves only
that coupling identity, not all of CHNS energy stability; kinetic advection,
viscosity, pressure/incompressibility, boundaries and discretization still need
their own compatibility. Soft PINN residuals do not impose an exact trajectory
energy law merely because operator identities hold.

Independent verification: fully-nonlocal-flow-check.py/.json alongside this
record run in0.012s with NumPy. n16,delta0.2 nodal radial cubic candidate:
moment identity error6.93e-18; constant-gradient5.55e-16; global divergence
sum0; adjoint gap1.42e-14; phase/capillary work gap2.13e-14. Low sine relative
derivative errors fall0.06635->0.01695 from n16,delta0.2 to n32,delta0.1,
an illustrative consistency check, not PINN or solution convergence evidence.

Important limitation discovered directly: alternating checkerboard gradient
max3.33e-16, while pair-Laplacian RMS497.07. Thus a naive symmetric nodal
gradient has an extra null mode on the even grid. D=-G^T and mass conservation
do not eliminate a pressure instability. Composition G^TG differs strongly
from the pair Laplacian (relative Frobenius0.9743 in this example); do not
replace the latter by this composition without analysis. This diagnostic
candidate is NOT proposed as a validated production NS operator. Need a
stable pair/discretization or justified stabilization and pressure nullspace
check. Analytic divergence-free velocity also needs a discrete divergence
audit when operators change.

Primary sources actually consulted:

- [Du et al. nonlocal vector calculus, author manuscript](https://ora.ox.ac.uk/objects/uuid%3A5d9607c1-46cb-48a5-b448-e3079cd04fec/files/mece8c9c0317b56e1cb9f4763c3c8344b).
- [Du/Tian nonlocal Stokes, author manuscript](https://arxiv.org/pdf/1805.08261),
  Section2 Eq2.7 gives negative adjointness; introduction and Section3 discuss
  kernel-dependent well-posedness. This is linear periodic Stokes, not a theorem
  covering our discrete nodal diagnostic, nonlinear NS or full CHNS.
- [Du/Tian nonlocal gradient stability](https://epubs.siam.org/doi/abs/10.1137/17M1139874),
  primary abstract confirms stability depends on kernel; no inaccessible full
  text is claimed as read.

No production/config/default/PDE/reference/checkpoint/training/Git change.
No new accuracy, speed or robustness claim. The diagnostic is independent
and does not import the training operators. Task-end memory: PROGRESS.md,
DECISIONS.md and this archived record. All63local links in three documents,
diagnostic JSON/source whitespace and git diff --check pass. No unrelated
test suite was run for this assessment.
