# Generalization of the latent phase residual

## Goal and acceptance criteria

Assess whether the melting tanh residual transformation can extend to broader
phase-transition models. Derive the general scalar transformation, distinguish
equation equivalence from training accuracy, and identify conservation,
pure-phase and coupling limitations with primary references. No implementation
or new numerical performance claims are requested.

## Context

Current melting uses phi=tanh(u), exact prescribed IC, and optimizes
r_u=r_phi/(1-phi^2), evaluated analytically in latent variables. Evidence so far
is limited to the recorded melting experiments. The user asks about broader
applicability, not a replacement of the validated default.

## Assessment

For scalar evolution R_phi=phi_t+F[phi,T,c]=0 and a smooth pointwise change
phi=g(u), g'(u)>0, define R_u=R_phi[g(u)]/g'(u). The pointwise zeros agree for
finite nonsingular latent values. This does not imply identical least-squares
objectives, conditioning, finite-error conservation or solution-error bounds.
Our MAE becomes mean(|R_phi|/g'); MSE would use weight1/(g')^2. The weight
depends on the network, so freezing it in differentiation defines another loss.

For constant a and phi_t-a*Laplacian(phi)+b*(phi^3-phi)+(1-phi^2)*q=0,
phi=tanh(u) yields

$$
r_u=u_t-a\Delta u+2a\tanh(u)|\nabla u|^2-b\tanh(u)+q.
$$

Here q can depend on x,t and physical coupled fields, evaluated consistently
with the original model. This identity holds in any spatial dimension and
does not prescribe geometry, speed, signed-distance structure or fixed width.
The hard IC is a separate choice; a generic strictly interior IC may use
u=g^{-1}(phi_0)+t*N. Exact endpoint data require special treatment because the
inverse is infinite. Other physically bounded variables can use a logistic
map or a model-specific monotone interface-profile map, with new derivations.

The isotropic, constant-W/tau specialization of PFHub benchmark3 is

$$
\tau\phi_t=W^2\Delta\phi+\phi-\phi^3
-\lambda U(1-\phi^2)^2,
\qquad U_t=D\Delta U+\tfrac12\phi_t.
$$

Its transformed phase equation is

$$
\tau u_t=W^2(\Delta u-2\tanh(u)|\nabla u|^2)
+\tanh(u)-\lambda U(1-\tanh^2(u)).
$$

The thermal source remains (1/2)*(1-phi^2)*u_t. This is a derivation for the
isotropic specialization, not the complete anisotropic benchmark. Coupling
terms and boundary/initial conditions must all remain consistent. Thermal
sharp layers can require additional treatment; phase scaling alone is not a
guarantee of temperature accuracy.

Cahn-Hilliard can formally use the substitution but requires chemical-potential
and flux/mass handling. A bounded map does not enforce mass conservation for
an approximate PINN. For no-flux boundaries and exact chemical potential,
d/dt integral(phi)=integral((1-phi^2)*r_u); an approximate mixed formulation
also has its chemical-potential error. Preserve the physical divergence form
and consider a mixed phi/mu formulation before deciding residual scaling.
Multiphase variables require coupled simplex constraints and a Jacobian on
independent coordinates; applying independent tanh maps does not ensure sum1.

Potential failure modes: exact pure states/out-of-range physical solutions,
sources not vanishing near saturation (division may blow up), stiffness and
strong penalties on physically tiny bulk errors, and competing smoothness
requirements near interface merging. Equal zero sets do not guarantee lower
terminal phase relative L2 or energy stability. The discrete nonlocal graph
must be transformed as defined; discretizing a transformed continuum PDE
generally gives a different discrete model.

## Primary sources checked

- Glasner, 2001, *Nonlinear Preconditioning for Diffuse Interfaces*, JCP174,
  695-711, doi:10.1006/jcph.2001.6933:
  https://math.arizona.edu/~kglasner/research/precond.pdf . Sections2-3 explicitly
  use atanh(phi), section4 studies coupled phase/temperature transformations.
  Its numerical-discretization gains are not PINN optimization guarantees.
- MOOSE developer documentation, *Linearized Interface Grain Growth Model*:
  https://mooseframework.inl.gov/modules/phase_field/Linearized_Interface_Grain_Growth.html .
  Documents multiphase grain-growth use of changed variables and Glasner origin.
- NIST PFHub benchmark3:
  https://pages.nist.gov/pfhub/benchmarks/benchmark3.ipynb/ . Gives anisotropic
  phase/temperature PDEs, heat source, no-flux conditions and observables.
- Mattey and Ghosh, *Gradient Flow Based Phase-Field Modeling Using Separable
  Neural Networks*, arXiv:2405.06119:
  https://arxiv.org/html/2405.06119v1 . Uses tanh-bounded neural output in a
  minimizing-movement method; distinguish this from our scaled strong residual.

## Suggested validation, not executed

Start with nonspherical/multiple-interface Allen-Cahn, then temperature-driven
melting/solidification, then a conserved phase-separation problem. For each,
compare raw output/original residual, bounded output/original residual, and
bounded output/transformed residual with identical IC enforcement, network,
samples, optimizers and budgets where meaningful. Assess full-domain terminal
phi relative L2, time history, interface geometry, coupled-field accuracy and
appropriate mass/energy balance, using independent converged references and
multiple seeds. Include width and parameter variations.

## Verification and changes

Complete; no blocker. Derived the identities by the scalar chain rule and
independently checked original versus transformed expressions using JAX AD
on nontrivial fields in1D/2D/3D and near-saturated values. Both scalar and
thermal identities passed tolerance1e-12 (maximum discrepancy2.220e-16,
float64, CPU). SymPy was unavailable;
no dependency was added. No training, PDE-solver experiment or production-code
change was performed. Updated this task, DECISIONS and PROGRESS; diff check
passed. This is an applicability assessment, not empirical generalization.

## Next action

None required for this discussion. A cross-model experiment remains a proposed
future task; preserve the current validated melting default.
