# Twin-model equations and feasibility

## Goal
Answer whether the proposed visually compelling mechanics/phase example is
too complex and explain governing equations for a feasible first model.

## Acceptance criteria
- Distinguish full finite-strain martensite from a proposed small-strain,
  two-variant, isothermal, quasistatic reduction.
- Derive phase driving force from the same elastic energy used in mechanics.
- Specify sufficient illustrative BC/IC and identify physical limitations.
- Explain nonlocal derivative needs without claiming automatic traction BCs
  or proven convergence/morphology.

## Context
No mechanics PINO, reference or constitutive parameters have been selected.
The preceding generated graphic is conceptual, not numerical evidence.
The user requests explanation and difficulty assessment, not implementation.

## Status
Complete (2026-11-05). Primary formulas and independent variation/BC review
confirm the proposed explanatory reduction. No implementation selection.

## Blocker
None for the explanatory reduction. Calibrated material/2D idealization remain
future implementation decisions.

## Next action
Await a future model selection. If implemented, fix the material, 2D mechanical
idealization, boundary profile family and interface resolution before reference
checks and neural training.


## Explicit explanatory reduction

Fixed reference domain; two existing variants; small total/transformation strain,
isothermal Allen--Cahn kinetics, quasistatic mechanical equilibrium, no body
force. This does not reproduce finite-strain/lattice-rotation/multivariant
features of the full cited martensite model. Let \(\eta=0,1\) denote variants A,B
and let \(\mathbf u=(u_x,u_y)\). Constant symmetric positive elastic tensor
\(\mathbb C\), positive \(W,\kappa,M\); plane-stress/plane-strain reduction must
be selected explicitly before implementation.

$$
\varepsilon=\tfrac12(\nabla\mathbf u+\nabla\mathbf u^T),\qquad
\Delta\varepsilon=\varepsilon_B-\varepsilon_A,\qquad
h(\eta)=\eta^2(3-2\eta),\quad
\varepsilon^{\rm tr}=\varepsilon_A+h(\eta)\Delta\varepsilon.
$$

$$
E=\int_\Omega\left[
W\eta^2(1-\eta)^2+\frac\kappa2|\nabla\eta|^2+
\frac12(\varepsilon-\varepsilon^{\rm tr}):\mathbb C:
(\varepsilon-\varepsilon^{\rm tr})\right]\,dx,
\qquad\sigma=\mathbb C:(\varepsilon-\varepsilon^{\rm tr}).
$$

At fixed displacement and constant stiffness, independent variation gives
\(\partial_\eta\psi_{\rm el}=-h'(\eta)\sigma:\Delta\varepsilon\).
Therefore the three scalar PDEs are

$$
\nabla\cdot\sigma=0,\qquad
\eta_t=M\left[\kappa\Delta\eta
-2W\eta(1-\eta)(1-2\eta)
+6\eta(1-\eta)\sigma:\Delta\varepsilon\right].
$$

This is nonconserved variant switching, not conserved concentration evolution.
No chemical-potential output or pressure/incompressibility constraint is needed.
If stiffness becomes phase dependent, its derivative contributes another term;
the above formula must not be reused unchanged.

An illustrative small-strain pair is
\(\varepsilon_A=\varepsilon_s\operatorname{diag}(1,-1)\),
\(\varepsilon_B=-\varepsilon_A\). It expresses different preferred extension
axes and admits small-strain compatible twin directions. It is not a calibrated
specific material model or proof of a particular branched morphology.

## Illustrative BC/IC and power balance

On a fixed rectangle: clamp bottom \(\mathbf u=0\); prescribe top
\(u_y=-a(t)g(x)\), with \(\sigma_{xy}=0\) for its free tangential component;
vertical sides \(\sigma n=0\); neutral phase boundary \(\partial_n\eta=0\)
on all edges. Use a common smooth ramp with \(a(0)=0\), a smooth displacement
profile supported away from corners and a fixed mixed-domain \(\eta_0\).
The whole-top displacement condition sets \(u_y=0\) where \(g=0\): those
segments are normally constrained, not traction-free surfaces outside an actual
contact patch. This idealized boundary profile does not solve unilateral contact.

The initial displacement must equilibrate the initial eigenstrain and boundary
conditions; zero external displacement does not imply \(\mathbf u_0=0\).
Since the well and interpolation derivatives vanish at both pure variants,
a perfectly homogeneous pure initial state cannot spontaneously nucleate in
this deterministic model. Fix mixed domains or explicit embryos consistently.
Continuum boundedness does not itself impose bounds on a neural output.

$$
\frac{dE}{dt}=-\int_\Omega\frac{\eta_t^2}{M}\,dx
+\int_{\partial\Omega}(\sigma n)\cdot\mathbf u_t\,ds.
$$

For the top prescription, the loading power is
\(-a'(t)\int_{\Gamma_{\rm top}}g(x)\sigma_{yy}\,ds\).
Energy need not decrease during loading; it dissipates with boundary positions
held fixed under the stated zero-traction/no-flux conditions.

## Difficulty and operator implications

The independent-order-parameter reduction has one first-order time equation and
spatial derivatives of order at most two. Compute strain and phase gradients,
then stress divergence and phase Laplacian with appropriately consistent
nonlocal approximations; time JVP remains applicable. Fixed geometry permits
precomputation. Mechanical traction and displacement need their own treatment;
scalar no-flux closure does not supply traction automatically.

Main challenges are resolved narrow walls, elasticity/phase scale disparity,
metastability, loading-history dependence, and genuine morphology sensitivity.
Fewer fields than NS--CH is not proof of easier training. The generated
illustration is not numerical evidence; use reference cases before promising
fine branching. Fix the ramp when conditioning only on spatial g.

## Traceability

- Levitas, Roy, Preston, PRB88,054113(2013), SectionIII equations12--16: two-
  variant double well, gradient energy and cubic interpolation; SectionV retains
  finite transformations/rotations and interfacial stress. The present fixed-
  reference small-strain energy is explicitly simplified, not full reproduction.
  https://harvest.aps.org/v2/journals/articles/10.1103/PhysRevB.88.054113/fulltext
- Bhattacharya/Zaeem, Scripta Materialia252,116273(2024), equations1--4 and
  mechanical equilibrium: three-variant small-strain TDGL/eigenstrain framework.
  https://www.imechanica.org/sites/default/files/Ferroelasticity_Nucleation_Scripta%20Materialia_2024.pdf
- MOOSE official mechanics coupling: elastic-energy feedback and variable
  eigenstrain/stiffness.
  https://mooseframework.inl.gov/modules/phase_field/Mechanics_Coupling.html

## Verification and changes

Independent analytical review confirmed phase-variation sign, derivative factors,
mechanical natural boundaries and external-power balance. Reviewed the cited
formulations. Only PROGRESS, DECISIONS and this archived task changed in this
turn; no code tests warranted. Local links and git diff whitespace checked.
