# Boundary-conditioned phase-field/displacement PINO feasibility

## Goal
Analyze harder two-dimensional phase-field/displacement neural operators with fixed geometry and initial state and parameterized boundary functions. The original request concerned spatial traction; the user refined it on 2026-11-03 to ask for several feasible alternatives, preferably boundary-condition parameterizations. Analysis only, not implementation or training.

## Acceptance criteria
- Identify the actual coupling and limitations of the existing precipitate2d PINN.
- Explain a physically meaningful load-to-microstructure operator, admissible boundary data, network conditioning and nonlocal discretization needs.
- Distinguish plausible morphology changes from guaranteed responses; expose weak load sensitivity and validation pitfalls.
- Compare displacement-driven coherent precipitation, boundary chemical-potential/flux-driven chemoelastic phase separation, and multivariant ferroelastic transformation.
- Recommend a staged pilot with independent reference checks, without selecting unverified parameters or launching jobs.

## Context
Four PINO README/publication tasks are complete. The previous elastic pilot uses conserved CH, phase-dependent eigenstrain, homogeneous moduli and exact periodic spectral mechanical equilibration. Its harder multiparticle version is an untuned moderate-error baseline, not a high-accuracy completed PINO. User now proposes arbitrary along-edge applied stress to make the mechanics/phase coupling more interesting.

## Status
Complete (2026-11-03). Existing implementation inspected and three boundary-conditioned candidates compared using primary papers and official mechanics-coupling documentation. The user refinement is within the same feasibility task. Recommendations below are design proposals, not validated numerical results or a selected implementation.

## Blocker
None for the completed analysis. Material model, plane-stress versus plane-strain idealization, boundary prior and computational budget must be explicit before future implementation.

## Next action
None required for analysis. If the user chooses a candidate, open a new implementation task and first check an independent small set of boundary-conditioned reference solutions for meaningful two-way coupling and distinguishable morphology. No source/config changes, training, reference solves or Git mutations were performed.


## Existing implementation and its limits

`examples/precipitate2d` uses conserved Cahn-Hilliard phase evolution, a mixed
chemical-potential network output, homogeneous isotropic elasticity, linear
phase-dependent eigenstrain and fixed-zero-macrostrain periodic boundaries.
Displacement is solved by differentiable Fourier equilibration; it is not
independently predicted. The harder three-particle opt-in baseline has
common-local pooled phase errors about8.03%nonlocal/12.90%local AD at the matched
final budget. It is not already a completed high-accuracy elastic PINO.

## Recommended candidates

1. **Spatial displacement boundary -> coherent precipitate evolution.** Fixed
   rectangle and several fixed unequal off-axis precipitates; clamp the left
   edge and prescribe a smooth two-component displacement curve on the right.
   Top/bottom can remain traction free, with neutral/no-flux phase conditions.
   Use one fixed compatible loading ramp and random spatial curves, rather
   than random loading histories in the first pilot. This is the most direct
   reuse of existing phase/eigenstrain physics. Dirichlet lifting can impose
   prescribed displacements exactly, while free-edge mechanical consistency
   still needs validation. Strain/stress can change diffusion-driven shape,
   growth competition and orientation, but no specific response is guaranteed.
   Boundary variations should be smooth and on lengths that penetrate to the
   particles, rather than many unresolved high-frequency modes.

2. **Boundary chemical potential or influx -> chemoelastic phase separation
   and swelling.** Fixed solid and initial concentration; prescribe a spatial
   chemical-potential function OR a species flux on an exposed edge, with
   fixed mechanical supports. Concentration-dependent stress-free strain
   causes deformation, and elastic free energy feeds back into chemical
   potential. A simplified intercalating-solid/absorbing-solid pilot has a
   meaningful context without claiming a full calibrated battery model.
   Different supply patterns may create different fronts and stress hot spots;
   diffusion can still smooth fine boundary details, so test response first.
   It is an open system: with outward flux J.n, total concentration changes
   by minus its boundary integral. Do not reuse the closed-system constant-mass
   projection. Chemical-potential data prescribe no known total influx in
   advance; its flux must be solved. Do not prescribe both flux and chemical
   potential on the same boundary component. Use compatible initial/ramped data.
   The cited LiFePO4 paper is reaction limited and anisotropic: it motivates
   coupling, not an assertion that this simpler boundary-diffusion proposal
   reproduces its equations or quantitative morphology.

3. **Boundary displacement -> ferroelastic/martensitic variant evolution.**
   Fixed initial mixed-variant seeds and geometry; random tension/shear
   displacement curves can alter favored variants and domain-wall motion.
   This is potentially the strongest visual example, but needs a traceable
   multivariant free energy/eigenstrain model, additional order parameters,
   treatment of metastability and fixed loading histories. It is not simply
   the current CH equation with a new label. Full ferroelectric material
   models may also require electric fields; a mechanical-only pilot should
   select an appropriate ferroelastic model explicitly. Higher development
   risk makes this a later option rather than the first extension.

Relative difficulty and visual promise are engineering judgments, not measured
performance. The first candidate is the recommended next mechanics-conditioned
PINO; the second is a useful distinct physical application. No user selection
or numerical parameter set has yet been inferred.

## Important physical and numerical checks

- Two-way coupling requires the elastic free energy in the phase chemical
  potential, not only phase-dependent mechanics after an uncoupled phase solve.
  For constant stiffness C and eigenstrain h(phi)E, its fixed-displacement
  phase derivative is -h'(phi)sigma:E.
- With constant C, linear h and a spatially uniform added stress, that added
  chemical potential is constant; closed-system CH applies its Laplacian
  and removes it. Thus uniform loading alone need not change morphology.
  This is a checked restricted limiting case, not a statement that all applied
  strain has no effect. Nonuniform loading or carefully justified material
  contrast can remove this degeneracy.
- A spatially varying traction is the vector sigma.n, not arbitrary independent
  prescription of every tensor stress component. Pure traction also requires
  force/moment compatibility and rigid-mode treatment; a support changes the
  physical problem and supplies reaction forces.
- Quasistatic mechanics needs no independent displacement initial condition.
  Fixed phase IC with nonzero eigenstrain generally has nonzero equilibrated
  displacement, even at zero applied boundary displacement. Do not hard-code
  full-domain u(t=0)=0.
- Phase, chemical potential and displacement require physical nondimensional
  output/residual scales; small displacement can coexist with meaningful
  elastic energy. Predict phi,mu,ux,uy as one possible mixed architecture;
  compute stress constitutively. Eliminating mu or solving mechanics exactly
  are later explicit comparisons, not silent implementation decisions.
- Nonlocal gradients/Laplacians can be used, but scalar no-flux graph closure
  does not automatically implement elasticity traction, prescribed displacement
  or nonzero chemical influx. Check rigid motions, affine strain, force/torque
  balance, quadrature/adjoint consistency and shrinking-horizon behavior before
  interpreting a neural training error. Fixed geometry permits precomputation.
- For fixed dead traction, the mechanically stationary potential includes
  external work (internal energy minus boundary traction work). Do not apply
  the old closed periodic internal-energy monotonicity audit unchanged.
  Time-varying displacement/traction also requires boundary-power accounting.

## Shared neural-operator design

Encode sampled boundary coordinates and the prescribed function values using
point-token lifting and attention. A displacement input contains the two
known displacement components; a chemical input contains the selected scalar
chemical potential or flux. Include normals/condition types when using a
heterogeneous boundary representation, without inventing unknown traction
values on clamped boundaries. Decode common (x,y,t) queries through cross
attention into the physical fields. Fixed geometry/IC need not be redundantly
encoded as sample-varying images. Use fresh smooth random function coefficients
online; no per-sample geometry/operator rebuild or training-solution library
is necessary. Fixed quadrature/operators and shared space/time preparation
can reuse existing tensor-split optimizations.

## Validation order for a future selected pilot

1. Select a published/derived constitutive model and explicit boundary family.
2. Check mechanical/phase boundary and operator consistency independently.
3. Refine a few independent local references and compare distinct boundary
   curves plus a no-elastic-feedback control; confirm morphology actually
   depends on the input, separately from drawing a deformed plotting grid.
4. Compare a direct nonlocal baseline to local references, then train PINO.
5. Evaluate reserved boundary functions: phase/displacement/stress errors,
   interface/phase-region errors, mass balance, force balance and correct
   energy/work accounting. Measure input-response error as well as total-field
   relative L2 so an average prediction cannot look like operator learning.

## Primary sources checked

- [MOOSE Mechanics Coupling](https://mooseframework.inl.gov/modules/phase_field/Mechanics_Coupling.html): bidirectional eigenstrain/material-energy coupling.
- [Zhang et al.2006](https://wulixb.iphy.ac.cn/article/doi/10.7498/aps.55.2433): coherent precipitation under applied strain; publisher abstract retrieved, subsequent page open timed out.
- [Cogswell and Bazant2012](https://arxiv.org/abs/1110.3877): coherency strain, intercalation phase separation and reaction-limited kinetics; not the same boundary model as the simplified proposal.
- [Xue et al.2016](https://arxiv.org/abs/1610.01420): imposed strain and ferroelastic phase/domain formation.
- [University of Alberta potential-energy derivation](https://engcourses-uofa.ca/books/introduction-to-solid-mechanics/variational-principles/the-principle-of-minimum-potential-energy-for-conservative-systems-in-equilibrium/): external work in traction-controlled mechanics.

## Completion and verification

Only repository memory was changed. No new reference, trained model, accuracy
claim, example default, source/config change, Git commit or push. Documentation
links and diff whitespace checked; no code tests are warranted for analysis-only
text. The deferred channel20000/lower-final-LR idea remains unlaunched.
