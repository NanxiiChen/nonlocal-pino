# Boundary-conditioned phase-field RVE candidates

## Goal
Recommend a phase-field/displacement fine-scale parametric operator whose
boundary loading is supplied by a prospective macroscopic mechanics model,
without implementing full multiscale coupling now.

## Acceptance criteria
- Explain admissible first-order homogenization boundary conditions and outputs.
- Compare physically motivated alternatives beyond the preceding twin sketch.
- Retain loading history and evolving internal state in operator semantics.
- Distinguish genuine two-way coupling from mechanically passive phase patterns.
- Identify limits of fracture localization, cell representativity and nonlocal
  discretization before claiming a homogenized constitutive surrogate.

## Context
The user seeks a fifth PINO combining phase and displacement, with fixed initial
state/geometry and parameterized loading. Visual morphology matters. Previous
mechanics examples remain conceptual; no constitutive model is selected.

## Status
Complete (2026-11-05). Primary-source comparison and independent boundary,
history and constant-stiffness limiting-case checks completed.

## Blocker
None for analysis; candidate and material selection precede implementation.

## Next action
Await candidate selection. Leading new candidate: elastic-loading-driven
polycrystal grain-boundary migration/texture evolution. Lower-reuse-cost
alternative: inhomogeneous-elastic two-phase directional coarsening. Before any
PINO, verify distinct load responses and averages on a few cell references.


## Operator and boundary interpretation

For a fixed small-strain cell Y and fixed initial internal state, the input is
macroscopic strain history, not just instantaneous strain:

$$
\mathcal G:\{\overline\varepsilon(s):0\leq s\leq t\}\mapsto
\{\eta_i(\mathbf x,t),\widetilde{\mathbf u}(\mathbf x,t),
\overline\sigma(t)\}.
$$

Classical first-order periodic loading uses

$$
\mathbf u=\overline\varepsilon(t)(\mathbf x-\mathbf x_c)
+\widetilde{\mathbf u},\qquad
\langle\widetilde{\mathbf u}\rangle_Y=0,\quad
\overline\sigma=\langle\sigma\rangle_Y.
$$

Opposite-face fluctuations are periodic and paired outward tractions are
anti-periodic. Phase fields are periodic as well for a repeated-cell pilot;
closed no-flux phase boundaries would be a different cell model. The mean strain
is the prescribed symmetric tensor. Tensor shear E12 differs from engineering
shear gamma12=2E12. Affine displacement on every boundary (KUBC) is another
valid choice, generally not identical to periodic fluctuations on a finite cell.
Arbitrary independent random boundary profiles are not the standard first-order
macrostrain-to-cell mapping. Finite-strain loading instead uses a consistent
Fbar/Pbar formulation; do not mix finite and infinitesimal conjugate measures.

Mechanical Hill--Mandel power:

$$
\overline\sigma:\dot{\overline\varepsilon}
=\langle\sigma:\dot\varepsilon\rangle_Y.
$$

This is not equality to stored-energy rate alone: phase evolution changes
internal energy and dissipates energy. Full phase-field homogenization may
introduce additional macrovariables/microforces; this pilot only establishes a
mechanically driven fine-scale response map, not full FE2.

## Leading new candidate: stress-driven grain growth / texture evolution

Fixed periodic 2D multigrain pattern, fixed crystal orientations and temperature.
Orientation-dependent elastic stiffness couples grain identity to stress.
Changing macroscopic extension/shear/mixed loading can select different grains
for growth, change the boundary network and alter the averaged mechanical
response. Output multiphase fields, displacement fluctuations, stresses,
orientation fractions and the average stress--strain/relaxation curves.

Primary evidence:
- Tonks/Millett2011, MSEA528,4086--4091, DOI10.1016/j.msea.2011.02.007:
  2D copper simulations under different deformations; preferred orientation
  depends on load; elastic driving weakens as texture develops.
  https://www.sciencedirect.com/science/article/pii/S0921509311001584
- Shahnooshi et al.2019, JCG518,18--29, DOI10.1016/j.jcrysgro.2019.04.015:
  stressed grain growth in a 2D polycrystalline RVE, including elongation/shear.
  https://pure.korea.ac.kr/en/publications/phase-field-modeling-of-stressed-grain-growth-effect-of-inclinati/
- Tonks et al.2010 explains elastic strain-energy driving and why a simplistic
  rule that the locally lower-energy grain always grows is wrong under
  nonuniform strain; the total elastic equilibrium redistribution matters.
  https://micro.stanford.edu/~caiwei/papers/Tonks10script-gbmig.pdf

Model limitations: same isotropic stiffness in every grain removes the desired
orientation-dependent elastic selection. Use a traceable anisotropic crystal
model. Multiple grain order parameters/constraints and stiffness interpolation
are additional numerical work. Elastic driven boundary migration is not a full
recrystallization model: plasticity, dislocation storage and nucleation are
additional physics. Morphology and macro-response separation must be measured;
later evolution may be mostly capillary-driven. No guaranteed pattern claimed.

Proposed first study: fixed periodic cell with a modest number (e.g.8--12) of
initial grains and fixed orientations, one temperature, small-strain mechanics,
moderate loading and explicit phase-field kinetics. This small illustrative cell
is not statistically representative solely because it is called an RVE.

## Alternative: two-phase directional coarsening / rafting

A fixed multiparticle or bicontinuous initial pattern can migrate/coalesce into
oriented structures when elastic moduli differ. This reuses CH/elasticity more
than a multigrain model; it is more collective than one isolated precipitate.
Classic source: Gururajan/Abinandanan2007, Acta55,5015--5026,
DOI10.1016/j.actamat.2007.05.021. Source indexed abstract explicitly identifies
modulus mismatch, homogenization-inspired equilibrium and uniaxial-load rafting.
https://www.sciencedirect.com/science/article/abs/pii/S1359645407003515
Supporting open primary texts:
https://www.math.purdue.edu/~shen7/pub/CSZT4.pdf
https://jmmab.com/wp-content/uploads/2019/03/1450-53391800009M-77c.pdf
The latter2019 study includes periodic boundaries and imposed axial strain.
Pure elastic diffusion-driven rafting is not a quantitatively complete
high-temperature superalloy creep/plasticity model.

Independent limiting-case audit: for constant C, affine eigenstrain Bc,
periodic mechanics and conserved closed CH, changing homogeneous Ebar adds
constant stress C:deltaE and a spatially constant chemical-potential shift
-(C:deltaE):B. CH annihilates this shift, so concentration histories may be
identical across loads. Moreover

$$
\overline\sigma=\mathbb C:(\overline\varepsilon-B\langle c\rangle),
$$

which is insensitive to morphology when mean c is conserved. This explains why
simply wrapping the existing homogeneous-elastic precipitation model in an RVE
boundary input can yield a trivial operator. Require physically justified C(c)
contrast or another explicitly derived mechanism; do not invent load terms.

## Other candidates and limits

- Composite microdamage/microcracks: particularly direct nonlinear constitutive
  application, visible different crack paths and stiffness degradation.
  Bharali/Larsson/Janicke2021 distinguishes selective fine-scale-only phase
  fields from fuller phase-field homogenization, with periodic inclusion cells:
  https://research.chalmers.se/publication/522540/file/522540_Fulltext.pdf
  Irreversibility, instability and cell-spanning localization complicate
  representativity/scale separation. Preserve the separate fracture-project
  scope; a fixed-cell operator is not automatically a general FE2 material law.
- Ferroelastic/martensitic RVE remains a suitable prior candidate for load-
  dependent variant selection and history effects, but not a new alternative.
- Crystal-plasticity/recrystallization is physically relevant but adds slip,
  dislocations and nucleation; defer as first implementation. Primary example:
  https://doi.org/10.1016/j.cma.2014.12.007
- Recent2026 FE2 phase-field preprint includes mechanical martensitic coupling,
  but is a preprint and has richer macro phase variables than this pilot:
  https://arxiv.org/abs/2607.09356

## Conditioning and validation recommendation

First use proportional paths Ebar(t)=a(t)E0 with one fixed ramp and random
extension direction, biaxiality and shear mixture. This is a low-dimensional
parameterized solution family; a large attention encoder is not inherently
necessary. To make a functional boundary-history operator, later use smooth
nonproportional strain paths and time tokens (s,E11,E22,E12); preserve prefix
causality and fixed initial state. Histories cannot be replaced by the current
strain once internal fields evolve. Source for history-aware homogenization:
https://arxiv.org/abs/2210.17443

Fixed geometry permits shared query coordinates and precomputed periodic
nonlocal neighborhoods. Approximate derivatives of periodic fluctuation fields;
add the affine macrostrain analytically rather than incorrectly wrapping an
unadjusted affine displacement through a periodic stencil. Mechanical traction,
quadrature and work/energy consistency still need independent validation.

Assess phase/boundary morphology, displacement/stress errors, average-stress
curves, loading sensitivity versus a mean-field predictor, and mechanical work
and phase dissipation. Full macro coupling additionally needs incremental state
handling and consistent tangent or equivalent update; it is outside this task.

## Completion verification

Primary papers/repository policy reviewed; independent agents checked the
boundary/history map and the constant-C counterexample. Documentation links and
git diff whitespace checked. Only PROGRESS, DECISIONS and this archived task
changed in this turn. No code tests, numerical solves, training or Git mutation.
