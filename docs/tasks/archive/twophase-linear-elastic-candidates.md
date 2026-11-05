# Two-phase interface candidates with simple linear elasticity

## Goal
Recommend visually clear, physically grounded two-phase interface evolution
cases coupled to small-strain linear elasticity, avoiding crystal/grain physics.

## Acceptance criteria
- Name the physical phases and a genuine interface migration mechanism.
- Identify a classic example and plausible boundary parameterization.
- Separate linear elastic mechanics from nonlinear moving-interface physics.
- State limits of candidate simplifications, without invented equations,
  guaranteed morphologies or starting code/simulations.

## Context
User prefers larger-scale interfaces, accepts simple linear elastic coupling,
and seeks alternatives to the previous crystal/grain and simple precipitate
cases. Existing corrosion/electropolish phase-field code may be reusable;
mechanics coupling and boundary treatment are not already validated there.

## Status
Complete (2026-11-05). Primary-source searches and both parallel candidate audits
completed; recommend ATG as the new classic visual candidate, with corrosion
and chemoelastic fronts as alternatives. Relative difficulty is an engineering
assessment, not measured PINO evidence.

## Blocker
None for recommendations; material/model selection precedes implementation.

## Next action
Await user candidate selection. A future implementation must select the actual
kinetics, loading/boundary setup and physical scales before numerical work.

## Preferred candidate: ATG surface instability

A continuum elastic solid in contact with its melt or another external phase
has a weakly corrugated interface. Under nonhydrostatic stress, elastic-energy
release competes with surface energy; selected disturbances grow into grooves,
which compete and coarsen. No grains/orientations are needed in the isotropic
model. The changing interface comes from mass transfer/phase change, not elastic
buckling or simply plotting a deformed mesh. Characteristic scales depend on
modulus, stress and interface energy; no arbitrary macroscopic length promised.

Primary literature admits quasistatic Hooke elasticity and a liquid with no
shear resistance at fixed pressure, without hydrodynamics. A bounded first
proposal is fixed temperature, fixed weakly wavy initial interface, uniform
external pressure and interface-controlled melt/solid kinetics, with mechanical
boundary strain amplitudes or smooth profiles as inputs. Fast thermal/fluid
equilibration are assumptions; this is not full Stefan/NS coupling. A solute
dissolution interpretation additionally requires explicit reservoir/transport
assumptions; it is not automatically a quantitative corrosion model.

The nonconserved interface-reaction phase field and conserved surface-diffusion
variant are different physical models. Surface diffusion needs appropriate
tangential/degenerate transport with a checked sharp-interface limit, not an
arbitrary ordinary CH replacement. For the initial phase-change variant, the
core fields can be phi and two displacement components; auxiliary pressure or
constraints may depend on the adopted liquid treatment, so no architecture is
fixed here.

Stress amplitude can affect unstable wavelengths as well as rates; it is not
generically a global clock rescaling. Symmetric isotropic models may depend on
stress squared, so changing tension to compression alone need not yield a new
morphology. Validate small-wave growth first, then finite groove competition;
stop before sharp tips invalidate small-strain/no-plasticity physics. No
guaranteed branching or fracture claim.

Numerical difficulty remains in the moving stress boundary, near-zero external
shear stiffness and energy scales. Artificial external stiffness is a model
regularization requiring convergence checks. Nonlocal derivatives do not
automatically enforce traction continuity or the correct solid/fluid limit.

## Alternatives

### Mechanically assisted dissolution/corrosion

Fixed metal/electrolyte geometry and initial pits, variable applied displacement
or loading history, with fixed chemical environment. Solid dissolution changes
load-bearing shape and stress; stress can alter interface kinetics. Reuses
existing corrosion concepts but requires a newly validated mechanics coupling.
The actual strength of morphology sensitivity under elastic-range loads must
be established; diffusion limitation can suppress kinetic stress effects.

Cui/Ma/Martinez-Paneda2021 separates a hydrostatic stress kinetic factor from
plastic strain and film effects, so an elastic/no-film limit can be defined.
This source deliberately does not insert mechanical free energy directly into
the dissolution phase driving term (discussion after equations34--35); it
uses stress-sensitive mobility and phase-dependent mechanical degradation.
Do not silently replace it with variation of total chemical-plus-elastic energy.
The full publication includes plasticity/passivation, and its dramatic
pit-to-crack morphologies are not evidence that the elastic subset reproduces
them. Reserve detailed SCC/plasticity/fracture for later material selection.

### Chemoelastic two-composition front

Rich/poor species phases with different stress-free strain are driven by a
boundary chemical potential, reaction or flux under fixed mechanical supports.
Elasticity feeds back on phase evolution. An intercalating-solid front is a
physical example; it is not grain migration, though calibrated battery models
can have material-specific anisotropy. Cogswell/Bazant2012 uses small-strain
elasticity and a reaction-limited LiFePO4 model, not an arbitrary isotropic CH
bulk-diffusion model. Avoid calling large-swelling gels or silicon insertion
linearly elastic over their full transformation strain. This option is most
natural for chemical-boundary conditioning, less directly the user's earlier
mechanical-boundary objective, so secondary here.

## Checked primary sources

- Kassner et al., PRE63,036117(2001), Phase-field modeling of stress-induced
  instabilities. Author full text, including Hooke elasticity, external phase
  without hydrodynamics and groove competition:
  https://arxiv.org/html/cond-mat/0005476
  https://doi.org/10.1103/PhysRevE.63.036117
- Kassner, How to model surface diffusion using the phase-field approach:
  https://arxiv.org/abs/cond-mat/0607823
  (conserved-model asymptotic caveats, checked by candidate reviewer).
- Misbah et al.(2004), Dynamics of a dissolution front for solids under stress:
  https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2003GL019136
  (linear elasticity plus interfacial dissolution and solute transport,
  experimental/geophysical scale context).
- Cui, Ma, Martinez-Paneda(2021), dissolution-driven SCC:
  https://arxiv.org/abs/2011.12068
  https://arxiv.org/pdf/2011.12068
- Cogswell/Bazant(2012), Coherency strain and kinetics of phase separation:
  https://arxiv.org/abs/1110.3877
  https://web.mit.edu/bazant/www/papers/pdf/Cogswell_2012_ACS_Nano_ASAP.pdf

## Completion verification

Only memory documents changed: PROGRESS, DECISIONS and this archived task.
Inspect record, verify dashboard links and git diff whitespace. No code tests,
default changes, numerical solve, training, image generation or Git operation.
