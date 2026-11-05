# Obstacle-conditioned two-phase flow candidates

Date: 2026-11-05

## Goal and acceptance

Recommend a cylinder-flow-like phase-field example with parameterized fixed
obstacle positions and shapes. Explain the physical meaning of the phase field,
how geometry can change interface morphology, and a feasible first pilot using
the existing coupled NS--CH experience. Cite primary evidence and distinguish
expected behavior from simulated results. No solver/training is requested.

## Context

User redirects from elastic ATG interfaces to fluid mechanics. Existing
channel_nsch2d provides a geometry-conditioned coupled two-fluid pilot, while
confined_ch2d has irregular domains with internal holes. A fixed solid obstacle
can be represented as domain geometry separately from the evolving phase field.
Need not introduce moving solids, elastic deformation or melting to include a
physically active phase field.

## Status

Complete. Recommend a fixed droplet in a two-fluid channel with one or two fixed
posts, varying centers, semiaxes and orientation within minimum-clearance and
initial-drop exclusion bounds. Keep initial fluid configuration, flux, material
properties and wall wetting common. The evolving phase field describes the two
fluids; solid geometry is a separate domain/encoder input. NS and CH are coupled
through advection and capillary force, not passive dye transport.

The useful progression is deflection/deformation first, then possible unequal
splitting or rejoining. Geometry affects trajectory, flow division and necking,
not only a scalar deformation amplitude. These are possible regimes, not a
promise that the proposed untested parameters exhibit all of them. Drops that
are very small relative to posts/gaps or highly surface-tension dominated may
give little morphology variation. Low-inertia viscous/capillary deformation is
a sensible first target; vortex shedding is a separate harder extension.

A related richer candidate is immiscible displacement through a small obstacle
array, with a common initial front and different solid layouts. It has pore-scale
displacement applications but adds many contact lines, thin films and potential
trapping events. Free-surface flow over a cylinder is another physical variant;
it adds gravity/density contrast and is not the first recommendation.

## Primary evidence

- Link et al. (2004), experimental geometry-mediated droplet breakup, including
  centered/off-center obstacles and unequal daughter droplets:
  https://doi.org/10.1103/PhysRevLett.92.054503
  Author full text independently read by the research agent:
  https://www.princeton.edu/~stonelab/Publications/pdfs/From%20Howard/PRLandPREandPRB/DLinkSAnnaPRLFeb04.pdf
  This is an experiment, not a phase-field solver. Surfactants stabilize daughter
  drops; do not promise identical long-term behavior in a surfactant-free model.
- Protiere et al. (2010), circular-post bypass/breakup experiments and reduced
  capillary/permeability model:
  https://web.mit.edu/bazant/www/papers/pdf/Protiere_2010_EPL_droplet_breakup.pdf
  Actual quasi-2D device has confinement/films; a 2D pilot is not a quantitative
  replication without modeling and resolution checks.
- Yang and Kim (2020), modified NS and ternary CH with fixed solid phase,
  triangle-obstacle droplet and porous-medium tests:
  https://mathematicians.korea.ac.kr/cfdkim/wp-content/uploads/sites/15/cfdkim/papers/TwoPhaseAribraryDomain.pdf
  DOI 10.1016/j.camwa.2019.10.008. This supports phase-field feasibility; it does
  not require adopting its third-field representation in this repository.
- Stoter et al. (2023), immersed NS--CH with generalized Navier slip and porous
  geometry breakup/coalescence:
  https://doi.org/10.1016/j.cma.2023.116483
  https://arxiv.org/abs/2305.19237

## Repository reuse and limits

Read examples/channel_nsch2d/README.md, independently audited by a second agent.
Reuse coupled NS--CH experience, time JVP, point geometry encoder, rolling
operator pool and interface/minority diagnostics. The current channel is
explicitly without holes; its height-mapped hard streamfunction lift cannot
simply impose internal-post no-slip. A multiply connected lift would need
correct boundary streamfunction constants/flow partition, or a validated
velocity/continuity formulation. No implementation choice made here.

Nonlocal neighborhoods need fluid-domain/visibility treatment near posts and
narrow gaps so they do not connect directly through solids when approximating
the local fluid PDE. Solid-wall velocity, wetting/contact-angle and chemical
no-flux conditions require explicit treatment. Nonlocal differentiation alone
does not impose all these laws. Thin films and necks need spatial, time,
interface-width and mobility sensitivity checks before interpreting breakup.
Current traction outlet already fixes pressure reference; no extra point anchor
is automatically needed in a channel extension with the same outlet law.

Validation: primary literature checked, two independent agent audits completed,
repository assumptions inspected. Documentation-only diff/link checks; no code
tests or numerical runs.

## Blocker and next action

No blocker. Await user's choice of example/schematic before implementation.
Existing default runs unchanged; no training, cleanup or Git history operation.
