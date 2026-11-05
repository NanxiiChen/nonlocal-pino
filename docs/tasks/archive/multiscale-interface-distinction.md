# Grain-boundary versus corrosion/fracture multiscale explanation

## Goal
Explain loaded polycrystal interface evolution using the user's familiar
mechanically influenced corrosion and fracture, separating physical mechanisms
from the architecture of multiscale coupling.

## Acceptance criteria
- Explain crystal orientation, grain boundaries and migration without implying
  cracks, material dissolution, crystal rotation or mere elastic deformation.
- Contrast physical interfaces, driving forces and feedback to macro mechanics.
- Show that both material mechanisms can use RVE homogenization or local models;
  identify extra environment inputs and localization/representativity limits.
- Give an intuitive example and practical implications for the proposed PINO.

## Context
Previous archived RVE candidate task recommends macrostrain-conditioned grain
growth; the user has not worked at grain scale. Latest request is explanation,
not candidate selection, coding, simulation or Git publication.

## Status
Complete (2026-11-05). Primary-source and independent conceptual audit complete;
Chinese explanation compares interface meaning, mechanical feedback and two
common scale-coupling architectures.

## Blocker
None.

## Next action
Await the user's preferred physical mechanism. No new mechanics example is
selected or authorized for implementation in this conceptual turn.

## Explanation and qualifications

- A grain is a region of common lattice orientation. In the proposed migration
  model, adjacent grains remain bonded solid and usually the same chemical
  phase. Boundary motion changes the region occupied by each orientation;
  it does not require mass loss, elastic elongation or rigid grain rotation.
  A continuum phase-field grain model does not explicitly simulate atoms.
- Corrosion moves a solid/environment interface through dissolution/transport;
  fracture creates separation and loses load transfer without necessarily
  removing material. The phase variables have different physical meanings.
- All can couple mechanical equilibrium to evolving internal fields. Grain
  energy includes boundary and orientation-dependent elastic terms; other
  models may add plastic storage. Corrosion needs environmental/transport
  conditions; fracture needs toughness and an appropriate irreversible law.
- Grain-boundary mobility and observation timescale must be plausible for the
  material/temperature. Ordinary macroscopic loading does not automatically
  produce visibly migrating grains. Fixed-grain crystal plasticity is another
  problem and is not the proposed grain-growth model.
- Compare two zoom regions of a loaded plate: representative interior cell
  returns average stress; a pit/crack submodel resolves a particular dangerous
  location. These are common uses, not exclusive physics assignments.
- RVE homogenization learns effective material response from macrostrain
  history and internal state; local submodels inherit surrounding boundary
  fields. One-way submodeling without feedback is not full FE2. Either method
  can use grain evolution or corrosion/fracture where physically justified.
- Distributed microdamage can be homogenized; a dominant macroscopic crack
  challenges conventional local scale separation. A surface corrosion patch
  has environmental boundaries unlike an all-periodic bulk cell. A small
  cell does not become statistically representative merely by naming it RVE.
- Existing corrosion/fracture experience is sufficient for a boundary/history
  conditioned fine-scale operator. Grain migration is an alternative physical
  choice, not a prerequisite for a valid multiscale-oriented PINO.

## Primary sources checked

- MOOSE elastic grain-growth model and mechanics/orientation coupling:
  https://mooseframework.inl.gov/modules/phase_field/Elastic_Driving_Force_Grain_Growth.html
- MOOSE multiphase grain model:
  https://mooseframework.inl.gov/modules/phase_field/Grain_Growth_Model.html
- MOOSE temperature-dependent grain-boundary mobility:
  https://mooseframework.inl.gov/source/materials/GBEvolution.html
- Cui, Ma, Martinez-Paneda (2021), dissolution-driven SCC, coupled phase,
  displacement and concentration with mechanics-sensitive kinetics and films:
  https://arxiv.org/abs/2011.12068
  DOI:10.1016/j.jmps.2020.104254
- Bharali, Larsson, Janicke (2021), computational homogenisation of phase-field
  fracture, including fine-scale-only and macro/fine phase alternatives:
  https://research.chalmers.se/publication/522540/file/522540_Fulltext.pdf
  DOI:10.1016/j.euromechsol.2021.104247

## Completion verification

No code, default, numerical data or Git history change. Documentation-only
checks: inspect changes, verify dashboard/task paths and git diff whitespace.
