# Visually distinctive phase-field/mechanics candidates

## Goal
Recommend classic coupled phase-field/displacement cases with stronger visual
morphology than the deferred coherent-precipitate proposal, ideally conditioned
on boundary loading while keeping geometry and initial state fixed.

## Acceptance criteria
- Ground candidate mechanisms and representative morphologies in primary sources.
- Distinguish solid phase transformation, chemoelastic transport and damage.
- Compare boundary-function sensitivity, reference construction and PINO risks.
- Recommend a feasible first pilot without inventing material parameters or
  promising unverified patterns.

## Context
Four existing PINOs are complete. The user reopened the deferred mechanics
candidate discussion on 2026-11-05, requesting alternatives rather than execution.
The repository is independent of the separate fracture project; no cross-project
implementation is authorized.

## Status
Complete (2026-11-05): primary-source comparison completed. Recommended the
mechanically driven variant/twin family, distinguishing visually richer finite-
strain martensitic transformation from a simpler small-strain ferroelastic pilot.

## Blocker
None for recommendation. A particular model and loading family have not been selected.

## Next action
Await the user's candidate selection. Before implementation, select a traceable
constitutive model and test a few fixed-IC boundary-load cases with an independent
reference. No simulation, source/config changes, or Git operation was performed.


## Source-grounded comparison

1. **Localized displacement-driven martensitic transformation and twinning.**
   Levitas, Roy and Preston (2013), Physical Review B88,054113, develops a
   finite-strain, lattice-rotation-aware phase-field/mechanics model. Figure6
   specifically prescribes increasing vertical displacement on a small boundary
   segment and exhibits two variants initially growing downward, followed by
   one turning toward a corner with a curved variant interface. Figure8 uses
   two PRESSURE-loaded indenters, not two displacement controls, and demonstrates
   interacting structures and a change of dominant variant on unloading one side.
   Figures2--4 also show tip splitting, bending and crossing under their own
   stated conditions. These are direct published morphology evidence, not
   predictions for a new smooth-boundary PINO setup.
   https://harvest.aps.org/v2/journals/articles/10.1103/PhysRevB.88.054113/fulltext

2. **Pure ferroelastic domain switching.** Bhattacharya and Zaeem (2024),
   Acta Materialia276,120039, and Scripta Materialia252,116273, treat variant
   evolution coupled to small-strain mechanical equilibrium. The latter uses
   three tetragonal variants in yttria-stabilized zirconia, with domain-wall
   motion and nucleation under a compression/tension path. This is a physically
   distinct, potentially more manageable first pilot than reproducing all
   finite rotations in the PRB model. Pure ferroelasticity need not introduce
   electric potential; ferroelectric polarization models generally do.
   https://doi.org/10.1016/j.actamat.2024.120039
   https://www.imechanica.org/sites/default/files/Ferroelasticity_Nucleation_Scripta%20Materialia_2024.pdf
   The large Acta PDF exceeded the web fetch limit; its indexed text and the
   companion Scripta full text were available. Do not claim a complete Acta PDF
   visual inspection.

3. **Chemoelastic phase separation.** Stanton and Bazant (2012) analyze CH-type
   phase separation with anisotropic coherency strain; Cogswell and Bazant
   (2012) demonstrate tilted stripes and surface elastic relaxation in LiFePO4.
   Visually appealing and related to existing spinodal/precipitate components,
   but boundary chemical potential/flux is a more directly motivated variable
   than assuming arbitrary mechanical loads produce distinct stripe patterns.
   The LFP paper's reaction-limited electrochemical model is not identical to
   a simplified closed-system CH model.
   https://arxiv.org/abs/1202.1626
   https://web.mit.edu/bazant/www/papers/pdf/Cogswell_2012_ACS_Nano_ASAP.pdf

4. **Phase-field fracture.** A classic displacement/damage coupling with
   visible crack turning and, in appropriate regimes, branching. The official
   PhaseFieldX single-edge-notched shear example documents the Miehe2010
   benchmark and implementation. Irreversibility and localized stiffness loss
   make it a different research track; the existing project split means this
   recommendation does not authorize adding fracture implementation here.
   https://phasefieldx.readthedocs.io/en/stable/auto_examples/PhaseFieldFracture/plot_1712.html

## Proposed operator and limits (engineering judgment)

Keep a fixed rectangle, material, initial mixed-variant/seed field and smooth
loading ramp. Sample a smooth prescribed displacement profile along a loaded
edge: first vary its amplitude and width, then location or multiple lobes once
response is verified. A local imposed displacement is an idealized actuator /
indentation boundary, not a solved unilateral contact problem. Do not move
seeds with the load and call the initial condition fixed.

The operator maps the complete prescribed spatial boundary profile (with fixed
ramp) to evolving variant fields and displacement. Boundary point tokens and
shared space-time queries fit the existing architecture; fixed geometry permits
operator precomputation. Coupled elastic energy produces stress-dependent
variant driving forces, rather than only warping a displayed mesh.

Important limitations: multivariant constitutive consistency; narrow twins;
metastability and path dependence; initialization and loading history selecting
solution branches; finite rotations if claiming the full PRB behavior. A
current-load-only input is insufficient for arbitrary hysteretic histories.
For an independent-order-parameter small-strain model, familiar spatial
first/second derivatives remain usable, but nonlocal support must resolve the
walls and mechanical boundary/rigid-motion/affine consistency must be checked.
No claim that scalar no-flux closure automatically supplies traction BCs.

Recommend the twin/variant family for boundary-conditioned visual diversity,
with a simpler pure ferroelastic pilot if feasibility is prioritized. Validate
3--5 actual load responses before committing to a PINO training campaign;
no pattern, training accuracy or speedup is yet established.

## Completion verification
Only repository memory changed: PROGRESS, DECISIONS and this archived task.
Documentation paths and diff whitespace checked; no code tests warranted.
