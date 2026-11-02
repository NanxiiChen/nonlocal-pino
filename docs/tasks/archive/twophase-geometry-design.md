# Parameterized two-phase geometry recommendation

## Goal and acceptance

Recommend a physically meaningful and visually clear family of varying outer
domains for the next phase-field operator case, without interior holes. Only
geometry is random; initial phase, material parameters and driving data stay fixed.
Explain what geometry changes physically and how condition tokens would encode it.
Keep the distinction between current prescribed-flow CH and fully coupled flow.

## Relevant context

Current `twophase2d` solves periodic prescribed incompressible advection plus CH;
it does not solve Navier--Stokes or pressure. Fourier flux differentiation and the
optional nonlocal gradient both assume periodic neighborhoods. General wall BC
work was explicitly deferred (`docs/issues/twophase-fft-periodic-boundary.md`).
This request authorizes conceptual case selection, not an unannounced PDE/BC
replacement or another full training experiment.

User prefers physical geometry parameters and a common point-token architecture,
not geometry-specific distance-attention modifications. Current irregular-domain
condition streams are boundary coordinates/normals/type and domain coordinates/IC,
with quadrature measures. Actual query points may differ between geometries.

## Status and recommendation

Complete as a conceptual recommendation. Prefer a smooth contraction--expansion
microchannel with fixed common inlet/outlet sections. Randomize throat minimum
width, contraction/taper length and optionally upper/lower wall asymmetry. Use
smooth bounded wall profiles, fixed topology, and reject self-intersection or
unresolved gaps. Keep a fixed initial droplet in the common upstream region,
fixed physical initial radius/interface width/material parameters/contact law,
and one fixed inlet-flow protocol. Geometry-dependent flow responses are not
independently parameterized velocities. Show transient deformation and recovery;
do not assume or tune for breakup before a conventional reference pilot proves it.

Other options: variable smooth closed confinement has less inlet/outlet work but
may give similar relaxed droplet shapes; Y/bifurcating channels give a more
challenging possible route/splitting response. No new family has been simulated.

For physical hydrodynamics, consider low-Re coupled Stokes--CH with capillary
feedback, or full NS--CH if inertia is part of the chosen regime. A precomputed
single-phase Stokes flow + CH would remain a one-way transport pilot and cannot
be presented as the same two-way model. No PDE or constitutive coefficients are
selected in this discussion.

Architecture can reuse generic boundary/domain point streams. Boundary type
labels would identify inlet, outlet and wall; domain samples evaluate the same
known initial function. Coordinates/measure/normal data describe geometry, with
no geometry-specific attention-distance bias.

## Numerical implications for a future implementation

- Nonperiodic walls invalidate the current full-grid Fourier flux derivative;
  the experimental nonlocal first derivative is also periodic. Resolve this
  declared numerical work before claiming an irregular-wall implementation.
- No-slip velocity and phase wetting/chemical flux conditions need explicit,
  separately validated treatment; a regional scalar Laplacian does not impose
  all flow/interface boundary conditions automatically.
- Open-domain phase mass obeys boundary flux balance. The present closed-grid
  fixed-mass projection cannot be copied without checking that balance.
- Fix upstream/downstream sections and keep the whole diffuse initial interface
  safely inside their common region. Comparing identical physical observation
  times remains useful; geometry affects passage time as well as shape. Interface
  deformation, trajectory and pressure/flow diagnostics supplement field L2.

## Primary evidence checked

- Trégouët et al., Phys. Rev. Fluids 3, 053603 (2018),
  https://journals.aps.org/prfluids/abstract/10.1103/PhysRevFluids.3.053603 .
  Opened primary abstract: constriction/flow-focusing deformation experiments
  and the need to resolve the flow stresses along the droplet trajectory.
- De Menech, phase-field T-junction droplet breakup (2005 preprint; related
  Phys. Rev. E 73, 031505), https://arxiv.org/abs/cond-mat/0511371 .
  Opened primary abstract: 3D phase-field microfluidic capillary/viscous-dominated
  breakup. Supports feasibility, not quantitative validation of this proposed2D
  constriction or a promised breakup regime.
- Bai et al. (2017), DOI10.1016/j.ijmultiphaseflow.2017.04.008, primary publisher
  search result describes 3D CHNS/FEM flow focusing with wall wetting/GNBC. Direct
  full-page fetch returned403; no unseen full-text formula is used.
- Current README, problem/derivative source and the deferred boundary issue
  confirm prescribed-flow periodic scope. No reinterpretation as an existing NS
  solver. An unrelated glob/path lookup found no `twophase2d/physics.py`; actual
  scope is documented in `problem.py`, `advection.py` and `operators.py`.

## Verification and changes

Discussion/source/document checks only. No numerical tests needed because no
runtime/config/model changes. `git diff --check` passes. Add this archived task,
update `docs/PROGRESS.md` and append the decision in `docs/DECISIONS.md`. Preserve
existing uncommitted diagram/HTML documentation. No training, reference solve,
media generation, dependency or Git-history operation.

## Blocker and next action

None. Recommendation delivered; no queued implementation or training. A future
implementation request must first declare the coupled/one-way scope, complete
boundary conditions, scales, operator consistency gates and reference route.
