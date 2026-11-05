# ATG solid/melt interface concept sketch

## Goal
Illustrate the proposed ATG two-phase moving-interface example with simple
linear elasticity, showing problem setup and possible boundary conditioning.

## Acceptance criteria
- Show loaded solid below a melt reservoir and a shared initially wavy interface.
- Distinguish phase migration from bulk elastic stretching or buckling.
- Mark fixed temperature/pressure assumptions and variable mechanical boundary
  input without applying solid shear traction to the melt.
- Label illustrated groove evolution as conceptual, with no fake time/error
  values or claim of computed morphology.

## Context
User says the newly proposed ATG direction sounds good and requests a schematic.
This authorizes illustration only, not a numerical implementation. Previous
candidate research is archived under twophase-linear-elastic-candidates.md.

## Status
Complete (2026-11-05). Built-in imagegen preview generated and visually inspected;
the diagram respects the independent physical-layout audit.

## Blocker
None.

## Next action
Await user feedback/model selection; no numerical implementation requested.

## Artifact

Preview image, retained at the built-in tool's direct output path:

`/root/.codex/generated_images/01a16d71-2d47-7201-9441-ab74e0523152/exec-ca22062e-9f6c-4ee0-98ba-9f05acad85cf.png`

The image was displayed inline by the generation tool. No repository asset or
README reference added. Built-in generation, no CLI/API-key fallback.

## Content and review

Chinese ivory/navy/cyan scientific infographic: upper fixed-temperature/pressure
melt over a lower linearly elastic solid; solid-only side displacement arrows;
surface-energy smoothing versus elastic-energy-release destabilization.
Three qualitative time panels depict initial small waves, groove formation and
later unequal rounded grooves. The figure explicitly calls these one initial
interface/loading condition evolving in time, not three input samples.

Footer identifies boundary displacement amplitude/smooth profile as the possible
operator input, with interface, displacement and stress outputs. The physical
setup is only schematic; complete boundary conditions and material/kinetic
parameters remain unselected. No exaggerated elastic deformation, clamps,
crystal grains, fluid velocity, quantitative stress/temperature/time, stable
terminal equilibrium, mass-conserving phase or fracture claim.

Exact title: “应力驱动的固体–熔体界面演化”.
Footer: “概念示意，非数值解；具体形态需计算验证。弹性变形未放大展示。”

## Completion verification

Visual inspection and agent review of physics constraints; PNG decoding and
dimensions checked, dashboard links checked and git diff whitespace checked.
Only PROGRESS, DECISIONS and this archived task changed in the illustration
turn. No code tests, simulations, training, cleanup or Git history operations.
