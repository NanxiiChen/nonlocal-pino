# Loaded polycrystal RVE concept sketch

## Goal
Respond to the user's request for a schematic of a fine-scale parametric
phase-field/mechanics operator appropriate to future multiscale homogenization.

## Acceptance criteria
- Show macrostrain-history input and averaged stress output.
- One common initial polycrystal cell; only loading varies across samples.
- Distinguish grain-boundary migration from crystalline variant switching.
- Label all evolved shapes as hypothetical and any macro component as context.
- Include a qualitative periodic-cell interpretation without fake solved fields.

## Context
The RVE candidate comparison is complete. The leading new candidate is elastic
loading-driven grain-boundary migration/texture evolution. User asks to visualize
it before choosing a numerical implementation.

## Status
Complete (2026-11-05). Built-in imagegen preview generated, inspected and
label-edited once to clarify reference configuration and future periodic setup.

## Blocker
None.

## Next action
None required for this schematic. Numerical candidate/model selection remains
open; future response differences must be established by reference solves.


## Delivered image and provenance

Final preview:
`/root/.codex/generated_images/01a16d71-2d47-7201-9441-ab74e0523152/exec-5a9f4053-30d8-4720-8f6a-33a56932ec77.png`.
Initial image, retained:
`/root/.codex/generated_images/01a16d71-2d47-7201-9441-ab74e0523152/exec-19488894-deb0-4a01-9f5d-80957c960aad.png`.
Opaque PNG1536x1024, built-in imagegen generation followed by one targeted label
edit using the initial image as the reference. Preview only, not copied into
tracked README assets.

Prompt brief: Chinese graphical abstract for strain-history-conditioned
polycrystal cell evolution. Macro material point is application context; one
common initial colored grain field feeds tension, shear and mixed-loading
scenarios; hypothetical evolved grain shapes and average-stress output. Fixed
colors denote fixed orientations. No numerical curves, times, accuracy or
stress colorbar. Subsequent edit explicitly labels reference configuration,
implementation-time periodic microstructure construction and future macro link.

## Visual review

One shared initial cell is present, all three loading families are labeled,
and the microstructure colors are described as orientations with changing
occupied areas. The hypothetical fields are conceptual illustrations, not
actual grain histories or a calibrated texture prediction. The drawn boundary
colors do not satisfy literal periodic tessellation matching; final text
explicitly states that a matching periodic microstructure is to be constructed
for implementation. Square outlines show the reference configuration and omit
elastic shape deformation. Feedback arrow is dashed and marked future macro
coupling; this task only studies the cell proposal. A small illustrated cell
is not evidence of statistical representativity.

Physics source for the candidate (not for drawn shapes): Tonks/Millett2011,
https://www.sciencedirect.com/science/article/pii/S0921509311001584 .
The broader candidate/boundary/history comparison is archived at
`phasefield-rve-operator-candidates.md`.

## Verification and changes

Generated preview visually inspected; PNG decoding verified. Independent
conceptual review identified the periodic-image and reference-coordinate notes,
which were incorporated. Updated PROGRESS, DECISIONS and this archived task;
local memory links and git diff whitespace checked. No numerical source,
training, reference solve, default or Git history operation.
