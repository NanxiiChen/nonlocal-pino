# ATG boundary loading and morphology

Date: 2026-11-05

## Goal and acceptance

Explain whether varying the proposed outer-solid displacement input changes
only groove depth or also interface morphology. Separate scalar loading from
spatial loading functions, early linear behavior from nonlinear competition,
and documented uniform-load theory from expectations for an untested finite
domain. No implementation or simulation requested.

## Context and sources

Previous task: [ATG sketch](atg-interface-concept-sketch.md).
The proposed model is a solid/melt interface, quasistatic linear elasticity,
fixed reservoir temperature/pressure and interface-controlled kinetics. Inputs
would be smooth displacement functions on fixed outer solid boundaries, with
a common temporal ramp/hold and the same initial phase field.

Kassner et al., author full text:
https://arxiv.org/html/cond-mat/0005476, equations 1--7 and groove-competition
discussion. Reopened during this task. A separate agent audits the interpretation.

## Status

Complete. Uniform-load, single-mode linear evolution can chiefly change depth
and rate. With multiple initial wavelengths, load amplitude also changes growth
selection because the elastic and capillary terms have different wave-number
dependence; it is not generally just a time rescaling. The fastest-growing
linear mode is not a prediction of the final groove count.

Spatially varying imposed displacements can change interfacial stress patterns
and bias groove positions, symmetry and competition. This is a physical
expectation for the proposed finite-domain case, not an existing numerical
result. Elastic transmission can attenuate fine spatial boundary variations,
particularly far from the loaded boundary. Do not equate a local displacement
peak to a groove directly above it, or promise branching/distinct late states.

For any future feasibility study, recommend one fixed weakly multimode initial
interface and a few low-frequency load profiles, with common physical times
and separate checks of groove depth, spacing and relative growth. This is a
recommendation, not authorization to run or a selected implementation.

Validation: primary-source equations and assumptions checked, independent
agent audit agrees; documentation-only change, diff/link checks before finish.

## Blocker and next action

No blocker for explanation. Numerical differences cannot be claimed until
boundary placement and consistent reference solves are chosen and validated.
Await user's next design choice. No solver, training or Git history operation
is authorized or needed.
