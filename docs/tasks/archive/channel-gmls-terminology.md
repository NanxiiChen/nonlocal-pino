# Clarify GMLS versus nonlocal differentiation terminology

## Goal and acceptance

Answer whether the current channel GMLS differentiation may be described as
nonlocal consistently with the other examples. Inspect the implemented target
functionals and primary literature; distinguish numerical neighborhood support
from a nonlocal continuum operator and record exact derivative exceptions.
No formulation, configuration, training, reference or Git changes requested.

## Context

Previous default-selection task is complete. Current code uses 36-neighbor,
degree3 QR-GMLS differential functionals, explicit boundary losses, time JVP
and AD streamfunction-to-velocity reconstruction. Sources under review:
Compadre's mathematical overview and primary GMLS/local/nonlocal papers.

## Findings and wording

- operators.py targets physical first derivatives and the Laplacian applied to
  polynomial basis functions. QR weighted reconstruction yields a finite sum
  of neighboring field values; no regional pair-integral target is selected.
  Neighborhood is the 36 nearest nodes, radius 1.25 times the farthest selected
  distance, degree3 and weight exponent16 in the inherited default.
- Compadre explicitly describes generating generalized finite differences on
  unstructured point clouds. The Sandia 2023 comparison calls this use of GMLS
  localized meshfree differentiation. GMLS is a general functional-reconstruction
  technique, so the name alone does not determine local/nonlocal physics.
- A discrete finite-neighborhood/nonlocal numerical-differentiation description
  is possible only with that restricted definition stated. Do not equate this
  implementation with the other examples' integral nonlocal operator, infer a
  nonlocal continuum PDE, or claim that all examples validate the same kernel.
- Preferred common wording: neighborhood-based meshfree differentiation, with
  integral nonlocal and GMLS implementations named separately.
- Retain actual exceptions: time JVP, streamfunction curl spatial AD, parameter
  gradients reverse AD; Neumann/outlet BCs are explicit, not supplied by GMLS.

## Primary sources

- [Compadre mathematical overview](https://github.com/sandialabs/compadre#generalized-moving-least-squares-gmls),
  checked weighted reconstruction, target functionals and stencil interpretation.
- [Jones et al. 2023, GMLS vs RBF-FD](https://www.sandia.gov/research/publications/details/generalized-moving-least-squares-vs-radial-basis-function-finite-difference-2023-10-01/),
  checked author-laboratory publication abstract and localized classification.
- [Trask et al. 2018, nonlocal quadrature](https://arxiv.org/abs/1801.04488),
  checked explicit compactly supported integral-equation targets, distinguished
  from the currently implemented differential targets.

## Verification and completion

Code/default/README inspection and primary-source verification completed.
Documentation-only task; no code tests, training, FEM, changes to numerical
choices or Git operations. git diff --check passes. Complete; no blocker,
remaining action or queued experiment.
