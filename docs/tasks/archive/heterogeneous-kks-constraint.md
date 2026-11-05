# KKS constraint in the heterogeneous corrosion PINO

Date: 2026-11-06. Status: complete clarification.

## Goal

Explain whether the earlier corrosion kks_constraint is retained in the current
heterogeneous_corrosion2d pilot. Distinguish KKS thermodynamics/algebraic output
reconstruction from optional concentration bounds, extra losses and hard IC.
No implementation or experiments requested.

## Context

Prior [particle discussion](inert-particle-corrosion-geometry.md) is a
proposed extension; the existing tested pilot varies positive kinetics only.
## Verified implementation

`examples/heterogeneous_corrosion2d/network.py:19` implements
phi=(1+tanh(z))/2, cL=cLe*s, c=cL+(cSe-cLe)*h(phi), with s unrestricted.
This is the same output coordinate choice as electropolish PINO physical_fields.
For the declared equal-curvature quadratic KKS free energies, reconstructing
cS=cL+(cSe-cLe) satisfies the chemical-equilibrium and mixture relationships
algebraically. No separate KKS loss is present. Stable q=cLe*(s-1) is evaluated
directly to avoid subtracting nearly equal mixture terms.

The shared legacy `src/nonlocal_pinn/nn/constraints.py:11` additionally tanh-bounds
cL to [0,1-cSe+cLe]. Under the current cSe=1 this is [0,cLe], forcing q<=0
and total c within [0,1]. The new pilot does not call this old function and
does not impose those concentration bounds; phase remains bounded. As recorded
in electropolish README's output-coordinates section, that operator relaxed
the bound because its reference can have positive q. The material pilot uses
the same unrestricted map; do not claim a newly measured advantage over bounded
concentration in this material pilot, since no such ablation was performed.

Nonlocal losses are AC, CH, mouth phase, normalized mouth concentration and
initial fields. Local AD additionally has the two wall normal losses. IC and
mouth conditions remain soft losses; hard algebraic KKS reconstruction is not
hard IC. The particle extension has not yet been implemented.

## Completion

Only task memory/decision/progress documents updated. Read-only code inspection
and documentation diff/link checks; no model changes, tests, training, solver
runs or Git operations. No blocker or outstanding work.
