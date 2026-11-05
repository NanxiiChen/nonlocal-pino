# Nonlocal internal-wall advantage in corrosion

Date: 2026-11-06. Status: complete conceptual clarification.

## Goal

Answer whether excluded inert particles demonstrate an advantage over the
current strong-form local-AD PINO by avoiding additional Neumann boundary loss.
Separate discrete flux closure, local-limit consistency and trained error.
No implementation or experiment requested.

## Context

Preceding [geometry assumptions](inert-particle-corrosion-geometry.md):
fixed impermeable particles, neutral phase boundary, zero normal q transport.
Existing stronger-material experiments remain preserved.

## Verified existing implementation and conclusion

User explicitly identifies `examples/confined_ch2d` as the intended analogy.
Read README lines13--29,130--196,309--315, `geometry.py:visible_pairs` and
`operators.py:RegionalLaplacian`. The existing regional scheme excludes hole
interiors, rejects segment-crossing interactions symmetrically, and uses actual
cell areas with a difference-form operator. No separate normal-derivative wall
loss is used by its nonlocal route. Its local-AD route uses both Hessians and
two explicit Neumann losses. Boundary points may still be used as encoder
tokens; absence of boundary loss does not mean no geometry/boundary inputs.

With positive volumes V_i and symmetric a_ij, (Lu)_i=V_i^-1 sum_j a_ij(u_i-u_j)
has sum_i V_i(Lu)_i=0. This is a discrete flux cancellation property, not a
proof that a trained continuous network obeys pointwise local normal derivative
zero. Residual training alone need not exactly conserve mass; corrosion also
has an exterior reservoir opening. The closed CH pilot's hard mass projection
must not be copied as constant-total-mass physics into open corrosion.

Thus the same structural approach can be reused for fixed inert particles,
replacing the CH phase/chemical-potential law with corrosion phase/q dynamics.
It avoids additional wall Neumann soft losses and their gradient balancing
relative to current strong-form AD. Geometry-aware precomputation is still
required. Local FEM/FV and weak-form or hard-constrained PINNs can also impose
zero-flux boundaries without these penalties; the advantage is not exclusive
to nonlocal methods. Speed, accuracy and finite-horizon wall consistency remain
experimental questions. Existing CH README reports nonzero AD normals for the
nonlocal network, so do not claim exact classical Neumann satisfaction.

Independent agent audit agrees. Primary context:
- Cortazar et al., regional nonlocal diffusion and local Neumann limit:
  https://mate.dm.uba.ar/~jrossi/cerwEP-20-04-06.pdf
- You et al., asymptotically compatible Neumann treatment:
  https://arxiv.org/abs/1908.03853

## Completion and resume point

Only memory documents updated, with diff/link checks. No code, training,
artifact cleanup or Git history action. No outstanding action; conceptual
reuse is supported, implementation requires a subsequent user request.
