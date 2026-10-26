# Nonlocal advantages assessment — completed 2026-10-26

## Question and conclusion

User asks whether nonlocal has advantages beyond the measured update speed.
Yes: lower spatial regularity requirements for bounded integrable kernels,
explicit reciprocal pair flux/positive quadratic form, architectural separation
of field representation from spatial differentiation, and finite-range physical
modeling. Bounded high-frequency amplification is a derived property and a
plausible optimization benefit, not evidence of better PINN conditioning/L2.
Current smooth FiLM corrosion/melting/electropolish have no completed matched
local-reference convergence study proving superior nonlocal accuracy/stability.
No new training, production/config/reference change or Git operation.

## Primary references read

1. Du & Yang2016, *Asymptotically Compatible Fourier Spectral Approximations
   of Nonlocal Allen–Cahn Equations*, SIAM J Numer Anal54,1899–1919,
   DOI10.1137/15M1039857.
   https://www.columbia.edu/~qd2125/Res/dy16sinum.pdf
   Periodic nonlocal/local limits and nonsmooth steady states for relevant
   integrable kernels. Its asymptotic compatibility/O(delta^2) theorems concern
   its stated Fourier scheme/assumptions, not this finite graph automatically.
2. Pang et al2020, *nPINNs: nonlocal Physics-Informed Neural Networks for a
   parametrized nonlocal universal Laplacian operator. Algorithms and Applications*.
   https://arxiv.org/abs/2004.04276
   Nonlocal parameter inference/local limits demonstrate modeling flexibility,
   but operator mimicking warns that nonlocal parameters may not be identifiable.
   Its singular/fractional operator family is not the current bounded cubic kernel.
3. Haghighat et al2020/2021, *A nonlocal physics-informed deep learning framework
   using the peridynamic differential operator*.
   https://arxiv.org/abs/2006.00446 ; DOI10.1016/j.cma.2021.114012.
   Authors report sharp-gradient elastoplastic forward/inverse improvements.
   Their PDDO/network construction differs from our scalar finite-horizon graph;
   this motivates experiments and does not prove transfer to corrosion.

Initial Du et al2012 overview PDF and a DG PDF host timed out; no uninspected
full-text claim relies on them. Du/Yang2016 full text and the two primary arXiv
abstracts above were accessible. No secondary source used as technical authority.

## Checked actual graph and derivations

Source: src/nonlocal_pinn/operators/__init__.py cubic kernel; current matched
reference /root/autodl-tmp/data/corrosion2d/results_matched_graph_20261024.npz.
[Check source](nonlocal-advantages-check.py), [raw output](nonlocal-advantages-check.json).
CPU SciPy/NumPy only; no GPU work.5151 nodes,60296 directed entries (zero-valued
horizon-end weights included). All weights nonnegative and exactly reciprocal.
Exact-edge/reference D=diag(sum_j w_ij)-W annihilates constants exactly here.

For nonnegative reciprocal weights and equal nodal volumes:

$$
u^T D u=rac12sum_{ij}w_{ij}(u_i-u_j)^2ge0,qquad {f1}^T D u=0.
$$

The random-vector energy identity has relative discrepancy0, internal-flux
relative cancellation error2.53e-18. These support dissipative linear diffusion
and pairwise internal exchange. They do not assert full nonlinear PINN energy
stability or closed-system concentration conservation. Current corrosion has
Dirichlet reservoirs, soft PDE/data losses and a stored FP32 diagonal in AC;
its max row-sum defect4.8828125e-4 remains. Stable edge CH has the direct
constant-annihilating form. These existing differences were not silently fixed.
Conservative local FD/FV/FEM/weak PINNs can also preserve analogous structures;
this is an advantage relative to an unstructured pointwise strong-AD residual,
not an exclusive property of every nonlocal method versus every local method.

For fixed finite-radius nonnegative integrable kernels and bounded error e:

$$
|D_delta e(x)|le2|e|_inftyint J_delta(x,y),dy.
$$

Graph analogue: max|De| <=2 max_i(sum_j w_ij) max|e|. Manufactured bounded
perturbation has max|De|54307.13 versus bound64057.49. A discontinuous step
also has a finite graph action. Contrast a continuum local perturbation
 e=epsilon*sin(kx), whose Laplacian magnitude grows epsilon*k^2. This suggests
less unbounded high-frequency amplification at fixed delta, not automatic
smoothing/conditioning/success. The bound grows as delta shrinks; nonlinear
reaction stiffness remains. High-frequency errors can also be under-penalized
or aliased. Local mesh discretizations likewise have finite matrix norms, so
finite graph dimension alone is not proof of unique benefit.

## Applicability and tradeoffs

- Regularity: bounded integral/edge differences do not require pointwise second
  spatial derivatives. Nonsmooth representations and sharp/discontinuous states
  can be accommodated where strong local AD is unsuitable. Current gelu/FiLM
  diffuse-interface examples do not fully exploit this flexibility. This does
  not prove equal physical models for local discontinuous solutions.
- Representation: a graph action needs field values, so spatial outputs could
  come from grid neural operators or other representations without a smooth
  coordinate-to-field map. Local FD/FV/spectral differentiation could also be
  used on grid outputs; the comparison is specifically with local spatial AD.
- Modeling: delta encodes finite interaction range if supported by physics/data.
  It is not free accuracy regularization. If the actual target is a local PDE,
  finite-delta model bias, quadrature and boundary treatment must be measured,
  with a validated local limit before claiming improved total accuracy.
- Learning: better late L2, transport gradients, robustness, and time-to-target
  error remain hypotheses. Same-graph reference error alone measures agreement
  with that graph, not superiority to a local continuum solution. Current Dq
  errors already warn against equating small field L2 with transport fidelity.

Most useful proposed test: matched physical references/boundaries and decreasing
interface width, same network/data, independent seeds, plot wall time to a common
phi/c/q/Dq error target and failure rates. This is proposed follow-up only;
no new convergence study or local model was constructed in this assessment.

## Status and verification

Complete, no blocker or remaining requested work. Independent current-graph
checks pass; primary-source scope and limits recorded. All three archived artifacts pass source AST,
strict JSON and relative-link checks; git diff --check passes. Production tests unnecessary
for this read-only assessment; existing artifacts and Git history unchanged.
