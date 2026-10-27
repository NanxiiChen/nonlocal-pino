# PINN residual-preconditioning prior-art search

Date: 2026-10-27. Status: complete. Blocker: none.

## Goal and acceptance

Explicit user request: search whether applying a preconditioner to the PDE
residual, then constructing a PINN loss, has already been done. Verify direct
formulas against original papers; distinguish adjacent methods, and assess what
remains of the proposed research direction. Acceptance met by the primary-source
comparison below. No implementation, training, reference, defaults, external
artifact-path or Git-history operation was performed.

## Context and conclusion

Previous discussion writes S=W R(G) and loss ||S||², separating field mapping,
left residual weighting and parameter optimization. Existing vortex controls
separate map-only interface gains from inverse-map residual scaling tradeoffs.
The broad residual-preconditioned PINN loss is already established prior art,
including a paper with the exact P^-1(Au-b) squared-norm objective. Neither
nonlinear/time-dependent extension nor phase-field application alone establishes
novelty. Search results do not prove priority or exhaustively exclude narrower
state-map/nonlocal/conserved phase-field combinations.

## Primary-source comparison

### Liu et al.: Preconditioning for Physics-Informed Neural Networks

- First arXiv submission: 2024-02-01. Do not infer an accepted conference venue
  from an under-review OpenReview manuscript.
- Metadata: https://arxiv.org/abs/2402.00531
- Original text: https://arxiv.org/html/2402.00531v1
- Section 4, Eq. (20):

$$
\mathcal L^\dagger(\theta)=\|P^{-1}(Au_\theta-b)\|^2
=\|\widehat U^{-1}\widehat L^{-1}(Au_\theta-b)\|^2.
$$

Sparse PDE discretization and ILU factorization define the preconditioner;
network outputs at mesh points enter the residual, which is preconditioned
before the loss. This is direct prior art, not just a parameter-gradient
optimizer. Boundary conditions enter the assembled discrete system. Appendix
B.3 discusses all-at-once/time-stepping treatments and nonlinear problems,
including preconditioning a linear part with nonlinear residual terms and
Newton linearizations. Do not characterize this work as exclusively linear.

### Song, Cao and Zhang: A matrix preconditioning framework for physics-informed neural networks based on adjoint method

- First arXiv submission: 2025-08-05; treated as a preprint here.
- Metadata: https://arxiv.org/abs/2508.03421
- Original PDF: https://arxiv.org/pdf/2508.03421
- Section 2.3, Eqs. (6), (8), (9): PDE-state Jacobian J=df/du, ILU-derived
  matrix M, transformed residual p=M^-1 f and loss p^T p/N_r.
- AD plus matrix coloring constructs the sparse PDE Jacobian; an adjoint method
  supplies training gradients involving triangular solve actions. Do not claim
  this necessarily differentiates through the ILU factorization/state-dependent
  preconditioner itself: that derivative detail was not established here.
- CNN-based PINNs, finite-difference operators and hard BC treatment; Poisson
  and steady incompressible Navier–Stokes experiments. Direct residual-loss
  preconditioning with a PDE Jacobian, distinct from inverse scalar map g'.

### Wen, Mishra and Zeinhofer: Preconditioned Physics-Informed Neural Operator Training

- First arXiv submission: 2026-09-28; recent preprint as of 2026-10-27.
- Metadata: https://arxiv.org/abs/2609.36216
- Original text: https://arxiv.org/html/2609.36216v1
- Section 2.1, Eq. (6): expected squared norm of P R_h(u(theta,rho);rho)
  in an explicit B inner product. Their P denotes the inverse action.
- Geometric/algebraic multigrid, approximate inverse Jacobians, and block
  residual norms. Experiments include Poisson, Allen–Cahn and stationary
  Stokes. Allen–Cahn uses a frozen pure-phase Jacobian approximation in an
  autoregressive loss (Appendix D); it is not our scalar inverse-map scaling.
- Neural operator/FNO training across PDE inputs, not an ordinary single-PDE
  PINN. Nevertheless, it directly overlaps residual-loss preconditioning,
  phase-field application and generic spatial/block preconditioning claims.
- Output-space conditioning statements do not remove neural parameter
  Jacobian effects or guarantee lower field relative L2 for our cases.

### Badia, Li and Martin: Finite element interpolated neural networks for solving forward and inverse problems

- First arXiv submission: 2023-06-09; v4 dated 2023-10-10.
- Metadata: https://arxiv.org/abs/2306.06304
- Original text: https://arxiv.org/html/2306.06304v4
- Section 2.5, Eq. (7): inverse Riesz/discrete operator applied to a weak
  residual before the norm. Section 6.1.3, Eq. (11):

$$
\mathscr L(\theta)=\|B^{-1}Au_h(\theta)-B^{-1}f\|_{\ell^2}.
$$

- FE interpolation of neural fields, with exact/operator and geometric
  multigrid preconditioning comparisons. Related neural PDE methodology
  predating the 2024 PINN paper, not conventional pointwise AD collocation.
  Its displayed norm need not be squared; distinguish dual/Riesz norms from
  arbitrarily replacing all inner products with Euclidean norms.

### De Ryck et al.: An operator preconditioning perspective on training in physics-informed machine learning

- First arXiv submission: 2023-10-09; v2 dated 2024-05-03.
- Metadata: https://arxiv.org/abs/2310.05801
- Original text: https://arxiv.org/html/2310.05801v2
- Operator/tangent-kernel conditioning theory, parameter/feature transforms
  and interpretations of established training choices. Useful theoretical
  context, but distinguish its explicit parameter preconditioning from the
  more direct residual-preconditioned objectives above. Main linearized/
  linear scope does not establish universal nonlinear conditioning guarantees.

## Implications for the project

For a linear discretization, u_theta=u_star+e gives residual A e. If P^-1
approximates A^-1, the transformed residual approximates e; this motivates an
error-relevant loss in an appropriate discrete norm. It does not guarantee
lower physical relative L2 for nonlinear coupled PINNs or arbitrary weights.
PDE-state Jacobian inversion is different from dividing by the derivative of
an output map, and an inverse may require BCs/nullspace treatment.

Possible questions requiring additional targeted prior-art review and evidence:

- A cheap analytic, state-dependent inverse-map metric versus global ILU or
  multigrid: cost, interface error, physical relative L2, and failure regimes.
- Nonlocal conserved CH/corrosion: preserve mass/constant modes, coupled
  chemical/KKS constraints, and investigate horizon/interface-width/grid
  dependence. Generic block/spatial treatment is already covered by prior art.
- Demonstrate a mathematical connection to chosen physical error measures,
  with controlled map-only/residual-only/combined ablations. Current six-run
  vortex results do not show universal superiority of residual scaling.

No assertion of first-ever work, nonlocal novelty, or publication sufficiency.
Previous Glasner2001 dependent-variable phase-field preconditioning is another
relevant antecedent, but not a PINN residual-loss paper.

## Validation and exact resume point

Original formulas, scope and arXiv first-public dates checked using the linked
primary papers/metadata. No secondary AI-generated paper summaries are relied
on. Documentation-only review: inspect this report and memory changes, verify
local task link, and run git diff --check. No numerical tests are appropriate.
Task complete; no workers or pending actions. Further work would be a separately
scoped targeted review/derivation, not an implicit training or default change.
