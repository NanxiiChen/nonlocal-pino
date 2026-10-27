# Residual transformation as nonlinear preconditioning

Date: 2026-10-27. Status: completed. Blocker: none.

## Goal and acceptance

User asks whether division by mapping Jacobian can generalize and whether it is
numerical preconditioning. Explain exact zero equivalence, nonlinear variable
coordinates versus residual left scaling, generalized scalar/vector evolution,
actual conditioning/parameter optimization and system/mass limitations. Relate
to observed melting/CH outcomes, without conflating map-only gains with scaling.
Read primary sources; independently check derivative/conditioning identities.
No code/default/reference/training/Git change requested or planned.

## Context and plan

Mandatory memory/development/numerical guidance read, prior task complete.
Current controls: CH unbounded map-only improves interfaces; inverse-map both
has small global/area gains and worsens interfaces versus map-only. Melting exact
tanh cancellation is more favorable. Existing archived generalization assessment
already identifies Glasner2001 nonlinear preconditioning of diffuse interfaces.
Start generic R(y)=0,y=G(z),B=DG, transformedS=W(z)R(G(z)); B is not the PDE
Jacobian A=D_yR nor the parameter Jacobian. Invertible W preserves zeros, not
least-squares minimizers at finite residual. Under aligned vector evolution,
W=B^-1 normalizes time tangent; mixed CH algebraic q row need not receive that
same phase factor. Verify root derivative W*A*B, full weight derivative away
from root and near-root Gauss-Newton Gram. Check the square similarity limitation
and exact Newton invariance for fixed nonsingular left scaling. Preserve original
nonlocal discretization before pointwise scaling. Final explain in Chinese with
primary source links and practical criteria, excluding hard-IC theory.


## General formulation and distinction between three Jacobians

Let the original residual operator be \(\mathcal R(y)=0\), with a pointwise
smooth locally invertible state map \(y=G(z)\). On the admissible state domain,
let \(B(z)=DG(z)\) and choose a nonsingular residual preconditioner \(W(z)\):

$$
\widetilde{\mathcal R}(z)=W(z)\mathcal R(G(z)).
$$

G changes dependent-variable coordinates (a nonlinear right-coordinate change);
W changes residual coordinates (nonlinear left residual scaling). This is a
restricted, state-dependent residual-preconditioning family, not a claim that it
implements every nonlinear Schwarz/inner-solver preconditioner used in PETSc.
Provided G covers the desired physical solution and W is nonsingular, the pointwise
zero sets agree under G. Finite nonzero least-squares minimizers need not agree.
No hard initial constraint is needed for this theory.

For a scalar first-order evolution residual
\(\mathcal R(\phi)=\phi_t+\mathcal F[\phi]\), a time-independent map
\(\phi=g(u)\) gives the natural normalized residual

$$
\widetilde r=\frac{\mathcal R(g(u))}{g'(u)}
=u_t+\frac{\mathcal F[g(u)]}{g'(u)}.
$$

This holds for local or nonlocal F, in any dimension, but does not imply F factors
g'. Transform the declared discrete operator directly; a local Laplacian chain
rule is not automatically a chain rule for a finite-horizon graph. Explicit x/t
map dependence must be retained in spatial derivatives; if G explicitly depends
on time, its partial G_t term must also be included.

Three different Jacobians must stay distinct:
- B=DG: pointwise state-map Jacobian (scalar g' for phase only).
- A=D_y R: full PDE/discrete residual Jacobian, including differential operators,
  nonlinear reactions, couplings and boundary/initial rows as applicable.
- J_theta=D_theta r: residual sensitivity to neural parameters, usually rectangular.

B^-1 is not A^-1. SOAP/Adam operate in parameter space, a separate design choice.
Variable mapping and residual scaling can coexist with parameter preconditioning.

## Conditioning: exactly what follows, and what does not

At an exact root z*, the product/chain rule gives

$$
D_z\widetilde{\mathcal R}(z_*)=W_* A_* B_*.
$$

Away from the root, a direction h also has the extra term
\((DW(z)[h])\mathcal R(G(z))\). Our implementation differentiates this weight;
stop_gradient(W) instead defines a different update/iteratively reweighted scheme.

For a finite square residual system where W=B^-1, the root Jacobian is the
similarity transform B^-1*A*B. Eigenvalues are unchanged. Singular values, Euclidean
condition number and non-normality can improve or worsen. A diagonal A commuting
with diagonal B is unchanged, so map-Jacobian division cannot be claimed to remove
all spatial/time stiffness. For continuous operators, domains/norms/boundary data
need analysis; the finite-dimensional statement is not a blanket spectral theorem
for arbitrary operator domains.

The neural parameter problem is generally rectangular, with a restricted trial
space. For MSE, \(L=\tfrac12\|W r_\theta\|^2\) and near a zero residual the
Gauss-Newton Gram is approximately \(J_\theta^T W^T W J_\theta\), where J_theta
already contains the state map. Far from a root, the weight derivative contributes.
Thus this is a state-dependent residual metric, not automatically a uniformly better
conditioned optimization problem. Melting's MAE variant has different norm powers;
when discussing MSE the residual multiplier w induces squared weight w^2.

A fixed nonsingular left row scaling cancels from an exactly solved square Newton
step: (WJ)^-1*(-Wr)=J^-1*(-r). It can still change iterative linear-solver conditioning,
finite precision/stopping and least-squares training. State-dependent full Newton
also includes DW*r, so the fixed-W statement does not imply identical global
nonlinear iterations for the transformed problem.

## Generalization to coupled phase fields and target error metrics

For aligned vector evolution y_t+F(y)=0 and nonsingular B, W=B^-1 normalizes the
time tangent. With a nontrivial mass matrix M(y), the time tangent is M(G)*B;
a singular differential-algebraic system cannot be normalized by blindly inverting
that product. Multiphase simplex maps have rank constraints and require independent
coordinates/tangent spaces rather than inverting a full rank-deficient softmax map.

For the mixed CH unknowns (phi,q), G(u,v)=(g(u),v) has B=diag(g',1).
Its inverse-coordinate residual scaling is diag(1/g',1): the phase transport row
naturally gets the factor. Dividing the algebraic chemical-potential row by g' as
well is an ADDITIONAL block-norm choice, not forced by that map's Jacobian.
The existing map_both experiment deliberately makes this further choice. It is
pointwise equivalent when nonsingular, but should not be labeled the full-vector
inverse-map Jacobian without this qualification. Coupling-aware block/Schur
preconditioners or spatial approximate inverses are legitimate broader candidates;
none has been implemented or validated by this discussion.

A useful physical-error design principle follows from a well-posed linearization,
including needed BC/IC/gauges: r approximately equals A*e for physical-state error e.
Then ||e|| approximately equals ||A^-1*r||. Choosing W to approximate the relevant
inverse/solution-error metric has a direct error motivation; choosing B^-1 alone
primarily compensates coordinates and need not approximate A^-1. Interface or
conservation objectives require their own error/adjoint/constraint analysis.
This is a local linearization argument, not a global nonlinear L2 guarantee.

Practical criteria:
- Cover the true admissible solution range, keep W nonsingular; uniform lower and
  upper singular-value bounds additionally yield norm equivalence/stability control.
- Derive all source, derivative, boundary and coupling terms consistently.
- Treat coupled rows/units/time-mass constraints explicitly, avoiding arbitrary
  identification of every row with the phase map's derivative.
- Preserve physical conservative operators and check their zero mode. For periodic
  divergence-free CH, integral(r_phi)=d/dt integral(phi); under scalar unnormalized
  r_u=r_phi/g', mass drift is integral(g'*r_u), not identically zero at finite error.
  Independent q constitutive error can damage dynamics even if exact transport
  still conserves mass.
- Evaluate full-field/interface/coupled-field/mass metrics and training Jacobian
  conditioning separately, with matched map-only controls and independent seeds.

## Interpretation of recorded cases

Melting has a favorable factorization: tanh time tangent and specific reactions
share 1-phi^2. Phase coordinate saturation can be analytically compensated. The
original discrete nonlocal residual is transformed, not replaced by a different
continuum discretization. Actual melting gains are recorded elsewhere; this is
not proof for all Allen-Cahn/CH models.

CH Lq does not generally carry g'; verified CH excursions also exclude strict
(-1,1) tanh representation. Our bounded/unbounded-map six-run experiment shows
map-only interface gains and norm-dependent tradeoffs: mapped-original common-local
band0.16045 versus map_both0.17455, despite slightly better full-field L2 under
map_both. This is expected for a modified residual metric on a restricted finite
network. No actual PINN condition numbers were measured by those runs; do not
infer that conditioning improved merely from loss/L2 or the name preconditioner.

## Primary sources checked

- Glasner2001, *Nonlinear Preconditioning for Diffuse Interfaces*, JCP174,695–711,
  DOI10.1006/jcph.2001.6933. Author-hosted PDF:
  https://math.arizona.edu/~kglasner/research/precond.pdf . Sections1–3 frame
  profile-based dependent-variable changes as nonlinear preconditioning and use
  atanh(phi). Evidence is discretization/truncation improvement, not PINN training
  or a general condition-number theorem. No claim that tanh/Jacobian division itself
  is a new concept.
- PETSc SNES official manual, nonlinear-preconditioning section:
  https://petsc.org/release/manual/snes/#nonlinear-preconditioning . Distinguishes
  left nonlinear-function modification and right function composition; general
  PETSc implementation uses inner nonlinear solvers and is broader than our local
  map/row-scaling family. Newton discussion also supports separating exact linear
  solves from iterative preconditioning. Algebraic formulas above are independently
  derived here rather than attributed wholesale to the manual.

## Independent verification and completion

Archived NumPy check verifies root W*A*B derivative (max1.15e-10) and full away-root
weight derivative (max2.92e-10), exact constant-left Newton-step invariance,
similarity eigenvalues and conditioning examples. A non-normal2x2 system has
condition~500002.5; coordinate scaling gives either2.618 or5e11 while preserving
eigenvalues. Commuting diagonal stiffness is unchanged. A restricted weighted
least-squares example moves minimizer0->99/101; A^-1*r=e verified for linear error.
These are algebraic sanity checks, NOT measured PDE/PINN conditioning or new runs.

[Check script](residual-preconditioning-theory-check.py),
[raw checks](residual-preconditioning-theory-checks.json).
No application code/config/default/reference/checkpoint/training or Git history
operation. No new dependencies. Memory changed: this archived task, PROGRESS and
DECISIONS. Documentation links/JSON/AST/diff verification recorded at completion.
Next action: none required; coupling/target-error preconditioning remains a research
proposal, not an automatically scheduled implementation.
