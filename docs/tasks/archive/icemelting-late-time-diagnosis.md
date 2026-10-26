# Ice melting late-time diagnosis — 2026-10-24

Location update: the historical run labelled `SOTA` in this diagnosis now lives
in `/root/autodl-tmp/tf-logs/icemelting3d/legacy-raw-phase-soft-ic`. Its checkpoint
contents and metrics are unchanged; see the [rename record](icemelting-run-tags.md).

## Status, goal and acceptance

Completed diagnosis. User confirms reference means the analytic profile in
`examples/icemelting3d/train.py`; checkpoints are under
`/root/autodl-tmp/tf-logs/icemelting3d`. Goal: explain late acceleration and
check reference correctness. Acceptance: trace equations, independently check
reference/discretisation, inspect real checkpoints, distinguish proven findings
from unresolved run-specific observations. No training, physical/config/API
changes, external artifact writes, checkpoint resume or Git history changes.

[Numerical evidence](icemelting-late-time-diagnosis.json) includes exact
checkpoint paths/SHA256, all six representative run measurements, derivative
checks, convergence results and historical replay comparisons.
[Scientific figure](icemelting-late-time-diagnosis.png).

## Main finding

The spherical linear-radius tanh is **not an exact solution** of the declared
3D PDE. Its error is small for the default parameters. The larger observed
late-time PINN drift accompanies interface broadening, in a strongly sensitive
IVP whose ice equilibrium is linearly unstable. Direct local and discrete
controls remain close to the nominal radius; local AD PINN shows the same drift
as the nonlocal PINN. A nonlocal quadrature defect is real but does not explain
the observed direction/magnitude alone.

Examined default checkpoints still have positive ice at \(t^*=1\); the user's
specific run with complete disappearance has not been identified. Do not claim
that all runs disappear or extrapolate the models beyond their trained interval.
Controlled, tiny IC perturbations do independently reproduce early loss of the
positive phase without any neural network.

## Source trace

- `train.py:63-75`: coefficients and length/time scales.
- `train.py:96-110,115-152`: all evaluation/snapshot reference values come from
  `losses.analytic_solution`; no FEM data is read.
- `losses.py:95-115`: nonlocal/local residual signs agree with the stated PDE.
- `losses.py:128-166`: volume-averaged PDE MAE and IC MSE.
- `train.py:318-344`: independent uniform times over [0.001,1], cycling fixed
  subgraph pool; all-time training, no time marching by default.
- `configs/train_base.py`: \(T_c=4\), \(\epsilon^*=0.0225\),
  \(\delta=0.01\), 128 points per coordinate, gamma=0, causal disabled,
  output constraint disabled.
- `nn/arch.py:102-126`: tanh is a hidden activation; final output is linear.
  The config comment "tanh output" does not bound the output.

## Independent analytical substitution

Suppress stars and let \(d=\sqrt{2}\epsilon\), \(r=|x-x_c|\),
\(q=\tanh((R_0-\lambda t-r)/d)\). For \(r>0\):

$$
q_t=-\frac{\lambda}{d}(1-q^2),\qquad
q_r=-\frac{1}{d}(1-q^2),\qquad
q_{rr}=\frac{q^3-q}{\epsilon^2}.
$$

The radial 3D Laplacian includes \(2q_r/r\). The implemented local residual is

$$
q_t-N_1\Delta q+N_2(q^3-q)+N_3(1-q^2)
=\frac{2N_1}{r d}(1-q^2)\ne0.
$$

Here \(N_2=N_1/\epsilon^2\), \(N_3=\lambda/d\). The profile also does not
exactly satisfy zero normal derivative on the finite cube faces, and its
nonzero radial derivative at the origin precludes a globally smooth exact
radial solution. Neither issue is silently corrected.

A thin-interface estimate is \(R'\simeq-\lambda-2N_1/R\), **not an exact
radius ODE** for a fixed tanh profile at finite thickness. Default values:
\(N_1=4\times10^{-5}\), \(N_2=0.079012345679\),
\(N_3=6.285393610547\), \(\lambda^*=0.2\), \(R_0^*=0.35\).
The curvature correction to speed is only 0.114% initially and 0.267% at R=0.15.
It cannot explain the measured ~20% terminal PINN speed excess by itself.

## Independent reference checks

1. Radial local control, independent of PINN operators: solve
   \(u_t=N_1(u_{rr}+2u_r/r)-N_2(u^3-u)-N_3(1-u^2)\).
   Domain r in [0,0.5], regularity at 0, Neumann at 0.5. This sphere is an
   **interior control, not the exact Neumann cube BVP**. The initial outer face
   is far from the interface relative to diffusion length sqrt(N1)~0.0063.
   Use centred differences; origin Laplacian 6(u1-u0)/h²; reflected outer
   ghost; SciPy BDF with explicit tridiagonal Jacobian, rtol=1e-9, atol=1e-11.
   Sample 21 evenly spaced times. With 500/1000/2000 radial intervals,
   terminal R = 0.1494710287 / 0.1494716251 / 0.1494717764. Last refinement
   changes R by 1.51e-7. Analytic R is 0.15 (about 0.352% high).
2. Default cubic stencil consistency: h=1/127, delta/h=1.27; sqrt(2)h exceeds
   delta, so an interior point has only **six axial neighbours**. The comment
   "delta/h~3.8, k_avg~242" is stale. For weight
   w=315/(2*pi*delta^5)*(1-h/delta)^2*h^3,
   the coefficient of -Laplacian is w*h²=0.6858547757, rather than 1.
   Actual SparseD on a 7³ patch gives D(|x|²)=-4.1151314 vs local -6.
   Three padded subgraphs reproduce full core matvecs within ~9.8e-4
   absolute float32 cancellation error before multiplication by N1.
3. Direct declared six-neighbour evolution: even-coordinate symmetry reduces
   the 128³ cube to a 64³ octant, with coordinates (i+0.5)h relative to centre.
   Edge ghosts implement mirror symmetry at centre planes and missing graph
   neighbours at exterior faces. Recompute declared kernel weights in float64;
   integrate using RK4 with dt=0.001 and 0.0005. Near-axis zero crossing at
   y=z=h/2 gives terminal x radius 0.15020458. Max radius difference between
   the two dt values is 3.10e-11. This sampled near-axis line has a small
   transverse offset: its initial analytic crossing is 0.34994824, not 0.35.
   Do not call this a full spatial-convergence study or exact local reference.
   It nevertheless rules out this stencil *alone* producing the measured
   R~0.146 terminal PINN radius. A float32 control pins the central plateau
   to exactly +1 and gives R~0.15065, illustrating precision sensitivity;
   the converged comparison uses float64.

## Strong forcing and unstable ice phase

At spatially uniform points the reaction is

$$
f(u)=(1-u^2)(N_2u-N_3),\qquad
f'(1)=2(N_3-N_2)=12.41276253>0.
$$

For u=1-eta, positive eta initially grows approximately as exp(12.41276*t).
The amplification at t=1 is 245920.24. Thus an IC MSE near 1e-6 is not evidence
that long-time phase evolution is accurate. This is a conditioning property of
the chosen equation/parameters; it does not by itself prove the trained network
is literally time-stepping its initial error.

Repeat the independent radial solve with
u0_perturbed=u0-eta0*(1+u0)/2. With eta0=1e-4, initial pointwise perturbation
is at most 1e-4 (MSE at most 1e-8), but the positive phase disappears between
sampled times 0.75 and 0.80. For eta0=1e-3 it disappears between 0.60 and 0.65.
These are sampled brackets, not event-resolved extinction times. Early radius
agreement remains good; this reproduces the qualitative late acceleration.
Outside [-1,1], the implemented 1-u² source also reverses sign. An unconstrained
PINN can overshoot even though the intended solution stays in this interval;
observed initial central overshoots ~0.001–0.002 are therefore material.

## Checkpoint evidence and provenance

Six representative runs were inspected by read-only inference. Most useful:

| Model | File | R along +x at t=1 | central phi at t=1 |
|---|---|---:|---:|
| analytic profile | formula | 0.15000000 | ~0.999839 |
| SOTA | ckpt_017600.eqx | 0.14619212 | 0.97187585 |
| local_ad | model_final.eqx | 0.14593588 | 0.97076863 |
| 20260715-205828 | model_final.eqx | 0.14611438 | 0.97110120 |
| baseline2 | model_final.eqx | 0.14811091 | 0.97401010 |

Measure zero crossings on 481 radial points in each of six coordinate directions;
record all directions in JSON. Derivative checks use +x roots, AD in x/t, and
respectively the six-neighbour stencil or AD Hessian for the actual method.

SOTA at t=1: phi_r=-25.93779 vs analytic -31.42697;
effective width 1/abs(phi_r)=0.03855378 vs 0.03181981 (+21.2%).
The zero-level inward speed phi_t/phi_r is 0.24005124 vs nominal 0.2.
Local AD gives 0.23747152 and width 0.03829087. Signed interface PDE residuals
are ~0.0713 and 0.1019 respectively, despite volume-averaged loss near 0.008.
At zero phase, the source magnitude is fixed at N3, so a flatter profile yields
faster level-set motion when the residual approximately balances that source.
Broadening and speed are measured; attributing their full origin to a particular
optimizer requires a controlled training ablation and is not claimed here.

The volume-based metric dilutes late ice errors: analytic ice occupies only
4*pi*0.15³/3~1.41% of the cube at t=1. Constant -1 has zero PDE residual.
Uniform volume sampling, no interface weighting, soft IC, and no causal/time
marching constraint create an optimisation weakness consistent with this drift;
none alone is proven to be the unique training cause.

Historical caveat: before the activation dispatch change, train.py used
getattr(jnp, activation) or tanh. GELU is in jax.nn, so gelu-labelled baseline2
and local_ad checkpoints actually used tanh. Verified in Git a555329's train.py.
Static activation is not stored in the Eqx array leaves. Using current GELU
silently reconstructs the wrong function despite matching shapes. Explicitly
reconstruct tanh for those historical runs, then validate against exact logged
8000-point evaluation sample (PRNGKey123, five times):

| Run/step | Replayed relative L2 | TensorBoard relative L2 |
|---|---:|---:|
| SOTA/17600 | 0.00957602728 | 0.00957601704 |
| local_ad/19800 | 0.00922454521 | 0.00922454614 |
| baseline2/19800 | 0.01016190927 | 0.01016190648 |

All differences are below 1.1e-8. Historical graph_conv configs lack a matching
current builder; missing-config or unverified experimental architectures are
excluded. An exploratory whole-directory sweep stopped on that schema instead
of inventing missing architecture settings; its unverified results are not used.

## FEM audit (not the user's analytic reference)

Weak-form signs agree with the local PDE. It is not presently a validated drop-in
reference: solver Tc=5 vs training Tc=4; eps is derived from mesh N, whereas PINN
uses explicit fixed eps. At N=64 FEM eps*=0.02251405, close but not identical.
Refining N changes eps and hence changes the PDE, so it does not establish
fixed-physics mesh convergence. MPI export saves only rank0's local vector/coords
with no global gather; printed relative error is likewise rank-local. The default
output NPZ is absent in the available data directory. Local AD training also
has no explicit Neumann residual; a natural boundary condition is not imposed
merely by omitting a boundary loss in a strong-form PINN. These issues need
separate corrections before using FEM/local AD as certified exact-cube references.

## Recommended next action and limits

If retaining the current PDE, replace the *claimed exact* analytic acceptance
criterion with a numerical reference verified at matched physical parameters,
boundaries and precision; keep the formula as an approximate comparison.
Track per-time radius, ice volume, interface width/speed and phase-conditioned
residual/IC errors. Then test short time windows or causal training and more
accurate IC/range constraints before tuning arbitrary optimizer weights.
Causal code currently averages zero-masked halo/padding rows over all rows,
so its core normalisation must be fixed/validated before enabling it for sparse
subgraphs. Importance sampling must preserve an explicitly defined objective.
Repair delta/h consistency with a stated quadrature/convergence plan; changing
delta alone changes the numerical operator and is not a justified universal fix.

If a strictly linear-radius sphere is the desired exact benchmark, a different
manufactured equation/source must first be agreed; do not silently change the
physical model to force agreement.

Relevant primary literature for training follow-up (not evidence for the above
measured numbers): [causal PINNs](https://arxiv.org/abs/2203.07404),
[propagation failure and R3 sampling](https://arxiv.org/abs/2207.02338).

## Session changes, verification and exact continuation

Changed only project memory and diagnostic artifacts: docs/PROGRESS.md,
append-only docs/DECISIONS.md, this archived task, JSON evidence and PNG figure.
No production code/config/README claims were silently changed. Mathematical
substitution, CPU checkpoint replay, independent radial spatial convergence,
float64 direct-stencil time convergence, real-operator moment/subgraph checks,
and figure inspection completed. git diff --check is the final repository check;
no unrelated test suite or GPU/FEM training was run.

No blocker for this diagnosis. A follow-up implementation task must choose the
retained PDE versus a manufactured exact benchmark, fix reference provenance,
and identify the precise run if full disappearance within [0,1] must be traced.
Start from the above matched controls and historical replay checks. Scratch
experiment programs remain in /tmp (paths in JSON); durable results and
reproduction parameters are stored here. Do not resume or rewrite old runs.
