# Matched coupled phase-field residual experiments — completed2026-10-24

## Goal and acceptance criteria

Compare original and transformed phase residuals in corrosion2d and
electropolish2d after auditing and regenerating model-aligned reference
solutions. Check PDE, coefficients, phase convention, IC/BC, physical times,
domain and discretisation explicitly. Keep the output representation and
soft IC treatment fixed; exclude hard IC changes. Use matched initialization,
samples, optimizer and5000-update budgets; report terminal/full-field L2 for
all physical fields, time histories and interface errors. Check reference
accuracy before attributing differences to optimization. Preserve old artifacts.

## Context

The corrosion audit derived cancellation for existing phi=(1+tanh(z))/2;
r_phi=2phi(1-phi)*r_z. Both g' and h' reactions cancel. Existing concentration
mixture/transport must be retained. The user warns old references may have
different BCs and explicitly requests alignment and fresh reference solves.
Electropolish2d is also authorized as a second experiment.

## Status

Complete: all four matched5000-update trials and both regenerated/refined
same-graph references are complete. Independent checkpoint/field replay and
KKS/pre-scale probes are complete. Production defaults remain unchanged.
Hard ICs were not used. No training/FEM process remains running.

## Blocker

None. The negative pilot does not settle the best performance after tuning
coupled residual scales or concentration parameterization.

## Next action

None required for this completed task. Any follow-up should isolate AC/CH
scale calibration from concentration-map changes, preserve soft IC/BC and
the same reference, measure gradients/cap activation, and report terminal L2.
Direct paths: `/root/autodl-tmp/data/<case>/results_matched_graph_20261024.npz`
and `/root/autodl-tmp/tf-logs/<case>/20261024-residual-{phase,latent}-seed42`.
The first zero-update electropolish setup failed in a newly added boundary
monitor (flat mask applied to a2D grid); corrected flattening, added a focused
metric regression, and preserved that directory as
`20261024-residual-phase-seed42-setup-failed`. The clean restart uses the planned
run name and fresh identical seed42 weights; no failed trial updates enter the
comparison. Six focused test methods passed in segments, including reference
metadata rejection and boundary-scoring regression. Existing expensive tests
were not rerun without changed inputs.

## Alignment audit and protocol

- Corrosion old FEM had only pit-mouth zero BCs; PINN has solid left/right/top
  one BCs too. Corrected solver. Small-mesh validation exposed facet marking
  missing a narrow mouth; use pointwise bottom-node selection and initial BCs.
- Both FEM loops clamped the final label while taking a full dt, and saves
  could overshoot. Set actual dt to the next save/final target. Old arrays
  lack model metadata and remain unchanged.
- Electropolish FEM used floored initial SDF lookup; now evaluates the same
  function as the PINN. Exact checks at non-grid points pass to 1e-14.
- Corrected electropolish README local CH sign and unsupported claims of exact
  pointwise Neumann enforcement/operator superiority; FD also has discrete BCs.
- Primary references independently integrate physical phi/c on the exact pair
  graph with prescribed nodal BCs and a derived sparse Jacobian. Refine BDF
  tolerances/max step before use. Local continuum convergence is not claimed.
- Retain physical coefficients/horizons, production architectures, KKS map and
  soft IC/BC. Preset: 101x51,delta0.02,K_core256,Nt16,pool100,SOAP,lr0.001,
  5000 updates,seed42,GradNorm every25; causal/stagger/interface weighting off.
- Latent arm: r_phi/[4phi(1-phi)] (r_z/2), equal scale at phi0.5; CH unchanged,
  no denominator floor/clipping or hard IC. No production default replacement.

## Results and limits

Terminal relative L2 uses every spatial grid node at the stated physical time,
including prescribed boundary nodes; errors are computed for physical phi/c,
not latent variables. The selected endpoint is exactly5000 completed updates.

| Case | Residual | Terminal time | phi relative L2 | c relative L2 |
|---|---|---|---|---|
| corrosion2d | original |50 s|0.03483970324176635|0.04135162629750127|
| corrosion2d | transformed |50 s|0.568385719526229|0.5458450196137687|
| electropolish2d | original |5 s|0.28604333038646146|0.2944289156240351|
| electropolish2d | transformed |5 s|0.34397927604422535|0.3792039111983612|

For each pair initial checkpoint bytes match exactly and saved configs differ
only in experiment.formulation and log.run_name. Times, deterministic graph
sample pools, seed, budgets, optimizer and loss machinery match. Corrosion's
last1000-update terminal-phi L2 medians are0.04629/0.56851; electropolish's
are0.29132/0.33663. Thus the endpoint comparison is not an isolated late dip.
No favorable earlier checkpoint is substituted for the matched endpoint.

Corrosion's transformed result barely grows the pit: terminal solid fraction
0.96816 versus original0.69462. Mean contour error worsens0.007089->0.20716.
Electropolish's mean contour distance improves0.04518->0.02927, but BOTH arms
have THREE vertical phi0.5 crossings in all101 columns; reference has ONE.
Global field L2 is worse after transformation and extra bands invalidate any
claim of overall geometry improvement from contour distance alone.

These are one-seed101x51 pilots with inherited scales, not evidence that no
retuned latent model can work. Electropolish's original result is itself poor.
No transformed residual is promoted to either production preset.

## KKS coupling: algebra versus optimization

Let z,v be the two raw network outputs, dc=c_Se-c_Le, C=1-c_Se+c_Le:

$$
\phi=p(z)=\frac{1+\tanh z}{2},\qquad
\ell(v)=C\frac{1+\tanh v}{2},\qquad
c=\ell(v)+dc\,h(\phi),\quad h=3\phi^2-2\phi^3.
$$

The driving variable q=c-dc*h(phi)-c_Le=ell(v)-c_Le. Consequently cancellation
of h'=6phi(1-phi) by p'=2phi(1-phi) is still exact. However, its derivative
with respect to v still contains ell'(v). The joint output Jacobian is

$$
J=\begin{pmatrix}p'&0\\dc\,h'\,p'&\ell'\end{pmatrix},\qquad
J^{-1}\binom{r_\phi}{r_c}
=\binom{r_\phi/p'}{(r_c-dc\,h' r_\phi)/\ell'}.
$$

The experiment changes only the phase residual (half the first component),
retaining r_c. It is partial preconditioning, not full inversion of this
matrix. Both have the original zero-residual solution when p',ell' are nonzero,
but full inversion also introduces division by the concentration Jacobian;
it is not an automatic cure. An independent float32 AD probe verifies J and
J*inverse_action for non-saturated outputs with1e-6 relative tolerance.

On corrosion's transformed final model, raw v ranges-15.4731..-6.9250 on all
grid nodes at normalized times0,0.5,1. The maximum ell is3.51e-8 and85.17%
of those nodes have exactly zero ell' in float32. Reference q stays inside
[-c_Le,0] to2.78e-17; the reference is not excluded by the KKS range. The
solid boundary requires ell=c_Le=0.0356643, whereas saturation at ell~0 leaves
c~dc=0.9643357 and loss~c_Le^2=0.00127194. The fixed-batch solid BC parameter
gradient is3.92e-11 at this final checkpoint. Both arms use the same KKS map:
this establishes a vanishing-gradient failure mode, not its sole cause.
Electropolish has no exactly-zero concentration derivatives in either arm;
the same saturation explanation must not be assigned to that result.

## Pre-scale audit

Saved corrosion configs inherit s_AC=1e-7 and s_CH=1; electropolish inherits1/1.
PDE losses are mean ABSOLUTE residuals (not residual squares), so

$$
L_{AC}^{old}=s_{AC}\langle|r_\phi|\rangle,\qquad
L_{AC}^{new}=s_{AC}\left\langle\frac{|r_\phi|}{4\phi(1-\phi)}\right\rangle.
$$

The divisor equals1 at phi0.5 but amplifies by25.25 at phi0.01 and250.25 at
phi0.001. A constant scale cannot restore the original spatial weighting.
The AC physical coefficients also remain large (N1=1.07e10,N2=1.76e9,N3=1.03e6;
N_CH=4.25). Deleting1e-7 is not justified by analytical cancellation.

The scaling replay uses the first deterministic graph core/time batch and
the same original-final checkpoint for both residuals. Parameter-gradient L1
norm (the quantity used by the implementation's GradNorm) changes as follows:

| Term | Original residual | Transformed residual |
|---|---|---|
| AC, after configured scale |1996.31665|25203.82813|
| CH |1794.52319|1794.52319|
| solid BC |0.0001745323|0.0001745323|
| IC |1.21612358|1.21612358|

At initialization AC norms are nearly equal (1997.39/1987.50), so the scale
distortion emerges as the solution separates into phases. On the transformed
final checkpoint, using the transformed residual, gradient norms in order
AC,CH,zero BC,one BC,IC are12252.2,0.06410,0.06935,3.92e-11,1.86011. Recomputed
GradNorm proposals hit the10000 weight cap for CH and both BCs; the solid BC
norm is below the1e-6 floor. Even the capped solid BC contributes only3.92e-7
in gradient norm, versus~2451 for AC. Adaptive weights cannot recover a zero
output derivative. These are proposals measured at saved checkpoints, not
historical optimizer weights, which were not saved.

Without norm/weight clipping or delayed updates, inverse-norm balancing can
compensate constant per-loss scales in RELATIVE gradient magnitudes, up to a
common factor; it cannot undo altered spatial weighting/gradient directions.
Here norms are clipped to[1e-6,1e6], weights capped, and refreshed every25
updates. Hypothetically removing AC pre_scale at initialization gives~1.99e10
gradient norm, above the1e6 ceiling, breaking that compensation immediately.
Thus inherited scaling merits a controlled retuning study; current data do
not establish that retuning alone removes KKS saturation or improves L2.

## Validation and artifacts

- Reference temporal-refinement max snapshot phi/c relative L2:
  corrosion3.6408888266410613e-9; electropolish1.523935937181892e-8.
  Float64 BDF uses independent physical-field RHS and sparse analytic Jacobian;
  tolerances tighten1e-7/1e-10->1e-8/1e-11 and max step0.01->0.005.
- Production residual/latent chain-rule identity, unchanged CH, finite/finite-
  difference gradients, soft IC gradients and compiled SOAP refresh/nonrefresh
  checks passed. Boundary metric and reference mismatch rejection passed.
- Actual small-mesh FEniCS solves checked pit/solid BCs, rough direct IC and
  save/final times[0,.004,.008,.01]. Full production FEM was not regenerated;
  the fresh quantitative comparison references are explicitly the graph solves.
- Independent CPU checkpoint replay differs from saved GPU fields by at most
  2.51e-6; terminal relative L2 recomputes within1e-12.
- Final git diff --check passes. AST syntax checks pass for eight changed/new
  Python modules/scripts; all245 registered external directories and eight
  direct experiment reference/report/checkpoint files exist. Report links
  resolve. Replayed terminal fields/curves were regenerated in the archive;
  the final field-figure colorbar layout was visually inspected. Scaling
  audit AD Jacobian max absolute discrepancy is5.96e-8. No training/FEM
  process remains. No Git history action was taken. Old artifacts are retained.

Direct report files (no hashes or manifest gates):

- [Replay JSON](coupled-latent-residual-experiments.json),
  [replay script](coupled-latent-residual-experiments.py),
  [training L2 curves](coupled-latent-residual-experiments.png),
  [terminal fields](coupled-latent-residual-experiments-fields.png).
- [Scaling/Jacobian audit JSON](coupled-latent-residual-experiments-scaling.json)
  and [script](coupled-latent-residual-experiments-scaling.py).
- Each case's direct run/ref/checkpoint paths and terminal metrics are indexed
  in artifacts/index.yaml under residual_comparison_20261024.

Diff-level changes: added experimental config/residual/runner/focused tests;
corrected two FEM IC/BC/time paths; updated example/root READMEs, artifact index
and memory; added paired replay/figures and scaling diagnostics. Existing
melting defaults and other earlier-session changes are preserved.
