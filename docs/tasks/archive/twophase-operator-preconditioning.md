# Two-phase CH accuracy and residual-preconditioning study

Date: 2026-10-27. Status: complete. Blocker: none. No workers remain.

## Goal and acceptance

User requested a more principled alternative to tanh(u)+0.1u, inspired by
residual-preconditioning literature, initially targeting approximately1% overall
phi relative L2. Subsequent explicit steering relaxes1% and prioritizes general,
simple improvements over problem-specific construction. Complete a bounded
controlled study, preserve physical PDE/reference and all old artifacts, expose
negative results, compare full-interval/terminal/interface errors and verify saved
models. This study meets that revised acceptance; the1% target is NOT reached.
No Git history operation was authorized or performed.

## Recommendation and measured result

Use examples/twophase2d/configs/train_vortex_conserved.py through train.py:
64x64 periodic grid, delta.04,4x64 GELU layers,3fixed integer spatial modes,
8open-stratified time samples,FP64/ordinary GPU,SOAP5000(lr1e-3) then at most
1000L-BFGS iterations/2000 function evaluations on32fixed midpoint times.
Residuals retain original equal weights; no inverse-Jacobian, Schur or causal
metric, no extra initial PDE time or added initial weighting. Use known phase
IC and grid mass, never future/reference field values in training.

| Final recommended cold start | Pooled phi L2 to common local | Terminal phi L2 to common local | Pooled interface-band L2 |
|---|---:|---:|---:|
| seed42 | 2.843% | 2.343% | 12.839% |
| independent seed2026 | 2.906% | 2.557% | 12.789% |

Matched nonlocal-reference pooled/terminal errors are3.0869%/2.5396% and
3.2070%/2.8525%. Mass projection imposes grid conservation to~2e-15; do not
present an enforced constraint as independently learned conservation. Whole-
field L2 does not imply equally small interface error. Two seeds of one
prescribed-velocity CH problem do not prove generic robustness or solved CHNS.
Recommended trainer times214.65s/213.00s include training JIT/eval/refinement,
exclude earlier setup, and are not directly comparable to5000-only controls.
Both L-BFGS runs stop at their iteration budget rather than gtol/ftol convergence.

Same-budget additive/conservative control is3.235% pooled/3.070% terminal.
The cubic coordinate provides a modest controlled gain. Prefer this physically
consistent model over the slightly lower pooled AD-hybrid cubic result2.769%
(terminal2.885%); global L2 alone does not select all physics/metrics.

Selected direct paths (no hashes):
- /root/autodl-tmp/tf-logs/twophase2d/20261027-vortex-conserved-conserved/model_final.eqx
- /root/autodl-tmp/tf-logs/twophase2d/20261027-vortex-conserved-conserved/config.json
- /root/autodl-tmp/tf-logs/twophase2d/20261027-vortex-short-seed2026/model_final.eqx

## Physical model and independent reference

The periodic prescribed reversed vortex is unchanged: U=2,T=2,L=1,
M=1e-4,epsilon=.02, initial center(.5,.75),level.2,width.06. No velocity,
pressure or Navier–Stokes equations are learned. With positive L approximately
minus Laplacian, physical residuals are

$$
r_\phi=\phi_t+\nabla\cdot(\mathbf v\phi)+MLq,\qquad
r_q=q-\phi^3+\phi-\varepsilon^2L\phi.
$$

Div(v)=0 makes conservative and advective forms equal continuously. The new
Fourier-flux discretization intentionally differs from the legacy AD point
slope; it matches the existing independent reference's conservative nodal flux.
The reference imports no JAX/PINN operator and was not regenerated or altered.
Existing validated metadata gives maximum-snapshot time refinement3.6154e-5
(graph)/2.5429e-5(local) and local spatial refinement1.9917e-7, mass drift<=2.22e-16.
Graph/local target gap is1.28047% pooled/1.16849% terminal. This includes finite-
grid/model discrepancies and is separate from optimization error; it is not a
strict lower bound when approximation errors can cancel. L2 uses all21 saved
times and common64grid; report matched and common-local targets separately.

## General improvements and independently checked derivation

### Monotone convexified-chemical coordinate

For F=(phi²-1)²/4 choose the strictly convex part Fc=phi⁴/4+phi²/2. Its gradient
u=Fc'(phi)=phi+phi³=F'(phi)+2phi is monotone with derivative1+3phi²>=1.
The analytic unbounded real inverse is

$$
g(u)=\frac2{\sqrt3}\sinh\!\left[\frac13\operatorname{asinh}\!\left(\frac{3\sqrt3}{2}u\right)\right],
\qquad g'(u)=\frac1{1+3g(u)^2}>0.
$$

Use u0=phi0+phi0³, s=(t/T)Ntheta, phi=phi0+[g(u0+s)-g(u0)]. Exact initial
data and verified CH overshoots are representable without Newton inversion,
a leakage coefficient, clipping or a prescribed evolved interface. This is one
fixed convexification choice, not a universally optimal coordinate. The map is
not copied from a reference trajectory or claimed to be a new general method.

### Compatible nodal derivatives and exact affine mass constraint

Legacy first derivatives came from AD of the point network, while nonlocal L
only saw grid values. Actual checkpoints show AD versus nodal Fourier-flux
advection RMS differences.25507(refined cubic) and.92627(failed96unit model).
The failed model's mean AD advection reaches.11436; Fourier mean<6e-17. This
supports discretization/aliasing freedom as an error source, not a unique causal
proof or a claim that AD itself is wrong. New explicit advection=spectral_flux
uses FFT div(v*phi) with zero even-grid Nyquist first derivative; no independently
adjustable point slopes enter advection. This is a spatial-discretization change,
not just residual scaling. Its sum is zero to roundoff.

phase_mass=exact_grid_mean subtracts mean(provisional_phi-phi0) on the fixed
uniform grid. This is the affine projection onto the actual conserved mass;
nonuniform meshes need quadrature weights and nonconserved equations must not
use it blindly. fields.predict_fields is the physical grid prediction API used
by training/time JVP/evaluation/checkpoint replay; raw model(x,t) is provisional.
Spatial query derivative of the fixed-grid shift is zero, time derivative is
included. Inverse point-map residual modes are rejected with this spatially
coupled projection because their claimed scalar Jacobian is no longer complete.

### Actual periodic-frequency bug fixed

The supposedly fixed integer modes were floating Equinox parameters, updated
by SOAP: a historical leaky model has[.76716584,1.8836843,2.9716012] instead of
[1,2,3], with opposing-edge differences.000945phi/-.048591q in one check.
Stop parameter gradients through modes. New trained models retain integers and
periodic values/derivatives. Preserve array leaves and historical stored values
for direct loading; never round/overwrite old models or imply they had exact BCs.
Shared temporal Fourier frequencies remain learnable. Base configuration remains
the original simple problem; this case-local bug fix affects new cold starts.

## Residual metrics tested and rejected as recommendations

Inspired by [Liu2024](https://arxiv.org/abs/2402.00531) and
[Wen2026 preprint](https://arxiv.org/abs/2609.36216), freeze bulk chemical
curvature c=2 and form A_tau=[[1/tau,ML],[-(c+eps²L),I]]. A Fourier block inverse
uses positive denominator1+tau*M*lambda*(c+eps²lambda). Transform
s=diag(1/tau,1)A_tau^-1r. A second causal version inverts the frozen backward-
Euler error system across sorted times: zphi_i=(zphi_prev+dt*(rphi-MLrq))/
(1+dt*MLB), zq=rq+B*zphi, B=c+eps²L, with phase normalization by explicit T.
Both preserve roots/invertible sampled intervals, retain spatial constant mode,
and omit advection/variable curvature from their approximate Jacobians. Dense
space and384x384 space-time solves and parameter finite differences verify the
algebra, not an accuracy theorem. Both fail to provide a reliable field benefit.

Adding one equally averaged t=0 PDE time (not an extra weight) slightly improves
pooled2.843->2.825%, but worsens terminal2.343->2.534% and q; do not adopt it.
It cannot be combined with the causal metric's zero-length interval. A proposed
separate initial_pde_weight option was never used in any16run and is removed
before completion to avoid an unused extra tuning parameter.

Larger96unit/8mode AD-hybrid models fail. Longer standard L-BFGS3000 lowers
loss to.000519/.000410 yet final local pooled L2 worsens to3.240%/6.336% for
42/2026. This demonstrates finite-collocation loss/error mismatch; no uniquely
isolated temporal-overfitting mechanism is claimed. Keep the simpler1000 limit;
no best-reference-epoch or seed selection. Long experiments stay reproducible
through the explicit research train_vortex_accuracy.py preset and saved sources.

## All completed controls

Pooled and terminal metrics below use the same common LOCAL reference, all21
times/final model. First4 runs precede the frequency fix and are explicitly
limited by incorrect exact-periodic claims. Budget is SOAP+accepted L-BFGS,
which have different costs; dtype/architecture/constraints are in the JSON.

| Stage / arm | Seed provenance | Budget | Pooled phi L2 | Terminal phi L2 |
|---|---:|---:|---:|---:|
| preconditioning/original | 42 | 5000+0 | 6.297% | 8.841% |
| preconditioning/cubic | 42 | 5000+0 | 4.780% | 6.505% |
| preconditioning/schur | 42 | 5000+0 | 7.036% | 9.488% |
| preconditioning/cubic_schur | 42 | 5000+0 | 4.553% | 6.216% |
| refinement/original | 42 | 5000+1000 | 4.942% | 5.366% |
| refinement/cubic | 42 | 5000+1000 | 2.769% | 2.885% |
| refinement/cubic_schur | 42 | 5000+1000 | 3.340% | 3.374% |
| resolution/cubic_resolved | 42 | 5000+1000 | 34.594% | 38.001% |
| resolution/causal | 42 | 5000+1000 | 33.588% | 37.365% |
| conserved/consistent | 42 | 5000+1000 | 3.235% | 3.070% |
| conserved/conserved | 42 | 5000+1000 | 2.843% | 2.343% |
| conserved/causal_conserved | 42 | 5000+1000 | 25.174% | 26.818% |
| initial-time/initial_time | 42 | 5000+1000 | 2.825% | 2.534% |
| confirmation/seed42 | 42 | 5000+3000 | 3.240% | 2.408% |
| confirmation/seed2026 | 2026 | 5000+3000 | 6.336% | 5.723% |
| short-confirmation/seed2026 | 2026 | 5000+1000 | 2.906% | 2.557% |


## Verification and source/provenance limits

- Existing20test affected suite passes90.822s; endpoint-transient test passes
  7.732s; final9preconditioning/mapping/mass/gradient tests after removing the
  unused weight pass52.816s. Together cover21distinct checks. No unchanged
  expensive reference/FEM/full-project suite was rerun.
- Independent NumPy dense block and space-time solves, analytic coordinate
  inversion/implicit derivatives, full parameter finite differences, constant
  physical roots, periodic gradients/updates/edges, manufactured conservative
  flux/zero mean, exact mass/time JVP and sharp initial-transient detection pass.
- All17presets validate physical identity/reference ordering/shapes and omit
  strategy seeds; runtime --seed remains provenance. Actual canonical train.py
  entry used for all16cold starts. No reference field targets enter loss.
- All16 saved final checkpoints replay on CPU; explicit first-cohort FP32 audit
  tolerance6e-6 covers measured maximum5.125999e-6. The historical evaluator
  default5e-6 is unchanged; first failure and full diagnostic are retained.
  Promoted/CPU/GPU evidence identifies backend rounding; no saved field is edited.
  Final FP64 CPU replay maximum1.398881e-14, tolerance1e-10.
- All16 frozen GPU replays are EXACT, with per-run expected initial arrays
  exact. The12post-fix trained models retain prescribed integer modes; maximum
  trained opposing-edge value gap is below5e-15. Old noninteger modes are
  recorded as known violations, never silently corrected.
- First4, fixed-FP64 triple, conservative triple and long42/2026 controls have
  byte-equal numerical source files within each cohort. The96unit exploratory
  pair has later inactive-wrapper source additions; do not call it a bitwise
  source-controlled comparison. Across stages, precision/architecture/sampling/
  optimizer changes are explicit and not attributed to the scalar map alone.
- Source snapshots, config, initial/final/optimizer-stage models, prediction
  arrays, complete histories and all failures remain at direct existing log
  roots. No hashes, migration, sibling dependencies or history changes.

## Diff-level implementation

network.py adds analytic inverse_cubic and fixes mode gradients while retaining
legacy leaves. fields.py adds optional conserved physical-grid projection;
advection.py precomputes conservative Fourier first-derivative multipliers.
losses.py exposes compatible advection and optional operator metrics/initial-time
collocation. preconditioning.py contains checked spatial/causal inverses.
problem.py validates optional schemas/combinations. refinement.py contains
explicit FP64 SciPy L-BFGS with convergence/evaluation metadata. train.py saves
SOAP/refined models/fields and stages separately, logs pooled common-local L2;
evaluate.py audits physical projected fields and combined stage counts. Research
configs and9focused tests cover these paths. Example README, development guide,
artifact index and project memory describe the final recommendation/limits.
No established other-example defaults or shared physics operators changed.

## Reproduction and artifacts

Recommended training:

```bash
source .venv/bin/activate
XLA_FLAGS='' python examples/twophase2d/train.py --config=examples/twophase2d/configs/train_vortex_conserved.py --config.log.run_name=my-vortex-conserved
```

Runtime --seed=2026 selects independent provenance, not a strategy parameter.
Archived launchers preserve exact commands for7cohorts. Analyzer accepts
--report-dir=reports/twophase-operator-<stage> with JAX_PLATFORMS=cpu. GPU verifier
supports --stage=<stage>; previous15 verified records are preserved separately
from the last single-run check. Summary script reads all saved paths without
hashes and regenerates comparison/figures. Reports remain under reports/;
archived essential outputs are linked below.

[Comparison](twophase-operator-comparison.json),
[validation](twophase-operator-validation.json),
[reference checks](twophase-operator-reference.json),
[source comparisons](twophase-operator-source-comparison.json),
[GPU15](twophase-operator-gpu-verification.json),
[GPU last](twophase-operator-short-gpu-verification.json),
[CPU roundoff](twophase-operator-replay-diagnostic.json),
[advection diagnostic](twophase-operator-advection-diagnostic.json),
[fields](twophase-operator-recommended-phi.png),
[convergence](twophase-operator-convergence.png),
[loss versus L2](twophase-operator-residual-vs-error.png).

## Exact resume point

Study complete. No further training, model design, default promotion or pending
approval. The1% target remains unmet; use the measured simpler recommendation.
Further research would require a separately scoped diagnosis of unsampled-time/
initial-transient residual control, not more unvalidated scaling. All workers
exit0, no blocker. Final verification, archived memory and local link/diff checks
are recorded in the adjacent validation JSON.
