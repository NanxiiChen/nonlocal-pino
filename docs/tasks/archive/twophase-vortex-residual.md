# Vortex residual-preconditioning experiment

Date: 2026-10-26. Status: completed. Blocker: none.

## Goal and acceptance

User asks whether melting-style residual transformation improves inadequate
accuracy and diffuse-looking interfaces in the reversed-vortex CH pilot.
Keep physical PDE/reference, original IC-only network, inputs/seed42/SOAP/5000
updates fixed. Compare fresh original, transport-only and both-residual controls;
audit common-local/matched phi L2, interface-band phi error, contour distance,
transition width, q, raw PDE residuals and soft area drift at21 saved times.
A negative result is acceptable evidence, not grounds to hide a run or tune seeds.
No default promotion without evidence, no Git operation authorized this turn.

## Relevant context and checked derivation

Previous pilot nonlocal t1/t2 common-local L2=0.05479/0.07889, interface-band
terminal L2=0.34266; 64x64,4shared times,4x64GELU,SOAP1e-3,FP32,ordinary.
Physical local CH: r_phi=phi_t+v.grad(phi)+M Lq, r_q=q-phi^3+phi-eps^2 Lphi,
L=-Delta or periodic positive finite-horizon graph. Melting divides by J=1-phi^2
under phi=tanh(u), but CH Lq need not contain J; strict tanh range also excludes
reference excursions (local min/max -1.09837/1.06254; graph -1.06899/1.04804).
Use an explicit empirical bounded continuation of inverse J, with eta0.1:
w(phi)=sqrt(1+eta^2)/sqrt(eta^2+min((1-phi^2)^2,1)).
For |phi|<1 as eta->0 it recovers1/(1-phi^2); w(0)=1 and
1<=w<=sqrt(1+eta^2)/eta=10.0499. Positive finite w preserves pointwise PDE
zeros, permits overshoots and caps squared residual amplification at101.
The denominator cap prevents reducing loss through large excursions. Full AD
through w is retained; this is a changed optimization norm, not new physics,
not exact tanh latent-chain form and not an interface weight (it peaks near±1).
No hard-IC or learned representation changes; no cancellation claim for CH.

## Plan and exact resume point

Mandatory memory/development/numerical/JAX guidance read; no different active task.
Next: add explicit optional residual config union (legacy absent=unchanged original
schema), focused zero/chain-rule/gradient/overshoot checks, independent fixed-model
weight allocation probe, then3 serial cold starts. Keep required mode/eta explicit
in new experiment preset and preserve old serialized config/checkpoint loading.
Reference already independently converged; do not regenerate unchanged data.

## Pre-training checks and launch

10focused tests pass in29.781s: prior7 plus weight/chain-rule limit/overshoots,
NumPy mixed-loss equality with full parameter finite differences, physical constant
zeros and legacy-schema equivalence/invalid eta. No physics/reference change.
Fresh serial original/transport/both launcher started;5000 updates each,eta0.1,
ordinaryFP32/SOAP/common42. Source frozen for cohort. Exact resume: inspect
reports/twophase-vortex-residual/manifest.json and per-arm logs, then CPU checkpoint
replay and geometry/raw-residual audits. Launcher stdout is in /tmp/twophase-vortex-
residual-launch.log, individual raw logs remain in the report directory.

## Second-stage decision

All3 initial controls finish. Bounded transport/both interface errors increase to
~0.776/0.954 vs original0.332. Full stage1 source parity verified and frozen at
reports/twophase-vortex-residual/bounded-source before further implementation.
CPU/geometry/raw residual audit is running from the frozen stage1 code.
Add3 explicit leaky-tanh map controls to distinguish actual latent representation
from residual norm: alpha0.1,8-step Newton inverse IC, exact hard IC via differences;
original/map_transport/map_both inverse-J norms. No physics/reference changes or
phase bounds. Validate Newton values/implicit derivative/full residual gradients
before launching the new serial5000-update cohort. Every result will be retained.

12focused tests now pass in41.840s, including Newton inverse/implicit derivative,
exact mapped-network IC/time-chain rule and full inverse-map residual gradients.
Stage2 serial mapped-original/map_transport/map_both5000-update controls started
under reports/twophase-vortex-latent (same original seed42/parameters and samples).
No further training-source change planned during this cohort. Stage1 CPU audit
passes; observed terminal common-local phi0.07997/0.21941/0.27171 for original/
transport/both. Width proxies0.1620/0.2835/0.6027 vs reference0.10918. Stage1 zero
contours survive but shrink/shift under transformed arms. Bulk squared-residual
share increases strongly under weighting (both-phase24.1%raw->83.9%weighted),
supporting bulk emphasis rather than improved interface resolution; not unique
causal proof. Geometry self/shifted-circle/tanh-width checks pass.
Exact resume: wait stage2 manifest then run residual-analyze.py --report-dir=
reports/twophase-vortex-latent on CPU; retain both stage analyses and compare.


## Final results and interpretation

All6 seed42 cold starts finish5000updates. Same architecture/optimizer/dtype/
compilation, grid/reference/random draws; all initial parameter leaf files are
byte-equal, including the prior pilot. Within each representation stage, initial
physical metrics agree exactly and configs differ only mode/run_name. Across
representations, initial positive-time physical outputs differ as an explicit
consequence of the map; do not claim identical physical initialization there.
No time windows, sampling/physics/reference change or additional evolved prior.

All following field/interface errors use the same refined LOCAL reference at t2.
Interface band means |reference phi|<0.9; it measures phase values, not positions.
Width proxy is transition area(|phi|<0.9)/zero-contour perimeter, reference0.109179.
It is resolution-dependent on64x64, not exact normal thickness. Distances use
linearly interpolated zero contours resampled at h/8 and periodic nearest points;
self/shifted-circle/tanh-band checks pass. Missing contours are explicit flags/nulls,
not invented zero errors; all6 actual final contours exist.

| Representation / residual | phi L2 | band phi L2 | width-proxy relative error | mean contour distance | max soft area drift |
|---|---:|---:|---:|---:|---:|
| additive/original | 0.07997183 | 0.30326958 | 0.484013 | 0.00480401 | 0.027438 |
| additive/transport | 0.21940974 | 0.75188035 | 1.596694 | 0.03674135 | 0.156511 |
| additive/both | 0.27170811 | 0.93427922 | 4.520600 | 0.08853536 | 0.248324 |
| leaky_tanh/original | 0.04192344 | 0.16044946 | 0.085725 | 0.00279750 | 0.042872 |
| leaky_tanh/map_transport | 0.04872216 | 0.20848897 | 0.203206 | 0.00482749 | 0.018227 |
| leaky_tanh/map_both | 0.04019318 | 0.17455449 | 0.177675 | 0.00447810 | 0.020714 |

Output representation is the clearest improvement: mapped original reduces
common-local terminal phi0.07997->0.04192, band0.30327->0.16045, width-proxy
error48.4%->8.57%, mean contour distance0.004804->0.002797. It has WORSE maximum
soft area drift2.74%->4.29%; not an all-metric winner. The mapped both-residual
version slightly improves global phi to0.04019 and q RMS0.06931->0.05488 and
area drift to2.07%, but worsens band error to0.17455, width-proxy error17.77%
and contour distance0.004478. Mapped transport only is intermediate/worse in
field metrics, with smaller area drift1.82%. Sharper-interface candidate is the
mapped original-residual FINAL5000 model; all outcomes remain in the report.
No best-checkpoint/seed search or strong claim from small L2 differences.
The prior ordinary-FP32 raw replay0.07889 vs fresh0.07997 illustrates path
variation; no statistical robustness is established for any candidate.

Physical-field bounded transport/both weights significantly worsen fields and
widths. Their final common-local L2 is0.21941/0.27171. Both weighting allocates
83.9% of pooled transport squared residual to reference bulk, versus24.1% raw,
supporting bulk emphasis rather than better interface resolution. This is not
proof of a unique failure mechanism. Inverse-map both also raises bulk shares
but improves raw pooled phase/q RMS0.13914/0.04708->0.13034/0.03596 relative to
mapped original; raw bulk improves while interface transport RMS worsens
0.35028->0.41106. A single aggregate residual/L2 does not rank all physical fields.

## Verification and changed files

12focused tests pass41.840s, including legacy7, bounded weight/zero/gradient
checks and inverse-map Newton/implicit derivative/network IC/time chain/full
parameter-gradient checks. All6 independent CPU checkpoint/field/metric replays
pass the existing explicit FP32 tolerance5e-6. The current network also reloads
the earlier additive model with max phase difference3.21865e-6; static fields
add no parameter leaves. Same physical reference identity all6, complete histories,
initial bytes and each stage's frozen numerical source are verified. Reference
remains the already-converged independent vortex solution; no expensive unchanged
reference regeneration. Old external paths/checkpoints are preserved.

Diff-level changes:
- losses.py exposes original residuals, adds bounded inverse-J weight and exact
  inverse-map modes with full weight AD; mass loss is unchanged.
- network.py adds the unbounded leaky-tanh optional map, checked fixed Newton
  inverse and pure phase_jacobian; original schema remains additive.
- problem.py validates explicit residual/map extension fields and inverse range.
- Two explicit configs enter through unchanged train.py; no established default
  or physical model changes. Added5 numerical checks in test_twophase_residual.py.
- Example README/development explain formulations, limits, modes and replay;
  artifact index registers6 new runs. PROGRESS/DECISIONS/task record updated.

## Reproduction and direct paths

Stage1: archived residual-launch.py runs original/transport/both with
configs/train_vortex_residual.py; reports/twophase-vortex-residual holds raw logs,
per-model CPU figures/raw physics/geometry and frozen bounded-source.
Stage2: archived latent-launch.py runs original/map_transport/map_both with
configs/train_vortex_latent.py; reports/twophase-vortex-latent holds raw results.
Each is5000updates and uses fresh run directories under the existing log root:
/root/autodl-tmp/tf-logs/twophase2d/20261026-vortex-residual-{original,transport,both}
and /root/autodl-tmp/tf-logs/twophase2d/20261026-vortex-latent-{original,map_transport,map_both}.
Each stores config/source/initial/final/checkpoints/loss/metrics/phi/q/summary.
No hashes, new seed setting, checkpoint import/resume or training on reference.

Analyze using the archived residual-analyze.py --report-dir=reports/<stage> with
JAX_PLATFORMS=cpu. Then run residual-verify.py and residual-summarize.py. Numerical
sources remain in each run; the bounded stage snapshot predates the map extension
and is explicitly checked against that stage, not silently compared to new code.

[Final summary](twophase-vortex-residual-summary.json),
[verification](twophase-vortex-residual-verification.json),
[bounded audit](twophase-vortex-residual-bounded-analysis.json),
[mapped audit](twophase-vortex-residual-latent-analysis.json),
[before/after fields](twophase-vortex-residual-representation-fields.png),
[combined curves/profiles](twophase-vortex-residual-combined-curves-profiles.png),
[all mapped fields](twophase-vortex-residual-latent-fields.png),
[negative bounded fields](twophase-vortex-residual-bounded-fields.png).

Status: complete; no blocker/next action required. All workers exited0, GPU idle.
No Git history operation this turn; previous uncommitted work is preserved.
Possible future work is mass-constrained phase representation/time causality and
cross-seed validation, rather than assuming inverse-J scaling always sharpens CH.
Memory changed: PROGRESS.md, DECISIONS.md, DEVELOPMENT.md and this archived task.
Final completion checks are in the adjacent validation JSON.
