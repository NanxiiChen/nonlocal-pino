# Electropolish online operator AC/CH balance diagnosis — complete

Date: 2026-10-28. Status: COMPLETE. Accuracy unresolved. Blocker: none.
All diagnostic workers exited; no training/default/model/reference/Git change.

## Goal and acceptance

Diagnose whether AC or CH dominates the completed continuously sampled B8/5000
Adam operator training, including raw/scaled/weighted losses, parameter gradient
norms/conflicts, IC/BC competition, and concentration/transport versus reference.
Use fixed saved checkpoint and recorded inputs; no training/default/physics or
reference changes. Report measured evidence separately from causal hypotheses.

## Context

Run: /root/autodl-tmp/tf-logs/electropolish2d/20261028-electro-operator-online-b8-adam-01.
Final checkpoint model_final.eqx; initial model_initial.eqx; loss.jsonl stores
five scaled parts and weights, but no gradient cosine history. Fresh continuous
six-mode profiles, 40000 actual inputs; evaluation reference bank separate.
Five losses: AC/CH scaled L1, top/bottom/IC physical MSE. Joint inverse-L1-gradient
balance every25updates, not staggered or relative-rate original GradNorm.
Heldout16 phi13.159% pooled/10.995% terminal; model smooths roughness too fast.
Prior completed work: [fixed/online controls](electropolish-operator-single-online.md).

## Completed diagnostic protocol

Read saved-source/current parity and adaptive update. Analyze all loss records.
Replay initial/final on fixed predeclared actual online batches and inspect
per-loss parameter gradients/weighted influence/cosines and cancellation.
Use independent NumPy/SciPy reference comparisons for concentration/transport,
interface roughness and physical equation terms. Save reproducible script,
raw JSON and diagnostic plots; validate focused arithmetic/finite differences
where needed; diff check; archive task and update PROGRESS/DECISIONS.

## Changes and results

Read saved/current source parity and verified actual sampler coefficients,
coordinates, times, initial objective and adaptive weights. All six source files
match saved run. Main GPU diagnostic exits0 in237.3s. Initial losses/weights pass
rtol5e-5; scalar gradient versus extracted weighted sum relative gap5.32e-5
(FP32 threshold2e-4). Four final GPU fields pass2e-5pointwise replay gate.

Last1000 logged AC/CH weighted loss ratio median.46747, min.11915/max2.37555.
Final AC/CH cosine.02503/.01740/.11927/.01038; fresh balanced gradient L2
AC.9695/.8770/1.0881/1.2333, CH1.3841/1.6453/1.2891/1.5172.
No orders-of-magnitude imbalance/strong negative conflict on these probes.
Both BC weights cap1000, but BC gradient shares remain small. Reusing final
weights across four batches gives IC share7.05..8.39%; fresh balance22.58..27.51%.
Saved weights last refreshed update4976 and are not historical weights for the
other batches. These are Euclidean diagnostics, not actual Adam directions.

All four initial phiL2 1.17..1.63%; at t*.25 predicted roughness.00195..00208
versus reference.01731..04802. At t*.05 phaseL2 jumps17.54..32.12% while roughness
collapses. Heldout16 terminal qL2.03414/DqL2.80704; across four DqL2.529..1.096.
Independent SciPy physical terms/residuals and full-grid timeJVP saved in NPZ.
Plots of logs/gradients/roughness/transport and q fields created. Dense early-time
residual/ensemble-condition-spread probe completed below; no training.

## Early transient and condition response

The original four actual time batches all have t*>.27. Their gradient results
cannot exclude near-initial-time conflicts. Additional frozen full-grid JVP
and independently assembled SciPy graph diagnostics resolve early times for
unseen16. Integrate spatial mean absolute physical residual, excluding boundary
nodes. Logarithmic0..05 and uniform.05..1 nodes refined41->81 per interval;
integrals change<=.508%, passing the predeclared3% gate.

|Residual|Integral0..05|Integral.05..1|Early fraction|Early/late time-average ratio|
|---|---:|---:|---:|---:|
|AC|.048708|.049086|49.81%|18.85|
|CH|.049210|.017305|73.98%|54.03|

AC absolute residual peaks4.691 at t*.00481, CH4.744 at.00535. First5% of time
contains most CH inconsistency. These are diagnostics, not new training losses.
No temporal finite difference of sparse reference snapshots is used.

Across four unseen inputs16..19, define field spread as the Frobenius norm
after subtracting the four-field ensemble mean. Predicted/reference phase
spread:102.93%at0,6.03%at.05,.943%at.25,.447%at1. Concentration spread is1.077%
at.25. This measures retention of input differences on four samples, not
relativeL2 prediction accuracy or a population generalization estimate.
Interface height is linearly interpolated at the unique phi=.5 crossing;
exactly one vertical crossing is checked for every time/column.

For displayed16, terminal total cL2=12.682%, qL2=3.414%, DqL2=80.704%.
q=c-(cSe-cLe)h(phi)-cLe contains a substantial bulk component; small global q
error does not guarantee transport accuracy. Dq error combines concentration
and interface-position error; this does not isolate concentration as a cause.

## Supplemental early gradient probe

Keep actual batch0 coefficients/queries and use NEW explicit times
(.001,.003,.005,.02). These are final-model diagnostic gradients, never
presented as historical training gradients. AC/CH losses3.8894/4.1722.

|Gradient pair|Cosine|
|---|---:|
|AC / CH|+.92657|
|AC / IC|-.59619|
|CH / IC|-.52807|

Early PDE gradients align with one another and conflict with IC improvement.
Saved-final weights give weighted gradient L2 AC9.582/CH17.995/IC.202;
IC contributes.70% of the sum of norms and has cosine-.55551 with total gradient.
Fresh balancing gives AC12.126/CH13.169/IC9.763; IC hits1000cap, contributes27.06%,
and still has cosine-.21081 with the combined gradient. Global gradient clipping
preserves direction; Adam moments/state are unavailable. An infinitesimal plain
negative-total-gradient step would increase IC loss on this probe, but no claim
is made about actual historical Adam directions or isolated causality.

## Interpretation, limits and future controlled work

Observed failure: fit initial geometry (~1..2% phaseL2), rapidly forget its
differences, and follow nearly one flat trajectory while missing early PDE
dynamics and concentration transport. Last1000 scalar AC/CH ratio and four
late gradient probes do not show persistent AC suppression of CH. Near the
missed transient, both PDE gradients align and compete with IC. Therefore
AC/CH staggering alone is not an evidence-backed remedy.

Whole-domain scalar loss and gradient balancing do not guarantee interface/
transport accuracy or correct temporal propagation of an initial condition.
Uniform four-time sampling does not prioritize the narrow early residual peak.
Sampling, conditional representation and optimization are not causally separated.
No new staggered/weighting/architecture control was trained or claimed better.

If optimization is resumed, first test one general early-time sampling change
with declared density/objective or correct importance weighting, same model/
budget, monitoring retained input dependence and Dq alongside physicalL2.
Condition injection through temporal decoder layers is a separate candidate.
Hard IC alone would not prove correct evolution. No future reference trajectory
or interface-motion prior is proposed. No next experiment is queued.

## Artifacts, validation and session end

Reproducible scripts (all diagnostics, never parameter updates):
[main](electro-online-balance-diagnose.py),
[dense early times/spread](electro-online-balance-early.py),
[early gradients](electro-online-balance-early-gradient.py),
[plots](electro-online-balance-plots.py).
Corresponding JSON/stdout files share the same stems (main JSON is
electro-online-balance-diagnosis.json); full physical terms in
electro-online-balance-fields.npz. Source comparison uses bytes, not hashes.

Figures: [overview](electro-online-balance-summary.png),
[four interface responses](electro-online-balance-condition-collapse.png),
[early residual/spread](electro-online-balance-early.png),
[early gradient conflict](electro-online-balance-early-gradient.png),
[q fields](electro-online-balance-q-fields.png).

Six source files match saved run. Actual coefficients match recorded arrays.
Initial losses/adaptive weights reproduce rtol5e-5/atol2e-6; independently
weighted scalar gradient versus extracted sum gap5.32e-5 passes2e-4 FP32 gate.
All four final GPU fields pass2e-5pointwise replay gate. Dense quadrature
refinement passes. Artifact syntax/JSON/numerical consistency/links/figures and
git diff checks pass; no numerical-core changes justify old expensive suites.

Changed: scripts/raw diagnostics/figures; example README validation link;
artifacts/index.yaml; docs/PROGRESS.md, append-only DECISIONS.md and this task.
Existing uncommitted work preserved. All workers exited; no training/model/
PDE/reference/default/optimizer implementation or Git history change.
Exact resume: none required; a future accuracy experiment needs user direction.
