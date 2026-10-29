# Electropolish conditional training and initial-distance audit

Date2026-10-28. Status COMPLETE; blocker none. No active jobs or further sweeps.

## User goal and acceptance

User expects PINO to fit the same physical equations as a successful PINN and
does not accept layered time sampling as the presumed remedy. Diagnose actual
training/conditional representation and explain exactly how the IC distance is
defined. No sampling/physical/reference/default change or Git operation.
Quantify initial-distance approximation, image/local-feature dependence and
per-condition gradient cancellation. Compare one/two/many fixed inputs using
available independently checked references, retaining all failed controls.

## Established context

Current SOTA single-IVP PINN is SOAP on its prescribed rough interface. Operator
single input0/Adam2000/uniform Nt4 already fits phi2.483%pooled/3.387%terminal;
SOAPsingle4.187%/3.214%. Finite16/B2/2000 and onlinecontinuousB8/5000 collapse
input dependence after t0. Matched causal-only5000 did not improve global L2.
Those facts do not isolate sampling as a cause. See archived operator-single-
online, operator-balance and operator-causal records.

Initial phi=.5*(1+tanh(k*d)), k=sqrt(omega/(2alpha))*Lc; c0=cSe*h(phi).
Both legacy NumPy and online JAX use2000-piecewise-linear-curve vertices and
33candidate segments around query x. Sign from below/above y=f(x); magnitude
is nearest Euclidean segment distance WITHIN that finite window, not global
exact SDF and not simply vertical gap. Top/bottom nodal overrides match refs.

## Predeclared protocol

1. Independent exhaustive segment-distance comparison on grid for source0,1,
   heldouts16..19 and first four actual online inputs. Report normal-interface
   phi/distance errors versus the current definition; do not change reference.
2. Frozen online-final and fitted-single checkpoints: swap condition images
   independently from known local initial features for pairs0/1 and16/17,
   quantify time-dependent physical response against true input differences.
3. Online-final actual batch0: separate AC/CH/IC parameter gradients for each
   of8conditions at recorded times and early(.001,.003,.005,.02). Quantify
   norm-of-mean versus mean-of-norm, cross-condition cosine and head-channel
   gradients. These are frozen diagnostics, not historical optimizer updates.
4. New two-fixed-input(0,1)/B2/2000Adam cold start, same model/init/physical
   losses/NT4uniformtime/core256/pool32/IC1024/LR.001/FP32highest/ordinary/5s.
   Compare available one-input and16-input2000 controls on the SAME inputs;
   differing B/exposure/work limitations explicit. No new supervised labels.

## Distance and fixed-count controls

Distance audit completes10profiles with exhaustive1999segment search. Six bank
ICphi relative differences from true global polyline distance.14..1.28%; vertical
distance differences1.01..4.15%. Current window field matches stored initial
arrays<=3e-7; no reference/IC change. k58.45918,10-90normalwidth.0375856.

Canonical two-fixed-input control exits0 in220.2s, source0phi12.547%pooled/
7.240%terminal, source1 6.392%/4.600%; meantrain9.470%/5.920%. Uniform Nt4,
2000Adam, same initialization as earlier1/16input controls. Single0phi2.483%/
3.387%; prior16input same0 21.538%/15.888%. Count/exposure/B differences remain.

Frozen online per-condition gradient probe completed. Batch0 recorded-time AC/CH
norm-of-mean divided by mean-of-norm .9851/.9861; average pair cosines.9658/.9697.
Early times retention.8576/.8928, cos.7066/.7782. Thus no strong cancellation
on these probes, including early times. Image/local-feature swaps and GPU audits
completed below. No runtime change was made at this stage; the subsequent
single-option ablation is described next. Scripts/configs use electro-training-*.

## Supplemental trained shortcut ablation

Frozen probes finish: online16/17 image-only phase response .00506%at0/.02760%
at.25 of true input difference; local-feature-only104.38%/1.152%. Pair-trained
0/1 image-only .197%/.224% versus local96.82%/2.280%. Encoder parameter gradients
are nonzero (~.6%of full gradient L2 at late times); no frozen-AD bug found.
This suggests a local-IC shortcut, but frozen sensitivity is not causal proof.

Add explicit optional arch.local_input=zero, zeroing only query feature channels
inside loss and predict_fields, preserving image, true IC targets, physical
output map and all parameter leaves. Defaultinitial and old missing-field
configs preserve legacy behavior. Focused supervision-preservation/inference
test passes29.03s; legacy-mode equality check passes10.89s. No SDF/time/physics/
reference/default-strategy change. Same two-input2000 uniform-time cold start
for zero-local mode completed at run pair-global-adam-01; log electro-training-
pair-global-training.txt. Actual inputs/init/mode delta and fields checked.

Zero-local two-input control exits0 in254.04s. Mean trainphi2.601%pooled/
2.704%terminal versus9.470%/5.920% with local input; heldout24.375%/15.808%
still poor after two inputs. Supervision-preservation and legacy-mode equality
tests pass29.03s/10.89s. Pair-global checkpoint/input/mode audit passes. One
matched onlineB8/5000 zero-local validation completed on the same40000profile
stream, uniform times and original losses, separating fitting from operator
generalization. No causal/strata/physics/default change.

## Matched feature-ablation results

All run paths share /root/autodl-tmp/tf-logs/electropolish2d:

-20261028-electro-operator-pair-adam-01: local input,2fixed cases,2000updates.
-20261028-electro-operator-pair-global-adam-01: zero-local,2cases,2000updates.
-20261028-electro-operator-online-global-adam-01: zero-local,onlineB8,5000updates.

Pair controls use identical sample order (each update includes0and1), identical
initial arrays and all configs except local_input/run name after explicitly
interpreting the missing historical field as initial. Same fitted input0:

|Training|phi pooled|phi terminal|
|---|---:|---:|
|Existing single-input|2.483%|3.387%|
|Two inputs, local features|12.547%|7.240%|
|Two inputs, image only|3.057%|3.433%|
|Existing16inputs|21.538%|15.888%|

Single versus two versus16 also differs in B/exposure/objective; do not claim an
isolated sample-count effect. The two-input feature ablation itself holds these
factors constant. Mean two-fit phi9.470/5.920% ->2.601/2.704%. Image-only pair
response to image0/1 swap recovers98.744%of true difference at0 and92.456%at.25;
local-feature response is exactly0. Training before ablation had image response
.197%/.224%. Frozen decoder-side initial copying is a plausible shortcut; the
matched trained ablation supports it as a major contributor under this scheme.

Online control holds the complete continuous stream and5000budget fixed:

|Four-unseen relative L2|Original local-query input|Image only|
|---|---:|---:|
|phi pooled|17.7477%|5.5733%|
|phi terminal|12.3551%|4.3718%|
|c pooled|19.3173%|6.4519%|
|c terminal|14.0295%|5.1241%|

315.832s including compile/evaluation, no causal/layered sampling/hard IC.
Same73initial arrays, all40000actual profiles/accepted counts and sampling
scheme; config differences exactly local_input and run_name after documented
legacy-schema interpretation. No test-based model selection (final5000).

Unseen16 phi1.834%pooled/1.690%terminal, q1.215% and Dq17.788%terminal versus
original q3.414%/Dq80.704%. At.25 its roughness.017229 versus ref.017310 (old.001997).
Across four heldouts, phase variation recovers80.358%of reference at.25 versus
old.943%; at1 it retains79.810%. Unseen18 is still difficult:10.436%pooled/
8.881%terminal phi, Dq53.61%. Other terminal Dq25.70%/32.78%. Initial phi errors
across four1.61..5.16%. Improved conditional evolution is not full accuracy or
broader-distribution/seed stability proof; one matched online pair is measured.

## Initial-field explanation and reference boundary

Random six-mode graph y=f(x),2000vertices. A query's signed distance magnitude
is minimum Euclidean distance to33nearby line segments, with projections
clipped to segment endpoints. Sign from y<f(x) (solid positive), opposite in
liquid; the vertical gap is not the magnitude. phi=(1+tanh(k*d_window))/2,
c0=cSe*(3phi^2-2phi^3), k58.45918. Top/bottom nodes are prescribed0/1. IC images,
query-local values, true IC supervision and current reference share this rule.

Exhaustive1999segment comparison reports bank phiL2 difference.143..1.275% from
the current field, pointwise max up to.08196. These are IC-definition differences,
not trained solution errors. A vertical-gap construction differs1.015..4.150%.
Both differ from the existing shared initial definition. It is appropriate to
call current data windowed signed-distance approximation, not exact global SDF.
Only docstrings/docs were clarified; no IC/reference numerical change.

## What changed and verification

Case-local `_query_initial` optionally zeros only feature channels in BOTH
make_loss_components and predict_fields. True IC labels remain original known
values. `arch.local_input=initial` is explicit in current operator config;
old saved configs without this optional field use legacy behavior. Network
parameters/templates and input tensor shapes remain identical. No frozen
learned preparation outside parameter AD or missing encoder gradient was found.
Two meaningful focused tests protect unchanged IC supervision, inference and
legacy mode. Defaultsingle-IVP/causal/original presets retained; reproduce the
best measured operator through train_operator_online.py with
--config.arch.local_input=zero. No automatic default promotion or history change.

Checks:
- NumPy exhaustive-segment IC audit10profiles, current fields match six stored
  initial conditions<=3e-7; distance monotonicity checked.
- Frozen gradient matrices/parameter groups finite; separately computed B1
  gradients average to original B8 gradient norms within.0346%AC/.2430%CH/
  .0000322%IC, below declared1% FP32 batching gate. No cancellation conclusion
  is extrapolated outside inspected batch/times.
- Both new pair runs replay all six fields<=2e-5;73initial arrays exactly match
  single and16case controls; actual inputs0/1/optimizer/configs audited.
- Online: exact73arrays/40000profiles/accepted counts; unchanged numerical/model/
  optimizer source bytes; generator executable AST equal except clarified
  docstrings. SixGPUfield replays max2.33e-6, targets exactly match refs. NumPy
  physical errors match logged means, SciPy independently computes Dq.
- Two focused tests pass29.03/10.89s. Final syntax/strict JSON, finite arrays,
  source/links/figures/archive/index and git diff checks pass. No repeated
  expensive unchanged suite, no FEniCS process or shared numerical-core change.

## Artifacts and exact session end

Scripts/raw JSON/stdout use electro-training-*:
[distance](electro-training-distance.py), [conditioning](electro-training-conditioning.py),
[pair audit](electro-training-pair-audit.py) (explicit --local-input=initial|zero),
[online audit](electro-training-online-audit.py), archived two-pair config files,
[fixed-input plots](electro-training-plots.py),
[online plots](electro-training-online-plots.py).
Field targets/checkpoints/actual streams remain at direct external paths.
Resolved config/summary/loss/metrics/selection/memory copies have full run prefixes.

Figures: [same input0 learning](electro-training-input-count.png),
[condition sensitivity](electro-training-conditioning.png),
[initial distance definitions](electro-training-distance.png),
[online convergence](electro-training-online-convergence.png),
[online physical phase](electro-training-online-fields.png),
[retained input dependence](electro-training-online-response.png).

Changes: optional local-input feature mode/validation/default metadata;
two focused tests; IC approximation docstrings; example README/DEVELOPMENT;
artifacts/index.yaml; PROGRESS/DECISIONS and this archived task; diagnostics,
three authorized fresh runs/configs/figures. Existing uncommitted work retained.
No time sampling, PDE, IC numerical definition, reference, optimizer algorithm,
shared raw network, old artifact/default or Git history change. All workers/GPU
idle; exact resume none required. Further accuracy/stability work is not queued.
