# Per-term adaptive gradient balancing for the stronger channel

## Goal and acceptance

After local checkpoint dev@b2dff4b, user authorizes continuing modifications to
replace hand-set loss coefficients by adaptive gradient-norm balancing. Extend
the repository's stopped inverse-L1-gradient/EMA method to all nine soft terms
and exact full-time40 gradients with eight-time memory blocks. This is not the
learned-weight objective of the original GradNorm paper. Retain physical output
scales, model, physics, GMLS/AD/time-JVP, geometry sampling and independent FEM.

Acceptance: legacy behavior preserved; new explicit termwise config with all
residual divisors1; complete mean-gradient/weight/SOAP and short actual training
tests; bounded production resource check; one finite cold5000 versus the saved
stronger control, with field/shape, time/input/cold-state and runtime checks.
No automatic retries/later queue, new reference, default/SOTA or Git action.

## Context

- Control: /root/autodl-tmp/tf-logs/channel_nsch2d/20261102-5090-gmls-time40-jvp-stronger-online-width128-01.
- Existing config: examples/channel_nsch2d/configs/train_operator_width128_gmls_time40_jvp_stronger_online.py.
- Current balance.py has four grouped weights, one VJP tape, stopped inverse-L1
  ratios, cap1000/epsilon1e-6/EMAnewfraction.9, refresh100. Current time blocks
  reject balance. Norms must be taken AFTER averaging each term's full40-time
  parameter gradient, not averaged over local block norms.
- Successful reference/control and failed older balancing studies are retained;
  new balance is an experiment, not presumed accuracy improvement.

## Status and outcome

COMPLETE. Seven focused tests, the finite102-update probe, cold5000 and all six
owned saved-run stages completed/exit0. No blocker, required resume work, later
experiment or new FEM solve. Do not promote a default/SOTA or start further
tuning automatically. The new scheme is usable as an opt-in research control,
but does not beat the existing fixed-scale recommendation at this budget.

| Relative error / diagnostic | Fixed scales | Raw + nine adaptive weights |
|---|---:|---:|
| Pooled phi L2 |1.9470%|2.8770%|
| Terminal phi L2 |1.6625%|2.6360%|
| Velocity L2 |3.0587%|5.1258%|
| Pressure L2 |3.9455%|8.2671%|
| Chemical-potential L2 |19.5989%|38.9275%|
| Terminal minority-field L2 |4.8688%|7.6886%|
| Worst saved minority-field L2 |8.7984%|13.3838%|
| Terminal contour distance / R0 |2.4662%|3.5241%|
| Terminal droplet IoU |93.0197%|91.6529%|

At peak flattening t4, x-span/y-span is1.51677 versus FEM1.29955(~16.7%bias),
while fixed-scale prediction1.40958 has~8.5%bias. The t4-inclusive comparison
was visually inspected; do not infer perfect shape accuracy from pooledphi2.9%.
One initialization/reused nominal development geometry only; no population or
cross-seed conclusion and no general claim about every GradNorm variant.

## Verification and interpretation

- Seven tests PASS136.435s: analytic gradient cancellation across time blocks,
  exact whole40-time nine-part gradients/weights, independent frozen-weight FD,
  physical-primal/oneSOAP parity, legacy balance and actual3step cold training/
  replay/all-input/weight-history checks. No unchanged expensive suite repeated.
-102-update production probe EXIT0, all measured gradients/weights finite;
  cold464.669s, held.288800s, warmrefresh2.794011s, meanwarm.313604s.
  Fitted probe state discarded. Its fixed-input peak includes compile/autotune.
- Actual production cold model AND complete SOAP bytes equal the control;
  initial/final input arrays and all actual time/condition/online draws match.
  All50refreshes, stopped EMA/clipped weights and logged weighted objectives
  independently replay;90saved Python source byte checks pass.
- All six owned stages exit0: training, validate_run, diagnose_phase,
  droplet_shape, phi_gallery, continuous_time_residual. The latter uses240fresh
  continuous times on full nominal and final-pool grids; these are canonical
  diagnostics, not a continuum or exact GMLS conservation certificate.
- Independent nominal240-time mean physical RMS: CH .006579 ->.011175 and
  chemical relation .028952 ->.038399. Wallmu .018607 ->.000380 and outletmu
  .015410 ->.001548. Boundaries improve while main equations/fields are less
  accurate. Gradient balancing is operating, but equal parameter-gradient norms
  are not equivalent to the most useful physical-error allocation at5000steps.
  This is evidence of a tradeoff, not proof of a universal causal mechanism.
- Training wall2165.832s(~36.1min), warm.315596s versus1926.685s/.288946s;
  mean warm overhead~9.2%. Recorded peak22583938304bytes(~21.0GiB) includes
  compilation/autotuning.32GiB5090 fits; no stronger GPU requested.
- Final9changed-Python AST,50configuration imports/serialization, strictJSON,
  task/artifact links, text/size/whitespace and gitdiff--check pass. All owned
  processes exited and GPU is free. No history change after dev@b2dff4b.

## Direct artifacts and regeneration

- Config: examples/channel_nsch2d/configs/train_operator_width128_gmls_time40_jvp_allterms_online.py.
- Run: /root/autodl-tmp/tf-logs/channel_nsch2d/20261102-5090-gmls-time40-jvp-allterms-online-width128-01.
- Journal: /root/autodl-tmp/data/channel_nsch2d/20261102-5090-gmls-time40-jvp-allterms-online-width128-01-process.json.
- Reference/screen root: /root/autodl-tmp/data/channel_nsch2d/stronger-reference-20261102-01.
- Resource result: /root/autodl-tmp/data/channel_nsch2d/allterms-resource-20261102-01/result.json.
- Comparison/figures: /root/autodl-tmp/data/channel_nsch2d/allterms-comparison-20261102-01.
- Regenerate plots via docs/tasks/channel_allterms_plot.py with explicit
  --control/--trial/--output; requires completed matched-control proof.
- [Compact receipt](../channel-allterms-gradnorm-validation.json).

## Task-end changes

Extended example balance/grouping, validation and canonicaltrain dispatch;
added explicit raw-adaptive preset, strict saved-run comparison, focused tests,
bounded probe/owner integration and scientific plotting helper. Updated example
README/development guidance and artifact index without changing current_run,
defaults or existing checkpoints. Updated PROGRESS/DECISIONS/this task and
archived completion. All weights/figures/arrays remain outside Git. No Git
operation or local artifact cleanup performed in this modification turn.
