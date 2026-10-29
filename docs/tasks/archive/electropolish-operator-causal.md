# Electropolish online causal control

Date2026-10-28. Status COMPLETE; blocker none. Accuracy remains unresolved;
no training/sweep queued, workers idle.

## Goal and acceptance

User explicitly requests existing causal training to address the missed early
transient/input forgetting. Reuse shared CausalWeightor in the canonical operator
entry; keep old noncausal behavior/loaders; add explicit opt-in preset, focused
formula/AD/core-mask/config checks, full matched B8/5000 cold start and independent
L2/transport/condition/field replay. Report success or failure without changing
physics/reference/architecture/defaults or hiding raw losses. No Git authorization.

## Context

Shared src/nonlocal_pinn/training/causal.py implements sorted per-time or chunks,
stopped exp(-eps*cumsum(previous per-time MAE)), mean(w*loss), adaptive eps.
Legacy electropolish uses shared min(epsAC,epsCH), separate AC/CH weights and
updates both eps. Current operator rejects causal. Frozen diagnosis archived in
electropolish-operator-balance.md: early PDE/IC competition, strong loss
of input differences, large Dq error; not AC against CH.

Baseline /root/autodl-tmp/tf-logs/electropolish2d/20261028-electro-operator-online-b8-adam-01:
Adam5000/B8/LR.001/Nt4/uniformtime/core256/pool32/IC1024/FP32highest/ordinary/
5s/seed42runtime; 40000 unique fresh profiles. Four testphi17.748%/12.355%,
c19.317%/14.029% pooled/terminal; original five losses inverse-L1 balance25/cap1000.

## Predeclared first control

Only add causal weighting and its adaptive state; preserve baseline condition,
spatial/time RNG and tensor split. Per-condition causal AC/CH applied BEFORE
batch mean; normalize core segments by actual free-core count, not padded halo
size. Reuse compute_weights/update_eps without changing shared module. All
IC/BC losses remain active. Dynamic eps enters JIT, aux contains per-condition
segments/weights; no parameter AD through weights. Log unweighted and weighted
parts, eps and weights. Mean condition weights drive shared adaptive eps updates,
matching legacy single-case policy at B1. Existing Nt4 uniform-time first control
does not guarantee an early sample each step; keep that limitation visible.

## Implementation and validation

Implemented opt-in causal config, five-part loss components/aux-aware causal
step and early config validation; noncausal uses the unchanged shared step.
Independent FP64 NumPy/core/time-order/condition/stopped-gradient formula check,
causal-step versus original balancing/optimizer parity, original noncausal
physical residual/directional-gradient check all pass. Config rejection check
passes separately(.022s). Raw test logs preserve two harness-only failures:
initial comparator tried subtracting static functions; second invalid-config
fixture was rejected by ConfigDict before the validator. Fixtures corrected;
unchanged expensive numerical checks not rerun. No numerical failure concealed.

Canonical causal B8/5000 cold start launched:
/root/autodl-tmp/tf-logs/electropolish2d/20261028-electro-operator-online-b8-causal-01.
Stdout: [training](electro-causal-training.txt). Run exits0 after5000updates,
323.976s including compile/evaluation. Final fixed-budget checkpoint, no test
selection. No positive-time reference fields enter optimizer inputs.

## Reused rule and exact behavior

Use `CausalWeightor.compute_weights` and `update_eps` without changing shared
code. Operator components form per-condition AC/CH per-time MAE, sorted by time,
normalized by actual free-core count. Weights are exp(-min(epsAC,epsCH)*prefix
MAE), first weight1, stopped in AD; loss is unnormalized mean(w*MAE) then mean
across conditions. No floor or normalization by sum(w) is added. All five losses
retain original inverse-L1 parameter-gradient balancing every25/cap1000/alpha1.
IC/BC do not enter causal prefixes and remain active in the joint gradient.

Dynamic two-component eps starts(1,1); multiplier1.1/max100/last threshold.9/
mean threshold.01. Batch-mean AC and CH weights drive their own eps updates,
shared minimum drives both equations. `has_aux=True` VJP returns diagnostics
from the same PRE-update forward; logged segments/weights and eps_used are
consistent with the losses used for that step, not recomputed after updating.
`unweighted_parts` retain physical residuals to prevent weighted loss hiding
unlearned later times. Noncausal payload/arithmetic/shared step are retained.

## Matched fixed-budget results

|Mean four-unseen physical relative L2|Baseline|Causal|
|---|---:|---:|
|phi pooled|.17747656|.17931640|
|phi terminal|.12355057|.13105258|
|c pooled|.19317300|.19520598|
|c terminal|.14029465|.14801502|

No measured accuracy improvement; no SOTA/default promotion. This is one
matched pair, not a stable optimizer/method ordering or repeatability study.
Same default runtime seed42 is provenance, never a selected strategy knob.

Unseen16 phi pooled/terminal13.866%/12.108%; baseline13.159%/10.995%.
Terminal q/Dq baseline3.414%/80.704% -> causal6.253%/88.266%. Its roughness at
t*.25 is.002123 versus reference.017310; baseline.001997. Other unseen Dq
errors57.83%,112.29%,89.55% likewise remain large. Across four unseen inputs,
predicted/reference phase spread at t*.25 increases.943%->1.540%, but remains
nearly collapsed; at t*.05 spread6.03%->18.04%. Early input response improves
somewhat without translating to improved pooled/terminal physical L2 here.

## What this control establishes and misses

Causal weighting is active and correctly connected: late-time last weights in
last1000logs average.648AC/.722CH. Final epsAC1.61051/epsCH100, effective shared
eps1.61051. There is no persistent residual history indexed by physical time;
the reused per-t rule only sorts the CURRENT batch's four random times.
Of51logged batches84.31%contain no time<.05, median earliest.19319. Under the
declared uniform[t_min,1) sampler, first5%time is absent with probability
((1-.05)/(1-.001))^4~81.8%. This explains a limitation of the current causal
input coverage, not proof of a unique cause for final errors or of general
causal ineffectiveness. No sampling change is silently bundled into this run.

If further optimization is requested, a predeclared early-covering temporal
sampling control is the next candidate. Declare its sampling density/objective
or correct importance weights; compare it separately, retaining pure causal
and original controls. Hard IC, new network, stagger or a future interface
trajectory are not tested or proposed as already validated remedies.

## Independent replay and checks

[Audit script](electro-causal-audit.py) and [JSON](electro-causal-audit.json):

- Config differences exactly loss.use_causal and log.run_name;73identical initial
  parameter arrays, all40000actual coefficient vectors and accepted counts exact.
- Shared model/operator/optimizer/residual/generator numerical sources byte-
  identical. Online config source gained only an explanatory reference-bank-only
  comment since baseline; AST equality verified and byte difference retained.
  Physical output-map AST unchanged; shared causal.py unchanged.
- All51records independently match NumPy causal prefix weights, weighted and
  unweighted losses, eps updates and original random time draws. Weights sorted
  nonincreasing with first1; no positive-time reference labels in loss payload.
- Six final GPU model evaluations agree with saved fields max6.26e-7, below
  2e-5 gate; targets match references exactly. Independent NumPy field errors
  reproduce logged four-case means; SciPy graph reconstructs free-node Dq.
- Three numerical/step/noncausal-directional-gradient checks pass; corrected
  config validation passes separately. Preserve harness-only failures in
  electro-causal-tests-initial.txt and electro-causal-tests.txt, and the clean
  config result in electro-causal-config-tests.txt. First audit byte gate failed
  on the comment above; raw failure retained, explicit AST check added.

No physical/numerical failure or tolerance relaxation is hidden. Final syntax,
strict JSON, links, finite fields, plot, source and git diff checks pass.

## Artifacts, changes and task end

[Plot script](electro-causal-plots.py), [L2 convergence](electro-causal-convergence.png),
[causal activity](electro-causal-weights.png), [phase fields](electro-causal-fields.png),
[input response](electro-causal-condition-response.png). Archive copies of
config/summary/loss/metrics/selection/memory have full20261028run prefixes;
model/source/actual coefficient arrays stay at direct external run paths.
No hashes, checkpoint migration, old artifact edits or sibling dependencies.

Code: operator_training.py components/causal step/logging/source snapshots;
operator_data.py explicit causal validation; configs/train_operator_causal.py;
tests/test_electropolish_operator_causal.py. Docs: example README, DEVELOPMENT,
artifacts/index.yaml, PROGRESS/append-only DECISIONS and this archived task.
Established single-IVP and noncausal configs/defaults/model templates retained.
All workers/GPU idle; no Git operation or further experiment queued. Exact
resume: none needed for this control; accuracy optimization awaits user direction.
