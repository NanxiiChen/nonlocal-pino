# Concentration fidelity and stagger audit — completed 2026-10-25

## Goal and corrected interpretation

The user's point is accepted: emphasizing concentration imbalance is not itself
a defect. Small phase error in bulk does not establish concentration/transport
accuracy. A large pure-phase loss allocation is an observation, not proof that
the transform should ignore that region. Solve the coupled problem rather than
use phase-field L2 alone to rank physical fidelity. Earlier original-residual
recommendations are comparative defaults within stagger-off controls, not proof
that the full original staggered strategy or concentration dynamics is solved.

This audit quantifies seven saved KKS runs and verifies production stagger.
No training, physical PDE/default/checkpoint change. Only a startup guard and
README clarification are added to prevent the research runner silently ignoring
a nonzero stagger setting.

## What stagger actually does

- Single-pit production config: first1000 training updates alternate25 AC-only
  then25 CH-only; both PDE losses are active afterwards. Two-pit production
  config instead uses10000 warmup updates/switch every10. IC/BC remain active.
- These are training-iteration windows, not physical time marching. Every
  minibatch still samples the prescribed physical time domain.
- Activation masks multiply loss gradients. No parameters or field evaluations
  are frozen; shared trunk and KKS mapping couple both outputs. AC-only can
  update concentration-related parameters and CH-only can alter phase.
- All saved original/latent/bounded research controls have stagger_steps0.
  residual_experiment.train/make_step does not implement AC/CH masks. A config
  field alone cannot activate them. The new guard rejects nonzero stagger_steps
  before reading references or creating output directories. It does not wire
  or enable a new method; valid existing joint runs are unaffected.

## Reproduced production weight-cache defect

Production make_loss_fn returns weights after applying temporary AC/CH masks,
and train.py stores that vector as w_ema. If a stagger switch occurs without a
GradNorm refresh, the newly active equation can inherit the zero from its
previous inactive period while the formerly active one is now masked out.
Both PDE coefficients become zero until refresh. Production single-pit switch25
and GradNorm refresh20 do not always coincide.

A minimal actual-production-loss probe with nonzero gradients reproduces:
AC-block refresh weights(AC,CH)=(2.18279,0); switch to CH without refresh=(0,0);
switch with refresh=(0,1.84414). IC/BC weights stay nonzero. Optimizer momentum
can retain past PDE influence; the claim is about current PDE loss-gradient
contributions, not all parameter motion. Full vectors are saved in the
[probe JSON](corrosion-concentration-stagger-audit-weights.json).

No production fix is applied in this audit. Before a new stagger experiment,
keep unmasked adaptive weights as state, form effective masked/biased weights
only for the current gradient, and verify switches even without refresh.
Forcing refresh on each switch is a narrower workaround; storing unmasked state
is the preferred explicit design. This is a control-state defect, not PDE math.

## Concentration decomposition

With h=3phi^2-2phi^3, dc=c_se-c_le and q=c-dc*h(phi)-c_le,

$$
c=c_L+\Delta c\,h(\phi),\qquad q=c_L-c_{le},
$$

$$
c-c_{\rm ref}=\Delta c\,[h(\phi)-h(\phi_{\rm ref})]
+(c_L-c_{L,\rm ref}).
$$

The decomposition is exact and independently checked to absolute1e-15 for all
saved snapshots. Component norms are correlated and are not additive error
percentages. c_L is a KKS auxiliary; q differs by a constant, so their absolute
error is equal but their relative-L2 denominators differ. Also report RMSE and
RMSE/c_le in the JSON. No reference value enters training.

Full-grid physical50 s metrics are below. c and c_L use all nodes; q/Dq and CH
use free interior nodes. Dq error is the relative L2 of the *graph transport
operator*, not flux-vector error or complete CH residual. Stable c_L/q comes
from reloaded raw network output, avoiding subtraction of two physical fields.

|Run|c relative L2|c_L relative L2|q relative L2|Dq relative L2|CH RMS|
|---|---:|---:|---:|---:|---:|
|historical original|0.041352|0.006762|0.036483|3.6862|2.122930|
|original seed42|0.047219|0.007474|0.040454|2.9398|1.688361|
|bounded seed42|0.019280|0.007919|0.042438|8.4737|4.890990|
|original seed7|0.031598|0.005176|0.027406|2.9089|1.675212|
|bounded seed7|0.043427|0.004788|0.025145|1.5810|0.898756|
|full inverse|0.545845|1.000000|5.337034|1.0000|0.000060|
|full inverse ac1e-9|0.449566|0.083942|0.458739|3.8978|2.173356|

At seed42 bounded weighting improves total c L2 from4.72% to1.93%, yet c_L
error slightly worsens0.747%->0.792%, and Dq error increases2.94->8.47 times
the reference norm. At seed7 total c worsens while c_L/q/Dq improve. Thus
ranking by total phase/concentration L2 can reverse the concentration-transport
ranking; the mixture geometry can mask another field's weakness. c_L value
errors are sub-percent in the original controls, not uniformly large; their
spatial transport operator can still be substantially inaccurate.

Reference terminal free-node Dq RMS=0.135825; stable Dq error RMS is0.399298
original seed42 and1.150941 bounded seed42. Native-q-versus-stable-q graph
reconstruction difference RMS is only~0.00060, so the large operator error is
not explained by saved-field cancellation. Actual c_t is obtained by JVP of
the selected network output map; reference c_t=-N_CH*Dq follows the independently
checked physical graph ODE on free nodes. Reference c_t RMS=0.577256.
Stable/native complete CH RMS agree closely for the accurate-field controls
(original1.68836, bounded4.89099 at seed42). This independently confirms a
transport-equation weakness despite improved total-field L2. No full FP64
training or spectral attribution is claimed.

The saturated full-inverse model has c_L relative error~100%, q error~534%,
but stable CH RMS only5.97e-5: nearly constant wrong concentration can satisfy
transport while violating BC/IC. Do not treat a small CH residual as reference
concentration accuracy. Its native CH floor0.00132 is larger due to finite
precision; that does not explain the physical collapse.

## Concrete next experiment, not executed here

First correct/wire stagger bookkeeping and masks in the research runner, then
compare a fresh2x2 original/full-latent x stagger-off/on study at the same
5000-update/50 s budget, same KKS, soft IC/BC, reference, AC/CH scales, seeds,
initialization and sample pools. Start with the inherited1000-update/25-block
schedule, not an unvalidated physical-time split. Log unmasked/effective
weights and activation bits. Monitor phi/c/c_L/q relative L2, graph Dq error,
full CH residual, BC/IC and saturation; do not declare success from interface
or transformed residual alone. Genuine frozen-field/block-coordinate training
would need separate derivation/parameter handling and must be labelled distinct
from the current production loss alternation.

## Reproduction, verification and task-end

[Audit script](corrosion-concentration-stagger-audit.py),
[all terminal/pooled/region metrics](corrosion-concentration-stagger-audit.json),
[production cache probe](corrosion-concentration-stagger-audit-weights.json).
Paths to all seven existing runs are explicit in script/JSON. Reproduction:

```bash
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 .venv/bin/python -u docs/tasks/archive/corrosion-concentration-stagger-audit.py
```

Metadata/grid/time/BC checks, decomposition identities, raw-checkpoint q/JVP
and production cache probes passed. Startup rejection was verified before any
reference/run-directory access. Final JSON/AST/whitespace/diff checks pass at
session end. No unchanged expensive suite or reference/FEM solve was repeated.

Diff: add audit/script/JSON/cache-probe record, add the research startup guard,
clarify corrosion README, append DECISIONS and update PROGRESS. No default,
production-step implementation, historical checkpoint or Git history changed.
No blocker/running process/remaining action for this audit. The above2x2
stagger experiment is a concrete separate next task; its benefit is unmeasured.
