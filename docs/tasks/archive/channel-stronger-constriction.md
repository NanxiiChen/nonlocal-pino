# Slightly stronger channel constriction — completed

## Goal and outcome

User requested a somewhat stronger smooth middle indentation after the mild
fullNS--CH operator worked well. Increase nominal upper/lower(.12,.08) ->
(.18,.12), minimum height.80 ->.70; prospective independent-amplitude prior
[.04,.14] ->[.06,.20], worst minimum height.60. The compact center2/half-length.9
profile, initial drop and all material/forcing/time parameters remain the same.
The fixed flowQ.2 implies nominal throat mean speed.25 ->.285714. Geometry is
the only physical/input-family change. No user Git/default/SOTA/cleanup action
was authorized or performed.

Status: COMPLETED finite trial. Independent reference/input screening, frozen
extrapolation, one cold5000 stronger-family training and every owned finite
audit exited0. No live owner/child/GPU work or later experiment is queued.
No blocker or required resume action. Peak-shape bias remains a documented
limitation of this first harder trial, not an automatically queued optimization.

## Exact configuration and direct artifacts

- Canonical training entry: examples/channel_nsch2d/train.py.
- Opt-in preset: examples/channel_nsch2d/configs/train_operator_width128_gmls_time40_jvp_stronger_online.py.
- Current run: /root/autodl-tmp/tf-logs/channel_nsch2d/20261102-5090-gmls-time40-jvp-stronger-online-width128-01.
- Model: same run/model_final.eqx. Optimizer/full pool/input records retained.
- Reference: /root/autodl-tmp/data/channel_nsch2d/stronger-reference-20261102-01.
- Frozen source: /root/autodl-tmp/tf-logs/channel_nsch2d/20261102-5090-gmls-time40-jvp-chemical-online-width128-01.
- Frozen stronger evaluation: /root/autodl-tmp/data/channel_nsch2d/stronger-frozen-parent-20261102-01.
- Reserved16 input-only geometry manifest: /root/autodl-tmp/data/channel_nsch2d/stronger-unseen16-20261102-01/manifest.json.
- [Full validation receipt](../channel-stronger-constriction-validation.json).
- [Preceding mild-case record](channel-derivative-clarification.md).

Width128/4heads, original-variable phi/mu/psi/p model, FP32 ordinary compilation,
SOAPbeta1=beta2=.9/lr1e-3, chemical output4/row1/boundary4, nine soft terms,
physical timeJVP with40 fresh continuous strata/update, eight-time gradient
blocks/one optimizer update, degree3/k36 GMLS spatial PDE/BC derivatives,
original AD streamfunction curl, rolling pool16/B2/replace2 every50updates.
No numerical network/PDE/loss/optimizer/sampler modification for this trial.
Seed42 is runtime provenance, not a tuned strategy field. No evolved labels
are used during training; no tanh/output projection/masks/new loss added.

## Independent reference and operator validation

Nominal(.18,.12) and prior-corner(.20,.20) production payloads pass GMLS domain/
boundary rank and polynomial checks. Full202 online prefix plus16 initial
geometries pass59.492s; maximumcondition2.8566e6<1e9, polynomialerror4.19e-11.
Corrected the screen's initial branch to retain boundary_neighbors int32, matching
canonical online construction; no operator weights changed.

Independent localFEM solves coarse40x16/dt.025, time64x24/dt.05, fine64x24/dt.025,
T6,13snapshots,96x40 physical midpoint queries. Source solver/equations unchanged.
All three passed original gates: spatial maximum-time phi1.6116%,u1.4329%,
p2.4019%; temporal phi.9265%,u.2813%,p1.9983%; maximum flux mismatch6.25e-14,
wall speed0. Chemical refinement error is separate and excluded from the
principal-field acceptance gate, as in the original validated case.

Finite reference owner388564 and every FEM child exited before frozen GPU work;
no FEniCS/JAX GPU overlap. Its owner/config and six actual operator/solver helper
sources are preserved directly, without hash-based loading.

## Frozen extrapolation versus new-family cold training

Both rows use the SAME new reference, physical queries, times and geometry:

| Stronger geometry evaluation | Phase pooled | Terminal phase | Velocity | Pressure |
|---|---:|---:|---:|---:|
| Original model, zero extra updates |3.2529%|4.4658%|2.9881%|14.4499%|
| New-family cold5000 |1.9470%|1.6625%|3.0587%|3.9455%|

Frozen source upper prior stopped at.14, so upper.18 is extrapolation. Strict
transfer permits only wall/prior values and artifact paths to change. Actual
source-field replay4.7684e-6 and new IC1.5873e-6/wall3.0219e-7/flux1.9634e-7
pass. Source model bytes unchanged, no fitted weights emitted/imported.
Pressure and minority-shape degradation justified the predeclared single
cold5000 stronger-family trial; no mid-run physics/method/hyperparameter changes.

Terminal minority-fraction L2:12.8749% ->4.8688%; mean contour distance:
6.7792%R0 ->2.4662%R0; IoU81.7901% ->93.0197%. Across13 saved snapshots,
new minority L2 max8.7984%, contour max4.1861%R0, IoU min88.0813%.
Auxiliary pooledmu L2 remains19.5989%. Velocity stays near3%; it did not improve
relative to the frozen baseline. Report the tradeoff rather than all-field gain.

Reference maximum x-span/y-span increases1.1562(mild) ->1.2996(stronger).
At peak t4, the new prediction is1.4096, approximately8.47% too flat by this
bounding-box descriptor. This is not a material strain tensor or continuum
shape certificate. Inspecting all13 snapshots caught a peak omitted by the
automatic0/1.5/3/4.5/6 gallery. The final common-query comparison explicitly
includes0/1.5/3/4/6; fields and both contour figures were visually inspected.

This is one initialization/nominal development geometry within the stronger
training prior. The16 reserved geometries have no evolved solutions in this
trial; no population or repeated-initialization accuracy claim is made.

## Training and numerical provenance

Owner428091, training428094 and all following children exited0. Existing finite
owner accepts --geometry-control separately from unchanged --comparison-control.
Strict comparison allows only four geometry values and artifact paths. Actual
initial model/fullSOAP bytes match the preceding mild cold initialization;
all indices/times/online drawIDs match over5000updates/10000conditiondraws.
Geometry arrays intentionally differ; both complete actual input streams are
independently reconstructed, with equal fixed shapes/dtypes.91 saved actual
source/owner/helper byte checks passed at launch.

Training wall1926.685s(~32.1min), warmupdate.288946s, activepeak7.996GiB;
JAX reserved pool is a different quantity.99rolling replacements installed198
geometries; producer wait.803s; no PDE/FEM solves or labels in training.

Saved-field replay2.8610e-6, physical timeJVP primal9.5367e-7, flux1.5208e-7,
wall3.1444e-7, divergence2.9802e-8 pass unchanged numerical gates. Actual5000
fresh-time/geometry histories replay. Phase balance diagnostics and240independent
continuous-time/full15633-node residual diagnostics on nominal/final-pool
geometries completed. These finite diagnostics do not prove exact GMLS mass or
continuum convergence; no gate or physical boundary treatment was relaxed.

## Figures and verification

All generated assets remain outsideGit:

- /root/autodl-tmp/data/channel_nsch2d/stronger-comparison-20261102-01/contours.png
  (black FEM, orange dashed frozen, blue new; explicit peak t4 included).
- Current run/phi-fields/phi-fields.png (reference/prediction/absolute phase error).
- Current run/phi-fields/convergence.png and droplet-shape/contours.png.

Two focused guard/state/time-tamper tests pass0.036s. Seven changed Python ASTs,
JSON/link/actual-figure checks and git diff --check pass. No unchanged expensive
NN suite was repeated; canonical numerical training code was unchanged.

Diff-level session changes: stronger config; finite reference owner; frozen
geometry evaluator plus cold geometry-only owner control; screen index dtype
fix; focused metadata guard tests; explicit peak contour display helper;
README/DEVELOPMENT/receipt/artifact/dashboard/decision/task records. Preserve
all preceding controls, references, weights and source snapshots. No commit,
push, cleanup, default or SOTA promotion performed.
