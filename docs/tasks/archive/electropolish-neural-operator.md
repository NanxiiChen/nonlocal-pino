# Electropolish neural operator — completed prototype

Date:2026-10-28. User explicitly switches from completed confined-CH default
work to a random-initial-interface neural operator inspired by PI-CViT, in
repository Equinox style, with shared-space/time optimization and a simpler
condition-injection comparison. Requested prototype/experiment COMPLETE.
Accuracy remains exploratory; no default/SOTA/general operator accuracy claim.
No pending job or blocker; no Git history operation authorized/performed.

## Delivered formulation and architecture

Existing electropolish six-mode rough-profile/SDF input family, material data,
5s/normalized0..1 horizon,101x51/sparse delta.02/cubic graph and top/bottom
Dirichlet data retained. Side closure remains in-domain pairs, not a pointwise
classical-Neumann guarantee. Inputs are16 prescribed training fields and four
predeclared disjoint test fields; dataset seed42 is provenance. Positive-time
references exist only for two train-monitor and four test cases, never enter
loss/step/payload. No test-based model selection: final2000-update checkpoints.

Canonical train.py supports configs/train_operator.py; train_base remains the
established single-IVP default. New reusable src/nonlocal_pinn/nn/conditional.py
has pure Equinox patch projection,2self-attention encoder blocks/width64/heads4,
spatial cross-attention OR pooled FiLM, then3pointwise time-FiLM layers. Initial
image1x32x64/patch4x8 gives64tokens. Spatial queries additionally receive known
local phi0/c0. No future geometry/motion/fields supplied; no hard IC. Frequencies
are stopped metadata. APIs: encode/prepare_encoded/prepare/decode/paired/call.
Preparation stays inside parameter AD, outside temporal JVP; no learned cache
persists across optimizer updates. Query images/sensors fixed; no resolution or
arbitrary-topology/material generalization claim.

Source read: https://github.com/NanxiiChen/PI-CViT/blob/main/models/cvit.py
This is an independently written separable spatial-attention/time-FiLM variant,
not an exact refactor of upstream time-dependent cross-attention. Simpler FiLM
is the condition-injection control, not a full FNO/DeepONet comparison.
Primary background: https://arxiv.org/abs/1709.07871 and
https://doi.org/10.1038/s42256-021-00302-5 .

Independent random-interface references show positive q maxima up to.00298,
violating old sigmoid-map q<=0 (c_Se=1), even though physical phi/c stay within
roundoff of0..1. New operator explicitly uses linear_liquid:
phi=(1+tanh(z))/2, c_L=c_Le*s with unrestricted s, c=c_L+dc*h(phi).
KKS mixture and physical PDE unchanged; artificial liquid bound removed only
in this new variant. Total-c range is monitored, not hard bounded.

## Data/reference and runs

Dataset root:
/root/autodl-tmp/data/electropolish2d/operator_random16_v1
Its dataset.json/conditions.npz record IDs, exact partition, input images/fields,
physical spec, graph grid and direct reference paths. Six physical-field SciPy
BDF/analytic sparse Jacobian references pass independent time refinement;
maximum1.93e-8, fixed acceptance1e-4. This isolates same-graph neural error;
no local-PDE mesh convergence claimed. No FEniCS/MPI job used.

Run root /root/autodl-tmp/tf-logs/electropolish2d:
-20261028-electro-operator-cross-01
-20261028-electro-operator-film-01
-20261028-electro-operator-entry-smoke-01 (one-update final API check, not accuracy)

Both pilots:2000Adam/LR.001/ordinaryFP32/highest matmul, two conditions/update,
Nt4,256core/32padded pools,1024IC nodes, five original physical losses:
AC/CH L1, top/bottom/IC physical MSE. Existing inverse-L1-gradient balance,
refresh25/cap1000; no new adaptive algorithm. Initial/Dirichlet constraints soft.
Prescribed nodal IC uses exact top/bottom overrides matching references.

Mean over FOUR held-out inputs (pooled includes t0):
| Scheme | phi pooled | phi terminal | c pooled | c terminal | Trainer s |
|---|---:|---:|---:|---:|---:|
|Cross|.171850265|.133051080|.186910131|.149292687|245.682|
|FiLM|.174616440|.119362611|.190404204|.135197392|160.897|
|Frozen IC|.318770073|.477211250|.328136105|.483092050|—|

Both learn evolution; neither approaches established single-IVP PINN precision.
Two monitored training inputs have pooled phase.17032/.15164, similar magnitude
to tests; do not infer a unique mechanism or claim all16 training fields tested.
Plots show excessive early smoothing of roughness. Terminal q relative error
cross.05735/film.05304, freeDq.81463/.76596. Small-family/single-seed pair is not
statistical architecture ordering. Current default remains cross-attention
as opt-in PI-CViT-inspired prototype; lighter film available and worth using as
baseline. No additional blind architecture/seed/optimizer sweep authorized by
inference; SOAP implementation side issue remains deferred.

## Verification and precision limits

Three distinct focused tests: split/paired output equivalence, time finite
differences and zero cross-time derivatives, condition response/frozen frequency
gradients; independently NumPy physical PDEs and finite time/trainable-parameter
direction derivatives; invalid config/shapes. All pass across focused runs.
The physical/parameter check passes30.31s. Initial harness failures are retained:
perturbing stopped frequency values, then a NumPy read-only tangent; corrected
checks perturb only trainable parameters. No PDE/gradient formula correction.
Range preflight initially rejected h(phi)=1+roundoff; explicit3e-7 allowance,
no clipping. Its only empty failed output directory was removed before retry.

55 common initial parameter-array leaves match exactly between two pilots;
configs identical except conditioning/run name.14/15 source files identical;
sole delta train.py success return. Same condition/time PRNG and core-pool
schedules, no hash processing. Final public layout puts dimensions/conditioning
under c.arch and data/action under c.operator. First two saved pilot configs
still store architecture in operator; preserved unchanged. Private template
builder accepts(architecture,image_shape,key), so archival replay passes
saved.operator explicitly and public training saved.arch. Exact same-backend
initial template equality and saved-leaf compatibility pass; do not require
bitwise CPU/GPU random initializer equality (max~1e-8 differences observed).

Canonical mocked dispatch verifies --seed runtime/copy and absl returnNone;
real reduced width8/seed2026 one-update canonical GPU check completes/saves with
exit0 after final public config cleanup. First cross trainer was numerically
complete/saved, but original new branch returned summary dict to absl, producing
CLI exit1; preparation similarly printed its tuple with exit1 after successful
reference creation. Fixed branch returnsNone; film and final smoke exit0.
No numerical run is silently relabeled failed/successful by that wrapper error.

GPU final replay at saved highest policy: cross exact, film max6.26e-7.
CPU highest: cross physical gap up to.0016244, film1.43e-6. Strict cross CPU
pointwise2e-5 probe FAILS and remains explicit. CPU cross pooled/terminal phi
changes2.29e-5/9.88e-5 absolute (<.01 percentage point); aggregate fields match
within relative1e-3, not bitwise. A default-policy GPU probe accidentally used
other matmul precision and is excluded; raw evidence retained.

Shared vs repeated space forward+timeJVP benchmark, actual trained models,
1024points/four times,50warmed calls, no PDE/parameter gradients/optimizer:
cross.6219 vs.7226ms (1.162x); film.5353 vs.5932ms (1.108x). FP32 cross raw
outputs/JVP relative gaps2.2846e-4/1.6233e-4; max.00233/.00538. Strict FP32
probe fails; do not claim exact finite-precision batching invariance. Promoted
trained FP64 weights/inputs give exact output/JVP equality in both probes.
No production normalization/precision change is used to conceal differences.

## Records, changed files and exact resume

Archive prefix electro-operator-* includes scripts, raw failures/success logs,
config/summary/loss/metric copies, provenance, separate CPU/GPU JSON audits,
benchmark, initial-family/convergence and3x5/all-test-terminal fields.
Reference/training checkpoints stay at their original new direct paths.
Full raw physical-field arrays are run/prediction.npz. Read backend provenance
in replay records; no hash gate. No full expensive unchanged test suite rerun.
Final checks: canonical real smoke/entry/template, strict reference/dataset,
JSON/schema/source parity, links, no external/sibling runtime dependencies and
git diff --check. No active process remains at completion.

Code: src/nonlocal_pinn/nn/conditional.py; examples/electropolish2d/operator_data.py,
operator_training.py, configs/train_operator.py; four-line canonical train.py
branch; tests/test_electropolish_operator.py. Stable docs: root/example README,
DEVELOPMENT and artifacts/index.yaml. Memory: PROGRESS, append-only DECISIONS,
this archived task; prior confined-ch-global top marks explicit switch/inactive.

Requested pilot COMPLETE; phase/transport accuracy and FP32 attention batching
sensitivity remain limitations, not claims of a solved general operator. Next
work requires user direction: e.g. controlled5000SOAP baseline or joint geometry/
time FiLM at every decoder layer; neither is queued or represented as tested.
