# Electropolish fixed-input and continuous-online controls — complete

Date2026-10-28. User asks actual condition batch usage, authorizes a single-input
CViT test, asks SOAP vs Adam/current architecture, then requests continuous
online random initial interfaces withB8/16 if memory allows. Both stages COMPLETE.
Accuracy remains exploratory. No active job/blocker/Git authorization or action.

## Actual model and prior training

Previous finite pilot used16 source training interfaces and B2; vmap computes
independent condition losses then means the five physical parts across B.
Established electropolish single-IVP PINN usesSOAP5000/LR.001; NO pilots useAdam.
No math/operator/reference/default change is needed to compare input regimes.
The recommended operator is an independent separable PI-CViT-inspired variant:
1x32x64 initial phase image,4x8patches/64tokens; encoder2blocks/width64/heads4,
FFN128, one spatial cross-attention block; three width64GELU time-FiLM layers.
Spatial32Fourier pairs +xy+knownlocalphi0c0 produce68features, temporal8pairs+rawt
produce17features. Output2raw channels map to boundedphi/unrestricted liquid
concentration via the declared KKS mixture.146178trainable array values plus64
frozen spatial-frequency values. One model covers all inputs/times, no windows.
Retain shared-space/time preparation and time JVP; raw frequency/position metadata
and no-future-label constraints are unchanged.

## Fixed-input controls

Canonical train.py loads configs/train_operator_single.py; explicit subset(0,),
B1 from unchanged source dataset, geometry identity1662057944. Optional subsets
reject empty/duplicate/noninteger/negative/heldout IDs and oversized batch.
Absent option preserves original all16 draws exactly. Excluded source monitors
are unused_train, never mislabeled as fitted accuracy. Actual2000x1 sampled ID
arrays in BOTH runs contain only0;73 initial parameter arrays match exactly.
Configs differ onlyoptimizer/name; model/operator/physical loss sources match.
Fresh seed42,2000updates/LR.001, ordinaryFP32/highest, Nt4/core256/pool32/IC1024,
5s, original five physical losses/gradient balance and no hard initial condition.
No positive-time labels enter training; final fixed-budget checkpoint is used.

Run root /root/autodl-tmp/tf-logs/electropolish2d:
-20261028-electro-operator-single-cross-01 (Adam,237.259s)
-20261028-electro-operator-single-soap-01 (SOAP,387.328s)

Same interface0 relative L2:
|Scheme|phi pooled|phi terminal|c pooled|c terminal|
|---|---:|---:|---:|---:|
|Prior16-input Adam2000/B2|.215383892|.158878528|.232730781|.176055334|
|SingleAdam2000/B1|.024828917|.033869553|.029650846|.040311951|
|SingleSOAP2000/B1|.041868677|.032135855|.049411626|.037002806|

SOAP slightly improves terminal while losing pooled/cost here; no universal
optimizer ranking or longer-budget conclusion. SOAP first compile produced slow
GEMM alarms but completed successfully; do not classify alarms as failed training.
The condition-count comparison also changes batch/work/exposure counts and
objective; not an isolated generalization ablation. Single-case fitting is not
NO generalization. SelectAdam for the first inexpensive online baseline; preserve
SOAP as explicit override and established PINN default.

## Continuous online input space

Canonical configs/train_operator_online.py uses B8/5000Adam/LR.001, same model/
physics/losses/time-grid schemes and5s. NEW continuous amplitudes[.3A,A),
frequencies[f_lo,f_hi), phases[0,2pi) for six modes are drawn each update.
Existing source dataset is only a reference/evaluation bank in this mode;
its two historical source-training monitors are monitor, four heldouts test.
No bank image/field enters the training payload; no train referenceL2 reported.

Input family is the existing six-mode height-profile form, not arbitrary
function spaces/materials/topologies. Fixed4B candidates, select firstB whose
2000vertices stay in(.005,.495), explicitly conditions the input distribution
on valid heights. If valid_count<B raise BEFORE any update; no silent padded
replacement, retry or batch reduction. JAX initial-field construction uses the
original2000vertex/±16segment SDF and exact top/bottom IC overrides. Nearest
curve indices/query/sensor geometry are precomputed per fixed-shape pool.
Construct images and known local IC from the SAME coefficient triples, outside
parameter AD. Time derivatives still act in decoder JVP; physical PDE unchanged.

Run: /root/autodl-tmp/tf-logs/electropolish2d/
20261028-electro-operator-online-b8-adam-01;327s including compile/evaluation.
Actual online_coefficients.npy shape5000x8x6x3,40000unique coefficient vectors;
accepted counts31..32 of32 per update. No sampled_conditions.npy/finite bank
selection is used. Four fixed unseen reference targets (strict independent
same-graph BDF/Jacobian/time refinement from prior task) give mean phi pooled/
terminal.177476563/.123550569, c.193172998/.140294653. No test checkpoint selection.
Prior finite pilot has different budget/batch/input distribution; no pure
online-accuracy advantage claim. Residual/shape/transport fitting remains open.

Memory.json first/25th/final measures peak active2311164672bytes (2.15GiB),
allocator pool5375000576bytes(5.01GiB); observedNVML~5.3GiB on reported48GiB.
B8 fits; B16 not run or claimed. GPU idle after training and audits.

## Verification and direct records

Focused subset/RNG/split validation passes1test1.31s; independent NumPySDF/FP64
identity plus fresh/replayableB8/BC/range/rejection tests pass2tests5.56s.
Old unchanged expensive numerical suite not rerun. GPU final checkpoints all
replay within1.43e-6; saved targets equal independently loaded references.
NumPy reconstructs phi/c L2 and independent SciPy graph transport diagnostics,
vertical-crossing/range metrics. All actual fixed IDs and40000-profile uniqueness,
accepted counts,32 independent NumPy profile-height checks pass. Known earlier
CPU FP32 cross-attention pointwise sensitivity remains recorded in original
operator task; this task does not claim CPU bitwise identity.

Archive scripts/JSON/logs/plots under prefix electro-single-* and electro-online-*:
-provenance.py/json records73identical initial leaves and exact fixed input use;
-single-online-audit.py/json reloads all three final models/direct fields;
-single-online-plots.py creates same-input optimizer curves, online heldout
curves,3x5phase fields and sixteen actual profiles from predeclared updates;
-raw config/summary/loss/metric/selection logs, generator/selector test output,
SOAP compile alarms, coefficient/accepted paths and allocator measurements.
External checkpoints and coefficient arrays stay at the explicit new run paths.
No hashes or sibling dependencies introduced, no old artifact altered.

## Changes, completion and resume

Code: optional training_condition_indices/sample_condition_indices/condition_split
in operator_data.py; train_operator_single/online configs; online_conditions.py;
conditional online payload/provenance/memory recording in operator_training.py.
New focused tests test_electropolish_online.py and subset method in existing
operator tests. No model/numerical-core/PDE/optimizer implementation change.
Docs: root/example README, DEVELOPMENT, artifacts/index.yaml; memory PROGRESS,
append-only DECISIONS and this archive. No Git history operation.

Acceptance achieved: actual batch clarification, fixed input/model fit and
matched optimizer evidence, continuous onlineB8/5000 run with verified40000new
inputs, measured memory, unchanged pure physics training, independent replay/
stream/L2/field records. Final syntax/JSON/link/diff/source/default checks pass.
Requested work complete; no jobs or further sweeps queued. Future accuracy work
needs user direction (e.g. condition/time coupling or transient sampling); do
not silently treat those as implemented, or tune heldout metrics as training.
