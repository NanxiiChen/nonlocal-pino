# FEM weak-residual PINO versus nonlocal and local AD

Date:2026-11-13. Status: comparisons and requested figures complete; follow-up
numerical diagnostics remain open. The user authorizes README/site publication
and commit/push of the current result. Channel RESOURCE probes are complete and
its long run is cancelled. Spinodal stays on its original full-field scheme.
No numerical default change is authorized.

## TODO and execution gate

- [x] User authorizes executing this comparison after channel resource tests,
  explicitly cancelling channel long training.
- [x] Verify the signed FEM weak-residual implementation and run the fixed-geometry
  feasibility check described below.
- [x] Run the paired nonlocal/local-AD/FEM comparison and64sample evaluation.
- [ ] Complete the frozen historical-source reproduction control and remaining
  discretization/interpolation diagnostics; do not infer a universal ranking.

The prior deferral is superseded by the user's explicit execution instruction.
Channel owner5245 was first stopped, then terminated only after final resource
child57170 exited0. Its journal now says `completed_resources_only`; all four
resource results are saved, no20k/30k run launched. Ceramic may now use the GPU.
Do not revive any other stopped experiments.

## Question and scope

Can the same geometry-conditioned, physics-only neural operator use a finite
element spatial residual, avoid spatial AD, and incorporate homogeneous Neumann
conditions naturally? Yes. The proposed comparison separates two benefits that
should not be conflated: avoiding high-order coordinate AD, and using regional
nonlocal exchange. Natural homogeneous Neumann treatment is not exclusive to
nonlocal methods.

Prefer ceramic-particle corrosion (`examples/particle_corrosion2d`) first:
six random smooth holes, a fixed small reservoir opening, existing paired
nonlocal/local-AD machinery and an existing64geometry local-reference cohort.
Use the accepted physical parameters and output map. Do not reopen the failed
Spinodal compact-output experiment or begin with coupled NS/pressure issues.

## Literature and what is being borrowed

- Badia, Li and Martín, *Finite element interpolated neural networks for solving
  forward and inverse problems*, CMAME418(2024),116505,
  [paper](https://arxiv.org/abs/2306.06304),
  [full text](https://arxiv.org/html/2306.06304v4).
  Interpolate neural predictions into FE spaces and minimize the assembled weak
  residual; residual norms/preconditioning matter. This is a forward/inverse
  PDE framework, not evidence that this repository's corrosion operator will
  achieve any particular accuracy.
- Rezaei et al., *A Finite Operator Learning Technique for Mapping the Elastic
  Properties of Microstructures to Their Mechanical Deformations*, IJNME126,
  e7637(2025), [paper](https://doi.org/10.1002/nme.7637).
  Physics-only parametric learning using FE algebra and natural Neumann
  conditions supplies an operator-learning precedent. Their elasticity energy
  loss is not to be copied as a transient corrosion residual.
- [FEniCSx mixed Dirichlet/Neumann tutorial](https://jsdokken.com/dolfinx-tutorial/stable/chapter3/neumann_dirichlet_code.html)
  documents integration by parts, the test space and the boundary integral.

The proposed mass-lumped, nodally integrated corrosion arm is inspired by this
family; it is not a verbatim implementation of the FEINN paper's formulation,
test spaces or Riesz preconditioner.

## Derivation and boundary meaning

For a representative scalar equation

$$
u_t-\kappa\Delta u+f(u)=0,
\qquad \partial_nu=0\text{ on }\Gamma_N,
$$

test with functions \(v\) whose trace vanishes on the Dirichlet boundary only:

$$
(v,u_t)_\Omega+\kappa(\nabla v,\nabla u)_\Omega+(v,f(u))_\Omega
-\kappa\langle v,\partial_nu\rangle_{\Gamma_N}=0.
$$

The last term is zero because the prescribed Neumann data are zero. Test
functions touching Neumann walls must remain. Nonzero prescribed flux would
require an explicit boundary load integral; it cannot simply be dropped.

Use standard P1 nodal hat functions \(N_i\) on conforming triangles as test and
interpolation functions. They are known functions, not another learned model:

$$
u_h=\sum_j u_\theta(x_j,t;a)N_j(x),\quad
M_{ij}=\int_\Omega N_iN_j,\quad
K_{ij}=\int_\Omega\nabla N_i\cdot\nabla N_j.
$$

Here \(a\) denotes the geometry condition. The consistent weak residual is
\(M\dot{\mathbf u}+\kappa K\mathbf u+\mathbf F(\mathbf u)\), where
\(F_i=\int N_i f(u_h)\). Differentiation of the analytic FE shape functions
is precomputed. Do not take an elementwise second derivative of the P1
interpolant: it is zero inside each element and is not the weak Laplacian.
Parameter backpropagation and time JVP remain; spatial coordinate AD does not.

For the first arm use the ALREADY IMPLEMENTED mass-lumped/nodal-quadrature
local discretization, with \(M_L=\operatorname{diag}(m_i)\),
\(m_i=\int N_i\), and \(D_{FE}=M_L^{-1}K\simeq-\Delta\).
This avoids introducing consistent-mass solves or new nonlinear quadrature at
the same time as the spatial-method comparison. It is a declared FE quadrature
choice, not exact integration of the nonlinear reaction.

Natural enforcement is a property of the weak equations and their test space.
It does not guarantee that an unconverged network or its off-mesh derivatives
satisfy zero normal flux pointwise. The analogous finite-horizon nonlocal
closure conserves regional exchange but is not exact classical pointwise
Neumann enforcement at arbitrary finite horizon either.

## Exact corrosion arm

Keep \(\phi=(1+\tanh z)/2\),
\(c=c_{Le}s+\Delta c\,h(\phi)\),
\(q=c_{Le}(s-1)\), with the same \(h,g\) and all physical coefficients.
At the selected mobility multiplier1, the mass-lumped weak vectors are

$$
R_\phi=M_L\dot\Phi+N_3K\Phi+
M_L\{N_2g'(\Phi)-N_1\Delta c\,Qh'(\Phi)\},
\qquad R_c=M_L\dot C+N_{CH}KQ.
$$

All nonlinear functions/products here are nodal. Only the rows associated with
Dirichlet mouth tests are removed; retain their columns and the network's mouth
values in neighboring rows. Preserve the existing soft mouth and initial losses
in all arms; do not give FEM an additional hard mouth constraint secretly.
Keep all other outer-wall and particle-wall test rows. Assemble the homogeneous
natural conditions on both \(\phi\) and \(q\), not incorrectly on \(c\).

Use \(KQ=c_{Le}Ks\) to avoid cancellation, after verifying the constant-null
identity at the chosen precision. The network output and KKS relation need no
modification. The current recipe does not divide residuals by a tanh Jacobian.

Normalize by lumped mass before reducing: \(r_i=R_i/m_i\). Retain the current
volume-weighted mean ABSOLUTE PDE residuals, unit AC/CH prescales and the same
five-loss gradient-balancing policy. Do not compare a raw Euclidean weak-vector
norm with the current pointwise loss: it introduces mesh-dependent scaling.
Any future squared/Riesz/consistent-mass loss is a separate matched ablation.
For sampled rows use the same free-node measure and inclusion correction as
nonlocal. Compare convergence using field errors, not incompatible loss values.

Existing building blocks inspected:

- `examples/particle_corrosion2d/mesh.py::assemble_p1`: symmetric stiffness,
  positive lumped volumes, constant-null check and explicit conforming holes.
- `solver.py`: independent BDF physical-field integration on \(M_L^{-1}K\),
  with identical reservoir/zero-flux conditions.
- `losses.py`, `local_ad.py`: shared physical fields and time JVP, original
  five losses versus two additional normalized wall losses for AD.
- `configs/train_physical_spaced_w256_method_{nonlocal,local_ad}.py`:
  inherited paired rolling method-study presets.

Important implementation constraint: general P1 stiffness on these meshes
need not have nonpositive off-diagonals. Use a general signed sparse/element
action; do not route FEM through a positive nonlocal-edge validator, clip
coefficients, or impose an unrequested M-matrix repair. FEM already uses these
meshes for the reference; learning from its residual involves no solved labels.

## Proposed staged comparison

| Arm | Spatial calculation | Homogeneous Neumann | Shared output / IC / mouth |
|---|---|---|---|
| Nonlocal | Current visible regional integral operator | Closed regional exchange | Current scheme |
| Local AD | Existing coordinate derivatives | Two explicit wall losses | Current scheme |
| FEM weak | P1 shape functions, lumped mass and stiffness | Natural weak boundary terms | Current scheme |

1. **Correctness before training.** Independently verify element assembly,
   orientation/constant nullspace, signed-row support and refinement. Check
   complete weak rows against full assembly, including rows on hole walls;
   check a manufactured nonzero-flux field with its explicit load to verify
   boundary signs, plus a homogeneous constant solution. Compare parameter
   gradients of compact/full versions of the same selected FEM rows. Time
   derivatives retain random-time JVP, not a time-step approximation.
2. **One fixed geometry, short cold feasibility.** Use the same predeclared
   geometry, model initialization and full-node queries to verify boundary
   behavior and resource feasibility. A proposed2000-step pilot diagnoses
   correctness/optimization only; it is not a final accuracy comparison.
3. **Matched geometry-operator comparison.** Three cold20k runs using width256,
   B4/Nt32, FP32 ordinary compilation, original Adam schedule and balancing.
   Geometry64slot rolling pool, replacement4/250; spatial/IC pools refresh4/50;
   paired256free-core draws, identical time/IC/mouth/encoder data. Use each
   method's complete required support: FEM element stars, nonlocal neighborhoods,
   AD coordinate points plus wall quadrature. Count all actual neural queries.
   Do not impose an extra Neumann penalty on FEM or nonlocal for symmetry.
4. **Only then test sampling/mesh sensitivity.** Begin with identical assembled
   meshes and core IDs. A full-versus-sampled FEM row control can diagnose
   sampling separately. Refine spacing once if FEM spatial error or thin-gap
   resolution is material. Do not jointly sweep architecture, optimizer,
   residual coordinates and sampling as in the earlier Spinodal pilot.

The plan keeps rolling geometry streams deterministic by explicit draw IDs and
installation schedules; different CPU preparation speeds must not change the
realized geometry prior. Persist model/optimizer initial arrays and replay
actual inputs, since a common integer seed alone does not establish equality.
First use one paired initialization; additional paired repetitions can be
planned if apparent differences are small. No claim of seed robustness from one.

FEM core rows require all incident triangles/nodes, including boundary rows;
do not sample arbitrary cell fragments or square unassembled element residuals
as though they were the assembled nodal residual. Complete support allows
rolling FEM row pools without a dense global metric or global mass projection.
This possibility is specific to this chosen lumped formulation.

## Evaluation, timing and interpretation

Use64common held-out geometries and a common refined local-reference solution,
never the same coarse training discretization alone as the accuracy target.
Audit existing reference provenance/refinement before reuse. Also report direct
coarse-FEM and nonlocal numerical trajectories against that target on a small
predeclared subset, separating spatial bias from learning error.

Preserve the user's primary metric: per-sample, per-channel relative L2 over
all common evaluation space-time points; average over channels, then samples.
Report channel scores and final-time phase separately. Retain the existing
unweighted common-point metric for compatibility; add volume-weighted diagnostics
explicitly rather than silently replace it. Report both direct continuous NN
queries (the shared primary inference route) and FE-interpolated predictions for
the FEM arm; discrepancies reveal off-node behavior/interpolation error.

Include independent near-wall field errors, pit/interface agreement and
concentration balance accounting for mouth exchange. Weak flux residuals and
AD normal derivatives of the continuous network are different diagnostics;
do not declare either identical or selectively report the easier one.

Measure on the then-current SAME GPU: compile/startup, warm update including
input preparation, cold total wall time, peak active memory, preprocessing/
assembly and per-geometry inference. Also plot error against elapsed training
time and compare time to predeclared accuracy thresholds. Sequential resource
measurements avoid GPU contention; previous-card durations are context only.
Save intermediate models at1000step intervals to permit common-cohort error
curves, without claiming a curve from final-only weights.

Proposed figures: final64case channel-mean distributions; phi/c error versus
time cost; four fixed geometries at representative times with shared local
reference contours. Follow PLOTTING_GUIDELINES when drawing. No figures yet.

FEM may be competitive or faster because it has compact element connectivity;
the comparison must allow that outcome. It tests whether a nonlocal integral
operator adds value beyond discrete physics and natural Neumann handling.
Current ceramic node generation/volumes already use a conforming mesh, so this
case cannot support a claim of completely mesh-free preprocessing for nonlocal.
Claims about arbitrary geometry require geometry/discretization qualification,
not just absence of an explicit boundary-loss term.

## Exact resume point

Read `/root/autodl-tmp/tf-logs/ceramic-fem-comparison-20261113-02/journal.json`
and its current stage log first. Continuation owner101250 reuses the completed
FEM20k arm from bundle01 after explicit regional-postprocessing recovery.
It runs nonlocal/localAD20k in that order, independently audits rolling inputs
and pairings, and evaluates each final model on the same64refined references.
The original bundle01 journal remains failed, preserving its exact history.
Failure stops the queue with retained evidence; never launch a duplicate owner
or silently retry. No FEM training/inference was repeated during this recovery.
No FEniCS solve, Git action or Spinodal experiment is authorized/needed here.

## Implementation and initial verification

`fem_weak.py` assembles signed mass-normalized P1 rows and preserves mouth trial
columns; no negative offdiagonal filtering, coefficient clipping or incomplete
support. Original graph residual evaluation and timeJVP are reused. The existing
`train.py` is the entry for all three explicit `train_fem_comparison_*` presets.
FEM validates signed coefficients separately from positive physical measures.
Initial and rolling input constructors preserve original physical core draws.
The paired audit allows only declared FEM support differences, checks fixed
IC/BC/encoder arrays, every actual core/weight/time/geometry draw and cold
model/optimizer bytes. Old nonlocal/localAD comparisons retain their stricter
stencil-identity checks. No production default changed.

All arms: width256, B4, Nt32 freshly sampled, 256cores,8sets/geometry,
64rolling slots, Adam20k, LR1e-3 halved every3000 with1e-5floor, original
five shared losses/gradient balancing. LocalAD adds its original two explicit
Neumann penalties. Checkpoints every1000; FEM padding2048/32 versus original
nonlocal3072/64, justified by complete incident-element rather than horizon
support. Different network-query counts are part of the method cost.

Known inputs: `/root/autodl-tmp/data/particle_corrosion2d/20261107-spaced-six-01/inputs64`.
Four development references: sibling `local_dev4`. Final evaluation:
`/root/autodl-tmp/data/particle_corrosion2d/local_test64_20261111-01`.
All68development/test geometries excluded from training. Final references use
actual mesh spacing.01/256boundary points, versus training.015/128. The reference
manifest's common config alone does not describe these refined meshes. Existing
validation passes: four spatial checks phi0.00090–0.00109,c0.00122–0.00146,
and time refinement near1e-9. Training consumes no solved target fields.

Tests cover obtuse-triangle signed stiffness, constant nullspace, weighted mass,
manufactured linear boundary loads/orientation, mouth columns, duplicate-core
parameter gradients, overflow rejection, real-geometry paired inputs and journal
semantics. Initial27tests passed; expanded38tests (FEM, method validation, rolling
pool and independent online audit) passed after the journal adjustment.
Full initial-bank preflight:64geometries/512sets, maximum1457halo nodes,
maximum10neighbors,477negative signed coefficients retained, exact fixed data
and actual cores. Receipt:
`/root/autodl-tmp/tf-logs/ceramic-fem-preflight-20261113-01/summary.json`.
CPUpreflight took5.327s. No mesh trajectory solved.

Fixed known-geometry full-row feasibility completed2000updates. Warm update
median0.028854s, active peak7,318,385,152bytes, process wall294.589s including
compilation/evaluation. Physical-weighted phase/concentration error moved from
50.68%/50.35% initially to26.36%/28.62%; phase was25.40% at1500. All updates
finite, but this is not a converged solution or generalization result. Preserve
this limitation; the formal matched20k comparison has no final accuracy yet.
Single-case outputs and prescribed inputs are in the queue's `-fixed-fem2k`
run directory under `/root/autodl-tmp/tf-logs/particle_corrosion2d/`.

Remaining after the finite queue: independent saved-field error reductions,
FE-interpolated inference diagnostic, direct coarse-solver bias on a small
predeclared subset, flux/mouth balance and common-cohort convergence plots as
appropriate. These diagnostics are not claimed complete by the training launch.

## Session handoff

Changed numerical/API files: newfem_weak and three opt-in configs; dispatch in
inputs/online/train/problem; signed payload validation; initial-bank rebuild;
online and method pairing audits; canonical comparison fields. Original
losses/network/optimizer/physical defaults are unchanged. Added focused FEM
tests and finite resource-finalizer/preflight/fixed-probe/comparison scripts.
Updated example README,DEVELOPMENT,PROGRESS,DECISIONS,this task and stopped
channel task. Existing unrelated research edits are preserved. Expanded38tests
passed; after limiting FEM provenance to its opt-in selector,17focused tests
passed again. `git diff --check` passes. No Git operation or file cleanup.
Keep this task active until the queued comparisons and requested validation
are reported; inspect current journal before assuming any stage completed.


## FEM20k result and explicit continuation — 2026-11-13 08:11UTC

FEM training exited0 after1260.563s (~21.01min); all20k updates and final model/
optimizer exist. Online reconstruction passes399installed events,316new
geometries,1520spatial refreshes,380distinct sampled geometries, all64final
slots, zero heldout overlap and zero solved training fields.

Final64common local references, unweighted full-space-time norms per channel
then sample means (the user's primary metric):

| Channel | Mean | Sample SD |
|---|---:|---:|
| Phase |0.05553496385564092|0.0074704847864646585|
| Concentration |0.0659368978302109|0.008871891024512494|
| Per-sample channel mean |0.0607359308429259|0.008165872385282196|

Distinct supplementary physical-volume/time-weighted64case means are
phi0.0333363682,c0.0396889790; do not confuse these with the primary scores.
The four-case physical-weighted development phase decreases4.6335% at5k,
3.6468% at10k,3.4953% at15k,to3.4242% at20k. These are not a64case convergence
curve. Regional primary phase:8.90385%near particles,2.25389%bulk; near band
contributes89.09%of primary squared phase error. Physical-weighted contribution
is61.97%; no boundary-treatment cause is established from these numbers alone.

FEM warm optimizer step median42.993ms, mean47.596ms (including gradient-weight
refresh updates, excluding first compilation). First update187.304s;
refresh median148.160ms, held median42.939ms. Active peak14.0466GiB; reserved
allocator pool is different. Input/gather16.45s, online updates4.18s,
coretrace20.84s, evaluation48.35s and total1258.17s are separately recorded.

The original queue failed AFTER saving all64predictions/headline scores,
in regional metric replay: relative threshold2e-13 rejected a1.10675e-14
absolute norm difference. CPU replay matches every saved headline bitwise.
Extended-precision accumulation gives max relative saved drift3.22396e-14,
indexed-order drift2.19697e-13. Explicit regional replay tolerance is now1e-11;
no headline data, physical model, optimizer, checkpoint or source in the training
pair's required set changed. Four focused regional tests pass, including a
large-array partition and rejection of a1e-6relative corruption. All64regional
replays now pass. Receipt: bundle01/reduction_roundoff_audit.json; original
failed journal/log untouched. No GPU re-inference or FEM retraining performed.

Queue now supports explicit `--continue-from BUNDLE`, requiring the exact
completed-FEM/failed-regional state, successful64case recovery and initial run
provenance. No implicit retry. New owner101250, child101255 nonlocal20k, started
08:11:34UTC /16:11:34Asia-Shanghai. Bundle02 imports the completed FEM record
and schedules only nonlocal then localAD with original matching sources/draws.
Outer log `/tmp/ceramic-fem-comparison-20261113-02.log`. Read the current journal
before resuming; do not restart bundle01 or either stopped channel/Spinodal owner.

Task-end this session: postprocessing reduction audit tolerance/helper and its
focused test; explicit continuation in the finite queue; PROGRESS,DECISIONS,
and this task updated. Changed modules compile; focused tests and diff check
pass. No publication figures, training-default changes, cleanup or Git operation.
Remaining work stays active: complete/audit two matched controls, report final
relative errors/timing, then planned discretization/interpolation diagnostics.


## Nonlocal reproduction investigation — 2026-11-13T08:38:13.290275+00:00

User suspects an accidental change because the old nonlocal run converged.
Current nonlocal latest evaluation step19000: physical-weighted4case phase
32.85512%,c34.80950%. Matched FEM same step phase
3.39701%; historical nonlocal same step phase
3.33524%. These are development
monitors, not the primary final64metric. Full snapshot is bundle02/
reproduction-progress.json. Recent throughput including routine work~68.5ms/update;
no NaN or process failure is observed. The current recipe has not reproduced
historical convergence; retain this result without attributing causality.

Audit source: `docs/tasks/ceramic_nonlocal_reproduction_audit.py`. CPU-only
execution passed in13.908s; receipt and two identical symbolic update MLIRs:
`/root/autodl-tmp/tf-logs/ceramic-fem-comparison-20261113-02/historical-reproduction-audit/`.
Identical loss/operator/network/optimizer/normalization sources, cold bytes,
initial payloads and all planned slot/pool/time draws. Actual first10000core
coordinates/weights match historical bytes;270installed journals through13500
match except worker timing/cache. Old/current independent builder payloads at
updates50,100,250,6550,13000match bitwise. Identical CPU-lowered update graph
length959374characters; default-off FEM and stratified sampling branches are
inactive. Recorded config differences are checkpoint1000vs5000 and comparison
selectorrolling_fem_v1vsrolling_spatial_v1; current adds reserved64exclusion,
which has not changed the actual geometry stream. No accidental mathematical
or input change found by these checks.

The first-update CH value differs7.8523626328vs7.8523669243; weights differ at
FP32roundoff scale. Old GPU RTX4090, new PRO6000. Old environment explicitly
XLA_FLAGS='', compilation cachefalse, allocator90%; current flags/cache-dir
unset and allocator80%. No inherited deterministic/TF32override found. Current
Python/CUDA package versions cannot be inferred for the older run. Therefore
cross-device FP32/update sensitivity is a hypothesis, not a confirmed cause.

Frozen old-source replay is queued, not yet training. Owner115187 waits for
bundle02statuscomplete; if dependency fails/stops, it stops without retry.
Journal `/root/autodl-tmp/tf-logs/ceramic-old-source-pro6000-20261113-01/journal.json`.
Uses historical snapshot as working directory, historical canonical train.py,
original preset/seed and explicit launcher flags. Original historical source
entrypoint imports/--help pass on CPU. After20k it independently reconstructs
inputs, compares old cold states and complete sampling/initial/final arrays,
verifies64heldout geometry disjointness, then evaluates. No source snapshot
mutation or concurrent GPU training. This is one bounded reproduction control,
not an open-ended hyperparameter sweep or default change.

Task-end: added two CPU-audit/background-replay scripts; PROGRESS,DECISIONS,
this task updated. Actual input/symbolic-update audit and changed-module
syntax checks pass; git diff --check passes. Existing experiment unchanged.
Exact resume: inspect both bundle02 and frozen-replay journals; do not launch
another owner or report queued/final64results as already complete.


### Completed nonlocal arm, before session handoff

Nonlocal20k training, independent online audit, strict FEM/nonlocal actual-input
pairing and64case evaluation/region audit all pass. Primary final64 means/SD:
phi0.3199168507713878/0.007847636551792778;
c0.3362076481756415/0.0079083357917266;
channel mean0.3280622494735147/0.007877034789190178.
Total training1620.636s (~27.01min); warm update median59.827ms, mean66.301ms;
active peak20.660GiB. Four-case physical-weighted final phase32.8451%,
concentration34.7927%. LocalAD has automatically started; original-source replay
remains waiting for this comparison owner. Never mix four-case physical-weighted
and64case primary norms in the same unlabeled comparison.

Completed historical input audit extends the earlier prefix check: all20000
sampling/time/pool arrays, actual core coordinates/weights, initial/final payload
banks and all399installed physical geometry journals match the successful old
nonlocal run exactly. Receipt: bundle02/historical-completed-input-audit.json.
Thus the suspect mathematical/actual-sampling modification is not supported by
current evidence. Same-card old-source replay remains needed to distinguish
execution reproducibility; no causality claim yet. PROGRESS and this task were
updated with the completed arm; no additional numerical change or Git action.


## Status query — 2026-11-13T08:49:31.405133+00:00

Comparison owner101250 remains healthy. Nonlocal all four stages passed; localAD
child117897 has logged update1000/20000. First update456.295s includes
compilation/autotuning; logs explicitly show a slow GPU GEMM-fusion compilation,
then finite updates at~50ms. This was not eight minutes of optimizer updates.
Latest development evaluation step1000; no completed AD or new64case
AD score. The frozen historical-source owner115187 is healthy but waiting for
comparison completion, so it has not tested source reproduction on this GPU yet.
No new convergence/reproducibility conclusion follows from this progress query.

No process, numerical/configuration, default or Git changes. Only PROGRESS and
this active task updated; git diff --check passes. Continue reading both journals
before action; no duplicate owners, no extra GPU jobs, no stopped channel/Spinodal
restart. Existing bounded order remains AD20k/audits/64eval, then frozen replay.

## Status query — 2026-11-13 09:01 UTC

Comparison owner101250, AD child117897 and waiting frozen-source owner115187
remain alive. Latest journal snapshot09:00:59UTC records AD12700/20000updates;
the last complete four-case physical-weighted evaluation is step12000:
phi0.3338852376296349,c0.3358838897307419. No clear recovery is observed.
The latest update takes0.048653s; GPU utilization is77% at the process check.
The final64AD evaluation does not exist yet. Frozen historical-source replay
has not begun training, so it cannot yet resolve the source/backend hypothesis.
Completed FEM/nonlocal final64scores remain unchanged. Do not compare the
intermediate weighted four-case monitor directly against those primary64scores.

Task-end: only PROGRESS and this task's status/resume record changed; no new
experiment, process control, numerical/default change or Git operation.
`git diff --check` passes. Resume by inspecting both queue journals: finish
AD20k/audits/64eval, then the already queued frozen-source20k replay and its
input/heldout/evaluation audits. Preserve all failed and successful evidence.

## Three-arm completion and recovered historical-source trajectory — 2026-11-13

Bundle02 completed09:09:50UTC, with all eight new train/online/pair/eval stages
passing and the prior successful FEM result reused. Final20k primary64scores
use unweighted full-space-time norms per channel, then sample means:

| Method | Phase mean | Concentration mean | Channel mean | Training total (s) | Warm mean/median (ms) |
|---|---:|---:|---:|---:|---:|
| FEM weak |0.055534963856|0.065936897830|0.060735930843|1258.170|47.596/42.993|
| Nonlocal current replay |0.319916850771|0.336207648176|0.328062249474|1620.636|66.301/59.827|
| Local AD |0.322069916644|0.326059676687|0.324064796666|1704.924|56.773/49.127|

AD sample SDs are0.005235920097/0.006863686102/0.006046844148 respectively.
Its whole training child took1707.964s; active peak11.1821GiB. Warm timings
exclude initial compilation; total includes setup, compilation, monitoring and
I/O, but not independent final64evaluation. Final64AD evaluation took59.930s.
AD did not recover convergence. This result alone does not establish intrinsic
AD derivative inaccuracy or a pointwise Neumann-boundary causal explanation.

Frozen-source owner115187 automatically started child138508 at09:10:15UTC.
The old source/preset/launcher flags on the SAME PRO6000 now converge: four-case
physical-weighted phase6.6233%at5k,3.9523%at10k,3.1016%at15k,3.0655%at19k;
concentration3.7258%at19k. Latest logged update19300. This rejects the claim
that the current GPU necessarily prevents convergence. It does not isolate
source changes, flags/cache, GPU autotuning or FP32 optimization sensitivity;
multiple execution details differ and final original-input audit is pending.
Primary64evaluation is not yet available. Preserve the failed current replay.

Exact resume: inspect the frozen-source journal, its online/cold-state/full
sampling/heldout audits and final64evaluation. No new run, numerical/default
change, stopped-task restart or Git action. Memory updates: PROGRESS, DECISIONS
and this task. Existing bounded replay continues; no additional sweep scheduled.

### Frozen-source20k and64evaluation complete — 2026-11-13 09:38:24 UTC

All three frozen-source stages passed: train1608.454s process wall,
online reconstruction16.300s, final64evaluation59.706s process wall.
Training-reported total1605.919s (~26.77min). Original config, model/optimizer
cold bytes, all planned/actual20k draws and initial/final input-bank arrays
match the historical run exactly; heldout64disjointness and regional norm
replay pass. The journal explicitly records original_inputs_and_cold_state_identical.

Final20k primary64score means/SD:

| Channel | Mean | Sample SD |
|---|---:|---:|
| Phase |0.038174886259585886|0.005164982659342687|
| Concentration |0.04620518849296874|0.006276693367092963|
| Per-sample channel mean |0.042190037376277315|0.005719438928867052|

The historical same-seed primary channel mean was4.554058%; recovered current
GPU run4.219004%, FEM6.073593%, failed current-source run32.806225%, AD32.406480%.
This is a single paired initialization, not a robust multi-seed method ranking.
Four-case weighted final phi3.072872%,c3.731944% must remain separate from64
primary scores. No default, README publication or method-wide conclusion follows.

Final artifacts: `/root/autodl-tmp/tf-logs/ceramic-old-source-pro6000-20261113-01/`
contains the completed journal, eval64/summary.json, per-sample predictions and
regional_metrics.json. Training source/config/weights and audit live in the
same-name run under `/root/autodl-tmp/tf-logs/particle_corrosion2d/`.

Task-end: PROGRESS, DECISIONS and this task updated; git diff --check passes.
No numerical/process/default/Git changes. Both authorized queues are complete,
with no successor training. Keep this investigation active: causal isolation of
source/launcher/repeatability differences and planned interpolation/discretization
checks remain. Inspect these completed receipts before any follow-up; preserve
the failed run and do not silently substitute the recovered one into its pair.

## Requested result figures — 2026-11-13

User requested final-error distributions and/or convergence. Produced both,
retaining all four arms including the failed first nonlocal replay. This is not
a silent replacement of its matched control by the later recovered run.

- `particle_corrosion2d_fem_comparison_distribution.pdf/.png`:100×64mm,
  final20k seed42,64common heldout samples per arm, each point is mean(phi,c)
  of per-channel full-space-time Euclidean relative L2. Means4.2190%,6.0736%,
  32.4065%,32.8062% in plot order. Natural-log-error KDE, Scott bandwidth,
  three bandwidths of tail extension, equal maximum violin widths, inner
  quartiles; no observations omitted, no invented tail observations.
- `particle_corrosion2d_fem_comparison_convergence.pdf/.png`:170×68mm,
  phase/concentration panels, actual evaluations0,1000,...,20000. Four local
  development geometries, volume/trapezoidal-time weighted full-space-time
  relative L2 per channel; mean±sampleSD, no smoothing. These are explicitly
  not64case primary-metric curves. FEM descends sooner; recovered nonlocal
  approaches/crosses it late, while the two failed arms plateau near0.3.

External output directory:
`/root/autodl-tmp/tf-logs/particle_corrosion2d/publication-fem-comparison-20261113/`.
Complete config snapshots, plotted sample values/monitor arrays and typography
receipts are in its `provenance.json`; compact repository record is
`docs/tasks/ceramic-fem-comparison-figures.json`. CPU-only reproducible renderer:
`docs/tasks/ceramic_fem_comparison_figures.py`.

```bash
PYTHONPATH=/tmp/pino-pdf-audit JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m docs.tasks.ceramic_fem_comparison_figures \
  --output /root/autodl-tmp/tf-logs/particle_corrosion2d/publication-fem-comparison-20261113
```

Uses existing external PyMuPDF1.26.7 and Helvetica; Poppler22.02 font-check
binary extracted under `/tmp/pino-poppler-audit/`, without system/repository
dependency changes. Download source is recorded. All256per-sample metrics
reduced again from saved numerator/denominator sums; final mean/SD and all84
monitor means independently match. Both PDFs have embedded CID TrueType fonts,
6–7pt editable text, no raster images, automatic y limits, no clipped labels;
temporary-copy programmatic ordinary/math-label edits pass (not a GUI claim).
PNG and independently rasterized PDF previews inspected. Four outputs total
294234bytes. Plotted errors remain ratios, with no percentage-axis conversion.

Task-end: added renderer/compact manifest; updated PROGRESS, DECISIONS and this
task. Rendering/metric/font checks and git diff --check pass. No training,
inference, reference solve, default/README/site change or Git operation. Keep
the overarching numerical investigation active with the prior unresolved
causal-reproducibility and discretization/interpolation diagnostics; the
requested figure deliverable is complete.

### Three-method presentation revision — 2026-11-13

User explicitly removes first run and replay terminology from the figures.
The current presentation now contains Nonlocal PINO, Local AD PINO, FEM PINO
in that order (NPG red/cyan/green), plus the separate text FEM numerical
reference. The reference is not plotted as a fourth training curve or zero
error. All192retained final sample scores,63monitor evaluations and associated
statistics are exactly unchanged from the four-arm figure receipt.

Revised PDF/PNG pairs and full provenance are under the previous external
directory's `three-methods/` subdirectory. Original four-arm figures, numerical
evidence, provenance and now frozen `four_arm_renderer.py`/`four_arm_manifest.json`
remain at the parent. The compact repository manifest and active renderer point
to the revised presentation. The source/launcher qualification remains recorded
without replay/first-run wording in either displayed PDF.

Task-end: renderer/compact manifest, PROGRESS, DECISIONS and this task updated.
Three labels and reference label verified in both PDFs, all embedded6–7pt font/
editable text checks and numerical-identity checks pass; PNGs visually inspected.
git diff --check passes. No new training/inference, data deletion, numerical
default, README/site publication or Git action. The previous investigation's
remaining diagnostics are unchanged; this requested figure revision is complete.

## Approved distribution publication — 2026-11-13

User explicitly requests the accepted three-method distribution in this
example's README and website, results documentation, commit and push.
Publish only its PNG/PDF pair (94,162bytes total) under assets/pino-studies,
with shared manifest artifacts/ceramic_fem_comparison_publication.json.
Both public descriptions distinguish FEM PINO from FEM numerical reference,
state the64case final20k channel-mean point norm, mean/sampleSD and same-GPU
timing scope, and retain single-seed/execution qualifications. The canonical
default and older comparisons are unchanged; convergence remains external.

Release scope includes the opt-in signed FEM implementation, exact comparison
presets/audits, reduction-roundoff fix/tests and completed-run/figure provenance.
Unfinished Spinodal/channel implementation is excluded. Follow dev commit ->
regular merge main -> push main only. Preserve unrelated working-tree bytes;
use a clean temporary main worktree if overlapping documentation hunks require it.
Next: stage and validate exact release tree, confirm focused FEM regressions,
nine-page build/local links and asset identity, then publish and verify Pages.

Pre-publication validation passes on the exact staged-only tree:42focused
FEM/method/rolling-input/regional tests, nine-page build,328relative HTML links,
README/source links, JSON/Python syntax and file-size checks. The approved
PNG/PDF pair is copied byte-for-byte. Chromium1440px/390px previews show the
figure/PDF link correctly, with no page overflow or JavaScript errors.
Receipt: `artifacts/ceramic_fem_publication_validation.json`. The numerical
default and unrelated working-tree source bytes are preserved. Next: content
commit/regular main merge/push, then exact-commit Pages and live asset verification.
