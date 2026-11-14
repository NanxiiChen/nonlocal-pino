# FEM weak-residual PINO versus nonlocal and local AD

Date:2026-11-14. Status: complete. Requested ceramic, 3D melting and full-coordinate
channel FEM PINO comparisons and validation are finished. Ceramic publication
was completed earlier; the new melting/channel additions remain local and have
no current Git authorization. The channel finite queue finished all eight stages
at2026-11-14T00:53:55UTC. GPU is idle. Production defaults are unchanged.
Optional FEM interpolation/residual-norm diagnostics remain deferred and do not
block completion; no automatic new experiment. Historical notes below preserve
their original execution context and authorization at the time.

## TODO and execution gate

- [x] User authorizes executing this comparison after channel resource tests,
  explicitly cancelling channel long training.
- [x] Verify the signed FEM weak-residual implementation and run the fixed-geometry
  feasibility check described below.
- [x] Run the paired nonlocal/local-AD/FEM comparison and64sample evaluation.
- [x] Complete the frozen historical-source reproduction control and its64case audits.
- [x] Publish the approved three-method distribution, results and provenance;
  commit/push and verify GitHub Pages.
- [x] Identify within-executable GPU update/scatter nondeterminism and verify
  old/current source parity with deterministic30update controls.
- [x] Correct the stale default physical preset and test fixed-linear adjoints.
- [x] Finish canonical-default cold20k, full-input audit and final64validation;
  promote only with acceptable convergence and record cost/limitations.
- [x] Publish the validated current default, update README/site SOTA summaries
  and verify the GitHub Pages deployment; preserve unrelated pending research.
- [ ] Deferred: remaining FEM discretization/interpolation diagnostics are separate from
  this default-repair acceptance; do not infer a universal ranking.

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

### Publication verification and task-end — 2026-11-13

Content commit `b00510f` was regularly merged and pushed to main as
`132b789cee557c181993dbd27789fd138efeadd4`. GitHub Pages run
[43416337906](https://github.com/NanxiiChen/nonlocal-pino/actions/runs/43416337906)
built and deployed successfully. All nine live HTML pages and the approved
PNG/PDF pair returned HTTP200 and match the inspected release build byte-for-byte.
[Published comparison](https://nanxiichen.github.io/nonlocal-pino/particle-corrosion.html#spatial-derivative-comparison).

The release includes the example README, website/build workflow, curated
94,162byte figure pair, direct-path source/config/metric manifests and the
related opt-in FEM code/audits. Numerical defaults are unchanged. All58unrelated
or partially staged files match their pre-publication byte copies, including
unreleased Spinodal sections of shared documentation. No weights/reference
arrays were added. Validation:42focused tests,328HTML links,89Markdown links,
desktop/mobile previews, syntax checks and git diff --check pass.
Machine-readable receipt: `artifacts/ceramic_fem_publication_validation.json`.

This requested publication is complete. Memory updated: PROGRESS, DECISIONS
and this task. The overall investigation remains active only for the previously
listed causal reproducibility and discretization/interpolation diagnostics;
no additional experiment is queued. Resume there rather than repeating final64
inference, redrawing the accepted figures or restarting stopped sampling runs.

## Reproduction-cause clarification — 2026-11-13

The user asks whether recovery from the frozen snapshot identified a new-code
bug. Re-read the original historical-reproduction and completed-input receipts,
the successful frozen-source journal, and both launcher implementations.
Answer: no specific faulty source change or causal execution setting has yet
been isolated. The successful control changed source/preset and launcher
settings together. Explicit compilation-cache disablement and90%allocation
versus inherited cache settings and80%allocation are actual differences, not
proven explanations. CPU-lowered graph identity is not a GPU executable audit.

Verified evidence remains: shared core source bytes, initial model/optimizer,
all20000actual sampling arrays and399physical input events match; same current
GPU produces32.8062%with the failed current-source execution and4.2190%with the
frozen-source/launcher combination on the identical64case metric. No claim of
a fixed bug, a source-only effect or general hardware failure is justified.

Exact next diagnostic, if continuing causal isolation: keep current source and
recorded inputs fixed, apply the successful launcher settings, and first compare
initial forward losses, gradients and updates in fresh processes against the
frozen source under identical settings. Then isolate any remaining source or
execution differences one at a time, including repeatability before interpreting
a single trajectory. This is a proposed next check, not a launched experiment.
Task-end: only this task and PROGRESS updated; no numerical edits, new training,
Git operation or change to the published figures. git diff --check passes.

## Authorized cause isolation and default repair — 2026-11-13

User now explicitly requests locating the cause and making the current default
entrypoint select the best validated nonlocal recipe. Acceptance: isolate source
versus launcher/repeatability using identical saved cold inputs, verify the actual
train.py/default path with cold training and the common64cohort, document limits
and add focused regressions. No Git action is authorized this turn.

Initial inspection finds a separate definite default-routing defect:
configs/train_default.py imports train_physical (eight particles, width192,
T_c5s/high mobility), whereas the accepted result uses train_physical_spaced_w256
(six separated particles, width256,T_c350s/moderate mobility). This does not explain
the failed explicitly configured20k comparison, but must be fixed for the user's
default-entrypoint requirement after validating the accepted current-source path.

Current GPU is idle PRO6000Blackwell97,887MiB;37GiB disk free. Preserve earlier
runs/weights and unrelated dirty work. First checks isolate current/frozen source
and launcher on recorded first batches, then use bounded complete cold validation.

### First isolated execution evidence

Bundle `/root/autodl-tmp/tf-logs/ceramic-execution-isolation-20261113-01/`.
Current source with the old launcher twice still yields different30step
trajectories. More decisively, five identical calls to one compiled first-update
function produce nonidentical state/model outputs (combined maximum absolute
difference6.61e-5..9.05e-5); this is within one process with no code, cache,
allocator or input change. The initial replay flags alone do not ensure equality.

An isolated graph_action VJP on the actual first core/halo stencil exposes
scatter-add in the backward JAXPR. Twenty identical executable calls differ
(19/19comparisons); relative error against a float64 CPU transpose oracle is
about6.08e-8, consistent with accumulation-order roundoff, not an incorrect
operator formula. This identifies one concrete source of execution perturbation,
not yet proof that it alone causes the32.8%versus4.2%final gap.

Deterministic first-update/source controls are running next. Official XLA
[determinism](https://openxla.org/xla/determinism) distinguishes autotuning from
execution-time atomic scatter; JAX [indexed updates](https://docs.jax.dev/en/latest/_autosummary/jax.Array.at.html)
notes nondeterministic conflicting sums. Use the measured controls before
attributing the final convergence or promoting a potentially slower policy.

### Source isolation passed; bounded fixed-adjoint control

With identical deterministic execution, current and frozen historical sources
produce byte-identical model AND optimizer states after1and30updates; all30loss
and balancing-weight vectors match. A second cold30update trajectory in the
same frozen-source executable is also identical. This includes the gradient
weight refresh at update26. Receipt: execution bundle/source-isolation.json.
Global deterministic updates cost1.35–1.38s, so do not promote that slow policy.

Added an opt-in dense_fixed adjoint candidate: original edge-difference forward
operator, D^Tfield pullbacks and C^Tcore-gather pullbacks with duplicate sampled
cores retained. Selected-batch matrices only; input pool and quadrature unchanged.
Independent float64 transpose and coupled-residual field-gradient oracles pass.
The candidate additionally declares autotune_level0 to eliminate compile-time
kernel-selection changes; inherited conflicting flags fail explicitly.
Full GPU repeatability/timing probe is now running before any cold20k trial.
Default import now selects the documented width256/six-particle original recipe;
this optional adjoint is not yet the default. Updated CLI regression passes.

### Canonical cold validation started

The fixed-adjoint/autotune0 probe passes all5identical first-update calls and
two identical cold30update trajectories, including byte-identical final models.
Warm update~0.105s, versus~0.062s ordinary atomic transpose and~1.35s global
determinism. Six focused config/adjoint tests pass, prior14loss/runtime tests pass.
Default is provisionally routed to train_physical_spaced_w256_fixed_adjoint;
no final convergence claim yet. Historical configs remain unchanged.

Owner203256 started the canonical CLI **without --config** from a fresh seed42
initialization. Run:
`/root/autodl-tmp/tf-logs/particle_corrosion2d/ceramic-default-fixed-adjoint-20261113-01/`.
Owner journal: execution bundle/default-cold/journal.json. It is bounded to20k,
independent online reconstruction, exact cold/full-input matching to the prior
failed run, and the same64case final evaluation. No reference labels enter loss.
Expected~35–40minutes; no other experiment queued. Resume by inspecting this
journal and train log; do not duplicate or report intermediate4case monitoring
as the final64metric. Keep default pending until the result is verified.

### Canonical default accepted — 2026-11-13 13:21 UTC

Owner203256 and all three children exited0; journal status is `complete`.
No training or inference remains queued. The actual no-config train.py entrypoint
completed20,000cold updates. Final model, common64local-FEM cohort:

| Run | Phase mean | Concentration mean | Channel mean |
|---|---:|---:|---:|
| Earlier unsuccessful current-source control |31.991685%|33.620765%|32.806225%|
| Historical-source recovered control |3.817489%|4.620519%|4.219004%|
| Current canonical fixed-adjoint default |3.696453%|4.483867%|4.090160%|

The metric is the unweighted Euclidean norm across the complete space-time
evaluation array per channel/sample, followed by channel and sample means.
Current channel-mean sample SD is0.548288percentage points. Four-case
volume/time-weighted development phase2.904612%,c3.538808% are separate values.

Acceptance checks:

- Canonical default's JSON-normalized config equals the saved training config;
  all93saved Python source files equal current source byte-for-byte.
- Initial model AND optimizer bytes, all20k planned/actual draws and complete
  initial/final input arrays equal the unsuccessful paired control.
- Independent online reconstruction passes:399installed events,316new geometries,
  1520spatial refreshes, all64final slots reconstructed, zero evaluation overlap
  and zero future-field labels. All64saved-field regional norm checks pass.
- Earlier source isolation and repeated fixed-adjoint GPU updates/trajectories
  pass;40combined targeted tests and a separate full-network float64 gradient
  oracle pass. Prior14loss/runtime checks also pass (overlapping test scopes,
  not55unique tests). The forward five losses agree exactly in the oracle;
  all network-parameter gradients agree to2e-11.

Current training process wall2563.101s (42.718min), internal2560.619s,
compile-first-update169.044s. Warm update median102.362ms/mean114.843ms;
including sampling/online preparation median103.150ms/mean115.948ms.
Active peak21,814,555,136bytes (20.3164GiB);90%allocator reservation is separate.
The historical recovered26.77min run remains faster. Do not silently reuse its
timing for this default or rewrite the already-published comparison figure.

Confirmed diagnosis: stale user-facing preset routing plus measured original
GPU reverse-accumulation nondeterminism. The stale route is NOT the cause of
the explicit-preset failure. Current/frozen source equality under controlled
execution rejects the claimed changed-PDE/optimizer explanation. Exact fixed
field adjoints and autotune0 together restore convergence with unchanged
forward physics and actual training inputs. Their individual contributions to
the old final-error gap remain unquantified. One seed/sameGPU validation is not
a cross-seed/device guarantee; the strategy still has no fixed seed field.

### Task-end record and exact resume point

Current request is complete. The actual default selects the six-particle
width256,350s,Adam20k,rolling-core/halo recipe with dense_fixed field adjoints
and autotune0. Added execution provenance, supported-shape validation and focused
default/gradient tests; retained historical presets. README,DEVELOPMENT and the
direct-path `artifacts/ceramic_execution_validation.json` record the results.
Updated PROGRESS and appended the promotion decision. No Git operation,
reference generation, figure replacement or stopped-task restart occurred.

The larger comparison task remains open only for optional FEM discretization/
interpolation diagnostics. Do not automatically launch them. Resume from this
receipt if the user asks; do not rerun the completed cold20k default validation.

### SOTA publication ready

Latest user authorizes commit/push and README/website synchronization. The
current SOTA is explicitly identified at the start of the example README and
website chapter; both overviews distinguish4case weighted phase2.9046%/5.4357%
from64case unweighted phase3.6965%,c4.4839%,channel mean4.0902%. Launch examples
use the canonical default and retain direct paths to historical presets.
Historical field/comparison PNG/PDF files and their numerical values are unchanged.

Scoped25files are isolated from57unrelated files plus the uncommitted Spinodal
sections of DEVELOPMENT. Backup/staged snapshot:
`/tmp/ceramic-sota-release-vayyozyu/`. All93training source files match the staged
tree;6focused entrypoint/gradient tests pass in3.801s. Prior full cold/gradient
validation remains valid. Nine pages build with330local links, unchanged
comparison images, desktop1440/mobile390without overflow or script errors.
Release receipt: `artifacts/ceramic_sota_publication_validation.json`.
Next: commit on dev, regular merge into main, push main; verify exact Pages
commit and live HTML, then record completion without starting experiments.

### SOTA publication complete and task-end

Content95bce5b merged normally to main4c2ef61 and pushed. Pages43436212919
completed build/deploy successfully. At2026-11-13T13:45:18UTC all9live HTML pages
and both original comparison PNG/PDF files returned200 and matched the inspected
build byte-for-byte. Current SOTA/default/results and reproduction commands are
now synchronized across the root/example READMEs and report overview/case page.

Published25scoped files include the prior validated default fix, execution and
gradient code,6focused release checks, direct validation provenance, website
text and memory. No new images, training, reference solves or dependencies.
All57unrelated files and the remaining partial DEVELOPMENT edits are unchanged.
The receipt now records deployment identity and URLs; PROGRESS/DECISIONS are
updated. This requested publication is complete. Optional FEM diagnostics remain
deferred in this same task; no experiments or further deployment are queued.


## Extension: 3D melting weak-operator pilot

User authorizes a second FEM-mesh PINO comparison, suggesting melting. This
extends the existing task, without reopening stopped channel/Spinodal work.
No commit/push authorization applies to this turn. Do not choose a deliberately
poor FEM mesh or tune outcomes to favor nonlocal.

Acceptance: original cube128^3vertices,6Kuhn tetrahedra/cell, exact P1element
stiffness and row-sum lumped mass. With nodal reaction quadrature, weak rows are

$$
M_L\dot\phi+N_1K\phi+M_L[N_2(\phi^3-\phi)+N_3(1-\phi^2)]=0.
$$

Normalize each row by its lumped mass and tanh Jacobian, as in the production
latent residual. Preserve stable pairwise tanh ratios, including at walls;
no explicit Neumann penalty. All6faces have natural zero flux. Interior
stiffness on this uniform Kuhn mesh reduces to axis-neighbor couplings; this is
not reason to assume a costly or inaccurate3D FEM PINO. Independently compare
assembled rows/mass to a brute-force tetrahedral assembly on a small cube;
check symmetry,constant nullspace,positive mass,total volume,affine energy,
interior quadratic consistency and actual transformed residual parity.

Run two cold10k arms through train.py, same architecture/Adam/B16/Nt8/rolling
256cores, identical seed42initialization and recorded core/condition/time draws.
Disable GPU kernel autotuning in both explicit comparison presets; original
production default stays unchanged. Probe time/memory before long runs. Save
final weights and actual draws; audit all inputs. Evaluate8native test cases
and existing localCG2case005separately with complete space-time norms; no
native-vs-native result is evidence of lower continuum error. Existing local
reference is accepted_by_user, not strict_refinement_passed; its time/space
uncertainty limits sub0.1%ranking. No new FEM MPI solve concurrent with GPU.

References: existing ceramic weak-form derivation, independent element checks,
[FEINN](https://arxiv.org/abs/2306.06304) and
[FEniCSx heat weak form](https://jsdokken.com/dolfinx-tutorial/stable/chapter2/heat_equation.html).
This is lumped/nodal FE algebra in an operator-learning loss, not a complete
reproduction of FEINN's Riesz norm or a direct numerical FEM time solve.

Initial status before implementation: operator validation and launch pending.


### Melting implementation and launch

Three independent FEM tests and the original complete-row/rolling-sampler
regression pass (4tests,total0.734s on final check). Production128^3assembly
uses12,290,298implicit tetrahedra and12,484,608directed nonzero couplings;
0.571s CPU assembly,positive masses summing to1.0000000000000009. LocalCG2and
native case005points/times/parameters match exactly. Native bank has2monitor
and8test cases; no new references were generated.

Owner6152 launched the finite probe/probe/nonlocal10k/FEM10k/evaluation queue:
`/root/autodl-tmp/tf-logs/melting-fem-comparison-20261113-01/journal.json`.
Prefix `melting-fem-comparison-20261113-01`; all runs under the existing
icemelting3d log root. Probe active-memory gate75GiB; current card96GB.
Latest user asks whether3DFEMmust be slower because it needs the whole mesh:
answered no; weak rows and their complete FE support can be sampled. A fullFE
versus sampled nonlocal comparison would confound implementation and sampling.
Both planned arms use256cores and identical complete halos; no full-mesh arm
is added merely to manufacture a disadvantage. Default/source snapshot tests
and final complete-input audit remain mandatory.


### Melting final results and task-end

The finite queue completed both cold 10k controls and the independent final
input/metric audit. All 41 saved Python source files are identical across the
two runs and the current checkout. Initial network/optimizer bytes and all
recorded condition/time/core inputs match; every core draw independently
reconstructs. Four focused operator/sampling tests pass. No source change was
made during training. The postprocessor additionally verifies local physical
parameters and finite nonzero norm denominators.

| Final-weight measurement | Nonlocal PINO | FEM PINO |
|---|---:|---:|
| Native eight-case phase mean | 0.178980% | 0.074465% |
| Native sample SD, percentage points | 0.035126 | 0.015594 |
| Local CG2 case 005 phase | 0.115568% | 0.193416% |
| Local CG2 case 005 terminal phase | 0.167437% | 0.241246% |
| Warm update median | 12.054 ms | 12.088 ms |
| Including preparation median | 14.098 ms | 14.216 ms |
| Whole training process wall | 276.726 s | 277.097 s |
| Active peak | 1.45475 GiB | 1.30757 GiB |

Norms use all five saved times and 24^3 diagnostic nodes, then average samples;
only phase is a solution channel here. The independent local reference is
case 005, accepted by the user with its strict refinement flag still false.
Native/local reference difference on that case is 0.147448%. The opposite
rankings against native and local targets do not establish general method
superiority or a lower converged error floor. A single final weight can contain
optimization error that partly cancels the numerical-reference difference.

Whole-process time includes startup, graph preparation, compilation, monitoring
and final outputs. Warm update excludes those costs; the inclusive warm measure
adds condition/time/core preparation only. GPU allocator reservation is not the
active peak. Both use full graph precomputation but only complete sampled rows
and their halos for neural queries. Six surviving stiffness neighbors on this
uniform Kuhn mesh explain the similar workload; neither method needs whole-mesh
neural evaluation per update. Production N128 FE assembly was 0.571 s on CPU.

Diff-level scope: new opt-in FEM graph, two explicit comparison presets, dispatch
and execution provenance hooks, three operator tests, finite launcher/evaluator,
README derivation/results and direct-path validation receipt. Updated
DEVELOPMENT, PROGRESS and DECISIONS. Original defaults, references, existing
figures/site and unrelated dirty files are unchanged. No Git action occurred.

Exact reproduction bundle:
`/root/autodl-tmp/tf-logs/melting-fem-comparison-20261113-01/`.
`journal.json` is complete; `comparison.json` retains every per-case error.
`artifacts/melting_fem_comparison_validation.json` carries direct checkpoint,
reference, config command and source paths without content hashes.

This requested melting pilot is complete; no jobs remain queued. Optional
further FEM diagnostics remain deferred. Resume only on a new request, from
this receipt, without repeating the completed runs. A broader local-reference
cohort or another mesh topology would be a separate experiment and must not be
presented as already measured.

Final checks: `git diff --check` passes; launcher/evaluator syntax, validation
JSON and both checkpoint paths pass. The GPU compute-process list is empty.


## Extension: full-coordinate random-wall NS–CH

User requests a different case, suggesting full-coordinate two-phase flow and
reuse of previous nonlocal results. Historical nonlocal full20k uses the same
PRO 6000 Blackwell model but GPU UUID c55d506d; current UUID is 1feff23f with
580.95.05 driver. Reuse the completed accuracy record with its limitations;
remeasure the unchanged nonlocal update on the current card before speed claims.
No existing stopped queue is resumed and no Git operation is authorized.

Plan: keep 192x80 subdivisions / 15,633 network nodes, width192/B8/Nt36/time18,
SOAP20k, hard IC/inlet/wall-flow lift, time JVP and rolling 16-slot geometry pool.
Use a 96x40 coarse cell mesh split into quadratic isoparametric triangles: its
P2 nodes coincide with all original queries. Phase, chemical potential and
velocity interpolate in P2; pressure uses P1 corner values. Spatial weak terms
follow the existing independent solver, including symmetric viscous stress,
conservative CH advection with outlet flux, and natural zero wall/outlet phase
and chemical flux plus zero outlet traction. Essential test rows vanish at the
inlet and, for velocity, at walls. Analytic streamfunction incompressibility
remains; no extra continuity penalty. Interpolated FE velocity has only
approximate discrete divergence; document and diagnose that distinction.

Use six-point-per-axis Duffy quadrature and squared weak rows in the inverse
consistent-mass-diagonal metric, normalized by mesh area and unchanged physical
scales. P2 vertex row-sum masses can vanish, so do not reuse P1 mass-lumping or
silently divide by a floor. This is an explicit FE residual norm, distinct from
pointwise MSE and not an exact mass-Riesz solve. Boundary conditions move from
five explicit penalties into four weak PDE terms; do not call it an identical
nine-loss objective. Keep the FE numerical reference independent.

Acceptance: independent Vandermonde P2 basis/element checks, positive geometry
Jacobians and mass diagonals, assembly/adjoint tests, constant-carrier outlet
flux balance and constant-pressure outlet traction checks; actual loss/parameter
gradient and finite GPU SOAP probe before cold training. Record paired initial
states, unchanged encoder/geometries/time draws and final same three reserved
local-FEM evaluations. No claim that FEM must be worse or slower.

Status: formulation and input audit complete; implementation next.


### Channel implementation and finite launch

Added `fem_weak.py`, the optional `train_operator_fem_weak.py` preset and a
nonlocal full-coordinate timing preset preserving the historical20k LR. Small
changes route the declared FE method through the canonical data/loss/online/
validation/training paths. Default numerical paths are unchanged. Four new FE
checks and12existing integral/budget/online regressions pass. FE checks include
an independent higher-order quadrature oracle, outlet carrier transport,
constant-pressure traction, reverse field derivatives and actual network
parameter gradients. Final quadrature geometry validation also checks vertex
Jacobian extrema; the mapped x-coordinate is affine, so det(J) is linear.

Finite owner33893 launched:
`/root/autodl-tmp/tf-logs/channel-fem-full-20261114-01/journal.json`.
Its stages are geometry screen, nonlocal60, FEM60, cold/input/memory qualification,
FEM cold20k, saved-run/online audits, same3reserved FEM cohort, final comparison.
No retries, new numerical reference solves or unbounded sweeps. Ordinary FP32,
97%allocator, no explicit XLA flags and compilation cache disabled; current
GPU UUID1feff23f, driver580.95.05. Actual active peak must be below95%of allocator
limit before long training. No change to production defaults or Git history.

The initial16 plus802online input screen passes in168.46s with no PDE solves;
minimum recorded quadrature Jacobian5.7163e-4 and mass diagonal1.7361e-5.
A subsequent validation-only vertex-extremum check is performed on every actual
production geometry; it does not change element arrays. The screen's reports
retain their original recorded minima. Nonlocal resource probe34352 is running.

Formulation references: the existing independently checked coupled solver,
[FEniCSx Navier–Stokes weak stress form](https://jsdokken.com/dolfinx-tutorial/stable/chapter2/navierstokes.html),
and [DOLFINx mixed CH weak form](https://docs.fenicsproject.org/dolfinx/v0.9.0/python/demos/demo_cahn-hilliard.html).
No IPCS time splitting from the tutorial is copied; physical time remains JVP.


### Channel resource progress and session resume

Nonlocal60 completed through canonical train.py. Current-card median update is
1.58446s, batch preparation0.01445s, inclusive1.59930s. Active peak85.3468GiB;
first update including compilation380.190s; full short child approximately
609s. This closely matches the historical1.608s warm update, but the historical
9.16h accuracy run remains separately labeled because its physical GPU differs.
Cold model/full SOAP bytes match exactly. No residual/physics fallback occurred;
slow GEMM compilation/autotuning warnings are retained in the log.

FEM production probe52267 is now running under owner33893. Upon success, the
owner compares both60-update condition/time/geometry streams and initial encoder
payloads to the historical run, checks active-memory headroom, then starts the
single cold20k FE run. It subsequently performs saved-field/hard-constraint and
full rolling-input replay, the three reserved references, independent norm
reductions and current-source checks. No additional run or automatic retry.

Task-end state: incomplete numerical comparison, healthy finite background
owner. Exact resume: inspect the journal above, then its active stage log. Do
not start a duplicate process or change numerical sources while it trains. If
failed, preserve the journal and fix only the identified stage. Final accuracy
and FEM timing remain unmeasured at this checkpoint.

Diff-level changes this session: opt-in FE assembly/weak loss and two configs;
canonical data/loss/online/validation/timing hooks; independent four-test suite;
finite queue and support audits; example README, DEVELOPMENT and direct-path
validation receipt. Memory updated: PROGRESS, DECISIONS and this task file.
Four final FE tests pass in10.812s;12existing integral/budget/online tests pass
in33.069s. No default promotion, reference solve, figure/site update, Git action
or deletion. All unrelated pending changes are preserved.


### 2026-11-14 channel status: resource checks passed, full training underway

Both production60 probes and qualification passed. Nonlocal/FEM inclusive warm
medians are1.59930/1.41693s, active peaks85.34683/80.81597GiB. The FE short update
is about11.4%less time; this does not establish faster convergence to equal L2.
FEM first-update compilation/call261.806s, whole probe446.846s. Cold network and
complete SOAP states, all60actual condition/time/geometry draws and initial
encoder fields match the historical full20k record.

Owner33893 advanced automatically to the authorized FEM cold20k; trainer66341
is healthy and using the GPU. Latest observed loss record: step200,
total0.87908679 (initial46.96917). The actual long-run cold states also
match the historical control byte-for-byte. Step25 peak80.78411GiB. The first
500step development evaluation has not yet occurred; no trained-model L2 is
reported from the initial step0 fields. Reserved three-case metrics remain a
post20k stage. A rough remaining duration from the measured warm rate is8hours,
not a guaranteed completion time.

Task-end: read-only process/log/state checks and documentation/receipt updates
only; numerical source and active processes are unchanged. Updated PROGRESS,
this task record and artifacts/channel_fem_comparison_validation.json. Resume
from the same journal, inspecting train-fem then validate/cohort/finish; do not
launch a duplicate run or restart completed probes. No Git action or tests
needed for this status-only update beyond diff/schema inspection.


### 2026-11-14 01:05 CST — first channel development evaluation

Owner33893/trainer66341 remain healthy. First500update evaluation uses the
same development local-FEM fields and13saved times as historical nonlocal:

| Relative L2 | Historical nonlocal500 | FEM PINO500 |
|---|---:|---:|
| Phase | 15.5824% | 15.8759% |
| Terminal phase | 18.1845% | 18.5917% |
| Velocity | 6.6569% | 7.0502% |
| Pressure | 13.2468% | 15.5704% |
| Chemical potential | 112.2325% | 114.7555% |

Phase includes all13times; velocity/pressure/mu exclude t=0, matching the
existing evaluator. The phase errors are similar at this early budget; FEM
has somewhat larger flow errors. This does not establish final method ranking.
FEM loss decreases46.9692 ->0.8791at200 ->0.3464at500. Current GPU short-probe
inclusive times remain1.5993s nonlocal/1.4169s FEM; do not replace them with
historical process timing or claim final time-to-accuracy improvement.

Task-end: status-only inspection and record update. Updated PROGRESS, this
task and the direct validation JSON with same-step historical metrics. No
source/process changes, retraining, reference solves, Git actions or new tests.
Exact resume remains the finite queue journal; await500-step monitoring and
final20k/three-case evaluation. No action needed while the healthy trainer runs.

Final live check at01:09CST: training has passed600updates, loss0.315931;
latest full L2 evaluation remains500. Owner heartbeat remains healthy.


### Status check 2026-11-13T17:27:43.825442+00:00

Owner 33893 / trainer 66341 remain healthy; latest logged step 1400/20000,
loss 0.21179584. Latest L2 evaluation is step 1000 on the same single
development geometry and local-FEM reference used by historical nonlocal.

| Relative L2 | Nonlocal at 1000 | FEM PINO at 1000 |
|---|---:|---:|
| Phase | 10.2156% | 10.1940% |
| Terminal phase | 12.8250% | 12.3131% |
| Velocity | 6.3173% | 6.1448% |
| Pressure | 10.0265% | 8.9004% |
| Chemical potential | 69.4148% | 72.5589% |

From 500 to 1000 FEM phase falls 15.8759% -> 10.1940%; pressure
15.5704% -> 8.9004%. Phase is almost equal to the same-step nonlocal,
velocity/pressure slightly lower, and chemical potential slightly higher.
These are early development results, not final generalization or method ranking.
The 500->1000 monitoring interval takes 714.701 s, or 1.4294 s/update
including interval evaluation overhead. Current-card short resource medians
remain 1.59930 s nonlocal / 1.41693 s FEM. Remaining training is roughly
7.4 hours at this rate; completion/evaluation overhead may extend that.

Task-end: only PROGRESS, this task record and the validation JSON changed this
session. No numerical/source/process/default changes, new tests, figures or Git
actions. JSON and diff-whitespace checks pass. Exact resume: inspect the same
finite queue journal and 500-step monitor records, then automatic validate,
three-case cohort and final audit after 20k. Keep historical stopped jobs stopped.


## Final channel comparison and task-end — 2026-11-14

Finite owner33893 completed all eight stages at00:53:55UTC; GPU and process
inspection confirms no remaining job. Both20k final weights use the same three
reserved geometries, numerical FEM references and evaluation points/times.

| Mean relative L2 | Nonlocal PINO | FEM PINO |
|---|---:|---:|
| Phase | 1.6435% | 4.6465% |
| Terminal phase | 2.1961% | 4.0764% |
| Velocity | 1.5172% | 4.3892% |
| Pressure | 5.2840% | 7.1489% |
| Chemical potential | 13.9541% | 43.7908% |

FEM per-geometry phase errors:4.3046%,4.7375%,4.8974%; nonlocal:
1.9860%,1.4293%,1.5152%. FEM pressure worst11.9478% versus nonlocal6.5885%.
All three phase/velocity cases favor nonlocal. Development-phase errors at
5k/10k/15k/20k: FEM4.696/4.471/4.350/4.372%; nonlocal3.377/2.610/2.037/1.450%.
The late FEM phase plateau is visible; its finite nonzero training residual
and interpolation/norm differences prevent assigning a unique cause.

Current-card resource60 probes: inclusive median1.59930s nonlocal/1.41693sFEM,
active85.34683/80.81597GiB. Full FEM process28910.367s(8.031h), training-wall
28885.714s, inclusive median1.418664s, final active peak80.91628GiB.
Historical nonlocal training-wall32985.807s(9.163h) uses a different physical
GPU UUID, so whole-run speed is qualified. This comparison shows an accuracy
advantage for this nonlocal recipe, not an update-time advantage.

Cold network and fullSOAP states, all20000condition/time/geometry draws,
initial encoder fields and114saved/current Python sources match. Final norm
reductions independently match both cohorts.399refreshes/798actualonline
inputs and final16slot bank reconstruct, with no reference labels or overlap.
Saved-field replay gap3.58e-6, JVPgap1.67e-6, wall speed6.42e-7 and neural
AD-curl divergence2.98e-8 pass the existing validation thresholds.

Retain the FE interpolation distinction: quadrature velocity-divergence RMS
at positive times is roughly0.0028--0.0044 (maximum0.8223), despite exact neural
streamfunction incompressibility. This is a diagnostic, not a failed hidden
pointwise-divergence gate or established explanation of the error gap. Four
weak losses/approximate dual norm and natural BCs differ from nine nonlocal
pointwise terms. Single initialization and three reused geometries do not
establish population ranking. Do not promote, retune or restart automatically.

Task-end changes: finalized direct-path validation receipt and channel README;
updated PROGRESS/DECISIONS and archived this completed task. The old task path
retains only a forwarding link so published provenance URLs remain valid.
No numerical source, defaults, processes, figures, website content or Git
history changed. Previously completed numerical checks are not rerun for this
documentation-only update; JSON, result-table, link and diff checks pass.
Exact future resume is the comparison.json and validation receipt; optional
norm/interpolant diagnostics require a new user request. Stopped sampling
experiments remain stopped. No required work remains from this comparison.
