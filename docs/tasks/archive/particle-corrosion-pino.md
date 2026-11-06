# Random ceramic-particle corrosion PINO

Date: 2026-11-06. Status: complete; stronger eight-particle nonlocal PINO,
independent local references, frozen-cohort evaluation and matched AD controls.

## User objective and acceptance

Complete nonlocal physics-only neural-operator corrosion with random inert
ceramic particles. Target mean relativeL2 1%; if repeated attempts cannot reach
it, accept qualified3%or5% results. Compare nonlocal versus local spatial AD
accuracy and timing under matched architecture/training choices. User's active
goal explicitly authorizes continuing beyond input previews. No new Git action
requested this turn. RTX5090,32607MiB available; FP32 ordinary compilation.

Evaluate both methods against the SAME independent local reference, separating
native nonlocal discretization from neural error. Use predeclared unseen
geometries, report phase/concentration pooled and terminal relativeL2, corroded
region/interface measures, compilation/warm timing and active memory. Positive
time reference fields must never enter training or checkpoint selection.

## Established context

Latest user steering: increase the particle count and/or concentrate particles
near the pit because the six-particle fields look too similar. Current prior is
eight particles, at least four within.12of the initial pit. Whole-geometry
rejection exhausted its explicit budget in two attempts; the new explicit
near_first placement conditions the first four placements, then shuffles all
particle identities. This is a documented changed sampling distribution, with
unchanged size/shape/clearance and PDE parameters. Eight independent test draws
pass in8.77s total (max3.10s), with four near particles, area11.33--15.61%, minimum
pairgap.03622. Old six-particle artifacts below are retained mild controls.

Prior [publication/preview](heterogeneous-publication-and-particle-preview.md)
has published material pilot at main c12b8c3; current dev source is uncommitted
geometry extension. Particle generator/config/preview and three focused tests
pass. Four actual inputs saved at
`/root/autodl-tmp/data/particle_corrosion2d/20261106-geometry-preview-01/manifest.json`.
Six smooth irregular fixed particles per1x0.6 rectangle, radius.045..075;
same initial pit r.12 and mouth halfwidth.055. Particle/domain normal distinction
and conservative clearance are independently checked. Geometry shape remains
the established prior unless a later change is explicitly recorded.

## Intended numerical route

Use existing KKS AC/concentration equations with homogeneous positive matrix
kinetics, phi∈[0,1], unrestricted scaled liquid concentration and algebraic KKS
reconstruction. Neutral particle/closed exterior walls; phi=c=0 on the mouth.
No constant-total-mass projection for this open system. Implement geometry-aware
visible nonlocal pairs with physical measures, time JVP and existing general
point-conditioning architecture. Validate mesh/horizon/interface scales before
interpreting training. Local reference uses independently assembled standard
weak-form FEM; avoid mixing FEniCS MPI with JAX GPU. Native direct solver and
local AD comparisons remain explicitly distinct. All fields/weights external.

## Plan / resume

1. Mesh/geometry and independent local/native direct solvers, fixed heldouts.
2. Focused operator/BC/reference qualification and expected interface contact.
3. Point-conditioned PINO and memory/speed probe, cold train and diagnose.
4. Improve convergence with general methods; preserve negative controls.
5. Matched cold local AD run, common-target comparison and documentation.

## Earlier implementation and mild-prior qualification

The canonical train.py, point-conditioned network, original five-loss nonlocal
path and seven-loss localAD control are implemented. Separate domain(x,y,phi0)
and boundary(x,y,nx,ny,kind) streams feed generic heterogeneous integral
attention; query coordinates are independent and spatial work is reused in time.
Raw z,s map to phi=.5(1+tanh z), c=cLe*s+(cSe-cLe)h(phi). Diffusion is evaluated
as cLe*D(s), dropping the constant analytically to protect FP32 differences.
No hard IC, mass projection, supervised future targets or distance attention.

Same128-edge smooth-contour polygons define conforming FEM and visible-pair
nonlocal geometry. Initial Gmsh sizing propagated tiny boundary edges and gave
19013nodes; explicit uniform target sizing fixes this to3903nodes, retaining
the complete boundary polygon. The rejected mesh remains external, unused.
First operator has41283visible pairs, maximum54neighbors; constant, weighted
symmetry/conservation and connectedness checks pass. Exact local Neumann values
are not claimed at finite horizon (wall quadratic moment differs byhalf).

First direct local BDF takes3.85s. Native delta.035 takes24.79s, with phase
pooled.9836%/terminal1.8433%, concentration1.2085%/2.2801%, and corroded-field
2.9667%/3.8655% versus same-mesh local FEM. Horizon refinement and independent
local mesh/time refinement are in progress before choosing the training recipe.

Fourier/timeJVP/localHessian/padding/core-halo and gradient tests pass. The64
independent known-geometry input bank completed in20.09s (fourCPUworkers),
halo3136/neighbor72 fixed capacities,20MiB compressed. Core and IC nodes use
mass-proportional sampling with replacement; each core has its complete halo.
No training/reference trajectories are present in this bank.

Direct artifacts:
- Inputs: /root/autodl-tmp/data/particle_corrosion2d/20261106-input-bank64-01
- Local qualification: /root/autodl-tmp/data/particle_corrosion2d/20261106-reference-qualification-02
- Native qualification: /root/autodl-tmp/data/particle_corrosion2d/20261106-native-qualification-01

The mild reference is qualified: h.01→.0075 at fixed256polygon phase changes
.0786%pooled/.1375%terminal; temporal change~1e-8. Native delta.026 versus refined
local on coarsepoints gives.9452%pooled/1.8652%terminal under trapezoidal time
weighting, so initial training selects delta.026. Mild GPUprobe B4/Nt8/width96
had3.197GiBactivepeak,16.52msheld,30.28msrefresh;235.73sfirstcompile. LocalADprobe
was stopped at240swhilecompiling,zero updates,noOOM; not a solver failure or
timing comparison. Subsequently replace unrolled per-loss norm evaluations with
sequential lax.map; CPU fullAdam parity passes for5/7losses and refresh/held.
Production compilation/timing remain to be measured after this exact-algorithm
change and the stronger geometry prior.

Completed stronger-prior artifacts:
- Four stronger inputs: /root/autodl-tmp/data/particle_corrosion2d/20261106-eight-particle-preview-01
- Training bank: /root/autodl-tmp/data/particle_corrosion2d/20261106-eight-input-bank64-01
- Four fine local references: /root/autodl-tmp/data/particle_corrosion2d/20261106-eight-local-reference4-01
- First-case native/finer local qualification: /root/autodl-tmp/data/particle_corrosion2d/20261106-eight-reference-qualification-01
- Eight independent reserved inputs: /root/autodl-tmp/data/particle_corrosion2d/20261106-eight-particle-reserved8-inputs-01

## Completed baseline and selected wider model

The stronger64geometry bank finished39.56s, halo2560/neighbor48. Four development
FEM references are complete; each reaches four particles byT1. Stronger first
case native delta.026 vs fine local phase1.1412%pooled/2.1371%terminal, c1.3789%/
2.5636%. Further local h.01→.0075 changes phase.0670%/.0817%, so reference
uncertainty is much smaller than current neural error.

Completed cold10000 width96 run:
/root/autodl-tmp/tf-logs/particle_corrosion2d/20261106-eight-nonlocal-w96-cold10000-01

Four unseen development geometries: phase4.18505%pooled/5.39596%terminal;
concentration5.02294%/6.45113%; corroded12.41784%/11.84690%. It plateaus near4%
after5k. Initial parameters320930; runtime seed1260183369. Total360.32s,
first update compile157.06s, median held11.897ms, refresh28.528ms,
activepeak2.19990GiB. A separate four-training-geometry diagnostic gives phase
4.2118%pooled/5.1359%terminal and c5.0405%/6.1376%, so there is no clear
generalization gap in this small cohort. Diagnostic reference fields were
generated after training and never entered the optimizer.

Matched localAD cold10000 completed:
/root/autodl-tmp/tf-logs/particle_corrosion2d/20261106-eight-local-ad-w96-cold10000-01
Actual initial model/optimizer bytes, every geometry/core/time draw and eight
common source files pass the strict paired audit. Phase28.9513% pooled/35.9741%
terminal, c27.8918%/36.3654%; optimization stagnated. Total675.51s, first update
382.40s, held19.281ms/refresh58.390ms, activepeak1.815GiB. This is not intrinsic
AD inaccuracy; derivative tests pass. Paired comparison is
/root/autodl-tmp/tf-logs/particle_corrosion2d/20261106-eight-w96-paired-comparison-01.

Completed wider nonlocal run:
/root/autodl-tmp/tf-logs/particle_corrosion2d/20261106-eight-nonlocal-w192-cold20000-01
Preset configs/train_wide.py changes width192/spatialfeatures96/20000updates,
LRhalving3000,eval1000/checkpoint5000; all PDE/loss/sampling structure retained.
Development phase1.95838% pooled/3.53662% terminal, c2.36704%/4.25570%.
Total747.924s; firstupdate205.533s, warmmean21.379ms, activepeak4.172GiB.
Width and budget change together; do not attribute improvement to width alone.

Finalreserved8 FEM bank also complete:
/root/autodl-tmp/data/particle_corrosion2d/20261106-eight-local-reserved8-01
It uses tightertime tolerances2e-8/2e-10/.0125; currentdevelopment uses1e-7/1e-9/
.025. Identical physics/spatialresolution; final cohort was first evaluated after
choosing the fixed20000 endpoint. Training4diagnostic is
/root/autodl-tmp/tf-logs/particle_corrosion2d/20261106-eight-w96-training4-diagnostic-01.

## Final reserved validation and negative controls

The fixed20000 width192 endpoint gives phase1.958708% pooled/3.423560% terminal,
c2.351003%/4.089795%, corroded5.835328%/7.525755%, terminalIoU96.11708% on the
eight frozen geometries. Individual phase pooled1.6436--2.3047%; all c pooled
below2.80%. No1%claim: accepted3%overall fallback is met, terminal3%is not.
Prediction range phi[5.066e-7,1], c[-.00022950,1.00073481], without clipping.
Evaluation: /root/autodl-tmp/tf-logs/particle_corrosion2d/20261106-eight-w192-reserved8-final-01.

Default train_default.py exactly equals the completed width192 config; old
presets remain. A direct h.01/delta.020 refinement gives phase1.1520%/2.0174%
versus1.1412%/2.1371%: no pooled improvement under respective quadratures.
Preserve configs/train_refined.py as an untrained control; no larger input bank
or neural run was launched. Qualification path:
/root/autodl-tmp/data/particle_corrosion2d/20261106-eight-refined-native-qualification-01.

First-case saved-field decomposition on4180coarsepoints: neural-minus-native
phase1.1308%/2.2687%, native-minus-local1.1412%/2.1371%, neural-minus-local
1.8784%/3.6198%, errorcosines+.3671/+.3494. Neural predictions here are FEM
interpolations of saved fine-query values, not new exact network queries.
Both discretization and fitting contribute; this is not a rigorous errorfloor.
Record: /root/autodl-tmp/data/particle_corrosion2d/20261106-eight-w192-error-budget-01.

Width96 diagnostics rejected uniform-attention collapse: learned boundary
attention KL.2156 (initial.00212), totalvariation20.60%; actual outputs agree
with dense-equivalent attention to2.36e-6. Geometry-centered error remains high,
but conditional signal is small and mean-field bias dominates global error.
Paths: 20261106-eight-w96-attention-audit-01 and
20261106-eight-w96-conditional-response-01 under the same data root. No new
architecture or geometry-distance bias was introduced.

Independent final read-only audit passes actual cold model/optimizer bytes,
eight common sources, all20000 geometry/pool/time arrays and full RNG replay
(runtime seed1607243318). Canonicalized particle records have zero duplicates
or cross-cohort intersections across64train/4dev/8reserved. All76geometry
encoder inputs reconstruct exactly from geometry/knownIC. Reserved metrics
recomputed from saved physical fields match every scalar exactly. No future
reference fields enter optimizer payloads. Focused geometry/operator/input/
loss/reference tests passed, including local physical Hessians and timeJVP,
complete halo and volume sampling, independent FEM Jacobian/flux/energy, and
sequential gradient-balancing full-Adam parity.

Curated assets/pino/particle_corrosion2d.png shows first4reserved, actualtimes
0,.2,.5,.8,1, predictedphase and localFEMinterface. Size229742bytes; all five
older PNGs and provenance entries unchanged byte-for-byte. Six PNG total
2281366bytes; aggregatecap2.5MiB, individualcap1MiB. Raw fields/weights external.

## Full-domain residual audit

Additional full-domain CPU audit (13.08s, no reference fields/parameter gradients)
uses firstdev4180nodes, all4171nonmouthPDErows and four independent times
.068427,.302947,.603608,.807510. Volume-weighted original AC/CH meanabsolute
residuals.019508/.068531, RMS.116683/.303985; independent8x256volume samples
give.019343/.071727. Boundary meanabsolute.035493/.105479 exceeds interior
.018707/.066680. Localmax4.6414/7.6762 and laterAC growth remain; no strict
pointwise convergence claim. RMSphi_t=.46567, RMSc_t=.55904.
IC RMSphi=.0039713,c/cLe=.0046532; mouthpositive RMSphi=.00028232,
c/cLe=.00020385. Record and reproduction script:
/root/autodl-tmp/tf-logs/particle_corrosion2d/20261106-eight-w192-fullgrid-audit-01.
CPU audit overlap UTC2026-11-05T19:33:22.279925--19:33:36.029874; GPU jobs
remain serial, but fullrun timings include this brief CPU diagnostic overlap.

## Final matched control and task end

Matched width192/localAD20000 completed at
/root/autodl-tmp/tf-logs/particle_corrosion2d/20261106-eight-local-ad-w192-cold20000-01.
Development phi97.26613% pooled/100%terminal, c97.07194%/99.99980%.
Same reserved8 gives phi97.26646%/100%, c97.07348%/99.99980%; saved field metrics
recompute exactly. Initialphi spansapproximately.0045--.997, but saved positive
times.1--1 have maxphi4.47e-7, terminalphi identically0. This is a degenerate
optimization outcome, not a derivative correctness failure or AD accuracy limit.
Both actual baselines and failed controls remain preserved. No retuning claimed.

Wide strict comparison validates initial model/fulloptimizer/common sources and
every actual draw. Nonlocal/AD median held19.6905/28.5241ms, balance refresh
50.7299/90.2855ms; warmmeans21.3786/31.6296ms. Firstupdatecompile205.5335/
493.4215s; total747.9236/1236.6816s; activepeak4.1717/3.6203GiB. Includes setup,
I/O/evaluation; allocator25,251,741,696bytes is separate. GPUjobs serial; brief
CPU overlap disclosed above. Different wall-loss terms/closures mean a complete
method comparison, not equal-accuracy timing or isolated AD versus graph error.
Final artifacts under the run root:
- 20261106-eight-w192-paired-comparison-01/comparison.json
- 20261106-eight-w192-local-ad-reserved8-final-01/metrics.json

Diff-level delivery: new self-contained particle example (geometry, conforming
mesh, independent local/native solvers, inputs/operators, two derivative paths,
canonical training/evaluation, explicit presets),22focused tests, one curated
figure/renderer/provenance, project/example README, artifact index and memory.
Shared numerical/network/optimizer modules are unchanged. Static AST30modules,
default/saved configuration equality,24localREADMElinks, previous5imagebytes
and old provenance entries, six-image2.5MiBbudget, artifact inventory and
gitdiffcheck pass. No raw fields or weights added to Git; no Git operation or
file cleanup performed in this task.

No running experiment or blocker. The original1%target remains unmet; the
explicitly allowed3%overall fallback is satisfied on the frozen8cohort. Future
work, if requested: robust localAD optimization, further wall/late-time residual
convergence, additional seeds and larger heldout populations. No automatic
follow-on run is queued. The completed record is archived by task-end.
