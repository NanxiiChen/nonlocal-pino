# Project Progress

## Private GitHub publication — 2026-10-24

Repository: [NanxiiChen/nonlocal-pinn](https://github.com/NanxiiChen/nonlocal-pinn).
Created as private; default branch main. Split code and publication records
follow dev commit -> regular merge into main -> push main only. Local dev
remains local. Original source and recovery backup are retained.

## Project split complete — 2026-10-24

Independent package, explicit configs and artifact index are delivered.
[Validation and recovery](migration/VALIDATION.md);
[completed task](tasks/archive/project-split-validation.md).
Original source/environment/data/log paths remain intact.
The split preserved physical formulations and existing artifact paths.

## Active task

None. Latest completed: [current code publication](tasks/archive/current-code-publication.md).
Code/records devcommit ae46746 -> regularmainmerge c359d7f pushed and exact
origin/main equality verified; final task-end receipt follows the same workflow.
No numerical/default changes, artifact deletion or training queued.

## Recent completed tasks

[Current operator code/records published](tasks/archive/current-code-publication.md).
337textual files include confined-CH geometry/online/SOTA and electro/melting
spatial controls plus explicit unresolved replay/query issues.148Python AST/
156strictJSON/45configuration factories pass;1851indexedfiles20.83MiB and no
new binary/generated payload or oversized file. Two whitespace-only edits have
identical texttokens/PythonAST. Prior numerical evidence reused, no retraining.
dev ae46746 -> ordinarymainmerge c359d7f -> origin/main verified; no pushdev,
history rewrite or cleanup. Publication validation/archived memory receipt
retains exact code refs; final documentation tip is authoritative in Git.

[Melting spatial-pool comparison](tasks/archive/melting-spatial-pool-comparison.md).
Matched5000Adam/B16/Nt8/128^3/ordinaryFP32 fixed/rolling: native8meanphi pooled
0.1631/0.4217%, terminal0.2218/0.2514%; warm12.737/13.247ms(+4.00%), cold190.06/
190.65s, activepeak1.4952/1.4955GiB. Coverage0.3898->45.6869%, producerwait0.0236s.
Initialmodel+Adam/80000actualICs/40000times/first32cores/allcorestreams/rawmetrics
and sameGPU1024queryreplays<=9.62e-7pass. Two focusedtests/FP64full-gradient
identity pass. Full3update probe15.44s/38.31GiB; user cancels full5000. Extra
65536queryfield gate fails0.00577versus1024; input/JVP/explicit/captured probes
retain it, cause unresolved, issue recorded and failed gate retained. No scalar
fullresidual ranking, partition-independent/stable accuracy claim or promotion.
README/DEVELOPMENT/index/scripts/receipts/figures archived, no cleanup/Git action.
Allworkers exited; no follow-up queued.

[Electropolish spatial sampling comparison](tasks/archive/electropolish-spatial-pool-comparison.md).
Full/fixed/rolling matched5000Adam/B8/Nt4/FP32ordinary, all40000actualICs/time/
ICBC/model+Adam state parity. Same64localFD101meanphi4.481/4.964/4.858%pooled,
3.774/3.833/3.651%terminal; warm effective15.83/13.22/15.08ms, cold4.109/3.915/
4.090min. Strict safe halo/SDF/FIFO/overflow tests4PASS; fixed3938/4949free
coverage versus full/rolling100%. GPUreplay<=4.09e-6and independent64field
metrics pass. ExtraCPUreplay fails0.3167; composedattention issue recorded with
NumPy/isolated CPU-GPU probes, cause unresolved. Interpret measured GPU paths
only, no robust/unique sampling advantage or default promotion. README/index/
scripts/receipts/figures archived; failedCPUgate retained, no reference solve,
cleanup orGit action. Allprocesses exited/GPUreleased; no follow-up queued.

[Rolling core+halo assessment](tasks/archive/core-halo-rolling-assessment.md).
Both NOs draw fresh ICs/times but cycle32static spatial sets. Exact CPU core
coverage: electropolish3938/4949free(79.57%), melting8172/2097152(0.3897%);
halo still participates in AD. Recommend reusable bounded subgraph producer
on immutable graph, deterministic uniform cores, complete halo, fixed safe
padding and coordinated query/SDF metadata refresh. Current observed pool caps
cannot safely bound arbitrary fresh sets. Pure CPU draw simulation illustrates
coverage only, not trained L2 or full producer throughput. No runtime/default/
reference/Git change; source/JSON/docs/diff checks pass. Discussion complete.

[Geometry-operator SOTA promotion](tasks/archive/confined-ch-operator-sota.md).
User-accepted width256/online/SOAP50k/FP32ordinary now selected by the opt-in
configs/train_operator.py; fixed50k model at4.8329% pooled/5.1935% terminal phi
on the same8native development geometries. Root/example README, DEVELOPMENT
and artifact index agree on the recipe/direct checkpoint/validation scope.
Original mild config retained explicitly; all37derived presets unchanged,
single-IVP window default and mild reference CLI semantics preserved. Exact
saved-config equivalence, actual entry/schema/path checks and3focused CPUtests
PASS26.288s; whitespace check passes. No training, external file cleanup/move,
new media or Git history operation. Promotion complete; no required resume.

[Width256 cold50k with online rolling inputs and finalphysicalfields](tasks/archive/confined-ch-width256-online-cold50k.md).
Fresh50000SOAP/FP32ordinary/B16Nt16updates complete; nativephasevalidation
4.8329%/terminal5.1935%, historical16monitor5.0529%. Same8reused development
geometries/oneinitialization, nativeonly/FEMdeferred; worstterminal10.5862%.
102.26mintrainingwall, fullrunGPUactivepeak20.4337GiB; GPUreleased.16000new
operator-onlyinputs installed/15994sampled, no trainingPDE/FEMsolves. Fullstate/
actualslot/time/geometry/pool and24CPUfieldreplaysPASS(phi7.15e-7/mu3.66e-6).
All10pairedcheckpoints present/final50kbytesexact. Finalmeanis prescribedbudget,
not lower49.5kmetricselection. Three low/median/worst3x5phi galleries validated/
inspected, originalmesh/holemasks/sharedscales, PNG/PDFexternal. Finalcurvehelper
reports truehistorycoverage after25k and stoppedcontrols without extrapolation.
User-approved249redundantintermediatefiles freed5.7428GiB; protectedstates/logs/
data retained. Results/audit/live/index and archive provide exactpaths; no
defaultpromotion, newenvironment orGit history change. Taskclosed, no resume.

[Geometry-operator capacity, precision and spatial controls](tasks/archive/confined-ch-width128-refinement.md).
Fixed30kcold FP32 width128->192/fourheads/Fourier128->192channels improves native
meanphase9.784/10.580->8.915/9.802%, dense61time9.319->8.346%; train16phase
5.089->3.970%, worstnative18.55->15.53%. Coldtraining64.65->79.21min(+22.5%),
warm92.33->119.43ms, activepeak8.43->9.38GiB. ExplicitSOAPcap512preserves wider
FFNcoverage; not an isolated head-dimension gain. All30000actualdraw batches
match, no labels/reserved draws, CPU24field gaps6.26e-7/2.82e-6 PASS; mass1.63e-8/
energy nonincrease. Six of eight densegeometries improve; two worsen, single
seed/reusednative8cohort/FEMdeferred. Finalpoint rebound, not fully converged;
fixed30k result retained, no best-heldout selection/longer run/default promotion.
Earlier FP64partials/hardware/two-native-spatial checks preserved. Final receipts
confined-ch-width192-results/audit.json and confined-ch-capacity-final.json,
curves/fields/checkpoints external. All jobs exited/GPUidle; no code/Git change
in completion session, no follow-up queued.


[Conservative disk cleanup](tasks/archive/disk-cleanup.md).
User-authorized package-cache cleanup plus nine verified orphan128MiB MPI pools
recovers measured9.805GiB; available4.529->14.334GiB. All numerical data/logs,
checkpoints, environments, Trash/history/backup retained.26protected metadata
checks, two reference NPZ reads, CPU JAX/autodiff and dolfin import PASS. No
source/config/Git change. Immediate small pilots fit; full refined512bank
estimated10-12GiB would leave tight room, optional30-50GiB expansion advised.
Exact paths/commands/scope and byte counts recorded in task/receipt.

[Density/capacity/GPU assessment](tasks/archive/confined-ch-density-capacity.md).
Recommend width128control at current h=.035, separately1-2same-geometry native
h=.030refinement checks; no run/config change. Matchedwidth64->96benefits already
measured; width128benefit/peak unmeasured. Current48GiB GPU/15.72GiB activepeak/
35.56GiB allocator leaves room for modest pilots; densemetric N^2storage7.07->
estimated13.10GiB when meanpoints1230->1674. Fullrefinedbank needs disk (4.6GiB
free). Native9.66%doesnot itselfprove underresolution or localoperator accuracy.
Evidence/estimates/limits in task/receipt; no largerGPU request or queuedjob.

[Random geometry-operator physical fields](tasks/archive/confined-ch-random-gallery.md).
Blind draw514/519/516/517/518 from eight reserved development geometries;
five shared-scale phase reference/prediction/absolute-error panels at0/.75/1.5/
2.25/3. Pooled phase17.26/14.65/7.71/13.67/4.78%; worst case retained. All eight
raw reference/geometry/score checks and five positive display reconstructions
pass; all panels inspected. Native graph/BDF only, no fresh test or training.
Script/receipt/index archived; PNG/PDF6.85MiB external, no Git action.

[Fixed10mode geometry CH operator optimized to approximately10%](tasks/archive/confined-ch-fourier-optimization.md).
Selected96width/512inputs/30k: nativephase9.66/10.51%,61time9.19%; training16
monitors5.92/6.30%, worst8geometry17.26%,mu21.06%. All24CPUfieldreplays<=4.34e-15,
fullstateimport gaps0, no heldoutdraws/futuretraininglabels; mass1.44e-16/energy
nonincrease. Geometryresponsecorr.977/amplitude.933. Same8reuseddevelopment
geometries/oneinitialization, nativeonly/FEMdeferred; terminal/allcases not all<10%.
Matchedcapacityhelps;1024/64/10k20.21/21.79% doesnotbeat512/64/10k18.95/20.26%.
Explicitcoldbest numericalpreset matchesmeasuredstagedrecipe, separatecold30k
notmeasured. Interruptedfirstattempt retained; detachedretry andpairedSOAP
snapshots fixrecovery, realrecovery/interruptedwrite/CPU21kbackup checksPASS.
README/DEVELOPMENT/index/receipts archived; largeartifacts external, noGitaction.
Alljobs exited; no requiredresume.

[Fixed small-scale10mode geometry operator](tasks/archive/confined-ch-smallscale10-operator.md).
Fresh24native refs PASS340.25s;512bank126s/6.08GiB/no newtraining trajectories.
UserdefersFEM. Matched20kcoldAdam/SOAP nativephi51.26/50.78->39.56/39.17%,
wall1435.61/2030.37s, warm42.79/69.82ms. SOAPmatchesAdamfinalat8500/1045.75s.
Bothstillfall; no stationarytrap/capacityfloor or low-errorclaim. Initialparams/
training-source/80kactualdrawsidentical; CPU24fieldreplays<=4.83/3.56e-15.
Geometryresponse91.71->80.07%error,28.42->41.95%amplitude; incompleteconditioning.
Width64/4heads/4conditionblocks/3decoder,224705params; width96preset and
1536-weight time4->16control untrained. Docs/index/scripts/compactreceipts
archived; generatedpayloadsexternal, defaultsunchanged, alljobs exited.


[Longer cold training of the512-domain CH operator](tasks/archive/confined-ch-operator-long-training.md).
Only5000->20000Adam budget changes; same512/8data/model/physics/LR, exact initial
parameters and actual shared5000drawprefix. Nativephi4.690/5.454->2.463/2.837%,
local2FEM8.976/10.431->8.111/9.473%; geometry response35.51->19.28%,correlation
.936->.982 on identical4627queries.80000draws visit all512inputs123..201times;
CPU24fieldreplays<=1.45e-15, no leakage and physical checks pass. Cold1401.54s,
warm41.89ms. Last5000phase2.333..2.791%,slowly falling; no complete-convergence,
validation-selected-best, multi-seed or continuum claim. Explicit long preset,
README/index/analysis/compact receipts archived; generated outputs external.
Old5000/windowdefaults/artifactpaths retained, no runtime/core/Git change.
All jobs exited; no follow-up queued.


[Expanded richer geometry-conditioned CH operator](tasks/archive/confined-ch-geometry-richer.md).
Six-worker512training/8validation bank ready124.70s (~6.11GiB), no newPDE/FEM
trajectories.496newoperator-only inputs plus16labeled train monitors; original8
validation domains retained. Same5000Adam/B4initial/model gives nativephi
9.184/11.200%->4.690/5.454%, local2FEM12.846/15.402%->8.976/10.431%; geometry
response error77.03->35.51%, correlation.645->.936 on identical4627queries.
Cold431->524s, warm43.02->43.22ms. All512visited20..68times/noheldout overlap;
CPU24fieldreplays<=1.64e-15, initial/reference arrays exact, focused2tests and
legacyload/operator/physical checks pass. Single-run reused development cohort;
native/local bias7-9% and seed/topology/continuum limitations remain. Explicit
preset/loader/evaluator/docs/index/scripts/compact receipts archived, media/data
external. Old defaults/artifactpaths retained; no reflection run or Git action.
All processes exited, no follow-up required.


[General geometry-conditioned CH operator](tasks/archive/confined-ch-geometry-operator.md).
Finite16/8random-hole bank, fixed ambient IC, physical quadrature, hard IC/mass,
PDE-only canonical train.py. User rejects distance bias; generic GNOT-inspired
boundary/domain-function streams improve longerT.75nativephi5.293/6.220% to
2.380/2.771%, geometry-response error84.42%->42.65%, amplitude43.54%->94.74%.
Local2meanphi6.886/7.907%->5.720/6.704%; operator bias remains. Cold314/407s;
all24CPUfield replays<=1.9e-15,20kactual geometry draws/config parity and focused
kernel/gradient/mask/query/physical checks pass. References/FEM pass; reused
development cohort, no arbitrary geometry/seed convergence claim. Scripts/
figures/metrics/source/index/docs archived; existing defaults retained, no Git
history action. All jobs exited; further geometry/operator work is optional.

[Irregular-domain initial-field encoding](tasks/archive/confined-ch-operator-encoding.md).
Fixed ellipse/twoholes has3682triangle-centroid inputs with physical areas.
Recommend point lifting/area-aware latent cross-attention, reusing shared
spatial/time decoder; mask-aware image patches remain a simpler control.
No automatic BC enforcement, geometry-transfer or long-time convergence claim;
window route retained. Literature/code/array checks complete; no implementation.


[Attached-paper CH versus NSCH clarification](tasks/archive/twophase-nsch-scope.md).
Direct PDF recheck confirms Qiu2022 vortex explicitly omits momentum/continuity,
while rising bubble solves coupled CHNS with capillary feedback and(u,v,p,C).
Current pilot corresponds only to the first scope, not exact paper reproduction.
Recommend true coupled model for full two-phase NO; random vortices then specify
initial velocity/forcing, not future prescribed velocities. No implementation.


[Physical random-flow family](tasks/archive/twophase-physical-flow-design.md).
Recommend controlled counter-rotating Gaussian vortex pair: random circulation,
core size, separation and offset; periodic zero-net-vorticity construction.
Physical prior replaces random Fourier coefficients; FFT remains a numerical
representation. Explicit one-way prescribed stirring, not free NSCH dynamics.
Four-roll mill is an alternative requiring device/BC distinctions. Three flow
schematics/independent divergence checks archived; no CH/NO training or defaults.


[Two-phase NO conditioning recommendation](tasks/archive/twophase-operator-design.md).
Current case is prescribed-flow CH, not NSCH. Prefer fixed droplet/random
low-frequency divergence-free streamfunction velocities first, then joint
random IC; reuse image encoder/shared queries/time JVP and grid-mass constraint.
Keep full-grid FFT assumptions, native/local validation and peak-deformation
metrics explicit. Discussion only, no implementation/default/training/Git action.


[Verified NO defaults and lightweight publication](tasks/archive/operator-sota-publication.md).
Electropolish canonical NO now onlineB8/5000Adam/image-only, melting retains
onlineB16/5000Adam/hardIC/latent. Exact saved-config comparison passes; five old
control presets preserved.27focused tests pass.248generated outputs untracked,
27new outputs excluded; all275local files plus byte-verified external copies
retained. Current tree~18.74MiB vs90MiB; new media ignored, restore manifest/tool
and content guard added. Historical Git blobs remain; no rewrite/forcepush.
Source/numerical/FEM/render records retained; no training or numerical tuning.


[Publication-style melting3Dinterfaces](tasks/archive/melting-operator-3d-render.md).
Prior random inputs5/9/2, unchanged Adam checkpoint: actual96^3/61time fields
and183closed phi=0meshes. Saved-field/IC/chunk checks<=1.32/4.15/.48e-6,
meshphi residual<=4.52e-6, volumes decrease. User rejects decorative preview;
recommended matte orthographic five-time panels and time-colored3Dcontours,
450dpiPNG/PDF/SVG with explicit vector/raster distinction. Optional display-only
cutaway labeled; corrected GIF/MP4 retained as preview. Temporary isolated
VTK/EGL setup, cache/depth-sort/layout issues and provenance recorded. README/
index/scripts/fields/media/memory archived; no solver/default/Git change,
workers idle, no further work queued.


[Melting local/nonlocal reference check](tasks/archive/melting-operator-local-fem.md).
One held-out input5, matched full-cube CG2 FEM/N96/dt.00125: native/local phi
pooled.1474%, terminal.1670%; slice.3575/.5351%. User accepts current reference
precision and deprioritizes final tiny-solid error; retain native monitoring,
no operator optimization. Original strict slice gate.0737%>.05% remains recorded,
status accepted_by_user. Fresh Adam/SOAP local pooledphi.1479/.1861%; native
.2383/.0709% can reverse ranking through cancellation. Five FEM tests/CPUgeometry/
GPUold-new parity/checkpoint replay/independent metrics pass; GPU strict geometry
threshold caveat and all raw failures archived. README/index/scripts/fields/
records updated; no default/Git change, workers idle, other samples deferred.


[Matched SOAP melting operator](tasks/archive/melting-operator-soap.md).
Same79initial arrays/80000actual online shapes/40000time draws,5000/B16/lr1e-3.
Eight-test meanphi pooled improves.2266%->.0785%, terminal worsens.0922%->.1291%;
solid terminal4.502%->6.504%, volume1.213%->4.016%. Cold300.57->492.06s,
warm22.43->33.66ms. Source/input/reference/metric checks and fresh checkpoint
replay<=9.54e-7 pass; curves/raw records/README/index archived. Keep Adam
operator default/SOAP opt-in; no numerical source, seed-stability, FEM or Git
history work. Workers idle.

[Random held-out melting samples](tasks/archive/melting-operator-random-samples.md).
Uniform pre-score draw IDs5/9/2; three five-time reference/prediction/error
galleries with shared color scales. Saved-reference equality and independent
3D diagnostic L2 replay pass; selection/script/metrics/figures archived.
No model/training/default or Git history change.

[Random-ellipsoid 3D melting operator](tasks/archive/icemelting-random-operator.md).
Canonical opt-in train_operator: volumetric512tokens/shared time preparation,
Euclidean rotated/shifted ellipsoid IC, hardIC/latent residual.80000distinct
online shapes/5000AdamB16exit0, full300.57s; peakactive1.332GiB/pool4GiB.
Eight held-out meanphi pooled/terminal.2266/.0922%, solid.6425/4.5017%; terminal
solid-fraction error1.2133%, max late solidL210.59%. Diagnostic24^3nodes and
128^2slices, full128^3native references10time checks<=1.422e-7; no full-grid/
continuum/seed-stability claim. Four new numerical/two2Dregressions, legacy
checkpoint/output, source/stream/field/replay audits pass. Scripts/results/
figures/README/index archived; single-IVP defaults retained, workers idle,
no Git history action.

[Phase-field publication record](tasks/archive/phasefield-results-publication.md).
User authorizes publication of accumulated confined-CH/electropolish work.
155Python AST/260strictJSON/100JSONL checks pass;823staged files, with20files
normalized only in whitespace and checked for identical Python/SVG/text meaning.
Git workflow is dev commit -> regular main merge -> origin/main only; branch
history and final remote-ref verification provide the publication receipt.
No retraining or numerical/default change in the publication task.

[Electropolish operator local AD/nonlocal comparison](tasks/archive/electropolish-operator-local-comparison.md).
Original rough ICs, matched5000Adam/B8; exact73initial arrays/40000profiles/
time/core/IC/BC parity. CommonlocalFD10164meanphi pooled NL4.427/AD8.595%,
terminal3.176/5.498%; early side4.219/12.223%. Cold process368/693s mostly
compilation; warm update16.71/19.35ms. All76reference time gates<=2.583e-8;
declared4FD201->401phi differences.068..076%, refined-target ranking retained.
Local explicit sixth Neumann loss gives tighter point normals despite worse
field fit; no exact-Neumann/unique-cause/seed-stability claim for nonlocal.
Focused tests, field/refinement/source verification and scientific figures
complete; scripts/results/README/index/memory archived. No IC/default/Git change;
all workers complete. Unrestricted concentration range check is derived/tested.

[Electropolish operator64-interface evaluation](tasks/archive/electropolish-operator-test64.md).
Same image-only final checkpoint,64fresh accepted six-mode profiles, original
IC/graph/reference gates: meanphi pooled/terminal4.506/3.502%, p906.909/5.850%,
worst10.541/8.210%; meanc5.323/4.163%. Allrefinements<=4.60e-8, no training-input
overlap/model selection. New test estimates, not a retraining gain. References
797s/fourworkers; cold eval26s with24sJIT, warm2.23ms/fulltrajectory. Existing
5000Adam/B8training315.8s unchanged. PDE-only loss+parameter-gradient probe
NL7.62/AD6.99ms on245freecores (2453/245queries), promoted Hessian check passes.
Small timing difference has CPU-load caveat; local compile wall includes pause.
No full local training/accuracy or stable speed-order claim; no production,
default, checkpoint or Git change. Scripts/raw results/README/index archived,
all workers idle.

[Electropolish conditional-training shortcut diagnosed](tasks/archive/electropolish-operator-training.md).
Uniform-time single fit succeeds; two-input local-feature fit degrades. Frozen
image swaps nearly ignored while local IC copies t0; no strong inter-input
gradient cancellation/frozen encoder found. Optional arch.local_input=zero
removes only query features, retains true IC supervision/image/physics. Matched
two-fit meanphi9.47/5.92% ->2.60/2.70%; same-stream onlineB8/5000phi17.748/12.355%
->5.573/4.372%, c19.317/14.029% ->6.452/5.124%,315.8s. Unseen16phi1.83/1.69%,
Dq80.70->17.79%; input variation at.25 recovers80.36%reference. Harder18still
10.44/8.88%; no universal/seed-stability or default-promotion claim.73initial
arrays/40000actual profiles exact; tests/gradient/field/source/transport checks
pass. IC is33segment windowed signed Euclidean distance, not vertical gap or
exact global SDF; exhaustive IC differences.14..1.28%, numerics/ref unchanged.
Scripts/results/figures/README/index archived; defaults/old artifacts/Git history
retained, workers idle. No causal/stratification/hard-IC intervention bundled.

[Electropolish online causal control completed](tasks/archive/electropolish-operator-causal.md).
Canonical opt-in train_operator_causal reuses shared stopped per-time weights/
eps adaptation per condition, correct free-core means and existing gradient
balance. MatchedB8/5000Adam takes324s; mean testphi17.748/12.355% ->17.932/13.105%,
c19.317/14.029% ->19.521/14.802% pooled/terminal: no improvement/default promotion.
73initial arrays/40000actual profiles/time draws exact;51NumPy causal-record
checks and sixGPUfield replays<=6.26e-7 pass. Three numerical/step/noncausal
gradient checks plus config rejection pass; test-harness/comment-parity failures
retained explicitly. Input dependence/transport still weak;84%of logged Nt4
batches miss first5%time, a sampling limitation not isolated causality. Raw and
causal losses/weights/eps, configs/checkpoints/curves/fields/index/README archived.
Physical/model/shared-causal/reference/default/Git history untouched; workers idle.

[Electropolish online balance diagnosed](tasks/archive/electropolish-operator-balance.md).
Last1000 logged AC/CH weighted ratio median.467; frozen four late batches show
cosines.010..119 and similar balanced gradient norms. Supplemental early probe
has AC/CH+.927 but AC/IC-.596 and CH/IC-.528: early PDE/IC competition, not
measured persistent AC/CH suppression. Four unseen inputs fit initial phi~1..2%
then collapse toward one flat trajectory; at t*.25 phase condition spread is
.943% of reference. Unseen16 terminal qL2 3.41% yet Dq80.70%; first5%time holds
49.81%AC/73.98%CH residual integral. Six-source/actual-input/initial-loss/weight/
GPUfield/weighted-gradient/quadrature checks pass; diagnostics/figures/README/
index archived. Frozen probes do not reconstruct Adam's historical directions
or prove a unique cause/remedy. No training/model/default/reference/Git change;
workers idle, no queued optimization.

[Electropolish fixed-input/online controls completed](tasks/archive/electropolish-operator-single-online.md).
Clarify prior16inputs/B2. New sample0/B1 Adam2000 phi2.483%pooled/3.387%terminal,
SOAP4.187%/3.214%,237/387s;73identical initial arrays/all actual sampled IDs0.
Adam chosen for online baseline, no universal optimizer ordering; established
PINNSOAP default intact. New onlineB8/5000Adam draws fresh continuous six-mode
coefficients,40000distinct actual inputs, never finite-bank training. Explicit
profile-validity distribution/fixed-shape SDF/no silent padding or batch reduction.
Fourheldout phi17.748%/12.355%, c19.317%/14.029%,326.7s; no test model selection
or claimed accuracy solution. Peakactive2.15GiB/pool5.01GiB, observed~5.3GiB on48GiB.
Three focused selector/generator checks, actual-stream/source/parameter parity,
GPUfinal replays<=1.43e-6, independent fields/transport/32height audits pass.
Configs/data/sampler/source/direct artifacts and scientific figures archived;
README/DEVELOPMENT/index/memory updated. No core/physics/default/reference/Git
change or extra optimization queued; GPU idle. No generalization claim from
fixed-input fit or pure online-vs-finite gain claim across different budgets.

[Electropolish neural-operator pilot completed](tasks/archive/electropolish-neural-operator.md).
Canonical train.py opt-in supports PI-CViT-inspired separable patch encoder/
spatial cross-attention or pooled FiLM, known local IC features and shared
temporal FiLM/JVP; c.arch owns architecture, c.operator owns input data.16train/
4held-out random SDF interfaces,5s, pure physical losses; six independent
same-graph references pass time refinement<1.93e-8. Fresh2000Adam cross/FiLM:
mean test phase pooled17.19/17.46%, terminal13.31/11.94%, trainer245.7/160.9s,
vs frozen-IC terminal47.72%. Early over-smoothing/Dq~.77-.81 remain; no accuracy/
stability/SOTA or arbitrary-topology claim. Unrestricted liquid output removes
an old map bound excluded by random references; PDE/KKS mixture retained.
Three focused tests/independent derivatives,55common initial arrays/14of15source
parity, GPU final replays, public CLI/template and real one-update final API
smoke pass. Cross CPU FP32 strict field probe fails at.00162gap but aggregate
phase drift<.01percentage point; trained promotedFP64 split/JVP equality passes.
Shared-space forward/JVP gains1.16x/1.11x exclude optimizer/PDE. Raw probes,
fields/curves, scripts/configs/index and precision/initial-CLI-return caveats
archived. Established single-IVP default/checkpoints unchanged; no Git action,
FEM process or queued GPU work. Resume accuracy only on explicit user direction.

[Confined CH default switch completed](tasks/confined-ch-global.md): verified
30-window original-complex-case default retained; all-field accuracy limitations
are deferred. No global optimization is queued.

[Confined CH diagnosis/optimization completed](tasks/archive/confined-ch-optimization.md).
Frozen richer T1.5 physical case/reference, PDE-only sequential30 windows:
independent42/2026 common-FEM phase pooled3.908%/3.866%, terminal5.754%/5.672%
vs original82.96%/94.55%. Native phase0.724%/0.678%; trainer281/288s with5000+
500 total updates. Complete CPU chains,31-source/config parity, mass/chemical
constraints and sampled decreasing energy pass. Same-budget15windows is slightly
worse; keep explicit train_coarsening_windows as richer-case recommendation.
Native mu pooled~30%, positive-only~52%, terminal~100% remains unresolved;
time derivatives jump and fixed-grid states lack off-grid normal extensions.
More stored networks/modes/metric/resets are a bundle, no isolated/local-AD
advantage claim. All failed controls, supervised-only probes, plots/raw results
registered and documented. Original simple default/reference/paths retained;
workers idle, no Git operation.

[Richer confined CH training experiment completed](tasks/archive/confined-ch-coarsening-training.md).
Matched seed42 SOAP5000+LBFGS500 both exit0 but fail phase learning: common
pooled NL82.96%/AD82.95%, terminal94.55%/99.41%, trainer174.65/696.63s. Best
both untrained step0. Strict final/best CPU replays and19-source/initial parity
pass; fixed17 initial chemical loss52.66%/39.46%, positive mu L2>119% remains.
PINN purity/energy checks expose wrong late evolution; mass is hard constrained.
Capacity/lifting/sampling/optimization not isolated. Curves/fields/raw results/
README/index archived, accuracy remains unsolved; no default/config/reference
change or extra tuning. CPU/GPU/FEM idle; no Git work.

[Sharper confined CH coarsening completed](tasks/archive/confined-ch-coarsening.md).
Opt-in ten-wave IC/epsilon.02/T1.5 gives clear restructuring and46.14% terminal
pure area vs0% in the original pilot. Preserve failed first mesh check; refined
.014/.01 reference passes fixed gates: max phi mesh/time.552%/.048%, mu3.014%/
.180%; mass/energy pass. Two focused tests, old replay, actual-grid operator,
reference loading, docs/index/diff checks pass. Final fields are FEM references,
no new PINN training or accuracy claim. Original default retained; CPU/GPU idle,
no Git operation. Config/README/scripts/figures/checks archived.

[Confined CH Neumann pilot completed](tasks/archive/confined-ch-neumann.md).
Prior results committed728e5b9 on dev; new ellipse+twohole example has canonical
train.py, physical-volume visible pairs and independently refined mixed-P2 FEM.
CN chemical-reference gate failed and was repaired with BDF2/analytic mu0,
without changing physics or loosening thresholds. Matched5000+500 cold starts:
common-local pooled phi NL4.1935%/AD5.6639%, terminal4.9998%/3.8354%, trainer
146.47/529.03s; native NL pooled.8898% against its separate target. Natural closed
pair flux does not make MLP local wall derivatives zero (NL RMS1.66-8.79 vs
AD.0158-.0255); pointwise classical-Neumann advantage remains unvalidated.
Initial mu remains weak; all5 focused checks, reference gates, initialized/
source parity, final CPU replays and physical audits pass. Fields/metrics/
README/index/normal diagnostics archived; no global accuracy or stability claim.
Original defaults unchanged; new work uncommitted, GPU/FEM idle, no push/merge.

[Neumann phase-field case recommended](tasks/archive/phasefield-neumann-candidates.md).
Propose confined constant-mobility CH coarsening in ellipse+two impermeable
holes, with local AD control and independent mixed-FEM/native references.
Primary papers support feasibility; pairwise natural no-flux at finite horizon
is distinct from exact pointwise classical Neumann. Quadrature, wall-crossing
bonds and boundary/local-limit refinement require checks before any advantage
claim. Geometry schematic and formulation/validation proposal archived; no
implementation, training/default/reference or Git change. Await case selection;
existing precipitate results and deferred two-phase FFT issue retained.

[More demanding precipitate case completed](tasks/archive/precipitate-multiparticle.md).
Three unequal interacting particles, stronger elasticity and T4 produce clear
elongation/disappearance, with independent refined reference and zero-elastic
control. Requested matched nonlocal/local AD cold starts complete5000+500:
common-local pooled phi8.0304%/12.8981%, terminal9.1820%/19.2435%, trainer
183.34/674.27s. Both final CPU replays,7focused tests, legacy loading and
mass/energy/mechanics audits pass. Initial q and late disappearance remain weak;
keep opt-in harder baseline, original default unchanged, no accuracy stability
claim or further tuning. README/config/index/fields/metrics archived; completed
spectral exploration excluded from primary comparison. GPU idle, no Git work.

[FFT periodic-boundary limitation](issues/twophase-fft-periodic-boundary.md)
is recorded and deferred at the user's request. Retain FFT first derivatives,
nonlocal pair Laplacians and time JVP in the recommended vortex preset. Both
current spatial implementations assume periodic geometry; no further boundary
or nonlocal optimization work until explicitly resumed. Documentation-only
[completed record](tasks/archive/twophase-fft-boundary-issue.md); configs unchanged.

[All-nonlocal flux control completed](tasks/archive/twophase-nonlocal-flux.md).
Optional nonlocal_flux adds moment-corrected periodic first derivatives with
independent native reference; default FFT flux retained. Matched fresh seed42
SOAP5000+LBFGS1000: common-local pooled phi FFT2.9236%/full-NL8.5448%, terminal
3.1270%/4.5147%; native pooled3.2435%/2.7625% uses different targets. Trainer
213.41/217.63s gives no speed benefit. New native/local gap9.219% and overshoots
already exist before PINN fitting; q/phase-band errors also worsen vs local.
Ten numerical checks, extra schema check,19configs, native dt refinement,
identical initialization/37sources and both CPU replays pass. Figures/results/
README/index archived; no promotion, NS, tuning or SOAP work. All trainers exit0.

[All-nonlocal spatial derivatives assessed](tasks/archive/fully-nonlocal-flow-assessment.md).
Feasible through paired gradient/divergence plus existing nonlocal Laplacian;
time JVP and parameter AD remain. Primary vector-calculus/Stokes sources and
independent NumPy checks support adjointness/mass/capillary-work identities.
Naive symmetric nodal G also has a checkerboard null mode; stability is not
automatic and -DG differs from the existing pair Laplacian. No production,
reference, default or training change, no full NS/accuracy/speed claim.

[Two-phase scope clarified](tasks/archive/twophase-ns-derivative-scope.md).
Current pilot solves prescribed-flow mixed CH for phi/q, with no NS/pressure
or solved velocity. Recommended advection uses conservative Fourier first
derivatives; legacy uses AD. Time uses JVP and diffusion uses nonlocal
Laplacians (AD Hessians in local control). README/module wording corrected;
no numerical/default/training change. Prior phase errors do not validate CHNS.

[Two-phase map is optional](tasks/archive/twophase-optional-map.md).
Base/conserved presets explicitly default to additive; enable cubic via
--config.phase.map=inverse_cubic. Hard IC/mass, flux, PDE losses and budget
remain the same; historical cubic research controls are explicit. All18configs,
both CLI parses, two focused checks and saved-config checkpoint replays pass.
README/development/current artifact metadata updated, old results preserved.
No retraining, changes to other examples or SOAP work; no Git operation.

[Fresh two-phase map ablation completed](tasks/archive/twophase-map-ablation.md).
Two matched seed42 nonlocal cold starts, current two-PDE loss and SOAP5000+
L-BFGS1000: direct/cubic common-local pooled phi3.2006%/3.2522%, terminal
3.6215%/2.9159%, terminal interface14.8632%/11.6362%; trainer212.38/217.83s.
Removing the map does not materially worsen pooled error; cubic improves late
times here but not all times. One pair and known rerun variation do not prove
stable ordering. Both CPU checkpoint replays, initial/36source/config parity,
two-loss and roundoff-mass checks pass; q errors~60% remain. README/index/raw
metrics/curves/fields archived; defaults unchanged, no SOAP investigation.

[Redundant hard-mass loss removed](tasks/archive/twophase-remove-redundant-mass-loss.md).
Two-phase exact-grid-mass presets now compute only two PDE losses and omit the
mass weight; legacy unprojected presets retain three soft-loss components. Old
saved hard weights are ignored, parameter loading unchanged, mass diagnostics
retained. Three focused checks and18configs pass; README/development updated.
No retraining or claimed L2 change; old artifacts preserved and SOAP deferred.

[Two-phase map/conservation explained](tasks/archive/twophase-map-conservation-explanation.md).
Inverse cubic is a convexified local-energy coordinate with finite pure-phase
sensitivity/unbounded range, not inverse-J residual weighting or a convex PINN
loss. Generality is conditional; map-only stable accuracy advantage unproven.
Distinguish conservative spectral flux/pair diffusion from exact mean projection;
finite residuals need not conserve mass without the constraint. Algebra/zero-mode
checks and figure archived, no production/default/reference/training changes.

[Precipitate local controls completed](tasks/archive/precipitate-local-comparison.md).
Same initialization/config/budget, spectral mechanics and SOAP5000+LBFGS500:
common-local pooled phi spectral0.0599%/AD0.0677% vs historical nonlocal0.5420%
(native0.0387%); trainer183/680/183s, compiled steps19.95/91.53/19.70ms.
Two full runs/replays exit0; manufactured derivatives/gradient, initial/source
parity and independent saved-field/energy checks pass. q initial errors~30%
remain; one-seed results are not a stability/statistical claim. Explicit presets,
README/index/plots/metrics archived; defaults unchanged, GPU idle, SOAP deferred.

[Elastic precipitate pilot completed](tasks/archive/elastic-precipitate-pilot.md).
Prior snapshot devfbe98d0 -> main regular mergefcdfe91 pushed before development;
returned to dev. New periodic equal-modulus coherent CH pilot uses phase PINN
and exact differentiable spectral elastic equilibration, with nonlocal phase
diffusion. Refined independent reference and SOAP5000+LBFGS500 run finish183s:
native pooled phi0.0387%, solved-u0.0185%, stress0.0354%; q3.81% remains weaker
at t0. Common-local phi0.542%; mild anisotropic shape change verified against
uncoupled control. Four numerical tests, corrected CLI smoke, CPU checkpoint
replay and energy/equilibrium/shape audits pass. Figures/config/index archived;
new development uncommitted on dev, GPU idle. SOAP stability remains deferred.

[Next phase-field candidates assessed](tasks/archive/phasefield-next-cases.md).
Recommend elastic precipitate shape relaxation for new mechanical coupling;
thermal dendrites for anisotropic morphology, fracture as a harder extension,
periodic spinodal decomposition as the cheapest control. Primary PFHub/model
sources checked; operator reuse and reference routes are proposals, not measured
advantages. Derivative replacement and native nonlocal physics distinguished.
No implementation selected or training/default change. SOAP remains deferred.

[Deferred SOAP issue](issues/soap-repeatability.md): low-priority optimizer
stability side investigation, deferred at the user's request; not a mainline
blocker. Resume only on explicit request. Evidence and current settings retained;
earlier proposed follow-ups below are historical, not active work.
[Documentation task](tasks/archive/soap-issue-record.md).

[SOAP implementation audited](tasks/archive/soap-implementation-audit.md).
Pinned SOAP_JAX0.2.2 source matches upstream/prior probe. Author PyTorch and
independent pytorch_optimizer each pass16gradient-stream controls; shared-basis
FP64parameter maxgap2.22e-16, moment5.88e-15; actual factory1005-step clipping/
LR/bias recurrence passes. No tested core translation/call defect explains
current instability. Rank-deficient initialization sensitivity reproduces in
references at their precision. Older stale-EMA defect is already fixed; nonzero
weight-decay ordering really differs but current zero-decay path is unaffected.
Native precision/reference-loading caveats explicit. Sources/licenses/raw states/
scripts/analysis archived; README/development/index updated. No dependency,
production/default/reference/full-training/Git change; workers exit0, task none.

[Same-seed vortex repeatability diagnosed](tasks/archive/twophase-repeatability.md).
Six GPU processes/12short paths/814updates hold initial/time arrays exact.
Original gather-transpose noise gives1.39e-17gradient differences; SOAP initial
Gram example64x64rank2 amplifies them into25–26% actual second-update changes.
Identical frozen-gradient optimizer repeats are exact; canonical paths diverge
at update2. Equivalent symmetric adjoint removes tested in-process100-step
noise, but identical-source independent processes still differ6.94e-18 in
initial gradients and diverge: partial isolation, not complete fix. Specific
cross-process kernel cause/final-L2 benefit unproven. Prioritize stable weak-
direction handling, two same-seed complete runs, then small predeclared paired
cohort. Defaults/production/reference/history untouched; README/index/decision/
raw evidence/scripts archived, checks pass. All workers exit0; no active task.

[Matched latest local AD/nonlocal vortex comparison completed](tasks/archive/twophase-matched-local-comparison.md).
Fresh serial seed42 pair holds cubic/original/mass/spectral-flux/FP64/ordinary/
SOAP5000+LBFGS1000 equal, changing only diffusion. Common-local pooled phi
local3.36%/nonlocal3.78%, terminal3.22%/3.27%; trainer924.23/211.15s,
compiled SOAP83.16/16.73ms. Nonlocal4.38x faster; historical2.84% nonlocal
rerun variation prevents stable accuracy-order claim. Exact initial/config/
36source parity,1manufactured combined test, CPU/GPU field replays and periodic
checks pass. Curves/scripts/raw metrics/config/README/index archived; defaults/
reference/old paths/Git history unchanged. Audit-only backend/bitwise assumptions
corrected with failure logs retained. All workers exit0; no active task/blocker.

[Two-phase accuracy study completed](tasks/archive/twophase-operator-preconditioning.md).
Recommend explicit train_vortex_conserved via train.py: inverse-cubic coordinate,
ORIGINAL residual, consistent conservative nodal flux, exact grid mass, fixed
integer modes,FP64/ordinary,SOAP5000+LBFGS1000. Independent42/2026 common-local
pooled phi2.84%/2.91%, terminal2.34%/2.56%;1% unmet and band errors~12.8%.
Discovered/repaired trainable periodic frequencies; retain old checkpoints with
BC caveat.16controls retain failed Schur/causal/larger-network/long-refinement
results; lower loss does not guarantee lower L2. Reference unchanged/converged;
21distinct focused checks,17config validations,16CPU/exact GPU replays and
trained periodic/initial/source audits pass. Plots/scripts/JSON archived, index/
README/development updated; no other defaults, paths or Git history changed.
All workers exit0; no active task or blocker.

[PINN residual-preconditioning literature search completed](tasks/archive/pinn-residual-preconditioning-literature.md).
Direct prior art confirmed: Liu2024 Eq20 uses ||P^-1(Au_theta-b)||² with ILU;
Song2025 builds PDE-Jacobian/ILU residual preconditioning with adjoint gradients.
Earlier FEINN2023 uses preconditioned weak residual norms; Wen2026 preprint
covers neural-operator residual losses including Allen–Cahn and block Stokes.
Broad idea/phase-field application alone is not new. Inverse-map versus PDE-
Jacobian distinctions and narrower nonlocal/conserved-field questions remain
research hypotheses, not novelty claims. Primary formulas/dates/scopes checked;
no training/code/default/reference/Git operation. No blocker or active task.

[Residual preconditioning theory clarified](tasks/archive/residual-preconditioning-theory.md).
General S=W R(G) separates state mapping, nonlinear left residual scaling and
parameter optimization; B=DG differs from PDE/neural Jacobians. For mixed CH,
chemical-row scaling is an additional block weight, not forced by diag(g',1).
Root derivative WAB and similarity limitation rule out automatic conditioning/
stiffness claims; finite neural least-squares errors change despite equal roots.
Glasner2001/PETSc primary sources and independent NumPy derivatives/Newton/
conditioning/weighted-minimum checks pass. Operator/block/target-error directions
remain proposals; no application/default/reference/training/Git change. Source/
checks/task archived, memory updated; no active task or blocker.

[Vortex residual/map experiments completed](tasks/archive/twophase-vortex-residual.md).
Six serial5000-update controls preserve CH/reference/seed42/parameters. Bounded
physical inverse-J weights worsen common-local phi0.07997->0.21941/0.27171.
Unbounded leaky-tanh +original residual improves phi0.04192, band0.30327->0.16045,
width-proxy error48.4%->8.57%, contour0.004804->0.002797; area worsens2.74%->4.29%.
Exact inverse-map both slightly improves globalphi0.04019/q/area2.07%, but worsens
band0.17455/width17.77%/contour0.004478 versus map-only. Attribute interface gain
to representation, not residual scaling; no all-metric/robustness/SOTA promotion.
12focused tests,6CPU model replays, exact initial leaves/reference/stage-frozen
sources and current legacy replay pass. All figures/scripts/metrics archived,
6new runs registered; defaults/reference/old paths/Git unchanged. Workers exit0,
GPU idle; README/development/artifact/memory updated, no blocker/active task.

[Reversed-vortex deformation pilot completed](tasks/archive/twophase-vortex-pilot.md).
Explicit periodic variant adds prescribed U2/T2 vortex, thin rounded droplet and
independent graph64/local128/256 Fourier references; time/grid errors<=3.62e-5/
1.99e-7, phase-mass drift<=2.22e-16. Matched5000-update controls learn crescents
and return. Common-local t1/t2 phi: NL0.05479/0.07889, AD0.07463/0.10186;
compiled3.30/10.31ms, trainer76.85/136.54s including JIT/eval. NL area drift3.37%
is worse thanAD2.90%; interface-band errors34.3%/37.4%, no high-accuracy or
statistical dominance claim.7focused tests, independent CPU replay, exact frozen
GPU fields and paired initial/config/source/reference audits pass. Explicit CPU
replay tolerance5e-6 records backend roundoff<=3.31e-6; no stored fields changed.
Figures/curves/scripts/results archived;2new runs registered. Constant-flow base
and old configs/references retained; no coupled NS, default/Git change or blocker.
All workers exit0/GPU idle. README/development/artifact/memory updated.

[Periodic prescribed-flow phase-field pilot completed](tasks/archive/twophase-transport-pilot.md).
Before development, prior results committed dev05780d4, main regular merge21fe6d9
pushed, then returned to dev. New twophase2d case has explicit periodic advective
CH, mixed local AD/nonlocal diffusion, independent Fourier reference and direct
saved-field audit. Both5000updates finish: own-targetphi0.002297/0.005121,
common-local-target0.008542/0.005121; no nonlocal accuracy-superiority claim.
Trainer66.7/91.1s includesJIT/eval,area drift0.264%/0.897%. Independent reference
refinements<8.3e-8/time,<2.1e-11/local-space; graph/local gap0.82%.5focused
checks,actualstartup,CPU reload/source/initial/config/field audits and6case319dir
inventory pass. Curves/fields/scripts/results archived; no workers/blockers.
Root/example README,index/navigation/memory updated. New development remains
uncommitted on dev; only the old snapshot was pushed. No coupled NS yet.

[Attached two-phase PINN paper reviewed](tasks/archive/two-phase-pinn-paper-review.md).
Qiu2022 uses local CHNS/four-output AD, sequential time windows and material
coefficient clipping; forward loss uses PDE+IC/BC. Reversed-vortex transport
then1000:1bubble are useful validation stages. Actual velocityL2 is3.56–15.12%,
area drift~1.12%; no exact conservation/sub-percent field claim. Rendered p7/p14
confirm printed potential/initial-profile inconsistencies; BC/chemical scaling/
coordinates require resolution before implementation. Same-group2022 MIM already
outputs chemical potential, so mixed AD is a required efficient comparator for
proposed nonlocal mixed form.35pages extracted/primary follow-ups/math/doc/diff
checks complete. No implementation/training/default/Git change or blocker.

[Next phase-field cases assessed](tasks/archive/phasefield-case-recommendations.md).
Recommend periodic2D CH spinodal decomposition for a scalar extension, or
matched-density CHNS droplet relaxation for two-phase flow. Primary PFHub/
CHNS/nonlocal-CHNS sources consulted; conservative transport/capillary energy
cancellation checked analytically. Existing Laplacian alone does not supply
compatible flow gradients/divergence; separate derivative approximation from
nonlocal interaction physics. No implementation/training/default change or
Git operation; source/doc/diff checks pass, no blocker or active task.

[Ordinary-policy defaults completed](tasks/archive/ordinary-policy-defaults.md).
Six new serial5000-update cold starts plus10KKS/1full-grid melting CPU reloads
support per-case defaults: corrosion ordinaryFP64, electro/melting ordinaryFP32.
Corrosion42/2026 phi0.002499/0.006833; electro0.004757/0.003399, while ordinary
FP64 electro42 fails0.299757. Melting new explicit ordinary full-gridphi3.9053e-5.
Original physics/residuals/data/SOAP/horizons retained; seed remains run provenance.
Corrosion late spikes/Dq persist; no uniform stability/global-optimum claim.
Shared explicit policy/conflict parsing, README/index, historical preset precision,
curves/fields/manifests/source/scripts archived.11focused tests +7affected-policy
checks,2canonical GPU smokes and default/reference/dtype/source audits pass.
All workers exit0; no Git history operation or blocker. Final JSON/link/artifact/
diff verification is recorded in the archived validation file.

[Other-case GPU policy audit completed](tasks/archive/phasefield-gpu-policy-audit.md).
Four actual-config FP32 workers finish125 updates each (50 warm-up/75 timed).
Electropolish deterministic/ordinary197.73/10.98ms (~18.0x); melting468.26/26.62ms
(~17.6x). Initial/all-pool/time/data arrays exactly match. HLO verifies3032-edge
and11929-edge loops, including backward; independent numerical errors<3e-7.
Electropolish default enforces deterministic flag; melting inherits environment
and is already ordinary when unset. Other two entries only statically inspected;
1D missing default config is documented without choosing physics. Source/raw
results/HLO/plots archived, READMEs/index updated. No long accuracy cohort,
production/default/physics/Git change; all workers exit0 and no active task.
Final21-artifact/26-source/strict-JSON/link/HLO/input/output/308-dir/diff checks pass.

[Ordinary FP64 accuracy control completed](tasks/archive/ordinary-fp64-accuracy.md).
One additional seed42 cold start finishes5000 updates/50s in196.5s through
train.py. Finalphi/c0.002499/0.002944 versus default0.008835/0.010426; observed
~6x faster with different cache histories. Latephi median0.004685 improves,
mean/max do not; Dq late spike18.665 prevents a uniform stability/transport claim.
All four initial parameter values match exactly; independent CPU reload/dtype/
config/reference/transport/topology/history/source checks pass. Curves, fields,
source/direct paths archived; README/index updated, workers exit0, no default
or Git history changes. No active task remains.
Final10-artifact/source/strict-JSON/link/output/5-case308-directory/diff checks pass.

[Compiler-policy explanation completed](tasks/archive/compiler-policy-explanation.md).
Frozen GPU/HLO probe confirms current deterministic graph3032-edge scatter
lowers to3032-iteration while; ordinary retains scatter. Ten repeats give exact
forward/input-gradient deterministic results and~6e-8 relative ordinary variation.
Both policies pass earlier SciPy checks at~1e-8. First full-run losses differ
1.5e-7, adaptive weights diverge44% by200: supports sensitive training paths,
not a guaranteed deterministic precision advantage or uniquely isolated cause.
Probe/source/HLO/metrics archived; README clarified. No retraining/default/Git
change; workers exit0 and no active task remains.

[Full local/nonlocal cold starts completed](tasks/archive/local-ad-full-training.md).
Four paired FP32/seed42 cold starts finish5000 updates/50s through train.py.
Own-reference finalphi/c: graph default0.008835/0.010426, ordinary0.020680/0.024565,
local tensor0.346112/0.373221, local joint0.411813/0.425739. Local adds matched
no-flux sixth loss; independent401 FD reference retains1.42%/1.68% spatial
refinement discrepancy, so no sub-percent continuum claim. Defaults unchanged;
ordinary faster but worsens field/late metrics while improving q/Dq. Twelve
focused checks, two canonical CPU smokes, four full runs and independent CPU
checkpoint/initial/config/transport/topology/roundoff audits pass. Curves, fields,
source and direct paths archived; all workers exit, no Git history operation.
Final source/strict-JSON/link/output/5-case307-directory and diff checks pass.

[Nonlocal advantages assessment completed](tasks/archive/nonlocal-advantages.md).
Primary-source review and actual5151-node graph checks support lower spatial
regularity requirements, reciprocal pair/positive-energy structure and finite
interaction-scale/representation flexibility. Bounded high-frequency operator
amplification is derived, not a demonstrated PINN convergence advantage.
No matched local accuracy/robustness superiority claimed; open Dirichlet
reservoirs, soft residuals, stored-diagonal roundoff and local-target bias are
explicit. Sources/check/raw JSON archived; no training/production/Git change.

[Nonlocal versus local-AD performance completed](tasks/archive/local-ad-performance.md).
All12 serial GPU workers exit0:6 primary,4 graph-policy kernels,2 ordinary
full-update controls. Same model/IC/BC/SOAP,3760–4032 active PDE pairs,
50 warm-up+75 measured updates. FP32 graph/tensor/joint229.31/11.98/16.12ms;
FP64266.11/31.32/47.77ms. Deterministic aggregation~38ms vs ordinary~0.17ms;
ordinary graph complete12.49/35.13ms shows policy overhead is a major cause.
Joint time-only grad/JVP controls agree; random pairing does not forbid JVP.
Manufactured derivatives/chain rule/data/mixed gradients, raw timing/refresh/
dtype/count/initial-loss checks pass; plots inspected. No convergence, boundary
equivalence or ordinary-policy reproducibility claim. Production code, defaults,
reference and Git history untouched. Sources/raw JSON/figures archived; GPU idle.
All27 archived files/17 JSONs, source/link/status/operator checks,5-case303-dir
inventory and git diff --check pass. Updated memory: this dashboard, DECISIONS.md
and the self-contained archived task.

[Time-JVP structure clarification completed](tasks/archive/time-jvp-structure.md).
Within-update times share a spatial subgraph; updates cycle the pool. All three
nonlocal phase-field paths already use time JVP. Independent CPU float64 toy
checks match vmapped time grad exactly and verify zero cross-time blocks.
No Nx*Nt-factor speedup claimed against properly vectorized grad; no timing,
training or production change. Evidence archived; git diff --check passes.

[Corrosion FP64 accuracy control completed](tasks/archive/corrosion-fp64-accuracy.md).
One paired seed42 cold start completes5000 updates/50s through train.py, with
identical promoted FP32 initialization/data/time draws/stored graph coefficients.
FP64 final phi/c0.002677/0.003207 vs FP32 0.008835/0.010426; free q halves,
Dq1.703->1.220. Late phi median improves~16%, but max/mean and Dq median worsen;
no stability/error-floor/cross-seed claim. Default FP32 retained; explicit
compute_dtype option, dtype records, README/index and archived scripts/results/
plots delivered. Seven focused tests, exact GPU initialization/graph parity,
independent CPU reload/AD/transport/topology and frozen-inference controls pass.
FP64 subprocess exits0 (~24.9min including compilation/evaluation); GPU idle.
All11 archived artifacts/JSON/source/link/run checks,5-case303-directory
inventory and git diff --check pass. Memory updated: this dashboard,
DECISIONS.md, DEVELOPMENT.md and the archived self-contained task.
No reference/default physics change or Git operation.

[Local FP64 performance benchmark completed](tasks/archive/fp64-performance.md).
8 serial GPU measurement workers and1 dtype-lowering audit finish. On reported
RTX3090/48GiB, same-size corrosion SOAP/GradNorm75 measured updates after50
warm-up average0.2033s FP32 vs0.2267s FP64 (~11.5% slower). JAX peak active
memory1.535->3.063GiB; compile~156s both.5000 update-only estimate16.94->18.89min,
excluding JIT/evaluation/saves. All model/PDE/SOAP tensors are true FP64; only
existing scalar Optax LR schedule stays FP32. Corrected GEMM/type controls,
finite/dtype/shape/refresh/JSON/syntax/link/diff checks pass. No full accuracy
run/default precision change or Git history operation. Sources/results archived;
GPU idle. FP64 L2 benefit remains untested.

[Corrosion robust-objective optimization completed](tasks/archive/corrosion-robust-objective.md).
Local dev snapshot e4a0764/f65e6f8 is committed; no merge/push. Four new5000-step
cold starts and a9-model CPU audit finish. Frozen bounded weighting rescues
failed7/2026 to phi~0.01 but regresses42/late metrics. Corrosion default now uses
original phase with unchanged raw-edge CH/nodal data/scales: observed42/7/2026
worst phi/c0.01074/0.01273, late phi median0.009514; historical bounded best remains
separate. Dq~1.70–1.76 remains unresolved.3 numerical/5 entry checks, default
strategy parity, paired initial/config checks,9 final phi/c reloads, topology,
strict JSON/plots/diff and artifact index5cases302dirs pass. All processes exit,
GPU idle. Sources/manifests/curves/fields archived; optimization changes uncommitted.

[Cross-seed mechanism diagnosis completed](tasks/archive/phasefield-seed-mechanism.md).
7 final/10 intermediate CPU probes show nodal IC/BC fit is not sufficient:
failed corrosion2026 BC error6.28e-5 is smaller than successful42's2.45e-3.
Same-data/initial-state seed7 original->bounded gives phi0.00422->0.422.
Failed PDE losses remain high on covered nodes;35 never-core nodes contain
only0.935% phase SSE. Stiff coupled objective/optimization-path sensitivity is
supported, unique mechanism/resampling benefit unproven. Electropolish failure
also has poor full-grid IC fit. Sources/metrics/plot archived; no training,
physical/default change or new remedy claimed. All probes/checks pass.

[Three-case independent-seed validation completed](tasks/archive/phasefield-seed-stability.md).
All3 predeclared seed2026 cold starts finish5000 updates. Corrosion bounded
phi/c0.4070/0.4310 fails with spurious late phase bands (seed7 also fails);
electropolish original/random0.002832/0.003427 succeeds but retains historical
seed42 failure; melting full-grid graph phi5.9857e-5 agrees with seed42.
Canonical configs now omit seed; run-only --seed(default42) is resolved into
saved metadata for replay. No seed shopping, math/reference changes or default
replacement. Full CPU/transport/topology audit,10 focused tests,18 legacy preset
parity checks and raw curves/field galleries pass. READMEs/index retain all
outcomes; all processes exit/GPU idle.

[Three-case SOTA defaults completed](tasks/archive/phasefield-sota-defaults.md).
All3 canonical train.py entries load train_base.py: corrosion bounded/raw-edge/
s_bc/nodal seed42(terminal phi/c0.001433/0.001719), electropolish original/
physical/random seed7(0.001138/0.001358), melting IC-only latent/SOAP seed42
(graph phi6.1097e-5). All5000 updates/lr1e-3; READMEs/index give exact configs,
paths and seed/transport caveats. Legacy configs/data/log paths preserved.
30 numerical +3 entry checks,3 real reduced CPU smoke runs, exact config/
same-device initialization/sampler audits and artifact checks pass. No full
retraining or Git history change; all processes exit.

[Electropolish3x5 best-phi panel completed](tasks/archive/electropolish-best-phi-panel.md).
Reference/prediction/absolute error at t*=0,.25,.5,.75,1 use one selected random
seed7 checkpoint005000, lowest terminal phi L2 among45 current-cohort saved
models(0.001138). Shared color scales, CPU reload, raw NPZ/JSON and PNG/SVG
exports verified. No training/reference/default changes.

[Electropolish convergence curves completed](tasks/archive/electropolish-convergence-curves.md).
Raw200-update histories for random/candidate seeds42/7 are plotted separately
for terminal5 s and pooled0–5 s phi/c L2. PNG/SVG/raw JSON preserve late
oscillations and the successful random seed7 counterexample. Histories/endpoints,
figure layouts and artifact checks pass; no retraining or default changes.

[Electropolish field optimization completed](tasks/archive/electropolish-field-optimization.md).
Nine fresh5000-update/5 s runs,11-model CPU audits and3 focused sampler checks
finish. Original residual/full-grid analytic IC/graph BC/s_bc gives seed42
phi/c0.002106/0.002518 vs current random0.290928/0.299013; seed7 candidate
0.002389/0.002869 is worse than successful random0.001138/0.001358. Candidate
q improves both seeds, but seed7 Dq/c_t regress; no universal/default promotion.
Both candidate seeds have one interface at all21 times. Temporal window helps
full latent0.5194->0.04677 but does not beat original. Sources/direct paths,
mixed outcomes and plots archived; all processes exit. Physics/reference unchanged.

[Electropolish representative physical fields completed](tasks/archive/electropolish-physical-field-gallery.md).
Saved seed42 original/transformed models at0/1/2.5/5 s: phi/c fields, signed
errors and final centerline profiles. Both positive-time fields have three
vertical interfaces in all101 columns; reference has one. Terminal phi/c
0.28604/0.29443 original versus0.34398/0.37920 transformed. CPU reload and
per-time L2 checks pass. These earlier pilots omit later corrosion s_bc/raw-edge
changes. No training, reference, numerical or default changes.

[Corrosion representative physical fields completed](tasks/archive/corrosion-physical-field-gallery.md).
Saved seed42 original/raw-edge models at0/10/25/50 s are plotted against the
unchanged graph reference: phi, c, driving q and signed errors. Positive-time
phi/c improve in this control; initial phi/c/q fits are worse. Consistent color
scales, interface contours, CPU field reload and terminal L2 checks pass.
No training, numerical or default changes.

[Transport optimization completed](tasks/archive/corrosion-transport-optimization.md).
Twelve5000-update/50 s controls,15-model CPU audit and22 focused checks finish;
full baseline and both pre-tail3000/pre-RMS4000 trajectories replay exactly.
Stable KKS/edge evaluation improves phi/c in both seeds, but seed7 q/Dq
regresses; no all-metric default promotion. Late-rate, bounded and cold/late
RMS controls do not give a robust joint field/transport improvement. Keep
original phase/KKS/scales/defaults. All processes exit; sources, commands,
metrics, plots and direct paths are archived. Remaining research: independent
concentration parameters/true field-block refinement, not yet tested.

[Normalized concentration data controls completed](tasks/archive/corrosion-normalized-data.md).
Six seed42 controls, paired seed7 validation and an exact5000-update baseline
replay finish at50 s;16 focused tests and eight independent CPU audits pass.
Original residual+s_bc lowers terminal phi/c L2 to0.01177/0.01402 (seed42)
and0.00620/0.00738 (seed7), versus~0.571/0.547 physical-data controls.
Matched graph BC/IC and explicit concentration normalization options are
delivered. Full latent/stronger IC do not improve later fields. Pit-mouth Dq
hotspots and late variability remain; no production-default or pre-scale
change. All processes exited; report, sources, metrics, plots and direct run
paths are archived. Future targeted work: near-mouth free-node transport.

[Boundary/initial drive audit completed](tasks/archive/corrosion-boundary-drive-audit.md).
Six saved-checkpoint CPU evaluations separate pit/solid BCs and initial q/Dq.
Collapsed original has pit c error~2e-9 but solid error0.035664 (whole drive
scale); initial q is nearly constant despite plausible phi. Reference IC
already supplies a q gradient. Frozen BC substitutions do not reconstruct it.
Finite tanh IC/zero-mouth corner mismatch is explicit. No training/PDE/BC/
default change. Next separate test: normalized concentration BC/IC accuracy
with explicit corner treatment and prior reproducibility limitation retained.

[Corrected stagger controls completed](tasks/archive/corrosion-stagger-experiment.md).
Production adaptive state now excludes temporary masks/bias; research stagger
is wired and verified. Four5000-update/50 s runs do not recover concentration:
phi~0.569, c~0.547, c_L near0 and Dq error~1. Fresh original joint and an
archived-step replay also fail to reproduce the earlier accurate control.
CPU first-batch gradients match; identical-input GPU gradient variation is
measured in old/new paths, without an isolated cause for final collapse.
Eleven focused tests and independent four-checkpoint/config/source/activation
and transport audits pass. Configuration defaults/pre-scales stay unchanged;
all processes exited. Next separate work: establish stable joint replay before
ranking further concentration optimizations.

[Concentration/stagger audit completed](tasks/archive/corrosion-concentration-stagger-audit.md).
Research controls all disable stagger; production single-pit uses1000/25 loss
alternation without freezing fields. Actual loss probe reproduces masked-weight
cache failure at switches without GradNorm refresh. At that audit stage the
research runner rejected unsupported nonzero stagger (now implemented above).
Seven-run decomposition shows better total
c L2 can hide worse c_L/q transport: seed42 cap11 Dq relative error2.94->8.47.
That audit made no production/default/training change; the subsequent correction
and paired experiment are recorded above.

[Residual visual explanation completed](tasks/archive/residual-visual-explainer.md).
Chinese inline compression/residual schematic and measured frozen-corrosion
loss allocation are created. Both phase ranges,12 interaction states and
schematic arithmetic pass local checks; no real browser is available. The
explanation distinguishes output mapping, PDE imbalance, weighting and physical
L2. No training/PDE/default changes.

[Bounded corrosion experiments completed](tasks/archive/corrosion-bounded-residual.md).
Eight new5000-update/50 s cold starts: eta0.1 (cap11) improves seed42 phi/c
0.03984/0.04722->0.01619/0.01928, but worsens seed7
0.02664/0.03160->0.03661/0.04343. Both no-scale arms have phi L2~0.512 and
lose the terminal interface. Retain original/KKS/pre_scale defaults; bounded
mode is research-only. Ten-checkpoint CPU audit and nine focused tests pass.
All processes exited; direct paths and figures are in the archived report.

[Corrosion latent mechanism diagnosis completed](tasks/archive/corrosion-latent-mechanism.md).
At a fixed accurate checkpoint, transformed AC allocates99.47% of absolute
residual to bulk (original8.54%); bulk/interface gradient-norm ratio0.030->2.89.
Exact graph/CH, concentration sensitivity and stiff near-solid-limit checks pass.
Evidence supports excessive bulk amplification and stiff coupling without proving
unique causality or a successful remedy. No training/PDE/default changes.
Bounded weighting was subsequently tested; see the completed experiment above.

[Corrosion KKS/scale ablation completed](tasks/archive/corrosion-kks-scale-ablation.md).
Four new5000-step/50 s controls reuse the validated graph reference. At AC1e-9,
terminal phi/c L2: KKS original0.04212/0.04992, KKS latent0.43679/0.44957;
direct-c original0.53846/0.53685, direct-c latent0.53645/0.52965. Smaller scaling
eliminates exactly-zero concentration derivatives in the KKS latent probe but
leaves extra phase bands. Direct c removes saturation yet fails to grow the pit.
No new setting beats prior KKS/original/AC1e-7, so defaults stay unchanged.
Explicit research CLI map/scale overrides and actual loss/weight logs are added;
seven focused tests and six-control checkpoint/config/initial-weight replay pass.
All GPU/FEM processes are idle; direct paths/figures are in the archived record.

[Matched corrosion/electropolish residual pilots completed](tasks/archive/coupled-latent-residual-experiments.md).
Both fresh same-graph references passed time-refinement checks; all four
matched5000-update runs finished with soft ICs. Terminal phi/c L2 original->latent:
corrosion50 s0.03484/0.04135->0.56839/0.54585; electropolish5 s
0.28604/0.29443->0.34398/0.37920. Defaults remain unchanged. Corrosion latent
concentration saturates (85.2% zero c_L derivatives); both electropolish arms
have extra interfaces. Inherited AC/CH scales were held fixed. Checkpoint probes
show phase preconditioning changes gradient balance and activates GradNorm
caps; neither scale retuning nor a causal KKS ablation was performed. All
processes exited. Detailed results, derivations, validation and direct paths
are in the completed task and replay artifacts.

[Corrosion reaction cancellation assessed](tasks/archive/corrosion-latent-residual-assessment.md).
Hard ICs excluded. Existing0-1 map is already(1+tanh(z))/2=sigmoid(2z).
Both double-well and concentration-driven phase reactions cancel its Jacobian;
the graph edge ratio is identical to melting. Derived bounded-reaction criteria
and preserved the original concentration residual. Actual1D/2D loss/operator
identity checks passed (float64~1.75e-15; FP32 convolution~9.51e-7), including
saturated-ratio verification with high precision. No corrosion training or
production configuration/code change; L2 benefits remain unmeasured.

[Residual-transformation generalization assessed](tasks/archive/icemelting-residual-generalization.md).
Scalar inverse-Jacobian residual scaling extends beyond geometry/dimension;
Allen-Cahn and thermal phase fields are natural validation targets. Glasner2001
provides direct nonlinear-preconditioning precedent. Conserved/multiphase models
need separate constraint/flux derivations. Chain-rule identities checked by AD
in1D/2D/3D (max discrepancy2.22e-16); no cross-model training was performed and
the validated melting implementation remains unchanged.

[Validated five-second default promoted](tasks/archive/icemelting-default-preset.md).
The canonical base preset is the selected5000-update SOAP configuration;
all training/model/operator/sampling parameters match the saved winner.
README/config and artifact provenance identify this default explicitly.
Configuration comparison, direct artifact paths and diff checks passed.

[Five-second SOAP versus Adam comparison completed](tasks/archive/icemelting-soap-adam-5s.md).
Current defaults: Tc5 s, lr_init0.001,5000 updates, SOAP, IC-only latent residual.
Matched5000-step full-grid terminal graph L2: SOAP6.11e-5, Adam7.40e-4;
last1000-update sampled medians differ by~5x. Adam's slightly lower analytic
error reflects cancellation against the approximate profile. Compiled step
times are close. SOAP's selected5000 checkpoint excludes later work before the
user-requested stop; Adam completed5000. No process remains running.
[Current melting strategy README completed](tasks/archive/icemelting-strategy-readme.md).
The example README foregrounds the IC-only latent default and documents loss,
sampling/optimizer/controls/L2/loading. Current defaults all use5 s; prior4 s
records remain explicitly historical. Documentation links/math/diff checks passed;
no numerical code, training or checkpoint change in this documentation task.
[Direct file-path loading without hashes completed](tasks/archive/icemelting-direct-path-loading.md).
User preference: explicit paths, no file hashing or hash-manifest dependencies.
Removed executable hash processing in the three melting runtime modules and two
replay scripts; preserved parameter/grid/shape/time checks. Two focused tests and
a config+checkpoint-only loading check passed. Old static hash records remain
historical data and are ignored by loaders. See docs/DEVELOPMENT.md for the rule.
[Melting run directory tags completed](tasks/archive/icemelting-run-tags.md):
best audited run is 20261024-ic-latent-coldstart-01-sota; former SOTA is now
legacy-raw-phase-soft-ic. All139 artifact files verified unchanged; current
paths, artifact index and replay scripts updated, original run identities retained.
[Default latent melting integration and cold starts completed](tasks/archive/icemelting-production-latent.md).
Production phase.formulation defaults to ic_latent; legacy/raw and hard-IC phase
controls remain selectable. Two fresh4 s runs finished2000 updates. Full-grid
terminal graph L2 is3.86e-4 with base batch/lr, and3.02e-5 with the bounded preset;
analytic L2 is1.40e-3 and1.03e-3 respectively. Curves show late oscillations.
Five numerical regression tests and independent checkpoint/IC/geometry checks
passed. These were historical4 s cold starts; the current5 s comparison is above.
[Original baseline metric audit completed](tasks/archive/icemelting-baseline-metric-audit.md):
historical SOTA full-grid 4 s L2=0.01412 vs analytic and 0.01466 vs the same-graph
numerical solution. Earlier phase-residual pilots also had hard IC and must not
be labelled the historical baseline. Logged pooled-time errors replayed.
[IC-only melting controls completed](tasks/archive/icemelting-weak-prior.md).
The user's strong-prior objection is accepted; the research runner now embeds
only the prescribed IC. Seven matched-budget pilots give 5 s transformed-residual
L2=8.7e-5–1.5e-4 versus 0.0038–0.0047 for phase-residual controls across three seeds.
Earlier motion-prior results remain conditional controls. The production
integration above keeps spatial discretisation and physical parameters unchanged.
