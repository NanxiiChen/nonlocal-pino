# Richer geometry-conditioned CH operator and smooth field visualization

Date:2026-10-29. Status: COMPLETE. Blocker: none. No Git history authorization.

User goal: modestly denser actual mesh, smooth physical-field plots,5-8fixed IC
waves/higher frequencies, and broader random circle positions/sizes, producing
more distinct physical evolutions for different geometry under one ambient IC.
Preserve previous runs and defaults; no explicit distance-bias architecture.

Context: completed generic point-stream model on T.75/two-mode IC has native
validationphi2.380/2.771% pooled/terminal, local2mean5.720/6.704%; response error
42.65%, amplitude94.74%. Prior mesh h.045 gives769..871centroid quadrature points.
Finite16train/8validation bank; all future fields are evaluation-only. Refer to
confined-ch-geometry-operator.md for source/validation/provenance.

Plan/acceptance:
- New opt-in canonical train.py preset: h.035,8explicit fixed ambient Fourier
  modes, broader two-hole positions/radii, same CH beta/M/epsilon/T and generic
  architecture/Adam5000/B4. This is a new physical benchmark, not a same-problem
  numerical-method gain. Same ambient function restricted to each domain,
  without a moving-hole envelope. Weak natural BC may adjust initially.
- Independently assembled native bank with time-refinement/mass/energy checks;
  representative matched local FEM checks before GPU work. Quantify true
  geometry spread on common physical probes before accepting the new case.
- Smooth plots use a documented topology-respecting linear reconstruction from
  physical cell values. Error metrics stay on original centroid values/areas;
  no interpolation across holes, no visual smoothing as extra numerical accuracy.
- Focused config/geometry/IC derivative and reconstruction checks, full5000cold
  training, saved-field/native/local/response evaluation, representative images.
- README/index/memory and task-end; only source/compact metrics tracked.

Next: implement explicit richer config, prepare native bank in fenicsenv, add
smooth visualization helper and focused checks while references run.

Implementation/validation so far:
- `train_operator_richer.py`: h.035,8fixed waves (amplitude bound.748), both
  centers x[.22,.78]/y[.27,.73], independent radii[.045,.12], clearance.025.
  No network/PDE/optimizer change. New bank root operator_geometry_richer_v1.
- `operator_plotting.SmoothMeshDisplay` uses positive incident-area projection
  and original-triangle linear interpolation onto400x320display grid; original
  topology and analytic circles mask holes. The evaluator explicitly selects
  flat/smooth, CLI default smooth; metrics always use original cells/areas.
- Two focused checks pass3.591s: new config/prior/fixed-IC NumPy-vs-AD Laplacian
  and local modal growth guide; display constants/bounds/linearity/no mutation/
  original-hole topology. Raw `/tmp/confined-geometry-richer-tests.txt`.
- Real smooth evaluation of old longer checkpoint succeeds in evaluation_smooth;
  all24condition metrics and train/test/local aggregates are EXACTLY unchanged.
  Raw `/tmp/confined-geometry-richer-plot-regression.txt`; figure inspected.
- Native preparation running; first21geometries pass time/mass/energy checks,
  with1152..1326actual cells so far. Next: finish bank, common-probe reference
  spread/reference gallery and two representative local FEM validations. No GPU
  training while FEniCS work runs.

Native preflight complete in568.867s:24geometries,1152..1326cells, phase temporal
refinement max8.04e-10; mass/energy checks pass. Reference common-probe response
on4627points has terminal relative spread15.2844%, RMSstd.0413934,maxstd.196482;
initial relative spread1.53e-16 after identical analytic IC correction. Old
two-wave/broader-time benchmark was4.3689%,RMSstd.0146921 on7556own common points.
These are different benchmark/distribution/support statistics, not a matched
method gain. New physical geometry dependence is clearly stronger. Maximum
reference phase.7601. Reference gallery16/20/23 inspected and saved under
`operator_geometry_richer_v1/reference_gallery/reference_fields.png`.

Local FEM currently running: first geometry16passes space/time gates
.317902%/.0003825%; mass7.46e-17, energy nonincreasing. Native/local phase bias
7.4923/8.8040% pooled/terminal is larger than old lower-frequency benchmark;
keep it explicit rather than claiming native-reference precision as local-PDE
accuracy. Geometry17pending; no GPU training until both are complete.

Both local validations now complete and passed. Geometry17mesh/time errors
.219579%/.0003037%; native/local phase7.0501/8.9340% pooled/terminal. All FEM jobs
exited before starting the5000Adam/B4cold run:
`/root/autodl-tmp/tf-logs/confined_ch2d/20261029-confined-geometry-richer-01`.
Raw training log `/tmp/confined-geometry-richer-training.txt`; GPU session64943.
Next exact resume: finish5000updates, run smooth evaluator and common-probe
model response diagnosis, then CPU replay/provenance audit and final artifacts/
docs/memory. Do not compare this changed benchmark L2 as a method-only gain.

Latest user clarification (authoritative): user accepts512precomputed training
geometries, without their FEM/native future solutions. Benchmark operator-only
preprocessing before estimating cost. Keep original8validation geometries out
of expanded training; full references only for monitor/validation. Earlier3.4h
estimate extrapolated the full native solve/refinement pipeline and does NOT
describe operator construction. Reflection augmentation code/test was added
but NO augmented run launched; it is not the selected route.

First richer run completed431.297s, native train2.016/2.229%, validation9.184/
11.200% pooled/terminal. Common-response error77.027%, amplitude74.423%,
correlation.6453; local2phi12.846/15.402%. Stronger physical geometry dependence
is real, but generalization to broad placement remains weak with16base inputs.
Smooth evaluator and response diagnosis complete; paths are under richer-01.
No GPU/FEM process is running. Next: operator-only construction timing and512
training-bank format/preparation with reserved original validation geometries.

Operator-only benchmark complete; user latest turn requests conceptual/time
alignment before further bulk work. Source/receipt:
[benchmark](confined-operator-precompute-benchmark.py),
[timings](confined-operator-precompute-benchmark.json).
Three geometries0/16/23,1162/1179/1180nodes, production-prep totals
1.353/1.335/1.282s. Means: mesh/quadrature.5182s, production assembly.0588s,
dense resolvent.2092s, compressed write.5368s. Serial512extrapolation677.37s,
compressed storage estimate5.43GiB; actual512size/time unmeasured. Independent
ordered assembler2.35s/input, with three probe fields agreeing against production
per geometry; excluded from production timing. No time integration/FEM solve.
First attempt failed because .venv lacks meshio; corrected script invokes
isolated fenicsenv meshing interpreter, then pure CPU production assembly.
Raw failure/success `/tmp/confined-operator-precompute-benchmark{,-fixed}.txt`.

Task-end for this alignment turn: no training/reference run queued;512bank
NOTgenerated yet. Exact resume: prepare512operator-only training inputs while
reserving original8validation geometries (new random draws exclude those IDs),
update bank loader/evaluator to distinguish operator-only inputs from labeled
monitor/validation references, then cold train/evaluate. Keep current physical
case/model; no reflection run. Preserve old checkpoint/dataset compatibility.
PROGRESS/DECISIONS updated; benchmark syntax and git diff checks passed.

2026-10-29 latest user authorizes parallel512precomputation (target~5min) then
training. Implement six spawned CPU workers, one BLAS/OMP thread each; no future
solves for496new inputs. Retain16labeled training monitors and8original validation
geometries, remapped512..519 with explicit accepted-draw provenance. Extend bank
loader/evaluator for operator-only entries, preserve legacy banks/checkpoints.
Next: focused partition/operator-only tests, generate512bank and measure wall,
then5000Adam/B4cold train/evaluate on identical8held-out domains.

Parallel preparation COMPLETE:124.703s, sixworkers/oneBLASthread,496newoperators
plus16existing training monitors and8held-out references,6559697287bytes (~6.11GiB
including meshes/configs). No trajectory/FEM solve. Exact canonical prepare
entry exits0; `/tmp/confined-geometry512-prepare.txt`. Format/config/loader tests
pass2in24.123s; first loader fixture failed because three toy points had no
neighbors, fixed to connected nearby points (failure retained in /tmp).
Training started canonical5000Adam/B4/seed42 under
`/root/autodl-tmp/tf-logs/confined_ch2d/20261029-confined-geometry-richer512-01`.
Raw `/tmp/confined-geometry512-training.txt`. Exact resume: finishrun, independent
smooth/raw/native/local evaluation + common-query geometry response, then CPU
field/initial-parameter/draw/reference-identity audit, README/index/memory.

Precompute production/operator check receipt: `confined-precompute512.json`.
New training inputs16/255/511 agree with independent ordered native assembly on
constant/sinusoidal/polynomial probes within1.82e-12. Legacy24bank reload passes;
all analytic IC arrays exactly equal stored initial fields. Raw
`/tmp/confined-geometry512-legacy-check.txt`. New cached shapes1129..1351nodes,
maximum96neighbors vsold1152..1326/max91. New bank~6.11GiBincludes auxiliary meshes.
Training initial evaluation occurs120.69s after start (loading/validation/padding/
first inference compile). At1500steps, train-monitor/testphi8.903/9.405% and
terminaltest11.084%, vsold same-step6.122/10.457% and12.536%; this is an interim
sign of reduced fitting gap, not final checkpoint selection. Budgetremains5000.

Cold512run completes524.416s (120.696setup,87.960firstupdate,43.217mswarm),
finalnative train-monitorphi4.314/5.064%,heldout4.690/5.454%,mu11.661/14.621%.
Compareold16testphi9.184/11.200%; worstvalidationphi13.715->6.725%. Native/raw
independent evaluator passes; local2phase12.846/15.402%->8.976/10.431%,positive
mu28.168->17.947%. All24labeled predicted trajectories preserve mass and sampled
energy is nonincreasing; no future arrays for496othertraininginputs. Devicepeak
active16.300GB (~15.18GiB), retainedpool38.184GB; no memoryfailure.
Native eval `/tmp/confined-geometry512-evaluation.txt` exits0. CPU checkpoint/
initial/draw/reference audit and independent common-query geometry diagnosis are
running concurrently (CPUJAX/GPUJAX, no FEM/MPI work). Next: finishboth, generate
comparison plot/metrics, inspect representativefield, finalREADME/index/memory.

Final task-end — COMPLETE, no blocker or queued training/optimization:
- Six-worker preparation124.703s meets user~5min target.496newoperator-only
  inputs +16reused training monitors; exact8held-out geometries remain excluded.
  Cache~6.11GiB, no new PDE/FEM evolution solves, directpaths/nohashes.
- New explicit richer512config/precompute CLI uses canonical train.py; spawned
  CPUworkers/oneBLASthread, isolated meshing. Versioned unlabeled-bank loader
  validates analytic IC/graph/metric/provenance, pads fixed shapes and rejects
  future fields in operator-only files. Evaluation uses16labeled train monitors
  and8held-out cases; old full-reference banks remain loadable.
- Cold5000Adam/B4complete524.416s, same physical/model/optimizer/initial weights.
  Nativephi9.184/11.200%->4.690/5.454%; local2FEM12.846/15.402%->8.976/10.431%.
  Independent field/chemical identity, hardIC/mass and sampledenergy checks pass.
- Common4627probes and reference variation exactly unchanged: geometry-response
  error77.027->35.514%, amplitude74.423->88.478%, correlation.6453->.9362.
- All20kactualdraws cover512inputs20..68times with no reserved validation input.
  CPU24fieldreplaymax1.638e-15; initial parameters and reference arrays identical.
  Focused2tests pass24.123s; legacy24reload passes; three independently assembled
  newoperator probes agree<=1.82e-12. Earlier disconnected toy-fixture failure
  retained and corrected; no physical/numerical fallback used.
- Native/local bias7-9% remains; eight reused development domains and one seed
  do not establish broad statistical/continuum/topology generalization. No
  reflection run or time-window/single-IVP default change. No Git action.
- Artifacts: richer512-01/model_final.eqx,config.json,training_draws.npz,
  evaluation/{metrics.json,fields_512/516/519.png,convergence.png},
  geometry_effect/{metrics.json,geometry_spread.png},comparison/{metrics.json,
  convergence.png/pdf}; source snapshots and bothold/newbanks remain external.
  Representative512field and comparison curve inspected. Only code/Markdown/
  compactJSON receipts are in repository; no new images/arrays/checkpoints.
- Verification logs: /tmp/confined-geometry512-{prepare,tests-fixed,legacy-check,
  training,evaluation,response,audit,comparison}.txt. Scripts/receipts archived
  alongside this task. Syntax/strictJSON/currentartifactlinks/diff checks pass.
- Memory files changed: docs/PROGRESS.md, docs/DECISIONS.md and this archived task.
  Exact resume: no required work remains. Further input-count/seed/mesh controls,
  faster cache loading or continuum-kernel refinement require a new user task.
