# Physics-only CH operator over parameterized perforated geometries

Date:2026-10-29. Status: COMPLETE; no blocker or queued training.
The chronological progress below preserves intermediate results and failures;
the final result/verification section supersedes earlier next-action paragraphs.
User requests an experimental random-geometry operator: fixed initial condition,
random positions/radii of two holes, simplified coarsening first. NSCH deferred.
No commit/push authorized. Preserve prior windowed single-IVP default/artifacts.

Acceptance:
- Opt-in through canonical train.py, explicit geometry distribution/validity,
  fixed ambient IC restricted to each domain, simpler short CH evolution.
- Geometry boundary point encoder; shared-space/time decoder; own physical
  mesh areas/visible regional operator per geometry, fixed padded JAX shapes.
- Initial pilot uses a finite train/held-out bank drawn from continuous geometry
  parameters; not online infinite geometry. No positive-time labels in training.
- Independent native references with time refinement; at least representative
  local FEM checks with matching weak natural BC/IC to separate operator bias.
- Meaningful numerical/gradient/mask tests, cold-start training and held-out
  field/mass/energy/geometry-sensitivity evaluation, local scientific figures.
- Record speed/memory and all failures, README/index/memory/task-end. No images/
  arrays/checkpoints in Git. Leave exact resume point if incomplete.

Implementation choices to verify: use Gmsh per sampled geometry before GPU,
physical triangle-centroid areas and existing regional kernel/visibility;
pad bank to common N/K, never treat padding as physical area. Boundary point
condition exposes outer/hole curves; same spatial coordinates across all t for
one geometry but different across geometries. Preserve parameter gradients
through learned encoder/preparation, never cache learned activations across
updates. Hard initial field/weighted mass, chemical elimination, PDE-only loss.
Fixed IC is genuinely geometry-independent ambient function, so local initial
normal derivative may mismatch moving holes; no hidden hole-dependent envelope.

Next: inspect reusable FEM/native/encoder APIs, implement explicit short-case
config and geometry/reference bank, tests, training pilot and diagnostics.

Implementation progress:
- PointFieldOperator added separately from image model: boundary point lifting,
  arc-measure-weighted latent cross-attention, latent self-attention and shared
  spatial preparation/time FiLM. Existing image model/checkpoints unchanged.
- Canonical train.py opt-in config dispatches geometry operator. Same fixed
  ambient two-wave cosine field (amplitude.25), epsilon.06,T.2,M.02;16train/
 8heldout geometries, Adam1e-3/5000/B4, ordinaryFP64;32latents,width48.
- Own Gmsh triangle-centroid area and visible regional graph per geometry,
  delta.11,h.045; 769..871nodes. Pad N/K with zero volumes/coefficients and
  mask physical output. Inverse biharmonic metric reuses existing definition.
- Independent SciPy assembly/BDF bank completed24cases in93.16s. Max phi
  time differences~3e-8,mu~1.5e-7, mass~5e-17, sampledenergy nonincreasing.
  Frozen-IC mean pooledphi12.397%, so evolution is measurable.
- Heldout16/17 independent mixed-P2 FEM local checks passed. Max phase mesh
  differences.1342/.1674%, time.00253/.00205%; native/local pooled3.099/3.229%,
  terminal3.786/4.037%. Geometry bias remains distinct from training error.
  Same ambient IC has incompatible classical wall derivatives at t0; local
  weak phase evolution valid, no classical initial-mu validation claim.
- Logs currently /tmp/confined-geometry-{prepare,local,tests}.txt; CPU focused
  tests running. No GPU training yet; no concurrent FEM/GPU job.
Next: finish tests, cold train through canonical entry, heldout/swap/local checks,
scientific plots and documentation/artifact registration. No blocker.

First cold run complete: 20261029-confined-geometry-operator-adam-01.
5000Adam/B4 FP64,259.34s, firstupdate70.09s, warm16.78ms; activepeak.527GiB.
Heldoutnativephi3.272/4.326%, mu8.228/10.287% pooled/terminal.11tests pass.
Critical diagnosis: wrong-geometry encoder swaps leave errors almost unchanged;
geometry information is not yet used adequately. Do not stop at scalar L2.
Next: direct boundary-token weighted self/cross-attention control, same bank/
physical scheme/budget; first run preserved. Independent evaluation in progress.

Selected mild direct-token5000run completes305.20s; warm21.08ms; native heldout
phi2.855/3.709%,mu7.599/9.343%; localFEM2meanphi4.526/5.733%. FrozenIC12.416%.
Wronggeom3.425%,7/8worse; activepeak.527GiB,pool1.064GiB. Fresh24field replay
maxgap0,independent metrics/all20kactual geometry draws match. Default setdirect.

Latest user steering: fields look almost identical across geometry; user allows
harder coarsening if simple case is weak. Do NOT close task yet. Build common-
query reference/prediction sensitivity diagnosis and longerT.75control, keeping
same initial/geometry/operator/epsilon/5000budget. Separate geometry contrast
from common mean-field accuracy. Existing8heldout inputs have been reused for
architecture development, not a fresh final test. Next: prepare longer references,
quantify physical response; train only after preflight shows stronger evolution.

Current mild reference common-point diagnosis: 120x96candidate probes, restricted
inside every heldout domain, centroid interpolation stencils crossing holes or
enclosing holes removed; interpolate evolution increments then add identical
analytic IC to remove initial interpolation variance. Terminal native geometry
spread is2.595% of mean-field norm, RMSstd.00587,maxstd.03973; maxphase.4444.
Thus visually similar fields largely reflect a mild physical case, but model
contrast fidelity still to measure. Removed interpolation initialstd.001293;
this diagnostic is not an exact shared-grid FEM solution. Longer-bank first
11geometries show frozenIC errors~35%, substantially stronger evolution.
New off-grid query helper retains original physical-grid mass projection;
focused query/IC/mass/time derivative test running. README CR escape corrected.

Current authoritative resume point (supersedes earlier next-action paragraphs):
- Mild direct-token shared-query geometry-response error84.25%; whole-field1.967%
  on7556common probes. Reference terminal RMSstd.0058704 vs prediction.0028728.
  Thus both mild physics and suppressed learned geometry dependence contribute.
- LongerT.75native bank complete, otherwise identical24geometries/IC/physics.
  Reference terminal RMSstd.0146921; geometry relative spread4.369%, maxphi.6389.
  Both independent local FEM checks passed: meshphi.1600/.2291%, time<.00034%;
  native/local pooled4.7777/4.8433%, terminal5.7621/5.7109%. FEM processes exited.
- User rejects explicit distance/locality bias; interrupted patch never landed.
  Primary GNOT/PerceiverIO/Transolver methods rechecked. Test a generic separate
  boundary/domain-IC point-stream model with repeated normalized attention;
  no hand-set query-to-wall distances. GNOT-inspired, not full paper reproduction.
- Next: implement/check weighted linear attention and padded-query/gradient
  invariants, then serial matched5000cold starts on longer bank. Evaluate whole
  field AND geometry contrast; no evolved labels, no test checkpoint selection.
  Eight held-out geometries remain reused development inputs, not final test.

General architecture implementation / current verification:
- `HeterogeneousPointOperator` is a separate pure Equinox class in the point
  module. Explicit input streams(5boundary,3domain-initial), shared coordinate
  Fourier metadata, four heterogeneous integral-attention layers, width64/heads4,
  and the existing three-layer temporal FiLM. GNOT-inspired equations3-8 with
  physical measures; no query self-attention/MoE or explicit pair-distance bias.
- Linear associative moment formula and its Q/K/V gradients agree with an
  independently materialized pairwise formula. Permutation, measure splitting,
  padding, exact query chunking, off-grid/full-grid agreement, hard IC/mass,
  time JVP/finite differences and PDE gradients reaching BOTH streams pass.
  Old latent-model PDE gradient/canonical dispatch also passes. Three tests
  complete144.997s; raw `/tmp/confined-geometry-heterogeneous-tests.txt`.
- Longer boundary-only5000cold start completed, native heldoutphi5.2933/6.2201%,
  local2mean6.8858/7.9069%. Shared-query response error84.4187%, amplitude43.5375%
  of reference, correlation.5477. Whole common-probe field4.0068%, meanfield2.6187%.
  Run `/root/autodl-tmp/tf-logs/confined_ch2d/20261029-confined-geometry-longer-direct-01`.
- Heterogeneous5000cold start currently running in separate serial GPU process,
  same longer bank/seed/sampler/optimizer/budget; first1000nativeheldoutphi5.0671%.
  Do not rank incomplete/intermediate checkpoints. Exact resume: finish this
  run, then saved-field/native/local and geometry-response diagnostics, paired
  CPU replay/provenance audit, lightweight artifact registration and task-end.

## Final result and verification

User-preferred general conditioning route is implemented and tested. No explicit
distance-bias patch landed. `train_operator_heterogeneous.py` selects separate
boundary/domain-IC point streams, repeated GNOT-inspired normalized integral
attention and the existing shared spatial/time decoder. It remains a research
preset; the mild operator and established windowed single-IVP defaults remain.

Same longerT.75bank, Adam1e-3/5000/B4, ordinaryFP64, same seed and sampler:

| Metric | Boundary-only | Heterogeneous |
|---|---:|---:|
| Native heldoutphi pooled/terminal |5.2933/6.2201% |2.3805/2.7710% |
| Native heldoutmu pooled/terminal |10.7630/13.0767% |5.2039/6.4580% |
| Local2meanphi pooled/terminal |6.8858/7.9069% |5.7202/6.7037% |
| Local positive mu |14.2818% |9.8073% |
| Common-query response relative error |84.4187% |42.6478% |
| Response amplitude/reference |43.5375% |94.7414% |
| Response correlation |.547688 |.905470 |
| Common-query whole-field error |4.00685% |2.12325% |
| Common-query mean-field error |2.61872% |1.47017% |
| Cold total trainer |314.313s |406.647s |
| Warm update after100 |21.458ms |36.291ms |
| Peak active bytes |438346496 |555710720 |

Terminal reference RMSstd.0146921, prediction.0067562/.0141158. Reference and
model conditional response is evaluated on7556common physical probes using
visible-triangle interpolation of native evolution increments; exact common
ambient IC is added to remove initial interpolation variance. Near-wall excluded
support and interpolation limit this diagnostic; it is not common-grid FEM.
Response L2 uses the centered response as denominator, not the full phase norm.
Eight training-unseen geometries have been reused for architecture development.
No fresh final test, geometry-count/topology transfer, mesh transfer or seed-
stability claim. Architecture changes streams/position/depth/width together;
neither a single-component cause nor generic convergence has been established.

Final checkpoints use fixed5000budget, without heldout checkpoint selection.
The4000heterogeneous monitor is slightly better in scalarphi but is not selected.
Native frozen-IC heldoutmean35.6687%; actual evolved-label fields never enter
the step payload. Hard initial/mass hold; maximum predicted mass drift4.17e-17.
Energy is nonincreasing at the five stored times, not a continuous-time proof.
Local/native reference differences~4.8/5.7% are separate operator-discretization
errors. Better native fitting does not establish subpercent continuum accuracy.
Both longer jobs used JAX's default35.56GiB allocator pool, distinct from active
peaks. Cold times include compilation/setup/evaluation; single observations are
not a statistical speed benchmark. All CPU/GPU/FEM jobs exited.

Direct artifacts:
- Mild native bank: `/root/autodl-tmp/data/confined_ch2d/operator_geometry_v1`.
- Longer bank: `/root/autodl-tmp/data/confined_ch2d/operator_geometry_longer_v1`.
  Each has manifest/config,24references and independently refined FEM16/17.
- Run root: `/root/autodl-tmp/tf-logs/confined_ch2d`.
  Mild runs `20261029-confined-geometry-operator-adam-01` (latent32) and
  `20261029-confined-geometry-operator-direct-01` (boundary tokens).
  Longer runs `20261029-confined-geometry-longer-direct-01` and
  `20261029-confined-geometry-longer-heterogeneous-01`.
  All retain source/config/initial/final checkpoints and raw loss/metric logs.
  Direct/heterogeneous runs have `evaluation/` fields/curves/metrics;
  longer runs have `geometry_effect/` common-coordinate diagnostics.
- [Original selected-mild replay](confined-geometry-audit.py),
  [receipt](confined-geometry-audit.json).
- [Paired longer CPU replay/provenance audit](confined-geometry-general-audit.py),
  [receipt](confined-geometry-general-audit.json). All24savedphi/mu arrays per run
  reproduce within1.36e-15/1.89e-15; all nonarchitecture config fields agree;
  actual20,000geometry draws identical/all16visited/noheldout. Actual time arrays
  were not saved; time parity scope is seed/implementation, not array evidence.
- [Focused test logs](confined-geometry-validation.txt): initial11tests pass,
  direct-token1pass, off-grid/time1pass; new weighted-linear forward/gradient
  and complete heterogeneous invariants plus old CLI/PDE-gradient3pass.
  One prior invalid-config fixture first failed in ConfigDict before validation;
  fixed test uses ignore_type, rerun passes. This is preserved as a harness
  failure, not hidden as a numerical solver failure.

Primary architecture references: [GNOT sections3.3-3.4](https://proceedings.mlr.press/v202/hao23c/hao23c.pdf),
[Perceiver IO read/process/write](https://arxiv.org/html/2107.14795v3),
[Transolver learned slices](https://arxiv.org/html/2402.02366v2).
Our model omits GNOT query self-attention and MoE for independent query chunks;
paper data-trained successes do not prove physics-only CH convergence.

Diff-level delivery: new point operators/configs, fixed ambient IC utilities,
native/FEM geometry-bank preparation, canonical train dispatch, PDE-only trainer,
saved-field and geometry-sensitivity evaluators, focused tests and regeneration
audits. Root/example README, DEVELOPMENT, artifact index, PROGRESS and DECISIONS
updated. Generated PNG/NPZ/checkpoints stay outside Git. No hashes introduced,
no legacy artifact paths moved, no commit/merge/push/history operation.
Future optional work: a fresh geometry cohort, seed checks, or native/local
operator refinement; none is queued automatically. Final syntax/path/index/diff
checks recorded before task-end.

Task-end checks passed:15changed/untracked Python ASTs, strict JSON parsing,
Markdown carriage-return check, all example/task local links,8case/425directory
artifact inventory and `git diff --check`. Existing1529indexed files total
18.77MiB;22pending source/record files contain no generated media/checkpoints or
file>1MiB. Source tests/integration runs above provide behavioral verification;
no unchanged expensive suite rerun. PROGRESS now has no active task and names
this archived result; DECISIONS records the selected general research control.
