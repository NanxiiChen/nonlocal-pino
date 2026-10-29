# Richer confined CH PINN training

Goal: train the validated ten-wave confined CH case through terminal time1.5,
as requested after independent reference delivery. Complete matched nonlocal
and local AD cold starts with the existing canonical train.py/preset, monitor
relative L2 and show fields/convergence/timing without changing physics.

Acceptance: both SOAP5000+L-BFGS500 runs finish or a concrete failure is diagnosed;
shared FEM reference for primary pooled/terminal phase errors, separate native
target metrics; best/final checkpoints distinguished; saved-config CPU replay,
initial/source/config parity and physical diagnostics; README/index updated.
No accuracy threshold, extra tuning, seed selection or Git operation requested.

Status: experiment complete2026-10-27; both runs and strict CPU artifact audits
finish, but phase learning fails at the declared budget. Blocker: none. Next
action: none in this task. Any future strategy optimization should start from
these failed controls; numerical accuracy remains an unresolved research issue.

Context: examples/confined_ch2d/configs/train_coarsening.py, epsilon.02,T1.5,
ten deterministic physical IC waves, direct phase output, original residuals,
weighted hard IC/mass, geometry mesh.02/3682 centroids, delta.03. FP64 ordinary,
initial LR.001, SOAP5000 and L-BFGS500. Refined reference at
/root/autodl-tmp/data/confined_ch2d/coarsening_multimode10/reference.npz passes
phase mesh/time.552%/.048%, mu3.014%/.180%, mass/energy checks. Native/local
phase gap3.779% predates training. First .02/.014 reference failed and remains
preserved. Two multimode checks/legacy replay already pass; do not repeat the
unchanged expensive suite. Prior task: [reference delivery](confined-ch-coarsening.md).

Predeclared runs:

- /root/autodl-tmp/tf-logs/confined_ch2d/20261027-coarsening10-nonlocal-01
- /root/autodl-tmp/tf-logs/confined_ch2d/20261027-coarsening10-local-ad-01

Use runtime --seed=42 provenance, not a strategy-config seed. No FEniCS process
active; never launch FEM alongside GPU training. No original artifact or default
is replaced. Natural native no-exchange is not pointwise local Neumann; local
AD boundary normals remain a separate diagnostic. Deferred SOAP work stays out.

Nonlocal session62086 completed SOAP5000+LBFGS500 in174.65s, mean compiled SOAP
19.55ms. Final common pooled/terminal phase82.96%/94.55%; L-BFGS500 reaches
budget normally (status1), loss.02408. Best shared-FEM phase checkpoint is
step0, pooled72.31%/terminal79.52%, so it is untrained initialization, not a
successful trained result. This harder case is not yet learned well. Local AD
session9645/log /tmp/confined-ch-coarsening10-local-ad-training.log launched
after nonlocal exit, with unchanged preset/method-only override. NL saved-config
CPU replay ran separately (CPU only, no FEM). Source/defaults were not tuned
during this matched pair. No Git work.

Early audit: both saved configs match except method/run name, initialized
checkpoint bytes identical,19 source files identical. NL CPU final replay
completed (session55017 exit0), report and phi_fields/convergence saved in its
run. Final own-native phase83.02%/94.63%, also poor: local/native reference gap
alone cannot explain the PINN error. Physical plot shows initial small regions
persist instead of the reference mergers; avoid claiming uniform-field collapse.
Local AD by step3750 remained~80% pooled. Archive analysis script
confined-ch-coarsening10-analyze.py was then run after local exit with
JAX_PLATFORMS=cpu .venv/bin/python, logging to a separate archive record. It
explicitly reused the already-validated NL final replay, evaluated local final,
replayed both best checkpoints, verified parity and reported17-time residual
allocation/energy/mass/normals/IC persistence. No source or strategy change.

Local AD session9645 exits0, all5500 updates in696.63s, mean SOAP93.48ms.
Final common pooled/terminal phase82.950%/99.409%; best checkpoint again step0.
NL corresponding82.960%/94.545%. Both L-BFGS runs reach500 accepted iterations;
do not label requested iteration-limit status as a crash or true convergence.
The first CPU archive analysis session87248 failed in audit initialization as
described below; corrected session34900 completed. The comparison remains poor even
though nonlocal trainer is3.99x faster; no accuracy/default promotion.

First archive audit attempt failed in best-checkpoint replay because reusing
the final NL report bypassed evaluate.py's global x64 setup. Warnings expose
FP64-to-FP32 truncation; max difference5.24e-6. This is audit initialization,
not a changed training checkpoint. Preserve failed log as
confined-ch-coarsening10-analysis-precision-init-failed.txt; explicitly enable
JAX x64 at audit main entry and rerun unchanged strict1e-8 comparisons. Do not
loosen tolerances or alter saved model/fields. Standalone final NL replay had
already passed to1.53e-14.

Final audit session34900 exits0 with x64 explicit; final NL/local CPU phase
max differences1.53e-14/8.77e-15, best replays7.20e-15 (both untrained step0).
Initial/source/config parity rechecked, strict1e-8 tolerances unchanged. The
archive status passed refers to replay/parity checks, not PDE prediction
accuracy. Descriptive scope was clarified after execution; numerical results
are unchanged. Raw failed precision-initialization log is retained.

| Final-budget common FEM metric | Nonlocal | Local AD |
|---|---:|---:|
| Pooled phase relative L2 | 82.9597% | 82.9502% |
| Terminal phase relative L2 | 94.5455% | 99.4092% |
| Positive-time mu L2 | 119.0950% | 123.2522% |
| Trainer wall seconds | 174.6488 | 696.6260 |
| Mean compiled SOAP ms | 19.5549 | 93.4811 |
| Best shared-FEM checkpoint step | 0 | 0 |

Both trained final models are worse in common pooled phase L2 than their
untrained initialization72.31% (terminal79.52%). L-BFGS500 budget exit is not
optimization convergence. Nonlocal is3.99x faster on this serial pair, but
neither model is accurate; no precision ordering/stability/SOTA claim.

Readonly diagnosis:
- Final pure-phase areas11.264%/8.379% vs FEM46.138%. Incorrect mergers and
  insufficient phase contrast persist. Fields are not frozen at the IC: their
  weighted relative changes from IC are.857/.827, vs reference.945. Correlations
  to the IC are.589/.593 vs reference.638, so visual initial-region persistence
  is not evidence for a uniquely frozen-IC cause.
- Fixed17-time objective = t0 plus16 positive midpoints with original weights.
  Initial chemical loss accounts52.657%/39.459% of total, all initial parts
  53.626%/40.238%. Positive-time mu errors>119% show fitting failure beyond t0.
  Capacity, global hard-IC lifting, time sampling and optimization are possible
  directions, not isolated mechanisms. No extra control was run here.
- Weighted mass drift3.47e-16/2.78e-16 follows the hard constraint. Sampled
  native energy increases7.30e-4/3.38e-4 are genuine model violations; final
  energies.11683/.13041 vs own reference.06927/.06994. Do not report all physical
  checks as passing just because artifact replay passes.
- Local AD terminal phi normal RMS.0476-.0939 vs native-NL12.28-21.96;
  classical pointwise Neumann is still not automatic for the nonlocal MLP.
  Reference gap3.779% and own-native phase83.02% rule out finite-horizon target
  bias as a sufficient explanation for this large fit error.

Artifacts: confined-ch-coarsening10-analysis.json/txt and reproducible
confined-ch-coarsening10-analyze.py; raw per-method configs, summaries, best
metrics, loss/validation histories, refinement reports, replay JSONs and fields.
Compact L2 curve uses the same raw histories (percent axes, no smoothing),
plus detailed comparison/time/energy/normals plots. Common-field comparison
shows t0,.15,.45,.9,1.5; best plots explicitly label step0. Failed audit log,
original reference and both training stdout logs preserved. Direct paths only,
no hashes or overwritten old results. README/index register failed outcomes.

Diff-level changes: new archive analysis/plot script and outputs; confined_ch2d
README adds matched failed-run results and caveats; artifact index registers two
new runs/comparison. No training implementation, strategy config, physical data,
operator, reference or optimizer dependency changed. Original default retained.

Task-end verification: both full5500-update runs, saved-source/initial/config
parity, two final and two best CPU replays and hard weighted-mass checks pass.
Residual/energy/wall diagnostics executed and physical failures recorded above.
Plot inspection, strict JSON/metadata/histories, source syntax, changed
documentation links, artifact inventory and git diff --check pass.
Unchanged expensive numerical suite not rerun. GPU/FEM/CPU workers idle; no Git
history operation. Memory updated: PROGRESS.md, DECISIONS.md and this record.
