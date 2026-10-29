# Matched SOAP control for the 3D melting operator

Date: 2026-10-29

## Goal and acceptance

User requests trying SOAP. Cold-start the existing online 3D random-ellipsoid
operator for 5000 updates with SOAP instead of Adam. Preserve architecture,
physics, hard IC/latent residual, lr1e-3 schedule, B16/core+halo/time sampling,
seed and held-out native graph references. Compare physical phase/solid L2,
terminal solid fraction, training/cold wall time and memory. Verify actual
initial parameters and recorded input streams; do not select checkpoints on
test scores. Retain Adam defaults pending evidence and no Git history actions.

## Context and artifacts

Adam baseline: /root/autodl-tmp/tf-logs/icemelting3d/20261029-melting-operator-online-b16-01.
SOAP run: /root/autodl-tmp/tf-logs/icemelting3d/20261029-melting-operator-online-b16-soap-01.
Reference: /root/autodl-tmp/data/icemelting3d/operator_ellipsoids_v1.
Uses existing upstream SOAP_JAX factory; b1=.9,b2=.99,refresh5,
max_precond_dim100000,precondition_1d=false,clip100, ordinary FP32/highest.
No new mathematical derivation or source modification is required.

## Outcome

COMPLETE. Canonical SOAP process exits0 after5000updates/80000online shapes,
492.063951s full cold process,448.456120s trainer including sampling/monitoring/
first compilation. First synchronized update247.116638s; subsequent updates
33.655437ms mean, sampling3.064558ms. Adam corresponding values300.574071s,
257.167458s,108.238333s,22.428261ms,3.411398ms. SOAP peakactive1.589519GiB,
Adam1.331575GiB, both4GiB JAXpools (driver/context usage separate).
No competing GPU/FEM job; identical environment, soap-jax0.2.2/JAX0.10.1.
Compiler emits two slow GEMM fusion compilation alarms but training completes
normally. No optimizer implementation or compilation-policy intervention.

Eight predeclared held-out shapes, final5000checkpoint, mean errors:

|Metric|Adam|SOAP|
|---|---:|---:|
|Pooled phi relative L2|0.226648%|0.078517%|
|Terminal phi relative L2|0.092213%|0.129073%|
|Pooled solid relative L2|0.642494%|0.223470%|
|Terminal solid relative L2|4.501659%|6.503614%|
|Terminal solid-fraction relative error|1.213330%|4.015645%|
|Interface-band phi absolute RMSE|0.0108761|0.00377592|
|Pooled central-slice phi relative L2|0.420117%|0.223106%|
|Terminal central-slice phi relative L2|0.286081%|0.412674%|

Solid=(phi+1)/2; interface reference band abs(phi)<.9. Maximum terminal solid
L2 worsens10.5936% ->15.3061%; maximum terminal solid-fraction error3.3453%
->8.4404%. Whole-field phi is background-dominated near disappearance.

|Dimensionless time|Adam mean phi L2|SOAP mean phi L2|
|---|---:|---:|
|0|0.000028%|0.000028%|
|.25|0.267508%|0.054688%|
|.50|0.331351%|0.063874%|
|.75|0.254790%|0.073738%|
|1|0.092213%|0.129073%|

SOAP improves intermediate fields and pooled error in this pair, while late
small-solid accuracy degrades and compute costs rise. Keep Adam operator
default, retain SOAP opt-in. This is not a universal optimizer ranking or
seed-stability investigation; existing deferred SOAP issue is not reopened.
No reference/test checkpoint selection or training-length tuning performed.

## Verification

- Saved configs differ only in train.optimizer and log.run_name.
- All79initial checkpoint arrays,24source snapshots/current files and graph
  JSON are byte-identical; initial monitor metrics also exactly equal.
- Actual recorded5000x16x10shape parameters and5000x8time arrays are identical;
  80000unique online ellipsoids. Same deterministic sub_sample(i,K) pool
  constructor, graph, arguments and cyclic order. Neither run saved raw
  subgraph arrays, so direct array replay of historical pool is unavailable.
- Independent FP64 NumPy saved-field metrics, per-time metrics and eight-input
  aggregates agree with training summary. Reference parameters/coordinates/
  fields/slice/time arrays exactly equal saved independent dataset; physical
  ranges, finite arrays and hardIC gate1e-5 pass.
- Fresh GPU loading of saved config/final checkpoint replays predeclared cases
  0and2 with maximum gap9.536743e-7 (gate2e-5); all final parameters finite.
- Convergence plot visually inspected: two fixed monitor inputs, separate from
  final8test metrics. Native128^3nonlocal references,24^3model diagnostic
  nodes and128^2slices, no full-grid PINN/local-FEM/continuum claim.

No numerical source/config defaults changed, so no unchanged numerical suite
rerun. The first artifact whitespace check caught compiler-alarm trailing
spaces in stdout. Preserve exact bytes at the external run/training_stdout.txt;
normalize trailing spaces only in the repository display copy and assert every
non-whitespace byte is preserved. Repeat artifact/JSON/index/diff checks after
that correction. Plot/script/README/memory checks complete at task-end.

## Artifacts and resume

All artifacts below are alongside this archived task:

- melting-operator-soap-coldstart.py / process.json / training.txt:
  canonical command, environment, cold wall measurement and whitespace-cleaned
  stdout display; exact original stdout is retained in run/training_stdout.txt.
- melting-operator-soap-comparison.py / .json / .txt:
  source/initial/input parity, independent metrics and performance comparison.
- melting-operator-soap-convergence.png: two monitor inputs, not final test set.
- melting-operator-soap-replay.py / .json / .txt: fresh GPU checkpoint replay.
- 20261029-melting-operator-online-b16-soap-01-{config,graph,loss,summary,timing}.json
  and -metrics.jsonl: exact direct copies of external run records.

README and artifacts/index.yaml register direct SOAP paths, mixed accuracy
outcome, opt-in flag and unchanged Adam default. Memory changed: this task,
docs/PROGRESS.md, docs/DECISIONS.md. No blocker/next action/queued training;
no Git history actions. GPU idle after replay.
