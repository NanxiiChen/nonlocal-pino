# Canonical ordinary-policy defaults — 2026-10-26

## Goal and acceptance

User requests confirmation and alignment of best explored defaults, especially
ordinary compilation, for corrosion2d, electropolish2d and icemelting3d. Complete
missing full controls before promotion, preserve physical PDE/BC/reference/data/
residuals/scales/model/optimizer/horizons, and distinguish speed/field accuracy
from stability/transport. Seed is runtime provenance, not a strategy parameter.
No Git history operation authorized or performed. Existing artifact paths retained.

## Completed selection

All canonical `train.py` entries still load `configs/train_base.py`. They now
explicitly select ordinary GPU policy. Corrosion uses FP64; electropolish and
melting FP32. All use SOAP, initial lr1e-3 and5000updates. Corrosion50s,
electropolish/melting5s. Original residual/data strategies remain unchanged:
corrosion phase/rawKKS/s_bc/nodal; electro phase/physical/random;
melting IC-only latent/exact initial condition.

Selection uses observed final and late field accuracy, physical topology and
throughput across predeclared42/2026 controls. It is not a global optimum,
statistical robustness proof, seed selection or uniform stability claim.
Historical individual field winners remain distinct from representative current
strategy checkpoints. Melting's new full replay becomes its selected5s result;
no existing directories are renamed, overwritten or resumed.

## Full training and independent results

Six new serial GPU cold starts each complete5000 updates via canonical train.py;
all exit0. Existing paired controls are reused. Independent CPU final checkpoint
reloads (10KKS models,1melting model) confirm:

| Case | Policy / precision | Run seed | Terminal phi L2 | Terminal c L2 | Late4000–5000 phi median |
|---|---|---:|---:|---:|---:|
| corrosion | previous deterministic FP32 | 42 | 0.00883538 | 0.01042560 | 0.00719096 |
| corrosion | previous deterministic FP32 | 2026 | 0.01074417 | 0.01272650 | 0.00951408 |
| corrosion | selected ordinary FP64 | 42,existing control | 0.00249850 | 0.00294446 | 0.00468453 |
| corrosion | selected ordinary FP64 | 2026,new | 0.00683255 | 0.00807736 | 0.00500373 |
| electro | previous deterministic FP32 | 42 | 0.29092776 | 0.29901269 | 0.29196189 |
| electro | previous deterministic FP32 | 2026 | 0.00283170 | 0.00342673 | 0.00463667 |
| electro | selected ordinary FP32 | 42,new | 0.00475716 | 0.00573811 | 0.00442108 |
| electro | selected ordinary FP32 | 2026,new | 0.00339923 | 0.00411778 | 0.00452465 |
| electro | rejected ordinary FP64 | 42,new | 0.29975673 | 0.30898136 | 0.29900886 |
| electro | rejected ordinary FP64 | 2026,new | 0.00355082 | 0.00428119 | 0.00601905 |

Late median uses all six200-update evaluations, including final5000; no
intermediate minimum is selected. Corrosion selected models retain one pit at
all21 reference times. Both selected electro runs retain exactly one vertical
interface per column at all21 times. FP64 electro42 retains spurious interfaces;
more precision does not repair its optimization trajectory.

Corrosion final free-node q/Dq L2 at42/2026 is0.007739/1.356993 and
0.010077/1.377873. Large Dq error is unresolved.
Seed42 late phi maximum0.014885 exceeds previous0.008835, with a Dq spike18.665;
new2026 late phi maximum0.014465 versus previous0.016322. Worst-seed late median
improves, but uniform stability and transport convergence are not established.
Electro selected free-node Dq L2 is0.059229/0.058106; previous successful2026
was0.054906. The new policy rescues42 in this realization; it does not improve
every successful old endpoint or guarantee every ordinary run succeeds.

Melting new ordinary FP32 seed42 gives **GPU full-grid terminal graph phi
L2=3.9052947454e-5**, independently CPU3.9058160588e-5 on2097152 nodes.
Solid-fraction L2 is3.54454768e-4; predicted/reference solid-volume node fractions
both0.0040779114. Sampled8000-point late graph phi median3.48832457e-5,
range2.89452009e-5–6.54558223e-5. Full-grid late monitoring has4000/5000 only.
The recorded earlier2026 result is **5.98573384e-5**, with external policy not established by that run's metadata.
Earlier42 full-grid6.10970108e-5 remains historical. Analytic-profile errors are
separate from this graph comparison; whole-cube and solid-fraction L2 differ.

## Speed scope

Prior synchronized steady-update benchmarks include sampling, SOAP refreshes
and GradNorm, excluding initial compilation, evaluation, logging and saving:

| Selected arithmetic/policy | Selected ms/update | Previous deterministic FP32 ms/update |
|---|---:|---:|
| corrosion ordinary FP64 | 35.13 | 229.31 |
| electro ordinary FP32 | 10.98 | 197.73 |
| melting ordinary FP32 | 26.62 | 468.26 |

Sources: [corrosion benchmark](local-ad-performance.md),
[other-case benchmark](phasefield-gpu-policy-audit.md).
New six-job end-to-end observed seconds are180.70,70.50,317.42,223.86,201.10,
218.91 in [manifest](ordinary-policy-defaults-manifest.json) order. They include
actual JIT/evaluation/save/process costs with reused, different compiler-cache
histories; these are not fresh-cache compilation benchmarks. Raw metric time
curves retain those histories and should not be interpreted as isolated timing
ratios. No full timing/default claims for elasticity or corrosion1d.

## Implementation and direct provenance

- Shared `src/nonlocal_pinn/training/runtime.py:configure_gpu_policy` runs before
  backend initialization. It preserves unrelated XLA flags, validates policy and
  rejects inherited deterministic flags under ordinary mode, including a bare
  Boolean flag. Explicit deterministic policy is retained. False/conflicting
  deterministic settings fail loudly. No JAX import occurs in the helper.
- Melting base adds explicit ordinary policy; entry calls shared helper and
  summary records policy and actual XLA_FLAGS. This makes the common previous
  unset-environment behavior explicit, without claiming all historical shells
  had ordinary policy.
- KKS canonical presets override the historical research strategy's precision/
  policy explicitly. `train_graph_ordinary.py` preserves FP32; local tensor/joint
  presets preserve FP32 deterministic. Existing controls therefore retain their
  previous meaning despite canonical promotion.
- Seed remains absent from strategy configs; runtime --seed and saved resolved
  metadata remain unchanged. Historical loading uses original architecture/
  physical template and parameter dtype; FP64 models need x64 and a promoted
  template. Historical in-place resume compatibility is not asserted.
- README root/three examples, DEVELOPMENT and artifacts/index.yaml now identify
  current precision/policy, direct selected paths and limitations. Index retains
  old default records and historical individual winners separately.

Representative current-default checkpoints (untuned seed42):

- `/root/autodl-tmp/tf-logs/corrosion2d/20261026-full-graph_ordinary-float64-seed42/ckpt_005000.eqx`
- `/root/autodl-tmp/tf-logs/electropolish2d/20261026-defaultcheck-ordinary-float32-seed42/ckpt_005000.eqx`
- `/root/autodl-tmp/tf-logs/icemelting3d/20261026-defaultcheck-ordinary-float32-seed42/ckpt_005000.eqx`

New runs all use the same `20261026-defaultcheck-ordinary-<dtype>-seed<seed>`
name under their existing case tf-logs root. Exact six commands/paths/statuses
are in the manifest. Numerical references remain unchanged at
`/root/autodl-tmp/data/<corrosion2d|electropolish2d>/results_matched_graph_20261024.npz`
and `/root/nonlocal-pinn/reports/icemelting-weak-prior-20261024/reference.npz`.

Raw logs/NPZ/audited fields/final source snapshot are at
`/root/nonlocal-pinn/reports/ordinary-policy-defaults-20261026`.
Each full run retains source/config/initial/final/checkpoints/metrics. The
verification locates100 run-snapshot Python files and nine final-default source
snapshots by direct path. Final helper additionally covers bare Boolean flag
parsing; empty-XLA full-run numerical behavior is unchanged. No hashing.

## Validation and evidence

- Six new5000-update full GPU runs, serial, all exit0.
- Ten KKS independent CPU physical/raw-q/time-derivative/transport/topology
  reloads plus one full-grid melting CPU reload, final==checkpoint5000 bytes,
 26evaluation rows per run, finite fields and physical reference identities.
- Paired initial parameter values agree exactly within seed across dtype/policy;
  generated common FP32 inputs are promoted, stored graph is not rebuilt in64.
  True parameter/data/gradient/SOAP dtypes checked; scalar LR schedule stays FP32.
- Current-vs-selected config parity allows run name/checkpoint cadence and
  optional melting evaluation reference only; numerical defaults match.
  Historical absent continuation fields are accepted only at checked disabled
  values; tuple/list differences are normalized through serialized JSON.
- `JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 XLA_FLAGS='' .venv/bin/python -m unittest tests.test_sota_training_entry tests.test_corrosion_precision tests.test_corrosion_local_ad -v`: **11 pass**.
  After bare-flag parsing was strengthened, the affected entry suite alone:
  **7 pass**. No unchanged expensive suite rerun.
- Two actual canonical GPU2-update starts without precision/policy overrides
  confirm new defaults, final optimizer/data/parameter dtypes and outputs;
  [smoke results](ordinary-policy-defaults-canonical-smokes.json).
- [KKS audit](ordinary-policy-defaults-kks-audit.json),
  [melting audit](ordinary-policy-defaults-melting-audit.json),
  [configuration/source verification](ordinary-policy-defaults-verification.json).
- [Corrosion convergence](ordinary-policy-defaults-corrosion2d-convergence.png),
  [electro convergence](ordinary-policy-defaults-electropolish2d-convergence.png),
  [melting convergence](ordinary-policy-defaults-icemelting3d-convergence.png).
  [Corrosion representative phi](ordinary-policy-defaults-corrosion2d-phi-fields.png),
  [electro representative phi](ordinary-policy-defaults-electropolish2d-phi-fields.png)
  use final5000 models and times0/.25/.50/.75/1, not best intermediate models.
- Launch/audit/smoke/plot/verify scripts are archived alongside this record.

## Status, blocker and next action

Completed. No worker, blocker, pending approval or required training remains.
Final strict-JSON, link, source/default parity, registered-artifact and diff
checks recorded in `ordinary-policy-defaults-validation.json`. No Git history
operation performed. Future work may target corrosion transport/late spikes or
broader statistical replication; neither is required for this default alignment.
