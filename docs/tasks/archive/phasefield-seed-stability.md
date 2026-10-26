# Three-case independent-seed validation —2026-10-26

## Goal and outcome

Complete. The user rejects seed as a strategy parameter and requests one
additional seed per current strategy. Predeclared seed2026 for all3 cases,
not present in saved log configs. All3 cold starts complete5000 updates,
return0 and use their final5000 checkpoints, without seed rerolls, intermediate
winner selection, early stopping, extra steps or strategy retuning.

Corrosion fails this held-out run; electropolish succeeds in this run but its
historical seed42 failure remains; melting agrees with its earlier seed42 run.
The previous best-observed defaults are not evidence of cross-seed stability.
No physical equation/reference/operator/BC/pre-scale change is made.

## Fixed strategies and stochastic scope

- Corrosion50 s: bounded eta0.1,raw-KKS edge CH/L1,s_bc,nodal IC/BC.
- Electropolish5 s: original residual,physical data losses,random IC/BC.
- Melting5 s: IC-only latent,SparseD128^3,Kcore2000,Nt64.

All SOAP,initial lr1e-3,5000 updates; all numerical/model/sampling fields match
the previous strategies. KKS changes initialization and temporal draws via
train.seed. Melting sparse temporal keys are fixed by iteration, so the change
mainly probes initialization. Geometry, data samplers, subgraph pools and
fixed evaluation streams remain prescribed. This is not an every-RNG or
statistical uncertainty study. GPU reports RTX3090/49152 MiB,CUDA13; at most3
workers, no preallocation, deterministic XLA flag, no FEniCS/MPI concurrently.
Concurrent wall times are not compared with previous hardware as speed results.

## Terminal physical-field relative L2

KKS errors use all5151 spatial nodes against the same-graph SciPy reference;
melting uses all2097152 cube nodes against the same-graph numerical reference.
These are final physical times50/5/5 s, not pooled errors or minimum history.

| Case | seed42 | seed7 | New predeclared seed2026 |
|---|---|---|---|
| corrosion phi/c | 0.00143305 /0.00171859 | 0.422001 /0.446140 | 0.40701613 /0.43102531 |
| electropolish phi/c | 0.290928 /0.299013 | 0.00113782 /0.00135782 | 0.00283171 /0.00342674 |
| melting full graph phi | 6.10970108e-5 | Not tested | 5.98573384e-5 |

Corrosion new phi error is~284 times its best observed seed42 endpoint. The
new electropolish endpoint is~2.49 times its best seed7 phi but remains small.
Melting new full-grid error is0.980 times the prior run. Seed-specific results
are observations, not parameters to select a favorable strategy outcome.

## Late-window and independent physical evidence

Late window comprises the six logged evaluations4000,4200,...,5000. No
smoothing, best-checkpoint replacement or favorable subdomain selection.

| New seed2026 | Late phi median(range) | Late c median(range) |
|---|---|---|
| corrosion | 0.404946(0.399760–0.408093) | 0.429274(0.424088–0.432407) |
| electropolish | 0.00463667(0.00275226–0.00712757) | 0.00560374(0.00329118–0.00855193) |
| melting, sampled graph | 4.97345e-5(2.73270e-5–5.92417e-5) | — |

Melting's late full-grid monitoring has only two snapshots(4000/5000), not six;
the six-point late summary above is explicitly sampled-graph phi. Final
headline comparison uses full-grid error. Independent CPU replay gives
5.98538783e-5. Predicted and reference solid-volume node fractions both equal
0.00407791; solid-fraction L2=5.43177e-4, distinct from bulk-dominated phase L2.
This supports initialization stability in two observed runs, not a guarantee.

Corrosion independent CPU phi/c=0.40701615/0.43102532, pooled phi/c=
0.267134/0.284634. Free-node q/Dq errors=0.310697/4.013898; c_t error2.80636,
stable CH RMS1.55963. Liquid components become2 at t*>=.6 rather than
reference1; final liquid area is9.9204% versus reference29.3341%. The final
phi gallery shows a spurious late phase band. The failure persists over the
late window, rather than being one bad endpoint or only a concentration metric.

Electropolish independent CPU phi/c=0.00283170/0.00342673, pooled phi/c=
0.00225795/0.00275241. Free-node q/Dq errors=0.0185384/0.0549065; c_t error
0.0482061, stable CH RMS0.0222724. Every101 column at all21 saved times has
exactly one interface. This new run reproduces accurate fields, while the
existing seed42 failure prevents calling this random-data strategy universal.
Live q errors use all nodes; independent q/Dq results here use free PDE nodes.

## Runtime seed separation

Canonical train_base configs omit train.seed. Public entries accept a run-only
--seed; common untuned default42. Resolved copied configs retain train.seed
metadata for old checkpoint compatibility and reproduction. Input config and
lock state are preserved. A supplied CLI seed overrides old serialized seed
metadata; without it, old configs retain their recorded value. Invalid/noninteger
or out-of-range seeds fail; legacy fixed-RNG KKS loops reject unsupported
--seed rather than silently ignore it.

```bash
python -m examples.corrosion2d.train --seed=2026
python -m examples.electropolish2d.train --seed=2026
python -m examples.icemelting3d.train --seed=2026
```

The live controls launched before this CLI refactor using the old
--config.train.seed=2026 syntax. Their already resolved configs/source
snapshots/PRNGs were unchanged; the manifest records actual argv, while current
launcher reproduction source uses --seed. No numerical function changed during
training. Retaining seed in a saved run record does not make it a strategy knob.

## Diff-level changes and verification

- New training/runtime.py resolves run metadata without mutating strategy
  configs, including absl-locked ConfigDicts. All3 train.py entries expose
  --seed; all3 canonical configs omit train.seed. Shared KKS snapshots include
  the runtime helper; melting recursively snapshots the package already.
- residual_pilot separates get_strategy_config(case,formulation) from legacy
  get_config(case,formulation,seed). All18 old research combinations(2 cases,
  3 formulations,3 seeds) retain exactly the same resolved config values.
- Five runtime/entry tests pass: both KKS direct/module argv and overrides,
  melting entry resolution, locked copies, invalid seeds, old metadata and
  unsupported legacy switches. Early checks exposed insertion into a locked
  copy; the corrected resolver unlocks only the copied train metadata.
- Five affected melting numerical/model/IC/evaluation/update tests pass after
  the canonical config schema change. No unrelated expensive suite repeated.
- [Preflight](phasefield-seed-stability-preflight.json) checks exact control
  config differences, distinct initial parameters and5 identical numerical
  KKS function ASTs per case. Corrosion adds three explicitly disabled schema
  fields; melting declares5000 instead of its historical planned20000 and
  enables the same monitoring reference by absolute path. No schedule depends
  on the old planned budget. All other strategy fields match.
- All3 final parameter files equal ckpt005000 bytes. Independent CPU replays
  reproduce KKS saved fields within2.86e-6/2.15e-6, and final norms match logs.
  The melting model is independently evaluated over the whole128^3 cube.
  Reference model/grid/time/BC and output-shape checks pass.
- PNG layouts visually inspected: raw convergence and3 phi galleries at
  t*=0,.25,.5,.75,1(reference/prediction/absolute error). Shared scales within
  each gallery, no quantile clipping. Melting gallery is a central grid slice
  at z*=0.503937; its displayed slice L2 differs from full-cube headline L2.
- Strict JSON, syntax, links, PNG/SVG integrity, direct artifact inventory,
  old metadata compatibility and git diff --check pass. GPU idle; all workers
  and independent audit processes exit successfully.

## Direct artifacts and reproduction

All new runs share name20261025-independent-seed2026-default-strategy under
the original /root/autodl-tmp/tf-logs/<case>/ roots. Each retains resolved config,
initial/final/checkpoint weights,source snapshots,metrics and full train.log.
No previous artifact is moved or overwritten; no hashes are needed to load.
The filename date reflects launch UTC25; this task finishes on2026-10-26.

- [Launcher](phasefield-seed-stability-launch.py).
- [Independent audit source](phasefield-seed-stability-audit.py).
- [Complete measurements and direct run paths](phasefield-seed-stability-results.json).
- [Convergence PNG](phasefield-seed-stability-results-convergence.png),
  [SVG](phasefield-seed-stability-results-convergence.svg).
- Phi galleries: [corrosion](phasefield-seed-stability-results-corrosion2d-phi.png),
  [electropolish](phasefield-seed-stability-results-electropolish2d-phi.png),
  [melting](phasefield-seed-stability-results-icemelting3d-phi.png).
- Launch manifest/stdout retained in reports/phasefield-seed-stability-20261025/;
  canonical strategy/seed policy and all3 new directories indexed in
  [artifacts/index.yaml](../../../artifacts/index.yaml).

Audit reproduction from repository root:
`JAX_PLATFORMS=cpu .venv/bin/python -m docs.tasks.archive.phasefield-seed-stability-audit`.
Launcher refuses existing report/run names; choose new explicit output paths
for another authorized study rather than overwrite or silently resume.

## Task end

Complete; no blocker, remaining run, or required next action. Current strategies
are retained for the completed fixed-control comparison; no replacement robust
strategy is invented. READMEs/index explicitly retain the corrosion failure and
electropolish counterexample. Changed memory files: docs/PROGRESS.md,
docs/DECISIONS.md and this archived task. No Git history operations.
