# Original complex CH: shared mixed losses and balancing — 2026-10-28

All four runs use the original ten-wave T1.5 problem, ellipse/two holes, epsilon.02, delta.03, mesh.02 and the same strictly validated reference. One shared spatial MLP64x4/modes8 outputs two groups of24 global log-Chebyshev coefficients: physical phi and mu. No time windows or frozen intermediate states. Phase IC/mass remain hard; mu is learned, not eliminated. Output mu scale.5 is the rounded known-IC RMS.4886. All cold starts use runtime seed42, SOAP5000/LR.001 + L-BFGS500, ordinary FP64. No target labels enter training.

First three are a matched cohort changing ONLY loss_balance: fixed unit weights; inverse-L1-parameter-gradient weights refreshed25/alpha1 with explicit bounds; alternating phase/chemical losses every25 updates for4000 then1000 joint. Shared parameters remain trainable, one optimizer state. This is loss alternation, not a frozen-field stagger solve. Gradient norm refers to the project corrosion rule, not original relative-training-rate GradNorm. L-BFGS uses joint fixed final weights (unit except adaptive).

Fourth is a later raw-residual guard control: same fields/initialization/physics/budget/adaptation, remove the spatial metric and divide evolution by100, chemical relation by.5. These scales use only prescribed IC/operator (native initial rate99.655, mu.4886). It changes residual norm/scales as a bundle, not an isolated inverse-only ablation. Later optional-loss implementation source differences are explicitly recorded.

| Final checkpoint | Phi pooled / FEM % | Phi terminal / FEM % | Mu pooled / native % | Mu terminal / native % | Trainer s |
|---|---:|---:|---:|---:|---:|
| Fixed joint / Neumann inverse | 55.8807 | 62.6196 | 67.7328 | 277.3272 | 265.60 |
| Gradient norm / Neumann inverse | 63.6294 | 71.5108 | 147.1197 | 475.9529 | 293.13 |
| Alternating losses / Neumann inverse | 68.8259 | 67.2824 | 68.1433 | 348.6527 | 282.42 |
| Raw normalized / gradient norm | 55.7275 | 64.2271 | 104.0866 | 364.3306 | 113.81 |

Reference/budget-matched historical benchmarks (not fresh ablations): phase-only global M5 R/t17.985%/29.368%;30windows3.908%/5.754% and3.866%/5.672%. The new controls do not beat either; no failed run is promoted. This does not rule out other shared representations, original GradNorm, field-block schemes or single-network methods in general. One seed/new cohort, known SOAP process variation: no stable universal ranking.

## Actual loss and physics checks

In the first-three common normalized metric, fixed objective parts are evolution.02053156 and chemical.03037566. Gradient balancing reduces chemical to.0000297221 while evolution rises to.20031953. Final weights .500143/1744.9413, maximum chemical weight2336.4912; no floor/ceiling/cap is active at recorded refreshes. The later raw control hits its phase-weight cap at3 of200 refreshes (no norm floor/ceiling hits); the preceding no-cap observation is scoped to the three-arm cohort. Thus inverse-gradient balancing improves algebraic constitutive agreement while sacrificing dynamics; it does not guarantee correct phi or mu. Chemical potential error increases rather than being hidden. Stagger final scalar.04600467 is below fixed.05090722 even though phase is worse. Do not compare the raw/adaptive weighted scalar losses across different norms/weights as field accuracy.

Best-common-phase checkpoints were also replayed:
- Fixed joint / Neumann inverse: step5000, pooled55.8448% / terminal62.2461%. This does not rescue the result.
- Gradient norm / Neumann inverse: step4500, pooled62.6653% / terminal70.1994%. This does not rescue the result.
- Alternating losses / Neumann inverse: step5500, pooled68.8259% / terminal67.2824%. This does not rescue the result.
- Raw normalized / gradient norm: step5500, pooled55.7275% / terminal64.2271%. This does not rescue the result.

| Final field audit (151 dense energy times) | Phase range | Max energy rise | Chemical max defect | Mass drift |
|---|---|---:|---:|---:|
| Fixed joint / Neumann inverse | [-1.048498,1.042720] | 3.98989e-05 | 1.12995 | 3.61e-16 |
| Gradient norm / Neumann inverse | [-1.026702,1.030778] | 2.52372e-05 | 0.0315022 | 2.78e-16 |
| Alternating losses / Neumann inverse | [-1.051785,1.040539] | 0.000504857 | 0.960034 | 1.32e-16 |
| Raw normalized / gradient norm | [-1.059629,1.051627] | 9.94584e-05 | 0.218278 | 2.01e-16 |

All four final/best CPU physical-field replays pass (phase <=7.11e-15, mu <=1.89e-14). Independent SciPy native matrix, lifted inverse where applicable, actual time derivatives and physical chemical law reconstruct each trained final weighted objective to ~1e-15. This validates evaluation/scales/weight metadata in the tested path, not all possible optimizer/model defects. Learned mu is not assumed equal to eliminated chemistry; defects are measured.

First-three initial parameter files are byte-identical, all57 source files identical, full5000x80 time arrays identical and same physical identity/reference. Stagger counts exactly2000 phase-only/2000 chemical-only/1000 joint. Adaptive weights refresh200times; independent recurrence matches recorded norms. The later raw run has identical initial bytes/times, with only README/losses.py/problem.py changed plus its new config; all other sources (network/train/balancing/operators) identical. Direct comparisons, no hashes.

Two focused test rounds pass:3tests26.630s, then2tests18.054s after adding raw normalization (one affected mixed regression repeated). Four distinct tests cover analytic parameter-gradient balancing/stopped weights, real mixed stagger gradients and stage counts, raw dense PDE scaling/parameter finite differences, and legacy shared/separate mixed dense residual/derivative behavior. No unchanged broad suite rerun.

Artifacts: chbalance-analysis.json; chbalance-provenance.json and chbalance-raw-provenance.json; chbalance-tests.txt/raw-tests.txt; all four saved config/summary/metrics/loss/balancing records; convergence.png, energy.png and final/best3x5 phase fields. Figures use shared phase limits +/-1, with true dense ranges reported above. Source snapshots and checkpoints load through explicit run paths under /root/autodl-tmp/tf-logs/confined_ch2d. Complete trajectory still required for window benchmark.

All trainers/replay/audit/plot workers exit0. No new FEM, changed reference, physical model, simple default, selected window fallback, old artifact path or Git history. This requested sub-study is complete; broader global/all-field accuracy remains OPEN. No more seed/weight/step sweeps launched. A further global study should first isolate representation/initial flux/temporal optimization against the recorded failed controls, without claiming a single cause from this table.
