# Compiler-policy precision explanation — completed 2026-10-26

## Goal and result

Explain why deterministic/ordinary GPU policy changes full-training L2 and
throughput with the same mathematical nonlocal operator. No training, parameter
update, numerical/default change or Git operation. Both serial frozen GPU
workers finish0; existing full-run data are reused. No blocker or active worker.

## Verified observations

- Existing paired full runs have exactly equal initial arrays; configs differ
  only in GPU policy and run name. Both retain FP32, reference, sampling, network,
  SOAP and GradNorm settings. The compiled arithmetic differs despite this.
- Frozen actual first subgraph:2559 nodes x16 times,3032 edges, JAX/jaxlib0.10.1.
  Optimized deterministic graph_action HLO replaces scatter with a while loop
  with known_trip_count3032; body performs successive row updates. Ordinary HLO
  retains scatter. This directly explains why this implementation can lose much
  parallelism. It does not imply every deterministic implementation is slow.
- Ten fixed-input repeats: deterministic forward/input-gradient differences0;
  ordinary relative differences6.4747e-8/6.0662e-8. Cross-policy relative
  differences7.5681e-8/8.2198e-8. These gradients are of the graph-kernel scalar
  squared norm with respect to field values, not full NN parameter gradients.
- Prior independent SciPy forward errors: deterministic diagonal/offdiagonal
  4.12e-8, ordinary7.34e-8; edge-difference5.25e-8 versus5.17e-8. Thus neither
  policy has percent-level single-operator error in these controls. Prior timing
  ~38ms versus~0.17ms is consistent with the newly inspected lowering; no new
  timing benchmark or extrapolated kernel-to-update speed ratio is claimed.
- At first full-training logged update, loss vectors differ1.4655e-7 relative;
  adaptive weight vectors2.9606e-7. At100, differences3.79%/2.64%; at200,
  0.503%/44.33%; at300, weight difference159.5%. Models are already evolving
  differently, so these later differences are not fixed-parameter operator errors.
- Full finalphi0.008835 versus0.020680, latephi median0.007191 versus0.012737;
  ordinary first sampledphi<1% at2800 versus4000 deterministic and improves
  terminal free q/Dq. One realization is not an expected-accuracy ranking.
  Existing independent CPU inference reproduces saved fields at tiny FP32
  roundoff scale, distinguishing training-path change from evaluation artifacts.

## Interpretation and boundaries

Floating-point addition is non-associative. Parallel scatter has colliding row
updates; ordering variation changes rounding. The fixed executable repeat probe
confirms execution-time nondeterminism in ordinary mode, independently of
compiler autotuning. Global flag effects can also change other lowerings; we
have not isolated the full-training difference uniquely to this one kernel.

Data support small numerical perturbations followed by sensitive nonlinear
optimization paths. Stiff coupled residuals, L1 sign changes, adaptive GradNorm
weights and SOAP state/preconditioning are plausible amplifiers. The logged
weight divergence demonstrates feedback paths differ, but does not establish
GradNorm or SOAP as the unique original cause. Neither higher arithmetic
precision nor a changed mathematical PDE follows from the determinism flag.
No proof of systematic deterministic accuracy advantage or ordinary failure.

A potential follow-up is fixed-neighbor/per-row deterministic reduction with
explicit deterministic transpose aggregation for backward propagation, validated
against the same stored operator. Merely optimizing forward and leaving gather
transpose scatter is insufficient to promise reproducibility. Not implemented
in this question/explanation; no promise of bitwise equivalence or identical L2.

## Evidence and primary sources

[Probe source](compiler-policy-explanation-probe.py),
[combined metrics](compiler-policy-explanation-audit.json),
[deterministic worker](compiler-policy-explanation-deterministic.json),
[ordinary worker](compiler-policy-explanation-ordinary.json),
[deterministic HLO](compiler-policy-explanation-deterministic-forward.hlo),
[ordinary HLO](compiler-policy-explanation-ordinary-forward.hlo).
Raw fixed arrays are retained in reports/compiler-policy-explanation-20261026/.
Prior independent performance/accuracy artifacts are referenced by the combined
metrics; full-training provenance remains in local-ad-full-training.md.

[JAX numerical FAQ](https://docs.jax.dev/en/latest/faq.html#jit-changes-the-exact-numerics-of-outputs)
explains compiler transformations and floating-point numerics.
[OpenXLA determinism guide](https://github.com/openxla/xla/blob/main/docs/determinism.md)
distinguishes autotuning/execution nondeterminism and deterministic slowdown.
[XLA scatter semantics](https://openxla.org/xla/operation_semantics#scatter)
defines colliding-update ordering. Current online guide names newer flags;
this experiment tests the existing accepted xla_gpu_deterministic_ops flag
rather than changing flags based on newer documentation.

## Changes and verification

Only research probe/evidence, example README and memory records changed in this
session; unrelated existing uncommitted work preserved. No production numerical
code/config change or test suite needed. Actual frozen GPU worker execution,
fixed-input equality, finite JSON, optimized-HLO inspection, source syntax,
Markdown links and git diff --check provide proportionate verification. Memory:
PROGRESS.md, DECISIONS.md and this archived task. No further work is required
for the explanation; any optimizer ablation or aggregation implementation is
separate work.
