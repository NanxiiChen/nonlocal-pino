# Melting GPU query-chunk consistency

## Observed2026-10-31

Supplemental full-spatial residual validation for the fixed/rolling comparison
fails its independent2e-5physical-field invariance gate. The standard saved
predictor uses1024queries per call. On the final fixed-pool checkpoint, the same
512selected points evaluated within a65536query call differ by max0.00576991
absolute phi. Original first failure0.005770858 is retained separately.

Same input volume and initial latent values are **exactly equal** between the
batched/per-condition SDF paths. Explicit encoded input/noJVP, explicit encoded
input/withJVP, dynamic image/inline encoder and captured image/inline encoder
all retain the discrepancy (0.00576943..0.00576991). These probes exclude those
specific input/JVP/capture choices as sufficient explanations; they do not
isolate the exact responsible operation, backend bug or numerical mechanism.
Do not label this ordinary FP32 roundoff without additional evidence.

The prior [electropolish replay issue](electropolish-attention-replay.md) also
involves execution-graph sensitivity, but a shared cause is not established.
No dependency, neural architecture, precision, compiler flags or checkpoint
was changed to hide this failure. This issue is recorded for separate work.

## What is verified

Both melting5000runs finish;80000actual ICs, alltimes, initial model/Adam and
first32cores match. Independent raw saved-field metrics pass. Fresh same-GPU
replay through the original1024query predictor passes on declared inputs2/5,
maxphi gap<=9.62e-7. Thus the measured fixed-versus-rolling accuracy/timing
comparison is retained **for that identical evaluation path**, not as a
backend/query-partition independent or uniquely sampling-causal ranking.

The failed extra full-spatial residual check is not marked passed; its scalar
residual values are not used to rank models because the large-query field
differs from the standard predictor. Full5000training was already canceled
by the user for cost; the3update full probe remains performance-only.

## Reproduce and evidence

- Script: [melting_spatial_chunk_probe.py](../tasks/melting_spatial_chunk_probe.py).
  Run with the project.venv, JAX_PLATFORMS=cuda, one BLAS/OMP thread and explicit
  XLA_PYTHON_CLIENT_MEM_FRACTION=.20. It loads existing direct paths only.
- Receipt: [melting-spatial-chunk-probe.json](../tasks/melting-spatial-chunk-probe.json).
- Original failed check: [receipt](../tasks/melting-spatial-full-residual-first-failure.json)
  and [script](../tasks/melting_spatial_full_residual.py).
- Passed standard replay: [melting-spatial-replay.json](../tasks/melting-spatial-replay.json).
- Checkpoint: /root/autodl-tmp/tf-logs/icemelting3d/20261031-melting-spatial-01-fixed_pool/model_final.eqx.
- Direct parameters: original reference bank manifest input2. Unit128cube,
  gridplanes y=63..66/127, all128x128XZnodes, selected512indices from
  NumPydefault_rng(12).choice(65536,512,replace=False), times .25/.5/.75/1.

No follow-up training or investigation is automatically queued.
