# Local FP64 performance benchmark — 2026-10-26

## Goal, status and conclusion

Complete. User requested measuring this machine's float64 performance before
considering full corrosion FP64 training. All8 serial GPU measurement workers
(2 micro,2 actual-step,4 GEMM policy controls) and1 lowering-only dtype audit
finish successfully. No full5000-step accuracy experiment, production precision
change, FEM work, external asset move or Git history operation was performed.
Prior uncommitted optimization work is preserved. No blocker or remaining job.

For the current corrosion workload, FP64 main computation is practical:
75 measured actual-size updates after50 warm-up give FP32 mean0.203262s,
FP64 mean0.226737s, slowdown1.11549 (~11.5%). This is a performance result,
not evidence that phi/c/q/Dq relative L2 will improve. The next meaningful
accuracy experiment would hold optimizer, graph, initial parameters and batches
fixed while changing arithmetic precision. Production defaults stay FP32.

## Machine, environment and controls

Driver reports NVIDIA GeForce RTX3090,49152MiB (48GiB),580.82.09. JAX/JAXLIB
0.10.1, Equinox0.13.8, Optax0.2.8, SOAP-JAX0.2.2 in repository .venv.
GPU initially idle, one GPU worker at a time; explicit CUDA backend requirement
fails instead of falling back to CPU. JAX preallocation off, matmul precision
highest. Primary micro/training workers retain production
--xla_gpu_deterministic_ops=true. Additional pure GEMM workers compare this
flag against ordinary XLA policy; large GEMM timings remain essentially equal
(57.39 vs57.40ms FP32;216.04 vs216.07ms FP64). No flag benefit is inferred.
These are measured local JAX workloads, not vendor peak throughput claims.

Fresh processes isolate precision/memory/compiler states. Compile/lowering,
first execution, warm-up and steady executions are separated. All timed outputs
are synchronized with block_until_ready. NumPy-generated common FP32 GEMM
inputs are promoted exactly to FP64; sampled outputs pass independent NumPy
FP64 checks (<5e-5 relative for FP32,<1e-11 for FP64). Primary timings are
per-call latency, not an asynchronous enqueue rate.

## Kernel performance

Medians of30 timed GEMMs after5 warm-ups; graph actions use100 after5.
GEMM values below use the corrected explicit deterministic-policy repeat.
Full-graph action is5151 nodes x16 fields, same stored graph coefficients in
both precisions. Pure square GEMM does not predict sparse training cost.

| Kernel | FP32 ms | FP64 ms | FP64/FP32 |
|---|---:|---:|---:|
| gemm_1024_1024_1024 | 0.644021 | 4.51762 | 7.015 |
| gemm_4096_4096_4096 | 57.3867 | 216.036 | 3.765 |
| gemm_16384_64_64 | 0.159014 | 0.314219 | 1.976 |
| graph_diagonal_plus_offdiagonal | 678.786 | 836.798 | 1.233 |
| graph_stable_edge_difference | 696.495 | 768.171 | 1.103 |

The4096 GEMM achieves~2.395TFLOP/s FP32 and~0.636TFLOP/s FP64 in this highest-
precision JAX test. Small-kernel timing varies between process runs; retain all
raw samples. The realistic75-step average, with three full refresh cycles, is
the primary feasibility measurement. Its lower slowdown is consistent with a
workload containing graph operations and small matrices; no full operator-level
performance profile or exact bottleneck attribution was performed.

## Actual corrosion training-step benchmark

Use the current default: original phase residual, raw-KKS CH edges, s_bc,
soft matched graph IC/BC, AC/CH scales1e-7/1, SOAP/lr1e-3, no stagger/time
window/tail/RMS switch. Grid101x51,delta0.02 cubic; Nt16/core256/pool100.
Actual tight padding:2559 nodes,3032 edges; full5151 IC nodes,9 pit BC nodes,
201 solid BC nodes. SOAP refresh5, GradNorm refresh25, same as production.

Load the same initial parameter file:
`/root/autodl-tmp/tf-logs/corrosion2d/20261025-transport-kks_raw_edges-chl1-s_bc-icmatched_graph-bcmatched_graph-phase-seed42/model_initial.eqx`.
Build identical FP32 graph/targets/batches and promote for FP64, preserving
coordinates, edge coefficients, parameters and125 temporal samples. Check graph
rows/weights against the unchanged matched reference. This measures FP64
computation on common stored inputs; it does not claim recovering precision lost
when those inputs/coefficients were originally stored, or construct a new
higher-precision spatial discretization/reference.

Both workers execute125 actual optimizer updates; first50 warm up SOAP/compute.
Measure iterations50..124:75 updates,15 SOAP refreshes including3 GradNorm
refreshes. State structure remains fixed. All losses/gradients/weights stay finite.

| Measurement | FP32 | FP64 |
|---|---:|---:|
| Mean update,including refreshes | 0.203262 s | 0.226737 s |
| Ordinary update mean | 0.181136 s | 0.198617 s |
| SOAP-only refresh mean | 0.186433 s | 0.221616 s |
| GradNorm+SOAP refresh mean | 0.713099 s | 0.809616 s |
| Three25-step block means | 0.201584/0.205174/0.203029 s | 0.226949/0.224704/0.228558 s |
| First JIT lowering+compile | 157.016 s | 155.712 s |
| First execution | 0.862 s | 0.988 s |
|5000-update time extrapolation | 16.94 min | 18.89 min |
| JAX peak active device bytes | 1.535 GiB | 3.063 GiB |
| JAX peak allocator-pool bytes | 2.029 GiB | 4.029 GiB |

The5000 estimate excludes compilation, evaluation, logging and checkpoint saves;
it is not an observed full training duration. JAX allocator statistics exclude
some CUDA context/driver allocations; do not call active bytes whole-process
VRAM. Device has ample capacity for this measured workload. Memory peaks and
compile times include the real update path, not a stripped forward-only pass.

## Precision audit and limitations

All18 network floating leaves,76 initial/final SOAP floating leaves and429 data
leaves have the selected dtype. FP64 parameter/state/data bytes are exactly2x
FP32. Gradients/PDE/activation/SOAP tensor operations are FP64 in StableHLO;
there are no non-scalar FP32 tensors in the FP64 training graph.

A precise exception remains: the existing Optax exponential learning-rate
schedule divides int32 counts, producing FP32 scalar rate arithmetic; that rate
is cast to FP64 before multiplying updates. Retain this optimizer behavior for a
fair production-path comparison. Thus this is FP64 core computation, not a claim
that every floating operation including the scalar schedule is double precision.
The excerpt/source and audit JSON document this explicitly. No neural/PDE/SOAP
matrix computation silently falls back to FP32; FP16/BF16 are absent.

The first micro-FP32 diagnostic regex missed xf32 in shaped GEMM types; its
input/output/numerical checks and timings were valid. Correct the regex and
repeat only GEMMs with explicit type checks; retain original records with a
note and use corrected repeats for the kernel table. Graph/training records
need no repeat. No failed numerical run, silent fallback or hidden dtype default
was accepted. Current train.py has explicit FP32 data casts, so enabling x64
alone would not reproduce this experiment; a full FP64 accuracy study needs
explicit handling of data/model/operator/optimizer precision and reference
compatibility. No L2 benefit or universal hardware slowdown is inferred here.

## Diff-level changes, validation and artifacts

Only isolated benchmark/controllers, machine-result records, memory and a
README precision note are added. Numerical production code/defaults/checkpoints
remain unchanged. Benchmark execution itself exercises the real compiled SOAP/
GradNorm path. Validate all statuses/dtypes/shapes/refresh counts, exact2x
floating storage, StableHLO audit, independent sampled GEMM results, strict JSON,
changed-source syntax, local links and git diff --check. No unchanged expensive
suite is rerun for this isolated probe. All processes exit/GPU0MiB/0%.

- [Benchmark/controller source](fp64-performance.py),
  [queued compiler-policy controls](fp64-performance-gemm-controls.py).
- [Summary](fp64-performance-summary.json),
  [primary manifest](fp64-performance-manifest.json),
  [GEMM-controls manifest](fp64-performance-gemm-controls-manifest.json).
- [FP32 training data](fp64-performance-train-float32.json),
  [FP64 training data](fp64-performance-train-float64.json).
- [FP32 kernels](fp64-performance-micro-float32.json),
  [FP64 kernels](fp64-performance-micro-float64.json).
- [FP64 dtype audit](fp64-performance-audit-float64.json),
  [FP32-scalar IR excerpt](fp64-performance-dtype-audit.txt).
- Corrected GEMM records: [strict FP32](fp64-performance-gemm-deterministic-float32.json),
  [strict FP64](fp64-performance-gemm-deterministic-float64.json),
  [ordinary FP32](fp64-performance-gemm-ordinary-float32.json),
  [ordinary FP64](fp64-performance-gemm-ordinary-float64.json).

All stdout logs and full StableHLO remain at the direct path
`/root/nonlocal-pinn/reports/fp64-performance-20261026/`.
To reproduce in a fresh output directory, run the benchmark main script and the
queued GEMM controller (the latter waits for primary completion); run
`--worker=float64 --suite=audit` with the same environment to inspect IR.
Existing output directories fail rather than overwrite results. GPU env and
exact subprocess commands are in the manifests; archive root discovery is
adjusted after moving sources. No sibling checkout/import dependency or hashes.

## Task end

Complete; no required continuation or running job. Next optional research is a
matched5000-step precision/L2 comparison, not performed under this request to
measure performance first. Current default remains FP32. Memory changed:
docs/PROGRESS.md,docs/DECISIONS.md and this archived task. No commit/merge/push.
