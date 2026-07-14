# Performance optimization results

Environment: NVIDIA B200, JAX 0.10.1, CUDA 13, float32.
Date: 2026-07-14.  All numbers from the scripts in this folder (median of
timed iterations after JIT warm-up, `block_until_ready` per step, optimizer
state carried forward like real training).

**Accuracy constraint**: every change in this PR is an exact reformulation of
the same FP32 math — no approximation, no precision loss.
`verify_equivalence.py` gates each one against the reference implementation
(loss values, full gradient pytrees, GradNorm weights).  Both new code paths
pin `Precision.HIGHEST` internally, so they stay exact even if the global
matmul precision is lowered.

## What's in this PR

1. **ConvD** (`nl.method="conv"`): the non-local operator evaluated as a
   stencil convolution on structured grids.  On a uniform meshgrid the kernel
   weight depends only on the integer offset, so
   `D @ u = diag*u - conv(u, W)` with `diag = conv(ones, W)` handling
   boundary truncation exactly.  Default for the 2D corrosion config.
2. **Tensor-split first layer** (`arch.tensor_split`, mlp trunk): the losses
   evaluate the net on a tensor product `vmap(x) x vmap(t)`; splitting the
   first trunk layer `W@[ex;et] = W_x@ex + W_t@et` lets vmap hoist each
   partial matmul out of the opposing batch dimension — the dominant
   `(N_x*N_t, 256, 64)` matmul disappears.  Off by default (see caveats).
3. **`train.matmul_precision` config flag**: defaults preserve current
   behavior exactly ("highest" = exact FP32 for the 2D scripts, JAX default
   for icemelting3d).  Provided so the measured TF32 speedup (below) can be
   enabled deliberately after validation, never silently.
4. Benchmark + verification scripts (this folder).

## Equivalence verification (verify_equivalence.py)

| change | loss rel dev | grad rel dev |
|---|---|---|
| ConvD vs SparseD (full loss, regular + GradNorm steps) | ≤2.5e-06 | ≤2.5e-06 |
| tensor-split vs concat net (outputs, d/dt JVP, loss, grads) | 0.0 | 7.5e-09 |

All at FP32 round-off level (tolerance 2e-5).  `operators.py` self-tests
additionally verify ConvD against DenseD/SparseD on 1D/2D/3D grids.
A 1000-iteration 3D training run with tensor-split on/off converges
identically (rel-L2 vs analytic @1000: 2.944e-2 vs 2.938e-2 — round-off-level
seed noise).

## Operator microbenchmark (D @ u and its VJP)

2D full grid 201x101 = 20 301 points, delta = 0.02 (nnz = 950 396,
k_avg = 47), Nt = 50, exact FP32:

| variant | fwd ms | vjp ms |
|---|---|---|
| sparse (segment_sum, baseline) | 0.110 | 0.110 |
| **conv (stencil)** | **0.042** | **0.047** |
| bcoo (jax.experimental.sparse SpMM) | 0.111 | 0.111 |
| dense | 1.537 | 1.977 |

## Training-step benchmarks

2D corrosion full grid (201x101, Nt = 50, FiLM 64x4, Adam), exact FP32,
ms/iter (regular / GradNorm-update step):

| variant | regular | update |
|---|---|---|
| sparse operator (baseline) | 13.91 | 19.53 |
| **conv operator** | **12.85** (1.08x) | **18.22** (1.07x) |

Tensor-split, isolated network evaluation (`bench_tensor_split.py`,
mlp trunk, exact FP32):

| shape | fwd+JVP | loss value+grad |
|---|---|---|
| 2D (20 301 x 50) | 4.05 → 2.29 ms (**1.77x**) | 15.94 → 8.13 ms (**1.96x**) |
| 3D (13 879 x 64) | 3.29 → 2.00 ms (**1.65x**) | 11.78 → 7.25 ms (**1.62x**) |

Full 2D training step with the mlp arch (Adam, conv operator, exact FP32):
regular 17.4 → 10.2 ms (**1.71x**), GradNorm-update 24.8 → 13.9 ms (**1.78x**).

3D icemelting step (128^3 grid, K_core = 2000 subgraph, Nt = 64, MLP 64x4,
SOAP), at `matmul_precision="highest"`:

| variant | regular | update |
|---|---|---|
| concat net (baseline) | 13.85 | 14.01 |
| **tensor-split net** | **12.05** (1.15x) | **8.90** (1.57x) |

### Tensor-split caveats (why it's off by default)

- Only applies to the **mlp trunk**.  The FiLM net (2D corrosion default) is
  already implicitly factorized — its trunk input depends only on x and its
  conditioning only on t, so vmap already hoists everything hoistable.
- The win depends on matmul precision.  Under icemelting3d's default
  (TF32-capable JAX default) precision the first-layer matmul is already
  cheap and the extra broadcast add makes the split roughly neutral for
  regular steps.  Under exact FP32 it is a large win.  Enable with
  `--config.arch.tensor_split=True` for mlp-trunk runs at
  `matmul_precision="highest"`.

## TF32 experiment (flag only; defaults unchanged)

1000-iteration side-by-side training runs, identical seeds:

| workload | metric | FP32 "highest" | TF32 | step time |
|---|---|---|---|---|
| 3D icemelting | rel-L2 vs analytic @1000 | 2.9996e-02 | 2.7470e-02 | 1.5x faster |
| 2D corrosion | loss trajectory | — | matches within noise | 1.2x faster |

TF32 tracked FP32 within run-to-run noise (3D final error slightly *lower*
under TF32), and the D operator always stays exact FP32 (precision pinned
internally).  Defaults remain exact FP32; enable explicitly with
`--config.train.matmul_precision=tensorfloat32` after validating on your
target problem to completion.

## Evaluated but not included

For the record, three further optimizations were implemented, verified
exact, benchmarked — and dropped for not clearing the bar:

- **ELL (gather) sparse matvec** instead of scatter-add `segment_sum`:
  0.98-1.00x on B200 — XLA's segment_sum is already efficient at this
  sparsity pattern.
- **GradNorm forward de-duplication** (one shared PDE forward + two VJP
  pullbacks): neutral — XLA already CSEs the two identical forwards inside
  one jit graph.
- **Tile-based subgraph sampling** (contiguous-block minibatch cores in 3D):
  2.1-2.4x faster per step (halo drops from 5.9x core to 0.8x), but the
  spatially correlated batches slowed per-iteration convergence enough at
  delta/h ≈ 1.3 (k_avg = 6) to lose on time-to-accuracy.  Worth revisiting
  if the horizon/grid ratio grows (halo ratios of 50x+ change the math).

## Where the time goes

The nonlocal operator is <2% of a training step.  ~95% is network
evaluation at N_x·N_t points (forward + time-derivative JVP + backward);
that is what the tensor-split (and, opt-in, TF32) attack.

## Reproduce

```bash
python benchmarks/verify_equivalence.py      # accuracy gate
python benchmarks/bench_operator.py
python benchmarks/bench_step.py --workload 2d [--tf32]
python benchmarks/bench_step.py --workload 3d --precision highest
python benchmarks/bench_tensor_split.py      # first-layer split, mlp trunk
```
