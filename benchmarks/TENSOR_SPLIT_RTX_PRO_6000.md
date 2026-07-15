# Tensor-split and GradNorm branches on RTX PRO 6000

Date: 2026-07-15.

This note is a follow-up to `benchmarks/RESULTS.md`.  Those results were
measured on an NVIDIA B200.  The measurements below were taken on an NVIDIA RTX
PRO 6000 Blackwell Server Edition with JAX 0.10.1 and
`matmul_precision="highest"`.  All timings synchronize the full returned pytree,
not only the scalar loss.

## Main Takeaways

`arch.tensor_split=True` still performs the intended algebraic rewrite: the
first trunk layer is evaluated as

```text
W @ [ex; et] + b  ->  W_x @ ex + W_t @ et + b
```

Under a tensor-product `vmap(x) x vmap(t)` evaluation this removes a large
first-layer matmul over `N_x * N_t` points.  However, full training-step time is
not determined only by that matmul.  Backward-pass codegen, loss grouping,
broadcast/reduce traffic, optimizer work, and XLA fusion choices can dominate.

The important distinction is the loss structure:

- **icemelting3d** has one expensive PDE loss and one relatively different IC
  loss.  Splitting the gradients by loss term can make the compiled program much
  faster for `tensor_split=True`.
- **electropolish2d/corrosion-like losses** have multiple PDE terms, especially
  AC and CH, that share the same expensive `_compute_all` result.  Splitting AC
  and CH into independent backward passes repeats the dominant network/JVP work
  and is slower.

So the practical rule is not "more separate gradients is always faster".  The
rule is: preserve sharing inside expensive PDE groups, and only split loss groups
when the smaller compiled graphs outweigh the extra forward/JVP work.

## Why More Backward Calls Can Be Faster

For fixed weights, these two gradients are mathematically equivalent:

```text
grad(w0 * L0 + w1 * L1) = w0 * grad(L0) + w1 * grad(L1)
```

But XLA does not necessarily lower them to the same GPU program.  The first form
creates one larger reverse-mode graph.  The second form exposes smaller
per-loss reverse-mode graphs followed by an elementwise gradient-pytree weighted
sum.

The number of Python backward calls is therefore not the unit of GPU cost.  The
actual cost is the compiled HLO/kernel program.  One large backward graph can be
slower than two smaller backward graphs if it creates worse fusion boundaries,
higher register pressure, longer-lived intermediates, or more expensive
broadcast/reduce scheduling.  This matters more after tensor-split because the
first layer introduces a broadcast add and corresponding backward reductions.

## icemelting3d

Small fixed-shape diagnostic:

- `K_core=512`, padded subgraph `N_pad=3557`
- loss terms: PDE and IC
- MLP + SOAP

Full step, current regular/update branches:

| network | branch | median step time |
|---|---|---:|
| concat | regular | 7.54 ms |
| tensor-split | regular | 8.09 ms |
| concat | GradNorm update | 7.68 ms |
| tensor-split | GradNorm update | 5.44 ms |

Forward/JVP-only diagnostics still show that tensor-split removes useful network
work at larger `N_x`:

| `N_x` | concat fwd/JVP | split fwd/JVP | speedup |
|---:|---:|---:|---:|
| 2048 | 0.819 ms | 0.807 ms | 1.02x |
| 4096 | 1.949 ms | 0.934 ms | 2.09x |
| 8192 | 4.187 ms | 2.366 ms | 1.77x |
| 13879 | 6.998 ms | 3.888 ms | 1.80x |

The surprising part is the full step: with `tensor_split=True`, the GradNorm
update branch is faster than the regular branch, even though it computes one
gradient per loss term.

A follow-up diagnostic separated the effect of updating GradNorm weights from
the effect of using separate per-loss gradients:

| variant | meaning | median step time |
|---|---|---:|
| current regular | `grad(w0 * L_pde + w1 * L_ic)` | 7.97 ms |
| separate fixed-weight grads | `w0 * grad(L_pde) + w1 * grad(L_ic)`, without updating weights | 5.24 ms |
| GradNorm update | per-loss grads, recompute GradNorm weights, then combine grads | 5.14 ms |

This suggests that the slow path is not the Python closure around `total_fn`;
Python closures are traced away before XLA lowering.  The important difference
is differentiating one combined weighted objective versus differentiating each
loss term separately.

For icemelting, this separate-gradient path is plausible because the IC loss is
not another expensive tensor-product PDE residual.  Splitting PDE and IC gives
XLA smaller, more regular programs without duplicating a second PDE
`_compute_all` path.

## electropolish2d

Electropolish has a different structure: AC and CH are two PDE losses that share
the same expensive `_compute_all` result.  The current regular branch computes
AC and CH together via `loss_pde`, so the network forward/JVP is shared.  The
current GradNorm update branch computes `loss_ac` and `loss_ch` separately, so
AC and CH each call `_compute_pde` and repeat `_compute_all`.

Small fixed-shape diagnostic:

- grid: `101 x 51 = 5151` points
- `Nt=20`
- sparse D with `nnz=60296`
- loss terms: AC, CH, BC top, BC bottom, IC
- MLP + SOAP

| network | regular | GradNorm update | naive separate fixed grads | grouped separate fixed grads |
|---|---:|---:|---:|---:|
| concat | 5.86 ms | 9.12 ms | 8.84 ms | 6.08 ms |
| tensor-split | 5.04 ms | 7.13 ms | 10.09 ms | 4.98 ms |

Definitions:

- **regular**: current `update_w=False`; AC and CH are computed together through
  `loss_pde`, sharing `_compute_all`.
- **GradNorm update**: current `update_w=True`; AC, CH, BC top, BC bottom, and
  IC are differentiated separately.  AC and CH repeat `_compute_all`.
- **naive separate fixed grads**: same per-loss split as update, but without
  recomputing GradNorm weights.  It still repeats AC/CH `_compute_all`.
- **grouped separate fixed grads**: AC+CH are kept together as a PDE group so
  they share `_compute_all`; BC and IC are differentiated separately.

The result is the opposite of the icemelting lesson if one splits too finely:
per-loss gradients are slow because they duplicate the dominant PDE network/JVP
work.  The grouped version is the right shape for fixed-weight regular gradients:
it preserves AC/CH sharing and is roughly tied with the current regular path for
tensor-split, while slightly slower for concat.

There is an important constraint for true GradNorm updates: AC and CH cannot be
fully collapsed into one opaque PDE group, because GradNorm needs separate
`grad(L_ac)` and `grad(L_ch)` norms to update `weight_ac` and `weight_ch`.  The
regular branch only needs the weighted total gradient, so grouping AC+CH is
valid there.  The update branch needs per-term gradients, but the implementation
should ideally share the primal `_compute_all` work between AC and CH rather than
calling `_compute_pde` separately for each term.  A possible future direction is
a shared-forward/per-term-VJP update path: compute AC and CH losses from one
`_compute_pde`, then pull back separate cotangents to recover per-term gradients
for GradNorm.

Tensor-split itself is still useful here: the regular branch improves from
`5.86 ms` to `5.04 ms` in this reduced electropolish case.  The slow part is not
tensor-split; it is the current per-loss GradNorm/update decomposition that
breaks AC/CH sharing by repeating `_compute_all`.

## Practical Guidance

1. Do not assume B200 tensor-split timings transfer directly to RTX PRO 6000.
   The algebra is exact, but XLA codegen and GPU balance can change the full-step
   result.
2. For icemelting-like losses, a separate fixed-weight gradient path can preserve
   regular-step semantics while exposing a faster graph structure.
3. For corrosion/electropolish-like regular steps, keep PDE terms that share
   `_compute_all` in the same group.  Splitting AC and CH into separate
   backward passes repeats the expensive forward/JVP path and is slower.
4. For true GradNorm update steps, still preserve separate AC/CH gradients for
   their weights; the optimization target is sharing the primal PDE evaluation,
   not merging the loss terms into one unseparable gradient.
5. A useful future option is not simply `separate_loss_grads=True`, but grouped
   gradient computation plus a possible shared-forward/per-term-VJP update path.
6. Any change should be validated on full-size target runs for both step time and
   time-to-accuracy; small diagnostics are for identifying codegen trends, not
   final performance claims.
