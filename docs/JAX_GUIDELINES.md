# JAX And Equinox Guidelines

This document contains the implementation details behind the fixed-shape and
functional-style rules in `AGENTS.md`.

## Equinox Fields

Use `eqx.field(static=True)` for Python callables, dispatch strings, booleans,
and small shape-defining integers. Use
`eqx.filter_value_and_grad(fn)(net)` so static fields are excluded from
automatic differentiation.

Network calls must be pure functions of their inputs. Do not mutate module
fields after construction; return updated modules or use `eqx.tree_at`.

## JIT Invariants

JAX recompiles primarily when argument shapes or dtypes change. Keep these
fixed inside a compiled training path:

- operator edge arrays and matrix shapes;
- grid/collocation array shapes;
- BC array shapes;
- network architecture and output shapes.

Build operators, grids, and fixed BC arrays once outside the step function. Do
not capture changing `D` or `x_points` in a newly created loss closure.

## Sampling

Variable-length core+halo or filtered point sets must be padded to a fixed
maximum and masked. Oversample-and-truncate filtered LHS points to a fixed
count. A sampled core loss changes the optimisation objective unless its
quadrature or inclusion-probability weighting is explicitly derived.

## Traced Values

Values that change across iterations, including flags, counters, EMA values,
and schedules, must enter JIT as JAX arrays rather than changing Python
primitives. Use `jax.lax.stop_gradient` for weights and buffers that should not
receive gradients.

## Computation Style

- Prefer `jax.vmap` to Python loops over array batches.
- JIT the training step, not the loss factory.
- Use full-batch checks for acceptance even when training uses subgraphs.
- Keep local and nonlocal paths explicit; do not silently fall back after an
  exception.
