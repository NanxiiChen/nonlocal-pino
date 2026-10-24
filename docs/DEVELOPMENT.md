# Development Guide

This document contains repository conventions that are broader than one
example but more detailed than the mandatory policy in `AGENTS.md`.

## Example Layout

Every example under `examples/<name>/` should use:

```text
examples/<name>/
├── README.md                  # physics, BCs, formulation, usage
├── train.py                   # grid, network, operator, training loop
├── losses.py                  # residuals, BC losses, loss factory
├── solver.py                  # independent FEM/reference solver if applicable
├── mesh.geo                   # geometry if applicable
└── configs/
    ├── train_base.py          # PINN configuration
    └── solver_base.py         # reference-solver configuration
```

Keep reference solvers independent from PINN operators when they are intended
to measure model or discretisation error.

## APIs And Naming

Use the shared loss-factory shape:

```python
def make_loss_fn(x_all, D, ...physical_params...,
                 ...bc_data..., alpha=0.9, gradnorm_cap=1000.0):
    """Return loss_fn(net, w_ema, bias, update_w, x_points=None)."""
```

- `loss_fn` receives `x_points` explicitly, never `batch_idx`.
- Capture fixed BC data in the factory; pass changing collocation data per step.
- `D is None` means local AD; `D is not None` means nonlocal mode.
- Hard constraints handle Dirichlet data; do not duplicate it as a loss term.
- Use `make_loss_fn`, `_make_step`, `_build_grid`, `_build_network`,
  `_build_D`, `_build_optimizer`, and `_eval_fem` where those roles apply.
- Use names such as `c.arch.hidden_dim`, `c.nl.delta`, `c.sample.Nx`, and
  `c.train.lr_init`.

## Configuration

- Use `ml_collections.ConfigDict`.
- Put material data in `c.physical`, geometry in `c.geometry`, nonlocal data in
  `c.nl`, architecture in `c.arch`, and training data in `c.train`.
- Declare every required field explicitly. Do not use `.get()` fallbacks or
  hidden function-argument defaults for required settings.
- Validate physical ranges, array shapes, grid ordering, operator assumptions,
  and required input files at startup.
- Fail loudly with context. Never catch an arbitrary exception and silently
  switch to another numerical method.

## Verification

Choose the smallest verification set that gives credible evidence for the
change. More checks are not automatically better.

- Documentation, agent instructions, and configuration-only edits normally
  need diff inspection, applicable syntax/schema validation, and
  `git diff --check`; they do not require the Python test suite or compileall.
- A localized behavior change normally needs one focused regression test or the
  narrowest existing test command that reaches the changed path. Add a test
  only when it protects meaningful behavior or a reproduced defect; do not add
  tests for wording, trivial wiring, or implementation details already covered.
- Compile only changed Python modules when syntax was not already exercised by
  the focused test. Do not compile unrelated trees by default.
- Run broader integration, numerical-equivalence, GPU, FEM, or full-suite checks
  only for changes that cross those boundaries, alter shared numerical code or
  public APIs, or when the user explicitly requests them.
- Avoid repeating expensive checks whose relevant inputs have not changed.
- Always run `git diff --check` and update the example README when its public
  API, formulation, configuration, or file structure changes.

## Markdown Math

Markdown intended for OpenCode Web uses inline `\(...\)` and display `$$`
delimiters, with each display delimiter on its own line. Do not use `$...$` or
`\[...\]`.

## Project split package layout

Reusable implementation lives in `src/nonlocal_pinn`; example directories own
case geometry, BCs, formulation and CLI. Imports use `nonlocal_pinn`, with no
`pinn` compatibility namespace and no sibling checkout dependency.
Operator implementation is preserved as one module inside `operators/` pending
separately justified subdivision.

See migration/source-map.json for old-to-new source ownership.
