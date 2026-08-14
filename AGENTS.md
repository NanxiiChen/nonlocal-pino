# AGENTS.md — AI Agent Instructions for nonlocal-KKS-PINN

This file is read by AI coding agents (Claude Code, etc.) working on this
repository.  These instructions are **mandatory** — violations will be
flagged and reverted.

---

## 1. Project Background

This is a **nonlocal Physics-Informed Neural Network (PINN)** framework for
phase-field and solid-mechanics problems.  The core idea is replacing the
local Laplacian \(\nabla^2\) with a precomputed sparse nonlocal operator
\(D_\delta\) built from peridynamic neighbour pairs.  Neumann BCs emerge
automatically from horizon truncation — no explicit BC loss terms needed.

For a detailed overview, see [`README.md`](README.md).  If the README is
outdated (missing examples, wrong file structure, stale API references),
**update it first** before proceeding with implementation work.

Key examples (see `examples/`):

| Example | Physics | Operator |
|---------|---------|----------|
| `icemelting3d` | Allen-Cahn, ice sphere melting | SparseD |
| `electropolish2d` | KKS phase-field, electropolishing | SparseD |
| `fracture1d` | AT2 phase-field fracture | SparseD |
| `elasticity2d` | Linear elasticity, plate with elliptical hole | ElasticD (correspondence peridynamics), local AD |
| `pino_elasticity2d` | PINO design for multi-geometry elasticity | ElasticD + mask weight-zeroing |

---

## 2. Code Style — Consistency Across Examples

### 2.1 File structure

Every example under `examples/<name>/` must follow this layout:

```
examples/<name>/
├── README.md                  # Physics, BCs, formulations, usage
├── train.py                   # Training script (grid, net, D, loop)
├── losses.py                  # PDE residual, BC losses, GradNorm, make_loss_fn
├── solver.py                  # FEM reference solver (if applicable)
├── mesh.geo                   # Gmsh geometry (if applicable)
├── configs/
│   ├── train_base.py          # PINN training hyperparameters
│   └── solver_base.py         # FEM solver config
```

### 2.2 API conventions

All `make_loss_fn` factories must follow the same signature pattern:

```python
def make_loss_fn(x_all, D, ...physical_params...,
                 ...bc_data..., alpha=0.9, gradnorm_cap=1000.0):
    """Factory returning loss_fn(net, w_ema, bias, update_w, x_points=None)."""
```

Key rules:
- `loss_fn` takes `x_points` as an explicit keyword argument, not `batch_idx`
- `make_loss_fn` captures BC arrays and boundary conditions in its closure
- PDE collocation points are passed per-step, never baked into the factory
- `D is None` signals local (all-AD) mode; `D is not None` signals nonlocal mode
- Hard constraint handles Dirichlet BCs; `loss_fn` never contains Dirichlet BC loss terms

### 2.3 Naming

- Config fields: `c.arch.hidden_dim`, `c.nl.delta`, `c.sample.Nx`, `c.train.lr_init`
- Helper functions in train.py: `_build_grid`, `_build_network`, `_build_D`, `_build_optimizer`, `_eval_fem`
- Loss functions: `loss_pde`, `loss_bc_top`, `loss_bc_hole`, etc.
- Factory: `make_loss_fn` (not `make_loss` or `build_loss`)
- Step function factory: `_make_step` (train.py internal)

### 2.4 Configs

- Use `ml_collections.ConfigDict`
- Physical parameters always go in `c.physical` (lam, mu, E, nu)
- Geometry always in `c.geometry` (Lx, Ly, hole params)
- Nonlocal params in `c.nl` (delta, kernel, method)
- Training params in `c.train`
- **No implicit defaults** — every field must be explicitly defined (see §5)

### 2.5 OpenCode Web math rendering

All user-facing responses and Markdown intended for OpenCode Web must use its
KaTeX-compatible delimiters:

- Inline math uses `\(...\)`.
- Display math uses `$$` delimiters, with each `$$` on its own line.
- Do not use `$...$` or `\[...\]`; they may not render in OpenCode Web.
- Do not place formulas that should render inside code fences.

---

## 3. No Guessing — Mathematical and Numerical Rigour

### 3.1 Absolute rules

- **ZERO tolerance for guessed formulas.**  Every PDE residual, every
  constitutive law, every kernel normalisation must be traceable to a
  published reference or a mathematically verifiable derivation.
- **If you are unsure about a numerical method**, do NOT implement it.
  Discuss it with the user first.  This includes:
  - Kernel normalisation constants (tophat, cubic, quartic, Wendland)
  - Non-dimensionalisation schemes
  - Nonlocal operator formulations (D_δ Laplacian, ElasticD correspondence)
  - Boundary condition enforcement (penalty vs hard constraint vs Nitsche)
  - Adaptive loss-weighting schemes (GradNorm, NTK, residual-based, etc.)
  - Optimizer hyperparameters for PINN-specific loss landscapes
- **Follow community standards for numerical methods.**  Examples:
  - Peridynamics: Silling et al. (2007) correspondence formulation
  - FEM: standard CG1 weak forms, not ad-hoc stiffness assembly
  - Phase-field: AT2/AT1 degradation functions, not invented variants
  - Kernel methods: standard J0 normalisation (cubic, quartic, Wendland C²)

### 3.2 When to stop and ask

Stop and ask the user if:
- You are about to invent a formula rather than look it up
- The reference you found describes a different setup (e.g., 1D vs 2D
  kernel normalisation, or dynamic vs static problem)
- There are multiple valid approaches and the choice is non-trivial
- You discover an inconsistency between the code and the equations it
  claims to implement — flag it, don't silently "fix" it

---

## 4. JAX / Equinox — Functional Style and Static Fields

### 4.1 Equinox `static=True` — quick reference

Fields marked `eqx.field(static=True)` are excluded from the PyTree and
not traced by `jax.grad`/`jax.jit`.  Use `static=True` for:
- Python callables (`activation`, `constraint`), dispatch strings
  (`trunk_type`, `kernel`), booleans (`multi_head`, `tensor_split`)
- Small ints that affect array shapes (`hidden_dim`, `num_layers`)

Use `eqx.filter_value_and_grad(fn)(net)` as the standard training-step
pattern — it filters static leaves, differentiates only through arrays.

### 4.2 Avoiding JIT recompilation — the real issues

The dominant source of recompilation in PINN training is **NOT** mis-marked
`static=True` — it's **shape changes and dtype changes** in arguments to
JIT-compiled functions.

#### 4.2.1 Shape stability

JAX JIT retraces when tensor shapes change between calls.  This is the
single most common recompilation pitfall.

**Pattern 1 — variable-length collocation points (core + halo sampling):**
```python
# ❌ WRONG — sub-sampled point count varies → recompile every step
x_sub, sub_data, core_mask = D.sub_sample(seed, K_core)
loss = loss_fn(net, x_points=x_sub)

# ✅ CORRECT — pad to a fixed maximum size, mask padded entries
x_sub, sub_data, core_mask = D.sub_sample(seed, K_core)
x_padded = jnp.pad(x_sub, ((0, K_max - len(x_sub)), (0, 0)))
# ... mask applied in loss computation to ignore padded entries
```

**Pattern 2 — LHS sampling with hole exclusion:**
```python
# ❌ WRONG — filtering changes the point count
pts = lhs_sampler.random(n=N)
pts = pts[is_in_domain(pts)]  # length varies

# ✅ CORRECT — oversample + truncate to fixed size
raw = lhs_sampler.random(n=int(N * 1.3))    # oversample 30%
pts = filter_and_truncate(raw, N)            # always returns exactly N points
```

**Pattern 3 — BC point arrays:**
```python
# ✅ CORRECT — precompute BC arrays once at setup, never change shape
x_bc_top = sample_bc_points(n_bc)    # fixed (n_bc, 2), created once
```

#### 4.2.2 Traced scalar values

Scalars that change across training steps (iteration counters, EMA decay
rates, GradNorm update flags) must be passed as **JAX arrays** (traced),
not Python ints/floats/bools (static).

```python
# ❌ WRONG — Python bool changes value → recompile
update_w = (it % gradnorm_every == 0)       # Python bool, toggles

# ✅ CORRECT — wrap as JAX scalar, stays traced
update_w = jnp.array(it % gradnorm_every == 0)  # jnp scalar — traced

# ❌ WRONG — Python float changes → value change may trigger recompile
alpha = 0.9 if it < 1000 else 0.99

# ✅ CORRECT — wrap in jnp.array; same shape → no recompile
alpha = jnp.array(0.9) if it < 1000 else jnp.array(0.99)
```

**Rule of thumb:**  If a value appears in a JIT-compiled function and its
Python value changes across iterations, it MUST be a JAX array (traced).
Only values that are truly constant throughout training can be Python
primitives (static).

#### 4.2.3 Precompute and freeze

- **Operator matrices** (`ElasticD`, `SparseD`): built once at setup, JIT
  traces the matmul with fixed COO shapes → no recompile as long as N is
  fixed.
- **BC point arrays**: sampled once, fixed shape, passed as arguments.
- **Grid points**: fixed (N, dim) array, constant across all steps.

### 4.2 Functional style

- **No mutation of module fields after `__init__`.**  Use `eqx.tree_at` or
  return updated equinox modules from training steps.
- **No side effects in `__call__`.**  Network forward passes must be pure
  functions of `(self, x, t)`.
- **`jax.vmap` over explicit Python loops** for batched computation.
- **JIT the step function, not the loss factory.**  The loss function
  closure should be compatible with `eqx.filter_value_and_grad`.
- **Use `jax.lax.stop_gradient`** for weights and EMA buffers that should
  not contribute to the gradient computation.

### 4.3 Avoiding recompilation

- **Fixed array shapes**: If collocation point counts vary, oversample and
  truncate to a fixed size rather than passing variable-length arrays.
- **Precompute static data**: Operator matrices (`ElasticD`, `SparseD`),
  BC point arrays, and neighbour tables are built once at setup time.
- **Don't put `D` or `x_points` inside the `loss_fn` closure** if they
  would change shape or dtype across calls.

---

## 5. Explicit Over Implicit — No Silent Behaviour

This is a **hard requirement** from the user.  Violations are treated as bugs.

### 5.1 Config access

```python
# ✅ CORRECT — explicit, fails loudly if key is missing
delta = config.nl.delta
Nx = config.sample.Nx

# ❌ WRONG — silent fallback hides misconfiguration
delta = config.get('delta', 0.01)
Nx = config.get('Nx', 200)
```

Every config field must be explicitly defined in the config file.  Never
substitute a default — if a value is missing, the program must crash with
an informative `AttributeError`.

### 5.2 Function arguments

```python
# ✅ CORRECT — explicit parameter, no default
def _build_grid(config):
    a_x = config.geometry.a_x

# ❌ WRONG — hidden default masks missing config
def _build_grid(config, Nx=200):
```

Only use defaults when there is a clear, physics-based reason (e.g.,
GradNorm `alpha=0.9` is the standard EMA rate; `eps=1e-6` prevents
division by zero).  Document why the default exists.

### 5.3 Error handling

- **Crash early, crash loud.**  If a required file is missing,
  `raise FileNotFoundError` with the expected path — don't fall back
  to a different method.
- **Validate assumptions at startup.**  Check that `Nx * Ny == len(x)`
  for structured grids, that `delta > 0`, that Lamé constants satisfy
  the physical constraints.
- **No `try/except` that swallows errors.**  If you catch an exception,
  either re-raise it with more context or handle only the specific
  case you understand.

### 5.4 Acceptable vs unacceptable defaults

Some fallback patterns are acceptable when they represent **well-defined
alternative computation paths** with clear semantics, not hidden error
concealment.

**Acceptable** — explicit dispatch with a documented fallback:

```python
# ✅ OK — full batch is a well-defined alternative to sub-sampling
if sub_data is None:
    loss = loss_fn(net, x_points=x_all)         # full batch
else:
    loss = loss_fn(net, x_points=x_sub, ...)    # core + halo

# ✅ OK — uniform dV is a valid choice for structured grids
dV = compute_barycentric_dV(x, tri) if tri is not None else uniform_dV(x)
```

**Forbidden** — patterns that silently conceal errors:

```python
# ❌ Catches everything, hides the real error
try:
    D = build_sparse(x, delta)
except Exception:
    D = build_dense(x, delta)

# ❌ .get() with default — masks missing config field
method = config.get('method', 'sparse')

# ❌ Try one path, silently fall back to another on any failure
try:
    result = fast_path(x)
except:
    result = slow_path(x)
```

The distinction:
- **Acceptable**: an explicit conditional choosing between two known,
  well-defined computation paths based on the availability of input data
  (e.g., "if no sub-graph is provided, use full batch")
- **Forbidden**: catching exceptions and silently switching to a different
  code path, or substituting default values for missing config fields

---

## 6. Git Workflow — Authorization Required

### 6.1 Hard rule: NEVER commit without explicit permission

**Do not run any git write operation** (`git commit`, `git merge`,
`git push`, `git tag`, `git stash`, `git branch -D`, `git rebase`,
`git reset --hard`, `git checkout --`, `git clean`, etc.) unless the
user has **explicitly and unambiguously** asked you to do so in the
current turn.  This includes:

- `git commit` / `git commit --amend`
- `git merge` / `git merge --squash`
- `git push` / `git push --force`
- `git stash` / `git stash pop`
- `git rebase` / `git rebase -i`
- `git reset` / `git checkout -- <file>`
- `git branch -D` / `git tag -d`
- Anything that modifies the repository state or history

**Implicit permission does not count.**  If the user said "commit and
push" two turns ago, that permission is expired — you must ask again
before doing any new git write.  If the user says "commit the changes"
without specifying which branch, ask — don't assume.

Read-only operations (`git status`, `git diff`, `git log`, `git branch`)
do not require permission.

### 6.2 Workflow

- **dev branch**: feature/experiment work, may have multiple WIP commits
- **main branch**: `git merge dev` — regular merge, no `--squash`
- Workflow: `dev` → commit → `main` → `git merge dev` → `git push origin main`
- Commit messages end with `Co-Authored-By: Claude <noreply@anthropic.com>`
- Push only `main` to origin; `dev` stays local

---

## 7. Environment

Two mutually exclusive Python environments:

| Environment | Purpose | Activation |
|-------------|---------|------------|
| `fenicsenv` (conda) | FEniCS FEM solvers | `conda activate fenicsenv` |
| `.venv` (uv, JAX) | PINN training | `source .venv/bin/activate` |

**Never activate both simultaneously.**  Deactivate one before activating
the other.  JAX GPU operations and FEniCS MPI conflict.

---

## 8. Pre-commit Checklist

Before committing any change, verify:

1. [ ] All config fields are explicitly defined (no `.get()` calls)
2. [ ] No silent fallbacks — errors are raised explicitly
3. [ ] Mathematical formulas match published references
4. [ ] Equinox static fields are correctly marked (`static=True`)
5. [ ] No JIT recompile risks from variable shapes or mutated statics
6. [ ] New examples follow the standard file structure (§2.1)
7. [ ] README.md is updated if file structure or API changes
8. [ ] No leftover debugging `print()` or commented-out dead code
