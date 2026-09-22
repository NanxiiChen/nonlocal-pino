# AGENTS.md — AI Agent Instructions for nonlocal-KKS-PINN

This file is the mandatory policy and navigation index for coding agents in
this repository. Detailed, changing guidance lives in the linked documents.

## 1. Scope And Required Reading

This project develops nonlocal PINNs for phase-field and solid-mechanics
problems. The repository README is the project overview; update it first when
its file structure or public API is stale.

Before implementation, read the relevant reference documents:

- [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) — file layout, APIs, naming,
  configuration, and verification conventions.
- [`docs/NUMERICAL_METHODS.md`](docs/NUMERICAL_METHODS.md) — mathematical
  rigor, formulation choices, references, and validation order.
- [`docs/JAX_GUIDELINES.md`](docs/JAX_GUIDELINES.md) — Equinox, JIT, fixed-shape,
  sampling, and functional-style rules.

The main examples are `icemelting3d`, `electropolish2d`, `fracture1d`,
`fracture2d`, `elasticity2d`, and `pino_elasticity2d`.

### Documentation Roles

- Example `README.md` files contain stable user-facing problem definitions,
  formulations, configuration, usage, and validation guidance.
- `docs/PROGRESS.md` is the short project-wide dashboard; keep details in the
  linked active task rather than duplicating results here.
- `docs/DECISIONS.md` is the append-only rationale/history log; do not use it as
  a current-results report.
- `docs/tasks/<task-id>.md` is the self-contained active-work record for status,
  validation outputs, blockers, and the exact resume point. Archive it when the
  task is complete.
- Do not add example-level `STATUS.md` files that duplicate an active task.
  Merge run-specific validation into the task file and keep example READMEs
  stable.

## 2. Agent Memory Protocol

Use repository memory files instead of relying on chat history. This applies to
implementation, investigation, and planning sessions.

### Start

1. Read `docs/PROGRESS.md` in full and the last five entries of
   `docs/DECISIONS.md`, or all entries if fewer exist.
2. Resume the current task file if it exists; otherwise create one under
   `docs/tasks/<task-id>.md` with goal, acceptance criteria, relevant context,
   status, blocker, and next action.
3. If `PROGRESS.md` names a different active task, stop and ask whether to
   resume, close, or explicitly switch. Never silently replace it.
4. Set exactly one active task in `PROGRESS.md` and restate the current state
   and plan to the user in 3-5 bullets before code or detailed planning.

### During And End

- Append non-trivial decisions to `docs/DECISIONS.md` immediately. Entries are
  dated and include the decision, reason, rejected alternatives, and relation.
- Keep the active task file self-contained and update its status, blocker, and
  next action as they change.
- Before finishing, run `/task-end`: record diff-level changes, archive a
  completed task under `docs/tasks/archive/`, or leave an incomplete task with
  an exact resume point; then update `PROGRESS.md` and output a short session
  summary listing changed memory files.

The `/task-start` and `/task-end` commands are operational templates; this
section is authoritative if they differ.

## 3. Non-Negotiable Technical Rules

- Do not guess mathematics. If a PDE, constitutive law, kernel normalisation,
  boundary treatment, nondimensionalisation, or adaptive method is uncertain,
  stop and ask rather than inventing it.
- Keep formulations traceable to references or an independently checked
  derivation. Do not silently correct inconsistencies between code and docs.
- Required configuration fields are explicit. No `.get()` defaults, hidden
  required-argument defaults, swallowed exceptions, or silent numerical
  fallbacks.
- Validate physical ranges, shapes, ordering, operator assumptions, and input
  files early; fail loudly with useful context.
- Preserve fixed array shapes and dtypes in JIT paths, and keep operators,
  grids, and fixed BC data precomputed.
- Preserve pure functional network calls and avoid mutation of Equinox fields.

See [`docs/NUMERICAL_METHODS.md`](docs/NUMERICAL_METHODS.md) and
[`docs/JAX_GUIDELINES.md`](docs/JAX_GUIDELINES.md) for the detailed rules.

## 4. Git Authorization And Workflow

Never commit, merge, push, tag, stash, rebase, reset, clean, or otherwise alter
Git history unless the user explicitly authorizes that operation in the
current turn. Read-only commands such as `git status`, `git diff`, `git log`,
and `git branch` do not require authorization.

When authorized, use this repository workflow:

```text
dev -> commit -> main -> git merge dev -> git push origin main
```

Use a regular merge, not squash. Commit messages end with:
`Co-Authored-By: Claude <noreply@anthropic.com>`. Push only `main`; keep `dev`
local.

## 5. Environments

Use mutually exclusive environments:

| Environment | Purpose | Activation |
|---|---|---|
| `fenicsenv` | FEniCS/FEM solvers | `conda activate fenicsenv` |
| `.venv` | JAX/PINN code | `source .venv/bin/activate` |

Never run FEniCS MPI and JAX GPU work concurrently or with both environments
activated.

## 6. Response And Completion Rules

For Markdown intended for OpenCode Web, use inline `\(...)` and display `$$`
delimiters on their own lines; do not use `$...$` or `\[...\]`.

Before completion, apply proportionate verification:

- documentation, agent-rule, and configuration-only changes: inspect the diff,
  validate syntax or schema when applicable, and run `git diff --check`; do not
  run code tests or compilation unless behavior can be affected;
- localized code changes: run the smallest focused test command that exercises
  the changed behavior; compile only changed Python modules when useful;
- broad, numerical-core, public-API, dependency, or release changes: expand to
  affected integration or full-suite checks when their added coverage justifies
  the cost;
- do not rerun an unchanged expensive suite in the same session unless earlier
  results are invalidated or the user requests it;
- `git diff --check` passes;
- configs and numerical choices are explicit and documented;
- README/docs match changed APIs or formulations;
- no debugging code or dead commented code remains.
