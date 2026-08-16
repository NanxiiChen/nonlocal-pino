# Task: Refactor Agent Instructions

## Goal

Reduce `AGENTS.md` to stable mandatory policy and move detailed engineering
guidance into focused, linked reference documents.

## Acceptance Criteria

- `AGENTS.md` is a concise policy/index file with no duplicated tutorials.
- Detailed example conventions, numerical rigor, and JAX guidance are preserved
  in focused documents under `docs/`.
- The memory protocol, Git authorization rules, environment rules, and critical
  no-guessing/fixed-shape requirements remain discoverable from `AGENTS.md`.
- All extracted documents are referenced from `AGENTS.md`.
- Markdown has no whitespace errors and no application code changes occur.

## Relevant Context

- The repository currently has a 1D fracture baseline and is planning a 2D
  notched fracture example.
- `AGENTS.md` currently mixes project background, memory lifecycle, API style,
  numerical policy, JAX implementation details, Git workflow, and checklists.
- `docs/PROGRESS.md` has no active task before this refactor.

## Status

- Complete: reduced `AGENTS.md` to mandatory policy and linked the detailed
  development, numerical-method, and JAX reference documents.
- Verification: `AGENTS.md` is 114 lines; `git diff --check` passed; no
  application code changed.
