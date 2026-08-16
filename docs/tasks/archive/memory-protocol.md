# Task: Standardize Agent Memory Protocol

## Goal

Make repository memory files and OpenCode task commands the reliable source of
project state across sessions.

## Acceptance Criteria

- `AGENTS.md` contains the mandatory session-start, during-task, and session-end protocol.
- `task-start` resumes an existing task file instead of overwriting it.
- `task-end` records decisions, updates progress, and archives only completed tasks.
- `PROGRESS.md` is a short current dashboard, not a changelog.
- `DECISIONS.md` remains append-only with dated entries.

## Relevant Context

- The 1D fracture example is an accepted engineering baseline with float32 as
  the default and float64 artifacts retained as reference runs.
- The next technical direction is a 2D notched fracture example; continuum,
  quadrature, notch, and mechanics choices are not yet frozen.
- `docs/DECISIONS.md` currently contains only its template.

## Status

- Complete: integrated the protocol into `AGENTS.md`, synchronized both
  OpenCode commands, and initialized the project dashboard and decision log.
- Verification: `git diff --check` passed; no application code was changed.
