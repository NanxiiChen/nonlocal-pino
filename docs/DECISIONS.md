# Decisions Log

> Append-only. Newest entries at the bottom. Never edit past entries — if a
> decision is reversed, add a new entry that references the old one.

<!--
Template for each entry:

## YYYY-MM-DD — <short title>
- Decision: <what was decided>
- Why: <reasoning>
- Alternatives rejected: <what else was considered, why not>
- Related: <task file / PR / issue if applicable>
-->

## 2026-08-16 — Adopt repository memory protocol
- Decision: Use `PROGRESS.md` as the short project dashboard, `DECISIONS.md` as
  an append-only technical decision log, and one `docs/tasks/<task-id>.md` file
  per active subtask; archive completed task files.
- Why: The project now has multiple numerical branches and unresolved 2D
  formulation choices that cannot be safely reconstructed from chat history.
- Alternatives rejected: one growing changelog, a single large task document,
  and relying on long-lived conversation context.
- Related: `docs/tasks/memory-protocol.md`, `.opencode/command/task-start.md`,
  `.opencode/command/task-end.md`.

## 2026-08-16 — Prevent silent task switching
- Decision: A new task may not replace a different active task named in
  `PROGRESS.md`; the agent must stop and ask whether to resume, close, or
  explicitly switch tasks.
- Why: Overwriting the dashboard link would orphan the previous resume point
  and make parallel or interrupted work indistinguishable.
- Alternatives rejected: silently replacing the active link and relying on
  filesystem scans to reconstruct task ownership.
- Related: `AGENTS.md`, `.opencode/command/task-start.md`.

## 2026-08-16 — Layer agent policy and engineering references
- Decision: Keep `AGENTS.md` as a concise mandatory policy and index; move
  detailed development conventions, numerical-method rules, and JAX guidance
  into focused documents under `docs/`.
- Why: Stable rules and frequently changing implementation guidance have
  different maintenance lifecycles; mixing them makes the agent contract hard
  to audit and causes stale instructions.
- Alternatives rejected: keeping one large `AGENTS.md`, or splitting every
  subsection into a separate document with no clear index.
- Related: `docs/tasks/agents-refactor.md`, `docs/DEVELOPMENT.md`,
  `docs/NUMERICAL_METHODS.md`, `docs/JAX_GUIDELINES.md`.
