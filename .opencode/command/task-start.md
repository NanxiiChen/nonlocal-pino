---
description: Start a new subtask with full memory context loaded
---

Before doing anything else:

1. Read `docs/PROGRESS.md` in full and the last 5 entries of
   `docs/DECISIONS.md` (or all entries if fewer than 5 exist).
2. The subtask for this session is: $ARGUMENTS
   - If empty, ask me what the subtask is before proceeding.
3. Derive a short kebab-case slug. If `docs/tasks/<slug>.md` already exists,
   read it and resume it; do not overwrite it or create a second task file.
   Otherwise create it with:
   - Goal (one sentence)
   - Acceptance criteria
   - Relevant context pulled from PROGRESS.md / DECISIONS.md (only what's actually relevant)
   - Status, blocker, and next action
4. If `docs/PROGRESS.md` names a different active subtask, stop and ask whether
   to resume, close, or explicitly switch tasks. Do not replace its link.
5. Update `docs/PROGRESS.md`: set exactly one "Current focus" to this
   subtask and link the task file.
6. Check for contradictions between the memory files and the user's request;
   flag them instead of silently choosing one.
7. Summarize the current state and plan back to me in 3-5 bullets before
   writing code or making a detailed implementation plan.

Do not begin implementation before steps 1-7 are complete. Keep the task file
terse, self-contained, and updated throughout the task.
