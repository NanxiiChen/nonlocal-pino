---
description: Close out the current subtask and sync memory files
---

Do the following, in order, before ending this session:

1. Summarize what was actually done this session (diff-level: files changed,
   behavior changed — not a conversational recap) in the active task file.
2. If any non-trivial decision made this session isn't already in
   `docs/DECISIONS.md`, append it now (with date, reasoning, alternatives rejected).
3. Determine if the subtask is complete:
   - If YES: update the task file with the final verification, move it from
     `docs/tasks/<task-id>.md` to `docs/tasks/archive/<task-id>.md`, and add a
     one-line entry to the "Recently completed" section of `docs/PROGRESS.md`.
   - If NO: leave it under `docs/tasks/` with the exact resume point, current
     blocker, and next concrete action, written for someone with zero memory
     of this conversation.
4. Update `docs/PROGRESS.md` "Current focus" and "Next up" accordingly. There
   must be exactly one active task file when work remains, and none when the
   task is complete.
5. Do not commit, merge, push, or otherwise modify Git history unless the user
   explicitly authorized that operation in the current turn.
6. Output a final "Session summary" block listing exactly what changed in the
   memory files, so I can verify it before closing.
