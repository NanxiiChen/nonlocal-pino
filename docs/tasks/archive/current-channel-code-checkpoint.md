# Save current channel code before further loss changes

## Goal and acceptance

User explicitly requests a local commit of current code before modifications.
Preserve current NS–CH implementation, configurations, tests and research
records. Inspect content/size, validate text syntax and whitespace, and commit
on local dev. No authorization for push, main merge, cleanup or numerical edits.

## Context

Starting branch main at 0dc54bc, with six tracked modifications and243untracked
text files (2344961bytes). Stronger-constriction trial and scheme inventory
are complete. Earlier numerical checks are recorded in their receipts; do not
repeat unchanged expensive training/test suites solely for publication.

## Status and task-end

Checkpoint content verified and archived for the containing local dev commit;
the commit identity is available from Git and the user report, avoiding a
self-referential ID. No blocker or numerical work required for this checkpoint.
No push/main merge, cleanup, runtime behavior change or training started.
GradNorm discussion/implementation is separate follow-up work.

## Scope and verification

-250changed text files: full examples/channel_nsch2d source/configurations,
  tests, diagnostic/regeneration scripts and compact validation/task records.
  Root README now accurately describes nominal-validation/online-training scope
  without claiming unseen-geometry population accuracy. Existing AGENTS and
  development/index updates are retained.
-202Python AST checks,49configuration factory imports/strict serialization,
  23strictJSON files(including JSON-formatted artifacts/index.yaml) pass.
  Reuse the recorded unchanged numerical/reference/training validations; no
  expensive suite, GPU training or FEM computation repeated for this checkpoint.
-Reviewed index:2095files/23.74MiB, no generated report payload or file>1MiB;
  no binary diff. Weights, figures and arrays remain at existing external paths.
-Staged/worktree whitespace checks pass. Main/dev starting trees were identical,
  so switching to existing localdev preserved all pending contents. No history
  rewrite, deletion or remote operation authorized or performed.
-Memory changed: docs/PROGRESS.md, docs/DECISIONS.md and this archived task.
  After the containing commit, verify localdev HEAD and clean worktree; main
  remains at0dc54bc, with no push or merge.
