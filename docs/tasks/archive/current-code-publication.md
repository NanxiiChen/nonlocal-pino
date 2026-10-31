# Publish current neural-operator code and research records

Date2026-10-31. User explicitly authorizes commit and push of current code and
records. Follow localdev commit -> main regularmerge -> origin/main only.

## Goal and acceptance

Publish all accumulated confined-CH geometry-operator implementation, online
rolling/SOTA configurations and validation records, plus electropolish/melting
spatial-sampling controls and known replay/query-consistency issues. Keep large
generated media/arrays/checkpoints outside Git; preserve existing external paths
and recorded failed numerical gates. Do not retrain, tune defaults, rewrite
history, forcepush, pushdev or delete local artifacts.

Acceptance: inspect pending files, syntax/strictJSON and staged content/whitespace
checks, reuse existing unchanged numerical validations, commit ondev, ordinary
merge tomain and push onlyorigin/main, verify exact remote/local ref equality
and clean worktree. Publish a final accurate memory/receipt update afterward.

## Context / current state

Start onmain4f02885=origin/main; localdev651cc63has the same tree, ancestor
ofmain.335pending source/Markdown/JSON/text files,3101737bytes (~2.96MiB),
largest381938byte artifact index. No generated binary/media candidate. Existing
ignore/content guard excludes generated outputs and files>1MiB. Current SOTA
and experiment results are already documented and verified; publication changes
no numerical behavior. Recent supplemental failed gates remain open issues.

## Status / blocker / next action

COMPLETE. Current code/records published onorigin/main and exact remote equality
verified. Codecommit ae467463399994edd36d9b0f163a6c2c7f5f3a88(localdev), regular
mainmerge c359d7f80f39194e1eb66b6719f4b16a8d81ba07, with prior main4f02885as
first parent. No remotedev branch pushed. Trees main/dev exact, clean worktree
at publication. This final accurate memory/receipt update follows the same
devcommit -> regularmainmerge -> origin/main workflow; the containing final
commit/merge IDs are recorded in Git and the user report, not self-referenced.
No blocker, remaining numerical work or queued training.

## Verification and published scope

-337textual changed files,55048insertions/44deletions in the codecommit; no
  binary changes. Includes point-conditioned geometry operators, deterministic
  bounded online refresh, capacity/precision/SOTA controls and spatial ablations.
-148Python AST modules,156strictJSON records before the additional publication
  receipt, and45configuration factories import/serialize successfully. Existing
  completed numerical suites/cold-training/reference/checkpoint evidence reused;
  no retraining or unchanged expensive suite repeated.
-1851indexedfiles/20.83MiB after publication receipt, no generated report payload
  or file>1MiB. Existing weights/figures/arrays stay at original external paths.
-Staged and worktree whitespace/content checks pass. Two whitespace-only fixes:
  docs/tasks/archive/confined-ch-width256-online-cold50k.md line trailing space
  and src/nonlocal_pinn/nn/point_conditional.py extra EOFnewline. Text tokens and
  Python AST exact; no numerical semantics changed by these formatting edits.
-Previously failed electropolish CPU replay and melting large-query invariance
  gates remain explicit issues/receipts. Publication doesnot relabel them passed.
-No local artifact deletion, default tuning, forced update, rebase, squash or
  history rewrite. Only main is pushed; dev remains local.

## Task-end memory

Updated docs/PROGRESS.md, docs/DECISIONS.md and this archived task. Compact
current-code-publication-validation.json records checks and published code refs.
No runtime/configuration change in this publication task beyond formatting.
