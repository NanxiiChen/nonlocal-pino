# Split nonlocal PINN and fracture PINN projects

## Goal and authorization

First preserve current code/docs in a dev commit. Then propose two independent
research projects, their package/example/docs/test boundaries, and external
reference-data/log/checkpoint ownership. User must confirm before execution.
No repository creation, file migration, numerical change, merge or push now.

## Acceptance criteria

- Existing tracked and new code/docs/tests are committed with focused validation.
- Proposal is grounded in actual imports, example formulations and artifact paths.
- Both target structures, shared-code policy, history retention, artifact strategy,
  staged migration and verification criteria are reviewable.
- No split is implemented before confirmation.

## Context

User explicitly switched from nonlocal-pinn-architecture on 2026-10-24;
that task is suspended and incomplete. Current checkout is dev. External
references live under /root/autodl-tmp/data/<example>; tf-logs/<example>
contains events, checkpoints, results and diagnostic/launch scripts as well.
Nonlocal fracture controls remain in fracture1d and fracture2d; shear/L-panel
use local FE-energy history-conditioned neural damage with local mechanics.

## Status

Pre-split snapshot ready: four irreversibility CPU regressions PASS (30.505s).
Actual dependency/artifact inspection underway; architecture not adopted.

## Blocker

No blocker to proposal. Execution is intentionally gated by user confirmation.

## Next action

Commit the current snapshot on dev; finish concrete proposal and present it.
