# Corrosion robust-objective optimization — 2026-10-26

## Goal and acceptance

First preserve the user-authorized current working tree in a local dev commit.
Then test a targeted remedy for corrosion's seed-sensitive bounded objective,
without selecting a seed, changing physics/reference, adding hard ICs, or
increasing the5000-update budget. Compare terminal/late-window phi/c errors
and independent concentration/transport diagnostics. Retain negative results;
promote defaults only with cross-seed evidence.

## Context and plan

See archive/phasefield-seed-mechanism.md and phasefield-seed-stability.md.
Corrosion bounded42 succeeds, bounded7/2026 fail despite full nodal IC/BC.
Same-initial-state original7 succeeds. Differentiating the bounded multiplier
adds |r| grad(w), a testable objective-conditioning difference.

1. Review/validate and commit existing changes locally on dev; no push/merge.
2. Add an explicit frozen-weight bounded control with unchanged forward
   residual; preserve legacy formulations and fixed JIT shapes.
3. Cold-start original2026 and frozen-weight bounded7/2026; retain existing
   original7 and differentiated controls, all5000 updates/lr1e-3/50s.
4. Audit full fields/transport and update docs/defaults as evidence allows.

## Status, blocker, next action

Active, no blocker. Reviewed19 tracked and204 untracked files (~13.6MB,
none >5MB), no external checkpoints; git diff --check passes. dev/main trees
agree. Next: snapshot commit, then implement and validate controls.
