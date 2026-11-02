# Select fixed residual normalization as the channel default

## Goal and acceptance

User chooses fixed-weight residual normalization after the matched nine-term
adaptive comparison. Make the actual train.py default select the validated
stronger-channel fixed-scale recipe, retain explicit adaptive/legacy presets,
and align stable README/default guidance. Verify resolved configuration against
the saved fixed-scale run, without new training/FEM or Git operations.

## Context

- Prior task: [nine-term comparison](channel-allterms-gradnorm.md).
- Fixed control: /root/autodl-tmp/tf-logs/channel_nsch2d/20261102-5090-gmls-time40-jvp-stronger-online-width128-01.
- Before this change, train.py defaulted to early configs/train_operator.py(width96,
  Nt4, shared chemical scale4). Many explicit research presets inherit that
  factory, so do not replace it with a circular alias to its own descendant.
- Use a separate train_default wrapper selecting the validated stronger recipe
  and a fresh automatic run name. Keep named historical configurations exact.
- Earlier uncommitted adaptive implementation/results are preserved.

## Changes

- Added configs/train_default.py selecting the saved stronger-channel recipe,
  with an empty run name for fresh automatic naming. Updated train.py's actual
  Abseil default to this file; historical configuration sources are unchanged.
- Fixed residual divisors: CH .2, chemical relation 1, momentum 1, phase
  Neumann 1, chemical Neumann 4, traction 1. Output mu amplitude remains 4.
  No adaptive balancing, causal weighting, stagger or phase transformation.
- Updated the root/example READMEs and docs/DEVELOPMENT.md to distinguish the
  actual default from explicit legacy/adaptive presets and matching FEM inputs.
- Updated artifacts/index.yaml default pointer/selection metadata; kept its
  saved best-run path and all historical entries.
- Recorded configuration checks in
  [the receipt](../channel-fixed-normalization-default-validation.json).

## Verification

- Actual Abseil default resolves and passes validate_config without training.
- Exact JSON parity with the saved fixed-scale run after excluding runtime
  train.seed provenance and using a fresh blank log.run_name; fixed divisors
  and absent adaptive options checked explicitly.
- All 51 configuration factories import and serialize. All 51 historical
  configuration source files match the prior saved source byte-for-byte.
- Existing FEM reference/validation and geometry manifest files are present.
- Changed Python syntax, JSON metadata and git diff --check pass.
- No new training, FEM, checkpoint modification or Git history operation.

## Status, blocker, next action

Complete. No blocker or remaining implementation step. Default selection uses
the existing nominal-geometry result; population/cross-seed accuracy remains
unverified. No further experiments are queued.
