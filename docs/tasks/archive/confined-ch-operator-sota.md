# Promote the confined CH geometry-operator SOTA

Date: 2026-10-31. Status: COMPLETE. No Git history authorization.

## Goal and acceptance

The user accepts the completed width256 online cold50k result and requests it
as this geometry-operator case's SOTA. Make the canonical opt-in operator preset
select that exact numerical configuration, document the measured final model,
and update the artifact index. Preserve the separate single-IVP window default,
historical physical configurations, direct artifact paths and checkpoint replay.

Acceptance: canonical operator config matches the saved winner numerically
(apart from runtime provenance); explicit old mild pilot remains available;
configuration inheritance has no cycles or changed historical controls; README,
development guide and artifact metadata agree. Apply proportionate config/entry
checks and git diff --check. No new training, cleanup or Git history operations.

## Context

Completed run:
/root/autodl-tmp/tf-logs/confined_ch2d/20261031-confined-width256-online-cold50k-01-train.
Selected model is model_final.eqx at the prescribed50000 updates, not the lower
49500 development score. Mean native phase L2 is4.8329%, terminal5.1935% across
the same8 reserved development geometries. Independent CPU audit passes. Native
reference only; local FEM remains deferred. See the completed cold50k task and
confined-ch-width256-online-results.json for full evidence and physical fields.

## Changes

- Root README updated first to list the accepted geometry operator alongside
  melting/electropolish operator defaults. The example README now leads its
  operator section with the complete width256/online50k recipe, direct final
  checkpoint, measured runtime/memory, native phi/mu metrics and validation scope.
- configs/train_operator.py aliases the measured online-width256 cold50k config.
  Its numerical fields match the saved run exactly after removing only runtime
  train.seed/log.run_name. No continuation, tuned seed or checkpoint loading is
  introduced into a fresh training launch.
- Original mild config preserved as train_operator_mild.py. The historical
  longer-control inheritance, reference CLI default and two existing test
  fixtures now import it. All37derived research configurations remain exactly
  equal to their pre-edit values, including the explicit online winner. The old
  smallscale10_best docstring identifies it as the historical FP64 width96 recipe;
  its numerical values remain unchanged.
- artifacts/index.yaml adds operator_default with canonical/explicit configs,
  accepted model/config/SOAP direct paths, metrics and scope; the measured winner
  is marked promoted. Old pilot metadata names its preserved mild preset and
  retains its launch-time config path. Existing completion receipts and old
  reports remain historical; no external artifact is renamed, moved or deleted.
- DEVELOPMENT explains the new canonical alias and preserved inheritance.
  PROGRESS and DECISIONS record this promotion separately from the training task.

## Validation / task-end

Receipt: docs/tasks/confined-ch-operator-sota-validation.json.
Configuration-only comparison passes for all37historical derivatives, and the
preserved mild pilot exactly matches the original canonical values. New canonical
config exactly matches saved config apart from runtime provenance; physical/
online schema validates. Actual train.py flags parse the new operator config
and dispatch to train_operator with runtime seed42. Parsing without --config
still selects the separate T1.5single-IVP windows. Reference CLI points explicitly
to the unchanged mild numerical defaults. Changed Python syntax, artifact JSON,
SOTA metric equality and all required checkpoint/config/gallery paths pass.

Three existing focused CPU unittests PASS26.288s: PDE gradient/CLI dispatch,
geometry-independent initial derivatives/prior, and immutable bank reuse. No new
test or expensive training/FEM suite is added. git diff --check passes. Model,
optimizer, data/reference and generated media remain external; no Git history
operation is performed.

## Status, blocker and next action

COMPLETE, no blocker or required resume. Recommended operator command:

```bash
python -m examples.confined_ch2d.train --config=examples/confined_ch2d/configs/train_operator.py
```

This task does not launch another run. Single-IVP30-window defaults and numerical
training implementation are unchanged. Memory files updated: PROGRESS.md,
DECISIONS.md, DEVELOPMENT.md and this archived task.
