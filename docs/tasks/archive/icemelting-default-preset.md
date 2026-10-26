# Validated icemelting default preset

## Goal and acceptance criteria

Make the best configuration from the matched 5 s SOAP/Adam comparison the
canonical production default. Check all training parameters against the selected
SOAP run, preserve the user-requested 5000-update budget, and document direct
checkpoint provenance and the default invocation.

## Context

The selected run is
`/root/autodl-tmp/tf-logs/icemelting3d/20261024-5s-soap-lr1e3-seed42`,
checkpoint `ckpt_005000.eqx`. Its terminal full-grid phase relative L2 against
the same-graph numerical solution is 6.10970108092214e-5. The run was originally
planned for 20000 updates and stopped after the user shortened the budget;
its original saved configuration must remain intact. See
[paired comparison](icemelting-soap-adam-5s.md).

## Status

Complete. The base config already matches every saved training, architecture,
physical, operator and sampling setting, with the requested 5000-update budget.

## Blocker

None.

## Changes and validation

- `examples/icemelting3d/configs/train_base.py`: identify the validated default,
  correct the stale tensor-split comment and remove a duplicate SOAP setting.
  Numerical field values are unchanged in this promotion.
- `examples/icemelting3d/README.md`: specify the canonical default, distinguish
  its 5 s provenance from the historical 4 s tag, and give the explicit optional
  graph-reference monitoring command.
- `artifacts/index.yaml`: add `default_training` with the config, selected
  checkpoint, evaluation reference, metric and comparison report paths.
- Memory: update `docs/DECISIONS.md`, `docs/PROGRESS.md` and this archived task.

Pure configuration comparison checked every field against the saved SOAP
config. Only the planned iteration count (saved20000/current5000), fresh run
name and optional evaluation reference differ. Asserted the selected default
values, config syntax, JSON artifact schema, checkpoint/reference/report paths
and metric consistency with the comparison entry. All passed. Inspected the
diff; `git diff --check` passed. No numerical code or trained artifacts changed;
no additional training or numerical tests were needed.

## Next action

None. The default invocation uses this preset for a fresh run. Optional graph
monitoring is enabled with the documented direct reference-path override.
