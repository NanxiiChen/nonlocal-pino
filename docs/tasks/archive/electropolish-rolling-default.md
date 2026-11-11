# Electropolish rolling-pool default

## Goal and acceptance

Apply the user's requested promotion of the successful rolling spatial recipe.
Update the example and root README, selected checkpoint/representative image
and report sources. Preserve historical preset values and checkpoint loading.
Validate exact default parity against the measured run, archived preset parity,
spatial sampling regression, image provenance, site build and whitespace.
No Git operation or public deployment was requested in this turn.

## Selected recipe

Current canonical configs/train_operator.py inherits the historical rolling
pool controls and sets width96/Nt16/Adam10000. Spatial settings are K256,
32initial sets,3328node/3072edgecapacity,oneCPUworker/32prefetch.
Runtime seed and run names are excluded from strategy settings.
The selected final checkpoint is under
/root/autodl-tmp/tf-logs/electropolish2d/pino-spatial-pools-20261111-01-electropolish2d-s42-rolling_pool.

Against the same64localFD101cases the selected phase pooled/terminal errors
are1.546571/1.325638%,concentration1.860404/1.580444%. The second paired seed
also favors rolling. Both pairs and same-device RTX4090times are documented.
The figure shows the same predeclared four cases0/21/42/63from that model.
The earlier fixed10kand historical5kmetadata are retained as historical evidence.

## Status

Completed. Default, READMEs, report source, selected metadata and image all
match the measured rolling-pool recipe. Final validation passed.

## Blocker

None.

## Next action

None for this promotion.

## Verification and task-end

The default equals the saved measured recipe after removing only runtime seed
and run name. The final checkpoint deserializes into the current template.
All13historical presets retain their pre-edit values. Four spatial sampling
and six operator regressions pass on CPU. No new training or reference solve
was needed.

The renderer checks the selected checkpoint, case identities, saved times,
coordinates and FD101phase errors. The replacement2025x1095PNG is406448bytes;
all six curated PNGs total2268852bytes. The other five PNGs and their provenance
records are byte-identical. The nine-page report builds locally, uses the new
metrics and copies the image byte-for-byte. All133checked local document links,
artifact JSON and whitespace checks pass.

Diff-level changes are the canonical preset parent, example/root README,
report index/case/training text, selected artifact pointer with history,
renderer/PNG/provenance and the development guide. Previous experiment source
and records remain separate workspace changes. No Git commit/push or public
Pages deployment was part of this request.

Memory files are docs/PROGRESS.md, docs/DECISIONS.md and this archived task.
The completed spatial study is archived alongside it, with optional regional
and Spinodal diagnostics explicitly deferred in PROGRESS.
