# Melting rolling-pool default

## Goal and acceptance

The user selects the rolling recipe as Melting's default and requests the
related README updates. Promote the measured cold10k recipe and preserve the
historical 5k controls without a circular config dependency. Update the
example/project docs and report sources with the selected checkpoint and
finite-cohort/endpoint qualifications. Verify saved configuration parity,
old presets, source links, checkpoint loading and affected regression/build.

## Status

Implementation and validation complete. The user explicitly authorized commit
and push in the final continuation request. No training is pending.

## Changes

- Canonical `examples/icemelting3d/configs/train_operator.py` selects Adam
  10,000, B16, Nt8, width48, FP32/highest and rolling K256/32 initial sets,
  with 1792-node/1536-edge buffers and one worker/32 prefetch entries.
- Freeze the old canonical factory in `train_operator_5k.py` and redirect the
  fixed control to it. All nine other existing operator presets retain their
  previous resolved values. The single-sphere PINN default is unchanged.
- Update the example/root README, artifact index and report sources with the
  selected endpoint, historical controls and finite-cohort qualifications.
  Keep the existing 3D figure explicitly labeled as the historical fixed-pool
  5,000-step model. No new dependency or rendering run was needed.
- Include the completed sampling study's optional channel compact-row and
  ceramic wall-stratum code, controlled presets, focused test and reproducible
  research scripts, plus the earlier Electropolish rolling-default promotion.
  Channel full-grid and ceramic area-sampling defaults remain selected.

## Selected result and provenance

The final model is
`/root/autodl-tmp/tf-logs/icemelting3d/melting-pools-cold10k-20261111-01-rolling_pool/model_final.eqx`.
Mean phase relative L2 on eight reused native-reference inputs is 0.1002%
pooled and 0.1711% terminal, evaluated at 24^3 query nodes over five times.
Fixed gives 0.1203%/0.1132%; five late checkpoint medians favor fixed.
The selected rolling default follows the user's choice, with endpoint,
cohort and ordinary-GPU repeatability limits preserved. Local-FEM results
and the unchanged 3D illustration concern the separate historical 5k model.

## Validation

- New canonical numerical config equals the selected saved config except
  runtime seed, run name and the optional research snapshot cadence. The
  saved model loads into the current template, with FP32 floating leaves.
- All nine historical presets and the frozen original canonical factory
  preserve the pre-promotion configuration snapshot.
- Six Melting CPU regression tests pass: two spatial sampling and four
  operator tests. Earlier unchanged study checks retain their recorded
  thirteen geometry and ten Electropolish focused-test results.
- All 27 changed/new Python files parse; modified JSON and the JSON-compatible
  artifact index parse. A math-aware Markdown parser validates 292 local
  file/image targets. The report builds nine pages with 245 local HTML targets.
- Six curated PNGs total 2,268,852 bytes, each below 1 MiB. Only the selected
  Electropolish image changes. No raw fields, logs or weights enter Git.
- `git diff --check` passes.

## Task-end and release

No implementation blocker remains. Archive this task and clear the active
implementation entry. Publish the reviewed changes using local dev commit,
regular merge to main and push of main only, then inspect the existing Pages
workflow. This session updates `docs/PROGRESS.md`, `docs/DECISIONS.md` and this
archive; earlier completed study records are included in the same release.
