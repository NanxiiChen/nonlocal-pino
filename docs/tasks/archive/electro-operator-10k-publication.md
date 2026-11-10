# Electropolish 10,000-step default and publication

## Goal and acceptance

Promote the user's selected width-96, 16-time, 10,000-step Electropolish
operator. Update the example's result text and curated phase evolution image,
preserve historical control settings and publish a regular dev-to-main merge.

Verify the default against the saved run configuration, the old presets
against their existing settings, and the figure against the saved predictions
and independent reference. Keep the shared image and its README and website
captions tied to the same checkpoint. Verify the pushed main tip.

## Promoted recipe

`configs/train_operator.py` now selects 16 shared times, width 96 and
10,000 Adam updates. The published checkpoint is the final model from
`/root/autodl-tmp/tf-logs/electropolish2d/20261111-electro-time-audit-s42-t16-w96`.
Its mean pooled and terminal phase errors are 1.740% and 1.540% against
64 independent local FD101 targets. The result covers one cold run.
The [time sampling and reference audit](electro-time-reference-audit.md)
contains the paired budget, seed and reference comparisons.

The historical width-64, four-time, 5,000-step recipe is available as
`configs/train_operator_5k.py`. The matched spatial and time-injection
controls use that preset. The no-argument single-interface PINN uses
`configs/train_base.py`. Run seed and names remain runtime settings.
The optional intermediate checkpoint is disabled in the recommended
preset. The previous experiments' controls and reproducibility records
are included with this release.

## Figure and documentation

The example and root README and the two report pages use the selected
metrics. The existing 2025×1095 PNG shows saved phase for the predeclared
inputs 0, 21, 42 and 63, with dashed local FD101 interfaces. The renderer
checks the checkpoint identity, aligned coordinates and times, case
identities and the four pooled errors. Direct source paths are in the
[figure manifest](../../../artifacts/pino_readme_figures.json).
The replacement image is 404,017 bytes, and the six-image total is
2,266,421 bytes. The release tracks the curated PNG and its provenance.

## Verification

- The recommended resolved configuration equals the saved training
  configuration after clearing run name, runtime seed and optional
  intermediate write. The promoted checkpoint deserializes into that
  template. Twenty-three saved sources match byte-for-byte. The remaining
  differences are the preset promotion, historical preset import and
  source-list update.
- All 12 previous configuration values are preserved under their
  original preset or the explicit 5,000-step replacement. The canonical
  CLI/seed and historical checkpoint parity check passes.
- Ten focused operator and spatial-sampling CPU tests pass. The report
  builds all nine pages and copies the replacement PNG byte-for-byte.
  The built index and Electropolish page both show the promoted metrics.
- The release uses a local dev commit and a regular main merge. The
  pushed main tip is verified against the remote in the release step.
