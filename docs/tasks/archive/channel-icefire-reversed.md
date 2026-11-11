# Reversed icefire and channel time labels

## Goal and acceptance criteria

User requests icefire_r and asks why the earlier channel figure did not use
nondimensional time. Render a separate reversed-palette preview, preserve the
accepted icefire figure, and label snapshots explicitly by t*=t/T. Keep the same
fields, errors, equal aspect, 170 mm width and editable 5–7 pt PDF typography.

## Context

The prior preview is complete: `archive/channel-main-figure.md`. Source model,
reference cohort and render specification are in `artifacts/channel_main_figure.json`.
`network.py::temporal_features` already uses times/terminal_time; time JVP
includes this chain rule. PDE and local FEM use original model-time coordinates,
T=6. README presents the dimensional-form NS–CH equations, but no physical
seconds calibration is established. Normalized plot time t/T is distinct from
a complete nondimensionalization of all PDE coefficients. No PDE/weight change.

## Status

Completed 2026-11-12: icefire_r and explicit t* headings rendered and verified.

## Blocker

None.

## Next action

None; preview delivered for user review. No Git operation.

## Changes and reproduction

- `scripts/render_channel_main_figure.py` accepts the exact reversed palette,
  validates the display-time convention and draws editable t* headings with
  7 pt sans-serif base text and a 5 pt superscript.
- `artifacts/channel_main_figure_icefire_r.json` records unchanged source fields
  and model/reference paths, the reversed palette and t*=t/T, T=6.
- `artifacts/channel_main_figure_icefire_r_validation.json` records scores,
  explicit display-time mapping, fonts, layers and independent checks.
- PROGRESS and DECISIONS record this completed revision; old icefire outputs
  and their specification/receipt remain available.

```bash
.venv/bin/python -m scripts.render_channel_main_figure \
  --spec artifacts/channel_main_figure_icefire_r.json
```

Outputs in `/root/autodl-tmp/tf-logs/channel_nsch2d/publication-main-20261112`:
`two_phase_flow2d_main_phase_icefire_r.pdf` (778836 bytes) and matching PNG
(300054 bytes). 170 x 59 mm, same 4 samples and 5 times, white dashed interfaces.

## Validation

All 20 scores and their complete source records exactly match the icefire
receipt. Five normalized labels match saved times divided by 6. Independent
PDF inspection confirms dimensions, all 5–7 pt text, embedded TrueType fonts,
no clipped text, 21 raster layers and identical vector wall/frame/interface
coordinates. PNG and rasterized PDF inspected. JSON/AST and diff checks pass.
No training/inference/solve, numerical defaults, README image or Git changes.
