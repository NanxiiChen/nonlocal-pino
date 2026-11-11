# Two-phase-flow main-figure preview

## Goal and acceptance criteria

User requests four samples/five times, phase-only prediction with reference
interfaces, following the previous publication galleries. Choose a suitable
colormap; retain170mmwidth, equal spatial scale, scientific instantaneous
phase-error annotations, one centered shared colorbar and editable5–7pttext.
Record exact checkpoint, source/reference data, sample roles and actual times.

## Context

Full NS–CH channel is `channel_nsch2d`, not prescribed-flow `twophase2d`.
Selected20kcheckpoint uses three predeclared reserved cases plus one nominal
development case. These four are available without a new reference solve;
do not describe all four as held out. References are local mixed FEM. Full
physical4x1channel must not be stretched/cropped to enlarge the droplet.
No training/default changes, README promotion or Git authorization.

## Status

Completed preview, 2026-11-12. User-selected icefire replaces the initial PuOr
preview. No numerical solve, inference, training, default change, curated
README-image replacement or Git operation.

## Data and layout

- Exact source paths, model/configuration, four sample roles and 20 raw scores
  are recorded in `artifacts/channel_main_figure.json` and its validation file.
- Selected final 20000-step model, runtime seed 42; one development geometry
  followed by all three predeclared reserved geometries, not four blind cases.
- Independent spatial/time-refined local mixed FEM/BDF2 reference.
- Actual saved times: 0, 1.5, 3, 4.5, 6. No temporal interpolation.
- Original mapped 96 x 40 physical queries; phase channel only. Instantaneous
  relative L2 uses equally weighted original query values, not displayed pixels.
- Exact Seaborn 0.13.2 icefire table, shared [-1.2, 1.2] scale containing the
  raw phase overshoots. White dashed reference zero contours for contrast.
- Full 4 x 1 channel at equal aspect; each panel is 26.8 x 6.7 mm. Gray solid
  walls follow the analytic geometry. Gouraud rendering is display-only.
- 170 x 59 mm PDF, 2007 x 696 PNG. One vertically centered colorbar with its
  label above; scientific error labels, no example title or interface legend.
- Editable embedded Helvetica and sans-serif italic L, all text 5–7 pt.
  Physical fields and colorbar gradient are raster; walls, contours and text
  remain vector.

## Changes and reproduction

- `scripts/render_channel_main_figure.py`: CPU-only saved-array rendering and
  source/model/configuration/query/time validation.
- `scripts/colormaps/icefire.json`: all 256 RGB entries and the original BSD
  license from the Seaborn 0.13.2 wheel; no package installation or new runtime
  dependency. Matplotlib loads the explicit palette file.
- `artifacts/channel_main_figure.json`: direct input/checkpoint paths, roles,
  metric and complete style/output specification.
- `artifacts/channel_main_figure_validation.json`: scores, geometry, font,
  layout and independent verification receipt.
- `docs/DECISIONS.md` records palette, reference and layout choices;
  `docs/PROGRESS.md` marks this preview completed.

```bash
.venv/bin/python -m scripts.render_channel_main_figure \
  --spec artifacts/channel_main_figure.json
```

Output directory:
`/root/autodl-tmp/tf-logs/channel_nsch2d/publication-main-20261112`.
Files: `two_phase_flow2d_main_phase.pdf` (789396 bytes) and matching `.png`
(305861 bytes). The prior PuOr outputs/specification/receipt are preserved in
`puor-preview/` beneath that directory. Repository curated images are unchanged.

## Validation

- All 20 instantaneous norms independently recomputed with Python math.fsum;
  scores exactly match the prior PuOr preview. Four pooled source metrics agree.
- All 256 palette RGB entries exactly match the source wheel; 40 solid-wall
  pixels are gray. PNG and independently rasterized PDF visually inspected.
- Actual PDF measures 170 x 59 mm; 5/6/7 pt dark text with embedded TrueType,
  no Type3 or clipped labels. Temporary text edits passed for a sample label
  and a scientific error annotation.
- 21 raster layers match 20 physical fields plus colorbar; 80 panel frame edges
  confirm equal physical scaling. White dashed reference paths remain vector.
- All text strings/positions/sizes and all vector path coordinates exactly
  match the initial preview. Only palette and reference line color changed.
- JSON/Python syntax, referenced-file existence and whitespace/diff checks pass.

## Blocker and next action

None. Preview delivered for user review; no further work scheduled.
