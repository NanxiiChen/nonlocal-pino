# Publication-style violin redraw

## Goal and acceptance criteria

Redraw Electropolish and particle-corrosion final64sample seed42 channel-mean
space-time error violins following `docs/PLOTTING_GUIDELINES.md`. Use Helvetica,
NPG categorical colors, 5–7 pt black/gray text, concise labels and editable
embedded-font PDF plus PNG. All256 sample scores and checkpoints unchanged.

## Context

Source: artifacts/pino_derivative_comparisons.json and its validation receipt;
renderer: scripts/render_pino_derivative_comparisons.py. No new training or
inference required. Existing field sheets are outside this redraw scope.
System font search found only Helvetica AFM metrics, not embeddable glyphs.
User explicitly asks to download Helvetica, declining an offered substitute.
Prior task records include the immutable metric definitions and external data.

## Changes

- Downloaded the notebook project's Helvetica.ttf from
  https://raw.githubusercontent.com/NanxiiChen/PI-CViT/main/helvetica/Helvetica.ttf
  to `/root/.local/share/fonts/pino/Helvetica.ttf`; metadata: Helvetica Regular,
  Version10.0d4e1, TrueType glyf, fsType0,317968bytes. No font committed.
- Updated `scripts/render_pino_derivative_comparisons.py`: explicit font-path
  checks, scoped publication styling, same-figure vector PDF/RGB PNG, artist
  text-size/bounds checks and `--figures distribution` to preserve field sheets.
- Updated the source manifest and rendering receipt with publication settings,
  direct PDF/font paths and unchanged sample values. Added independent PDF
  validation at `artifacts/pino_violin_pdf_validation.json`.
- Replaced only the two curated distribution PNGs. Updated `assets/README.md`
  regeneration/style/format notes; existing example image links display these.
- Updated `docs/DECISIONS.md` and `docs/PROGRESS.md`; archived this task.

## Artifacts and exact command

Output/backup root:
`/root/autodl-tmp/tf-logs/pino-derivative-eval64-20261111-01/publication-violins-20261112`

Two external PDFs: `electropolish2d_distribution.pdf` (22294bytes) and
`particle_corrosion2d_distribution.pdf` (22479bytes). Their300dpi PNGs at the
existing assets/pino-comparisons paths are64881and45793bytes respectively.
Both PDFs are85x65mm; all text6or7pt, Helvetica, color#222222. No raster images.
NPG red nonlocal/cyan AD, same64points/ScottKDE/jitter/medians/IQR and linear axes.
All four curated comparison PNGs total1160968bytes, inside the2MiBbudget.
Previous10presentation/source files are in `before_redraw/` under that root.
Independent audit source is `validate_pdf.py` there, with command in the receipt.
PDF audit dependency PyMuPDF1.26.7 was installed only under `/tmp/pino-pdf-audit`;
no project dependency/lock/environment change.

```bash
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/render_pino_derivative_comparisons.py \
  --case all --figures distribution
```

## Validation

- Both full saved-field/config/reference audits pass onCPU; no inference/solve.
- All256sample records, channel norms/statistics and evaluation grids exactly
  equal the archived receipt; seed42and final10k/20k selections unchanged.
- Two physical-field PNGs byte-identical to backup; original SOTA assets unchanged.
- Actual PDF objects contain embedded TrueType Helvetica, extractable labels,
  only6or7pt spans and dark text, no clipped text and no raster images.
- PyMuPDF edit roundtrip replaces ordinary/numeric labels on temporary copies;
  both edited labels extract correctly. Original PDFs preserved. No claim of
  testing Illustrator/Acrobat GUI or complex formula editing (no formulas here).
- Both PNGs visually inspected; independent PDF raster preview also inspected.
- JSON, renderer static syntax, asset links and `git diff --check` pass.
- No training/default changes, Git operation, deletion or paused-process action.

## Status

Complete. Ready for user visual review; source/font and PDF/PNG paths recorded.

## Blocker

None. Genuine Helvetica resolved from the user-owned notebook project.

## Next action

None pending. Further layout changes can use the same immutable sample scores.
