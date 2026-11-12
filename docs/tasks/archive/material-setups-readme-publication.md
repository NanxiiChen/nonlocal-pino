# Promote five approved material setup figures into documentation

Completed 2026-11-12. The user approved the five refined setup schematics and
requested example README previews plus retained paper PDFs. This was concurrent
documentation work alongside the active fixed-particle horizon-ratio study.

## Delivered

- Five byte-identical PNG/PDF pairs in `assets/pino-setups/`: melting v2,
  Electropolish v4, spinodal v2, channel v1 and ceramic corrosion v3. The approved
  material colors, short arrows and simplified labels are preserved.
- Five example READMEs embed the new PNG and link its PDF, source/style
  specification and collection reproduction instructions. Captions describe
  one prescribed initial state instead of the historical three-input SVGs.
- Root README, asset index, new collection README and development asset policy
  document the separate 2 MiB collection / 1 MiB per-file budget. Legacy SVGs
  remain for historical links with their original budget.
- `artifacts/pino_setup_figures.json` records direct specifications, rendering
  receipts, sizes and retained destinations. The existing generic publication
  copier is reused; only its docstring was generalized.
- Five local website chapters use the same PNG/PDF pairs. The build validates
  the setup manifest, asset paths and byte budgets; Pages watches its manifest.
  This prepares a future release, without committing, pushing or deploying.

No numerical configuration, checkpoint, evaluation result or main-field image
was changed by this request. Earlier uncommitted work is preserved.

## Validation

The receipt is `artifacts/pino_setup_publication_validation.json`.

- All ten retained files are byte-identical to the approved external exports;
  total 1,458,907 bytes (about 1.39 MiB), with all per-file limits satisfied.
- Actual PDF sizes match specifications. All extracted text is 7 pt, all fonts
  are embedded TrueType, and each file retains text objects. Only melting
  shading and spinodal field colors have raster layers; the other PDFs are
  all vector. Previous per-figure visual reviews remain applicable because
  no image was redrawn.
- All five README PNG/PDF/specification/index links exist. The asset indexes
  have 47 checked local links, plus the newly written validation receipt.
- `npm --prefix website run build` succeeds for nine pages. All 255 local HTML
  links exist; ten website setup assets match the repository copies. None of
  the five current chapters refers to the historical setup SVG.
- `git diff --check` passes.

Status: complete. Blocker: none. Next: none for this request. Numerical work
remains in `docs/tasks/electropolish-fixed-particle-horizon-ratio.md`; Git and
live website deployment require a separate explicit user request.

Memory changed: this archived record, `docs/PROGRESS.md`, `docs/DECISIONS.md`
and the active numerical task's concurrent-work note.
