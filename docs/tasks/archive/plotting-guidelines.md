# Publication plotting guidelines

## Goal and acceptance criteria

Write `docs/PLOTTING_GUIDELINES.md` and reference it from `AGENTS.md`.
Specify legible/editable text, explicit Helvetica paths, sans-serif mathematics,
5–7 pt at final size, NPG colors, concise labels, embedded TrueType PDF and
matching PNG previews. Use the two attached notebooks as visual references,
with the current user requirements taking precedence over their 9 pt labels.
No existing figure redraw or numerical change is requested.

## Context and sources

- `/root/.codex/attachments/ff13c9e2-b8ba-4621-a815-a79333d2f26c/eval.ipynb`
  (50 cells; style cell 0; violin cell 44).
- `/root/.codex/attachments/ff13c9e2-b8ba-4621-a815-a79333d2f26c/eval (1).ipynb`
  (35 cells; style cell 1; field layouts cells 8, 11, 13, 27).
- Both use Helvetica from `./helvetica/`, NPG ten colors, 0.5 pt axes/ticks,
  inward ticks, frameless legends and PDF/PNG export. Field example uses
  `icefire`, equal aspect, rasterized field artists and smooth display only.
- Repository asset budgets/provenance remain in `assets/README.md` and
  `docs/DEVELOPMENT.md`. Previous metric/figure task is complete.
- System fontconfig listing does not expose Helvetica; this task records the
  acquisition/path requirement, without claiming installation or font readiness.

## Changes

- Added `docs/PLOTTING_GUIDELINES.md` in Chinese: final-size typography,
  explicit font acquisition/path requirements, sans-serif math, exact notebook
  NPG palette and named continuous derivatives, sparse labels, field/violin
  conventions, current space-time/channel/sample metric, editable PDF/PNG,
  reproducibility records and delivery checks.
- Added the required-reading reference to `AGENTS.md`, plus navigation from
  `docs/DEVELOPMENT.md` and the root README research-record index.
- Appended the rationale in `docs/DECISIONS.md`; updated `docs/PROGRESS.md`.
- Existing numerical sources, figures, defaults and external artifacts untouched.
  Pre-existing uncommitted experiment changes remain in the working tree.

## Validation

- Inspected the style cells and plotting/layout cells in both notebooks without
  running either notebook or reading their data/checkpoints.
- Exact ten-color NPG sequence matches both notebook style cells.
- Reviewed Matplotlib official font/mathtext documentation and ggsci palette
  reference; links are included near the relevant requirements.
- Three Python snippets pass static AST parsing only; no snippet executed.
- Math/code fences and all new local document/navigation targets checked.
- Documentation diff inspected; `git diff --check` passes.
- No behavioral tests, compilation, training, inference, figure generation,
  font download/install, deletion or Git/history operation required/performed.

## Status

Complete. Specification and navigation are ready for future redraw tasks.

## Blocker

None for the requested documentation. Font availability and real editor/PDF
acceptance remain per-render checks, not an assertion about existing figures.

## Next action

Await a separately requested redraw. Use the specification before creating any
publication figure, resolve licensed Helvetica paths, and retain saved metrics.
