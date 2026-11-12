# PINO architecture and training master figure

Goal: redesign the architecture illustration as a publication master figure,
using the supplied visual reference for composition. Cover regular patch and
irregular two-stream point encoding, spatial reads, per-layer temporal FiLM,
geometry parameterization, background rolling pools, core/halo sampling and
nonlocal physics training. No numerical architecture/training changes.

Acceptance: source-traceable diagram with correct separate domain/boundary
K/V and shared spatial Q; time introduced in subsequent decoder MLP layers;
nonlocal spatial and time JVP roles separated. Case-dependent preconditioning,
balancing/causal and optimizer choices explicitly qualified. 170mm-wide,
editable embedded-font5–7ptPDF, PNG preview and editable SVG; layout visually
inspected and source/typography/output provenance retained. Preserve previous
architecture export before replacement; local website references remain valid.

Sources: `src/nonlocal_pinn/nn/conditional.py`, `point_conditional.py`, selected
example networks/losses/online producers; `website/content/architecture.md`,
`training.md`; prior `attention-film-order` and `two-stream-architecture-diagram`
audits. Style reference is the user attachment
`/root/.codex/attachments/72f8c93e-a1f6-4107-b277-7015656b7d87/codex-clipboard-30926045-3bb0-4e51-ab1a-bf571a0895a4.png`.

Status: complete. New170×150mmmaster figure has7panels: parameterized known
inputs/CPU workers/rolling pool/core+halo; patch path; two-stream point path;
three temporal FiLM layers; nonlocal stencil/timeJVP; physics loss; weighting
and Adam/SOAP. Explicit note qualifies core/halo, preconditioning/balancing and
optional causal use. Input/output textures are illustrations, not solver data.

Implementation audit: `ConditionalFieldOperator.encode/prepare_encoded/_point`
and `HeterogeneousPointOperator.encode/prepare_encoded/decode`,
`HeterogeneousAttention.__call__` and `normalized_integral_attention` confirm
pointwise lifts, separate K/V/measures, shared spatial query, equal message
mean, residual projection/MLP updates, then separate per-layer time FiLM.
Checked channel/particle decoders, scalar nonlocal operator sign and background
particle-pool producer. Caption retains physical-wall and channel AD-curl
qualifications; no inference that every case uses the same options.

Sources/outputs:
- `scripts/render_pino_master_figure.py` and `artifacts/pino_master_figure.json`.
- `artifacts/pino_master_figure_validation.json` with fonts, text boxes,
  source paths, PDF edit probes, asset parity and browser checks.
- `assets/pino-architecture/pino_master_figure.png` (444,488bytes) and PDF
  (141,390bytes); editable `website/architecture.svg` (276,769bytes).
- External previews, previous SVG and source snapshot:
  `/root/autodl-tmp/tf-logs/pino-master-figure-20261112/`.

Diff-level changes: new renderer/spec/receipt and two curated assets; replace
website SVG; embed same PNG/PDF in rootREADME and both report placements;
add precise training-panel scope, asset guide and website regeneration guide;
build copies PNG/PDF and Pages trigger includes architecture asset directory.
No numerical source/default/weights changes or Git/live deployment.

Validation: actual rendering and AST/JSON/XML checks; no overlapping text boxes;
all PDF text5–7pt, dark, embedded TrueType and inside page. Ordinary and Greek
labels edited in an isolated PyMuPDF probe. Only3schematic textures are raster;
70SVGtext elements retained. PNG/PDF and browser image visually inspected.
Nine-page build and260relative links pass; index/architecture at1440and390px
load correctly without overflow/errors. All repository/build copies and frozen
source inputs match byte-for-byte. Combined artwork862,647bytes. Final
`git diff --check` passes. Prior SVG preserved externally.

Memory updated: PROGRESS, DECISIONS and this archived task.
Blocker:none. Next:user design feedback; no required implementation remains.
