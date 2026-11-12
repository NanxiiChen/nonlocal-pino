# Ceramic styling and website main-figure synchronization

## Scope and acceptance

User-requested presentation work alongside the active fixed-particle horizon
queue. Use ordinary gray ceramic interiors with dark outlines; show the five
approved README main figures on the website. Preserve all fields, errors,
references and numerical defaults. Publish only these changes after explicit
user authorization, then check GitHub Pages and its served assets.

## Implementation and local validation

The final ceramic style is fill #A0A0A0, outline #333333, width 0.45 pt. The
intermediate #555555 fill was rejected by the user as too dark. Both previous
previews remain outside Git. The current source/style specification and its
validation receipt are in `artifacts/particle_corrosion_main_figure*.json`.
CPU regeneration reads the original saved fields; no inference or solve.

All 20 instantaneous phase norms reproduce independently. The original and
new PDF text/spans, path coordinates and figure layout match. Only the 20
background fills and 30,720 ceramic edge styles change. All 120 particle-center
pixels have the new fill; pixels outside the panels are unchanged. The PDF
retains embedded TrueType, editable 5–7 pt text and 21 raster field/bar layers.
PDF and PNG were visually inspected.

`website/build.mjs` reads `artifacts/pino_main_figures.json` and copies the same
10 PNG/PDF files used by the READMEs, checking approved sizes and budgets. All
five captions now match their model, sample roles, reference type and snapshot
metric. Image dimensions and editable-PDF links are current. The Pages workflow
watches `assets/pino-main/**` and the manifest; the prior historical image
directory is no longer the website's gallery source. Figure-link CSS keeps PDF
links inline in captions. The case and asset READMEs document reproduction.

The build passes. Chromium checked all nine chapters at 1440, 390 and 320 px,
offline with GPU disabled: no overflow, missing local links/images, or KaTeX
errors; all five PNG dimensions and PDF links match. Browser screenshots of
the ceramic gallery were inspected. All 10 website assets are byte-identical
to their repository sources; the other eight publication files are unchanged.
Collection size is 8,463,535 bytes, within the 9 MiB budget. `git diff --check`
passes. Compact receipt: `artifacts/pino_main_website_sync_validation.json`.

## Status and exact next action

Local implementation and validation complete. User explicitly authorized
commit/merge/push/deployment, excluding experimental code. Publish via dev →
main, inspect the Pages action, verify live PNG/PDF bytes, then archive this
side-work record. The background horizon-ratio study remains the sole active
numerical task and its running processes must not be interrupted.
