# Material-aware spinodal setup

User requests the spinodal schematic in the established compact material-aware
style. Show one real prescribed two-hole geometry inside the fixed ellipse;
use the actual fixed ten-mode initial function, not evolved phases. Blue/orange
continuous tones show signed initial fluctuations, white hole interiors and
gray hatched wall bands denote excluded impermeable boundaries. Only hole
positions/radii are random. Short7ptlabels and neutral/no-flux BC equation;
no flow/mechanics or material identities added. Keep equal spatial scales,
vector text/geometry, PDF/PNG/SVG and direct provenance.

Status: complete. Blocker: none. Next: user artwork feedback. Concurrent
presentation request; retain the numerical horizon-ratio task as active.
No new simulation, inference, canonical-asset replacement or Git publication.


## Task-end and reproduction

Renderer: `scripts/render_spinodal_setup_materials.py`; explicit inputs/style:
`artifacts/spinodal_setup_materials.json`; compact check receipt:
`artifacts/spinodal_setup_materials_validation.json`.

```bash
.venv/bin/python -m scripts.render_spinodal_setup_materials \
  --spec artifacts/spinodal_setup_materials.json
```

Exports PDF/PNG/SVG use the direct paths under
`/root/autodl-tmp/tf-logs/confined_ch2d/publication-setup-materials-20261112/`.
110×81mmcanvas,66×52.8mmellipse, exact physical scale1.25. Initial values use
801×641cell-centered analytic samples of the production fixed ten-mode function;
explicit limits[-.3,.3], softblue/neutral/NPGorange palette. No sample-dependent
rescaling or evolved shape. White holes with1.1mmhatched bands inside the excluded
regions; outer band outside the physical ellipse. Decorative wall thickness is
not a new modeled layer. Full original geometry and source coefficients retained.

Verified valid two-hole geometry and prior bounds, independent25point initial
formula equality to1e-15, both hole-center pixels white, actual PDF size and
all7ptdark glyphs, embedded TrueType/noType3, one raster field layer with vector
walls/text, editable SVG text. PNG/PDF inspected; no clipping/overlap.
`git diff --check` passes. No numerical, main-figure, website or Git change.
Memory: this archive, PROGRESS, DECISIONS and active-ratio side-work note.
