# Material-aware ceramic-corrosion setup preview

User accepts the compact Electropolish style and requests a ceramic-corrosion
setup with appropriate material colors. Use the same gray metal/light-blue
electrolyte and warm ivory, dark-outlined inert ceramic. Single first existing
six-particle prescribed layout; keep fixed initial pit and narrow reservoir
mouth, no fabricated evolution. The remaining outer and particle walls are
closed. Label zero partial normal derivatives of phi and c: equivalent to
phi_n=q_n=0 since q_n=c_n-Delta_c*h'(phi)*phi_n. This denotes continuum boundary
data, not exact pointwise satisfaction by a finite regional graph.

Plan: 100mmwide vector PDF/PNG/SVG with7ptlabels, actual physical5:3aspect,
short initial-pit and random-particle callouts. Validate source geometry with
the production geometry-only function; no JAX or PDE/reference solve. Keep
published setup and result assets unchanged while previewing this new style.

Status: complete; blocker: none. Next: await user feedback on the material palette.
This presentation request is concurrent with the active horizon-ratio study.


## Completed artifacts and reproduction

Renderer: `scripts/render_particle_corrosion_setup_materials.py`.
Explicit source/layout/font/palette/output specification:
`artifacts/particle_corrosion_setup_materials.json`.
Receipt: `artifacts/particle_corrosion_setup_materials_validation.json`.

```bash
.venv/bin/python -m scripts.render_particle_corrosion_setup_materials \
  --spec artifacts/particle_corrosion_setup_materials.json
```

PDF/PNG/SVG are saved at the direct output paths in the specification under
`/root/autodl-tmp/tf-logs/particle_corrosion2d/publication-setup-materials-20261112/`.
Actual domain50×30um and fixed6umradiuspit, six existing smooth particles;
geometry-validation clearances pass. The narrow opening is x=.445..555 only.
All text uses actual7pt, embedded TrueType Helvetica/sans mathematics;
100×75mmPDF contains only vector elements. Editable SVG text retained.
Independent PDF extraction, six ceramic-center color checks and visual PNG/PDF
inspection pass. Gray matrix/light-blue liquid/warm-ivory ceramic with dark
outlines are distinct; x/y physical scales are equal.

Only presentation source/spec/receipt and memory changed. Existing published
assets, field palettes, training and references are unchanged; no Git history
operation. Active horizon-ratio task retained. No inference or PDE solve.
