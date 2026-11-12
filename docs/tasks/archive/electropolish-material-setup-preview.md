# Material-aware Electropolish setup preview

User requests new concise setup schematics, first one Electropolish preview.
Use gray metal and blue electrolyte, position IC/BC labels on their physical
features, and show the random initial-interface family. Preserve the current
published assets until this preview is reviewed. The horizon-ratio queue remains
the sole active experiment; this is authorized concurrent CPU illustration work.

Acceptance: existing prescribed profile coefficients, physical 2:1 aspect,
correct top/bottom Dirichlet and closed side walls, concise 5–7pt dark editable
labels, embedded Helvetica PDF plus matching PNG, source/provenance recorded.
No future field prediction, PDE/reference solve, numerical change or Git action.

Sources: artifacts/pino_setup_schematics.json, current Electropolish README,
operator_local.py normal_derivatives, corrosion2d/residual_experiment.py
initial_fields. For c_se=1 the known initial concentration is h(phi_0); zero
normal derivatives of phi and q imply that of c. Regional pair closure does
not imply exact pointwise Neumann traces at a finite horizon. Label walls as
closed/no flux and do not claim exact boundary differentiation in the artwork.

## Completed preview

One large initial cross-section plus two smaller alternative prescribed
profiles; gray metal with subtle illustrative strokes, blue electrolyte and
hatched inactive side fixtures. Eleven short labels cover the materials,
initial interface, initial concentration, boundary values, closed walls and
dimensions. Reuse all three prior prescribed profiles in order, without new
random draws, amplification or future fields.

Renderer: `scripts/render_electropolish_setup_materials.py`.
Specification: `artifacts/electropolish_setup_materials.json`.
Receipt: `artifacts/electropolish_setup_materials_validation.json`.
External exports: `/root/autodl-tmp/tf-logs/electropolish2d/publication-setup-materials-20261112/electropolish2d_setup_materials.{pdf,png,svg}`.

Reproduce from the repository root:

```bash
.venv/bin/python -m scripts.render_electropolish_setup_materials \
  --spec artifacts/electropolish_setup_materials.json
```

Validation: PDF is 170×80 mm; all actual text is 6–7 pt, dark and embedded in
TrueType Helvetica/DejaVu Sans. No Type 3 fonts or raster image layers; SVG has
11 editable text nodes. Physical aspect is 2:1 in all three diagrams. Plain
text and a formula glyph were successfully edited in a temporary PDF copy.
PDF/PNG inspected; prescribed coefficient/height checks and diff check pass.
File sizes: PDF139118, PNG99112, SVG182430 bytes. No published asset, README,
website, training code or numerical input changed. No Git operation.

Status: requested one-example preview complete, awaiting user style feedback
before further examples. Blocker: none. Task-end memory: this archived record,
PROGRESS, DECISIONS and the active horizon-ratio task's cross-reference.
