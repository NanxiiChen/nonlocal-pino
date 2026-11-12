# Problem setup figures

These are the five approved material-aware setup diagrams. Each PNG is the
example README preview; the matching PDF retains editable text and embedded
TrueType fonts for paper writing. Both depict prescribed inputs and boundary
conditions, not learned or evolved solutions.

| Example | Preview | Paper PDF | Original size | Source/style specification |
|---|---|---|---|---|
| Ice melting | [PNG](icemelting3d.png) | [PDF](icemelting3d.pdf) | 100 × 83 mm | [Specification](../../artifacts/melting_setup_materials.json) |
| Electropolish | [PNG](electropolish2d.png) | [PDF](electropolish2d.pdf) | 100 × 58 mm | [Specification](../../artifacts/electropolish_setup_materials.json) |
| Spinodal decomposition | [PNG](confined_ch2d.png) | [PDF](confined_ch2d.pdf) | 90 × 81 mm | [Specification](../../artifacts/spinodal_setup_materials.json) |
| Two-phase channel flow | [PNG](channel_nsch2d.png) | [PDF](channel_nsch2d.pdf) | 110 × 65 mm | [Specification](../../artifacts/channel_setup_materials.json) |
| Ceramic-particle corrosion | [PNG](particle_corrosion2d.png) | [PDF](particle_corrosion2d.pdf) | 100 × 75 mm | [Specification](../../artifacts/particle_corrosion_setup_materials.json) |

Text is 7 pt at the listed PDF dimensions. Use these dimensions when inserting
the diagrams into LaTeX to retain the intended type size. Text, arrows and
boundary curves remain vector objects; only the melting surface shading and
spinodal initial-field colors are rasterized. The diagrams follow the
[plotting guidelines](../../docs/PLOTTING_GUIDELINES.md).

Gray metal and blue electrolyte distinguish the corrosion materials; ceramic
particles use ivory with dark outlines. The channel uses blue carrier fluid
and an orange droplet, without assigning specific chemical materials. Spinodal
colors show small initial fluctuations; its gray wall bands are illustrative.
Melting shading distinguishes ice and water, not temperature.

## Reproduction and provenance

The [publication manifest](../../artifacts/pino_setup_figures.json) records each
renderer, specification, validation receipt, file size and output path. The
specifications refer directly to the
[prescribed-input parameters](../../artifacts/pino_setup_schematics.json),
including sample selection, geometry, colors and annotation positions.
No checkpoints, reference solves or training data are needed.

From the repository root, reproduce the final exports on CPU:

```bash
.venv/bin/python -m scripts.render_melting_setup_materials --spec artifacts/melting_setup_materials.json
.venv/bin/python -m scripts.render_electropolish_setup_materials --spec artifacts/electropolish_setup_materials.json
.venv/bin/python -m scripts.render_spinodal_setup_materials --spec artifacts/spinodal_setup_materials.json
.venv/bin/python -m scripts.render_channel_setup_materials --spec artifacts/channel_setup_materials.json
.venv/bin/python -m scripts.render_particle_corrosion_setup_materials --spec artifacts/particle_corrosion_setup_materials.json
```

Each specification names its external PDF/PNG/SVG output paths and in-repository
validation receipt. The explicit Helvetica and sans-serif math font paths must
exist; font files remain external and are not silently substituted. A fresh
machine must install the fonts and explicitly update these paths before rendering.
After checking the new exports, copy approved PDF/PNG pairs into this directory:

```bash
.venv/bin/python -m scripts.publish_main_figures --manifest artifacts/pino_setup_figures.json
```

The copier checks approved byte counts and preserves each source file exactly.
Intentional rendering changes require a refreshed manifest and visual review.
The five pairs total about 1.39 MiB, under a separate 2 MiB collection budget
and a 1 MiB per-file limit. Alternate previews and font binaries remain external.
The [publication validation](../../artifacts/pino_setup_publication_validation.json)
records the retained PDFs, README links and local website build checks.

## Legacy SVGs

The six earlier three-input SVG schematics remain for historical references,
including the heterogeneous-material example. They retain their original
320 KiB collection budget and can be regenerated with
`scripts/render_pino_setup_schematics.py`. They are separate illustrations,
not vector duplicates of the approved PDF/PNG pairs above. See the
[asset index](../README.md) for their individual links.
