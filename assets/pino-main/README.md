# Publication main figures

Five approved final galleries, each at170mmwidth with5–7ptembedded editable
TrueType text. Dense field layers are raster; reference interfaces and text
remain vector. These PDF/PNG pairs are exact copies of the approved external
outputs, not screen captures. No weights, meshes or raw fields are stored here.

| Example | Preview | Paper PDF | Source specification |
|---|---|---|---|
| icemelting3d | [PNG](icemelting3d.png) | [PDF](icemelting3d.pdf) | [Config](../../artifacts/melting_main_figure.json) |
| electropolish2d | [PNG](electropolish2d.png) | [PDF](electropolish2d.pdf) | [Config](../../artifacts/electropolish_main_figure.json) |
| confined_ch2d | [PNG](confined_ch2d.png) | [PDF](confined_ch2d.pdf) | [Config](../../artifacts/spinodal_main_figure.json) |
| channel_nsch2d | [PNG](channel_nsch2d.png) | [PDF](channel_nsch2d.pdf) | [Config](../../artifacts/channel_main_figure_icefire_r.json) |
| particle_corrosion2d | [PNG](particle_corrosion2d.png) | [PDF](particle_corrosion2d.pdf) | [Config](../../artifacts/particle_corrosion_main_figure.json) |

Captions, sample roles, references and metric details are in each example
README and the linked specifications/validation receipts. Melting uses three
samples with paired3D/slice rows; the other four use four samples. Each has
five actual snapshots. Publication does not change any numerical defaults.
Ceramic interiors use gray (#A0A0A0) with dark-gray (#333333) outlines,
separate from the phase colormap.
The website reads the same manifest and copies these PNG/PDF files without
rerendering; Pages watches this directory and the manifest for updates.

Keep only one accepted PDF/PNG pair per example: each file<2MiB and the ten
files together<9MiB. This separate paper-figure budget does not change the
older six-PNG collection or comparison-image budgets. Historical renderings
and alternative palettes remain external or at their existing documented paths.

After regenerating a changed preview with its recorded renderer, review it and
update the approved source byte counts in the manifest before publishing.
Re-copy unchanged approved figures from their direct external paths:

```bash
.venv/bin/python -m scripts.publish_main_figures \
  --manifest artifacts/pino_main_figures.json
```
