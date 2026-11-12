# README figures

`pino/` contains six representative evolution figures used by the PINO
example READMEs:

| Figure | Example |
|---|---|
| [icemelting3d.png](pino/icemelting3d.png) | Random initial ellipsoids, 3D melting |
| [electropolish2d.png](pino/electropolish2d.png) | Random initial interfaces, 2D electropolishing |
| [confined_ch2d.png](pino/confined_ch2d.png) | Random holes, 2D spinodal decomposition |
| [channel_nsch2d.png](pino/channel_nsch2d.png) | Random channel walls, coupled 2D NS–CH flow |
| [heterogeneous_corrosion2d.png](pino/heterogeneous_corrosion2d.png) | Random material kinetics, activation-controlled corrosion |
| [particle_corrosion2d.png](pino/particle_corrosion2d.png) | Six random inert ceramic particles, geometry-conditioned corrosion |

The channel figure uses the selected20000-step final model and all three
predeclared reserved geometries, with the same five times and FEM contour
overlays as the earlier10000-step figure. Its source fields and per-case L2
values are recorded in the provenance manifest below.

The particle figure uses the user-accepted six-particle SI case's width256
20000-step final model on the first three of four development geometries,
with direct nonlocal interface overlays at actual times0,.2,.5,.8,1 (0..350s).
Independent local FEM and native errors are both audited and reported separately.
The legacy setup SVG uses the same three prescribed layouts. Historical eight-particle
regimes, controls, raw fields and checkpoints remain in external artifacts.
The numerical train_default selector is separate from this documented preset.

Keep only curated documentation images here: each PNG below 1 MiB and the six
images below 2.5 MiB in total. Raw fields, checkpoints, full galleries and videos
remain external artifacts.

## Publication setup figures

[pino-setups](pino-setups/README.md) contains the five approved material-aware
setup diagrams: PNG previews embedded in the example READMEs and matching
editable PDFs for paper writing. Each shows one prescribed initial state,
materials and concise boundary/initial-condition labels. The
[publication manifest](../artifacts/pino_setup_figures.json) records the final
source/style specifications and rendering receipts. These ten files total
about 1.39 MiB, under a separate 2 MiB budget and 1 MiB per-file limit.
The local website uses the same approved pairs.

## Historical setup and evolution galleries

`pino-setups/` also retains six lightweight legacy SVG setup schematics for
the six cases below. Each shows three prescribed inputs, the fixed physics/boundaries,
and the varying condition. They contain no predicted/evolved fields. The six
SVGs total less than320KiB and use portable vector geometry/text:

| Setup | Varying input |
|---|---|
| [icemelting3d.svg](pino-setups/icemelting3d.svg) | Initial ellipsoid center, semiaxes and orientation |
| [electropolish2d.svg](pino-setups/electropolish2d.svg) | Initial rough interface |
| [confined_ch2d.svg](pino-setups/confined_ch2d.svg) | Two hole positions/radii; same initial field |
| [channel_nsch2d.svg](pino-setups/channel_nsch2d.svg) | Upper/lower walls; same initial droplet and driving |
| [heterogeneous_corrosion2d.svg](pino-setups/heterogeneous_corrosion2d.svg) | Spatial reaction kinetics; same pit, domain and reservoir |
| [particle_corrosion2d.svg](pino-setups/particle_corrosion2d.svg) | Smooth inert ceramic positions/shapes; same pit, matrix and reservoir |

[Known-input parameters](../artifacts/pino_setup_schematics.json) preserve the
accepted illustration samples. Regenerate without training, model inference,
external arrays or reference solves:

```bash
JAX_PLATFORMS=cpu .venv/bin/python scripts/render_pino_setup_schematics.py
```

[Evolution-figure provenance](../artifacts/pino_readme_figures.json) records the exact samples,
times, source fields, checkpoints and references. Regenerate from those existing
artifacts without training or solving a reference:

```bash
.venv/bin/python scripts/render_pino_readme_figures.py --case all
```

## Nonlocal/local-AD comparisons

`pino-comparisons/` contains two final-weight error distributions and two
physical-field sheets, separate from the selected SOTA figures above.
No checkpoint or training default is promoted by this comparison.

| Figure | Content |
|---|---|
| [Electropolish distribution](pino-comparisons/electropolish2d_distribution.png) | Final10k, seed42, 64 local-FD inputs; channel-averaged space-time relative L2 |
| [Electropolish fields](pino-comparisons/electropolish2d_fields.png) | Seed42, first three inputs, five times; local FD / nonlocal / local AD |
| [Particle distribution](pino-comparisons/particle_corrosion2d_distribution.png) | Final20k, seed42, 64 fresh local-FEM geometries; channel-averaged space-time relative L2 |
| [Particle fields](pino-comparisons/particle_corrosion2d_fields.png) | Seed42, first three fresh geometries, five times; local FEM / nonlocal / local AD |

For each input, normalize phase and concentration separately over all points
of the declared space-time evaluation grid, then average the two relative L2
errors. The violin shows these 64 input-level values; the mean averages them.
Use ordinary unweighted point norms, not channel concatenation or averages of
instantaneous errors. Electro uses 21 uniform times and 101×51 nodes; particle
uses 11 uniform times and all physical FEM nodes. For the irregular FEM mesh,
this point measure differs from earlier volume/time-quadrature diagnostics.

Scott-bandwidth KDE uses all64inputs in natural-log error coordinates.
Thin dashed lines mark the25th,50thand75thpercentiles; no dots or mean labels
are overlaid. The axis shows original error ratios, without multiplication
by100 or a percentage label. Density is per unit log-error, with area normalization
within each case. Smooth tails extend three KDE bandwidths past the observed
log extrema; these are density estimates, not additional observations. Both
cases use the first strictly audited seed42 pair, not the best-performing seed.
Other seeds, old sparse convergence plots and the original Electro43 audit
failure remain preserved in the external archive named by the source manifest.
Field sheets and their spatial phase-error labels are unchanged.

Each PNG stays below1MiB and the four together below2MiB. The two field sheets
retain256-color encoding; the two publication-style violins use lossless RGB.
Raw fields and weights stay external. The violins follow the
[plotting guidelines](../docs/PLOTTING_GUIDELINES.md): two separate55×45.72mm
figures, Helvetica6–7pt, NPG red/cyan and black text. Example names appear
in filenames and README context, not as in-figure titles. Matplotlib automatically
sets each log-axis range from its data; no y limits are prescribed.
PDF and300dpi PNG come from the same figure;
the field sheets have not been restyled.
[Source/configuration manifest](../artifacts/pino_derivative_comparisons.json)
retains direct run/model/data/source paths and all8original configurations,
plus explicit metric/grid/plot settings. [Rendering receipt](../artifacts/pino_derivative_comparison_validation.json)
retains every displayed input's per-channel errors and norms, channel mean,
quantiles, image sizes, font paths and library versions. Each case's
`publication_pdf` records its external editable PDF; the distribution's
`publication` settings record the exact style and font source. No hashes or
implicit file lookup.

Helvetica was downloaded from the user's PI-CViT `helvetica/Helvetica.ttf`
asset to `/root/.local/share/fonts/pino/Helvetica.ttf`; its direct download URL
is in the manifest. The font binary remains outside this repository. A fresh
machine needs that font at the explicitly configured path; the renderer rejects
missing or substituted fonts. Both PDFs embed TrueType glyphs and retain text
objects. PDF font/size and label-edit checks are recorded in
[the PDF validation receipt](../artifacts/pino_violin_pdf_validation.json).

References and inference commands are in the
[64-case evaluation record](../docs/tasks/archive/pino-derivative-evaluation64.md).
The [distribution record](../docs/tasks/archive/pino-final-violin-figures.md)
documents the latest metric and archived figures. Regenerate on CPU:

```bash
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/render_pino_derivative_comparisons.py --case all
```

Add `--figures distribution` to redraw only the two PDF/PNG violin pairs,
preserving the physical-field images and their receipt. Previous presentation
files are preserved at the manifest's `publication_redraw_archive` and
`compact_redraw_archive` and `title_free_redraw_archive` paths.

Use `--case electropolish2d` or `--case particle_corrosion2d` for one pair of
images. The renderer checks source configs, coordinate/time identities and
original field metrics before deriving the new errors; no training, inference
or solve is performed. Missing external data raises an error. Numerical values
remain readable in the receipt; full field regeneration requires the arrays.

## Publication main figures

The approved170mmPDF/PNG main-figure pairs now live in
[pino-main](pino-main/README.md), with exact model/data/style records in
[the publication manifest](../artifacts/pino_main_figures.json). Each example
README now embeds this collection. The older `pino/` entries above remain
historical galleries with their original selections and reference conventions.
The separate10-file publication budget is9MiB, each file below2MiB; raw fields,
meshes and weights remain external. See the collection README for regeneration.
