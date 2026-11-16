# Publication main figures

Five accepted phase galleries, each at 170 mm width with 5–7 pt embedded,
editable TrueType text. Dense field layers are rasterized; reference interfaces
and text remain vector. PDF/PNG pairs are byte-identical copies of the approved
external exports. Raw fields, meshes and weights remain external.

Each gallery shows **Best / Middle / Worst**, selected by equal-point,
phase-only relative L2 over the complete declared space-time evaluation grid.
Middle is the lower central actual sample after sorting by (error, sample ID).
Snapshot labels report instantaneous spatial phase error; for Melting they use
the complete original 3D volume, not only the displayed slice. These are not
channel-averaged metrics or averages of per-frame relative errors.

| Example | Cohort | Best / Middle / Worst IDs | Preview | Paper PDF |
|---|---:|---|---|---|
| icemelting3d | 64 shapes | 52 / 33 / 56 | [PNG](icemelting3d.png) | [PDF](icemelting3d.pdf) |
| electropolish2d | 64 profiles | 30 / 5 / 56 | [PNG](electropolish2d.png) | [PDF](electropolish2d.pdf) |
| confined_ch2d | 8 reserved geometries | 1027 / 1025 / 1026 | [PNG](confined_ch2d.png) | [PDF](confined_ch2d.pdf) |
| channel_nsch2d | 3 reserved geometries | 1 / 2 / 0 | [PNG](channel_nsch2d.png) | [PDF](channel_nsch2d.pdf) |
| particle_corrosion2d | 64 geometries | 50 / 53 / 30 | [PNG](particle_corrosion2d.png) | [PDF](particle_corrosion2d.pdf) |

All figures retain their original main-figure checkpoints. Melting and Spinodal
use native nonlocal references; Electropolish uses local FD101; Channel and
Ceramic use independent local FEM. The Channel development case is excluded.
The cohorts are not equally large or newly blind test sets. Historical summary
metrics and current-default comparisons retain their separate declared scopes.

Melting pairs a 3D surface row with an XY slice row for each selected sample;
the other four figures have three sample rows. All use five actual snapshots.
Melting uses GPU inference verified against ten cached historical fields; the
CPU diagnostic is not selected. Its best sample has no positive-phase surface
at the final frame, explicitly labelled `Melted`. No surface smoothing is used.
Ceramic interiors retain warm ivory (#EFE7D9) with dark-gray (#333333) outlines.

- [Selection, checkpoints, direct data paths and reproduction commands](../../artifacts/pino_ranked_main_preview.json)
- [Publication manifest: direct exports, sample IDs and byte budgets](../../artifacts/pino_main_figures.json)
- [Local integration validation](../../artifacts/pino_ranked_main_publication.json)

The renderer is `scripts/render_ranked_main_figures.py`. It uses existing field
arrays; no model inference or reference solve is needed to redraw a prepared
case. Original style specifications and old random galleries remain historical
records. For the four 2D cases, select `electropolish`, `ceramic`, `spinodal` or
`channel`; the Melting command uses its qualified GPU preparation directory:

```bash
JAX_PLATFORMS=cpu .venv/bin/python -m scripts.render_ranked_main_figures \
  --case electropolish --root /root/autodl-tmp/tf-logs/pino-ranked-main-20261115-01
JAX_PLATFORMS=cpu .venv/bin/python -m scripts.render_ranked_main_figures \
  --case melting --root /root/autodl-tmp/tf-logs/pino-ranked-main-20261115-01/melting-gpu
```

After reviewing a changed export, update the approved byte counts and provenance.
Copy the accepted source files into the shared README/website paths with:

```bash
.venv/bin/python -m scripts.publish_main_figures \
  --manifest artifacts/pino_main_figures.json
```

The website checks the same manifest and copies the same files without
rerendering. Keep one selected PDF/PNG pair per example: each file below 2 MiB,
all ten below 9 MiB. This does not expand any other figure collection's budget.
No numerical defaults change when the selected galleries are updated.
