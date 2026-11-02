# README figures

`pino/` contains the four representative evolution figures used by the main
PINO example READMEs:

| Figure | Example |
|---|---|
| [icemelting3d.png](pino/icemelting3d.png) | Random initial ellipsoids, 3D melting |
| [electropolish2d.png](pino/electropolish2d.png) | Random initial interfaces, 2D electropolishing |
| [confined_ch2d.png](pino/confined_ch2d.png) | Random holes, 2D spinodal decomposition |
| [channel_nsch2d.png](pino/channel_nsch2d.png) | Random channel walls, coupled 2D NS–CH flow |

Keep only curated documentation images here: each PNG below 1 MiB and the four
images below 2 MiB in total. Raw fields, checkpoints, full galleries and videos
remain external artifacts.

[Provenance](../artifacts/pino_readme_figures.json) records the exact samples,
times, source fields, checkpoints and references. Regenerate from those existing
artifacts without training or solving a reference:

```bash
.venv/bin/python scripts/render_pino_readme_figures.py --case all
```
