# README figures

`pino/` contains the four representative evolution figures used by the main
PINO example READMEs:

| Figure | Example |
|---|---|
| [icemelting3d.png](pino/icemelting3d.png) | Random initial ellipsoids, 3D melting |
| [electropolish2d.png](pino/electropolish2d.png) | Random initial interfaces, 2D electropolishing |
| [confined_ch2d.png](pino/confined_ch2d.png) | Random holes, 2D spinodal decomposition |
| [channel_nsch2d.png](pino/channel_nsch2d.png) | Random channel walls, coupled 2D NS–CH flow |

The channel figure uses the selected20000-step final model and all three
predeclared reserved geometries, with the same five times and FEM contour
overlays as the earlier10000-step figure. Its source fields and per-case L2
values are recorded in the provenance manifest below.

Keep only curated documentation images here: each PNG below 1 MiB and the four
images below 2 MiB in total. Raw fields, checkpoints, full galleries and videos
remain external artifacts.

`pino-setups/` contains four additional lightweight SVG setup schematics, one
per example. Each shows three prescribed inputs, the fixed physics/boundaries,
and the varying condition. They contain no predicted/evolved fields. The four
SVGs total less than256KiB and use portable vector geometry/text:

| Setup | Varying input |
|---|---|
| [icemelting3d.svg](pino-setups/icemelting3d.svg) | Initial ellipsoid center, semiaxes and orientation |
| [electropolish2d.svg](pino-setups/electropolish2d.svg) | Initial rough interface |
| [confined_ch2d.svg](pino-setups/confined_ch2d.svg) | Two hole positions/radii; same initial field |
| [channel_nsch2d.svg](pino-setups/channel_nsch2d.svg) | Upper/lower walls; same initial droplet and driving |

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
