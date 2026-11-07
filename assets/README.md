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
The setup SVG uses the same three prescribed layouts. Historical eight-particle
regimes, controls, raw fields and checkpoints remain in external artifacts.
The numerical train_default selector is separate from this documented preset.

Keep only curated documentation images here: each PNG below 1 MiB and the six
images below 2.5 MiB in total. Raw fields, checkpoints, full galleries and videos
remain external artifacts.

`pino-setups/` contains six additional lightweight SVG setup schematics for
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
