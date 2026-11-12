# Material-aware two-phase channel schematic

User asks for the next schematic in the compact material-aware publication
style, with colors chosen for the two-phase-flow case. Use pale-blue carrier,
muted NPG orange droplet and gray hatched solid walls. These denote generic
immiscible phases, not specific oil/water or different density/viscosity.
Single first existing prescribed random-wall input; fixed initial disk, actual
4:1aspect. Keep short condition labels,7ptHelvetica/sans math, editable PDF/SVG
and PNG. No fabricated evolved shapes or velocity results; arrows indicate
inlet forcing direction after the ramp, not initial/rest velocity.

BCs verified against solver.py and losses.py: inlet prescribed parabolic U(y,t),
phi=1,mu=0; stationary walls u=0 and partial_n phi=partial_n mu=0; outlet zero
hydrodynamic traction sigma*n=0 and partial_n phi=partial_n mu=0. Here
sigma=-pI+eta*(grad u+grad u transpose); do not replace traction with p=0.
Geometry source: artifacts/pino_setup_schematics.json, channel_nsch2d, sample0.
Production NumPy wall profiles/geometry validation; no JAX or FEM solve.

Status: complete. Blocker: none. Next: user feedback on the setup preview.
Concurrent user-authorized presentation work; ratio study remains active.
No main-field palette changes, canonical asset replacement or Git publication.


## Task-end and reproduction

Renderer `scripts/render_channel_setup_materials.py`, explicit source/palette/
font/BC configuration `artifacts/channel_setup_materials.json`, direct-path
receipt `artifacts/channel_setup_materials_validation.json`.

```bash
.venv/bin/python -m scripts.render_channel_setup_materials \
  --spec artifacts/channel_setup_materials.json
```

PDF/PNG/SVG outputs live under
`/root/autodl-tmp/tf-logs/channel_nsch2d/publication-setup-materials-20261112/`.
110×65mm canvas,80×20mm actual4:1channel,8.8mmdiameter initial circle. Generic
carrier #C9EAF3, droplet NPG #F39B7F, hatched walls #D4D7DA, dark labels/edges.
All16coefficients reuse the first prescribed input; production geometry
validation checks minimum gap/slope/curvature. The disk is the initial phi=0
contour upstream of the random segment. No predicted deformation is drawn.

PNG/PDF inspected: labels do not overlap; inlet, wall and traction labels
match actual code. Independent PDF checks confirm110×65mm, allglyphs7pt,
embedded TrueType Helvetica/DejaVuSans (noType3), all-vector output, editable
SVG text, circular8.8×8.8mmbounds and exact orange/blue center colors.
`git diff --check` passes. No training/reference solve/inference, field palette,
canonical README/site asset or Git change. Memory: this archive, PROGRESS,
DECISIONS and active-ratio task cross-reference.
