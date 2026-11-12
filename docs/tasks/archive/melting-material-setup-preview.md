# Material-aware three-dimensional melting setup

User asks how to draw the melting example after the other material setup
schematics. Create one orthographic equal-scale cube and the actual third
previously prescribed ellipsoid (more visibly unequal axes); this illustrative
choice is unrelated to solution error. Show ice-blue shaded ice, pale-blue
water and a wireframe computational cube. No container wall material, thermal
boundary/heat arrow or fabricated evolution. Only initial center/axes/orientation
vary; actual phase-only driven Allen-Cahn model and continuum zero-normal-phase
boundary. The native finite graph implements closed interactions approximately.

Use7ptHelvetica/sans math, editable vector labels/edges and a rasterized dense
surface if needed; explicit source/camera/light/colors and PDF/PNG/SVG. Reuse
production rotation/prior validation on CPU, no SDF inference/training/solver.

Status: complete. Blocker: none. Next: user artwork feedback. Concurrent
presentation work; active numerical ratio task and canonical assets unchanged.
No Git publication.


## Task-end and reproduction

Renderer `scripts/render_melting_setup_materials.py`, explicit source/camera/
light/palette/fonts `artifacts/melting_setup_materials.json`, direct receipt
`artifacts/melting_setup_materials_validation.json`.

```bash
.venv/bin/python -m scripts.render_melting_setup_materials \
  --spec artifacts/melting_setup_materials.json
```

PDF/PNG/SVG outputs:
`/root/autodl-tmp/tf-logs/icemelting3d/publication-setup-materials-20261112/`.
100×83mmcanvas and7ptlabels. Third original prescribed ellipsoid; no geometry
modification. Orthographic projection with common38mmworld scale, same rotation
as production. Three pale-blue back planes suggest water; wire edges identify
the fixed cube. Ice-blue rasterized Lambert/highlight shading with slight
translucency; exact analytic vector silhouette and visible surface curves.
Those colors/light lines are illustrative, not temperature or phase magnitude.

Checks pass: original prior and unit quaternion, strict cube containment,
closed16384minus256face surface with every edge in two faces/Euler2, orthonormal
rotation/projection and exact projected silhouette enclosing all mesh vertices.
PDF dimensions100×83mm, actualdarkglyphs7pt, embedded TrueType/noType3,
one raster shading layer and vector edges/labels; SVG text retained.
PNG/PDF inspected; text positions adjusted to avoid cube edges. `git diff
--check` passes. No PDE solve, inference, training, mainfigure/README/site
promotion or Git change. Memory: this archive, PROGRESS, DECISIONS and active
ratio-task side-work cross-reference.
