# Melting schematic leader placement

User requests improved arrow lengths and positions. Align the initial caption
above the ice, point downward to its upper-right silhouette, shorten the BC
leader and aim it diagonally at the cube boundary. Preserve geometry/view,
shading/colors,7pttype and coordinate triad. Put editable annotation positions
in the explicit configuration and save prior outputs/sources beside v1exports.

Status: complete; blocker: none. Next: user artwork feedback. Concurrent presentation task;
no numerical, canonical-asset, website or Git changes.


Completed: initial leader now5.26mm, pointing nearly vertically down to the
exact upper-right ice silhouette; BC leader6.77mm points diagonally to the cube
edge. Caption/leader positions and target direction are explicit in the JSON
specification; the receipt records computed targets/lengths. Coordinate triad
retained after inspection. V2PDF/PNG/SVG outputs use direct paths in
`artifacts/melting_setup_materials.json`; prior exports/renderer/spec/receipt
remain beside v1. No physical or aesthetic changes beyond these annotations.

Checked identical parameters, rotation, projection, cube vertices, silhouette,
mesh counts, text content and7pttype. Embedded raster shading is byte-identical
to v1; actual PDF size and embedded TrueType/noType3 retained. PDF/PNG visually
inspected, no overlapping captions; `git diff --check` passes. No numerical,
README/site or Git changes. Memory: this archive, PROGRESS, DECISIONS and
active-ratio side-work note.
