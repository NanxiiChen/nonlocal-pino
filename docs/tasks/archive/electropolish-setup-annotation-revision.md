# Electropolish setup annotation revision

Follow-up preview alongside the active horizon-ratio study. User requests:
remove both alternative profiles; use the two-line label "Randomly generated /
initial interface"; remove the top Initial state/time heading and the phrase
"Initial concentration:"; replace both No flux labels with partial-derivative
notation. Retain c=h(phi), which was outside the selected deletion region.

Keep the original main profile, materials and 2:1 aspect. Crop the empty sidebar
to a 132×70mm page with unchanged 98×49mm physical panel size. Use display-style
fractions so their numerator/denominator text stays at 7pt. The depicted
zero-normal-derivative equations are continuum BCs; finite-horizon regional
closure is their approximation, and the rough IC can have incompatible traces.
No PDE or numerical interpretation is changed by the label revision.

Preserve previous exports/specification/receipt/renderer externally in the v1
output directory. Use the v2 output directory and the current materials source
specification/renderer. No source assets, README/website or Git changes requested.

Status: complete. One original profile, the exact two-line interface text and
two mathematical side labels are present. Removed headings/prefix/small panels
are absent. The main-panel physical size, profile and material colors are
unchanged. The 132×70mm PDF contains actual6–7pt embedded TrueType text, vector
geometry and no raster images; SVG has10editable text nodes. PDF/PNG inspected;
source/JSON/diff checks pass. All outputs are under the original external
directory name with `-v2` appended. Sizes: PDF71253/PNG75635/SVG72234 bytes.

Source/config/receipt: scripts/render_electropolish_setup_materials.py,
artifacts/electropolish_setup_materials.json and its `_validation.json` sibling.
The prior v1 folder also preserves `render-v1.py`, `specification-v1.json` and
`validation-v1.json`, alongside its unchanged three-profile exports.

Task-end: requested preview revision delivered; no publication or Git action.
Memory updated: this archived record, PROGRESS, DECISIONS, and the ratio study's
presentation note. Next: user feedback or extension to the remaining examples;
the background numerical experiment remains the sole active task.
