# Remove initial concentration formulas from setup artwork

User asks to remove c=h(phi) from both Electropolish and ceramic-corrosion
material schematics, leaving the explanation to the paper. Acceptance: only
these two labels disappear; same geometry, BCs, materials, scale and7pttext.
Export PDF/PNG/SVG and preserve previous sources/receipts. No numerical or
Git action. Concurrent presentation request; horizon-ratio task stays active.

Status: complete. Blocker: none. Next: user artwork review.


Task-end: removed one label call and its explicit configuration key from each
material renderer/spec. Both PDF/PNG/SVG outputs regenerated: Electropolish v4,
particle corrosion v3 under their existing publication-setup-materials-20261112
roots. Prior source/spec/receipt files retained beside prior exports. Updated
per-case validation receipts record direct previous PDF paths. PDF vector
geometries/colors/page dimensions unchanged, formula absent, allremainingtype7pt,
embedded TrueType/noType3, PNG previews inspected. `git diff --check` passes.
No changes to numerical models, main field figures, website or Git history.
Memory changed: this archived task, PROGRESS, DECISIONS and active-ratio note.
