# Add the middle particle grid and refine bubble styling

User requests an approximately10k-particle third curve, NPG C0/C1/C2 colors,
larger differences between four marker sizes, white marker edges and thicker
lines. Predeclare11,476particles(151×76), with the same delta/h1.5/2/2.5/3.
Ratios2and3already exist; cold train only ratios1.5(delta.01)and2.5(delta1/60).
Keep frozen width96/Adam/B8/Nt16/core256/rolling32/seed42/10k scheme and common
64local final-weight evaluation. Run serially for comparable timing.

Final plot:3curves×4means, no spread; NPG C0#E64B35,C1#4DBBD5,C2#00A087 in
ascending particle count. Four explicitly keyed marker diameters5/8/11/14pt
give stronger separation; this is an ordered size key, no longer area-linear
in delta/h. White edges~1pt, data lines~1.4pt. Preserve the two-curve export.

Status: complete. Owner559807 finished; both cold10k models, their64-case
evaluations and final audit exited successfully (five stages).
Root: `/root/autodl-tmp/tf-logs/electropolish2d/cost-error-three-curves-cold10k-20261112-01`.
`owner.json`, `journal.jsonl`, `owner.log` and `logs/` retain recovery evidence;
do not restart an active owner. Both halos fit the already completed same-grid
ratio3 capacities. GPU model/memory/driver matches prior timing records. First
initial model and full Adam state match the ratio2 baseline byte-for-byte.

Renderer supports2or3groups with explicit marker-size mode. Old two-curve PNG
regenerates byte-identically in `/tmp/electropolish-two-curves-regression.png`;
its coordinates, sizes, means and automatic limits are unchanged. New12-run
selection/style is `artifacts/electropolish_cost_error_three_curves.json`, frozen
along with rendering source in the new study's source directory before results.

First ratio1.5completed10k+64evaluation: channel-mean0.0495549595241353,
SD0.0045949582939045185, phi0.045316780993960734, c0.053793138054309866.
Full cold process539.067693s; separate evaluation31.595709s. Both stages exit0.
Second ratio2.5started at2026-11-12T06:49:38UTC.

Second ratio2.5completed: mean0.023821115078708065, SD0.008098514757545272,
phi0.021725098479799924, c0.025917131677616206. Cold process842.954272s;
evaluation31.100564s. Final audit3.595140s. Both cold states and all10000
same-grid draws match;128saved-field sample norms replayed. Entire new queue
completed in24.14min at2026-11-12T07:04:16UTC.

Final source/receipt: `artifacts/electropolish_cost_error_three_curves.json`
and `artifacts/electropolish_cost_error_three_curves_validation.json`.
`artifacts/electropolish_cost_error_medium_completion.json` records results,
all stage times and direct run/config/model/evaluation paths. External outputs:
`/root/autodl-tmp/tf-logs/electropolish2d/publication-cost-error-three-curves-20261112/electropolish_cost_error_three_curves.{pdf,png}`.

Validation:17unique source models/one duplicate;12selected final10kmodels,
768sample means, exact timing replay, four circles per NPG C0/C1/C2 curve.
PDF paths confirm5/8/11/14pt diameters,1ptwhite edges and1.4ptlines; embedded
editable6–7pttext,130×85mmvector PDF24,239bytes/PNG94,046bytes. PNG and
rasterized PDF inspected. Near-coincident measured points remain unshifted.
Previous two-curve export remains byte-reproducible. README now records the
three-grid table, encoding, timing scope, single-seed caveat and reproduction.

Diff-level changes: generic2/3group renderer and explicit sizing options;
old style gains explicit equivalent fields; new cold-control runner, new plot
spec/receipt/completion record; README and task memory updated. Numerical model
and production defaults unchanged. Blocker:none. Next: user figure review;
no training remains and no Git/Pages action is authorized or performed.
