# Nonlocal/local-AD README figures and reproducibility

## Goal and acceptance

User requests convergence and physical-field comparison figures for both
Electropolish and ceramic-particle corrosion in their example READMEs, plus
exact data/configuration provenance for future reproduction. Publish four
compact PNGs in `assets/pino-comparisons/`, a CPU-only renderer and manifests
with direct data/checkpoint/source paths, full saved configurations, seeds,
sample IDs, actual times, metric definitions and display conventions.
Keep existing SOTA figures/defaults and all unrelated pending work intact.
No Git/history operation is authorized in this turn.

## Context

Completed eight-arm study: [paired results](pino-rolling-local-ad-controls.md).
Particle gallery: [field inspection](particle-method-field-gallery.md).
External root: `/root/autodl-tmp/tf-logs/pino-rolling-ad-cold-20261111-01`.
Three strict pair audits pass. Electro seed43 retains its known FP32 input
witness mismatch and failed raw journal; label it explicitly in both seeds'
convergence results. Physical-field galleries use the strict seed42pair.
Particle monitors4localFEM cases. Electro monitors4native cases; its64localFD
cohort has only5k/10kanchors. Never conflate these different measures/cohorts.

## Status, blocker and next action

Complete. Both example READMEs now include the current paired study's results,
time/memory scope, convergence and three-input reference/nonlocal/local-AD
phase galleries. Production SOTA figures/defaults remain unchanged. Root README
links the studies; assets/development docs specify a separate comparison-image
budget and reproduction procedure. No pending work or GPU work.

## Outputs and provenance

Four curated PNGs, total1,228,600bytes(1.17MiB), each below1MiB and combined
below the declared2MiBcomparison budget:

|Path under assets/pino-comparisons|Bytes|Pixels|
|---|---:|---|
|particle_corrosion2d_convergence.png|61488|1728x656|
|particle_corrosion2d_fields.png|575201|2112x2512|
|electropolish2d_convergence.png|122142|1728x1184|
|electropolish2d_fields.png|469769|2112x2304|

`scripts/render_pino_derivative_comparisons.py` reads external saved fields and
logs without model inference. `artifacts/pino_derivative_comparisons.json`
(116916bytes) declares exact direct paths, final/midpoint checkpoints, frozen
source inventories, all8full configuration snapshots, device/timing records,
reference identities, samples, times and plot settings. The separate
`artifacts/pino_derivative_comparison_validation.json`(120660bytes) retains
all plotted scalar histories/anchors, final casewise metrics, per-panel errors,
saved field ranges, image sizes and render-library versions. No file hashes,
raw arrays or model weights are added to Git candidates.

Field selection is seed42 and cohort indices0,1,2 in both examples. Particle
times0,.2,.5,.8,1 correspond to0,70,175,280,350s. Electro times0,.25,.5,.75,1
correspond to0,1.25,2.5,3.75,5s; original surface identities are354943939,
34428158,1162395345. Both field selections use strict-pair passes. Geometry
holes and original FEM triangles are preserved. All panels share0..1phase
colors; black0.5contours identify the displayed field, white dashed0.5contours
the local reference. Gouraud interpolation is display-only; PNG palette
quantization does not change numerical arrays or metrics.

Particle curves use all recorded4local-FEM monitors. Electro top panels use
the4native monitor-bank entries16--19; bottom panels use only the separate
64local-FD5k/10kanchors. Both seeds are retained and Electro43's failed strict
input-value audit remains explicit. Time curves use corresponding recorded
trainer elapsed values; separate cold-process/warm-step figures are scoped in
the README tables/text. No causal boundary attribution or universal method
superiority is claimed; reused-development and finite-budget limits retained.

## Verification

- Executed the documented CPU renderer on both cases; all8saved configuration
  snapshots match actual configs, frozen source inventories exist, actual
  spatial/time identities agree, and every final physical-field norm matches
  the saved per-case and aggregate records.
- Inspected all4rendered images; source metrics and phase conventions match
  the README tables/captions. Existing6SOTA PNGs and setup SVGs have no diff.
- Repeated the full regeneration command and compared file bytes directly:
  all4PNGs are byte-identical, without hashes. Only this CPU documentation
  pipeline was repeated; no numerical training tests were needed.
- PNG integrity/palette/dimensions, both JSON files, source AST,151local links
  and image budgets pass. The original raw owner journal still says failed,
  as required by the Electro43qualification. `git diff --check` passes.

Command from repository root:
```bash
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/render_pino_derivative_comparisons.py --case all
```

Per-case commands are documented in each README. Missing external data fails
explicitly; stored curve values remain independently readable in the receipt.

## Session changes

Added the four compact PNGs, renderer and two manifests; updated both example
READMEs, root README navigation, `assets/README.md` and `docs/DEVELOPMENT.md`.
Updated `docs/DECISIONS.md`, archived this task and returned `docs/PROGRESS.md`
to no active task. No training/default/source-numerics changes, external
artifact cleanup, Git/history action or changes to the published Pages site.
