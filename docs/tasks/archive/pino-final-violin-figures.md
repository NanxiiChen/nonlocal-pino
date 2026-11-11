# Final-checkpoint channel-averaged error distributions

## Goal and acceptance

User requests final-weight error distributions for one seed, using violin
plots instead of sparse convergence curves. Latest steering changes the metric:
compute each channel's relative L2 over the full space-time evaluation set,
then average channels per sample; aggregate reported error averages samples.
Use seed42, the first strictly audited pair (not selected by error), Electro10k
and particle20k, on64common inputs each. Both channels are physical(phi,c).
Use ordinary discrete Euclidean norms over all points in the declared grid:
Electro101x51nodes/21uniform times0..1, particle all physical FEM nodes/11uniform
times0..1. This intentionally differs from previous particle quadrature-weighted
metrics; retain old metrics as historical evidence, not as the new violin data.
The reduction order is norm over space-time -> relative norm per channel ->
arithmetic mean over channels -> (summary only) arithmetic mean over samples.
No averaging of instantaneous relative errors and no channel concatenation
before normalization. Add64points, median/IQR and optional overall-mean label.
Retain all old evaluations, caveats and source data. Update renderer, README,
manifest/receipt; rederive new metrics from physical arrays. No new inference,
reference solve, training/default change or Git operation.

## Context

Previous completed task: `archive/pino-derivative-evaluation64.md`.
Source manifest/receipt: `artifacts/pino_derivative_comparisons.json` and
`artifacts/pino_derivative_comparison_validation.json`.
64case evaluation root:
`/root/autodl-tmp/tf-logs/pino-derivative-eval64-20261111-01`.
Electro42 and particle42 both strictly paired. Electro43's failed input-value
audit stays archived and is not reclassified. SOTA/defaults remain untouched.

## Status, blocker and next action

Complete. Both example READMEs now display only the final seed42 channel-mean
error distribution in place of the sparse convergence curves. Two violin PNGs
render on CPU from saved fields. No pending work, new model inference,
reference solve, training, default change, Git/history action or Pages update.

## Metric and implementation

For each sample and each physical channel(phi,c), flatten the declared
space-time evaluation set and compute norm(pred-ref)/norm(ref) in FP64.
Normalize channels separately, then take their arithmetic mean. The violin
contains64such sample-level values; the headline takes their arithmetic mean.
The result is not the norm of concatenated channels, the RMS of channel errors,
an average of instantaneous relative errors, or a pooled-across-samples norm.

Use unweighted evaluation-point norms. Electro retains the declared uniform
101x51x21grid including0/1; the4extra early diagnostic times in its saved25time
array are outside that grid. Particle uses every physical FEM node and11uniform
times including0/1, without the old nodal-volume/trapezoidal weighting. Thus the
particle measure changes as well as the channel aggregation. Nonuniform FEM
node density contributes to this discrete point measure; it is not reported
as a physical-volume integral. This choice is explicit in both READMEs and the
v3manifest. Underlying predictions and references are unchanged.

Renderer first validates saved configs, frozen source inventories, physical
coordinates/times, strict seed42pair status and all256original casewise field
metrics, then derives the new channel errors from original physical arrays.
No dependence on intermediate checkpoints remains in the published figures.
The full8run manifest and original evaluation roots preserve both seeds and
all checkpoints, including Electro43's original failed strict-input audit.

## Results

Final64input averages, percent:

| Case | Method | Channel-mean error | Median | Q1–Q3 |
|---|---|---:|---:|---:|
| Electropolish10k | Nonlocal | 2.046708 | 1.837781 | 1.622250–2.473499 |
| Electropolish10k | Local AD | 4.630900 | 4.245946 | 3.460146–5.505190 |
| Particle20k | Nonlocal | 4.554058 | 4.420097 | 4.203530–4.956986 |
| Particle20k | Local AD | 32.514890 | 32.500321 | 32.182474–32.945137 |

Electro per-channel phase/concentration means are1.862481/2.230936%(NL) and
4.261495/5.000306%(AD). Particle per-channel point-norm means are4.120417/
4.987700%(NL) and32.456553/32.573227%(AD). These latter phase scores differ from
the archived volume/time-weighted phase scores; they must not be mixed.

## Figures and provenance

- `assets/pino-comparisons/electropolish2d_distribution.png`:1008x752,29898bytes.
- `assets/pino-comparisons/particle_corrosion2d_distribution.png`:1008x752,24802bytes.
- Existing two physical-field sheets stay byte-identical, including the
  explicitly labeled phase-only instantaneous spatial norms in their panels.
- Total four comparison PNGs:1,104,994bytes(1.05MiB), each<1MiB,total<2MiB.
- Violin KDE uses Scott bandwidth/256points, confined to sample min/max, with
  linear percentage axes. All64points are shown using fixed golden-ratio
  horizontal offsets by sample index. No jitter changes an error value.
  White dot=median; thick bar=IQR; text=overall sample mean. KDE width is display
  density, not extra observations or a confidence interval. Same method colors
  as previous comparisons; no y-axis break or per-method rescaling.
- Manifest and rendering receipt are schema v3. Exact direct model/config/
  reference/prediction paths, uniform-time indices, two channels and aggregation
  weights, KDE settings, per-channel numerator/denominator norms, all256sample
  values and final statistics are stored. All8historical configs remain.
- Before replacement, both READMEs, root/assets guide, renderer, v2manifest/
  receipt and all4PNGs were copied to
  `/root/autodl-tmp/tf-logs/pino-derivative-eval64-20261111-01/before_final_violin`.
  Only the2superseded convergence PNGs were retired from the curated asset
  folder, after their archived bytes were checked. No weights, raw fields or
  experimental history were deleted. Archived convergence paths are explicit
  absolute paths in the v3manifest.

## Verification

- Hand-constructed unequal-channel/time fixture checks the stated reduction
  order and distinguishes it from global concatenation and per-time averaging.
  Independent rescaling of each channel leaves its relative error invariant.
  Zero reference norms and nonfinite arrays are rejected.64sample aggregation
  matches the known arithmetic mean.
- All256final sample metrics recomputed from physical arrays; original field
  metrics still match the archived evaluation. Every plotted value equals the
  mean of its two normalized channel errors. Figures contain only seed42.
- Both new violins visually inspected. No off-range density tail, clipped
  outlier or missing sample. Two original field PNGs byte-identical to archive.
- Full CPU regeneration repeated once: all4PNGs reproduce byte-for-byte.
  This verifies the plotting pipeline, not a repeated numerical training test.
- AST/JSON/image budgets and `git diff --check` pass. All269local Markdown
  link targets resolve; only seed42appears in displayed metric records. The
  original failed audit journal is preserved. Existing SOTA PNGs and setup
  SVGs are unchanged.

Reproduce from repository root using the existing external arrays:

```bash
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/render_pino_derivative_comparisons.py --case all
```

Use `--case electropolish2d` or `--case particle_corrosion2d` for one case. No
new inference or solve is required. The v3receipt retains all plotted values
even when raw external arrays are unavailable.

## Session changes

Updated renderer, both provenance JSONs, root and both example READMEs,
assets guide and the development guide. Replaced2convergence PNGs with2violin
PNGs and retained2field sheets. Memory: `docs/DECISIONS.md`, this archived task
and `docs/PROGRESS.md`. User's latest channel-averaging instruction supersedes
the interrupted phase-only plan; that earlier decision remains history, not
the current metric. No Git/history operation authorized or performed.
