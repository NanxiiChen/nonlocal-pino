# Reproduce the five PINO derivative comparisons

The comparison is between **trained PINOs** using nonlocal operators, local
spatial automatic differentiation, and a finite-element weak residual.
An independently solved FEM reference is a different object; it is never the
third trained model. Original SOTA configurations remain separate from these
controlled experiments.

Spinodal is the explicit exception: on2026-11-15the user omitted Local AD after
resource measurements. Its completed comparison contains only Nonlocal and
FEM PINO, with `comparison_kind=spinodal_nonlocal_fem` and `nonlocal_fem`
filenames. An explicit omission in the final audit is required; a missing run
alone never changes the method set. The other four cases retain three methods.

Use [`PLOTTING_GUIDELINES.md`](PLOTTING_GUIDELINES.md) for the approved visual
style. The common entry point is
[`scripts/render_pino_three_method.py`](../scripts/render_pino_three_method.py).
It reads saved results on CPU, recomputes final physical-field errors, and writes
editable PDF/PNG pairs. It does not train, infer, solve a reference, select a best
checkpoint, or overwrite the curated README/site figures.

## Data and metric scope

Each panel shows a separate physical field. For each sample, first take the
relative norm over the declared space-time evaluation array; then average these
sample errors for the convergence curve. The violin retains one error per
sample. Do not pool different channels into one norm or average relative errors
computed independently at each time.

| Example / `--case` | Final weights | Convergence records / cohort | Final distribution / reference |
|---|---:|---|---|
| Ceramic corrosion / `ceramic` | 20k | 21 records, 4 local-FEM development geometries | 64 local-FEM geometries; phase and concentration |
| Electropolish / `electropolish` | 10k | 41 records, all 64 local-FD test inputs | The same 64 inputs; phase and concentration |
| Melting / `melting` | 10k | 41 records, all 64 native-reference test ellipsoids | The same 64 ellipsoids; phase |
| Spinodal / `spinodal` | 50k | 101 records, 8 reused local-FEM geometries | The same 8 geometries; phase and chemical potential |
| Coupled two-phase flow / `channel` | 20k | 41 records, 1 local-FEM development geometry | 3 reserved local-FEM geometries; phase, velocity, pressure and chemical potential |

All declared training arms and their final field audits are complete, including
the explicit Spinodal AD omission. Consult the [suite task](tasks/pino-three-method-suite.md) and
[direct-path inventory](../artifacts/pino_three_method_suite.json) for execution
state. The renderer requires a passed final audit and complete actual records;
missing declared methods or steps are errors, not permission to interpolate observations.

New controls evaluate their **complete existing test cohort at every evaluation**,
including step zero. Electropolish and Melting evaluate every 250 updates;
Spinodal evaluates every 500. References remain read-only and cannot enter the
loss, optimizer or sampling payload. Report final fixed-budget weights, without
choosing checkpoints by these curves. Repeatedly inspected test cohorts are not
blind validation. Already running channel and completed ceramic experiments
retain their historical monitoring scope. New total timings include full-test
evaluation and cannot be inferred from older subset-monitoring wall times.

Preserve these qualifications in both README and website captions:

- Ceramic convergence uses the historical volume/time-weighted development
  metric. Its final distribution uses equal-node space-time norms. The two are
  explicitly labeled and are not the same test statistic.
- Electropolish uses the 21 uniformly spaced saved times from 0 to 1 on the
  common local-FD grid. Extra early-time diagnostic snapshots are not given
  extra weight in the primary metric.
- Melting's new 64-case distribution uses the native nonlocal reference. The
  accepted independent local CG2 check for case 005 is reported separately;
  one local case is not a local-reference distribution. The original strict
  local-refinement gate remains unpassed.
- Spinodal's local reference must pass its independent space/time and physical
  checks before any method comparison. The primary chemical-potential error
  includes the explicitly declared analytic initial trace; retain the separate
  positive-time chemical diagnostic. These eight geometries have been reused
  for monitoring and are not a blind test set.
- Channel phase uses all 13 saved times; velocity, pressure and chemical
  potential retain the established 12-positive-time convention. Velocity uses
  a joint vector norm. Three reserved cases support a descriptive violin, not
  a precise population density. The historical nonlocal long run used the same
  GPU model but a different physical GPU UUID; do not describe its historical
  wall time as a new same-device run.

## Cold training wall-clock cost

Primary training cost covers the complete cold process: imports/device and model
initialization, in-process input/operator preparation, compilation, all updates
and rolling-pool waits, in-training evaluation, final output/checkpoint writing,
and teardown. Use the parent's timer from before process creation through exit;
warm-update measurements are auxiliary and must not replace this total.
Separately prepared input/reference banks, qualification probes, and separately
launched post-training evaluations/audits/plots are outside this interval.

| Example | Updates | Nonlocal | Local AD | FEM PINO |
|---|---:|---:|---:|---:|
| Electropolish | 10,000 | 15.63 min | 18.27 min | 15.44 min |
| Melting 3D | 10,000 | 6.89 min | 10.37 min | 7.35 min |
| Ceramic corrosion | 20,000 | 26.81 min | 28.47 min | 21.01 min |
| Two-phase flow | 20,000 | approximately 9.18 h* | 10.16 h | 8.03 h |
| Spinodal | 50,000 | 2.2464 h | Omitted after resource probes | 1.9035 h |

*The historical Nonlocal channel interval is reconstructed from external UTC
records (33,044.11 s), with approximately ten-second exit polling plus monitor
work and a timestamp immediately after process creation. It is approximate,
not a strict sub-second pre-spawn timer, and uses a different physical GPU from
the current AD/FEM controls. Ceramic's three controls share an older physical
RTX PRO 6000; Electropolish/Melting and current channel AD/FEM use the current
RTX PRO 6000. Same model name alone does not establish same-card timing.
The other table values use recorded pre-launch/post-exit parent timers.
Spinodal uses this same current physical GPU for both completed controls. Different examples retain their
declared monitoring cohorts; these totals include that evaluation workload.

For the two 10k temporal controls (both nonlocal), complete cold times are:

| Example | Spatial attention + decoder FiLM | Joint space-time query |
|---|---:|---:|
| Electropolish | 15.63 min | 23.65 min |
| Melting 3D | 6.89 min | 14.99 min |

[Exact parent journals, stage identities, seconds and timing scopes](../artifacts/pino_training_wall_costs.json).

## Spinodal cross-reference norms

After both50k methods pass the original final audit, run the independent CPU
reduction below on their saved physical fields. It reports equal-node norms
against both native and local FEM references, with historical physical-area
weights as a separate diagnostic. It verifies matching coordinates, times,
geometry and cohort, and reproduces every audited local per-case/channel value.
No inference, training or reference solve is performed. The final audit must
exist and pass; this command does not replace that gate.

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m docs.tasks.spinodal_reference_norms \
  --final-report /root/autodl-tmp/tf-logs/spinodal-nonlocal-fem-20261115-01/final.json \
  --output artifacts/spinodal_final_reference_norms.json
```

Primary norms include all saved times. The positive-time chemical diagnostic
separately handles the different initial chemical-potential conventions; never
silently drop the initial trace from the main result. Do not subtract reported
relative-error magnitudes to attribute the native/local discrepancy. Qualification
against all8old SOTA fields is recorded in
[`spinodal_reference_norm_qualification.json`](../artifacts/spinodal_reference_norm_qualification.json).
The completed new fields pass the same independent reductions in
[`spinodal_final_reference_norms.json`](../artifacts/spinodal_final_reference_norms.json).
The selected two-method figures are at
`/root/autodl-tmp/tf-logs/spinodal-nonlocal-fem-figures-20261115-02/spinodal/`;
the preceding failed render is retained. Its PDF-edit checker lacked the
Nonlocal/FEM legend format, now tested alongside the existing formats;
training data and numerical results were unaffected.

## Curves, intervals and violins

All methods use the same NPG colors across figures: Nonlocal C0, Local AD C1,
and FEM PINO C2. Thin solid lines show actual recorded case means; thicker solid
lines show log-LOWESS trends. The 401-point display grid is a fitted curve, not
401 checkpoints. Final tables and violins always use unsmoothed errors.

For histories with multiple cases, resample **complete case trajectories** so
that temporal dependence is retained. Enumerate all ordered resamples for two
or four cases; use 2,000 fixed-seed resamples for larger cohorts, including
eight and 64 cases. Reuse the same
sample-index draws across methods. The shaded 2.5–97.5 percentile band is
conditional on this empirical cohort and one training trajectory; particularly
for two or four cases, it does not establish population or cross-seed 95%
coverage. It does not correct smoothing or model-selection bias.

Channel's one-case history retains the previously approved circular-block
residual bootstrap: span 0.22, block length 4, 2,000 replicates, seed 20261114.
This is an approximate pointwise trend interval under smooth-trend and
short-memory residual assumptions, not uncertainty across geometries or seeds.
Other spans are 0.30 for ceramic, 0.22 for Electropolish/Melting, and 0.12 for
Spinodal; they use at least four neighboring records in each local fit.

Violins use all actual samples, Gaussian KDE in log10 error, Scott bandwidth,
three-bandwidth tails, equal maximum width, and internal quartiles. They do not
multiply errors by 100, manually restrict y limits, add artificial samples, or
jitter identical values. Undefined log/KDE cases stop with an explicit error.

## Reproduction

Run from the repository root with `.venv`. The current machine's PDF-audit
dependency is at `/tmp/pino-pdf-audit`; Helvetica is loaded explicitly from
`/root/.local/share/fonts/pino/Helvetica.ttf`, and the embedded-font checker is
`/tmp/pino-poppler-audit/usr/bin/pdffonts`. Missing dependencies/font files must
be installed or supplied explicitly, never silently replaced.

Use a new output directory for each invocation. For the already completed
ceramic comparison:

```bash
PYTHONPATH=/tmp/pino-pdf-audit OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m scripts.render_pino_three_method \
  --case ceramic --output /tmp/ceramic-three-method-figures
```

For the other cases, use the corresponding **passed final report** below with
`--comparison`. These paths identify the queued comparisons; the command must
not be run as a completed comparison before its report exists and passes.

| `--case` | `--comparison` |
|---|---|
| `electropolish` | `/root/autodl-tmp/tf-logs/electro-threeway-20261114-03/comparison.json` |
| `melting` | `/root/autodl-tmp/tf-logs/melting-threeway-20261114-04/comparison.json` |
| `spinodal` | `/root/autodl-tmp/tf-logs/spinodal-nonlocal-fem-20261115-01/final.json` |
| `channel` | `/root/autodl-tmp/tf-logs/channel-sampled-ad-20261114-03/comparison.json` |

For example, after the channel AD final audit passes:

```bash
PYTHONPATH=/tmp/pino-pdf-audit OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m scripts.render_pino_three_method \
  --case channel \
  --comparison /root/autodl-tmp/tf-logs/channel-sampled-ad-20261114-03/comparison.json \
  --baseline-comparison /root/autodl-tmp/tf-logs/channel-fem-full-20261114-01/comparison.json \
  --output /tmp/channel-three-method-figures
```

Only channel needs `--baseline-comparison`, for its retained nonlocal/FEM pair.
Ceramic preserves the previously approved three-model selection. It does not
silently switch to a newer default model.

Each output directory contains the PDF/PNG pairs, actual observations and
bootstrap arrays, plus `provenance.json` with direct checkpoint/configuration/
reference paths, field-norm checks, cohort/time scope, font/PDF verification,
and the exact command. Multi-field figures are 170 × 60 mm; the one-field
Melting figure is 85 × 60 mm, with editable 6–7 pt text.

Inspect final figures and their metric qualifications before copying selected
assets into the repository and updating both README and local website. Keep raw
fields, weights and alternate figures external. Publication and Git operations
require separate current-turn authorization.

For a visually checked, completed derivative comparison, copy the four selected
files and append its compact direct-path record with:

```bash
.venv/bin/python -m scripts.publish_pino_three_method \
  --provenance /tmp/channel-three-method-figures/provenance.json \
  --visual-review 'Legend, labels and all field panels inspected; no clipping or overlap.'
```

Use the actual reviewed output path and an accurate review note. This updates
only local assets and `artifacts/pino_three_method_figures.json`; it does not
edit captions, train, select a checkpoint, commit or deploy. It requires the
declared final budget, all declared methods, complete cohorts and existing direct
inputs, and checks file sizes/byte identity before adding a case. Repeating the
same selected input is read-only; a different replacement is rejected rather
than silently overwriting a published result. Update the example README and
website chapter together, run `npm run build` from `website/`, and check their
figures, metric qualifications and links. Ceramic retains its separate accepted
component manifest and is not replaced by this helper.

The Spinodal two-method figure pair uses the same derivative collection budget
and publisher, retaining Nonlocal C0 and FEM PINO C2. Its final audit is produced
by `docs/tasks/spinodal_nonlocal_fem_audit.py`; the existing `--case spinodal`
renderer reads that explicit scope. All8reserved cases at all101actual records
and both final50k models remain mandatory. The two-method suffix and omission
reason are retained in the publication manifest and website checks.


## Nonlocal time-feature fusion controls

Electropolish and Melting reuse the new nonlocal10k arm above as the spatial
cross-attention + decoder-FiLM baseline. Each adds a cold10k joint(x,t)query arm
without FiLM. Both monitor all64test inputs at every250updates, including zero.
Melting first extends its native reference bank; the original files and accepted
single local-CG2case005 are preserved. Never describe this as a64case local-FEM
comparison. Shared backbone initialization and every actual training draw are
audited; different temporal-module capacities are explicitly reported.

The two methods use NPG C0/C1 and the same solid raw curves, LOWESS and paired
whole-case-trajectory bootstrap as the derivative study. Component violins use
final unsmoothed64sample errors. These are repeated monitoring cohorts and one
initialization, not blind tests or cross-seed uncertainty. The scalar overall
error first averages channels within each sample, then samples; component
panels retain their own physical channels.

After the audit passes, render with:

```bash
PYTHONPATH=/tmp/pino-pdf-audit OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m scripts.render_pino_time_fusion \
  --comparison /root/autodl-tmp/tf-logs/pino-time-fusion-20261114-02/electropolish-comparison.json \
  --output /tmp/electropolish-time-fusion-figures \
  --x-axis cold-wall-time \
  --clock-evidence artifacts/pino_time_fusion_wall_clock.json
```

Use `melting-comparison.json` for the second case. The finite owner queues
resource probes, training, independent final audits and external PDF/PNG previews.
Do not promote synthetic layout qualification data or incomplete run histories
to the README or website. Full-process times include complete-test inference;
synchronized warm updates and peak active arrays are reported separately.

After inspecting the actual completed plots, select the four local assets with:

```bash
.venv/bin/python -m scripts.publish_pino_time_fusion \
  --provenance /tmp/electropolish-time-fusion-figures/provenance.json \
  --visual-review 'Use a truthful note describing the actual visual inspection.'
```

`artifacts/pino_time_fusion_figures.json` retains both methods, their shared
initialization/sampling checks, capacities, timing and separate local diagnostic.
The two-case collection has a2MiBbudget and1MiBper file. An empty selection does
not claim a completed experiment. Update both example captions and website
chapters after selection, then build locally; this command does not deploy.

## Cold wall-clock axes for the temporal studies

The selected two temporal convergence figures use real elapsed seconds, not
steps or steps multiplied by a warm-update estimate. Other derivative figures
retain their existing step axis. All original per-case errors remain unchanged.
The final existing observation is placed at the measured complete parent-process
exit, including final saves and teardown. No observation is inserted at time zero.

The historical monitor clocks start inside the training function. Their offset
is reconstructed using the original launch log's filesystem birth time (created
before Popen) and the final metric-file write time. Independently checking the
first metric-file creation against its trainer counter gives discrepancies below
0.004 s in all four runs. This is a consistency check, not a bound on absolute
clock accuracy: launch and metric serialization overhead remain, so the plotted
intermediate times do not claim subsecond precision. Electropolish additionally
counts the following local-monitor evaluation; Melting's counters already include
evaluation. Only the final point uses the directly measured complete-process total.

[`pino_time_fusion_wall_clock.json`](../artifacts/pino_time_fusion_wall_clock.json)
freezes original file timestamps, parent stage, counters, reconstruction and
plotted coordinates. Rendering reads this evidence rather than reinterpreting
timestamps of subsequently copied files. Initial calibration tolerance is 0.1 s;
inconsistent/missing evidence is an error. `--x-axis step` explicitly reproduces
the old axis and takes no clock-evidence argument.

For a reviewed replacement, `scripts.publish_pino_time_fusion` accepts
`--replace-provenance` with the exact prior selected provenance. It requires
unchanged methods, weights, samples, steps and errors, retains the superseded
source record, and checks the same asset budget. It cannot replace another
comparison kind. Cold wall times include all training-process work; offline
reference banks and later independent audits remain separate.
