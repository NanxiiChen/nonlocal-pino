# Fixed condition-library size comparison

## Goal
User selected Electropolish and Ceramic, fixed libraries of64/128/256/512inputs,
identical update budgets within each example, final relativeL2mean and sampleSD.
Investigate finite-library generalization versus the rolling/continuous defaults.

## Acceptance
- Nested fixed inputs from an unchanged physical prior; no positive-time labels
  in training and no overlap with the fixed evaluation cohort.
- Only condition library diversity changes. Keep time/core-halo resampling,
  PDE/output/BC/optimizer/architecture, common initialization and per-case budget.
- Final weights, equal-point complete-space-time norms per physical channel;
  arithmetic channel mean per sample, then64sample mean and ddof1standard deviation.
- Full cold process timing; direct input/config/source/weight provenance.
- Fixed shape/dtype and audited sampling; preserve existing defaults and unrelated
  local experimental files. No Git/history/publication authorization this turn.

## Context and scope
GPU: RTX PRO6000Blackwell Server,97887MiB,UUID85fe6412-636e-5d3c-d5f0-d1e9523c3faa;
idle at start. Ceramic is a true geometry-conditioned fast case (~44min/20k).
User confirmed geometries or initial conditions both qualify, and selected only
Electropolish+Ceramic with64/128/256/512inputs;1024is deferred. Electropolish
uses10kupdates, Ceramic20k, each preserving its established64case monitor.
Expected complete GPU training is about4hours on the current device, excluding
input preparation, resource probes and final audits. No Spinodal/Melting run.
Spinodal's earlier local/nonlocal-reference gap remains deferred, not this task.

## Status
Complete. All eight trainings and independent final64case audits passed;
the finite queue finished2026-11-16at03:37Asia/Shanghai. All owned trainers exited.
Final statistical/visual/font/editability checks passed2026-11-16. No new training
is queued. Ceramic128/256/512retain poor32%final convergence; audit success
confirms input/metric integrity, not solution accuracy. Cause remains unestablished.
Completion/provenance: artifacts/fixed_condition_library_results.json.

Owner PID857580; bundle /root/autodl-tmp/tf-logs/fixed-condition-library-20261115-01;
launch log /root/autodl-tmp/tf-logs/fixed-condition-library-20261115-01.launch.log.
Frozen numerical/source/config tree is bundle/source; do not edit running sources.
Input root: /root/autodl-tmp/data/fixed-condition-library-20261115-01.

## Exact resume point
No required experiment or figure work remains. Preserve all endpoints, including
poorly converged arms, and the completed journal. Never restart the finite owner.
Any future Ceramic convergence diagnosis should start from the retained128/256/512
loss histories, actual geometry/time/spatial draws and final fields; no cause is
currently established. Further training, promotion or publication is not queued.
Defaults remain unchanged; Git operations require current-turn authorization.

## Preparation and focused validation
- Confirmed two cases only: Electro10k/Ceramic20k; sizes64/128/256/512.
- Input root: /root/autodl-tmp/data/fixed-condition-library-20261115-01.
  Electro512accepted profiles took1.38s; Ceramic512nested known-input geometries
  took127.13s using4CPU workers. No PDE/FEM solutions generated. No evaluation
  overlap. Ceramic largest padded bank is677355520bytes (~646MiB).
- Fifteen focused tests passed in9.61s: finite-prefix field identity/replay,
  unchanged default online draws, shared prescribed fields, geometry preservation
  under real spatial rebuilding, FIFO schedules, complete halos and error handling.
  Test log: /tmp/fixed-condition-library-tests.log.
- GPU owner will freeze only the two examples, their corrosion dependency andsrc,
  then qualify largest512library workloads. Eight full trainers run serially;
  each is followed by final independent evaluation/input audit. Mean±sampleSD
  PDF/PNG renders only after all eight audits pass. No automatic retries.

## Diff-level changes and session checks
- Optional fixed geometry mode in Ceramic online producer/config; old rolling
  mode preserves schedule and all numerical operators. Its summary labels fixed
  geometries explicitly; only spatial versions refresh in this experiment.
- Optional Electro coefficient-library loader and branch in existing online
  prescribed-field sampler; runtime source/selection/summary record this mode.
- Input-only preparation, finite frozen queue, independent norm/sampling audit,
  final mean±SD PDF/PNG renderer, focused tests and research README notes.
- All15focused tests, nested bank byte/identity checks, modified-module compile
  and diff whitespace checks pass. Defaults are not edited. Unrelated Melting
  and earlier sampling investigations remain untouched.
- Largest geometry bank646MiB; expected8run training about4hours. No large raw
  artifact is added toGit and no files deleted.29GiBdisk was free at start.

## Largest-library GPU qualification
Ceramic512passed26actual finite optimizer updates plus all64reference evaluations.
Peak active GPU memory20.87345GiB; first update including compilation159.50s;
64case evaluation51.35s. This is a resource/update check, not a trained-error
result. Receipt: bundle/ceramic-probe.json. Next stage:probe-electropolish-512,
51updates to include the first spatial-pool replacement; logs in the bundle.

## Session handoff — formal training started
Both resource probes passed. Electro512completed51finite updates, spatial pool
replacement and full64case monitoring; peak active12.02426GiB. Ceramic512peak
20.87345GiB. Current stage is train-electropolish-64 in the existing owner journal.
Do not interpret short-probe errors as trained results. The actual first formal
run is /root/autodl-tmp/tf-logs/electropolish2d/fixed-library-20261115-01-electro-n64.
Follow the existing frozen owner; no repeated probes or duplicate trainers.
All modified modules compile;15tests and nested inputs pass; diff check passes.
No commit, push, default promotion or file deletion was performed. Memory files
updated:docs/PROGRESS.md,docs/DECISIONS.md,and this incomplete active task.

## Concurrent user-requested figure preview (2026-11-15)

Keep the background fixed-library owner running. Create five ranked main-figure
previews (best, middle, worst phase-only equal-point full-space-time L2), retaining
the published main-figure checkpoints and physical-field styles. Use all available
held-out cases; extend existing CPU-only final-checkpoint inference onto established
64-case Ceramic/Melting references where compatible. Spinodal has 8 reserved cases;
Channel has 3 reserved geometries (exclude development geometry).
Middle is the lower central actual sample after sorting by (error, sample ID).
Save direct input paths, all scores, selections and PDF/PNG externally. No training,
reference solves, default changes, publication or Git operations for this preview.
Resume: complete ranked preparation/rendering and visual/font/metric checks.

### Ranked main-figure preview completed

Five visually reviewed PDF/PNG pairs; 203 phase scores independently recomputed.
Direct checkpoints, all score lists, selected IDs, exact times and output paths:
`artifacts/pino_ranked_main_preview.json` and its external provenance files.

| Case | Cohort | Best / middle / worst IDs | Full-space-time phase errors |
|---|---:|---|---|
| Electropolish |64|30 /5 /56|0.00606960 /0.01391288 /0.03832229|
| Ceramic |64|50 /53 /30|0.02671394 /0.03490255 /0.04840499|
| Spinodal |8|1027 /1025 /1026|0.01707254 /0.04052802 /0.09980281|
| Channel |3|1 /2 /0|0.01429263 /0.01515234 /0.01986002|
| Melting |64|52 /33 /56|0.000704959 /0.000968559 /0.001323363|

All original main-figure models/reference methods retained. Ceramic and Melting
extended onto already available64case references; no new reference solve.
Spinodal/Melting references remain native nonlocal. Channel excludes development.
All figure widths170mm; fonts5–7pt, embedded TrueType, noType3; ordinary/formula
text edits in disposable PDF copies pass. Fields rasterized, text/contours vector.
All physical aspect ratios preserved. Selected Melting case52 has no positive
phase at final time and is explicitly labelled Melted; no fabricated isosurface.
Use Lorensen edge interpolation for all Melting surfaces: closed-edge checks pass
and maximum interpolated zero-level residual1.059e-6; no surface smoothing.

Melting CPU diagnostics did not pass original GPU replay tolerance: maximum
relative field difference4.377e-4 and score difference3.328e-4 across10cached
fields, despite frozen/currentCPU arrays being identical. CPU results are NOT
selected. User confirms GPU inference is appropriate; actualGPU inference64plus
three dense volumes took56.765s after backend setup. GPU replay across10cached
fields passes (maxabsolutephi1.431e-6); final selected median is33, notCPU5.
The short GPU overlap with the live fixed-library arm is recorded externally;
that arm's wall time is not isolated-performance evidence. GPU preallocation was
disabled;9050MiBwas free before inference. Background owner remains untouched.

Diff-level sidework: three new preparation/rendering scripts, preview artifact
manifest, and updated PROGRESS/DECISIONS/this task. No default, README, website,
asset or Git changes for this preview. Original and failed diagnostic arrays
retained externally; no cleanup performed. Focused selection/norm checks, changed
module compilation and diff checks pass. Figure subtask has no pending work.

Exact parent resume: inspect the existing fixed-library owner journal; last task-
end observation was `train-electropolish-256` with status `running`. Do not launch
a duplicate owner. Continue final independent audits and the library-size plot
when the existing queue completes.

### Accepted gallery integration (2026-11-16)

User accepts the five ranked previews and requests README/site updates. Replace
the existing five selected PDF/PNG pairs in place, update all example captions
and the overview, and retain raw arrays externally. Update publication provenance
and site checks to the explicit ranked manifest. Validate the complete local site
and all asset copies; preserve background owner and unrelated work. No current-
turn Git authorization was given, so this integration does not commit or deploy.
Status: updating local documentation, selected assets and website.

### Local README/site gallery integration complete

- Replaced exactly the five existing selected PDF/PNG pairs in assets/pino-main.
  Total7,022,408bytes; each below2MiB and collection below9MiB. All byte-identical
  to the accepted external exports; no raw fields or weights added.
- Updated five exampleREADME captions/alt text/reproduction links, root overview,
  website's five chapter captions/dimensions and overview, and asset/site guides.
- Manifestv2 records explicit source paths, original style/source records, sample
  IDs, cohort counts, phase scores and actual image dimensions. Copier and site
  builder validate rank metadata and file budgets. Historical results unchanged.
- `npm --prefix website run build`:9pages.395local links/anchors pass;10file
  source/repository/site identities and allPNGdimensions pass. Copier rejects
  malformed sample selection. Changed publisher compilation, node syntax and
  git diff whitespace checks pass. Figures were already visually reviewed.
- Exact provenance/checks: artifacts/pino_ranked_main_publication.json. No Git
  history operation or livePages deployment performed: user requested updates,
  without explicit current-turn commit/push authorization required byAGENTS§4.
- Memory updated:PROGRESS,DECISIONS,this active task. The gallery integration
  subtask is complete; no model/default/ongoing training changes. Resume parent
  work from the existing ownerjournal. Last observed stage: `train-ceramic-64` (`running`).

### Progress snapshot — 2026-11-16, Asia/Shanghai

All four completed Electro arms share10kfinal weights,64test samples and the
user's equal-point per-channel full-space-time norm, then channel/sample means.
SD below is across64samples (ddof1), not a seed confidence interval.

| Fixed inputs | Phase mean | Concentration mean | Channel mean | Sample SD |
|---:|---:|---:|---:|---:|
|64|0.12159633|0.13585157|0.12872395|0.03437927|
|128|0.02428838|0.02893670|0.02661254|0.01091263|
|256|0.02620127|0.03130804|0.02875465|0.01055959|
|512|0.01829047|0.02198254|0.02013651|0.00797700|

The64input arm is substantially worse;512is best in this single initialization.
128→256is not monotonic. Do not infer overfitting or a universal library-size law
without further evidence. Preserve all endpoints; no retuning/restarts.
Latest Ceramic progress is in bundle/progress-20261116-0043.json. Recent throughput
including monitoring is0.1214s/update, giving roughly2.80hours
remaining before final audits/figures. Continue existing queue; default15minute
polling cadence. This session changed only PROGRESS and this task, plus an
external progress snapshot. No code/default/Git/training mutation.

### Online-baseline clarification — 2026-11-16

User asks whether online/rolling conditions must yield the lowest error. Checked
the existing final10k weights and64sample metrics, without retraining:

| Initial-condition sampling | Phase mean | Concentration mean | Channel mean | Sample SD |
|---|---:|---:|---:|---:|
| Fixed512 |0.01829047288|0.02198254226|0.02013650757|0.00797700048|
| Matched online |0.01971288757|0.02356285882|0.02163787320|0.00779638416|
| Historical selected online main model |0.01546570901|0.01860404387|0.01703487644|0.00679751780|

Direct run roots under `/root/autodl-tmp/tf-logs/electropolish2d/`:

- Fixed512: `fixed-library-20261115-01-electro-n512`;
  `fixed_library_audit.json` records the independently audited final norms.
- Matched online: `electro-threeway-20261114-03-nonlocal`;
  final row of `local_metrics.jsonl`, step10000,64conditions.
- Historical main: `pino-spatial-pools-20261111-01-electropolish2d-s42-rolling_pool`;
  `spatial_evaluation64.json`, records[].local101_64.pooled_phi/pooled_c.

Matched configs differ only in run name and the added condition_library block.
Both model_initial.eqx and optimizer_initial.eqx are byte-identical. The historical
main also has identical initial states, but lacks the host_float64_shared input
evaluation option and all64 in-training comparison monitor; checkpoint logging
and method-control metadata also differ. Keep its better endpoint separate from
the matched comparison. All listed metrics use21regular times, equal-point
full-space-time norms and the same64local-FD reference cases. Sample SD is over
channel-averaged sample errors, not a cross-seed uncertainty estimate.

These experiments vary the initial-condition library; spatial core/halo pools
continue rolling in every arm. Online conditions offer broader coverage but
do not impose a finite-step accuracy ordering: revisitation and stochastic
optimization differ. The matched channel-mean gap is only0.15014percentage points;
one initialization cannot establish a stable ranking. No overfitting diagnosis
or new experiment is implied. Preserve ongoing Ceramic queue and defaults.

This session changes only this task and PROGRESS; source/config/weights and Git
history are untouched. Resume from the existing owner's journal, without
duplicating or restarting training. Final library-size figures remain pending
completion of the Ceramic arms.

### Exact initial-input implementation difference — 2026-11-16

Follow-up source inspection clarifies the previous shorthand "different initial
evaluation path": the physical initial condition was NOT changed. Historical
main and matched-online frozen `online_conditions.py` files are byte-identical.
Both retain six random Fourier modes, identical amplitude/frequency/height prior,
2000 FP32-rounded curve vertices, the ±16-segment signed-distance search,
tanh profile, concentration relation, physical coefficients and scales.

The old preset evaluates encoder images and prescribed IC/Dirichlet arrays in
the original GPU/JAX FP32 sampler. The comparison preset wraps that same sampler:
`comparison_sampling.prescribed_shared_fields` recomputes ONLY the encoder
sensors and common IC/Dirichlet data with NumPy FP64, then casts once to FP32.
Profile draws/acceptance and the variable-query values remain on the original
path. Model, residual and optimizer training remain FP32. This is not a different
initial-field formula, distribution, thickness or full-training precision.

Reason: earlier derivative controls with differing query layouts failed exact
input pairing (image discrepancy8.94e-7, IC discrepancy1.19e-6). Independent
double-precision checks differed by at most4.2e-6; they support roundoff, without
proving a particular compiler cause. The shared wrapper passed exact common-input
identity across64updates ×8profiles ×3layouts. Evidence already exists in
artifacts/electro_input_roundoff_diagnostic.json and
artifacts/electro_shared_input_qualification.json; no new GPU tests were run.

Other configuration differences are read-only64case monitoring, checkpoint
frequency and comparison metadata. Do not attribute the historical1.7035% versus
matched2.1638% final error gap to the input wrapper without an isolated ablation.
The historical model remains a valid same-physics reference point, but not an
otherwise identical control for the fixed-library study. Fixed arms and the
matched-online baseline all use the SAME wrapper. No defaults or running jobs
changed. This session updates PROGRESS and this task; diff whitespace checked.

### Ceramic progress — 2026-11-15T17:07UTC

User retains the Electropolish results and asks for Ceramic status. Existing
owner857580 and trainer937428 are alive; GPU utilization100%. No new job launched.
First arm (fixed64geometries) has reached13900/20000updates. Other sizes128/256/512
remain queued and will launch serially under the same owner.

All evaluations below cover64test geometries and use the requested equal-point
space-time channel norms, then channel/sample averaging:

| Step | Phase mean | Concentration mean | Channel mean |
|---:|---:|---:|---:|
|2500|0.27254664|0.29247206|0.28250935|
|5000|0.17080673|0.19476437|0.18278555|
|7500|0.08554778|0.10158806|0.09356792|
|10000|0.06120302|0.07306998|0.06713650|
|12500|0.05097295|0.06121439|0.05609367|
|13500|0.05448367|0.06549660|0.05999013|

Longer-term improvement with recent fluctuations;12500is the best observed
intermediate channel mean, not the selected final endpoint. Retain20kbudget.
Recent measured rate including evaluations is0.12195s/update; estimated current
remainder12.4minutes. Including three further cold runs/startups and final checks,
rough total remainder2.5hours. No causal claim about fixed-library generalization
until final audited endpoints are available. Default15minute monitoring cadence.
Sources: run metrics.jsonl/loss.jsonl and bundle journal.json; no GPU inference
or training mutation performed for this status check. Memory changes:PROGRESS
and this task; git diff whitespace check passes. Resume existing queue only.

### Ceramic64 complete,128 plateau — 2026-11-15T17:52UTC

The fixed64geometry run completed20000updates in2654.681451s (44.2447min),
measured from cold process start to process exit. Its14.4791s independent audit
passed separately: all64geometries sampled,1596spatial refreshes, exact final
slots reconstructed, no evaluation overlap or future-field training labels.
Final equal-point64case channel errors are0.04714028506(phi) and0.05682836595(c),
channel mean0.05198432551 and sampleSD0.00738840600.

The128geometry arm started17:19:44UTC, has13700logged updates and the following
64case channel-mean history:

| Step | Channel mean |
|---:|---:|
|2500|0.29342258|
|5000|0.30584323|
|7500|0.32439736|
|10000|0.32457606|
|12500|0.32301852|
|13500|0.32282789|

At13500, phi0.31442001903 and c0.33123575643. This is poor convergence so far,
not a completed endpoint. Read-only checks confirm byte-identical initial model
states and saved configs differing only in operator.geometry_count (64vs128).
Finite logged losses, live trainer/owner and100%GPUutilization show no crash;
they do not prove a cause for the error plateau. Do not infer that increasing
the library generally harms accuracy or silently discard/retry this arm.
Preserve20kbudget and queued256/512arms; next final audit remains automatic.

Recent inclusive rate0.12297s/update implies about13minutes for the active arm;
using the completed44.24minute cold run for each remaining arm gives roughly
1h40m until all trainings/audits/plots. Numerical defaults and running source
remain untouched. This status session updates PROGRESS and this task only;
diff whitespace check passes. Resume the same owner, no duplicate jobs.

## Completed results and delivery — 2026-11-16

Every entry uses final fixed-budget weights and64test cases; channel errors are
equal-point full-space-time norms, averaged over physical phi/c channels and
then samples. SD is across64channel-averaged sample errors, ddof1.

| Case | Inputs | Steps | Phase | Concentration | Mean | Sample SD | Full cold training seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
|Electropolish|64|10000|0.12159633|0.13585157|0.12872395|0.03437927|903.2544|
|Electropolish|128|10000|0.02428838|0.02893670|0.02661254|0.01091263|929.7993|
|Electropolish|256|10000|0.02620127|0.03130804|0.02875465|0.01055959|928.2373|
|Electropolish|512|10000|0.01829047|0.02198254|0.02013651|0.00797700|898.6339|
|Ceramic|64|20000|0.04714029|0.05682837|0.05198433|0.00738841|2654.6815|
|Ceramic|128|20000|0.31971686|0.33614643|0.32793164|0.00803954|2686.6736|
|Ceramic|256|20000|0.31658639|0.33307352|0.32482995|0.00789291|2673.4912|
|Ceramic|512|20000|0.31973747|0.33609240|0.32791493|0.00788010|2708.7729|

Cold timing is actual training-process start to exit, including initialization,
compilation, updates, rolling waits, monitoring, writes and teardown. Separate
input preparation/probes/final audits/figures are excluded. Earlier short Melting
GPU inference overlaps Electro256; do not claim isolated timing for that arm.

All four Ceramic input audits reconstruct final slots exactly, cover all declared
geometry identities, record1596spatial refreshes each, and verify zero evaluation
overlap/positive-time training labels. Initializations and spatial/time draws
match the first run under the declared audit. All stages passed; no replacement
seed, omitted result, retry, retuning or default promotion. The three large Ceramic
banks did not recover by20k. This is poor convergence under this protocol, not a
demonstrated generalization law or proof of a specific optimizer/implementation bug.

Final selected figure directory:
`/root/autodl-tmp/tf-logs/fixed-condition-library-20261115-01/figures-reviewed/`.
Files: `electropolish_ceramic_fixed_library_error.pdf` and matchingPNG, with
`provenance.json`. Panels(a)/(b) are Electropolish/Ceramic; four actual endpoints
and mean±sampleSD only, no smoothing or manually set ylimits. Original automatic
preview remains in figures/. Reviewed output removes example titles per style
guide.170×62mm, Helvetica6/7pt, NPGcolors, editable embeddedCIDTrueType, vector
only, PDF12232bytes/PNG100063bytes. Visual and disposable-copy text/label edits
pass; all512sample-mean values reaggregate exactly from audited channel errors.

Renderer now accepts both audited and complete journals for reproduction, while
requiring passed final-budget64case audits. A first local redraw invocation lacked
the existingPDF-check dependency path; rerun with the queue's /tmp/pino-pdf-audit
path passed. No package changes or GPU work. Reproduction command/environment,
all direct runs/audits and checks: artifacts/fixed_condition_library_results.json.

/task-end: fixed-library study complete and archived. Updated example study notes,
PROGRESS,DECISIONS,renderer and compact result manifest. No website/public asset
promotion, numerical/default mutation, Git history operation or live deployment.
Prior accepted ranked-gallery integration remains local and undeployed. No
scheduled monitoring remains for this completed queue. Diff check passes.
