# Integral nonlocal differentiation control for the channel operator

## Goal and acceptance

User requests replacing channel GMLS derivatives with nonlocal operators similar
to the other examples and measuring accuracy. Preserve fixed-normalization
default/control, network/physics/reference/sampling/optimizer and old results.
Derive and independently validate boundary-compatible integral stencils before
one finite cold-training comparison. Report field/minority/shape errors, timing
and any changed numerical assumptions. No Git/history/default promotion/cleanup.

## Context

- Prior default task: [fixed default](channel-fixed-normalization-default.md).
- Fixed control: /root/autodl-tmp/tf-logs/channel_nsch2d/20261102-5090-gmls-time40-jvp-stronger-online-width128-01.
- Current derivatives: degree3/36-neighbor GMLS PDE/BC gradients and Laplacians,
  AD streamfunction curl, physical time JVP and reverse parameter AD.
- Existing channel_nonlocal_probe.py demonstrates integral correspondence
  gradients and positive regional Laplacians, but exposes the incorrect bare
  natural-flux closure for no-slip viscous layers/open boundaries. Do not
  silently reuse that closure or change the declared continuum boundary model.
- Existing dirty adaptive/default/terminology work must be preserved.
- Hardware checked: idle RTX5090 32GiB, approximately34GiB free disk. No FEM
  generation needed; saved stronger local FEM remains the independent target.

## Status, blocker and next action

Complete; no blocker, queued experiment or remaining action. Owner PID527773
and canonical train PID527780 exited after one cold5000 and six successful owned
stages. Added an explicit
integral_moment preset, radial horizon.075/capacity128/degree3, and QR moment
correction of the original radial gradient/Laplacian coefficients. Derivation
and sources are recorded in DECISIONS and the example README; boundaries retain
their original physical constraints, with no natural-flux substitution.

Three initial CPU tests passed1.754s: truncated/corner/off-node polynomial
reproduction and independent constrained-solve coefficients; smooth refinement
and quadratic no-slip layer; invalid support/capacity/config failures. Initial
independent test mistakenly counted padded node0 duplicates as real neighbors;
fixed the test's support mask without changing the operator. Extended actual
timeJVP-loss gradient finite-difference test and two existing GMLS guards are
running. One legacy payload check initially used rounded FP32 amplitudes and
correctly failed; rerunning against original prediction FP64 amplitudes.

The input-only220-geometry screen, seven-update production probe and all six
focused tests have passed. Actual cold-state/unchanged-input parity passed.
Matched curves/contours are generated and inspected, final results recorded
below. Default, FEM reference and existing results remain unchanged.

## Qualified finite launch

- Six tests PASS158.862s, including actual timeJVP loss parameter gradient versus
  finite differences and two unchanged GMLS guards. All52 config factories import
  and serialize; all original GMLS evaluation arrays reproduce byte-for-byte
  when using the original FP64 geometry amplitudes.
- Input-only screen PASS382.523s for16initial+204online constructions, maxQR
  condition250.930, maxpolynomial error4.911e-12; no PDE/FEM solve. Receipt:
  /root/autodl-tmp/data/channel_nsch2d/integral-stronger-input-screen-20261102-01.json.
- Production7-update probe PASS: first step352.745s including compile; warm
  mean.429631s; all full40-time gradients finite, fresh random time replay exact,
  actual peak8467511296bytes (~7.886GiB). No fitted probe state exported/imported.
  /root/autodl-tmp/data/channel_nsch2d/integral-stronger-resource-20261102-01/result.json.
  Explicitly corrected a legacy descriptive space='unchanged_GMLS' metadata
  label to integral_moment, preserving the correction provenance in that result;
  actual rebuilt inputs/saved config were integral throughout.
- Single cold5000 launched through canonical train.py and finite owner with
  --operator-control. Name:20261102-5090-integral-time40-jvp-stronger-online-width128-01.
  Process journal: /root/autodl-tmp/data/channel_nsch2d/20261102-5090-integral-time40-jvp-stronger-online-width128-01-process.json.
  Owner logs/stage logs share that prefix. Remaining stages: checkpoint replay,
  phase metrics, droplet shape, field gallery, matched actual-input comparison,
  and240fresh continuous-time residual probes. Then plot matched curves/contours.
  No automatic second run, new FEM, default change or Git operation.

## In-progress diagnostics

- Owner's actual cold-state/input check passed: complete model/SOAP bytes and
  all initial geometry/encoder/quadrature arrays match the GMLS control; only
  the seven declared operator arrays change.
- Analytic derivatives of the actual compact smoothstep initial interface give
  gradient/Laplacian errors4.208%/15.141% for GMLS and12.138%/38.479% for the
  wider integral kernel. These are derivative diagnostics, not solution L2 or
  reference errors, and were not used to tune the fixed horizon. Report:
  /root/autodl-tmp/data/channel_nsch2d/integral-initial-interface-consistency-20261102-01.json.
- At500/1000/1500 steps integral phi10.634%/5.800%/5.031%, versus GMLS
  11.087%/5.324%/4.872%. Integral velocity/pressure are slightly better at these
  monitors; final budget/shape comparison remains pending.
- During training, extended only postprocessing diagnostics to label integral
  residuals correctly and evaluate its actual phase-flux branch. Numerical
  training files remain frozen. Actual updated validate_run.py/diagnose_phase.py
  snapshots are separately recorded in the run's postprocess_source directory
  and postprocess-source.json; initial training snapshots remain untouched.

## Final matched result

All six owned stages exit0: train5000, validate_run, diagnose_phase,
droplet_shape, phi_gallery and continuous_time_residual. Strict operator-control
audit passes actual initial model/fullSOAP, every time/condition/drawID, and all
unchanged initial/final/evaluation geometry/encoder/quadrature arrays. Online
audit rebuilds16final inputs and all198actual installed/sampled draw identities.
Operator arrays are intentionally different. No evolved labels or new FEM.

| Metric | GMLS control | Integral + moment correction |
|---|---:|---:|
| Pooled phi L2 (%) |1.9470|2.0714|
| Terminal phi L2 (%) |1.6625|1.8767|
| Positive-time velocity L2 (%) |3.0587|2.6915|
| Positive-time pressure L2 (%) |3.9455|3.8079|
| Positive-time chemical L2 (%) |19.5989|18.0335|
| Terminal minority fraction L2, area weighted (%) |4.8688|5.4998|
| Worst saved minority fraction L2 (%) |8.7984|8.7730|
| Terminal negative-phase IoU (%) |93.0197|91.6397|
| Terminal mean contour distance (% initial radius) |2.4662|2.9207|
| Warm update (s) |.288946|.425587|
| Cold training wall (s) |1926.685|2619.880|

The integral control retains comparable field accuracy, with slightly worse
phase/shape and better velocity in this one initialization/nominal geometry.
Its pressure difference is small. It is not evidence of stable cross-seed or
population advantages. Warm updates are47.3%slower; actual active peak is
9053896192bytes (~8.432GiB), so the32GiBGPU fits. No default/SOTA promotion.

Peak deformation at t4 remains too flat: x/yspan integral1.42038, GMLS1.40958,
FEM1.29955. This time is explicitly included in the inspected contour figure.
Terminal shape differences and localized errors remain visible rather than
being inferred solely from the small background-dominated global phase L2.

Replay maxfieldgap2.861e-6, maxwallvelocity3.168e-7, maxfluxerror8.033e-7,
AD physical divergence3.725e-9. Approximate integral discrete divergence reaches
.04487 and is a diagnostic, not the AD-curl constraint or a training loss.
Both physical continuity and the stated residual derivatives must be described
accurately; do not call this an entirely spatial-AD-free or exactly conservative
nonlocal incompressible model.

Independent240continuous-time/full-spatial probes on nominal/final-pool geometry
complete. Total normalized MSE integral .003504/.003378 versus GMLS
.003420/.003231; nominal CH .001126 versus .001132 and chemical .000846 versus
.001045. Different derivative operators define different losses; these totals
are not a local continuum error comparison or a stability certificate.

Training-source byte checks:89case Python files and all3shared runtime/network
files match their actual snapshots; the2updated diagnostic sources have separate
explicit actual snapshots. Finite-owner source matches. Source/state recovery
uses direct paths and byte comparison, no hashes. Six focused tests and the
finite probe remain valid; no unchanged expensive suite was repeated.

## Artifacts and completion

- Run: /root/autodl-tmp/tf-logs/channel_nsch2d/20261102-5090-integral-time40-jvp-stronger-online-width128-01.
- Comparison: /root/autodl-tmp/data/channel_nsch2d/integral-comparison-20261102-01
  (convergence.png, contours.png, t4-shape.json, provenance.json, plot_source.py).
- Full independent-time report: run/continuous-time-residual/validation.json.
- Strict comparison: run/operator-control-validation.json.
- Compact [receipt](../channel-integral-nonlocal-control-validation.json).
- Changed numerical/example files: integral_operators.py, integral_validation.py,
  explicit integral config, data/loss/problem selectors and train summary;
  online/validate_run/diagnose_phase/diagnose_time diagnostics. Extended finite
  owner/probe controls, added tests and comparison plot helper. Updated stable
  README/DEVELOPMENT and artifact index; existing dirty adaptive/default work
  was preserved. Figures/weights/data remain external and untracked.
- Final syntax/JSON/link/source/figure/diff checks pass. GPU idle. No Git/history,
  cleanup, reference replacement or further training action. Archive complete.
