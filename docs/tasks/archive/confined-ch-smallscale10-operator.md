# Fixed small-scale 10-mode geometry operator: Adam/SOAP control

Date: 2026-10-30. Status: COMPLETE (experiments and architecture assessment).
The harder-case accuracy remains insufficient. No further run is queued.
No Git history operation was authorized or performed. Local FEM is deferred.

## User goal and final scope

Use one fixed small-scale 10-mode ambient initial field, with random two-hole
geometry as the only input parameter. The user corrected an earlier mixed-band
interpretation, then explicitly prioritized nonlocal references over FEM.
Complete a fresh 512-domain physics-only operator run, compare SOAP with Adam,
and assess modest architecture growth. Do not equate a completed experiment
with a solved accuracy problem or select a checkpoint by validation minimum.

## Formulation and unchanged scheme

T=3, beta=1, epsilon=.06, mobility=.02, mesh h=.035, visible regional horizon=.11.
Fixed IC: .036 sum_j cos(pi*k_j*z+phase_j), z=(x-center)/axes, amplitude bound .36,
shortest wavelength .235294. Directions/phases are explicit in source preset.
The same ambient function is restricted to each domain, without a wall envelope.

Generic heterogeneous boundary/domain point streams, area-aware conditioning,
shared spatial preparation/time JVP, hard analytic IC and physical-volume mass,
chemical elimination and one frozen biharmonic-metric PDE loss. No time windows,
reference-field training labels or geometry-specific attention bias. Adam/SOAP
both use B4, Nt4 plus t0, FP64/ordinary, 20000 optimizer calls, LR .001*.95**(i/1000)
with 1e-5 floor. Seed42 is common runtime provenance, not a tuned strategy.
SOAP b1=.9, b2=.99, precondition frequency5, max dimension128, no1D, clip10.
The operator loop does not run the inherited single-IVP L-BFGS settings.

## References and preparation

Native source: /root/autodl-tmp/data/confined_ch2d/operator_geometry_smallscale10_v1
Training bank: /root/autodl-tmp/data/confined_ch2d/operator_geometry_smallscale10_512_v1

- Fresh 24 independent ordered-graph/BDF references PASS in340.249509s; maximum
  phase/chemical time-refinement differences1.0811e-9/9.3878e-10.
- Native terminal relative spread across8geometries on4627common queries37.79%.
- Six-worker bank preparation PASS125.764312s,6524920765bytes (~6.08GiB),496new
  operator-only inputs and24fresh labeled monitor/validation inputs. New inputs
  require no PDE/FEM trajectories. Reserved accepted draws16..23 become512..519;
  training uses0..15,24..519. Future fields never enter optimizer payloads.
- User deferred FEM. Owned workers/parent were stopped and absence verified
  before GPU work; coarse/fine-dt partial fields retained without a local gate.
  Selected512preset explicitly requests no local indices. Source preset retains
  (16,17)foroptional later study; no FEM job is queued.

Native-only infrastructure change: empty explicit local_validation_indices means
not_requested; precompute does not read/copy FEM files, evaluator reports
not_evaluated rather than NaN. Nonempty legacy requests retain mandatory passed
FEM checks; --local rejects an empty request. Focused two-test control PASS6.581s,
including required-local missing-file failure. Raw test log:
/tmp/confined-smallscale10-native-only-tests.txt. Existing10mode analytic
Laplacian/component check previously passed3.805s; unchanged checks not rerun.

## Cold-run results

Run root: /root/autodl-tmp/tf-logs/confined_ch2d/
Adam: 20261030-confined-geometry-smallscale10-01
SOAP: 20261030-confined-geometry-smallscale10-soap-01

| Metric | Adam | SOAP |
|---|---:|---:|
| Native8mean phase pooled5snapshots (%) |51.2599 |39.5614 |
| Native8mean phase terminalT3 (%) |50.7835 |39.1695 |
| Native16train-monitor phase pooled (%) |50.9591 |37.9941 |
| Native8mean chemical pooled (%) |47.5280 |37.7076 |
| Cold wall (s) |1435.614 |2030.369 |
| Setup (s) |118.835 |119.007 |
| First compiled update (s) |94.309 |201.786 |
| Warm synchronized update after100 (ms) |42.785 |69.817 |
| Geometry response error (%) |91.7112 |80.0706 |
| Geometry response amplitude recovery (%) |28.4231 |41.9528 |
| Geometry response correlation |.42165 |.63747 |

SOAP relative pooled-phase error reduction22.8219%. First matches both Adam
final phase metrics at8500calls/1045.747s (~17.4min vs23.9min). Full SOAP budget
is slower, but time to that accuracy is shorter in this pair. Last5k phase
55.07->51.26% (Adam),44.00->39.56% (SOAP); slopes-.729/-.824pp per1000calls.
User's interpretation is supported: slow incomplete fitting, not a proven
stationary trap or a demonstrated model-capacity floor. Final-budget checkpoints
are used in both. One cold pair/eight reused development geometries do not
establish optimizer stability or local-PDE accuracy.

Both initial parameter sets and saved numerical training sources match exactly.
All80000 actual condition draws match, every training domain visited123..201times,
no held-out draw or evolved optimizer input. CPU24field replay maximum phase/mu
differences4.8295e-15/3.5527e-15 (Adam/SOAP). Independent raw metric replay agrees;
IC/mass and sampled5timeenergy checks pass, but do not certify field accuracy.
Geometry-response rows center all trajectories across geometries on4627common
queries; they have a different denominator from physical-area field L2.

## Architecture assessment: measured counts, untrained alternatives

Current width64,heads4,conditionblocks4,decoderlayers3, spatial Fourier24 atscale1,
time Fourier4,outputscale.1. Inputs are boundary(x,y,nx,ny,type) and domain(x,y,phi0).
224705trainable scalars plus48fixed Fourier entries. Width96:493729trainable;
width128:867201. Width96preset is prepared but NOT trained or memory-benchmarked.

Temporal input angles are j*pi*t/T. At4frequencies, max angular frequency drops
16.755->4.189rad/s when T changes.75->3. time_features16 restores the old maximum
physical input frequency and adds1536weights (226241total). Nonlinear MLPs are
not band-limited: this is an optimization hypothesis, not a proven cause.
The existing --config.arch.time_features=16 provides this control; NOT trained.
No wider/time-feature accuracy claim or additional job is queued. Current
accurate older defaults remain unchanged; these are explicit research presets.

## Changes, artifacts and regeneration

Changed runtime: operator_problem validation, operator_reference empty-request
guard, operator_precompute local deferral, operator_evaluate no-local reporting.
Added selected10mode/SOAP/width96presets, focused test, plotting/audit receipts;
README/DEVELOPMENT/root navigation/artifact index updated. No optimizer-core,
physical equation, kernel or old checkpoint/path migration change.

See example README for canonical train.py/reference preparation commands.
Both run directories contain config/source/model_initial/model_final, actual
training_draws.npz,24predicted fields, raw losses/metrics, evaluation figures,
training_report and geometry_effect. SOAP comparison contains complete curves
versus calls and cold elapsed time, metrics.json and direct-source/input audits.
Generated data/media/checkpoints remain external, not Git payloads.

Archived helpers/receipts in this directory:
- confined-smallscale10-audit.py, -audit.json, -soap-audit.json
- confined-smallscale10-optimizer-comparison.py/.json
- confined-smallscale10-training-report.py, -architecture.json
- confined-smallscale10-precompute.json, -preview.json
- confined-smallscale10-symbols.py/.json
- confined-multiscale-preview.py, -preview.json, -preview-v2.json,
  confined-multiscale-initial-bands.py (superseded candidates)

For a replay, set PYTHONPATH=. JAX_PLATFORMS=cpu and BLAS/OMP1, invoke archived
confined-smallscale10-audit.py with --run, --prior pointing to
20261029-confined-geometry-richer512-long-01, --output and --config explicit module
examples.confined_ch2d.configs.train_operator_smallscale10_512[_soap].
Comparison helper requires --baseline/--candidate/--output; output directories
must be new. Saved configs/direct paths define templates; no hash loading.

## Preserved superseded candidates

T2/20mode and T3/24mixed-band previews were superseded by the user's fixed10mode
correction. Configs and external operator_multiscale_preview_v1/v2 and
operator_geometry_multiscale_v1 retained, no neural training/512bank for them.
The24modeFEMchecks finished before attempted stop; both passed. They cannot
validate the10mode case. Infinite-medium kernel-symbol diagnostic independently
matches2Dquadrature within1.71e-13;10mode symbol deficits9.08..17.26% exclude
walls/mesh/nonlinearity and do not uniquely explain local bias. User deferred
that issue; do not resume FEM or alter the horizon without a new instruction.

## Completion and exact resume

All requested native/optimizer experiments, source/input/checkpoint audits,
plots and architecture assessment are complete. Accuracy is still unsatisfactory;
possible future work is an explicitly measured time-feature or width control.
All research processes exited. No queued run, blocker, Git action or active task.

Final task-end verification: git diff --check PASS; 15 archived helper/receipt
files parse as Python/strict JSON and remain below1MiB; current report/config/
checkpoint/figure/audit paths exist. Width96 preset validates and differs only
in width. No unchanged expensive suite was rerun. All requested jobs have exited.
