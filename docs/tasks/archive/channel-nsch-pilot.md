# Mild geometry-conditioned NS--CH pilot

## Goal and acceptance

Implement and actually test the user's full NS--CH channel with mild velocity
and smooth independent upper/lower indentations, no holes or explicit offset.
Only geometry is parameterized. Keep fixed initial phase/materials/driving,
velocity no-slip and a pressure-referenced open outlet without an extra pressure
anchor. Preserve the old periodic prescribed-flow example. Canonical new
training entry must be train.py. Deliver measured fields/errors/timings rather
than a design-only result. Progress through fixed-geometry validation to a small
geometry-conditioned physics-only operator trial.

## Context and resources

User authorized implementation/training, not Git or cleanup. Prior conceptual
tasks specify full unsteady NS, not Stokes substitution.85.65GBreported RTX6000D
idle; disk8.9GiB free. Existing generated-media exclusions/direct artifact paths
apply. Use .venv for JAX and fenicsenv for independent mixed FEM, sequentially.
No subagents requested or permitted. Current other uncommitted changes are docs.

## Formulation and validation route

Matched-property incompressible NS with capillary force -phi*grad(mu), CH
transport and mu=A*(phi^3-phi)-kappa*lap(phi). All material coefficients positive
and fixed. Use a smooth stationary graph channel and common straight ends.
Velocity inlet has fixed smooth ramp; walls no-slip; open outlet prescribed
hydrodynamic traction. Phase inlet carrier value, neutral/no-flux solid walls,
declared natural outflow. No global constant-mass projection in this open domain.

Independent FEM uses mixed Taylor--Hood velocity/pressure and mixed phase/mu,
implicit full coupling. Check spatial/time refinement, flux, no-slip, pressure
and phase evolution. Training remains PDE/IC/BC-only; references are evaluation.
Spatial nonlocal second differentiation needs its own boundary consistency gate
rather than copying periodic FFT or regional velocity-Neumann operators.

## Status / blocker / next action

Completed initial implementation/experiment; coupled neural accuracy is not
achieved. No further run is queued. New example/problem/config/FEM and point-conditioned physics trainer are
implemented. All FEM processes have exited. BDF2 space/time gates pass without
relaxation: maximum-time phi1.560/0.767%, velocity1.543/0.225%, pressure3.186/
2.014%. Chemical potential is reported separately and is less well converged.
Accepted reference is bdf-fine.npz; reference-validation.json records the gates.
Five original focused CPU tests pass54.69s. Streamfunction-specific hard initial,
wall/inlet, query/time JVP, exact divergence and cross-section flux tests pass
61.58s. Full streamfunction parameter-gradient test passes25.20s, log
/tmp/channel-nsch-stream-gradient.log. New invalid-configuration checks pass.

First exploratory BE coarse24×10/dt.05 completes17.4s;40×16/dt.02563.5s,
64×24/dt.0588.2s anddt.025165.9s. Flux/velocity walls pass, but time phi pooled
2.506%above maximum-time gate2.5%, so use BDF2 production controls. Exploratory
files lack embedded identity and are retained only as nonaccepted pilots;
new BDF2 files embed physical identity. No metadata retrofit or gate relaxation.

GMLS checks:32×12/degree3/weight4 polynomial error1.12e-13 but actual initial
Laplacian error96.3%;64×24still87.7%. Degree5/48neighbor rank failure explicit.
More local exponent16 gives equilibrium-tanh Lap errors45.85/17.12/7.78/4.97%
at32×12/64×24/96×40/128×48. Exponent32 fails condition threshold. Choose96×40,
degree3/36neighbors/weight16 for the experimental training; disclose that this is
GMLS differentiation, not regional nonlocal physics, and that initial C2 profile
is harder. No t0 chemical-residual sample; hard phi/u initial values remain exact.
Affected smooth-stencil test passes after the default change.

Two cold single-geometry Adam5000 controls finish, with width64 and unchanged
known-input arrays/initial model. Direct linear lifting: phi41.99%, u38.63%,
p63.08%, cold261.82s/warm8.76ms. Direct normalized lifting: phi43.35%, u98.86%,
p101.51%, cold261.52s/warm8.42ms. These are failures, not accepted solutions.
Directories under /root/autodl-tmp/tf-logs/channel_nsch2d are
20261031-channel-single-gmls-01 and20261031-channel-single-normalized-01.
Independent GPU audit of the latter confirms primal/JVP agreement3.58e-7 but
flux falls from0.19995at inlet to0.00180at outlet at t3, versus prescribed0.2.
Thus soft sampled continuity has allowed a leaky-flow minimum. The audit is
flow-audit.json in that run; all failed artifacts are retained.

Current explicit base uses a2Dstreamfunction, exact divergence/flux/no-slip/
inlet/initial velocity and no duplicate continuity loss. Raw4channels(phi,mu,
psi,p) become physical5channels(phi,mu,ux,uy,p). Normalized phase/chemical lifts
remain; no prescribed future droplet motion. Missing velocity/lifting records
preserve earlier checkpoints; train_linear.py/train_normalized_direct.py retain
their exact earlier recipes. Cold16geometry/B2/SOAP5000 width96/decoder3 run
20261031-channel-operator-stream-01 completes5000updates: pooledphi30.594%,
terminalphi33.735%,u4.819%,p22.963%,mu150.69%. Physical-area-weighted values
are phi29.693%,u4.849%,p23.050%. This is a failed coupled-accuracy trial, with
early interface dissolution, not a SOTA. Cold1000.99s,warm52.89ms/update,
active peak2.800GiB; allocator pool62.287GiB is a separate reservation.
Log: /root/autodl-tmp/data/channel_nsch2d/train-operator-stream-01.log.

Independent saved-checkpoint validation passes: field gap7.15e-7,primal/JVPgap
2.98e-7,fluxgap2.50e-6,wall8.85e-8,div3.54e-8. All16conditions visited592..668
times; actual draws/times/known-input schema pass. Predicted phase area declines
0.15203->0.05587, versus approximately0.15209 in FEM; this is not merely a small
interface displacement. Actual phi/speed/pressure3x5fields and convergence plots
are saved in the run's figures directory;32x12original queries, display-only
linear interpolation, not new dense field measurements.

Coarse-grid open-boundary diagnosis at t1.5 finds weak phase-balance defect
0.17551 while integrated GMLS strong residual is3.855e-5. Decomposition:
AD-advection volume/surface quadrature gap=-0.175446; mobility-scaled GMLS
Laplacian Gauss gap=2.51e-5. Thus the measured discrepancy is dominated by the
transport-volume versus boundary balance, not by the Laplacian Gauss gap alone.
Refined diagnostic completes: hold3977encoder inputs/model fixed, integrate on
192x80(15633queries); advection gap becomes+.02580, weak storage balance remains
.17544. Three strongest transport-gradient queries agree with central finite
differences to4.15e-5/6.23e-5relative at steps.001/.0005. Evidence supports a
collocation/quadrature-aliasing concern, not a large AD error at those probes or
a unique-cause proof. Diagnostic quadrature is not yet fully converged. A direct
native coupled GMLS baseline remains unvalidated. This bundled pilot is not an
isolated optimizer/architecture comparison; one diagnostic FEM geometry is not
a generalization estimate. All training/audit/FEM workers have exited.

## Diff-level changes and verification

New independent channel_nsch2d example includes explicit physics/geometry/
configs, QR-GMLS/data, multichannel point network, physical losses, canonical
train.py, independent FEM/refinement, saved-run replay, phase-balance diagnosis
and actual-field/convergence plots. Shared neural/operator/optimizer APIs and
established examples are unchanged. Canonical entry selects the tested SOAP
operator preset; original linear and normalized-direct numerical recipes match
saved controls after excluding only provenance and unused evaluation-count
metadata. Their initial model bytes/actual known inputs/time/condition draws
are exactly identical. Optional records preserve original checkpoint arithmetic.
New reports correct inherited solid/solid_area labels to dispersed_phase/
dispersed_area; old artifact contents are untouched.

Seven unique focused tests pass across staged invocations: geometry/initial,
polynomial reproduction/smooth refinement, NS/capillary signs, invalid settings,
exact IC/inlet/wall/query/time jets, exact cross-section flux, finite complete
parameter gradients. FEM gates and independent GPU saved-field/metrics/flow
checks pass; these do not override the failed accuracy result. Solver CLI --help
passes after all GPU workers exit; new source AST, strict artifact JSON,
canonical-recipe comparison and whitespace checks pass. Actual phi/speed/pressure
panels inspected, including early dissolution. Small source/text receipt only;
media/checkpoints/data remain outside Git. No package change, deletion or Git
history action. README, DEVELOPMENT, navigation, index and memory are updated.

Receipt: docs/tasks/channel-nsch-pilot-validation.json. Final run root and three
control paths above are direct filesystem paths; no file hashing or sibling
imports. Latest phase diagnosis log is
/root/autodl-tmp/data/channel_nsch2d/phase-balance-refined-operator-stream-01.log.

## Optional future resume

No user blocker or required follow-up remains for the requested first trial.
Accuracy work remains unresolved: start with the frozen final model and actual
open-boundary phase law, check overlapping/coarsened/refined query gradients and
converged quadrature, then design conservative transport/spatial-collocation
controls and a direct native baseline before enlarging network/training budget.
Do not silently change the NS--CH BCs, pressure convention or old regional
kernel, impose closed-domain constant mass, or call this operator solved/SOTA.
