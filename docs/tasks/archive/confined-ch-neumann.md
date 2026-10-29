# Confined CH with Neumann boundaries

Goal: implement and try the agreed ellipse+two impermeable holes phase-field
case, then compare nonlocal natural no-flux and local AD against independent
refined mixed FEM. Preserve existing precipitate results.

Acceptance: train.py entry, explicit configs/README; physical quadrature and
symmetric visible bonds; constant/weighted symmetry/mass/energy checks;
independent FEM time/mesh refinement; native baseline separated from continuum
bias; actual matched cold starts and CPU checkpoint replays with field/error/
boundary/energy/mass/time reports. No promised accuracy advantage, NS, mechanics
or deferred SOAP/FFT redesign. Keep one active task.

Status: complete. Both cold starts exit0, final CPU replays and physical
audits pass. Results retain explicit finite-horizon/Neumann limitations.
Blocker: none. Next action: none for this first implementation/pair. Further
joint horizon/grid refinement, pointwise boundary-extension diagnosis or initial
mu-fitting improvement would be a new task. No GPU/FEM job remains running.
No change to prior example defaults or SOAP/FFT deferred issues.


Git: user explicitly authorizes commit this turn. Saved prior working tree on
dev as728e5b9 (172 files). No merge/push requested or performed. Staged check
reported one trailing blank in an old archived failure log; raw log preserved.
All new changes begin after that commit. No new result commit is implied.

Plan: regional finite-horizon negative Laplacian with physical cell/node
quadrature, nonnegative symmetric pairs and analytic segment visibility around
circular holes. For2D radial kernel120/(pi delta^4)*(1-r/delta)^2,
int K(z) z_i z_j dz=2delta_ij by radial integration, so interior -Delta limit.
Avoid row rescaling that alters physical mass. At finite delta native closure
is not exact pointwise local Neumann. Compare local AD Hessians with explicit
normal BC residuals. Both arms share hard initial phase and weighted mass,
without redundant mass loss. Use direct output coordinate; no inverse-cubic
map or transformed residual. Deterministic smooth initial perturbation should
be Neumann-compatible before time evolution; no reference field labels train.

Preflight: ellipse center(.5,.5),axes(.5,.4), holes(.35,.54,.075)/(.68,.43,.055),
collocation mesh size.025 ->2369 triangle centroids, area.6014558924. T1,
epsilon.05,M.02,beta1,initial amplitude.3 with wall width.04. Smooth phase
bands reach walls and become distinct by T1; coarse-P2 FEM range[-.7846,.8128],
initial->terminal field change183.2%, energy.14933->.11794, mass drift1.65e-16.
Freeze this physical case before training; preflight not yet reference accuracy.
Full FEM coarse/fine and dt refinement running exec22652 at
/tmp/confined-ch-reference.log. Focused CPU tests exec38274 at
/tmp/confined-ch-tests.log; no concurrent JAX GPU worker. Native independent
SciPy functions factored into native_reference.py with unchanged mathematics.

Environment correction: loading Gmsh and DOLFIN in one process loaded conflicting
MPI/PETSc libraries. Gmsh meshing now runs in a separate subprocess; repeated
full-reference startup no longer prints that conflict. Existing HDF5 build/
runtime warning remains visible in raw log (single-process solver, not MPI);
validate stored datasets/numerical refinement rather than suppress it.

Four focused tests pass54.492s, including independent graph matrix parity,
constant/physical-volume symmetry and mass/energy identities, line obstruction,
analytic initial phi/mu Neumann compatibility, hard IC/weighted time JVP and
finite nonzero parameter gradients for both residual modes. Training CLI loads
its canonical default config (--helpshort prints usage and conventional exit1).
Changed modules compile; diff whitespace check passes. No GPU training yet.

First full reference FAILED its gate: local max phi spatial.00581/time.01214;
mu spatial.5637/time.6982. Native BDF time checks pass(1.64e-7 max mu), graph/local
pooled phi gap.04209. Failed CN target kept at reference_cn_failed.npz/json;
no PINN trained against it. Revising reference to BE-started BDF2 and analytic
continuum mu0 (independent NumPy chain/product Laplacian), with unchanged
physical IC and tolerance. Need renewed reference refinement before training.

Independent analytic IC Laplacian vs JAX Hessian test passes5.094s (fifth
numerical check). BDF2 mixed construction also verified against lifex tutorial07
https://lifex.gitlab.io/lifex-public/classlifex_1_1tutorials_1_1Tutorial07__AD.html;
no stabilized-BDF energy theorem is claimed. Full new reference exec20351 logs
/tmp/confined-ch-reference-bdf2.log. FEM stage arrays now save separately.

Reference now PASSES, process exits0. Local max spatial phi.005810 (t0
interpolation),mu.002051; max temporal phi.0003769,mu.001449. Native max temporal
phi2.82e-8,mu1.64e-7. Local/native mass drift9.19e-17/5.38e-16 and no saved energy
increase. Shared analytic geometry+IC unchanged. Pooled native/local reference
gap phi4.20767%,mu32.3404% includes horizon/boundary/quadrature bias; never label
native error as common continuum error. Source/failed/success logs/checks archived.
Starting nonlocal GPU cold run20261027-confined-nonlocal-01; no FEM process
remains. Pending: finish both5000+500 cold starts, CPU final replay/plots and
all-field/common/native errors, physical diagnostics and artifact registration.

Nonlocal run exec32094 at /tmp/confined-ch-nonlocal.log has reached4250 SOAP
updates: native phi~0.00977, common-local~0.04298. Source case files frozen until
the local AD snapshot is captured. No physics/horizon/weights tuned to results.
Independent manufactured-IC integral audit (2369 vs4696 physical triangle areas)
records fixed-delta quadrature difference1.06% atdelta.07. Fine integral vs
analytic -Lap IC differs58.25% overall (steep IC boundary layer); decreasing
delta.09/.07/.05 gives68.06%/58.25%/44.57%, while coarse/fine quadrature grows
0.56%/1.06%/3.04%. This is a consistency diagnostic, not an evolution convergence
proof or a theorem for arbitrary kernels. Boundary/horizon errors remain real;
shrinking delta without mesh refinement is not a free accuracy improvement.
Artifact: archive/confined-ch-boundary-operator.py/json.

Nonlocal run completes exit0:146.474s,compiled14.779ms,SOAP5000+LBFGS500;
final common phi pooled.0419353/terminal.0499981,native pooled.00889803.
Native/common mu pooled.131421/.416370; hard-mass drift4.86e-17. Local AD
exec63776 runs at /tmp/confined-ch-local-ad.log. Both initialized checkpoint
bytes and18 frozen source files match; configs differ only nl.method/run_name.
Artifact index registers the new case, validated/failed reference paths and both
new run directories; pending final analysis link until local completion.

Local AD reaches3000 SOAP updates, common phi L2.08105 and still falling.
Nonlocal completed native/common errors and matching snapshots are above;
await full AD budget before ordering. Both raw physical fields and boundary
AD-normal RMS will be audited; native graph zero external exchange does not
assert the extrapolating MLP's pointwise boundary derivative is zero.

## Final implementation and validation

Added independent examples/confined_ch2d: canonical train.py, explicit config,
shared problem/validation/analytic IC+Laplacian, Gmsh subprocess geometry,
physical-volume regional pair operator, nonperiodic direct-output network,
original mixed residuals and local AD Neumann losses, independent P2 mixed FEM
and SciPy native references, direct checkpoint evaluation/plots. Tests cover
geometry, normalization, conservation/energy, IC/normal compatibility, coupled
parameter gradients and independent analytic derivative parity. Root README,
DEVELOPMENT and artifact index document the API. Prior examples unchanged.

All5 distinct focused checks pass (first4 in54.492s,new analytic derivative in
5.094s); after stricter duplicate/support validation the operator check alone
passes again2.607s. Both actual canonical training paths and CPU evaluation
execute successfully; no unchanged full suite is repeated.

Cold starts: /root/autodl-tmp/tf-logs/confined_ch2d/
20261027-confined-nonlocal-01 and20261027-confined-local-ad-01, each5000SOAP+
500LBFGS, seed42, same initialized bytes/18source snapshots. Full GPU sample
trace not stored, but seed/shapes/sampling code identical. Both LBFGS stop at
the500 budget, not converged tolerances. Final checkpoints used for main table.

| Common refined local reference, final update5500 | Nonlocal | Local AD |
|---|---:|---:|
| Pooled phi L2 | 4.19353% | 5.66394% |
| Terminal phi L2 | 4.99981% | 3.83545% |
| Boundary-band pooled phi L2 | 5.26337% | 6.73197% |
| Interface-band terminal phi L2 | 9.05836% | 7.05492% |
| Pooled mu L2 | 41.6370% | 46.0890% |
| Positive-time pooled mu L2 | 6.70558% | 10.1082% |
| Terminal mu L2 | 13.3199% | 21.1296% |
| Initial mu L2 | 83.0407% | 91.1334% |
| Trainer wall | 146.474s | 529.026s |
| Compiled SOAP mean after100 | 14.7787ms | 65.4080ms |

Single-pair speed3.6118x wall/4.4258x compiled. Timer includes JIT/evaluation/
refinement, excludes operator setup/reference/replay. Not a dedicated kernel or
robustness benchmark. Nonlocal lower pooled error, local AD lower terminal
error; no universal ordering or boundary-only causal attribution. Boundary band
is distance<delta, interface band|phi_local|<.5. Native nonlocal pooled/terminal
phi0.889803%/0.646899%, native mu13.1421%/3.98776%. Pooled continuum gap4.20767%
already exists before PINN fitting. Best common-local checkpoints: NL750 pooled
4.12891%/terminal4.69763%; AD5500 as final. Best-phase selection may worsen mu;
no reference field is a training target. Both source snapshots remain frozen;
later README result additions do not change replay mathematics.

## Critical Neumann finding and remaining limitations

Native symmetric visible graph closure avoids external exchange and explicit
normal-derivative losses, but the trained MLP local normal derivatives do NOT
automatically approach zero at finite delta. At t1, per-wall phi normal RMS
(outer/hole1/hole2): NL1.656/4.672/8.790 vs AD.01577/.02212/.02554. Corresponding
mu normal RMS: NL.5258/.3489/.2735 vs AD.00638/.01396/.01189. This includes
unconstrained MLP extension from cell centroids and finite-horizon effects; the
current evidence does not uniquely separate them. Do not claim exact classical
Neumann, nonzero prescribed-flux handling or a universal boundary-free method.
Further boundary/horizon/mesh refinement is required before that claim. Local
weak FEM already naturally handles homogeneous Neumann as well.

Initial mu is still poorly learned. At final t0+16midpoint audit, initial
chemical residual contributes79.27% NL and84.60% AD of total loss. AD t0
chemical MSE.110669 vs positive-time.0008561, with smaller boundary penalties
.00005591(phi)/.00002924(mu). Its slower training cannot be attributed entirely
to Neumann penalties without controlled ablation. Both include the same initial
PDE-time policy; no post-result training or weight/seed selection was added.
Retain pilot results, no all-field precision/SOTA or stability claim.

## Physical/replay audits and artifacts

Both final CPU field replays match saved phi/mu to2.45e-15. Predicted weighted
mean drift<=6.94e-17 and sampled native free energies decrease. Actual graph:
2369 points,63841visible pairs,oneconnected component,weighted symmetry defect
4.16e-17,weighted column-sum defect6.25e-16. Exact ODE pair conservation is
separate from the shared PINN hard mass constraint; no mass loss is present.
FEM integral mass is roundoff-conserved, while sampling that FE field on coarse
centroid quadrature gives initial mean offset4.5369e-6 and mean drift1.4721e-4.
Do not interpret this sampling quadrature error as loss of FE mass or compare
PINN mass to that drifting sampled reference mean. Field L2 uses fixed physical
quadrature consistently. Native energy curves use each model's own functional,
not a shared energy criterion to rank different operators.

[Physical fields](confined-ch-common-fields.png),
[convergence/time/energy](confined-ch-comparison.png),
[native/local reference gap](confined-ch-reference-gap.png),
[full audit](confined-ch-analysis.json),
[nonlocal reference/prediction/error](confined-ch-nonlocal-phi_fields.png),
[local AD reference/prediction/error](confined-ch-local_ad-phi_fields.png),
[validated reference checks](confined-ch-reference.json),
[failed CN checks](confined-ch-reference_cn_failed.json),
[operator boundary audit](confined-ch-boundary-operator.json),
[first4 tests](confined-ch-tests.txt),
[analytic Laplacian check](confined-ch-laplacian-test.txt),
[post-validation operator check](confined-ch-operator-recheck.txt).
Configs, summary, best/metrics/loss/refinement/replay JSONs and figures are
archived alongside this task; full checkpoints/frozen sources remain at direct
run paths. Failed CN and stage references remain at the unchanged new data root.
Failed/successful FEM and both training stdout are copied as.txt for Git
visibility. Existing HDF5 runtime warning stays visible; no read/write or
refinement failure is hidden. No FEniCS MPI was launched; environments separate.

Closure records diff-level changes in this section. Memory files updated:
docs/PROGRESS.md,docs/DECISIONS.md,this task archived from its active path.
Old728e5b9 snapshot is committed; NEW case work remains uncommitted on dev.
No push/merge or further history operation. Further checks at closure: artifact
inventory, documentation links/JSONs and git diff --check.

Additional [per-wall normal diagnostic](confined-ch-boundary-normals.png) makes
the local-vs-native boundary distinction explicit. Plot is reconstructed from
already audited JSON, without rerunning checkpoint or expensive tests.

Final closure: git diff --check passes,112 local documentation links resolve,
all archived new JSON parses, and new text files have no trailing whitespace.
Artifact inventory:8cases/362directories. No active task or worker remains.
