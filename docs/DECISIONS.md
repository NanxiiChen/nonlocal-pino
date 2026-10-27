# Decisions

## 2026-10-24 — Independent research project with preserved recovery baseline

Decision: user approves splitting nonlocal and fracture research; create
independent packages from nonlocal-KKS dev61352f1 while preserving the source
checkout, environment, all external artifacts and a verified history/worktree
backup. Reason: separate research responsibilities without losing validated
numerics or historical continuation. Rejected: in-place deletion, immediate
artifact relocation, third common runtime package and history rewriting.
Relation: approved proposal and docs/migration/source-map.json.

Numerical bodies and explicit config values are preserved; imports/package
ownership and complete source-snapshot collection change. Old run identities
are not rewritten or bypassed. Original mixed decisions remain in Git history
and /root/nonlocal-KKS/docs/DECISIONS.md, rather than being selectively edited
into a false new history.

## 2026-10-24 — Validate split and close migration without new simulations

Decision: accept code organization after owned regression suites, exact
configuration/fixed-input comparison, isolated imports, artifact checks and
verified bundle recovery. FEniCS checks run in its separate environment; GPU
training/performance is outside migration scope. Record historical resume
limitations instead of weakening identity guards. Network timeouts justified
independent copies of exact baseline dependency payloads, followed by standard
PEP660 project installation and frozen/offline lock audits. Switch only the
package index to official PyPI; keep all53 source dependency versions and
make the already-used Gmsh4.15.2 an explicit optional mesh extra. Rejected:
version upgrades, shared-source path injection and declaring old checkpoints
transparently compatible without validation. Relation: migration/VALIDATION.md.

## 2026-10-24 — Publish the validated split as a private repository

User requests a private GitHub repository for each split project. Prepare the
validated split commit on dev and a regular merge into main; push only main
after CLI browser authentication. Retain original code/history/data/backup and
all existing numerical validations. No default or numerical changes. Repository:
https://github.com/NanxiiChen/nonlocal-pinn
Reason: private remote backup and independent project development. Rejected:
empty repositories, pushing dev or changing the original project's remote.
Relation: completed split; publication coordinator record in the source project.

## 2026-10-24 — Confirm private GitHub publication

Created NanxiiChen/nonlocal-pinn as private under the authenticated NanxiiChen
account. Configured only this project's origin and host-scoped local credential
helper; published main after a regular dev merge. Verified private visibility,
main default branch, only main remote branch and local/remote SHA equality.
Reason: user requested private repositories after approving the split and
completed browser authorization. Rejected: pushing dev, modifying original
source origin, or relaxing privacy before publication. No numerical changes
or repeated expensive tests; existing validations remain valid.

## 2026-10-24 — Diagnose melting against the implemented IVP before tuning

Decision: audit the claimed analytic profile by substitution, evaluate existing
checkpoints read-only, and compare a converged independent radial local control
before proposing training or physical changes. User confirms reference means the
analytic profile and checkpoints are under the existing melting log root.
Reason: spherical curvature makes the linear-radius profile inexact, while its
magnitude must be separated from optimisation and strong forcing sensitivity.
Rejected: calling every discrepancy a PINN failure, treating the analytic profile
as exact, silently changing the PDE/BCs or resuming historical checkpoints.
Relation: docs/tasks/icemelting-late-time-diagnosis.md.

## 2026-10-24 — Separate melting reference bias from unstable-phase PINN error

Decision: retain the declared PDE and report the linear-radius tanh profile as an
inexact comparison, with independently checked local/discrete controls. Default
N1=4e-5, N2=0.079012345679, N3=6.285393611 imply fprime(+1)=12.41276253:
small ice-phase errors grow strongly. Refined radial control gives R(1)=0.14947178;
SOTA gives 0.14619212 and a widened interface with inward speed 0.24005.
Six-neighbour cubic quadrature has second-moment coefficient 0.68585478, but its
float64 direct evolution stays near R=0.1502; local AD shows similar PINN bias.
Reason: quantify distinct reference, quadrature and approximation errors instead
of attributing visible acceleration to curvature alone. Rejected: arbitrary
mobility/source changes, claiming all checkpoints disappear within training,
or enabling causal weighting without checking its core-mask normalisation.
Relation: icemelting-late-time-diagnosis; diagnosis only, no physical/API changes.

Historical evaluation requires numerical semantics as well as checkpoint leaf
shapes: pre-2026-07-16 train.py used getattr(jnp, activation) with tanh fallback,
so gelu-labelled baseline2/local_ad checkpoints used tanh (a555329 source).
Read-only reconstruction records this explicitly; config/identity guards and
external artifacts remain unchanged. Graph-conv experiment configs cannot be
reconstructed with the present Fourier-only builder and are excluded.

Provenance clarification: the fallback statement refers to the early July15
implementation a555329 and the specifically replay-validated baseline2/local_ad
runs, not every run before July16; later July15 source 29d8c26 already passes
activation strings to the registry. Completed diagnosis and durable evidence
are archived at docs/tasks/archive/icemelting-late-time-diagnosis.md.

## 2026-10-24 — Explore stable latent phase representation without changing PDE

Decision: retain the original six-neighbour cubic graph and compare isolated
5 s pilots using raw phase, a hard-IC tanh latent distance, and its analytically
transformed residual; consider an expanding-time-horizon pilot. Use direct
float64 integration of the same graph for validation, including physical 4 s.
Reason: tanh latent values represent exponentially small bulk phase deficits
without asking a raw network to resolve 1 minus tiny numbers; exact initial
geometry removes soft-IC ambiguity. Record the linear front motion in the
trial function as a deliberate physics prior, and compare its untrained error
so training gains are not confused with that prior. No Eikonal penalty or
reference labels enter training. Rejected: changing forcing/mobility, arbitrary
residual clipping/division by a numerical floor, using analytic targets for
training, altering delta during optimisation, and overwriting historical runs.
Relation: docs/tasks/icemelting-late-time-accuracy.md. This is a bounded research
pilot, not acceptance of a production default or a general-geometry claim.

## 2026-10-24 — Preserve latent dynamics rather than only bounding the phase

Decision: prioritize the exactly transformed nonlocal residual in the hard-IC
tanh pilot. For phi=tanh(u), the residual is
u_t + N1*sum_j w_ij*sinh(u_i-u_j)*cosh(u_i)/cosh(u_j) - N2*tanh(u_i) + N3.
Evaluate the cosh ratio using logaddexp, without division by rounded 1-phi^2.
Pointwise residual zeros agree analytically; its MAE is a different optimisation
objective, explicitly emphasizing bulk latent evolution. No arbitrary floor or
residual clipping is introduced. The trial function subtracts the known constant
latent forcing N3*t and embeds the given initial signed distance exactly.
Reason: the first 5 s pilot gives relative L2=6.31e-5 and radius error -0.0252%;
tanh/hard-IC with original phase residual gives 5.47e-4 and +0.235%, worse than
the untrained prior's 4.25e-4 field error. Expanding-horizon pilot gives 7.99e-5,
so it has no demonstrated additional benefit here. Rejected: treating output
bounding alone as a cure or attributing the initial prior's accuracy to learning.
Relation: icemelting-late-time-accuracy. Seed repetitions are pending; unchanged
six-neighbour discretisation error is outside these accuracy claims.

## 2026-10-24 — Close five-second accuracy exploration with replicated evidence

Decision: recommend opt-in production integration of hard-IC latent values and
the exact transformed nonlocal residual. Eight isolated 2000-step pilots retain
the same physical equation and six-neighbour graph. Seeds 42/7/123 give 5 s
full-grid relative L2=6.31e-5–1.02e-4 and worst six-direction radius error 0.0563%;
original-phase latent controls give 4.36e-4–5.47e-4. The untrained prior is 4.25e-4,
so bounding alone did not improve its terminal field accuracy. Direct float64
dt-halving error is 3.46e-9. Reason: replicated geometry/field checks support the
transformed objective, while single-seed curriculum has no extra benefit here.
Rejected: claiming continuous-PDE convergence from a fixed graph, relying on
volume-averaged residual alone, crediting the prior to learning, or changing
production defaults before a matched integration/full-budget validation.
Relation: archived docs/tasks/archive/icemelting-late-time-accuracy.md and JSON
with configuration/source/checkpoint provenance. No historical run was resumed;
new generated artifacts remain in distinct local reports/ directories.

## 2026-10-24 — Suspend strong-prior recommendation and test learned evolution

Decision: accept the user's objection to embedding -lambda*t in the trial
function. Earlier pilots demonstrate correction of a strong motion prior,
not autonomous PINN discovery of melting evolution. Suspend the production
recommendation until controls with u=u0+t*N and no explicit forcing/motion term
are tested. Retain the exact initial condition, which is prescribed problem
data, and derive both objectives from the unchanged nonlocal PDE. Compare at
matched budget and evaluate only against the already verified graph reference.
Reason: separate conditioning improvements from knowledge of the target motion.
Rejected: defending low errors as general solver evidence, hiding the prior,
or replacing it with a prescribed radius/velocity/width penalty.
Relation: docs/tasks/icemelting-weak-prior.md; earlier records remain preserved.

## 2026-10-24 — Initial-condition-only control learns motion from the PDE

Decision: keep speed-free controls as the default research experiment. The trial
function u=u0+t*N is independent of lambda at fixed network weights; its zero
network remains stationary (5 s untrained relative L2=0.7718035). The exact IC
is prescribed data, not future motion. Seed42 at 2000 steps gives phase-residual
L2=0.00466244 and transformed-residual L2=8.829e-5 at 5 s, with no reference labels
in training. Reason: this directly tests evolution learning rather than only
correcting a near-reference trajectory. Additional seeds are pending. Rejected:
crediting earlier prior accuracy to the network, fixing an evolved radius or
width, and assuming one seed proves broad generalisation.
Relation: docs/tasks/icemelting-weak-prior.md. The older strong-prior controls
remain explicit and their artifacts are untouched.

## 2026-10-24 — Close speed-free controls and narrow the recommendation

Decision: prefer IC-only trial functions for any subsequent integration; keep
strong-motion methods only as explicit historical controls. Seven 2000-step runs
retain all other physics/graph/architecture/optimizer/sample settings. Three
paired seeds give transformed 5 s L2=8.66e-5–1.53e-4 versus phase-residual
0.003804–0.004662 (28.5–52.8x paired reduction), starting from a stationary field
with L2=0.7718. Worst six-direction radius error is 0.0724% vs 3.0894%.
Reason: gains survive removal of near-reference motion, addressing the user's
methodological objection. Rejected: reinstating the nominal speed, extrapolating
to general PDE performance, or confusing fixed-graph and continuous-PDE error.
Relation: docs/tasks/archive/icemelting-weak-prior.md plus durable JSON/figure.
Changed only the isolated research runner's default; production training and
historical artifacts remain untouched. Production integration remains opt-in
follow-up work, with matched full-budget and spatial validation still needed.

## 2026-10-24 — Disambiguate the original baseline and terminal metric

Decision: audit the historical no-hard-IC checkpoints separately from the new
hard-IC phase-residual controls. Compare at physical 4 s and label analytic versus
same-graph references explicitly. Reason: the user correctly noted that the new
0.0038–0.0047 terminal errors cannot describe the original production model.
Rejected: relabelling the hard-IC control as the historical baseline, treating
the logged pooled-time metric as terminal-only, or asserting a matched ablation
across different historical budgets. Relation: icemelting-baseline-metric-audit.

## 2026-10-24 — Integrate the IC-only latent formulation by explicit request

Decision: default production melting to the IC-only tanh representation and
stable exact transformed residual, with explicit formulation selection retaining
raw-phase/soft-IC and hard-IC/phase controls. Keep the legacy raw network builder
unchanged for read-only historical reconstruction; new production wrapping is
explicit. Train a fresh run and label analytic and graph-reference L2 separately.
Reason: the user now explicitly authorizes production integration and cold start
after reviewing the numerical controls. Rejected: resuming historical weights,
building in nominal motion, silently dividing by saturated sech squared, and
presenting analytic discrepancy as numerical solver error.
Relation: icemelting-production-latent; archived weak-prior and metric audits.

## 2026-10-24 — Monitor a bounded production cold start before full-budget use

Decision: preserve the base 4 s/20000-update sampling and optimizer budget, and
provide a separate 2000-update production cold-start preset with the validated
K_core256/Nt16/lr0.001 settings. Evaluate every200 updates, full-grid terminal
errors every1000 and at initialization/finalization, against separately labelled
analytic and verified graph targets. Match the existing 5 s reference at physical
0/1/2/3/4 s after checking common dimensional parameters and spatial identity.
Reason: validate the actual production entry point with a reviewable L2 history
without misrepresenting the bounded run as a full-budget validation. Rejected:
silently changing the full training budget, interpolating or relabelling reference
snapshots, dropping the original pooled metric, redundant IC GradNorm terms, and
masking numerical failure with nan_to_num. Relation: icemelting-production-latent.

## 2026-10-24 — Close production integration with two fresh measured histories

Decision: retain ic_latent as production default and keep base sampling/lr/budget
otherwise unchanged. Two2000-update GPU cold starts at4 s complete: base
K_core2000/Nt64/lr0.005 gives full-grid terminal graph L2=3.862e-4 and analytic
L2=0.001397; the separate bounded preset256/16/0.001 gives3.018e-5 and0.001028.
Reason: actual production-entry-point runs, numerical regression checks, and
independent reload/IC/radius checks verify the integration. Both share identical
fresh initial weights and show nonmonotonic late L2. Rejected: reporting only the
better preset as default-batch performance, claiming monotonic convergence,
attributing differences solely to lr when batching also differs, or presenting
2000-update/4 s runs as full20000-update/5 s validation. Relation: archived
icemelting-production-latent task, JSON with exact provenance and L2 figure.

## 2026-10-24 — Promote the best audited melting run by directory tag

Decision: append -sota to 20261024-ic-latent-coldstart-01 and rename the old SOTA
directory to legacy-raw-phase-soft-ic. Preserve run contents and original config/
source/event identities, recording the rename and current artifact locations.
Reason: the user explicitly requests filesystem naming changes; the promoted
run has the lowest terminal full-grid4 s error among the audited production
cold starts against both references. Rejected: promoting the worse default-batch
run, comparing unlike horizons/strong-prior controls for this tag, overwriting a
destination, rewriting checkpoint/config/event bytes, or retaining a SOTA symlink
that duplicates TensorBoard runs. Relation: icemelting-run-tags.

## 2026-10-24 — Prefer explicit artifact paths over file hashes

Decision: remove file-hash creation and verification from melting runtime and
replay Python, record source/model/reference paths, and load config/checkpoint/
NPZ data directly. Reason: the user explicitly does not want hash-based artifact
handling; normal loaders already use paths and do not need these extra manifests.
Keep dimensional/grid/shape/time compatibility checks because they establish
which PDE the reference solves. Rejected: retaining a hidden checksum gate,
introducing a replacement hash scheme, or rewriting historical source/checkpoint
contents merely to erase prior recorded metadata. Static historical reports and
migration/rename evidence remain past records, not loading requirements.
Relation: icemelting-direct-path-loading. This preference applies to future work.

## 2026-10-24 — Describe the current production strategy before research history

Decision: organize the melting README around the current IC-only latent default,
with explicit residual/loss/sampling/optimizer/evaluation and loading semantics.
Separate 4 s production defaults from the 5 s FEM/research configurations, and
link archived run evidence rather than duplicating transient performance tables.
Reason: the user requests a self-contained record of the current case strategy;
the existing README foregrounds pilot history and mixes solver/training defaults.
Rejected: documenting stale parameters, implying tanh output alone is the new
objective, embedding a future-motion prior, or changing code while documenting it.
Relation: icemelting-strategy-readme. No formulation or configuration is changed.

## 2026-10-24 — Set five-second defaults and compare configured optimizers

Decision: change production Tc from4 to5 s (normalized horizon stays0–1) and
compare fresh SOAP/Adam production runs at the full20000-update base budget.
Keep IC-only latent formulation, architecture, seed42, graph, spatial/time
sampling and lr0.005/schedule identical; vary optimizer selection only. Use
the existing direct5 s graph integration strictly for labelled evaluation.
Reason: the user explicitly requests the default horizon change and an empirical
optimizer comparison. Rejected: extrapolating4 s weights, changing both algorithm
and representation/lr unnoticed, selecting different random batches, hashing
artifacts, or claiming a single shared-lr experiment proves globally optimal
performance across optimizers. Relation: icemelting-soap-adam-5s.

## 2026-10-24 — Lower the default and comparison learning rates before launch

Decision: use lr_init=1e-3 in the production base config and both5 s optimizer
runs, preserving the same schedule, sampling and20000-update budget. Reason:
the user explicitly requests1e-3 and the steering arrived before either run
started. This supersedes the0.005 rate in the preceding planned protocol.
Rejected: comparing optimizers at different rates, continuing with the old rate,
or changing the experiment mid-run. Relation: icemelting-soap-adam-5s.

## 2026-10-24 — Respect the reduced five-thousand-update budget

Decision: set base iterations to5000 and compare both optimizers at5000 updates.
Stop the running SOAP process and use its already-saved ckpt_005000.eqx; run
Adam fresh for5000 only. Reason: the user explicitly reduces the budget while
SOAP is running. The later SOAP iterates are preserved as history but excluded
from the paired comparison and timings. Rejected: finishing20000 anyway,
comparing unequal update counts, deleting valid earlier checkpoints, or
overwriting the interrupted run's original saved config. Relation:
icemelting-soap-adam-5s; this supersedes the earlier20000-update plan.

## 2026-10-24 — Close the equal-budget five-second optimizer comparison

Decision: retain SOAP as default with Tc5 s, lr0.001 and5000 updates. Selected
5000-step checkpoints give terminal full-grid graph L2=6.110e-5 for SOAP and
7.395e-4 for Adam; tail sampled medians differ by4.99x. Speeds after compilation
are close (~27–28 ms/update). Reason: direct paired production measurements,
identical initial weights and independent field/geometry checks favor SOAP's
late accuracy/stability under these settings. Adam's lower analytic error is
partly cancellation against that approximate profile (error cosine~-0.973),
not lower fixed-graph solution error. Rejected: presenting12.1x endpoint ratio
as universal, counting interrupted SOAP work beyond5000 in paired timing,
retagging unlike4 s/5 s runs by an unqualified score, or changing the optimizer
default on analytic error alone. Relation: archived icemelting-soap-adam-5s
task with direct paths, JSON evidence and comparison figure.

## 2026-10-24 — Identify the validated five-second SOAP preset as canonical

Decision: explicitly designate train_base.py as the default configuration from
the selected 5 s SOAP ckpt_005000.eqx, recording its direct path in the artifact
index. Retain ic_latent, SOAP, lr0.001, 5000 updates, seed42, K_core2000/Nt64
and the matching network/operator settings, which already match the winner.
Keep the graph-reference path optional and document the explicit monitoring
override; it affects evaluation only. Reason: the user requests the best result
as the default. The saved run's 20000-update plan is historical; the selected
checkpoint and current default budget are5000. Rejected: rewriting that saved
config, automatically loading trained weights for a fresh run, adding a required
local reference artifact, or conflating the old4 s SOTA tag with this5 s default.
Relation: icemelting-default-preset and archived icemelting-soap-adam-5s.

## 2026-10-24 — Frame the latent residual as model-dependent preconditioning

Decision: describe the generalization as an invertible bounded-variable change
combined with inverse-Jacobian residual scaling. Prioritize nonconserved scalar
Allen-Cahn and thermally coupled tests; treat conserved and multiphase models
as separate derivations with conservation/constraint checks. Reason: scalar
chain rules generalize independently of geometry/dimension, and Glasner (JCP,
2001, doi:10.1006/jcph.2001.6933) already studies atanh nonlinear preconditioning
for Allen-Cahn and nonisothermal phase fields. This is relevant precedent, not
cross-model evidence for our PINN loss. Rejected: claiming invention of the
variable change, universal L2 improvement, automatic conservation/energy
stability, or automatic suitability of tanh for fields that reach/cross its
range limits. Keep the validated implementation unchanged. Relation:
icemelting-residual-generalization.

## 2026-10-24 — Separate reaction cancellation from hard ICs in corrosion

Decision: exclude hard initial conditions from this assessment and retain the
existing 0-1 corrosion phase map phi=(1+tanh(z))/2=sigmoid(2z). Its derivative
is2*phi*(1-phi). Both implemented reactions g'=2*phi*(1-phi)*(1-2phi) and
h'=6*phi*(1-phi) cancel analytically; their transformed terms are respectively
N_AC2*(1-2phi) and -3*N_AC1*q*dc0. The nonlocal edge ratio is exactly the same
tanh ratio already used for melting. Reason: direct inspection of both corrosion
loss modules and the shared KKS constraint, plus the equivalent sigmoid/tanh
identity. Rejected: inventing a new -1..1 corrosion model, changing physical
parameters, adding hard ICs to explain this cancellation, or dividing the
concentration equation by the phase Jacobian. Phase-map saturation, graph
ratios, coupled stiffness and exact endpoint BCs still need numerical care.
Relation: corrosion-latent-residual-assessment; this is algebraic suitability,
not a measured corrosion accuracy improvement.

## 2026-10-24 — Align coupled-model references before residual ablation

Decision: use explicit research presets for corrosion2d single-pit and rough
electropolish2d, preserving each case's physical parameters, horizon, geometry,
KKS output map and soft IC/BC treatment. Use a101x51 sparse cubic graph with
delta0.02 for bounded pilots and regenerate references by independent float64
implicit integration of that graph with prescribed Dirichlet nodes and
interior-only evolution. Check tighter integration tolerances before training.
Matched5000-update pairs use SOAP/lr0.001/seed42, K_core256/Nt16, and identical
sample pools. Normalize the latent phase residual as r_phi/[4phi(1-phi)] so
its interface-centre scale matches the original; keep concentration transport.
Reason: old corrosion FEM omits the PINN's solid side/top BCs, both FEM solvers
mislabel the final clamped step's interval, and rough FEM IC lookup differs
from PINN evaluation. Local FEM and nonlocal graphs also differ spatially.
Correct solver discrepancies; use regenerated graph integration as the main
ablation reference, without claiming continuum-PDE convergence. Rejected:
reusing old arrays, changing mobility/time horizon, conflating model mismatch
with optimization, adding hard ICs, denominator floors or overwriting old data.
Relation: coupled-latent-residual-experiments; presets remain experimental.

## 2026-10-24 — Make FEM boundary checks and initial traces explicit

Decision: mark corrosion pit-mouth Dirichlet conditions pointwise on bottom
nodes, retain explicit solid side/top BCs, and apply prescribed BC values to
both FEM solvers' initial vectors. Reason: a small-mesh smoke solve found the
old topological facet predicate could miss a narrow mouth entirely; the PINN
and graph reference define it by position. Also correct the electropolish README
CH local-Laplacian sign to match both implementations (minus N_CH*Laplacian(q)).
These are traceable alignment corrections, not a new constitutive model.
Relation: coupled-latent-residual-experiments; historical files are retained.

## 2026-10-24 — Retain original coupled-model defaults after negative pilots

Decision: do not promote the transformed phase residual for corrosion2d or
electropolish2d. At matched5000 updates/seed42 with soft ICs, terminal phi/c
L2 is0.03484/0.04135 versus0.56839/0.54585 for corrosion, and0.28604/0.29443
versus0.34398/0.37920 for electropolish (original versus transformed). Reason:
fresh same-graph references passed temporal-refinement checks; paired configs
and initial weights match and CPU checkpoint reload reproduces GPU fields.
Corrosion's transformed concentration branch is nearly c_L=0, with85.2% zero
output derivatives on the probe grid/times. Both electropolish arms produce
three vertical interfaces where reference has one, so its better transformed
contour-distance score cannot establish overall accuracy. Rejected: selecting
metrics to claim success, comparing unlike budgets, attributing poor results
to unverified old references, claiming the variable change is mathematically
invalid, or extrapolating one-seed pilots to all hyperparameters/models.
Future studies should isolate coupled-loss balance and concentration saturation
before adding other constraints. Relation: coupled-latent-residual-experiments.

## 2026-10-24 — Audit coupled KKS gradients and inherited residual scales

Decision: interpret the paired pilot as a transfer test with inherited AC/CH
scales, not a tuned comparison of each formulation. Corrosion retains
ac_scale=1e-7/ch_scale=1 (electropolish1/1). Phase-only inverse-Jacobian
preconditioning remains algebraically valid under c=c_L+dc*h(phi), but the
concentration tanh Jacobian and cross-coupling are not removed. Do not apply
the phase divisor to CH or automatically invert the full coupled Jacobian.
Reason: AD verifies the triangular joint Jacobian. On the same original-final
corrosion checkpoint and first deterministic training batch, switching only
the residual raises scaled AC parameter-gradient L1 norm1996.32->25203.83;
other terms are identical. On the transformed-final checkpoint the recomputed
GradNorm proposal caps CH and both BC weights; solid BC gradient is3.92e-11
despite loss0.00127194. Removing pre_scale would push the AC gradient above
the existing1e6 norm ceiling even at initialization (~1.99e10). These are
checkpoint probes, not reconstructed historical optimizer weights. Rejected:
claiming unchanged pre_scale ensures unchanged loss balance, removing it
without measuring gradients, treating KKS as an algebraic incompatibility,
claiming saturation proves the sole cause, or asserting a retuned L2 gain
without a new controlled experiment. Relation: coupled-latent-residual-experiments.

## 2026-10-24 — Separate concentration output constraints from AC pre-scaling

Decision: add research-only direct concentration c=raw[1] with the phase tanh
retained, and run paired original/latent controls at AC1e-9 for both KKS and
direct-c maps. Keep CH scale1, soft IC/BC, all physical parameters and the
validated graph reference. Reason: the user authorizes removing the output
constraint or changing scale; previous gradient probes motivate testing a
100-fold smaller AC scale. Existing AC1e-7/KKS pairs provide the first scale
control. Rejected: changing the physical KKS free energy, unbounding phase
while still using the tanh-derived residual, clipping concentration outputs,
adding hard IC, removing all scales without a measured reason, or promoting
unvalidated settings. Treat physical-field initialization differences across
maps explicitly and report concentration ranges. Relation:
corrosion-kks-scale-ablation.

## 2026-10-24 — Keep coupled defaults after KKS/scale ablation

Decision: retain bounded-KKS/original-residual defaults; expose direct-c and
positive AC/CH scale overrides only in the research runner. Four new5000-step
controls at AC1e-9 yield terminal phi/c L2 KKS phase0.04212/0.04992, KKS
latent0.43679/0.44957, direct phase0.53846/0.53685 and direct latent
0.53645/0.52965. Reason: no new setting beats prior KKS/AC1e-7 original
0.03484/0.04135. Smaller scaling removes exactly-zero concentration output
derivatives in the KKS latent probe but produces extra phase bands and poor
L2. Direct output has no concentration saturation yet both residuals barely
grow the pit. Independent config/initial-weight/CPU-checkpoint replay passes.
Rejected: declaring fundamental KKS incompatibility, treating zero saturation
as sufficient for accuracy, selecting an earlier transient dip, promoting a
new default from negative controls, or claiming this one-scale/one-seed pilot
exhausts tuning. The physical KKS free energy remains unchanged. Relation:
corrosion-kks-scale-ablation.

## 2026-10-25 — Diagnose latent bulk amplification before further tuning

Decision: explain negative corrosion transfer by measured objective/gradient
redistribution, stiff near-pure-phase dynamics and concentration coupling;
distinguish observations from unique causal proof. At a fixed accurate model,
bulk carries99.47% of transformed AC residual versus8.54% original, despite
2.36e-4 bulk phase MAE. Bulk/interface gradient-norm ratio changes0.030->2.89.
Near uniform solid, original reaction vanishes as epsilon but transformed
reaction tends to-N2/2=-8.8e8. Graph/CH identities, physical-c finite differences
and the pure-phase limit pass. Reason: algebraic cancellation need not improve
coupled optimization; reduced scaling/removing KKS failed to restore L2.
Rejected: claiming loss shares prove optimizer direction, assuming latent
updates always oppose physical error (the probed full gradient does not),
blaming simple FP32 residual roundoff, requiring hard IC in the theory, or
declaring a universal impossibility. Bounded/partial inverse weighting remains
an untested candidate, not a promoted default or demonstrated gain. Relation:
corrosion-latent-mechanism, coupled-latent-residual-experiments and
corrosion-kks-scale-ablation.

## 2026-10-25 — Test bounded inverse phase weights before removing scales

Decision: add explicit research-only bounded residual with eta1/0.1/0.01,
corresponding to spatial weight caps2/11/101. Screen at seed42 with the existing
AC1e-7/CH1 and5000-update/50 s budget, unchanged PDE/reference/KKS/soft IC/BC.
If both terminal physical-field errors improve, validate a second seed and
matched AC=CH=1 controls before any default or scale removal. Reason: the user
authorizes this candidate and conditional removal, while diagnosis separates
spatial bulk amplification from coupled equation scaling. Rejected: silently
regularizing the exact latent residual, changing physical coefficients,
reference-guided training, adding hard IC, selecting transient dips, or assuming
bounded spatial weights remove the need for equation balancing. These eta
values are explicit exploratory hyperparameters, not physically derived optima.
Relation: corrosion-bounded-residual and corrosion-latent-mechanism.

## 2026-10-25 — Replay original control for the bounded experiment

Decision: add a fresh5000-step/seed42 original-residual cold start with all
current pilot settings and a separate explicit run name. Retain the historical
original in the report. Reason: bounded cap2 exhibits late-step fluctuations;
current-runtime replay is a stronger control than comparison only to a saved
historical checkpoint. Rejected: selecting its4400-step dip, changing budgets,
or replacing/deleting prior results. Relation: corrosion-bounded-residual.

## 2026-10-25 — Keep corrosion defaults after bounded and no-scale validation

Decision: retain original residual/KKS/AC1e-7 defaults; keep bounded eta only
as an explicit research option. Eight new5000-step cold starts show eta0.1
(cap11) improves terminal phi/c at seed42 from0.03984/0.04722 to
0.01619/0.01928, but worsens validation seed7 from0.02664/0.03160 to
0.03661/0.04343. Both AC=CH=1 controls have phi L2~0.512, no terminal0.5
contour and large BC errors. Reason: limiting pure-phase amplification has
conditional value but is not a sufficient coupled preconditioner; bounded
spatial weights do not replace equation-scale balancing. Cap101 reproduces
wrong-endpoint concentration saturation. Rejected: promoting the screened
seed42 win, hiding the second-seed reversal/last-window errors, cancelling
pre_scales after failed controls, attributing all failure to KKS or saturation,
or claiming universal impossibility. Fresh original is primary; historical
controls remain labelled and untouched. Nine focused tests and ten-checkpoint
CPU/config/initialization audits pass. Relation: corrosion-bounded-residual.

## 2026-10-25 — Explain residual scaling with schematic and measured visuals

Decision: use a normalized latent-residual schematic with compression factor
sech^2(u) shared by melting and interface-centre-normalized corrosion, and a
separate measured frozen-checkpoint AC allocation graphic. Show bounded cap11
as a distinct finite weight, not the exact transform. Reason: the user asks
for intuition and illustrations rather than more algebra. Rejected: claiming
physical-field error equals residual, suggesting tanh alone changes the loss,
assuming gradient saturation is completely removed in a coupled network,
drawing fabricated training trajectories, or hiding seed-dependent corrosion
results. Relation: residual-visual-explainer and corrosion-bounded-residual.

## 2026-10-25 — Audit concentration fidelity and qualify stagger comparisons

Decision: separate geometry-induced concentration error from c_L/q errors and
verify actual stagger execution before attributing transformed failure to bulk
weighting. Reason: the user correctly notes that small phase error does not
make concentration imbalance unimportant. Research controls all disable stagger
and their runner has no stagger activation path; production alternates PDE
losses only during warmup, without freezing fields/parameters. Rejected:
calling amplified concentration error a defect of the transform, equating
transformed AC imbalance with reference-c error, claiming current controls test
the full original staggered scheme, or silently assuming a config-only toggle
works. This audit does not introduce a new formulation or launch training.
Relation: corrosion-concentration-stagger-audit, corrosion-latent-mechanism.

## 2026-10-25 — Preserve concentration/transport accuracy as a separate target

Decision: qualify earlier bulk-weighting interpretation and retain c_L/q/Dq/CH
metrics independently of phi/total-c L2. Seven saved-run audits show seed42
bounded total c L2 improves0.04722->0.01928 while c_L slightly worsens and graph
Dq error2.94->8.47; stable raw-output checks rule out simple reconstruction
roundoff. Saturated wrong constant concentration can instead have tiny CH
residual, so both residuals and field/boundary errors matter. Also reproduce
production stagger cache defect: masked weights saved as EMA can zero both
PDE coefficients at switches without refresh. Research runner now rejects
nonzero stagger until scheduling is actually wired. Reason: user seeks to
solve concentration imbalance, not hide it; all prior controls had stagger off.
Rejected: declaring bulk attention harmful, labelling shared-network loss
alternation frozen-field solves, treating config-only toggles as working,
claiming concentration value errors are uniformly large, or asserting a
stagger benefit before a matched2x2 trial. Production/defaults stay untouched.
Relation: corrosion-concentration-stagger-audit and corrosion-bounded-residual.

## 2026-10-25 — Execute corrected stagger/residual paired controls

Decision: repair persistent adaptive weights to exclude temporary masks/bias,
wire explicit activation bits into the research step, and run four fresh
seed42 original/full-latent x stagger0/1000 controls at5000 updates/50 s.
Use inherited25-update AC-first blocks and existing GradNorm refresh25 in
research, while testing nonaligned switches against production refresh behavior.
Keep KKS, soft IC/BC, physical/reference model and AC1e-7/CH1 unchanged.
Reason: user requests execution and concentration fidelity rather than hiding
bulk imbalance. Rejected: config-only toggles, forced-refresh-only bug masking,
reusing old off controls as the only baseline, treating alternating losses as
frozen fields, removing scales simultaneously, or choosing metrics/checkpoints
to claim improvement. Production default schedule is not changed. Relation:
corrosion-stagger-experiment and corrosion-concentration-stagger-audit.

## 2026-10-25 — Diagnose failed joint replay before interpreting stagger

Decision: retain all four completed paired runs and add one explicit archived
joint-step replay, using the current evaluator but the saved original make_step.
Reason: even the new stagger-off phase control collapses (phi/c0.5690/0.5468),
unlike the previous0.03984/0.04722 result despite identical saved config apart
from run name and identical initialization bytes. A real full-sized first-batch
CPU FP32 probe finds exactly equal loss, weights and parameter gradients with
the new all-true mask; this does not prove GPU trajectory reproducibility.
Rejected: interpreting failed baseline as a stagger benefit, blaming dynamic
mask/fusion without evidence, overwriting runs, or promoting a new default.
The supplemental control isolates the training step from evaluator changes;
it uses the same5000-update budget and preserves explicit override provenance.
Relation: corrosion-stagger-experiment.

## 2026-10-25 — Retain defaults after negative corrected stagger controls

Decision: retain original/KKS/pre-scales and existing schedule defaults, while
keeping the independent production raw-weight-state bug fix. Four corrected
5000-update/50 s controls finish but all fail: terminal phi~0.569, c~0.547,
c_L near zero, free q relative error~5.337 and Dq error~1. Actual on-schedule
counts500/500/4000 and positive adaptive weights pass independent verification.
Reason: loss alternation does not rescue concentration in this trial; even the
fresh original joint arm fails to reproduce the earlier accurate trajectory.
An archived original-step replay also fails. CPU first-batch gradients match
exactly; repeated identical-input GPU gradients vary in both old/new steps
(max relative1.12e-5/4.41e-4 in the recorded probe). This measures numerical
variation but does not isolate its source or prove causality for collapse.
Rejected: promoting small loss/transient dips, removing pre-scales, claiming
stagger is generally ineffective, replacing historical good results, or blaming
the new mask uniquely. Next separate diagnostic: establish stable current-
runtime joint replay before another optimization ranking. All outputs and
failed GPU equivalence-probe thresholds are retained in the archived report.
Relation: corrosion-stagger-experiment and corrosion-concentration-stagger-audit.

## 2026-10-25 — Separate boundary and initial-interface transport drive

Decision: audit pit-mouth and outer-solid Dirichlet errors separately, alongside
curved-pit IC and q/Dq, in accurate/collapsed saved checkpoints. Use exact-BC
substitution only as a frozen-field sensitivity probe, not a new solver or
training improvement claim. Reason: the user identifies boundary precision as
a possible cause of absent concentration gradients. The implemented KKS map
has q=c_L-c_le; pit targets imply q=-c_le, solid targets q=0, and analytic IC
c=c_se*h(phi) implies q=c_le*(h(phi)-1). A globally wrong c_L~0 can satisfy the
pit mouth while losing solid/initial gradients. Check these identities and
possible IC/Dirichlet corner mismatch explicitly. Rejected: treating total-c
BC MSE as concentration-drive accuracy, assigning a permanent Dirichlet value
to the moving arc, assuming pit-mouth precision alone is sufficient, or
changing physical BCs without evidence. Relation: corrosion-boundary-drive-audit.

## 2026-10-25 — Prioritize scaled concentration drive, not pit precision alone

Decision: qualify the boundary hypothesis with split checkpoint evidence and
propose an explicit normalized-s concentration BC ablation before any promotion.
Here s=c_L/c_le, with pit/solid targets0/1 and IC targeth(phi_0). Reason: the
collapsed original already has sampled pit c error1.99e-9 but solid c error
0.0356643 (the whole transport range); its initial Dq RMS is1.69e-6 versus
reference4.29762. The accurate checkpoint has solid error2.63e-4 and initial
arc q RMSE/c_le0.03496 versus collapsed0.70496. Frozen pit clamping has almost
no effect; solid clamping creates a boundary layer without directly restoring
the initial interface. Also disclose finite tanh IC/zero-mouth corner mismatch
(phi up to0.04949,c0.007105 on sampled mouth points), without silently changing
physics or claiming the checked reference is wrong. Rejected: requiring only
higher pit-mouth accuracy, interpreting total-c gradients as CH transport,
claiming BC causality from terminal association, guaranteeing normalization
avoids tanh saturation, or conflating loss normalization with a physical change.
No training/default/numerical implementation change. Relation:
corrosion-boundary-drive-audit and corrosion-stagger-experiment.

## 2026-10-25 — Execute normalized concentration data controls

Decision: add research-only physical/s-BC/s-BC+IC modes, keeping the legacy
make_loss_vector API physical for archived callers. New normalized factory
uses s=(1+tanh(raw_c))/2 directly, validates KKS/c_se1/c_le>0, keeps phase
targets and both PDE residuals unchanged. All six new controls use graph-node
analytic IC with declared Dirichlet nodes overriding at0 s, exactly the
existing independently solved graph initial state. This changes IC sampling,
not the declared discrete problem; no new continuous IC or hard constraint is
invented. Keep legacy random-interior IC as an explicit option. Probe XLA GPU
deterministic-ops before training; do not silently fall back if unsupported.
Reason: user requests action on missing concentration drive and corner issue.
Rejected: comparing normalized runs only against older random-IC controls,
normalizing BC and IC together without a BC-only arm, changing physical BCs,
clamping the moving arc, removing pre-scales simultaneously, using reference
trajectory targets or promoting a one-seed result. Relation:
corrosion-normalized-data and corrosion-boundary-drive-audit.

## 2026-10-25 — Require measured deterministic GPU controls

Decision: run the new cohort with explicit XLA_FLAGS=--xla_gpu_deterministic_ops=true
and a separate compilation cache, recording the flag in each summary. A small
scatter/JVP test and the actual full-batch archived old/new GradientRecorder
probe both execute on this GPU. Four identical-input calls in each path have
exactly zero repeated loss/gradient differences, unlike earlier unflagged
probes. Reason: reduce an established uncontrolled variation before judging
normalization. Rejected: silently changing precision/operator, assuming the
flag guarantees every future trajectory, or hiding historical results. A
separate same-configuration trajectory replay is still needed. Relation:
corrosion-normalized-data and corrosion-stagger-experiment.

## 2026-10-25 — Overlap independent deterministic controls

Decision: replace the serial launcher only, preserving/adopting its original
unchanged phase/physical worker PID47641, and run at most three independent
GPU workers with preallocation disabled. Same commands, seeds, data, flags and
budgets; no FEniCS work. Reason: measured deterministic training costs~0.20 s
per update on RTX3090 while using~2.6 GiB and~54% GPU utilization; CPU benchmark
is slower (~0.389 s/update), so independent concurrency reduces elapsed waiting.
The interrupted serial launcher is preserved as corrosion-normalized-serial-start.py;
its training child remains running. Rejected: truncating budgets, changing
physical precision/operator, restarting the first training run, or comparing
concurrent wall times as optimizer speed evidence. Whole-trajectory replay
still required; roundoff across configurations is not assumed identical.
Relation: corrosion-normalized-data.

## 2026-10-25 — Align BC samples with graph initial data before formal controls

Decision: stop and preserve the three incomplete phase pilot workers, record
last logged/evaluated updates and reason, and use graph boundary nodes as well
as graph IC nodes for the formal six controls under new run names. Add an
explicit random/matched_graph boundary-sampling field, preserving legacy paths.
Reason: graph IC fixes phi0.05103 at the non-Dirichlet mouth endpointx0.45,
while a random zero-BC sample lies atx0.450552. Their t0 target jump requires
a spatial slope~92.4, versus the graph's resolved endpoint-to-neighbor spacing
0.01. Keeping off-grid BC with nodal IC adds a sharper constraint than the
chosen reference discretization. This correction is based on target geometry,
not observed winning/losing training metrics. No smooth physical IC or moving
boundary condition is invented. Rejected: silently treating partial runs as
5000-update controls, overwriting/restarting their folders, leaving the extra
off-grid corner constraints in an allegedly matched graph study, or changing
BC values/physical PDE. Final cohorts all keep BC/IC point sets identical.
Relation: corrosion-normalized-data.

## 2026-10-25 — Probe initial transient without changing the training problem

Decision: independently integrate the same graph ODE over5e-6 s and0.05 s,
using only an equivalent Tc rescaling, while the formal controls run unchanged.
All four rate coefficients scale linearly with Tc (asserted), so this samples
the original physical autonomous ODE earlier rather than changing its physics.
Reason: normalized IC restores initial q/Dq, while later transport error remains
large; check the early layer and current c_L output-map representability before
blaming residual scaling alone. This diagnostic does not replace reference,
feed trajectory targets into training or affect the matched cohort. Rejected:
guessing an early time scale, changing t_min mid-cohort, or claiming an early-
transient cause before measurement. Relation: corrosion-normalized-data.

## 2026-10-26 — Validate BC-only gain without hiding transport error

Decision: finish independent six-run audit, fully replay the physical seed42
baseline and run paired seed7 phase/physical versus phase/s_bc5000-step
controls. Formal seed42 terminal phi/c: phase physical0.57072/0.54629,
s_bc0.01177/0.01402,s_bc_ic0.41937/0.44152; latent physical0.56722/0.54568,
s_bc0.42001/0.44036,s_bc_ic0.44363/0.46614. Reason: BC-only gives a strong
field gain, but terminal Dq error12.13 is poor, while stronger IC only improves
initial drive. Latent data controls do not justify promotion. Rejected:
promoting a one-seed field win, hiding Dq with total-c L2, assuming accurate
initial data guarantees later dynamics, or treating quota-blocked tool reads
as completed actions. The earlier tool approval-service usage-limit failure
is resolved by a normal successful retry after the user's continue request;
no approval check is bypassed. No production default change yet. Relation:
corrosion-normalized-data.

## 2026-10-25 — Date correction and spatial transport audit

Correction: the preceding entry headed2026-10-26 was written on2026-10-25
(verified UTC host clock); its numerical records are unchanged. Decision:
locate the BC-only candidate's remaining Dq errors and reconstruct the exact
fixed100-batch core coverage, including boundary/interface/bulk and never-
core regions. Reason: independent CPU audit confirms terminal Dq relative
error12.13 and native CH RMS~7 despite low field L2. Held-out BC errors are
also retained. Rejected: dismissing transport error as reconstruction noise,
assuming every graph row is trained by a finite fixed pool, or changing
sampling before measuring where the error resides. This is read-only; no
cohort or validation parameter changes. Relation: corrosion-normalized-data.

## 2026-10-25 — Complete normalized-data controls and retain transport qualification

Decision: deliver original-residual+s_bc as an explicit field-improving research
option using matched graph BC/IC; preserve production and legacy research
defaults, KKS and AC1e-7/CH1 scales. Completed5000-update/50 s seed42 phi/c
falls0.57072/0.54629->0.01177/0.01402; seed7 falls0.57042/0.54746->
0.00620/0.00738. Full latent and stronger IC normalization are not promoted.
Reason: the field gain repeats, but free-node terminal Dq relative errors
12.13/1.10 and late fluctuations remain. For seed42,99.9507% of squared Dq
error lies in the worst1% of nodes, chiefly near mouth endpoints; the dominant
row is sampled and has nonzero output derivative. This does not establish a
unique optimization remedy. Eight independent CPU audits,16 focused tests and
a full exact5000-update baseline trajectory replay validate the recorded
comparison without resolving earlier historical divergence. Rejected: hiding
transport error behind total-c L2 or favorable regional metrics, removing
pre-scales, claiming a transformed-residual win, or replacing production
defaults on field accuracy alone. Archive all sources/metrics/figures and
preserve every completed, replay and explicitly stopped run at direct paths.
Relation: corrosion-normalized-data and corrosion-boundary-drive-audit.

## 2026-10-25 — Separate transport evaluation from outlier weighting

Decision: optimize the normalized-BC research candidate with explicit raw-KKS
edge-difference transport and CH RMS loss controls, preserving physical PDE,
reference, KKS output, AC1e-7/CH1, nodal data and5000-update budget. Derivation:
q=cle*(tanh(v)-1)/2 follows directly from c=cle*s+dc*h(phi), and graph action
sum_j w_ij(q_i-q_j) equals diag*q+off for the fixed graph. RMS and L1 have
residual units but different gradients; an explicit negligible smooth-zero
floor will prevent undefined RMS gradients at zero. Reason: localized Dq
hotspots and native/stable residual differences were independently measured.
Rejected: assume floating-point cancellation is the unique cause, change
physical BCs or graph, feed reference gradients into training, oversample
hotspots without accounting for an objective change, or claim RMS is a proven
fix before matched experiments. Legacy code paths/defaults must stay explicit
and compatible. Relation: corrosion-transport-optimization.

## 2026-10-25 — Freeze checked transport controls with cancellation scope measured

Decision: execute a complete phase2x2 evaluation/norm screen plus latent/raw/RMS
using identical s_bc/nodal data,50 s,5000updates and deterministic GPU flags.
Include a fresh legacy-L1 baseline rather than relying only on historical
trajectories. Reason:19 tests pass and actual GPU legacy first-batch losses,
weights and parameter gradients match saved sources exactly. Frozen accurate
model native-versus-stable CH evaluation error falls0.003003->0.000229, but
CH RMS remains~7.00367; evaluation cancellation is not the principal hotspot.
Rejected: present reduced roundoff as improved trained physics, truncate
unfavorable controls, change sampling or scales within the screen, or use
concurrent wall times as optimizer-speed evidence. Sources remain frozen
through the cohort. Relation: corrosion-transport-optimization.

## 2026-10-25 — Distinguish nodal BC accuracy from neighboring free-node transport

Decision: retain the planned loss/evaluation controls rather than add a hard-BC
training change in response to the remaining hotspot. Reason: frozen s_bc
seed42/7 model q substitution by exact prescribed graph BC values leaves Dq
relative errors12.13297->12.13328 and1.10441->1.10456, respectively. This
isolates the finite nodal BC-value contribution, not all effects of boundary
geometry on fitting. Free-node concentration structure remains defective.
The new legacy baseline also matches the previous1000-update checkpoint bytes
exactly. Rejected: claim tighter boundary values alone resolve the remaining
transport defect, infer successful training from a frozen substitution, or
rank early RMS transport improvement without phase/c accuracy. Relation:
corrosion-transport-optimization.

## 2026-10-25 — Test bounded phase weighting after concentration-data correction

Decision: add one existing cap11 bounded-phase/raw-edge/L1 control, queued
until the first three workers exit, keeping maximum three GPU jobs. Reason:
prior bounded tests used different physical/random data losses; the new s_bc
cohort corrects the concentration-drive fitting scale. Test this combination
against the present phase/raw-edge/L1 arm and full latent/raw-edge/RMS arm,
without claiming a pure phase-formulation comparison against the latter.
The bounded-versus-phase pair holds CH norm/evaluation fixed. No new numerical
source, loss formula, extra scale removal or reference prior is introduced.
All six sources remain identical; this adds one5000-update cold start to use
the last wave's free GPU slot. Rejected: assume the earlier mixed-seed bounded
outcome generalizes across this changed data objective or silently switch the
production residual. Relation: corrosion-transport-optimization.

## 2026-10-25 — Prepare a late-rate control without altering frozen workers

Decision: prepare a separate research-only optimizer helper preserving the
original SOAP/exponential schedule through3000updates, then continuously
reducing its rate to1e-5 at5000. Do not wire or launch it until all six current
workers have captured their frozen sources; test schedule and actual SOAP
updates first. Reason: seed42 baseline has phi/c0.00295/0.00347 at3800 but
0.01458/0.01734 at4200, and its unchanged final learning rate is~77% of the
initial1e-3. This motivates a controlled late-refinement test if current
variants do not give stable joint gains. It adds no updates, physics, hard IC,
reference prior or optimizer-state reset. Rejected: choose a favorable transient
checkpoint as the new final result, change active workers, or simultaneously
change the loss norm/scales when testing the schedule. Relation:
corrosion-transport-optimization.

## 2026-10-25 — Validate raw-edge gain and a controlled late-rate refinement

Decision: after all six initial workers capture identical original sources,
wire the separately tested research-only late-rate helper and queue three
more cold starts: raw/L1 tail seed42, raw/L1 legacy-rate seed7, raw/L1 tail
seed7. First3000updates retain1e-3*0.95^(step/1000); the remaining2000 decay
continuously to1e-5, with no state reset or additional updates. Reason:
raw/L1 final5000 phi/c0.008835/0.010425,q0.013319,Dq1.70254 improves all
listed seed42 terminal metrics, while late variability remains and cold RMS
fails fields. Both seeds/rate arms are predefined before tail results; retain
all negatives. At most three GPU workers. The core optimizer stays unchanged;
new snapshots explicitly add the helper and identify runner/config wiring
differences, with original numerical function ASTs checked. Rejected: modify
active frozen workers, silently mix source versions, resume model-only files
with invented optimizer state, add training steps, or claim roundoff reduction
alone proves the cause of trained gains. Relation: corrosion-transport-optimization.

## 2026-10-25 — Validate bounded/raw/L1 field gain without hiding q tradeoff

Decision: add a seed7 cap11/raw/L1 cold start after all three later workers
have started and a slot frees, maintaining maximum three GPU jobs and no
source changes. Reason: bounded seed42 final phi/c0.001433/0.001719 is far
below raw original0.008835/0.010425, but q0.017305 is above raw0.013319 and
Dq1.5004 remains appreciable. This is a field candidate, not a validated full
transport winner. Preserve5000updates,50 s and all other data/scales. Final
audit expands to13 models (ten new, three prior); original rate/tail and
bounded alternatives remain separately ranked. Rejected: promote one final
field dip without seed validation, omit concentration checks, or silently
stop the already queued late-rate tests. Relation: corrosion-transport-optimization.

## 2026-10-25 — Test late RMS after second-seed concentration regression

Decision: run a final paired L1->RMS continuation control for seeds42/7: first
4000joint updates identical to raw/L1, last1000 switch only CH reduction to
RMS, with the same optimizer/state/BC/IC/scales/legacy rate and5000-update
budget. Add explicit disabled-by-default ch_rms_after field and validate an
initial L1 norm and GradNorm-refresh alignment. Reason: raw seed7 final phi/c
0.004225/0.005006 improves prior0.006202/0.007379, but q/Dq0.009607/1.7298
regresses from0.006203/1.1044; tail seed7 also regresses q/Dq0.010414/2.7319.
Cold RMS failed geometry, whereas later concentration outliers still need
attention. The final two controls test whether a correct field basin permits
RMS refinement. All prior ten sources were checked frozen before wiring;
no prior worker changes. Maximum three GPU jobs. Rejected: hide seed7
transport regression, promote field error alone, assume a late RMS benefit
without measurement, extend an existing run beyond5000 or add trajectory
supervision. After these12 new controls, report the full tradeoff rather than
continue an open-ended parameter search. Relation: corrosion-transport-optimization.

## 2026-10-25 — Close transport study with field gain and explicit concentration tradeoff

Decision: retain original phase/KKS/scales/defaults; deliver raw-KKS edge
transport as an optional field/roundoff improvement, without all-metric
promotion.12 fresh5000-update controls and15-model CPU audits finish. Raw/L1
phi/c improves seed42 0.01177/0.01402->0.00884/0.01042 and seed7
0.00620/0.00738->0.00422/0.00501; q/Dq improves seed42 but regresses seed7
0.00639/1.1044->0.00990/1.7298 (free nodes). Tail improves seed42 but seed7
Dq becomes2.7319. Bounded gives seed42 phi0.00143 but seed7 fails0.4220;
its seed42 late median0.01813 is worse than raw0.00719. Final4000L1+1000RMS
controls give phi0.10172/0.04770 and do not solve concentration/phase jointly.
Reason: scalar residual emphasis alone does not yield a reliable physical win.
22 focused tests, exact5000 baseline replay and exact pre-tail3000/pre-RMS4000
parameter/loss/evaluation prefixes pass. All outputs, adverse controls and
source schemas remain at direct paths. Rejected: call a field win solved
transport, select one seed/dip as default, remove pre-scales, promote full
latent or late RMS, or continue an open-ended parameter search. Archive this
completed controlled study. A separate next study may test concentration-only
parameter blocks/true field freezing; its benefit is unmeasured. Relation:
corrosion-transport-optimization and corrosion-normalized-data.

## 2026-10-25 — Display matched corrosion fields without selecting a favorable arm

Decision: show seed42 final5000 original and stable raw-KKS/edge L1 fields at
physical0/10/25/50 s, with unchanged reference, shared color scales, interface
contours, signed errors and separately scaled q. Reason: the user requests
representative current physical fields after the L2 comparison. The original
already uses s_bc/matched IC/BC. CPU reload and terminal-summary checks pass.
Keep the worse initial q fit visible (0.06039->0.26282) alongside positive-time
phi/c improvement. Rejected: present only the favorable tail/bounded seed42
terminal dip, conceal initial mismatch, or infer all-seed transport success
from visually similar total concentration. No training or defaults change.
Relation: corrosion-physical-field-gallery and corrosion-transport-optimization.

## 2026-10-25 — Show electropolish extra bands and distinguish the older pilot

Decision: display existing seed42 original/transformed final5000 fields at
0/1/2.5/5 s against the validated graph reference, with shared scales, phi0.5
contours, signed errors and5 s centerline profiles. Reason: the user requests
the analogous electropolish view. Both models have three vertical crossings
per column at all selected positive times, reference one; transformed terminal
phi/c L2 worsens0.28604/0.29443->0.34398/0.37920. CPU reload and all selected
norms reproduce saved results. Explicitly identify these as earlier physical-
data original/full-latent pilots, without later corrosion s_bc/raw-edge changes.
Rejected: imply corrosion's latest improvements have been tested here, rank
solutions only by nearest-interface distance, or infer a unique cause from
the images. No new training, physical/default changes or Git history operation.
Relation: electropolish-physical-field-gallery and coupled-latent-residual-experiments.

## 2026-10-25 — Optimize electropolish with data and temporal-window controls

Decision: isolate matched nodal IC/BC and normalized concentration BC objectives
for original/full-latent phase residuals at5 s/5000 SOAP updates/lr1e-3. Add a
separate explicitly validated time-window curriculum if band defects remain,
with second-seed validation of promising settings. Reason: initial fits are
good while both pilots create extra bands at positive times. Corrosion data
normalization is a motivated test, not an established electropolish remedy.
Temporal sampling can strengthen early evolution without prescribing interface
motion or using reference trajectory values. Keep KKS/PDE/scales/reference and
soft ICs; retain q/Dq and topology alongside phi/c L2. Rejected: impose a
reference-shaped output, monotonicity constraint, extra labels or longer budget,
or assume the original residual is already accurate. Relation:
electropolish-field-optimization.

## 2026-10-25 — Recover historical electropolish context without mixing controls

Decision: evaluate historical SOTA checkpoints read-only on the current matched
101x51 reference and retain them as context, separate from the new controlled
cohort. Reason: SOTA ckpt005000 gives phi/c L2 0.02608/0.03106 and one vertical
interface per column, unlike the poor reduced pilots; ckpt005800 gives
0.01996/0.02388. Training used301x151 full spatial batches, lr0.005, Nt50,
stagger1000 and a6000-update configuration, so these are cross-discretization
evaluations, not evidence for a single optimization cause or equal budgets.
Production zero-based checkpoint labels also differ from completed-update
research labels. Physical parameters/rough surface/time scales are checked
identical; no checkpoint/artifact is altered. Rejected: present pilot defects
as the best result ever available, or attribute the historical gain uniquely
to stagger/grid/lr. Keep the predefined current6 controls running. Relation:
electropolish-field-optimization.

## 2026-10-25 — Predeclare current-baseline replay and independent electropolish validation

Decision: after six seed42 controls complete, run current-code random-IC/random-BC
original controls in seeds42/7 and one seed7 nodal-data candidate. Choose the
candidate by lowest final5000 terminal phi+c L2 among those with exactly one
vertical crossing in every column at all21 saved times; if none qualify, stop
selection and diagnose. Reason: historical poor pilots differ in sampling and
runtime/source history, and older SOTA differs in grid/batching/rate/stagger.
Contemporary random-data controls isolate the nodal-data cohort under the
same current source/deterministic runtime; seed7 tests selection robustness.
No earlier checkpoint substitution, new physical prior or budget extension.
Retain q/Dq separately because the declared field criterion is not a transport
criterion. All queued workers wait until six complete; max3 GPU jobs. Relation:
electropolish-field-optimization.


## 2026-10-25 — Close electropolish study with topology recovery and explicit seed tradeoff

Decision: retain original phase/KKS/scales/production defaults and deliver
full-grid analytic IC/graph-node BC/s_bc as an optional single-interface field
candidate. Nine5000-update/5 s runs,11-model CPU audits and3 focused sampler
tests complete. Current random seed42 phi/c0.290928/0.299013 becomes
0.002106/0.002518; seed7 candidate0.002389/0.002869 is worse than successful
random0.001138/0.001358. q free improves both seeds (seed42 0.15809->0.00310,
seed7 0.01151->0.00500), but seed7 Dq0.03776->0.05838 and c_t0.02941->0.06225
regress. Candidate late phi median improves in both observed seeds despite
seed7's worse endpoint; do not substitute it for the terminal comparison.
Both candidate seeds retain one interface at all21 times. Time-window
curriculum reduces full-latent phi0.5194->0.04677/eliminates extra bands, yet
remains less accurate than original; original-window endpoints/transport do
not beat no-window counterparts. Reason: large seed42 field/topology recovery
is verified without demonstrating all-metric/every-seed superiority. Grid ICs
use more known initial-value samples, not reference trajectory labels, so equal
update budgets do not imply equal work. Source/config/initial/checkpoint,
finite JSON/images/links/syntax/diff checks pass; all processes exit. Archive
full controls, historical context and counterexamples. Rejected: universal
promotion, conceal successful random seed7 or transport regression, attribute
the combined IC/BC strategy to one unique cause, call lower residual solved
physics, or extend into open-ended tuning. Relation: electropolish-field-optimization.


## 2026-10-25 — Plot raw electropolish training L2 with both seeds and metric scopes

Decision: show four current-source random/candidate histories in separate
phi/c panels, with shared logarithmic scales, raw200-update evaluations and
explicit terminal5 s versus pooled21-time definitions. Keep the successful
random seed7 control and visible late oscillations. Reason: seed42 improves
strongly but seed7's candidate endpoint is worse, so one selected trajectory
or smoothing would obscure the observed tradeoff. Rejected: identifying this
original-residual candidate as a full-latent result, replacing endpoint L2
with late-window medians, or treating equal updates as equal compute. Four
26-point histories and saved summary endpoints pass checks; plots and direct
source paths archived without retraining/default changes. Relation:
electropolish-convergence-curves and electropolish-field-optimization.


## 2026-10-25 — Select a saved phi checkpoint for the requested3x5 electropolish panel

Decision: rank all45 saved1000-update checkpoints across the completed current
nine-run cohort by logged terminal phi relative L2. The selected random-data
seed7 checkpoint005000 has0.001137819, lower than the nodal-data candidate.
Reason: the user now requests the best phi checkpoint; this figure's criterion
is explicit and differs from the earlier seed42 phi+c strategy selection.
Reload on CPU, verify against saved predictions, then show only phi reference,
prediction and pointwise absolute error at t*=0,.25,.5,.75,1 with common scales.
Rejected: choose a favorable time/checkpoint separately for each column, label
the chosen random baseline as the improved strategy, or claim it is best over
unsaved models and all possible metrics. No training/default changes. Relation:
electropolish-best-phi-panel.


## 2026-10-25 — Complete electropolish3x5 phi comparison with shared error scaling

Decision: archive the requested three-row/five-time phi panel using selected
random seed7 checkpoint005000. CPU replay agrees with saved GPU phi within
1.61e-6 and reproduces terminal L2=0.001137826. Common field0–1 and absolute
error0–0.01740267 scales preserve comparisons across times. Reason: one
checkpoint and one absolute-error scale avoid time-by-time model or visual
selection. Five per-time metrics and all45 ranked checkpoints/direct paths
are retained with raw plotted fields. Rejected: per-panel error normalization
or implying a terminal-selected baseline proves a universally better strategy.
All figure/artifact checks pass; no training/default changes. Relation:
electropolish-best-phi-panel.


## 2026-10-25 — Promote observed terminal-field SOTA through canonical train.py entries

Decision: honor the new user request to promote best verified terminal-field
presets, including seed: corrosion bounded eta0.1/raw-edge L1/s_bc/nodal data,
seed42; electropolish original/physical/random data,seed7; retain melting
IC-only latent/SOAP,seed42. Rank KKS final5000 checkpoints by phi+c, melting
by graph-reference phi. Reason: the user now requests default SOTA rather
than the previous policy of retaining legacy defaults; this authorization
supersedes earlier no-promotion decisions. Both KKS winners have seed and
transport limitations, which remain explicit in READMEs/index. Preserve
legacy presets and dispatch canonical train.py to the validated KKS engine
instead of reimplementing its optimizer/loss loop. Extract unchanged pure
samplers to avoid importing CLI modules from that engine. Rejected: claim
universal/cross-seed or transport SOTA, silently use stronger motion priors,
change physical models/pre-scales, overwrite saved runs, or rerun costly
training solely for reorganization. Relation: phasefield-sota-defaults.


## 2026-10-25 — Complete three-case SOTA default and train.py consolidation

Decision: promote the user-authorized observed terminal-field winners through
canonical train.py/train_base.py in all3 cases, retaining explicit legacy
configs and original shared numerical updates. READMEs/index give resolved
settings, winning seeds, direct checkpoints/references and mixed robustness/
transport evidence. Corrosion bounded seed42 phi/c0.001433/0.001719 is not a
seed7 or Dq remedy; electropolish original/random seed7 0.001138/0.001358 does
not rescue seed42. Melting IC-only latent/SOAP retains graph phi6.1097e-5.
Reason: fulfill the requested best-observed defaults without converting one
endpoint into a universal strategy claim. Source/data helpers avoid importing
CLI modules; fresh named outputs and deterministic GPU configuration preserve
the declared entry behavior. Configs/same-CPU initial arrays/samplers match the
saved studies; CPU-vs-GPU initial draw rounding max4.77e-7 is explicit. 30
numerical and3 final entry tests,3 actual reduced CPU entry/save smoke checks,
syntax/paths/links/JSON/artifact inventory/diff validation pass. A new test
stub initially returned MagicMock instead of the trainer's None; only that
affected entry suite was rerun after correction. Rejected: overwrite old
models, rename external artifacts, silently ignore unsupported legacy switches,
claim full cold-start reproduction without rerunning5000 updates, or repeat
unchanged expensive numerical tests. Relation: phasefield-sota-defaults.


## 2026-10-25 — Stop treating winning seeds as strategy parameters; predeclare2026 validation

Decision: test current corrosion/electropolish/melting strategies once each
at new seed2026, exactly5000 updates and the same physical horizons/configs.
Treat random seed as recorded run provenance, not a tuned strategy component.
Reason: user rejects seed-dependent SOTA selection and asks for independent
stability evidence. Keep fixed geometry/data/pool/evaluation RNGs to isolate
the existing train.seed-controlled initialization/time streams, documenting
this scope. Rank/report final states and late variability, with q/Dq/topology
for KKS; do not substitute intermediate checkpoints or reroll failed runs.
Retain serialized seed metadata for direct checkpoint compatibility. Rejected:
choose a successful seed after looking, reduce budgets/horizons, silently tune
losses/sampling, or call one extra success a cross-seed guarantee. Relation:
phasefield-seed-stability and phasefield-sota-defaults.


## 2026-10-26 — Move canonical random seed from strategy configs to run CLI

Decision: canonical train_base configs omit train.seed; all3 train.py entries
accept --seed, resolved into copied saved-run metadata with a common untuned
default42. Old serialized config seeds remain loadable, with explicit CLI
seed overriding them. Reason: user explicitly rejects seed as a strategy
parameter; selecting seed7 to make electropolish's default appear accurate
would continue the selection bias. Preserve that best run as history, not a
promised default result. Runtime resolution handles locked absl configs and
preserves the caller. Controls already launched at2026 before this API edit;
their loaded configs/source snapshots/math stay unchanged. Actual argv is
retained and the launcher reproduction source uses --seed. Rejected: discard
seed provenance, change RNG algorithms/geometries/pools during validation,
or claim moving the CLI repairs stochastic optimization. Relation:
phasefield-seed-stability.


## 2026-10-26 — Complete predeclared seed validation; reject stable-SOTA claims for failing strategies

Decision: retain and report all3 final5000 seed2026 states, no rerolls or
intermediate substitution. Corrosion bounded phi/c0.407016/0.431025, late phi
median0.404946 and false phase band/2 liquid components demonstrate failure
(seed7 also failed). Independent Dq free error4.01390. Electropolish original/
random phi/c0.00283171/0.00342674 and one interface at all21 times reproduce
low fields, but earlier seed42 failure remains. Melting full-grid graph phi
5.98573e-5 matches prior6.10970e-5; this supports two observed initializations,
not a statistical guarantee. Reason: best-seed promotion was inadequate as a
strategy stability criterion. Canonical configs now exclude seed; run-only
--seed has common untuned default42, recorded only in resolved metadata with
legacy config compatibility. Preserve current mathematical settings for this
fixed-control comparison, without inventing a robust replacement.10 focused
runtime/melting tests,18 legacy preset parity checks, source/config isolation,
all3 independent CPU field/transport/topology replays, raw curves and galleries,
paths/JSON/syntax/image/link/diff checks pass. All workers finish/GPU idle;
3 new run directories indexed, train.log retained with checkpoints; old assets
unchanged. Rejected: hide corrosion failure behind seed42, describe the new
electropolish success as universal stability, conflate sampled/slice/full L2,
or claim seed alone uniquely explains all stochastic optimization mechanisms.
Relation: phasefield-seed-stability and superseding seed-based SOTA selection
in phasefield-sota-defaults.


## 2026-10-26 — Diagnose fixed-data randomness hypothesis using common saved-model probes

Decision: distinguish fixed IC/BC data, fixed PDE core pools and changing
initialization/time streams; compare successful and failed models on common
full-grid/time/BC/IC/residual diagnostics before attributing seed sensitivity
to sample randomness. Reason: user asks for the mechanism, while corrosion
already uses all nodal IC/BC points and successes share the same fixed data.
No new training or changed physical/default strategy in this diagnostic turn.
Rejected: treat missing randomness as proven cause, infer a local minimum from
L2 alone, or promise resampling cures stiff coupled optimization without a
controlled run. Relation: phasefield-seed-mechanism and phasefield-seed-stability.


## 2026-10-26 — Attribute cross-seed evidence to optimization sensitivity, not proven loss of IC/BC randomness

Decision: report frozen common-grid evidence and causal limits. All nodal
corrosion IC/BC data are present; failed2026 BC error6.28e-5 is smaller than
successful42's2.45e-3, with both initial phi errors~1e-3 or less. Same initial
parameter bytes and configs differing only phase/bounded+run name at seed7
give terminal phi0.00422 versus0.422. Failed bounded2026 full/pool AC losses
1.194/1.139 remain high;35 never-core nodes contain only0.935% phase SSE.
Intermediate successful42 removes extra bands by3000 while2026 retains them.
Reason: this supports state/trajectory-sensitive optimization in the stiff
coupled loss, rather than a sufficient IC/BC missing-randomness explanation.
Electropolish failed42 additionally has full-grid initial phi error0.0792
versus successful~0.0016, so mechanisms need not be identical. Preserve
localized-gradient/sampling hypotheses and distinguish strict local minima
from insufficient fixed-budget convergence.7 final/10 intermediate CPU norms,
reference/coverage normalization, product-rule AD, JSON/syntax/plot/link/diff
checks pass; no new training or production/default change. Rejected: claim
unique causality, declare fixed sampling harmless in all cases, blame GradNorm
caps alone, or promise random resampling cures the problem. Relation:
phasefield-seed-mechanism and phasefield-seed-stability.


## 2026-10-26 — Preserve current results before controlled objective optimization

Decision: user explicitly authorizes a local commit of current modifications;
commit on dev without merge/push. Then test original2026 and frozen-weight
bounded7/2026 with unchanged5000-update budget, spatial samples, physics and
reference. Reason: same-seed7 original/bounded contrast and the verified
|r| grad(w) term motivate an isolated gradient intervention. Frozen weights
retain the forward residual/zero set but define a different update direction;
no improvement or gradient of the original scalar objective is claimed.
Rejected: seed shopping, changing multiple sampling/physics mechanisms at once,
or treating fixed BC data as the established cause. Relation:
corrosion-robust-objective.


## 2026-10-26 — Validate frozen weighting at the previously successful seed too

Decision: add bounded_frozen seed42, queued until an original worker exits,
keeping at most3 GPU workers. Declare it before any new final5000 result.
Reason: improvements on failed7/2026 alone cannot show the modification retains
successful behavior. Compare all3 objectives at42/7/2026 with final checkpoints
and six late-window evaluations. Reject choosing a different seed per objective
or treating good total-field L2 as proof of accurate q/Dq transport. Relation:
corrosion-robust-objective.


## 2026-10-26 — Select corrosion default by observed cross-seed reliability

Decision: change corrosion's canonical formulation from bounded to original
phase, retaining raw KKS CH edges, s_bc, full nodal IC/BC, SOAP/lr1e-3/5000
and every physical parameter. The historical best bounded42 checkpoint remains
recorded separately. Reason: original42/7/2026 all finish with phi<=0.01075 and
c<=0.01273; bounded fails7/2026 near0.4. Original's worst late phi median is
0.00952, whereas frozen7 alone already has0.01463; remaining frozen42 cannot
reverse that worst-case comparison. Frozen weighting rescues the failed bounded
seeds and improves some final transport values, but does not consistently beat
original late field/transport errors; retain it as an explicit research control.
No seed is included in the strategy config, and no single-run L2 or universal
robustness guarantee is claimed. Rejected: leave the known2/3-failing default,
choose a winner by best intermediate checkpoint, or hide q/Dq tradeoffs. Relation:
corrosion-robust-objective.


## 2026-10-26 — Complete controlled frozen-weight study with mixed outcomes

Decision: retain original phase as the revised corrosion default and frozen
weighting as an explicit research option. Four new5000-update/50s runs finish;
all9 final models pass independent CPU field/transport/topology/config audits.
Reason: frozen7/2026 reduce prior bounded phi~0.4 to~0.01, but frozen42 gives
phi/c0.01457/0.01727 and Dq3.01. Original three-seed worst phi/c0.01074/0.01273,
q0.01372, Dq1.757 and late phi median0.009514 are all smaller than corresponding
frozen worst cases. All original/frozen fields have one liquid component over
21 times. Dq errors remain substantial, so no concentration-gradient cure is
claimed. Reject selecting the lowest intermediate model, masking frozen42
regression, attributing failures solely to fixed IC/BC samples, or promoting a
universal generalization claim. Retain historical bounded42 as an individual
field-best record; no old assets move. Archive sources/manifests/raw metrics/
plots and close task; GPU idle. Relation: corrosion-robust-objective.


## 2026-10-26 — Benchmark local FP64 before committing to full training

Decision: honor the user's request to measure this machine first. Run serial
isolated FP32/FP64 workers on identical promoted stored geometry/parameters,
with JAX highest matmul precision, synchronized timings, explicit floating dtype
checks, graph kernels and the real corrosion SOAP/GradNorm step. Report compile,
warm-up, steady timing and memory separately; no full training or default
precision change. Reason: large GEMM alone does not predict this sparse,
small-network, multi-loss workload, and current explicit FP32 data would defeat
an x64-flag-only comparison. Reject an advertised peak-throughput estimate as
a substitute for measurement or claiming L2 improvement from timing probes.
Relation: fp64-performance.


## 2026-10-26 — Separate native GEMM throughput from production compiler policy

Decision: supplement the serial realistic benchmark with GEMM-only runs under
production deterministic=true and the ordinary XLA policy, both highest precision.
Reason: measured strict FP32/FP64 large GEMM ratio alone can conflate hardware
throughput with compiler restrictions; actual sparse training remains the
primary decision metric. Repeat GEMM-only strict32 also verifies a corrected
IR-type diagnostic (initial regex missed shaped xf32 tokens; input/output and
independent numeric checks already passed). No training-policy change or graph
rebenchmark is needed. Relation: fp64-performance.


## 2026-10-26 — FP64 core is feasible for the measured corrosion workload

Decision: report measured FP64 feasibility and retain FP32 production defaults
until a separate accuracy experiment. Eight serial GPU workers and1 IR audit
finish: realistic SOAP/GradNorm mean0.203262->0.226737s/update (~11.5% slower),
peak active1.535->3.063GiB, compile157.0/155.7s.5000-update-only estimates
16.94/18.89min exclude compilation/evaluation/saves. Large4096 GEMM is57.39/
216.04ms (~3.76x); ordinary/deterministic policies give similar values, so do
not attribute the actual result to that flag. Model/data/PDE/gradient/SOAP tensors
are FP64 with exact2x floating storage; only the unchanged Optax scalar LR
schedule remains FP32. Reason: actual measured workload matters more than square
GEMM ratio and this device has memory headroom. Reject claiming an L2 improvement,
a complete5000-step run, literal all-operations FP64, or automatically changing
training precision. Sources/raw samples/dtype evidence archived; checks pass,
GPU idle. Relation: fp64-performance.


## 2026-10-26 — Isolate FP64 arithmetic before changing graph coefficients

Decision: run the approved5000-update corrosion seed42 accuracy control through
the canonical train.py with common FP32-generated parameters/data/time draws
and stored graph arrays promoted exactly. Retain the stored diagonal and audit
its row-sum rounding defect separately against the exact-row-sum reference.
Reason: regenerating geometry, initial RNG draws or diagonal coefficients would
confound arithmetic precision. Reject x64-flag-only training, silent operator
correction and precision default promotion from one run. Model/PDE/SOAP arrays
are FP64; unchanged Optax scalar LR schedule remains FP32. Relation: corrosion-fp64-accuracy.


## 2026-10-26 — Record rather than silently repair stored-diagonal roundoff

Preflight verifies exact GPU initialization replay and identical reference edges/
weights. Stored FP32 diagonal differs from float64 row sums by at most4.8828e-4
(relative3.49e-8); AC coefficient amplifies constant-field defect to502.93,
or5.03e-5 after AC L1 scale. Decision: retain it in the predeclared arithmetic-only
control, report the limitation and do not infer high-accuracy operator equivalence
from matching edge weights alone. Stable CH edge evaluation annihilates constants.
A diagonal-consistent control, if needed, is a separate numerical evaluation
change, not an unreported FP64-only fix. Relation: corrosion-fp64-accuracy.


## 2026-10-26 — Retain FP64 as an explicit control, not the stable default

Decision: close the paired5000-update seed42 FP64 accuracy control and retain
FP32 default. Deliver train.compute_dtype with common FP32 inputs promoted,
unchanged stored diagonal/scales/residuals/SOAP and explicit dtype records.
Reason: independent CPU final phi/c improve0.008835/0.010426 to0.002677/0.003207;
free q halves, Dq1.703->1.220 and CH RMS0.982->0.704. But late phi median only
improves~16%, mean/max worsen, and late Dq median1.706->1.852 worsens. FP32
at4800 already reaches phi0.002689. Frozen-model inference precision changes
fields<1.7e-6, indicating different learned trajectories rather than evaluation
precision alone. Reject final-checkpoint-only claims of lower error floor,
resolved concentration gradients, improved late stability or cross-seed benefit.
One pair is complete; further paired seeds or separately specified row-sum
correction remain potential follow-up. Source/results/plots archived, all
training exits0 and GPU idle; no Git operation. Relation: corrosion-fp64-accuracy.


## 2026-10-26 — Distinguish batched time JVP from point-count speedup claims

Decision: confirm the current shared spatial batch and already-enabled time JVP;
do not claim Nx*Nt-fold acceleration over properly jitted/vmapped pointwise grad.
Reason: all-ones JVP extracts the desired derivatives because cross-time blocks
vanish, while network/AD batch work still scales with point count. Independent
CPU float64 two-field checks match pointwise grad exactly. Python loops/full
Jacobians are different baselines; parameter reverse AD is still necessary for
the current many-parameter loss. No timing, retraining or production change.
Relation: time-jvp-structure.


## 2026-10-26 — Benchmark local AD with matched active residual counts

Decision: compare current corrosion graph/time-JVP against local tensor and
joint-uniform-space/time AD at equal active PDE counts, with same model,
initialization, SOAP/GradNorm, nodal IC/BC, data scales and precision. Measure
FP32/64 separately; include sampling dispatch in synchronized update timings,
report halo overhead, compile and isolated residual timing. Local signs follow
the existing verified local PDE; direct stable-q Laplacian is independently
checked against the mixture chain rule. Reject claiming equal physics/reference
accuracy from this throughput study or assuming random pairing forbids JVP.
No production/default change, full convergence experiment or Git operation.
Relation: local-ad-performance.


## 2026-10-26 — Preserve time JVP in the local tensor control

Decision: local tensor sampling uses the same batched time JVP principle as
current nonlocal training, with AD only replacing the spatial operator. Joint
space/time sampling uses per-field time grad as requested; its time-only JVP
control verifies alternative differentiation remains valid. Reason: needlessly
using separate reverse time sweeps in the shared-time local control would conflate
spatial locality and avoidable time-AD overhead. Updated independent float64
manufactured/mixture/data/mixed-gradient preflight passes. Relation: local-ad-performance.


## 2026-10-26 — Diagnose the unexpectedly large graph/local throughput gap

FP32 graph229.3ms/update versus local tensor12.0ms (~19x), both same active
PDE/data counts and validated AD. Decision: after all six serial training-step
workers finish, add isolated graph-action/edge-action kernel controls under
production deterministic=true and ordinary compiler policy, same shapes/values
and independent SciPy checks. Reason: halo overhead alone does not establish
the cause; scatter determinism may contribute and needs direct measurement.
This is a read-only microbenchmark, not a training-policy change or a claim of
ordinary-policy reproducibility/whole-step performance. Relation: local-ad-performance.


## 2026-10-26 — Measure ordinary full updates before extrapolating kernel gains

Isolated kernels validate against SciPy: production deterministic graph
aggregation~38ms versus ordinary~0.17ms for the actual padded batch, both
precisions. Decision: add two serial ordinary-policy graph full-step workers
(FP32/64), same125-update protocol, after all main/operator workers end.
Reason: kernel policy explains a major PDE-forward cost but is not itself a
whole-step acceleration measurement. Preserve original default deterministic
policy and do not claim ordinary-policy training reproducibility or equal L2.
Relation: local-ad-performance.


## 2026-10-26 — Separate deterministic graph bottleneck from local-AD cost

Decision: close the throughput study with defaults unchanged. All12 serial
workers exit0. At equal active PDE counts and shared network/data/SOAP, FP32
current graph/local tensor/local joint means229.31/11.98/16.12ms; FP64
266.11/31.32/47.77ms. Isolated deterministic graph actions~38ms versus ordinary
~0.17ms, independently correct against SciPy. Actual ordinary-policy graph
updates12.49/35.13ms confirm a major compiler-policy-sensitive graph bottleneck;
ordinary graph and local tensor are close. Joint sampling is slower at equal
count here; time grad and JVP remain both valid. Reject explaining the large
default-policy gap by AD mode or equations alone, extrapolating kernel ratios
into training speed, claiming equal local accuracy/boundaries or promoting
ordinary-policy long-run stability from125-step timing runs. Archive source/raw
records/plots; no production/config/reference/Git history changes. Potential
next work is deterministic aggregation optimization or separately matched local
accuracy experiments. Relation: local-ad-performance.


## 2026-10-26 — Distinguish nonlocal structural value from untested PINN gains

Decision: identify defensible benefits as lower spatial regularity demands for
bounded integrable kernels, explicit reciprocal pair/energy structure, field-only
spatial-operator interfaces and finite-range physical modeling. Independently
verify actual reference graph symmetry/nonnegative weights, constant nullspace,
pair energy and internal flux cancellation. Treat bounded high-frequency
amplification as mathematical structure and improved optimizer conditioning as
an untested hypothesis. Reject claims that current nonlocal PINNs are proven
more accurate/stable, automatically conserve mass under Dirichlet reservoirs,
have fully corrected AC row sums, or necessarily outperform a local target
after finite-delta/quadrature/boundary bias. Du/Yang2016 and primary nPINN/PDDO
references inform scope; external formulations are not transferred as proof of
corrosion performance. No training/default/production/Git change.
Relation: nonlocal-advantages.


## 2026-10-26 — Complete local boundary/reference before full L2 controls

Decision: run four5000-step FP32/seed42 cold starts (default graph, ordinary
graph, local tensor, local joint) with shared initialization/data/scales/SOAP.
Local reference and PINN explicitly impose natural phi_n=q_n=0 on non-mouth
bottom, traceable to existing FEM weak form; do not rank a missing-BC local
benchmark against graph reference. Use independently refined local MOL/BDF,
retain graph reference and measure model-reference differences separately.
Extend canonical train.py research options without promoting new defaults;
ordinary compiler-policy L2 remains an experimental full-run control.
Relation: local-ad-full-training.


## 2026-10-26 — Retain measured local reference uncertainty

Decision: retain all four independent local MOL solves and explicitly qualify
local accuracy against the401x201 FD reference. Temporal201-grid refinement
changes fields by at most2.75e-9, but201->401 terminal phi/c differences remain
0.01417/0.01681 (101->201:0.02886/0.03423). Spatial convergence below1e-3 is
false and recorded as such. Reject calling this a spatially converged continuum
truth or using sub-percent local errors to rank methods. Full5000-update runs
remain informative for training failure/success and policy controls; report
reference-model bias and refinement separately. No extrapolated or silently
corrected reference. Relation: local-ad-full-training.


## 2026-10-26 — Keep graph default after four full accuracy controls

Decision: keep deterministic nonlocal FP32 defaults. All four paired seed42
cold starts finish5000 updates/50s with identical initial arrays. Finalphi/c:
default0.008835/0.010426; ordinary0.020680/0.024565; local tensor0.346112/0.373221;
local joint0.411813/0.425739 against own-model references. Ordinary is much
faster but worsens final/late field errors, while improving q/Dq; no universal
accuracy regression or robustness guarantee follows. Local inherited scales/
SOAP plus six-loss no-flux objective fail this budget; reject claiming intrinsic
nonlocal superiority from one seed/different boundary objectives, or blaming
34–43% field error on1.4–1.7% reference refinement. Retain explicit research
presets, separate common-reference scores, cache-dependent observed timings,
all curves and independent CPU reload/transport audit. Relation: local-ad-full-training.


## 2026-10-26 — Explain policy effects as lowering plus training sensitivity

Decision: separate execution determinism, single-operation numerical accuracy
and trained-model L2. New frozen JAX/jaxlib0.10.1 controls with3032 edges show
deterministic scatter lowering to a3032-iteration while; ordinary retains scatter.
Ten repeats are exact for deterministic forward/input gradient, versus relative
variation6.47e-8/6.07e-8 ordinary. Existing SciPy comparisons give O(1e-8) errors
for both. Full-run first loss/weight vectors differ only1.47e-7/2.96e-7 relative,
but weight vectors differ44% at step200. This supports floating-point perturbation
and sensitive training paths; reject treating0.884% versus2.068% finalphi as a
guaranteed deterministic accuracy benefit, or attributing amplification uniquely
to SOAP/GradNorm/scatter without isolation. Fixed-neighbor deterministic forward
and transpose aggregation is a potential performance direction, not implemented
or promised numerically identical. No training/default/Git change.
Relation: compiler-policy-explanation.


## 2026-10-26 — Test ordinary FP64 without combining separate benefits by assumption

Decision: complete one ordinary-policy FP64 seed42 cold start at5000 updates/50s,
same common FP32 inputs promoted exactly and same stored graph/scales/SOAP.
Reuse three completed policy/precision controls. Reason: ordinary FP64 throughput
is measured (~35ms/update), but deterministic FP64 accuracy does not establish
ordinary FP64 accuracy. Reject inferring additive speed/accuracy benefits or
promoting defaults before the actual full-run/late-window/transport audit.
No extra seeds, mathematical changes or Git operation. Relation: ordinary-fp64-accuracy.


## 2026-10-26 — Retain ordinary FP64 as a promising, nonuniform one-seed control

Decision: keep defaults unchanged after the completed ordinary FP64 control.
5000-update/50s finalphi/c0.00249850/0.00294446, wall196.5s vsdefault1179.8s:
this realization is faster with lower final and median field errors. Initial
values match all four exactly; CPU reload/dtype/config/source/transport checks
pass. However, latephi max0.014885 and mean0.006661 exceed default0.008835/
0.006348; late Dq spike18.665 is much worse than default max1.915. Final q/Dq
improve versus default but regress versus deterministic FP64. Reject claiming
uniform stability/transport/cross-seed superiority or a fair fresh-cache compile
speedup. Archive complete curves and direct checkpoint; no extra optimization,
seed trials, numerical/default or Git history changes. Relation: ordinary-fp64-accuracy.


## 2026-10-26 — Audit each case's actual policy before generalizing the slowdown

Decision: measure canonical electropolish and ic_latent melting FP32 workloads
under both policies with identical initial/data/graph arrays and production step
functions,125 updates (50 warm-up/75 synchronized measured), plus frozen graph
forward/input-gradient/HLO controls and independent numerical checks. Electropolish
default adapter enforces determinism; melting entry inherits environment (currently
unset), so distinguish current default behavior from artificially forced policy.
Inspect elasticity/corrosion1d statically; do not silently repair missing1d default
train_pinn.py or choose a different physical regime. No full accuracy cohort,
production/default/physics/Git change. Relation: phasefield-gpu-policy-audit.


## 2026-10-26 — Distinguish an enforced KKS bottleneck from inherited melting policy

Decision: close the other-case audit with defaults and numerical code unchanged.
All four serial FP32 workers complete125 updates,50 warm-up/75 measured with
identical complete pool/initial/time arrays. Electropolish197.73/10.98ms and
melting468.26/26.62ms deterministic/ordinary (~18.0x/~17.6x). Current optimized
HLO shows3032-edge forward/gradient loops for electropolish and11929-edge
forward/three-gradient loops for melting; independent frozen forward/transpose
errors stay below3e-7. Electropolish adapter enforces deterministic default;
melting inherits environment and already uses ordinary policy when unset.
Reject saying melting is necessarily slow by default, inferring equal final L2,
transferring measured ratios to elasticity/1D, or automatically choosing a1D
physical regime to repair its missing default config. Document explicit options,
separate compilation/steady timing and preserve all raw evidence.
Relation: phasefield-gpu-policy-audit.


## 2026-10-26 — Validate ordinary defaults against complete field trajectories

Decision: run four ordinary electropolish controls (FP32/64 x seeds42/2026),
ordinary FP64 corrosion2026 and ordinary FP32 melting42 before selecting
canonical defaults. Existing42/2026 controls are reused. Preserve the established
cross-seed field-reliability criterion alongside final/late L2 and throughput;
do not choose a seed or declare the fastest short benchmark numerically optimal.
Centralize policy parsing in training.runtime, keep explicit conflict rejection
and unrelated flags, and add ordinary policy explicitly to melting before any
backend/filesystem work. This makes its existing ordinary behavior explicit;
KKS candidate runs use explicit overrides while canonical promotion awaits data.
Historical checkpoints still load using their physical/architecture templates;
no saved metadata/artifact path is rewritten or in-place resume promised.
Relation: ordinary-policy-defaults.


## 2026-10-26 — Select ordinary precision per case from complete controls

Decision: canonical corrosion uses ordinary FP64; electropolish ordinary FP32;
melting explicitly ordinary FP32. All six predeclared5000-update jobs finish;
independent CPU reloads, exact paired initial values, allowed-config/reference/
dtype/history/final-checkpoint/topology checks pass. Corrosion42/2026 phi/c
0.002499/0.002944 and0.006833/0.008077 improve final fields versus deterministic
FP32; worst late median0.005004 versus0.009514. Late maxima/Dq spikes remain
worse, so selection is a speed/field-accuracy compromise, not uniform stability.
Electro ordinary FP3242/2026 phi0.004757/0.003399 rescues historical42 failure;
ordinary FP64 still fails42(phi0.299757), reject universal precision promotion.
Melting explicit ordinary full-grid phi3.9053e-5 agrees with independent CPU.
Keep physical models/references/data/losses/scales/SOAP/horizons unchanged;
seed is run provenance. Preserve historical local FP32 deterministic and
graph-ordinary FP32 research presets explicitly so default changes do not alter
those controls. Retain deterministic override with conflict validation, direct
old loading and fresh run paths. No Git history operation.
Relation: ordinary-policy-defaults.


## 2026-10-26 — Close validated per-case ordinary defaults

Decision: finish promotion after six full GPU jobs,10KKS and1full-grid melting
CPU reloads,11focused tests,7affected policy-parser checks and2actual canonical
GPU starts. Shared parser additionally rejects a bare deterministic Boolean
flag in ordinary mode; empty-flag numerical behavior of completed runs remains
unchanged. Current-vs-saved selected defaults match numerically; only run name,
checkpoint cadence and optional melting evaluation reference differ. Preserve
historical FP32 ordinary graph and deterministic local research presets, prior
default/index records and old artifact paths. Archive all failures/late spikes
alongside selected results; no additional seed search or global-optimum claim.
README/index/memory synchronized; task closed, no workers or Git operation.
Relation: ordinary-policy-defaults.


## 2026-10-26 — Recommend phase-field benchmarks and matched-density two-phase flow

Decision: recommendations only; first scalar extension is periodic2D CH spinodal
decomposition, while preferred first coupled-flow case is equal-density/viscosity
CHNS elliptical-droplet relaxation without gravity. Primary PFHub1–4 and
Jacqmin1999 / Colli-Frigeri-Grasselli2011 / Zeng et al2023 sources support models.
Existing scalar negative-Laplacian machinery covers diffusion/chemical potential;
flow requires gradient/divergence, pressure gauge and compatible conservative
transport/capillarity. Distinguish local-model nonlocal derivative approximation
from a native nonlocal interaction-energy model. Do not claim arbitrary operator
replacement preserves mass/energy or removes actual CHNS training stiffness.
Reject starting with high density-ratio rising bubbles, unvalidated contact lines
or presuming full-local AD must use fourth derivatives (mixed mu lowers order).
No case implemented, train started, default/reference/physics changed or Git
history operation. Relation: phasefield-case-recommendations.


## 2026-10-26 — Use Qiu2022 as a checked local-AD baseline, include later mixed form

Decision: attached35-page Qiu et al PF-PINNs manuscript is a useful local CHNS
reference: four outputs(u,v,p,C), AD including fourth-order CH, sequential time
windows, material-coefficient clipping only, PDE+IC/BC forward losses. Confirm
two benchmarks: prescribed reversed vortex, then coupled1000:1 rising bubble.
Treat reported u/v errors5.55–15.12%/3.56–7.92%, ~1.12% area drift and published
reference comparisons as evidence, not sub-percent field accuracy or exact
conservation. Rendered pages7/14 confirm potential/initial-condition formula
inconsistencies; no silent correction or assumption about actual author code.
Chemical-potential scale, phase/flux BCs, coordinates/gravity and clipping
derivative compatibility require explicit resolution before implementation.
Same-group2022 MIM follow-up already outputs chemical potential, so include
AD mixed formulation as the efficient comparator for nonlocal mixed form.
Prefer reversed-vortex pilot before full CHNS; keep physical reference/model/
windowing/optimizer/budget fixed between derivative methods. No training,
default, PDE/code modification or Git history operation.
Relation: two-phase-pinn-paper-review.


## 2026-10-26 — Publish prior results before developing the simple CH transport pilot

Decision: user explicitly authorized commit/push then dev development. Prior
results committed05780d4; regular main merge21fe6d9 pushed origin/main; switched
back to clean dev before new-case changes. Generated archives only had whitespace
normalization needed for staged diff check;48Python AST/65JSON/314directory
checks pass, existing numerical validations retained without unnecessary rerun.
First example is periodic2D CH transport by constant prescribed velocity, with
explicit pilot data and mixed(C,q) outputs, not a silent repair/reproduction of
Qiu's unspecified wall/flux/initial formulas or full high-density CHNS. Simple
periodic IC-only hard condition introduces no prescribed future motion. Compare
same model/initial network/optimizer/points with spatial AD versus periodic
finite-horizon negative Laplacian. Normalize discrete isotropic kernel moment
(sum w*z_x^2=sum w*z_y^2=2), record this derivative approximation choice and
validate positivity/nullspace/Fourier consistency. Independent Fourier IMEX
reference and graph/local refinements distinguish errors. No new-case commit
or publication planned automatically; develop on dev.
Relation: twophase-transport-pilot.


## 2026-10-26 — Accept the end-to-end prescribed-flow pilot without an accuracy-superiority claim

Decision: close first periodic CH route after both predeclared5000update runs and
CPU checkpoint reloads. Mixed nonlocal/AD own-target terminalphi0.002297/0.005121,
trainer wall66.7/91.1s, observed full-step2.16/3.18ms excluding sampling/eval;
initial models byte-equal and configs differ only method/name. Independent
reference time refinement<8.3e-8, local spatial refinement<2.1e-11; graph/local
terminal discrepancy0.008198. Against common local target nonlocal0.008542
is worse thanAD0.005121, so reject claiming intrinsic accuracy superiority.
Area drifts0.264%/0.897%, energy decreases at21saved times; not exact learned
conservation or statistical robustness. Phase excursions also occur in refined
reference and are shown without clipping. Keep this case explicitly prescribed
flow/hybrid differentiation, not full two-phase NS or a Qiu reproduction.
Four numerical checks, strengthened manufactured AD/nonlocal residual check,
actual startup and source/config/initial/reference/reload checks pass.
Prior snapshot main21fe6d9 remains published; new development remains on dev
uncommitted for review. Relation: twophase-transport-pilot.


## 2026-10-26 — Add an explicit periodic reversed-vortex deformation control

Decision: preserve constant-flow train_base and old reference identity; introduce
train_vortex with explicit structured velocity in place of the constant vector.
This discriminated union keeps existing serialized configs exact, without hidden
required defaults. Use Qiu Eq23/24 velocity U0=2,T0=2 on periodic unit square,
but declare own periodic rounded-droplet IC, epsilon0.02,M0.0001 and T=2.
Variable advection is conservative Fourier flux divergence, explicitly extrapolated
with nonlinear CH in BDF2; only biharmonic diffusion stays implicit. Independently
check reference time/grid refinement and mass. Strain supplies energy, so reject
using monotone phase-field energy or exact return to IC as acceptance conditions.
Compare mixed AD/nonlocal under same model/initial draws/5000updates before any
accuracy claim. No coupled NS or exact paper reproduction claim; no Git operation.
Relation: twophase-vortex-pilot.


## 2026-10-26 — Record cross-backend replay roundoff explicitly for vortex audit

Decision: CPU audit reports per-field replay max-absolute/relative-L2 differences
and uses an explicit5e-6 FP32 tolerance (FP64 stays1e-10). Initial3e-6 check
fails at one nonlocal phase value3.0398e-6; full two-model diagnostic finds max
3.3007e-6. Frozen GPU replay matches both stored phi/q arrays exactly; CPU FP32
relative phase differences<4.08e-7 and promoted FP64 inference also stays within
8e-6 of saved fields. Evidence supports backend/precision rounding rather than
wrong checkpoints; no model/data/training or source changes to the finished
cohort. Reject silently ignoring replay errors or modifying saved predictions.
Only evaluator validation/reporting changes after training, recorded in task.
Relation: twophase-vortex-pilot.


## 2026-10-26 — Accept the deformation route with explicit accuracy limits

Decision: both prescribed-vortex5000-update controls learn crescent deformation
and reverse-return. Common local-target phi L2 at t1/t2 is nonlocal0.05479/0.07889
versus mixedAD0.07463/0.10186. Own-target nonlocal terminal0.08410 remains separate
from graph/local reference gap0.01168. Nonlocal compiled update3.30ms versus10.31ms,
trainer76.85/136.54s including JIT/eval, but soft-area drift3.37% is worse than
AD2.90%. Reject all-metric/global/stability claims or declaring high accuracy:
interface-band phi L2 remains34.3%/37.4%. Keep constant-flow base default; retain
new vortex preset as an explicit harder experiment. Checkpoint CPU audit/frozen
GPU parity, exact paired inputs, independent reference refinement and7focused
tests support route correctness. Source/results/scripts/plots archived; no new
training, full NS, Git history operation or default promotion.
Relation: twophase-vortex-pilot.


## 2026-10-26 — Test bounded inverse-Jacobian preconditioning on vortex CH

Decision: user requests melting residual idea for sharper/more accurate vortex
interfaces. Strict phi=tanh(u) excludes verified CH excursions (local min-1.09837,
graph-1.06899), and Lq does not factor J=1-phi^2. Preserve physical network/IC/PDE/
reference and use explicit positive bounded weight w=sqrt(1+eta²)/sqrt(eta²+
min((1-phi²)²,1)), eta0.1, range[1,10.0499]. This empirical optimization norm
recovers inverse-J inside(-1,1) away from endpoints as eta->0; its floor prevents
downweighting large excursions, cap avoids unbounded bulk stiffness. Keep full
AD of the weight; no exact latent-map/cancellation or interface-weight claim.
Compare three same-initialization5000-update arms: original, phase transport only,
both mixed residuals. Original configs without the optional explicit residual
record retain their original schema/behavior; new preset requires mode/eta.
Reject changing CH to enforce a maximum principle or conflating output mapping
with residual preconditioning. No Git operation or default promotion planned.
Relation: twophase-vortex-residual.


## 2026-10-26 — Separate CH phase representation from exact inverse-map residual scaling

Decision: first3 bounded-weight controls finish; both transformed arms worsen
interface/phase/mass metrics. Before judging melting's actual latent method,
add3 same-budget controls with an overshoot-compatible bijection f(u)=tanh(u)+
alpha*u,alpha0.1. Unlike strict tanh, f covers all real phase values and f'>=alpha.
Compute u0=f^-1(phi0) by8 fixed Newton steps (independently check value/implicit
AD); use phi=phi0+tanh(u0+s)-tanh(u0)+alpha*s, s=(t/T)N. This guarantees the
same numerical IC exactly and encodes no subsequent motion. The phase map is
monotone unbounded; chemical potential stays an independent output. Test
original/transport/both inverse-map residuals, w=(1+alpha)/f'(u); positive
bounded w is the exact inverse-map Jacobian up to constant normalization. No
claim Lq/reaction automatically cancels. Preserve physics/reference/architecture/
random parameters/5000steps and compare map-only against transformed map arms.
Stage1 source snapshot is frozen/verified before these changes. No default/Git
operation. Relation: twophase-vortex-residual.


## 2026-10-26 — Keep vortex map-only interface gain separate from residual-scaling tradeoff

Decision: all6 predeclared5000update seed42 controls/replays complete. Physical-
field bounded inverse weights worsen common-local phi from0.07997 to0.21941/
0.27171. Unbounded leaky-tanh map with ORIGINAL residual improves phi to0.04192,
common-local band0.30327->0.16045, width proxy error48.4%->8.57%, contour mean
0.004804->0.002797, but area drift worsens2.74%->4.29%. Exact inverse-map BOTH
residuals slightly improve globalphi0.04019, qRMS0.05488 and area2.07%, while
worsening band0.17455, width17.77% and contour0.004478. Do not attribute map-only
gain to residual transformation, claim all-metric dominance or robust superiority
from one ordinary-FP32 pair. Keep both controls and negative results; sharper
interface experimental setting is latent config with mode=original, final5000.
Established/canonical defaults unchanged.12focused tests,6CPU replays, same
initial parameter bytes/reference and stage-frozen sources pass; original model
still reloads with new static-only fields. Sources/results/curves/fields archived,
6new runs registered. No Git history operation or further workers.
Relation: twophase-vortex-residual.


## 2026-10-27 — Classify inverse-map residual scaling as restricted nonlinear preconditioning

Decision: answer the user's generalization question with S(z)=W(z)R(G(z)),
separating state-coordinate/right transformation, nonlinear left residual norm
and parameter-space optimizer preconditioning. B=DG is distinct from full PDE
Jacobian A and neural parameter Jacobian. Scalar inverse-g' normalizes time;
for mixed CH G=(g(u),v), full map inverse is diag(1/g',1), while scaling chemical
row too is an additional block-norm choice. Preserve both existing research modes;
clarify terminology without changing code or formulas. At roots D S=W*A*B;
when square W=B^-1 this is a similarity, not a general conditioning/stiffness
improvement theorem. Invertible weights preserve roots but finite least-squares
solutions and physical-field errors change. Propose coupling/operator/target-error
metrics as broader directions, with constraints and no universal accuracy claims.
Primary Glasner2001/PETSc sources checked; independent NumPy derivative/Newton/
conditioning/least-squares identities pass. No new training/default/PDE/Git change.
Relation: residual-preconditioning-theory.


## 2026-10-27 — Treat residual-preconditioned PINN loss as established prior art

Decision: the explicit literature search finds direct prior art, including
Liu2024 Eq20 ||P^-1(Au_theta-b)||² and Song2025 PDE-Jacobian/ILU residual
preconditioning. Earlier FEINN2023 applies inverse operators to weak residuals;
Wen2026 neural-operator preprint includes Allen–Cahn, multigrid and block norms.
Do not claim novelty for placing preconditioning in PINN loss, general nonlinear
extension, generic block/spatial treatment or phase-field application alone.
Distinguish inverse scalar output-map derivative from full PDE-state Jacobian
and parameter preconditioning. Narrow nonlocal/conserved phase-field metrics,
cost/error tradeoffs and map-based analysis need further targeted prior-art
review and controlled evidence; search absence does not prove novelty. Reject
renaming standard residual weights as a new general framework or presenting
recent preprints as confirmed peer-reviewed publications. Primary formulas,
dates and scopes are recorded in the archived report. No numerical, default,
reference or Git-history changes.
Relation: pinn-residual-preconditioning-literature.


## 2026-10-27 — Separate convexified-chemical phase mapping from CH block preconditioning

Decision: target the user's approximately1% overall phi L2 with explicit controls.
Replace the heuristic leaky map experimentally by u=phi+phi³=F'(phi)+2phi,
F=(phi²-1)²/4; derivative1+3phi²>=1 yields an unbounded analytic inverse
g(u)=2/sqrt(3)*sinh(asinh(3sqrt(3)u/2)/3). Preserve exact IC via differences.
This convexified chemical coordinate is an independently derived trial mapping,
not claimed to be a published optimal representation. Separately freeze bulk
F''(+-1)=2 and temporal scale tau to form A_tau=[[1/tau,ML],[-(2+eps²L),I]].
Its Fourier block inverse gives zphi=(rphi-ML*rq)/(1/tau+ML(2+eps²L)),
zq=rq+(2+eps²L)zphi. Optimize norm(zphi/tau,zq), with explicit positive tau;
this normalizes temporal units while preserving roots/zero mode. It omits
advection and variable curvature, so no exact nonlinear/time inverse claim.
Use literature-inspired frozen Jacobian, not inverse output-map weights.
Check dense inversion/spectral operator equivalence and parameter gradients
before cold-start controls. Preserve old schemas/checkpoints/physics/reference,
retain all negative results and use paired budgets/initial parameters. Common-
local terminal reference bias~1.17% is reported separately; no data supervision,
reference replacement, default promotion or Git operation.
Relation: twophase-operator-preconditioning.


## 2026-10-27 — Add explicit PDE-only deterministic L-BFGS refinement controls

Decision: first5000-step original/cubic runs remain materially underconverged
while the frozen Schur metric alone does not improve them. Do not claim a
representation or preconditioning success. Freeze stage1 numerical sources
before implementing an optional second optimizer stage through train.py. Use
SOAP5000 followed by explicit full-grid midpoint-time L-BFGS (SciPy), no field
reference targets, with convergence/line-search status and function-evaluation
budget recorded. Use FP64 for this refinement cohort to avoid FP32 line-search
resolution, and clearly separate its budget/precision from5000-step controls.
Retain original/cubic/Schur comparisons so any refinement gain is not attributed
to the map alone. Evaluate unchanged21 reference times, distinct from the fixed
midpoint training times; choose final iterates, not best reference checkpoints.
No default/Git change.
Relation: twophase-operator-preconditioning.


## 2026-10-27 — Repair trainable integer periodic frequencies before further accuracy claims

Decision: actual checkpoint audit discovers PhaseNetwork.modes is an inexact
Equinox parameter, so SOAP changes integer periodic modes. Historical mapped
original stores[0.76716584,1.8836843,2.9716012]; edge field differences confirm
periodicity is not exact. This is an implementation defect against the declared
periodic formulation, not a new boundary model. Stop parameter gradients through
these fixed feature modes in raw(), preserving array leaves/dtypes/values so old
checkpoint inference and spatial/time derivatives retain their saved values.
Do not silently round old modes, overwrite checkpoints or present historical
runs as verified hard-periodic solutions. Cold starts retain exact integer modes.
Test mode gradients, optimizer updates and opposing-edge values/derivatives.
Finish/freeze the four pre-fix controls and disclose the limitation; new controls
are a distinct fixed-frequency cohort. Shared learnable FourierEmbedding remains
unchanged: only case-local integer modes must stay fixed. L-BFGS parameter
partition must likewise exclude or have zero derivatives on this leaf.
Relation: twophase-operator-preconditioning.


## 2026-10-27 — Test causal space-time block inverse and higher spatial-feature resolution

Decision: fixed-periodic FP64 refinement does not yet reach1%; distinguish
representation limitations from the pointwise pseudo-time metric. Preserve
completed/running refinement source snapshots. Add a causal lower-bidiagonal
backward-Euler approximation of the frozen mixed CH Jacobian across sorted
training times: zphi_i=(zphi_(i-1)+dt_i*(rphi_i-ML*rq_i))/(1+dt_i*ML*B),
zq_i=rq_i+B*zphi_i, B=2+eps²L, zphi_0=0. This inverse is exact for that
discrete frozen block system and approximates PDE error propagation, omitting
advection/variable curvature; it changes only the residual norm. Normalize
phase by explicit terminal time, retain constant spatial mode, test against
an independent full space-time dense solve and full parameter finite differences.
Positive distinct time samples use explicit open-stratified sampling; fixed
midpoints are used in refinement, not reference fields. Separately increase
fixed integer spatial modes3->8/width64->96 to resolve thin-interface corrections,
with original residual control. Report architecture/dtype/budget changes rather
than attributing them to scalar mapping or preconditioning. No physics/reference/
canonical default/Git change.
Relation: twophase-operator-preconditioning.


## 2026-10-27 — Add exact grid-mass projection and initial-PDE collocation controls

Decision: the resolved raw-loss run develops~24% soft-area drift and~41%
terminal L2, supporting an optimization/initial-transient failure rather than
a monotone capacity benefit. Preserve this negative control. Add explicit
optional phase_mass=exact_grid_mean: subtract the mean of predicted phase minus
known initial phase on the fixed training grid. This is the orthogonal projector
onto the conserved-mass affine space, adds no trajectory/reference prior and
permits CH overshoots. Differentiate its time-dependent mean in JVP; its spatial
query derivative is zero since the quadrature grid is fixed. All training/eval/
checkpoint audit paths must use the same physical field wrapper. Also optionally
collocate the ORIGINAL PDE at t=0 via separately logged initial phase/chemical
loss terms, avoiding a sampled initial transient bypass. Compare original and
causal residual metrics on the previous64unit/3mode cubic representation, with
identical mass/anchor constraints/budgets. These controls change training beyond
a scalar mapping; do not attribute any accuracy gain solely to preconditioning.
No default, physics/reference/history change.
Relation: twophase-operator-preconditioning.


## 2026-10-27 — Align periodic conservative advection with nodal nonlocal residuals

Decision: checkpoint diagnostics show large AD versus nodal spectral-flux
advection discrepancies: refined cubic RMS.2551; failed resolved SOAP model
.9263, with AD mean advection up to.11436 despite exact periodic frequencies.
The latter's full-time AD phase residual RMS.15065 versus spectral residual
.94098 supports a collocation/derivative inconsistency, not proof of a unique
cause. Add explicit advection=spectral_flux: FFT divergence of v*phi, using
zero Nyquist first-derivative multipliers on the even periodic grid. This
represents the conservative derivative of the interpolated nodal flux, removes
independently adjustable AD point slopes and annihilates the constant mode
exactly to roundoff. Continuous equivalence follows div(v)=0; finite-grid
advection differs intentionally from the legacy hybrid. Existing independent
NumPy/SciPy reference already uses this conservative spectral flux; no data or
reference change is needed. New explicit presets compare original/causal metrics
with identical cubic coordinate, exact grid mass, initial-PDE anchor and budget.
Verify manufactured analytic flux, discrete mean, residual/parameter gradients
before runs. Keep the legacy AD path/default and all failed controls. No claim
that a changed spatial discretization is merely a scalar residual scaling.
Relation: twophase-operator-preconditioning.


## 2026-10-27 — Prioritize general consistency and low complexity after user steering

Decision: user explicitly relaxes the1% target and requests general rather than
overdesigned optimization. Finish existing controls, then compare additive,
cubic and causal-cubic using the SAME conservative spectral flux, exact grid
mass,64unit architecture,SOAP5000+LBFGS1000 and reference. Omit the proposed
extra t=0 PDE weight from these presets, retaining it only as an explicitly
tested optional research control. This makes the main comparison about
discretization consistency/conservation, representation and residual metric
without more hand-designed weights. Recommend the simplest measured winner;
do not add new problem-specific maps or optimize seeds/checkpoints. Distinguish
matched nonlocal and common-local errors and report negative experiments.
Tool review quota interruption has recovered; no safety rejection or bypass.
Relation: twophase-operator-preconditioning.


## 2026-10-27 — Check initial-time PDE collocation as a sampling-only improvement

Decision: consistent cubic final errors are highest just after initialization
(matched3.64–4.12% at t.1–.3 versus2.54% terminal), while current uniform
interior samples never include t=0. Add one optional explicit
sample.include_initial_time=True that concatenates t=0 to the PDE collocation
times in both SOAP and L-BFGS loss evaluation, with the SAME equal per-time
mean and no new loss weight or reference data. This is ordinary PDE endpoint
collocation, distinct from hard prescribing future dynamics. Reject combining
this zero interval with the experimental causal inverse, which requires strictly
positive times. Compare a final same-budget conserved-cubic cold start with
and without this one initial PDE point, then confirm the selected simple preset
on a predeclared independent seed2026. Retain earlier initial_pde_weight as a
separate research option; do not combine both forms of initial collocation.
Relation: twophase-operator-preconditioning.


## 2026-10-27 — Retain simple original residual and test optimizer convergence budget

Decision: equal-budget consistent additive/cubic/causal gives local pooled
3.23485%/2.84283%/25.17413%, terminal3.06985%/2.34319%/26.81794%. Extra
initial PDE sampling gives2.82532% pooled but worsens terminal2.53405% and q;
do not adopt this small mixed result. The selected simple candidate uses cubic
coordinate, ORIGINAL residual, conservative nodal flux and exact grid mass.
Its L-BFGS is still improving at the1000 iteration limit, so the final accuracy
check changes ONLY the explicit optimizer budget to3000 L-BFGS/5000 function
evaluations after the unchanged5000 SOAP. Predeclare seed42 and independent2026
cold starts, same strategy; no seed tuning, new mapping/weights or reference
change. This tests ordinary optimizer convergence rather than overdesigning
preconditioning. Report added compute separately and keep all shorter controls.
Relation: twophase-operator-preconditioning.


## 2026-10-27 — Reject longer refinement after residual/L2 divergence

Decision:3000-LBFGS controls lower training losses to.000519/.000410 but
common-local pooled L2 ends3.24048%/6.33613% for seeds42/2026. More optimizer
iterations are not a reliable accuracy improvement under finite temporal
collocation. Keep original1000-iteration refinement as the simpler recommendation;
no further model/preconditioner design. Because the2026 short-budget intermediate
was only logged, run one actual seed2026 cold start of train_vortex_conserved
(5000SOAP+1000LBFGS) to save/audit the recommended final model independently.
Final long-run history at step6000 is3.40366% pooled but is not a saved1000-
refinement checkpoint. Do not present the long final as short-budget validation
or select unsaved reference-minimum epochs. All negative long runs retained.
Relation: twophase-operator-preconditioning.


## 2026-10-27 — Remove unused initial-PDE weighting option before completion

Decision: the main comparison intentionally never enabled initial_pde_weight;
the tested endpoint-sampling flag provides the needed initial-time ablation
without extra weights. Remove the unused weighted-loss option and its docs/
validation to keep the solver interface simpler. All16 actual training configs
omit that field, so their physical residuals, predictions and checkpoints are
unchanged; frozen run sources remain preserved. Retain the numerical mass/time-
derivative/full-gradient test with the actual3-part loss, and retain endpoint
sampling's sharp-transient test. No additional training/model design.
Relation: twophase-operator-preconditioning.


## 2026-10-27 — Recommend simple conserved cubic original-residual vortex preset

Decision: all16 controlled cold starts and independent short-budget2026 final
checkpoint are complete. Recommend train_vortex_conserved.py via train.py:
convexified-chemical inverse cubic, original PDE residual, conservative Fourier
flux consistent with nodal L, exact grid mass and fixed integer periodic modes;
FP64/ordinary,SOAP5000+LBFGS1000. Common-local pooled phi2.84283%/2.90598%,
terminal2.34319%/2.55726%; band~12.8%, so1% is not achieved. Prefer the modest
controlled representation gain plus physical consistency; reject Schur/causal
metrics, larger AD-hybrid network, extra initial-time point and3000refinement
as default recommendations because measured tradeoffs/failures do not justify
added complexity. Long residual decreases accompany worse L2; no universal
conditioning/generalization claim. All16 frozen GPU predictions match exactly,
CPU replay passes explicit tolerances, initial arrays and12post-fix periodic
models pass.21distinct focused checks/17config identities verified; reference
unchanged, source caveats and all negatives archived. User-relaxed general/simple
objective addressed; further work would be a separately scoped investigation.
No Git operation, path migration, other-example default promotion or workers.
Relation: twophase-operator-preconditioning.


## 2026-10-27 — Match latest local AD and nonlocal vortex training

Decision: the user requests local accuracy and both training times with other
schemes held equal. Run a fresh seed42 serial pair of train_vortex_conserved
and an inherited local control changing only nl.method. Keep conservative nodal
spectral flux in BOTH arms, exact grid-mass/time differentiation, cubic map,
initialization, samples, FP64 ordinary and SOAP5000+L-BFGS1000 unchanged. Local
negative Laplacians are AD Hessian traces of the provisional point network:
the full-grid mass shift is spatially constant, so its spatial Hessian is zero.
Compare both against the same converged local reference, and native references
separately. Reject comparing historical local controls with newer nonlocal
schemes or calling spectral advection all-AD. Include JIT/evaluation in complete
timings and distinguish first/steady SOAP and L-BFGS costs; one pair does not
establish seed robustness. No default/reference/Git change.
Relation: twophase-matched-local-comparison.


## 2026-10-27 — Retain fresh nonlocal outcome in matched timing/accuracy pair

Decision: fresh recommended nonlocal seed42 gives common-local pooled3.77757%
/terminal3.26634%, trainer211.153s (external217.424s), versus historical2.84283%
/2.34319%. Direct checks show identical config except run name, identical
initial checkpoint bytes and identical first loss10.29859750798121; histories
already differ at step100. Saved source changes are only inactive endpoint/
removed-anchor branches and validation/metadata for this config. Ordinary GPU
compilation and adaptive training do not promise identical trajectories; this
comparison does not isolate a unique numerical cause. Report the fresh result
alongside the fresh local arm, disclose historical variation, and do not replace
it with the more favorable historical number or tune seeds/budget. Same-source
pair remains controlled. No default/reference/history change.
Relation: twophase-matched-local-comparison.


## 2026-10-27 — Complete same-scheme local AD versus nonlocal accuracy/timing pair

Decision: both fresh seed42 runs complete SOAP5000+L-BFGS1000 with identical
initial bytes,36saved sources and all config except method/name. Common-local
pooled/terminal phi: local3.36431%/3.22328%, nonlocal3.77757%/3.26634%.
Trainer wall924.230/211.153s, external931.018/217.424s, compiled SOAP83.1561/
16.7317ms; L-BFGS425.197/83.6103s. Local's pooled error is modestly lower and
terminal nearly equal; nonlocal is4.38x faster by trainer time and4.97x by SOAP
step. Historical same-config nonlocal2.84% means this one pair does not establish
stable accuracy ordering; retain all results rather than choose a favorable run.
Both stages stop at budget, not tolerance; no defaults/reference/budget tuning.
One new manufactured local-Hessian+mass+spectral-flux/gradient test passes;
CPU field replay<=9.99e-15 and GPU<=4.44e-16 under existing1e-10 tolerance,
GPU initialization exact and trained periodic modes/edges pass. Initial audit
mistakenly required exact CPU/GPU random initialization and bitwise ordinary
GPU mass reduction; preserve failed audit logs and correct to same-backend exact
initial reconstruction plus pre-existing FP64 prediction tolerance. No checkpoint
or saved field edit and no numerical gate was loosened from the established
replay contract. Comparison/config/curves/README/index/memory archived.
No Git history operation; no workers/blockers.
Relation: twophase-matched-local-comparison.


## 2026-10-27 — Diagnose same-seed numerical repeatability before larger cohorts

Decision: user explicitly prioritizes same-seed rerun diagnosis, then deciding
full comparison size. Freeze actual recommended initializer/source/time draws;
repeat objective/gradient/full SOAP updates and restart short trajectories.
Compare ordinary and deterministic policy to isolate numerical execution versus
random sampling, without changing physics/default or assuming determinism means
better L2. Inspect nonlocal gather pullback and actual upstream SOAP operations;
report measured differences and amplification without attributing all final L2
variation to one mechanism unless isolated. No long cohort or Git operation.
Relation: twophase-repeatability.


## 2026-10-27 — Use symmetry only as an isolated adjoint diagnostic

Decision: periodic L has L_ij=-w_ij for reciprocal pairs w_ij=w_ji and diagonal
sum_j w_ij, so L^T=L and the exact real-valued reverse pullback is L times
the cotangent. A probe-only custom VJP can replace the gather's scatter-add
transpose by the SAME gather/reduce forward action. Check dot-product identity
and gradient agreement before interpreting repetitions. This changes floating
execution but not the algebraic operator; it is not valid blindly for truncated/
row-normalized/nonsymmetric operators. Do not modify production implementation
or select a new method on the basis of a short repeatability probe. Compare
autodiff and symmetric pullbacks with frozen gradients/SOAP amplification.
Relation: twophase-repeatability.


## 2026-10-27 — Observe FP64 gradient noise amplified by SOAP initialization

Decision: actual fixed-input ordinary GPU probe gives identical losses while
gradients differ max6.94e-18..1.39e-17. SAME frozen gradient repeatedly fed to
SOAP yields exactly equal GG/Q/updates, ruling out random eigensolver variation
for these tested identical inputs. Repeated near-identical gradients instead
change initial GG by~1e-21 and Q byO(1); after feeding an IDENTICAL second
gradient to each state, actual second updates differ25.3–26.5% relative. Thus
Q differences are not only harmless sign/gauge differences. Original JIT
restarts share arrays/times, parameters first differ on update2 and reach32.1%
relative difference by100. Test exact symmetric adjoint/deterministic controls
and inspect covariance spectrum before assigning a general remedy. No long
run or new defaults; causal chain isolated locally, final L2 attribution limited.
Relation: twophase-repeatability.


## 2026-10-27 — Equivalent symmetric pullback removes tested within-process divergence

Decision: probe-only custom pullback L^T*bar=L*bar retains the EXACT forward
operator and loss, replacing scatter accumulation only. Under ordinary policy,
all5frozen gradient evaluations become bitwise equal; two restarted original
JIT100-update trajectories have exactly identical parameters/GG/Q/loss at every
checked milestone, whereas standard pullback diverges from update2. This is a
controlled intervention supporting gather-transpose accumulation as a trigger
for the observed short-path SOAP divergence. Preserve production untouched;
independent-process reproducibility, covariance spectrum, full-run L2 and
performance are separate claims not yet established. Check deterministic policy
on a bounded5-step diagnostic without invoking a long cohort.
Relation: twophase-repeatability.


## 2026-10-27 — Retain cross-process failure and distinguish diagnosis from a fix

Decision: identical-source v2 symmetric-adjoint probes in two independent GPU
processes retain identical initializer/time arrays/frozen loss but first frozen
gradients differ max6.94e-18, GG max2.54e-21, and100-step parameters differ29.7%
relative. Each process's own two restarts remain exactly equal. Thus replacing
nonlocal transpose scatter removes one within-process trigger, not full-process
repeatability. Exact cross-process compiler/kernel source is not isolated;
do not claim a finished reproducibility fix or convert to a production custom
VJP on this evidence. Spectrum independently shows output-weight64x64 Gram
rank2 (62near-null directions); frozen near-identical gradients change actual
second updates25.3–26.5%, while optimizer repeats with an identical frozen
gradient are exact. The measured key amplifier is SOAP's initial weak-direction
basis/coordinate normalization, not new random sampling. Prioritize validating
stable preconditioner initialization/weak-direction handling, then two same-seed
full runs before a small predeclared paired-seed accuracy cohort. No evidence
that this yields a lower final L2 yet; no full training/default/physics change.
Relation: twophase-repeatability.


## 2026-10-27 — Audit SOAP against primary implementations before blaming algorithm

Decision: user asks whether the installed SOAP implementation is incorrect.
Verify actual pinned dependency provenance and compare author PyTorch, JAX
pinned/current and independent implementation under matched arguments/gradient
streams, including initialization/step counting/QR refresh/moment coordinates.
Distinguish formula defects, intentional choices and finite-precision rank
deficiency. The prior sensitivity probe alone is insufficient evidence of
a SOAP implementation bug. No package/default replacement or long training.
Relation: soap-implementation-audit.


## 2026-10-27 — SOAP core parity passes; distinguish precision and inactive decay differences

Decision: actual author PyTorch, installed pinned SOAP_JAX0.2.2 and independent
pytorch_optimizer source classes complete16matched gradient-stream cases each
(1D/2D/3D, partial axes, bias toggle, three refresh boundaries). With shared
orthogonal basis tape in FP64, max parameter gap2.22e-16, physical first-moment
gap5.88e-15. This directly checks count/projection/refresh/moment transport; no
core translation defect is found in tested paths. Native differences align
with author FP32 eig/QR and independent FP32 QR versus JAX parameter precision.
Author's nativeFP32 one-ULP initial-gradient perturbations yield62–78% second-
update differences; independent nativeFP64 measured perturbations yield74–84%.
Thus rank-deficient initialization sensitivity is not unique to JAX. The
second gradient is synthetic/common in these single-block sensitivity controls,
not a new PINN L2 result. Real nonzero-weight-decay ordering difference exists:
author decays the updated parameter; JAX decays the old parameter. Current
factory uses default0decay, so it cannot explain current training variation.
Record differences rather than silently substitute libraries or claim a fix.
Relation: soap-implementation-audit.


## 2026-10-27 — Current SOAP already includes upstream stale-EMA fix; factory passes recurrence

Decision: pinned upstream README explicitly documents an older non-refresh-step
EMA staleness fix. Current source and48stream runs verify this is not present
in installed0.2.2; installed file also exactly matches prior diagnosis's
archived upstream source. Do not conflate known historical upstream defects
with current behavior. A separate1005-step NumPy-coordinate recurrence checks
the actual project factory's global clipping, continuous LR decay, first-step
skip, effective bias count and epsilon placement; max parameter gap1.79e-9
is within the declared FP32 LR-schedule rounding allowance2e-8. Author source
also reproduces FP64 matrix/default-FP32 Gram dtype error; reference runs
explicitly align Torch default dtype with parameter dtype, retaining author's
FP32 eig/QR casts. No installed dependency or runtime default is modified.
Relation: soap-implementation-audit.


## 2026-10-27 — Defer SOAP stability work outside the research mainline

Decision: user explicitly asks to record this issue in a standalone Markdown
file and revisit only when time permits. Create docs/issues/soap-repeatability.md
as low-priority/deferred, with symptoms, measured evidence, implementation-audit
limits, direct artifact/script links and a future resume checklist. Existing
diagnoses remain archived evidence; prior proposed follow-up protocols are not
active instructions. Do not continue initialization changes, dependency swaps
or larger training cohorts unless the user explicitly resumes this issue.
Preserve current strategy/reference/checkpoints and stop the side investigation.
Relation: soap-issue-record; supersedes priority of earlier SOAP follow-up plans.

## 2026-10-27 — Recommend elastic precipitates as the next new coupling

Decision: after the user closes the current two-phase pilot and requests further
phase-field/nonlocal examples, recommend elastic precipitate shape relaxation
(PFHub4) to join existing phase and elasticity capabilities. Thermal dendrites
(PFHub3), AT2 fracture and periodic spinodal decomposition are alternatives
with respectively anisotropic-operator, irreversibility and lower-implementation
cost tradeoffs. This is a recommendation, not authorization to implement a new
physical model. PFHub4 uses fictitious CH relaxation; its phase-dependent,
anisotropic misfit elasticity is not a drop-in constant ElasticD call. Preserve
the distinction between nonlocal differentiation and native integral physics;
no automatic advantage or novelty claim. Current prescribed-flow pilot remains
distinct from coupled CHNS, and SOAP investigation remains explicitly deferred.
Relation: phasefield-next-cases; extends phasefield-case-recommendations.

## 2026-10-27 — Publish the completed transport and optimizer-audit snapshot

Decision: user explicitly authorizes commit and push of current contents before
elastic-precipitate development. Save all current tracked/untracked project work
(including existing research archives) on dev, regular-merge dev into main and
push origin main only, then return to dev. Snapshot validation: 22 focused
two-phase unittest checks pass, artifact inventory passes 6 cases/345 directories,
and git diff --check passes. pytest is unavailable; use the existing unittest
suite without changing dependencies. No new numerical case implementation is
included in this pre-development snapshot. Relation: elastic-precipitate-pilot.

Snapshot packaging note: staging exposes upstream whitespace in the verbatim
SOAP source/license evidence. Preserve those exact source bytes and declare a
narrow .gitattributes whitespace exemption for that archive's sources directory;
project code and all other documentation keep normal whitespace validation.

## 2026-10-27 — Start a reduced periodic coherent-precipitate pilot

Decision: use a homogeneous-modulus, periodic 2D small-strain elastic/CH pilot,
with fixed zero macroscopic strain and zero-mean periodic displacement. Define
h=(1+phi)/2 and eigenstrain E*h; quartic chemical energy, isotropic phase-gradient
energy and elastic energy 1/2(eps(u)-E*h):C:(eps(u)-E*h). Differentiate this
single energy for chemical potential and equilibrium. Isotropic equal moduli
allow exact differentiable Fourier elimination of mechanical equilibrium;
PINN learns phase/mixed chemical potential rather than a separate displacement
network. A uniaxial E generates directional shape preference. Nonlocal pair
Laplacian enters phase gradient energy and CH diffusion, while mechanics remains
spectral: explicitly a hybrid, two-way coupled reduced formulation. Independent
NumPy/SciPy reference uses separately assembled Fourier Green coefficients and
space/time refinement. Reject an immediate full PFHub4 implementation because
variable anisotropic moduli/free boundaries would obscure first-route validation;
this changes model/BC/potential relative to PFHub4 and will be labeled as a
simplified pilot, not its reproduction. SOAP issue remains deferred.
Relation: elastic-precipitate-pilot; sources PFHub4 and Hu/Chen2001.

Pilot parameter preflight: the first independent reference used E_xx=0.35,
lambda=2,mu=1 as an algebraic energy-scale check. Preserve it at the separate
results_periodic_pilot path. Before training choose E_xx=0.01 and scale both
moduli by1225, retaining identical C*E^2 and phase trajectories while using1%
misfit. Regenerated results_periodic_smallstrain_pilot is the canonical
reference; it passes all-field time/grid controls. The old large-eigenstrain
reference is a preflight, not a trained result or material calibration.

## 2026-10-27 — Close the first elastic-precipitate route pilot with explicit limits

Decision: retain the validated periodic homogeneous-modulus default as a pilot,
not as an exact PFHub4 or full mechanical-network result. SOAP5000+LBFGS500
native pooled phi0.0387%, solved-u0.0185%, stress-component0.0354%, q3.807%;
common refined local phase0.5420%. q at t0 remains16.42% inaccurate despite
exact phase IC, so no all-field subpercent claim. Independent time/grid,
energy derivative, acoustic equilibrium and CPU saved-checkpoint checks pass;
reference/predicted terminal shape ratios0.95860/0.95876 versus uncoupled1.0.
L-BFGS stops at its iteration budget. One seed and modest deformation do not
establish robustness or a nonlocal advantage. Do not expand scope into harder
coupling or SOAP fixes automatically. New development remains uncommitted on
dev after the explicitly requested pre-development main publication. The initial
full-run CLI return status was corrected and separately validated by an exit0
entry smoke without rerunning unchanged numerical training.
Relation: elastic-precipitate-pilot.

## 2026-10-27 — Compare local spectral and local AD precipitate diffusion explicitly

Decision: user asks for local results. Existing local_spectral is a local-PDE
Fourier discretization, not a point-network AD baseline. Run it unchanged as
one control and add an explicitly named local_ad mixed-Hessian control so the
result distinguishes these algorithms. Keep prescribed phase IC/mass/network,
exact spectral mechanical elimination, FP64/ordinary, initializer/time samples
and SOAP5000+LBFGS500 fixed; only phase spatial differentiation and run name
change. Both local paths use the already-refined local reference. The projected
phase has a uniform spatial-query shift, so its point-network Hessian is valid;
time derivative still differentiates the full-grid projection. No reference,
default, SOAP issue or Git history changes. Relation: precipitate-local-comparison.

## 2026-10-27 — Report precipitate local controls on a common reference

Decision: complete local spectral and AD controls with identical initializer,
config except method/name and SOAP5000+LBFGS500. Common-local pooled phi
spectral0.0599%/AD0.0677%/nonlocal0.5420%; native nonlocal0.0387% remains valid
against its different target. Reference native/local gap0.5378% explains the
principal common-target error floor; do not compare unlike targets as though
all were continuum errors. Trainer times183/680/183s show AD cost, not a
nonlocal speed advantage over FFT. Source/initial byte parity, manufactured
mixed residual/parameter gradient, CPU checkpoint and independent saved-field
checks pass. Initial q~30% and one-seed/historical timing scope remain explicit.
Retain existing default and defer horizon/optimizer/stability investigations;
user requested results, not a promotion or broader optimization campaign.
Relation: precipitate-local-comparison.

## 2026-10-27 — Explain cubic coordinates separately from conservative constraints

Decision: ground the requested explanation in current source. z=phi+phi^3 is
Fc' for Fc=phi^4/4+phi^2/2, giving a monotone unbounded coordinate and finite
sensitivity at pure phases; this is a chosen convexification, not a guarantee
of easier neural optimization or a new inverse-J residual loss. State the
conditional weakly-convex-potential extension and actual overshooting reference
range. Explain spectral conservative flux, reciprocal pair diffusion and exact
grid-mean projection as three distinct mechanisms. Finite PINN residuals can
still cause mass drift without projection; enforced mass is not learned
conservation, zero-contour area or energy conservation. Projection globally
couples the scalar map, so its full Jacobian is P*diag(g'), not g' alone.
Historical map ablation improvement is modest and same-seed variability prevents
claiming stable general benefit. No new training or production strategy change.
Relation: twophase-map-conservation-explanation.

## 2026-10-27 — Remove the redundant soft mass term under hard conservation

Decision: user points out that exact mass projection makes an additional mass
loss unnecessary. Current two-phase factory still evaluates a third mean-square
penalty, although projection reduces it to roundoff. With explicit phase_mass,
construct only the two PDE losses/weights; do not read mass_weight or compute
its penalty. Remove the unused weight from the canonical hard preset and make
validation require it only for the unprojected legacy three-term path. Preserve
old serialized hard configs with the now-ignored extra weight and unchanged
network leaves/artifacts. Precipitate already uses two PDE terms. Prefer actual
removal over retaining a computed third term with weight0; no full retraining
or claims of improved L2/reproducibility. Relation: twophase-remove-redundant-mass-loss.

## 2026-10-27 — Test cubic-map removal with a fresh matched pair

Decision: user asks whether removing the cubic coordinate greatly worsens
accuracy. Run the existing train_vortex_consistent additive preset and current
train_vortex_conserved cubic preset serially with seed42 and the same nonlocal
operator, exact IC/mass, conservative flux, FP64/ordinary and SOAP5000+LBFGS1000.
Both use the newly cleaned two-term PDE loss; do not mix that change into a
comparison against only an old run. Same underlying initial network parameters
and IC do not imply identical provisional future fields under different maps;
that is part of the parameterization change. Keep all results and final-budget
metrics, no tuning or default promotion, no SOAP reproducibility investigation.
One pair can quantify this run, not establish statistical superiority.
Relation: twophase-map-ablation.

## 2026-10-27 — Treat the cubic coordinate as optional after the fresh ablation

Decision: report both full-budget seed42 cold starts and independent CPU replays.
Direct/cubic common-local pooled phi3.2006%/3.2522% is effectively comparable;
terminal3.6215%/2.9159% and terminal band14.8632%/11.6362% favor cubic, while
pooled band13.7798%/14.2428% favors direct. Trainer212.38/217.83s. Removing
the map does not cause a large pooled-error regression in this pair, but late
accuracy changes measurably. Do not describe the map as necessary, uniformly
better, or a proven optimization preconditioner. One matched seed with known
trajectory variation is not a stability conclusion. Keep both explicit presets
and existing recommendation unchanged; reject favorable old-checkpoint-only
comparisons, further tuning or an unsolicited SOAP investigation. Pooled q
errors60.0%/58.1% remain explicit. Both exact-mass runs use only two PDE losses.
Relation: twophase-map-ablation.

## 2026-10-27 — Default two-phase output to additive with explicit cubic opt-in

Decision: user prioritizes cross-example generality and requests default-off.
Declare phase.map=additive in train_base, let the recommended conserved vortex
inherit it, and allow the existing --config.phase.map=inverse_cubic override.
Extend validation to accept explicit additive; retain missing-phase legacy
configs unchanged. Prefer one existing string switch over another boolean,
new mapping implementation or duplicate preset. Hard IC/mass, conservative flux,
original residual and optimizer budgets are independent and unchanged. Freeze
historical causal/initial-time/long-refinement cubic controls explicitly so the
default switch does not silently alter their experiment definitions. Preserve
recorded run config/model paths and identify old cubic results as historical;
use the fresh additive run as current default evidence. No retraining, SOAP
work, changes to other examples, or Git history operation.
Relation: twophase-optional-map; twophase-map-ablation.

Validation detail: the prior hard-mass gradient test perturbed the phase output
bias. With additive output this is a spatial constant removed exactly by the
mass projection, leaving a zero derivative whose relative finite-difference
check is dominated by cancellation (4.44e-10 finite difference versus2.78e-17
AD). Probe an output weight with spatial variation instead; keep the existing
3e-6 relative tolerance and physics unchanged. Normalize tuples/lists through
JSON when comparing a live config with a saved config. These are verification
corrections, not a numerical fallback or altered constraint.

## 2026-10-27 — Clarify that the current two-phase pilot has no solved NS system

Decision: answer the user's NS/first-derivative question from current source.
The pilot solves only mixed advective CH for phi/q with a prescribed analytic
divergence-free velocity. Recommended spectral_flux computes the divergence of
nodal velocity-phase flux by FFT; legacy advection uses point-network AD.
Time differentiation is a JVP of the full physical projected fields; only the
negative Laplacians use the finite-horizon nonlocal operator in nonlocal mode.
Local-AD controls retain the selected advection scheme. Correct the stale
general statement that all first derivatives use AD; do not add NS, redesign
operators or relabel phase error as full CHNS accuracy. Fourier differentiation
is a global evaluation of a local derivative, distinct from finite-horizon
nonlocal interaction physics. No numerical/default/training/Git change.
Relation: twophase-ns-derivative-scope.

## 2026-10-27 — Assess all-nonlocal spatial differentiation with paired operators

Decision: all spatial derivatives are feasible through finite-horizon gradient,
divergence and Laplacian operators; retain time JVP and parameter AD. Current
prescribed-flow CH only needs a new flux divergence, while future NS needs
pressure/velocity/viscous/capillary compatibility. Consult Du/Tian arXiv1805.08261
Sections2–3 (periodic nonlocal Stokes and conditional kernel stability), plus
Du et al.2013 vector calculus. Do not infer that all kernels or arbitrary
discrete substitutions give a stable coupled CHNS solver.
An independent NumPy nodal-periodic candidate uses moment-corrected G and
D=-G^T. Constants, summed divergence, adjointness and phase-advection/capillary
work cancellation pass at~1e-14, but an alternating checkerboard has G nearly0
while the pair Laplacian acts strongly. Thus adjointness alone does not ensure
pressure stability, and -DG is not the existing pair Laplacian. Reject blind
Laplacian factorization or an accuracy/speed promise. This is a feasibility
discussion with a diagnostic, not implemented first derivatives or NS, a
default switch or training request. Production/reference/artifacts unchanged.
Relation: fully-nonlocal-flow-assessment.

## 2026-10-27 — Test nonlocal first derivatives on prescribed-flow CH first

Decision: user requests an actual effect test. Add optional nonlocal_flux with
periodic moment-corrected nodal gradient G and negative-adjoint divergence D,
using the same cubic radial horizon as the pair Laplacian. Retain that positive
pair Laplacian rather than substituting -DG: known gradient null modes remain
explicit, and no pressure/NS stability claim is made. Start with direct output,
same hard IC/mass and original two PDE losses, FP64/ordinary, SOAP5000+LBFGS1000.
Run fresh spectral/nonlocal-flux controls with seed42 and same initial network.
Create a separate native reference with independent Fourier symbols and dt
refinement; use the SAME already-refined local PDE reference for cross-method
errors. New-mode identity explicitly includes advection, while old identities
remain unchanged. Old data/checkpoint paths and defaults stay intact. No
unrequested horizon/kernel tuning, stabilization, full NS or SOAP work.
Relation: twophase-nonlocal-flux; fully-nonlocal-flow-assessment.

Pre-training validation: ten focused numerical/legacy transport checks pass.
New native dt2e-4/1e-4 reference differences are1.3384e-5 max phase and1.4734e-4
max chemical; unchanged common-local fields are reused explicitly from their
validated direct path, rather than regenerated with a different target. Native
phase differs9.219% pooled/3.858% terminal from common local, versus the old
pair-only diffusion reference gap~1.28%/1.17%. Keep this measured model/
discretization bias separate from PINN optimization; it is not a strict error
floor. Both new cold-start raw initial checkpoints and37source snapshots
match. After their snapshots were captured, add validation-only rejection of
the unsupported nonlocal-flux/local-diffusion combination; no executed run
branch or numerical formulation changes. Its focused schema/common-target
check passes. Retain the same horizon/kernel; no post-result tuning.

## 2026-10-27 — Retain spectral flux after the complete first-derivative control

Decision: both fresh seed42 runs finish SOAP5000+LBFGS1000 and pass independent
CPU field replays. Common-local pooled/terminal phi FFT2.9236%/3.1270% versus
all-spatial-nonlocal8.5448%/4.5147%; native pooled3.2435%/2.7625% uses DIFFERENT
targets and must not be labeled a shared accuracy gain. Trainer213.41/217.63s
shows no speed benefit. Native nonlocal first derivatives already change the
reference by9.219% pooled and cause large overshoots[-1.546,1.586], reproduced
in kind by the trained phase[-1.415,1.378]. q common-local errors55.3%/147.6%
remain explicit. This is evidence of substantial transport discretization/model
bias at64x64/delta0.04 with this centered cubic-moment operator; it does not
prove all-nonlocal methods are worse or uniquely attribute the result to a
checkerboard mode. Keep optional research preset and the original FFT default.
Reject hidden stabilization, more training or horizon/kernel tuning within this
matched control. No NS/pressure or robustness claim, no SOAP/Git work.
Relation: twophase-nonlocal-flux.

## 2026-10-27 — Defer nonperiodic boundary work and retain FFT first derivatives

Decision: user explicitly asks to record the FFT periodic-boundary limitation,
defer deeper work and keep the current FFT first derivative. Record a standalone
issue linked from README/PROGRESS. The recommended conserved vortex keeps
spectral_flux, independent nonlocal pair Laplacians, time JVP and direct phase
output. Current Fourier differentiation and nonlocal neighborhoods both assume
periodic geometry; network BC constraints alone do not generalize those
operators. Do not launch a boundary redesign, resume nonlocal first-derivative
optimization or add NS without an explicit new request. Existing configs,
references and artifacts stay intact. This is a documentation-only deferral,
not a default/physics change or a blocked numerical task.
Relation: twophase-fft-boundary-issue; twophase-nonlocal-flux.

## 2026-10-27 — Extend coherent precipitation through unequal particle interactions

Decision: user asks for a less trivial precipitate case. Retain the declared
quartic CH energy, homogeneous isotropic elastic equilibration, periodic
zero-macrostrain boundary and1% anisotropic eigenstrain. Add an explicit
multiple-particle IC as a smooth union of periodic tanh particles; only the
initial field is constrained. Preflight unequal particle sizes/non-symmetric
positions over a longer horizon and stronger elastic/interface competition
before choosing the final case. This supplies morphology/coarsening/elastic
interaction without inventing a heterogeneous constitutive law or claiming
PFHub4 reproduction. Keep old schema/arithmetic/checkpoint leaves and the
simple default, use separate config/reference/log paths. Refined references
and zero-eigenstrain control must show actual complexity before PINN accuracy
claims. Exact mechanical elimination remains explicit, not a learned-u claim.
Sources: existing energy derivation and PFHub4 overview (multiple-precipitate
shape/coarsening motivation); its detailed BC/potential/material choices are
different and are not silently adopted. No Git operation or deferred work.
Relation: precipitate-multiparticle.

Reference-only preflight selects three initial centers(.35,.32),(.67,.36),
(.40,.68), levels0.18/0.11/0.065, shared width0.035/background-0.95, T4 and
fourfold elastic moduli(lambda9800,mu4900), keeping eigenstrain1%. At native63
the smallest component disappears and the largest elongates; uncoupled control
instead merges the lower pair. No future shape is prescribed. Freeze this
physical case before training. A t0 spatial precheck finds q difference0.2948%
between127/255, above the existing0.2% criterion; increase the coarse local
reference validation grid to191, keep fine255 and unchanged tolerance/IC.
Retain paired nonlocal and local_spectral training controls with the same
current strategy/budget; no local AD run is needed to establish this extension.

User steering supersedes that comparison choice: use local AD as the PRIMARY
control, with the same multi-particle model and SOAP5000+LBFGS500 budget.
The local spectral run completed just before the instruction; retain its
artifacts as exploratory history, exclude it from the main comparison and do
not rerun it. Launch local AD through train_multiple.py with explicit
--config.nl.method=local_ad, keeping the existing source snapshots unchanged.
New analysis must measure mass relative to the directly sampled hard-IC mean;
the common fine-grid local field transfer differs2.88078e-6 in mean and0.2091%
initial phase L2. This is reported discretization mismatch, not conservation
failure and not grounds to relax tolerances or rewrite the reference.

## 2026-10-27 — Keep harder precipitate results as an opt-in baseline

Decision: complete the user-requested nonlocal/local AD comparison with the
frozen three-particle,T4,fourfold-modulus case. Same initialized bytes,21 source
files and SOAP5000+LBFGS500 produce common-local final pooled phi8.0304%/
12.8981%, terminal9.1820%/19.2435%, trainer183.34/674.27s. Both independent CPU
replays and physical audits pass. Native nonlocal pooled phi7.3559%, so the
1.2353% native/local target gap alone cannot explain remaining training error.
Do not promote this untuned harder case as the default or claim stable operator
accuracy ordering from one seed. Retain original simple pilot and expose the
new config through train.py. Best validation checkpoints occur before LBFGS
completion; record best and final-budget results separately, using the latter
for primary matched fields. Reject extra optimization, seed tuning or SOAP
investigation in this case-extension task. Late small-particle disappearance
and high initial q remain limitations. Solved mechanics remains spectral for
both phase differentiation schemes. Completed local spectral exploration is
preserved but excluded from the requested primary comparison. No Git operation.
Relation: precipitate-multiparticle.

## 2026-10-27 — Recommend confined CH for the Neumann geometry study

Decision: assess, rather than implement, a new phase-field case. Recommend
constant-mobility quartic CH phase separation/coarsening in a smooth elliptical
container with two unequal impermeable circular holes, neutral surface energy
and no mass flux on every wall. Compare nonlocal pair diffusion with local AD,
using independent mixed-FEM local reference plus a separately audited native
nonlocal solve. Avoid mechanics/NS/anisotropic wetting coupling in the first
boundary-focused example; retain existing examples and deferred FFT issue.
Natural graph/regional closure eliminates external exchange and a separate
boundary-normal loss for the declared nonlocal problem; it is NOT exact
pointwise classical Neumann enforcement at finite horizon. Symmetric pair
weights, physical quadrature volumes, geometric visibility and local-limit
boundary convergence require validation before an advantage claim. FEM already
incorporates homogeneous Neumann weakly, so do not claim uniqueness to nonlocal.
Published complex-domain CH examples and nonlocal Neumann/local-limit theory
support feasibility, not guaranteed PINN accuracy or arbitrary-kernel convergence.
Relation: phasefield-neumann-candidates.
