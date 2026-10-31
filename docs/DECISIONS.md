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

## 2026-10-27 — Begin confined CH pilot after preserving prior results

Decision: user authorizes trying the proposed ellipse+twohole case and first
committing current work. Commit728e5b9 on dev preserves prior results; no
merge/push. New independent example confined_ch2d uses physical triangle-centroid
areas, kernel120/(pi delta^4)*(1-r/delta)^2 and symmetric line visibility to
exclude paths crossing impermeable holes. Integral second moment is2I in2D
(radial integral delta^4/60), fixing interior negative-Laplacian normalization;
no boundary row normalization is introduced. Local-limit boundary consistency
is measured, not assumed. Independent P2 mixed FEM uses standard theta=.5 CH
weak forms per DOLFIN demo; native SciPy BDF graph reference uses its own
assembly. Smooth deterministic IC vanishes to order4 at every analytic wall,
so phi and chemical-potential normal derivatives are compatible initially.
Both PINNs share hard initial phi and physical weighted mass, direct output,
original mixed residuals, FP64 ordinary/SOAP; local AD alone has both boundary
normal losses. Geometry/physical scales remain preflight choices until frozen.
Reject bounding-box area/N quadrature, cross-hole bonds, periodic FFT reuse,
new mechanics/NS coupling and predeclared accuracy superiority.
Relation: confined-ch-neumann; phasefield-neumann-candidates.

Preflight confirms the first confined morphology at T1 with epsilon.05,M.02,
beta1,IC amplitude.3/flattening width.04. Freeze these physical choices before
training. Initial phi and local mu normal compatibility follows fourth-order
wall vanishing, not any future shape constraint. Gmsh is isolated into its
own subprocess because its shared MPI/PETSc stack conflicts with DOLFIN when
loaded together; do not mix environments or suppress actual solver errors.

Reference gate fails for CN/P2: phase max space/time0.581%/1.214%, mu56.4%/
69.8%; native BDF refinement passes1.64e-7 max mu. Do not train or loosen
thresholds. Stiff startup and mass-inverted interpolant derivatives are plausible
sources, not proven unique causes. Replace CN by BE-started BDF2 on the same
mixed weak form, and compute continuum t0 mu from an independently derived
analytic product-rule Laplacian (checked against JAX). Initial phase/model/
geometry/horizon and acceptance thresholds remain fixed. Save failed reference
as reference_cn_failed.npz/json and all subsequent FEM stage fields for diagnosis.

BDF2 reference passes fixed gates: max phi space/time.00581/.0003769,mu.002051/
.001449; native mu time1.64e-7, mass drift<=5.38e-16, saved energies decrease.
Native/local phi gap4.20767% and mu32.3404% are measured pretraining baseline
bias, not a guaranteed error floor or a PINN benefit. Proceed with the frozen
physical case and declared horizon without fitting it to the FEM target.

## 2026-10-27 — Retain confined pilot with a precise Neumann claim

Decision: both seed42 cold starts complete SOAP5000+LBFGS500 and CPU replay.
Common-local final pooled phi NL4.1935%/AD5.6639%, terminal4.9998%/3.8354%;
trainer146.47/529.03s. Native NL pooled/terminal.8898%/.6469%, with pretraining
native/local gap4.2077%. Keep the first pilot; do not claim a stable precision
ordering or attribute faster convergence solely to Neumann treatment. Same
initial bytes/18sources/config except method/name; original defaults untouched.

Important limitation: local AD wall-normal phi RMS at t1 is.0158-.0255; native
NL MLP RMS1.66-8.79 is appreciably nonzero. This validates a closed physical-
volume pair graph with no extra normal losses, not automatic pointwise local
Neumann. Native horizon effects vs unconstrained MLP boundary extension are
not isolated; weak FEM already handles natural Neumann as well. Further joint
horizon/mesh/boundary validation precedes stronger claims. Shared hard mass
also prevents attributing neural conservation solely to graph closure.

Initial mu common error83.0%/91.1% and initial chemical residual79.3%/84.6% of
fixed final loss are the major current fitting weakness. Positive-time mu errors
6.71%/10.11% remain. No unrequested new tuning, seed selection or SOAP work.
All5 numerical checks, independent refined references, exact initialized/source
pair audit, CPU final replay and physical diagnostics pass. No new-result Git
commit: only the explicitly requested prior snapshot728e5b9 was committed.
Relation: confined-ch-neumann.

## 2026-10-27 — Enrich confined CH morphology before more optimization

Decision: user finds the weak initial phase and late diffuse interfaces too
simple. Add a separate explicit multimode coarsening preset, retain base pilot
and artifacts. First use deterministic six-wave tanh-saturated physical initial
phase, existing fourth-order wall compatibility, smaller epsilon and longer
horizon; keep geometry/model/flux unchanged. Validate reference morphology,
phase purity, topology and resolution before training. Initial tanh is an IC
shape definition, not a learned-output map or future-motion prior. Analytic
chain/product-rule Laplacian must agree with independent AD, and static IC
metadata must preserve old checkpoint trainable leaves. No Git operation,
SOAP work or pointwise automatic-Neumann claim is authorized by this critique.
Relation: confined-ch-coarsening; confined-ch-neumann.

The six-wave T3 preflight sharpens phases (64.6% pure) but retains about5 large
bands, so do not promote it as sufficiently complex. Analytic initial-only
screen yields12 significant components for both10/12 balanced waves; choose10
with fewer explicit terms. User then limits the horizon to1.5s: honor T1.5,
use dt.005/.0025 and21 snapshots, no T3 training. Preserve the6-wave candidate
path/config and write the selected10-wave case under coarsening_multimode10.

Selected10-wave T1.5 preflight shows27.1% ->46.3% pure-phase area, nontrivial
early restructuring and10 ->8 significant coarse-FEM sign regions. Exact
analytic initial data has12 such regions on the same query grid; the count is
threshold/resolution sensitive. Freeze the physical IC/epsilon/horizon now,
before inspecting PINN errors. Keep original preset as the baseline and new
case opt-in. Coarse initial interpolation error1.131% motivates independent
mesh/time validation; never infer reference accuracy from an appealing plot.
All selected multimode derivative/compatibility/gradient tests pass and old
checkpoint fields agree within2.45e-15. Reference refinement is in progress;
no new trained accuracy or automatic classical-Neumann claim is made.

The first .02/.014 reference phase mesh gate necessarily fails at t0: their
same-IC P2 interpolants differ1.1257%, above1%. Refine to .014/.01 without
changing physical choices or tolerances. Existing .014 stages are reused only
by explicit archived reconstruction with physical identity/mesh/request checks;
new .01 stages use the same independent FEM routine and both original dt values.
Separate single-process FEM stages may run concurrently on CPUs; no JAX GPU or
MPI run is active. Preserve the failed first reference before replacing it.

Finer .014/.01 reference passes the original gates: max phase mesh/time.55193%/
.04796%, mu3.01351%/.17959%; local/native mass drift<=1.05e-15 and saved energies
decrease. Freeze and retain the richer case as an opt-in physical benchmark:
terminal pure fraction46.14% and12 ->9 significant sign regions, with threshold/
mesh caveats. Stop this case-extension task after validated reference delivery;
no new PINN pair or accuracy improvement is claimed. Full-grid IC quadrature
change5.66% atdelta.03 and native/local phase gap3.779% remain separate
finite-horizon/discretization diagnostics. Original default/results retained;
final reference, failed first pass, sources and plots are explicitly indexed.
All workers idle, no Git operation. Relation: confined-ch-coarsening.

## 2026-10-27 — Train the validated richer confined CH case

Decision: user asks to continue PINN training. Run the frozen train_coarsening
preset through canonical train.py: paired seed42 nonlocal/local AD cold starts,
FP64 ordinary, SOAP5000+L-BFGS500, same initialization/temporal rule/budget and
refined FEM common target. Keep native errors separate, report best/final and
actual runtime with phase/mu/physical audits. Reject spectral comparison, new
physical tuning, transformed loss or output map, seed shopping and SOAP side
work in this task. Original pilot/default/artifacts remain; new unique run
paths. No Git authorization or FEM/GPU concurrency. Relation:
confined-ch-coarsening-training; confined-ch-coarsening.

Both frozen cold starts finish5500 updates but fail phase learning: common
pooled NL82.96%/AD82.95%, terminal94.55%/99.41%, trainer174.65/696.63s; both
best-validation checkpoints are untrained step0. Retain and index this failure,
do not promote defaults or tune the physical case after seeing errors. Own-
native NL83.02% vs native/local target gap3.779% excludes bias alone as a
sufficient explanation. Strict final/best CPU replays,19sources/initial/config
parity pass. Corrected audit-only missing x64 initialization without changing
models or loosening tolerances; failed audit log retained.

Initial chemical loss contributes52.66%/39.46% of the fixed17-time objective,
but positive mu errors119.1%/123.3% and wrong purity/energy evolution show a
broader fit failure, not an isolated t0 defect. Fields change substantially
from IC; do not call them uniformly collapsed/frozen. Capacity, hard lifting,
sampling and optimization need separate future controls, not unsupported
causal claims. End this requested training/diagnosis experiment with accurate
failed results and unresolved accuracy explicit. No extra tuning, SOAP work,
physics/reference change or Git action. Relation: confined-ch-coarsening-training.

## 2026-10-27 — Diagnose and optimize failed richer CH training

Decision: user explicitly requests diagnosis and improvement. Freeze the physical
case/reference and retain failed controls; directly measure PDE gradient balance
and representation fit before attributing failure to one cause. Test general
IC lifting, declared row metrics and temporal strategies with separate ablations,
then full-budget cold-start/field/replay validation. Supervised reference fitting,
if used, is a representation diagnostic only and must never be reported as a
PINN improvement. Exponential IC blending, if tried, fixes only t0 and leaves
arbitrary positive-time fields; no future shape/trajectory is imposed. Reject
changing IC/epsilon/T to ease fitting, seed shopping, reference training labels
or reviving SOAP implementation work. No Git operation authorized. Relation:
confined-ch-optimization; confined-ch-coarsening-training.

Direct probe finds initial phase RMS/PDE-speed timescale.006755 vs first
uniform stratum width.1875. At the failed final model PDE gradient cosine is
-.826 (positive-only-.737); gradient norms are comparable, not an unmeasured
orders-of-magnitude phase dominance. Additive lift requires raw-phase RMS6.22
at first snapshot.075 vs.797 for exponential tau.15. However blend-only pilot
2000+100 worsens common pooled phase91.53%: reducing lift amplification alone
does not solve causal learning. Proceed to explicit early-time sampling and
expanding-horizon curriculum, record its nonuniform residual measure. No physical
timescale or future solution is imposed by these optimizer coordinates.

Supervised diagnostic demonstrates same64x4 architecture can fit phi4.24%
(additive) or2.62%(blend), not a PINN result/capacity impossibility proof.
PDE log-time/horizon pilot improves common phase to52.98% vs82.96%; log-time
coordinate and frozen biharmonic metric alone remain around50%. Add an exact
native chemical elimination control: mu=beta(phi^3-phi)+epsilon^2 A phi,
train only phi_t+M A mu. This is exactly the same graph CH ODE used by the
independent native solver, not a future-reference prior. Delete chemical loss
and unused mu head. Keep a frozen resolvent P=I+tau*M*epsilon^2 A^dagger A in
physical-volume L2; P is SPD, so P^-1 r=0 iff r=0. Its weighted Cholesky solve
is independently checked; it changes finite residual metric, not PDE roots.
Native eliminated mu is only defined on quadrature nodes; do not fabricate
pointwise boundary-normal diagnostics for it. Local AD elimination is not
implemented; fail explicitly instead of silently changing its model.

Elimination/log-blend/modes8 pilot reaches common phi44.79%/terminal55.51%,
an improvement but inadequate. Test actual sequential time windows: each
window fixes the previous PINN endpoint (never reference data), learns a small
phase increment, derives chemical potential exactly and solves the same ODE.
For local coordinate s in[0,1], phi=phi_start+s*N_theta and residual
phi_s+Delta_t*M*A*mu. This is exactly Delta_t times the physical residual.
Fixed weighted mass and state continuity are hard; derivative/energy accuracy
is checked, not assumed. Use30 log-spaced windows to resolve early stiffness,
5000 total SOAP (650 first,150 each remaining) and500 total L-BFGS
(36 first,16 remaining), warm-start weights but reset optimizer explicitly.
Precompute per-window SPD resolvents; keep dynamic state/factors fixed-shape
JAX inputs to avoid window-by-window compilation. Save an explicit checkpoint
trajectory manifest; no single final-window model is the whole trajectory.
Nonlocal-only initial implementation; no local AD/pointwise boundary-extension
claim. This is a general causal method-of-lines PINN, not supervised rollout.

First full window run completes exactly5000 SOAP+500 L-BFGS: common pooled
phase3.90795%, terminal5.75378%, vs prior82.96%/94.55%. Freeze this configuration
and now independently validate fixed seed2026, with every outcome retained;
no seed selection or hyperparameter change between runs. Replay the complete
state/checkpoint chain and inspect native errors, energy and temporal joins
before promotion. Reference/physical case unchanged. Relation:
confined-ch-optimization.

Independent2026 cold start confirms phase improvement (common pooled3.86634%,
terminal5.67185%). First42 replay/links/mass/energy pass; native phi.724% but
native mu31.0% (terminal102.1%) remains weak. Do not hide derivative-sensitive
chemical error behind phase accuracy. Test one same-budget15-window control
with more updates per window (first800+14x300=5000 SOAP,52+14x32=500 L-BFGS,
maxfun104+14x64=1000), same physics/spatial modes/operator/seed42. This is a
budget-allocation control, not more total training or reference-label fitting.
Keep the validated30-window two-seed result regardless of the control outcome.

### 2026-10-27 — Retain30 windows for richer confined CH; phase success, mu open

Both frozen30-window cold starts pass complete CPU trajectory replay and
source/config parity (31 files; only seed/name differ). Common-FEM phase pooled
3.908%/3.866%, terminal5.754%/5.672%, trainer281.25/288.23s; native phase pooled
0.724%/0.678%. Same5000+500-budget15-window control is slightly worse on phase
(4.008%/5.936%) and native mu (31.35% pooled), so reject fewer windows as the
recommended allocation. No further tuning or reference changes are warranted
for this task. Recommend explicit train_coarsening_windows through train.py;
preserve original simple-case default and every failed global control.

Do not claim all-field convergence: native mu pooled31.01%/29.99%, positive-only
52.98%/51.23%, terminal102.11%/97.87%. Terminal error RMS is comparable to mu
reference RMS, not merely a tiny-denominator artifact. Chemical algebra/mass and
sampled energy pass, but derivative jumps and fixed-grid wall limitations remain.
The bundle includes larger spatial modes,30 stored networks, optimizer resets,
exact elimination and residual metric; equal update counts do not isolate a
single cause or match total parameter capacity. No matched local AD window
advantage claim. Archive full results and plots, separate supervised probes,
register paths and complete diagnosis task. No Git operation. Relation:
confined-ch-optimization.


## 2026-10-27 — User requires one global-time confined CH network

Decision: replace the recommended direction with a single network covering
all[0,1.5]; do not propagate/freeze intermediate states or create slab networks.
Retain historical window artifacts only as comparison. Physics/reference fixed;
chemical elimination and independently derived residual/time conditioning may
be tested without labels. All times enter one objective; any time discretization
or changed residual measure must be explicit and independently validated.
No Git action or SOAP side investigation. Relation: confined-ch-global.

## 2026-10-28 — Test one global space-time residual metric

The resumed baseline full5000+500 run completed after the tool interruption.
Next test an independently checked space-time left residual metric. Given
continuous AD residual r_i at sorted positive times, define e_0=0 and
(I+h_i K)e_i=e_{i-1}+h_i r_i, K=M epsilon^2 A^2. This invertible block lower
triangular map leaves collocation residual zeros unchanged. It approximates a
stable linear evolution inverse; it does not integrate predicted phase or
propagate/freeze network states. All residuals come from one global network,
all parameter gradients flow through the complete metric. Include terminal T
as a residual query and use sum h_i||e_i||_V^2/T. No PDE time derivative is
discretized: phi_t remains AD; backward Euler only defines the loss metric.
Precompute the weighted-self-adjoint A eigenbasis, explicitly check symmetry,
and independently verify weighted dense block solves and gradients first.
This is an optimization hypothesis, not a proven improvement or novelty claim.

Full space-time differential residual metric fails (common phase97.97%, native
97.98%); late fields nearly vanish. Reject as a recommended strategy. Hypothesis:
finite residual quadrature can miss the transient carrying initial-state
information. Test the mathematically integrated CH equation instead:
R(t)=phi(t)-phi0+M A integral_0^t mu(s)ds. Continuous R=0 for all t is equivalent
to the original IVP, and explicitly retains phi0 at every query. Approximate
only the mu integral with one global log-Chebyshev polynomial quadrature matrix,
independently checked for moments and manufactured evolution convergence.
Every temporal value is produced by the same network in one objective; no
field stepping, solver-generated training target, or fixed intermediate state.
Use (I+t M epsilon^2 A^2)^-1 as a declared residual metric. Unlike differential
collocation, finite quadrature introduces a time approximation; check refinement
and continuous differential residual separately before any accuracy claim.

Global integration quadrature validates against an independent native IVP:
32/64/128-node preconditioned defects max1.46e-4/2.65e-6/3.71e-9. Training
still struggles early. Test an IC-preserving increment coordinate
phi=phi0+(1-exp(-t/tau))*N(x,logtime), with known scalar IC as one optional
network input. Zero increment retains IC; no future dynamics is encoded.
Legacy configs retain old features/leaves. Also test Neumann inverse residual
A^-1 r/M=A^-1 phi_t/M+(mu-mean(mu)) on the hard-mean subspace. Lift only the
constant nullspace with its physical-volume projector; independently check
dense solves/gradients. Include t0 evolution residual with exact chemical law.
One global network, all positive-time strata over full interval, no curriculum.

Test a general separable global-time network: one spatial MLP outputs a_k(x),
phi=phi0+s sum_k a_k(x) T_k(2s-1), s=log1p(t/tau)/log1p(T/tau). Each time
basis has global support; optimize all coefficients together, no frozen states.
This amortizes the spatial trunk and supplies temporal resolution without30
networks. Rank24 gives phase degree24 and exact eliminated mu degree72 in s;
80 global quadrature intervals avoid chemical interpolation aliasing. Small
output initialization retains IC as starting guess. This is a trial-space and
conditioning bundle, not an isolated metric gain. One checkpoint covers all t.

Global basis improves phase (~38% at4500) but native mu pooled exceeds700%.
Test a thermodynamic balance term, not phase-only selection: with native
E=sum V[beta/4(phi^2-1)^2+epsilon^2/2 phi Aphi], weighted symmetry gives
dE/dt=<mu,phi_t>_V=-M<mu,A mu>_V for the same CH equation. Add the squared
normalized integral defect (E(t)-E0+M integral<mu,A mu>)/E0, weight1, alongside
the integrated PDE norm. E0 is known IC energy, no future/reference label.
With phase degree24, dissipation degree144; use160 quadrature intervals so
this nonlinear temporal integral is unaliased too. Verify energy variation,
independent dense-matrix objective and gradients. No freezing or time windows.

One further normalization control is warranted by the integral residual's
R(t)=O(t): squaring R with physical dt underweights the initial evolution by
t^2 even though initial state is explicit. Test the time-average residual R/t
(strictly positive queries), with the same spatial resolvent/global basis and
no energy term. This restores finite initial-rate information without initial
trajectory labels, changes only the least-squares metric, and keeps continuous
roots. Record separately from the energy control; do not combine untested
heuristics. After this comparison select/validate a fixed global strategy or
report its accuracy limitation; avoid endless controls.

The R/t global control materially improves phase but still leaves large chemical
errors. Before closing, test one optimizer-only continuation direction: exact
Hessian-vector Newton-CG after the same5000 SOAP updates, using the unchanged
global objective and no new physical constraint. Matrix-free products are AD
of the scalar loss; no external state solve/targets. Limit200 Newton iterations,
record actual value/gradient/Hessian calls rather than equating them to LBFGS
updates. A manufactured time-PDE solve must pass first. This is generic
optimization, not another ansatz/loss or a guarantee of accuracy. No more
formulation variants in this task; validate fixed winner or report limits.


## 2026-10-28 — Reject costly Newton control; check observed terminal oscillations

Newton-CG interrupted at554.979s total inside its CG Hessian product. Only1
iteration logged; no complete Newton result or accuracy claim. Retained SOAP
stage independently replays. A new physical concern emerged in field review:
R/t global model follows morphology early but develops spatial grid-scale
oscillations at terminal time (native mu2716%). For high spatial eigenvalue k,
the integral metric behaves like averaged-in-time perturbation plus a small
instantaneous term ~delta_phi/(t*k); small integrated residual need not bound
instantaneous oscillations. This motivates one targeted recombination, not a
new formulation: same global basis24/N80, use the already independently tested
differential residual and Neumann A^-1 norm, with exactt0 evolution included.
No new shape/architecture/reference prior. All samples/states remain jointly
trained. Compare final phase/mu and energy; do not select based on loss alone.


The instantaneous global-basis control reduces overshoots/chemical defects but
still fails morphology. Test a single literature-backed causal weighting control,
only Eq3.2 of Wang/Sankaran/Perdikaris2022 https://arxiv.org/pdf/2203.07404.
Weights w_i=exp(-strength sum_{j<i}loss_j) use stopped parameter gradients.
One network/all nodes/all coefficients throughout, no time marching or frozen
states. Explicit strength1/minimum1e-8; normalize by sum(weights), a declared
variation from the paper's1/N. SOAP uses adaptive weights; L-BFGS uses uniform
weights so its scalar objective and gradient are consistent (do not recompute
stopped weights inside a line search). Validate this gradient rule independently.
Source is a general training heuristic, not a new physical law or a promise of
window-free accuracy. No reference labels or new architecture. This is the
last causal optimization control on the existing strong model, not a new PDE.


## 2026-10-28 — Isolate capacity after temporal approximation audit

Causal control completes but fails phase56.77%/64.46%; SOAP terminal weight
only.0196, so the fixed5000 budget has not activated full causal coverage.
Do not label it converged. Validation-only free nodal coefficients show rank24
log-time basis can approximate native phase0.05746% pooled/.08887% terminal,
mu.793%/4.322%. This is not a PINN/learned result. Initial derivative defect
still62.99 RMS, so temporal derivative sensitivity is not ruled out; spatial
neural capacity/optimization are also unisolated. Do one capacity control only:
same global basis24, instantaneous Neumann-inverse PDE and81 fixed time queries,
change hidden width64->128. No new loss/constraint/geometry/trajectory prior.
Keep5000+500 budget/ordinary FP64, record parameter/runtime cost. If useful,
freeze and check2026; otherwise retain all failures without SOTA promotion.


## 2026-10-28 — End current global sweep without claiming a solved case

Width128 control completes5000+500,308.02s, phase65.635%/73.925%, native
mu59.396%/188.274%; it does not resolve the one-network problem. Ten completed
single-seed controls now exist plus one interrupted Newton cost control.
Lowest completed phase remains R/t21.54%/43.18%, with native mu258.1%/2716.0%,
energy increase.0671 and overshoots. Do not promote SOTA/default or run seed2026
merely to validate a clearly inaccurate candidate. Preserve every result and
original physical/reference/default paths. Single-network implementation and
sweep are delivered; high-accuracy objective remains incomplete, not blocked
by permissions/data. Next resume must isolate spatial neural fit versus temporal
initial-derivative resolution with diagnostics before further loss variants.
Unconstrained time-basis fitting is not a neural result. Simple terminal-error
correlation with IC is nearzero (phi cosine-.0109; Aphi cosine-.113), insufficient
to support a uniform leftover-IC explanation or invent a new skip constraint.
No Git operation. Relation: confined-ch-global.

## 2026-10-28 — Restore global mixed fields and distinguish scaling from elimination

User asks to inspect phi/mu architecture, field scales and the two PDE losses.
Original learned mode is one shared MLP/two channels; the recent lowest-phase
trial eliminates mu and minimizes one fourth-order evolution residual. Known-IC
native RMS values are phi0=.674445, mu0=.488573, phi_t0=99.6552; validation-only
native terminal mu RMS=.043815. These differ in both amplitude and time scale.
Test globally supported mixed time expansions with explicitly learned mu and
two residuals. Use Neumann A^+r_phi/M, which has chemical-potential units, and
r_mu=mu-beta(phi^3-phi)-epsilon^2 Aphi; dividing both by the same declared mu
scale preserves their relative weighting. Explicit mu output scale=.5 is a
rounded known-IC RMS estimate, independent of future reference labels. Compare
shared versus separate field MLPs at identical ranks/physics/time nodes/budget;
separate means field networks trained jointly on the entire interval, never
sequential time windows. Do not infer a stable ranking from a single seed or
hide mu errors. Existing eliminated checkpoints must keep their templates.
Relation: confined-ch-global; extends the prior diagnostic plan at user request.

## 2026-10-28 — Test an independently checked mixed block inverse

Scaled shared mixed control completes5000+500: common phase58.490%/63.504%,
native mu70.278%/301.481%,265.58s. Separate-field control is running with
identical case sources. Scaling/two outputs alone do not resolve phase accuracy.
One further controlled conditioning experiment will invert the known linear
mixed Volterra block, not alter the PDE or freeze any temporal states. For
R=phi-phi0+M A W mu and C=mu-f(phi)-epsilon^2 Aphi, the block
L=[[I,M A W],[-epsilon^2 A,I]] has inverse applied as
Ephi=(I+M epsilon^2 A^2 W)^-1(R-M A W C),
Emu=C+epsilon^2 A Ephi. W is the existing global log-time integration matrix.
Ephi/Emu have phase/chemical units; use scales1/.5. Check against an independently
assembled dense2x2 block and parameter finite differences. This is ordinary
linear residual preconditioning with globally coupled nodes, not a time-window
solver or future reference prior. Roots preserved at fixed quadrature; finite
polynomial quadrature error is reported separately. Reject promotion if mu or
energy fail even when phase improves. Relation: confined-ch-global.

## 2026-10-28 — Mixed fields improve terminal/chemical metrics but do not solve global training

Three physics-only5000+500 cold starts complete with final/best CPU replays.
Shared/separate/block common phase pooled58.490/71.242/46.106%, terminal
63.504/71.301/38.602%; native mu70.278/74.059/50.509%, block terminal130.065%.
Block has sampled energy max rise0 and improves terminal phase versus prior
R/t43.178%, but pooled phase remains worse than21.540%. Do not promote SOTA/
default or call output scaling alone effective. Separate fields have32,688
parameters versus shared17,904; shared/separate40-source snapshots are identical,
and separate/block initial checkpoint files match directly (no hashes). All
three physical targets/operators/references/budgets unchanged; block also changes
integral residual and norm, so this is not an inverse-only causal ablation.

SAME-network supervised-only field capacity diagnostic (identical initial
parameters,5000SOAP) reaches native phi.43846%/mu2.7160%; no diagnostic weights
enter any PINN. Its independently evaluated block PDE loss8.985e-5 is30.745x
below physics-trained.0027624. Thus this architecture contains a much better
joint field/residual approximation; current block physics optimization path
fails to reach it. This does not prove global optimality or sole optimizer cause.
Its strong loss.45948 remains large, so accurate fields do not guarantee accurate
time derivatives for another norm. Free nodal time projections with known initial
rate/mu reduce strong loss18.57x while retaining sub.1% phase, but are neither
neural nor residual-minimizing fits. No single-source explanation established.

Block boundary band~21.10% area carries57.39% terminal native squared error;
bulk native relative phase28.47% still remains. This is not reference-guided
training reweighting. Next investigate a reference-free initial-linear/temporal
conditioning or continuation path, using the established feasible block objective;
do not widen networks or choose seeds blindly, transfer supervised weights, or
return to frozen time windows. Record proposed formulas/validation before any
next implementation. Accuracy remains OPEN; no Git action. Relation: confined-ch-global.

## 2026-10-28 — User selects eliminated phi, tanh/Jacobian and Fourier controls

User retains phase-only/chemical elimination and explicitly asks to test tanh
with the corresponding residual transform and inspect Fourier frequencies.
No tanh output/residual trial exists in confined_ch2d; previous tanh there was
only the prescribed IC. Frozen references slightly overshoot unit wells (FEM
[-1.02634,1.02517], native[-1.02776,1.02486]); strict tanh cannot represent all
those values. Use declared amplitude1.1 (fixed10% margin about the well scale,
not a fitted future trajectory); retain CLI entry1.0 with its range limitation.

Use u=atanh(phi0/a)+s sum c_k T_k, phi=a*tanh(u+alpha(t)). Solve the scalar
alpha so physical-volume mean is exactly prescribed. A post-tanh additive mean
shift would change the map/range/Jacobian, so reject it. Implicit derivative is
alpha_t=-sum(v J u_t)/sum(v J), J=a*sech²(u+alpha). Eliminated physical residual
r=phi_t+M A[beta(phi³-phi)+eps²Aphi] is transformed to r/J, BEFORE the existing
invertible spatial metric. A does not commute with this multiplication; no
Allen-Cahn-style cancellation of the entire CH spatial term is claimed. For
r/J the mean need not be zero, so use the actual invertible(A+Pi0)^-1, never
silently project away its constant component. Test full chain and parameter
finite differences independently; no numerical Jacobian floor/hidden clipping.

Predeclare4 seed42 cold starts5000SOAP+500LBFGS: fresh eliminated strong modes8
control; tanh/original residual modes8; tanh/Jacobian residual modes8; same
Jacobian scheme modes5. Use instantaneous residuals: tanh composition invalidates
the previous polynomial unaliasing argument for the integral recipe. Physical/
reference/IC/time horizon unchanged, one global model, no frozen states or
supervised weights. Fixed source before the four controls. Fourier features
are sin/cos(pi*k*z) for k1..8, not2pi*k*x; wavelengths x.125/y.1. Initial
fundamental coordinate wave numbers reach5, with higher harmonics from tanh.
Current global basis has NO temporal Fourier embedding (static spatial MLP);
time is log-Chebyshev rank24. Lower to5 experimentally, no assumption that8
is intrinsically excessive; changed input dimension/initialization is recorded.
Relation: confined-ch-global; supersedes the next mixed-block probe at user request.

## 2026-10-28 — Add frequency-only control on the retained best integral recipe

Primary instantaneous controls show map-only phi58.56% versus fresh59.01%;
Jacobian-before-metric modes8 worsens to73.07% (best evaluated71.00%). An
independent step1000 audit finds transformed constant mode only4.25% of its
loss, so reject an unsupported explanation that mean-mode amplification alone
caused failure. Midpoint dense-time loss matches training nodes, no large
between-node defect at that checkpoint. Do not invent another operator ordering
or declare tanh universally ineffective from this single strong-residual recipe.
User also asks whether Fourier frequencies are too high. In addition to the
modes8/5 Jacobian pair, run ONE frequency-only modes5 control on the retained
best phase-only integral R/t recipe, through its existing CLI override. No new
mapping/loss/formulation/default. This tests frequency reduction on the actual
retained strategy; it is not a claim of an identical-parameter/seed-invariant
ablation (input dimension/initialization change). Existing modes8 integral run
remains historical, no fresh matching rerun or stability claim. Keep5000+500,
FP64/ordinary, seed42 and frozen physics/reference. Name
20261028-chtanh-average-m5-01; launch after4primary controls. Relation: confined-ch-global.

## 2026-10-28 — Retain eliminated phi; tanh transform fails these controls, lower band helps R/t

All5 new cold starts5000SOAP+500LBFGS complete. Strong original/tanh-only/
tanh-Jac-m8/tanh-Jac-m5 phase FEM pooled59.012/58.563/73.073/69.271%, terminal
65.162/64.977/78.424/75.100%. Tanh-only phase difference is small versus known
process variability; direct r/J worsens this recipe. Not a universal statement
about tanh/CH, and not a test of tanh with the best integral loss. No hidden
Jacobian clipping, map derivatives/implicit mass or exact chemical law defect
found:2focused tests pass, trained independent derivative max2.3e-12, allCPU
final/best replays pass. Mean mode contributes only.052%/.020% final Jacobian
loss; midpoint objectives agree within1%. Reject claims that either explains
all failure. Constant25ish inverse-J weights can still alter the optimization
path; exact CH diffusion terms cannot globally cancel J.

Frequency-only current-best integral R/t modes5 reaches phi17.9854%/29.3681%,
native mu129.292%/836.597%,190.78s, versus historical modes8phi21.540%/43.178%,
mu258.072%/2716.044%. Phase[-1.34285,1.24822], energy max rise.007078 (old.067052)
remain poor. No secondseed is warranted as a SOTA promotion while these defects
remain. Add explicit train_coarsening_average_lowfreq working preset, identical
to the completed CLI override; keep original simple default and opt-in tanh.
Preserve the old integral/mixed results. User's phase-only preference supersedes
mixed-block as the next primary route. No default/SOTA/Git or reference changes.

Modes8 features have wavelengths.125/.1,9--11 median neighbor spacings: no clear
input sampling violation. Native A² feature gains are2.46--3.30 times modes5;
higher frequency can still stiffen learning. Lower input count/initialization
and one-seed/historical-control limitations prevent a strict frequency-only
causal or stable ordering claim. Keep source-parity records (mapped controls
identical51 files; baseline active functions identical, unused tanh helpers were
stabilized before mapped runs). Next preserve phase-only modes5R/t and address
late chemical/high-frequency errors via a derived single-field full space-time
linear inverse, rather than blindly adding maps/capacity/seeds/time windows.
Relation: confined-ch-global; accuracy remains OPEN.

Clarification for this session's finite-weight observation: inverse-J weights
are state/time dependent, not constant or stopped. Their sampled maximum is
about23.3; the earlier "Constant25ish" wording referred only to finite magnitude.
No weighting floor, clipping or numerical derivative saturation is used.

## 2026-10-28 — User requests an intermediate-difficulty global CH control

Retain the ellipse/two holes, physical parameters (including epsilon .02),
regional delta .03 and collocation mesh .02. Reduce ten initial waves to three
integer waves (1,1), (2,-1), (1,-2), with amplitudes .8/.7/.6 and phases 0/.8/1.6;
keep saturation .98, scale .55 and wall width .03. Shorten T1.5 to .75, eleven
reference snapshots. This changes the physical initial-value problem; regenerate
and independently refine FEM/native references in a new moderate_multimode3
child. Preserve all rich-case artifacts and strict phi/mu refinement gates.
Use the retained phi-only M5 global R/t strategy, rank24, rebuilt80 temporal
nodes, 5000SOAP+500LBFGS, seed42 runtime provenance. No tanh, windows, new loss,
reference supervision or default promotion. This first control tests the bundled
effect of simpler IC/shorter horizon; it cannot isolate either cause alone.
Rejected reverting to the original unsaturated weak-perturbation pilot, because
the user still wants visible phase separation and nontrivial perforated walls.
Relation: confined-ch-global; supersedes the next space-time-metric probe.

## 2026-10-28 — Easier CH improves phase but does not solve global accuracy

New three-wave/T.75 strict independent reference passes (phase mesh/time max
.1627%/.0154%, mu .1857%/.0371%, mass/energy pass). Matched retained M5 R/t
strategy cold-start5000+500 takes188.15s: common-FEM phase5.1557%/9.7665%, native
phase4.2891%/8.9375%, versus rich-case common phase17.9854%/29.3681%. This changes
IC AND horizon, so reject isolated causality/same-IVP method/stable-order claims.
A frozen-IC baseline on new case remains40.84%/47.65%; reference pure area grows
18.75% to42.92% with60.8% field change, preserving nontrivial dynamics.
Native mu45.06%/156.50%, actual error RMS.1981 vs target.1266, phase overshoots
[-1.116,1.174] and151-time energy max rise.0003325 persist. Do not conceal these
or promote SOTA/default. Keep this explicit intermediate benchmark and all richer
results. CPU final/best replays, independent chemical/mass audits pass; no core
formula changes or Git action. Relation: confined-ch-global; difficulty sub-study
complete, global accuracy OPEN. Next controlled option is band5 versus3 on same
new IVP/reference, or separate ten-wave T.75 for IC/time attribution.

## 2026-10-28 — User selects time windows again; document actual switching

User explicitly says to adopt time windows, superseding the earlier no-window
requirement. Select existing validated train_coarsening_windows for the original
rich T1.5 case; preserve original simple default, global/moderate experiments,
physics/reference/old paths. No new training or adaptive algorithm requested.
Boundaries are fixed log1p spacing with tau.005 and30 windows, not loss-triggered.
First650SOAP+max36LBFGS, later150+max16; both archived runs hit L-BFGS iteration
caps in all30windows. Status is logged but success/residual/endpoint accuracy is
not a prerequisite to propagate. Reject claiming each window is certified
converged. Final predicted grid phase is frozen as next initial state, with hard
continuity/mass and exact chemical law; weights warm start and optimizer resets.
Errors can propagate/amplify despite conservation. Native terminal phase.809%/
.748% includes cumulative fitting/propagation at the tested horizon, not a
long-time guarantee; chemical defects persist. Independent residual/endpoint-
stability acceptance is a future candidate only, not silently added now.
Relation: confined-ch-global task retained as the same investigation; goal
updated explicitly to selected window route with all-field accuracy still open.

## 2026-10-28 — User requests global shared two-field loss balancing

User requests one full-horizon network with phi/mu heads, scale normalization
and stagger or GradNorm, on the original ten-wave T1.5 physical case. Existing
shared GlobalMixedNetwork uses one trunk/24 coefficients per physical head,
not sequential states; fixed-weight scaled mixed runs exist, adaptive/stagger
controls do not. Predeclare three NEW matched seed42 cold starts: fixed joint,
gradient-norm balance, loss alternation. Keep original validated reference,
rank24/shared64x4/modes8, known-IC-scale mu output.5, chemical-unit two losses
[(A+Pi0)^-1 r_phi/(M*.5), r_mu/.5], fixed full-horizon nodes, hard phase IC/mass,
ordinary FP64, SOAP5000/LR.001+LBFGS500. No chemical elimination or time windows.

Reuse project's corrosion gradient-norm balance (inverse whole-parameter L1
gradient norms, NOT original relative-training-rate GradNorm): refresh25,
alpha1, norm floor1e-6/ceiling1e6, weight floor1e-6/cap10000; stop weight AD.
Stagger activates phase/chemical losses alternately in25-update blocks for4000
updates, then1000 joint; all parameters remain shared/trainable with one optimizer
state. It is loss alternation, not a frozen-field block solve. No compounded
bias/active-mask state. Refinement is joint in every arm; adaptive arm freezes
final positive weights for a consistent L-BFGS objective. Log parts/weights/
active masks/norms and exact phase/chemical/joint update counts. Mathematical
PDEs/normalization stay as independently checked mixed tests; check new gradient
recurrence/masks/frozen weighted objective with focused independent tests before
three serial full runs. Reference labels remain monitoring/checkpoint selection
only. Preserve selected window fallback and all old checkpoints/defaults.
Relation: confined-ch-global; new active sub-study, not a default promotion or
SOAP investigation. No Git authorization.

## 2026-10-28 — Add one raw-residual normalization guard control

Three matched global shared-head controls complete; fixed/gradient_norm/stagger
common phase55.88/63.63/68.83%, all poor. Independent actual loss reconstruction
passes; gradient norm reduces normalized chemical loss .03038 ->.00002972 while
evolution rises .02053 ->.20032. All three retained the older Neumann-inverse
spatial metric. Do not attribute their failure to every two-head/GradNorm scheme.
Add ONE raw-residual+gradient_norm control, same network/IC/reference/budget,
with explicit evolution_scale100 (rounded known-IC native phi_t RMS99.655) and
chemical_scale.5 (known-IC mu RMS.4886). Losses are mean[(phi_t+M A mu)/100]^2
and mean[(mu-beta(phi^3-phi)-eps^2 Aphi)/.5]^2. No inverse spatial metric or
future target scale. Extend optional validator/loss scale, independently check
the raw residuals/parameter derivative and unchanged legacy mixed case, then
fresh5000+500. It changes residual norm relative to first three, not physics.
Preserve first-three identical57-source/initial/time records; fourth snapshot
records later optional-loss source change separately, no four-way source parity
claim. No defaults/Git/reference changes. Relation: confined-ch-global.

## 2026-10-28 — Four global shared-head controls fail to improve complex CH

Complete fixed/gradient_norm/loss-stagger controls on original ten-wave T1.5:
common phase55.8807/63.6294/68.8259%, terminal62.6196/71.5108/67.2824%; native
mu67.73/147.12/68.14%, terminal277.33/475.95/348.65%; trainers265.60/293.13/
282.42s. Later raw-normalized gradient control gives55.7275%/64.2271%, native
mu104.09%/364.33%,113.81s. All5000+500, identical initial parameter files.
Best-phase checkpoints do not rescue accuracy. No promotion over prior global
phase-only17.985%/29.368% or windows3.91%/5.75%.

Actual SciPy reconstructed objectives equal logged scalars; final/best CPU
replays max phase7.12e-15/mu1.89e-14. First-three57source files/full time arrays
identical; later raw changes only optional loss/validator/README/new config,
not network/train/balancing/operators. Adaptive refresh200times and stagger
2000/2000/1000 counts verified. First-three norm bounds/weight caps not active;
raw hits phase weight cap3times, recorded separately. Gradient balancing reduces
chemical loss .03038 to.00002972 but evolution rises .02053 to.20032; learned
mu consistent with wrong phi can still be physically inaccurate. Stagger's
smaller scalar does not imply smaller field error. Dense energy rises persist.
Do not infer a universal impossibility of global/shared-head/GradNorm methods.
Loss alternation is not a frozen-field stagger solve; project gradient balancing
is not original relative-rate GradNorm. Preserve these limits and all controls.
Selected window fallback remains; no additional blind seed/step/weight sweeps,
reference/default/history changes or SOAP investigation. Relation: confined-ch-
global; requested experiment sub-study complete, all-field accuracy still OPEN.

## 2026-10-28 — User makes verified complex CH time windows the CLI default

Explicit user request: retain time-window route and make it default. Default
train.py now selects existing train_coarsening_windows (original ten-wave T1.5,
30log windows,5000SOAP+500LBFGS total, FP64 ordinary); solver.py defaults to the
same physical/reference config. Keep train_base as the explicit legacy simple
foundation: changing it would cascade into coarsening and every research preset
or introduce an import cycle. No physical or numerical parameter retuning.
Canonical train entry resolves a blank window run_name to a timestamp plus
nonlocal-windows on a copied runtime config; direct window trainer retains its
explicit-name requirement. Existing named runs/checkpoints and loading unchanged.
README quick start/legacy commands, DEVELOPMENT exception, project overview and
artifact default/legacy metadata updated. No new training, FEM, SOTA/all-field
accuracy guarantee or Git action; all previous experiments retained. Verify real
CLI default resolution, window dispatch/name handling with mocked trainer,
existing31 config schemas/reference and archived selected trajectory. No expensive
unchanged numerical suite required for this entry/default change.
Relation: confined-ch-global; explicit new default supersedes historical notes.

Delivered/verified: real no-config CLI/default historical-strategy parity and
window dispatch (mocked to avoid training), automatic copied-config name and
explicit-name preservation, explicit legacy CLI/reference, solver parser default,
all31schemas, strict reference/selected30-window trajectory and docs/index/diff
checks pass. Numerical strategy unchanged; no full training/FEM rerun needed.
Default switch complete; remaining chemical/global accuracy limitations deferred,
not claimed solved. No Git action. Records: archive/chwindow-default-selection.json.

## 2026-10-28 — Explicit switch to random-interface electropolish operator

User selects electropolish neural operator, referencing PI-CViT cvit.py, and
requests repository style plus shared-space tensor split. Mark completed CH
default work inactive; retain all its unresolved limits. Prototype the same KKS
PDE/graph on a distribution of existing SDF rough profiles, with disjoint runtime
IC IDs and no positive-time labels in the physics objective. Preserve single-IVP
CLI default and checkpoints; opt in through train.py. Use PI-CViT-inspired patch
encoder/cross-attention; arrange expensive spatial condition decoding before
pointwise temporal FiLM so it can be reused exactly across all times. This is a
separable variant, not an assertion of equivalence to upstream time-dependent
cross-attention. Simpler pooled-condition FiLM is an explicit comparator, not a
claimed universal improvement. Validate cached/uncached outputs and time/parameter
AD before training and independently refine held-out same-graph ODE references.
No future-interface geometry, reference labels, new PDE or Git history operation.
Relation: electropolish-neural-operator.

## 2026-10-28 — Random-interface references exceed old liquid-map bound

First two independently refined random-interface graph references have positive
q maxima .00298/.00148, i.e. c_L>c_Le. Old sigmoid KKS map artificially enforces
q<=0 when c_Se=1 and cannot represent these targets. Do not alter reference or
clip it. For the NEW operator use explicit concentration_map=linear_liquid:
phi=(1+tanh(z))/2, c_L=c_Le*s with unbounded network s, c=c_L+dc*h(phi).
This retains KKS mixture algebra and the same physical phi/c PDE; removes only
an artificial output bound. Original single-IVP default/checkpoints unchanged.
Known local input features remain prescribed phi0/c0, not evolved labels.
Check physical residual/time/parameter AD independently and report c_L/q and
physical ranges; no assumption that total-c constraints follow automatically.
Relation: electropolish-neural-operator; no SOAP or physics/reference retuning.

## 2026-10-28 — Public operator config follows arch/data ownership

Move neural-operator model dimensions/conditioning under c.arch, leaving dataset
family/image shape/action under c.operator. The first two pilot processes already
captured the initial all-in-operator experimental schema. Preserve their exact
saved configs/source/checkpoints unchanged. Private _build_network now accepts an
explicit architecture record and image_shape; archival replay passes the saved
operator record explicitly, new training passes c.arch. No implicit schema or
numerical fallback, state conversion, field values or artifact relocation. Check
new/public-template versus saved pilot-template equality and canonical dispatch.
Relation: electropolish-neural-operator; configuration ownership cleanup only.

## 2026-10-28 — Keep FP32 trained-attention replay limits explicit

Both2000Adam operator pilots complete. GPU replay with saved highest-matmul
policy recovers cross fields exactly and film within6.26e-7. A default-policy
GPU probe was wrong arithmetic; retain its log, use saved policy. Even with
highest, CPU FP32 cross fields differ by1.62e-3 and shared-vs-repeated-query GPU
raw outputs by~.00233; film CPU gap1.43e-6. Aggregate cross phase pooled/terminal
changes only~2.3e-5/9.9e-5 absolute. Do not silently label strict pointwise CPU
probe passed or infer a code/PDE defect. Small random FP64 split/time checks pass.
For trained-network reuse, record FP32 differences explicitly and verify the
algebra again using promoted trained weights/inputs inFP64; timing remainsFP32.
No production arithmetic/normalization retuning or SOAP investigation. Store
per-backend outputs/metric differences and avoid bitwise/batch-invariance claims.
Relation: electropolish-neural-operator.

## 2026-10-28 — Electropolish operator prototype and two pilots delivered

Two matched2000Adam original-physics pilots complete: mean held-out phase pooled
cross17.185%/FiLM17.462%, terminal13.305%/11.936%, vs frozenIC47.721% terminal;
trainer245.682/160.897s. Same16train/4test,5s,55common initial arrays,14of15exact
sources; sole snapshot delta fixes initial new-CLI return-dict/absl exit1. Cross
trainer finished/saved despite wrapper code; film and final c.arch width8/seed2026
one-update smoke exit0. Existing single-IVP defaults unchanged. Six references
max refinement1.93e-8; no positive-time labels or test checkpoint selection.
Three distinct focused numerical checks and real canonical/template/provenance
checks pass; retain failed harness/probe logs. No universal architecture ranking.

Spatial cache forward+timeJVP gains1.162x/1.108x on1024x4, excluding PDE/optimizer.
Trained promotedFP64 shared/paired values and JVP exactly equal in both cases.
FP32 cross batching relative gaps2.28e-4/1.62e-4 remain; CPU physical gap.00162,
aggregate phase drift<.01percentage point. GPU saved-highest cross exact/film
6.26e-7; CPU film1.43e-6. Strict cross CPU field probe is NOT passed. Do not hide
it, change production arithmetic or overstate the speedup. Early excessive
roughness smoothing and freeDq~.77-.81 show remaining physics-fit limits.

Archive requested prototype task with direct paths/raw evidence, scientific
figures, README/DEVELOPMENT/index and idle-state resume. Future5000SOAP or joint
geometry/time FiLM are proposals only, not queued; deferred SOAP issue untouched.
No Git history action or old artifact migration. Relation: electropolish-neural-
operator COMPLETE as prototype, accuracy research remains open on user request.

## 2026-10-28 — User requests fixed single-interface CViT fitting control

Clarify previous training used16interfaces with2conditions/update and vmap/mean,
not one input. User authorizes initial single-sample test. Predeclare source
training index0/B1, same cross-attention/Adam2000/LR.001/seed42/5s/FP32highest/
PDE/IC/BC/graph and sampling sizes; new cold start, not resume. Reuse strict
existing dataset/reference unchanged. Add explicit optional operator.training_
indices: absence uses entire source train partition; present tuple is a unique
nonempty subset of training IDs, never test IDs. Preserve absent-option random
schedule. Report selected input as train, excluded source-train monitors as
unused_train, independent tests as test. Compare sample0 to same sample0 of prior
16-input run, not its test mean. One input changes condition batch/optimization
objective/work, not a pure generalization ablation or proof of NO generality.
No blind optimizer sweep, math/reference/default/Git changes. Relation:
electropolish-operator-single.

## 2026-10-28 — User requests optimizer alignment and online input sampling

Established electropolish PINN usesSOAP5000/LR.001; earlier operator pilots used
Adam2000. Fixed-inputAdam2000 now complete at phi2.483%pooled/3.387%terminal.
User asks optimizer choice, retains single-input test first, then explicitly
requests online sampling from continuous initial-interface space, B8/16 as memory
allows. Run one matchedSOAP2000 fixed-input control before selecting optimizer.
Online extension will draw continuous amplitude/frequency/phase coefficients each
update, not replay a finite seed list. Keep PDE, known SDF definition, graph and
held-out reference bank; declare any initial-profile validity rejection explicitly.
B8 first; measure actual memory. Preserve prior experiments/defaults/precision
limits and reject generalization claims from fixed-input fitting. Relation:
electropolish-operator-single, continued scope with authorized online stage.

## 2026-10-28 — Continuous online SDF distribution and validation declared

Online input triples draw a in[.3A,A), f in[f_lo,f_hi), phase in[0,2pi), the
existing six-mode family. Fixed4B candidates/update, accept firstB whose2000-
vertex profile lies strictly within(.005,.495). This explicitly conditions the
input distribution on staying inside the physical domain; no amplitude clipping,
finite-seed-bank replay or future-solution prior. If valid_count<B fail before
any optimizer update; no padded sample silently used. GPU JAX SDF preserves the
existing2000vertex/±16segment query formula; nearest segment indices and grids
precomputed per fixed-shape spatial pool. Independent NumPy SDF identity and
fresh/replayableB8/range/BC tests pass (2tests5.56s). No parameter AD through
random target construction; time JVP still only in field decoder. Reference
bank used only for monitoring/test under online mode; no train-L2 label for it.
Plan explicitonlineB8/5000 baseline after optimizer control, record coefficient
triples per update and allocator peak at first/25th/final updates. Relation:
electropolish-operator-single continued; user authorized online stage.

## 2026-10-28 — Choose Adam for the first online operator baseline

Matched single-input2000 controls complete: Adamphi2.4829%pooled/3.3870%terminal,
SOAP4.1869%/3.2136%; c2.9651%/4.0312% vs4.9412%/3.7003%; trainer237.26/387.33s.
Both sample0/B1, same initial arrays/model/PDE/time samples pending final parity
check. SOAP slightly better terminal, worse pooled and higher compile/runtime;
not a universal optimizer ranking or statement about establishedSOAP PINN.
Select Adam/LR.001 for onlineB8/5000 as the clearer inexpensive baseline; retain
SOAP as explicit override and both results. Profile actual peakmemory, never
change batch silently. User authorized continuous-online second stage. No SOTA/
physics/default/Git or SOAP implementation changes. Relation: ongoing task.

## 2026-10-28 — Fixed-input and continuous-online controls completed

All three new cold starts exit0. Fixed-inputAdam/SOAP2000 use sample0 at every
actual update and73identical initial arrays; onlyoptimizer/name differ. Same-
input old16-case phi21.538%pooled/15.888%terminal falls to2.483%/3.387%Adam or
4.187%/3.214%SOAP. Not an isolated condition-count ablation (B changes), nor NO
generalization or universal optimizer ranking. Preserve Adam online baseline.

OnlineB8/5000 generates40000distinct actual coefficient vectors, not a finite
training bank. Minimum31/32candidates pass declared profile bounds; no fallback
or batch reduction. Fourheldout phi17.748%/12.355%, c19.317%/14.029%,326.71s;
early shape/transport accuracy remains open. Prior finite pilot has different
budget/batch/distribution, so no pure online-accuracy-gain claim. Peak active
JAX2.15GiB, pool5.01GiB, observedNVML~5.3GiB on reported48GiB; B16 not measured.
GPU final replays all within1.43e-6; independent fields/actual-input use and
32NumPy height checks pass. New subset RNG/exclusion test and two independent
continuousSDF/fresh-batch tests pass. Existing numerical core/source/read-only
reference bank/defaultPINNSOAP retained. Save raw logs, source/config/checkpoints,
actual coefficients/sampled IDs, scientific curves/fields and index/README.
Archive current task; no further training/optimization or Git operation queued.
Relation: electropolish-operator-single, extended online stage COMPLETE.

## 2026-10-28 — Diagnose online electropolish equation balance before changing training

User authorizes AC/CH dominance, gradient-conflict and concentration diagnosis
after inspecting unseen-interface16 fields. Keep the completed onlineB8/5000
Adam final checkpoint, PDE, sampler, references and all defaults unchanged.
Analyze recorded five-part losses/weights; supplement absent gradient history
with explicitly labeled frozen-checkpoint probes, not inferred historical
gradients. Predeclare actual online batch iterations0,1249,2499,4999, retaining
the original coordinates/time draws and comparing stale saved-final versus
newly proposed inverse-L1 weights. Inspect initial checkpoint on batch0 as an
initialization control. Compare heldout16 and all four heldout fields with
independent SciPy graph concentration transport, phase-interface roughness,
and physical RHS terms. Gradient directions/caps alone cannot establish unique
causality; no optimizer sweep or training modification until evidence warrants
it. Relation: electropolish-operator-balance diagnostic.

## 2026-10-28 — Frozen online diagnostics do not support persistent AC/CH dominance

Same-source initial/final replay on actual predeclared batches0/1249/2499/4999
passes; initial losses/weights reproduce logs. Last1000 logged AC/CH weighted
loss ratio median.467(range.119..2.376). Final AC/CH gradient cosines.010..119,
fresh inverse-L1 weighted L2 norms AC.877..1.233/CH1.289..1.645; no strong
negative conflict or orders-of-magnitude dominance on these probes. BC weights
cap1000 yet their weighted gradients stay relatively small. Saved-final IC
gradient norm share~7..8% versus freshly balanced~23..28% on these batches;
this illustrates stale stochastic weights, not reconstructed historical failure.

Four heldouts fit initial phi~1.17..1.63% but lose distinct roughness by t*.05.
At t*.25 model std(height)~.002 across all four, reference.0173..0480. Heldout16
terminal qL2.03414 yet DqL2.80704; independent SciPy action and GPU fields agree
with existing reference/checkpoint. This is shape/transport and condition-
retention failure; do not rank stagger as a remedy without a trained control,
or treat q value accuracy as transport accuracy. Dense early residual probe
and condition-spread quantification underway; no training/default/physics change.
Relation: electropolish-operator-balance, measured diagnostic evidence.

## 2026-10-28 — Supplement late-batch balance with near-initial-time diagnostics

The four predeclared actual batches all have t*>.27; their gradient balance
cannot rule out early conflicts. Independent dense time-JVP/SciPy graph checks
show [0,.05] holds49.81%AC/73.98%CH integrated absolute residual, with mean
residual18.85x/54.03x late time. Quadrature41->81 nodes per interval changes
integrals<=.508%, below the predeclared3% gate. Ensemble phi spread prediction/
reference changes102.93% at0 to6.03%at.05/.943%at.25/.447%at1 (only four heldouts).
Add explicitly non-historical final-model gradient probe using actual batch0
geometry/conditions and declared times(.001,.003,.005,.02), near the observed
early residual peaks. Preserve original four probes and distinguish this
supplement from recorded training times; no optimizer/model update or new
reference. Relation: electropolish-operator-balance diagnostic scope refinement.

## 2026-10-28 — Early frozen probe exposes PDE versus initial-data conflict

Supplemental early batch0 probe exits0: AC/CH cosine+.92657, versus IC-.59619/
-.52807. Early AC/CH L1 losses3.889/4.172; fresh balancing gives weighted L2
12.126/13.169 and IC9.763 (IC hits1000cap). Even with fresh weights, IC cosine
with the combined gradient is-.21081; with saved-final weights it is-.55551 and
IC contributes only.70% of sum of gradient norms. Thus early conflict is between
coupled PDE improvement and maintaining IC, not AC against CH, on this frozen
probe. Negative total-gradient IC alignment means an infinitesimal negative-
gradient step increases IC loss; Adam's actual state/direction is unavailable.
Do not claim the causal optimizer trajectory or a proven remedy. Prefer a
future explicitly controlled early-time sampling/condition-retention experiment
over assuming AC/CH staggering solves this failure. Finish as diagnostic only,
retain raw evidence and all defaults; no training or Git history change.
Relation: electropolish-operator-balance, acceptance achieved pending record checks.

## 2026-10-28 — Online balance diagnosis completed and archived

All diagnostic processes exit0; six-source/actual-coefficient parity, initial
loss/weight replay, weighted scalar-gradient linearity, four GPU field replays,
dense temporal quadrature, gradient-matrix/finite-array checks and strict
artifact JSON/syntax/links/figures/diff checks pass. Archive self-contained task
with scripts/raw JSON/NPZ/stdout and five scientific figures; update README
validation guidance, artifact index and dashboard(activeNone). Frozen probes
support early PDE/IC competition, loss of condition dependence and transport
error, not a unique causal optimizer trace or a validated correction. All
workers/GPU idle; no training, model/default/reference/Git history change and
no queued experiment. Relation: electropolish-operator-balance COMPLETE.

## 2026-10-28 — User requests shared causal training for online electropolish

Reuse existing per-time CausalWeightor compute_weights/update_eps, not a new
weighting law or window scheme. First matched B8/5000 Adam control changes
only causal weighting/state, retaining actual condition/time/spatial RNG,
model/physical losses/reference/default. Compute correctly normalized per-time
free-core MAE per condition before causal weighting; do not inherit legacy
masked-to-padded mean dilution. Stop weights in parameter AD. Separate AC/CH
weights use shared min(epsAC,epsCH) as legacy; update shared eps values with
batch-mean causal weights, reducing to legacy at B1. Keep IC/BC active and
record both physical raw and weighted losses/weights/eps. Per-time Nt4 uniform
sampling may still miss the narrow initial layer; do not bundle a hidden
sampling change or promise improvement. Independent checks precede canonical
train.py cold start, then replay errors/transport/input dependence. No shared
causal/PDE/default/reference/Git change. Relation: electropolish-operator-causal.

## 2026-10-28 — Matched online causal control completed without L2 improvement

Canonical causalB8/5000 Adam run exits0 in323.98s. Same73initial array leaves,
exact40000actual profiles/accepted counts and recorded time draws; config
diffs onlyloss.use_causal/run_name. Shared numerical/model/generator sources
match; online preset source differs by one explanatory comment, AST equal.
Four heldout mean phi pooled/terminal baseline17.748%/12.355% -> causal17.932%/
13.105%; c19.317%/14.029% ->19.521%/14.802%. Unseen16 terminal q3.414->6.253%,
Dq80.704->88.266%; roughness still collapses. Keep causal opt-in, no promotion.

Independent51record stopped-prefix/weighted/raw-loss/eps/time-RNG audit passes,
final GPU replay max6.26e-7;73array/stream/config/source/physical-map parity passes.
Three numerical/step/noncausal-gradient checks and corrected config rejection
check pass; harness-only static-function/ConfigDict fixture failures retained.
Initial byte-parity audit failed on a comment; record explicit AST exception,
never silently drop numerical source checks. No PDE/reference/architecture/
shared-causal/default/Git change or seed/stability claim.

Observed limit: per-time causal sorts only4sampled times;84.31%of51logged batches
contain no t*<.05, median earliest.1932. No persistent per-time residual memory
or IC prefix exists in this reused rule. Final epsAC1.61051/epsCH100, effective
min1.61051; causal weights are active, late-time mean weights~.65AC/.72CH in
last1000 logs. A future explicit early-covering temporal sampling control may
address this limitation but is a separate change, not a measured benefit here.
Finish and archive this requested causal comparison; no further sweep queued.
Relation: electropolish-operator-causal complete pending final record checks.

## 2026-10-28 — Causal operator integration and comparison archived

Final changed-module syntax/whitespace, saved/current source parity, strict
JSON, finite model fields, recurrence/time-stream/initial/GPU replay records,
scientific PNGs, task links/index/archive and git diff checks pass. Four focused
checks are covered by the numerical/step/regression and separate corrected
config commands; raw harness/comment-parity failures remain documented.
Archive task, mark dashboard activeNone, keep opt-in causal preset and measured
negative comparison visible in README/index. No second sampling control, default
promotion, additional training, shared-causal modification or Git history action;
all workers/GPU idle. Relation: electropolish-operator-causal COMPLETE.

## 2026-10-28 — User redirects diagnosis to conditional training and IC distance

Accept that observed early residuals/causal batch coverage do not establish
sampling as the root cause. Existing single-input CViT fits with uniform Nt4;
freeze sampling/physics/defaults and inspect training/condition dependence.
IC uses signed minimum Euclidean distance over33nearby segments of2000vertex
profile, with y-f(x) used only for sign; it is not globally exact SDF. Quantify
that approximation against exhaustive search without silently changing IC/ref.
Predeclare source0/1, heldouts16..19 and first four actual online profiles for
distance audit; condition image/local-feature swaps0/1 and16/17; per-condition
AC/CH/IC gradients of frozen online final batch0 at recorded and explicit early
times. Add one two-fixed-input(0,1)/B2/2000Adam/uniform-time cold start using
canonical train.py and existing reference bank; compare existing same-input
single/16-case controls with exposure/batch limitations. No stratified sampling,
new equation/reference, optimizer sweep, default promotion or Git change.
Relation: electropolish-operator-training.

## 2026-10-28 — Distance approximation and two-input control narrow the diagnosis

Exhaustive1999segment comparison on ten declared profiles passes current-window
initial-field replay<=3e-7. Six bank phase differences from global SDF.14..1.28%,
vertical-gap versions1.01..4.15%; do not call existing33segment search exact SDF
or silently regenerate physics. Two-fixed-input uniformNt4/2000Adam exits0 in
220.2s: same source0phi12.547%/7.240% vs fittedsingle2.483%/3.387% and prior16case
21.538%/15.888%. This is a multi-condition fitting issue under current schemes,
not evidence that layered sampling is necessary. No isolated count effect is
claimed across B/exposure changes. Frozen final batch0 AC/CH individual gradient
mean retains98.5/98.6% of mean norms (early85.8/89.3%); high positive inter-input
cosines contradict strong cancellation on these probes. Finish condition-path
sensitivity/head audits before attributing the collapse to architecture or optimizer.
Relation: electropolish-operator-training, measured intermediate evidence.

## 2026-10-28 — Local-IC query shortcut requires a trained ablation

Frozen online16/17 image-only swaps change phase by~.005%of true input difference
at0/.0276%at.25, versus local-IC-only104.38%/1.152%. Two-trained-input model
also largely ignores images (.197%at0/.224%at.25), fits IC via local features.
Encoder gradients are nonzero; no stopped-AD/frozen-parameter bug identified.
This is a concrete representation/training shortcut hypothesis, not a proven
cause from frozen probes alone. Add opt-in arch.local_input=zero to zero ONLY
query-local feature channels, preserving all true IC targets, image data,
model parameter leaves/init and physical losses. Explicit initial mode retains
current strategy; old saved architecture without this optional field retains
legacy initial mode. One matched two-input/B2/2000Adam/uniform-time cold start
tests this single change; no SDF/reference/sampling/default strategy change.
Focused supervision-preservation/inference test before training, and replay
must use saved option. Relation: electropolish-operator-training diagnostic ablation.

## 2026-10-28 — Removing local-query IC shortcut improves matched two-input fitting

Zero-local two-input/B2/2000Adam/uniform-time run exits0 in254.04s. Mean fitted
phi pooled/terminal9.470%/5.920% ->2.601%/2.704%, with true IC supervision and
all physics/sampling/initial arrays retained. This supports the shortcut
hypothesis, but four heldout phi24.375%/15.808% remain poor after only two
training inputs; do not extrapolate fitting to operator generalization.
Prescribed finite-window SDF is unchanged; documentation now says approximation
explicitly. Launch one matched continuous-onlineB8/5000Adam zero-local control
on the ORIGINAL40000profile stream to validate the same intervention at operator
scale, using canonical train.py/online config and explicit arch.local_input=zero.
No causal/stratification/initial hard constraint/optimizer/default promotion,
reference regeneration or Git change. Relation: electropolish-operator-training,
positive diagnostic ablation extended to matched online validation.

## 2026-10-28 — Same-stream online image-only conditioning improves physical accuracy

UniformNt4/onlineB8/5000Adam zero-local control exits0 in315.83s. Four unseen
mean phi pooled/terminal17.748%/12.355% ->5.573%/4.372%; c19.317%/14.029% ->
6.452%/5.124%. No causal, stratification, hard IC, modified SDF/PDE/reference,
optimizer change or extra budget. Fitted pair0 image-only swap response is98.7%
of true initial difference and92.5%at.25, versus original~.2% image response;
two-input fit reproduces single-PINN-like accuracy after removing local shortcut.
This controlled intervention supports a conditional-training shortcut as a
major contributor, without proving universality or seed stability. Default
initial mode stays unchanged as predeclared; zero is explicit reproducible
best measured online control. Finish same73arrays/40000profiles/config/source/
field/transport audit and archive. Relation: electropolish-operator-training.

## 2026-10-28 — Conditional training/IC diagnosis completed with verified improvement

All three new controls and frozen/audit workers exit0; exact73initial arrays/
40000online profiles/config delta, source executable parity and six final GPU
field replays<=2.33e-6 pass. Independent Dq at unseen16 improves80.70->17.79%;
phase input variation at.25 recovers80.36%of reference. Two supervision/legacy
tests pass, exhaustive IC check and within1%separate/B8 gradient check pass;
final syntax/strictJSON/finite-field/PNG/links/index/current-source/diff checks
pass. Preserve the successful zero-local option, original initial default and
all old runs; archive task/results/configs/figures and mark activeNone. Explain
windowed signed-distance approximation candidly; no initial/reference numerics
changed. No causal/layered sampling/extra budget/Git history action or queued
optimization; all workers/GPU idle. Best measured online meanphi5.573/4.372%
remains a one-pair result with harder-case and seed-stability limits. Relation:
electropolish-operator-training COMPLETE.

## 2026-10-28 — Freeze operator before expanding to64 independent test interfaces

User requests training cost, a larger64or100 test set, and local AD comparison
context. Choose64 fresh profiles, predeclare identities before solving, retain
the six-mode continuous marginal law conditioned on the same2000-vertex height
acceptance as online training. Existing seed-defined IC is preserved; no new
distance formula. Freeze final image-only checkpoint, no test-based selection.
Validate all references with original coarse/fine tolerances, report per-input
and aggregate errors/dispersion. Four CPU processes, one BLAS thread each,
separate direct-path dataset. Current operator trainer has no local AD branch;
explain encode-once feasibility and unmeasured derivative/parameter-gradient
cost rather than asserting a speed ratio or ordinary PINN timing transfer.
No local full training, defaults, old artifacts or Git history changes.
Relation: electropolish-operator-test64.

## 2026-10-28 — Bound the local operator cost comparison to an explicit benchmark

While64references solve, measure nonlocal versus coordinate-AD physical PDE
loss plus parameter gradient at the same B8/Nt4/free-core queries. Share the
encoder inside parameter AD, compute coordinate Hessian traces with directional
JVPs (not a dense cross-query Jacobian), and omit nonlocal halo for local AD.
Check directional derivatives against single-query Hessians. Report compilation
and synchronized warm timings separately; no IC/BC/optimizer/adaptive-balance or
full local-training accuracy/speed claim. This sharpens the user's cost question
without introducing a production local PDE/BC/reference branch.
Relation: electropolish-operator-test64.

## 2026-10-28 — Larger frozen test passes; local component cost is comparable

All64fresh profiles/reference refinements pass (maximum4.60e-8). Meanphi
pooled/terminal4.506/3.502%, p906.909/5.850%, worst10.541/8.210%; meanc5.323/4.163%.
Keep this separate from old four-input mean5.573/4.372%; same frozen final
checkpoint, no training gain/seed-stability/new distribution claim. References
797s/fourworkers, cold evaluation26s(~24sJIT), warm complete trajectory2.23ms.
Existing training5000Adam/B8/315.8s is unchanged; large bank is final-test only.
Directional coordinate-Hessian versus full single-query check passes3.30e-16;
PDE value+parameter-gradient warm medians NL7.62ms/AD6.99ms, same245freecores,
2453halo versus245local queries. No clear AD slowdown; halo/AD costs can offset.
Nonlocal warm timing overlaps CPU solvers; local follows their exit, and its
compile wall includes an explicit temporary process pause. Retain raw evidence;
do not rank an8%difference or compare pure compile costs/full training/accuracy.
No production local branch, new BC/reference formulation, default promotion,
checkpoint modification or Git action. Archive validated probes/results;
dashboard activeNone and all workers idle. Relation: electropolish-operator-test64 COMPLETE.

## 2026-10-28 — Preserve incompatible rough initial fields in full operator comparison

User explicitly retains physically meaningful endpoint slopes and authorizes
the original-case local/nonlocal cold starts. Do not redesign ICs for a cleaner
comparison. Local parabolic evolution has positive-time Neumann closure and
an initial boundary adjustment; impose no Neumann loss at t0. Add an opt-in
local_ad operator path, same image-only model/initial arrays,40000profiles,
times/core/IC draws,Adam5000/B8. Explicit sixth normalized phi_n/q_n loss closes
side walls; nonlocal keeps five terms and pair closure. Report this necessary
objective difference. Use final checkpoints only, common64graph targets and
independent local FD101targets with original time tolerances; predeclare space
refinement/extra-early indices0,21,42,63 at201/401 grids. Do not silently call
finite-horizon pair boundaries classical Neumann or infer stable ordering from
one pair. Serial GPU timing, CPU SciPy reference work only, no IC/reference
replacement/default promotion/optimizer sweep/Git operation.
Relation: electropolish-operator-local-comparison.

## 2026-10-29 — Separate phase invariance from concentration reference diagnostics

Original localFD101test42 reaches c1.000019071 and fails the inherited global
0..1range heuristic;63other101and3of4FD201complete. Total c has no unit-interval
maximum principle: at c=1 on an interior patch and a maximum of h(phi),
c_t=-N_CH*dc*Laplacian h(phi)>0. Independently check this counterexample against
the physical RHS. New operator_reference validates finite fields and AC phase
range, records concentration extrema without clipping or an invented c cap.
This matches the already documented unrestricted linear-liquid operator map;
equations/IC/BC/time-refinement tolerances are unchanged. Original graph targets,
legacy solver and loaders are untouched. Retain failed raw records and all
validated partial references; an explicit --resume mode will check and reuse
them, rather than silently rerunning or skipping failed inputs.
Relation: electropolish-operator-local-comparison reference validation.

## 2026-10-29 — Interpret matched operator timing and boundaries separately

Both original-IC5000Adam/B8cold starts complete with exact73initial arrays,
40000profiles and time/core/IC/BC samples. Nonlocal/local process368.35/693.03s;
first update223.35/533.87s, warm mean16.71/19.35ms. Report compilation separately
instead of transferring the1.88x cold-start ratio to steady updates. Reference
workers were stopped throughout both training timings, then resumed. Common
localFD10164meanphi pooled4.427/8.595%, terminal3.176/5.498%; common graph gives
the same ranking. Local normal RMS is smaller despite worse early side fields,
so do not call pair closure exact classical Neumann or infer a unique boundary
mechanism/stable seed ordering. Keep final checkpoints/no test selection and
existing defaults. Last FD401/interface42 reference was interrupted during the
service interruption; explicit --resume validates75outputs and fills only the
missing one. Earlier automatic approval review failed due to exhausted usage,
not an unsafe-action finding; resumed user authorization permits retry after
tools are available. Relation: electropolish-operator-local-comparison.

## 2026-10-29 — Complete matched operator comparison with refined common targets

Last FD401/interface42 solve completes on explicit resume; all76coarse/fine time
checks pass, maximum2.583e-8. On declared4inputs, FD201->401pooledphi differences
.068..076%; againstFD401 mean NL/localphi5.065/8.773%, terminal2.945/4.886%,
early side3.829/10.336%. Thus commonlocal64ranking survives refinement on this
subset; do not call all64continuum-converged or extrapolate training seeds.
Nativegraph/FD401pooledphi.56..1.01% remains a finite-horizon discrepancy.
All64/refined4field norms/reference gaps/aggregates/normals/input/source checks
and76reference BC/range/time gates independently pass; total c max1.00031175
is recorded unmodified, consistent with its no-maximum-principle derivation.
Retain optional local_ad/six-loss control and existing default settings. Archive
62new task artifacts without overwriting old records, update README/DEVELOPMENT/
artifact index and mark activeNone. CPU/GPU workers complete; no queued work,
new optimizer sweep, IC flattening, checkpoint selection or Git history action.
Relation: electropolish-operator-local-comparison COMPLETE.

## 2026-10-29 — User authorizes publication of accepted current results

Commit/push is explicitly authorized in this turn. Preserve the coherent
accumulated confined-CH/electropolish implementation and archived results since
dev728e5b9, including documentation and agent navigation updates. Stage only
repository files; external artifacts keep direct paths. Inspect syntax/strict
records/whitespace and existing numerical-verification evidence; do not rerun
unchanged expensive studies. Use dev commit, regular merge into main, push
origin main only, then verify remote equality and clean worktree. Reject force
push, history rewriting, dev publication or default retuning as unnecessary.
Relation: phasefield-results-publication.



## 2026-10-29 — Normalize publication whitespace without changing results

Staged inspection covers823files and reveals trailing spaces in generatedSVG/
stdout plus one extra PythonEOF line. Normalize20files only in whitespace;
independent PythonAST/SVGtree/attribute and whitespace-free text equality pass.
Numerical source/field values and previous verification evidence are retained.
155Python syntax/260strictJSON/100JSONL format checks pass. Archive publication
preparation record with branch history as the authoritative delivery receipt;
verify main remote refs after the authorized regular merge and push. No extra
training or unchanged expensive test suite is warranted by whitespace cleanup.
Relation: phasefield-results-publication.

## 2026-10-29 — Start 3D random-ellipsoid melting operator

User requests online random initial melting shapes and3Dconditional encoder
with memory validation. Retain single-IVP defaults/physical PDE and validated
IC-only latent residual; input volumes condition the encoder, spatial work is
shared across all times. Begin semiaxes.24..36, centers.45..55, Haar rotation,
1x32^3input/4^3patches/512tokens/width48, core256/Nt8. Measure B8/B16training
peaks explicitly before selecting batch. Use Euclidean signed distance, checked
Lagrange-root/axis degeneracies against Eberly/SciPy; reject implicit radial
distance mislabeled as SDF. Rotated/shifted ellipsoids require full-cube direct
references, not sphere analytic motion or octant symmetry. Preserve old2D
parameter leaves/outputs and external artifact paths. Local dev only; no new
commit/push authorization. Relation: icemelting-random-operator.

## 2026-10-29 — Stabilize exact ellipsoid distance at near-axis queries

Independent geometry checks expose a fixed-bisection conditioning issue when
rotation leaves a smallest-axis coordinate~machine epsilon. A tiny Lagrange
root cannot be resolved relatively by56iterations; direct q_min/root then gives
large projection error. Reconstruct the minimal-axis projection norm from the
ellipsoid surface constraint for the small-root branch, an equivalent identity,
without a geometric distance floor or radial surrogate. Unchanged sphere/
multi-start SciPy closest-point/rotation tolerances now pass.3Dtime/parameter
AD and independently derived graph stencil tests also pass. Retain failed raw
records and original single-IVP arithmetic. Relation: icemelting-random-operator.

## 2026-10-29 — Choose B16 after measured volumetric training memory

Both8-update probes use actual128^3graph,512input tokens,1788core/halo queries,
Nt8, full parameter backward and Adam. Peakactive/pool B8 .625/2GiB, B16
1.331/4GiB;7warm update means14.811/21.759ms, sampling2.567/3.294ms. Select
B16for this new opt-in operator, keep single-IVP default. Do not equate active
allocation to driver usage or short timings to long-run benchmarks. B8CLI
initially exited1because a returned dict reached absl; corrected call/None
return and B16canonical command exits0. Legacy2Darray/output/checkpoint and
FP32SDFchecks pass. Prepare10full-cube held-out references on CPU before fresh
GPU training, no sphere/octant target or future-field labels. Relation:
icemelting-random-operator.

## 2026-10-29 — Complete 3D random-ellipsoid operator pilot

Canonical fresh5000Adam/B16/Nt8run exits0 in300.57s full process;80000distinct
online ellipsoids, no future-field labels/test checkpoint selection. Peakactive
1.332GiB/pool4GiB, warm update22.43ms+sampling3.41ms. All10full128^3native
reference time gates<=1.422e-7;8testmeanphi pooled/terminal.2266/.0922%, solid
.6425/4.5017%, terminal solid-fraction error1.2133%. Report24^3diagnostic node
errors and128^2slice checks separately from reference integration grid; do not
claim full-grid/continuum/seed stability. Maximum late solidL210.59% remains a
concrete limit. Preserve original single-sphere defaults/checkpoints; selected
B16is only the new opt-in NO preset. Four new numerical tests, two2Dregressions,
legacy array/output/checkpoint, source/stream/reference/field/aggregate and
fresh GPUreplays pass. Archive scripts/raw outcomes/figures/direct artifacts;
README/DEVELOPMENT/index/memory updated, workers idle, no Git history action.
Relation: icemelting-random-operator COMPLETE.

## 2026-10-29 — Archive verified 3D melting operator delivery

All final AST/strict-record/whitespace/figure/link/direct-path/current-source
checks pass;38task artifacts archived without overwriting old outputs. Single
failed-test stdout trailing spaces normalized only, preserving every nonspace
character. Project dashboard activeNone; GPUidle1MiB/0%, no worker or queued
training. Preserve the successful first prototype and explicit late-solid/
diagnostic-grid limits rather than adding an unrequested optimizer sweep.
Relation: icemelting-random-operator COMPLETE, no Git history action.

## 2026-10-29 — Show randomly selected held-out melting trajectories

User requests several random results. Draw three uniformly without replacement
from the eight declared test inputs before opening scores; retain IDs5/9/2
and reuse saved final-checkpoint predictions. Prefer common phase/error color
scales and distinguish central128^2slice pictures from24^3diagnostic L2.
Replay original reference equality and independently calculate phase/solid
metrics; near-disappearance background-dominated phase L2 requires the solid
metric as context. Reject error-based selection, retraining and new checkpoint
selection for this visualization request. All three plots inspected; archive
selection/script/raw metrics and completed task. No Git history action.
Relation: melting-operator-random-samples COMPLETE.

## 2026-10-29 — Run a matched SOAP melting neural-operator control

User requests SOAP after clarifying the successful3Doperator used Adam.
Change only optimizer and output run name via canonical train.py flags; retain
5000updates/B16/lr1e-3schedule/seed42/ordinaryFP32/core+halo/hardIC/latent PDE
and the eight predeclared native-graph tests. Reuse the existing upstream
SOAP_JAX factory, default b1.9/b2.99/refresh5/clip100. Verify actual initial
arrays, shape/time streams and graph/source parity; compare physical phase and
late solid errors plus cold timing/memory. Reject test checkpoint selection,
optimizer retuning, reference regeneration and default promotion before
evidence. No commit/push authorized. Relation: melting-operator-soap.

## 2026-10-29 — Retain Adam after mixed SOAP melting control

SOAP5000/B16cold start completes normally in492.06s versus Adam300.57s.
All79initial arrays,24source files,80000actual shapes/40000time draws match;
config differs only optimizer/run name. Eight native-reference testmeanphi
pooled.2266%->.0785%, terminal.0922%->.1291%; terminalsolid4.502%->6.504%,
solid-fraction1.213%->4.016%. Early/intermediate accuracy improves, terminal
small-solid accuracy and speed worsen. Keep Adam opt-in operator preset
default and SOAP selectable through train.py; reject promotion based solely
on pooled phi or retrospective test checkpoint selection. Independent metrics/
reference identity, initial/source/stream audits and fresh GPUreplay<=9.54e-7
pass. One pair does not establish seed stability; deferred optimizer issue
remains deferred. Archive results/curve/scripts/task, update README/index,
clear dashboard; no numerical source or Git history action.
Relation: melting-operator-soap COMPLETE.

## 2026-10-29 — Treat local PDE as the melting operator accuracy target

User requests local FEM evaluation of existing checkpoints. Compare total
local-reference errors, retaining native-reference fitting errors and the
native/local gap as distinct diagnostics; their scalar norms need not add.
Start with previously randomly chosen held-out IDs5/9/2, not error-selected
inputs. Old sphere FEM derives epsilon from mesh.N, so do not silently reuse
it for fixed-physics ellipsoid refinement. Add an explicit independent
full-cube local weak-form solver with epsilon=.0225, same exact Euclidean IC,
natural zero-normal flux, grid/time checks. No retraining, changed operator,
checkpoint selection or speed-advantage claim from this evaluation. Preserve
old reference paths, mutually exclusive FEniCS/JAXGPU environments and Git
history. Relation: melting-operator-local-fem.

## 2026-10-29 — Rewrite canonical melting FEM from fixed PINO metadata

User explicitly rejects the old FEM implementation and requests a rewrite
with physical/interface parameters aligned to PINN/PINO. Replace solver.py
and solver_base.py with explicit saved-config/manifest loading, fixed epsilon,
full-cube consistent CG2 tetrahedral FEM and implicit-midpoint polynomial
reaction, natural homogeneous Neumann on all faces. Weak diffusion sign is
independently checked by a zero-normal polynomial; reaction by constants,
MPI query collection by a quadratic field, IC by existing saved arrays.
Use CG/Jacobi Newton solves with explicit tight tolerances, degree8quadrature,
no mass lumping or operator import. Move only JAX imports into JAX-specific
ellipsoid helpers so FEniCS can use the identical NumPy SDF without importing
JAX; preserve all numerical arithmetic. Predeclare spatial/time gates5e-4phi
and1e-2solid, and mesh levels24/32/48/64 before examining checkpoint scores.
Legacy solver source remains in historical run snapshots/Git; no old data is
overwritten. Relation: melting-operator-local-fem.

## 2026-10-29 — Use constant-matrix IMEX BDF2 for economical local references

The midpointN32pilot needs379s/200steps; cubic-mass/Jacobian reassembly is
unnecessary for this mild reaction time scale. Retain it as an independent
time-integrator control, and add standard SBDF2: implicit local diffusion,
explicit2G(phi_n)-G(phi_nm1), consistent CG2 mass and exact polynomial reaction
quadrature. BE/explicit-reaction startup supplies O(dt^2)initial error; verify
second-order temporal convergence independently on the homogeneous ODE. Reuse
constant mass/stiffness matrices, never lump the mass or change epsilon/PDE.
Primary source: Ascher/Ruuth/Wetton1995, doi10.1137/0732037, which analyzes
IMEX multistep reaction-diffusion methods and favors BDF damping over CNAB
in its reported regimes. Add explicit time.scheme=imex_bdf2; preserve midpoint
as a named option. All fixed spatial/time gates remain unchanged. Hardware
CPUquota12requires sequential12-rank solves despite96advertised cores.
Relation: melting-operator-local-fem.

## 2026-10-29 — User narrows local FEM evaluation to one input

After measured96-mesh719万DOF cost, user explicitly selects completing sample5
before expanding to3. Defer9/2; preserve all physical/refinement gates. Pause
only the orchestration parent viaSIGSTOP while its independent active MPI
dt.0025child continues, avoiding cancellation/recomputation and automatic
launch of deferred cases. Once child ends, terminate stopped parent and rerun
controller with explicit indices=(5,), reusing validated cached outputs. No
checkpoint/model/default or Git action. Relation: melting-operator-local-fem.


## 2026-10-29 — Extend explicit local FEM time refinement if needed

N96 dt.005->.0025 fails the unchanged phase/solid gates (max.2523%/1.7617%).
The independent homogeneous-ODE test establishes second-order SBDF2; another
single halving need not meet the.05% phase threshold. Before seeing dt.00125
results, extend the explicit list to.000625 as a conditional next level. Run
it only if the existing adjacent-step gate still fails. Retain all earlier
outputs/failures, one sample5 and one12-rank job at a time. Reject loosening
thresholds or treating a Richardson estimate as the declared adjacent-step
check. No checkpoint/training/default change. Relation: melting-operator-local-fem.


## 2026-10-29 — Accept current FEM accuracy for the reference-gap question

User explicitly accepts N96/dt.00125 results and says not to over-refine. They
clarify the primary question is native nonlocal versus local FEM reference
agreement, to decide whether operator optimization is necessary. Stop only the
unneeded dt.000625job and its pipeline before JAX stages; preserve cancellation
receipt and partial log. Use status=accepted_by_user, strict_refinement_passed
=false: original.05% slice gate fails at.0737336%, while volumephi.0294622%
and solid.547736% pass. Spatial maxphase.0377924%/solid.2136203%. Keep these
uncertainties explicit, do not relabel strict checks as passed. Primary report
is reference-gap by time/phase/solid/interface, model fits auxiliary; no operator
change or further FEM refinement. Relation: melting-operator-local-fem.


## 2026-10-29 — Retain native melting reference for current whole-field goal

User clarifies that late tiny-solid accuracy is not the current objective.
One-input native/local FEM pooled/terminalphi differences.1474/.1670% (slice
.3575/.5351%) support native reference for routine monitoring. Keep independent
local validation for finer model comparisons: fresh Adam pooledphi native/local
.2383/.1479%, SOAP.0709/.1861%, with error-field cancellation changing ranking.
Do not equate graph fitting error to local accuracy or scalar-add norms. No
operator optimization, training/default change or extra sample/refinement queued.
Archive accepted precision/raw failures/source/replay/fields; close task and
clear dashboard. Relation: melting-operator-local-fem COMPLETE; no Git action.


## 2026-10-29 — Render actual melting checkpoint isosurfaces

User requests three-dimensional rendered shrinking ice instead of slices. Reuse
previous pre-score random IDs5/9/2 and unchanged Adam checkpoint, evaluating
actual physical phi on a dense regular XYZgrid at intermediate times. Extract
phi=0 interfaces; do not animate a prescribed ellipsoid scale or add geometry
not predicted by the model. Keep fixed cameras/coordinate scales across time,
record mesh/time/source identity and render-only resolution. Use an isolated
optional rendering environment because training.venv lacks VTK/marching-cubes/
video tools; do not change project dependencies, physics or checkpoints. No new
FEM/training/Git history action. Relation: melting-operator-3d-render.


## 2026-10-29 — Switch melting display to publication-style scientific figures

User rejects the decorative ice-material rendering and requests paper-suitable
3Dplots. Keep actual predicted isosurfaces, replace specular material with matte
scientific shading, fixed orthographic axes and thin silhouettes; also provide
time-colored contour overlays. A display-only cutaway can reveal nested times
and must be labeled as a visualization crop, not changed physical geometry.
Export high-resolution PNG and PDF/SVG with vector labels/curves; dense filled
surfaces may be rasterized and must not be called fully vector. Reuse verified
meshes, no neural/physics/model changes. Relation: melting-operator-3d-render.


## 2026-10-29 — Close melting publication visualization without numerical changes

Actual96^3/61time checkpoint fields produce183closed interfaces; all three
geometric volumes decrease. Replay/IC/chunk and meshphi checks pass. User's
paper preference selects matte fixed-scale snapshots and vector3Dcontours;
retain explicitly labeled display-cutaway and decorative animation as alternatives.
Correct framebuffer refresh and per-triangle depth sorting, with draft issues
recorded; geometry/fields unchanged. Archive scripts/provenance/formats/media,
register direct data paths and clear active task. No new solver/reference,
training/default/seed or Git history action. Relation: melting-operator-3d-render
COMPLETE.


## 2026-10-29 — Publish validated NO presets and stop versioning generated outputs

User authorizes commit/push and requests SOTA defaults plus necessary-only Git
content. Electropolish generic operator preset still points to early finite-input
pilot; promote canonical operator preset to saved validated online image-only
AdamB8/5000, preserve historical preset semantics through explicit legacy base.
Melting retains AdamB16/5000 based on accuracy/cost/local-FEM compromise rather
than native pooled error alone. Keep canonical train.py single-IVP mode defaults.
Untrack generated docs/tasks media/arrays/checkpoints/HLO with local files and
external copies preserved; retain code/configs/metrics/provenance and restore
manifest. No history rewrite or forcepush: existing historical blobs remain.
Follow devcommit -> regular mainmerge -> push origin/main only. Relation:
operator-sota-publication.


## 2026-10-29 — Complete verified-default and artifact publication preparation

Exact saved-run comparison confirms both canonical NO strategies, allowing only
run-name and runtime seed provenance. Preserve five historical presets rather
than changing their meaning.27focused tests pass; no new training/FEM needed.
Current Git tree drops from90.00 to18.74MiB by removing only generated outputs
from tracking. All275original local files and byte-verified external backups
remain; fresh-copy restoration passes. Retain scripts/configs/compact metrics,
no new binary files or history rewrite. Archive publication task and audits,
clear dashboard, then perform the authorized devcommit/regularmainmerge/mainpush;
Git branch and remote equality provide the final receipt. Relation:
operator-sota-publication COMPLETE for implementation and verification.


## 2026-10-29 — Recommend velocity-conditioned phase transport operator

For the next NO discussion, prefer fixed IC with random low-frequency spatial
streamfunction velocity structure, complementing two existing IC-conditioned
operators; random IC/joint conditioning remains a later extension. This is a
recommendation, not user approval of a numerical distribution. Retain a known
time envelope, exact divergence-free construction, current periodic FFT flux,
full-grid mass/IC constraints and mixed PDE losses. Reject amplitude-only sweeps
as the main intended demonstration and do not call prescribed CH transport a
coupled NSCH solution. Check peak deformation as well as final fields; retain
independent local/native reference separation. No implementation, training,
default change or Git authorization in this turn. Relation:
twophase-operator-design COMPLETE; archived recommendation and next steps.


## 2026-10-29 — Give random flow inputs a controlled vortex encounter meaning

User wants physical interpretation for the velocity distribution. Recommend a
counter-rotating finite-core vortex pair with circulation/core/separation/offset
randomness, motivated by vortex-drop encounter studies. Use periodic unit-mass
Gaussian cores and equal opposite circulation for Poisson compatibility; FFT
is a representation, not the physical prior. Distinguish externally maintained
prescribed stirring from freely evolving NS vortices and two-way drop dynamics.
Four-roll mill remains a physical alternative with explicit wall/device mismatch
to current periodic geometry. No numerical bounds frozen or model implemented.
Archive only lightweight schematic script/receipts plus ignored local PNG;
NumPy divergence/zero-mean checks pass. Relation:
twophase-physical-flow-design COMPLETE; no training/default/Git action.


## 2026-10-29 — Anchor two-phase scope to the supplied PINN manuscript

Direct re-read of Qiu2022attachmentp12confirms reversed vortex solves onlyCH;
p8/p9andp15-20confirm rising bubble solves NS+continuity+CH with capillary and
material coupling, four network outputs and sequential windows. Current code
implements only prescribed-flow scope. Recommend full NSCH for the user's
coupled two-phase NO objective, with random vortex conditions interpreted as
initial velocity or forcing. Equal-property2Dperiodic model is a proposed first
step, not a silent default change; future variable-density formulation and
reference choices must be explicit. Correct overly broad use of two-phase-flow
for the transport pilot. Archive page/visual verification; no implementation,
training or Git action. Relation: twophase-nsch-scope COMPLETE.


## 2026-10-29 — Design irregular-domain conditioning; defer NSCH

User defers coupled two-phase work and returns to confined CH initial-field
encoding. Fixed geometry already supplies3682triangle-centroid samples/areas.
Recommend shared point lifting into a small area-aware latent cross-attention
encoder, preserving spatial/time splitting; regular image patches plus occupancy
mask are a valid simpler control. No need to force a rectangular physical domain
or treat hole filling as phase. Volume measure belongs in aggregation/mass;
geometry/general-time convergence and local Neumann behavior require separate
validation. Existing30-window difficulty is not resolved merely by an encoder.
Archive recommendation/primary references; no implementation, default, training
or Git action. Relation: confined-ch-operator-encoding COMPLETE.


## 2026-10-29 — Start simplified geometry-conditioned CH operator

User authorizes randomizing two hole positions/radii while keeping the outer
ellipse and initial function fixed, reducing coarsening difficulty. Start an
opt-in finite random geometry-bank pilot with Gmsh physical triangle-centroid
quadrature, own visible regional operator and fixed-shape padding per domain.
Boundary point encoder supplies geometry, with shared spatial/time decoding.
Keep the same ambient IC restricted to each domain: do not multiply by moving
hole envelopes and call it fixed. A local initial Neumann mismatch is explicit.
Native reference refinement and representative local FEM checks precede claims.
No evolved labels in training, no old default/history change. Relation:
confined-ch-geometry-operator ACTIVE; NSCH deferred.


## 2026-10-29 — Test direct boundary tokens after latent geometry collapse

First5000Adam geometry pilot reaches heldoutphi3.272%/4.326% pooled/terminal,
mu8.228%/10.287%,259.34s; however swapping encoded geometry changes phierror
by almost nothing. Do not call this successful geometry dependence. Test a
simpler direct boundary-token encoder (no32-token compression), preserving
arc-measure attention, physicalbank/IC/PDE/metric/budget/sampling/optimizer.
Keep the first run and source snapshot. This is an architecture control, not
an isolated optimizer or general convergence claim. Relation:
confined-ch-geometry-operator ACTIVE.


## 2026-10-29 — Check geometry sensitivity before accepting scalar operator error

User observes nearly identical final fields and authorizes a somewhat harder
coarsening control. Direct boundary tokens improve native heldoutphi2.855/3.709%
and wrong-geometry2.855->3.425% (7/8worse), but this alone does not establish
strong conditional fidelity. Diagnose reference/model variation at identical
physical queries, with common-domain support and no cross-hole interpolation.
Keep the mild run; extend only T.2->.75 with same initial function,24geometries,
mesh/kernel/epsilon/optimizer/budget first. Off-grid neural queries must retain
the original volume-weighted mass correction, not recompute it on probe points.
Quantify mean-field versus geometry-response errors separately. Relation:
confined-ch-geometry-operator ACTIVE; harder extension authorized.

## 2026-10-29 — Prefer general heterogeneous point conditioning over distance bias

User rejects an explicit query-to-wall distance attention bias. No such patch
landed. Test a GNOT-inspired heterogeneous normalized integral-attention model:
separate boundary and domain-initial-field streams, repeated conditional reads,
shared coordinate Fourier features, and the existing separable time decoder.
Use GNOT equations3-8 with physical quadrature in the normalized sums, checked
against an explicit pairwise implementation. This is not a full GNOT reproduction:
omit query-set self-attention/MoE so arbitrary query chunks remain independent.
Do not infer that boundary-only encoding lacks sufficient information, or that
the earlier small latent control rules out Perceiver IO. Keep PDE, IC/mass,
bank, optimizer, samples and budget fixed within the longerT.75 comparison.
Report this as an architecture bundle, not an isolated cause or guaranteed gain.
Longer local FEM checks16/17 passed; native/local phase discrepancy~4.8% pooled,
~5.7% terminal remains separate. Relation: confined-ch-geometry-operator ACTIVE.

## 2026-10-29 — Retain the general point-stream control after matched cold starts

Two5000Adam/B4 longerT.75runs match all nonarchitecture configuration and20,000
actual geometry draws. GNOT-inspired heterogeneous streams reduce native heldout
phase5.293/6.220%->2.380/2.771% pooled/terminal; common-probe response error
84.42%->42.65%, amplitude43.54%->94.74% and correlation.548->.905. This supports
improved conditioning for this development cohort without explicit distance bias;
it does not isolate the cause among stream/position/depth/width changes. CPU
replays of all24fields agree with GPU saves within1.9e-15. Local FEM meanphi
improves6.886/7.907%->5.720/6.704%, so native-reference gains are not continuum
accuracy claims. Cold314/407s and warm21.46/36.29ms trade speed for accuracy.
Keep the stronger generic model as an explicit research preset, preserve the
mild pilot and windowed single-IVP defaults; no user-requested default/history
change here. Further geometry/mesh/seed generalization and kernel refinement
are separate experiments. Relation: confined-ch-geometry-operator COMPLETE.

## 2026-10-29 — Increase physical geometry response without tailored attention

User authorizes modest mesh refinement, richer fixed IC and broader random hole
distribution plus smooth plots. Add a separate h.035/eight-mode preset, same
beta1/M.02/epsilon.06/T.75 and generic model/Adam5000/B4. Both hole centers share
a broad box with independent radii. Keep clearance and rejection validation;
no hole-dependent IC or forced future shape. Frequencies are checked against
linearized local CH growth M*k^2*(beta-epsilon^2*k^2), not increased blindly.
Independent new native/local references and common-probe variation precede
training. For display, area-weight neighboring physical cells to mesh vertices
then interpolate linearly inside original triangles; mask holes. Metrics use
unmodified centroid fields. No cubic overshoot or smoothing-as-accuracy claim,
no comparison of different physical cases as an isolated method gain. Relation:
confined-ch-geometry-richer ACTIVE; old artifacts/defaults retained.

## 2026-10-29 — Test PDE-consistent geometry augmentation for the broader prior

Richer-case5000run fits16training geometries at2.016% but validates at9.184%,
so broader-prior generalization remains weak. Test axis-reflection augmentation
without changing architecture/physics/budget/base draws. Axis-aligned ellipse,
isotropic pair distances, segment visibility and physical volumes are reflection
invariant; the new broad prior boxes are invariant too. Reuse A/resolvent under
the same point indexing, reflect coordinates/boundary normals, and RECOMPUTE the
same fixed ambient IC at reflected coordinates. Do not reflect old phi0/future
solutions: IC is not assumed symmetric. Check independent matrix covariance and
initial identity before training. Draw two reflection bits per update/condition,
record actual bits, keep original8validation domains untouched and check their
exclusion from all augmented inputs. This is symmetry-based data augmentation,
not geometry-specific attention or proof that finite16coverage is the sole cause.
Relation: confined-ch-geometry-richer ACTIVE; preserve first run and references.

## 2026-10-29 — Separate training-operator precomputation from reference solves

User clarifies512precomputed random training geometries are acceptable and asks
only about constructing their operators. The earlier3.4hour extrapolation mixed
full native trajectory/refinement work into training preprocessing; it is not
an estimate of512operator-only inputs. Reflection control is not launched.
Benchmark mesh/quadrature, production visible-pair assembly, existing dense
residual metric and storage separately, with no time integration/FEM solve.
Only fixed monitoring/validation inputs need future-field references. Preserve
the existing8validation geometries when expanding training; no labels or input
overlap. Online construction is a performance choice, not mathematically
impossible. Relation: confined-ch-geometry-richer ACTIVE,512bank work authorized.

## 2026-10-29 — Measure operator-only cost before expanding the geometry bank

Three current~1160-1180node inputs, CPU/OpenBLAS1/OMP1: mesh/quadrature.518s,
production pair assembly.0588s, dense residual metric.209s, compressed storage
.537s; total1.323s/input.512serial extrapolation677s (~11.3min), not an actual
512measurement. Three independent ordered-assembly probes agree; no PDE/FEM
solve performed. Independent reference assembler costs2.35s/input separately
and is not necessary for every production-only training input. Meshing runs in
separate fenicsenv interpreter because .venv lacks meshio; initial failed
environment probe is retained in /tmp. Online construction is feasible but slow
relative to~43ms GPU updates, hence caching is pragmatic. No512future reference
generation is needed. Relation: confined-ch-geometry-richer ACTIVE.

## 2026-10-29 — Parallel operator-only bank with unchanged held-out domains

Use six spawned CPU workers (one BLAS/OMP thread each) for512training domains;
construct only mesh/quadrature, the unchanged production visible-pair operator
and existing dense biharmonic residual metric. No evolution integration or FEM
solve for new training inputs. Reuse the16existing labeled training monitors;
reserve original accepted draws16..23 as the same eight validation geometries,
reindex them512..519, and use draws0..15,24..519 for training. Reuse their exact
validated fields/local FEM files after physical/config/geometry checks. New
explicit operator-only bank format records label availability and monitor IDs;
legacy all-labeled banks/checkpoints remain loadable. Reference arrays never
enter the optimizer payload. Keep the same5000Adam/B4model/budget and compare
on the identical development validation domains; larger coverage is a hypothesis,
not a guaranteed generalization gain. Time complete preparation and training
separately; five minutes is a target, not an assumed result. Relation:
confined-ch-geometry-richer ACTIVE, user authorizes parallel generation/training.

## 2026-10-29 — Parallel preparation meets the wall-time target

Six spawned CPU workers complete the expanded512training/8validation bank in
124.703s (496newoperators,24reused labeled monitors/validation files), well below
the requested~5min target. Three new-domain actions agree with the independent
ordered assembler within1.82e-12. Reuse the exact old references/local FEM;
no training trajectories are integrated. Keep compressed cache storage (~6.11GiB)
and full padded device bank for this run; first training setup takes120.69s,
so preprocessing and cache-loading/compilation times must be reported separately.
No change to physics/architecture/optimizer or optional reflection augmentation.
Relation: confined-ch-geometry-richer ACTIVE;5000step cold comparison in progress.

## 2026-10-29 — Expanded geometry coverage improves the matched cold comparison

Same5000Adam/B4budget/model/initial parameters and identical8development validation
domains:512training inputs reduce native phase9.184/11.200% to4.690/5.454%
(pooled/terminal), while original16training-monitor phase rises2.016->4.314%.
Common4627physical probes retain exactly the same reference variation; response
error77.027->35.514%, amplitude74.423->88.478%, correlation.6453->.9362. Local2
FEM phase12.846/15.402->8.976/10.431%, so improvement is not solely native-target
cancellation. Native/local operator bias7-9% remains distinct. Actual20kdraws
visit all512inputs20..68times, never oldheldout draws16..23. CPU final24field
replay agrees<=1.64e-15 and initial parameters/reference arrays are identical.
Cold trainer431.30->524.42s reflects larger cache setup; warm43.02->43.22ms.
This supports broader coverage/reduced overfitting for one matched run, not
monotone sample-count gains, seed stability or an independent final-test claim.
Retain the512preset as opt-in research route; leave windowed single-IVP default
and prior controls/artifact paths intact. Relation: confined-ch-geometry-richer
COMPLETE; no further optimization or Git action authorized/queued.

## 2026-10-29 — Extend the512-domain operator from a matched cold initialization

User requests a longer cold run because5000step validation curves still fall.
Predeclare20000Adamsteps, changing only iteration count from the512preset; reuse
same cached512/8partition, physical/operator/model/payload and runtime seed42.
Keep the exponential LR rule .001*.95**(step/1000), floor1e-5, unchanged rather
than stretching it: the common5000step prefix remains directly comparable.
Fresh network/Adam state, canonical train.py, independent run directory. Keep
500step evaluation/1000step saves and fixed-budget final checkpoint, no held-out
selection. Compare full/terminal and localFEMphi plus geometry response and tail
trends; more steps are not assumed to guarantee convergence. Relation:
confined-ch-operator-long-training ACTIVE; no Git operation authorized.

## 2026-10-29 — Longer cold budget improves fitting but does not prove a plateau

Fixed20000step final reaches native validationphi2.463/2.837% versus5000final
4.690/5.454%, and chemical6.971% versus11.661%. Local2FEMphi8.976/10.431%->
8.111/9.473%, with unchanged independently checked native/local bias7-9%.
Initial parameters are exact and common5000metrics differ<=4.31e-16; checkpoint
parameter relative difference7.28e-15. Keep final-budget selection and expose
the complete fluctuating curve: last5000phase2.333..2.791%, fitted slope
-.0585percentagepoints/1000steps, so no claim of complete convergence or a
validation-selected best model. Cold1401.54s includes setup/compile, warm41.89ms.
Retain5000control and add opt-in20000preset, without changing the physical
horizon/model/operators or windowed single-IVP default. Relation:
confined-ch-operator-long-training ACTIVE pending replay/geometry diagnostics.

## 2026-10-29 — Complete the longer-budget control with unchanged inputs

Independent24field replay agrees<=1.45e-15; all80kactualdraws cover512domains
123..201times and the first5000updates' condition IDs match the prior run exactly.
No future arrays enter optimizer payloads; raw field/chemical/mass/sample-energy
checks pass. Same4627common queries retain exactly the same reference variation;
geometry-response error35.514->19.284%, amplitude88.478->93.316%, correlation
.9362->.9825. Archive full curves, final native/local metrics, source/config/
initial/prefix identities and compact receipts. Keep final20000model as the
explicit longer-budget result; retain the old5000control and all artifact paths,
with no numerical-core/default/Git change. Relation:
confined-ch-operator-long-training COMPLETE; no work queued.

## 2026-10-29 — Increase physical complexity with longer multiscale coarsening

User requests longer evolution and richer large/small initial ripples. Test a
separate T2.0/20explicit-wave fixed ambient IC preset, keeping epsilon.06,geometry
prior/h.035/delta.11 and generic point architecture unchanged. Keep phase bound
strictly below1; distribute energy across low/mid/high modes with weaker high
frequencies. Local linearized M*k²*(beta-epsilon²*k²) identifies likely growing
and damped bands as a guide only; higher modes may physically decay early.
Preview existing validation domains first, then recompute ALL monitoring/
validation trajectories/local checks for changed physical inputs. No reuse of
old future fields, no hole-dependent envelope, no shrinking epsilon blindly or
new tailored attention. Parallel CPU solves/precomputation precede cold20kAdam;
changed-case error comparisons are not isolated algorithmic improvements.
Relation: confined-ch-multiscale-operator ACTIVE; no Git authorization.

## 2026-10-29 — Balance modal amplitudes after the first physical preview

T2/20modepreview increases late purity to4-8% (|phi|>=.8), but initial plots
remain dominated by the previous large lobes. Compare fixed candidate ICs on
one unchanged geometry using a shared color scale; choose24more evenly weighted
modes (first8weights.20,last16weights.18, amplitude.20, strictbound.896).
Add four missing directions up to(3,+/-3), shortest wavelength.208vsmesh.035;
keep epsilon.06/delta.11 and all geometric/model assumptions unchanged. Preview
T3to permit clearer late phase separation after high-band early relaxation.
This is selection of a physical benchmark from reference fields, not network
seed/error tuning. Preserve T2/20preview/config/results as untrained candidates;
new preview uses an independent path and must precede full production bank.
Relation: confined-ch-multiscale-operator ACTIVE.

## 2026-10-29 — Parallelize independent reference domains without changing solvers

The old local checks cost~206s/domain atT.75; T3would multiply serial wall.
Add explicit optional reference_workers=6 and local_reference_workers=2 for
this new case. Each spawned CPUprocess owns one domain; native ordered-row/BDF
and local mixedP2/BE-startedBDF2/refinement thresholds are unchanged. Parent
orders records/handles manifests and rejects failures. Local worker imports
DOLFIN itself (parent stays uninitialized), meshing remains separately isolated;
all FEM workers finish before GPU. Legacy configs retain serial orchestration.
No coarsened FEM time step or relaxed tolerance is introduced. Relation:
confined-ch-multiscale-operator ACTIVE.

## 2026-10-29 — User corrects IC scope: fixed small-scale10modes only

User clarifies that the initial state is fixed and selects the previously shown
small-scale component, reduced from12to10modes. Stop the superseded24modeFEM
work and retain completed/partial outputs as historical candidates, no training.
Keep T3/epsilon.06 and other physical parameters initially; select the first10
of the old highband, omitting the two highest-|k| diagonal modes(3,+/-3), preserving
all retained amplitudes/phases (.036physical modal amplitude). No mixed bands,
new scale-control knob or parameterized IC. Preview actual evolution because
these high modes are locally damped aroundphi0, while walls/nonlinearity can seed
other scales. Do not assume late separation or change physics without evidence.
New source/training presets/paths identify10mode small-scale case explicitly.
Relation: confined-ch-multiscale-operator ACTIVE (corrected scope).

## 2026-10-30 — Retain tested FEM scheduling for the corrected10mode case

Approval quota interrupted an earlier proposed six-resolution scheduling edit;
no such change executed. On resumption, retain the working two-domain spawned
FEM checks rather than extending reference infrastructure for this experiment.
The24new10mode native references all pass in340.25s; local refinement remains
the same two geometries/three resolutions/original gates. Proceed to512operator-
only preparation and cold20kAdam only after local checks. No mixed24futurefields
are reused. Relation: confined-ch-multiscale-operator ACTIVE; no Git action.

## 2026-10-30 — User prioritizes native-only operator training over FEM

User explicitly requests training/evaluation against nonlocal references first,
deferring FEM. Stop owned FEMparent/workers, preserve partial outputs without a
passed-local claim. Permit an explicit empty operator.local_validation_indices
to mean no requested local checks for this bank; set this in the selected512
preset only. Precompute must not read/copy FEM files in that case, and evaluation
must report not_evaluated rather than NaN or a fabricated local metric. Existing
nonempty presets retain their required passed local checks. Native24refinement
gates remain mandatory. Sourcepreset retains two FEMindices for optional later
resumption. No physical/model/optimizer change; no Git operation. Relation:
confined-ch-multiscale-operator ACTIVE, native512cold20k now takes priority.

## 2026-10-30 — Matched SOAP control before changing model width

User requests SOAP and asks whether a larger architecture would help. Adam20k
still decreases nativephi55.07->51.26%inlast5k, so label slow incomplete fitting
rather than a proven stationary trap/capacity limit. First coldstart SOAP20k
with exactly the same architecture/LR/bank/physics/sampling/runtime seed, changing
only optimizer. Use existing explicit b1.9,b2.99,frequency5,maxdim128,no1D,clip10.
Preserve fixed-budget selection and compare wall/step time as well as L2. Next
capacity control may widen64->96 with heads4/depth/Fourierbandwidth unchanged;
measure memory before a full widened run. Do not tune seed or resume Adam state
for the primary cold comparison. Local FEM remains deferred and old defaults
remain until evidence. Relation: confined-ch-multiscale-operator ACTIVEextended.

## 2026-10-30 — Adam native-only result is slowly improving but inaccurate

Complete20kAdam1435.614s: native8meanphi51.260/50.784%pooled/terminal,16train
monitors50.959%, mu47.528%. Last5kphi55.072->51.260%, slope-.729pp/1000: slow
incomplete fitting, not demonstrated trapping. CPU24fieldreplay4.83e-15, nofuture
payload, sameinitialparams/80kactualdraws/all512inputs123..201visits verified.
PhysicalIC/massandsampled5timeenergychecksdo not implyfieldaccuracy. Geometry
responseerror91.711%,amplituderecovery28.423%,correlation.42165 onsame4627queries
shows weakconditionalfidelityaswell. Keepprioraccuratedefaults, archive thisas
a harder-case baseline after requestedSOAPcomparison, not a newSOTA. No local
PDEaccuracy claim orFEMfollowupqueued. Relation: confined-ch-multiscale-operator ACTIVE.

## 2026-10-30 — Temporal Fourier inputs change physical scale with the horizon

Direct code inspection confirms angles=j*pi*s with s=t/T, not2*pi*s. Four
input frequencies therefore have maximum angular frequency4*pi/T:16.755rad/s
atoldT.75 versus4.189rad/s atT3. Increasing time_features4->16 would restore the
old physical input band and add only1536trainable weights atwidth64. Nonlinear
FiLM/MLPs can produce higher frequencies, so this is an inductive-bias/optimization
hypothesis, not a strict output bandlimit or proven cause. Record as a separate
generic architecture control from width64->96. Do not change the in-flight SOAP
control, which must retain its matched4frequency input. No extra architecture
run has started. Relation: confined-ch-multiscale-operator ACTIVE.

## 2026-10-30 — Complete matched SOAP control without claiming convergence

Same architecture, initial parameters, numerical source and 80000 condition draws:
SOAP20k reduces native phase pooled/terminal51.260/50.784% to39.561/39.170%.
Cold wall1435.614->2030.369s and warm42.785->69.817ms; first matching both Adam
final metrics is SOAP8500calls/1045.747s. Last5kstillfall44.00->39.56%, supporting
slow incomplete convergence rather than a demonstrated trap/capacity limit.
All24SOAPfields replay on CPU within3.56e-15, no future-label payload or held-out
input draw. Geometry response improves but remains weak (80.07%error/41.95%
amplitude). Retain explicit SOAP research preset; do not promote this still-poor
harder case over older defaults. Width96 and time16 controls are concrete options,
not measured remedies or queued jobs. FEM remains deferred per user direction.
Archive records/source/compact receipts; generated outputs stay external. Relation:
confined-ch-smallscale10-operator COMPLETE, no active task or Git operation.

## 2026-10-30 — Optimize generic Fourier features toward the requested10% target

User authorizes wider Fourier embeddings and continued optimization to about10%
relative phase L2. First test spatial64frequency vectors(128sin/cos channels),
time32frequencies(64channels), spatial Gaussian scale2, keeping64wide backbone,
heads/depths/outputscale.1 and physical/data/loss/sampling settings. This bundles
feature count and bandwidth and cannot identify either alone. Increase SOAP
max_precond_dim128->256 so133wide lifting inputs remain preconditioned; otherwise
embedding expansion would also disable that input covariance axis. Common seed42
remains provenance, not tuned. Preserve prior20kcontrols/defaults and pure physics.
Save final optimizer state for possible explicit future continuation, changing
no numerical update. Native-only/FEMdeferral continues. Relation:
confined-ch-fourier-optimization ACTIVE; no Git authorization.

## 2026-10-30 — Diagnose growing-mode errors before global time curriculum

The Fourier64/time32/scale2 control is still learning (5.5kphase64.29%), but
slower early than priorSOAP. Frozen2kCPUdiagnostic on geometry512 finds phase
RMS decaying .0831->.0481 while native reference grows to .4476; growing-subspace
error at T is94.49%. Exact weighted graph linearization about phi=0 gives
mode growth gamma=M*lambda*(beta-epsilon^2*lambda), fastest gamma1.388 and
exp(T*gamma)64.32. This is an infinitesimal diagnostic, not a nonlinear solution
or a proof of the only bottleneck. Existing biharmonic metric weights that mode
.1857; average transformed RMS residual is .0213 early-half vs .00842 late-half.
Test the repository's existing log-time/global-horizon curriculum next, keeping
one network and the same physical T3/IC/geometry/metric. Set explicit .05time
scale, .15initial horizon and8kwarmup; same4positive times plus t0, no extra labels
or time windows. Preserve the historical operator sampler bit-for-bit when this
optional record is absent. Report this as an optimization-measure experiment,
not a physics correction or a guaranteed route to10%. Relation: active Fourier task.

## 2026-10-30 — Separate temporal Fourier width from bandwidth

Add optional static arch.time_max_frequency to the generic point operators.
Without it retain the exact arange(1,K+1) expression, RNG keys and parameter
leaves. With it use K frequencies linearly spaced from1 to the explicit maximum
in units of pi*t/T; require K>=2 and finite maximum>=1. A wide embedding need
not increase its highest frequency. This is a reusable representation control,
not an asserted cure. Keep the in-flight and queued controls at their original
implicit integer band. New bounded-band configurations will be separate cold
runs only after measuring the current curriculum. Verify old checkpoints by
independent CPU replay; no old files or defaults are rewritten. Migration README
read; no in-place checkpoint resume. Relation: active Fourier optimization.

## 2026-10-30 — Test a stable-part resolvent without weakening growing modes

Curriculum at11.5kstill74.73%, not evidence of an improvement. Prepare a separate
reference-free CH residual norm from the frozen homogeneous state m0=mean(phi0):
S=V^(1/2) A V^(-1/2), q(lambda)=M*lambda*(epsilon^2*lambda+beta*(3*m0^2-1)).
The linearized evolution is y_t+q(S)y=0. Use P=(I+T*q(S)_+)^(-1), pulled back to
physical coordinates, where the positive part is defined spectrally. All weights
are in(0,1]; growing modes(q<0)retain1, while stable stiff modes are damped.
This is an SPD residual norm with the same zero set, not a change to PDE/BC/IC
or a nonlinear reference solution. It is a frozen linearization, not claimed
optimal or guaranteed accurate. Independently verify closed-form3node spectra,
finite-difference CH Jacobian and volume symmetry/constants before any new run.
Build matrices once with explicit6CPUworkers from input-only graph/IC data; reuse
the physical bank without duplicate dense caches or labels. Keep old metric
paths/default arithmetic unchanged. Bound-band control remains separately queued;
new metric is only a candidate until validated/measured. Relation: active task.

## 2026-10-30 — Prioritize validated stable-part norm after the curriculum control

Full520domain CPUpreflight passes input-only construction, PDEbackprop/nonzero
encoder gradients and hardIC/mass, with all metric weightspositive. Curriculum
17.5k40.91% is finally competitive but far from10%. Prioritize a cold bounded
32/max4 temporal band + stable-part metric + historical uniform-sampling bundle
next if curriculumfinal stillmisses either10%target. Cancel only the owned
waiting bounded-only launcher (no trainingstarted) and retain its explicit
unrun preset. This avoids spending another fullcontrol solely on bandwidth
before testing the diagnosed growing-mode weighting. No isolated causal claim:
new run changes band/norm/sampling relative to curriculum and band/norm relative
to uniformFourier. Same243649weights/physicalbank/SOAP/budget/seed. It is not
reference supervision, geometry-specific input design or a PDE change. Relation:
active Fourier task, no Git history action.

## 2026-10-30 — Preserve state and global draws for explicit operator continuation

All completed controls still have meaningful training error, and optimizer state
is now saved. Add an explicit train.continuation.parent_run extension to the
canonical geometry trainer, for a NEW run with a higher total global iteration
budget. Require complete parent model/state/config/summary, identical numerical
configuration including architecture/sampler/optimizer/seed, and a strictly
larger total budget. Restore model AND SOAP state; global fold-in counters and
curriculum remain continuous. Replay all imported parent saved physical fields
before any new update; reject mismatch. No in-place resume, hidden state reset,
new seed or schedule retuning. Save direct-path parent provenance and separate
stage/cumulative wall times. Fresh configs retain the historical initialization,
iteration/RNG sequence and loss. Independently test real SOAP/PDE steps through
serialization against uninterrupted CPU steps, including a refresh boundary.
This adds training-budget continuity, not a claimed accuracy improvement or time
window method. Use only if the measured cold candidate remains above target and
continued convergence justifies it. Relation: active Fourier task.

## 2026-10-30 — Reuse generic logarithmic time lifting for the initial transient

Input-only initial RHS across8domains has RMS .907..987. With phi=phi0+s*raw_c,
s=t/T, the required centered raw(0)=T*F(phi0) has RMS2.72..2.96, despite phase
initialRMS~.08. Reuse the existing single-IVP logarithmic_additive convention:
s=log1p(t/tau)/log1p(T/tau) for BOTH decoder time and the IC increment, tau=.05.
Then raw_c(0)=tau*log1p(T/tau)*F(phi0), RMS.186..203 (factor14.595 smaller).
This follows exactly by differentiating the hard ansatz at0; uses no evolved
reference. A positive monotone time coordinate preserves the free solution class,
hardIC/mass and physical-time JVP/PDE. It is input/lifting conditioning, not a new
PDE residual Jacobian division. Optional phase record only; absent preserves the
historical arithmetic and old checkpoints use saved configs. Independently check
closed-form physical initial derivative, mass and off-grid query agreement.
Prepare a separate cold stable/log-time control if current20kstillmisses10%;
no outcome or isolated optimum is assumed. No model growth or geometry-specific
prior. Relation: active Fourier task.

## 2026-10-30 — Prepare a unit-output-gain control before further model growth

The existing point decoder uses output_scale=.1. Initial derivative scaling shows
centered correction magnitudes can be much larger than the initialized decoder.
Prepare an explicit unit_output variant of the log-time preset changing ONLY
that existing static gain to1.0, the phase field's natural unit scale. No bound,
new parameters, different loss/labels or geometry feature is introduced. This
is not yet a measured/queued run, and the current log-time20k must finish first.
Check initial physical ranges and finiteness from input-only data before any
trial. Prefer this inexpensive parameterization check to speculative geometry-
specific conditioning or an immediate large-model change; if used, report the
controlled gain change and actual metrics, not an assumed improvement. Relation:
active Fourier task.

## 2026-10-30 — Prefer the simpler time coordinate for the gain-only control

Log-time17.5k23.43% is comparable to linear-time23.2%; no clear global benefit.
User values general/simple schemes. After the complete20k comparison, select
log only if BOTH primary phase metrics improve over linear by at least one
absolute percentage point; otherwise use the historical linear coordinate.
This is a practical development-selection threshold, not statistical significance.
The fifth cold control will change ONLY output_scale .1->1 relative to that
explicit selected baseline. Preserve both explicit variants; cancel/recreate only
the idle unit-log waiting launcher, not the active20kcontrol. Reference fields
still never enter optimizer loss/payload. Same512/8development cohort, no fresh
test claim. Relation: active Fourier task.

## 2026-10-30 — Independently check a linear causal defect preconditioner

The unit-gain control at12k26.29% remains comparable to priorparameterizations.
Investigate loss conditioning in time before claiming a representationfloor.
For frozen modal q, solve z_t+q*z=r with z(0)=0 on piecewise-linear nodal r.
Exact interval coefficients are a=h[1-(1+d)exp(-d)]/d^2,
b=h[d-1+exp(-d)]/d^2,d=q*h; use checked Taylor limits around0. Retain an
additional first output(h1/2)*r0 so the discrete map is injective, while the
propagated initial response remains0. This is one global linear residual map,
not time-window training or a nonlinear reference solver/output prior.
Independent SciPy quadrature differs<=1.64e-16; scalar constant/linearforcing
checks and positive singularvalues pass. Frozengeometry512 stable20k has
physical errorRMS.03946, spatialdefectRMS.03192, causaldefectRMS.08357;
terminalerror.07589 vs causaldefect.24019. These are diagnostic scales, not an
assertion that linearized defect equals nonlinear error. Prototype only, no
productionintegration/queuedtraining yet. Compare another frozenmodel and
validate fixed-time quadrature/AD/physicalgradients before deciding. Relation:
active Fourier task.

## 2026-10-30 — Measure minibatch noise before adding model/loss complexity

Frozen final-output-weight gradients on16traininginputs/8time draws show, at
Nt4, geometryvariance .0004976 and shared-timevariance .00008021. Estimated
B4totalvariance .0002046. Nt32 reduces shared-timevariance to3.187e-7 and B4total
to.00008697; B16/Nt32 estimate.00002198 (9.3x lower). This partial-gradient
finite-cohort probe is not a full-network proof, but motivates a simpler batch/
time-count control before integrating the causal prototype or growing the model.
Keep the SAME expected objective by explicitly preserving the legacy Nt4+0
initial-PDE fraction1/5: L=.2*l(0)+.8*mean_positive(l(t)). For stratified uniform
positive times, its expectation is independent of Nt. This is not an IC loss.
Add optional initial_residual_fraction; absent retains exact old mean arithmetic.
Measure Nt16 separately, then consider B16/Nt16 with a fixed10k cold budget.
No training has used these new counts yet; no change to PDE, input priors, seed
strategy or geometry/query architecture. Relation: active Fourier task.

## 2026-10-30 — Cold-test B16/Nt16 with preserved PDE expectation

The separate Nt16 frozen-output-gradient measurement gives geometryvariance
.00036054,timevariance1.973e-6. Estimated B16/Nt16 variance2.451e-5 versus
B4/Nt4 .0002046 (8.35x reduction). This is a measured partial-gradient estimate,
not a promised full-network L2 gain. Cold-test B16/Nt16 for10000updates, keeping
same64wide243649parameter model, linear-time/unitgain/stablemetric/seed/LR.
Explicit initial_residual_fraction=.2 preserves the legacyexpected objective.
New weighting value/fullparametergradient agrees with old Nt4mean, and toy
B16/Nt16backprop plus invalidfractions pass (2tests32.76s). Record first-update
allocator stats to assess actual full physics/backward/SOAP memory. No fixed
nodes, new PDE, future-field labels or geometry-specific features. Causal defect
prototype remains unintegrated; its scalar tests/diagnosis do not assert a
nonlinear error equality. Consider strictcontinuation only after this fixed
pilot finishes and actual errors are known. Relation: active Fourier task.

## 2026-10-30 — Prefer state-preserving budget extension while larger batches improve

At6.5k, B16/Nt16 phase21.59/23.15% improves on prior20k22.73/24.30%; both
training and development errors are still falling. Frozen stable20k modal
diagnosis places mean90.94% pooled/90.63% terminal error energy in the growing
subspace of the homogeneous-initial-mean CH Jacobian. This is a linearized
diagnostic, not a nonlinear-mode classification theorem or an error estimator.
It does not support simply raising the highest Fourier frequencies. Finish
fixed10k; if either primary target still exceeds.1 and BOTH primary errors
drop by at least.01 over its last5k, continue the complete state to20k TOTAL.
This is a practical development-budget rule, not a significance test or
minimum-error checkpoint selection. Use explicit batch_continue config/new run;
retain model/SOAP state, scheduler/global draw counters and physical objective.
No wider model, new seed, LR reset or changed residual is bundled with this
extension. If the rule fails, stop that extension and diagnose before another
architecture/loss experiment. Relation: active Fourier optimization task.

## 2026-10-30 — Prepare a width-only control without raising Fourier frequencies

B16/Nt16 at14.5k continues to improve (16.86/18.14% phase, train12.61%).
Frozen10k PDEloss/field-error association remains positive; no proof of a
capacityfloor or ineffective objective. Input-only width64/96/128 CPUpreflight
has243649/522145/905089trainableweights and finite SOAP states. Prepare an
explicit batch_wide preset changing ONLY backbonewidth64->96 from the10kB16
cold preset. Keep128spatial/64temporal Fourier channels, frequencyband/scales,
fourheads/depths, SOAP/B16/Nt16/LR/physical inputs and expectedobjective.
The preparation is not a trained accuracy/memory claim and is not queued;
finish current20k first. This reserves a conventional capacity test instead of
geometry-specific input features, another residual design or speculative
highest-frequency growth. Relation: active Fourier optimization task.

## 2026-10-30 — Permit an explicit state-preserving learning-rate reduction control

At16.5k the unchanged B16 trajectory is16.29/17.53%, still improving more slowly.
Prepare a conventional short learning-rate fine-tuning alternative before paying
for a wider cold restart. Default continuation stays strict. An explicit positive
factor<1 may allow ONLY lr_init and lr_min to scale by that declared factor;
require both values to match, reject unknown options and every other numerical
change. Import complete model/SOAP state and original schedule count, so this
scales the ongoing schedule rather than restarting it. Installed SOAP0.2.2
source applies learning rate after its adaptive transformation; independently
test equal-gradient update ratios and equal adaptive states through serialization.
Candidate factor.3/5000newupdates is a prepared control, not a measured benefit
or queued job. Physics, loss, model, RNG and input sampling stay fixed. The active
20k control finishes first; no runtime mutation. Relation: active Fourier task.

## 2026-10-30 — Select the short learning-rate control after the complete20k run

Unchanged-schedule B16/Nt16 finishes20k with phase15.330/16.469%, train10.636%,
mu28.500%; cumulative5609.284s. Target remains unmet. First try the prepared
.3 schedule scaling for5000newupdates (25kTOTAL), preserving all states/global
draw counters and input/objective/model. Actual initial fine-stage LR is about
1.075e-4 because the inherited counter remains20000. This lower-cost ordinary
optimization test precedes a96wide cold restart, not a promised improvement.
Use a new run/fixedfinal25k and report it even if worse; no min-validation or
seed selection. Three focused continuation checks pass across the combined run
and corrected scalar-ratio test; the existing Optax scalar-schedulefloat32
roundoff is accounted for without changing factory arithmetic. Relation: active
Fourier task; wider candidate remains prepared, unrun and unqueued.

## 2026-10-30 — Conditional width96 cold control if fine-tuning still misses the target

Fine-control first1000updates are15.105/16.244%, only a small improvement over
20k15.330/16.469%; complete its fixed5000newupdates. If either primary phase
metric still exceeds.1 at25kFINAL, cold-test prepared width96 for10000updates.
Only backbonewidth changes from the B16/Nt16 cold preset: same fixed Fourier
directions/bandwidth, sampling counts/expectedobjective, SOAP/LR, physical bank,
global interval and common runtime seed. No hand-coded geometry feature or new
residual accompanies the capacity test. Input-only initializer/finite SOAP state
preflights passed; actual fullGPUmemory/time and accuracy remain to be measured.
Run sequentially after the owned fine process exits, record conditional selection
and every outcome. Do not select a minimum-error checkpoint or silently fall back
on different batch/dtype after a failure. Relation: active Fourier task.

## 2026-10-30 — Diagnose a real geometry-generalization gap and prepare known-RHS conditioning

Frozen25k model: production PDEobjective on all512training inputs averages
9.43384e-5, versus.00118502 on8unseen inputs (12.56x). Seven unseen cases exceed
the training maximum; the eighth is at99.80percentile. 32/64time-quadrature
checks differ<=.6142%; original16and new496training cohorts have similarmean
9.030/9.447e-5. Input physical specifications/operators remain checked; no
unique cause is proved, but remaining error is not solely a step-budget issue.
Keep the independent width96 run unchanged. Prepare an OPTIONAL known-input
encoder control: [x,y,phi0,T*F_g(phi0)], where
F_g(phi0)=-M*A_g[beta*(phi0^3-phi0)+epsilon^2*A_g*phi0]. This is one evaluation
of the prescribed PDE on prescribed initial data, not a trajectory label,
future-field prior or hard initial-derivative constraint. It supplies discrete
operator response beyond coordinates/initialvalues; the principle applies to
first-order autonomous evolution systems. No loss/PDE/horizon change. Default
off; require explicit heterogeneous domainchannel count4 and independent formula,
mass, loader/no-label and legacy replay checks before any experiment. Reject
unsupported augmentation combinations rather than using stale known features.
Relation: active Fourier/geometry-operator optimization task.

## 2026-10-30 — Use idle CPU for a separately labeled known-RHS pilot

Known-input feature checks pass: independent CH/BDF derivative, volume-mean-zero,
hardIC/mass, offgrid/mask and gradient checks, plus original16operatorregressions.
Real geometry512 initialization is finite, exactIC/mass6.97e-18; knownT*F0RMS
2.8727/max10.3611,243713weights (only64more thanbase). Keep the GPU width96
cold control unchanged. Run an explicit1000step CPU-only pilot with samephysical
scheme/B16/Nt16/SOAP/commonseed,100stepmonitoring/500stepcheckpoints,8CPUaffinity
and1BLASthread. No FEniCS/MPI work. Cgroup memorylimit96.6GB/current32.2GB leaves
headroom. This is a preliminary backend-distinct diagnostic, not a matched GPU
accuracy ranking or SOTA promotion. No optimizer/numerical fallback if slow or
failed; preserve/log its actual outcome. Canonical train.py remains the entry.

## 2026-10-30 — Stop the CPU screening pilot after measuring inadequate throughput

CPU first100updates take~5.3s each, projecting over1h for the intended1000step
screening (memory remains within96.6GBlimit). Lastreported100step phase66.05/
65.80% is unconverged and not an accuracy ranking. Stop only the owned CPU
PID100148 and its waiting postprocessPID100505, retain rawlogs/config/initial
checkpoint/termination record. No complete1000step checkpoint or summary is
fabricated, no numerical failure or feature-effect conclusion is inferred.
GPU width96PID91175 continues. Evaluate the optional known-initial-RHS input
with a separate matched GPUcold control if target remains unmet; do not reuse
this incomplete CPU probe as SOTA or a backend performance comparison. Relation:
active Fourier task, prior CPU-screening decision.

## 2026-10-30 — Conditional matched-GPU initial-RHS encoder control

If either primaryphase L2 still exceeds.1 after fixed width96cold10k, run the
prepared initial_rhs10000step GPUcold preset. Relative to the B16/Nt16 cold
baseline, only the known initial tangent input and required domaininputdim3->4
change;64body/243713weights, sameFourier/bandwidth/SOAP/LR/physics/loss/sampling/
seedpolicy. Keep querycoordinates and free output/hardIC/mass unchanged. Do not
transfer partialCPUweights or rank the abortedCPUprobe. Use new run and final
10kcheckpoint; retain all outcomes. Input derivation, masking, physicalderivative,
labelseparation and original16regressions pass; realinputinitfinite. The optional
feature remains defaultoff until actual GPU/native/61time/replay evidence supports
selection. This is a test of a general known-PDE response representation, not a
new physical hypothesis or reference-shape prior. Relation: active Fourier task.

## 2026-10-30 — Prepare an incremental1024input-bank control for geometry coverage

The all512/unseen residual gap12.56x motivates a conventional data-coverage
control, not another physical formulation. Extend purephysics inputs512->1024,
preserving the exact original16monitors/8validation references and physical
prior/IC/PDE. Reuse all520existing records by accepted draw identity, including
operator-only inputs; build only512new operator/quadrature records, zero new
trajectory/FEM solves. An explicit hardlink reuse mode avoids duplicating the
~6GiB old payload on the~12GBfree filesystem. Existing defaultcopy behavior and
native-source preparation remain valid. Require identical physical specs,
unmodified original held-out draws, safe source paths and correct label roles;
no silent I/O fallback if hardlinks fail. Source inspection confirms isolated
Gmsh/meshio-only meshing and CPUJAX operators, no DOLFIN/FEniCSMPI solve; keep
GPU training independent. Verify logical remapping/reference availability and
inode identity in focused tests before generation. New training/bank-growth
accuracy is not claimed or queued yet. Relation: active Fourier task.

## 2026-10-30 — Continue the verified wider model with full state to20k

Width96 cold10k passes native/61time/CPUreplay/no-label gates:15.3286/16.4277%
vs width64 matched10k18.9452/20.2599%, at428.82 vs219.25ms warmupdates.
Keep all otherphysical/model/sampling/optimizeroptions unchanged and continue
its exact finalmodel+SOAPstate/globaldrawcounter for10000newupdates (20kTOTAL).
Use the established strict continuation path and new run; no state reset/LR
change/window/newconditionfeature. Serialize after owned initial-RHS GPU10k
completes. This tests further optimization of the promising general capacity
control, not a guaranteed10%result. No checkpoint/min-validationselection.
Incremental1024inputbank preparation remains separate and changes no current
trainingdata. Relation: active Fourier optimization task.

## 2026-10-30 — Queue a matched1024coverage control after the wide continuation

User asks whether training/unseen field errors justify enlarging the bank.
Best25k has nativephase9.3178/9.8983% on16training monitors vs14.3782/15.4513%
on8unseen geometries (pooled/terminal), ratios1.543/1.561. This supports a
coverage control but does not prove the sole cause; most training inputs have
no fieldreferences. Frozen512physics gap12.56x is a squared-lossratio, notL2.
Incremental1024bank nowPASS:129.565s,520unchanged hardlinked oldrecords/5816
exactarrays,512newknowninputs, original8devdraws16..23 preserved, zeroPDE/FEM
solves. Nmax1353/Kmax97, estimatedpaddedpayload16.804GB (~doubleduplication
33.6GB under38.184GBGPUlimit; actualfirstupdate muststillbechecked).
Queue canonical1024cold10k after owned width96continuation20k exits. It changes
only inputbank size/sourceprovenance relative64B16/Nt16cold10k; sameIC/PDE/T3/
architecture/loss/optimizer/seedpolicy. NoRHSfeature, warmstart ornewlabels.
Retain fixedfinal result even if worse. Alloldreferences/dense61arrays are
identical and only indices512..519->1024..1031 remap; explicit bank-growth audit
is required, never compare arbitrarydifferent references. NoGPUconcurrency,
noautomaticbatch/dtypefallback. Relation: active optimization and latestquestion.

## 2026-10-30 — Capacity improves near10%; larger bank alone does not at fixed10k

Both queued runs completed and fullnative/61time/CPU24field/no-label/physical
gates PASS. Width96/512inputs/20k:phase11.3836/12.2805%,61time10.92995%,
training16monitor8.0986/8.6376%; cumulative9972.284s,CPUgap4.219e-15. It beats
matched64/512/20k15.3301/16.4691% and prior64/25k14.3782/15.4513%.
1024inputs/64width/10k:20.2116/21.7872%,61time19.3169%,49.14min;
matched512/64/10k18.9452/20.2599%. Trainingmonitor15.3082->18.2987% also worsens.
The smaller train/unseen gap alone does not establish bettergeneralization;
optimization/capacity and finite per-input exposure remain confounded with
coverage (same160kdraws overtwiceas many inputs). No larger-bank SOTApromotion.
Actual1024GPUpeak32.441GB<38.184limit,live18.051GB; nofallback/failedrun.

Width96 last17500->20000 nativephase12.124->11.384% fallssteadily. Continue its
exact complete model/SOAPstate/counters and unchanged scheduledLR to30kTOTAL
(10000newupdates), rather than adding features or changing physics/loss. This is
the existing general budget-extension mechanism, fixedfinal/no seed/checkpoint
selection. No1024followupqueued now. Preserve both negativecoverage and positive
capacity controls in docs/index; current10%target remainsunmet andone-seed/
8reuseddevelopment limitations remain. Relation: active optimization/lateststatus.

## 2026-10-30 — Recover an incomplete30k run without discarding SOAP history

On explicit userresume, PID132521/watch132891 are gone, GPUidle, no30ksummary/
finalmodel/finaloptimizer; rawlogends26500 withphase10.2406/11.1150%. NoPython
traceback or cgroupOOM/oom_kill. Exitcause isunconfirmed; previousapprovalreview
quotaerrordoesnotprove it killedtraining. Latestmodelcheckpoint26000 lacksits
SOAPstate, so it cannot be advertised as an exact-state continuation. Keepthat
run/log/checkpoints incomplete. Restart samefixed30k recipe from independently
validated20kfullstate in aNEWrun, no optimizerreset or physicalchange. Ordinary
GPUpolicy may produce rerunvariation; don'tclaim byteidenticalfutureupdates.

Add intermediate optimizer snapshots and a step/model/state marker alongside
existing every1000step modelcheckpoints. Publish markeronlyafterbothpayloads
areclosed/atomicallyrenamed; no new loss/update/configchange. Verify realSOAP
recovery acrosscheckpoint with focusedtinytrainer beforelaunch. Explicitly
launch ownedtraining/postprocess in detachedOSsessions withdirectlogpaths/PIDs
so normaltoolsessioncompletion doesnotterminateauthorizedlongrunningwork.
No broadprocesskill/FEM/Gitaction. Recordlastincomplete outcome, notminselected
checkpointpromotion; fullfixedfinal/native61/CPUreplay gatesstillrequired.
Relation: activeoptimization, usercontinue, interrupted30kstage.

## 2026-10-30 — Finish fixed30k and prepare a portable selected-recipe entry

Detachedretry successfully completes30k: phase9.66090/10.51306% (pooled/terminal),
trainingmonitors5.92428/6.30245%, native saved-field checksPASS; dense61/CPUaudit
stillrunning at thisdecision. This meets the requested roughly10% pooledphase
aim, not a claim that terminal/allgeometries/allphysicalfields are below10%.
Prepare train_operator_smallscale10_best.py as a cold entry with SAME numerical
settings, removing only parent-run dependency. Exact configdictionary comparison
against actualsaved30krecipe passes after removing runtime seed/runname and
continuationprovenance. Do not change canonical time-window defaults or old
controlpresets. Report measured error/time as the staged10k+10k+10k result;
a separate cold30000run/seedstability has NOT been measured. Finalbestpromotion/
taskclosure waits fullnative61/CPUreplay/physicalgates. Relation: active task,
usercontinuecompletion, no architecture/PDE/optimizerupdate change.

## 2026-10-30 — Close the approximately10% pooledphase optimization after full validation

Fixed30kfinal is9.66090% pooledphase/10.51306%terminal, dense61time9.18872%;
requestedapproximately10% overallphase aim is reached. Terminal/allgeometries/
mu are not allbelow10% (worstgeometry17.255%,meanmu21.056%). Independentnative
metrics/61time massenergy/CPU24fieldreplay4.334e-15 andfullstate/draw/no-label
checksPASS; no FEM/freshfinaltest/multi-seed claim. Promote this fixedfinal as
currentbest, retain1024negativecontrol andoriginalincompleteattempt. Selected
coldconfig matchesnumericalsettings; measurednumberscome fromstaged10k+10k+10k,
not aseparatecold30k. Canonicalwindowdefaults/oldpresets unaffected.

Archive task underdocs/tasks/archive/confined-ch-fourier-optimization.md; update
README/index/PROGRESS toactualcomplete/noactivejobs. Allrunner/training/CPUmonitor
processes exited. Furthergeometrycoverage/seed/FEM studies are optionalfuture
work, not a reason to keep this boundedpilot active. NoGitcommit/push/tag/history
change authorized orperformed. Relation: completedFourieroptimization task.

## 2026-10-30 — Blind random gallery of the completed CH geometry operator

For the requested additional physical fields, draw five of eight reserved
development geometries uniformly without replacement and persist IDs before
reading errors. Use the selected fixed30k final checkpoint's saved, already
CPU-replayed fields; do not select favorable examples or retrain. References
remain native graph/BDF, not FEM. Use common phase/error scales and original
mesh interpolation only for display. Interpolate raw absolute errors through
the positive reconstruction, rather than taking absolute value after signed
interpolation (which can cancel errors). Relative L2 stays on original physical
cell values with area weights. Relation: confined-ch-random-gallery task.

## 2026-10-30 — Recommend separated capacity/refinement pilots before new hardware

User asks whether more points/network capacity and another GPU are needed.
Measured matched20k width64->96 phase15.33->11.38% supports recommending a
width128control at unchanged physical scheme and h=.035. It does not prove a
capacity floor or future gain; width128GPU training remains unmeasured. Recommend
a separate h=.030 native spatial-refinement check on1-2same geometries before
building a full refined bank; keep epsilon/delta fixed and use common queries.
Current9.66% is error against the same native discretization, so it alone does
not implicate point density. Refining h at fixed delta does not remove local
finite-horizon bias. Avoid simultaneous capacity/mesh changes as a first control.

Verified system48GiB; selectedrun peak active15.72GiB versus allocator35.56GiB.
No current need to request another GPU for modest pilots; full-run feasibility
requires an actual short GPU probe. Dense residual metrics for all520padded
geometries use7.07GiB now; h^-2 point scaling estimates13.10GiB at h=.030,
before activations/workspace. Network parameters grow522145->905089. Disk only
4.6GiB free, so a separate full refined bank needs storage preparation. This is
an assessment, not a launch/default/solver/migration/Git change. Relation:
confined-ch-density-capacity assessment.

## 2026-10-30 — Reclaim package caches without deleting numerical evidence

User explicitly authorizes conservative removal of genuinely unused/redundant
files. Inventory finds about11GiB uvcache,2.3GiB npm content cache,0.5GiB pip HTTP
cache, while the protected experiment data/logs occupy about73GiB. Reclaim
package downloads through supported cache commands, preserving installed
environments and all numerical artifacts. Scan of three project .venvs,
miniconda3 and managed uv installs finds no symbolic-link targets in candidate
cache roots. Honor uv's in-use checks; do not force cache-lock removal or edit
uv internal files manually. Record actual filesystem recovery rather than
equating logical cache sizes with reclaimed physical blocks.

Do not delete Trash (contains historical experiment outputs), OpenCode history,
IDE runtimes, migration backup, checkpoints, references or tf-logs. Conda
tarball/index/log dry-run reports zero recoverable bytes; do not clean extracted
package stores or installed environments. No new training, relocation or Git
operation. Relation: active disk-cleanup task. Official cache semantics checked
at https://docs.astral.sh/uv/concepts/cache/.

## 2026-10-30 — Remove nine verified orphan MPI shared-memory pools

After package-cache cleanup, inspect exact nine128MiB shared_mem_cuda_pool files
under /tmp/ompi.autodl-pro-7775bcfa7d82.0/jf.0. Each is a regular file older than
3600hours. Complete readable /proc maps/fd/cwd/root audit finds no references
and no process with libmpi loaded; no denied process inspection. Record exact
paths/inodes/sizes/times before deletion, recheck identity and process usage at
deletion. Remove only those nine stale pool files (about1.125GiB), retaining
session directories and all other files. Do not run ompi-clean or terminate
processes. Relation: disk-cleanup; user-authorized unused temporary-file cleanup.

## 2026-10-30 — Complete conservative cleanup and retain unique numerical outputs

Normal uv/pip/npm cache commands and nine individually verified orphan MPI
pool deletions recover measured9.805GiB: free4.529->14.334GiB. Package-manager
logical size reports exceed physical recovery because installed hardlinks keep
shared blocks.26protected artifact metadata checks, reference514/518readability,
project CPU JAX/autodiff and separate dolfin2019.1.0 import PASS. Retain numerical
data/tf-logs, checkpoints, environments, migration backups, Trash with old
experiments, app histories and IDE runtimes; no more deletions are necessary
for immediate modest pilots. Full new refined512bank estimated10-12GiB would
leave tight headroom, so optional30-50GiB extra capacity is preferable to losing
unique experiment evidence. No data migration/training/Git action. Archive task
and return dashboard to idle; exact receipt docs/tasks/disk-cleanup-receipt.json.

## 2026-10-30 — Execute separated width128 and small spatial-refinement controls

User explicitly approves proceeding. Add an opt-in width128cold30k preset which
differs from selected cold recipe only in backbone width; preserve512inputs,
B16/Nt16, Fourier arrays/depth, physical scheme, FP64ordinarySOAP and schedule.
Run a full-bank production-loss/optimizer GPU probe before launching canonical
train.py for fixed30k. Retain fixedfinal/paired checkpoints and durable detached
runner; no default promotion or old artifact deletion. Same runtime seed is
provenance rather than a tuned strategy option.

Before GPU work, independently assemble/refine native h=.030 references for
existing development geometries514/518 (previously observed worst/low-error
diagnostic cases, not a new random/generalization test). Preserve holes, IC,
epsilon=.06, delta=.11 and time refinement. Compare on common uniform physical
queries with both visible-centroid P1 and original-triangle P0 reconstruction;
report IC reconstruction errors/support explicitly. Do not subtract IC errors,
hide boundary support exclusions or identify reconstruction differences as pure
operator error. No full512refinedbank or localFEM yet; tasks are separated to
attribute capacity and mesh effects. Relation: confined-ch-width128-refinement.

## 2026-10-30 — Width128 GPU probe passes; start fixed30k with existing hardware

Two native h=.030 geometry514/518 references pass time/mass/energy gates before
GPU work. P1commonquery differences1.728/1.338%, P0differences11.569/11.273%;
initial reconstruction/supported-area differences show these are not pure
operator-error bounds. Preserve both metrics; no fullrefinedbank now.

Fullbank width128/12production SOAP updates pass exact Fourier/geometry/time
stream identity and finite-gradient/no-label checks. Activepeak17.354GiB versus
35.56GiB allocator; warm10updates .575272s, firstcompile359.232s. Current48GiB
card is sufficient for this unchanged-mesh control. Discard probe updates and
start fresh canonical fixed30k under detached runner; no batch/precision/seed/
policy change or default promotion. Estimated5-6hours includes validation, not
measured completion. Setup attempt01's missingmeshio error is preserved; use
existing explicitFEniCS mesher in attempt02 before GPU, no environment install
or training-seed retry. Task remainsactive until full outcome and audits.

## 2026-10-30 — User requests a cold beta2=.9 SOAP control

User explicitly proposes beta2=.9 and asks to rerun directly. Extend the current
width128 task with a new opt-in preset differing only in train.soap_b2 .99->.9.
Inherited shampoo_beta=-1 follows b2, so both rotated Adam variance and Gram
EMA change; describe this as coupled beta2/Gram-memory control, not an isolated
Gram-only test. Keep architecture, LR, batch/time streams, FP64/ordinary policy,
data, seed provenance and fixed30k budget. Do not resume the deferred optimizer
implementation/repeatability branch or change unrelated SOAP settings.

At a confirmed closed paired checkpoint, intentionally stop the owned .99
runner/trainer and preserve its incomplete run, model/state pairs and online
metrics. This follows the user's request to restart; no concurrent GPU jobs or
unrequested wait for the original30k. Record exact stop point, verified process
identities, previous journal and deliberate reason; don't label the partial
run a completed30k or change its saved config. Start .9 from the same fresh
initialization and RNG counters in a new directory. Audit initial parameters,
actual draw prefixes and same-step L2, then compare fixed30k to old96widthbest.
Currentbest/defaults unchanged pending completed validation. Relation: active
confined-ch-width128-refinement task, current user steering.

## 2026-10-30 — Audit encoder capacity and hardware before another scale increase

User asks whether more network capacity or a more efficient encoder is needed
and offers RTX5090. Extend the active width128/beta2 task with a CPU-only
architecture/shape and traced dot-work audit while its authorized GPU run
continues. Do not interrupt it, launch competing GPU work or change its saved
architecture/optimizer. Separate computational reuse from representation:
geometry/spatial conditioning is shared over time and normalized attention uses
associative feature moments, but per-head feature size is finite. Existing
geometry-swap/response checks establish input dependence, not sufficient
capacity. Parameter growth alone cannot establish a convergence benefit.

Inspect widths128/192/256, component parameter counts and shape-dependent SOAP
preconditioner coverage. Keep FLOP estimates explicitly distinct from measured
GPU timings; no component bottleneck ranking from analytical counts alone.
Check official RTX5090 memory/FP64 specifications before a migration
recommendation. No new mesh library, reference labels, precision change,
full-training/default promotion or GPU transfer is authorized by this audit.
Relation: confined-ch-width128-refinement, encoder/capacity/hardware follow-up.

## 2026-10-30 — Preserve FP64 feasibility result within its measured scope

User recalls the earlier good FP64 benchmark and asks whether48GiB indicates a
modified3090. Re-read the archived actual timings and dtype/IR checks rather
than discard them based on vendor peaks: corrosion203.262/226.737ms FP32/FP64
is valid for that legacy deterministic workload. Later serial graph-lowering
evidence prevents generalizing its11.5%slowdown to current NO or a hardware
FP32:FP64 ratio. Pure FP64 GEMM .636TFLOP/s is compatible with a normally boosted
3090; FP32 kernel throughput there was well below its nominal peak.

Use metadata-only CUDA Driver API during training, with no context/allocation/
kernels. It reports82SM,CC8.6, single:double precision performance ratio64.
NVML48GiB is nonstandard versus official3090 24GB, so expansion/platform
customization is a plausible inference, not a confirmed physical inspection.
Do not claim enhanced FP64 units, identify board modifications or infer a
measured5090 ranking. Preserve old benchmarks; record direct device receipt,
source definitions and uncertainty. Relation: active width128/capacity task.

## 2026-10-30 — User requests FP32 on the existing GPU

User explicitly asks to try FP32 because current NO training is slow, and
clarifies to keep the current GPU. Extend the active task with an explicit
width128/beta09 FP32 preset and complete compute-dtype plumbing. Preserve
FP64 reference/operator/metric construction and validation on the host; round
the finalized model and floating training inputs once to FP32. Keep JAX x64
enabled for unchanged initialization/time-RNG/reference construction, not for
network/PDE/SOAP arithmetic. Generate the same FP64 time draws and cast them;
do not silently change random streams, architecture, batch, physics or SOAP.

First run focused CPU precision/backward-compatibility checks. At a verified
closed model/optimizer pair, intentionally stop the owned FP64 runner/trainer
and preserve their partial status/checkpoints and journal. Run a production-size
FP32 finite/dtype/time/memory probe, then a fresh canonical30k run only if it
passes. No GPU replacement, mixed-precision fallback, continuation across
dtypes or default promotion. Save matched-step/elapsed comparisons and validate
against the original unrounded native reference. Precision-aware independent
IC/chemical checks must use declared rounding bounds, never obsolete FP64-only
thresholds or blanket relaxation. Relation: confined-ch-width128-refinement.

## 2026-10-31 — Keep FP32 SOAP updates from transient promotion

Production-size preflight catches a pinned SOAP detail with globalx64 enabled:
Python beta powers produce a FP64 bias-correction scalar, multiply every FP32
update in FP64, then cast back. Model/state dtype checks alone miss this.
Keep the shared optimizer and upstream SOAP implementation unchanged. The local
operator precision adapter supplies FP32 beta1/beta2 scalars only for FP32SOAP;
FP64/Adam paths delegate unchanged. Nominal hyperparameters remain the same.
CPU7update/refresh and lowered-IR checks now pass with no nonscalarFP64 tensors.
Scalar configuration literals that are cast toFP32 are permitted and counted;
they are not a mixed-precision network/matrix/QR fallback. Retain failed probes:
01config tuple/list mismatch,02IR promotion, both before optimizer updates.
Attempt03 uses the same numerical config and seed, with this precision fix;
no training-seed selection, changed algorithm or new hardware.

## 2026-10-31 — User authorizes moderate FP32 operator capacity increase

Continue the active capacity task with a fresh width192/four-head control:
head dimension32->48, spatial Fourier directions64->96 (128->192sin/cos
channels), unchanged four cross-attention blocks and three FiLM decoder layers.
Keep the fixed512/8geometry bank, PDE/constraints/metric, FP32/ordinary policy,
SOAP beta2=.9, LR, B16/Nt16, sampler keys and fixed30k final budget. No learned
weights or optimizer state imported. Reject changing geometry density, deeper
networks or jumping directly to256 while testing this moderate enlargement.
This jointly enlarges backbone and Fourier feature count, not a width-only
ablation. Independent initialization follows the same global seed policy;
different-shaped weights/features cannot be called identical initial arrays.

Set explicit SOAP max_precond_dim512 to retain both FFN axes at width384;
leaving256 would silently remove eight preconditioned axes. For the128baseline,
256 and512 have identical coverage because no learned axis lies between them;
the temporal output axis768 remains excluded under both. Record this coupled
setting and verify shape/state coverage, instead of attributing any measured
gain solely to head dimension or promising a precision improvement.

First finish/check the prior30k FP32 audit, then use exclusive production-size
25update GPU/IR/finite/memory and independent CPU replay gates. Discard the probe
and cold-start canonical train.py only on success; automatically assess native,
dense61, geometry response and CPU/input-draw audits. Preserve all old results,
do not promote defaults or select a best seed/checkpoint. Relation: current
confined-ch-width128-refinement task, explicit user approval of proposed192width.

## 2026-10-31 — Close the moderate capacity control at its prescribed final budget

Completed width192/fourheads/Fourier192feature cold30k improves native mean phase
9.7841->8.9148% and terminal10.5799->9.8017% against completed width128FP32;
61time phase9.3192->8.3458%, coldtraining wall+22.52%. Actual30000geometry batches
are identical, full CPU/state/input/physical audits pass. This supports moderate
joint capacity growth within this benchmark; it cannot isolate head dimension
or establish a multi-seed/continuum result. Two of eight dense geometries worsen.
Retain fixed30k final despite a lower29500validation point, reject automatic
best-heldout checkpoint selection, immediate256/longer training or default
promotion. No further run is queued. Preserve previous selected defaults and all
partial/control artifacts; archive the completed task with direct result paths.
Relation: completed confined-ch-width128-refinement capacity/precision study.

## 2026-10-31 — User authorizes width256 with a memory gate and 50k budget

Create a new active task for a fresh width256/fourheads(D64), spatial256feature
operator and fixed50k SOAP budget after the192/30k study. Keep four cross blocks,
three FiLM layers, FP32/ordinary, beta2=.9, axis cap512, LR schedule, B16/Nt16,
bank/physics/constraints/metric and runtime-seed policy. Cap512 already covers
the new512wide FFN axes; the1536wide time-output axis remains excluded. Do not
change the GPU, reduce the batch, silently mix precision, tune seeds or import
previous learned states. User explicitly requires GPU-fit confirmation first.

Reuse/generalize the research-only gates/runner with explicit experiment case
metadata, retaining the prior192CLI behavior. Audit actual production25updates
and preconditioner refreshes plus independent CPU replay before starting50k.
Compute a conservative disk budget for50paired intermediate states; keep all
old runs. Retain30k native and dense61metrics to compare capacity at a common
budget, then report256/30k->50k as extra-training benefit. Reject attributing
192/30k versus256/50k differences solely to architecture. No automatic default
promotion or best-heldout checkpoint selection. Relation: new
confined-ch-width256-50k task, explicit current user instruction.

## 2026-10-31 — User switches to width192 and expanded bank, cold50k

Stop the owned256runner/trainer as explicitly requested, preserve40000complete
model/SOAP pair and40900partial logs, and cancel its50kdownstream pipeline. Close
the previous task by user switch, not original-objective completion. Test doubled
512->1024geometry inputs using existing immutable checked bank; no reason to
recompute operators or solve training trajectories. Retain accepted draws16..23
as the exact same reserved physical domains and16originaltraining monitors.

User supersedes the proposed30kbudget with explicitcold50k. Preserve width192/
Fourier96directions/fourD48heads, FP32ordinary, SOAPb2.9cap512, B16/Nt16 and all
physics/loss/LR/seedpolicy. Initialize exactly as prior192coldrun, not continuation.
Retain30kcheckpoint/dense61comparison versus192/51230k; report added50ktraining
separately. Expanding randint support changes condition-ID draws, so reject a
false identical-draw-prefix requirement; audit actual newdraws/globalkeys and
no heldout overlap. Preserve time stream and exactsame initial arrays. Native
validation IDs are relabeled1024..1031, so use checked draw mapping and immutable
hardlinked dense61view, not unchecked numeric-index aliases or newreferences.

Require CPUconfiguration/bank/label/reference/storage and full1032bankGPU25step
FP32finite/IR/memory plus independentCPUreplay gates before coldlaunch. Reject
changing architecture, priors, point density, optimizer, seed or default while
isolating bank coverage. More inputs need not improve reused8development-error;
no fresh-test or multi-seed claim. Relation: confined-ch-width192-bank-growth.

## 2026-10-31 — Explain duplicate cold compilation without perturbing bank control

Preflight compiles541.6s, then the separate productionprocess firstupdate spends
~549.7s including compile; original192/512firstupdate504.2s. Actual JAX0.10.1
persistent cache directory isNone, so cache-enableTrue alone does not build a
cross-process cache. Fullstep network/PDEJVP/parametergrad/SOAP compilation and
GEMMfusion alarms are verified; exact codegen/autotuning/SOAP cost not isolated.
PaddedNx/degree also changes with bank growth, preventing unchanged-shape reuse.
Record the overhead and continue the current experiment after compile finishes;
reject restarting the nowrunning control or changing compiler/SOAP/numerics just
to improve startup. A future shared compiled entry/persistent-cache improvement
must verify identicalHLO/cache hits and fresh model/optimizer state; cache setup
alone does not guarantee reuse across different programs. Reference: official
JAX persistent-compilation-cache documentation and local installed source.
Relation: confined-ch-width192-bank-growth, status/compile explanation.

## 2026-10-31 — Assess online geometry inputs without changing the active control

User defers a joint larger-network/larger-bank experiment until the current
192/1024cold50k finishes and asks about continuously sampled geometry inputs.
Keep that run unchanged. Distinguish geometry/operator/known-input generation
from reference trajectories: online physical training needs no evolved labels.
Historical six-worker512input preparation takes129.56s, but its cached metric
is not the current linear-stable metric rebuilt during loading. A lightweight
CPU-only three-input/nine-build production metric probe measures0.516s/build;
combine stage costs only as an estimate, not a measured online pipeline speed.

For later evaluation, prefer asynchronous generation into a bounded refreshed
pool with reuse over a synchronous all-fresh B16 each update; the estimated
producer throughput is far below GPU input-use throughput at current resources.
Visible96logical CPUs have a12CPU cgroup budget, so do not promise unlimited
worker scaling. New values need fixed padded shapes/dtypes and explicit overflow
handling, not per-geometry JIT recompilation or silent clipping of the prior.
This is a deferred design proposal, not an implemented/default sampler or proof
of lower validation error. Relation: confined-ch-width192-bank-growth.

## 2026-10-31 — User requests25k continuation with asynchronous rolling geometry inputs

User explicitly supersedes waiting for the192/1024cold50k endpoint: stop that
run and import checkpoint25000 into an online-geometry continuation. Owned
runner339235/trainer346501 stop by SIGTERM06:08:33Z, last26900logged/26500evaluated;
preserve all artifacts and close that control by user switch, not50kcompletion.
Restore the paired25kmodel and SOAP state including schedule counters; retain
the existing total50kbudget, all physics, architecture, loss metric and reference
cohorts. No learning-rate restart, weight-only warm start or default promotion.

Implement optional input-only asynchronous CPU generation and bounded rolling
pool; initially1024old inputs, replace32slots every100updates with6workers and
bounded prefetch. Consume a deterministic generated order at fixed refresh
steps, waiting explicitly if necessary rather than letting completion timing
choose training inputs. Separate frozen16monitor/8validation payloads from
replaceable training slots. Use fixed declared padding with hard overflow errors,
no biased retry/clipping for an oversized valid geometry. Never solve training
PDE/FEM trajectories or store an unbounded new bank. Extend strict continuation
only for an explicit paired-checkpoint path and declared online sampling option;
preserve legacy complete-parent behavior and direct-path provenance. Verify full
state import, PDE/operator/metric parity, label isolation, deterministic refresh,
actualGPUfinite/headroom and measured producer throughput before full launch.
Relation: new confined-ch-online-rolling-pool task, explicit user scope switch.

## 2026-10-31 — Gate rolling inputs in the production process and retain compiler limits

CPU production producer64input/6worker gate passes at5.66inputs/s, exact paired
25kstate/counter25000 and two legacy-mesh comparisons pass. Gmsh's copied .venv
Python binding lacks its native symbol, so use the existing explicit meshing
interpreter for Gmsh/NumPy only; no DOLFIN/FEniCSMPI integration or package change.
Workers explicitly set JAX platform configuration toCPU, in addition to process
environment, because JAX may be imported before a spawn initializer runs.

Use the first25actual updates of the same canonical production process for
GPUfinite/precision/headroom/timing gates, retaining those valid updates. Reject
another separately compiled throwaway probe, which duplicated~9min startup in
the previous control. Shapes1536/128 support online variation with strict overflow
failure; they are larger than the old1353/97empirical bank maxima. Zero-padding
preserves the physical graph/objective, but changed shapes can change FP32
reduction kernels and SOAP trajectories. Initial96score parity is checked;
that alone does not make historical fixed-bank versus online results a fully
isolated sampling ablation. Such attribution would require matching padded
shapes and continuation budget; no extra control is authorized or queued here.
Relation: confined-ch-online-rolling-pool implementation and experiment limits.

## 2026-10-31 — Fresh width256 rolling-pool run and approved snapshot cleanup

User explicitly requests cold50000updates with width256 and online rolling
inputs after replacing GPU with RTX6000D/85651MiB. Previous192online process
exited after32900logged/32500evaluated; close by scope switch, retain paired32000
and all online artifacts rather than claiming completion or restoring its state.
Reuse only known geometry/operator inputs in the initial1024pool; allow explicit
origin_step0 without continuation and retain deterministic fresh-geometry stream,
fixed evaluation payloads, physics, FP32ordinary, SOAPbeta2.9 and batching. Reject
weight-only warm starts and unmeasured default promotion.

Before deleting, present exact249-file plan in three inactive fixed-bank runs;
user approves it explicitly. Delete only paired nonessential snapshots and their
markers after checking unchanged inode/size/mtime and all13protected groups.
Measured6166249472bytes recovered. Keep all logs/config/source/initial/final
weights and every data/reference file; no broad directory/cache deletion. New
run checkpoint_every5000 limits future growth to10paired groups, plus final and
initial states, without deleting new checkpoints. Relation: confined-ch-width256-
online-cold50k, cleanup proposal/receipt and archived previous online task.

## 2026-10-31 — Plot explicit history coverage beyond25k

The current256online cold run passes25k, where the192online continuation adds
a fifth exact same-budget record; the earlier four-row plotting assertion is
therefore invalid. Determine per-run availability from recorded first/last
global steps, require exactlyone record for any target inside that interval,
and report stages notyetstarted or alreadyended explicitly. Keep eachcurve at
its observed start/end and clip to requestedbudget, with no interpolation or
last-value extension. Require the currentrun's endpoint record. Preserve full
raw histories and add a recent8kzoom from the same points. Reject silently
dropping stopped controls or fabricating an onlinecoldprefix before25k.
This changes research reporting only; training/config/physics/default unchanged.
Relation: confined-ch-width256-online-cold50k, convergence refresh past25k.

## 2026-10-31 — Close cold50k at the fixed budget and show field-error strata

The fresh256online run completes50000updates and independentCPU actualinput/
RNG/state/24field replay passes. Report model_final.eqx at the prescribedbudget,
not the lower49.5kdevelopment-score record; no heldoutcheckpoint selection or
defaultpromotion. User requests physicalfields. Select low-error, uppermedian
and maximum-error geometries from the same8reserved development cases, label
this error-stratified selection explicitly, and show all five savedtimes with
reference/prediction/absolute-error rows. Include the worstcase rather than
claiming allterminalerrors below10%.

Reuse original-mesh area-weighted display interpolation and analytic hole masks,
with shared phase/error scales across all8cases. Recompute scores on unchanged
cellvalues/areas, not displaypixels. SavedFP32 hardIC differs from FP64reference
by rounding; bound the initial absolutegap by4compute-dtypeeps and record the
measured gap. Keep previousFP64 gallery semantics separate, no changed training
math or regenerated reference. GeneratedPNG/PDF remain external toGit.
Relation: confined-ch-width256-online-cold50k completion and field gallery.

## 2026-10-31 — Promote the accepted online geometry-operator result

The user accepts the completed width256 online cold50k physical fields and asks
to retain this result as the case's SOTA. Promote the opt-in canonical operator
preset to that exact cold numerical recipe, with the fixed50000-update final
model as its recorded checkpoint. Preserve the separate single-IVP30window
default and the original mild operator as an explicit historical preset. Move
historical inheritance to the preserved baseline before redirecting the public
operator config, avoiding a circular dependency or silently altered controls.

Reject selecting the lower49500 development metric, changing the physical
problem/optimizer, running new training, renaming external artifacts, or treating
this single-initialization reused8geometry native comparison as a fresh local-FEM
or statistical stability result. Generated media/weights stay outside Git and
all artifact loading remains by direct path. Relation: confined-ch-operator-sota;
supersedes the prior no-promotion status for this operator only.

## 2026-10-31 — Separate rolling residual-point pools from condition sampling

The user asks to extend bounded background production to the fixed-domain
electropolish/melting core+halo sets. Source inspection confirms both canonical
NOs already draw fresh initial conditions but cycle32static spatial subgraphs.
Exact existing-RNG coverage is3938/4949free electropolish nodes and8172/2097152
melting nodes as PDE cores. Halo nodes participate in network/gradient evaluation
but do not contribute their own residual rows. This motivates a spatial-sampling
experiment, not a diagnosis that the existing low-error models are invalid.

Recommend a shared bounded input-producer/scheduler with a fixed-graph subgraph
producer: reuse immutable full graph/weights/BCs, draw uniform cores, gather the
complete residual dependency neighborhood, and refresh query/edge/mask metadata
together. Preserve all-times shared spatial preparation and online condition/time
streams. Electropolish query-to-curve nearest-index data must follow changed node
IDs; precompute that mapping on the full fixed grid. Learned encoder/prepared
features must still be recomputed inside parameter AD.

Reject blindly reusing the observed32entry padding maxima: new subgraphs can
exceed them and the old generic padding helper has truncation branches. For the
current six-neighbor melting graph, K256 gives proven caps1792nodes/1536edges;
use explicit fixed dtype/shapes, complete edges and hard overflow validation.
Generic residuals may require more than one halo layer. Consume predetermined
draw IDs independently of producer completion order, with separate PRNG streams
and bounded queues. Measure full producer/transfer/wait/compile costs before
claiming negligible slowdown or improved L2. Current RandomState core-only draw
is~41ms on the2Mnode grid, so wrapping the existing builder without examining
sampling/remapping is not automatically cheap; the alternative CPU selection
simulation is not a full pipeline benchmark. No runtime/default change or new
training in this assessment. Relation: core-halo-rolling-assessment.

## 2026-10-31 — Compare electropolish full/fixed/rolling residual sampling

User chooses electropolish and requests time/accuracy for allcoordinates, a
precomputed pool, and a rolling pool. Use three cold5000-update current NO
controls (Adam/B8/Nt4/width64/ordinaryFP32), one untuned common runtime seed,
identical model initialization, IC/BC/query condition and time streams, five
physical losses and LR. Preserve the original model, residual and defaults.
Uniform cores remain sampled on the original graph then Dirichlet cores masked.

Match fixed/rolling padding to a proven one-hop degree bound rather than old
observed maxima. Keep the old first32core draws, then deterministic fresh draw
IDs independent of worker completion. Background workers produce NumPy-only
geometry metadata, never initialize a GPU or cache learned network features.
Full mode uses complete free residual rows and the original graph weights.
Check subgraph core values against the full graph before training. All times
retain shared spatial queries and JVP. Separate producer/packing/transfer wait,
compilation, setup, warm updates and end-to-end cost; common padding may cost
more than the historical tight pool, so do not treat the old timing as matched.

Reuse existing64native/localFD reference inputs and fixed final checkpoints;
no heldout selection or new PDE/FEMtraining trajectories. Report phi and c,
pooled and terminal, with native and common local errors separated. Reject
claiming single-run statistical superiority or automatically promoting a new
default. Relation: electropolish-spatial-pool-comparison.

## 2026-10-31 — Retain GPU sampling measurements with a separate replay issue

All3matched cold5000runs, actual draw/state/time/condition parity,64native/local
evaluations and independent saved-field metric calculations pass. Full has the
lowest pooled mean error; rolling is slightly better terminally and~14%slower
than fixed in warm effective steps. Differences are small and one initialization;
leave SOTA defaults unchanged. Fixed/rolling share proven safe padding.

The additional CPU checkpoint audit fails at0.3167absolute despite exact inputs/
parameters. Source/NumPy probes localize the first large mismatch to composed
cross-attention; isolated identical-q/kv attention agrees CPU/GPU. Exact cause
unresolved; explicit/captured model, FP32/FP64, cuDNNFMHA/Triton flags and q/kv
barrier probes do not resolve it. Record an open reproducible portability issue,
retain all failures, and restrict this comparison to the verified GPU execution.
Reject silently relaxing the2e-5CPUgate, calling it passed, altering the neural
architecture/compiler/physics just to complete this sampling ablation, or
claiming backend-independent/seed-stable accuracy. GPU checkpoint replay passes;
the requested empirical time/accuracy comparison is complete, with this material
limitation explicitly linked. Relation: electropolish-spatial-pool-comparison and
issues/electropolish-attention-replay.md.

## 2026-10-31 — Compare melting spatial pools without changing the3Dproblem

User requests full/fixed/rolling controls analogous to electropolish. Preserve
canonical128^3graph, ellipsoid distribution, hardIC/latentL1PDE, Adam/B16/Nt8 and
shared time/spatial preparation. Fixed/rolling share proven1792node/1536edge
capacity; background workers prepare NumPy-only metadata in deterministic order.
Use an efficient uniform without-replacement core generator rather than a full
2Mnode permutation per fresh draw, and document its RNG change explicitly.
Both controls must share that generator and initial32sets. Rejected silently
truncating halos, changing the grid/horizon/physics or treating random full-grid
subsets as a full objective. Full training may accumulate exact mean-loss
gradients over complete disjoint spatial chunks before a single Adam update;
measure memory/cost and check small-grid equivalence first. Long-run scope must
be communicated from actual measurements, not inferred from2Dcost. Original
defaults remain unchanged. Relation: melting-spatial-pool-comparison.

## 2026-10-31 — User limits melting comparison to the two core+halo pools

Exact full128^3/B16/Nt8 probe finishes3updates,64complete32768core chunks per
update:15.44s warm effective/38.314GiB activepeak;5000estimated21.4hours. User
explicitly declines full training and requests fixed versus rolling only.
Retain the full probe/optional implementation as performance evidence, not a
full-budget accuracy control. Run both sampled modes cold5000with identical
model/Adam and actual condition/time streams, no other numerical changes.
Focused FP64 full loss/gradient/one-Adam identity passes. Initial FP32 loss/
gradient agreement but strict near-zero-gradient Adam step failure is retained;
do not claim bitwise trajectories from different accumulation orders.
Relation: melting-spatial-pool-comparison; supersedes3full-budget acceptance.

## 2026-10-31 — Retain measured melting pool results and record query sensitivity

Both matched5000cold controls and initial-state/actual IC/time/core/saved-field/
same-GPU1024query checkpoint checks pass. Fixed/rolling native8meanphase pooled
0.1631/0.4217%, terminal0.2218/0.2514%; warm12.737/13.247ms, about4%rollingcost.
Coverage0.3898->45.6869%doesnot yield lower final whole-field error in this pair.
Histories fluctuate and a single terminal selection doesnot establish stable
ordering or a unique cause; retain the canonical default.

An extra full-spatial PDE validation fails field invariance:65536versus1024
queries differ0.00577absolutephi at identical selected points. Input images/
initial latents match exactly; explicitencoded, inlineencoded, captured/dynamic
images and JVP/noJVP probes retain the gap. Cause unresolved. Preserve the failed
2e-5gate, record a separate reproducible issue and do not use those full-spatial
residual scalars to rank models or claim partition-independent precision. Do not
change production model/compiler/physics to finish this sampling comparison.
Measured same-path GPU results remain reviewable; no further work is queued.
Relation: melting-spatial-pool-comparison and issues/melting-query-chunk-consistency.md.

## 2026-10-31 — Publish accumulated operator work and diagnostic records

User explicitly requests committing and pushing current code and records.
Include accumulated confined-CH geometry conditioning/online rolling/SOTA,
electropolish/melting spatial controls, compact validation evidence and open
replay/query-consistency issues.335current text files total~3MiB; preserve
generated-media/array/weight exclusions and existing external artifacts. Use
existingdev (same tree as startingmain), commit -> regularmainmerge -> push
origin/main only. Reuse completed unchanged numerical checks; add syntax/schema,
staged-content and whitespace checks. Record accurate published refs in a final
memory update and publish it through the same workflow; do not self-reference
the containing commit. No new training/default tuning, deletion, history rewrite
or pushdev. Relation: current-code-publication.

## 2026-10-31 — Verify publication and close the memory record

Published code/record commit ae467463399994edd36d9b0f163a6c2c7f5f3a88onlocaldev
via regularmainmerge c359d7f80f39194e1eb66b6719f4b16a8d81ba07; origin/main exact
equality verified, main/dev trees identical and worktree clean.337changed text
files, no binaries;1851indexedfiles20.83MiB.148AST/156strictJSON/45config factory
checks and staged content/whitespace pass. Only2formatting-only source/Markdown
fixes, checked identical AST/texttokens. Reuse prior numerical gates including
their explicitly recorded unresolved failures; no retraining/default changes.
Archive the completed task and publish this accurate receipt through the same
devcommit -> regularmainmerge -> pushmain workflow. Do not embed the containing
commit's own ID or rewrite history to add it later. Relation: current-code-publication.
