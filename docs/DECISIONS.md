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
