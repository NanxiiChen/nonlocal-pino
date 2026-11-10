# Project Progress

## Active task

None.

## Deferred work

The old high-mobility horizon process group618216 remains explicitly paused.
Recovery details are preserved in the archived physical study below; do not
automatically resume or discard it.

Phase-field/displacement boundary-conditioned precipitate example is recorded
but explicitly deferred by the user; no implementation, training or reference
solve. [Feasibility](tasks/archive/traction-conditioned-elastic-pino-analysis.md),
[concept sketch](tasks/archive/elastic-boundary-response-sketch.md).
The channel20000/lower-LR study is complete and archived below.

## Latest completed validation

[GitHub Pages repository rename](tasks/archive/pages-repository-rename.md).
The public report uses `/nonlocal-pino/`. The README and all nine generated
pages use the renamed repository, with 308 relative references, eight
source links and 60 font references checked in the local build.

[Electropolish 10,000-step default and publication](tasks/archive/electro-operator-10k-publication.md).
The operator preset uses width 96, 16 shared times and 10,000 Adam steps.
The README and report show 1.740% pooled and 1.540% terminal phase L2 on
64 local FD101 cases and reuse the updated 404,017-byte phase gallery.
Historical control configurations, checkpoint loading, ten focused
tests and the nine-page report build pass.

[Electropolish temporal sample count and reference audit](tasks/archive/electro-time-reference-audit.md).
Paired 5,000/10,000 checkpoints reach 2.483%, 2.555%, and 1.740% pooled
phase L2 on 64 local FD101 cases for 4/64, 16/64 and 16/96 respectively.
Equal-width sampling differences at 5,000 are small and change sign across
seeds. Native and local FD refinement checks preserve the old run ranking.

[Electropolish time sampling, width and training budget](tasks/archive/electro-time-sampling-width.md).
At 5,000 steps, 16 times raises pooled phase L2 from 3.985% to 4.612% at
width 64. Width 96 recovers 3.897%. The separate 10,000-step four-time
control gives 4.015% on the same 64 local FD101 cases.

[Electropolish temporal conditioning ablation](tasks/archive/electro-temporal-conditioning.md).
Matched cold 5,000-step runs on 64 local FD101 targets give pooled phase L2 of
3.985% for decoder FiLM, 5.908% for query FiLM, and 8.085% for joint queries.
Training draws and all saved-field norms were independently checked.
The [case README update](tasks/archive/electro-time-injection-readme.md) records
the retained architecture, the attention query count, and warm step timing.

[Attention and temporal FiLM ordering audit](tasks/archive/attention-film-order.md).
The published figure matches all five training paths. Spatial attention is
computed before the three time-FiLM layers. The Pages SVG was checked directly.

[Temporal FiLM Pages release](tasks/archive/publish-temporal-film.md).
The release build contains both three-layer decoder panels and the rendered
layerwise FiLM equation. The existing Pages workflow publishes these assets.

[Temporal FiLM clarification](tasks/archive/temporal-film-diagram.md).
The local diagram and report show three individually modulated decoder layers
and a separate output head. The build and four-width checks pass.

[Two-stream architecture diagram](tasks/archive/two-stream-architecture-diagram.md).
The overview and architecture chapter show both weighted context reads, their
shared query, and their per-block average. The build and four-width checks pass.

[Point-operator architecture clarification](tasks/archive/point-operator-diagram.md).
The local report distinguishes pointwise encoder lifts from weighted spatial
cross-attention. The SVG, affected pages and KaTeX build pass validation.

[Technical report GitHub Pages release](tasks/archive/research-report-pages.md).
The nine-page report is live at the [project site](https://nanxiichen.github.io/nonlocal-pino/).
The first Actions build and deploy succeeded, and the public homepage and all
84 published files match the locally built artifact byte for byte.

[Nonlocal PINO technical report site](tasks/archive/research-report-site.md).
The selected technical-reference design builds nine static chapters under
`website/dist/`, with local KaTeX typesetting and five complete cases. The locked
build and four-width Chromium checks, including offline fonts, pass.

[Research-site layout previews](tasks/archive/research-site-prototypes.md).
Three local designs reused the five primary case figures and passed desktop
and narrow-screen Chromium checks. The technical report supersedes the chooser.

[Six-particle corrosion homepage publication](tasks/archive/publish-six-particle-corrosion.md).
The 350 s case appears in the root results and training strategy tables. The
release includes the documented width-256 recipe, reproducibility sources and
curated figures. All 107 particle-corrosion CPU regressions, saved preset
parity, 79 frozen source comparisons and documentation checks passed.

[Accepted six-particle corrosion SI study and documentation](tasks/archive/particle-corrosion-physical-optimization.md):
width256/Adam cold20k on50x30um/350s, original joint residuals and rolling
geometry/core-halo inputs. Same4development cases: local phase2.933%/5.757%,
native1.450%/2.781%; modest improvement over192 at22% longer run and20.69GiB
active peak. All input/field audits pass. README/root overview now present the
accepted setup, explicit recipe and separate native/local metrics. Centralized
PNG(first3layouts,5times)+SVG total242287bytes; other cases preserved. All146
local links,79frozen numerical-source files,AST/JSON/XML and figure budgets
pass. All older
controls and the unmet2% aspiration remain qualified in the archive. No numerical
default change, new solve/training, Git action or cleanup.
[Compact record](tasks/archive/particle-corrosion-physical-optimization.json).


[Physical particle case and strategy audit](tasks/archive/particle-corrosion-physical-pilot.md):
actualL2e-6,Lc50um,Tc5s; derivedgroups1.7/1070/176/.103,50x30um region.
All5owner stages complete; final8localFEM phase35.4226%pooled/46.8794%terminal,
c34.7789%/45.6977%. First same-production-native phase33.6062%/46.2413% shows
neural convergence failure in addition to native/local discretization error.
22.94min,49.04msmedian,15.3945GiBpeak. All20kdraws/64finalslots audited; all8
saved-field metrics independently recomputed.24focused tests retained.
Both particle and successful electroPINO use joint originalL1/Adam/no causal/
unitPDEscales. Differences include cLe-normalized IC/BC and loss grouping,
faster LR decay,architecture,batch and geometry/operator measures. Early
interface residuals remain large despite small mouth errors. Physical-default
change is not a SOTA promotion; old dimensionless results remain separate.
[Compact receipt](tasks/archive/particle-corrosion-physical-pilot.json).

[Particle dimensional audit](tasks/archive/particle-corrosion-dimensional-audit.md):
effectiveN3/Nch disagrees with electropolishL*alpha/D by545.294times, independent
of characteristic scales. Holding electroA,omega,alpha,D yields Lc51.3179um,
Tc247.861s(original)/495.722s(speed2), requiring different actualL1.83387e-9.
Full identical material mapping has no solution; AC-only.454545s is invalid as
a complete interpretation. Forward/inverse scalar checks, independent SI/PINO
residuals(max1.1e-14),2xRHS/Jacobian and four saved-run config checks pass.
README unit errors corrected; numerical parameters/defaults remain unchanged.
This identifies missing calibration, not a proven cause of neural collapse.

[Particle-corrosion32random times](tasks/archive/particle-corrosion-time32-diagnosis.md):
node-aligned speed2 cold20k gives reused8phase41.2407%pooled/63.2984%terminal,
versus8times41.2232%/63.2892%; no useful improvement.23.33min,49.09msheld,
15.68GiB activepeak. Boundary targets align; independent native early solution
expands, but PINO still collapses and early interface PDE residuals remain high.
All20000draws/64finalinput slots audited; final8metrics recomputed, five owner
stages pass. Fresh seeds limit attribution. Original1xdefault retained.

[Twice-speed particle corrosion](tasks/archive/particle-corrosion-speed2.md):
t_max1 with all four rates doubled gives2.0514times the mean final corrosion
area, qualified by first-case FEM space/time refinement and eight overlap-time
identities. Cold20k phase42.0821%pooled/65.8807%terminal; aligning soft mouth
losses to operator nodes removes a large observed BC gap but a second cold20k
still gives41.2232%/63.2892%. Early interface PDE residuals remain large. Both
full online audits and all19focused tests pass; negative fields/results retained.
Each run about17min/4.92GiB. Default remains original1x rolling configuration;
fresh seeds/reused8cases limit attribution. No Git action, cleanup or pending jobs.

[Published particle corrosion](tasks/archive/particle-corrosion-publication.md):
current rolling core/halo default and full-grid comparison documented, with a
three-layout setup SVG and four-layout/five-time phase evolution PNG from the
default model. New images269387bytes combined; existing ten images unchanged.
All8saved-field metrics,53local links,11case/518artifact directories and renderer/
image/diff checks pass. All33numerical source files match the completed audited
run; prior42focused/regression tests reused. Dev896bbca merged normally into
maina27ce31 and pushed; independent remote lookup and clean-worktree check pass.
Related pending particle source/tests/records are included. No weights/raw
fields/logs copied; no new solves, training, numerical-default changes or deletion.

[Whole-mesh versus rolling core/halo](tasks/archive/particle-corrosion-full-grid.md):
samewidth192/B4/Nt8/Adam20k,coldstate,380geometry stream and IC/BC/time draws.
Full-grid same8localFEM meanphi1.7451%pooled/2.8658%terminal versus1.9827%/3.1209%;
c2.1093%/3.4481% versus2.3964%/3.7405%. All8phase pooled errors improve,7terminal.
Held32.53vs22.56ms,total1065.80vs959.04s(+11.13%),activepeak7.18vs4.93GiB;
producerwait25.54vs158.82s is already included. Full20kidentity/64final-input
replay and independent128initial/final whole-mesh CSR/volume/mouth-padding audits
pass.16newtests+26legacy regressions pass. Core/halo default and prior figures
remain unchanged; full is a completed comparison, not automatic promotion.

[Rolling particle-corrosion inputs](tasks/archive/particle-corrosion-online.md):
default now uses a64slot background geometry/core-halo pool, same8particle
physics/width192/Adam20k. Invalid candidates regenerate without aborting training;
the exact formerly failing draw251 now installs successfully. Cold20krun02 and
all audits complete:316newgeometries,1520spatialrefreshes,380sampledgeometries,
all64final slots rebuilt exactly, no evaluation overlap/labels;33focused tests
pass. Same8reused localFEM cases give phase1.9827%pooled/3.1209%terminal and
concentration2.3964%/3.7405%. Fixed pooled errors remain slightly lower; terminal
errors improve. Keep old fixed preset, lowest-pooled result and labeled figure.
Held22.56ms,total959.04s,activepeak4.93GiB;164.31sonline overhead includes158.82s
producer wait. Different padding and one paired initialization limit attribution.

[Eight-particle geometry PINO](tasks/archive/particle-corrosion-pino.md),
[compact receipt](tasks/archive/particle-corrosion-validation.json): eight smooth
inert particles, four close to the same initial pit. Historical fixed nonlocal width192/
Adam20000 gives reserved8 phase1.9587% pooled/3.4236% terminal and concentration
2.3510%/4.0898% against independently refined localFEM. Allowed3%overall fallback
met;1%unmet. Full-domain residual audit still finds wall/late-time tails. Matched
localAD20k collapses (97.2665%phase); derivative/state/sampling audits pass, so
this is failed optimization, not intrinsic AD inaccuracy. Held updates19.69/
28.52ms, total747.92/1236.68s on5090.22focused tests pass, cohorts64/4/8disjoint,
all76conditioning inputs/all20kdraws replayed, saved metrics reproduced. One
compact4samplefigure; this historical task's pending code/records are now included
in the particle publication above, whose figure shows the rolling default.

[Matched heterogeneous corrosion controls](tasks/archive/heterogeneous-corrosion-local-ad.md):
four new cold10000 runs and55checkpoint evaluations against one local FV bank.
Adam nonlocal/localAD phase1.227%/.872%, terminal2.086%/1.516%; uncontended warm
14.60/17.92ms, peaks3.93/3.02GiB. Same initial state and every input draw verified.
Both SOAP1e-3 controls fail by saturation; SOAP3e-4 remains22.67%phase. Preserve
negative controls and current defaults. Ten new focused tests and complete
input/field audits pass; [compact audit](tasks/archive/heterogeneous-corrosion-control-audit.json).
One paired initialization/four heldout maps; AD adds explicit wall losses and
cold costs partly overlap. No pending jobs, Git action or deletion.

[Heterogeneous material-map corrosion pilot](tasks/archive/heterogeneous-corrosion-pilot.md):
fresh online positive kinetics, fixed pit/geometry, activation-dominated KKS.
Mild cold5000 native/local phase0.795%/1.010%. Stronger kappa0.5..4 with zero
latent constant mode, width96/Adam cold10000: phase0.865%/1.227%, terminal1.306%/2.086%.
Four held-out inputs only; material-centered endpoint error8.31%, reference/PINO
mirrorasymmetry15.08/14.96%. Independent FV refinements,11focused tests and fullgrid
residual audits pass; early concentration transport and unvisited boundary cores
remain imperfect. Explicit presets/records,380s finalrun,3.93GiB peak. No Git action.

[Promoted20kdefault](tasks/archive/channel-20k-sota-promotion.md):
user-selected actual20kendpoint and all3predeclared reserved cases. Native
phase errors1.9860/1.4293/1.5152%; new curated PNG275257bytes and four-PNG total
1878013bytes. Other three PNGs/provenance unchanged. Current default exactly
matches saved20kconfig apart from runtime seed/run name; old10kcontrols retained.
47local links,9case/452artifact roots, JSON/AST/image/diff checks pass.

[Cold20k endpoint comparison](tasks/archive/channel-cold20k-and-four-pino-setups.md):
20k final development phi1.4496%,terminal1.9020%,u1.3312%,p4.6703%; same3reserved
means phi1.6435%,terminal2.1961%,u1.5172%,p5.2840%,minority4.3306%,mu13.9541%.
Against old10k reserved means: phase10.7%and velocity24.7%relative reductions;
terminalphase3.3%and pressure2.4%. Some cases worsen; pressure worst9.0316%->6.5885%.
Original owner stopped at240/18diagnostic-count rejection after successful
training. Separate explicit252probe completion finished all8remaining stages;
original failure journal preserved. Future18route now records252; legacy20/8
retain240. Exact saved-field paired replay and105source bytes pass. Train
wall9.1627h; subsequently promoted as current default by explicit user request. One initialization,
three reused reserved cases; changedLR/budget and36vs40times are coupled.
Direct completion: /root/autodl-tmp/data/channel_nsch2d/cold20k-restart-nt36-time18-20261103-04/final-completion-20261103-210103/completion.json.

[Random smooth-wall NS–CH operator](tasks/archive/channel-random-wall-pilot.md),
[compact receipt](tasks/channel-random-wall-pilot-validation.json).
Previous10kselection used width192/B8/SOAP10000/ordinaryFP32,
integral spatial derivatives, physical timeJVP40 and online rolling geometry
inputs. Timeblock20on PRO6000uses88.06GiB peak and explicit97%allocator; README
includes smaller block8/block2 presets. Automatic fresh run names; runtime seeds.

Cold10000and all ten owned final stages exited0;101source bytes checked before
promotion. Nominal phi2.6781%,terminal4.0550%,u1.6700%,p5.8931%,mu14.1030%.
Three reused reserved cases averagephi1.8396%,terminal2.2708%,u2.0160%,p5.4118%
(worst9.0316%),minority4.8474%,mu15.9168%. Terminal areaIoU86.14/94.25/93.64%.
This is finite-cohort evidence, not population/cross-seed validation. Unsaved
8500scalar phase minimum2.0476%is not a checkpoint. Pressure response to geometry
has larger relative error than the total pressure field; preserve this limit.

Default/saved-run config parity, all55historical presets,4focused tests and
complete raw convergence figures verified. Full records and external paths are
in the archived task. Source/compact records prepared for the user-authorized
local dev commit only; no merge/push, generated weights/images or cleanup.

## Project and publication

The public repository is `NanxiiChen/nonlocal-pino`, with `main` as its default branch.
[Split validation/recovery](migration/VALIDATION.md);
[completed split task](tasks/archive/project-split-validation.md).
Preserve existing external data/run paths. Git history changes require explicit
current-turn authorization; the documented workflow pushes main only.

## Recent completed tasks

- [Physical particle-corrosion cold20k and electropolish strategy comparison](tasks/archive/particle-corrosion-physical-pilot.md): corrected SI setup, qualified reference, failed neural evolution and exact scheme differences preserved.

- [Particle-corrosion dimensional coefficients and conditional physical time](tasks/archive/particle-corrosion-dimensional-audit.md): valid2xrescaling, but different material regime from electropolish; full derivation and reproducible CPU audit retained.

- [Particle-corrosion boundary audit and32random-time control](tasks/archive/particle-corrosion-time32-diagnosis.md): completed20k, no accuracy recovery from increasing uniform time samples alone.

- [Twice-speed reference validation and two cold PINO trials](tasks/archive/particle-corrosion-speed2.md): reference grows correctly; node-aligned BC fixes a sampling gap, but early-time PDE convergence remains unresolved.

- [More corrosion within normalized unit time](tasks/archive/particle-corrosion-faster-normalized-time.md): propose multiplying all AC coefficients and concentration diffusion by2 with kappa2/t_max1 unchanged; RHS/Jacobian scaling verified, no new morphology result or configuration change.

- [Particle-corrosion time horizon and kinetic coefficients](tasks/archive/particle-corrosion-time-kinetics.md): verified dimensionless0..1, uniform kappa2, n_ch80; no seconds/mobility calibration or parameter changes.

- [Published particle-corrosion setup, results and illustrated README](tasks/archive/particle-corrosion-publication.md).

- [Whole-mesh particle-corrosion PINO versus rolling core/halo](tasks/archive/particle-corrosion-full-grid.md).

- [Rolling particle geometries/core-halo pools and rejection recovery](tasks/archive/particle-corrosion-online.md).

- [Stronger random ceramic-particle corrosion PINO and matched AD controls](tasks/archive/particle-corrosion-pino.md).

- [Published heterogeneous corrosion and smooth particle inputs](tasks/archive/heterogeneous-publication-and-particle-preview.md).

- [Current corrosion KKS relation versus legacy bounded constraint](tasks/archive/heterogeneous-kks-constraint.md).

- [Nonlocal internal-wall advantage and confined-CH reuse](tasks/archive/nonlocal-internal-wall-advantage.md).

- [Inert ceramic particles as corrosion geometry](tasks/archive/inert-particle-corrosion-geometry.md).

- [Heterogeneous corrosion: paired SOAP and local-AD controls](tasks/archive/heterogeneous-corrosion-local-ad.md).

- [Activation-controlled heterogeneous corrosion implementation and stronger material prior](tasks/archive/heterogeneous-corrosion-pilot.md).

- [Heterogeneous pitting-corrosion candidate](tasks/archive/heterogeneous-pitting-candidate.md).

- [Obstacle-conditioned two-phase flow candidates](tasks/archive/obstacle-two-phase-candidates.md).

- [ATG boundary loading and morphology](tasks/archive/atg-boundary-morphology.md).

- [ATG interface concept sketch](tasks/archive/atg-interface-concept-sketch.md).

- [Two-phase interfaces with simple elasticity](tasks/archive/twophase-linear-elastic-candidates.md).

- [Temperature versus initial-grain parameterization](tasks/archive/grain-temperature-versus-initial.md).

- [Grain migration time and loading interpretation](tasks/archive/grain-migration-time-loading.md).

- [Grain-boundary versus corrosion/fracture multiscale explanation](tasks/archive/multiscale-interface-distinction.md).

- [Loaded polycrystal RVE concept sketch](tasks/archive/polycrystal-rve-concept-sketch.md).

- [Phase-field RVE operator recommendations](tasks/archive/phasefield-rve-operator-candidates.md).

- [Twin-model equations and feasibility](tasks/archive/twin-model-equations.md).

- [Twin boundary-condition concept sketch](tasks/archive/twin-boundary-concept-sketch.md).

- [Visually distinctive phase-field/mechanics candidates](tasks/archive/mechanical-phase-morphology-candidates.md).

- [20k SOTA code/docs/image publication](tasks/archive/channel-20k-sota-publication.md).

- [20k channel SOTA/default and README image promotion](tasks/archive/channel-20k-sota-promotion.md).

- [Final20k physical fields on four geometries](tasks/archive/channel-final20k-field-gallery.md).

- [Cold20k endpoint comparison and four PINO setup diagrams](tasks/archive/channel-cold20k-and-four-pino-setups.md).

- [Displacement-boundary morphology-response concept sketch](tasks/archive/elastic-boundary-response-sketch.md).

- [Boundary-function coupled phase/displacement PINO recommendations](tasks/archive/traction-conditioned-elastic-pino-analysis.md).

- [Centralized README figures and verified publication](tasks/archive/pino-readme-assets-publication.md).

- [Four main PINO README summaries and field figures](tasks/archive/pino-readme-overview.md).

- [Random smooth-wall operator and default promotion](tasks/archive/channel-random-wall-pilot.md).

- [Richer random-wall channel design](tasks/archive/channel-random-wall-design.md).
- [Integral nonlocal channel control](tasks/archive/channel-integral-nonlocal-control.md).
- [GMLS versus nonlocal terminology](tasks/archive/channel-gmls-terminology.md).
- [Fixed normalization default](tasks/archive/channel-fixed-normalization-default.md).
- [Nine-term raw-residual adaptive weighting](tasks/archive/channel-allterms-gradnorm.md).
- [Local channel code checkpoint](tasks/archive/current-channel-code-checkpoint.md).
- [Current channel scheme inventory](tasks/archive/channel-scheme-inventory.md).
- [Slightly stronger channel constriction](tasks/archive/channel-stronger-constriction.md).
- [Channel timeJVP/resampling and nominal droplet-shape optimization](tasks/archive/channel-derivative-clarification.md).
- [Mild geometry-conditioned full NS--CH pilot](tasks/archive/channel-nsch-pilot.md).
- [Independent-wall NS--CH feasibility analysis](tasks/archive/twophase-independent-wall-nsch.md).
- [Constricted-channel concept sketch](tasks/archive/twophase-channel-sketch.md).
- [Parameterized two-phase geometry recommendation](tasks/archive/twophase-geometry-design.md).
- [Conditional operator architecture schematic](tasks/archive/conditional-operator-architecture.md).
- [Conceptual geometry explanation](tasks/archive/geometry-operator-concept.md).
- [Parameterized-geometry operator illustrated](tasks/archive/geometry-operator-html-explainer.md).
- [Current operator code/records published](tasks/archive/current-code-publication.md).
- [Melting spatial-pool comparison](tasks/archive/melting-spatial-pool-comparison.md).
- [Electropolish spatial sampling comparison](tasks/archive/electropolish-spatial-pool-comparison.md).
- [Rolling core+halo assessment](tasks/archive/core-halo-rolling-assessment.md).
- [Geometry-operator SOTA promotion](tasks/archive/confined-ch-operator-sota.md).

All earlier task links and full historical dashboard text are preserved in the
[dated snapshot](tasks/archive/project-progress-through-20261102.md). Detailed
completed results remain in the archived task and linked receipts.
