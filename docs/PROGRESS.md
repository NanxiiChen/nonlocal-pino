# Project Progress

## Active task

[Publish particle corrosion README and figures](tasks/particle-corrosion-publication.md).
User authorizes recording the comparisons, a setup schematic and multi-sample
evolution figure, then commit/merge/push through dev→main. Preserve the current
rolling core/halo default, external artifacts and other examples' images.
No training or cleanup requested.

## Deferred work

Phase-field/displacement boundary-conditioned precipitate example is recorded
but explicitly deferred by the user; no implementation, training or reference
solve. [Feasibility](tasks/archive/traction-conditioned-elastic-pino-analysis.md),
[concept sketch](tasks/archive/elastic-boundary-response-sketch.md).
The channel20000/lower-LR study is complete and archived below.

## Latest completed validation

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
compact4samplefigure; historical images unchanged. Working tree not committed.

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

Independent repository: private `NanxiiChen/nonlocal-pinn`, default branch main.
[Split validation/recovery](migration/VALIDATION.md);
[completed split task](tasks/archive/project-split-validation.md).
Preserve existing external data/run paths. Git history changes require explicit
current-turn authorization; the documented workflow pushes main only.

## Recent completed tasks

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
