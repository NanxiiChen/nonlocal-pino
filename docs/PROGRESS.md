# Project Progress

## Active task

[Centralize and publish PINO README assets](tasks/pino-readme-assets-publication.md).
Move the four images to root assets/pino, update README/generator/provenance and
verify exact bytes/links/size. User explicitly authorizes this documentation
commit/push via dev -> main regular merge -> origin/main. Prior README work is
included. No numerical/default changes, new training or reference solves.

## Deferred work

Channel NS--CH: consider20000updates and a lower final learning rate, per the
user's explicitly deferred proposal. Exact LR and continuation-versus-cold-start
choice remain undecided. No run/automation is launched or queued. See
[the recorded idea](../examples/channel_nsch2d/README.md#deferred-experiment).

## Latest completed validation

[Random smooth-wall NS–CH operator](tasks/archive/channel-random-wall-pilot.md),
[compact receipt](tasks/channel-random-wall-pilot-validation.json).
Canonical channel train.py now defaults to width192/B8/SOAP10000/ordinaryFP32,
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
