# Project Progress

## Current task

**Active:** [Electropolish15k library and time-fusion controls](tasks/electro-library-extension.md).

Finish five new cold15k controls (fixed64/128/256/512, then joint query), reuse
completed fixed32/online15k, update Electropolish-only library/error and temporal
cold-wall-time figures, README/site, then authorized dev→commit→main→merge→push
and verify Pages. Ceramic stability remains deferred; defaults are unchanged.

Completed recovery02 includes independent reloaded10k/15k inference and audits
on all64local-FD tests. Final channel means: online1.53351%, fixed3216.44550%;
phase1.39597%/15.65992%, concentration1.67106%/17.23107%. SampleSD0.580879/
3.639694percentage points; complete cold times1283.698/1281.388s.
The incomplete original CUDA-capture attempt remains preserved, not selected.

Expanded bundle:`/root/autodl-tmp/tf-logs/electro-15k-completion-20261116-01`.
All seven cold15ktrainings and independent10k/15k64case audits are complete.
Fixed32/64/128/256/512means16.44550/12.27110/5.45571/2.12162/2.02321%; online
FiLM1.53351%, jointquery1.69628%. Paired complete cold times1283.698/2146.986s.
No training/figureowner remains active; numericaljournalcomplete, figurequeue
ready_for_visual_review, and all threePNG/PDFpairs have now been visually reviewed.
The internal first-step timing warnings were nonfatal; no GPUchange/restart.

Selected figures and final stats are integrated into local example/rootREADME,
website chapters and asset records. Ceramic's library graph is removed from
current display; its data/TODOremain. Explicit15kbuild validation and9page build
pass, publicationguard tests and selected-version read-only replay pass.
Remaining: final source/asset/link/staged-tree checks, authorizeddevcommit→main
regularmerge→pushmain, then verify Pages and archive the completed task.
[Library selection](../artifacts/electropolish_library15k_results.json) ·
[Temporal selection](../artifacts/pino_time_fusion_figures.json).

## Latest published evidence

[Completed library/ranked-gallery publication](../artifacts/fixed_library_pages_publication.json):
dev1d3e68a→mainb752981, Pages43808630409passed; six liveHTML and12assets verified.
Fixed10kElectropolish means for64/128/256/512:12.8724/2.6613/2.8755/2.0137%.
Fixed20kCeramic means:5.1984/32.7932/32.4830/32.7915%; larger banks poorly converged.
[Historical study](tasks/archive/fixed-condition-library-size.md).
The five selected main galleries use best/lower-median/worst phase-only ranking.

[Method/temporal comparison guide](PINO_COMPARISON_FIGURES.md) records cohorts,
reference solvers, metric definitions and full cold timing. Melting64native
references are not64local-FEM solutions; accepted localCG2case005is separate.
Spinodal has8qualified local-FEM cases and no AD control by user instruction.
Channel method controls use3reused cases and a historical Nonlocal GPU;
Ceramic method convergence and final distributions use distinct declared cohorts.
No universal speed/accuracy or cross-seed conclusion follows.

## Deferred and preserved work

- [TODO](TODO.md): Ceramic training stability and Spinodal discretization gap.
- [Stopped core/halo work](tasks/pino-unified-rolling-core-halo.md): no new launch.
- [Stopped Melting soft-IC](tasks/archive/melting-soft-initial-condition.md): hard-IC defaults retained.
- Unrelated dirty diagnostic/soft-IC work stays outside this publication.
- Data, weights and full galleries stay at existing external paths.

## Navigation and memory

- [Decisions](DECISIONS.md), [development](DEVELOPMENT.md), [plotting](PLOTTING_GUIDELINES.md).
- [Selected method figures](../artifacts/pino_three_method_figures.json),
  [temporal figures](../artifacts/pino_time_fusion_figures.json).
- [Prior dashboard and detailed publication history](tasks/archive/project-progress-through-20261116-1435.md).
- [Migration/recovery rules](migration/VALIDATION.md).

This session changesPROGRESS,DECISIONS,the active task and its experiment/
publication helpers. No current-task Git action or deployment has occurred yet.
