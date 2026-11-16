# Project Progress

## Current state

**Active task:** [Electropolish15k library and time-fusion controls](tasks/electro-library-extension.md).

User expanded the active task: cold15kfixed64/128/256/512thenjoint query,
reuse completed32/online15kcontrols, update Electropolish-only library and temporal
cold-wall-time figures/README/site, then commit/push. No Ceramic rerun/default change.

Recovery bundle02 completed all training and independent10k/15k64case audits
at2026-11-16 14:23Asia/Shanghai. Final phase/concentration/channel means:
online1.39597%/1.67106%/1.53351%, fixed3215.65992%/17.23107%/16.44550%.
Across-case sampleSD0.580879/3.639694percentage points; cold1283.698/1281.388s.
All seven declared controls retain seed42 and the same64local-FD tests.

New bundle `electro-15k-completion-20261116-01`, owner84522, is running fixed64
(start14:23:30,trainer93395), then128,256,512,joint_queryserially. All5preflights
passed;173numerical files match the completed baseline. Runtime capture stays
explicitly disabled, one40%GPUreservation; no automatic retries. Each arm includes
10kand15kweight reloads and physical-field audits before proceeding.
CPU-only figure owner105355waits for numerical completion, then prepares library
and temporal plots. It never trains or publishes. Exact states are in the bundle's
journal.json and figure_queue.json; do not start duplicate owners.

The first15kpair failed at13:19Asia/Shanghai during CUDA stream capture after
online update9900; the owner stopped fixed32. Last complete64case monitor9750:
fixed32mean14.3527%; online1.9581%(phase1.7868%,concentration2.1293%). These are
incomplete-run observations; no10k/final checkpoint is available.

Manual cold recovery started13:37Asia/Shanghai in bundle
`electro-library-extension-15k-20261116-02`, owner57842. Serial online→fixed32,
explicit CUDA command capture disabled;173numerical files byte-identical to the
failed attempt, same seed/source/LR/64tests and15kbudget with10kcheckpoint.
Preflight and two controller tests passed. Preserve failed bundle01and never
restart it. Root runtime trigger remains unproven; fixed32and final audits remain pending.
Only one40%GPUreservation is active at a time. No extra studies or20kextension.

The prior README/site publication is complete and verified:
[publication receipt](../artifacts/fixed_library_pages_publication.json).
Defaults remain unchanged; Ceramic stability remains deferred inTODO.

The preceding [fixed condition-library study](tasks/archive/fixed-condition-library-size.md) is complete.

All eight cold runs and independent64case audits completed2026-11-16at03:37
Asia/Shanghai; no trainer or queued work remains. Electropolish10k channel means
for64/128/256/512inputs:12.8724%,2.6613%,2.8755%,2.0137%. Ceramic20kmeans:
5.1984%,32.7932%,32.4830%,32.7915%. The three larger Ceramic banks retain poor
convergence; cause is unestablished, and audit success is not an accuracy claim.
Preserve all arms, defaults and original budgets; do not restart the owner.
Final mean±sampleSD figure, direct paths and complete cold times are checked:
[completion record](../artifacts/fixed_condition_library_results.json).
One initialization only; no claim that increasing library size generally harms
generalization. Further diagnosis is deferred. The reviewed figure is included in the current publication.

Baseline check: the configuration-matched online Electropolish run has2.1638%
channel-mean error versus2.0137% for fixed512; both10k/64tests and byte-identical
initial model/optimizer. The older selected online main model has1.7035%, but its
input-evaluation path differs. Online sampling does not guarantee lower error at
a finite update budget; all these arms retain rolling spatial core/halo pools.
The input-path difference is only common encoder/IC/Dirichlet array evaluation:
GPU FP32 versus CPU FP64 followed by one FP32 cast for exact cross-method pairing.
Physical initial conditions and training precision are unchanged; this has not
been isolated as the cause of the historical/current error gap.

The five accepted ranked main figures are now integrated into the local example
READMEs, project overview and website chapters. All10selected PDF/PNG files are
byte-identical to the approved exports; total7,022,408bytes remains below9MiB.
[Selection/reproduction](../artifacts/pino_ranked_main_preview.json) ·
[Local integration checks](../artifacts/pino_ranked_main_publication.json).
Nine-page build,395local links/anchors, rank metadata and asset dimensions pass.
This earlier local integration is now included in the user-authorized publication.
dev1d3e68a was merged as mainb752981 and pushed. Pages run43808630409
passed build/deploy; six live HTML pages and all12figure files match exactly.
Original main weights/defaults remain unchanged. The previous short MeltingGPU
inference overlap with the library study is recorded; no GPU job ran for this
documentation integration.

**Previous publication:** [Ceramic temporal controls](tasks/archive/ceramic-time-fusion-publication.md).
README, overview and website include the cold-time convergence and final64case
violins, with shared editable PDF/PNG pairs. dev6180ce7 -> mainfb45cf7 was pushed;
Pages run43727138734 passed build/deploy. Live overview/case HTML and all four
figure files match the selected build byte-for-byte.
[Publication receipt](../artifacts/ceramic_time_fusion_publication.json).

**Latest completed:** [Ceramic Time-FiLM versus joint query](tasks/archive/ceramic-time-fusion-comparison.md).
Both cold20k runs, all41×64equal-point histories, paired sampling/source/final-field
audit and four reviewed PDF/PNG figures are complete. Final channel means:
FiLM4.09016%,joint4.48324%; complete cold times2629.487s /12475.490s (4.74446x),
same physicalGPU. NewFiLM reproduces all64historical final predictions exactly.
Defaults remainFiLM. Results/reproduction are in the exampleREADME and
[completion manifest](../artifacts/ceramic_time_fusion_validation.json).
Selected statistical pairs are published from assets/pino-studies; full physical
galleries, fields and weights remain at their recorded external paths.

Melting soft-IC was stopped by user: Nonlocal/AD completed, FEM interrupted,
joint query unstarted. All owned processes stopped and original hard-IC defaults
verified. [Stopped experiment record](tasks/archive/melting-soft-initial-condition.md).

## Previously completed publication

The five-main-PINO spatial-method comparison and two
nonlocal temporal-feature studies are complete and published. The temporal
convergence figures now use elapsed cold wall-clock time.
[Latest publication task](tasks/archive/pino-time-fusion-wall-clock.md).
[Archived task](tasks/archive/pino-three-method-suite.md) ·
[Completion audit](../artifacts/pino_three_method_completion_validation.json).

| Study | Completed controls | Final evaluation / convergence scope |
|---|---|---|
| Ceramic corrosion | Nonlocal / Local AD / FEM PINO,20k |64local-FEM cases /4development cases |
| Coupled two-phase flow | Nonlocal / Local AD / FEM PINO,20k |3local-FEM cases /1development case |
| Electropolish | Nonlocal / Local AD / FEM PINO,10k |64local-FD cases at all41records |
| Melting3D | Nonlocal / Local AD / FEM PINO,10k |64native cases at all41records; separate accepted localCG2case005 |
| Spinodal | Nonlocal / FEM PINO,50k; AD omitted by user |8qualified local-FEM cases at all101records |
| Time fusion | Electropolish and Melting, two10k nonlocal pairs |64cases at all41records; spatial-attention/FiLM versus joint(x,t)query |

All28selected convergence/distribution PDF/PNG files appear in the corresponding
READMEs and local website. Build9pages,387local links/anchors, final-field metrics,
asset byte identity and16complete cold training intervals pass. Defaults remain
unchanged; comparisons retain one initialization and their declared cohorts.
Published via dev e8ce417 -> main merge e0e154c -> push origin main.
Pages run 43693239443 passed build and deploy. Both live temporal chapters and
four convergence PDF/PNG assets match the selected captions/files.
[Publication receipt](../artifacts/pino_comparison_pages_publication.json).
Standalone core/halo investigations remain local; no raw weights/reference arrays
or intermediate galleries were uploaded.

## Main result qualifications

Spinodal final phase error against independent local FEM is31.9216% Nonlocal /
5.5838% FEM PINO. Against native nonlocal numerical data, the identical equal-node
metric gives4.5990% /29.3864%; numerical targets differ by31.9633%. Complete cold
times are8086.912901s /6852.669361s on the same physical GPU. Both50k trainers
and final audit are complete. The first plot failed a PDF-label edit probe; the
repaired second output passes and is selected. Do not restart the old queues.

Melting64native references are not64local-FEM solutions; case005retains its
original unpassed strict-refinement flag and explicit practical acceptance.
Channel uses3reused reserved cases and a historical Nonlocal run on another
physical GPU. Ceramic convergence and final distributions use distinct declared
cohorts/weights. Temporal modules have different parameter counts. No universal
accuracy/speed, blind-test or cross-seed claims follow.

Primary time includes the full cold training process: initialization, preparation,
compilation, updates/rolling waits, in-training evaluation, writes and teardown.
Separate offline banks/probes and post-training audits/plots are excluded.
[Exact timings and hardware scopes](../artifacts/pino_training_wall_costs.json).

## Deferred work

The user explicitly deferred Spinodal horizon/resolution/boundary diagnostics:
[research TODO](TODO.md). Do not launch further Spinodal studies without a new
request. The older [sampling task](tasks/pino-unified-rolling-core-halo.md) remains
stopped; Spinodal retains full-field training.

## Navigation

- [Figure/metric/reproduction guide](PINO_COMPARISON_FIGURES.md).
- [Derivative assets](../artifacts/pino_three_method_figures.json),
  [ceramic assets](../artifacts/ceramic_component_comparison_figures.json),
  [temporal assets](../artifacts/pino_time_fusion_figures.json).
- [Completed temporal task](tasks/archive/pino-time-fusion-comparison.md).
- [Final Spinodal norms](../artifacts/spinodal_final_reference_norms.json),
  [publication validation](../artifacts/spinodal_nonlocal_fem_publication_validation.json).
- [Detailed prior dashboard](tasks/archive/project-progress-through-20261115-0439.md),
  [older dashboard](tasks/archive/project-progress-through-20261114.md).
- [Split/recovery guidance](migration/VALIDATION.md); preserve external data and
  unrelated worktree changes.

## Latest session checks

Current publication: staged9page build,405local links/anchors, all12selected
source/repository/site assets and18liveHTML/figure checks pass. Completed library
endpoints, ranked galleries, optional fixed samplers and reproducibility helpers
are committed and pushed. Raw arrays/weights and unrelated stopped experimental
work remain external or uncommitted. Defaults are unchanged. The expanded15klibrary/time-fusion suite remains
the sole active numerical task; its two baseline controls are fully audited and
five added controls are queued/running.
Memory changed:PROGRESS,DECISIONS,TODO,active extension task and archived library
study. [Verified deployment](../artifacts/fixed_library_pages_publication.json).


Fixed-library completion: all8final audits and512sample-mean reductions verified;
two-panel170mmfigure visually reviewed,6/7ptembeddedHelvetica and editable PDF
label checks pass. Renderer reproduces from a completed journal; example notes
and archived task agree. No numerical default change; the current publication includes these checked results.

Ranked gallery integration: explicit publication manifestv2, five updated chapter
captions/alt text/dimensions, five exampleREADMEs and overview, shared asset guide
and reproducible copier. Build9pages,395links, all10source/repository/site file
identities, PNG dimensions and malformed-selection rejection pass; diff check
passes. Local publication receipt records that Git/Pages deployment is pending.


Ceramic complete-pair audit verifies identical common initialization and actual
sampling, both reconstructed final pools,256final channel norms,10496actual
plotted channel observations,40physical-panel labels and complete cold endpoints.
Four final figures were visually reviewed; PDF fonts/text/layers pass. NewFiLM
and joint queues/observer are terminal. Numerical source matches the frozen
qualified snapshot. ExampleREADME, completion manifest and archived task agree;
defaults unchanged. Relevant implementation and reproduction are now committed.
Previously passed focused numerical tests remain valid.

Earlier five-case/temporal publication checks remain in the archived tasks above.
The new Ceramic study preserves prior selected assets and adds its own explicit
20k/64case temporal comparison. Staged/local9page builds,395links, source/metric
checks, live HTML and four image/PDF byte comparisons pass. Publication task is
archived; no work or monitoring remains scheduled.
