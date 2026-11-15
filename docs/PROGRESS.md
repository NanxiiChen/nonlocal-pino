# Project Progress

## Current state

**Active task:** [time-fusion wall-clock figures and comparison publication](tasks/pino-time-fusion-wall-clock.md).
The five-main-PINO spatial-method comparison and two nonlocal temporal-feature
studies are complete; this task changes figure timing and publishes their records.
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
The user now authorizes commit and push of the updated figures and results.
The latest deployed revision remains main f2b9127 / Pages 43556036346 until this
publication task succeeds.

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
