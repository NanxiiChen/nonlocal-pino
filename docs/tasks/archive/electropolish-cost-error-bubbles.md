# Electropolish cost–error bubble plot

User requests a diagram patterned after an attached cost/error chart: x=time,
y=relative L2, curve color=particle count, circle size=nonlocal horizon delta.
Combine the completed horizon/density and fixed-particle ratio studies without
duplicating the reused20,301/delta.01baseline. All14unique runs use final10k
weights and the same64local-FD sample/channel-averaged space-time metric.

Acceptance: complete cold training subprocess wall time from each study's
journal (includes startup, compilation, monitored training and final save;
excludes separate64-input evaluation/reference construction); colors by N,
circle area proportional to delta, lines in delta order without smoothing,
sample±1SD shading. Single-point N groups do not invent curves. Record the
direct run, metric, clock, configuration and plot paths. PDF/PNG with embedded
editable Helvetica/sans math,5–7pt and NPG colors; inspect both outputs.

## Delivered and verified

Complete. `scripts/render_electropolish_cost_error.py` reads explicit source
reports/journals and per-run configs/evaluation files. It verifies the shared
physical/training scheme, same64input identities/reference paths/times and
final checkpoint rule, and recomputes every per-sample channel mean and sample
SD. Duplicate run paths must have matching values; the shared baseline appears
once. Successful journal training-end records supply the plotted wall times.

`artifacts/electropolish_cost_error.json` fixes source paths and style;
`artifacts/electropolish_cost_error_validation.json` retains all14runs,
896sample reductions, clock records, particle/horizon encodings and output
checks. The example README explains the interpretation and reproduction.

External output directory:
/root/autodl-tmp/tf-logs/electropolish2d/publication-cost-error-20261112

- electropolish_cost_error.pdf:21,937bytes,170×95mm, vector-only PDF.
- electropolish_cost_error.png:108,124bytes,300dpi preview of the same plot.

Both axes are logarithmic, auto limits and raw error ratios. Five particle
counts retain14observations; singleton groups1,326and8,385use vertical sampleSD
bars rather than artificial bands in time. Multi-point groups use straight
segments and sampleSD shading. Actual nonmonotonic small timing differences
are retained. Circle area=144×delta/0.04pt² with four representative size keys.
The20,301/delta.01point has cold-process time655.048s and mean0.0172813.

Real renderer execution passed. Independent clock,896sample-reduction,
deduplication, size-mapping and group-order checks passed. Actual PDF text is
6–7pt, embedded TrueType with sans-serif math and no raster layers; all spans
are inside the page. PDF and PNG were visually inspected. Diff/link checks
pass. No new simulation, inference, reference solve, default promotion, Git
or website deployment. Prior figures remain unchanged.

Status: complete. Blocker: none. Next: user visual feedback; no further work
required for this request. Memory updated: this archive, PROGRESS and DECISIONS.
