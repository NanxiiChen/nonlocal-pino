# Two matched four-point cost–error curves

User requests2–3curves, four horizon settings per curve, means only without
SD bands, and authorizes missing experiments. User allows either fixed delta
or fixed delta/h. Before the new result, select two grids:101×51(5,151)and
201×101(20,301), with delta/h={1.5,2,2.5,3}on each. This matches relative support
across densities and avoids the known oversized finest-grid ratio4case.

Seven final10k/64local evaluations already exist. Missing only5,151/delta.025;
cold train from the saved n101-d020recipe with seed42and otherwise unchanged
scheme. Use the same card and process-clock definition; evaluate final10000
weights on the common64local cohort. Audit cold state and same-grid draws.
Do not select settings/checkpoints according to the new test error.

Figure: two NPG-colored lines, four circles each; circle area proportional to
delta/h with four common legend sizes. Raw mean space-time/channel error only,
no SD shading/bars. Retain the previous14-point figure/spec as historical.
Export editable5–7ptHelvetica PDF and PNG with direct provenance.

Status: complete. Blocker: none. All eight final10k/64local results and the
mean-only PDF/PNG are available; seven prior models reused, one missing model
cold-trained. No PDE/default change, reference solve, Git or website publication.

## Execution record

Root: /root/autodl-tmp/tf-logs/electropolish2d/cost-error-two-curves-cold10k-20261112-01
Owner552492; detached at2026-11-12T06:13:10UTC. Finite owner invokes canonical
train.py, the existing local64evaluator, then full cold-state/draw/norm audit.
Config: configs/n101-ratio250.json; run: runs/n101-ratio250. Only delta.025,
5151padded nodes/5120edges and log paths differ from frozen n101-d020baseline.
Completed ratio3same-grid halo is larger, so no additional resource probe.

Initial model and completeAdam bytes match baseline; source/config checks pass.
Early actual updates took~67ms. At step1200, nvidia-smi allocation was
35,527MiB on the same49,140MiB RTX4090. No other active training job observed.
Figure selection and renderer snapshot preserved under the study root before
this new final evaluation. Legacy14-row collector still passes after moving its
positive lower-SD check into the shaded renderer; means-only collection now
requires only positive mean values.

The finite owner completed with all three stages exit0. Regenerate the final
figure from the saved scalar records:

```bash
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m scripts.render_electropolish_cost_error_compact \
  --spec artifacts/electropolish_cost_error_compact.json
```

## Results and validation

The new n101-ratio250arm completed10000steps from cold initialization. Mean
space-time/channel error0.0304654109660; phase mean0.027675635, concentration
mean0.033255187. SampleSD0.007182087is retained in provenance but not plotted.
Full cold training subprocess806.109116s; separate64case evaluation29.868401s;
independent audit2.322492s. Warm update mean61.181ms. Same card model, memory and
driver string as the earlier study. Owner has finished; do not restart it.

Initial model/completeAdam and all10000same-grid condition/time/core/IC/BC draw
identities match frozen n101-d020. The new64prediction/reference norms were
independently replayed by the final audit. Exact input/config/state differences
remain restricted to horizon, halo capacities and output paths; FP32 execution
shapes may differ. The result is one paired initialization, not seed robustness.

The declared selection is5151:{n101-d015,n101-d020,n101-ratio250,n101-d030}and
20301:{ratio150,n201-d010,ratio250,ratio300}. Both follow ratios1.5/2/2.5/3.
Selection was frozen before the new final result; no post-result point picking.
`artifacts/electropolish_cost_error_compact.json` and its validation receipt
retain the eight weights/configs,512sample scores, actual delta, time and size
mapping. `artifacts/electropolish_cost_error_completion.json` records the new
experiment and exact train/eval/audit times. README explains both new and old
presentations and provides reproduction commands.

Outputs under:
/root/autodl-tmp/tf-logs/electropolish2d/publication-cost-error-two-curves-20261112

- electropolish_cost_error_two_curves.pdf:23,268bytes,130×85mm, vector-only.
- electropolish_cost_error_two_curves.png:79,833bytes,300dpi.

Real renderer execution, all512sample means, eight selected clock/ratio values
and ratio-linear circle areas verified. Actual PDF contains four colored data
circles for each curve, no SD fill, embedded editable6–7ptTrueType text and no
raster layers. PDF/PNG inspected; syntax, new README links and diff checks pass.
Historical14-point figure unchanged. Next: visual feedback only. Memory changed:
this archive, PROGRESS and DECISIONS. No Git action in this turn.
