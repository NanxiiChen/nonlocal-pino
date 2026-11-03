# Final20k channel physical-field gallery

## Goal
Show the completed20k final physical fields on the development geometry and
three predeclared reserved geometries, with independent FEM comparisons.

## Acceptance criteria
- Use actual completed20k `prediction.npz` arrays, not earlier partial weights
  or an unsaved scalar minimum; preserve reference/query provenance.
- Reuse existing five-time phase/error panels for all four geometries and the
  development zero-contour overlay; render missing speed/pressure panels on CPU.
- Visually inspect delivered figures; use absolute image paths in the response.
- No training, FEM solve, default promotion, cleanup or Git operation.

## Context
Completed task: `docs/tasks/archive/channel-cold20k-and-four-pino-setups.md`.
Run: `/root/autodl-tmp/tf-logs/channel_nsch2d/20261103-pro6000-integral-random-walls-online-width192-b8-time18-nt36-cold20k-lr1e5-04`.
Supplemental completion and paired independent10k/20k replay passed. All105
frozen training source files unchanged. Development phi1.4496%,terminal1.9020%,
u1.3312%,p4.6703%; reserved phi1.9860/1.4293/1.5152%, all native96x40queries.

## Status
Completed: verified completed20k saved arrays and native L2 on all four
geometries, rendered development speed/pressure on CPU, inspected all seven
delivered images. No remaining work or active process.

## Blocker
None.

## Next action
None. Preserve direct final-state image/provenance paths below; no additional
training, field evaluation or default change is started.

## Results and task-end

All fields correspond to actual completed20000weights. Phase shows t0/1.5/3/
4.5/6with three rows(FEM,prediction,absolute error), native3840queries x13times;
development/000/001/002relativephiL2=1.4496/1.9860/1.4293/1.5152%. Initial
conditions remain exact. Development zero contours use uncorrected physical
positions and equal zoom; blackFEM/orangeprediction, terminal areaIoU93.4280%.
Residual phase error is concentrated near the interface, including late front/
rear differences. Speed panels show magnitude; its plotted absolute error is
the difference of magnitudes, while reported velocityL2uses both components.
Speed/pressure use saved positive times0.5/1.5/3/4.5/6. Pressuret0is not fixed.
Mapped bilinear interpolation is for display only; no time interpolation or
changes to the saved native metrics/contours.

Four phase images and development contours were reused from the qualified
run's `phi-fields`, `unseen3/case_000..002/phi-fields`, and `droplet-shape`.
New CPU-rendered speed/pressure PNG/PDFs and exact source/log/provenance:
`/root/autodl-tmp/data/channel_nsch2d/cold20k-restart-nt36-time18-20261103-04/final-field-display-20261103-211553`.
render-source.py/evaluate-source.py preserve the actual source; gallery-validation.json
records the final model, direct input/image paths, metrics and times.
display-validation.json records all seven visually inspected image paths.

Verification: CPU renderer exits0, saved13x3840x5reference/prediction shapes,
finite fields, exact saved time nodes and independently recomputed nativephiL2
pass for all four cases. Seven PNG files inspected and decoded; provenance JSON,
archive/dashboard links and whitespace pass. No numerical code changes, code
tests, network/model inference, training, FEM, cleanup or Git operation.
Task archived; PROGRESS and this task updated. Completed20kresults/defaults and
existing README assets remain unchanged; no active job or remaining action.
