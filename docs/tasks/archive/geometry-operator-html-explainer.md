# Explain parameterized-geometry CH operator in an illustrated HTML

## Goal and acceptance

User requests a Chinese illustrated HTML explaining how confined-CH random
geometry is implemented, especially how neural inputs accommodate nonrectangular
varying domains. Deliver a self-contained offline standalone HTML, grounded in
the current width256onlineSOTA source. Use actual precomputed geometry points/
features/weights, interactive input inspection and clear architecture diagrams.
Do not invent evolved fields/attention weights or alter training/defaults.

## Source-checked context

Fixed ellipse center(.5,.5), axes(.5,.4), two random circles; centers in
[.22,.78]x[.27,.73], radii[.045,.12], clearance.025 plus explicit rejection.
Boundary192x5=(x,y,nx,ny,type), outer96+2x48; domain Nx3=(x,y,phi0), triangle
centroid coordinates and physical areas, fixed10mode ambient IC across domains.
Normal points out of physical domain, including into each hole. Outer/inner
types0/1, no per-hole ID. Different geometries have distinct quadrature/query
points; pad to1536zero-measure rows. Eight known-input training arrays include
operator/metric, but only two point/measure streams enter the neural encoder.

HeterogeneousPointOperator: separate pointwise lifts,258coordinate Fourier
features, width256/4heads/4spatial conditioning blocks/3temporal FiLM layers.
Normalized linear integral attention uses separate stream weights and an equal
stream mean, without query-hole distance bias, query self-attention, patches or
Perceiver latent bottleneck. Query is spatial coordinates only; initial field
also enters the external hardIC/mass wrapper. CurrentSOTA has no optional phase
record: lifting is linear t/T, not the experimental log-time mode. Predict phi;
derive mu algebraically through regional operator. Same geometry's times share
prepared spatial features; online pool1024,6workers,32replacements/100updates.

## Status / blocker / next action

COMPLETE. Standalone offline Chinese HTML generated from actual geometry inputs
0/256/768 (1162/1229/1223physical points), source features/weights/normals exact.
Desktop and390px mobile Chromium interaction/layout checks pass, no JS errors,
no external requests or horizontal overflow. Actual diagrams inspected; one
low-contrast caption corrected. No runtime/default/training/Git change or
blocker. Browser processes closed; no follow-up queued.

The visualize skill was inspected; its first rule explicitly excludes standalone
file requests, so the inline-fragment contract is not used for this deliverable.
Keep generated HTML out of Git per repository media policy; retain a compact
generator/source-facts receipt and update/archive memory at completion.

## Deliverables

- HTML: /root/nonlocal-pinn/docs/tasks/geometry-operator-explainer.html.
- Durable chat copy: /root/.codex/visualizations/2026/10/24/01a16d71-2d47-7201-9441-ab74e0523152/geometry-operator-explainer.html.
- Source: docs/tasks/geometry-operator-explainer.html.in and
  docs/tasks/geometry_operator_explainer.py; regenerate with project.venv onCPU.
- Browser checks: docs/tasks/geometry_operator_explainer_browser.py.
- Receipt: docs/tasks/geometry-operator-explainer-validation.json.
- Desktop/network/mobile PNG previews under docs/tasks with the same prefix.

The746402byte HTML includes its own numeric data, CSS/SVG and JavaScript; no
CDN/font/image downloads. Generated HTML and PNG stay ignored by Git; source
template/regenerator and compact receipt remain reviewable. Native file tab
requested viaopen_in_codex, returnedqueued; actual opening was not asserted.

## Content and verification

Seven illustrated sections: actual random geometry, boundary/domain tokens,
zero-measure padding, generic weighted attention/point lifts, spatial preparation
shared across time, hardIC/mass/chemical elimination and separate regional graph,
bounded online geometry pool. Three actual geometries can be switched; points
clicked to inspect real rows; normals/actual local bonds toggled. Scalar token
splitting demonstrates measure invariance without pretending to show trained
attention weights. Time/pool controls explicitly demonstrate workflow rather
than fabricate physical evolution or launch computation.

Boundary192x5/normal directions/arcweights match source arrays; domain initial
values recomputed byactual fixed_initial. Separate areas and arcweights rather
than extra concatenated neural channels are explicit. Padding counts374/307/313
and hardIC/weightedmass arithmetic pass (zero-time exact, mass gap<1e-15).
Input array/shape/range, AST/strictJSON, equal output-copy/offline-resource and
git diff --check pass. No costly numerical suite/training/reference solve.

Existing isolated Playwright/Chromium uses explicit child LD_LIBRARY_PATH and
FONTCONFIG_FILE for already present libasound/libgbm/Noto CJK. Initial missing
library launch was corrected without installs. First real click test found a
decorative normal line intercepting clicks; disabling pointer events on
decorative SVG elements fixed it. Subsequent full interaction checks pass.
Initial source inspection requested an absent optionalphase field; actual SOTA
uses linear t/T and HTML follows it, without changing configuration.

## Task-end memory

Update PROGRESS.md, DECISIONS.md, artifact index and this archived task. No
commit/push authorization in this turn, so keep the new explanation uncommitted.
