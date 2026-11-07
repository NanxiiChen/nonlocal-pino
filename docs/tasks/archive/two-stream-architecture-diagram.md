# Two-stream architecture diagram

Goal: draw both source streams and the complete spatial query flow, then
commit and push the diagram and related report corrections.

Acceptance: the diagram distinguishes pointwise lifts, independent K/V
projections, a shared Q, separately normalized attention, an equal stream
mean, query update, and temporal decoding. The SVG and both embedded figures
render cleanly and the authorized push reaches `main`.

Context: [prior source audit](point-operator-diagram.md). The earlier
clarification is already in the working tree and belongs to this publication.

Changes: redrew the [SVG](../../../website/architecture.svg) as separate fixed-
and parameterized-domain panels. The geometry panel shows two pointwise lifts,
separate K/V routes, a shared Q, arc-length and area normalization, the mean of
both messages, the repeated query update, and temporal decoding. Updated the
captions, alt text and dimensions in both placements, the method prose, and
the spatial-block labels in the three geometry case tables.

Validation: confirmed the layout against `HeterogeneousAttention` and the
fixed-domain `ConditionalFieldOperator`. The nine-page build and
`git diff --check` pass. Chromium checked SVG labels, links, KaTeX and layout
at 1440, 768, 390 and 320 px. Inspected the rendered diagram and adjusted label
padding. Release follows the normal dev-to-main merge and the existing Pages
workflow.
