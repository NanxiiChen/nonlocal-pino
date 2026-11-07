# Point-operator architecture clarification

Goal: answer how the selected parameterized-domain models use cross-attention
and clarify the report diagram where its placement was ambiguous.

Acceptance: the diagram and method text distinguish point lifts, weighted
coordinate-to-context reads, and the time decoder. The description matches the
actual projections, normalization and stream combination in the shared class.

Status: complete locally. Audited the shared class and call sites for
spinodal, channel flow and particle corrosion. Relabeled the [SVG diagram](../../../website/architecture.svg),
explained the weighted head and two-stream mean in the architecture chapter,
and clarified the block labels in the three affected case tables.

Validation: the nine-page build and `git diff --check` pass. Chromium verified
SVG label containment and desktop and narrow-screen layouts and math for the
five affected pages. No numerical code or public deployment changed.
