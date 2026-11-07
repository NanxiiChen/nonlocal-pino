# Temporal FiLM clarification

Goal: explain the position of time FiLM in the architecture diagram and
clarify the layerwise modulation in the local report.

Acceptance: describe the reusable spatial features, time MLP outputs,
three hidden layers and separate output head consistently with the code.
Check the updated SVG, equation and embedded figures.

Status: completed locally. Verified the fixed-grid decoder, shared point
decoder and both multi-output point decoders. Expanded the temporal portion
of the [diagram](../../../website/architecture.svg) for both paths and
updated both embeds. The report now gives the implemented affine, FiLM, and
GELU sequence and its three layer-specific time parameter pairs.

Validation: nine-page build and `git diff --check` pass. Chromium checked
the SVG labels and both embedded figures at 1440, 768, 390 and 320 px.
Inspected the rendered decoder panel. No numerical or Git history changes.
