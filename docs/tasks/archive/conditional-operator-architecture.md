# Conditional operator architecture schematic

## Goal and acceptance

Provide the user with a concise schematic of the current regular-grid and
parameterized irregular-domain operator architectures. Show patch/self-attention
encoding, separate boundary/domain pointwise encoding, query cross-attention,
equal stream fusion, and later time conditioning. Preserve the distinction
between architecture conventions and independently trained example parameters.

## Context and source

- User requests an illustrative diagram, without implementation detail.
- `src/nonlocal_pinn/nn/conditional.py`: `encode`, `prepare_encoded`, `decode`.
- `src/nonlocal_pinn/nn/point_conditional.py`: `HeterogeneousAttention`,
  `HeterogeneousPointOperator.encode` and `prepare_encoded`.
- `examples/confined_ch2d/operator_training.py`: `encode`, `prepare`, bank padding.
- `examples/electropolish2d/operator_training.py`: shared `times`/`queries` payload
  and condition-wise vmap; initial-field conditions remain separate.
- `examples/icemelting3d/operator_training.py`: one pool entry and time vector
  per update, reused across random initial ellipsoid conditions.
- Read `docs/PROGRESS.md`, the last five decisions, relevant development guidance,
  numerical/JAX rules and full visualize skill. Use the skill's Mermaid route for
  a static structure; no HTML artifact or generated media is needed.

## Status and verification

Complete. Deliver the static Mermaid source from
`docs/tasks/conditional-operator-architecture.mmd` inline in the response.
Source inspection confirms patch self-attention, separately lifted irregular
streams, mean fusion and temporal decoding after spatial preparation. Both
regular-grid training loops draw one spatial set/time vector outside the
condition-wise computation. The diagram intentionally omits repeated blocks,
normalization, physical wrappers and training losses to meet the user's request
for a conceptual illustration. Shared downstream nodes denote design roles,
not common trained weights between example models.

`git diff --check` passes. Diagram syntax and the shown arrows/labels were
inspected; no browser/rendering test or numerical test is claimed or required
for this documentation-only change. No HTML, image, checkpoint or dataset is
generated. No runtime, default, dependency, training or Git-history change.

## Query clarification

Within one current regular-grid training update, all initial-condition samples
share the sampled spatial core/halo and time vector. These samples still have
distinct learned conditional features. Spatial/time samples can change between
updates. Irregular-domain geometries have different physical point coordinates;
padding aligns lengths only. Within each geometry, times share spatial preparation.

## Blocker and next action

None. Completed; no required resume or queued training.

## Diff-level record

- Add the compact Mermaid source with the two encoder designs and cross-attention.
- Record the decision and completed task; update the progress dashboard.
- Preserve earlier uncommitted conceptual/detailed HTML documentation work.

Changed memory files: `docs/PROGRESS.md`, `docs/DECISIONS.md`, and this archived task.
