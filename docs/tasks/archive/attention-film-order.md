# Audit attention and temporal FiLM ordering

Goal: check whether the published architecture interleaves time FiLM with
cross-attention or applies FiLM only during subsequent temporal decoding.

Acceptance: trace the relevant production call sites, distinguish temporal
modulation from pooled condition FiLM, and report how the diagram maps to
the implementation and its spatial reuse.

Context: [temporal decoder diagram](temporal-film-diagram.md) and
[release verification](publish-temporal-film.md).

Finding: `ConditionalFieldOperator.prepare_encoded` applies its spatial
attention before `decode` introduces time. `HeterogeneousPointOperator`
completes all spatial cross-attention blocks in `prepare_encoded` and then
passes the prepared features to the temporal decoder. Verified the selected
melting, electropolish, confined CH, channel and particle corrosion call sites.
The channel and particle decoders have the same layerwise temporal order.
The optional pooled `condition_film` in the fixed-domain model is a separate
condition-injection branch.

Validation: the [live Pages SVG](https://nanxiichen.github.io/nonlocal-pino/assets/architecture.svg)
matches the audited checkout byte for byte. GitHub reports the repository as
`NanxiiChen/nonlocal-pino`. The old Pages path returned HTTP 404 and the new
architecture page returned HTTP 200. The audit required no website or model
edits. `git diff --check` passed.
