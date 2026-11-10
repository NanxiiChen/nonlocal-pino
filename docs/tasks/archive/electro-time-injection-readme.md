# Document the Electropolish time-injection decision

The user confirmed retaining spatial cross-attention followed by layerwise
time FiLM. This task records the decision, accuracy, and timing in the example
README.

The [README overview](../../../examples/electropolish2d/README.md#important-strategies)
now links to the time-injection ablation. The ablation explains spatial reuse,
the query count for both time-dependent variants, the matched experiment,
and the limits of its one-seed and unequal-capacity comparisons.

All 15 table values match the paired audit JSON from the
[matched ablation](electro-temporal-conditioning.md). The section anchor,
archive link, and `git diff --check` pass. This was a documentation update.
