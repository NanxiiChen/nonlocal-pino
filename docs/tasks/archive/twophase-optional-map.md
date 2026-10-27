# Optional inverse-cubic phase coordinate

Goal: favor cross-example simplicity by disabling the two-phase cubic phase
coordinate by default, with an explicit opt-in training entry.

Acceptance: explicit phase.map=additive in canonical configs; conserved vortex
inherits it; --config.phase.map=inverse_cubic enables the existing map without
changing other schemes. Keep old serialized configs loadable and historical
cubic-only research controls explicit. Update current README/recommendation
metadata without relabeling old results or changing stored artifacts.

Context: fresh paired ablation has comparable pooled phi3.20%/3.25%, though
cubic improves late times. User values common schemes more than this difference
and explicitly requests default-off with an enabling entry. No retraining or
Git operation is requested. SOAP remains deferred.

Status: complete (2026-10-27).
Blocker: none.
Next action: none. Cubic remains optional; no new training or Git operation.

Changes: train_base declares phase.map=additive; train_vortex_conserved no
longer overrides it. The local control inherits the same map. Validator allows
explicit additive while retaining missing-phase legacy configs and invalid-map
rejection. No network/PDE/operator or field wrapper changes. Existing
train_vortex_accuracy, train_vortex_causal_conserved and train_vortex_initial_time
explicitly set inverse_cubic to preserve their research definitions.

Enable through the canonical entry by appending
--config.phase.map=inverse_cubic to the conserved preset command. Disable with
--config.phase.map=additive. Both actual CLI argument parses pass via
--only_check_args (exit0), including the new override on the locked ConfigDict.
No additional boolean or duplicate preset is introduced.

Case README explains the default, opt-in/out, current additive checkpoint and
historical cubic/local controls. DEVELOPMENT documents explicit mapping and
saved-config loading. artifacts/index.yaml gains a current vortex_default
record with direct checkpoint/config paths, while older studies are explicitly
labeled with launch-time map semantics; stored runs and paths are untouched.

Verification: all18presets validate. Default matches the previously tested
additive strategy exactly after normalizing JSON containers and omitting only
explicit phase-map declaration and runtime seed/run-name provenance. The
archived check script rejects an invalid map and reloads both full-grid
initial/terminal checkpoint fields using saved configs. FP64 maximum field
differences6.66e-15(additive)/5.33e-15(cubic), below unchanged1e-10 tolerance.

Two focused existing checks pass: cubic inverse/derivative/exact IC (initial
two-test command) and hard-mass field/time/parameter-gradient/two-loss check
(corrected run17.406s). The initial gradient check failed because a uniform
phase bias is removed by exact projection under the new additive default;
finite differences of the zero direction left4.44e-10 roundoff. The test now
perturbs a spatially varying output weight with the original3e-6 tolerance.
The initial failure log is retained. This changes verification, not physics.

Artifacts alongside this record: twophase-optional-map-check.py,
twophase-optional-map-validation.json, twophase-optional-map-tests.txt and
twophase-optional-map-tests-initial-failure.txt. Source diffs reviewed; inventory,
documentation links (5documents/76links) and git diff --check pass. New source
and documentation whitespace pass; raw test logs preserve their original
traceback formatting. No full suite,
GPU/FEM job or full training was needed. SOAP remains deferred.

Task-end memory files: PROGRESS.md, DECISIONS.md, DEVELOPMENT.md and this record.
