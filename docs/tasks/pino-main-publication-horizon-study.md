# Main-figure publication and fixed-thickness horizon/grid study

## Goal and acceptance criteria

User explicitly authorizes commit/push of the accepted five main figures, then
requests a new PINO experiment varying nonlocal horizon and particle density
at fixed phase thickness. Use final-step weights and64common test samples;
plot relative L2 mean plus one sample standard deviation. Prior metric applies:
per-channel relative norm over all evaluation space-time points, channel mean
per sample, then mean/std over64samples. No percentage rescaling.

## Context

Five completed publication previews have direct specifications/receipts under
artifacts/*_main_figure*.json and external PDF/PNG outputs. Existing pending
Melting thin/extreme numerical code is unrelated to this figure release.
Git starts on main c48fa06, dev7b89e89, origin points to nonlocal-pino.
Required publication workflow: dev commit -> main regular merge -> push main.

Electropolish is the candidate because local-FD64targets already exist and
parameterized geometry does not confound mesh/horizon controls. Keep initial
interface distribution, physical coefficients/thickness, architecture, optimizer,
core count and training budget fixed. Audit quadrature/moment quality and actual
halo sizes before choosing affordable δ/h combinations. No new training labels.

## Status

Publication inventory and experiment feasibility inspection underway. GPU idle.

## Blocker

None. Exact study matrix and costs to be established from operator probes.

## Next action

Copy only final five PDF/PNG pairs to a dedicated publication-assets directory,
update README captions and provenance, verify, commit/merge/push. Then finalize
operator audit/matrix, implement a finite experiment runner, cold train and
independently evaluate/plot64final-case errors. Record exact process/recovery.
