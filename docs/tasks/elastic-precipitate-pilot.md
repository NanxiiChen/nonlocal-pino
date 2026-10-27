# Elastic precipitate phase-field pilot

Goal: publish the current research snapshot first, then implement and run a
small two-dimensional elastic-precipitate phase-field PINN pilot.

Acceptance: authorized dev commit -> main regular merge -> push main -> return
to dev precedes new numerical implementation. New example uses train.py, an
explicit cited/derived two-way elastic/phase model, independent refined
reference, operator/energy checks, a cold-start training run and field/error
plots. Report phase and mechanical accuracy and all physical scope limits.

Context: existing transport pilot is closed; SOAP stability remains deferred.
No requirement to reproduce the full anisotropic/no-flux PFHub4 benchmark in
the first pilot. Preserve existing cases, defaults, references and log paths.

Status: preparing current snapshot publication. Blocker: none.
Next action: inspect current diff/artifacts, validate snapshot, commit/merge/push
as authorized; then specify the simplest meaningful periodic coupled model.
