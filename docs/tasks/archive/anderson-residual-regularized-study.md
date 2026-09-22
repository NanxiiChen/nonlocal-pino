# Fixed-Mu Regularized Residual-Anderson Study

## Goal

Implement and verify the distinct experimental
`anderson_residual_regularized_backtracking` policy for the stabilized
engineering-grip controller without changing the approved mechanics, damage
equations, committed-history map, or historical policy payloads.

## Acceptance Criteria

- Require explicit finite positive `anderson_regularization_mu` in the
  stabilized configuration and workflow CLI. Do not provide an adaptive rule,
  hidden default, or silent fallback.
- Preserve the existing residual-backtracking controller contract exactly:
  Type-II consecutive-difference history, depth `5`, `beta=0.5`, alpha schedule
  `(1, 0.5, 0.25, 0.125, 0.0625)`, gate order, threshold, trial-local
  rejection, history updates, and exact `beta=0.5` Picard fallback.
- Solve the only changed proposal equation with augmented pivoted QR:
  `gamma = argmin ||f-DF gamma||_2^2 + lambda ||gamma||_2^2`,
  `lambda = mu ||f||_2^2`, using `[DF; sqrt(lambda) I]` and `[f; 0]`.
- Preserve lean v5 historical payload shapes; make new policy configuration,
  archive metadata, scalar diagnostics, and summaries strict about policy,
  merit identity, and `mu` provenance.
- Add independent manufactured numerical, controller-parity, CLI/config,
  provenance/forgery, compatibility, and coefficient-norm tests.
- Do not run a physical checkpoint or create external artifacts.

## Status

Implementation and non-physical verification are complete. The fixed-mu
contract, augmented-QR proposal, controller parity, CLI/config/binding
provenance, final public diagnostic names, README, and compatibility coverage
are implemented. No physical run is authorized.

## Blocker

No implementation or numerical-test blocker remains. External verification of
the Ouyang parameter-history claim remains unresolved; the code and
documentation retain bounded-study wording and make no convergence claim.

## Verification

- Expanded stabilized/workflow suites: `118/118` passed in `431.568 s`.
- Affected legacy engineering/JAX/baseline suites: `81/81` passed in
  `35.761 s`.
- `.venv/bin/python -m compileall -q examples/fracture2d tests`: passed.
- `git diff --check`: passed.
- Expected argparse usage/error output occurred only inside negative CLI tests.
- No physical checkpoint, continuation, or external artifact was run or
  created.

## Next Action

Primary Sol review should independently inspect the augmented-QR solve,
current-residual/lambda identity, controller parity and solve-count test,
regularized scalar/v5 provenance, and preservation of historical payloads.
