# Numerical Methods Policy

This is the detailed reference for the numerical rules summarized in
`AGENTS.md`.

## No Guessed Mathematics

- Every PDE residual, constitutive law, boundary treatment, kernel
  normalisation, and nondimensionalisation must come from a cited reference or
  a derivation that is checked independently.
- Do not copy a formula from a different dimension, model, or boundary-value
  problem without verifying that its assumptions match this example.
- Do not invent optimizer settings or adaptive weighting rules and present them
  as physical or generally valid numerical methods.

## Stop Conditions

Stop and ask the user when:

- a kernel normalisation or nonlocal operator formulation is uncertain;
- multiple valid formulations have materially different physics;
- a reference describes a different dimensionality, loading, or boundary model;
- code and documentation disagree about the implemented equations;
- a proposed correction would change the declared variational or history-field
  formulation.

## Required Standards

- Use published phase-field fracture formulations such as AT1/AT2 rather than
  ad-hoc degradation laws.
- Use standard FEM weak forms rather than hand-written stiffness shortcuts.
- Use established peridynamic correspondence formulations when that model is
  explicitly selected.
- Treat kernel normalisation, boundary truncation, quadrature, and adaptive
  sampling as model/discretisation choices, not implementation details.

## Validation Order

Separate these errors and validate them in order:

1. operator and quadrature consistency;
2. independent reference-solver discretisation error;
3. direct nonlocal baseline error;
4. PINN representation and optimisation error;
5. sampling or adaptive-refinement error;
6. physical-model and parameter error.

Record non-trivial choices in `docs/DECISIONS.md` before implementation grows
around them.
