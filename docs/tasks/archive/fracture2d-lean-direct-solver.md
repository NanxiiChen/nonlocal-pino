# Fracture2D Lean Direct Solver

## Goal

Reduce `examples/fracture2d/` to the adopted direct nonlocal scheme only:

- exactly one direct nonlocal solver file in the example folder,
  `examples/fracture2d/nonlocal_baseline.py` (entry + `run`/`verify`/`plot`
  CLI), formerly `direct_nonlocal_solver.py`;
- move the implementation library out of the example directory into the shared
  package `src/pinn/nonlocal2d/` so the PINN stage can reuse the same graph,
  energy, force, Amor history, and AT2 residual;
- park (untrack + `.gitignore`, keep locally) the durable workflow and every
  research variant that the adopted scheme does not use;
- keep numerical identity: the moved code must not change any equation, gate,
  parameter, or control flow.

## Acceptance Criteria

- `git ls-files examples/fracture2d/` lists only `nonlocal_baseline.py` as the
  direct nonlocal solver (local FEM reference files `solver.py`, `geometry.py`,
  `mesh.geo`, `configs/` stay).
- `src/pinn/nonlocal2d/` contains the moved stack with no stale
  `examples.fracture2d.nonlocal_*` imports anywhere in the tracked tree.
- `tests/test_fracture2d_direct_nonlocal_solver.py` (renamed if needed) passes.
- Focused suites pass for the moved stack.
- A fresh `run` smoke (`--load-final 2e-4`) reproduces the archived early
  states to rounding (reaction `25.3468356139` at `0.0002`).
- `verify`/`plot` still read the user's fresh compact solution and the legacy
  stabilized archive.
- README, `docs/PROGRESS.md`, `docs/DECISIONS.md`, and task files updated;
  `git diff --check` clean.

## Relevant Context

- Adopted scheme: original BA/Amor/AT2 on the 3395-node `delta=0.048` cloud,
  fixed damped-Picard `omega=0.5`, `du=1e-4`, `du_min=6.25e-6`, <=144 sweeps,
  `stabilized_warning` acceptance, `0 -> 0.0075`.
- Accepted artifacts: fresh compact solution SHA256 `5f3bc12b...`; research
  archive SHA256 `9545bed7...`; peak `641.34017` at `0.0061`, terminal
  `69.15426`, drift `2 -> 31`.
- The implementation stack is ~45k lines in 9 tracked modules; the workflow
  (`engineering_grip_workflow.py`, ~6.6k lines) is not on the adopted path.
- The monolithic/arc/reduced bundle is already untracked and gitignored.

## Status

Active. Completed in the first pass:

- implementation moved to `src/pinn/nonlocal2d/` (5 modules + `__init__.py`);
- reachability prune from the adopted entry: `45,346 -> 15,937` lines
  (`nonlocal_geometry.py` and `nonlocal_mechanics_jax.py` deleted;
  `nonlocal_mechanics.py` and `nonlocal_baseline.py` merged into their only
  consumers);
- entry renamed to `examples/fracture2d/nonlocal_baseline.py`; all imports,
  `_SOURCE_BUNDLE_FILES`, tests, and README references updated;
- durable workflow, monolithic/arc/reduced bundle, and the research test suites
  parked under `_legacy/` (ignored);
- static identity preserved: `model_digest`, `static_digest`, `material_hash`,
  and `union_hash` match the pre-prune implementation exactly;
- CPU smoke reproduces the archived early states (`0.0001 -> 12.67494679`,
  `0.0002 -> 25.34683561`); entry helper tests `4/4`; `verify`/`plot` reproduce
  the fresh compact solution (76 states, 10 warnings, peak `641.34017`).

## Blocker

None.

## Next Action

Deep-trim `nonlocal_engineering_grip_stabilized.py` (still 8,459 lines) to the
adopted `fixed` + `stabilized_warning` path only:

1. replace `StabilizedEngineeringGripConfig` with a fixed/ warning-only config;
2. remove the Anderson proposal/merit/scalar-diagnostic machinery, the
   matched-load schedule machinery, and the archive-budget machinery;
3. simplify `_run_stabilized_attempt` and `run_stabilized_engineering_grip`
   policy/schedule/archive branches, then re-run the reachability prune;
4. delete the JAX backend public diagnostics that the adapter never calls;
5. re-validate with the CPU smoke and one full `run` comparison against the
   archived statistics before considering any further merge.

## 2026-09-27 — Second pass: merged back into one example file and GPU-only

- The user required a single direct solver file under `examples/fracture2d/`
  that keeps only the GPU path. `src/pinn/nonlocal2d/` was merged back with a
  mechanical, behavior-preserving linker (per-module private-name prefixes
  `_g_/_a_/_j_/_e_/_s_`, intra-file import resolution) into
  `examples/fracture2d/nonlocal_baseline.py` (`16,259` lines), and the
  `src/pinn/nonlocal2d/` package was deleted.
- GPU-only: `mechanics_adapter` is now a required `JaxA2CPUReplayAdapter` for
  `run_zero_damage_work_check` and `run_stabilized_engineering_grip`; the CPU
  CG reference chain (`solve_a2_fixed_outer_displacement`,
  `solve_engineering_grip_fixed_outer_displacement`, `solve_fixed_outer_displacement`,
  `_a_solve_outer`) was removed. CPU replay evaluation remains, because it is
  what makes the acceptance gates CPU-authoritative.
- Verification: `model_digest`/`static_digest` match the archived compact
  solution exactly; entry tests `4/4`; GPU CLI smoke `0 -> 2e-4` returns
  `25.34683561371731` (archived `25.3468356139`); `verify`/`plot` work on the
  full archived solution.
- Remaining optional trim inside the single file: the Anderson proposal/merit/
  scalar-diagnostic machinery, matched-load schedule, and archive-budget
  machinery (roughly 3,000 lines of top-level definitions plus dead branches
  in the config/runner/attempt loop). Those policies can no longer be selected
  only after that surgery; until then they remain present but unused by the CLI.

### Safe trim and warning-path regression (same day)

- Removed duplicated module-level imports (first occurrence kept) and added a
  configuration guard in `StabilizedEngineeringGripConfig.__post_init__` that
  rejects every non-`fixed` relaxation policy. Non-adopted policies are now
  impossible to select; the remaining Anderson/schedule/archive definitions are
  unreachable dead code.
- Regression protocol on GPU: strict `0 -> 2e-4` returns
  `25.346835626210126` (archived `25.3468356139`); warning `0 -> 2e-4` with
  `--max-stagger-iterations 1` gives the same `stabilized_warning_accepted`
  statuses as the pre-trim file with field differences `<= 3.2e-13`.
- Next (remaining removal): delete the now-unreachable Anderson proposal/merit/
  scalar-event machinery, `StabilizedLoadSchedule`, archive-budget helpers, and
  their dead branches in the attempt loop and runner, then re-run the same
  strict + warning GPU regression. Expected reduction is roughly 3-4k lines
  from the current 16,263.

### Dead-code removal completed (validated)

- Removed duplicated imports, rewrote the stabilized config to a fixed-only
  dataclass, froze the policy flags, and eliminated every Anderson /
  matched-load-schedule / archive-budget top-level helper and dead branch via
  iterative constant folding plus reachability pruning (92 top-level symbols
  removed): `16,263 -> 12,188` lines in
  `examples/fracture2d/nonlocal_baseline.py`.
- Non-`fixed` relaxation policies are rejected by the configuration.
- GPU regression: strict `0 -> 2e-4` final reaction `25.3468356138`
  (archived `25.3468356139`); warning `0 -> 2e-4` with one sweep gives
  bit-identical reactions and statuses versus the pre-removal file, with field
  differences `<= 5.7e-12`. Static `model_digest`/`static_digest` unchanged.
- Retained tests pass `10/10`; `verify` on the archived full solution still
  reports 76 states, 10 warnings, peak `641.34017`.

## 2026-09-27 — Closed: default behavior confirmed against the archived solution

- Final code: `examples/fracture2d/nonlocal_baseline.py`, `11,374` lines,
  GPU-only single file.
- Second risky pass removed the Anderson scalar-event validation chain and the
  unused JAX public diagnostics (`12,188 -> 11,374` lines) with the same GPU
  strict + warning regression evidence (warning reactions/statuses bit-identical
  to the pre-pass file; `model_digest`/`static_digest` unchanged).
- Default-path confirmation: a fresh run `0 -> 0.001` (10 strict steps, current
  defaults) matches the archived fresh solution to `4.3e-19` on loads, all
  `stabilized_converged` with matching warning flags, reaction max relative
  difference `4.2e-10`, displacement `3.2e-13`, damage `4.3e-11`, history
  `5.7e-9`.
- Coverage trace of the real strict + warning + `verify` + `plot` paths reaches
  every top-level definition (`0` fully unexecuted), so no free dead-code
  reduction remains.
- Archived. Next stage: [`fracture2d-nonlocal-pinn.md`](fracture2d-nonlocal-pinn.md).
