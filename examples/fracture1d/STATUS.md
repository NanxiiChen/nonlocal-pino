# Fracture1D Implementation Status

Last verified: 2026-08-14

This file records the live implementation state. The formulation and
implementation contract are in `README.md`.

## Scope

- FEM path: independent local AT2 phase-field fracture reference.
- PINN path: exact 1D condensed mechanics with `SparseD` in the damage term.
- Material: uniform `Gc=2.7` with a modulus imperfection
  `amplitude=0.1`, `center=0.5`, `width=0.05`.
- Base FEM loading: fixed `du=1e-5`.

## Current Conclusion

### Overall Status

- The direct nonlocal baseline is accepted and complete. It uses `Nx=1601`,
  `delta=0.006`, fixed `du=1e-5`, and `cell_mass_lumped` history.
- The full PINN continuation is **not accepted**. It has not yet passed the
  strict fixed-history solve at the localization load and therefore must not
  be used to claim a converged post-peak continuation.
- The active localization test is the direct-baseline snapshot at
  `u_bar=0.01175`.
- The corrected selected baseline is
  `/root/autodl-tmp/data/fracture1d/nonlocal_baseline/results_direct_cellhistory_nx1601_metricfix.npz`.
  It uses float64 operator coordinates and evaluates the final residual against
  the committed projected history.
- The corrected transition diagnostic plot is
  `/root/autodl-tmp/data/fracture1d/corrected_pinn_d_strain_diagnostic.png`.
  It compares the direct field/condensed strain at `u_bar=0.01176` with the
  final, not-yet-converged cell-history PINN state and marks its single
  stress-strain point.
- At `u_bar=0.01176`, that PINN state has damage `(MAE,L-infinity)` of
  approximately `(2.46e-5,6.15e-5)` versus the corrected direct field. Its
  reaction is `2.392e-7` versus the direct `2.559e-7`, and its peak condensed
  strain is `11.284` versus `11.028`. The direct baseline is the only accepted
  full stress-strain curve at present.
- `config.train.gate_mode` is now explicit. The base PINN training mode is
  `warning`: finite candidates continue to the configured final load and all
  failed tolerances are saved as `gate_warning`. Set it to `strict` for the
  rollback/cutback fixed-point validation behavior.
- A full-load best-architecture PINN diagnostic was run with width `200`, swish,
  spatial scale `16`, linear output, `33` trainable hats, residual MSE, Jacobi
  scaling, Adam, and `80000` maximum fixed-history updates. It used the
  corrected cell-history baseline and target-free continuation.
- The strict run rejected the first load step because the neural damage change
  was `9.19e-6`, above the `5e-6` gate. An explicitly labelled curve diagnostic
  then used relaxed outer gates only (`atol_damage_change=2e-4`, scaled residual
  `(2e-5,2e-4)`, history `0.5`, mechanics `1e-4`). It recorded `1481` steps up
  to `u_bar=0.01481`, then stopped at `0.01482` after the residual/history
  feedback became unstable. This curve is not an accepted converged solution.
- The recorded PINN curve peaks at approximately
  `(u_bar,reaction)=(0.01155,7.3075e-3)`, versus the direct peak
  `(0.01156,7.3064e-3)`. Over the recorded path, the maximum reaction
  difference is `9.06e-6`; after localization, the mean difference is about
  `4.37e-8`.
- Curve archive:
  `/root/autodl-tmp/tf-logs/fracture1d/best_cellhistory_fullcurve_width200_hat33_adam80k_diagnostic/results_best_cellhistory_fullcurve_diagnostic_partial.npz`.
  Full and localization plot:
  `/root/autodl-tmp/data/fracture1d/stress_strain_pinn_best_cellhistory_diagnostic.png`.
  Post-peak zoom:
  `/root/autodl-tmp/data/fracture1d/stress_strain_pinn_best_cellhistory_postpeak.png`.
- The completed warning-mode full run with the current best architecture is
  `/root/autodl-tmp/tf-logs/fracture1d/best_cellhistory_full_width200_hat33_adam80k_warning_curve_tb/`.
  It reaches `u_bar=0.02` in `2000` unique load steps. The run has one
  `gate_warning`, at the localization step `u_bar=0.01175`; all other load
  steps pass the explicitly relaxed curve-diagnostic gates. TensorBoard stores
  `train/damage_field` and `train/stress_strain_curve` at every `eval_every`
  step, plus `eval/direct_nonlocal/d_l2_abs` and
  `eval/direct_nonlocal/d_l2_rel` scalars.
- Static post-processing outputs for the completed run are
  `/root/autodl-tmp/data/fracture1d/d_fields_representative_best_cellhistory.png`,
  `/root/autodl-tmp/data/fracture1d/stress_strain_full_best_cellhistory_warning_tb.png`,
  and `/root/autodl-tmp/data/fracture1d/d_l2_error_best_cellhistory_warning_tb.png`.
- Two serial float32 ablations were completed to `u_bar=0.015` with the same
  width-200/swish/33-hat architecture. The `spatial_emb_dim=32`,
  `spatial_scale=16` run has peak `(u_bar,reaction)=(0.01156,7.30756e-3)`,
  maximum full-path reaction error `1.03e-5`, post-transition maximum error
  `1.22e-7`, final absolute damage L2 `6.83e-4`, and two gate warnings.
- The float32 `spatial_emb_dim=64`, `spatial_scale=4` run has a similar
  pre-localisation peak, but delays the reaction drop by one step at
  `u_bar=0.01175`. Its full-path reaction error is `7.15e-3` and final
  absolute damage L2 is `3.14e-3`; it is not a candidate for the default.
- The `32/16` float32 run is now the default configuration and preferred
  acceleration path. The float64 full run remains the high-precision reference;
  the faster but less reliable `64/4` run is not a default candidate.
- `temporal_emb_dim=8` and `temporal_scale=1.0` are shared architecture fields,
  but temporal embedding is disabled in fracture1d and the static network path
  receives constant `t=0`. The load coordinate is handled by explicit
  continuation and history state, so these fields are inert for this example.

Float32 artifacts:

```text
/root/autodl-tmp/tf-logs/fracture1d/float32_best_spatial32_scale16_u0015/
/root/autodl-tmp/tf-logs/fracture1d/float32_best_spatial64_scale4_u0015/
/root/autodl-tmp/data/fracture1d/float32_spatial_embedding_comparison_u0015.png
```

### Acceptance And Reference

- With Jacobi scaling, the strict fixed-history gate is scaled residual
  `MAE <= 5e-6` and `L-infinity <= 2e-5`; the strict staggered damage-change
  gate is `5e-6`. Warning mode retains these values for reporting but does not
  reject a finite load-step candidate.
- At `u_bar=0.01175`, the saved direct target evaluated with its committed
  history has scaled residual approximately `(6.79e-9, 2.81e-8)` for
  `(MAE,L-infinity)`. Solving the same fixed linear system directly gives
  approximately `(5.0e-17,5.6e-16)`.
- The tolerance is therefore not the primary problem or a reference-noise
  artifact. The saved target/history stagger mismatch is about `1.94e-6` in
  damage L-infinity, below the damage-change gate.

### What Has Been Ruled Out

- The former nodal-history grid spike is not the active production reference;
  the selected baseline now uses DG0 cell strain energy with CG1 lumped-mass
  projection. The former nodal archive remains diagnostic-only.
- The failure is not caused by JAX recompilation, variable-shape batching,
  CUDA OOM, deadlock, or a mechanics reconstruction error. Fixed-shape compile
  logging and the regression suite are clean.
- Increasing width alone is not sufficient. Width `200` improves the mean
  residual and damage profile, while width `400` is harder to optimize and is
  worse at the tested budgets.
- Exact elimination of local hat coefficients is mathematically correct but
  does not solve the broad trunk shoulder outside the local patch.

### Best Controlled Results

- Best trainable candidate so far: width `200`, linear output, swish, 33 trainable
  CG1 hats, joint Adam, 80,000 residual updates; scaled residual is
  `(1.04e-6,2.46e-5)`. It narrowly fails only the pointwise L-infinity gate.
- Width `200`, linear output without hats reaches `(1.78e-6,4.44e-5)` after
  60,000 updates.
- The exact post-processing correction of the width-200 field with 33 hats
  reaches scaled L-infinity `8.83e-6`, but that correction is not a neural
  forward path and cannot be treated as production convergence.
- The exact alternating hat pilot is worse than joint Adam: width `200` with
  33 hats reaches scaled L-infinity `5.35e-5` after 60,000 updates.
- The native Optax RPROP implementation was added as a default-off diagnostic
  with the Manav et al. (2024) settings: initial step `1e-5`, `eta_minus=0.5`,
  `eta_plus=1.2`, and step bounds `[1e-10,50]`. On the frozen cell-history
  localization case at `u_bar=0.01175`, full-network residual-only RPROP passes
  the strict gate after 34,800 updates with scaled
  `(MAE,L-infinity)=(1.278e-6,1.998e-5)`. The largest residual is at the center
  node `x=0.5`; this is the first non-postprocessed neural solve to pass the
  frozen pointwise gate.
- The earlier RPROP transition replays were invalid as controlled comparisons:
  they initialized from a projected cell-history snapshot but generated trial
  history from nodal strain energy. For the same `d` at `u_bar=0.01176`, this
  hybrid path gives history maximum about `54.6`, while the configured
  cell-history update gives about `25.0`. Their outer history-error values must
  not be used to judge the corrected model.
- The corrected cell-history RPROP replay from `0.01175 -> 0.01176` still does
  not converge in 200 outer staggers. Its minimum observed damage, cell-history,
  and mechanics changes are approximately `2.05e-5`, `8.13e-3`, and
  `2.77e-6`; the final values are `1.55e-4`, `4.63e-1`, and `4.25e-5`.
  This is now a valid same-discretization failure.
- A direct SparseD control replay from the accepted `u_bar=0.01175` state with
  `du=1e-6` completes all ten increments to `0.01176`, with three outer
  staggers per increment. Its final damage differs from the existing direct
  `du=1e-5` target by only `2.21e-10`; the final reaction is
  `2.5589e-7`. The smaller load step therefore does not change the discrete
  reference solution.
- The former PINN `du=1e-6` replay also used the hybrid nodal-history path and
  is invalidated by the same issue. The direct `du=1e-6` control remains valid,
  but the corrected PINN small-step replay has not yet been rerun.
- The fixed-history defect diagnostic now solves `A(H)e=r` with the same direct
  SparseD system used by the baseline. The accepted frozen RPROP field has
  defect `L-infinity≈1.75e-5`, which is above the outer damage-change gate
  `5e-6`; this quantifies why the scaled residual gate is insufficient.
- A full-network scaled-residual LM refinement after 60,000 RPROP updates
  accepted six steps but did not reduce the defect below `1.75e-5`. A second
  20,000-RPROP/20-LM run reduced defect `L-infinity` from `1.93e-5` to
  `1.92e-5` and still failed the fixed-history gate.
- A defect-objective LM refinement after 20,000 RPROP updates accepted 14 of
  20 steps and reduced defect `L-infinity` from `1.93e-5` to `1.90e-5`. The
  improvement is real but far from the `5e-6` outer requirement; the remaining
  defect is oscillatory at the localization-center grid scale.

### Current Blocker

- The remaining residual is concentrated in a few low-history flank nodes just
  outside the hat patch, approximately `x=0.48875` and `x=0.51125`. This is a
  broad trunk/shoulder representation and optimization issue coupled to the
  localized history profile, not a missing global tolerance adjustment.
- The exact fixed-history defect of the RPROP/LM fields peaks near
  `x=0.495625` and changes sign across adjacent grid nodes. This is a local
  pointwise representation defect, not an outer load-step-size problem.
- The metric audit found and corrected two implementation errors: PINN history
  now remains in DG0 cell space and is projected only for damage rows, and the
  final residual gate is recomputed at the history that will be committed.
  Solve-history and committed-history residuals are archived separately.
- The default curve-training configuration now uses the best tested
  width-`200` swish network with `33` trainable hats, linear output, residual
  MSE, and the warning-mode diagnostic gates. The strict gate mode remains
  available for a separately configured validation run.

### Next Decision Gate

1. Rerun the corrected cell-history `0.01174 -> 0.01175` transition and the
   corrected `du=1e-6` diagnostic; do not compare new results to the invalidated
   hybrid-history outer metrics.
2. Defect diagnostics and full-network LM have now been tested; LM improves the
   residual objective but does not close the pointwise defect gate.
3. Do not promote LM or change the production optimizer. The next diagnostic
   should address the measured local representation defect directly, most
   likely with a defect-aware trainable local enrichment. A joint
   displacement/damage network remains a separate fallback and is not justified
   by the current evidence.

## Implemented

- The FEM solver uses a staggered mechanics/history/damage iteration at each
  displacement increment.
- `H_prev` is immutable during an attempted load step. The history used in an
  inner iteration is `max(H_prev, psi_plus)` and only the converged `H_next` is
  committed.
- Tensile energy and history are evaluated on DG0 cell values from the DG0
  strain field.
- The damage equation is solved directly as a linear variational problem with
  fixed `H_used`. The raw solution is the next damage iterate.
- The PINN condensed displacement is reconstructed by integrating the strain
  once; the interior field is no longer multiplied by stress a second time.
- PINN fixed-history acceptance checks damage residual MAE, L-infinity norms, and
  the maximum neural damage change; the latter is a hard staggered acceptance
  condition, not only a diagnostic. With Jacobi scaling enabled, scaled
  residuals are the acceptance quantities and raw residuals remain diagnostics.
- The base PINN configuration resets Adam/SOAP optimizer state at the start of
  each fixed-history damage solve while retaining the network warm start.
- The base PINN uses an unconstrained linear damage output, float32 arrays, and
  Jacobi-scaled residual MSE. Float64 remains an explicit reference option.
  Pilot scripts may select residual MAE, and
  `loss.objective` also supports the full-grid nonlocal quadratic energy.
- Raw and Jacobi-scaled damage residuals are both recorded. The scaled system
  diagonal is `1 + N_l**2 * D.diagonal + (2*N_l/N_G)*H`.
- The FEM reference does not impose damage box bounds, pointwise
  `d_previous <= d`, active-set iterations, under-relaxation, clipping, or
  Anderson acceleration.
- PINN healing is recorded as `min(d_n-d_prev)` and logged diagnostically; it
  is not used as a second damage constraint.
- Rollback/cutback loading, reaction reconstruction, DG0 stress diagnostics,
  and `stress_strain.png` output are implemented.
- Solver output includes `x_H`, `history_space`, `history_method`, loading
  metadata, and fixed-point tolerances.
- The MPI finalize warning from the serial FEniCS process was removed by
  disabling mpi4py finalization after DOLFIN owns process shutdown.
- The accepted FEM archive now saves the initial state and every fixed
  `du=1e-5` load state, enabling load-aware field comparison.
- `SparseD.to_scipy()` exists only for the direct nonlocal baseline: it converts
  the fixed operator once to a SciPy CSR matrix for the sparse linear solve.
  The PINN training path never calls it.
- The PINN hot path uses fixed-shape JIT helpers for network evaluation,
  mechanics, residual checks, and convergence reductions. Loss scalars are no
  longer copied from GPU to host after every optimizer update.

## Historical Diagnostics

The following files were produced before the classical-only refactor and are
not accepted as the final FEM reference:

- `/root/autodl-tmp/data/fracture1d/results_fracture1d_base_nx100.npz`
- `/root/autodl-tmp/data/fracture1d/results_fracture1d_corrected_nx100.npz`

Those runs used temporary bound-constrained damage updates. A previous
`mesh.nx=400` run lost outer fixed-point convergence near `u_bar ~= 0.01175`,
but that failure was not reproduced by the direct classical solve below.

The older `/root/autodl-tmp/data/fracture1d/results_fracture1d.npz` file is a
homogeneous, `amplitude=0` result and is not the current localized reference.

## Verified Classical Runs

All runs use the base fixed increment and the direct classical history-field
damage solve:

- `nx=100`: `/root/autodl-tmp/data/fracture1d/classical_nx100/results_classical_nx100.npz`
  completed 2000 steps with zero cutbacks; final reaction stress is
  approximately `1.052e-6`.
- `nx=400`: `/root/autodl-tmp/data/fracture1d/classical_nx400/results_classical_nx400.npz`
  completed 2000 steps with zero cutbacks; final reaction stress is
  approximately `3.170e-7`.
- `nx=2000`: `/root/autodl-tmp/data/fracture1d/classical_nx2000/results_classical_nx2000.npz`
  completed 2000 steps with zero cutbacks; final reaction stress is
  approximately `3.920e-7`.
- All archives report `history_space=DG0_classical_max` and
  `history_method=Miehe_max_history`, and all saved fields are finite.
- The unconstrained classical solve reaches saved damage maxima of about
  `1.0021` (`nx=100`), `1.0007` (`nx=400`), and `0.99997` (`nx=2000`). The
  fine-grid solution remains inside the physical damage range; the small
  coarse-grid excursions are recorded diagnostics, not suppressed by clipping
  or rejected by an extra constraint.
- The peak reactions are approximately `7.3095e-3` at `u_bar=0.01159`
  (`nx=100`), `7.3066e-3` at `u_bar=0.01157` (`nx=400`), and `7.3064e-3`
  at `u_bar=0.01156` (`nx=2000`). The fine-grid run is the accepted local
  reference for the current PINN comparison.

## Direct Nonlocal Baseline

- Pilot: `/root/autodl-tmp/data/fracture1d/nonlocal_baseline/pilot_nx81.npz`.
- Selected full baseline:
  `/root/autodl-tmp/data/fracture1d/nonlocal_baseline/results_direct_cellhistory_nx1601_metricfix.npz`.
- The selected baseline uses `Nx=1601`, `delta=0.006`, fixed `du=1e-5`, a
  direct sparse linear solve for every fixed-history damage update, and
  `cell_mass_lumped` history: DG0 cell strain energy is projected to nodal
  values with the CG1 lumped-mass rule.
- The former nodal-history archive remains available for comparison at
  `/root/autodl-tmp/data/fracture1d/nonlocal_baseline/results_direct_nonlocal_nx1601.npz`.
- It completes 2000 load steps with zero cutbacks and at most 171 staggered
  iterations.
- Peak reaction is approximately `7.30637e-3` at `u_bar=0.01156`, matching the
  fine FEM peak while retaining the finite-horizon nonlocal model.
- Maximum committed-history residuals for the selected cell-history baseline
  are approximately `2.97e-7` (MAE) and `4.12e-6` (L-infinity). The maximum
  immediate solve-history residual L-infinity is approximately `1.46e-10`; all
  saved fields are finite.
- The baseline is now the first target for PINN optimization. Its post-peak
  fields are not expected to equal the local FEM fields because
  `delta/N_l=0.25` is a finite-horizon choice.
- Stress-strain comparison plot:
  `/root/autodl-tmp/data/fracture1d/stress_strain_comparison.png`. The local FEM
  and direct baseline curves are nearly coincident before localization, with
  peak stress about `7.306e-3`; both show the abrupt drop at the localized load.
  The plotted PINN trace is the old pre-fix exploratory TensorBoard trace and is
  marked diagnostic-only because it advanced without the restored damage-change
  gate.

## Loaded Profile Width

- The dimensional scale is `L0=1 mm`, so `l0=0.024 mm`, `delta=0.006 mm`, and
  the `Nx=1601` grid spacing is `h=0.000625 mm`. Thus `l0/h=38.4`,
  `delta/h=9.6`, and `delta/l0=0.25`; the nominal diffuse interface is not
  under-resolved by the grid.
- For the local stationary AT2 exponential profile, the full width at
  `d=exp(-1)` is `2*l0=0.048 mm`, and the width at `d=exp(-2)` is
  `4*l0=0.096 mm`.
- The loaded `u_bar=0.01175` field has a nonzero background damage of about
  `0.182`. After normalizing the excess damage above that background, the
  `exp(-1)` full width is approximately `0.0509`, `0.0465`, and `0.0391 mm`
  for `Nx=801`, `1601`, and `3201`, respectively. The broad `0.1` level width
  is approximately `0.145`, `0.143`, and `0.140 mm`.
- The former nodal-history path produced a separate grid-scale core with
  approximately `H_max=39.8`, `159`, and `636` on `Nx=801`, `1601`, and `3201`.
  These values and the associated adjacent damage jump are retained as a
  historical diagnostic, not as the selected reference.
- The selected cell-history baseline removes that nodal-history comparison from
  the production target. Its `u_bar=0.01175` profile has a smooth diffuse
  envelope with a concentrated multi-layer history peak; the remaining PINN
  difficulty is now recorded as a fixed-history optimization problem.
- Refinement archives:
  `/root/autodl-tmp/data/fracture1d/nonlocal_baseline/results_direct_nonlocal_nx801_u01175.npz`
  and
  `/root/autodl-tmp/data/fracture1d/nonlocal_baseline/results_direct_nonlocal_nx3201_u01175.npz`.

## PINN Pilot

- A corrected-mechanics pilot completed 20 fixed load steps at `Nx=321` and
  `u_final=2e-4` using `delta=0.006` and 500 Adam updates per fixed history.
- Output:
  `/root/autodl-tmp/tf-logs/fracture1d/legacy-data/pinn_pilot/results_pinn_pilot_nx321_v2.npz`.
- The pilot has maximum residuals approximately `3.84e-7` (MAE) and
  `5.56e-7` (L-infinity), reaction error approximately `4.24e-12`, and final
  relative damage error approximately `2.80e-3` against the load-interpolated
  local FEM field.
- The first `Nx=161` pilot was correctly rejected because `dx > delta` left
  the operator without neighbors. The grid was increased rather than
  weakening the operator guard.

## PINN Continuation Exploration

- The full PINN uses the same `Nx=1601`, `delta=0.006`, and fixed `du=1e-5` as
  the direct nonlocal baseline. The local FEM reference remains `nx=2000`.
- Compile logging on a two-load-step run showed one compilation each for the
  fixed-shape `step`, `mechanics`, residual `check`, and convergence `check`
  executables. No load-dependent JAX recompilation was observed.
- The previous strict run reached approximately `step=597` before the process
  was interrupted. It had no cutback or numerical exception; the last recorded
  stagger count was `23`, with residual MAE `4.76e-6` and L-infinity `1.98e-5`.
- The exploratory run used only command-line overrides, leaving the strict
  base config unchanged:
  `atol_damage_residual=2e-5`, `atol_damage_residual_inf=1e-4`,
  `damage_check_every=200`, `damage_iterations=5000`, and `eval_every=20`.
- Output directory:
  `/root/autodl-tmp/tf-logs/fracture1d/legacy-data/pinn_explore_coarse_nx1601/`.
- This coarse run passed the former slow region through `step=1100` and reached
  `step=1175` (`u_bar=0.01175`). It then failed at `u_bar=0.01176`, close to
  the FEM/direct-baseline localization peak:
  `damage=1.395e-5`, residual MAE `1.995e-5`, residual L-infinity
  `1.730e-4`, stress change `4.284e-8`, history change `4.344e-9`, and
  mechanics change `3.632e-8`.
- Healing diagnostics began near `step=1159`; they are warnings only and are
  not hidden by clipping or converted into a second irreversibility constraint.
- The failure is therefore a localized damage optimization/residual problem,
  not a shape change, repeated compilation, GPU OOM, or deadlock. No final
  exploratory `.npz` was produced; checkpoints reached approximately `step=1150`.

## Historical Frozen-History Localization Pilots

- These archived runs use the former nodal-history baseline and are retained to
  explain the original localization failure. Current cell-history results are
  recorded in the later diagnostic sections.
- Pilot code: `examples/fracture1d/frozen_pilot.py`. Each case uses an exact
  direct-baseline snapshot at `u_bar=0.01172`, `0.01174`, or `0.01175`, a fresh
  fixed-history solve, and the same full `Nx=1601` SparseD grid.
- Exact MAE/reset output:
  `/root/autodl-tmp/tf-logs/fracture1d/legacy-data/frozen_pilots/pilot_mae_reset_exact_5000.npz`.
  At the three loads, residual `(MAE, L-infinity)` was respectively
  `(1.60e-3, 5.54e-3)`, `(1.35e-3, 6.39e-3)`, and
  `(7.63e-2, 1.24e1)` after 5000 updates.
- Exact MSE/reset output:
  `/root/autodl-tmp/tf-logs/fracture1d/legacy-data/frozen_pilots/pilot_mse_reset_5000.npz`.
  MSE improves the pre-peak cases to `(1.18e-4, 6.71e-4)` and
  `(2.70e-5, 2.09e-4)`, but the localization case remains
  `(2.86e-1, 2.02e0)`. Carrying Adam moments gives nearly the same pre-peak
  values and also saturates at localization:
  `/root/autodl-tmp/tf-logs/fracture1d/legacy-data/frozen_pilots/pilot_mse_carry_5000.npz`.
- A 50,000-step supervised fit to the direct target was used only as a
  representation diagnostic. For the localization snapshot, Fourier scales
  `4`, `64`, and `256` gave damage L-infinity errors `1.08e-2`, `6.28e-3`, and
  `7.22e-3`; the corresponding PDE residual L-infinity values were
  `6.40e3`, `3.73e3`, and `4.29e3`:
  `/root/autodl-tmp/tf-logs/fracture1d/legacy-data/frozen_pilots/representation_scale4_warm50000.npz`,
  `/root/autodl-tmp/tf-logs/fracture1d/legacy-data/frozen_pilots/representation_scale64_warm50000.npz`,
  `/root/autodl-tmp/tf-logs/fracture1d/legacy-data/frozen_pilots/representation_scale256_warm50000.npz`.
- The direct target at `u_bar=0.01175` has `d(0.5)=0.999914`, while the two
  adjacent values are approximately `0.8265`; its history maximum is about
  `159`. This one-node spike explains why a finite-capacity continuous network can
  match the mean damage while failing the pointwise residual by orders of
  magnitude. The target itself evaluated through the PINN float32 SparseD has
  residual approximately `(4.00e-6, 4.40e-5)` due to the float32 matvec, versus
  `(1.27e-6, 4.72e-6)` in the direct SciPy baseline.
- The restored fixed-point gate has not yet been followed by a full
  continuation rerun; it will now reject the previously accepted wrong branch
  instead of silently advancing past localization.

## Historical Linear/Scaled Optimization Exploration

- The first entries in this section use the former nodal-history target; the
  current cell-history linear results are listed under the width and trainable
  hat diagnostics below.

- `SparseD` now preserves the dtype of the input grid. The float64 target probe
  at `u_bar=0.01175` has raw residual approximately `(1.83e-6, 7.06e-6)` in
  JAX, below the current raw diagnostic tolerances.
- Linear damage output with Jacobi-scaled residual MSE reaches the scaled gate
  at `u_bar=0.01172` and `0.01174` in approximately 600 and 550 optimizer
  iterations. Output:
  `/root/autodl-tmp/tf-logs/fracture1d/linear_scaled_x64_frozen_5000_v2.npz`.
- The same frozen solve at `u_bar=0.01175` remains unresolved after 5000
  updates: raw residual is about `(9.97e-1, 1.47e3)` and scaled residual about
  `(2.90e-4, 8.24e-3)`. The output damage reaches `max(d)≈0.9975` but the
  pointwise localized equation is not solved.
- Full-grid quadratic-energy pilots reduce the localized raw residual compared
  with raw MSE and trigger the branch in the transition replay, but do not yet
  satisfy the fixed-point gate:
  `/root/autodl-tmp/tf-logs/fracture1d/linear_energy_x64_frozen_20000.npz`.
- `transition_pilot.py` replays `0.01174 -> 0.01175` using the direct-baseline
  previous field. The hybrid pilot reaches `max(d)>0.99` around staggered
  iterations `90-115`, confirming that the local branch can be triggered, but
  the post-localization field still oscillates and is not accepted. Gaussian
  enrichment and a deterministic L-BFGS post-optimizer were tested as pilot
  options but are not enabled in the base configuration.
- The pilot now supports explicit fixed-history continuation with
  `H(lambda)=H_prev + lambda*(H_used-H_prev)`. With four continuation
  substeps and residual MSE, all substeps pass through stagger `119`; the
  localized branch begins near stagger `116`, then the final substep fails at
  stagger `120` with `max(d)≈0.554` and scaled residual L-infinity
  `≈2.04e-5`. Eight substeps delay that failure to stagger `115`, reaching
  `max(d)≈0.619` with scaled residual L-infinity `≈2.15e-5`. The method
  tracks the branch farther without loosening the gate, but does not yet
   produce an accepted transition.

## Pseudo-Time Exploration

- `pseudo_time_pilot.py` implements the fixed-history implicit residual
  `q_tau=(d_new-d_old)/tau+R(d_new;H)` and a direct vector reference using
  `(I+tau*A_H)d_new=d_old+tau*b_H`. The production history update remains
  unchanged.
- Direct vector pseudo-time converges rapidly for the localized target with
  `tau=1e-3` or `1e-2`; after 5000 steps the damage L-infinity error is about
  `2.54e-5` and `1.94e-6`, respectively, relative to the direct target.
- The PINN fixed-history raw-MSE pilot does not pass the gate. The
  `tau=1e-5` fixed case ends with scaled residual L-infinity about
  `2.26e-2` and damage L-infinity error about `1.96e-1`:
  `/root/autodl-tmp/tf-logs/fracture1d/pseudo_time_tau1e-5_raw_5000.npz`.
- Jacobi-scaling the entire pseudo-time residual is worse in this pilot; its
  final scaled residual L-infinity is about `6.59e-1`:
  `/root/autodl-tmp/tf-logs/fracture1d/pseudo_time_tau1e-5_jacobi_5000.npz`.
- The finite-difference adaptive step estimate was tested with raw MSE. It
  shrinks `tau` from `1e-3` to roughly `2.2e-5--6.1e-5`, but does not pass the
  gate; final scaled residual L-infinity is about `1.92e-2`:
  `/root/autodl-tmp/tf-logs/fracture1d/pseudo_time_adaptive_raw_5000.npz`.
- The direct reference validates the implicit pseudo-time equation, while the
  PINN result still indicates a localized representation/optimization
  bottleneck rather than a failure of the fixed-history algebraic system.
- After switching the configured baseline to `cell_mass_lumped` history, the
  frozen linear/scaled PINN still passes `0.01172` and `0.01174`, but the
  `0.01175` case remains rejected after 5000 updates with scaled residual
  `(MAE,L-infinity)≈(1.50e-4,1.09e-2)`:
  `/root/autodl-tmp/tf-logs/fracture1d/frozen_cellhistory_linear_scaled_x64_5000.npz`.
  The reference is smoother, but pseudo-time/ordinary global MLP optimization
  is still not sufficient for the strict localized solve.

## Output Parameterization and Activations

- The diagnostic output `d=exp(-raw**2)` was tested with `d_init=0.01`,
  `spatial_scale=16`, and the new cell-history reference. At 5000 residual
  updates, scaled residual `(MAE,L-infinity)` was approximately
  `(2.40e-5,5.55e-4)` for `tanh`, `(1.18e-5,7.85e-4)` for `gelu`, and
  `(1.04e-5,4.28e-4)` for `swish`.
- Extending `swish` to 20000 updates reduced the scaled residual to
  `(3.87e-6,1.24e-4)`; `spatial_scale=64` gave `(1.07e-5,1.10e-4)`.
  Neither passed the `2e-5` L-infinity gate. The bounded output improves the
  localized solve substantially, but its center gradient also saturates as
  `d` approaches one. It remains a diagnostic mode, not the production default.

## Width Capacity Diagnostic

- The width sweep used the same `u_bar=0.01175` cell-history snapshot,
  `spatial_scale=16`, Jacobi-scaled residual MSE, and 20,000 supervised warm
  start updates. It did not change the production configuration or acceptance
  tolerances.
- With `exp_square`/swish, width `200` improved the 20,000-update result to
  scaled `(MAE,L-infinity)≈(5.27e-6,1.89e-4)` and damage MAE `1.79e-5`. After
  60,000 residual updates it reached `(1.60e-6,3.58e-5)`; a 100,000-update
  run became unstable and regressed.
- With linear output and the swish diagnostic activation, width `200` reached scaled
  `(2.56e-6,5.83e-5)` after 20,000 updates and `(1.78e-6,4.44e-5)` after
  60,000 updates. Damage MAE at 60,000 updates was `2.40e-6`.
- Width `400` did not improve the fixed-history solve at the same budget:
  `exp_square`/swish with the original learning rate stalled during warm start;
  lowering the rate produced scaled L-infinity `3.12e-4`, and production-linear
  output produced `1.01e-4`.
- Width `200` improves representation and mean residuals, but the remaining
  error is still concentrated on a few low-history flank nodes near the
  localization core. The current warning-mode curve configuration uses width
  `200`; strict acceptance remains unresolved.

## Local Correction Diagnostic

- Applying the existing CG1 hat-basis least-squares correction to the best
  linear width-`200` field reduces the scaled residual L-infinity from
  `4.44e-5` to `8.83e-6` with a 33-node patch spanning `0.02` in `x`.
- A 17-node patch remains just above the gate at `2.45e-5`; larger patches do
  not materially reduce the maximum after 33 nodes. This confirms that the
  remaining defect is locally correctable rather than a global tolerance
  mismatch.
- The exact least-squares correction remains a post-processing diagnostic only;
  the separate trainable branch below is not used for production acceptance.

## Trainable Hat Enrichment Pilot

- The PINN now has a default-off `arch.use_local_hat_enrichment` branch with
  explicit odd `arch.local_hat_count`; it adds trainable CG1 coefficients to
  the damage output and records the branch metadata in result archives.
- Width `200`, linear output, swish, and `33` trainable hats reached scaled
  residual `(MAE,L-infinity)≈(1.88e-6,5.83e-5)` after 20,000 updates,
  `(1.19e-6,2.80e-5)` after 60,000, and `(1.04e-6,2.46e-5)` after 80,000.
  It remains narrowly outside the strict pointwise gate; the largest errors
  sit one grid layer beyond the patch support.
- Increasing the patch to `65` hats at 20,000 updates gives L-infinity
  `9.29e-5`; the `37`-hat run also destabilizes its supervised warm start.
  More coefficients are therefore not automatically better.
- Width `64` with `33` hats is substantially worse at 20,000 updates. The
  diagnostic supports width `200` plus local enrichment as the next candidate,
  but neither is enabled in the production base configuration yet.
- The pilot now supports `--hat-update=exact` for linear output. It freezes hat
  gradients during Adam updates and applies the existing weighted least-squares
  coefficient solve at each residual check. Width `200` with 33 hats reaches
  scaled L-infinity `9.53e-5` after 20,000 updates and `5.35e-5` after 60,000;
  65 hats reaches `1.26e-4` after 20,000. The exact subsolve is stable and
   mathematically correct, but the trunk shoulder optimization is worse than
   joint Adam in this pilot. It remains diagnostic-only.

## Hard-Point Loss Weighting Pilot

- The residual loss now supports a residual-only top-k term controlled by
  `loss.hard_point_count`, `loss.hard_point_weight`, and
  `loss.hard_point_mode`. The default weight is `0.0`, so the production loss
  is unchanged. The selected rows come from the current scaled residual; no
  reference damage or prescribed crack location is used.
- Direct dynamic top-k selection is not stable with the current Adam setup.
  At width `200`, linear output, swish, `top_k=16`, and 20,000 updates, weights
  `1.0` and `10.0` give scaled L-infinity approximately `8.25e-4` and
  `3.83e-3`; weights `0.1` and `0.01` give approximately `5.78e-3` and
  `8.18e-4`. A lower-learning-rate `0.01` run also regresses to approximately
  `6.83e2` raw scaled L-infinity. The active set changes every optimizer step,
  so the objective is piecewise-discontinuous and Adam chases moving rows.
- We also tested weaker dynamic weighting and block-frozen row selection. The
  best tested weak dynamic case (`top_k=64`, weight `0.01`) gives scaled
  `(MAE,L-infinity)≈(3.06e-6,6.19e-5)`, while the unweighted width-200 baseline
  at the same budget gives `(2.56e-6,5.83e-5)`. Frozen selection with
  `top_k=16`, weight `0.01` gives `(6.59e-6,1.24e-4)`; with weight `0.001`
  and checks every 200 steps it gives `(3.46e-6,1.06e-4)`.
- Therefore extra hard-point weighting does not currently solve the pointwise
  gate. The failure is not evidence that the idea is invalid for 2D, but this
  naive top-k/Adam implementation is not a production candidate. A future
  version would need a smooth residual weighting or a trust-region/active-set
  optimizer rather than a hard row switch inside every Adam step.

## Smooth Weighting And Frozen-Fourier Ablations

- A smooth residual-aware mode was added. It uses a bounded `tanh` score of the
  current scaled residual magnitude, normalized to mean one, with a detached
  weight so the objective remains differentiable. At width `200`, linear output,
  swish, `temperature=4`, and weight `0.5`, the 20,000-update result is scaled
  `(MAE,L-infinity)≈(5.18e-6,2.35e-4)`, worse than the unweighted baseline
  `(2.56e-6,5.83e-5)`.
- A separate `arch.train_spatial_embedding=False` ablation freezes the Fourier
  kernel while training the trunk. With the same width, activation, and budget,
  it gives scaled `(7.95e-6,5.04e-4)`. The trainable Fourier kernel is therefore
  useful for this current representation; freezing it is not a solution.
- Neither ablation passed the frozen gate, so no target-free transition was run
  for them. Both options remain default-off diagnostics and production settings
  are unchanged.

## Local-Equilibrium Correction Pilot

- A default-off `arch.local_equilibrium` branch was implemented. It uses the
  analytically derived pointwise equilibrium
  `d_loc=(beta*H)/(1+beta*H)` and trains only an additive correction. The
  correction receives bounded `d_loc` and previous damage as immutable context;
  no current-load target is used.
- The first implementation exposed an input-scale inconsistency: the loss used
  raw `beta*H` while the residual check used bounded `d_loc`. That invalid pilot
  was corrected, and direct loss/residual/check consistency was verified.
- The valid cold-start pilot uses width `200`, swish, `spatial_scale=16`, linear
  output, zero supervised warm-start steps, and 20,000 residual-MSE updates.
  It reaches scaled `(MAE,L-infinity)≈(2.65e-4,5.45e-3)`, compared with the
  ordinary width-200 baseline `(2.56e-6,5.83e-5)` at the same budget. Reducing
  the learning rate by 100 gives `(3.95e-4,4.46e-3)`, so this is not merely a
  step-size issue.
- The decomposition is mathematically valid, but the correction is not small:
  at the shoulder the target damage is about `0.94` while `d_loc` is about
  `0.36`, requiring a correction near `0.58`. It therefore does not remove the
  difficult global/nonlocal shoulder for this history field. It remains
  diagnostic-only and was not advanced to target-free transition.

## Target-Free Incremental Transfer Pilots

- `transition_pilot.py` now supports explicit architecture settings, residual-only
  warm start at `H_previous`, optional output-layer-only staging, and adaptive
  history midpoint insertion. The duplicate `local_equilibrium` argument in the
  production `run_training` loss factory was also removed. The output-only
  gradient path is covered by regression tests.
- The residual-only warm start at `u_bar=0.01174` reaches scaled residual
  `(MAE,L-infinity)≈(6.12e-7,3.14e-6)` and differs from the previous-load direct
  damage by only `9.77e-6` in L-infinity. No damage target is used in the loss.
- Full-network residual transfer with width `200`, swish, spatial scale `16`,
  linear output and four history substeps follows the branch to
  stagger `119`, where it fails with `max(d)≈0.887` and scaled L-infinity
  `2.67e-5`. This is the target-free counterpart of the earlier continuation
  failure and confirms that previous-load supervision was not the cause.
- An optimizer audit found that the former `post_lr=2e-4` argument did not
  affect pure residual runs: they used the optimizer built first at the base
  `2e-3` rate. The relative comparisons above remain valid because they all
  followed that path, but they must not be described as `2e-4` experiments.
  The pilot now uses explicit `--residual-lr` for every residual Adam phase and
  `--continuation-optimizer` for full-network history substeps, while the
  original configured rate is retained only for an energy phase. Warm start
  and output-only staging remain Adam operations.
- Output-layer-only staging once per history substep suppresses the transition:
  50 and 200 staged updates end at `max(d)≈0.411` and `0.418`, with scaled
  L-infinity `1.06e-5` and `7.91e-6`, respectively. Staging once per outer
  stagger is even more restrictive: 10 and 200 updates end at `max(d)≈0.398`
  and `0.365`. These are not accepted solutions; the damage branch is being
  delayed rather than solved. This transfer schedule is therefore not a
  production candidate.
- Adaptive history bisection does track farther along the branch. With a maximum
  of 8 history intervals it inserts `lambda=0.625` and `0.6875` before failing
  at scaled L-infinity `2.53e-5`; with capacity 16 it reaches
  `lambda=0.814453125` and fails at `2.18e-5`. Finer history homotopy improves
  progress but does not pass the `2e-5` pointwise gate.
- Incremental transfer and history homotopy are useful diagnostics and remove
  target-supervision ambiguity, but neither solves the post-localization
  fixed-history optimization. Further staged-count scans are not justified; the
  next method should address residual least-squares conditioning or local
  representation directly.

## Frozen-Trunk Output-Head Least Squares

- `frozen_pilot.py --output-head-lstsq` now freezes a plain MLP trunk and solves
  the exact Jacobi-scaled residual least-squares problem over its 200 output
  weights and scalar bias. The fixed-history equation and linear output make
  this a 1601-by-201 linear problem; no damage target enters the subsolve.
- A target-free cold residual trunk trained for 20,000 updates changes from
  scaled `(MAE,L-infinity)=(7.50e-6,3.66e-4)` to
  `(8.54e-6,2.57e-4)` after the exact head solve. The matrix has full column
  rank and condition number `7.75e4`.
- A separate representation diagnostic with the existing 20,000-step
  supervised warm start followed by 20,000 residual updates changes from
  `(8.28e-6,3.10e-4)` to `(8.64e-6,2.17e-4)`. It also has full column rank,
  with condition number `7.29e4`. This run does not reproduce the older
  archived Adam result `(2.56e-6,5.83e-5)` despite matching its saved warm-start
  damage errors, so its pre-refinement comparison is recorded separately rather
  than replacing the archive.
- In both new runs, global residual MSE decreases but the strict MAE and
  pointwise gates fail; the largest raw residual remains at `x=0.499375`, one
  grid point beside the localization center. Last-layer optimization is
  therefore not the missing solver. The trunk must change, and any next
  deterministic refinement should be full-network or explicitly local.

## Verification State

- Python compilation succeeds for the fracture Python modules.
- After the defect, LM, and configuration regression changes, the complete
  unittest suite passes 60 tests.
- A 20-step `nx=40` smoke run and full `nx=100`/`nx=400` runs complete with
  fixed `du=1e-5`; the configured `nx=2000` run also completes with the same
  increment.
- The FEM reference loader validates material, loading, mesh, history metadata,
  and zero-load snapshot coverage.
- `configs/train_base.py` points to the selected `nx=2000` classical local FEM
  reference and the selected `Nx=1601` cell-history direct nonlocal baseline.
- `jax.devices()` reports `CudaDevice(0)`. The earlier CUDA OOM was caused by
  launching unittest and a JAX pilot concurrently; subsequent runs are serial.

## Next Work

1. Specify and test a default-off defect-aware local enrichment on the frozen
   `u_bar=0.01175` history. It must use trainable local basis coefficients,
   target-free defect diagnostics, and the same full-network acceptance gate;
   the existing hat pilot provides the representation baseline.
2. Keep local-equilibrium, hard-point weighting, and smooth weighting disabled,
   keep the Fourier embedding trainable, and use the width-`200` linear-output
   configuration for warning-mode curve diagnostics. The float32 `32/16` run is
   a viable acceleration candidate, while `64/4` is not promoted because it
   delayed localisation. Neither configuration is a strict fixed-point
   certification.
3. Add full-state checkpoint/resume so long continuation runs do not restart
   from `u_bar=0` after an interruption.
4. Repeat strict near-peak and post-peak fixed-history pilots, then run the full
   continuation with the linear output and scaled residual acceptance.
5. Report local FEM versus direct baseline separately from baseline versus PINN;
   keep additional mesh/profile refinement as a separate diagnostic study.
6. Do not add a second damage irreversibility or box constraint to the classical
   history-field solve.

## Legacy 2D Warning

The legacy 2D fracture script is not a convergence reference for this solver.
Its apparent convergence uses a pre-existing Dirichlet crack, a loose global
L2 tolerance, and non-strict inner history updates.
