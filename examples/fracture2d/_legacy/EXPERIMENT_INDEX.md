# Historical fracture2d experiment index

This catalogue was moved from the repository README during sampling cleanup.
Filenames and configurations below describe historical experiments.
Use the current example README for supported entry points and saved run source
snapshots for exact past experiments.


The historical filenames below now live under `examples/fracture2d/_legacy/`;
their configurations are in its `configs/` directory. Use the two consolidated
entries above for current training and direct solves.

`examples/fracture2d/local_damage_trial.py` is an independent isolation trial:
standard local P1 plane-strain mechanics on the same material/volume-grip mesh,
the existing nonlocal AT2 damage operator and maximum committed history.
It compares direct and neural damage without changing the old A2 paths.
Its local strain/driver quadrature is explicit and independently checked against
FEniCS. Convergence comparisons do not establish continuum accuracy or PINN
speed benefits. The example README documents selected-load and zero-start
direct checks and the separate neural diagnostic.
`examples/fracture2d/damage_optimization.py` and `damage_optimizer_trial.py`
provide separately identified fixed-history optimizer comparisons. The quadratic
damage objective is checked against the same AT2 residual; it is not a joint
mechanical/fracture energy or an imported direct training target.
`local_damage_corrected_trial.py` separately tests a neural predictor followed
by checked nonlocal nodal Jacobi corrections. Its fields are hybrid numerical
solutions; raw neural and corrected residuals are reported separately.
`local_damage_comparison.py` provides a separately identified local P1-gradient
damage control with the same nodal-lumped history/reaction, volume grips and
local mechanics. `local_damage_fem_audit.py` independently rebuilds UFL weak
forms to verify saved accepted fields; this does not establish mesh convergence.
`conditioned_damage.py` and `conditioned_damage_trial.py` test a neural damage
field conditioned on geometry and its own known history/neighborhood data.
These features change the ansatz, not the nonlocal AT2 equation; direct damage
is evaluated only after training and no nodal damage correction is applied.
`compact_damage_features.py` and `compact_damage_trial.py` enrich that frozen
neural trunk with continuous compact radial hidden units and trainable readouts.
Centers are selected from geometry and the neural residual, without direct
labels; coefficient normalization changes parameters rather than the equation.
`compact_damage_loading.py` tests these learned features/readouts from zero
physical damage/history with direct local FEM mechanics, explicit inner/outer
gates, and an additional raw-network residual check at the updated own H.
Its explicit `loading.train_trunk` option compares frozen-trunk readouts with
jointly trained conditioned MLP and compact readouts in separate run identities.
`compact_damage_fork.py` copies only verified accepted own-neural prefixes into
an explicit optimizer-policy fork, preserving field/network hashes and history.
`bounded_damage_output.py` and `output_damage_trial.py` compare nonnegative,
sigmoid and exponential-square outputs, including exact operator-diagonal/H
reconstruction of a bounded neural neighbor field. The controls share an own-
neural warm baseline, matching physical initialization and explicit tangent
normalization. They preserve AT2/history and do not commit diagnostic fields.
`output_damage_loading.py` separately compares certified bounded heads with
direct FEM mechanics from a verified accepted own-neural physical parent.
`output_damage_continue.py` preserves the complete certified sigmoid decoder
and accepted ancestry for later own-history loading, with restart checks and
fixed-layout compiled L-BFGS reuse in `cached_damage_optimization.py`.
`output_damage_safe_continue.py` declares an accepted-parent fork that caps
only the frozen initializer predictor before the finite-logit seed. The
unknown sigmoid damage, original AT2 equation/history and gates remain intact.
`plain_sigmoid_damage.py` removes that reference field entirely, retaining
the conditioned MLP and compact readouts as the direct sigmoid unknown.
`plain_sigmoid_trial.py` tests own-network parameter conversion and original
AT2 training at fixed own history before any declared physical continuation.
`plain_sigmoid_cold.py` tests fresh network weights and an own-zero physical
state with the first imposed displacement0.005, preserving the original
inner/coupled gates in a separate one-load experiment.
`plain_sigmoid_gradual.py` instead begins at displacement0.0001 with fresh
weights and exact zero physical fields. Its separate decoder uses unscaled
trainable logits, physical detached history features and fixed initial AT2
Jacobian parameter scales; only accepted states initialize later loads.
`plain_damage_head_trial.py` compares fresh sigmoid, exponential-square and
unconstrained identity outputs at exactly the same audited own history,
matching initial field and parameter sensitivities. Original range/residual
gates still apply; these diagnostic fields do not advance physical loading.
`range_damage_trial.py` continues the same linear-output checkpoint at fixed
own history, comparing extra residual training with inequality augmented
Lagrangian optimization of the damage range. It reports absolute/forcing-
normalized residuals and damage activity separately; outputs are not projected.
`linear_damage_loading.py` tests that certified linear head in a separate
own-zero coupled path with direct local FEM mechanics, fixed-H range
optimization and original accepted-parent/history/outer gates. The current-H
raw neural range is explicitly checked as well as its residual.
`linear_damage_continue.py` recovers that same critical load in a separate
branch with explicit user-accepted damage-range tolerance `2e-6`. It retains
the last accepted loading parent, resumes completed outer work with the saved
neural parameters, and preserves the original equation and coupled gates.
`linear_damage_budget_continue.py` continues a certified same-load accuracy
stop with a larger explicit inner training budget and unchanged final gates.
`linear_damage_relaxed_continue.py` attempts the full remaining path with
explicit negative tolerance `1e-3` and unchanged upper tolerance `2e-6`;
equation and coupled-equilibrium gates still apply.
`linear_damage_pending_continue.py` restores a certified interrupted physical
iterate in a fresh root without promoting it to an accepted loading parent.

`damage_residual_weighting.py` and `weighted_damage_trial.py` provide a separate
fixed-history comparison of detached static/damage-dependent residual weights,
plus exact core+halo row extraction for sampling diagnostics. They retain
full-domain physical acceptance and do not modify the running loading solver.
`provisional_damage_loading.py` adds explicitly user-authorized advancement
of budget-limited load states while retaining original convergence diagnostics.
`core_halo_adam.py` and `core_halo_adam_trial.py` test Adam with persistent
optimizer state across fresh random cores, with full-domain accuracy controls.
The `configs/core_halo_damage_resample.py` trial uses fresh random crack-biased
cores every8/32 optimizer updates, following elasticity distance refinement.
`core_halo_damage.py` and `core_halo_damage_trial.py` evaluate the network only
on fixed-shape halo batches, with frozen sampling probabilities, explicit
importance correction and full-domain validation. A cached full-graph control
separates feature-cache savings from sampling savings at the same checkpoint.
`examples/fracture2d/nonlocal_baseline.py` is the frozen GPU direct baseline.
The current user-accepted engineering comparison model uses the fine material
cloud with mechanics/damage radii both `0.008` and AT2 length `0.024`:
`configs/scale_separation_short_damage.py`. Its direct result retains a strict
52-load prefix and a separately marked warning tail through `0.0075`.
The PINN stage starts fresh on this model; direct arrays are comparisons only.
`configs/split_pinn_short_damage.py` explicitly binds the user-accepted warning
reference policy and starts with two loads. Warning reference provenance is
included in neural reports/checkpoint identities and prevents strict labeling.
Experimental acceleration controls use explicit ordered direct-reference indices
for nonuniform loading and separately configured inner-block stopping. Neural
checkpoint ordinals remain distinct from reference indices; each actual load
updates learned history. Fixed and accelerated paths retain separate identities.
`configs/split_pinn_full_warning.py` explicitly treats reference reaction/damage
differences as diagnostics and continues a separate complete warning path;
original comparison thresholds and rejected runs remain recorded.
The direct–PINN stagger uses `hybrid_pinn.py` and
`configs/hybrid_pinn_short_damage.py`: direct nonlocal mechanics, one nonlocal
AT2 damage network, and cumulative history from its own staggered fields. Each
subproblem freezes the other field; the networks/fields are not jointly optimized. It has
a separate checkpoint schema and starts fresh on the same fine-cloud model.
`configs/hybrid_pinn_swish_fourier.py` is a separate larger-network candidate:
256x6 Swish with64 trainable Fourier frequencies at scale16, inspired by
fracture1d's200x4 Swish/Fourier method. It requires fresh neural checkpoints.
`configs/hybrid_pinn_swish_fourier_128.py` is the user-selected narrower128x6
Swish/Fourier trial, starting fresh at0.0048 with its own loading history;
`--last-index 28` selects its full path through0.0075.
`configs/hybrid_pinn_stagger_gated.py` adds convergence-controlled outer stagger:
fixed-H/current damage residuals, adjacent-field stability and physical checks
must pass consecutive confirmations before loading advances. Exhausted budgets
stop the trial. A separate gated checkpoint schema preserves restart state;
explicit matching-prefix forks retain warning ancestry.
The original common-horizon PINN branch remains available through
`legacy_coupling_refine.py`.
`legacy_coupled_loading.py` advances one load from a SHA-verified diagnostic
predecessor only after rechecking three coupled-readiness confirmations.
An explicit `--warning-continue` policy supports diagnostic trend studies with
uncoupled ancestry disclosed and the original physical/mechanics hard gates.
`subproblem_matrix.py` compares direct/neural mechanics and damage at identical
stored history floors to diagnose inner accuracy versus outer coupling error.
`direct_path_probe.py` tightens the common-floor direct benchmark and compares
convergence-gated sequential loading step sizes without changing the kernel.
`objective_pair_probe.py` audits subproblem energy gradients and coupled mixed
derivatives, then compares fixed-d neural force-residual and original A2 energy
objectives from identical networks with fresh Adam; direct fields are comparisons.
The independent-horizon research branch uses `split_reference.py` for strict
comparison states and `split_pinn.py` for FP64 neural loading. Configuration B
(`configs/scale_separation_b.py`) retains the fine mechanics graph and halves
the damage radius; its direct run explicitly selects `--damage-backend gpu`.
Fresh direct roots use sparse trial retention (first, latest four and every
50th generation), while keeping all committed states and scalar reports.
`warning_reference.py` separately explores a diagnostic tail from a strict
prefix, retaining original gate failures and excluding it from strict PINN references.
The PINN starts
from zero and uses its own learned predecessor history; reference fields enter
comparisons only. Damage-field RMSE and reaction error lead the research report;
other unmet checks retain their values, limits and warning provenance. Warning
advancement is not strict field-equation validation. See the
[example README](../README.md) for commands and the
[active task](../../../docs/tasks/fracture2d-nonlocal-pinn.md) for current results.
The [2026-10-12 provisional-loading stage report](../../../docs/reports/fracture2d-provisional-loading-20261012.md)
preserves the higher-load experiment, matched-budget direct control, audited
results and figures. It is a dated research snapshot; the active task remains
the source for current status and next actions.
`bounded_damage_morphology_trial.py` with `configs/bounded_damage_loading.py`
loads directly with fresh no-reference sigmoid/exp-square networks and the
certified52 physical parent. Extra same-head adaptation precedes loading;
its residual is not a screening gate. Optional history-comparison mode keeps
direct diagnostic values out of training. Background damage, crack geometry,
reaction and AT2 residual are reported together.
`sigmoid_damage_accuracy_trial.py` isolates fixed-history budget/sampling
controls at the saved critical load, then extends the same-load staggered
iterations while retaining the committed previous-load history and gates.
`low_order_damage.py` supplies MAE/MSE residual objectives for exact random
core+halo controls with persistent Adam or pinned upstream SOAP_JAX 0.2.2.
`precomputed_damage_sampling.py` caches fixed operator support rows for fresh
random batches without rebuilding or validating the entire CSR each time.
`prebuilt_damage_pool.py` prepares blocks of exact sampled supports for
GPU batch selection; new blocks retain fresh random draws.
It retains the original nonlocal AT2 equation and separates mean losses from
whole-field acceptance; run-specific controls and outcomes live in the active task.
`l1_damage_loading.py` with `configs/l1_damage_loading_from_zero.py` starts
fresh sigmoid weights and exact zero physical fields, then uses L1 random
Adam with fresh GPU core+halo blocks on an explicit two-stage loading grid.
Budget-limited states and their descendants are labelled provisional; physical
readiness and optimizer state are recorded separately.
`sigmoid_damage_load_refined.py` separately restarts physical/sigmoid52 with
quarter-load increments through0.0053, keeping the model and readiness gates.

`restart_damage_loading.py` separately restarts certified load52 with its own
network/fields and128 alternating updates per load through75. Inner512-update
budgets and physical equations remain; immutable intermediate checkpoints
compare outer8/32/64/128 without changing the previous-load history.
The optional `configs/split_pinn_fast.py` continues a verified learned prefix
with measured operator optimizations and a reduced small-load update budget;
the example README documents its scope and the paired quality/runtime checks.
