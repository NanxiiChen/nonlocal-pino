"""Direct discrete nonlocal baseline for the 1D fracture PINN route.

The baseline uses the same condensed mechanics and classical history-field
staggering as the PINN, but solves the fixed-history SparseD damage equation
directly as a sparse linear system. It is a model/discretisation reference,
not a replacement for the neural solver.
"""

import os

import jax
import numpy as np
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve

from pinn.operators import SparseD


def _grid_data(n_points):
    if int(n_points) != n_points or n_points < 2:
        raise ValueError(f"n_points must be an integer >= 2, got {n_points}")
    x = np.linspace(0.0, 1.0, int(n_points), dtype=np.float64)
    spacing = 1.0 / (int(n_points) - 1)
    weights = np.full(int(n_points), spacing, dtype=np.float64)
    weights[[0, -1]] *= 0.5
    return x, weights


def _modulus_field(config, x):
    values = 1.0 - config.imperfection.amplitude * np.exp(
        -(x - config.imperfection.center) ** 2
        / (2.0 * config.imperfection.width**2)
    )
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("The configured modulus imperfection must give e(x) > 0")
    return values


def _cell_history_from_displacement(displacement, x, e_cell, H_previous):
    """Update DG0 tensile history from element strain and cell modulus."""
    displacement = np.asarray(displacement, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    modulus = np.asarray(e_cell, dtype=np.float64)
    previous = np.asarray(H_previous, dtype=np.float64)
    if displacement.shape != x.shape or len(displacement) < 2:
        raise ValueError("displacement and x must have the same length >= 2")
    if modulus.shape != (len(x) - 1,):
        raise ValueError("e_cell must have one value per grid cell")
    if previous.shape != modulus.shape:
        raise ValueError("H_previous must have one value per grid cell")
    if not all(np.all(np.isfinite(values)) for values in
               (displacement, x, modulus, previous)):
        raise ValueError("cell history inputs must be finite")
    cell_strain = np.diff(displacement) / np.diff(x)
    psi_plus = 0.5 * modulus * np.maximum(cell_strain, 0.0) ** 2
    return np.maximum(previous, psi_plus)


def _mass_lumped_cell_history_to_nodes(H_cell):
    """Project DG0 cell history to nodal values by the CG1 lumped mass rule."""
    history = np.asarray(H_cell, dtype=np.float64)
    if history.ndim != 1 or len(history) < 1 or not np.all(np.isfinite(history)):
        raise ValueError("H_cell must be a finite one-dimensional array")
    nodal = np.empty(len(history) + 1, dtype=np.float64)
    nodal[0] = history[0]
    nodal[-1] = history[-1]
    if len(history) > 1:
        nodal[1:-1] = 0.5 * (history[:-1] + history[1:])
    return nodal


def condensed_mechanics(d, x, e_field, u_bar, k_res, quadrature_weights):
    """Return condensed mechanics using nodal trapezoidal quadrature."""
    d = np.asarray(d, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    e_field = np.asarray(e_field, dtype=np.float64)
    quadrature_weights = np.asarray(quadrature_weights, dtype=np.float64)
    if d.shape != x.shape or e_field.shape != x.shape:
        raise ValueError("d, x, and e_field must have identical one-dimensional shapes")
    if quadrature_weights.shape != x.shape:
        raise ValueError("quadrature_weights must match x shape")
    degradation = (1.0 - d) ** 2 + k_res
    compliance_density = 1.0 / (e_field * degradation)
    compliance = float(np.dot(quadrature_weights, compliance_density))
    if not np.isfinite(compliance) or compliance <= 0.0:
        raise FloatingPointError(f"Invalid compliance {compliance}")
    sigma = float(u_bar) / compliance
    strain = sigma * compliance_density
    increments = 0.5 * (strain[1:] + strain[:-1]) * np.diff(x)
    displacement = np.concatenate(([0.0], np.cumsum(increments)))
    displacement[-1] = float(u_bar)
    psi_plus = 0.5 * e_field * np.maximum(strain, 0.0) ** 2
    stress = e_field * degradation * strain
    if not all(np.all(np.isfinite(values)) for values in
               (strain, displacement, psi_plus, stress)):
        raise FloatingPointError("Condensed baseline mechanics is non-finite")
    return sigma, strain, displacement, psi_plus, compliance


def build_fixed_history_system(operator, N_l, N_G, H_points):
    """Build the linear fixed-history system and its Jacobi diagonal."""
    history = np.asarray(H_points, dtype=np.float64)
    if history.ndim != 1 or not np.all(np.isfinite(history)):
        raise ValueError("H_points must be a finite one-dimensional array")
    if np.any(history < 0.0):
        raise ValueError("H_points must be non-negative")
    if not np.isfinite(N_l) or N_l <= 0.0:
        raise ValueError(f"N_l must be positive and finite, got {N_l}")
    if not np.isfinite(N_G) or N_G <= 0.0:
        raise ValueError(f"N_G must be positive and finite, got {N_G}")
    if not hasattr(operator, "shape") or not hasattr(operator, "dot"):
        raise TypeError("operator must be a SciPy sparse matrix")
    if operator.shape != (len(history), len(history)):
        raise ValueError(
            f"D shape {operator.shape} does not match history shape {history.shape}"
        )
    history_factor = 2.0 * N_l / N_G
    system = (
        eye(len(history), format="csr")
        + N_l**2 * operator
        + diags(history_factor * history, format="csr")
    )
    rhs = history_factor * history
    diagonal = np.asarray(system.diagonal(), dtype=np.float64)
    if not np.all(np.isfinite(diagonal)) or np.any(diagonal <= 0.0):
        raise FloatingPointError("Fixed-history system diagonal is invalid")
    return system, rhs, diagonal


def solve_fixed_history_damage(D, N_l, N_G, H_points):
    """Solve the unconstrained linear SparseD damage equation for fixed history."""
    if not hasattr(D, "to_scipy"):
        raise TypeError("D must provide the SparseD.to_scipy() baseline interface")
    history = np.asarray(H_points, dtype=np.float64)
    system, rhs, _ = build_fixed_history_system(
        D.to_scipy(), N_l, N_G, H_points
    )
    damage = np.asarray(spsolve(system, rhs), dtype=np.float64)
    if damage.shape != history.shape or not np.all(np.isfinite(damage)):
        raise FloatingPointError("Direct SparseD damage solve returned non-finite values")
    residual = np.asarray(system @ damage - rhs, dtype=np.float64)
    residual_mae = float(np.mean(np.abs(residual)))
    residual_inf = float(np.max(np.abs(residual)))
    return damage, residual_mae, residual_inf


def solve_fixed_history_defect(operator, N_l, N_G, damage, H_points):
    """Solve ``A(H) defect = A(H) damage - b(H)`` without target supervision."""
    values = np.asarray(damage, dtype=np.float64)
    history = np.asarray(H_points, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("damage must be a finite one-dimensional array")
    if values.shape != history.shape:
        raise ValueError("damage and H_points must have identical shapes")
    system, rhs, _ = build_fixed_history_system(operator, N_l, N_G, history)
    residual = np.asarray(system @ values - rhs, dtype=np.float64)
    defect = np.asarray(spsolve(system, residual), dtype=np.float64)
    solve_residual = np.asarray(system @ defect - residual, dtype=np.float64)
    if not np.all(np.isfinite(defect)) or not np.all(np.isfinite(solve_residual)):
        raise FloatingPointError("Fixed-history defect solve returned non-finite values")
    return defect, residual, solve_residual


def solve_pseudo_time_damage(D, N_l, N_G, H_points, d_previous, tau, operator):
    """Take one implicit pseudo-time step for a fixed history field."""
    history = np.asarray(H_points, dtype=np.float64)
    previous = np.asarray(d_previous, dtype=np.float64)
    if history.ndim != 1 or not np.all(np.isfinite(history)):
        raise ValueError("H_points must be a finite one-dimensional array")
    if np.any(history < 0.0):
        raise ValueError("H_points must be non-negative")
    if previous.shape != history.shape or not np.all(np.isfinite(previous)):
        raise ValueError("d_previous must be finite and match H_points shape")
    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError(f"tau must be positive and finite, got {tau}")
    if not np.isfinite(N_l) or N_l <= 0.0:
        raise ValueError(f"N_l must be positive and finite, got {N_l}")
    if not np.isfinite(N_G) or N_G <= 0.0:
        raise ValueError(f"N_G must be positive and finite, got {N_G}")
    if not hasattr(D, "to_scipy"):
        raise TypeError("D must provide the SparseD.to_scipy() baseline interface")
    if operator is None:
        raise ValueError("operator must be the precomputed SparseD SciPy matrix")
    if operator.shape != (len(history), len(history)):
        raise ValueError(
            f"D shape {operator.shape} does not match history shape {history.shape}"
        )
    history_factor = 2.0 * N_l / N_G
    fixed_system = (
        eye(len(history), format="csr")
        + N_l**2 * operator
        + diags(history_factor * history, format="csr")
    )
    system = eye(len(history), format="csr") + tau * fixed_system
    rhs = previous + tau * history_factor * history
    damage = np.asarray(spsolve(system, rhs), dtype=np.float64)
    if damage.shape != history.shape or not np.all(np.isfinite(damage)):
        raise FloatingPointError(
            "Direct pseudo-time damage solve returned non-finite values"
        )
    residual = fixed_system @ damage - history_factor * history
    residual_mae = float(np.mean(np.abs(residual)))
    residual_inf = float(np.max(np.abs(residual)))
    return damage, residual_mae, residual_inf


def run_incremental(config, output_path, log_every):
    """Run the historical nodal-history nonlocal baseline."""
    return _run_incremental(config, output_path, log_every, "nodal")


def run_incremental_with_history(config, output_path, log_every, history_mode):
    """Run the direct baseline with an explicit history discretisation mode."""
    return _run_incremental(config, output_path, log_every, history_mode)


def _run_incremental(config, output_path, log_every, history_mode):
    """Run the direct nonlocal baseline along the configured load path."""
    jax.config.update("jax_enable_x64", config.train.dtype == "float64")
    if int(log_every) != log_every or log_every < 1:
        raise ValueError(f"log_every must be a positive integer, got {log_every}")
    if history_mode not in ("nodal", "cell_mass_lumped"):
        raise ValueError(
            "history_mode must be 'nodal' or 'cell_mass_lumped'"
        )
    if int(config.baseline.max_stagger_iters) != config.baseline.max_stagger_iters:
        raise ValueError("baseline.max_stagger_iters must be an integer")
    if config.baseline.max_stagger_iters < 1:
        raise ValueError("baseline.max_stagger_iters must be positive")
    x, quadrature_weights = _grid_data(config.sample.Nx)
    e_field = _modulus_field(config, x)
    x_cell = 0.5 * (x[:-1] + x[1:])
    e_cell = _modulus_field(config, x_cell)
    N_l = float(config.physical.l0 / config.geometry.L0)
    N_G = float(config.physical.Gc / (config.physical.E * config.geometry.L0))
    x_operator = x[:, None].astype(np.dtype(config.train.dtype))
    D = SparseD(x_operator, config.nl.delta, kernel=config.nl.kernel)
    scipy_operator = D.to_scipy()

    d_previous = np.zeros_like(x)
    if history_mode == "cell_mass_lumped":
        H_previous = np.zeros(len(x) - 1, dtype=np.float64)
    else:
        H_previous = np.zeros_like(x)
    u_bar = 0.0
    du_current = float(config.train.du)
    cutbacks_for_step = 0
    softening_started = False
    previous_reaction = None
    step = 0

    u_bar_history = []
    du_history = []
    reaction_history = []
    stagger_iterations_history = []
    cutback_history = []
    softening_history = []
    damage_residual_mae_history = []
    damage_residual_inf_history = []
    solve_history_residual_mae_history = []
    solve_history_residual_inf_history = []
    damage_increment_min_history = []
    snap_u_bar = [0.0]
    d_snapshots = [d_previous.copy()]
    u_snapshots = [np.zeros_like(x)]
    H_snapshots = [
        _mass_lumped_cell_history_to_nodes(H_previous)
        if history_mode == "cell_mass_lumped"
        else H_previous.copy()
    ]
    H_cell_snapshots = (
        [H_previous.copy()] if history_mode == "cell_mass_lumped" else []
    )

    while u_bar < config.train.u_final:
        if np.isclose(u_bar, config.train.u_final, rtol=0.0, atol=1.0e-14):
            u_bar = float(config.train.u_final)
            break
        trial_du = min(du_current, config.train.u_final - u_bar)
        trial_u_bar = u_bar + trial_du
        d_iter = d_previous.copy()
        H_next = H_previous.copy()
        converged = False
        err_damage = np.inf
        err_sigma = np.inf
        err_history = np.inf
        err_mechanics = np.inf
        residual_mae = np.inf
        residual_inf = np.inf
        solve_residual_mae = np.inf
        solve_residual_inf = np.inf
        H_next_damage = np.zeros_like(x)
        sigma = np.nan
        stagger_index = -1

        for stagger_index in range(config.baseline.max_stagger_iters):
            mechanics_before = condensed_mechanics(
                d_iter, x, e_field, trial_u_bar, config.physical.k_res,
                quadrature_weights,
            )
            sigma_before, _, u_before, psi_before, _ = mechanics_before
            if history_mode == "cell_mass_lumped":
                H_used_reference = _cell_history_from_displacement(
                    u_before, x, e_cell, H_previous
                )
                H_used = _mass_lumped_cell_history_to_nodes(H_used_reference)
            else:
                H_used_reference = np.maximum(H_previous, psi_before)
                H_used = H_used_reference
            d_next, residual_mae, residual_inf = solve_fixed_history_damage(
                D, N_l, N_G, H_used
            )
            solve_residual_mae = residual_mae
            solve_residual_inf = residual_inf
            mechanics_next = condensed_mechanics(
                d_next, x, e_field, trial_u_bar, config.physical.k_res,
                quadrature_weights,
            )
            sigma, _, u_next, psi_next, _ = mechanics_next
            if history_mode == "cell_mass_lumped":
                H_next = _cell_history_from_displacement(
                    u_next, x, e_cell, H_previous
                )
                err_history = float(
                    np.max(np.abs(H_next - H_used_reference))
                )
            else:
                H_next = np.maximum(H_previous, psi_next)
                err_history = float(np.max(np.abs(H_next - H_used)))
                H_next_damage = H_next
            if history_mode == "cell_mass_lumped":
                H_next_damage = _mass_lumped_cell_history_to_nodes(H_next)
            committed_residual = d_next + N_l**2 * (scipy_operator @ d_next)
            committed_residual -= (
                (2.0 * N_l / N_G) * (1.0 - d_next) * H_next_damage
            )
            residual_mae = float(np.mean(np.abs(committed_residual)))
            residual_inf = float(np.max(np.abs(committed_residual)))
            err_damage = float(np.max(np.abs(d_next - d_iter)))
            err_sigma = abs(sigma - sigma_before)
            err_mechanics = float(np.max(np.abs(u_next - u_before)))
            d_iter = d_next
            if (
                err_damage <= config.baseline.atol_damage
                and residual_mae <= config.train.atol_damage_residual
                and residual_inf <= config.train.atol_damage_residual_inf
                and err_sigma <= config.train.atol_sigma
                and err_history <= config.train.atol_history
                and err_mechanics <= config.train.atol_mechanics
            ):
                converged = True
                break

        if not converged:
            at_minimum = trial_du <= config.train.du_min * (1.0 + 1e-12)
            if at_minimum or cutbacks_for_step >= config.train.max_cutbacks:
                raise RuntimeError(
                    f"Direct nonlocal load step at u_bar={trial_u_bar:.8e} did not "
                    f"converge: damage={err_damage:.3e}, residual={residual_mae:.3e}, "
                    f"residual_inf={residual_inf:.3e}, sigma={err_sigma:.3e}, "
                    f"history={err_history:.3e}, mechanics={err_mechanics:.3e}"
                )
            cutbacks_for_step += 1
            du_current = max(config.train.du_min, trial_du * config.train.cutback_factor)
            continue

        step += 1
        u_bar = trial_u_bar
        minimum_change = float(np.min(d_iter - d_previous))
        d_previous = d_iter
        H_previous = H_next
        if (
            previous_reaction is not None
            and sigma < previous_reaction - config.train.atol_sigma
        ):
            softening_started = True
        reaction_history.append(sigma)
        u_bar_history.append(u_bar)
        du_history.append(trial_du)
        stagger_iterations_history.append(stagger_index + 1)
        cutback_history.append(cutbacks_for_step)
        softening_history.append(softening_started)
        damage_residual_mae_history.append(residual_mae)
        damage_residual_inf_history.append(residual_inf)
        solve_history_residual_mae_history.append(solve_residual_mae)
        solve_history_residual_inf_history.append(solve_residual_inf)
        damage_increment_min_history.append(minimum_change)
        snap_u_bar.append(u_bar)
        d_snapshots.append(d_previous.copy())
        u_snapshots.append(condensed_mechanics(
            d_previous, x, e_field, u_bar, config.physical.k_res,
            quadrature_weights,
        )[2])
        if history_mode == "cell_mass_lumped":
            H_snapshots.append(
                _mass_lumped_cell_history_to_nodes(H_previous)
            )
            H_cell_snapshots.append(H_previous.copy())
        else:
            H_snapshots.append(H_previous.copy())
        if step % log_every == 0 or np.isclose(u_bar, config.train.u_final):
            print(
                f"Step {step:5d} u_bar={u_bar:.8e} reaction={sigma:.8e} "
                f"max_d={np.max(d_previous):.6e} stagger={stagger_index + 1}"
            )
        cutbacks_for_step = 0
        previous_reaction = sigma
        if softening_started:
            du_current = min(config.train.du_softening,
                             trial_du * config.train.growth_factor)
        else:
            du_current = min(config.train.du_max,
                             trial_du * config.train.growth_factor)

    output_directory = os.path.dirname(output_path)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
    np.savez_compressed(
        output_path,
        x=x,
        x_H=x,
        u_bar=np.asarray(u_bar_history),
        du_history=np.asarray(du_history),
        stagger_iterations=np.asarray(stagger_iterations_history),
        cutbacks=np.asarray(cutback_history),
        softening=np.asarray(softening_history),
        snap_u_bar=np.asarray(snap_u_bar),
        u=np.asarray(u_snapshots),
        d=np.asarray(d_snapshots),
        reaction_star=np.asarray(reaction_history),
        force=np.asarray(reaction_history),
        H=np.asarray(H_snapshots),
        x_H_cell=x_cell if history_mode == "cell_mass_lumped" else np.empty(0),
        H_cell=(
            np.asarray(H_cell_snapshots)
            if history_mode == "cell_mass_lumped"
            else np.empty((0, 0))
        ),
        damage_residual_mae=np.asarray(damage_residual_mae_history),
        damage_residual_inf=np.asarray(damage_residual_inf_history),
        damage_residual_solve_history_mae=np.asarray(
            solve_history_residual_mae_history
        ),
        damage_residual_solve_history_inf=np.asarray(
            solve_history_residual_inf_history
        ),
        damage_increment_min=np.asarray(damage_increment_min_history),
        E0=float(config.physical.E),
        Gc=float(config.physical.Gc),
        l0=float(config.physical.l0),
        L0=float(config.geometry.L0),
        k_res=float(config.physical.k_res),
        N_l=N_l,
        N_G=N_G,
        imperfection_amplitude=float(config.imperfection.amplitude),
        imperfection_width=float(config.imperfection.width),
        imperfection_center=float(config.imperfection.center),
        delta=float(config.nl.delta),
        particle_volume=float(1.0 / (config.sample.Nx - 1)),
        mesh_nx=int(config.sample.Nx),
        loading_du_initial=float(config.train.du),
        loading_du_min=float(config.train.du_min),
        loading_du_max=float(config.train.du_max),
        loading_du_softening=float(config.train.du_softening),
        loading_cutback_factor=float(config.train.cutback_factor),
        loading_growth_factor=float(config.train.growth_factor),
        output_save_every=float(config.train.du),
        baseline_max_stagger_iters=int(config.baseline.max_stagger_iters),
        history_space=(
            "nodal"
            if history_mode == "nodal"
            else "nodal_mass_lumped_from_DG0"
        ),
        history_method=(
            "Miehe_max_history"
            if history_mode == "nodal"
            else "Miehe_max_history_DG0_cell_strain"
        ),
        history_mode=history_mode,
        baseline_method="direct_sparse_linear",
    )
    return output_path
