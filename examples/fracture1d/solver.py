"""Independent local FEniCS reference for the 1D AT2 fracture benchmark.

The computational domain is the non-dimensional interval ``[0, 1]``. The
dimensional length ``geometry.L0`` is retained only when constructing
``N_l = l0 / L0`` and ``N_G = Gc / (E0 * L0)``. Mechanics and damage are solved
with a staggered history-field scheme at each displacement increment.

Run in the FEniCS environment, for example::

    python examples/fracture1d/solver.py
    mpirun --allow-run-as-root -np 4 python examples/fracture1d/solver.py
"""

import gc
import os
import sys

# Ensure DOLFIN pkg-config is discoverable before importing FEniCS.
_pkgconfig_dir = os.path.join(sys.prefix, "lib", "pkgconfig")
if os.path.isdir(_pkgconfig_dir):
    os.environ.setdefault("PKG_CONFIG_PATH", "")
    if _pkgconfig_dir not in os.environ["PKG_CONFIG_PATH"]:
        os.environ["PKG_CONFIG_PATH"] = (
            _pkgconfig_dir + ":" + os.environ["PKG_CONFIG_PATH"]
        )

# DOLFIN finalizes MPI during interpreter shutdown; prevent mpi4py from
# issuing a second post-finalize communicator query in the same process.
from mpi4py import rc as _mpi4py_rc
_mpi4py_rc.finalize = False
from mpi4py import MPI as _MPI
import numpy as np
from absl import app as _absl_app, flags as _absl_flags
from ml_collections import config_flags as _config_flags

from fenics import (
    Constant,
    DirichletBC,
    Function,
    FunctionSpace,
    IntervalMesh,
    TestFunction,
    TrialFunction,
    assemble,
    dx,
    project,
    set_log_active,
    solve,
)


def analytical_homogeneous_damage(eps, N_l, N_G):
    """Zero-residual-stiffness homogeneous AT2 damage relation."""
    alpha = N_G / N_l
    return eps**2 / (eps**2 + alpha)


def analytical_homogeneous_stress(eps, N_l, N_G):
    """Zero-residual-stiffness homogeneous stress relation."""
    d = analytical_homogeneous_damage(eps, N_l, N_G)
    return (1.0 - d) ** 2 * eps


def analytical_peak_strain(N_l, N_G):
    """Zero-residual-stiffness homogeneous peak strain."""
    return np.sqrt(N_G / (3.0 * N_l))


def analytical_peak_stress(N_l, N_G):
    """Zero-residual-stiffness homogeneous peak stress."""
    return (3.0 * np.sqrt(3.0) / 16.0) * np.sqrt(N_G / N_l)


def analytical_crack_profile(x, xc, N_l):
    """Infinite-domain stationary AT2 profile, for a separate diagnostic."""
    return np.exp(-np.abs(x - xc) / N_l)


def analytical_stationary_profile(x, xc, N_l):
    """Finite-domain stationary profile with natural endpoint conditions."""
    left = xc - x[0]
    right = x[-1] - xc
    if not np.isclose(left, right):
        raise ValueError("The finite stationary profile requires a symmetric domain")
    radius = left
    distance = np.abs(x - xc)
    return np.cosh((radius - distance) / N_l) / np.cosh(radius / N_l)


def _validate_config(config):
    """Validate every solver assumption before constructing FEniCS objects."""
    phys = config.physical
    geom = config.geometry
    imp = config.imperfection
    loading = config.loading
    solver_cfg = config.solver
    output = config.output

    positive_values = {
        "physical.E": phys.E,
        "physical.Gc": phys.Gc,
        "physical.l0": phys.l0,
        "geometry.L0": geom.L0,
        "imperfection.width": imp.width,
        "loading.du": loading.du,
        "loading.du_min": loading.du_min,
        "loading.du_max": loading.du_max,
        "loading.du_softening": loading.du_softening,
        "loading.u_final": loading.u_final,
        "solver.atol_u": solver_cfg.atol_u,
        "solver.atol_d": solver_cfg.atol_d,
        "solver.atol_H": solver_cfg.atol_H,
        "solver.atol_sigma": solver_cfg.atol_sigma,
        "solver.atol_stress_constancy": solver_cfg.atol_stress_constancy,
        "output.save_every": output.save_every,
    }
    for name, value in positive_values.items():
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite, got {value}")

    if not np.isfinite(phys.k_res) or phys.k_res < 0.0:
        raise ValueError(f"physical.k_res must be non-negative and finite, got {phys.k_res}")
    if not 0.0 <= imp.amplitude < 1.0:
        raise ValueError(
            "imperfection.amplitude must satisfy 0 <= amplitude < 1, "
            f"got {imp.amplitude}"
        )
    if not 0.0 <= imp.center <= 1.0:
        raise ValueError(
            "imperfection.center must lie in the non-dimensional interval [0, 1], "
            f"got {imp.center}"
        )
    if int(config.mesh.nx) != config.mesh.nx or config.mesh.nx < 2:
        raise ValueError(f"mesh.nx must be an integer >= 2, got {config.mesh.nx}")
    if not (
        loading.du_min <= loading.du_softening <= loading.du_max
    ):
        raise ValueError(
            "loading.du_softening must lie between loading.du_min and "
            "loading.du_max"
        )
    if loading.du_min > loading.du or loading.du > loading.du_max:
        raise ValueError(
            "loading.du must lie between loading.du_min and loading.du_max"
        )
    if not 0.0 < loading.cutback_factor < 1.0:
        raise ValueError("loading.cutback_factor must lie strictly between zero and one")
    if loading.growth_factor < 1.0:
        raise ValueError("loading.growth_factor must be at least one")
    if int(loading.max_cutbacks) != loading.max_cutbacks:
        raise ValueError("loading.max_cutbacks must be an integer")
    if loading.max_cutbacks < 1:
        raise ValueError("loading.max_cutbacks must be positive")
    if int(solver_cfg.max_stagger_iters) != solver_cfg.max_stagger_iters:
        raise ValueError("solver.max_stagger_iters must be an integer")
    if solver_cfg.max_stagger_iters < 1:
        raise ValueError("solver.max_stagger_iters must be positive")
    if solver_cfg.linear_solver not in ("default", "lu"):
        raise ValueError("solver.linear_solver must be 'default' or 'lu'")
    if int(output.log_every) != output.log_every or output.log_every < 1:
        raise ValueError("output.log_every must be a positive integer")


def _gather_1d(field, space, comm):
    """Gather a scalar FEniCS field to rank zero, sorted by coordinate."""
    local_coords = space.tabulate_dof_coordinates()[:, 0]
    local_values = field.vector().get_local()
    coords_list = comm.gather(local_coords, root=0)
    values_list = comm.gather(local_values, root=0)

    if comm.rank != 0:
        return None, None

    all_coords = np.concatenate(coords_list)
    all_values = np.concatenate(values_list)
    rounded = all_coords.round(decimals=10)
    _, unique_idx = np.unique(rounded, return_index=True)
    sort_idx = np.argsort(all_coords[unique_idx])
    return all_coords[unique_idx][sort_idx], all_values[unique_idx][sort_idx]


def _assign_local(field, values):
    """Assign an array with the local FEniCS vector shape."""
    values = np.asarray(values, dtype=np.float64)
    if values.shape != field.vector().get_local().shape:
        raise ValueError(
            f"Cannot assign shape {values.shape} to FEniCS vector shape "
            f"{field.vector().get_local().shape}"
        )
    field.vector().set_local(values)
    field.vector().apply("insert")


def _global_max_abs(values, comm):
    local = float(np.max(np.abs(values))) if len(values) else 0.0
    return float(comm.allreduce(local, op=_MPI.MAX))


def _global_min(values, comm):
    local = float(np.min(values)) if len(values) else np.inf
    return float(comm.allreduce(local, op=_MPI.MIN))


def _global_max(values, comm):
    local = float(np.max(values)) if len(values) else -np.inf
    return float(comm.allreduce(local, op=_MPI.MAX))


def _history_values(e_values, strain_values):
    """Return the dimensionless tensile energy density 0.5*e*<eps>_+^2."""
    tensile_strain = np.maximum(strain_values, 0.0)
    return 0.5 * e_values * tensile_strain**2


def _modulus_values(config, coordinates):
    """Evaluate the configured dimensionless modulus at given coordinates."""
    values = 1.0 - config.imperfection.amplitude * np.exp(
        -(coordinates - config.imperfection.center) ** 2
        / (2.0 * config.imperfection.width**2)
    )
    if np.any(values <= 0.0) or not np.all(np.isfinite(values)):
        raise ValueError("The configured modulus imperfection must give e(x) > 0")
    return values


def _reaction_star(u_field, d_field, e_field, k_res, domain_measure):
    """Return the volume-averaged reconstructed dimensionless stress."""
    degradation = (1.0 - d_field) ** 2 + k_res
    stress_integral = assemble(e_field * degradation * u_field.dx(0) * dx)
    return float(stress_integral / domain_measure)


def _stress_diagnostics(u_field, d_field, e_field, k_res, space, comm):
    """Project constitutive stress to DG0 and return its global spread."""
    stress_field = project(
        e_field * ((1.0 - d_field) ** 2 + k_res) * u_field.dx(0),
        space,
    )
    stress = stress_field.vector().get_local()
    if not np.all(np.isfinite(stress)):
        raise FloatingPointError("Reconstructed FEM stress contains non-finite values")
    minimum = _global_min(stress, comm)
    maximum = _global_max(stress, comm)
    return minimum, maximum, maximum - minimum


def _solve_elasticity(a_u, L_u, u_field, bcs, linear_solver):
    solve(a_u == L_u, u_field, bcs,
          solver_parameters={"linear_solver": linear_solver})


def _attempt_staggered_step(
    u_bar,
    u_bar_constant,
    a_u,
    L_u,
    a_d,
    L_d,
    bcs_u,
    u_new,
    u_committed,
    u_previous,
    d_iter,
    d_candidate,
    d_previous,
    H_prev_local,
    H_used_field,
    e_local,
    e_field,
    phys,
    Vhist,
    domain_measure,
    comm,
    solver_cfg,
):
    """Try one displacement increment from the last committed state.

    A failed attempt deliberately returns without committing any state. The
    caller may then reduce the displacement increment and call this function
    again; the first assignments restore the exact previous converged state.
    """
    u_bar_constant.assign(float(u_bar))
    d_iter.assign(d_previous)
    d_candidate.assign(d_previous)
    u_previous.assign(u_committed)

    converged = False
    err_u = np.inf
    err_d = np.inf
    err_H = np.inf
    err_sigma = np.inf
    reaction = np.nan
    H_next_local = H_prev_local.copy()
    stagger_it = -1

    for stagger_it in range(solver_cfg.max_stagger_iters):
        # Mechanics for the current damage iterate determines the history
        # used by this damage solve. H_prev_local remains immutable here.
        _solve_elasticity(
            a_u, L_u, u_new, bcs_u, solver_cfg.linear_solver
        )
        reaction_before = _reaction_star(
            u_new, d_iter, e_field, phys.k_res, domain_measure
        )
        strain_before = project(u_new.dx(0), Vhist).vector().get_local()
        psi_before = _history_values(e_local, strain_before)
        H_used_local = np.maximum(H_prev_local, psi_before)
        if not np.all(np.isfinite(H_used_local)):
            raise FloatingPointError("FEM H_used contains non-finite values")
        _assign_local(H_used_field, H_used_local)

        # H_used is fixed while the classical linear damage equation is solved.
        solve(
            a_d == L_d,
            d_candidate,
            [],
            solver_parameters={"linear_solver": solver_cfg.linear_solver},
        )
        d_candidate_values = d_candidate.vector().get_local()
        if not np.all(np.isfinite(d_candidate_values)):
            raise FloatingPointError("FEM damage contains non-finite values")
        d_iter_values = d_iter.vector().get_local().copy()
        residual_values = d_candidate_values - d_iter_values
        err_d = _global_max_abs(residual_values, comm)

        # Recompute mechanics and history from the updated damage field.
        d_iter.assign(d_candidate)
        _solve_elasticity(
            a_u, L_u, u_new, bcs_u, solver_cfg.linear_solver
        )
        reaction = _reaction_star(
            u_new, d_iter, e_field, phys.k_res, domain_measure
        )
        strain_after = project(u_new.dx(0), Vhist).vector().get_local()
        psi_after = _history_values(e_local, strain_after)
        H_next_local = np.maximum(H_prev_local, psi_after)
        if not np.all(np.isfinite(H_next_local)):
            raise FloatingPointError("FEM H_next contains non-finite values")

        err_u = _global_max_abs(
            u_new.vector().get_local() - u_previous.vector().get_local(), comm
        )
        err_sigma = abs(reaction - reaction_before)
        err_H = _global_max_abs(H_next_local - H_used_local, comm)

        u_previous.assign(u_new)
        if (
            err_u <= solver_cfg.atol_u
            and err_d <= solver_cfg.atol_d
            and err_H <= solver_cfg.atol_H
            and err_sigma <= solver_cfg.atol_sigma
        ):
            converged = True
            break

    return {
        "converged": converged,
        "err_u": err_u,
        "err_d": err_d,
        "err_H": err_H,
        "err_sigma": err_sigma,
        "reaction": reaction,
        "H_next_local": H_next_local,
        "stagger_iters": stagger_it + 1,
    }


def solve_fracture(config):
    """Run the local staggered FEM reference and write its snapshots."""
    _validate_config(config)
    comm = _MPI.COMM_WORLD
    rank = comm.rank

    phys = config.physical
    L0 = config.geometry.L0
    N_l = phys.l0 / L0
    N_G = phys.Gc / (phys.E * L0)
    peak_strain = analytical_peak_strain(N_l, N_G)
    peak_stress = analytical_peak_stress(N_l, N_G)

    if rank == 0:
        print(f"N_l = {N_l:.8e}")
        print(f"N_G = {N_G:.8e}")
        print(f"homogeneous peak strain (k_res=0) = {peak_strain:.8e}")
        print(f"homogeneous peak stress (k_res=0) = {peak_stress:.8e}")
        print(f"k_res = {phys.k_res:.8e}")

    nx = int(config.mesh.nx)
    mesh = IntervalMesh(comm, nx, 0.0, 1.0)
    Vu = FunctionSpace(mesh, "CG", 1)
    Vd = FunctionSpace(mesh, "CG", 1)
    V0 = FunctionSpace(mesh, "CG", 1)
    Vhist = FunctionSpace(mesh, "DG", 0)
    Vstress = FunctionSpace(mesh, "DG", 0)

    u_new = Function(Vu)
    u_committed = Function(Vu)
    u_previous = Function(Vu)
    d_iter = Function(Vd)
    d_candidate = Function(Vd)
    d_previous = Function(Vd)
    e_field = Function(V0)
    H_used_field = Function(Vhist)

    x_elastic_local = V0.tabulate_dof_coordinates()[:, 0]
    e_elastic_local = _modulus_values(config, x_elastic_local)
    _assign_local(e_field, e_elastic_local)
    x_hist_local = Vhist.tabulate_dof_coordinates()[:, 0]
    e_hist_local = _modulus_values(config, x_hist_local)

    H_prev_local = np.zeros_like(x_hist_local)
    _assign_local(H_used_field, H_prev_local)

    u_bar_constant = Constant(0.0)
    bc_left = DirichletBC(Vu, Constant(0.0), "near(x[0], 0.0)")
    bc_right = DirichletBC(Vu, u_bar_constant, "near(x[0], 1.0)")
    bcs_u = [bc_left, bc_right]

    u_trial = TrialFunction(Vu)
    u_test = TestFunction(Vu)
    d_trial = TrialFunction(Vd)
    d_test = TestFunction(Vd)

    # These forms reference mutable Function objects. Their values are updated
    # between staggered solves, while the variational forms remain fixed.
    degradation = (1.0 - d_iter) ** 2 + phys.k_res
    a_u = e_field * degradation * u_trial.dx(0) * u_test.dx(0) * dx
    L_u = Constant(0.0) * u_test * dx

    history_factor = 2.0 * N_l / N_G
    a_d = (
        N_l**2 * d_trial.dx(0) * d_test.dx(0)
        + (1.0 + history_factor * H_used_field) * d_trial * d_test
    ) * dx
    L_d = history_factor * H_used_field * d_test * dx

    domain_measure = assemble(Constant(1.0) * dx(domain=mesh))
    if not np.isfinite(domain_measure) or domain_measure <= 0.0:
        raise RuntimeError(f"Invalid assembled domain measure: {domain_measure}")

    reaction_history = []
    u_bar_history = []
    du_history = []
    stagger_iterations_history = []
    cutback_history = []
    u_snapshots = []
    d_snapshots = []
    H_snapshots = []
    snap_u_bar = []
    next_save = config.output.save_every

    u_bar = 0.0
    step = 0
    du_current = config.loading.du
    cutbacks_for_step = 0
    softening_started = False
    previous_reaction = None
    softening_history = []
    _, u_initial = _gather_1d(u_committed, Vu, comm)
    _, d_initial = _gather_1d(d_previous, Vd, comm)
    _, H_initial = _gather_1d(H_used_field, Vhist, comm)
    if rank == 0:
        if u_initial is None or d_initial is None or H_initial is None:
            raise RuntimeError("Failed to gather the initial FEM snapshot")
        u_snapshots.append(u_initial)
        d_snapshots.append(d_initial)
        H_snapshots.append(H_initial)
        snap_u_bar.append(0.0)
    while u_bar < config.loading.u_final:
        if np.isclose(u_bar, config.loading.u_final, rtol=0.0, atol=1.0e-14):
            u_bar = float(config.loading.u_final)
            break
        remaining = config.loading.u_final - u_bar
        trial_du = min(du_current, remaining)
        trial_u_bar = u_bar + trial_du
        attempt = _attempt_staggered_step(
            trial_u_bar,
            u_bar_constant,
            a_u,
            L_u,
            a_d,
            L_d,
            bcs_u,
            u_new,
            u_committed,
            u_previous,
            d_iter,
            d_candidate,
            d_previous,
            H_prev_local,
            H_used_field,
            e_hist_local,
            e_field,
            phys,
            Vhist,
            domain_measure,
            comm,
            config.solver,
        )

        if not attempt["converged"]:
            at_minimum = trial_du <= config.loading.du_min * (1.0 + 1e-12)
            if at_minimum or cutbacks_for_step >= config.loading.max_cutbacks:
                raise RuntimeError(
                    f"Load step at u_bar={trial_u_bar:.8e} did not converge "
                    f"after {attempt['stagger_iters']} staggered iterations "
                    f"and {cutbacks_for_step} cutbacks: "
                    f"err_u={attempt['err_u']:.3e}, "
                    f"err_d={attempt['err_d']:.3e}, "
                    f"err_H={attempt['err_H']:.3e}, "
                    f"err_sigma={attempt['err_sigma']:.3e}, "
                    f"du={trial_du:.3e}"
                )
            cutbacks_for_step += 1
            du_current = max(
                config.loading.du_min,
                trial_du * config.loading.cutback_factor,
            )
            if rank == 0:
                print(
                    f"Cutback before u_bar={trial_u_bar:.8e}: "
                    f"du={trial_du:.3e} -> {du_current:.3e}, "
                    f"err_d={attempt['err_d']:.3e}, "
                    f"err_sigma={attempt['err_sigma']:.3e}"
                )
            continue

        step += 1
        u_bar = trial_u_bar
        reaction = attempt["reaction"]
        H_next_local = attempt["H_next_local"]
        d_values = d_iter.vector().get_local()

        H_prev_local = H_next_local.copy()
        _assign_local(H_used_field, H_prev_local)
        d_previous.assign(d_iter)
        u_committed.assign(u_new)

        stress_min, stress_max, stress_spread = _stress_diagnostics(
            u_new, d_iter, e_field, phys.k_res, Vstress, comm
        )
        if stress_spread > config.solver.atol_stress_constancy:
            raise RuntimeError(
                f"Reconstructed stress is not spatially constant at load step "
                f"{step}: min={stress_min:.6e}, max={stress_max:.6e}, "
                f"spread={stress_spread:.3e}"
            )

        reaction_history.append(reaction)
        u_bar_history.append(float(u_bar))
        du_history.append(float(trial_du))
        stagger_iterations_history.append(int(attempt["stagger_iters"]))
        cutback_history.append(int(cutbacks_for_step))
        if (
            previous_reaction is not None
            and reaction < previous_reaction - config.solver.atol_sigma
        ):
            softening_started = True
        softening_history.append(bool(softening_started))

        is_final = np.isclose(
            u_bar, config.loading.u_final, rtol=0.0, atol=1e-14
        )
        if is_final:
            u_bar = float(config.loading.u_final)
        should_save = u_bar >= next_save - 0.5 * trial_du or is_final
        if should_save:
            _, u_full = _gather_1d(u_new, Vu, comm)
            _, d_full = _gather_1d(d_iter, Vd, comm)
            _, H_full = _gather_1d(H_used_field, Vhist, comm)
            if rank == 0:
                u_snapshots.append(u_full)
                d_snapshots.append(d_full)
                H_snapshots.append(H_full)
                snap_u_bar.append(float(u_bar))
            while next_save <= u_bar + 0.5 * trial_du:
                next_save += config.output.save_every

        max_d_global = _global_max(d_values, comm)
        if rank == 0 and step % config.output.log_every == 0:
            print(
                f"Step {step:5d}  u_bar={u_bar:.8e}  "
                f"du={trial_du:.3e}  reaction_star={reaction:.8e}  "
                f"max_d={max_d_global:.6e}  "
                f"iters={attempt['stagger_iters']}  "
                f"cutbacks={cutbacks_for_step}  "
                f"softening={softening_started}  "
                f"stress_spread={stress_spread:.3e}"
            )

        cutbacks_for_step = 0
        previous_reaction = reaction
        if softening_started:
            du_current = min(
                config.loading.du_softening,
                trial_du * config.loading.growth_factor,
            )
        else:
            du_current = min(
                config.loading.du_max,
                trial_du * config.loading.growth_factor,
            )

    x_full, _ = _gather_1d(d_iter, Vd, comm)
    x_H_full, _ = _gather_1d(H_used_field, Vhist, comm)
    if rank == 0:
        if x_full is None:
            raise RuntimeError("Rank zero did not receive the gathered FEM coordinates")
        if x_H_full is None:
            raise RuntimeError("Rank zero did not receive the history coordinates")
        if not u_snapshots or not d_snapshots or not H_snapshots:
            raise RuntimeError("No snapshots were collected")

        os.makedirs(config.output.dir, exist_ok=True)
        result_path = os.path.join(config.output.dir, config.output.filename)
        u_snapshots_array = np.asarray(u_snapshots)
        d_snapshots_array = np.asarray(d_snapshots)
        H_snapshots_array = np.asarray(H_snapshots)
        u_bar_array = np.asarray(u_bar_history)
        reaction_array = np.asarray(reaction_history)
        np.savez_compressed(
            result_path,
            x=x_full,
            x_H=x_H_full,
            u_bar=u_bar_array,
            du_history=np.asarray(du_history),
            stagger_iterations=np.asarray(stagger_iterations_history),
            cutbacks=np.asarray(cutback_history),
            softening=np.asarray(softening_history),
            snap_u_bar=np.asarray(snap_u_bar),
            u=u_snapshots_array,
            d=d_snapshots_array,
            reaction_star=reaction_array,
            force=reaction_array,
            H=H_snapshots_array,
            E0=float(phys.E),
            Gc=float(phys.Gc),
            l0=float(phys.l0),
            L0=float(L0),
            k_res=float(phys.k_res),
            N_l=float(N_l),
            N_G=float(N_G),
            imperfection_amplitude=float(config.imperfection.amplitude),
            imperfection_width=float(config.imperfection.width),
            imperfection_center=float(config.imperfection.center),
            mesh_nx=int(config.mesh.nx),
            loading_du_initial=float(config.loading.du),
            loading_du_min=float(config.loading.du_min),
            loading_du_max=float(config.loading.du_max),
            loading_du_softening=float(config.loading.du_softening),
            loading_cutback_factor=float(config.loading.cutback_factor),
            loading_growth_factor=float(config.loading.growth_factor),
            output_save_every=float(config.output.save_every),
            solver_atol_u=float(config.solver.atol_u),
            solver_atol_d=float(config.solver.atol_d),
            solver_atol_H=float(config.solver.atol_H),
            solver_atol_sigma=float(config.solver.atol_sigma),
            solver_atol_stress_constancy=float(config.solver.atol_stress_constancy),
            solver_max_stagger_iters=int(config.solver.max_stagger_iters),
            history_space="DG0_classical_max",
            history_method="Miehe_max_history",
        )
        print(f"Saved {result_path}")

        _plot_comparison(
            config,
            x_full,
            u_bar_array,
            reaction_array,
            u_snapshots_array,
            d_snapshots_array,
            np.asarray(snap_u_bar),
            N_l,
            N_G,
        )
        _plot_stress_strain(config, u_bar_array, reaction_array)
        _plot_stationary_profile(config, x_full, N_l)

    comm.Barrier()
    del a_u, L_u, a_d, L_d, degradation
    del u_trial, u_test, d_trial, d_test
    del bc_left, bc_right, bcs_u, u_bar_constant
    del mesh, Vu, Vd, V0, Vhist, Vstress
    del u_new, u_committed, u_previous, d_iter, d_candidate, d_previous
    del e_field, H_used_field
    gc.collect()


def _plot_comparison(
    config,
    x,
    u_bar,
    reaction_star,
    u_snapshots,
    d_snapshots,
    snap_u_bar,
    N_l,
    N_G,
):
    """Plot loaded response and homogeneous validation without false labels."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(config.output.plot_dir, exist_ok=True)
    peak_strain = analytical_peak_strain(N_l, N_G)
    peak_stress = analytical_peak_stress(N_l, N_G)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.plot(u_bar, reaction_star, "b-", linewidth=0.8, label="FEM staggered")
    eps = np.linspace(1e-8, max(float(u_bar[-1]), peak_strain) * 1.1, 500)
    ax.plot(eps, analytical_homogeneous_stress(eps, N_l, N_G), "r--",
            linewidth=1.5, label="homogeneous AT2, k_res=0")
    ax.axvline(peak_strain, color="gray", linestyle=":", linewidth=1,
               label=f"peak strain = {peak_strain:.4g}")
    ax.axhline(peak_stress, color="gray", linestyle="--", linewidth=1,
               label=f"peak stress = {peak_stress:.4g}")
    ax.set_xlabel("applied displacement u_bar*")
    ax.set_ylabel("reaction stress sigma*")
    ax.set_title("Loaded force-displacement")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(x, d_snapshots[-1], "b-", linewidth=1.0,
            label=f"loaded field, u_bar*={snap_u_bar[-1]:.4g}")
    ax.set_xlabel("x*")
    ax.set_ylabel("d")
    ax.set_title("Loaded damage field")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[2]
    center_idx = int(np.argmin(np.abs(x - config.imperfection.center)))
    center_damage = d_snapshots[:, center_idx]
    eps_fine = np.linspace(0.0, max(float(snap_u_bar[-1]), 1e-8), 300)
    ax.plot(snap_u_bar, center_damage, "b.-", markersize=2, linewidth=0.8,
            label="FEM center damage")
    ax.plot(eps_fine, analytical_homogeneous_damage(eps_fine, N_l, N_G),
            "r--", linewidth=1.5, label="homogeneous AT2")
    ax.axvline(peak_strain, color="gray", linestyle=":", linewidth=1,
               label=f"peak strain = {peak_strain:.4g}")
    ax.set_xlabel("applied displacement u_bar*")
    ax.set_ylabel("d at imperfection center")
    ax.set_title("Homogeneous validation / localization")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    path = os.path.join(config.output.plot_dir, "analytical_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Comparison plot saved to {path}")


def _plot_stationary_profile(config, x, N_l):
    """Write the separate finite-domain stationary-profile diagnostic."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(config.output.plot_dir, exist_ok=True)
    xc = 0.5
    finite_profile = analytical_stationary_profile(x, xc, N_l)
    infinite_profile = analytical_crack_profile(x, xc, N_l)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, finite_profile, label="finite stationary cosh profile")
    ax.plot(x, infinite_profile, "--", label="infinite-domain exponential")
    ax.set_xlabel("x*")
    ax.set_ylabel("d")
    ax.set_title("Separate stationary crack-profile diagnostic")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(config.output.plot_dir, "stationary_profile.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Stationary profile plot saved to {path}")


def _plot_stress_strain(config, strain_star, stress_star):
    """Write the dimensionless reaction-stress versus average-strain curve."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(config.output.plot_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(strain_star, stress_star, "b-", linewidth=1.0,
            label="FEM reaction stress")
    ax.set_xlabel("average strain $\\bar{\\varepsilon}^* = \\bar{u}^*$")
    ax.set_ylabel("reaction stress $\\sigma^*$")
    ax.set_title("Stress-strain response")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(config.output.plot_dir, "stress_strain.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Stress-strain plot saved to {path}")


_DEFAULT = os.path.join(os.path.dirname(__file__), "configs", "solver_base.py")
_CONFIG = _config_flags.DEFINE_config_file("config", _DEFAULT, "Config file")
FLAGS = _absl_flags.FLAGS


def _main(_):
    set_log_active(False)
    solve_fracture(FLAGS.config)


if __name__ == "__main__":
    _absl_app.run(_main)
