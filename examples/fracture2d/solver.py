"""Local FEM reference for the rounded-notch fracture2d benchmark.

The solver is intentionally independent of the repository's nonlocal
operators.  It uses the hybrid AT2/history formulation from Kristensen and
Martinez-Paneda (2020), plane-strain linear elasticity, and incremental
displacement loading.

Run in ``fenicsenv`` from the repository root::

    python examples/fracture2d/solver.py \
        --config=examples/fracture2d/configs/solver_base.py
"""

import gc
import json
import os
import subprocess
import sys

_pkgconfig_dir = os.path.join(sys.prefix, "lib", "pkgconfig")
if os.path.isdir(_pkgconfig_dir):
    os.environ.setdefault("PKG_CONFIG_PATH", "")
    if _pkgconfig_dir not in os.environ["PKG_CONFIG_PATH"]:
        os.environ["PKG_CONFIG_PATH"] = (
            _pkgconfig_dir + ":" + os.environ["PKG_CONFIG_PATH"]
        )

import meshio
import numpy as np
from absl import app as _absl_app, flags as _absl_flags
from fenics import (
    Constant,
    DirichletBC,
    Function,
    FunctionSpace,
    Mesh,
    MeshFunction,
    Measure,
    SubDomain,
    TestFunction,
    TrialFunction,
    VectorFunctionSpace,
    XDMFFile,
    assemble,
    derivative,
    dot,
    dx,
    grad,
    inner,
    near,
    project,
    set_log_active,
    solve,
    sym,
)
from ml_collections import config_flags as _config_flags
from ufl import Identity, as_tensor, conditional, gt
from mpi4py import MPI as _MPI

try:
    from .geometry import make_rounded_slot
except ImportError:
    from geometry import make_rounded_slot


def _finite_positive(name, value):
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value}")


def _validate_config(config):
    """Validate geometry, physical ranges, and all solver assumptions."""
    geom = config.geometry
    phys = config.physical
    boundary = config.boundary
    mesh_cfg = config.mesh
    loading = config.loading
    solver_cfg = config.solver
    output = config.output

    for name, value in {
        "geometry.Lx": geom.Lx,
        "geometry.Ly": geom.Ly,
        "geometry.L0": geom.L0,
        "geometry.notch_tip_x": geom.notch_tip_x,
        "geometry.notch_center_y": geom.notch_center_y,
        "geometry.notch_width": geom.notch_width,
        "geometry.notch_tip_radius": geom.notch_tip_radius,
        "physical.E": phys.E,
        "physical.Gc": phys.Gc,
        "physical.l0": phys.l0,
        "mesh.lc_far": mesh_cfg.lc_far,
        "mesh.lc_near": mesh_cfg.lc_near,
        "loading.du": loading.du,
        "loading.du_min": loading.du_min,
        "loading.du_max": loading.du_max,
        "loading.du_softening": loading.du_softening,
        "loading.growth_factor": loading.growth_factor,
        "loading.u_final": loading.u_final,
        "solver.atol_u": solver_cfg.atol_u,
        "solver.atol_d": solver_cfg.atol_d,
        "solver.atol_history": solver_cfg.atol_history,
        "solver.rtol_history": solver_cfg.rtol_history,
        "solver.atol_reaction": solver_cfg.atol_reaction,
        "solver.rtol_reaction": solver_cfg.rtol_reaction,
        "solver.max_newton_iters": solver_cfg.max_newton_iters,
        "solver.newton_atol": solver_cfg.newton_atol,
        "solver.newton_rtol": solver_cfg.newton_rtol,
        "solver.newton_relaxation": solver_cfg.newton_relaxation,
        "output.save_every": output.save_every,
    }.items():
        _finite_positive(name, value)

    if not 0.0 <= phys.nu < 0.5:
        raise ValueError(f"physical.nu must lie in [0, 0.5), got {phys.nu}")
    if not np.isfinite(phys.k_res) or phys.k_res < 0.0:
        raise ValueError(f"physical.k_res must be non-negative, got {phys.k_res}")
    if solver_cfg.gate_mode not in ("strict", "warning"):
        raise ValueError("solver.gate_mode must be 'strict' or 'warning'")
    if boundary.mode != "bottom_vertical_point_horizontal":
        raise ValueError(
            "fracture2d requires boundary.mode='bottom_vertical_point_horizontal'"
        )
    if not np.isclose(boundary.anchor_x, 0.0) or not np.isclose(boundary.anchor_y, 0.0):
        raise ValueError("The horizontal anchor must be at the lower-left point (0, 0)")
    if not np.isclose(geom.Lx, 1.0) or not np.isclose(geom.Ly, 1.0):
        raise ValueError("fracture2d is defined on the unit square")
    if not np.isclose(geom.L0, 1.0):
        raise ValueError("geometry.L0 must equal one for this unit-square baseline")
    if not np.isclose(geom.notch_width, 2.0 * phys.l0, rtol=0.0, atol=1.0e-12):
        raise ValueError("geometry.notch_width must equal 2 * physical.l0")
    if not np.isclose(
        geom.notch_tip_radius,
        0.5 * geom.notch_width,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError("geometry.notch_tip_radius must equal half the notch width")
    slot = make_rounded_slot(
        geom.notch_tip_x,
        geom.notch_center_y,
        geom.notch_width,
        geom.notch_tip_radius,
    )
    if slot.tip_x >= geom.Lx:
        raise ValueError("The notch tip must lie strictly inside the right boundary")
    if slot.center_y - slot.radius <= 0.0 or slot.center_y + slot.radius >= geom.Ly:
        raise ValueError("The notch must not intersect the top or bottom boundary")
    if mesh_cfg.lc_near >= mesh_cfg.lc_far:
        raise ValueError("mesh.lc_near must be smaller than mesh.lc_far")
    if not loading.du_min <= loading.du <= loading.du_max:
        raise ValueError("loading.du must lie between loading.du_min and loading.du_max")
    if not loading.du_min <= loading.du_softening <= loading.du_max:
        raise ValueError(
            "loading.du_softening must lie between loading.du_min and loading.du_max"
        )
    if not 0.0 < loading.cutback_factor < 1.0:
        raise ValueError("loading.cutback_factor must lie strictly between zero and one")
    if loading.growth_factor < 1.0:
        raise ValueError("loading.growth_factor must be at least one")
    if solver_cfg.newton_relaxation > 1.0:
        raise ValueError("solver.newton_relaxation must not exceed one")
    if not isinstance(solver_cfg.linear_solver, str) or not solver_cfg.linear_solver:
        raise ValueError("solver.linear_solver must be a non-empty string")
    field_grid_loads = np.asarray(output.field_grid_loads, dtype=float)
    if field_grid_loads.ndim != 1 or len(field_grid_loads) == 0:
        raise ValueError("output.field_grid_loads must be a non-empty 1D sequence")
    if not np.all(np.isfinite(field_grid_loads)) or np.any(field_grid_loads < 0.0):
        raise ValueError("output.field_grid_loads must be finite and non-negative")
    if len(field_grid_loads) > 1 and np.any(np.diff(field_grid_loads) <= 0.0):
        raise ValueError("output.field_grid_loads must be strictly increasing")
    allowed_linear_solvers = {
        "bicgstab",
        "cg",
        "default",
        "gmres",
        "lu",
        "minres",
        "mumps",
        "petsc",
        "richardson",
        "superlu",
        "superlu_dist",
        "tfqmr",
        "umfpack",
    }
    if solver_cfg.linear_solver not in allowed_linear_solvers:
        raise ValueError(
            f"Unsupported solver.linear_solver={solver_cfg.linear_solver!r}; "
            f"expected one of {sorted(allowed_linear_solvers)}"
        )
    if int(solver_cfg.max_newton_iters) != solver_cfg.max_newton_iters:
        raise ValueError("solver.max_newton_iters must be an integer")
    if solver_cfg.max_newton_iters < 1:
        raise ValueError("solver.max_newton_iters must be positive")
    if int(loading.max_cutbacks) != loading.max_cutbacks or loading.max_cutbacks < 1:
        raise ValueError("loading.max_cutbacks must be a positive integer")
    if int(solver_cfg.max_stagger_iters) != solver_cfg.max_stagger_iters:
        raise ValueError("solver.max_stagger_iters must be an integer")
    if solver_cfg.max_stagger_iters < 1:
        raise ValueError("solver.max_stagger_iters must be positive")
    if int(output.log_every) != output.log_every or output.log_every < 1:
        raise ValueError("output.log_every must be a positive integer")
    if not isinstance(mesh_cfg.rebuild, bool):
        raise ValueError("mesh.rebuild must be a boolean")
    return slot


def _lame_constants(E, nu):
    """Return the 3D Lamé constants used by the plane-strain model."""
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return lam, mu


def _positive_part(value):
    return conditional(gt(value, 0.0), value, 0.0)


def _hybrid_split(strain, lam, mu):
    """Return the paper's Amor-history / isotropic-mechanics split.

    The positive history driving energy is the volumetric-deviatoric split
    used by Kristensen and Martinez-Paneda (2020), while the displacement
    problem uses the full undamaged elastic energy multiplied by ``g(d)``.
    This is the paper's hybrid implementation and keeps the fixed-damage
    mechanics problem linear.
    """
    identity_2 = Identity(2)
    identity_3 = Identity(3)
    trace = strain[0, 0] + strain[1, 1]
    strain_3 = as_tensor(
        (
            (strain[0, 0], strain[0, 1], 0.0),
            (strain[1, 0], strain[1, 1], 0.0),
            (0.0, 0.0, 0.0),
        )
    )
    bulk = lam + 2.0 * mu / 3.0
    strain_dev_3 = strain_3 - (trace / 3.0) * identity_3
    trace_plus = _positive_part(trace)
    psi_plus = 0.5 * bulk * trace_plus**2 + mu * inner(strain_dev_3, strain_dev_3)
    psi_0 = 0.5 * lam * trace**2 + mu * inner(strain_3, strain_3)
    sigma_0 = lam * trace * identity_2 + 2.0 * mu * strain
    return psi_plus, psi_0, sigma_0


def _generate_mesh(config):
    """Generate Gmsh and convert its triangle cells to XDMF for DOLFIN."""
    geo_file = os.path.abspath(config.mesh.geo_file)
    msh_file = os.path.abspath(config.mesh.msh_file)
    xdmf_file = os.path.abspath(config.mesh.xdmf_file)
    for path in (geo_file, msh_file, xdmf_file):
        if path != geo_file:
            os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.isfile(geo_file):
        raise FileNotFoundError(f"Gmsh geometry file does not exist: {geo_file}")

    command = [
        "gmsh",
        "-2",
        "-setnumber",
        "Lx",
        str(config.geometry.Lx),
        "-setnumber",
        "Ly",
        str(config.geometry.Ly),
        "-setnumber",
        "tip_x",
        str(config.geometry.notch_tip_x),
        "-setnumber",
        "center_y",
        str(config.geometry.notch_center_y),
        "-setnumber",
        "r_tip",
        str(config.geometry.notch_tip_radius),
        "-setnumber",
        "lc_far",
        str(config.mesh.lc_far),
        "-setnumber",
        "lc_near",
        str(config.mesh.lc_near),
        geo_file,
        "-o",
        msh_file,
    ]
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "Gmsh failed while generating the rounded-notch mesh:\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )

    gmsh_mesh = meshio.read(msh_file)
    triangles = gmsh_mesh.get_cells_type("triangle")
    if triangles is None or len(triangles) == 0:
        raise RuntimeError(f"Gmsh mesh contains no triangle cells: {msh_file}")
    meshio.write(
        xdmf_file,
        meshio.Mesh(
            points=np.asarray(gmsh_mesh.points[:, :2]),
            cells=[("triangle", np.asarray(triangles, dtype=np.int32))],
        ),
    )
    with open(xdmf_file + ".metadata.json", "w", encoding="utf-8") as handle:
        json.dump(_mesh_signature(config), handle, indent=2, sort_keys=True)
    return xdmf_file


def _mesh_signature(config):
    return {
        "Lx": float(config.geometry.Lx),
        "Ly": float(config.geometry.Ly),
        "notch_tip_x": float(config.geometry.notch_tip_x),
        "notch_center_y": float(config.geometry.notch_center_y),
        "notch_tip_radius": float(config.geometry.notch_tip_radius),
        "lc_far": float(config.mesh.lc_far),
        "lc_near": float(config.mesh.lc_near),
    }


def _ensure_mesh(config):
    """Generate the configured mesh or fail clearly if a stale mesh is used."""
    xdmf_file = os.path.abspath(config.mesh.xdmf_file)
    geo_file = os.path.abspath(config.mesh.geo_file)
    metadata_file = xdmf_file + ".metadata.json"
    stale = config.mesh.rebuild or not os.path.isfile(xdmf_file)
    if not stale and not os.path.isfile(metadata_file):
        stale = True
    if not stale:
        try:
            with open(metadata_file, encoding="utf-8") as handle:
                stale = json.load(handle) != _mesh_signature(config)
        except (OSError, json.JSONDecodeError):
            stale = True
    if stale:
        return _generate_mesh(config)
    if os.path.getmtime(xdmf_file) < os.path.getmtime(geo_file):
        return _generate_mesh(config)
    return xdmf_file


def _assign_local(function, values):
    values = np.asarray(values, dtype=float)
    local = function.vector().get_local()
    if values.shape != local.shape:
        raise ValueError(
            f"Cannot assign shape {values.shape} to local FEniCS shape {local.shape}"
        )
    function.vector().set_local(values)
    function.vector().apply("insert")


def _global_max_abs(values, comm):
    local = float(np.max(np.abs(values))) if len(values) else 0.0
    return float(comm.allreduce(local, op=_MPI.MAX))


def _global_min(values, comm):
    local = float(np.min(values)) if len(values) else np.inf
    return float(comm.allreduce(local, op=_MPI.MIN))


def _global_max(values, comm):
    local = float(np.max(values)) if len(values) else -np.inf
    return float(comm.allreduce(local, op=_MPI.MAX))


def _gather_scalar(function, comm):
    """Gather a scalar field to rank zero, deduplicate and lexicographically sort."""
    space = function.function_space()
    coordinates = space.tabulate_dof_coordinates().reshape((-1, 2))
    values = function.vector().get_local()
    gathered = comm.gather((coordinates, values), root=0)
    if comm.rank != 0:
        return None, None

    all_coordinates = np.concatenate([entry[0] for entry in gathered], axis=0)
    all_values = np.concatenate([entry[1] for entry in gathered], axis=0)
    rounded = np.round(all_coordinates, decimals=12)
    _, unique_indices = np.unique(rounded, axis=0, return_index=True)
    coordinates = all_coordinates[unique_indices]
    values = all_values[unique_indices]
    order = np.lexsort((coordinates[:, 1], coordinates[:, 0]))
    return coordinates[order], values[order]


def _gather_displacement(function, comm):
    ux, uy = function.split(deepcopy=True)
    coordinates_x, values_x = _gather_scalar(ux, comm)
    coordinates_y, values_y = _gather_scalar(uy, comm)
    if comm.rank != 0:
        return None, None
    if not np.allclose(coordinates_x, coordinates_y, rtol=0.0, atol=1.0e-12):
        raise RuntimeError("Vector displacement components have inconsistent coordinates")
    return coordinates_x, np.column_stack((values_x, values_y))


def _gather_triangle_coordinates(mesh, comm):
    """Gather each mesh triangle as coordinates for rank-independent plotting."""
    coordinates = np.asarray(mesh.coordinates(), dtype=float)[:, :2]
    cells = np.asarray(mesh.cells(), dtype=np.int64)
    if cells.ndim != 2 or cells.shape[1] != 3:
        raise ValueError(f"Expected triangular cells, got shape {cells.shape}")
    local_triangles = coordinates[cells]
    gathered = comm.gather(local_triangles, root=0)
    if comm.rank != 0:
        return None
    triangles = np.concatenate(gathered, axis=0)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 2):
        raise RuntimeError(f"Invalid gathered triangle coordinates: {triangles.shape}")
    return triangles


def _triangle_indices(triangle_coordinates, coordinates):
    """Map gathered triangle coordinates onto the sorted CG1 vertex coordinates."""
    rounded = np.round(np.asarray(coordinates, dtype=float), decimals=12)
    indices = {tuple(point): index for index, point in enumerate(rounded)}
    triangles = np.round(np.asarray(triangle_coordinates, dtype=float), decimals=12)
    try:
        cells = np.asarray(
            [[indices[tuple(point)] for point in triangle] for triangle in triangles],
            dtype=np.int32,
        )
    except KeyError as error:
        raise RuntimeError(f"Mesh triangle vertex is missing from CG1 coordinates: {error}") from error
    if np.any(np.diff(np.sort(cells, axis=1), axis=1) == 0):
        raise RuntimeError("Mesh contains a degenerate triangle")
    return cells


def _align_cell_fields(snapshot, triangle_indices):
    """Align DG0 values with the archived triangle connectivity order."""
    centroids = snapshot["coords"][triangle_indices].mean(axis=1)
    rounded_centroids = np.round(centroids, decimals=12)

    def reorder(values, coordinates, name):
        rounded = np.round(np.asarray(coordinates, dtype=float), decimals=12)
        mapping = {tuple(point): index for index, point in enumerate(rounded)}
        order = [mapping.get(tuple(point)) for point in rounded_centroids]
        missing = [index for index, value in enumerate(order) if value is None]
        if missing:
            distances = np.sum(
                (centroids[missing, None, :] - rounded[None, :, :]) ** 2,
                axis=2,
            )
            nearest = np.argmin(distances, axis=1)
            nearest_distance = np.sqrt(distances[np.arange(len(missing)), nearest])
            if np.max(nearest_distance) > 1.0e-9:
                raise RuntimeError(
                    f"{name} DG0 coordinates do not match mesh cell centroids; "
                    f"maximum nearest distance={np.max(nearest_distance):.3e}"
                )
            for index, nearest_index in zip(missing, nearest):
                order[index] = int(nearest_index)
        order_indices = np.asarray(
            [value if value is not None else -1 for value in order],
            dtype=np.int64,
        )
        if np.any(order_indices < 0):
            raise RuntimeError(f"{name} DG0 reordering left unmatched cells")
        return np.asarray(values)[order_indices]

    snapshot["H"] = reorder(snapshot["H"], snapshot["H_coords"], "History")
    snapshot["sigma_xx"] = reorder(
        snapshot["sigma_xx"], snapshot["sigma_coords"], "Stress"
    )
    snapshot["sigma_yy"] = reorder(
        snapshot["sigma_yy"], snapshot["sigma_coords"], "Stress"
    )
    snapshot["sigma_xy"] = reorder(
        snapshot["sigma_xy"], snapshot["sigma_coords"], "Stress"
    )
    snapshot["H_coords"] = centroids
    snapshot["sigma_coords"] = centroids


def _solve_mechanics(F, J, displacement, bcs, config, load_value, stagger):
    parameters = {
        "newton_solver": {
            "absolute_tolerance": float(config.solver.newton_atol),
            "relative_tolerance": float(config.solver.newton_rtol),
            "maximum_iterations": int(config.solver.max_newton_iters),
            "relaxation_parameter": float(config.solver.newton_relaxation),
            "linear_solver": str(config.solver.linear_solver),
        },
    }
    try:
        solve(F == 0, displacement, bcs, J=J, solver_parameters=parameters)
    except RuntimeError as error:
        raise RuntimeError(
            f"Nonlinear mechanics failed at u_bar={load_value:.8e}, "
            f"stagger={stagger}: {error}"
        ) from error
    values = displacement.vector().get_local()
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("Mechanics solution contains non-finite values")


def _project_history(psi_plus, history_space):
    field = project(psi_plus, history_space)
    values = field.vector().get_local()
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("Projected tensile energy contains non-finite values")
    return field, values


def _attempt_staggered_step(
    trial_u_bar,
    u_bar_constant,
    F_u,
    J_u,
    u_trial,
    u_committed,
    u_previous,
    d_iter,
    d_candidate,
    d_previous,
    H_previous,
    H_used,
    psi_plus,
    a_d,
    L_d,
    bcs_u,
    reaction_form,
    config,
    comm,
):
    """Try one load step without modifying committed fields."""
    u_trial.assign(u_committed)
    u_previous.assign(u_committed)
    d_iter.assign(d_previous)
    d_candidate.assign(d_previous)

    H_previous_values = H_previous.vector().get_local().copy()
    H_next_values = H_previous_values.copy()
    converged = False
    err_u = np.inf
    err_d = np.inf
    err_H = np.inf
    err_H_relative = np.inf
    err_reaction = np.inf
    err_reaction_relative = np.inf
    reaction = np.nan
    stagger = 0

    for stagger in range(1, int(config.solver.max_stagger_iters) + 1):
        _solve_mechanics(
            F_u,
            J_u,
            u_trial,
            bcs_u,
            config,
            trial_u_bar,
            stagger,
        )
        reaction_before = float(assemble(reaction_form))
        H_from_before, H_before_values = _project_history(psi_plus, H_used.function_space())
        del H_from_before
        H_used_values = np.maximum(H_previous_values, H_before_values)
        _assign_local(H_used, H_used_values)

        try:
            solve(
                a_d == L_d,
                d_candidate,
                [],
                solver_parameters={"linear_solver": str(config.solver.linear_solver)},
            )
        except RuntimeError as error:
            raise RuntimeError(
                f"Damage solve failed at u_bar={trial_u_bar:.8e}, "
                f"stagger={stagger}: {error}"
            ) from error
        candidate_values = d_candidate.vector().get_local()
        previous_values = d_iter.vector().get_local().copy()
        if not np.all(np.isfinite(candidate_values)):
            raise FloatingPointError("Damage solution contains non-finite values")
        err_d = _global_max_abs(candidate_values - previous_values, comm)

        d_iter.assign(d_candidate)
        _solve_mechanics(
            F_u,
            J_u,
            u_trial,
            bcs_u,
            config,
            trial_u_bar,
            stagger,
        )
        reaction = float(assemble(reaction_form))
        _, H_after_values = _project_history(psi_plus, H_used.function_space())
        H_next_values = np.maximum(H_previous_values, H_after_values)
        err_u = _global_max_abs(
            u_trial.vector().get_local() - u_previous.vector().get_local(),
            comm,
        )
        err_H = _global_max_abs(H_next_values - H_used_values, comm)
        history_scale = max(
            _global_max(H_used_values, comm),
            _global_max(H_previous_values, comm),
            _global_max(H_next_values, comm),
        )
        err_H_relative = (
            0.0
            if history_scale > 0.0 and err_H == 0.0
            else np.inf
            if history_scale <= 0.0
            else err_H / history_scale
        )
        err_reaction = abs(reaction - reaction_before)
        reaction_scale = max(abs(reaction_before), abs(reaction), 1.0)
        err_reaction_relative = err_reaction / reaction_scale
        u_previous.assign(u_trial)

        if (
            err_u <= config.solver.atol_u
            and err_d <= config.solver.atol_d
            and (
                err_H <= config.solver.atol_history
                or err_H_relative <= config.solver.rtol_history
            )
            and (
                err_reaction <= config.solver.atol_reaction
                or err_reaction_relative <= config.solver.rtol_reaction
            )
        ):
            converged = True
            break

    return {
        "converged": converged,
        "stagger_iters": stagger,
        "err_u": err_u,
        "err_d": err_d,
        "err_H": err_H,
        "err_H_relative": err_H_relative,
        "err_reaction": err_reaction,
        "err_reaction_relative": err_reaction_relative,
        "reaction": reaction,
        "H_next_values": H_next_values,
    }


def _snapshot(u_field, d_field, H_field, sigma, history_space, comm):
    """Gather fields needed for a reproducible load-state archive."""
    coordinates, displacement = _gather_displacement(u_field, comm)
    d_coordinates, damage = _gather_scalar(d_field, comm)
    H_coordinates, history = _gather_scalar(H_field, comm)
    sx = project(sigma[0, 0], history_space)
    sy = project(sigma[1, 1], history_space)
    sxy = project(sigma[0, 1], history_space)
    sigma_coordinates, sigma_xx = _gather_scalar(sx, comm)
    _, sigma_yy = _gather_scalar(sy, comm)
    _, sigma_xy = _gather_scalar(sxy, comm)
    if comm.rank != 0:
        return None
    if not np.allclose(coordinates, d_coordinates, rtol=0.0, atol=1.0e-12):
        raise RuntimeError("Displacement and damage coordinates do not match")
    return {
        "coords": coordinates,
        "u": displacement,
        "d": damage,
        "H_coords": H_coordinates,
        "H": history,
        "sigma_coords": sigma_coordinates,
        "sigma_xx": sigma_xx,
        "sigma_yy": sigma_yy,
        "sigma_xy": sigma_xy,
    }


def _write_plots(config, results):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    plot_dir = os.path.abspath(config.output.plot_dir)
    os.makedirs(plot_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(results["u_bar"], results["reaction_y"], "k-")
    ax.scatter([0.0], [0.0], color="tab:blue", s=18, zorder=3)
    ax.set_xlabel(r"top displacement $\bar{u}_y$")
    ax.set_ylabel("reaction force")
    ax.set_title("Local FEM rounded-notch response")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "reaction_displacement.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(results["nominal_strain"], results["nominal_stress"], "k-")
    ax.scatter([0.0], [0.0], color="tab:blue", s=18, zorder=3)
    ax.set_xlabel(r"nominal strain $\bar{u}_y/L_y$")
    ax.set_ylabel("nominal stress")
    ax.set_title("Nominal stress-strain response")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "stress_strain.png"), dpi=150)
    plt.close(fig)

    final = results["snapshots"][-1]
    triangulation = mtri.Triangulation(
        results["coords"][:, 0],
        results["coords"][:, 1],
        triangles=results["triangles"],
    )

    def draw_vertex_field(axis, values, title, cmap):
        values = np.asarray(values, dtype=float)
        vmin = float(np.min(values))
        vmax = float(np.max(values))
        if vmin == vmax:
            padding = max(abs(vmin) * 1.0e-12, 1.0e-12)
            vmin -= padding
            vmax += padding
        artist = axis.tripcolor(
            triangulation,
            values,
            shading="gouraud",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        axis.set_aspect("equal")
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        return artist

    final_displacement = np.asarray(final["u"])
    final_magnitude = np.linalg.norm(final_displacement, axis=1)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), constrained_layout=True)
    for axis, values, title, cmap in (
        (
            axes[0],
            final_displacement[:, 0],
            "Final $u_x$",
            "coolwarm",
        ),
        (
            axes[1],
            final_displacement[:, 1],
            "Final $u_y$",
            "coolwarm",
        ),
        (axes[2], final_magnitude, "Final $|u|$", "viridis"),
        (axes[3], final["d"], "Final damage $d$", "magma"),
    ):
        artist = draw_vertex_field(axis, values, title, cmap)
        fig.colorbar(artist, ax=axis)
    fig.savefig(os.path.join(plot_dir, "final_fields.png"), dpi=150)
    plt.close(fig)

    history_artist = plt.figure(figsize=(6, 4))
    history_axis = history_artist.add_subplot(111)
    history_plot = history_axis.tripcolor(
        triangulation,
        facecolors=np.asarray(final["H"]),
        shading="flat",
        cmap="plasma",
    )
    history_axis.set_aspect("equal")
    history_axis.set_title("Final tensile history (DG0 cell field)")
    history_axis.set_xlabel("x")
    history_axis.set_ylabel("y")
    history_artist.colorbar(history_plot, ax=history_axis)
    history_artist.tight_layout()
    history_artist.savefig(os.path.join(plot_dir, "final_history_cellwise.png"), dpi=150)
    plt.close(history_artist)

    snapshot_u = np.asarray(results["snapshot_u_bar"])
    snapshot_damage_max = np.max(np.asarray(results["d"]), axis=1)
    peak_load = float(results["u_bar"][np.argmax(results["reaction_y"])])
    representative = [("initial", 0)]
    precrack = np.flatnonzero(snapshot_damage_max <= 0.5)
    representative.append(("before_crack", int(precrack[-1]) if len(precrack) else 0))
    representative.append(
        ("crack_development", int(np.argmin(np.abs(snapshot_u - peak_load))))
    )
    near_complete = np.flatnonzero(snapshot_damage_max >= 0.95)
    if len(near_complete):
        representative.append(("near_complete", int(near_complete[0])))
    representative.append(("fully_open", len(snapshot_u) - 1))

    unique_representative = []
    seen_indices = set()
    for label, index in representative:
        if index in seen_indices:
            continue
        seen_indices.add(index)
        unique_representative.append((label, index))
    for label, index in unique_representative:
        snapshot = results["snapshots"][index]
        displacement = np.asarray(snapshot["u"])
        magnitude = np.linalg.norm(displacement, axis=1)
        load_value = float(snapshot_u[index])
        fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
        fields = (
            (axes[0, 0], displacement[:, 0], "$u_x$", "coolwarm"),
            (axes[0, 1], displacement[:, 1], "$u_y$", "coolwarm"),
            (axes[1, 0], magnitude, "$|u|$", "viridis"),
            (axes[1, 1], snapshot["d"], "$d$", "magma"),
        )
        for axis, values, name, cmap in fields:
            artist = draw_vertex_field(axis, values, name, cmap)
            fig.colorbar(artist, ax=axis)
        fig.suptitle(f"{label}: $\\bar{{u}}_y={load_value:.6g}$")
        fig.savefig(
            os.path.join(plot_dir, f"representative_fields_{label}.png"),
            dpi=150,
        )
        plt.close(fig)

    selected_loads = np.asarray(config.output.field_grid_loads, dtype=float)
    selected_snapshot_u = np.asarray(results["snapshot_u_bar"], dtype=float)
    selected_indices = np.asarray(
        [int(np.argmin(np.abs(selected_snapshot_u - load))) for load in selected_loads],
        dtype=np.int64,
    )
    selected_errors = np.abs(selected_snapshot_u[selected_indices] - selected_loads)
    if np.all(selected_errors <= 1.0e-10):
        fig, axes = plt.subplots(
            3,
            len(selected_loads),
            figsize=(4.0 * len(selected_loads), 13),
            squeeze=False,
            constrained_layout=True,
        )
        row_labels = (r"$u_x$", r"$u_y$", r"$d$")
        row_cmaps = ("coolwarm", "coolwarm", "magma")
        for column, (load, index) in enumerate(
            zip(selected_loads, selected_indices)
        ):
            snapshot = results["snapshots"][int(index)]
            displacement = np.asarray(snapshot["u"])
            fields = (
                displacement[:, 0],
                displacement[:, 1],
                np.asarray(snapshot["d"]),
            )
            for row, (values, cmap) in enumerate(zip(fields, row_cmaps)):
                axis = axes[row, column]
                artist = draw_vertex_field(axis, values, "", cmap)
                axis.set_title(rf"$\bar{{u}}_y={load:.5f}$" if row == 0 else "")
                fig.colorbar(artist, ax=axis, fraction=0.046, pad=0.02)
        for row, label in enumerate(row_labels):
            axes[row, 0].set_ylabel(f"{label}\ny")
        fig.suptitle("Selected fracture2d physical fields", fontsize=16)
        fig.savefig(
            os.path.join(plot_dir, "selected_fields_grid.png"),
            dpi=150,
        )
        plt.close(fig)


def _plain_config(config):
    return config.to_dict()


def _prepend_history(values, initial):
    values = np.asarray(values)
    return np.concatenate((np.asarray([initial], dtype=values.dtype), values))


def solve_fracture(config):
    """Run the local FEM history-field continuation and save its archive."""
    slot = _validate_config(config)
    comm = _MPI.COMM_WORLD
    rank = comm.rank
    mesh_error = None
    if rank == 0:
        try:
            _ensure_mesh(config)
        except Exception as error:
            mesh_error = f"Mesh preparation failed: {error}"
    mesh_error = comm.bcast(mesh_error, root=0)
    if mesh_error is not None:
        raise RuntimeError(mesh_error)
    comm.Barrier()
    xdmf_file = os.path.abspath(config.mesh.xdmf_file)

    mesh = Mesh()
    with XDMFFile(xdmf_file) as xdmf:
        xdmf.read(mesh)
    if rank == 0:
        print(
            f"Mesh: {mesh.num_vertices()} vertices, {mesh.num_cells()} cells; "
            f"notch tip={slot.tip.tolist()}, radius={slot.radius:.6g}"
        )
    triangle_coordinates = _gather_triangle_coordinates(mesh, comm)

    V_u = VectorFunctionSpace(mesh, "CG", 1)
    V_d = FunctionSpace(mesh, "CG", 1)
    V_H = FunctionSpace(mesh, "DG", 0)
    u_trial = Function(V_u)
    u_committed = Function(V_u)
    u_previous = Function(V_u)
    d_iter = Function(V_d)
    d_candidate = Function(V_d)
    d_previous = Function(V_d)
    H_previous = Function(V_H)
    H_used = Function(V_H)
    _assign_local(d_iter, np.zeros_like(d_iter.vector().get_local()))
    _assign_local(d_candidate, np.zeros_like(d_candidate.vector().get_local()))
    _assign_local(d_previous, np.zeros_like(d_previous.vector().get_local()))
    _assign_local(H_previous, np.zeros_like(H_previous.vector().get_local()))
    _assign_local(H_used, np.zeros_like(H_used.vector().get_local()))

    u_bar_constant = Constant(0.0)

    class _Bottom(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], 0.0)

    class _Top(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], config.geometry.Ly)

    class _Anchor(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], config.boundary.anchor_x) and near(
                x[1], config.boundary.anchor_y
            )

    bottom = _Bottom()
    top = _Top()
    anchor = _Anchor()
    bcs_u = [
        DirichletBC(V_u.sub(1), Constant(0.0), bottom),
        DirichletBC(V_u.sub(0), Constant(0.0), anchor, "pointwise"),
        DirichletBC(V_u.sub(1), u_bar_constant, top),
    ]

    facets = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
    top.mark(facets, 1)
    ds_top = Measure("ds", domain=mesh, subdomain_data=facets)

    lam, mu = _lame_constants(config.physical.E, config.physical.nu)
    strain = sym(grad(u_trial))
    psi_plus, psi_0, sigma_0 = _hybrid_split(strain, lam, mu)
    degradation = (1.0 - d_iter) ** 2 + config.physical.k_res
    sigma = degradation * sigma_0
    elastic_energy = degradation * psi_0 * dx
    F_u = derivative(elastic_energy, u_trial, TestFunction(V_u))
    J_u = derivative(F_u, u_trial, TrialFunction(V_u))
    reaction_form = sigma[1, 1] * ds_top(1)

    d_test = TestFunction(V_d)
    d_trial = TrialFunction(V_d)
    a_d = (
        (config.physical.Gc / config.physical.l0) * d_trial * d_test
        + config.physical.Gc
        * config.physical.l0
        * dot(grad(d_trial), grad(d_test))
        + 2.0 * H_used * d_trial * d_test
    ) * dx
    L_d = 2.0 * H_used * d_test * dx
    fracture_energy = (
        config.physical.Gc
        * (
            d_iter**2 / (2.0 * config.physical.l0)
            + config.physical.l0 / 2.0 * dot(grad(d_iter), grad(d_iter))
        )
        * dx
    )

    u_bar = 0.0
    step = 0
    du_current = float(config.loading.du)
    previous_reaction = None
    cutbacks_for_step = 0
    softening_started = False
    reaction_history = []
    u_bar_history = []
    accepted_du_history = []
    stagger_history = []
    cutback_history = []
    damage_min_history = []
    damage_max_history = []
    healing_min_history = []
    damage_error_history = []
    history_error_history = []
    history_error_relative_history = []
    reaction_error_history = []
    reaction_error_relative_history = []
    gate_warning_history = []
    elastic_energy_history = []
    fracture_energy_history = []
    snapshots = []
    snapshot_u_bar = []
    next_save = float(config.output.save_every)

    initial_snapshot = _snapshot(
        u_committed, d_previous, H_previous, sigma, V_H, comm
    )
    triangle_indices = None
    if rank == 0:
        if initial_snapshot is None:
            raise RuntimeError("Rank zero did not receive the initial FEM snapshot")
        triangle_indices = _triangle_indices(
            triangle_coordinates,
            initial_snapshot["coords"],
        )
        _align_cell_fields(initial_snapshot, triangle_indices)
        snapshots.append(initial_snapshot)
        snapshot_u_bar.append(0.0)

    while u_bar < config.loading.u_final - 1.0e-14:
        remaining = float(config.loading.u_final - u_bar)
        trial_du = min(du_current, remaining)
        trial_u_bar = u_bar + trial_du
        u_bar_constant.assign(trial_u_bar)
        attempt = _attempt_staggered_step(
            trial_u_bar,
            u_bar_constant,
            F_u,
            J_u,
            u_trial,
            u_committed,
            u_previous,
            d_iter,
            d_candidate,
            d_previous,
            H_previous,
            H_used,
            psi_plus,
            a_d,
            L_d,
            bcs_u,
            reaction_form,
            config,
            comm,
        )
        if not attempt["converged"]:
            if config.solver.gate_mode == "strict":
                if (
                    trial_du <= config.loading.du_min * (1.0 + 1.0e-12)
                    or cutbacks_for_step >= config.loading.max_cutbacks
                ):
                    raise RuntimeError(
                        f"Load step u_bar={trial_u_bar:.8e} failed after "
                        f"{attempt['stagger_iters']} stagger iterations and "
                        f"{cutbacks_for_step} cutbacks: err_u={attempt['err_u']:.3e}, "
                        f"err_d={attempt['err_d']:.3e}, err_H={attempt['err_H']:.3e}, "
                        f"err_H_rel={attempt['err_H_relative']:.3e}, "
                        f"err_reaction={attempt['err_reaction']:.3e}, "
                        f"err_reaction_rel={attempt['err_reaction_relative']:.3e}"
                    )
                cutbacks_for_step += 1
                du_current = max(
                    config.loading.du_min,
                    trial_du * config.loading.cutback_factor,
                )
                if rank == 0:
                    print(
                        f"Cutback at u_bar={trial_u_bar:.8e}: "
                        f"du={trial_du:.3e} -> {du_current:.3e}"
                    )
                continue
            if rank == 0:
                # This is an intentional curve-diagnostic policy: the damage
                # field was solved with the previous staggered history, while
                # the committed history is the last finite post-mechanics value.
                print(
                    f"Gate warning at u_bar={trial_u_bar:.8e}: committing finite "
                    f"candidate after {attempt['stagger_iters']} stagger iterations "
                    f"(err_d={attempt['err_d']:.3e}, "
                    f"err_H_rel={attempt['err_H_relative']:.3e}, "
                    f"err_reaction_rel={attempt['err_reaction_relative']:.3e})"
                )

        step += 1
        u_bar = trial_u_bar
        reaction = float(attempt["reaction"])
        old_damage = d_previous.vector().get_local().copy()
        d_previous.assign(d_iter)
        u_committed.assign(u_trial)
        _assign_local(H_previous, attempt["H_next_values"])
        _assign_local(H_used, attempt["H_next_values"])

        d_values = d_previous.vector().get_local()
        healing = d_values - old_damage
        damage_min = _global_min(d_values, comm)
        damage_max = _global_max(d_values, comm)
        healing_min = _global_min(healing, comm)
        elastic_value = float(assemble(elastic_energy))
        fracture_value = float(assemble(fracture_energy))
        if not np.all(np.isfinite([reaction, elastic_value, fracture_value])):
            raise FloatingPointError("Accepted FEM state contains non-finite diagnostics")

        if previous_reaction is not None and reaction < previous_reaction:
            softening_started = True
        previous_reaction = reaction
        reaction_history.append(reaction)
        u_bar_history.append(u_bar)
        accepted_du_history.append(trial_du)
        stagger_history.append(attempt["stagger_iters"])
        cutback_history.append(cutbacks_for_step)
        damage_min_history.append(damage_min)
        damage_max_history.append(damage_max)
        healing_min_history.append(healing_min)
        damage_error_history.append(attempt["err_d"])
        history_error_history.append(attempt["err_H"])
        history_error_relative_history.append(attempt["err_H_relative"])
        reaction_error_history.append(attempt["err_reaction"])
        reaction_error_relative_history.append(attempt["err_reaction_relative"])
        gate_warning_history.append(not attempt["converged"])
        elastic_energy_history.append(elastic_value)
        fracture_energy_history.append(fracture_value)

        save_state = (
            u_bar >= next_save - 0.5 * trial_du
            or u_bar >= config.loading.u_final - 1.0e-14
        )
        if save_state:
            saved = _snapshot(u_committed, d_previous, H_previous, sigma, V_H, comm)
            if rank == 0:
                _align_cell_fields(saved, triangle_indices)
                snapshots.append(saved)
                snapshot_u_bar.append(u_bar)
            while next_save <= u_bar + 0.5 * trial_du:
                next_save += config.output.save_every

        if rank == 0 and step % config.output.log_every == 0:
            print(
                f"Step {step:5d} u_bar={u_bar:.8e} reaction_y={reaction:.8e} "
                f"max_d={damage_max:.6e} healing_min={healing_min:.3e} "
                f"history_err={attempt['err_H']:.3e} "
                f"history_rel={attempt['err_H_relative']:.3e} "
                f"reaction_rel={attempt['err_reaction_relative']:.3e} "
                f"stagger={attempt['stagger_iters']} cutbacks={cutbacks_for_step}"
            )

        cutbacks_for_step = 0
        if softening_started:
            du_current = min(config.loading.du_softening, trial_du * config.loading.growth_factor)
        else:
            du_current = min(config.loading.du_max, trial_du * config.loading.growth_factor)

    if rank == 0:
        if not snapshots:
            raise RuntimeError("No FEM snapshots were collected")
        result_dir = os.path.abspath(config.output.dir)
        os.makedirs(result_dir, exist_ok=True)
        result_path = os.path.join(result_dir, config.output.filename)
        final = snapshots[-1]
        u_bar_values = _prepend_history(u_bar_history, 0.0)
        reaction_values = _prepend_history(reaction_history, 0.0)
        results = {
            "coords": final["coords"],
            "triangles": triangle_indices,
            "u_bar": u_bar_values,
            "reaction_y": reaction_values,
            "nominal_strain": u_bar_values / float(config.geometry.Ly),
            "nominal_stress": reaction_values / float(config.geometry.Lx),
            "accepted_du": _prepend_history(accepted_du_history, 0.0),
            "stagger_iterations": _prepend_history(stagger_history, 0),
            "cutbacks": _prepend_history(cutback_history, 0),
            "damage_min": _prepend_history(damage_min_history, 0.0),
            "damage_max": _prepend_history(damage_max_history, 0.0),
            "healing_min": _prepend_history(healing_min_history, 0.0),
            "damage_error": _prepend_history(damage_error_history, 0.0),
            "history_error_abs": _prepend_history(history_error_history, 0.0),
            "history_error_rel": _prepend_history(history_error_relative_history, 0.0),
            "reaction_error_abs": _prepend_history(reaction_error_history, 0.0),
            "reaction_error_rel": _prepend_history(reaction_error_relative_history, 0.0),
            "gate_warning": _prepend_history(gate_warning_history, False),
            "elastic_energy": _prepend_history(elastic_energy_history, 0.0),
            "fracture_energy": _prepend_history(fracture_energy_history, 0.0),
            "snapshot_u_bar": np.asarray(snapshot_u_bar),
            "u": np.asarray([item["u"] for item in snapshots]),
            "d": np.asarray([item["d"] for item in snapshots]),
            "H_coords": final["H_coords"],
            "H": np.asarray([item["H"] for item in snapshots]),
            "sigma_coords": final["sigma_coords"],
            "sigma_xx": np.asarray([item["sigma_xx"] for item in snapshots]),
            "sigma_yy": np.asarray([item["sigma_yy"] for item in snapshots]),
            "sigma_xy": np.asarray([item["sigma_xy"] for item in snapshots]),
            "E": float(config.physical.E),
            "nu": float(config.physical.nu),
            "Gc": float(config.physical.Gc),
            "l0": float(config.physical.l0),
            "k_res": float(config.physical.k_res),
            "L0": float(config.geometry.L0),
            "notch_tip_x": float(config.geometry.notch_tip_x),
            "notch_center_y": float(config.geometry.notch_center_y),
            "notch_width": float(config.geometry.notch_width),
            "notch_tip_radius": float(config.geometry.notch_tip_radius),
            "history_method": "Kristensen_Martinez_Paneda_hybrid_Amor_positive_history",
            "plane_assumption": "plane_strain",
            "boundary_mode": str(config.boundary.mode),
            "horizontal_anchor": [
                float(config.boundary.anchor_x),
                float(config.boundary.anchor_y),
            ],
            "gate_mode": str(config.solver.gate_mode),
            "staggered_state_policy": (
                "strict_fixed_point"
                if config.solver.gate_mode == "strict"
                else "warning_last_finite_candidate"
            ),
            "initial_zero_state": True,
        }
        np.savez_compressed(result_path, **results)
        with open(os.path.join(result_dir, "config.json"), "w", encoding="utf-8") as handle:
            json.dump(_plain_config(config), handle, indent=2, sort_keys=True)
        _write_plots(config, results | {"snapshots": snapshots})
        print(f"Saved {result_path}")

    comm.Barrier()
    del F_u, J_u, a_d, L_d, elastic_energy, fracture_energy
    del u_trial, u_committed, u_previous, d_iter, d_candidate, d_previous
    del H_previous, H_used, mesh
    gc.collect()


_DEFAULT = os.path.join(os.path.dirname(__file__), "configs", "solver_base.py")
_CONFIG = _config_flags.DEFINE_config_file("config", _DEFAULT, "Config file")
FLAGS = _absl_flags.FLAGS


def _main(_):
    set_log_active(False)
    solve_fracture(FLAGS.config)


if __name__ == "__main__":
    _absl_app.run(_main)
