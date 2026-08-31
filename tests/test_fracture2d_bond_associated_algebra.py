"""Float64 algebra checks for the bond-associated model.

This is intentionally a self-contained manufactured check.  It does not
construct a production particle operator or a fracture solver.
"""

import unittest

import numpy as np


def _build_case():
    """Build the regular cloud used by the derivation check."""
    n_side = 5
    coordinates = np.asarray(
        [(i, j) for i in range(n_side) for j in range(n_side)],
        dtype=np.float64,
    )
    n_points = len(coordinates)
    delta = np.float64(2.01)
    relative = coordinates[None, :, :] - coordinates[:, None, :]
    distance = np.linalg.norm(relative, axis=2)
    edges = (distance > 0.0) & (distance < delta)
    omega = np.where(
        edges,
        120.0 / (np.pi * delta**4) * (1.0 - distance / delta) ** 2,
        0.0,
    )
    volumes = np.ones(n_points, dtype=np.float64)

    supports = {}
    deficient_supports = {}
    shape_tensors = {}
    condition_numbers = []
    for i in range(n_points):
        for j in np.flatnonzero(edges[i]):
            common = np.flatnonzero(edges[i] & edges[j])
            common = common[(common != i) & (common != j)]
            # The explicit target term is the discrete endpoint quadrature
            # contribution corresponding to the Chen--Spencer bond support.
            support = np.concatenate(([j], common))
            deficient_support = common
            supports[i, j] = support
            deficient_supports[i, j] = deficient_support
            shape = sum(
                omega[i, k]
                * volumes[k]
                * np.outer(
                    coordinates[k] - coordinates[i],
                    coordinates[k] - coordinates[i],
                )
                for k in support
            )
            shape_tensors[i, j] = shape
            condition_numbers.append(np.linalg.cond(shape))

    return {
        "coordinates": coordinates,
        "volumes": volumes,
        "edges": edges,
        "omega": omega,
        "supports": supports,
        "deficient_supports": deficient_supports,
        "shape_tensors": shape_tensors,
        "condition_numbers": np.asarray(condition_numbers),
    }


def _plane_strain_tangent(lame_lambda, shear_mu):
    """Return the row-major matrix for the displayed plane-strain map."""
    tangent = np.zeros((4, 4), dtype=np.float64)
    for column in range(4):
        perturbation = np.zeros((2, 2), dtype=np.float64)
        perturbation.flat[column] = 1.0
        strain = 0.5 * (perturbation + perturbation.T)
        stress = (
            lame_lambda * np.trace(strain) * np.eye(2)
            + 2.0 * shear_mu * strain
        )
        tangent[:, column] = stress.ravel()
    return tangent


def _kinematic_matrix(case, i, j, support):
    """Return ``L`` such that ``vec(F-I) = L @ u`` for one BA support."""
    coordinates = case["coordinates"]
    volumes = case["volumes"]
    omega = case["omega"]
    shape = sum(
        omega[i, k]
        * volumes[k]
        * np.outer(
            coordinates[k] - coordinates[i],
            coordinates[k] - coordinates[i],
        )
        for k in support
    )
    shape_inverse = np.linalg.inv(shape)

    n_points = len(coordinates)
    kinematic = np.zeros((4, 2 * n_points), dtype=np.float64)
    for k in support:
        weight = omega[i, k] * volumes[k]
        xi = coordinates[k] - coordinates[i]
        for component in range(2):
            block = weight * np.outer(np.eye(2)[component], xi)
            block = block @ shape_inverse
            kinematic[:, 2 * k + component] += block.ravel()
            kinematic[:, 2 * i + component] -= block.ravel()
    return kinematic


def _make_kinematics(case):
    kinematics = {}
    for (i, j), support in case["supports"].items():
        kinematics[i, j] = _kinematic_matrix(case, i, j, support)
    return kinematics


def _energy(displacement, damage, case, kinematics, lame_lambda, shear_mu,
            k_res):
    """Evaluate the declared center-partition mechanics energy."""
    omega = case["omega"]
    volumes = case["volumes"]
    energy = 0.0
    degradation = (1.0 - damage) ** 2 + k_res
    for (i, j), kinematic in kinematics.items():
        gradient_increment = (kinematic @ displacement).reshape(2, 2)
        strain = 0.5 * (gradient_increment + gradient_increment.T)
        psi = (
            0.5 * lame_lambda * np.trace(strain) ** 2
            + shear_mu * np.sum(strain * strain)
        )
        eta = omega[i, j] * volumes[j] / np.sum(omega[i] * volumes)
        energy += volumes[i] * degradation[i] * eta * psi
    return float(energy)


def _mechanics_residual_explicit(displacement, damage, case, lame_lambda,
                                 shear_mu, k_res):
    """Assemble Section 0.5 directly, without ``L`` or the Hessian.

    Each directed BA term recomputes its current moment, deformation gradient,
    small strain, and first Piola stress.  The vector
    ``P @ K^{-T} @ xi`` is then accumulated with the explicit target/source
    endpoint signs from the displacement difference.
    """
    coordinates = case["coordinates"]
    volumes = case["volumes"]
    omega = case["omega"]
    residual = np.zeros_like(displacement, dtype=np.float64)
    degradation = (1.0 - damage) ** 2 + k_res

    for (i, j), support in case["supports"].items():
        shape = sum(
            omega[i, k]
            * volumes[k]
            * np.outer(
                coordinates[k] - coordinates[i],
                coordinates[k] - coordinates[i],
            )
            for k in support
        )
        shape_inverse = np.linalg.inv(shape)
        current_moment = sum(
            omega[i, k]
            * volumes[k]
            * np.outer(
                coordinates[k] - coordinates[i]
                + displacement[2 * k:2 * k + 2]
                - displacement[2 * i:2 * i + 2],
                coordinates[k] - coordinates[i],
            )
            for k in support
        )
        deformation_gradient = current_moment @ shape_inverse
        gradient_increment = deformation_gradient - np.eye(2)
        strain = 0.5 * (gradient_increment + gradient_increment.T)
        stress = (
            lame_lambda * np.trace(strain) * np.eye(2)
            + 2.0 * shear_mu * strain
        )
        eta = omega[i, j] * volumes[j] / np.sum(omega[i] * volumes)
        center_factor = volumes[i] * degradation[i] * eta

        for k in support:
            xi = coordinates[k] - coordinates[i]
            if k == j:
                beta = omega[i, j] * volumes[j]
            else:
                beta = omega[i, k] * volumes[k]
            endpoint_force = beta * (stress @ shape_inverse.T @ xi)
            residual[2 * k:2 * k + 2] += center_factor * endpoint_force
            residual[2 * i:2 * i + 2] -= center_factor * endpoint_force
    return residual


def _boundary_particle_constraints(case):
    """Return exact boundary masks and the affine-lifting selector data."""
    coordinates = case["coordinates"]
    n_dofs = 2 * len(coordinates)
    bottom_particles = np.flatnonzero(coordinates[:, 1] == 0.0)
    top_particles = np.flatnonzero(coordinates[:, 1] == 4.0)
    anchor_candidates = np.flatnonzero(
        np.all(coordinates == np.asarray([0.0, 0.0]), axis=1)
    )
    if len(bottom_particles) != 5 or len(top_particles) != 5:
        raise AssertionError("exact 5x5 boundary masks were not constructed")
    if len(anchor_candidates) != 1:
        raise AssertionError("the lower-left anchor is not unique")
    anchor_particle = int(anchor_candidates[0])

    bottom_y_dofs = 2 * bottom_particles + 1
    top_y_dofs = 2 * top_particles + 1
    anchor_x_dof = np.asarray([2 * anchor_particle], dtype=np.int64)
    bottom_x_dofs = 2 * bottom_particles
    top_x_dofs = 2 * top_particles
    constrained_dofs = np.concatenate(
        (bottom_y_dofs, top_y_dofs, anchor_x_dof)
    )
    if len(np.unique(constrained_dofs)) != len(constrained_dofs):
        raise AssertionError("hard-constraint DOF masks overlap")
    all_dofs = np.arange(n_dofs, dtype=np.int64)
    free_dofs = np.setdiff1d(all_dofs, constrained_dofs, assume_unique=True)
    expected_free_boundary_x = np.concatenate(
        (np.setdiff1d(bottom_x_dofs, anchor_x_dof, assume_unique=True), top_x_dofs)
    )
    if not np.all(np.isin(expected_free_boundary_x, free_dofs)):
        raise AssertionError("non-anchor top/bottom horizontal DOFs are constrained")
    boundary_x_dofs = np.concatenate((bottom_x_dofs, top_x_dofs))
    expected_constrained_boundary_x = np.setdiff1d(
        boundary_x_dofs, expected_free_boundary_x, assume_unique=False
    )
    np.testing.assert_array_equal(expected_constrained_boundary_x, anchor_x_dof)
    P_f = np.eye(n_dofs, dtype=np.float64)[:, free_dofs]
    q_0 = np.zeros(n_dofs, dtype=np.float64)
    a_T = np.zeros(n_dofs, dtype=np.float64)
    a_T[top_y_dofs] = 1.0
    a_B = np.zeros(n_dofs, dtype=np.float64)
    a_B[bottom_y_dofs] = 1.0
    a_A = np.zeros(n_dofs, dtype=np.float64)
    a_A[anchor_x_dof] = 1.0
    return {
        "bottom_particles": bottom_particles,
        "top_particles": top_particles,
        "anchor_particle": anchor_particle,
        "bottom_y_dofs": bottom_y_dofs,
        "top_y_dofs": top_y_dofs,
        "anchor_x_dof": anchor_x_dof,
        "free_boundary_x_dofs": expected_free_boundary_x,
        "constrained_dofs": constrained_dofs,
        "free_dofs": free_dofs,
        "P_f": P_f,
        "q_0": q_0,
        "a_T": a_T,
        "a_B": a_B,
        "a_A": a_A,
    }


def _solve_boundary_particle_mechanics(hessian, boundary, u_bar):
    """Solve fixed-damage mechanics by DOF elimination and affine lifting."""
    P_f = boundary["P_f"]
    q_prescribed = boundary["q_0"] + boundary["a_T"] * float(u_bar)
    K_ff = P_f.T @ hessian @ P_f
    rhs = -P_f.T @ hessian @ q_prescribed
    q_f = np.linalg.solve(K_ff, rhs)
    displacement = P_f @ q_f + q_prescribed
    return displacement, K_ff


def run_checks():
    """Run all corrected-support checks and return reproducible metrics."""
    case = _build_case()
    coordinates = case["coordinates"]
    edges = case["edges"]
    omega = case["omega"]
    volumes = case["volumes"]
    supports = case["supports"]
    deficient_supports = case["deficient_supports"]
    n_points = len(coordinates)

    support_sizes = np.asarray([len(support) for support in supports.values()])
    np.testing.assert_equal(len(supports), 204)
    np.testing.assert_equal(np.min(support_sizes), 3)
    np.testing.assert_equal(
        min(np.linalg.matrix_rank(shape) for shape in case["shape_tensors"].values()),
        2,
    )
    np.testing.assert_equal(np.max(case["condition_numbers"]) < 10.0, True)

    # A target-only displacement must change the corrected F_{i|j}; it is
    # invisible to the old common-neighbor support by construction.
    target_i, target_j = 12, 13
    target_displacement = np.zeros(2 * n_points, dtype=np.float64)
    target_displacement[2 * target_j:2 * target_j + 2] = [1.0, -2.0]
    corrected_kinematic = _kinematic_matrix(
        case, target_i, target_j, supports[target_i, target_j]
    )
    deficient_kinematic = _kinematic_matrix(
        case, target_i, target_j, deficient_supports[target_i, target_j]
    )
    corrected_target_sensitivity = np.linalg.norm(
        corrected_kinematic @ target_displacement
    )
    deficient_target_sensitivity = np.linalg.norm(
        deficient_kinematic @ target_displacement
    )
    if not corrected_target_sensitivity > 1.0e-8:
        raise AssertionError("corrected BA support does not sense the target bond")
    np.testing.assert_allclose(deficient_target_sensitivity, 0.0, atol=1.0e-14)

    kinematics = _make_kinematics(case)
    youngs_modulus = 1.0
    poisson_ratio = 0.3
    lame_lambda = youngs_modulus * poisson_ratio / (
        (1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio)
    )
    shear_mu = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
    tangent = _plane_strain_tangent(lame_lambda, shear_mu)

    rng = np.random.default_rng(1234)
    damage = 0.2 + 0.6 * rng.random(n_points)
    k_res = 1.0e-8
    degradation = (1.0 - damage) ** 2 + k_res
    hessian = np.zeros((2 * n_points, 2 * n_points), dtype=np.float64)
    for (i, j), kinematic in kinematics.items():
        eta = omega[i, j] * volumes[j] / np.sum(omega[i] * volumes)
        hessian += (
            volumes[i]
            * degradation[i]
            * eta
            * kinematic.T
            @ tangent
            @ kinematic
        )

    # Boundary-particle hard constraints are checked at fixed damage only;
    # no history or damage update enters this mechanics/reaction diagnostic.
    boundary = _boundary_particle_constraints(case)
    boundary_particles = np.concatenate(
        (boundary["bottom_particles"], boundary["top_particles"])
    )
    for particle in boundary_particles:
        if not np.any(edges[particle]):
            raise AssertionError("boundary particle is absent from the graph")
        if not any(
            particle in support for support in supports.values()
        ):
            raise AssertionError("boundary particle is absent from BA supports")
        if not any(i == particle for i, _ in supports):
            raise AssertionError("boundary particle is absent as an energy center")
    boundary_u_bar = 0.08
    displacement, free_stiffness = _solve_boundary_particle_mechanics(
        hessian, boundary, u_bar=boundary_u_bar
    )
    np.testing.assert_allclose(
        displacement[boundary["bottom_y_dofs"]], 0.0, atol=1.0e-14
    )
    np.testing.assert_allclose(
        displacement[boundary["top_y_dofs"]], boundary_u_bar, atol=1.0e-14
    )
    np.testing.assert_allclose(
        displacement[boundary["anchor_x_dof"]], 0.0, atol=1.0e-14
    )
    free_eigenvalues = np.linalg.eigvalsh(free_stiffness)
    free_minimum_eigenvalue = float(free_eigenvalues[0])
    free_maximum_eigenvalue = float(free_eigenvalues[-1])
    free_spd_tolerance = 1.0e-12 * max(1.0, free_maximum_eigenvalue)
    if not free_minimum_eigenvalue > free_spd_tolerance:
        raise AssertionError(
            f"hard-constraint free block is not strictly SPD: "
            f"min={free_minimum_eigenvalue}, tolerance={free_spd_tolerance}"
        )
    free_condition_number = free_maximum_eigenvalue / free_minimum_eigenvalue

    explicit_boundary_residual = _mechanics_residual_explicit(
        displacement,
        damage,
        case,
        lame_lambda,
        shear_mu,
        k_res,
    )
    free_residual = boundary["P_f"].T @ explicit_boundary_residual
    free_residual_norm = float(np.linalg.norm(free_residual))
    if not free_residual_norm < 1.0e-12:
        raise AssertionError(f"free hard-constraint residual is {free_residual_norm}")

    top_reaction = float(boundary["a_T"] @ explicit_boundary_residual)
    top_residual_sum = float(
        np.sum(explicit_boundary_residual[boundary["top_y_dofs"]])
    )
    top_reaction_sum_error = abs(top_reaction - top_residual_sum)
    np.testing.assert_allclose(top_reaction, top_residual_sum, atol=1.0e-14)
    bottom_reaction = float(boundary["a_B"] @ explicit_boundary_residual)
    anchor_reaction = float(boundary["a_A"] @ explicit_boundary_residual)
    vertical_reaction_balance_error = abs(top_reaction + bottom_reaction)
    if not vertical_reaction_balance_error < 1.0e-12:
        raise AssertionError(
            f"top/bottom vertical reaction balance error is "
            f"{vertical_reaction_balance_error}"
        )
    if not np.all(np.isfinite([top_reaction, bottom_reaction, anchor_reaction])):
        raise AssertionError("hard-constraint reactions are not finite")

    reaction_epsilon = 1.0e-5
    displacement_plus, _ = _solve_boundary_particle_mechanics(
        hessian, boundary, u_bar=boundary_u_bar + reaction_epsilon
    )
    displacement_minus, _ = _solve_boundary_particle_mechanics(
        hessian, boundary, u_bar=boundary_u_bar - reaction_epsilon
    )
    reduced_energy_fd = (
        _energy(
            displacement_plus,
            damage,
            case,
            kinematics,
            lame_lambda,
            shear_mu,
            k_res,
        )
        - _energy(
            displacement_minus,
            damage,
            case,
            kinematics,
            lame_lambda,
            shear_mu,
            k_res,
        )
    ) / (2.0 * reaction_epsilon)
    reaction_energy_relative_error = abs(reduced_energy_fd - top_reaction) / max(
        abs(reduced_energy_fd), abs(top_reaction), 1.0e-30
    )
    if not reaction_energy_relative_error < 1.0e-9:
        raise AssertionError(
            f"top reaction/reduced-energy FD error is "
            f"{reaction_energy_relative_error}"
        )

    # Affine reproduction, including a nonzero translation.
    prescribed_f = np.asarray([[1.12, 0.035], [-0.018, 0.93]], dtype=np.float64)
    translation = np.asarray([0.13, -0.22], dtype=np.float64)
    affine_displacement = (
        coordinates @ (prescribed_f - np.eye(2)).T + translation
    ).ravel()
    affine_errors = [
        np.max(
            np.abs(
                np.eye(2)
                + (kinematic @ affine_displacement).reshape(2, 2)
                - prescribed_f
            )
        )
        for kinematic in kinematics.values()
    ]
    affine_max_error = float(np.max(affine_errors))
    if not affine_max_error < 5.0e-14:
        raise AssertionError(f"affine reproduction error is {affine_max_error}")

    # Translation and infinitesimal rotation are the three expected rigid
    # modes of the small-strain plane-strain energy.
    translation_x = np.zeros(2 * n_points, dtype=np.float64)
    translation_x[0::2] = 0.37
    translation_y = np.zeros(2 * n_points, dtype=np.float64)
    translation_y[1::2] = -0.29
    skew = np.asarray([[0.0, -1.0], [1.0, 0.0]], dtype=np.float64)
    infinitesimal_rotation = (1.0e-4 * coordinates @ skew.T).ravel()
    rigid_metrics = []
    for mode in (translation_x, translation_y, infinitesimal_rotation):
        mode_energy = _energy(
            mode,
            damage,
            case,
            kinematics,
            lame_lambda,
            shear_mu,
            k_res,
        )
        mode_gradient = hessian @ mode
        rigid_metrics.append((mode_energy, np.linalg.norm(mode_gradient)))
        if not mode_energy < 1.0e-25:
            raise AssertionError(f"rigid mode energy is {mode_energy}")
        if not np.linalg.norm(mode_gradient) < 1.0e-12:
            raise AssertionError("rigid mode has a nonzero mechanics gradient")

    # Energy-gradient directional finite difference and Hessian JVP.
    displacement = 1.0e-2 * rng.normal(size=2 * n_points)
    direction = rng.normal(size=2 * n_points)
    direction /= np.linalg.norm(direction)
    explicit_residual = _mechanics_residual_explicit(
        displacement,
        damage,
        case,
        lame_lambda,
        shear_mu,
        k_res,
    )
    hessian_gradient = hessian @ displacement
    residual_hessian_relative_error = np.linalg.norm(
        explicit_residual - hessian_gradient
    ) / max(
        np.linalg.norm(explicit_residual), np.linalg.norm(hessian_gradient), 1.0e-30
    )
    if not residual_hessian_relative_error < 1.0e-12:
        raise AssertionError(
            "explicit mechanics residual disagrees with Hessian @ displacement: "
            f"{residual_hessian_relative_error}"
        )
    epsilon = 1.0e-3
    finite_difference_energy = (
        _energy(
            displacement + epsilon * direction,
            damage,
            case,
            kinematics,
            lame_lambda,
            shear_mu,
            k_res,
        )
        - _energy(
            displacement - epsilon * direction,
            damage,
            case,
            kinematics,
            lame_lambda,
            shear_mu,
            k_res,
        )
    ) / (2.0 * epsilon)
    analytic_directional_energy = float(explicit_residual @ direction)
    energy_gradient_relative_error = abs(
        finite_difference_energy - analytic_directional_energy
    ) / max(
        abs(finite_difference_energy), abs(analytic_directional_energy), 1.0e-30
    )
    if not energy_gradient_relative_error < 1.0e-10:
        raise AssertionError(
            f"energy-gradient finite-difference error is "
            f"{energy_gradient_relative_error}"
        )

    residual_plus = _mechanics_residual_explicit(
        displacement + epsilon * direction,
        damage,
        case,
        lame_lambda,
        shear_mu,
        k_res,
    )
    residual_minus = _mechanics_residual_explicit(
        displacement - epsilon * direction,
        damage,
        case,
        lame_lambda,
        shear_mu,
        k_res,
    )
    jvp_finite_difference = (
        residual_plus
        - residual_minus
    ) / (2.0 * epsilon)
    jvp_analytic = hessian @ direction
    hessian_jvp_relative_error = np.linalg.norm(
        jvp_finite_difference - jvp_analytic
    ) / max(np.linalg.norm(jvp_analytic), 1.0e-30)
    hessian_symmetry_relative_error = np.linalg.norm(
        hessian - hessian.T
    ) / np.linalg.norm(hessian)
    if not hessian_jvp_relative_error < 1.0e-10:
        raise AssertionError(f"Hessian JVP error is {hessian_jvp_relative_error}")
    if not hessian_symmetry_relative_error < 1.0e-13:
        raise AssertionError(
            f"Hessian symmetry error is {hessian_symmetry_relative_error}"
        )

    # Independently check the directed-edge AT2 coefficient and the volume
    # self-adjoint relation on a nonuniform volume vector.
    damage_volumes = np.linspace(0.8, 1.2, n_points, dtype=np.float64)
    damage_operator = (
        np.diag(omega @ damage_volumes)
        - omega * damage_volumes[None, :]
    )
    volume_matrix = np.diag(damage_volumes)
    volume_adjoint_error = np.linalg.norm(
        volume_matrix @ damage_operator - damage_operator.T @ volume_matrix
    ) / np.linalg.norm(volume_matrix @ damage_operator)
    if not volume_adjoint_error < 1.0e-14:
        raise AssertionError(f"volume self-adjoint error is {volume_adjoint_error}")

    damage_rng = np.random.default_rng(9876)
    damage_field = 0.1 + 0.7 * damage_rng.random(n_points)
    history_field = 0.01 + 0.02 * damage_rng.random(n_points)
    fracture_toughness = 2.7
    regularization_length = 0.024
    graph_energy = 0.25 * np.sum(
        damage_volumes[:, None]
        * damage_volumes[None, :]
        * omega
        * (damage_field[None, :] - damage_field[:, None]) ** 2
    )
    graph_quadratic = 0.5 * damage_field @ volume_matrix @ damage_operator @ damage_field
    np.testing.assert_allclose(graph_energy, graph_quadratic, rtol=1.0e-14, atol=1.0e-14)

    def damage_energy(values):
        return float(
            np.sum(
                damage_volumes
                * (
                    fracture_toughness / (2.0 * regularization_length) * values**2
                    + history_field * (1.0 - values) ** 2
                )
            )
            + fracture_toughness
            * regularization_length
            / 4.0
            * np.sum(
                damage_volumes[:, None]
                * damage_volumes[None, :]
                * omega
                * (values[None, :] - values[:, None]) ** 2
            )
        )

    damage_strong_gradient = (
        fracture_toughness / regularization_length * damage_field
        + fracture_toughness * regularization_length * damage_operator @ damage_field
        + 2.0 * history_field * damage_field
        - 2.0 * history_field
    )
    damage_direction = damage_rng.normal(size=n_points)
    damage_direction /= np.linalg.norm(damage_direction)
    damage_epsilon = 1.0e-5
    damage_fd = (
        damage_energy(damage_field + damage_epsilon * damage_direction)
        - damage_energy(damage_field - damage_epsilon * damage_direction)
    ) / (2.0 * damage_epsilon)
    damage_analytic = float(
        (damage_volumes * damage_strong_gradient) @ damage_direction
    )
    damage_gradient_relative_error = abs(damage_fd - damage_analytic) / max(
        abs(damage_fd), abs(damage_analytic), 1.0e-30
    )
    if not damage_gradient_relative_error < 1.0e-8:
        raise AssertionError(
            f"directed-edge AT2 gradient error is {damage_gradient_relative_error}"
        )

    eigenvalues = np.linalg.eigvalsh(hessian)
    spectral_scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    psd_tolerance = 1.0e-12 * spectral_scale
    rigid_spectrum_tolerance = 1.0e-10 * spectral_scale
    minimum_eigenvalue = float(eigenvalues[0])
    if minimum_eigenvalue < -psd_tolerance:
        raise AssertionError(
            f"mechanics Hessian is not PSD within tolerance: "
            f"min={minimum_eigenvalue}, tolerance={psd_tolerance}"
        )
    rigid_zero_modes = int(
        np.count_nonzero(np.abs(eigenvalues) <= rigid_spectrum_tolerance)
    )
    np.testing.assert_equal(rigid_zero_modes, 3)
    first_nonrigid_eigenvalue = float(eigenvalues[rigid_zero_modes])
    if not first_nonrigid_eigenvalue > 10.0 * rigid_spectrum_tolerance:
        raise AssertionError(
            f"first non-rigid eigenvalue {first_nonrigid_eigenvalue} is not "
            f"separated from rigid-spectrum tolerance {rigid_spectrum_tolerance}"
        )

    return {
        "directed_edges": len(supports),
        "support_size_min": int(np.min(support_sizes)),
        "support_size_max": int(np.max(support_sizes)),
        "shape_rank_min": min(
            np.linalg.matrix_rank(shape)
            for shape in case["shape_tensors"].values()
        ),
        "shape_condition_min": float(np.min(case["condition_numbers"])),
        "shape_condition_max": float(np.max(case["condition_numbers"])),
        "target_sensitivity_corrected": float(corrected_target_sensitivity),
        "target_sensitivity_deficient": float(deficient_target_sensitivity),
        "bottom_boundary_particle_count": len(boundary["bottom_particles"]),
        "top_boundary_particle_count": len(boundary["top_particles"]),
        "hard_constraint_count": len(boundary["constrained_dofs"]),
        "free_dof_count": len(boundary["free_dofs"]),
        "free_boundary_horizontal_dof_count": len(boundary["free_boundary_x_dofs"]),
        "free_stiffness_minimum_eigenvalue": free_minimum_eigenvalue,
        "free_stiffness_condition_number": free_condition_number,
        "free_residual_norm": free_residual_norm,
        "top_reaction": top_reaction,
        "top_residual_sum_error": top_reaction_sum_error,
        "bottom_reaction": bottom_reaction,
        "anchor_reaction": anchor_reaction,
        "vertical_reaction_balance_error": vertical_reaction_balance_error,
        "reaction_energy_relative_error": float(reaction_energy_relative_error),
        "affine_max_error": affine_max_error,
        "translation_x_energy": rigid_metrics[0][0],
        "translation_y_energy": rigid_metrics[1][0],
        "infinitesimal_rotation_energy": rigid_metrics[2][0],
        "translation_x_gradient_norm": rigid_metrics[0][1],
        "translation_y_gradient_norm": rigid_metrics[1][1],
        "infinitesimal_rotation_gradient_norm": rigid_metrics[2][1],
        "energy_gradient_relative_error": float(energy_gradient_relative_error),
        "residual_hessian_relative_error": float(residual_hessian_relative_error),
        "hessian_jvp_relative_error": float(hessian_jvp_relative_error),
        "hessian_symmetry_relative_error": float(hessian_symmetry_relative_error),
        "volume_self_adjoint_relative_error": float(volume_adjoint_error),
        "directed_edge_at2_gradient_relative_error": float(
            damage_gradient_relative_error
        ),
        "rigid_zero_modes": rigid_zero_modes,
        "mechanics_hessian_psd_tolerance": psd_tolerance,
        "rigid_spectrum_tolerance": rigid_spectrum_tolerance,
        "minimum_eigenvalue": minimum_eigenvalue,
        "first_nonrigid_eigenvalue": first_nonrigid_eigenvalue,
        "maximum_eigenvalue": float(eigenvalues[-1]),
    }


class Fracture2DBondAssociatedAlgebraTest(unittest.TestCase):
    def test_corrected_support_and_energy_algebra(self):
        run_checks()


if __name__ == "__main__":
    for name, value in run_checks().items():
        print(f"{name}={value:.16e}" if isinstance(value, float) else f"{name}={value}")
