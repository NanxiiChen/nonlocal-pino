"""Local 3D Allen-Cahn FEM reference with fixed PINN/PINO physical parameters.

Run in fenicsenv, never concurrently with JAX GPU work. Parameters and geometry
come from explicit saved training config and reference manifest paths. This
solver does not import the training residual or nonlocal spatial operator.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ['PKG_CONFIG_PATH'] = str(Path(sys.prefix) / 'lib/pkgconfig') + os.pathsep + os.environ.get('PKG_CONFIG_PATH', '')

import dolfin as df
import numpy as np
from mpi4py import MPI

from examples.icemelting3d.ellipsoids import signed_distance_numpy


def load_problem(config_path, manifest_path, index):
    """Read the precise fixed physical problem; never infer epsilon from mesh."""
    c = json.loads(Path(config_path).read_text())
    manifest = json.loads(Path(manifest_path).read_text())
    if not isinstance(index, int) or index not in manifest['test_indices']:
        raise ValueError('FEM case must be an explicitly declared held-out input')
    for key in ('physical', 'nondim', 'initial'):
        if c[key] != manifest['identity'][key]:
            raise ValueError(f'Training/reference {key} mismatch')
    if c['initial']['kind'] != 'ellipsoid_sdf':
        raise ValueError('This reference requires the explicit ellipsoid_sdf IC')
    p, nd = c['physical'], c['nondim']
    values = np.array([p['M'], p['lam'], p['eps_star'], nd['L_c'], nd['T_c']])
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError('Physical coefficients/scales must be finite positive values')
    n1 = p['M'] * nd['T_c'] / nd['L_c']**2
    return dict(physical=p, nondim=nd, initial=c['initial'], index=index,
                parameters=manifest['parameters'][index],
                coefficients=[n1, n1 / p['eps_star']**2,
                              p['lam'] * nd['T_c'] / (nd['L_c'] * np.sqrt(2) * p['eps_star'])],
                boundary='homogeneous Neumann on all six faces',
                equation='phi_t=N1*Delta(phi)-N2*(phi^3-phi)-N3*(1-phi^2)')


def initial_field(points, problem):
    return np.tanh(signed_distance_numpy(points, problem['parameters'],
                   problem['initial']['reference_sdf_iterations'])
                   / (np.sqrt(2) * problem['physical']['eps_star']))


def weak_residual(phi, previous, test, dt, coefficients, measure):
    """Implicit midpoint, consistent CG mass, natural Neumann diffusion.

    Integrating -N1*Delta(phi_mid)*v by parts gives +N1*grad(phi_mid).grad(v);
    zero normal flux removes only the boundary integral. The physical reaction
    polynomial is integrated by quadrature, not nodally collocated/lumped.
    """
    n1, n2, n3 = coefficients
    midpoint = (phi + previous) / 2
    return ((phi - previous) * test / dt
            + n1 * df.dot(df.grad(midpoint), df.grad(test))
            + n2 * (midpoint**3 - midpoint) * test
            + n3 * (1 - midpoint**2) * test) * measure


def make_imex_step(phi, V, dt, coefficients, measure, controls):
    """Consistent FEM SBDF2, BE/explicit startup; no nodal reaction surrogate.

    (1.5 M+dt*N1*K)p[n+1] = 2Mp[n]-.5Mp[n-1]
                            -dt*(2G(p[n])-G(p[n-1])).
    G_i(p)=integral (N2*(p^3-p)+N3*(1-p^2))*v_i.
    See Ascher/Ruuth/Wetton1995, doi:10.1137/0732037.
    """
    trial, test = df.TrialFunction(V), df.TestFunction(V)
    n1, n2, n3 = coefficients
    mass = df.assemble(trial * test * measure)
    stiffness = df.assemble(df.dot(df.grad(trial), df.grad(test)) * measure)
    first = mass.copy()
    first.axpy(dt*n1, stiffness, True)
    later = mass.copy()
    later *= 1.5
    later.axpy(dt*n1, stiffness, True)
    solvers = []
    for matrix in (first, later):
        solver = df.PETScKrylovSolver(controls.linear_solver, controls.preconditioner)
        solver.set_operator(matrix)
        solver.parameters['absolute_tolerance'] = controls.krylov_atol
        solver.parameters['relative_tolerance'] = controls.krylov_rtol
        solver.parameters['maximum_iterations'] = controls.krylov_maximum_iterations
        solver.parameters['error_on_nonconvergence'] = True
        solver.parameters['nonzero_initial_guess'] = True
        solvers.append(solver)
    reaction_form = (n2*(phi**3-phi)+n3*(1-phi**2))*test*measure
    reaction = df.assemble(reaction_form)
    current_mass = mass * phi.vector()
    previous_mass = current_mass.copy()
    previous_reaction = reaction.copy()

    def advance(step):
        nonlocal current_mass, previous_mass, previous_reaction
        rhs = current_mass.copy()
        if step == 1:
            rhs.axpy(-dt, reaction)
        else:
            rhs *= 2
            rhs.axpy(-.5, previous_mass)
            rhs.axpy(-2*dt, reaction)
            rhs.axpy(dt, previous_reaction)
        iterations = solvers[0 if step == 1 else 1].solve(phi.vector(), rhs)
        phi.vector().apply('insert')
        previous_mass, current_mass = current_mass, mass * phi.vector()
        previous_reaction = reaction.copy()
        df.assemble(reaction_form, tensor=reaction)
        return iterations
    return advance


def query_plan(mesh, points):
    tree = mesh.bounding_box_tree()
    return np.array([i for i, x in enumerate(points)
                     if tree.compute_first_entity_collision(df.Point(*x)) < mesh.num_cells()], dtype=int)


def sample_field(phi, points, local_ids, comm):
    """Collect actual FE evaluation at all queries, including other MPI ranks."""
    local = np.array([phi(df.Point(*points[i])) for i in local_ids])
    parts = comm.gather((local_ids, local), root=0)
    result = None
    failure = None
    if comm.rank == 0:
        sums, counts = np.zeros(len(points)), np.zeros(len(points), dtype=int)
        lo, hi = np.full(len(points), np.inf), np.full(len(points), -np.inf)
        for ids, values in parts:
            sums[ids] += values
            counts[ids] += 1
            lo[ids] = np.minimum(lo[ids], values)
            hi[ids] = np.maximum(hi[ids], values)
        if np.any(counts == 0) or np.max(hi-lo) > 1e-8:
            failure = 'MPI point evaluation has missing/inconsistent FE values'
        else:
            result = sums / counts
    failure = comm.bcast(failure, root=0)
    if failure is not None:
        raise ValueError(failure)
    return result


def solve(c, index, subdivisions, dt, output):
    """One explicit resolution/time step, no automatic numerical fallback."""
    comm = MPI.COMM_WORLD
    problem = load_problem(c.reference.training_config, c.reference.manifest, index)
    if (subdivisions < 4 or c.mesh.degree != 2 or c.mesh.quadrature_degree < 8
            or not np.isfinite(dt) or dt <= 0 or c.reference.snapshots < 2):
        raise ValueError('Reference requires CG2, quadrature degree>=8, subdivisions>=4, snapshots>=2 and positive dt')
    steps = round(1 / dt)
    stride = steps // (c.reference.snapshots - 1)
    if steps * dt != 1 or steps % (c.reference.snapshots - 1):
        raise ValueError('dt must exactly divide the final time and saved intervals')
    if Path(output).exists():
        raise FileExistsError(output)
    native_path = Path(c.reference.manifest).parent / f'condition_{index:03d}.npz'
    with np.load(native_path) as native:
        points = native['points']
        slice_points = native['slice_points']
        times = native['time']
        np.testing.assert_array_equal(native['parameters'], problem['parameters'])
    np.testing.assert_array_equal(times, np.linspace(0, 1, c.reference.snapshots))
    query = np.concatenate([points, slice_points])
    begin = time.perf_counter()
    mesh = df.UnitCubeMesh(comm, subdivisions, subdivisions, subdivisions)
    V = df.FunctionSpace(mesh, 'CG', c.mesh.degree)
    phi, previous = df.Function(V), df.Function(V)
    coordinates = V.tabulate_dof_coordinates().reshape(-1, 3)
    count = phi.vector().local_size()
    phi.vector().set_local(initial_field(coordinates[:count], problem))
    phi.vector().apply('insert')
    previous.assign(phi)
    measure = df.dx(metadata={'quadrature_degree': c.mesh.quadrature_degree})
    test = df.TestFunction(V)
    if c.time.scheme == 'implicit_midpoint':
        residual = weak_residual(phi, previous, test, dt, problem['coefficients'], measure)
        jacobian = df.derivative(residual, phi, df.TrialFunction(V))
        nonlinear = df.NonlinearVariationalProblem(residual, phi, [], jacobian)
        solver = df.NonlinearVariationalSolver(nonlinear)
        options = solver.parameters['newton_solver']
        options['linear_solver'] = c.solver.linear_solver
        options['preconditioner'] = c.solver.preconditioner
        options['absolute_tolerance'] = c.solver.atol
        options['relative_tolerance'] = c.solver.rtol
        options['maximum_iterations'] = c.solver.maximum_iterations
        options['error_on_nonconvergence'] = True
        options['krylov_solver']['absolute_tolerance'] = c.solver.krylov_atol
        options['krylov_solver']['relative_tolerance'] = c.solver.krylov_rtol
        options['krylov_solver']['maximum_iterations'] = c.solver.krylov_maximum_iterations
        options['krylov_solver']['error_on_nonconvergence'] = True
        advance = None
    elif c.time.scheme == 'imex_bdf2':
        advance = make_imex_step(phi, V, dt, problem['coefficients'], measure, c.solver)
    else:
        raise ValueError('time.scheme must be implicit_midpoint or imex_bdf2')
    df.set_log_level(40)
    local_ids = query_plan(mesh, query)
    fields = [sample_field(phi, query, local_ids, comm)]
    volume = [float(df.assemble((phi + 1) / 2 * measure))]
    if comm.rank == 0:
        print('FEM_START', json.dumps(dict(index=index, N=subdivisions, dt=dt,
              dofs=V.dim(), ranks=comm.size, setup_seconds=time.perf_counter()-begin,
              initial_min=float(fields[0].min()), initial_max=float(fields[0].max()))), flush=True)
    iterations = []
    for step in range(1, steps + 1):
        if advance is None:
            previous.assign(phi)
            n, converged = solver.solve()
            if not converged:
                raise RuntimeError(f'Newton failed at step {step}')
        else:
            n = advance(step)
        iterations.append(int(n))
        if step % stride == 0:
            fields.append(sample_field(phi, query, local_ids, comm))
            volume.append(float(df.assemble((phi + 1) / 2 * measure)))
            if comm.rank == 0:
                print(json.dumps(dict(index=index, N=subdivisions, dt=dt, step=step,
                                      steps=steps, seconds=time.perf_counter()-begin)), flush=True)
    failure = None
    if comm.rank == 0:
        values = np.array(fields)
        finite = bool(np.isfinite(values).all())
        range_pass = finite and values.min() >= -1.005 and values.max() <= 1.005
        if not range_pass:
            failure = f'Invalid FEM range: finite={finite}, min={values.min()}, max={values.max()}'
        info = dict(problem=problem, mesh_subdivisions=subdivisions, degree=c.mesh.degree,
                    quadrature_degree=c.mesh.quadrature_degree, dt=dt,
                    scheme=c.time.scheme+', consistent CG2 tetrahedral FEM',
                    global_dofs=V.dim(), mpi_ranks=comm.size, seconds=time.perf_counter()-begin,
                    solver_iterations=iterations, config_path=c.reference.training_config,
                    manifest_path=c.reference.manifest, validated_reference=False,
                    finite_field=finite, range_gate_pass=bool(range_pass),
                    phase_min=float(values.min()) if finite else None,
                    phase_max=float(values.max()) if finite else None)
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, phi=values[:, :len(points)], slice_phi=values[:, len(points):],
                            points=points, slice_points=slice_points, time=times,
                            parameters=problem['parameters'], solid_fraction=volume,
                            metadata_json=np.array(json.dumps(info, allow_nan=False)))
        Path(output).with_suffix('.json').write_text(json.dumps(info, indent=2, allow_nan=False)+'\n')
        print('FEM_COMPLETE', output, info['seconds'], flush=True)
    failure = comm.bcast(failure, root=0)
    if failure is not None:
        raise FloatingPointError(failure)
    comm.Barrier()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='Dotted Python configuration module')
    parser.add_argument('--index', type=int, required=True)
    parser.add_argument('--subdivisions', type=int, required=True)
    parser.add_argument('--dt', type=float, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        solve(importlib.import_module(args.config).get_config(), args.index,
              args.subdivisions, args.dt, args.output)
    except BaseException as error:
        # Preserve the traceback and fail the MPI job instead of hanging when
        # an unhandled rank-local error reaches MPI_Finalize on only one rank.
        from mpi4py.run import set_abort_status
        set_abort_status(error)
        raise
