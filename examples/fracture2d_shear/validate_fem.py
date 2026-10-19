"""Independent UFL operator, edge-BC and horizontal-reaction checks; fenicsenv."""
from pathlib import Path
import json
import time
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, save_npz
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
from fenics import (Mesh, MeshEditor, VectorFunctionSpace, FunctionSpace,
    Function, TrialFunction, TestFunction, assemble, inner, sym, grad, div,
    dx, project, as_backend_type, set_log_active)
from ufl import conditional, gt
from examples.fracture2d.local_mechanics import LocalMechanics


def checked_csr(form, size):
    assembled = assemble(form)
    indptr, indices, values = as_backend_type(assembled).mat().getValuesCSR()
    return csr_matrix((values.copy(), indices.copy(), indptr.copy()), shape=(size, size))


def validate(config):
    root = Path(config.mesh.prepared_dir)
    metadata = json.loads((root/'identity.json').read_text())
    p = metadata['config']
    if p['boundary']['mode'] != 'horizontal_clamped_edges' or p['grip_extent'] != 0:
        raise ValueError('independent check requires material-only clamped edge shear')
    with np.load(root/'mesh.npz', allow_pickle=False) as z:
        a = {k: z[k].copy() for k in z.files}
    n, nc = int(a['n_material']), int(a['n_material_cells'])
    if len(a['coordinates']) != n or len(a['triangles']) != nc:
        raise ValueError('shear cache must not contain added grip regions')
    m = LocalMechanics(a['coordinates'], a['triangles'], n, nc, a['volumes'],
        E=p['E'], nu=p['nu'], k_res=p['k_res'], constrained=a['constrained'],
        load_direction=a['load_direction'], range_atol=config.validation.range_atol)
    started = time.perf_counter()
    set_log_active(False)
    mesh, editor = Mesh(), MeshEditor()
    editor.open(mesh, 'triangle', 2, 2)
    editor.init_vertices(n); editor.init_cells(nc)
    for i, point in enumerate(m.coordinates):
        editor.add_vertex(i, point)
    for i, cell in enumerate(m.triangles):
        editor.add_cell(i, cell.astype(np.uintp))
    editor.close()
    np.testing.assert_array_equal(mesh.coordinates(), m.coordinates)
    np.testing.assert_array_equal(np.sort(mesh.cells(), axis=1), np.sort(m.triangles, axis=1))
    space = VectorFunctionSpace(mesh, 'CG', 1)
    scalar = FunctionSpace(mesh, 'CG', 1)
    cell_space = FunctionSpace(mesh, 'DG', 0)
    tree = cKDTree(m.coordinates)
    error, nodes = tree.query(space.tabulate_dof_coordinates().reshape(-1, 2))
    np.testing.assert_allclose(error, 0., atol=1e-14)
    mapping = np.empty(space.dim(), np.int64)
    for component in (0, 1):
        dofs = space.sub(component).dofmap().dofs()
        mapping[dofs] = 2*nodes[dofs]+component
    if len(np.unique(mapping)) != 2*n:
        raise ValueError('UFL vector/material DOF mapping is not bijective')
    error, scalar_nodes = tree.query(scalar.tabulate_dof_coordinates().reshape(-1, 2))
    np.testing.assert_allclose(error, 0., atol=1e-14)
    scalar_order = np.argsort(scalar_nodes)
    if len(np.unique(scalar_nodes)) != n:
        raise ValueError('UFL scalar/material node mapping is not bijective')
    cell_dofs = np.array([cell_space.dofmap().cell_dofs(i)[0] for i in range(nc)])
    damage = Function(scalar)
    d = .75*np.exp(-((m.coordinates[:, 1]-.5)/.04)**2)
    damage.vector().set_local(d[scalar_nodes])
    u, v = TrialFunction(space), TestFunction(space)
    elastic = m.lam*div(u)*div(v)+2*m.mu*inner(sym(grad(u)), sym(grad(v)))
    matrix = checked_csr(((1-damage)**2+p['k_res'])*elastic*dx(
        metadata={'quadrature_degree': 2}), space.dim())
    q = np.random.default_rng(42).normal(size=2*n)*1e-4
    host_force, ufl_force = m.matrix(d)@q, matrix@q[mapping]
    stiffness_error = float(np.max(abs(ufl_force-host_force[mapping]))/np.max(abs(host_force)))
    np.testing.assert_allclose(ufl_force, host_force[mapping], rtol=2e-11, atol=2e-8)
    q_field = Function(space); q_field.vector().set_local(q[mapping])
    strain, trace = sym(grad(q_field)), div(q_field)
    trace_plus = conditional(gt(trace, 0.), trace, 0.)
    # Plane strain embedded in 3D: |dev eps|²=|eps_2d|²-tr(eps)²/3.
    positive = (.5*(m.lam+2*m.mu/3)*trace_plus**2
                + m.mu*(inner(strain, strain)-trace**2/3))
    projected = project(positive, cell_space, solver_type='cg', preconditioner_type='jacobi')
    ufl_driver = np.asarray(m.projection@projected.vector().get_local()[cell_dofs])/m.volumes
    host_driver = m.driver(q)
    driver_error = float(np.max(abs(ufl_driver-host_driver))/np.max(abs(host_driver)))
    np.testing.assert_allclose(ufl_driver, host_driver, rtol=1e-10, atol=1e-7)

    trial, test = TrialFunction(scalar), TestFunction(scalar)
    S = checked_csr(inner(grad(trial), grad(test))*dx, n)[scalar_order][:, scalar_order].tocsr()
    gradients = np.stack((m.B[:, 0, 0::2], m.B[:, 1, 1::2]), axis=2)
    elements = m.areas[:, None, None]*np.einsum('eia,eja->eij', gradients, gradients)
    rows = np.broadcast_to(m.triangles[:, :, None], elements.shape)
    columns = np.broadcast_to(m.triangles[:, None, :], elements.shape)
    expected = coo_matrix((elements.ravel(), (rows.ravel(), columns.ravel())), shape=(n, n)).tocsr()
    defect = S-expected
    scalar_error = float(np.max(abs(defect.data))/np.max(abs(S.data))) if defect.nnz else 0.
    if scalar_error > 1e-11:
        raise AssertionError('UFL scalar stiffness differs from material-P1 operator')
    save_npz(root/'local_scalar_stiffness.npz', S)

    # Independently derive edge constraints from UFL DOF coordinates.
    coordinates = space.tabulate_dof_coordinates().reshape(-1, 2)
    edge_dofs = abs(coordinates[:, 1]) < 1e-12
    edge_dofs |= abs(coordinates[:, 1]-1.) < 1e-12
    np.testing.assert_array_equal(np.sort(mapping[edge_dofs]), m.constrained)
    direction = np.zeros(2*n)
    x_dofs = np.array(space.sub(0).dofmap().dofs())
    top = x_dofs[abs(coordinates[x_dofs, 1]-1.) < 1e-12]
    direction[mapping[top]] = 1.
    np.testing.assert_array_equal(direction, m.load_direction)
    K = checked_csr((1+p['k_res'])*elastic*dx, space.dim())
    order = np.argsort(mapping); K = K[order][:, order].tocsr()
    load = 1e-4
    displacement = load*direction
    rhs = -(K@displacement)[m.free]
    displacement[m.free] = spsolve(K[m.free][:, m.free], rhs)
    force = K@displacement
    reaction = float(force@direction)
    host_matrix = m.matrix(np.zeros(n))
    host = load*direction
    host[m.free] = spsolve(host_matrix[m.free][:, m.free], -(host_matrix@host)[m.free])
    np.testing.assert_allclose(displacement, host, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(reaction, (host_matrix@host)@direction, rtol=1e-8, atol=1e-7)
    np.testing.assert_allclose(displacement[m.constrained], load*direction[m.constrained], atol=1e-15)
    if reaction <= 0 or np.linalg.norm(force[m.free]) > 1e-9*max(1., np.linalg.norm(rhs)):
        raise AssertionError('UFL horizontal reaction/free equilibrium failed')
    np.savez_compressed(root/'elastic_check.npz', q=displacement, load=np.float64(load),
                        reaction=np.float64(reaction))
    result = dict(pass_checks=True, independent_ufl=True, material_only=True,
        nodes=n, cells=nc, boundary=p['boundary'], stiffness_relative_inf=stiffness_error,
        driver_relative_inf=driver_error, scalar_relative_inf=scalar_error,
        elastic_load=load, horizontal_reaction=reaction,
        seconds=time.perf_counter()-started)
    result['pass'] = True
    (root/'fenics_parity.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
    return result


def main():
    from absl import app, flags
    from ml_collections import config_flags
    config_flags.DEFINE_config_file('config', str(Path(__file__).with_name('configs')/'train_base.py'),
                                   'Independent shear FEM validation', lock_config=True)
    def execute(_):
        validate(flags.FLAGS.config)
    app.run(execute)


if __name__ == '__main__':
    main()
