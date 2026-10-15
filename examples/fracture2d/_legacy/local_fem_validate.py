"""Independent UFL validation of a prepared local-damage mesh; use fenicsenv."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from fenics import (
    Mesh, MeshEditor, VectorFunctionSpace, FunctionSpace, Function,
    TrialFunction, TestFunction, assemble, inner, sym, grad, div, dx,
    project, as_backend_type, set_log_active,
)

from examples.fracture2d._legacy.local_mechanics import LocalMechanics
from examples.fracture2d._legacy.solver import _hybrid_split


def validate(root):
    metadata = json.loads((root/'identity.json').read_text())
    if hashlib.sha256((root/'mesh.npz').read_bytes()).hexdigest() != metadata['mesh_sha256']:
        raise ValueError('prepared mesh SHA differs')
    with np.load(root/'mesh.npz', allow_pickle=False) as f:
        arrays = {name: f[name].copy() for name in f.files}
    p = metadata['config']
    n, nc = int(arrays['n_material']), int(arrays['n_material_cells'])
    m = LocalMechanics(arrays['coordinates'], arrays['triangles'], n, nc,
        arrays['volumes'], E=p['E'], nu=p['nu'], k_res=p['k_res'],
        constrained=arrays['constrained'], load_direction=arrays['load_direction'])
    start = time.perf_counter()
    set_log_active(False)
    mesh, editor = Mesh(), MeshEditor()
    editor.open(mesh, 'triangle', 2, 2)
    editor.init_vertices(len(m.coordinates))
    editor.init_cells(len(m.triangles))
    for i, point in enumerate(m.coordinates):
        editor.add_vertex(i, point)
    for i, triangle in enumerate(m.triangles):
        editor.add_cell(i, triangle.astype(np.uintp))
    editor.close()
    np.testing.assert_array_equal(mesh.coordinates(), m.coordinates)
    np.testing.assert_array_equal(np.sort(mesh.cells(), axis=1), np.sort(m.triangles, axis=1))
    space = VectorFunctionSpace(mesh, 'CG', 1)
    scalar = FunctionSpace(mesh, 'CG', 1)
    cells = FunctionSpace(mesh, 'DG', 0)
    lookup = cKDTree(m.coordinates)
    error, nodes = lookup.query(space.tabulate_dof_coordinates().reshape(-1, 2))
    if np.max(error) > 1e-14:
        raise ValueError('UFL vector nodes differ from prepared mesh')
    mapping = np.zeros(space.dim(), dtype=np.int64)
    for component in (0, 1):
        indices = space.sub(component).dofmap().dofs()
        mapping[indices] = 2*nodes[indices]+component
    if len(np.unique(mapping)) != 2*len(m.coordinates):
        raise ValueError('UFL vector DOF mapping is not bijective')
    error, scalar_nodes = lookup.query(scalar.tabulate_dof_coordinates().reshape(-1, 2))
    if np.max(error) > 1e-14:
        raise ValueError('UFL scalar nodes differ from prepared mesh')
    cell_dofs = np.array([cells.dofmap().cell_dofs(i)[0] for i in range(len(m.triangles))])
    material = Function(cells)
    phase = np.zeros(cells.dim())
    phase[cell_dofs[:nc]] = 1
    material.vector().set_local(phase)
    damage = Function(scalar)
    d = .85*np.exp(-((m.coordinates[:n, 1]-.5)/.025)**2)
    all_damage = np.zeros(len(m.coordinates))
    all_damage[:n] = d
    damage.vector().set_local(all_damage[scalar_nodes])
    g = material*((1-damage)**2+p['k_res'])+(1-material)*(1+p['k_res'])
    u, v = TrialFunction(space), TestFunction(space)
    form = g*(m.lam*div(u)*div(v)+2*m.mu*inner(sym(grad(u)), sym(grad(v))))*dx(
        metadata={'quadrature_degree': 2})
    assembled = assemble(form)
    indptr, indices, data = as_backend_type(assembled).mat().getValuesCSR()
    matrix = csr_matrix((data, indices, indptr), shape=(space.dim(), space.dim()))
    rng = np.random.default_rng(42)
    q = rng.normal(size=len(mapping))*1e-4
    host = m.matrix(d) @ q
    ufl_force = matrix @ q[mapping]
    stiffness_relative = float(np.max(abs(ufl_force-host[mapping]))/np.max(abs(host)))
    np.testing.assert_allclose(ufl_force, host[mapping], rtol=2e-11, atol=2e-8)
    q_field = Function(space)
    q_field.vector().set_local(q[mapping])
    positive, _, _ = _hybrid_split(sym(grad(q_field)), m.lam, m.mu)
    cell_positive = project(positive, cells, solver_type='cg', preconditioner_type='jacobi')
    cell_values = cell_positive.vector().get_local()[cell_dofs[:nc]]
    projected = np.asarray(m.projection @ cell_values)/m.volumes
    host_driver = m.driver(q)
    driver_relative = float(np.max(abs(projected-host_driver))/np.max(abs(host_driver)))
    np.testing.assert_allclose(projected, host_driver, rtol=1e-10, atol=1e-7)
    result = dict(mesh_sha256=metadata['mesh_sha256'], stiffness_relative_inf=stiffness_relative,
        driver_relative_inf=driver_relative, seconds=time.perf_counter()-start,
        union_nodes=len(m.coordinates), cells=len(m.triangles), independent_ufl=True)
    result['pass'] = True
    (root/'fenics_parity.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--geometry-dir', required=True, type=Path)
    validate(parser.parse_args().geometry_dir)
