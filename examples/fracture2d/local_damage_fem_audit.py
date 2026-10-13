"""Independent UFL replay of matched local/nonlocal damage paths; fenicsenv.

Rebuild mechanics, material lumped mass, nodal Amor projection and scalar P1
stiffness from weak forms. Check every saved accepted field and selected fixed-H
damage solutions with SciPy sparse LU. No JAX/production assembly is imported.
This verifies a discrete fracture path, not spatial/continuum convergence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.sparse import csr_matrix, diags, load_npz
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
# The independent reference entry configures DOLFIN's pkg-config path before
# importing FEniCS, in addition to supplying the checked constitutive split.
from examples.fracture2d.solver import _hybrid_split
from fenics import (Mesh, MeshEditor, VectorFunctionSpace, FunctionSpace,
    Function, TrialFunction, TestFunction, assemble, inner, sym, grad, div, dx,
    as_backend_type, set_log_active)

from examples.fracture2d.coupling_diagnostics import atomic_json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sparse(matrix, size):
    indptr, indices, data = as_backend_type(matrix).mat().getValuesCSR()
    return csr_matrix((data, indices, indptr), shape=(size, size))


def at2(d, H, D, gc0, gc1):
    parts = (gc0 * d, gc1 * (D @ d), 2 * H * (d - 1))
    return float(np.max(abs(sum(parts))) / sum(np.max(abs(p)) for p in parts))


def audit(geometry_dir, paths, output, oracle_loads):
    start = time.perf_counter()
    set_log_active(False)
    geometry = json.loads((geometry_dir / 'identity.json').read_text())
    if sha(geometry_dir / 'mesh.npz') != geometry['mesh_sha256']:
        raise ValueError('independent audit mesh SHA differs')
    with np.load(geometry_dir / 'mesh.npz', allow_pickle=False) as f:
        data = {k: f[k].copy() for k in f.files}
    points, tri = data['coordinates'], data['triangles']
    n, nc = int(data['n_material']), int(data['n_material_cells'])
    p = geometry['config']
    mesh, editor = Mesh(), MeshEditor()
    editor.open(mesh, 'triangle', 2, 2)
    editor.init_vertices(len(points))
    editor.init_cells(len(tri))
    for i, point in enumerate(points):
        editor.add_vertex(i, point)
    for i, triangle in enumerate(tri):
        editor.add_cell(i, triangle.astype(np.uintp))
    editor.close()
    np.testing.assert_array_equal(mesh.coordinates(), points)
    np.testing.assert_array_equal(np.sort(mesh.cells(), axis=1), np.sort(tri, axis=1))
    Vu, Vd, Vc = VectorFunctionSpace(mesh, 'CG', 1), FunctionSpace(mesh, 'CG', 1), FunctionSpace(mesh, 'DG', 0)
    lookup = cKDTree(points)
    error, scalar_nodes = lookup.query(Vd.tabulate_dof_coordinates().reshape(-1, 2))
    if np.max(error) > 1e-14 or len(np.unique(scalar_nodes)) != len(points):
        raise ValueError('UFL scalar mapping differs')
    scalar_inverse = np.argsort(scalar_nodes)
    error, vector_nodes = lookup.query(Vu.tabulate_dof_coordinates().reshape(-1, 2))
    if np.max(error) > 1e-14:
        raise ValueError('UFL vector coordinates differ')
    mapping = np.zeros(Vu.dim(), np.int64)
    for component in (0, 1):
        dofs = Vu.sub(component).dofmap().dofs()
        mapping[dofs] = 2 * vector_nodes[dofs] + component
    if len(np.unique(mapping)) != 2 * len(points):
        raise ValueError('UFL vector mapping is not bijective')
    inverse = np.argsort(mapping)
    cell_dofs = np.array([Vc.dofmap().cell_dofs(i)[0] for i in range(len(tri))])
    material = Function(Vc)
    phase = np.zeros(Vc.dim())
    phase[cell_dofs[:nc]] = 1
    material.vector().set_local(phase)
    d_field, q_field = Function(Vd), Function(Vu)
    u, v = TrialFunction(Vu), TestFunction(Vu)
    ds, vs = TrialFunction(Vd), TestFunction(Vd)
    lam, mu = p['E'] * p['nu'] / ((1 + p['nu']) * (1 - 2 * p['nu'])), p['E'] / (2 * (1 + p['nu']))
    measure = dx(metadata={'quadrature_degree': 2})
    g = material * ((1 - d_field)**2 + p['k_res']) + (1 - material) * (1 + p['k_res'])
    mechanics_form = g * (lam * div(u) * div(v) + 2 * mu * inner(sym(grad(u)), sym(grad(v)))) * measure
    positive, _, _ = _hybrid_split(sym(grad(q_field)), lam, mu)
    driver_form = material * positive * vs * measure
    mass = assemble(material * vs * measure).get_local()[scalar_inverse][:n]
    np.testing.assert_allclose(mass, data['volumes'], rtol=1e-12, atol=1e-15)
    S_all = sparse(assemble(material * inner(grad(ds), grad(vs)) * measure), Vd.dim())
    S = S_all[scalar_inverse[:n]][:, scalar_inverse[:n]].tocsr()
    cached = load_npz(geometry_dir / 'local_scalar_stiffness.npz')
    difference = S - cached
    stiffness_error = float(np.max(abs(difference.data)) / np.max(abs(S.data))) if difference.nnz else 0.
    if stiffness_error > 1024 * np.finfo(float).eps:
        raise ValueError('independent UFL scalar stiffness differs from cached P1 control')
    local_D = diags(1 / mass) @ S
    constrained = np.sort(np.concatenate((2 * np.flatnonzero((np.arange(len(points)) >= n) & (points[:, 1] < 0)) + 1,
        2 * np.flatnonzero((np.arange(len(points)) >= n) & (points[:, 1] > 1)) + 1, [0])))
    np.testing.assert_array_equal(constrained, data['constrained'])
    direction = np.zeros(2 * len(points))
    direction[2 * np.flatnonzero((np.arange(len(points)) >= n) & (points[:, 1] > 1)) + 1] = 1
    np.testing.assert_array_equal(direction, data['load_direction'])
    free = np.ones(len(direction), bool)
    free[constrained] = False
    results = []
    for path in paths:
        identity = json.loads((path / 'identity.json').read_text())
        if identity['geometry'] != geometry or identity['mechanics_code_sha256'] != sha(
                Path(__file__).with_name('local_mechanics.py')):
            raise ValueError('audited path geometry/mechanics identity differs')
        c = identity['config']
        local = identity['schema'] == 'local-gradient-damage-control-v1'
        if not local and identity['schema'] != 'local-mechanics-nonlocal-damage-trial-v1':
            raise ValueError('unsupported damage-path schema')
        D = local_D if local else load_npz(geometry_dir / 'damage_operator.npz')
        if not local and sha(geometry_dir / 'damage_operator.npz') != geometry['operator_sha256']:
            raise ValueError('nonlocal audit operator SHA differs')
        gc0, gc1 = c['physical']['Gc'] / c['physical']['l0'], c['physical']['Gc'] * c['physical']['l0']
        original = json.loads((path / 'result.json').read_text())
        if not original['complete'] or any(r['status'] != 'ready' for r in original['curve']):
            raise ValueError('independent reference replay requires a completed ready path')
        rows, parent_sha = [], None
        H_parent, d_parent = np.zeros(n), np.zeros(n)
        for expected in original['curve']:
            folder = path / f"state_{expected['index']:04d}"
            report = json.loads((folder / 'result.json').read_text())
            if (report['parent_sha256'] != parent_sha or report['state_sha256'] != sha(folder / 'fields.npz')
                    or report != expected or not report['outer_convergence']['converged']):
                raise ValueError('accepted path report/parent/SHA differs')
            with np.load(folder / 'fields.npz', allow_pickle=False) as f:
                fields = {k: f[k].copy() for k in f.files}
            q, d, H = (fields[k] for k in ('q', 'd', 'H'))
            np.testing.assert_array_equal(fields['H_parent'], H_parent)
            np.testing.assert_array_equal(fields['d_parent'], d_parent)
            np.testing.assert_array_equal(q[constrained], report['load'] * direction[constrained])
            all_d = np.zeros(len(points))
            all_d[:n] = d
            d_field.vector().set_local(all_d[scalar_nodes])
            q_field.vector().set_local(q[mapping])
            K_ufl = sparse(assemble(mechanics_form), Vu.dim())
            force_ufl = K_ufl @ q[mapping]
            force = force_ufl[inverse]
            norm, reaction = float(np.linalg.norm(force[free])), float(force @ direction)
            factor = c['validation']['roundoff_factor']
            # Independent assembly can differ in accumulation order. Bound its
            # free-force roundoff explicitly, separately from the solve gate.
            rounding = factor * np.finfo(float).eps * np.linalg.norm((abs(K_ufl) @ abs(q[mapping]))[inverse][free])
            force_limit = report['mechanics']['convergence_limit'] * (1 + 1e-3)
            if norm > force_limit + rounding:
                raise RuntimeError('UFL equilibrium replay exceeds solve gate plus roundoff bound')
            driver = assemble(driver_form).get_local()[scalar_inverse][:n] / mass
            H_expected = np.maximum(H_parent, driver)
            np.testing.assert_allclose(H, H_expected, rtol=1e-10, atol=1e-10)
            current = at2(d, H, D, gc0, gc1)
            image = at2(fields['damage_image'], fields['H_used'], D, gc0, gc1)
            if (abs(current - report['coupled_AT2_relative']) > factor * np.finfo(float).eps
                    or abs(image - report['fixed_H_image_AT2_relative']) > factor * np.finfo(float).eps
                    or current > c['reporting']['coupled_AT2_relative_limit']):
                raise RuntimeError('independent damage residual/report/gate differs')
            row = dict(index=report['index'], load=report['load'], reaction=reaction,
                reaction_abs_difference=abs(reaction-report['reaction']),
                free_force_norm=norm, solve_limit=force_limit, assembly_roundoff_bound=rounding,
                driver_relative_inf=float(np.max(abs(H-H_expected)) / max(np.max(H), 1e-12)),
                current_AT2_relative=current, image_AT2_relative=image,
                current_report_abs_difference=abs(current-report['coupled_AT2_relative']))
            if any(abs(report['load'] - value) < 1e-12 for value in oracle_loads):
                A = diags(mass * (gc0 + 2 * fields['H_used'])) + gc1 * (diags(mass) @ D)
                rhs = 2 * mass * fields['H_used']
                oracle = spsolve(A.tocsc(), rhs)
                image_residual = A @ fields['damage_image'] - rhs
                oracle_residual = A @ oracle - rhs
                # A >= gc0*min(mass)*I: standard P1 S and reciprocal graph
                # mass*D are PSD. This gives an independently checkable bound
                # on field differences for the declared weighted CG norm gate.
                field_bound = np.linalg.norm(image_residual - oracle_residual) / (gc0 * np.min(mass))
                cg_limit = max(c['validation']['cg_atol'], c['validation']['cg_rtol'] * np.linalg.norm(rhs))
                image_rounding = factor * np.finfo(float).eps * np.linalg.norm(
                    abs(A) @ abs(fields['damage_image']) + abs(rhs))
                row['fixed_H_LU_image_inf_difference'] = float(np.max(abs(oracle-fields['damage_image'])))
                row['fixed_H_LU_relative_residual'] = float(np.linalg.norm(oracle_residual) / np.linalg.norm(rhs))
                row['fixed_H_image_residual_norm'] = float(np.linalg.norm(image_residual))
                row['fixed_H_CG_limit'] = float(cg_limit)
                row['fixed_H_image_assembly_roundoff_bound'] = float(image_rounding)
                row['fixed_H_SPD_field_inf_error_bound'] = float(field_bound)
                if (row['fixed_H_LU_image_inf_difference'] > field_bound * (1 + 1e-3)
                        or row['fixed_H_image_residual_norm'] > cg_limit * (1 + 1e-3) + image_rounding
                        or row['fixed_H_LU_relative_residual'] > 1e-10):
                    raise RuntimeError(f'independent LU/CG norm/error-bound check differs: {row}')
            rows.append(row)
            H_parent, d_parent, parent_sha = H, d, report['state_sha256']
            print(json.dumps(dict(root=path.name, index=report['index'], load=report['load'],
                                  current_AT2=current, UFL_force=norm)), flush=True)
        results.append(dict(root=str(path), identity_sha256=sha(path / 'identity.json'),
                            result_sha256=sha(path / 'result.json'), local_gradient=local, rows=rows, passed=True))
        atomic_json(output, dict(complete=False, paths=results, mesh_sha256=geometry['mesh_sha256']))
    result = dict(complete=True, passed=True, independent_ufl=True,
        interpretation='matched lumped/nodal discrete path verification; no independent trajectory or mesh convergence',
        audit_code_sha256=sha(__file__), reference_split_code_sha256=sha(Path(__file__).with_name('solver.py')),
        mesh_sha256=geometry['mesh_sha256'], scalar_stiffness_relative_inf=stiffness_error,
        seconds=time.perf_counter()-start, paths=results)
    atomic_json(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--geometry-dir', type=Path, required=True)
    parser.add_argument('--roots', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--oracle-loads', type=float, nargs='+', required=True)
    args = parser.parse_args()
    audit(args.geometry_dir, args.roots, args.output, args.oracle_loads)
